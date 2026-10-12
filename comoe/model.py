"""CoMoE: Codec-Mixture-of-Experts.

The paper configuration (``fusion: late_mean``, ``use_semantic: false``) is a
four-stream token classifier: two lower-level streams -> Expert-L, two
higher-level streams -> Expert-H, each an axial Transformer with masked mean
pooling, followed by independent linear heads whose logits are averaged.

The module also keeps the variants used in ablations:
  - Expert-S: pooled MERT embedding -> projection  (``use_semantic``; the
    ``MLP (MERT)`` baseline is Expert-S alone)
  - ContinuousMultiLayerExpert: linear projection of continuous MERT frames in
    place of the codebook embedding (``data.continuous: true``, MERT-continuous)
  - fusion variants ``concat`` / ``cross_attn`` / ``residual_semantic``
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _HAS_SDPA_KERNEL = True
except ImportError:
    _HAS_SDPA_KERNEL = False


class AxialTransformerBlock(nn.Module):
    """Depth-axis attn → Temporal-axis attn → FFN."""
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn_depth = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_time = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """x: (B, L, T, D). key_padding_mask: (B, T) True=pad."""
        B, L, T, D = x.shape

        # Depth attention — within each time step, attend across layers.
        # (B*T, L, D): L is tiny (2-8) but B*T can exceed flash/mem-eff limits
        # on torch 2.10+cu128, so force the MATH kernel here — L×L cost is trivial.
        h = x.permute(0, 2, 1, 3).reshape(B * T, L, D)   # (B*T, L, D)
        if _HAS_SDPA_KERNEL:
            with sdpa_kernel([SDPBackend.MATH]):
                a, _ = self.attn_depth(h, h, h, need_weights=False)
        else:
            a, _ = self.attn_depth(h, h, h, need_weights=False)
        h = self.ln1(h + a)
        h = h.reshape(B, T, L, D).permute(0, 2, 1, 3)    # (B, L, T, D)
        x = x + (h - x)  # equivalent to x = h, but keeps residual pattern

        # Temporal attention — within each layer, attend across time
        h = x.reshape(B * L, T, D)
        kpm = None
        if key_padding_mask is not None:
            kpm = key_padding_mask.unsqueeze(1).expand(B, L, T).reshape(B * L, T)
        a, _ = self.attn_time(h, h, h, key_padding_mask=kpm, need_weights=False)
        h = self.ln2(h + a)
        h = h.reshape(B, L, T, D)

        # FFN
        h = self.ln3(h + self.ffn(h))
        return h


class TokenExpert(nn.Module):
    """Embed + Axial Transformer + [CLS] pooling."""

    def __init__(
        self,
        codebook_size: int,
        layer_indices: List[int],
        d_model: int = 256,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.2,
        max_frames: int = 2250,
    ):
        super().__init__()
        self.layer_indices = layer_indices
        L = len(layer_indices)
        self.tok_embed = nn.Embedding(codebook_size, d_model)
        self.layer_embed = nn.Embedding(L, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, max_frames, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.blocks = nn.ModuleList([
            AxialTransformerBlock(d_model, n_heads, dropout) for _ in range(n_blocks)
        ])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor):
        """tokens: (B, L_total, T). Select layer_indices."""
        B, _, T = tokens.shape
        L = len(self.layer_indices)
        sel = tokens[:, self.layer_indices, :]    # (B, L, T)
        h = self.tok_embed(sel)                    # (B, L, T, D)

        layer_ids = torch.arange(L, device=h.device)
        h = h + self.layer_embed(layer_ids)[None, :, None, :]
        h = h + self.pos_embed[:, :, :T, :]

        # Build key_padding_mask
        idx = torch.arange(T, device=h.device)[None, :].expand(B, T)
        pad_mask = idx >= lengths[:, None]        # True = pad

        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad_mask)

        # Masked mean-pool across time, then mean across layers
        mask = (~pad_mask).float()[:, None, :, None]   # (B, 1, T, 1)
        h_time = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)   # (B, L, D)
        h_pool = h_time.mean(dim=1)               # (B, D)
        return self.ln_out(h_pool)


class ContinuousMultiLayerExpert(nn.Module):
    """Continuous multi-layer frame expert (drop-in continuous analog of TokenExpert).

    Input: (B, L_total, T, in_dim) float (e.g., MERT-v0 frames at 4 layers, in_dim=768).
    Mirrors TokenExpert: select layer_indices, axial transformer, masked mean-pool over
    time then mean over layers.  Only difference is a linear projection in place of
    the codebook embedding.
    """

    def __init__(
        self,
        in_dim: int,
        layer_indices: List[int],
        d_model: int = 256,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.2,
        max_frames: int = 2250,
    ):
        super().__init__()
        self.layer_indices = layer_indices
        L = len(layer_indices)
        self.in_proj = nn.Linear(in_dim, d_model)
        self.layer_embed = nn.Embedding(L, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, max_frames, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            AxialTransformerBlock(d_model, n_heads, dropout) for _ in range(n_blocks)
        ])
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, frames: torch.Tensor, lengths: torch.Tensor):
        """frames: (B, L_total, T, in_dim).  Select layer_indices."""
        B, _, T, _ = frames.shape
        L = len(self.layer_indices)
        sel = frames[:, self.layer_indices, :, :]    # (B, L, T, in_dim)
        h = self.in_proj(sel)                        # (B, L, T, D)

        layer_ids = torch.arange(L, device=h.device)
        h = h + self.layer_embed(layer_ids)[None, :, None, :]
        h = h + self.pos_embed[:, :, :T, :]

        idx = torch.arange(T, device=h.device)[None, :].expand(B, T)
        pad_mask = idx >= lengths[:, None]            # True = pad

        for blk in self.blocks:
            h = blk(h, key_padding_mask=pad_mask)

        mask = (~pad_mask).float()[:, None, :, None]
        h_time = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        h_pool = h_time.mean(dim=1)
        return self.ln_out(h_pool)


class LatentExpert(nn.Module):
    """Continuous pre-RVQ EnCodec latent → Transformer → pooled embedding.

    Input shape (B, in_dim, T) with in_dim=128 for EnCodec 24kHz.
    """
    def __init__(
        self,
        in_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.2,
        max_frames: int = 2250,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=False,
        )
        self.blocks = nn.TransformerEncoder(enc_layer, num_layers=n_blocks)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, latent: torch.Tensor, lengths: torch.Tensor):
        # latent: (B, in_dim, T)  → (B, T, in_dim)
        B, _, T = latent.shape
        x = latent.transpose(1, 2).contiguous()
        x = self.in_proj(x) + self.pos_embed[:, :T, :]
        idx = torch.arange(T, device=x.device)[None, :].expand(B, T)
        pad_mask = idx >= lengths[:, None]        # True = pad
        h = self.blocks(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).float()[:, :, None]
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.ln_out(pooled)


class SemanticExpert(nn.Module):
    """Projection head for pre-computed MERT embedding."""
    def __init__(self, in_dim: int = 768, d_model: int = 256, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, mert: torch.Tensor):
        return self.proj(mert)


class SemanticMultiLayerExpert(nn.Module):
    """Layer-grouped MERT expert.

    Input: (B, L_total, 768) per-layer mean-pooled MERT hidden states (L_total=13 for MERT-v1-95M).
    Each group pools its layers (mean) → linear → d_group, then groups are concatenated and
    projected to d_model.  Produces a single d_model vector per sample (same role as S stream).
    """
    def __init__(
        self,
        in_dim: int = 768,
        d_model: int = 256,
        dropout: float = 0.2,
        groups=((0, 1, 2), (5, 6, 7), (10, 11)),
    ):
        super().__init__()
        self.groups = [list(g) for g in groups]
        d_group = d_model
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, d_group),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_group),
            ) for _ in self.groups
        ])
        self.out_proj = nn.Sequential(
            nn.Linear(d_group * len(self.groups), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, mert_all: torch.Tensor):
        # mert_all: (B, L_total, 768)
        feats = []
        for g, proj in zip(self.groups, self.group_projs):
            idx = torch.tensor(g, device=mert_all.device, dtype=torch.long)
            pooled = mert_all.index_select(1, idx).mean(dim=1)   # (B, 768)
            feats.append(proj(pooled))
        return self.out_proj(torch.cat(feats, dim=-1))


class CoMoE(nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        low_layers: List[int] = [0, 1],
        high_layers: List[int] = [5, 6, 7],
        d_model: int = 256,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.2,
        max_frames: int = 2250,
        mert_dim: int = 768,
        use_semantic: bool = True,
        use_low: bool = True,
        use_high: bool = True,
        use_latent: bool = False,
        latent_dim: int = 128,
        mert_multilayer_groups=None,
        fusion: str = "concat",
        n_classes: int = 2,
        continuous_frames: bool = False,
        continuous_in_dim: int = 768,
    ):
        super().__init__()
        self.use_semantic = use_semantic
        self.use_low = use_low
        self.use_high = use_high
        self.use_latent = use_latent
        self.mert_multilayer_groups = mert_multilayer_groups
        self.fusion = fusion
        self.continuous_frames = continuous_frames

        if use_low:
            if continuous_frames:
                self.expert_L = ContinuousMultiLayerExpert(
                    continuous_in_dim, low_layers, d_model, n_heads, n_blocks, dropout, max_frames
                )
            else:
                self.expert_L = TokenExpert(
                    codebook_size, low_layers, d_model, n_heads, n_blocks, dropout, max_frames
                )
        if use_high:
            if continuous_frames:
                self.expert_H = ContinuousMultiLayerExpert(
                    continuous_in_dim, high_layers, d_model, n_heads, n_blocks, dropout, max_frames
                )
            else:
                self.expert_H = TokenExpert(
                    codebook_size, high_layers, d_model, n_heads, n_blocks, dropout, max_frames
                )
        if use_semantic:
            if mert_multilayer_groups:
                self.expert_S = SemanticMultiLayerExpert(
                    in_dim=mert_dim,
                    d_model=d_model,
                    dropout=dropout,
                    groups=mert_multilayer_groups,
                )
            else:
                self.expert_S = SemanticExpert(mert_dim, d_model, dropout)
        if use_latent:
            self.expert_Z = LatentExpert(
                latent_dim, d_model, n_heads, n_blocks, dropout, max_frames
            )
        n_streams = int(use_low) + int(use_high) + int(use_semantic) + int(use_latent)
        assert n_streams >= 1, "At least one expert must be enabled"

        if fusion == "concat":
            in_dim = d_model * n_streams
            self.head = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes),
            )
        elif fusion == "cross_attn":
            self.fuse_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes),
            )
        elif fusion == "late_mean":
            # Independent per-expert classifiers; final logit = mean of head outputs.
            def _head():
                return nn.Sequential(
                    nn.Linear(d_model, 256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, n_classes),
                )
            self.heads = nn.ModuleDict()
            if use_low: self.heads["L"] = _head()
            if use_high: self.heads["H"] = _head()
            if use_semantic: self.heads["S"] = _head()
            if use_latent: self.heads["Z"] = _head()
        elif fusion == "residual_semantic":
            assert use_semantic, "residual_semantic requires use_semantic=True"
            assert use_low or use_high, "residual_semantic requires at least one token expert"
            self.head_S = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes),
            )
            tok_dim = d_model * (int(use_low) + int(use_high))
            self.head_tok = nn.Sequential(
                nn.Linear(tok_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes),
            )
            self.alpha = nn.Parameter(torch.tensor(0.1))
        else:
            raise ValueError(fusion)

    def forward(self, tokens, lengths, mert=None, latent=None, latent_lengths=None, return_stream=False):
        streams = []
        stream_map = {}
        if self.use_low:
            h_L = self.expert_L(tokens, lengths)
            streams.append(h_L); stream_map["L"] = h_L
        if self.use_high:
            h_H = self.expert_H(tokens, lengths)
            streams.append(h_H); stream_map["H"] = h_H
        if self.use_semantic:
            assert mert is not None, "mert embedding required"
            h_S = self.expert_S(mert)
            streams.append(h_S); stream_map["S"] = h_S
        if self.use_latent:
            assert latent is not None, "latent required"
            ll = latent_lengths if latent_lengths is not None else lengths
            h_Z = self.expert_Z(latent, ll)
            streams.append(h_Z); stream_map["Z"] = h_Z

        if self.fusion == "concat":
            h = torch.cat(streams, dim=-1) if len(streams) > 1 else streams[0]
            logits = self.head(h)
        elif self.fusion == "late_mean":
            logit_list = [self.heads[k](v) for k, v in stream_map.items()]
            logits = torch.stack(logit_list, dim=0).mean(dim=0)
        elif self.fusion == "residual_semantic":
            tok_streams = []
            if self.use_low:
                tok_streams.append(stream_map["L"])
            if self.use_high:
                tok_streams.append(stream_map["H"])
            h_tok = torch.cat(tok_streams, dim=-1) if len(tok_streams) > 1 else tok_streams[0]
            logits = self.head_S(stream_map["S"]) + self.alpha * self.head_tok(h_tok)
        else:  # cross_attn: use last stream as query
            h_stack = torch.stack(streams, dim=1)             # (B, n_streams, D)
            q = streams[-1][:, None, :]
            fused, _ = self.fuse_attn(q, h_stack, h_stack)
            logits = self.head(fused.squeeze(1))

        if return_stream:
            return logits, stream_map
        return logits
