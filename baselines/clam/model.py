"""CLAM (Contrastive Learning for Audio Matching) — reimplementation.

Architecture (per arxiv 2512.00621 Fig. 1 + §3.2):
  - Two frozen pretrained encoders: MERT-v1-95M (music view) and Wav2Vec2-base (timbral view).
  - Per-stream WCA module:
      Conv1D(L→1, k=1) over hidden layers   (learned layer aggregation)
      Conv1D over time                      (local temporal context)
      Multi-Head Self-Attention             (intra-stream)
      mean-pool over time → linear          (compress to embed_dim)
  - Concat(d_music, d_vocal) → Linear → Dropout → Linear → 1 logit.
  - Loss = BCE + λ · Triplet; triplet uses real songs:
        anchor=d_music_real[i], pos=d_vocal_real[i], neg=d_vocal_real[j].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

MERT_NAME = "m-a-p/MERT-v1-95M"
W2V2_NAME = "facebook/wav2vec2-base"


class WCAStream(nn.Module):
    """Single-stream Weighted Cross-Aggregation."""

    def __init__(self, in_dim: int, n_layers: int, hidden: int = 384,
                 n_heads: int = 4, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # Learned layer aggregation: 1×1 conv over the layer axis.
        self.layer_agg = nn.Conv1d(n_layers, 1, kernel_size=1, bias=False)
        nn.init.constant_(self.layer_agg.weight, 1.0 / n_layers)

        self.proj_in = nn.Linear(in_dim, hidden)
        self.conv1d = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.ln1 = nn.LayerNorm(hidden)
        self.mha = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(hidden, embed_dim)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        """hs: (B, L, T, D) all-layer features → embed (B, embed_dim)."""
        B, L, T, D = hs.shape
        # Aggregate layers via 1×1 conv: treat (T*D) as conv1d "spatial" dim.
        x = hs.reshape(B, L, T * D)            # (B, L, T*D)
        x = self.layer_agg(x).squeeze(1)       # (B, T*D)
        x = x.view(B, T, D)                    # (B, T, D)

        x = self.proj_in(x)                    # (B, T, hidden)
        x = self.ln1(x)
        x = x + self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        a, _ = self.mha(x, x, x, need_weights=False)
        x = self.ln2(x + self.dropout(a))

        x = x.mean(dim=1)                      # mean-pool over time
        return self.proj_out(x)                # (B, embed_dim)


class CLAM(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden: int = 384, n_heads: int = 4,
                 dropout: float = 0.2, freeze_encoders: bool = True):
        super().__init__()
        self.mert = AutoModel.from_pretrained(MERT_NAME, trust_remote_code=True)
        self.w2v2: Wav2Vec2Model = Wav2Vec2Model.from_pretrained(W2V2_NAME)
        if freeze_encoders:
            for m in (self.mert, self.w2v2):
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()

        # MERT: 13 hidden states (1 conv + 12 transformer); W2V2-base: 13 likewise.
        self.n_layers_mert = self.mert.config.num_hidden_layers + 1
        self.n_layers_w2v2 = self.w2v2.config.num_hidden_layers + 1
        d_mert = self.mert.config.hidden_size
        d_w2v2 = self.w2v2.config.hidden_size

        self.stream_m = WCAStream(d_mert, self.n_layers_mert, hidden, n_heads, embed_dim, dropout)
        self.stream_v = WCAStream(d_w2v2, self.n_layers_w2v2, hidden, n_heads, embed_dim, dropout)

        self.head = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode_streams(self, wav_24k: torch.Tensor, wav_16k: torch.Tensor):
        # MERT (24 kHz). Use no_grad since frozen.
        with torch.no_grad():
            m_out = self.mert(input_values=wav_24k, output_hidden_states=True)
            v_out = self.w2v2(input_values=wav_16k, output_hidden_states=True)
        m_hs = torch.stack(m_out.hidden_states, dim=1)  # (B, L, Tm, Dm)
        v_hs = torch.stack(v_out.hidden_states, dim=1)  # (B, L, Tv, Dv)
        d_m = self.stream_m(m_hs.float())
        d_v = self.stream_v(v_hs.float())
        return d_m, d_v

    def forward(self, wav_24k: torch.Tensor, wav_16k: torch.Tensor):
        d_m, d_v = self.encode_streams(wav_24k, wav_16k)
        z = torch.cat([d_m, d_v], dim=-1)
        logit = self.head(z).squeeze(-1)
        return logit, d_m, d_v


def triplet_loss_real(d_m: torch.Tensor, d_v: torch.Tensor, label: torch.Tensor,
                      margin: float = 1.0) -> torch.Tensor:
    """label: 1=fake, 0=real. Apply triplet only to real samples in the batch.

    For each real i: anchor=d_m[i], positive=d_v[i], negative=d_v[j] with j a *different* real.
    Returns mean of max(0, ||a-p||² - ||a-n||² + margin) over usable real pairs;
    0 if fewer than 2 reals in the batch.
    """
    real_mask = label == 0
    if real_mask.sum() < 2:
        return d_m.new_zeros(())
    a = d_m[real_mask]
    p = d_v[real_mask]
    # Negative: random derangement of positives (avoids fixed points so neg != anchor).
    B = a.size(0)
    perm = torch.randperm(B, device=a.device)
    idx = torch.arange(B, device=a.device)
    fixed = (perm == idx).nonzero(as_tuple=True)[0]
    if fixed.numel() > 0:
        perm[fixed] = perm[(fixed + 1) % B]
    n = p[perm]
    dap = (a - p).pow(2).sum(-1)
    dan = (a - n).pow(2).sum(-1)
    return F.relu(dap - dan + margin).mean()
