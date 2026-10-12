"""MoM-open dataset for CoMoE.

  - Tokens: one ``<token_dir>/<utt_id>.npy`` per clip, int16 ``(L, T)`` for
    discrete token spaces or fp16 ``(L, T, D)`` for the MERT-continuous ablation.
  - Split file: ``{"fake": [utt_ids], "real": [utt_ids]}``.
  - Label comes from split membership, source from the metadata CSV.
  - Optional semantic stream: aggregated ``.npz`` (``mert_v0_pooled.npz``) or
    per-clip ``.npy`` under ``semantic_dir`` (only used by the MLP baseline and
    the HLS ablation).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

csv.field_size_limit(sys.maxsize)


class MoMCoMoEDataset(Dataset):
    def __init__(
        self,
        token_dir: str,
        meta_csv: str,
        split_file: str,
        semantic_npz: Optional[str] = None,
        semantic_dir: Optional[str] = None,
        max_frames: int = 860,
        crop_frames: Optional[int] = 860,
        crop_random: bool = False,
        continuous: bool = False,
    ):
        self.token_dir = Path(token_dir)
        self.max_frames = max_frames
        self.crop_frames = crop_frames
        self.crop_random = crop_random
        self.continuous = continuous

        with open(split_file) as f:
            sp = json.load(f)
        items: list[tuple[str, int]] = [(u, 1) for u in sp["fake"]]
        items += [(u, 0) for u in sp["real"]]

        self.src_of: dict[str, str] = {}
        with open(meta_csv) as f:
            for r in csv.DictReader(f):
                self.src_of[r["utt_id"]] = r["source"]

        n0 = len(items)
        items = [(u, l) for u, l in items if (self.token_dir / f"{u}.npy").exists()]
        if len(items) < n0:
            print(f"[MoM ds] dropped {n0 - len(items)} utts missing tokens in {self.token_dir.name}")

        self.sem_dict: Optional[dict[str, np.ndarray]] = None
        self.semantic_dir: Optional[Path] = None
        if semantic_npz and Path(semantic_npz).exists():
            sd = np.load(semantic_npz, allow_pickle=True)
            embeds = sd["embeds"]  # NpzFile[key] re-reads each call — load once.
            utt_ids = sd["utt_ids"]
            self.sem_dict = {str(u): embeds[i] for i, u in enumerate(utt_ids)}
            n1 = len(items)
            items = [(u, l) for u, l in items if u in self.sem_dict]
            if len(items) < n1:
                print(f"[MoM ds] dropped {n1 - len(items)} utts missing semantic in {Path(semantic_npz).name}")
        elif semantic_dir and Path(semantic_dir).exists():
            self.semantic_dir = Path(semantic_dir)
            n1 = len(items)
            items = [(u, l) for u, l in items if (self.semantic_dir / f"{u}.npy").exists()]
            if len(items) < n1:
                print(f"[MoM ds] dropped {n1 - len(items)} utts missing semantic .npy in {self.semantic_dir.name}")

        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def _crop_pad(self, t: np.ndarray, length: int) -> tuple[np.ndarray, int]:
        P = self.crop_frames if self.crop_frames is not None else self.max_frames
        if t.shape[1] > P:
            if self.crop_random and length > P:
                start = int(np.random.randint(0, length - P + 1))
            else:
                start = max(0, (length - P) // 2)
            t = t[:, start:start + P]
            length = min(length - start, P)
        if t.shape[1] < P:
            pad_shape = list(t.shape)
            pad_shape[1] = P - t.shape[1]
            pad = np.zeros(pad_shape, dtype=t.dtype)
            t = np.concatenate([t, pad], axis=1)
            # length unchanged (padded region is invalid)
        return t, length

    def __getitem__(self, i):
        uid, label = self.items[i]
        source = self.src_of.get(uid, "unknown")

        t = np.load(self.token_dir / f"{uid}.npy")  # int16 (L,T) or fp16 (L,T,D)
        length = int(t.shape[1])
        t, length = self._crop_pad(t, length)

        if self.continuous:
            tokens_tensor = torch.from_numpy(t.astype(np.float32))  # (L, T, D)
        else:
            tokens_tensor = torch.from_numpy(t.astype(np.int64))    # (L, T)

        item = {
            "tokens": tokens_tensor,
            "length": length,
            "label": label,
            "utt_id": uid,
            "source": source,
        }
        if self.sem_dict is not None:
            item["mert"] = torch.from_numpy(self.sem_dict[uid].astype(np.float32))
        elif self.semantic_dir is not None:
            sem = np.load(self.semantic_dir / f"{uid}.npy")
            item["mert"] = torch.from_numpy(sem.astype(np.float32))
        return item


from typing import Dict, List  # noqa: E402


def collate_fn(batch: List[Dict]):
    tokens = torch.stack([b["tokens"] for b in batch], dim=0)   # (B, L, T)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    out = {
        "tokens": tokens,
        "lengths": lengths,
        "labels": labels,
        "utt_ids": [b["utt_id"] for b in batch],
        "sources": [b["source"] for b in batch],
    }
    if "mert" in batch[0]:
        out["mert"] = torch.stack([b["mert"] for b in batch], dim=0)
    if "latent" in batch[0]:
        out["latent"] = torch.stack([b["latent"] for b in batch], dim=0)
        out["latent_lengths"] = torch.tensor(
            [b["latent_length"] for b in batch], dtype=torch.long
        )
    return out
