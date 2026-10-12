"""MoM-open raw-audio dataset for the CLAM baseline.

Returns dual-rate waveforms (24 kHz for MERT, 16 kHz for Wav2Vec2),
center-cropped/padded to `clip_sec` seconds.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from comoe.paths import resolve_audio_path

csv.field_size_limit(sys.maxsize)


class MoMCLAMDataset(Dataset):
    def __init__(
        self,
        meta_csv: str,
        split_file: str,
        clip_sec: float = 30.0,
        sr_mert: int = 24000,
        sr_w2v2: int = 16000,
        random_crop: bool = False,
        max_per_class: int = 0,
        cache_dir: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.clip_sec = float(clip_sec)
        self.sr_mert = int(sr_mert)
        self.sr_w2v2 = int(sr_w2v2)
        self.random_crop = random_crop
        self.n_mert = int(self.sr_mert * self.clip_sec)
        self.n_w2v2 = int(self.sr_w2v2 * self.clip_sec)

        with open(split_file) as f:
            sp = json.load(f)
        real, fake = sp["real"], sp["fake"]
        if max_per_class > 0:
            real, fake = real[:max_per_class], fake[:max_per_class]

        meta: dict[str, dict] = {}
        with open(meta_csv) as f:
            for r in csv.DictReader(f):
                meta[r["utt_id"]] = r

        self.items: list[tuple[str, str, int, str]] = []
        for u in real:
            if u in meta:
                self.items.append((u, str(resolve_audio_path(meta[u]["filepath"])), 0, meta[u]["source"]))
        for u in fake:
            if u in meta:
                self.items.append((u, str(resolve_audio_path(meta[u]["filepath"])), 1, meta[u]["source"]))

    def __len__(self) -> int:
        return len(self.items)

    def _load_cached(self, uid: str, n_samples: int, crop_offset: Optional[float]) -> torch.Tensor:
        cache_path = self.cache_dir / uid[:2] / f"{uid}.npy"
        arr = np.load(cache_path, mmap_mode="r")
        if arr.size >= n_samples:
            if crop_offset is None:
                start = (arr.size - n_samples) // 2
            else:
                start = int(crop_offset * (arr.size - n_samples))
            arr = np.array(arr[start:start + n_samples])
        else:
            arr = np.pad(np.array(arr), (0, n_samples - arr.size))
        return torch.from_numpy(arr.astype(np.float32) / 32768.0)

    def _load(self, path: str, target_sr: int, n_samples: int, crop_offset: Optional[float] = None):
        try:
            wav, sr = torchaudio.load(path)
        except Exception:
            return torch.zeros(n_samples)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav = wav.squeeze(0)
        if wav.numel() >= n_samples:
            if crop_offset is None:
                start = (wav.numel() - n_samples) // 2
            else:
                # crop_offset given as a fraction in [0, 1] for shared crop across rates.
                start = int(crop_offset * (wav.numel() - n_samples))
            wav = wav[start:start + n_samples]
        else:
            wav = F.pad(wav, (0, n_samples - wav.numel()))
        return wav.contiguous()

    def __getitem__(self, i):
        uid, fp, label, source = self.items[i]
        if self.random_crop:
            offset = float(np.random.rand())
        else:
            offset = None
        # Decode audio once at MERT rate (24 kHz); Wav2Vec2's 16 kHz view is
        # produced by GPU-side resampling in the training loop to halve decode cost.
        if self.cache_dir is not None:
            wav24 = self._load_cached(uid, self.n_mert, offset)
        else:
            wav24 = self._load(fp, self.sr_mert, self.n_mert, offset)
        return {
            "wav_24k": wav24,
            "label": label,
            "utt_id": uid,
            "source": source,
        }
