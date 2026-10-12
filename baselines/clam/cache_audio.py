"""Pre-decode MoM-open audio to a per-clip int16 .npy cache for faster CLAM training (optional).

Usage:  python -m baselines.clam.cache_audio --split_tag fakeA_sunov35 --cache_dir $COMOE_DATA_ROOT/clam_cache_24k_30s

Cache layout (sharded by 2-char prefix to avoid huge directories):
  <cache_dir>/<prefix>/<utt_id>.npy  shape (n_samples,) int16

n_samples = sr * clip_sec  (default 24000 * 30 = 720000 → ~1.44 MB per utt).

Multi-process (joblib). Skips already-cached utts. Total disk for full MoM:
  127K utts × 1.44 MB ≈ 180 GB.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from joblib import Parallel, delayed

from comoe.paths import REPO_ROOT, data_root, resolve_audio_path

csv.field_size_limit(sys.maxsize)


def cache_one(uid: str, fp: str, out_dir: Path, sr: int, n_samples: int) -> str:
    out = out_dir / uid[:2] / f"{uid}.npy"
    if out.exists():
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        wav, file_sr = torchaudio.load(fp)
    except Exception as e:
        return f"err:{e.__class__.__name__}"
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    wav = wav.squeeze(0)
    if wav.numel() >= n_samples:
        start = (wav.numel() - n_samples) // 2
        wav = wav[start:start + n_samples]
    else:
        wav = torch.nn.functional.pad(wav, (0, n_samples - wav.numel()))
    arr = (wav.clamp(-1.0, 1.0).numpy() * 32767).astype(np.int16)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.rename(tmp, out)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=str, default=str(REPO_ROOT / "data/splits/metadata_mom_norm.csv"))
    ap.add_argument("--split_dir", type=str, default=str(REPO_ROOT / "data/splits"))
    ap.add_argument("--split_tag", type=str, default="", help="same meaning as in baselines.clam.train")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "ood"])
    ap.add_argument("--cache_dir", type=str, default=str(data_root() / "clam_cache_24k_30s"))
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--clip_sec", type=int, default=30)
    ap.add_argument("--n_jobs", type=int, default=24)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_samples = args.sr * args.clip_sec

    fp_of: dict[str, str] = {}
    with open(args.meta) as f:
        for r in csv.DictReader(f):
            fp_of[r["utt_id"]] = str(resolve_audio_path(r["filepath"]))

    uids: list[str] = []
    for sp_name in args.splits:
        suffix = f"_norm_{args.split_tag}" if args.split_tag else "_norm"
        sp = json.load(open(Path(args.split_dir) / f"split_mom_{sp_name}{suffix}.json"))
        uids.extend(sp["real"]); uids.extend(sp["fake"])
    uids = sorted(set(uids))
    print(f"[plan] {len(uids)} unique utts → {cache_dir} ({n_samples} samples each, int16)")

    todo = [(u, fp_of[u]) for u in uids if u in fp_of]
    print(f"[plan] {len(todo)} have filepaths, {len(uids) - len(todo)} missing")

    counts = {"ok": 0, "skip": 0, "err": 0}
    res = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(cache_one)(u, fp, cache_dir, args.sr, n_samples) for u, fp in todo
    )
    for r in res:
        if r == "ok": counts["ok"] += 1
        elif r == "skip": counts["skip"] += 1
        else: counts["err"] += 1
    print(f"[done] ok={counts['ok']} skip={counts['skip']} err={counts['err']}")


if __name__ == "__main__":
    main()
