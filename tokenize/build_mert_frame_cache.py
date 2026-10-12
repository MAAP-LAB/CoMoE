#!/usr/bin/env python3
"""Build the MERT-v0 frame cache used to fit the k-means codebooks.

Samples --n_clips_per_source clips per source (balanced; a source with fewer
clips contributes all of them) from the metadata CSV, runs MERT-v0-public with
output_hidden_states and stores the full (13, T, 768) fp16 stack per clip.

This reproduces the sampling of the original analysis script: with the default
seed and 50 clips per source over all ten sources the cache holds 477 clips.
Pass ``--sources fma_medium jamendo`` to build the real-only cache (100 clips)
behind assets/mert_kmeans_centroids_REAL_ONLY_L0_L1_L11_L12.npz.

Usage:
  python tokenize/build_mert_frame_cache.py                      # -> $COMOE_DATA_ROOT/mert_frame_cache
  python tokenize/build_mert_frame_cache.py --sources fma_medium jamendo --cache_dir $COMOE_DATA_ROOT/mert_frame_cache_real
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from comoe.paths import REPO_ROOT, data_root, resolve_audio_path  # noqa: E402

MERT_SR = 16_000
MERT_SEC = 30
MODEL_NAME = "m-a-p/MERT-v0-public"
MIN_CLIPS_PER_SOURCE = 20

csv.field_size_limit(sys.maxsize)


def load_wav(p: Path):
    import torch  # noqa: F401
    import torchaudio
    try:
        info = torchaudio.info(str(p))
        n = int(info.sample_rate * MERT_SEC) + info.sample_rate
        wav, sr = torchaudio.load(str(p), num_frames=n)
    except Exception:
        try:
            wav, sr = torchaudio.load(str(p))
        except Exception:
            return None
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != MERT_SR:
        wav = torchaudio.functional.resample(wav, sr, MERT_SR)
    wav = wav.squeeze(0)[: MERT_SR * MERT_SEC].contiguous().float().numpy()
    if wav.shape[-1] < MERT_SR // 10:
        return None
    return wav


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", default=str(REPO_ROOT / "data" / "splits" / "metadata_mom_norm.csv"))
    ap.add_argument("--cache_dir", default=str(data_root() / "mert_frame_cache"))
    ap.add_argument("--n_clips_per_source", type=int, default=50)
    ap.add_argument("--sources", nargs="*", default=None,
                    help="restrict to these sources (default: every source with >= 20 clips)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    random.seed(args.seed)
    rows = list(csv.DictReader(open(args.meta_csv)))
    by_src: dict[str, list[str]] = {}
    fp_by_uid: dict[str, str] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r["utt_id"])
        fp_by_uid[r["utt_id"]] = r["filepath"]

    sample_by_src: dict[str, list[str]] = {}
    for src, uids in by_src.items():
        if args.sources is not None and src not in args.sources:
            continue
        if len(uids) < MIN_CLIPS_PER_SOURCE:
            print(f"[skip] {src}: only {len(uids)} clips")
            continue
        sample_by_src[src] = uids if len(uids) <= args.n_clips_per_source \
            else random.sample(uids, args.n_clips_per_source)
    n_total = sum(len(v) for v in sample_by_src.values())
    print(f"[plan] {len(sample_by_src)} sources x {args.n_clips_per_source} clips = {n_total} clips total")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [(src, uid) for src, uids in sample_by_src.items() for uid in uids
            if not (cache_dir / f"{uid}.npy").exists()]
    print(f"[mert] {len(todo)} clips to extract", flush=True)
    if not todo:
        return 0

    import torch
    from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    cfg.conv_pos_batch_norm = False  # newer transformers expect the field; MERTConfig lacks it
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, config=cfg, trust_remote_code=True).to(args.device).eval()

    t0 = time.time()
    n_done = 0
    for start in range(0, len(todo), args.batch_size):
        batch = todo[start:start + args.batch_size]
        wavs, uids = [], []
        for _, uid in batch:
            w = load_wav(resolve_audio_path(fp_by_uid[uid]))
            if w is None:
                continue
            wavs.append(w)
            uids.append(uid)
        if not wavs:
            continue
        inputs = fe(wavs, sampling_rate=MERT_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                                    enabled=str(args.device).startswith("cuda")):
            out = model(**inputs, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=1).half().cpu().numpy()  # (B, 13, T, 768)
        for i, uid in enumerate(uids):
            np.save(cache_dir / f"{uid}.npy", hs[i].astype(np.float16))
            n_done += 1
        if n_done % 50 == 0 or n_done == len(todo):
            rate = n_done / max(time.time() - t0, 1e-6)
            print(f"[mert] {n_done}/{len(todo)}  {rate:.1f} clips/s", flush=True)
    print(f"[done] cache at {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
