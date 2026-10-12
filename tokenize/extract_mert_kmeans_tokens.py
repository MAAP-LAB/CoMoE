#!/usr/bin/env python3
"""Extract MERT k-means tokens: MERT-v0 frames -> nearest centroid per layer.

Usage:  python tokenize/extract_mert_kmeans_tokens.py [--centroids assets/...REAL_ONLY...npz --out_dir ...]

For each clip:
  1. Load 30s waveform @ 16 kHz
  2. MERT-v0 forward, output_hidden_states=True -> (13, T, 768)
  3. Pick layers [0, 1, 11, 12] -> (4, T, 768)
  4. K-means assign each layer using saved centroids -> (4, T) int16
  5. Save to out_dir/{utt_id}.npy

Sharded by hashing utt_id -> (n_shards, shard_idx).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `comoe` imports without pip install
from comoe.paths import REPO_ROOT, data_root, resolve_audio_path  # noqa: E402

LAYERS = [0, 1, 11, 12]
K = 1024
MERT_SR = 16_000
MERT_SEC = 30
MODEL_NAME = "m-a-p/MERT-v0-public"

csv.field_size_limit(sys.maxsize)


def assign_kmeans(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Vectorised nearest-centroid assignment.
    X: (T, D) float32, centroids: (K, D) float32 -> (T,) int64
    """
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c
    x2 = (X * X).sum(axis=1, keepdims=True)        # (T, 1)
    c2 = (centroids * centroids).sum(axis=1)        # (K,)
    d = x2 + c2[None, :] - 2.0 * X @ centroids.T   # (T, K)
    return d.argmin(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", default=str(REPO_ROOT / "data" / "splits" / "metadata_mom_norm.csv"))
    ap.add_argument("--centroids",
                    default=str(REPO_ROOT / "assets" / "mert_kmeans_centroids_L0_L1_L11_L12.npz"))
    ap.add_argument("--out_dir",
                    default=str(data_root() / "tokens_mom_norm" / "mert_v0_kmeans1024_L0_L1_L11_L12"))
    ap.add_argument("--gpu_idx", type=int, default=0)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)

    import torch
    import torchaudio
    from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load centroids
    cd = np.load(args.centroids)
    cents = [cd[f"L{L}"].astype(np.float32) for L in LAYERS]  # list of (K, 768)
    print(f"[shard {args.shard_idx}/{args.n_shards}] centroids loaded: "
          f"{[c.shape for c in cents]}", flush=True)

    # Plan workload
    rows = list(csv.DictReader(open(args.meta_csv)))
    todo = []
    for r in rows:
        uid = r["utt_id"]
        h = int(hashlib.md5(uid.encode()).hexdigest(), 16) % args.n_shards
        if h != args.shard_idx:
            continue
        if (out_dir / f"{uid}.npy").exists():
            continue
        todo.append((uid, str(resolve_audio_path(r["filepath"]))))
    print(f"[shard {args.shard_idx}] {len(todo)} clips to extract "
          f"(of {len(rows)} total)", flush=True)
    if not todo:
        print("[shard] nothing to do")
        return

    # Load MERT
    print(f"[shard {args.shard_idx}] loading {MODEL_NAME}", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    cfg.conv_pos_batch_norm = False
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, config=cfg, trust_remote_code=True
    ).to("cuda").eval()

    def load_wav(p: Path):
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

    t0 = time.time()
    n_done = 0
    n_fail = 0
    for batch_start in range(0, len(todo), args.batch_size):
        batch = todo[batch_start: batch_start + args.batch_size]
        wavs, uids_b = [], []
        for uid, fp in batch:
            w = load_wav(Path(fp))
            if w is None:
                n_fail += 1
                continue
            wavs.append(w)
            uids_b.append(uid)
        if not wavs:
            continue
        inputs = fe(wavs, sampling_rate=MERT_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of 13 tensors (B, T, 768)
        for i, uid in enumerate(uids_b):
            tokens = np.zeros((len(LAYERS), out.hidden_states[0].shape[1]), dtype=np.int16)
            for L_idx, L in enumerate(LAYERS):
                X = out.hidden_states[L][i].float().cpu().numpy()  # (T, 768)
                tok = assign_kmeans(X, cents[L_idx])
                tokens[L_idx, :tok.size] = tok.astype(np.int16)
            np.save(out_dir / f"{uid}.npy", tokens)
            n_done += 1
        if n_done % args.log_every == 0 or n_done == len(todo):
            rate = n_done / max(time.time() - t0, 1e-6)
            eta = (len(todo) - n_done) / max(rate, 1e-6)
            print(f"[shard {args.shard_idx}] {n_done}/{len(todo)}  "
                  f"{rate:.2f} clips/s  eta={eta/60:.1f}min  fail={n_fail}", flush=True)
    print(f"[shard {args.shard_idx}] done. n_done={n_done} fail={n_fail}")


if __name__ == "__main__":
    main()
