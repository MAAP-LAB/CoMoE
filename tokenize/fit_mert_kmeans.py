#!/usr/bin/env python3
"""Fit MiniBatch k-means (K=1024) on MERT-v0 frames for layers [0, 1, 11, 12].

Reads the frame cache written by tokenize/build_mert_frame_cache.py (one
(13, T, 768) fp16 array per clip), samples --fit_frames_per_clip frames per clip
and layer, fits one codebook per layer and saves the centroids.

The shipped assets/mert_kmeans_centroids_L0_L1_L11_L12.npz was fit on a
balanced sample of 50 clips per source (477 clips, all sources including
held-out generators); assets/mert_kmeans_centroids_REAL_ONLY_L0_L1_L11_L12.npz
on 50 clips from each of the two real corpora only (100 clips).

Usage:  python tokenize/fit_mert_kmeans.py --cache_dir $COMOE_DATA_ROOT/mert_frame_cache --out assets/<name>.npz
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `comoe` imports without pip install
from comoe.paths import REPO_ROOT, data_root  # noqa: E402

LAYERS = [0, 1, 11, 12]
K = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=str(data_root() / "mert_frame_cache"))
    ap.add_argument("--fit_frames_per_clip", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(REPO_ROOT / "assets" / "mert_kmeans_centroids_L0_L1_L11_L12.npz"))
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    files = sorted(cache_dir.glob("*.npy"))
    print(f"[cache] {len(files)} frame-level npys")
    if not files:
        raise SystemExit("empty cache")

    centroids = {}
    for L_idx, L in enumerate(LAYERS):
        rng = np.random.default_rng(args.seed + L)
        buf = []
        for f in files:
            arr = np.load(f, mmap_mode="r")  # (13, T, 768)
            T = arr.shape[1]
            n = min(args.fit_frames_per_clip, T)
            idx = rng.choice(T, size=n, replace=False)
            buf.append(arr[L, idx, :].astype(np.float32))
        X = np.concatenate(buf, axis=0)
        print(f"[L{L}] fit on {X.shape[0]:,} frames  dim={X.shape[1]}", flush=True)
        t0 = time.time()
        km = MiniBatchKMeans(
            n_clusters=K, batch_size=8192, n_init=3, max_iter=100,
            reassignment_ratio=0.005, random_state=args.seed + L,
        ).fit(X)
        dt = time.time() - t0
        print(f"[L{L}] kmeans done  inertia={km.inertia_:.0f}  {dt:.1f}s", flush=True)
        centroids[f"L{L}"] = km.cluster_centers_.astype(np.float32)
        del X, buf, km

    np.savez(args.out, **centroids,
             layers=np.array(LAYERS, dtype=np.int32),
             K=np.array([K], dtype=np.int32))
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
