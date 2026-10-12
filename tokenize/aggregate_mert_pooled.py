"""Aggregate per-clip pooled MERT-v0 features into one mert_v0_pooled.npz.

Input:  $COMOE_DATA_ROOT/feats_mom_norm/mert_v0/<utt_id>.npy   (13, 768) fp16
Output: $COMOE_DATA_ROOT/feats_mom_norm/mert_v0_pooled.npz   (utt_ids, embeds, labels, sources)

Usage:  python tokenize/aggregate_mert_pooled.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `comoe` imports without pip install
from comoe.paths import REPO_ROOT, data_root  # noqa: E402

csv.field_size_limit(sys.maxsize)

ROOT = data_root() / "feats_mom_norm"
META = REPO_ROOT / "data" / "splits" / "metadata_mom_norm.csv"

IN_DIR = ROOT / "mert_v0"
OUT_PATH = ROOT / "mert_v0_pooled.npz"
SHAPE = (13, 768)
DTYPE = np.float16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-csv", type=str, default=None)
    ap.add_argument("--out-root", type=str, default=None)
    args = ap.parse_args()
    global META, ROOT, IN_DIR, OUT_PATH
    if args.meta_csv:
        META = Path(args.meta_csv)
    if args.out_root:
        ROOT = Path(args.out_root)
        IN_DIR = ROOT / "mert_v0"
        OUT_PATH = ROOT / "mert_v0_pooled.npz"
    print(f"[cfg] META={META}  IN_DIR={IN_DIR}  OUT={OUT_PATH}")

    rows = list(csv.DictReader(META.open()))
    keep = [r for r in rows if (IN_DIR / f"{r['utt_id']}.npy").exists()]
    print(f"[mert_v0] {len(keep)}/{len(rows)} utt_ids have features")

    embeds = np.zeros((len(keep),) + SHAPE, dtype=DTYPE)
    utt_ids, labels, sources = [], [], []
    for i, r in enumerate(keep):
        embeds[i] = np.load(IN_DIR / f"{r['utt_id']}.npy").astype(DTYPE)
        utt_ids.append(r["utt_id"])
        labels.append(1 if r["label"] == "fake" else 0)
        sources.append(r["source"])
        if (i + 1) % 20000 == 0:
            print(f"  loaded {i+1}/{len(keep)}")

    np.savez_compressed(
        OUT_PATH,
        utt_ids=np.asarray(utt_ids),
        embeds=embeds,
        labels=np.asarray(labels, dtype=np.int8),
        sources=np.asarray(sources),
    )
    sz = OUT_PATH.stat().st_size / (1024**3)
    print(f"[mert_v0] wrote {OUT_PATH}  shape={embeds.shape}  dtype={DTYPE}  size={sz:.2f} GB")


if __name__ == "__main__":
    main()
