"""Build the real-source-restricted splits (Real-FMA / Real-Jamendo) on top of the base splits.

Usage:  python data/build_real_loso_splits.py

Produces two variants (A/B):
  A) train_real=FMA-only,     val_real=Jamendo-only
  B) train_real=Jamendo-only, val_real=FMA-only

Fake splits (train/val/ood) and OOD real are left untouched.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `comoe` imports without pip install
from comoe.paths import REPO_ROOT  # noqa: E402

SPLITS = REPO_ROOT / "data" / "splits"
META = SPLITS / "metadata_mom_norm.csv"
IN = {
    "train": SPLITS / "split_mom_train_norm.json",
    "val":   SPLITS / "split_mom_val_norm.json",
    "ood":   SPLITS / "split_mom_ood_norm.json",
}

rows = {r["utt_id"]: r for r in csv.DictReader(META.open())}
src_of = lambda u: rows[u]["source"]

def filter_real(uids, keep_source):
    return [u for u in uids if u in rows and (src_of(u) == keep_source or rows[u]["label"] != "real")]

for tag, (train_src, val_src) in [
    ("losoA_fma2jam",   ("fma_medium", "jamendo")),
    ("losoB_jam2fma",   ("jamendo",   "fma_medium")),
]:
    for split_name, path in IN.items():
        d = json.load(path.open())
        if split_name == "train":
            d = {"real": [u for u in d["real"] if src_of(u) == train_src],
                 "fake": list(d["fake"])}
        elif split_name == "val":
            d = {"real": [u for u in d["real"] if src_of(u) == val_src],
                 "fake": list(d["fake"])}
        # ood untouched
        out = SPLITS / f"split_mom_{split_name}_norm_{tag}.json"
        out.write_text(json.dumps(d, indent=2))
        print(f"[{tag}] {split_name}: real={len(d['real'])} fake={len(d['fake'])} → {out}")
    print()
