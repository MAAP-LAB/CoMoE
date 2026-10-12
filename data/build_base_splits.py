"""Build the base train/val/OOD splits of MoM-open (CLAM protocol).

The shipped data/splits/split_mom_{train,val,ood}.json are the output of this
script (seed 42).  The original run also dropped clips whose EnCodec/DAC
tokens had failed to extract (corrupted FMA mp3s); pass --require_tokens with
the token directories to reproduce that filter.

Usage:  python data/build_base_splits.py [--require_tokens DIR ...]

Mirrors CLAM Table 4 protocol:
  - Train+Val: 47,911 fake (subsample from 53,563 train_fake) + 47,911 real
               (proportional Jamendo/FMA from real_pool). 10% val carved
               stratified by source.
  - OOD test:  15,505 fake (all of test_fake = Riffusion+Yue+Suno v3) +
               15,505 real (fresh, no overlap with train/val).

Subsampling rule: keep all minority fake sources (Udio, Diffrythm, Suno v2)
intact; downsample only Suno 3.5 to hit 47,911 total. This preserves the
small classes that matter for cross-family generalization.

Output (each file: {"fake": [utt_ids], "real": [utt_ids]}):
  data/musiccodecfake/split_mom_train.json
  data/musiccodecfake/split_mom_val.json
  data/musiccodecfake/split_mom_ood.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `comoe` imports without pip install
from comoe.paths import REPO_ROOT  # noqa: E402

ROOT = REPO_ROOT / "data" / "splits"
META = ROOT / "metadata_mom_raw.csv"
OUT_TRAIN = ROOT / "split_mom_train.json"
OUT_VAL = ROOT / "split_mom_val.json"
OUT_OOD = ROOT / "split_mom_ood.json"

SEED = 42
TRAIN_FAKE_TARGET = 47_911   # CLAM Table 4
VAL_FRAC = 0.10


def stratified_sample(rng, by_key: dict[str, list[str]], n: int) -> list[str]:
    """Proportional sample of size n across keys (preserves source mix)."""
    total = sum(len(v) for v in by_key.values())
    out: list[str] = []
    keys = sorted(by_key)
    # Compute per-key target; assign remainder to largest pool to hit n exactly.
    targets: dict[str, int] = {}
    for k in keys:
        targets[k] = int(round(len(by_key[k]) / total * n))
    # Adjust for rounding drift.
    drift = n - sum(targets.values())
    largest = max(keys, key=lambda k: len(by_key[k]))
    targets[largest] += drift
    for k in keys:
        pool = sorted(by_key[k])
        rng.shuffle(pool)
        out.extend(pool[: targets[k]])
    return out


def stratified_carve_val(rng, ids: list[str], src_of: dict[str, str], frac: float) -> tuple[list[str], list[str]]:
    """Stratified val carve by source. Returns (train, val)."""
    by_src: dict[str, list[str]] = defaultdict(list)
    for u in ids:
        by_src[src_of[u]].append(u)
    train, val = [], []
    for src in sorted(by_src):
        pool = sorted(by_src[src])
        rng.shuffle(pool)
        n_val = int(round(len(pool) * frac))
        val.extend(pool[:n_val])
        train.extend(pool[n_val:])
    return sorted(train), sorted(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require_tokens", nargs="*", default=[], type=Path,
                    help="drop clips without <dir>/<utt_id>.npy in every listed directory")
    args = ap.parse_args()

    rng = random.Random(SEED)
    rows = list(csv.DictReader(META.open()))

    # Drop utt_ids whose tokens failed to extract (FMA-medium corrupted mp3s).
    n_before = len(rows)
    rows = [
        r for r in rows
        if all((d / f"{r['utt_id']}.npy").exists() for d in args.require_tokens)
    ]
    print(f"[filter] dropped {n_before - len(rows)} utt_ids with missing tokens")

    src_of = {r["utt_id"]: r["source"] for r in rows}

    # ---- partition by split ----
    train_fake_by_src: dict[str, list[str]] = defaultdict(list)
    test_fake: list[str] = []
    real_pool_by_src: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["split"] == "train_fake":
            train_fake_by_src[r["source"]].append(r["utt_id"])
        elif r["split"] == "test_fake":
            test_fake.append(r["utt_id"])
        elif r["split"] == "real_pool":
            real_pool_by_src[r["source"]].append(r["utt_id"])

    print("[input]")
    for s, ids in sorted(train_fake_by_src.items()):
        print(f"  train_fake/{s}: {len(ids)}")
    print(f"  test_fake: {len(test_fake)}")
    for s, ids in sorted(real_pool_by_src.items()):
        print(f"  real_pool/{s}: {len(ids)}")

    # ---- TRAIN+VAL fake: keep minorities, downsample Suno 3.5 ----
    minor = ["udio", "diffrythm", "suno_v2"]
    minor_ids = sorted(uid for s in minor for uid in train_fake_by_src[s])
    n_remaining = TRAIN_FAKE_TARGET - len(minor_ids)
    suno35 = sorted(train_fake_by_src["suno_v3.5"])
    rng.shuffle(suno35)
    train_fake_full = sorted(minor_ids + suno35[:n_remaining])
    assert len(train_fake_full) == TRAIN_FAKE_TARGET, f"got {len(train_fake_full)}"
    print(f"[train+val fake] {len(train_fake_full)} (kept {len(minor_ids)} minority, "
          f"sampled {n_remaining} from suno_v3.5)")

    # ---- TRAIN+VAL real: proportional Jamendo/FMA, 47,911 total ----
    train_real_full = stratified_sample(rng, real_pool_by_src, TRAIN_FAKE_TARGET)
    assert len(train_real_full) == TRAIN_FAKE_TARGET
    train_real_set = set(train_real_full)
    print(f"[train+val real] {len(train_real_full)}")

    # ---- carve 10% val (stratified by source within each label) ----
    train_fake, val_fake = stratified_carve_val(rng, train_fake_full, src_of, VAL_FRAC)
    train_real, val_real = stratified_carve_val(rng, train_real_full, src_of, VAL_FRAC)
    print(f"[train] fake={len(train_fake)}  real={len(train_real)}")
    print(f"[val]   fake={len(val_fake)}  real={len(val_real)}")

    # ---- OOD: all test_fake + matched real count, no overlap w/ train+val ----
    ood_fake = sorted(test_fake)
    remaining_real_by_src = {
        s: sorted(set(real_pool_by_src[s]) - train_real_set)
        for s in real_pool_by_src
    }
    ood_real = stratified_sample(rng, remaining_real_by_src, len(ood_fake))
    assert not (set(ood_real) & train_real_set), "OOD real overlaps train!"
    assert not (set(ood_fake) & set(train_fake) & set(val_fake)), "OOD fake overlap!"
    print(f"[ood]   fake={len(ood_fake)}  real={len(ood_real)}")

    # ---- per-source sanity ----
    def by_src_count(ids):
        c: dict[str, int] = defaultdict(int)
        for u in ids:
            c[src_of[u]] += 1
        return dict(sorted(c.items()))

    print("\n[train fake by source]", by_src_count(train_fake))
    print("[train real by source]", by_src_count(train_real))
    print("[val fake by source]  ", by_src_count(val_fake))
    print("[val real by source]  ", by_src_count(val_real))
    print("[ood fake by source]  ", by_src_count(ood_fake))
    print("[ood real by source]  ", by_src_count(ood_real))

    # ---- write ----
    OUT_TRAIN.write_text(json.dumps({"fake": train_fake, "real": train_real}, indent=2))
    OUT_VAL.write_text(json.dumps({"fake": val_fake, "real": val_real}, indent=2))
    OUT_OOD.write_text(json.dumps({"fake": ood_fake, "real": ood_real}, indent=2))
    print(f"\nWrote {OUT_TRAIN}")
    print(f"Wrote {OUT_VAL}")
    print(f"Wrote {OUT_OOD}")


if __name__ == "__main__":
    main()
