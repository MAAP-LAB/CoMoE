"""Build the open-weight vs commercial (Fake-LOSO by licence) split JSONs.

Motivated by ICML feedback: Suno and Udio watermark their output, so a detector
trained on them might be reading the watermark rather than a generative
artefact.  Both existing single-family conditions (Fake-A = Suno-v3.5,
Fake-B = Udio) train on *commercial* audio, so the open-weight direction has
never been tested.

Two mirrored conditions plus a family-count control:

  Fake-Open   train fake = {DiffRhythm, YuE}      held-out = every commercial family
  Fake-Comm   train fake = {Suno-v3.5, Udio}      held-out = every open-weight family
  Fake-Comm1  train fake = {Suno-v3.5}            held-out = same as Fake-Comm

Fake-Comm1 holds N and the held-out target fixed against Fake-Comm and varies
only the number of training families, which is the comparison the paper's
existing splits cannot make: they are all single-family, so family count never
varies there.

The commercial side is downsampled to the open-weight pool size and to the same
number of families (2).  Without that, a weak Fake-Open result could not be told
apart from plain data starvation: the paper's own Fake-LOSO C (DiffRhythm only,
4,134 clips) collapses every token-based variant, and the open-weight pool is
only ~8.9k.  Matching N and family count is what makes the two directions
comparable.

Riffusion sits in the held-out pool of *both* conditions and in neither training
set: the name covers both the 2022 open Stable-Diffusion spectrogram model and
today's commercial service, and neither the MoM metadata nor its README says
which produced these clips.  Keeping it out of both training sets means it can
never decide the headline number; per_source_acc still reports it separately.

    python data/build_license_splits.py            # dry run
    python data/build_license_splits.py --write
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from comoe.paths import REPO_ROOT  # noqa: E402

BASE = REPO_ROOT / "data" / "splits"
META = BASE / "metadata_mom_norm.csv"
IN = {
    "train": BASE / "split_mom_train_norm.json",
    "val": BASE / "split_mom_val_norm.json",
    "ood": BASE / "split_mom_ood_norm.json",
}

OPEN_WEIGHT = ["diffrythm", "yue"]
COMMERCIAL = ["suno_v2", "suno_v3", "suno_v3.5", "suno_v4", "udio"]
COMM_TRAIN = ["suno_v3.5", "udio"]  # two families, mirroring the two open-weight ones
AMBIGUOUS = ["riffusion"]

# YuE currently lives entirely in the OOD pool, so it has to be re-partitioned
# before it can be trained on.  Same ratio the original splits use (~90/10).
YUE_VAL_FRAC = 0.10
SEED = 42


def load_splits():
    return {k: json.load(p.open()) for k, p in IN.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write files (default: dry run)")
    args = ap.parse_args()

    if not META.is_file():
        print(f"{META} not found")
        return 1

    src_of = {r["utt_id"]: r["source"] for r in csv.DictReader(META.open())}
    d = load_splits()

    def pick(split: str, side: str, source: str) -> list[str]:
        return [u for u in d[split][side] if src_of[u] == source]

    def pick_any(side: str, sources) -> list[str]:
        """All clips of these sources, across train+val+ood."""
        out = []
        for split in ("train", "val", "ood"):
            for s in sources:
                out += pick(split, side, s)
        return out

    rng = random.Random(SEED)

    # ---- open-weight training pool -------------------------------------
    # DiffRhythm keeps its existing train/val assignment; YuE is split fresh.
    yue_all = sorted(pick_any("fake", ["yue"]))
    rng.shuffle(yue_all)
    n_yue_val = round(len(yue_all) * YUE_VAL_FRAC)
    yue_val, yue_train = yue_all[:n_yue_val], yue_all[n_yue_val:]

    open_train = sorted(pick("train", "fake", "diffrythm") + yue_train)
    open_val = sorted(pick("val", "fake", "diffrythm") + yue_val)

    # ---- commercial training pool, matched to the open-weight size ------
    def sample_matched(split: str, target: int) -> list[str]:
        """Take `target` clips, split evenly over the two commercial families."""
        per = target // len(COMM_TRAIN)
        out, short = [], 0
        for i, s in enumerate(COMM_TRAIN):
            pool = sorted(pick(split, "fake", s))
            want = per + (target - per * len(COMM_TRAIN) if i == 0 else 0) + short
            if len(pool) < want:  # family too small -- borrow from the next one
                short = want - len(pool)
                want = len(pool)
            else:
                short = 0
            out += random.Random(SEED + i).sample(pool, want)
        return sorted(out)

    comm_train = sample_matched("train", len(open_train))
    comm_val = sample_matched("val", len(open_val))

    def sample_single(split: str, target: int, source: str) -> list[str]:
        """`target` clips from one family, for the 1-vs-2-family control."""
        pool = sorted(pick(split, "fake", source))
        return sorted(random.Random(SEED).sample(pool, min(target, len(pool))))

    # Same N and same held-out target as Fake-Comm, but a single family in
    # training.  Only Suno-v3.5 is large enough to reach N on its own, so the
    # control exists on the commercial side only.
    comm1_train = sample_single("train", len(open_train), "suno_v3.5")
    comm1_val = sample_single("val", len(open_val), "suno_v3.5")

    # ---- held-out pools -------------------------------------------------
    riff = sorted(pick_any("fake", AMBIGUOUS))
    all_commercial = sorted(pick_any("fake", COMMERCIAL))
    all_open = sorted(pick_any("fake", OPEN_WEIGHT))

    conditions = {
        # train on open weights, meet every commercial generator at test time
        "fakeOpen_ow": {
            "train": {"real": list(d["train"]["real"]), "fake": open_train},
            "val": {"real": list(d["val"]["real"]), "fake": open_val},
            "ood": {"real": list(d["ood"]["real"]), "fake": all_commercial + riff},
        },
        # the mirror: train on commercial at matched N, meet open weights
        "fakeComm_matched": {
            "train": {"real": list(d["train"]["real"]), "fake": comm_train},
            "val": {"real": list(d["val"]["real"]), "fake": comm_val},
            "ood": {"real": list(d["ood"]["real"]), "fake": all_open + riff},
        },
        # 1-family control: identical N and held-out target to Fake-Comm, so
        # the only thing that changes is how many families training saw.
        "fakeComm1_suno": {
            "train": {"real": list(d["train"]["real"]), "fake": comm1_train},
            "val": {"real": list(d["val"]["real"]), "fake": comm1_val},
            "ood": {"real": list(d["ood"]["real"]), "fake": all_open + riff},
        },
    }

    # ---- report + sanity checks ----------------------------------------
    ok = True
    for tag, splits in conditions.items():
        print(f"=== {tag} ===")
        for split, payload in splits.items():
            by_src = {}
            for u in payload["fake"]:
                by_src[src_of[u]] = by_src.get(src_of[u], 0) + 1
            print(
                f"  {split:<5} real={len(payload['real']):>6} fake={len(payload['fake']):>6}"
                f"  {dict(sorted(by_src.items()))}"
            )
        tr, va, od = (set(splits[s]["fake"]) for s in ("train", "val", "ood"))
        for a, b, names in ((tr, va, "train/val"), (tr, od, "train/ood"), (va, od, "val/ood")):
            if a & b:
                print(f"  LEAK {names}: {len(a & b)} shared utt_ids")
                ok = False
        print()

    n_open, n_comm, n_c1 = len(open_train), len(comm_train), len(comm1_train)
    print(f"train-fake size  open={n_open}  commercial={n_comm}  commercial-1fam={n_c1}")
    if max(n_open, n_comm, n_c1) - min(n_open, n_comm, n_c1) > 1:
        print("  WARNING: pool sizes not matched -- the two directions are not comparable")
        ok = False
    if not ok:
        print("\nsanity checks FAILED -- not writing")
        return 1
    print("sanity checks passed")

    if args.write:
        for tag, splits in conditions.items():
            for split, payload in splits.items():
                out = BASE / f"split_mom_{split}_norm_{tag}.json"
                out.write_text(json.dumps(payload))
                print(f"wrote {out.name}")
    else:
        print("\ndry run -- pass --write to create the split files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
