"""Build the fake-source-restricted splits on top of the base splits.

  Fake-Suno3.5  (fakeA_sunov35)   train fake = Suno-v3.5 only
  Fake-Udio     (fakeB_udio)      train fake = Udio only
  Fake-DiffRhythm (fakeC_diffrythm) train fake = DiffRhythm only

The real side is identical to the base split.  The held-out generators that
were in train/val move to the OOD fake pool, together with the original OOD
fakes (Riffusion, Suno-v3, Suno-v4, YuE).

Usage:  python data/build_fake_loso_splits.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from comoe.paths import REPO_ROOT  # noqa: E402

SPLITS = REPO_ROOT / "data" / "splits"
META = SPLITS / "metadata_mom_norm.csv"
IN = {
    "train": SPLITS / "split_mom_train_norm.json",
    "val":   SPLITS / "split_mom_val_norm.json",
    "ood":   SPLITS / "split_mom_ood_norm.json",
}

CONDITIONS = [
    # tag,               train source,  sources moved to OOD
    ("fakeA_sunov35",    "suno_v3.5",   ["diffrythm", "suno_v2", "udio"]),
    ("fakeB_udio",       "udio",        ["diffrythm", "suno_v2", "suno_v3.5"]),
    ("fakeC_diffrythm",  "diffrythm",   ["suno_v2", "suno_v3.5", "udio"]),
]


def main() -> int:
    rows = {r["utt_id"]: r for r in csv.DictReader(META.open())}
    src_of = lambda u: rows[u]["source"]  # noqa: E731

    def fakes_from(uids, keep_source):
        return [u for u in uids if src_of(u) == keep_source]

    for tag, train_src, holdout_srcs in CONDITIONS:
        d_tr = json.load(IN["train"].open())
        d_va = json.load(IN["val"].open())
        d_od = json.load(IN["ood"].open())

        # train/val: all real, fakes = train_src only (val stays in-distribution)
        tr = {"real": list(d_tr["real"]), "fake": fakes_from(d_tr["fake"], train_src)}
        va = {"real": list(d_va["real"]), "fake": fakes_from(d_va["fake"], train_src)}
        # ood: original OOD real; fakes = held-out generators from train+val + original OOD fakes
        ood_fake_extra = []
        for src in holdout_srcs:
            ood_fake_extra += fakes_from(d_tr["fake"], src)
            ood_fake_extra += fakes_from(d_va["fake"], src)
        od = {"real": list(d_od["real"]), "fake": ood_fake_extra + list(d_od["fake"])}

        for name, obj in [("train", tr), ("val", va), ("ood", od)]:
            out = SPLITS / f"split_mom_{name}_norm_{tag}.json"
            out.write_text(json.dumps(obj, indent=2))
            print(f"[{tag}] {name}: real={len(obj['real'])} fake={len(obj['fake'])} -> {out.name}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
