#!/usr/bin/env python3
"""Per-source AUC for the licence conditions, computed from saved predictions.

`ood_result.json` only stores per-source *accuracy*, which is read at one global
threshold.  Under single-family training that threshold is badly calibrated --
the paper's own finding (ii) -- so a source can look catastrophic on accuracy
while its scores still separate cleanly.  AUC per source avoids that: each
generator is scored against the same real clips, so the numbers say how well
the detector ranks that generator above real audio, independent of threshold.

    python scripts/per_source_auc.py

Needs <run_dir>/preds_ood.csv (produced by python -m comoe.predict).
"""
import csv
import pathlib

RUNS = pathlib.Path(__file__).resolve().parents[1] / "runs"
REAL = {"fma_medium", "jamendo"}
AMBIGUOUS = {"riffusion"}

CONDITIONS = [
    ("mert_kmeans_hl", "MERT k-means"),
    ("encodec_tokenonly", "EnCodec"),
    ("yue_xcodec_hl", "YuE xcodec"),
]
TAGS = [("fakeOpen_ow", "Fake-Open"), ("fakeComm_matched", "Fake-Comm")]


def auc(pos, neg):
    """Rank-based AUC: P(score(pos) > score(neg)), ties counted as half."""
    if not pos or not neg:
        return None
    marked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    # average ranks over ties so exact score ties contribute 0.5
    ranks, i = {}, 0
    while i < len(marked):
        j = i
        while j + 1 < len(marked) and marked[j + 1][0] == marked[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, lab) in enumerate(marked) if lab == 1)
    n_p, n_n = len(pos), len(neg)
    return (rank_sum - n_p * (n_p + 1) / 2) / (n_p * n_n)


def load(run: pathlib.Path):
    f = run / "preds_ood.csv"
    if not f.is_file():
        return None
    by_src = {}
    for r in csv.DictReader(f.open()):
        by_src.setdefault(r["source"], []).append(float(r["prob_fake"]))
    return by_src


def main() -> int:
    for stem, label in CONDITIONS:
        for tag, tag_label in TAGS:
            run = RUNS / f"mom_{stem}_norm_{tag}"
            by_src = load(run)
            if by_src is None:
                print(f"{label} / {tag_label}: no preds_ood.csv yet")
                continue
            real = [s for k, v in by_src.items() if k in REAL for s in v]
            fakes = {k: v for k, v in by_src.items() if k not in REAL}
            per = {k: auc(v, real) for k, v in fakes.items()}
            core = [v for k, v in per.items() if k not in AMBIGUOUS and v is not None]
            print(f"\n=== {label} / {tag_label} ===")
            for s, v in sorted(per.items(), key=lambda x: -(x[1] or 0)):
                mark = "  (ambiguous licence)" if s in AMBIGUOUS else ""
                print(f"   {s:<14} AUC={v:.4f}  n={len(fakes[s]):>6}{mark}")
            if core:
                print(f"   {'mean (excl. Riffusion)':<14} {sum(core)/len(core):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
