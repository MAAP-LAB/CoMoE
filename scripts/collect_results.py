#!/usr/bin/env python3
"""Build the main results matrix straight from runs/*/ood_result.json.

The workshop version shipped with a ``--'' in the YuE base cell even though
the run had finished two days before the camera-ready compile.  Regenerating
the table from disk instead of hand-editing it makes that class of mistake
impossible: anything that has an ood_result.json shows up here.

    python scripts/collect_results.py            # status matrix
    python scripts/collect_results.py --latex    # tabular body
"""
import argparse
import json
import pathlib

RUNS = pathlib.Path(__file__).resolve().parents[1] / "runs"

# display name -> run-name prefix.  Order matches the paper's table.
MODELS = [
    (r"\textsc{CoMoE-HLS} (EnCodec)", "CoMoE-HLS (EnCodec)", "mom_encodec_mert_v0_norm"),
    (r"\textsc{CoMoE-HLS} (DAC)", "CoMoE-HLS (DAC)", "mom_dac_mert_v0_norm"),
    (r"\textsc{CoMoE-HL}  (EnCodec)", "CoMoE-HL (EnCodec)", "mom_encodec_tokenonly_norm"),
    (r"\textsc{CoMoE-HL}  (MERT $k$-means)", "CoMoE-HL (MERT km)", "mom_mert_kmeans_hl_norm"),
    (r"\textsc{CoMoE-HL}  (MERT $k$-m., real-only)", "CoMoE-HL (MERT km real-only)", "mom_mert_kmeans_hl_norm_REAL_ONLY"),
    (r"\textsc{CoMoE-HL}  (MERT continuous)", "CoMoE-HL (MERT cont.)", "mom_mert_continuous_hl_norm"),
    (r"\textsc{CoMoE-HL}  (YuE xcodec)", "CoMoE-HL (YuE)", "mom_yue_xcodec_hl_norm"),
    (r"\textsc{CoMoE-HLS} (YuE xcodec)", "CoMoE-HLS (YuE)", "mom_yue_xcodec_hls_norm"),
    (r"\textsc{CoMoE-S}   (MERT-v0 pooled)", "CoMoE-S (MERT pool)", "mom_semantic_only_mert_v0_norm"),
]

# column label -> run-name suffix
COLUMNS = [
    ("base", ""),
    ("Real-A", "_losoA_fma2jam"),
    ("Real-B", "_losoB_jam2fma"),
    ("Fake-A", "_fakeA_sunov35"),
    ("Fake-B", "_fakeB_udio"),
    ("Fake-C", "_fakeC_diffrythm"),
]


def auc(prefix: str, suffix: str):
    """OOD AUC for one cell, or None if that run has not finished.

    Two naming conventions are in use for the base split: the older runs have
    no suffix at all (mom_encodec_mert_v0_norm) while the YuE runs spell it
    out (mom_yue_xcodec_hl_norm_base).  Accept either.  Matching is on the
    exact directory name, never a prefix, so that mom_mert_kmeans_hl_norm does
    not pick up mom_mert_kmeans_hl_norm_REAL_ONLY_*.
    """
    candidates = [f"{prefix}{suffix}"] + ([f"{prefix}_base"] if suffix == "" else [])
    for name in candidates:
        f = RUNS / name / "ood_result.json"
        if f.is_file():
            return json.loads(f.read_text()).get("auc")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="emit a LaTeX tabular body")
    args = ap.parse_args()

    if not RUNS.is_dir():
        print(f"{RUNS} not found")
        return 1

    table = {p: {c: auc(p, s) for c, s in COLUMNS} for _, _, p in MODELS}

    if args.latex:
        # Bold the column-wise best so the table stays self-consistent.
        best = {
            c: max((v for row in table.values() if (v := row[c]) is not None), default=None)
            for c, _ in COLUMNS
        }
        for tex, _, prefix in MODELS:
            cells = []
            base = table[prefix]["base"]
            for c, _ in COLUMNS:
                v = table[prefix][c]
                if v is None:
                    cells.append("--")
                    continue
                s = f"\\textbf{{{v:.4f}}}" if v == best[c] else f"{v:.4f}"
                if base is not None and c != "base":
                    s += f" (${(v - base) * 100:+.1f}$)".replace("+", "")
                cells.append(s)
            print(f"{tex:<44} & " + " & ".join(cells) + r" \\")
        return 0

    head = f"{'model':<30}" + "".join(f"{c:>9}" for c, _ in COLUMNS)
    print(head)
    print("-" * len(head))
    missing = 0
    for _, name, prefix in MODELS:
        row = f"{name:<30}"
        for c, _ in COLUMNS:
            v = table[prefix][c]
            row += f"{v:>9.4f}" if v is not None else f"{'--':>9}"
            missing += v is None
        print(row)
    print(f"\n{missing} empty cells ({missing * 75 / 60:.1f} h to fill at 75 min/run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
