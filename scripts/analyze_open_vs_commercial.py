#!/usr/bin/env python3
"""Regroup existing per-source accuracies into open-weight vs commercial.

Motivated by ICML feedback: Suno and Udio watermark their output, so a
detector might be picking up the watermark rather than a generative artefact.
If that were the whole story, a model trained on a watermarked commercial
family should transfer to *other commercial* families and fail on open-weight
ones.  Every finished run already stores per_source_acc, so this costs no GPU.

    python scripts/analyze_open_vs_commercial.py
"""
import json
import pathlib

RUNS = pathlib.Path(__file__).resolve().parents[1] / "runs"

# Riffusion is deliberately left out of both buckets: the name covers both the
# 2022 open Stable-Diffusion spectrogram model and today's commercial service,
# and the MoM metadata does not say which one produced these clips.
COMMERCIAL = {"suno_v2", "suno_v3", "suno_v3.5", "suno_v4", "udio"}
OPEN_WEIGHT = {"diffrythm", "yue"}
REAL = {"fma_medium", "jamendo"}


def bucket_means(per_source):
    """Mean accuracy over held-out fake sources in each bucket."""
    out = {}
    for name, members in (("open", OPEN_WEIGHT), ("comm", COMMERCIAL)):
        vals = [v for s, v in per_source.items() if s in members]
        out[name] = sum(vals) / len(vals) if vals else None
    return out


def main() -> int:
    rows = []
    for d in sorted(RUNS.iterdir()):
        f = d / "ood_result.json"
        if not f.is_file():
            continue
        res = json.loads(f.read_text())
        per = res.get("per_source_acc") or {}
        # per_source_acc mixes real and fake sources; keep the fakes only.
        fakes = {k: v for k, v in per.items() if k not in REAL}
        b = bucket_means(fakes)
        if b["open"] is None or b["comm"] is None:
            continue  # condition does not hold out both buckets
        rows.append((d.name, res.get("auc"), b["open"], b["comm"], b["open"] - b["comm"]))

    if not rows:
        print("no run holds out both an open-weight and a commercial generator")
        return 1

    print(f"{'run':<48}{'AUC':>8}{'open':>8}{'comm':>8}{'Δ':>8}")
    print("-" * 80)
    for name, auc, o, c, d in sorted(rows, key=lambda r: r[4]):
        a = f"{auc:.4f}" if auc is not None else "  --  "
        print(f"{name:<48}{a:>8}{o:>8.3f}{c:>8.3f}{d:>+8.3f}")

    deltas = [r[4] for r in rows]
    print(f"\nΔ = open-weight acc − commercial acc, over {len(rows)} runs")
    print(f"mean {sum(deltas)/len(deltas):+.3f}   min {min(deltas):+.3f}   max {max(deltas):+.3f}")
    print("\nΔ << 0 across the board would support the watermark-shortcut story;")
    print("Δ >= 0 argues the detector is not merely reading a commercial watermark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
