"""Re-run inference for a finished CoMoE run and save per-clip predictions.

Usage (from the repository root, with COMOE_DATA_ROOT set):
  python -m comoe.predict --run_dir runs/mom_yue_xcodec_hl_norm_fakeA_sunov35 [--split ood|val]

Output: <run_dir>/preds_<split>.csv with columns utt_id,label,prob_fake,source.
scripts/per_source_auc.py reads these files.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F

from comoe.data import MoMCoMoEDataset, collate_fn
from comoe.paths import resolve_cfg
from comoe.train import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="ood", choices=["ood", "val"])
    args = ap.parse_args()

    run_dir = args.run_dir
    cfg = resolve_cfg(yaml.safe_load((run_dir / "config.yaml").read_text()))
    ckpt = torch.load(run_dir / "best.pt", weights_only=False, map_location="cpu")

    d = cfg["data"]
    split_file = d["split_ood"] if args.split == "ood" else d["split_val"]
    ood = MoMCoMoEDataset(
        token_dir=d["token_dir"],
        meta_csv=d["meta_csv"],
        split_file=split_file,
        max_frames=d["max_frames"],
        crop_frames=d["crop_frames"],
        semantic_dir=d.get("semantic_dir"),
        semantic_npz=d.get("semantic_npz"),
        crop_random=False,
        continuous=bool(d.get("continuous", False)),
    )
    loader = DataLoader(ood, batch_size=args.batch_size, num_workers=4,
                        shuffle=False, collate_fn=collate_fn)

    use_sem = bool(d.get("semantic_dir") or d.get("semantic_npz"))
    model = build_model(cfg, use_sem).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    rows = []
    with torch.inference_mode():
        for b in loader:
            tokens = b["tokens"].to(args.device)
            lengths = b["lengths"].to(args.device)
            mert = b["mert"].to(args.device) if "mert" in b else None
            logits = model(tokens, lengths, mert=mert)
            prob_fake = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            uids = b["utt_ids"] if "utt_ids" in b else b.get("uids", [None] * len(prob_fake))
            srcs = b.get("sources", ["?"] * len(prob_fake))
            for uid, lbl, p, src in zip(uids, b["labels"].cpu().numpy(),
                                         prob_fake, srcs):
                rows.append((uid, int(lbl), float(p), src))

    out_csv = run_dir / f"preds_{args.split}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "label", "prob_fake", "source"])
        w.writerows(rows)
    print(f"saved {len(rows):,} preds -> {out_csv}")


if __name__ == "__main__":
    main()
