"""CLAM baseline (re-implementation) on MoM-open.

Joint loss = BCE + lambda * Triplet (real-only).  Encoders frozen (MERT-v1-95M,
Wav2Vec2-base); only the WCA streams and the classifier head are trained.

The paper runs used 1 epoch, 30 s clips, seed 42, on each split definition:
  python -m baselines.clam.train --split_tag ""              --out_dir runs/clam/base
  python -m baselines.clam.train --split_tag fakeA_sunov35   --out_dir runs/clam/fakeA_sunov35
(optionally --cache_dir <dir> after baselines/clam/cache_audio.py to skip mp3 decoding).
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from baselines.clam.dataset import MoMCLAMDataset
from baselines.clam.model import CLAM, triplet_loss_real
from comoe.paths import REPO_ROOT


# Cached on-GPU resampler (24 kHz → 16 kHz).
_RESAMPLER: torchaudio.transforms.Resample | None = None


def to_16k(wav24: torch.Tensor) -> torch.Tensor:
    global _RESAMPLER
    if _RESAMPLER is None:
        _RESAMPLER = torchaudio.transforms.Resample(24000, 16000).to(wav24.device)
    return _RESAMPLER(wav24)


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        w24 = batch["wav_24k"].to(device, non_blocking=True)
        w16 = to_16k(w24)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logit, _, _ = model(w24, w16)
        prob = torch.sigmoid(logit).float().cpu().numpy()
        for u, l, p, s in zip(batch["utt_id"], batch["label"].tolist(), prob.tolist(), batch["source"]):
            rows.append((u, int(l), float(p), s))
    df = pd.DataFrame(rows, columns=["utt_id", "label", "prob", "source"])
    y, p = df["label"].to_numpy(), df["prob"].to_numpy()
    auc = roc_auc_score(y, p)
    eer = equal_error_rate(y, p)
    f1 = f1_score(y, (p >= 0.5).astype(int))
    acc = float(((p >= 0.5).astype(int) == y).mean())
    return df, dict(auc=auc, eer=eer, f1=f1, acc=acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=str, default=str(REPO_ROOT / "data/splits/metadata_mom_norm.csv"))
    ap.add_argument("--split_dir", type=str, default=str(REPO_ROOT / "data/splits"))
    ap.add_argument("--split_tag", type=str, default="",
                    help='"" (base), losoA_fma2jam, losoB_jam2fma, fakeA_sunov35, fakeB_udio, fakeC_diffrythm')
    ap.add_argument("--clip_sec", type=float, default=30.0)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--lambda_triplet", type=float, default=0.5)
    ap.add_argument("--triplet_margin", type=float, default=1.0)
    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--max_train_per_class", type=int, default=0)
    ap.add_argument("--max_val_per_class", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="runs/clam/base")
    ap.add_argument("--cache_dir", type=str, default=None,
                    help="If set, load int16 .npy from <cache_dir>/<uid[:2]>/<uid>.npy instead of decoding")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda")
    print(f"[device] {device} (n={torch.cuda.device_count()})", flush=True)

    suffix = f"_norm_{args.split_tag}" if args.split_tag else "_norm"
    train_ds = MoMCLAMDataset(args.meta, f"{args.split_dir}/split_mom_train{suffix}.json",
                              clip_sec=args.clip_sec, random_crop=True,
                              max_per_class=args.max_train_per_class,
                              cache_dir=args.cache_dir)
    val_ds = MoMCLAMDataset(args.meta, f"{args.split_dir}/split_mom_val{suffix}.json",
                            clip_sec=args.clip_sec, random_crop=False,
                            max_per_class=args.max_val_per_class,
                            cache_dir=args.cache_dir)
    ood_ds = MoMCLAMDataset(args.meta, f"{args.split_dir}/split_mom_ood{suffix}.json",
                            clip_sec=args.clip_sec, random_crop=False,
                            max_per_class=args.max_val_per_class,
                            cache_dir=args.cache_dir)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} ood={len(ood_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False, drop_last=True,
                              persistent_workers=args.num_workers > 0,
                              multiprocessing_context="forkserver" if args.num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False,
                            persistent_workers=args.num_workers > 0)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False,
                            persistent_workers=args.num_workers > 0)

    print("[model] building CLAM (MERT-95M + Wav2Vec2-base, frozen)…", flush=True)
    model = CLAM(embed_dim=args.embed_dim, hidden=args.hidden, n_heads=args.n_heads,
                 dropout=args.dropout, freeze_encoders=True).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable {n_train/1e6:.2f}M / total {n_total/1e6:.1f}M", flush=True)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    scaler = torch.amp.GradScaler("cuda")
    best_val_auc, best_path = 0.0, out / "best.pt"
    history = []

    step = 0
    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep encoders in eval to lock dropout/BN.
        model.mert.eval(); model.w2v2.eval()
        ep_loss, ep_bce, ep_tri = 0.0, 0.0, 0.0
        n_batch = 0
        t0 = time.time()
        for batch in train_loader:
            w24 = batch["wav_24k"].to(device, non_blocking=True)
            w16 = to_16k(w24)
            y = batch["label"].float().to(device, non_blocking=True)

            for g in optim.param_groups:
                g["lr"] = lr_at(step)
            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logit, d_m, d_v = model(w24, w16)
                bce = F.binary_cross_entropy_with_logits(logit, y)
                tri = triplet_loss_real(d_m, d_v, batch["label"].to(device), margin=args.triplet_margin)
                loss = bce + args.lambda_triplet * tri

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=5.0
            )
            scaler.step(optim)
            scaler.update()

            ep_loss += loss.item(); ep_bce += bce.item(); ep_tri += float(tri)
            n_batch += 1
            step += 1
            if step % args.log_every == 0:
                el = time.time() - t0
                rate = (n_batch * args.batch_size) / el
                print(f"  ep{epoch} step{step}/{total_steps} "
                      f"loss={ep_loss/n_batch:.4f} bce={ep_bce/n_batch:.4f} "
                      f"tri={ep_tri/n_batch:.4f} lr={lr_at(step):.2e} "
                      f"rate={rate:.1f} clip/s", flush=True)

        train_stats = dict(epoch=epoch, train_loss=ep_loss / n_batch,
                           train_bce=ep_bce / n_batch, train_tri=ep_tri / n_batch,
                           train_sec=time.time() - t0)
        print(f"[ep{epoch}] train: {train_stats['train_loss']:.4f} "
              f"({train_stats['train_sec']/60:.1f} min)", flush=True)

        val_df, val_m = evaluate(model, val_loader, device)
        ood_df, ood_m = evaluate(model, ood_loader, device)
        print(f"[ep{epoch}] val  AUC={val_m['auc']:.4f} EER={val_m['eer']:.4f} "
              f"F1={val_m['f1']:.4f} ACC={val_m['acc']:.4f}", flush=True)
        print(f"[ep{epoch}] ood  AUC={ood_m['auc']:.4f} EER={ood_m['eer']:.4f} "
              f"F1={ood_m['f1']:.4f} ACC={ood_m['acc']:.4f}", flush=True)

        rec = {**train_stats, **{f"val_{k}": v for k, v in val_m.items()},
               **{f"ood_{k}": v for k, v in ood_m.items()}}
        history.append(rec)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)

        if val_m["auc"] > best_val_auc:
            best_val_auc = val_m["auc"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "val": val_m, "ood": ood_m}, best_path)
            val_df.to_csv(out / "preds_val_best.csv", index=False)
            ood_df.to_csv(out / "preds_ood_best.csv", index=False)
            print(f"  ↳ new best val AUC {best_val_auc:.4f} → {best_path}", flush=True)

    print(f"[done] total {(time.time() - t_train)/60:.1f} min, best val AUC = {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
