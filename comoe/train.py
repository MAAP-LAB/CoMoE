"""Train one CoMoE run on MoM-open and evaluate it on the held-out (OOD) split.

The model is selected on validation F1 and evaluated once on the OOD split of
the chosen split definition (base / Real-* / Fake-*).

Usage (from the repository root, with COMOE_DATA_ROOT set):
  python -m comoe.train --config configs/mom_yue_xcodec_hl_norm_fakeA_sunov35.yaml
  python -m comoe.train --config configs/<name>.yaml --override train.epochs=1 train.save_dir=runs/smoke

Outputs in train.save_dir: config.yaml (as given, paths unresolved), best.pt,
train_log.json, ood_result.json.  Set wandb.enabled: true in the config to log
to Weights & Biases; it is off by default.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from comoe.data import MoMCoMoEDataset, collate_fn
from comoe.model import CoMoE
from comoe.paths import resolve_cfg

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_cfg(p: Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


def make_loaders(cfg: dict):
    d = cfg["data"]
    sem_npz = d.get("semantic_npz")
    sem_dir = d.get("semantic_dir")
    use_sem_data = (sem_npz and Path(sem_npz).exists()) or (sem_dir and Path(sem_dir).exists())

    continuous = bool(d.get("continuous", False))

    def _ds(split_file: str, crop_random: bool) -> MoMCoMoEDataset:
        return MoMCoMoEDataset(
            token_dir=d["token_dir"],
            meta_csv=d["meta_csv"],
            split_file=split_file,
            semantic_npz=sem_npz if use_sem_data else None,
            semantic_dir=sem_dir if use_sem_data else None,
            max_frames=d["max_frames"],
            crop_frames=d.get("crop_frames"),
            crop_random=crop_random,
            continuous=continuous,
        )

    train_ds = _ds(d["split_train"], crop_random=True)
    val_ds = _ds(d["split_val"], crop_random=False)
    ood_ds = _ds(d["split_ood"], crop_random=False)

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]

    def _ld(ds: MoMCoMoEDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=nw,
            collate_fn=collate_fn, pin_memory=True, drop_last=shuffle,
            persistent_workers=(nw > 0),
        )

    return _ld(train_ds, True), _ld(val_ds, False), _ld(ood_ds, False), use_sem_data


def build_model(cfg: dict, use_sem_data: bool) -> CoMoE:
    m = cfg["model"]
    return CoMoE(
        codebook_size=cfg["data"]["codebook_size"],
        low_layers=m["low_layers"],
        high_layers=m["high_layers"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_blocks=m["n_layers"],
        dropout=m["dropout"],
        max_frames=cfg["data"]["max_frames"],
        mert_dim=cfg["data"].get("semantic_dim", 768),
        use_semantic=m.get("use_semantic", True) and use_sem_data,
        use_low=m.get("use_low", True),
        use_high=m.get("use_high", True),
        use_latent=False,
        mert_multilayer_groups=m.get("mert_multilayer_groups", None),
        fusion=m.get("fusion", "concat"),
        continuous_frames=bool(cfg["data"].get("continuous", False)),
        continuous_in_dim=cfg["data"].get("continuous_in_dim", 768),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    all_logits, all_labels, all_sources = [], [], []
    crit = nn.CrossEntropyLoss()
    for b in loader:
        tokens = b["tokens"].to(device)
        lengths = b["lengths"].to(device)
        labels = b["labels"].to(device)
        mert = b["mert"].to(device) if "mert" in b else None
        logits = model(tokens, lengths, mert=mert)
        loss = crit(logits, labels)
        loss_sum += float(loss) * labels.size(0)
        pred = logits.argmax(-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_sources.extend(b["sources"])
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = logits.argmax(-1)
    try:
        auc = roc_auc_score(labels, logits[:, 1] - logits[:, 0])
    except Exception:
        auc = float("nan")
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": auc,
        "n": total,
        "logits": logits,
        "labels": labels,
        "sources": all_sources,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    for ov in args.override:
        k, v = ov.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        cur = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v

    cfg_raw = copy.deepcopy(cfg)  # what gets saved: ${COMOE_DATA_ROOT} left unexpanded
    cfg = resolve_cfg(cfg)

    set_seed(cfg["train"]["seed"])
    device = cfg["train"]["device"]

    train_ld, val_ld, ood_ld, use_sem = make_loaders(cfg)
    print(f"Train={len(train_ld.dataset)} Val={len(val_ld.dataset)} OOD={len(ood_ld.dataset)}")
    print(f"Use semantic expert: {use_sem}")

    model = build_model(cfg, use_sem).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.2f}M")

    wb_cfg = cfg.get("wandb", {}) or {}
    wb_enabled = bool(wb_cfg.get("enabled", False)) and _WANDB_AVAILABLE
    if wb_enabled:
        run = wandb.init(
            project=wb_cfg.get("project", "comoe"),
            name=wb_cfg.get("name") or Path(cfg["train"]["save_dir"]).name,
            mode=wb_cfg.get("mode", "online"),
            config={**cfg, "params_M": n_params / 1e6, "use_semantic_data": use_sem},
            dir=str(Path(cfg["train"]["save_dir"]).parent),
        )
        wandb.watch(model, log=None)  # don't log gradients (heavy)
        print(f"[wandb] {run.name} → {run.url}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    total_steps = len(train_ld) * cfg["train"]["epochs"]
    warmup = cfg["train"]["warmup_steps"]

    def lr_at(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_at)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"].get("amp", True))

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_raw, f)

    best_val = -1.0
    best_epoch = -1
    log = []
    step = 0
    log_every = int(cfg["train"].get("log_every_steps", 50))

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        t0 = time.time()
        tot_loss, tot_n, tot_correct = 0.0, 0, 0

        for b in train_ld:
            tokens = b["tokens"].to(device, non_blocking=True)
            lengths = b["lengths"].to(device, non_blocking=True)
            labels = b["labels"].to(device, non_blocking=True)
            mert = b["mert"].to(device, non_blocking=True) if "mert" in b else None

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg["train"].get("amp", True)):
                logits = model(tokens, lengths, mert=mert)
                loss = crit(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1

            bsz = labels.size(0)
            tot_loss += loss.detach().item() * bsz
            tot_n += bsz
            tot_correct += (logits.argmax(-1) == labels).sum().item()

            if wb_enabled and (step % log_every == 0):
                wandb.log({
                    "train/loss_step": loss.detach().item(),
                    "train/lr": opt.param_groups[0]["lr"],
                    "epoch": epoch + step / max(len(train_ld), 1),
                }, step=step)

        train_acc = tot_correct / max(tot_n, 1)
        train_loss = tot_loss / max(tot_n, 1)
        val_res = evaluate(model, val_ld, device)
        dt = time.time() - t0
        print(
            f"[Epoch {epoch+1:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_res['loss']:.4f} val_acc={val_res['acc']:.4f} "
            f"val_f1={val_res['f1']:.4f} val_auc={val_res['auc']:.4f} | {dt:.1f}s"
        )
        log.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_res["loss"], "val_acc": val_res["acc"],
            "val_f1": val_res["f1"], "val_auc": val_res["auc"], "time_s": dt,
        })
        if wb_enabled:
            wandb.log({
                "train/loss_epoch": train_loss,
                "train/acc_epoch": train_acc,
                "val/loss": val_res["loss"],
                "val/acc": val_res["acc"],
                "val/f1": val_res["f1"],
                "val/auc": val_res["auc"],
                "epoch": epoch + 1,
                "time/epoch_s": dt,
            }, step=step)
        if val_res["f1"] > best_val:
            best_val = val_res["f1"]
            best_epoch = epoch + 1
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg_raw,
                "epoch": epoch + 1,
                "val_f1": best_val,
            }, save_dir / "best.pt")

    ckpt = torch.load(save_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    ood_res = evaluate(model, ood_ld, device)
    print(f"\n[OOD] best_epoch={best_epoch} acc={ood_res['acc']:.4f} "
          f"f1={ood_res['f1']:.4f} auc={ood_res['auc']:.4f}")

    by_src: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    preds = ood_res["logits"].argmax(-1)
    for s, l, p in zip(ood_res["sources"], ood_res["labels"], preds):
        by_src[s]["n"] += 1
        by_src[s]["correct"] += int(l == p)
    print("\n[OOD per-source acc]")
    for s, v in sorted(by_src.items()):
        print(f"  {s:<20} n={v['n']:>5}  acc={v['correct']/v['n']:.4f}")

    with open(save_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    with open(save_dir / "ood_result.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "acc": ood_res["acc"], "f1": ood_res["f1"], "auc": ood_res["auc"],
            "per_source_acc": {s: v["correct"]/v["n"] for s, v in by_src.items()},
        }, f, indent=2)
    print(f"\nSaved to {save_dir}/")

    if wb_enabled:
        wandb.log({
            "ood/acc": ood_res["acc"], "ood/f1": ood_res["f1"], "ood/auc": ood_res["auc"],
            "ood/best_epoch": best_epoch,
        }, step=step)
        for s, v in by_src.items():
            wandb.summary[f"ood/per_source_acc/{s}"] = v["correct"] / v["n"]
        wandb.summary["best/val_f1"] = best_val
        wandb.summary["best/epoch"] = best_epoch
        wandb.finish()


if __name__ == "__main__":
    main()
