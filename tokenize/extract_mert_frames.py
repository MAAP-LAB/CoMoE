#!/usr/bin/env python3
"""Extract MERT-v0 frame-level continuous features for layers {0,1,11,12}.

For each clip:
  1. Load 30s waveform @ 16 kHz
  2. MERT-v0 forward -> hidden_states (13, T_native, 768)
  3. Pick layers {0,1,11,12} -> (4, T_native, 768)
  4. Average-pool every POOL_FACTOR consecutive frames -> (4, T_out, 768)
  5. Save as fp16 npy at out_dir/{utt_id}.npy

This produces continuous-valued analog of the k-means token version, for
the CoMoE-HL (MERT continuous) ablation.

Sharded by hashing utt_id -> (n_shards, shard_idx).

Usage:  python tokenize/extract_mert_frames.py [--shard_idx i --n_shards n]
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `comoe` imports without pip install
from comoe.paths import REPO_ROOT, data_root, resolve_audio_path  # noqa: E402

LAYERS = [0, 1, 11, 12]
MERT_SR = 16_000
MERT_SEC = 30
MODEL_NAME = "m-a-p/MERT-v0-public"

csv.field_size_limit(sys.maxsize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", default=str(REPO_ROOT / "data" / "splits" / "metadata_mom_norm.csv"))
    ap.add_argument("--out_dir",
                    default=str(data_root() / "feats_mom_norm" / "mert_v0_frames_L0_L1_L11_L12_T300"))
    ap.add_argument("--pool_factor", type=int, default=5,
                    help="Average-pool every N native MERT frames (~50fps -> ~10fps when N=5).")
    ap.add_argument("--t_out", type=int, default=300,
                    help="Target output time length (truncate/pad after pooling).")
    ap.add_argument("--gpu_idx", type=int, default=0)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--log_every", type=int, default=500)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)

    import torch
    import torchaudio
    from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.meta_csv)))
    todo = []
    for r in rows:
        uid = r["utt_id"]
        h = int(hashlib.md5(uid.encode()).hexdigest(), 16) % args.n_shards
        if h != args.shard_idx:
            continue
        if (out_dir / f"{uid}.npy").exists():
            continue
        todo.append((uid, str(resolve_audio_path(r["filepath"]))))
    print(f"[shard {args.shard_idx}/{args.n_shards}] {len(todo)} clips to extract "
          f"(of {len(rows)} total)  pool_factor={args.pool_factor}  t_out={args.t_out}",
          flush=True)
    if not todo:
        print("[shard] nothing to do")
        return

    print(f"[shard {args.shard_idx}] loading {MODEL_NAME}", flush=True)
    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    cfg.conv_pos_batch_norm = False
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, config=cfg, trust_remote_code=True
    ).to("cuda").eval()

    def load_wav(p: Path):
        try:
            info = torchaudio.info(str(p))
            n = int(info.sample_rate * MERT_SEC) + info.sample_rate
            wav, sr = torchaudio.load(str(p), num_frames=n)
        except Exception:
            try:
                wav, sr = torchaudio.load(str(p))
            except Exception:
                return None
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != MERT_SR:
            wav = torchaudio.functional.resample(wav, sr, MERT_SR)
        wav = wav.squeeze(0)[: MERT_SR * MERT_SEC].contiguous().float().numpy()
        if wav.shape[-1] < MERT_SR // 10:
            return None
        return wav

    P = args.pool_factor
    T_out = args.t_out
    t0 = time.time()
    n_done = 0
    n_fail = 0
    for batch_start in range(0, len(todo), args.batch_size):
        batch = todo[batch_start: batch_start + args.batch_size]
        wavs, uids_b = [], []
        for uid, fp in batch:
            w = load_wav(Path(fp))
            if w is None:
                n_fail += 1
                continue
            wavs.append(w)
            uids_b.append(uid)
        if not wavs:
            continue
        inputs = fe(wavs, sampling_rate=MERT_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of 13 tensors (B, T_native, 768)
        T_native = out.hidden_states[0].shape[1]
        # Average-pool every P frames along time
        T_pooled = T_native // P
        for i, uid in enumerate(uids_b):
            # Build (4, T_pooled, 768) fp16
            stacked = np.zeros((len(LAYERS), T_out, 768), dtype=np.float16)
            for L_idx, L in enumerate(LAYERS):
                X = out.hidden_states[L][i, :T_pooled * P].float().cpu().numpy()  # (T_pooled*P, 768)
                X = X.reshape(T_pooled, P, 768).mean(axis=1)  # (T_pooled, 768)
                # Truncate or zero-pad to T_out
                T_keep = min(T_pooled, T_out)
                stacked[L_idx, :T_keep, :] = X[:T_keep, :].astype(np.float16)
            np.save(out_dir / f"{uid}.npy", stacked)
            n_done += 1
        if n_done % args.log_every == 0 or n_done == len(todo):
            rate = n_done / max(time.time() - t0, 1e-6)
            eta = (len(todo) - n_done) / max(rate, 1e-6)
            print(f"[shard {args.shard_idx}] {n_done}/{len(todo)}  "
                  f"{rate:.2f} clips/s  eta={eta/60:.1f}min  fail={n_fail}",
                  flush=True)
    print(f"[shard {args.shard_idx}] done. n_done={n_done} fail={n_fail}")


if __name__ == "__main__":
    main()
