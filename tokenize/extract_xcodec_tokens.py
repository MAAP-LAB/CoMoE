#!/usr/bin/env python3
"""Extract X-Codec mini (YuE ``xcodec_mini_infer``) RVQ tokens for every clip.

For each clip: 30 s mono @ 16 kHz -> xcodec_mini encode at 6 kbps (12 codebooks,
50 Hz) -> keep codebooks [0, 1, 10, 11] -> (4, T) int16 -> out_dir/<utt_id>.npy.

The codec code and weights are not redistributed here.  Download them once:
  git lfs install
  git clone https://huggingface.co/m-a-p/xcodec_mini_infer third_party/xcodec_mini_infer
and point --xcodec_dir (or $XCODEC_MINI_DIR) at that directory.

Usage:  python tokenize/extract_xcodec_tokens.py [--shard_idx i --n_shards n]
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `comoe` imports without pip install
from comoe.paths import REPO_ROOT, data_root, resolve_audio_path  # noqa: E402

# 12-RVQ codebooks @ 6kbps, 16k sr, 50Hz frame rate
KEEP_LAYERS = [0, 1, 10, 11]  # low=[0,1], high=[10,11]
XCODEC_SR = 16_000
XCODEC_SEC = 30
TARGET_BW = 6.0
CODEBOOK_SIZE = 1024
XCODEC_DIR = os.environ.get("XCODEC_MINI_DIR", str(REPO_ROOT / "third_party" / "xcodec_mini_infer"))

csv.field_size_limit(sys.maxsize)


def load_xcodec(device: str = "cuda"):
    """Load SoundStream + checkpoint. Assumes cwd is XCODEC_DIR."""
    import torch
    spec = importlib.util.spec_from_file_location("ss", "models/soundstream_hubert_new.py")
    ss = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ss)
    m = ss.SoundStream(n_filters=32, D=256)
    sd = torch.load("final_ckpt/ckpt_00360000.pth", map_location="cpu", weights_only=False)["codec_model"]
    m.load_state_dict(sd, strict=False)
    return m.eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", default=str(REPO_ROOT / "data" / "splits" / "metadata_mom_norm.csv"))
    ap.add_argument("--out_dir",
                    default=str(data_root() / "tokens_mom_norm" / "yue_xcodec_L0_L1_L10_L11"))
    ap.add_argument("--xcodec_dir", default=XCODEC_DIR,
                    help="clone of https://huggingface.co/m-a-p/xcodec_mini_infer")
    ap.add_argument("--gpu_idx", type=int, default=0)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)

    # Resolve paths before chdir: the codec code expects to run from its own directory
    # (it loads ./xcodec_mini_infer/semantic_ckpts/hf_1_325000 relative to cwd).
    xcodec_dir = os.path.abspath(os.path.expanduser(args.xcodec_dir))
    if not os.path.isfile(os.path.join(xcodec_dir, "final_ckpt", "ckpt_00360000.pth")):
        raise SystemExit(f"xcodec_mini_infer not found at {xcodec_dir}; see the module docstring")
    meta_csv = os.path.abspath(args.meta_csv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    cwd_orig = os.getcwd()
    os.chdir(xcodec_dir)
    sys.path.insert(0, xcodec_dir)
    sys.path.insert(0, os.path.join(xcodec_dir, "RepCodec"))
    if not os.path.exists("./xcodec_mini_infer"):
        os.symlink(".", "./xcodec_mini_infer")

    import torch
    import torchaudio

    out_dir.mkdir(parents=True, exist_ok=True)

    # Plan workload
    rows = list(csv.DictReader(open(meta_csv)))
    todo = []
    for r in rows:
        uid = r["utt_id"]
        h = int(hashlib.md5(uid.encode()).hexdigest(), 16) % args.n_shards
        if h != args.shard_idx:
            continue
        if (out_dir / f"{uid}.npy").exists():
            continue
        todo.append((uid, str(resolve_audio_path(r["filepath"]))))
    print(f"[shard {args.shard_idx}/{args.n_shards}] {len(todo)}/{len(rows)} clips to extract", flush=True)
    if not todo:
        print("[shard] nothing to do")
        return

    print(f"[shard {args.shard_idx}] loading xcodec_mini ...", flush=True)
    model = load_xcodec(device="cuda")
    print(f"[shard {args.shard_idx}] model ready. n_q={model.n_q} fr={model.frame_rate} sr={model.sample_rate}", flush=True)

    def load_wav(p):
        try:
            info = torchaudio.info(str(p))
            n = int(info.sample_rate * XCODEC_SEC) + info.sample_rate
            wav, sr = torchaudio.load(str(p), num_frames=n)
        except Exception:
            try:
                wav, sr = torchaudio.load(str(p))
            except Exception:
                return None
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != XCODEC_SR:
            wav = torchaudio.functional.resample(wav, sr, XCODEC_SR)
        wav = wav.squeeze(0)[: XCODEC_SR * XCODEC_SEC].contiguous().float()
        # pad to exact 30s
        if wav.shape[-1] < XCODEC_SR * XCODEC_SEC:
            pad = XCODEC_SR * XCODEC_SEC - wav.shape[-1]
            wav = torch.nn.functional.pad(wav, (0, pad))
        if wav.shape[-1] < XCODEC_SR // 10:
            return None
        return wav

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
        # (B, 1, T)
        x = torch.stack(wavs, dim=0).unsqueeze(1).cuda()
        with torch.inference_mode():
            codes = model.encode(x, target_bw=TARGET_BW)  # (n_q, B, T)
        # codes: (12, B, T) int64
        codes = codes.to(torch.int16).cpu().numpy()
        keep = codes[KEEP_LAYERS]   # (4, B, T)
        for i, uid in enumerate(uids_b):
            np.save(out_dir / f"{uid}.npy", keep[:, i, :].copy())  # (4, T)
            n_done += 1
        if n_done % args.log_every == 0 or n_done == len(todo):
            rate = n_done / max(time.time() - t0, 1e-6)
            eta = (len(todo) - n_done) / max(rate, 1e-6)
            print(f"[shard {args.shard_idx}] {n_done}/{len(todo)}  "
                  f"{rate:.2f} clips/s  eta={eta/60:.1f}min  fail={n_fail}", flush=True)
    print(f"[shard {args.shard_idx}] done. n_done={n_done} fail={n_fail}")
    os.chdir(cwd_orig)


if __name__ == "__main__":
    main()
