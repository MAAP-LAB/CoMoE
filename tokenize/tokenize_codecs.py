"""Tokenize MoM-open with EnCodec 24 kHz (6 kbps, 8 codebooks) and DAC 44.1 kHz (9 codebooks).

Outputs, one file per clip (int16, full RVQ stack; the config selects the streams):
  $COMOE_DATA_ROOT/tokens_mom_norm/encodec24k/<utt_id>.npy  (8, T)  75 Hz
  $COMOE_DATA_ROOT/tokens_mom_norm/dac44k/<utt_id>.npy      (9, T)  86 Hz

Usage (from the repository root; shards run in parallel on separate GPUs):
  CUDA_VISIBLE_DEVICES=0 python tokenize/tokenize_codecs.py --shard 0 --num-shards 2 &
  CUDA_VISIBLE_DEVICES=1 python tokenize/tokenize_codecs.py --shard 1 --num-shards 2 &
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `comoe` imports without pip install
from comoe.paths import REPO_ROOT, data_root, resolve_audio_path  # noqa: E402

import dac
from audiotools import AudioSignal
from encodec import EncodecModel

csv.field_size_limit(sys.maxsize)
torch.backends.cudnn.benchmark = True

CLIP_SECONDS = 90
DAC_SR = 44100
ENC_SR = 24000
DAC_DIR = data_root() / "tokens_mom_norm" / "dac44k"
ENC_DIR = data_root() / "tokens_mom_norm" / "encodec24k"
META = REPO_ROOT / "data" / "splits" / "metadata_mom_norm.csv"


def load_audio_mono(path: Path, target_sr: int, max_seconds: int) -> torch.Tensor | None:
    """Load → mono mix → resample → trim to max_seconds. Returns 1D float32 tensor."""
    try:
        wav, sr = torchaudio.load(str(path))  # (C, N)
    except Exception:
        return None
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    max_n = target_sr * max_seconds
    if wav.shape[-1] > max_n:
        wav = wav[..., :max_n]
    return wav.squeeze(0).contiguous().float()  # (N,)


def load_for_both(path: Path) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Load once at 44.1k, then derive 24k via resample."""
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        return None, None
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != DAC_SR:
        wav44 = torchaudio.functional.resample(wav, sr, DAC_SR)
    else:
        wav44 = wav
    max_n44 = DAC_SR * CLIP_SECONDS
    if wav44.shape[-1] > max_n44:
        wav44 = wav44[..., :max_n44]
    wav24 = torchaudio.functional.resample(wav44, DAC_SR, ENC_SR)
    return wav44.squeeze(0).contiguous().float(), wav24.squeeze(0).contiguous().float()


@torch.inference_mode()
def encode_dac_batch(model, wavs: list[torch.Tensor], device: str) -> list[np.ndarray]:
    """Pad to max-length, encode (fp16 autocast), then trim back to per-clip valid length."""
    lens = [w.shape[-1] for w in wavs]
    L = max(lens)
    hop = model.hop_length
    L = ((L + hop - 1) // hop) * hop
    padded = torch.zeros(len(wavs), 1, L, device=device)
    for i, w in enumerate(wavs):
        padded[i, 0, : w.shape[-1]] = w.to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        sig = model.preprocess(padded, model.sample_rate)
        z, codes, latents, _, _ = model.encode(sig)
    out = []
    for i, raw_n in enumerate(lens):
        T_valid = (raw_n + hop - 1) // hop
        out.append(codes[i, :, :T_valid].to(torch.int16).cpu().numpy())
    return out


@torch.inference_mode()
def encode_encodec_batch(model, wavs: list[torch.Tensor], device: str) -> list[np.ndarray]:
    """EnCodec 24k expects (B, 1, N). fp16 autocast. Returns codes (B, n_q, T)."""
    lens = [w.shape[-1] for w in wavs]
    L = max(lens)
    hop = int(round(model.sample_rate / model.frame_rate))
    L = ((L + hop - 1) // hop) * hop
    padded = torch.zeros(len(wavs), 1, L, device=device)
    for i, w in enumerate(wavs):
        padded[i, 0, : w.shape[-1]] = w.to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        encoded = model.encode(padded)
    codes = torch.cat([c for c, _ in encoded], dim=-1)
    out = []
    for i, raw_n in enumerate(lens):
        T_valid = (raw_n + hop - 1) // hop
        out.append(codes[i, :, :T_valid].to(torch.int16).cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--io-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--meta-csv", type=str, default=None)
    ap.add_argument("--dac-dir", type=str, default=None)
    ap.add_argument("--enc-dir", type=str, default=None)
    ap.add_argument("--clip-sec", type=int, default=None)
    ap.add_argument("--skip-dac", action="store_true", help="only extract EnCodec (faster)")
    ap.add_argument("--skip-encodec", action="store_true", help="only extract DAC")
    args = ap.parse_args()

    global META, DAC_DIR, ENC_DIR, CLIP_SECONDS
    if args.meta_csv: META = Path(args.meta_csv)
    if args.dac_dir: DAC_DIR = Path(args.dac_dir)
    if args.enc_dir: ENC_DIR = Path(args.enc_dir)
    if args.clip_sec: CLIP_SECONDS = args.clip_sec
    print(f"[cfg] META={META} DAC={DAC_DIR} ENC={ENC_DIR} CLIP={CLIP_SECONDS}s "
          f"skip_dac={args.skip_dac} skip_enc={args.skip_encodec}")

    if not args.skip_dac:
        DAC_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_encodec:
        ENC_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    rows = []
    with META.open() as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows = rows[args.shard :: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(rows)} clips")

    # Load codecs (skip the ones we don't need)
    dac_model = None
    enc_model = None
    if not args.skip_dac:
        print("[load] DAC 44.1k 8kbps...")
        dac_model = dac.DAC.load(dac.utils.download(model_type="44khz")).to(args.device).eval()
    if not args.skip_encodec:
        print("[load] EnCodec 24k 6kbps...")
        enc_model = EncodecModel.encodec_model_24khz()
        enc_model.set_target_bandwidth(6.0)  # 8 codebooks
        enc_model = enc_model.to(args.device).eval()

    # Async I/O via thread pool
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=args.io_workers)

    t0 = time.time()
    n_done = n_skip = n_fail = 0
    last_print = t0

    BS = args.batch_size

    def submit_load(row):
        return pool.submit(load_for_both, resolve_audio_path(row["filepath"]))

    # Pipeline: prefetch BS items ahead
    pending = []
    cursor = 0
    while cursor < BS and cursor < len(rows):
        pending.append((rows[cursor], submit_load(rows[cursor])))
        cursor += 1

    while pending:
        # Drain to BS items, also skip those whose outputs already exist
        batch_rows = []
        batch_w44 = []
        batch_w24 = []
        while pending and len(batch_rows) < BS:
            row, fut = pending.pop(0)
            uid = row["utt_id"]
            dac_out = DAC_DIR / f"{uid}.npy"
            enc_out = ENC_DIR / f"{uid}.npy"
            dac_done = args.skip_dac or dac_out.exists()
            enc_done = args.skip_encodec or enc_out.exists()
            if dac_done and enc_done:
                fut.cancel()
                n_skip += 1
                if cursor < len(rows):
                    pending.append((rows[cursor], submit_load(rows[cursor])))
                    cursor += 1
                continue
            w44, w24 = fut.result()
            if w44 is None or w44.numel() < 4410:  # < 0.1 s of audio
                n_fail += 1
                if cursor < len(rows):
                    pending.append((rows[cursor], submit_load(rows[cursor])))
                    cursor += 1
                continue
            batch_rows.append(row)
            batch_w44.append(w44)
            batch_w24.append(w24)
            if cursor < len(rows):
                pending.append((rows[cursor], submit_load(rows[cursor])))
                cursor += 1

        if not batch_rows:
            continue

        dac_codes_list = [None] * len(batch_rows)
        enc_codes_list = [None] * len(batch_rows)
        try:
            if not args.skip_dac:
                dac_codes_list = encode_dac_batch(dac_model, batch_w44, args.device)
            if not args.skip_encodec:
                enc_codes_list = encode_encodec_batch(enc_model, batch_w24, args.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Fallback: process one-by-one
            dac_codes_list = [None] * len(batch_rows)
            enc_codes_list = [None] * len(batch_rows)
            for i, (w44, w24) in enumerate(zip(batch_w44, batch_w24)):
                if not args.skip_dac:
                    dac_codes_list[i] = encode_dac_batch(dac_model, [w44], args.device)[0]
                if not args.skip_encodec:
                    enc_codes_list[i] = encode_encodec_batch(enc_model, [w24], args.device)[0]

        for row, dc, ec in zip(batch_rows, dac_codes_list, enc_codes_list):
            uid = row["utt_id"]
            if not args.skip_dac:
                np.save(DAC_DIR / f"{uid}.npy", dc)
            if not args.skip_encodec:
                np.save(ENC_DIR / f"{uid}.npy", ec)
            n_done += 1

        now = time.time()
        if now - last_print > 10:
            total = n_done + n_skip + n_fail
            rate = n_done / max(now - t0, 1e-6)
            remaining = len(rows) - total
            eta_h = remaining / max(rate, 1e-6) / 3600
            print(
                f"  [shard {args.shard}] {total}/{len(rows)} "
                f"done={n_done} skip={n_skip} fail={n_fail}  "
                f"{rate:.1f} clips/s  eta={eta_h:.2f}h",
                flush=True,
            )
            last_print = now

    pool.shutdown(wait=True)
    print(
        f"[shard {args.shard} done] elapsed={(time.time()-t0)/3600:.2f}h "
        f"done={n_done} skip={n_skip} fail={n_fail}"
    )


if __name__ == "__main__":
    main()
