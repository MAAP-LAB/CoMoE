"""Extract time-pooled MERT-v0-public features (13 layers x 768) for every clip.

Used by the ``MLP (MERT)`` baseline and the HLS ablation (semantic stream).
MERT-v0-public: 16 kHz input, 12 Transformer layers + input embedding = 13
hidden states of size 768.

Output: $COMOE_DATA_ROOT/feats_mom_norm/mert_v0/<utt_id>.npy  (13, 768) fp16
Then run tokenize/aggregate_mert_pooled.py to build mert_v0_pooled.npz.
Resumable (skips existing files).

Usage:  python tokenize/extract_mert_pooled.py [--shard i --num-shards n]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchaudio

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `comoe` imports without pip install
from comoe.paths import REPO_ROOT, data_root, resolve_audio_path  # noqa: E402

csv.field_size_limit(sys.maxsize)
torch.backends.cudnn.benchmark = True

MERT_SR = 16_000
MERT_SEC = 30

OUT_ROOT = data_root() / "feats_mom_norm"
MERT_DIR = OUT_ROOT / "mert_v0"
META = REPO_ROOT / "data" / "splits" / "metadata_mom_norm.csv"
MODEL_NAME = "m-a-p/MERT-v0-public"

# patched by main() if --out-root / --meta-csv given


def load_wav(path: Path) -> np.ndarray | None:
    try:
        info = torchaudio.info(str(path))
        n = int(info.sample_rate * MERT_SEC) + info.sample_rate
        wav, sr = torchaudio.load(str(path), num_frames=n)
    except Exception:
        try:
            wav, sr = torchaudio.load(str(path))
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


@torch.inference_mode()
def encode_batch(model, fe, wavs: list[np.ndarray], device: str) -> np.ndarray:
    inputs = fe(wavs, sampling_rate=MERT_SR, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(**inputs, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=1)  # (B, 13, T, 768)
    pooled = hs.mean(dim=2)                     # (B, 13, 768)
    return pooled.half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--io-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--meta-csv", type=str, default=None,
                    help="override metadata CSV (must have utt_id,filepath,...)")
    ap.add_argument("--out-root", type=str, default=None,
                    help="override feature root; per-utt files go to <root>/mert_v0/")
    args = ap.parse_args()

    global META, OUT_ROOT, MERT_DIR
    if args.meta_csv:
        META = Path(args.meta_csv)
    if args.out_root:
        OUT_ROOT = Path(args.out_root)
        MERT_DIR = OUT_ROOT / "mert_v0"
    print(f"[cfg] META={META}  MERT_DIR={MERT_DIR}")

    MERT_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(META.open()))
    rows = rows[args.shard :: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(rows)} clips")

    from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor

    print(f"[load] {MODEL_NAME}...")
    cfg = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Newer transformers' HubertEncoder expects conv_pos_batch_norm; MERTConfig lacks it.
    cfg.conv_pos_batch_norm = False
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = (
        AutoModel.from_pretrained(MODEL_NAME, config=cfg, trust_remote_code=True)
        .to(args.device)
        .eval()
    )
    assert fe.sampling_rate == MERT_SR, f"MERT-v0 SR mismatch: {fe.sampling_rate}"

    pool = ThreadPoolExecutor(max_workers=args.io_workers)

    def submit_load(row):
        return pool.submit(load_wav, resolve_audio_path(row["filepath"]))

    BS = args.batch_size
    pending: list[tuple[dict, object]] = []
    cursor = 0
    while cursor < BS and cursor < len(rows):
        pending.append((rows[cursor], submit_load(rows[cursor])))
        cursor += 1

    t0 = time.time()
    n_done = n_skip = n_fail = 0
    last_print = t0

    while pending:
        batch_rows = []
        batch_w = []
        while pending and len(batch_rows) < BS:
            row, fut = pending.pop(0)
            uid = row["utt_id"]
            out_path = MERT_DIR / f"{uid}.npy"
            if out_path.exists():
                fut.cancel()
                n_skip += 1
                if cursor < len(rows):
                    pending.append((rows[cursor], submit_load(rows[cursor])))
                    cursor += 1
                continue
            w = fut.result()
            if w is None:
                n_fail += 1
                if cursor < len(rows):
                    pending.append((rows[cursor], submit_load(rows[cursor])))
                    cursor += 1
                continue
            batch_rows.append(row)
            batch_w.append(w)
            if cursor < len(rows):
                pending.append((rows[cursor], submit_load(rows[cursor])))
                cursor += 1

        if not batch_rows:
            continue

        try:
            feats = encode_batch(model, fe, batch_w, args.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            feats = np.concatenate(
                [encode_batch(model, fe, [w], args.device) for w in batch_w], 0
            )

        for row, f in zip(batch_rows, feats):
            uid = row["utt_id"]
            np.save(MERT_DIR / f"{uid}.npy", f.astype(np.float16))
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
