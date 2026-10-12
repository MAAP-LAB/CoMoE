"""Format-normalise the MoM-CLAM audio into MoM-open (identical container for every clip).

Input : data/splits/metadata_mom_raw.csv  (filepath relative to $COMOE_RAW_ROOT)
Output: $COMOE_DATA_ROOT/normalized_30s/<source>/<utt_id>.mp3
        data/splits/metadata_mom_norm.csv  (filepath relative to $COMOE_DATA_ROOT)
        data/splits/split_mom_{train,val,ood}_norm.json  (base splits filtered to surviving clips)

The shipped metadata_mom_norm.csv and split JSONs are the outputs of this
script; re-running it is only needed to rebuild the audio.

Usage:  python data/normalize_mom.py [--workers 48] [--durations_npz ...]

Pipeline per file:
  decode → float32 mono @ 24 kHz (librosa, soxr_hq)
  → drop if native duration < 29.5 s (mp3 frame rounding gives 29.98 for "30 s" FMA)
  → center-crop to exactly 30.0 s, or pad with zeros at tail (≤0.5 s pad)
  → pyloudnorm LUFS integrated → target -14 LUFS
  → hard clip [-1, 1]
  → ffmpeg libmp3lame -b:a 192k -map_metadata -1  (CBR, no tags)

All files become identical-size (~720 KB). Drops the `size` and `ext` leaks.

Resumable: skip files whose destination mp3 already exists.
Uses durations.npz (pre-filter <29.5 s) to avoid touching too-short clips.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `comoe` imports without pip install
from comoe.paths import REPO_ROOT, data_root, raw_root, resolve_audio_path  # noqa: E402

csv.field_size_limit(sys.maxsize)
warnings.filterwarnings("ignore")

SPLITS = REPO_ROOT / "data" / "splits"
META = SPLITS / "metadata_mom_raw.csv"
OUT_ROOT = data_root() / "normalized_30s"
OUT_META = SPLITS / "metadata_mom_norm.csv"
OUT_SPLITS = {
    "train": SPLITS / "split_mom_train_norm.json",
    "val":   SPLITS / "split_mom_val_norm.json",
    "ood":   SPLITS / "split_mom_ood_norm.json",
}
IN_SPLITS = {
    "train": SPLITS / "split_mom_train.json",
    "val":   SPLITS / "split_mom_val.json",
    "ood":   SPLITS / "split_mom_ood.json",
}

TARGET_SR = 24000
TARGET_DUR = 30.0
MIN_DUR = 29.5
TARGET_LUFS = -14.0
TARGET_LEN = int(TARGET_SR * TARGET_DUR)


def normalise_one(args: tuple[str, str, str]) -> tuple[str, int, str]:
    """Returns (utt_id, status, msg). status: 0=ok, 1=skip_exist, 2=too_short, 3=err."""
    uid, src_path, source = args
    dst_path = OUT_ROOT / source / f"{uid}.mp3"
    if dst_path.exists() and dst_path.stat().st_size > 10_000:
        return uid, 1, "exist"
    try:
        import librosa
        import pyloudnorm as pyln
        wav, _ = librosa.load(src_path, sr=TARGET_SR, mono=True, res_type="soxr_hq")
        native_len = len(wav)
        if native_len < int(MIN_DUR * TARGET_SR):
            return uid, 2, f"too_short {native_len/TARGET_SR:.2f}s"
        if native_len > TARGET_LEN:
            start = (native_len - TARGET_LEN) // 2
            wav = wav[start:start + TARGET_LEN]
        elif native_len < TARGET_LEN:
            wav = np.pad(wav, (0, TARGET_LEN - native_len))
        try:
            meter = pyln.Meter(TARGET_SR)
            loudness = meter.integrated_loudness(wav.astype(np.float64))
            if np.isfinite(loudness):
                wav = pyln.normalize.loudness(wav, loudness, TARGET_LUFS)
        except Exception:
            pass
        wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "f32le", "-ar", str(TARGET_SR), "-ac", "1", "-i", "pipe:0",
             "-c:a", "libmp3lame", "-b:a", "192k",
             "-map_metadata", "-1", "-fflags", "+bitexact",
             "-id3v2_version", "0", "-write_xing", "0",
             str(dst_path)],
            input=wav.tobytes(), capture_output=True, timeout=60,
        )
        if proc.returncode != 0:
            return uid, 3, f"ffmpeg rc={proc.returncode}: {proc.stderr[:200].decode(errors='ignore')}"
        return uid, 0, "ok"
    except Exception as e:
        return uid, 3, f"{type(e).__name__}: {str(e)[:120]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="process only first N (for testing)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--durations_npz", type=str, default=None,
                    help="optional cache {utt_ids, durations}; otherwise durations are read from the files")
    args = ap.parse_args()

    # Load metadata + durations
    meta_rows = list(csv.DictReader(META.open()))
    print(f"metadata: {len(meta_rows)} rows")

    if args.durations_npz:
        durs = np.load(args.durations_npz)
        dur_map = {u: d for u, d in zip(durs["utt_ids"], durs["durations"])}
        print(f"durations loaded: {len(dur_map)} entries")
    else:
        import librosa
        print("reading native durations (pass --durations_npz to skip this step next time)")
        dur_map = {}
        for i, r in enumerate(meta_rows):
            try:
                dur_map[r["utt_id"]] = librosa.get_duration(path=str(resolve_audio_path(r["filepath"], raw_root())))
            except Exception:
                dur_map[r["utt_id"]] = -1.0
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(meta_rows)}", flush=True)

    # Filter: keep only native_dur >= MIN_DUR
    kept = []
    dropped_dur = 0
    for r in meta_rows:
        d = dur_map.get(r["utt_id"], -1.0)
        if d < MIN_DUR:
            dropped_dur += 1
            continue
        kept.append(r)
    print(f"after duration filter (>={MIN_DUR}s): {len(kept)} kept, {dropped_dur} dropped")

    if args.limit:
        kept = kept[: args.limit]
        print(f"--limit: processing only {len(kept)} files")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for src in sorted({r["source"] for r in kept}):
        (OUT_ROOT / src).mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[dry-run] would process:", len(kept))
        return

    # Pre-populate work (skip completed), shuffle so all sources advance in parallel
    work = []
    already = 0
    for r in kept:
        dst = OUT_ROOT / r["source"] / f"{r['utt_id']}.mp3"
        if dst.exists() and dst.stat().st_size > 10_000:
            already += 1
            continue
        work.append((r["utt_id"], str(resolve_audio_path(r["filepath"], raw_root())), r["source"]))
    import random
    random.Random(42).shuffle(work)
    print(f"already normalised: {already}  remaining: {len(work)} (shuffled)")

    t0 = time.time()
    n_ok = already
    n_short = 0
    n_err = 0
    errors: list[str] = []

    if work:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(normalise_one, item): item for item in work}
            done = 0
            for fu in as_completed(futs):
                uid, status, msg = fu.result()
                done += 1
                if status == 0 or status == 1:
                    n_ok += 1
                elif status == 2:
                    n_short += 1
                else:
                    n_err += 1
                    if len(errors) < 20:
                        errors.append(f"{uid}: {msg}")
                if done % 500 == 0:
                    dt = time.time() - t0
                    rate = done / dt
                    eta_min = (len(work) - done) / max(rate, 1e-6) / 60
                    print(f"  {done}/{len(work)}  rate={rate:.1f} it/s  eta={eta_min:.1f} min  "
                          f"ok={n_ok}  short={n_short}  err={n_err}", flush=True)

    dt = time.time() - t0
    print(f"\n=== DONE in {dt/60:.1f} min ===")
    print(f"ok (incl. pre-existing): {n_ok}")
    print(f"too short (dropped):     {n_short}")
    print(f"errors:                  {n_err}")
    if errors:
        print("first errors:")
        for e in errors:
            print(f"  {e}")

    # Build normalised metadata + split JSONs
    print("\nBuilding normalised metadata + splits...")
    dropped_uids: set[str] = set()
    norm_rows = []
    for r in kept:
        uid = r["utt_id"]
        dst = OUT_ROOT / r["source"] / f"{uid}.mp3"
        if not (dst.exists() and dst.stat().st_size > 10_000):
            dropped_uids.add(uid)
            continue
        r_new = dict(r)
        r_new["filepath"] = str(dst.relative_to(data_root()))
        norm_rows.append(r_new)
    print(f"norm metadata rows: {len(norm_rows)}  (dropped {len(dropped_uids)} failed/short)")

    # write metadata_mom_norm.csv
    with OUT_META.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(norm_rows[0].keys()))
        w.writeheader()
        for r in norm_rows:
            w.writerow(r)
    print(f"wrote {OUT_META}")

    # filter split JSONs
    norm_uids = {r["utt_id"] for r in norm_rows}
    for split_name, in_path in IN_SPLITS.items():
        d = json.load(in_path.open())
        nd = {lab: [u for u in ids if u in norm_uids] for lab, ids in d.items()}
        out_path = OUT_SPLITS[split_name]
        out_path.write_text(json.dumps(nd, indent=2))
        print(f"  {split_name}: {sum(len(v) for v in d.values())} → "
              f"{sum(len(v) for v in nd.values())}  → {out_path}")


if __name__ == "__main__":
    main()
