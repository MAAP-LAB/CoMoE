# CoMoE — Probing Token Spaces under Generator Shift in AI-Generated Music Detection

Code, split definitions and k-means codebooks for

> **Probing Token Spaces under Generator Shift in AI-Generated Music Detection**
> ICML 2026 Workshop on Machine Learning for Audio (non-archival) · [arXiv:2606.08663](https://arxiv.org/abs/2606.08663)

CoMoE (Codec-Mixture-of-Experts) is a small (~10 M parameter) classifier with a
fixed four-stream token interface.  Keeping the classifier fixed and swapping
only its input token space (EnCodec, DAC, X-Codec mini, MERT k-means, or
continuous MERT) isolates the effect of the token space.  Models are trained and
evaluated on **MoM-open**, an openly redistributable reconstruction of the
MoM-CLAM benchmark whose real side is FMA-medium + MTG-Jamendo, under the
standard split and under source-restricted (generator-shift) splits.

> **Status.** The code and split definitions are final.  Trained checkpoints
> and the per-run prediction files will be attached to a GitHub release; the
> results table in this README will be filled in at the same time.

## Repository layout

| Path | Contents |
|---|---|
| `comoe/` | model (`model.py`), dataset + collate (`data.py`), training (`train.py`), inference (`predict.py`), path handling (`paths.py`) |
| `configs/` | one YAML per run: `mom_<token space>_<variant>_norm[_<split>].yaml` |
| `tokenize/` | token / feature extraction for the five token spaces, k-means fitting |
| `data/` | MoM-open construction: format normalisation, split builders, and the shipped splits in `data/splits/` |
| `baselines/clam/` | re-implementation of CLAM (MERT-v1-95M + Wav2Vec2 streams, frozen encoders) |
| `scripts/` | result collection, per-source AUC, open-weight vs commercial analysis |
| `assets/` | the two MERT k-means codebooks (4 layers × 1024 centroids × 768) |

## Installation

```bash
git clone https://github.com/MAAP-LAB/CoMoE
cd CoMoE
pip install -r requirements.txt
pip install -e .          # makes `comoe` and `baselines` importable from anywhere
```

Every script expects to be run from the repository root.  Large artefacts
(audio, token caches, features) live outside the repository under one directory:

```bash
export COMOE_DATA_ROOT=/path/to/mom_open      # default: ./data_root
```

Config files refer to it as `${COMOE_DATA_ROOT}`; see `comoe/paths.py` for the
expected sub-directory layout.

## 1. Data: MoM-open

MoM-open keeps the fake side and the fake-generator protocol of
[MoM-CLAM](https://huggingface.co/datasets/...) and replaces the YouTube-derived
real audio with [FMA-medium](https://github.com/mdeff/fma) (CC-BY) and
[MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset) (CC-BY-NC).  Audio is
**not** redistributed here.  To rebuild it:

1. Download MoM-CLAM, FMA-medium and MTG-Jamendo.  `data/splits/metadata_mom_raw.csv`
   lists every clip with a path relative to `$COMOE_RAW_ROOT` (default: the parent
   of `$COMOE_DATA_ROOT`), e.g. `MoM-CLAM/suno_audio/<id>.mp3`, `fma_medium/<id>.mp3`.
2. Normalise every clip to the same container (mono, 24 kHz, 30 s centre crop,
   −14 LUFS, 192 kbps CBR mp3 without tags), which removes file-size and
   container shortcuts:
   ```bash
   python data/normalize_mom.py --workers 32
   ```
   This writes `$COMOE_DATA_ROOT/normalized_30s/<source>/<utt_id>.mp3` and
   regenerates `data/splits/metadata_mom_norm.csv`.

The split definitions are shipped in `data/splits/` as
`split_mom_{train,val,ood}_norm[_<tag>].json` (`{"real": [...], "fake": [...]}`):

| tag | condition | training fakes |
|---|---|---|
| *(none)* | base | all training generators |
| `losoA_fma2jam` / `losoB_jam2fma` | Real-FMA / Real-Jamendo | all, real side restricted to one corpus |
| `fakeA_sunov35` / `fakeB_udio` / `fakeC_diffrythm` | Fake-Suno3.5 / Fake-Udio / Fake-DiffRhythm | one generator family |
| `fakeOpen_ow` | Fake-Open | DiffRhythm + YuE (open weights), N = 8,884 |
| `fakeComm_matched` | Fake-Comm | Suno-v3.5 + Udio, N = 8,884 |
| `fakeComm1_suno` | Fake-Comm1 | Suno-v3.5 only, N = 8,884 (family-count control) |

The builders that produced them (`data/build_base_splits.py`,
`build_real_loso_splits.py`, `build_fake_loso_splits.py`,
`build_license_splits.py`) are included for reference; all use seed 42.

## 2. Token spaces

Each extractor writes one `<utt_id>.npy` per clip under `$COMOE_DATA_ROOT`
(resumable; shard across GPUs with the `--shard`/`--num-shards` or
`--shard_idx`/`--n_shards` options).

| token space | command | output |
|---|---|---|
| EnCodec 24 kHz, DAC 44 kHz | `python tokenize/tokenize_codecs.py` | `tokens_mom_norm/encodec24k`, `tokens_mom_norm/dac44k` |
| X-Codec mini (YuE) | `python tokenize/extract_xcodec_tokens.py` | `tokens_mom_norm/yue_xcodec_L0_L1_L10_L11` |
| MERT k-means | `python tokenize/extract_mert_kmeans_tokens.py` | `tokens_mom_norm/mert_v0_kmeans1024_L0_L1_L11_L12` |
| MERT k-means (real-only codebook) | `python tokenize/extract_mert_kmeans_tokens.py --centroids assets/mert_kmeans_centroids_REAL_ONLY_L0_L1_L11_L12.npz --out_dir $COMOE_DATA_ROOT/tokens_mom_norm/mert_v0_kmeans1024_REAL_ONLY_L0_L1_L11_L12` | |
| MERT continuous (ablation) | `python tokenize/extract_mert_frames.py` | `feats_mom_norm/mert_v0_frames_L0_L1_L11_L12_T300` |
| pooled MERT (MLP baseline) | `python tokenize/extract_mert_pooled.py && python tokenize/aggregate_mert_pooled.py` | `feats_mom_norm/mert_v0_pooled.npz` |

X-Codec mini is loaded from a clone of
[`m-a-p/xcodec_mini_infer`](https://huggingface.co/m-a-p/xcodec_mini_infer)
(`third_party/xcodec_mini_infer` or `$XCODEC_MINI_DIR`).  EnCodec, DAC and
MERT weights are downloaded by their libraries on first use.

The MERT k-means codebooks in `assets/` were fit with
`tokenize/fit_mert_kmeans.py` on frames cached by
`tokenize/build_mert_frame_cache.py` (50 clips per source).  The real-only
codebook used `--sources fma_medium jamendo`.

## 3. Training and evaluation

```bash
python -m comoe.train --config configs/mom_yue_xcodec_hl_norm_fakeA_sunov35.yaml
```

A run takes about 75 min on one H100 (12 epochs, batch 96, AdamW 2e-4, label
smoothing 0.05, seed 42).  The model is selected on validation F1 and evaluated
once on the OOD split; `runs/<name>/` receives `best.pt`, `train_log.json`,
`ood_result.json` (AUC, F1, accuracy, per-source accuracy) and a copy of the
config.  Any config value can be overridden on the command line, e.g.
`--override train.epochs=1 train.save_dir=runs/smoke`.

Per-clip scores (for threshold-free analyses) and the summary tables:

```bash
python -m comoe.predict --run_dir runs/mom_yue_xcodec_hl_norm_fakeA_sunov35   # -> preds_ood.csv
python scripts/collect_results.py            # OOD-AUC matrix over runs/
python scripts/per_source_auc.py             # per-generator AUC for the Fake-Open / Fake-Comm runs
python scripts/analyze_open_vs_commercial.py # open-weight vs commercial held-out accuracy
```

### Config naming

| config prefix | paper row | lower / higher streams |
|---|---|---|
| `mom_yue_xcodec_hl_norm` | CoMoE (X-Codec) | codebooks (0,1) / (10,11) |
| `mom_encodec_tokenonly_norm` | CoMoE (EnCodec) | codebooks (0,1) / (6,7) |
| `mom_dac_tokenonly_norm` | CoMoE (DAC) | codebooks (0,1) / (7,8) |
| `mom_mert_kmeans_hl_norm` | CoMoE (MERT k-means) | layers (0,1) / (11,12) |
| `mom_mert_kmeans_hl_norm_REAL_ONLY` | real-only codebook control | layers (0,1) / (11,12) |
| `mom_mert_continuous_hl_norm` | MERT-continuous | same layers, linear projection instead of the codebook embedding |
| `mom_semantic_only_mert_v0_norm` | MLP (MERT) baseline | pooled MERT → MLP |
| `mom_*_mert_v0_norm`, `mom_yue_xcodec_hls_norm` | HLS ablation | tokens + pooled-MERT semantic stream |

Suffixes `_losoA_fma2jam`, `_losoB_jam2fma`, `_fakeA_sunov35`, `_fakeB_udio`,
`_fakeC_diffrythm`, `_fakeOpen_ow`, `_fakeComm_matched`, `_fakeComm1_suno`
select the split; no suffix (or `_base`) is the base split.

### Baselines

* **MLP (MERT)**: `configs/mom_semantic_only_mert_v0_norm*.yaml` (needs the pooled MERT features).
* **CLAM**: `python -m baselines.clam.train --split_tag fakeA_sunov35 --out_dir runs/clam/fakeA_sunov35`
  (1 epoch, 30 s clips; `baselines/clam/cache_audio.py` pre-decodes audio to speed this up).

## 4. Checkpoints

To be attached to a GitHub release (CoMoE runs, ~40 MB each) and the Hugging
Face Hub (CLAM runs, ~0.8 GB each).  Each archive contains `best.pt`,
`config.yaml`, `ood_result.json` and `preds_ood.csv`, so the tables can be
recomputed without training.  Load a checkpoint with `python -m comoe.predict --run_dir <dir>`.

## Citation

```bibtex
@inproceedings{park2026comoe,
  title     = {Probing Token Spaces under Generator Shift in AI-Generated Music Detection},
  author    = {Park, Joonyong and others},
  booktitle = {ICML 2026 Workshop on Machine Learning for Audio},
  year      = {2026},
  note      = {arXiv:2606.08663}
}
```

## License

Code: *to be decided before the first release*.  The shipped split
definitions and metadata inherit the licenses of the underlying corpora
(MoM-CLAM: CC-BY-NC-4.0; MTG-Jamendo: CC-BY-NC; FMA: CC-BY).  X-Codec mini is
Apache-2.0 (m-a-p); EnCodec and MERT weights are CC-BY-NC-4.0.
