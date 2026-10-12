"""Path resolution for CoMoE.

Large artefacts (audio, token caches, continuous features) are *not* part of the
repository.  They live under one directory pointed to by the environment
variable ``COMOE_DATA_ROOT`` (default: ``<repo>/data_root``)::

    $COMOE_DATA_ROOT/
      normalized_30s/<source>/<utt_id>.mp3                 format-normalised MoM-open audio
      tokens_mom_norm/encodec24k/<utt_id>.npy              (8, T)  int16
      tokens_mom_norm/dac44k/<utt_id>.npy                  (9, T)  int16
      tokens_mom_norm/yue_xcodec_L0_L1_L10_L11/<utt_id>.npy (4, T) int16
      tokens_mom_norm/mert_v0_kmeans1024_L0_L1_L11_L12/<utt_id>.npy            (4, T) int16
      tokens_mom_norm/mert_v0_kmeans1024_REAL_ONLY_L0_L1_L11_L12/<utt_id>.npy  (4, T) int16
      feats_mom_norm/mert_v0/<utt_id>.npy                  (13, 768) fp16  (pooled, MLP baseline)
      feats_mom_norm/mert_v0_pooled.npz                    aggregate of the above
      feats_mom_norm/mert_v0_frames_L0_L1_L11_L12_T300/<utt_id>.npy (4, 300, 768) fp16 (MERT-continuous)
      mert_frame_cache/<utt_id>.npy                        (13, T, 768) fp16, k-means fitting only

Config files refer to that directory as ``${COMOE_DATA_ROOT}``.  Every other
path in a config (split JSONs, metadata CSV, run directory) is relative to the
repository root.

``COMOE_RAW_ROOT`` (default: the parent of ``COMOE_DATA_ROOT``) is only needed by
``data/normalize_mom.py``: it is the directory holding the un-normalised
MoM-CLAM download and FMA-medium, as listed in ``data/splits/metadata_mom_raw.csv``.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT_VAR = "COMOE_DATA_ROOT"
RAW_ROOT_VAR = "COMOE_RAW_ROOT"

PathLike = Union[str, os.PathLike]


def data_root() -> Path:
    return Path(os.environ.get(DATA_ROOT_VAR, REPO_ROOT / "data_root")).expanduser().resolve()


def raw_root() -> Path:
    return Path(os.environ.get(RAW_ROOT_VAR, data_root().parent)).expanduser().resolve()


def resolve(p: Optional[PathLike]) -> Optional[str]:
    """Expand ``${COMOE_DATA_ROOT}`` and make relative paths repo-relative."""
    if p is None:
        return None
    s = str(p)
    s = s.replace("${" + DATA_ROOT_VAR + "}", str(data_root()))
    s = s.replace("$" + DATA_ROOT_VAR, str(data_root()))
    s = os.path.expandvars(os.path.expanduser(s))
    q = Path(s)
    return str(q if q.is_absolute() else REPO_ROOT / q)


def resolve_audio_path(p: PathLike, root: Optional[Path] = None) -> Path:
    """Audio paths in the metadata CSVs are relative to the data root."""
    q = Path(p)
    if q.is_absolute():
        return q
    return (root if root is not None else data_root()) / q


# config keys that hold paths
PATH_KEYS = {
    "data": ["token_dir", "meta_csv", "split_train", "split_val", "split_ood",
             "semantic_npz", "semantic_dir"],
    "train": ["save_dir"],
}


def resolve_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``cfg`` with every path key resolved to an absolute path."""
    out = copy.deepcopy(cfg)
    for section, keys in PATH_KEYS.items():
        d = out.get(section) or {}
        for k in keys:
            if d.get(k):
                d[k] = resolve(d[k])
    return out
