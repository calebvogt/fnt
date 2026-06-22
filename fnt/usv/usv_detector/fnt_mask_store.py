"""Shared HDF5 storage for FNT mask data (used by both CAD and MAD).

The **CSV remains the canonical tabular output** for every tool (boxes,
features, harmonic links). These HDF5 files only hold *pixel data* — per-call
binary masks (for drawing on the spectrogram and re-editing) and, for MAD, the
full-grid probability map (so predictions can be re-thresholded without
re-running inference).

Two kinds of store:

1. **Per-wav sibling** ``<base>_FNT_masks.h5`` — companion to a wav's CSV::

       /                      attrs: sample_rate, nperseg, noverlap, nfft,
                                     n_freq_bins, n_time_frames, schema_version
       /calls/<call_id>       uint8 [h, w] cropped binary mask (gzip)
                              attrs: f_off, t_off  (offset on the full spec grid)
       /prob                  float16 [F, T] full-grid probability map (optional)

   ``call_id`` is the stable id stored in the CSV, so a CSV row and its mask
   are linked without relying on row order.

2. **Consolidated training store** ``models/training_data.h5`` (MAD) — every
   confirmed per-call example in one file (replaces thousands of PNG/JSON
   triplets)::

       /examples/<example_id>/spec   uint8 [H, W] normalized spec patch (gzip)
       /examples/<example_id>/mask   uint8 [H, W] binary mask (gzip)
       /examples/<example_id>        attrs: meta_json (JSON string)

Heavy import (``h5py``) is module-level; it's already a project dependency.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import h5py
    _HAS_H5 = True
except Exception:  # pragma: no cover
    _HAS_H5 = False

SCHEMA_VERSION = 1
MASKS_SUFFIX = "_FNT_masks.h5"
TRAINING_STORE_NAME = "training_data.h5"
_GRID_KEYS = ("sample_rate", "nperseg", "noverlap", "nfft",
              "n_freq_bins", "n_time_frames")


def _require_h5():
    if not _HAS_H5:
        raise RuntimeError(
            "h5py is required for FNT mask storage. Install with: pip install h5py"
        )


# ======================================================================
# Per-wav sibling mask store  (<base>_FNT_masks.h5)
# ======================================================================
def masks_sibling_path(wav_path: str) -> str:
    """Return the ``<base>_FNT_masks.h5`` path next to ``wav_path``."""
    p = Path(wav_path)
    return str(p.with_name(p.stem + MASKS_SUFFIX))


def set_grid_attrs(h5_path: str, **params) -> None:
    """Record spectrogram-grid params at the file root (idempotent)."""
    _require_h5()
    with h5py.File(h5_path, "a") as f:
        for k in _GRID_KEYS:
            if k in params and params[k] is not None:
                f.attrs[k] = int(params[k])
        f.attrs["schema_version"] = SCHEMA_VERSION


def get_grid_attrs(h5_path: str) -> Dict:
    """Return the recorded grid params, or {} if the file/attrs are absent."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return {}
    out: Dict = {}
    with h5py.File(h5_path, "r") as f:
        for k in _GRID_KEYS:
            if k in f.attrs:
                out[k] = int(f.attrs[k])
        if "schema_version" in f.attrs:
            out["schema_version"] = int(f.attrs["schema_version"])
    return out


def write_call_mask(h5_path: str, call_id, mask: np.ndarray,
                    f_off: int, t_off: int) -> None:
    """Store one call's cropped binary mask (in-place; overwrites if present)."""
    _require_h5()
    m = (np.asarray(mask) > 0).astype(np.uint8)
    with h5py.File(h5_path, "a") as f:
        grp = f.require_group("calls")
        key = str(call_id)
        if key in grp:
            del grp[key]
        ds = grp.create_dataset(key, data=m, compression="gzip",
                                compression_opts=4)
        ds.attrs["f_off"] = int(f_off)
        ds.attrs["t_off"] = int(t_off)


def read_call_mask(h5_path: str, call_id) -> Optional[Dict]:
    """Return ``{mask(bool), f_off, t_off}`` for ``call_id`` or None."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return None
    with h5py.File(h5_path, "r") as f:
        grp = f.get("calls")
        if grp is None or str(call_id) not in grp:
            return None
        ds = grp[str(call_id)]
        return {
            "mask": ds[()].astype(bool),
            "f_off": int(ds.attrs.get("f_off", 0)),
            "t_off": int(ds.attrs.get("t_off", 0)),
        }


def read_all_call_masks(h5_path: str) -> Dict[str, Dict]:
    """Return ``{call_id: {mask, f_off, t_off}}`` for every stored call."""
    _require_h5()
    out: Dict[str, Dict] = {}
    if not os.path.isfile(h5_path):
        return out
    with h5py.File(h5_path, "r") as f:
        grp = f.get("calls")
        if grp is None:
            return out
        for key in grp:
            ds = grp[key]
            out[key] = {
                "mask": ds[()].astype(bool),
                "f_off": int(ds.attrs.get("f_off", 0)),
                "t_off": int(ds.attrs.get("t_off", 0)),
            }
    return out


def delete_call_mask(h5_path: str, call_id) -> None:
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    with h5py.File(h5_path, "a") as f:
        grp = f.get("calls")
        if grp is not None and str(call_id) in grp:
            del grp[str(call_id)]


def list_call_ids(h5_path: str) -> List[str]:
    _require_h5()
    if not os.path.isfile(h5_path):
        return []
    with h5py.File(h5_path, "r") as f:
        grp = f.get("calls")
        return list(grp.keys()) if grp is not None else []


def write_prob(h5_path: str, prob: np.ndarray) -> None:
    """Store the full-grid probability map as float16 (for re-thresholding)."""
    _require_h5()
    p = np.asarray(prob).astype(np.float16)
    with h5py.File(h5_path, "a") as f:
        if "prob" in f:
            del f["prob"]
        f.create_dataset("prob", data=p, compression="gzip", compression_opts=4)


def read_prob(h5_path: str) -> Optional[np.ndarray]:
    """Return the float32 probability map, or None if absent."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return None
    with h5py.File(h5_path, "r") as f:
        if "prob" not in f:
            return None
        return f["prob"][()].astype(np.float32)


# ======================================================================
# Consolidated MAD training-example store  (models/training_data.h5)
# ======================================================================
def training_store_path(models_dir: str) -> str:
    return os.path.join(models_dir, TRAINING_STORE_NAME)


def td_save_example(h5_path: str, spec_patch: np.ndarray,
                    mask_patch: np.ndarray, meta: Dict,
                    example_id: Optional[str] = None) -> str:
    """Persist one confirmed-call example; returns its id (mirrors
    :func:`fnt.usv.usv_detector.mad_examples.save_example`)."""
    _require_h5()
    import uuid
    if example_id is None:
        stem = Path(str(meta.get("source_wav", "ex"))).stem
        example_id = f"{stem}_{uuid.uuid4().hex[:10]}"
    spec = np.asarray(spec_patch)
    if spec.dtype != np.uint8:
        spec = (np.clip(spec, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    mask = (np.asarray(mask_patch) > 0).astype(np.uint8)
    meta = dict(meta)
    meta["id"] = example_id
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    with h5py.File(h5_path, "a") as f:
        ex = f.require_group("examples")
        if example_id in ex:
            del ex[example_id]
        g = ex.create_group(example_id)
        g.create_dataset("spec", data=spec, compression="gzip", compression_opts=4)
        g.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)
        g.attrs["meta_json"] = json.dumps(meta)
    return example_id


def td_iter_examples(h5_path: str) -> Iterator[Dict]:
    """Yield ``{meta, spec(float[0,1]), mask(float{0,1})}`` per example."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    with h5py.File(h5_path, "r") as f:
        ex = f.get("examples")
        if ex is None:
            return
        for key in ex:
            g = ex[key]
            try:
                meta = json.loads(g.attrs.get("meta_json", "{}"))
                spec = g["spec"][()].astype(np.float32) / 255.0
                mask = (g["mask"][()] > 0).astype(np.float32)
            except Exception:
                continue
            yield {"meta": meta, "spec": spec, "mask": mask}


def td_count(h5_path: str) -> int:
    _require_h5()
    if not os.path.isfile(h5_path):
        return 0
    with h5py.File(h5_path, "r") as f:
        ex = f.get("examples")
        return len(ex) if ex is not None else 0


def td_list_ids(h5_path: str) -> List[str]:
    """Return the example ids stored in the consolidated h5 (or [])."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return []
    with h5py.File(h5_path, "r") as f:
        ex = f.get("examples")
        return list(ex.keys()) if ex is not None else []


def td_delete(h5_path: str, example_id: str) -> None:
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    with h5py.File(h5_path, "a") as f:
        ex = f.get("examples")
        if ex is not None and example_id in ex:
            del ex[example_id]


def td_update_mask(h5_path: str, example_id: str, mask_patch: np.ndarray,
                   meta_updates: Optional[Dict] = None) -> None:
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    mask = (np.asarray(mask_patch) > 0).astype(np.uint8)
    with h5py.File(h5_path, "a") as f:
        ex = f.get("examples")
        if ex is None or example_id not in ex:
            return
        g = ex[example_id]
        if "mask" in g:
            del g["mask"]
        g.create_dataset("mask", data=mask, compression="gzip", compression_opts=4)
        if meta_updates:
            meta = json.loads(g.attrs.get("meta_json", "{}"))
            meta.update(meta_updates)
            g.attrs["meta_json"] = json.dumps(meta)


def td_update_class(h5_path: str, example_id: str, new_class: str) -> None:
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    with h5py.File(h5_path, "a") as f:
        ex = f.get("examples")
        if ex is None or example_id not in ex:
            return
        g = ex[example_id]
        meta = json.loads(g.attrs.get("meta_json", "{}"))
        meta["class"] = new_class
        g.attrs["meta_json"] = json.dumps(meta)


def td_list_classes(h5_path: str) -> List[str]:
    seen: List[str] = []
    for ex in td_iter_examples(h5_path):
        c = ex["meta"].get("class")
        if c and c not in seen:
            seen.append(c)
    return seen
