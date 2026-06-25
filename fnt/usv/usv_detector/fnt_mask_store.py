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
# Optional per-wav redirect for the mask store (see mad_labels for the CSV
# equivalent). MAD points "browsed-in-place" files at a scratch h5 so their
# masks/predictions don't litter the original recording folder. Empty by
# default → unchanged behavior for CAD and graduated files.
_MASK_PATH_OVERRIDES: Dict[str, str] = {}


def set_mask_path_override(wav_path: str, h5_path: str) -> None:
    _MASK_PATH_OVERRIDES[os.path.normpath(wav_path)] = h5_path


def clear_mask_path_override(wav_path: str) -> None:
    _MASK_PATH_OVERRIDES.pop(os.path.normpath(wav_path), None)


def clear_all_mask_path_overrides() -> None:
    _MASK_PATH_OVERRIDES.clear()


def masks_sibling_path(wav_path: str) -> str:
    """Return the ``<base>_FNT_masks.h5`` path for ``wav_path`` (an override
    location if one is registered, else the sibling next to the wav)."""
    ov = _MASK_PATH_OVERRIDES.get(os.path.normpath(wav_path))
    if ov is not None:
        return ov
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


# ----------------------------------------------------------------------
# Per-blob prediction-mask crops  (/pred_calls/<blob_id>)
# ----------------------------------------------------------------------
# These are the small, thresholded pixel masks for each predicted call,
# carved out of the probability grid **once** at inference time. They let a
# file's predictions be redrawn on switch by reading a few MB of crops instead
# of decompressing the full-grid /prob map (~1 GB). MAD does not support
# re-thresholding predictions after inference (re-run inference instead), so
# the full /prob grid is no longer persisted — see MAD_README.md.
PRED_GROUP = "pred_calls"


def write_pred_masks(h5_path: str, crops: List[Dict]) -> None:
    """Replace the stored prediction crops with ``crops`` (one file open).

    Each crop is a dict ``{blob_id, mask, f_off, t_off}`` where ``mask`` is the
    cropped binary (thresholded) blob and ``f_off``/``t_off`` are its top-left
    offsets on the full spec grid. Also caches ``n_pred_blobs`` so file lists
    can show the count without reading any crop.
    """
    _require_h5()
    os.makedirs(os.path.dirname(h5_path) or ".", exist_ok=True)
    with h5py.File(h5_path, "a") as f:
        if PRED_GROUP in f:
            del f[PRED_GROUP]
        grp = f.require_group(PRED_GROUP)
        for c in crops:
            m = (np.asarray(c["mask"]) > 0).astype(np.uint8)
            key = str(c["blob_id"])
            ds = grp.create_dataset(key, data=m, compression="gzip",
                                    compression_opts=4)
            ds.attrs["f_off"] = int(c.get("f_off", 0))
            ds.attrs["t_off"] = int(c.get("t_off", 0))
        f.attrs["n_pred_blobs"] = int(len(crops))


def read_all_pred_masks(h5_path: str) -> Dict[str, Dict]:
    """Return ``{blob_id: {mask(bool), f_off, t_off}}`` for every stored
    prediction crop (cheap — only small per-blob arrays, never the grid)."""
    _require_h5()
    out: Dict[str, Dict] = {}
    if not os.path.isfile(h5_path):
        return out
    with h5py.File(h5_path, "r") as f:
        grp = f.get(PRED_GROUP)
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


def has_pred_masks(h5_path: str) -> bool:
    """True if the file holds per-blob prediction crops (cheap metadata read)."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            grp = f.get(PRED_GROUP)
            return grp is not None and len(grp) > 0
    except Exception:
        return False


def delete_pred_mask(h5_path: str, blob_id) -> None:
    """Remove one prediction crop by blob_id (no-op if absent). Used when a
    prediction is *deleted* (vs rejected, which keeps its crop)."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    try:
        with h5py.File(h5_path, "a") as f:
            grp = f.get(PRED_GROUP)
            if grp is not None and str(blob_id) in grp:
                del grp[str(blob_id)]
    except Exception:
        pass


def delete_pred_masks(h5_path: str, blob_ids) -> None:
    """Remove several prediction crops by blob_id in one file open (batch form
    of :func:`delete_pred_mask`). Updates the cached ``n_pred_blobs`` count."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    want = {str(b) for b in blob_ids}
    if not want:
        return
    try:
        with h5py.File(h5_path, "a") as f:
            grp = f.get(PRED_GROUP)
            if grp is None:
                return
            for key in list(grp.keys()):
                if key in want:
                    del grp[key]
            f.attrs["n_pred_blobs"] = int(len(grp))
    except Exception:
        pass


def clear_pred_masks(h5_path: str) -> None:
    """Delete all stored prediction crops (no-op if absent)."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    with h5py.File(h5_path, "a") as f:
        if PRED_GROUP in f:
            del f[PRED_GROUP]
        if "n_pred_blobs" in f.attrs:
            del f.attrs["n_pred_blobs"]


def delete_prob(h5_path: str) -> None:
    """Drop the legacy full-grid probability map and **reclaim the disk space**
    (no-op if absent). MAD no longer writes /prob; this runs when migrating old
    files to per-blob crops.

    ``del f["prob"]`` only unlinks the dataset — HDF5 leaves the freed bytes as
    unusable slack inside the file, so the ~1 GB grid would still occupy disk.
    To actually shrink, we repack: copy every object **except** ``prob`` (and
    the root attrs) into a fresh temp file, then atomically replace the original.
    """
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    try:
        with h5py.File(h5_path, "r") as f:
            if "prob" not in f:
                return
            keys = [k for k in f.keys() if k != "prob"]
        tmp = h5_path + ".repack.tmp"
        with h5py.File(h5_path, "r") as src, h5py.File(tmp, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            for k in keys:
                src.copy(k, dst, name=k)
        os.replace(tmp, h5_path)
    except Exception:
        # Best-effort: if repack fails, fall back to a plain unlink so the file
        # at least stops being read as a prob source.
        try:
            if os.path.isfile(h5_path + ".repack.tmp"):
                os.remove(h5_path + ".repack.tmp")
        except Exception:
            pass
        try:
            with h5py.File(h5_path, "a") as f:
                if "prob" in f:
                    del f["prob"]
        except Exception:
            pass


def write_prob(h5_path: str, prob: np.ndarray) -> None:
    """Store the full-grid probability map as float16 (for re-thresholding)."""
    _require_h5()
    p = np.asarray(prob).astype(np.float16)
    with h5py.File(h5_path, "a") as f:
        if "prob" in f:
            del f["prob"]
        f.create_dataset("prob", data=p, compression="gzip", compression_opts=4)
        # Stale blob count from a previous prob map; recomputed on demand.
        if "n_pred_blobs" in f.attrs:
            del f.attrs["n_pred_blobs"]


def read_prob(h5_path: str) -> Optional[np.ndarray]:
    """Return the float32 probability map, or None if absent."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return None
    with h5py.File(h5_path, "r") as f:
        if "prob" not in f:
            return None
        return f["prob"][()].astype(np.float32)


def has_prob(h5_path: str) -> bool:
    """True if the file holds a probability map (cheap — no array read)."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            return "prob" in f
    except Exception:
        return False


def get_prob_blob_count(h5_path: str) -> Optional[int]:
    """Return the cached prediction-blob count for the prob map, or None if
    it has not been computed yet.

    This is a cheap metadata read (an HDF5 attribute) — it never decompresses
    the probability grid, so it is safe to call for every file when populating
    a file list. The count is populated lazily by :func:`set_prob_blob_count`
    (at inference-write time, or the first time a file's predictions are
    loaded), so opening a project does not have to scan multi-GB prob maps.
    """
    _require_h5()
    if not os.path.isfile(h5_path):
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            # The count is a root attr cached at write time. It stays valid even
            # after the /prob grid is dropped (predictions now live as crops),
            # so don't gate the read on /prob's presence.
            v = f.attrs.get("n_pred_blobs")
            return int(v) if v is not None else None
    except Exception:
        return None


def set_prob_blob_count(h5_path: str, n: int) -> None:
    """Cache the prediction-blob count as a root attribute (cheap to read back
    later). No-op if the file is missing."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    try:
        with h5py.File(h5_path, "a") as f:
            f.attrs["n_pred_blobs"] = int(n)
    except Exception:
        pass


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


def td_iter_file_examples(h5_path: str, wav_name: str) -> Iterator[Dict]:
    """Yield examples for a single wav file, skipping heavy array reads for
    non-matching entries."""
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
            except Exception:
                continue
            src = os.path.basename(str(meta.get("source_wav", "")))
            if src != wav_name:
                continue
            try:
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


def td_count_by_source_wav(h5_path: str) -> Dict[str, int]:
    """Return ``{wav_basename: n_examples}`` reading **only** each example's
    metadata attribute — never decompressing the spec/mask arrays. Cheap enough
    to call for a whole project when populating a file list."""
    _require_h5()
    out: Dict[str, int] = {}
    if not os.path.isfile(h5_path):
        return out
    try:
        with h5py.File(h5_path, "r") as f:
            ex = f.get("examples")
            if ex is None:
                return out
            for key in ex:
                try:
                    meta = json.loads(ex[key].attrs.get("meta_json", "{}"))
                except Exception:
                    continue
                bn = os.path.basename(str(meta.get("source_wav", "")))
                if bn:
                    out[bn] = out.get(bn, 0) + 1
    except Exception:
        return out
    return out


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
