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
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import h5py
    _HAS_H5 = True
except Exception:  # pragma: no cover
    _HAS_H5 = False

SCHEMA_VERSION = 1

#: MAD's own document. HDF5 inside — the extension buys identity, not different
#: behaviour: locking, atomicity and space reclamation are properties of the
#: container format, not its name. What it does buy is a schema this project
#: owns and versions (``FORMAT_NAME`` / ``FORMAT_VERSION`` below), a file that
#: reads as "the document" rather than "some data", and a name that does not
#: invite hand-editing in a generic HDF5 viewer — which matters, because an
#: external program holding the file open is exactly what blocks MAD's writes.
#: Same idea as SLEAP's ``.slp``.
MAD_SUFFIX = "_FNT.mad"
#: Namings this file has had, newest first. All are read forever; only
#: :data:`MAD_SUFFIX` is written for a new recording — see
#: :func:`masks_sibling_path`. ``.mad`` was a brief interim naming; keeping it
#: readable costs one ``exists`` call and saves anyone who labelled during it.
LEGACY_MAD_SUFFIXES = (".mad", "_FNT_masks.h5")
#: The oldest naming, kept as its own constant because other modules import it.
LEGACY_MASKS_SUFFIX = "_FNT_masks.h5"
#: Back-compat alias for callers that imported the old constant.
MASKS_SUFFIX = LEGACY_MASKS_SUFFIX
#: Stamped at the root of every store so a file can say what it is, and so a
#: future layout change can be migrated deliberately instead of guessed at.
FORMAT_NAME = "FNT-MAD"
FORMAT_VERSION = 1


def _fnt_version() -> str:
    """Installed FNT version, or 'unknown' outside an installed package."""
    try:
        from importlib.metadata import version
        return version("fnt")
    except Exception:
        return "unknown"
TRAINING_STORE_NAME = "training_data.h5"
_GRID_KEYS = ("sample_rate", "nperseg", "noverlap", "nfft",
              "n_freq_bins", "n_time_frames")


def _require_h5():
    if not _HAS_H5:
        raise RuntimeError(
            "h5py is required for FNT mask storage. Install with: pip install h5py"
        )


# ----------------------------------------------------------------------
# Write-failure reporting
# ----------------------------------------------------------------------
# A failed write used to vanish. Most callers wrap store calls in
# ``except Exception: pass`` -- 28 of them -- which was tolerable while the CSV
# held a parallel copy of everything and is not once the h5 is the only record
# of a label. HDF5 also takes a whole-file lock, so a write fails outright if
# anything else has the file open (HDFView, a script, a second MAD): a real,
# recoverable condition that must never look like success.
#
# Every write reports through this hook BEFORE the exception propagates, so a
# caller that swallows the error cannot swallow the notification with it.
_WRITE_FAILURE_HOOK = None


def set_write_failure_hook(fn) -> None:
    """Register ``fn(operation, path, exception)``, called on every failed write.

    The GUI uses this to surface a failed save; without it a lock collision is
    indistinguishable from a successful one.
    """
    global _WRITE_FAILURE_HOOK
    _WRITE_FAILURE_HOOK = fn


def _reporting_write(fn):
    """Report a store write's failure through the hook, then re-raise."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            hook = _WRITE_FAILURE_HOOK
            if hook is not None:
                try:
                    path = args[0] if args and isinstance(args[0], str) else ""
                    hook(fn.__name__, path, exc)
                except Exception:
                    pass          # a broken hook must not mask the real error
            raise
    return wrapper


def is_locked_error(exc: Exception) -> bool:
    """True when a write failed because something else holds the file open.

    Worth separating from a genuine I/O error: it is transient and the fix is
    "close the other program", not "your data is damaged".
    """
    return "unable to lock file" in str(exc).lower()


# ----------------------------------------------------------------------
# Durability
# ----------------------------------------------------------------------
BACKUP_SUFFIX = ".bak"


def backup_store(h5_path: str) -> Optional[str]:
    """Copy the store aside before an operation that rewrites the whole file.

    HDF5 has no journal. An interrupted rewrite does not lose the last record,
    it can leave the container structurally unreadable — and these files live on
    a mapped network drive, where interruptions are a fact of life. One
    generation of backup is enough to turn "all labels gone" into "reopen
    yesterday's copy".
    """
    import shutil
    if not os.path.isfile(h5_path):
        return None
    dest = h5_path + BACKUP_SUFFIX
    try:
        shutil.copy2(h5_path, dest)
        return dest
    except Exception:
        return None            # a missing backup must not block the operation


def repack_store(h5_path: str, *, keep_backup: bool = True) -> Dict:
    """Rewrite the file without its dead space, atomically.

    Deleting an HDF5 object unlinks it but leaves the bytes as unusable slack:
    measured, deleting all 200 examples from a 3.7 MB store reclaimed exactly
    zero. Left alone the file grows forever, which is tolerable for a cache and
    not for the only copy of the labels.

    The rewrite goes to a temp file and is swapped in with ``os.replace``, so an
    interruption leaves the original untouched rather than half-written.
    """
    _require_h5()
    out = {"before": 0, "after": 0, "reclaimed": 0, "ok": False}
    if not os.path.isfile(h5_path):
        return out
    out["before"] = os.path.getsize(h5_path)
    tmp = h5_path + ".repack.tmp"
    try:
        with h5py.File(h5_path, "r") as src, h5py.File(tmp, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            for k in src.keys():
                src.copy(k, dst, name=k)
        if keep_backup:
            backup_store(h5_path)
        os.replace(tmp, h5_path)
        out["after"] = os.path.getsize(h5_path)
        out["reclaimed"] = out["before"] - out["after"]
        out["ok"] = True
    except Exception as exc:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except Exception:
            pass
        hook = _WRITE_FAILURE_HOOK
        if hook is not None:
            try:
                hook("repack_store", h5_path, exc)
            except Exception:
                pass
    return out


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
    """The MAD store beside ``wav_path`` (or a registered override).

    New recordings get ``<stem>.mad``. A recording that already has the older
    ``<stem>_FNT_masks.h5`` keeps using it: existing projects hold thousands,
    and quietly starting a second store next to a populated one would split a
    recording's labels across two files with neither being complete. Converting
    is a separate, explicit step — see :func:`migrate_to_mad_suffix`.
    """
    ov = _MASK_PATH_OVERRIDES.get(os.path.normpath(wav_path))
    if ov is not None:
        return ov
    p = Path(wav_path)
    current = p.with_name(p.stem + MAD_SUFFIX)
    if current.exists():
        return str(current)
    for suffix in LEGACY_MAD_SUFFIXES:
        older = p.with_name(p.stem + suffix)
        if older.exists():
            return str(older)
    return str(current)


def store_paths_for(wav_path: str) -> List[str]:
    """Every store path that could belong to ``wav_path``, newest naming first.

    For callers that clean up or move a recording's siblings and must not miss
    one because it predates the rename.
    """
    p = Path(wav_path)
    return [str(p.with_name(p.stem + s))
            for s in (MAD_SUFFIX,) + LEGACY_MAD_SUFFIXES]


def is_mad_store(path: str) -> bool:
    """True if ``path`` looks like a MAD store by name."""
    n = str(path).lower()
    return any(n.endswith(s.lower())
               for s in (MAD_SUFFIX,) + LEGACY_MAD_SUFFIXES)


def stamp_format(h5_path: str) -> None:
    """Record what this file is and which layout it uses."""
    _require_h5()
    try:
        with h5py.File(h5_path, "a") as f:
            f.attrs["format"] = FORMAT_NAME
            f.attrs["format_version"] = FORMAT_VERSION
    except Exception:
        pass          # stamping is informational; never block a real write


def read_format(h5_path: str) -> Dict:
    """``{format, format_version}`` as stored, or ``{}``."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return {}
    try:
        with h5py.File(h5_path, "r") as f:
            out = {}
            if "format" in f.attrs:
                v = f.attrs["format"]
                out["format"] = v.decode() if isinstance(v, bytes) else str(v)
            if "format_version" in f.attrs:
                out["format_version"] = int(f.attrs["format_version"])
            return out
    except Exception:
        return {}


def migrate_to_mad_suffix(wav_path: str) -> Optional[str]:
    """Rename a recording's legacy store to ``<stem>.mad``.

    A rename, not a rewrite: the bytes are already in the right format. Returns
    the new path, or None if there was nothing to move (or the target exists,
    in which case the two are left alone rather than one clobbering the other).
    """
    p = Path(wav_path)
    target = p.with_name(p.stem + MAD_SUFFIX)
    if target.exists():
        return None
    for suffix in LEGACY_MAD_SUFFIXES:
        older = p.with_name(p.stem + suffix)
        if not older.is_file():
            continue
        try:
            os.replace(str(older), str(target))
            stamp_format(str(target))
            return str(target)
        except Exception:
            return None
    return None


@_reporting_write
def set_grid_attrs(h5_path: str, **params) -> None:
    """Record spectrogram-grid params at the file root (idempotent)."""
    _require_h5()
    with h5py.File(h5_path, "a") as f:
        for k in _GRID_KEYS:
            if k in params and params[k] is not None:
                f.attrs[k] = int(params[k])
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["format"] = FORMAT_NAME
        f.attrs["format_version"] = FORMAT_VERSION
        # Provenance, written once and never overwritten: which FNT wrote this
        # file, when, and which recording it belongs to. A .mad separated from
        # its wav (copied to another machine, or found years later) otherwise
        # says nothing about where it came from — and "which version produced
        # this?" is unanswerable after the fact.
        if "created" not in f.attrs:
            f.attrs["created"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["fnt_version"] = _fnt_version()
        wav = params.get("source_wav")
        if wav and "source_wav" not in f.attrs:
            f.attrs["source_wav"] = str(wav)
        f.attrs["updated"] = datetime.now().isoformat(timespec="seconds")


#: Root attrs describing the last inference run over this recording. Written
#: even when the run found nothing — "inference ran and found 0 calls" and
#: "inference never ran" look identical without it, and the first is a real
#: result the user should not have to re-derive by re-running.
_RUN_KEYS = ("last_infer_at", "last_infer_model", "last_infer_threshold",
             "last_infer_min_blob", "last_infer_merge", "last_infer_n")


def set_infer_run_attrs(h5_path: str, **params) -> None:
    """Record who/what/when produced the predictions now in this file."""
    _require_h5()
    with h5py.File(h5_path, "a") as f:
        for k in _RUN_KEYS:
            v = params.get(k)
            if v is None:
                continue
            if isinstance(v, str):
                f.attrs[k] = v
            elif k in ("last_infer_min_blob", "last_infer_n"):
                f.attrs[k] = int(v)          # counts are counts, not 3286.0
            elif k == "last_infer_merge":
                f.attrs[k] = int(bool(v))
            else:
                # Thresholds come from a spin box; 0.15000000000000002 is
                # binary-float noise, not precision anyone chose.
                f.attrs[k] = round(float(v), 6)


def get_infer_run_attrs(h5_path: str) -> Dict:
    """The last run's settings, or {} when inference has never been run."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return {}
    out: Dict = {}
    try:
        with h5py.File(h5_path, "r") as f:
            for k in _RUN_KEYS:
                if k in f.attrs:
                    v = f.attrs[k]
                    out[k] = (v.decode() if isinstance(v, bytes)
                              else v.item() if hasattr(v, "item") else v)
    except Exception:
        return {}
    return out


def was_inferred(h5_path: str) -> bool:
    """True if inference has ever been run over this recording.

    Prefers the explicit run stamp, but falls back to the evidence a run
    leaves regardless of version — the cached blob count, or the pred_calls
    group itself, both of which only exist because a run wrote them. Without
    the fallback every file analyzed before the stamp existed would read as
    never analyzed.
    """
    _require_h5()
    if not os.path.isfile(h5_path):
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            if any(k in f.attrs for k in _RUN_KEYS):
                return True
            if "n_pred_blobs" in f.attrs:
                return True
            return PRED_GROUP in f
    except Exception:
        return False


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


@_reporting_write
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


@_reporting_write
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


#: Per-detection fields carried alongside a prediction crop. These used to live
#: only in the CSV, which made the CSV a second source of truth: the review
#: UI's min-score filter and the "which model produced this" record could not be
#: answered from the h5 at all. Storing them here is what lets the CSV become a
#: derived export rather than something that has to be kept in sync.
PRED_ATTRS = ("score", "class", "model_name", "threshold", "min_blob_pixels")


@_reporting_write
def write_pred_masks(h5_path: str, crops: List[Dict]) -> None:
    """Replace the stored prediction crops with ``crops`` (one file open).

    Each crop is a dict ``{blob_id, mask, f_off, t_off}`` where ``mask`` is the
    cropped binary (thresholded) blob and ``f_off``/``t_off`` are its top-left
    offsets on the full spec grid. Any of :data:`PRED_ATTRS` present is stored
    with it. Also caches ``n_pred_blobs`` so file lists can show the count
    without reading any crop.
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
            for k in PRED_ATTRS:
                v = c.get(k)
                if v is None or v == "":
                    continue
                # h5py cannot store None, and a mixed-type attr is a trap for
                # readers — keep numbers numeric and everything else a string.
                ds.attrs[k] = v if isinstance(v, (int, float)) else str(v)
        f.attrs["n_pred_blobs"] = int(len(crops))


def read_all_pred_masks(h5_path: str) -> Dict[str, Dict]:
    """Return ``{blob_id: {mask(bool), f_off, t_off}}`` for every stored
    prediction crop (cheap — only small per-blob arrays, never the grid).

    Any of :data:`PRED_ATTRS` stored with a crop (score, class, model
    provenance) comes back alongside it, so a pending detection can be fully
    described without consulting the CSV."""
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
            rec = {
                "mask": ds[()].astype(bool),
                "f_off": int(ds.attrs.get("f_off", 0)),
                "t_off": int(ds.attrs.get("t_off", 0)),
            }
            for k in PRED_ATTRS:
                if k in ds.attrs:
                    v = ds.attrs[k]
                    rec[k] = (v.decode() if isinstance(v, bytes)
                              else v.item() if hasattr(v, "item") else v)
            out[key] = rec
    return out


def list_pred_ids(h5_path: str) -> List[str]:
    """Prediction-crop ids only — no arrays read.

    For counting what is still unreviewed without decompressing every crop.
    """
    _require_h5()
    if not os.path.isfile(h5_path):
        return []
    try:
        with h5py.File(h5_path, "r") as f:
            grp = f.get(PRED_GROUP)
            return list(grp.keys()) if grp is not None else []
    except Exception:
        return []


def read_pred_attrs(h5_path: str) -> List[Dict]:
    """Per-crop attributes only — no masks read.

    Enough to answer "was this recording analysed by this model at these
    settings?", which is the resume question, without decompressing anything.
    """
    _require_h5()
    if not os.path.isfile(h5_path):
        return []
    out: List[Dict] = []
    try:
        with h5py.File(h5_path, "r") as f:
            grp = f.get(PRED_GROUP)
            if grp is None:
                return []
            for key in grp:
                ds = grp[key]
                rec = {"blob_id": key}
                for k in PRED_ATTRS:
                    if k in ds.attrs:
                        v = ds.attrs[k]
                        rec[k] = (v.decode() if isinstance(v, bytes)
                                  else v.item() if hasattr(v, "item") else v)
                out.append(rec)
    except Exception:
        return []
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


@_reporting_write
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


@_reporting_write
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


@_reporting_write
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


@_reporting_write
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


@_reporting_write
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


@_reporting_write
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


@_reporting_write
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


def td_iter_file_examples(h5_path: str, wav_name: str,
                          with_spec: bool = True,
                          kinds: Optional[Sequence[str]] = None,
                          ) -> Iterator[Dict]:
    """Yield examples for a single wav file, skipping heavy array reads for
    non-matching entries.

    ``with_spec=False`` skips the spectrogram patch entirely and returns the
    mask as bool rather than float32. Callers that only need geometry -- every
    path that builds annotations for the overlay -- pay roughly half as much,
    and on a file with a thousand labels sitting on a network share that is
    seconds per file switch.

    ``kinds`` filters on the example's kind *before* touching any array, so
    picking out a handful of rejected calls costs a metadata scan rather than a
    full read of the store.
    """
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
            if kinds is not None and example_kind(meta) not in kinds:
                continue
            try:
                spec = (g["spec"][()].astype(np.float32) / 255.0
                        if with_spec else None)
                mask = (g["mask"][()] > 0)
                if with_spec:
                    mask = mask.astype(np.float32)
            except Exception:
                continue
            yield {"meta": meta, "spec": spec, "mask": mask}


def td_read_example(h5_path: str, example_id: str) -> Optional[Dict]:
    """Read ONE example by id: ``{meta, spec(float[0,1]), mask(float{0,1})}``.

    Targeted rather than a scan, because the caller is the undo path: it grabs
    an example's content immediately before that example is deleted, on a
    per-keystroke review action. Iterating the whole store to find one would
    decompress every other example's arrays for nothing.
    """
    _require_h5()
    if not os.path.isfile(h5_path) or not example_id:
        return None
    try:
        with h5py.File(h5_path, "r") as f:
            ex = f.get("examples")
            if ex is None or str(example_id) not in ex:
                return None
            g = ex[str(example_id)]
            return {
                "meta": json.loads(g.attrs.get("meta_json", "{}")),
                "spec": g["spec"][()].astype(np.float32) / 255.0,
                "mask": (g["mask"][()] > 0).astype(np.float32),
            }
    except Exception:
        return None


def example_kind(meta: Dict) -> str:
    """An example's role in training: ``'label'`` (painted call, the default for
    anything written before negatives existed) or ``'negative'`` (a rejected
    detection, kept as an explicit hard negative)."""
    return str((meta or {}).get("kind") or "label")


def td_iter_meta(h5_path: str) -> Iterator[Dict]:
    """Yield every example's metadata, never touching the spec/mask arrays."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    try:
        with h5py.File(h5_path, "r") as f:
            ex = f.get("examples")
            if ex is None:
                return
            for key in ex:
                try:
                    yield json.loads(ex[key].attrs.get("meta_json", "{}"))
                except Exception:
                    continue
    except Exception:
        return


def td_count(h5_path: str, kind: Optional[str] = None) -> int:
    """Number of stored examples; ``kind`` filters to 'label' or 'negative'."""
    _require_h5()
    if not os.path.isfile(h5_path):
        return 0
    with h5py.File(h5_path, "r") as f:
        ex = f.get("examples")
        if ex is None:
            return 0
        if kind is None:
            return len(ex)
        n = 0
        for key in ex:
            try:
                meta = json.loads(ex[key].attrs.get("meta_json", "{}"))
            except Exception:
                continue
            if example_kind(meta) == kind:
                n += 1
        return n


def td_count_by_source_wav(h5_path: str,
                           kind: Optional[str] = None) -> Dict[str, int]:
    """Return ``{wav_basename: n_examples}`` reading **only** each example's
    metadata attribute — never decompressing the spec/mask arrays. Cheap enough
    to call for a whole project when populating a file list.

    ``kind`` filters to 'label' or 'negative'; None counts both."""
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
                if kind is not None and example_kind(meta) != kind:
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


@_reporting_write
def td_delete(h5_path: str, example_id: str) -> None:
    _require_h5()
    if not os.path.isfile(h5_path):
        return
    with h5py.File(h5_path, "a") as f:
        ex = f.get("examples")
        if ex is not None and example_id in ex:
            del ex[example_id]


@_reporting_write
def td_set_kind(h5_path: str, example_id: str, kind: str) -> bool:
    """Change one example's kind in place, keeping its pixels.

    This is what makes Reject reversible. Rejecting a confirmed call used to
    *delete* its example, which stopped it training the model -- correct -- but
    also destroyed the traced mask, so reopening the file showed a bare bounding
    box and re-accepting would have saved that box as the label. Demoting to
    ``kind='rejected'`` achieves the same training outcome while keeping the
    geometry, so the decision can be taken back. Delete still deletes.

    Returns True if the example existed and was changed.
    """
    _require_h5()
    if not os.path.isfile(h5_path) or not example_id:
        return False
    try:
        with h5py.File(h5_path, "a") as f:
            ex = f.get("examples")
            if ex is None or str(example_id) not in ex:
                return False
            g = ex[str(example_id)]
            meta = json.loads(g.attrs.get("meta_json", "{}"))
            meta["kind"] = str(kind)
            g.attrs["meta_json"] = json.dumps(meta)
            return True
    except Exception:
        return False


@_reporting_write
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


@_reporting_write
def td_update_meta(h5_path: str, example_id: str, updates: Dict) -> bool:
    """Merge ``updates`` into one example's metadata, leaving its pixels alone.

    How anything derived about a call — its harmonic assignment, say — gets
    recorded on the call itself rather than in a side file that has to be kept
    in step.
    """
    _require_h5()
    if not os.path.isfile(h5_path) or not example_id:
        return False
    with h5py.File(h5_path, "a") as f:
        ex = f.get("examples")
        if ex is None or str(example_id) not in ex:
            return False
        g = ex[str(example_id)]
        meta = json.loads(g.attrs.get("meta_json", "{}"))
        meta.update(updates)
        g.attrs["meta_json"] = json.dumps(meta)
        return True


@_reporting_write
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
