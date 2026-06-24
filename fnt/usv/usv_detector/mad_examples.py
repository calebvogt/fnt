"""MAD per-call training-example store.

Each confirmed call is saved as a small, self-contained example under the
project's ``models/training_data/`` directory — a normalized spectrogram
**patch** (full frequency height x a time window around the call), its binary
mask, and a metadata sidecar. Training reads these patches directly, so the
project no longer needs the original WAVs (inference still runs on whatever
recordings the user points at).

This replaces the old WAV + sibling-PNG + committed-bands path
(:mod:`fnt.usv.usv_detector.mad_dataset`) for the GUI labeling flow.

Layout (per example ``<id>``)::

    <dataset_dir>/<id>.png        normalized spec patch, uint8 (freq x time)
    <dataset_dir>/<id>_mask.png   binary mask, uint8 {0, 255}
    <dataset_dir>/<id>.json       metadata (class, source wav, time/freq, params)
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .mad_dataset import TILE_FREQ_BINS, TILE_TIME_FRAMES, _crop_or_pad

try:
    from PIL import Image as _PILImage
    _PILImage.MAX_IMAGE_PIXELS = None
    _HAS_PIL = True
except Exception:  # pragma: no cover - PIL is a project dep but be safe
    _HAS_PIL = False


def _require_pil():
    if not _HAS_PIL:
        raise RuntimeError(
            "PIL/Pillow is required to read/write MAD training examples. "
            "pip install pillow"
        )


# ----------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------
def save_example(
    dataset_dir: str,
    spec_patch: np.ndarray,
    mask_patch: np.ndarray,
    meta: Dict,
    example_id: Optional[str] = None,
) -> str:
    """Persist one confirmed-call example.

    Args:
        dataset_dir: ``<project>/models/training_data``.
        spec_patch:  normalized spectrogram patch, float in [0, 1] OR uint8.
        mask_patch:  binary mask, same shape as ``spec_patch`` (any nonzero =
                     positive).
        meta:        metadata dict (class, source_wav, time/freq, params, ...).
        example_id:  optional explicit id; a uuid-based one is generated
                     otherwise.

    Returns the example id.

    New examples are written to the consolidated ``training_data.h5`` store;
    legacy PNG/JSON triplets remain readable (see :func:`iter_examples`).
    """
    from . import fnt_mask_store as _ms
    d = Path(dataset_dir)
    d.mkdir(parents=True, exist_ok=True)
    return _ms.td_save_example(_store_path(dataset_dir), spec_patch, mask_patch,
                               meta, example_id)


def _store_path(dataset_dir: str) -> str:
    """Path to the consolidated HDF5 example store for a dataset dir."""
    return str(Path(dataset_dir) / "training_data.h5")


# ----------------------------------------------------------------------
# Read
# ----------------------------------------------------------------------
def _load_png_gray(path: Path) -> np.ndarray:
    _require_pil()
    return np.asarray(_PILImage.open(path).convert("L"))


def _iter_legacy_examples(dataset_dir: str) -> Iterator[Dict]:
    """Yield examples from legacy PNG/JSON triplets (pre-h5 storage)."""
    d = Path(dataset_dir)
    if not d.is_dir():
        return
    for meta_path in sorted(d.glob("*.json")):
        ex_id = meta_path.stem
        spec_path = d / f"{ex_id}.png"
        mask_path = d / f"{ex_id}_mask.png"
        if not spec_path.is_file() or not mask_path.is_file():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            spec = _load_png_gray(spec_path).astype(np.float32) / 255.0
            mask = (_load_png_gray(mask_path) > 0).astype(np.float32)
        except Exception:
            continue
        yield {"meta": meta, "spec": spec, "mask": mask}


def iter_examples(dataset_dir: str) -> Iterator[Dict]:
    """Yield example dicts ``{meta, spec, mask}`` from the consolidated h5 store
    **and** any legacy PNG/JSON triplets (h5 wins on id collision).

    ``spec`` is float32 in [0, 1]; ``mask`` is float32 {0, 1}.
    """
    from . import fnt_mask_store as _ms
    seen = set()
    for ex in _ms.td_iter_examples(_store_path(dataset_dir)):
        seen.add(ex["meta"].get("id"))
        yield ex
    for ex in _iter_legacy_examples(dataset_dir):
        if ex["meta"].get("id") not in seen:
            yield ex


def count_examples(dataset_dir: str) -> int:
    from . import fnt_mask_store as _ms
    n = _ms.td_count(_store_path(dataset_dir))
    # add legacy triplets not already in the h5 (ids are unique uuids)
    d = Path(dataset_dir)
    if d.is_dir():
        n += sum(
            1 for j in d.glob("*.json")
            if (d / f"{j.stem}.png").is_file()
            and (d / f"{j.stem}_mask.png").is_file()
        )
    return n


def count_by_source_wav(dataset_dir: str) -> Dict[str, int]:
    """Return ``{wav_basename: n_confirmed_examples}`` cheaply (metadata only,
    no spec/mask decompression). Combines the consolidated h5 store with any
    legacy PNG/JSON triplets."""
    from . import fnt_mask_store as _ms
    out = dict(_ms.td_count_by_source_wav(_store_path(dataset_dir)))
    d = Path(dataset_dir)
    if d.is_dir():
        for meta_path in d.glob("*.json"):
            ex_id = meta_path.stem
            if not (d / f"{ex_id}.png").is_file() or \
                    not (d / f"{ex_id}_mask.png").is_file():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue
            bn = Path(str(meta.get("source_wav", ""))).name
            if bn:
                out[bn] = out.get(bn, 0) + 1
    return out


def list_classes(dataset_dir: str) -> List[str]:
    """Distinct class labels present in the example store (in first-seen order)."""
    seen: List[str] = []
    for ex in iter_examples(dataset_dir):
        c = ex["meta"].get("class")
        if c and c not in seen:
            seen.append(c)
    return seen


# ----------------------------------------------------------------------
# Edit / delete
# ----------------------------------------------------------------------
def delete_example(dataset_dir: str, example_id: str) -> None:
    """Remove an example from the h5 store and/or its legacy triplet."""
    from . import fnt_mask_store as _ms
    _ms.td_delete(_store_path(dataset_dir), example_id)
    d = Path(dataset_dir)
    for suffix in (".png", "_mask.png", ".json"):
        p = d / f"{example_id}{suffix}"
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def update_example_class(dataset_dir: str, example_id: str, new_class: str) -> None:
    """Rewrite an example's class label (h5 store, else legacy sidecar)."""
    from . import fnt_mask_store as _ms
    sp = _store_path(dataset_dir)
    if example_id in _ms.td_list_ids(sp):
        _ms.td_update_class(sp, example_id, new_class)
        return
    p = Path(dataset_dir) / f"{example_id}.json"
    if not p.is_file():
        return
    with open(p) as f:
        meta = json.load(f)
    meta["class"] = new_class
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)


def update_example_mask(
    dataset_dir: str, example_id: str, mask_patch: np.ndarray,
    meta_updates: Optional[Dict] = None,
) -> None:
    """Overwrite an example's mask (h5 store, else legacy PNG) + merge metadata."""
    from . import fnt_mask_store as _ms
    sp = _store_path(dataset_dir)
    if example_id in _ms.td_list_ids(sp):
        _ms.td_update_mask(sp, example_id, mask_patch, meta_updates)
        return
    _require_pil()
    d = Path(dataset_dir)
    mask = (np.asarray(mask_patch) > 0).astype(np.uint8) * 255
    _PILImage.fromarray(mask, mode="L").save(d / f"{example_id}_mask.png")
    if meta_updates:
        p = d / f"{example_id}.json"
        meta = {}
        if p.is_file():
            with open(p) as f:
                meta = json.load(f)
        meta.update(meta_updates)
        with open(p, "w") as f:
            json.dump(meta, f, indent=2)


# ----------------------------------------------------------------------
# Legacy → HDF5 migration
# ----------------------------------------------------------------------
def has_legacy_examples(dataset_dir: str) -> bool:
    """True if any legacy PNG/JSON example triplets are present."""
    d = Path(dataset_dir)
    if not d.is_dir():
        return False
    for j in d.glob("*.json"):
        if (d / f"{j.stem}.png").is_file() and (d / f"{j.stem}_mask.png").is_file():
            return True
    return False


def migrate_legacy_to_h5(dataset_dir: str) -> int:
    """Move legacy triplets into ``training_data.h5`` and archive the originals
    to ``legacy_pre_h5/``. Returns the number of examples migrated."""
    from . import fnt_mask_store as _ms
    import shutil
    d = Path(dataset_dir)
    if not d.is_dir():
        return 0
    sp = _store_path(dataset_dir)
    archive = d / "legacy_pre_h5"
    n = 0
    for ex in list(_iter_legacy_examples(dataset_dir)):
        meta = ex["meta"]
        ex_id = meta.get("id")
        try:
            _ms.td_save_example(sp, ex["spec"], ex["mask"], meta, ex_id)
            archive.mkdir(exist_ok=True)
            for suffix in (".png", "_mask.png", ".json"):
                p = d / f"{ex_id}{suffix}"
                if p.exists():
                    shutil.move(str(p), str(archive / p.name))
            n += 1
        except Exception:
            continue
    return n


# ----------------------------------------------------------------------
# Per-file annotations (one object per saved example)
# ----------------------------------------------------------------------
def _examples_to_annotations(examples, wav_name, grid_shape):
    """Shared: convert example dicts into annotation dicts clipped to *grid_shape*."""
    n_freq, n_time = grid_shape
    target = Path(str(wav_name)).name
    for ex in examples:
        meta = ex["meta"]
        if Path(str(meta.get("source_wav", ""))).name != target:
            continue
        m = ex["mask"] > 0
        if not m.any():
            continue
        t_off = int(meta.get("patch_t_off") or 0)
        f_off = int(meta.get("patch_f_off") or 0)
        fs = np.where(m.any(axis=1))[0]
        ts = np.where(m.any(axis=0))[0]
        lf0, lf1 = int(fs[0]), int(fs[-1]) + 1
        lt0, lt1 = int(ts[0]), int(ts[-1]) + 1
        f0 = max(0, f_off + lf0)
        f1 = min(n_freq, f_off + lf1)
        t0 = max(0, t_off + lt0)
        t1 = min(n_time, t_off + lt1)
        if f1 <= f0 or t1 <= t0:
            continue
        local = m[lf0:lf0 + (f1 - f0), lt0:lt0 + (t1 - t0)]
        yield {
            "id": meta.get("id", ex.get("meta", {}).get("id")),
            "category": meta.get("class", ""),
            "f0": f0, "f1": f1, "t0": t0, "t1": t1,
            "mask": np.ascontiguousarray(local),
        }


def iter_file_annotations(
    dataset_dir: str, wav_name: str, grid_shape: Tuple[int, int],
):
    """Yield one annotation dict per example belonging to ``wav_name``.

    Uses a fast path that skips heavy array reads for non-matching examples.
    """
    from . import fnt_mask_store as _ms
    h5_path = _store_path(dataset_dir)
    fast = list(_ms.td_iter_file_examples(h5_path, Path(str(wav_name)).name))
    legacy = [ex for ex in _iter_legacy_examples(dataset_dir)
              if Path(str(ex["meta"].get("source_wav", ""))).name
              == Path(str(wav_name)).name]
    seen = {ex["meta"].get("id") for ex in fast}
    combined = fast + [ex for ex in legacy
                       if ex["meta"].get("id") not in seen]
    yield from _examples_to_annotations(combined, wav_name, grid_shape)


# ----------------------------------------------------------------------
# Per-file confirmed-mask reconstruction (for the GUI overlay)
# ----------------------------------------------------------------------
def reconstruct_file_mask(
    dataset_dir: str, wav_name: str, grid_shape: Tuple[int, int],
) -> np.ndarray:
    """Rebuild the confirmed-positive mask for one source file.

    Pastes every matching example's mask back onto the full-file spec grid at
    its saved time offset. ``grid_shape`` is ``(n_freq_bins, n_time_frames)``.
    Returns a uint8 {0, 1} array.
    """
    n_freq, n_time = grid_shape
    out = np.zeros((n_freq, n_time), dtype=np.uint8)
    target = Path(str(wav_name)).name
    for ex in iter_examples(dataset_dir):
        meta = ex["meta"]
        if Path(str(meta.get("source_wav", ""))).name != target:
            continue
        t_off = int(meta.get("patch_t_off") or 0)
        f_off = int(meta.get("patch_f_off") or 0)
        m = ex["mask"] > 0
        h, w = m.shape
        t0 = max(0, t_off)
        t1 = min(n_time, t_off + w)
        f0 = max(0, f_off)
        f1 = min(n_freq, f_off + h)
        if t1 <= t0 or f1 <= f0:
            continue
        sub = m[(f0 - f_off):(f1 - f_off), (t0 - t_off):(t1 - t_off)]
        out[f0:f1, t0:t1][sub] = 1
    return out


# ----------------------------------------------------------------------
# Training-tile assembly
# ----------------------------------------------------------------------
def collect_training_examples(
    dataset_dir: str,
    tile_time_frames: int = TILE_TIME_FRAMES,
    tile_freq_bins: int = TILE_FREQ_BINS,
    progress=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble ``(specs, targets, weights)`` stacks from saved examples.

    Each example patch is sliced into one or more tiles of shape
    ``(tile_freq_bins, tile_time_frames)``. The whole patch is supervised, so
    ``weight`` is 1 everywhere a tile overlaps the patch (and 0 in any pad).
    Mirrors the return contract of
    :func:`fnt.usv.usv_detector.mad_dataset.collect_training_tiles`.
    """
    examples = list(iter_examples(dataset_dir))
    specs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    n = len(examples)
    for i, ex in enumerate(examples):
        if progress is not None:
            progress(i, n, ex["meta"].get("id", ""))
        spec = ex["spec"]
        mask = ex["mask"]
        # Patch occupancy = 1 (supervised) everywhere the real patch exists;
        # used as the tile weight so any zero-padding stays unsupervised.
        occ = np.ones_like(spec, dtype=np.float32)
        H, W = spec.shape
        # Slide tiles across time if the patch is wider than a tile.
        t = 0
        while True:
            specs.append(_crop_or_pad(spec, tile_freq_bins, tile_time_frames, 0, t))
            targets.append(_crop_or_pad(mask, tile_freq_bins, tile_time_frames, 0, t))
            weights.append(_crop_or_pad(occ, tile_freq_bins, tile_time_frames, 0, t))
            t += tile_time_frames
            if t >= W:
                break
    if progress is not None:
        progress(n, n, "done")
    if not specs:
        empty = np.zeros((0, tile_freq_bins, tile_time_frames), dtype=np.float32)
        return empty, empty.copy(), empty.copy()
    return (
        np.stack(specs, axis=0),
        np.stack(targets, axis=0),
        np.stack(weights, axis=0),
    )
