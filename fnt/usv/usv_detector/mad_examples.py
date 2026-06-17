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
    """
    _require_pil()
    d = Path(dataset_dir)
    d.mkdir(parents=True, exist_ok=True)

    if example_id is None:
        stem = Path(str(meta.get("source_wav", "ex"))).stem
        example_id = f"{stem}_{uuid.uuid4().hex[:10]}"

    spec = np.asarray(spec_patch)
    if spec.dtype != np.uint8:
        spec = np.clip(spec, 0.0, 1.0)
        spec = (spec * 255.0).round().astype(np.uint8)
    mask = (np.asarray(mask_patch) > 0).astype(np.uint8) * 255

    _PILImage.fromarray(spec, mode="L").save(d / f"{example_id}.png")
    _PILImage.fromarray(mask, mode="L").save(d / f"{example_id}_mask.png")
    meta = dict(meta)
    meta["id"] = example_id
    with open(d / f"{example_id}.json", "w") as f:
        json.dump(meta, f, indent=2)
    return example_id


# ----------------------------------------------------------------------
# Read
# ----------------------------------------------------------------------
def _load_png_gray(path: Path) -> np.ndarray:
    _require_pil()
    return np.asarray(_PILImage.open(path).convert("L"))


def iter_examples(dataset_dir: str) -> Iterator[Dict]:
    """Yield example dicts ``{meta, spec, mask}`` for every example on disk.

    ``spec`` is float32 in [0, 1]; ``mask`` is float32 {0, 1}.
    """
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


def count_examples(dataset_dir: str) -> int:
    d = Path(dataset_dir)
    if not d.is_dir():
        return 0
    return sum(
        1 for j in d.glob("*.json")
        if (d / f"{j.stem}.png").is_file() and (d / f"{j.stem}_mask.png").is_file()
    )


def list_classes(dataset_dir: str) -> List[str]:
    """Distinct class labels present in the example store (in first-seen order)."""
    seen: List[str] = []
    for ex in iter_examples(dataset_dir):
        c = ex["meta"].get("class")
        if c and c not in seen:
            seen.append(c)
    return seen


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
        t_off = int(meta.get("patch_t_off", 0))
        f_off = int(meta.get("patch_f_off", 0))
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
