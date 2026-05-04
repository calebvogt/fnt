"""MAD dataset / tile extraction.

Converts audio + sibling mask PNG pairs into `(spec_tile, target_tile,
weight_tile)` training samples. The key constraint is that tiles must
overlap **committed columns** — the only regions where we have
supervision (positives or certified negatives).

The spec tile is a single-channel float32 image in [0, 1], normalized
from dB against the project's `db_min` / `db_max`. Target and weight
tiles are derived from the painted mask via
:mod:`fnt.usv.usv_detector.mad_labels`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .mad_labels import (
    committed_band_runs, load_mask_png, mask_sibling_path,
    positive_target, supervision_weight,
)
from .spectrogram import compute_spectrogram, load_audio


# Standard tile size — both dims divisible by 32 for U-Net.
TILE_FREQ_BINS = 512
TILE_TIME_FRAMES = 256


# ----------------------------------------------------------------------
# Spectrogram → normalized float image
# ----------------------------------------------------------------------
def spec_to_image(spec_db: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
    """Normalize dB spectrogram to float32 in [0, 1]."""
    if db_max <= db_min:
        db_max = db_min + 1e-3
    out = (spec_db - db_min) / (db_max - db_min)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def compute_full_spec_image(
    audio: np.ndarray, sample_rate: int,
    nperseg: int, noverlap: int, nfft: int,
    db_min: float, db_max: float,
) -> np.ndarray:
    """Return normalized full-file spec image, shape (n_freq_bins, n_time_frames)."""
    _, _, sxx_db = compute_spectrogram(
        audio, sr=sample_rate,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
    )
    return spec_to_image(sxx_db, db_min=db_min, db_max=db_max)


# ----------------------------------------------------------------------
# Tile cropping / padding helpers
# ----------------------------------------------------------------------
def _crop_or_pad(arr: np.ndarray, h: int, w: int, f_off: int, t_off: int,
                 fill: float = 0.0) -> np.ndarray:
    """Extract an (h, w) region starting at (f_off, t_off), padding as needed."""
    H, W = arr.shape
    out = np.full((h, w), fill, dtype=arr.dtype)
    f0 = max(0, f_off)
    t0 = max(0, t_off)
    f1 = min(H, f_off + h)
    t1 = min(W, t_off + w)
    if f1 <= f0 or t1 <= t0:
        return out
    out_f0 = f0 - f_off
    out_t0 = t0 - t_off
    out[out_f0:out_f0 + (f1 - f0), out_t0:out_t0 + (t1 - t0)] = arr[f0:f1, t0:t1]
    return out


# ----------------------------------------------------------------------
# Tile generation
# ----------------------------------------------------------------------
@dataclass
class TileWindow:
    wav_path: str
    f_off: int
    t_off: int
    h: int = TILE_FREQ_BINS
    w: int = TILE_TIME_FRAMES


def iter_training_tiles_from_file(
    wav_path: str,
    nperseg: int, noverlap: int, nfft: int,
    db_min: float, db_max: float,
    tile_time_frames: int = TILE_TIME_FRAMES,
    tile_freq_bins: int = TILE_FREQ_BINS,
    overlap_fraction: float = 0.25,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield `(spec_tile, target, weight)` tuples for one file.

    Only tiles that intersect a committed band are emitted — the rest
    have no supervision. Each tile is shape (tile_freq_bins,
    tile_time_frames), float32 in [0,1] for spec and {0,1} float32 for
    target and weight.
    """
    png_path = mask_sibling_path(wav_path)
    if not Path(png_path).is_file():
        return
    try:
        mask = load_mask_png(png_path)
    except Exception:
        return
    runs = committed_band_runs(mask)
    if not runs:
        return

    # Compute the full-file spec image (matches mask shape in width /
    # height as long as params agree with what the GUI used).
    audio, sr = load_audio(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    spec = compute_full_spec_image(
        audio.astype(np.float32), sr,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        db_min=db_min, db_max=db_max,
    )

    # Align mask to spec width — masks saved before nperseg changed may
    # drift by a few frames.
    T_spec = spec.shape[1]
    T_mask = mask.shape[1]
    if T_mask != T_spec:
        aligned = np.zeros((mask.shape[0], T_spec), dtype=mask.dtype)
        w_copy = min(T_mask, T_spec)
        aligned[:, :w_copy] = mask[:, :w_copy]
        mask = aligned

    target = positive_target(mask)
    weight = supervision_weight(mask).astype(np.float32)

    step_t = max(1, int(tile_time_frames * (1 - overlap_fraction)))

    # For each committed run, slide tiles that overlap it. Freq axis is
    # not tiled — we keep the full freq range cropped to tile_freq_bins
    # starting at bin 0 (drops Nyquist when tile_freq_bins == 512 < 513).
    for t_start, t_end in runs:
        t = max(0, t_start - tile_time_frames // 4)
        t_stop = t_end
        while t < t_stop:
            yield (
                _crop_or_pad(spec, tile_freq_bins, tile_time_frames, 0, t),
                _crop_or_pad(target, tile_freq_bins, tile_time_frames, 0, t),
                _crop_or_pad(weight, tile_freq_bins, tile_time_frames, 0, t),
            )
            t += step_t


def collect_training_tiles(
    wav_paths: List[str],
    nperseg: int, noverlap: int, nfft: int,
    db_min: float, db_max: float,
    tile_time_frames: int = TILE_TIME_FRAMES,
    tile_freq_bins: int = TILE_FREQ_BINS,
    overlap_fraction: float = 0.25,
    progress=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate tiles across files into 3-D stacks.

    Returns:
        specs:   (N, tile_freq_bins, tile_time_frames) float32
        targets: (N, tile_freq_bins, tile_time_frames) float32
        weights: (N, tile_freq_bins, tile_time_frames) float32
    """
    specs, targets, weights = [], [], []
    for i, wav in enumerate(wav_paths):
        if progress is not None:
            progress(i, len(wav_paths), Path(wav).name)
        for s, t, w in iter_training_tiles_from_file(
            wav, nperseg, noverlap, nfft, db_min, db_max,
            tile_time_frames, tile_freq_bins, overlap_fraction,
        ):
            specs.append(s)
            targets.append(t)
            weights.append(w)
    if progress is not None:
        progress(len(wav_paths), len(wav_paths), 'done')
    if not specs:
        empty = np.zeros((0, tile_freq_bins, tile_time_frames), dtype=np.float32)
        return empty, empty.copy(), empty.copy()
    return (
        np.stack(specs, axis=0),
        np.stack(targets, axis=0),
        np.stack(weights, axis=0),
    )
