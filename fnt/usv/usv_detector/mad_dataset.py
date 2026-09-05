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

import warnings
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


# Per-recording dB normalization ---------------------------------------------
# Percentiles, not min/max: the floor should track the noise bed (not the single
# quietest bin) and the ceiling should sit just under the loudest calls without
# being pinned by one broadband click.
DB_PCT_LO, DB_PCT_HI = 5.0, 99.9
DB_MIN_SPAN = 20.0          # never map a near-flat recording onto full contrast


def estimate_db_range(
    audio: np.ndarray, sample_rate: int, nperseg: int, noverlap: int,
    nfft: int, lo_pct: float = DB_PCT_LO, hi_pct: float = DB_PCT_HI,
    n_windows: int = 24, window_s: float = 1.0,
) -> tuple:
    """Estimate a recording's own dB range from evenly spaced probe windows.

    A project-wide fixed range assumes every recording shares a noise floor and
    gain. Across trials with different mics, distances and ambient conditions
    they do not, so the same call arrives at the model at different input
    intensities — which is a large part of why a model trained on a few
    recordings generalizes poorly to the rest.

    Probes are **evenly spaced and deterministic**, never random: labeling and
    inference must derive an identical range for the same audio, or a call
    would be normalized one way when it trains and another way when it is
    detected. Sampling ~24 s of a long recording is enough for percentiles and
    avoids computing a full-file spectrogram just to pick two numbers.
    """
    audio = np.asarray(audio, dtype=np.float32)
    win = max(int(window_s * sample_rate), nperseg * 4)
    if audio.size <= win:
        starts = [0]
    else:
        n = max(1, int(n_windows))
        step = max(1, (audio.size - win) // n)
        starts = [i * step for i in range(n)]
    noverlap_safe = min(noverlap, nperseg - 1)
    chunks = []
    for st in starts:
        seg = audio[st:st + win]
        if seg.size < nperseg:
            continue
        _f, _t, sxx = compute_spectrogram(
            seg, sr=sample_rate, nperseg=nperseg, noverlap=noverlap_safe,
            nfft=nfft)
        chunks.append(sxx.ravel())
    if not chunks:
        return (-100.0, -20.0)
    vals = np.concatenate(chunks)
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if hi - lo < DB_MIN_SPAN:
        hi = lo + DB_MIN_SPAN
    return (lo, hi)


def db_range_for(
    db_norm: str, db_min: float, db_max: float,
    audio=None, sample_rate: int = 0, nperseg: int = 512,
    noverlap: int = 384, nfft: int = 1024,
) -> tuple:
    """Resolve the dB range to normalize with: the project's fixed pair, or this
    recording's own. Falls back to fixed when there is no audio to measure."""
    if db_norm == 'per_file' and audio is not None and sample_rate:
        try:
            return estimate_db_range(audio, sample_rate, nperseg, noverlap, nfft)
        except Exception:
            pass
    return (float(db_min), float(db_max))


def compute_full_spec_image(
    audio: np.ndarray, sample_rate: int,
    nperseg: int, noverlap: int, nfft: int,
    db_min: float, db_max: float,
    chunk_frames: int = 20000,
    as_uint8: bool = False,
) -> np.ndarray:
    """Return normalized full-file spec image, shape (n_freq_bins, n_time_frames).

    Computed in time chunks, with the dB conversion and normalization fused
    into the same pass and written straight into the destination.

    The obvious version — full STFT, then ``10*log10``, then normalize — is
    three separate passes that each allocate a whole-file array. On a 600 s
    250 kHz recording that is 2.4 GB three times over plus scipy's own
    temporaries, and the cost is memory traffic rather than arithmetic: it
    measured 2.5x slower than this, and threading the STFT made it *worse*
    because the bandwidth was already saturated. Only one output array is
    allocated here; the working buffer stays small enough to be cache-friendly.

    Output is bit-identical to the unfused version — the same operations in the
    same order and dtype, just applied per chunk and in place.
    """
    audio = np.asarray(audio, dtype=np.float32)
    hop = max(1, int(nperseg) - int(noverlap))
    n_freq = int(nfft) // 2 + 1
    if len(audio) < nperseg:
        return np.zeros((n_freq, 0), dtype=np.float32)
    n_frames = (len(audio) - int(nperseg)) // hop + 1
    if db_max <= db_min:                 # mirrors spec_to_image
        db_max = db_min + 1e-3
    span = db_max - db_min

    from scipy import signal as _signal
    # uint8 costs a quarter of the memory and is the precision the model was
    # actually trained at — td_save_example stores every example as
    # (clip(x,0,1)*255).round().astype(uint8), so a float32 input at inference
    # is finer-grained than anything the weights ever saw.
    out = np.empty((n_freq, n_frames), dtype=np.uint8 if as_uint8 else np.float32)
    f0 = 0
    while f0 < n_frames:
        f1 = min(n_frames, f0 + int(chunk_frames))
        # Overlapping sample span for exactly the frames [f0, f1).
        s0 = f0 * hop
        s1 = (f1 - 1) * hop + int(nperseg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, sxx = _signal.spectrogram(
                audio[s0:s1], fs=sample_rate,
                nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                window='hann', scaling='density')
            sxx = sxx[:, :f1 - f0]
            # dB, then [0,1] normalization, in place on the chunk.
            np.log10(sxx + 1e-10, out=sxx)
            sxx *= 10.0
            sxx -= db_min
            sxx /= span
            np.clip(sxx, 0.0, 1.0, out=sxx)
            if as_uint8:
                sxx *= 255.0
                np.round(sxx, out=sxx)
        out[:, f0:f1] = sxx
        f0 = f1
    return out


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
