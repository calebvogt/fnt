"""Brightness ridge of a detected call: its frequency contour from the pixels under its mask.

The mask says where a call is; its shape is a poor contour. A fast trill under a blob-shaped
mask has a flat mean-frequency line, while the spectrogram inside the mask still shows the
oscillation. The ridge is the path through the call's own masked pixels, one frequency bin
per time frame, that maximises the summed spectrogram level minus ``lam`` levels per bin of
frequency change between frames (dynamic programming) -- the brightest continuous line.

Turns are reversals of the ridge by at least ``h_khz`` (zigzag pivots); the rate is turns / 2
/ duration (Hz) and the depth the median kHz between successive pivots. A trill is at least
:data:`TRILL_TURNS` turns at :data:`TRILL_HZ` or faster.

A mask in several pieces (8-connected), as when two detections are merged into one call, gets
one ridge per piece: the path never jumps the gap between pieces, which would count a frequency
step that is not in the call. Turns add up over the pieces, depths pool the pivot steps within
each piece, and the duration leaves out the empty columns between pieces. A one-piece mask gives
exactly what it did before pieces were handled.

This is the same algorithm, with the same constants and tie-breaking, as ``ridge_of`` and
``zigzag`` in the 2021_8x8_Mus analysis (R/09_usv.R), so measures from either agree. Input
levels are MAD's uint8 0-255 spectrogram (the /pred_specs crops, or a training example's
patch); arrays are frequency x time, as MAD stores them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

RIDGE_LAMBDA = 4.0       # levels per bin of frequency change between frames
RIDGE_H_KHZ = 3.0        # a turn needs this excursion
TRILL_TURNS = 4
TRILL_HZ = 20.0
RIDGE_ATTRS = ("ridge_turns", "ridge_rate_hz", "ridge_depth_khz", "ridge_trill")
#: The ridge rule a file's ridges were completed under (its ``ridge_version`` root attr, see
#: fnt_mask_store.mark_ridges_complete). 2: one ridge per piece of a mask (2026-10-21); a file
#: without the attr was written before it, and its multi-piece masks are recomputed.
RIDGE_VERSION = 2
_EIGHT = np.ones((3, 3), dtype=int)


def ridge_path(level: np.ndarray, mask: np.ndarray, lam: float = RIDGE_LAMBDA) -> Tuple[np.ndarray, np.ndarray]:
    """The ridge through ``mask`` on ``level`` (both frequency x time).

    Returns ``(frames, bins)``: the time columns that hold any mask pixel, and the
    frequency row of the ridge in each. Empty arrays for an empty mask."""
    m = np.asarray(mask, bool).T                       # time x freq, as the R code
    s = np.asarray(level, float).T
    on = np.flatnonzero(m.any(axis=1))
    if not len(on):
        return on, on
    cols = np.flatnonzero(m.any(axis=0))
    c0, c1 = int(cols.min()), int(cols.max())
    S = np.where(m, s, -1e6)[on][:, c0:c1 + 1]
    nt, nf = S.shape
    idx = np.arange(nf)
    D = lam * np.abs(idx[:, None] - idx[None, :])
    V = S[0].copy()
    back = np.zeros((nt, nf), dtype=np.int64)
    for t in range(1, nt):
        M = V[None, :] - D                             # M[f, g] = V[g] - lam |f - g|
        g = np.argmax(M, axis=1)                       # first maximum, as max.col "first"
        V = S[t] + M[idx, g]
        back[t] = g
    path = np.empty(nt, dtype=np.int64)
    path[-1] = int(np.argmax(V))                       # first maximum, as which.max
    for t in range(nt - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return on, path + c0


def zigzag(x, h: float) -> np.ndarray:
    """Pivots of a contour: extremes between which it moves by at least ``h``."""
    x = list(np.asarray(x, float))
    piv = []
    if not x:
        return np.asarray(piv)
    direction, ext, start = 0, x[0], x[0]
    for v in x[1:]:
        if direction == 0:
            if v >= start + h:
                direction, ext = 1, v
            elif v <= start - h:
                direction, ext = -1, v
        elif direction == 1:
            if v > ext:
                ext = v
            elif v <= ext - h:
                piv.append(ext)
                direction, ext = -1, v
        else:
            if v < ext:
                ext = v
            elif v >= ext + h:
                piv.append(ext)
                direction, ext = 1, v
    return np.asarray(piv)


def mask_pieces(mask: np.ndarray) -> List[np.ndarray]:
    """The 8-connected pieces of ``mask`` (frequency x time), each a bool mask of the same
    shape, in time order (first column, then lowest bin). One piece: the mask itself."""
    m = np.asarray(mask, bool)
    from scipy import ndimage
    lab, n = ndimage.label(m, structure=_EIGHT)
    if n <= 1:
        return [m] if n else []
    pieces = [lab == k for k in range(1, n + 1)]
    first = [(int(np.flatnonzero(p.any(axis=0))[0]), int(np.flatnonzero(p.any(axis=1))[0]))
             for p in pieces]
    return [pieces[i] for i in sorted(range(n), key=first.__getitem__)]


def ridge_measures(level: np.ndarray, mask: np.ndarray, df_khz: float, dt_ms: float,
                   f_off: int = 0, lam: float = RIDGE_LAMBDA,
                   h_khz: float = RIDGE_H_KHZ) -> Optional[Dict]:
    """Ridge, turns, rate, depth and trill call of one crop.

    ``f_off`` is the crop's first frequency bin on the full grid (bin b is b x df_khz).
    Duration is the crop's width in time frames x dt_ms, less any empty columns between the
    mask's pieces. Returns None for an empty mask. The ``khz`` entry is the ridge per time
    column of the crop (NaN where the mask is empty there); when two pieces share a column (a
    merged harmonic, say) it is one row per piece instead (pieces x columns). ``pieces`` is the
    number of pieces."""
    pieces = mask_pieces(mask)
    if not pieces:
        return None
    width = np.asarray(mask).shape[1]
    rows, turns, steps = [], 0, []
    covered = np.zeros(width, bool)
    for p in pieces:
        frames, bins = ridge_path(level, p, lam)
        khz_on = (bins + int(f_off)) * df_khz
        piv = zigzag(khz_on, h_khz)
        turns += int(len(piv))
        if len(piv) >= 2:
            steps.extend(np.abs(np.diff(piv)))
        row = np.full(width, np.nan, dtype=np.float32)
        row[frames] = khz_on
        rows.append(row)
        covered[frames[0]:frames[-1] + 1] = True       # a piece spans whole columns
    on = np.flatnonzero(covered)
    gaps = int((~covered[on[0]:on[-1] + 1]).sum())
    dur_s = (width - gaps) * dt_ms / 1000.0
    rate = turns / 2.0 / dur_s if dur_s > 0 else 0.0
    depth = float(np.median(steps)) if steps else float("nan")
    k = np.stack(rows)
    if (np.isfinite(k).sum(axis=0) <= 1).all():
        k = np.fmax.reduce(k, axis=0)                  # one row while pieces don't overlap
    return {"khz": k, "ridge_turns": turns, "ridge_rate_hz": float(rate),
            "ridge_depth_khz": depth,
            "ridge_trill": int(turns >= TRILL_TURNS and rate >= TRILL_HZ),
            "pieces": len(pieces)}


def grid_units(sample_rate: float, nperseg: int, noverlap: int, nfft: int) -> Tuple[float, float]:
    """(kHz per frequency bin, ms per time frame) of a MAD spectrogram grid."""
    return sample_rate / nfft / 1000.0, (nperseg - noverlap) / sample_rate * 1000.0
