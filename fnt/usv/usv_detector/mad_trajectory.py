"""Calls as paths through a feature space — the data behind MAD's 3D view.

A spectrogram shows a call as a shape against time. Plotting its per-frame
features against *each other* instead turns it into a path: each spectrogram
frame is one point, and the call traces a trajectory through (pitch, timbre,
motion). Upsweeps, downsweeps, trills and flat calls then trace visibly
different shapes, which is hard to see side by side on a spectrogram.

Everything here is mask-gated. Features come from :func:`call_frame_features`
over the call's own pixels, so the broadband noise under a USV or a second call
overlapping it in time never enters the path. That is the point of doing this
in MAD rather than on raw audio.

Pure numpy, no Qt: the window in ``fnt.usv.mad_trajectory_view`` only draws what
this produces, so the numbers can be tested without a display.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Selectable axis features: key -> (label, unit). Units are what the arrays
#: returned by :func:`call_trajectory` are expressed in.
FEATURES: Dict[str, Tuple[str, str]] = {
    'pitch':      ('Pitch', 'kHz'),
    'pitch_rate': ('Pitch rate', 'kHz/ms'),
    'entropy':    ('Spectral entropy', ''),
    'tonality':   ('Tonality', ''),
    'bandwidth':  ('Bandwidth', 'kHz'),
    'centroid':   ('Spectral centroid', 'kHz'),
    'power':      ('Power', 'dB'),
    'time':       ('Time in call', 'ms'),
}

#: X, Y, Z. Pitch, timbre and motion: spectral entropy stands in for timbre
#: (tonal vs noisy), and the rate of pitch change for motion — with pitch it
#: makes a phase portrait, where a sweep's direction and a trill's oscillation
#: become the shape of the path.
DEFAULT_AXES: Tuple[str, str, str] = ('pitch', 'entropy', 'pitch_rate')

#: Frames of centred smoothing (~2.5 ms at 0.5 ms/frame). The per-frame peak is
#: quantised to one frequency bin (~244 Hz at nfft 1024 / 250 kHz), so the raw
#: pitch staircases and its derivative is mostly quantisation noise without it.
DEFAULT_SMOOTH_FRAMES = 5


def label_for(key: str) -> str:
    """'Pitch (kHz)' — an axis title for a feature key."""
    name, unit = FEATURES.get(key, (key, ''))
    return f"{name} ({unit})" if unit else name


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average with edge-hold padding (no shrink, no lag)."""
    n = y.size
    k = min(int(k), n if n % 2 else n - 1)
    if k < 3:
        return y.astype(np.float64, copy=True)
    if k % 2 == 0:
        k -= 1
    half = k // 2
    padded = np.concatenate([np.full(half, y[0]), y, np.full(half, y[-1])])
    return np.convolve(padded, np.ones(k) / k, mode='valid')


def call_trajectory(
    frames: Dict, dt: float, frame_offset: int = 0,
    smooth_frames: int = DEFAULT_SMOOTH_FRAMES,
) -> Optional[Dict[str, np.ndarray]]:
    """One call's path: every :data:`FEATURES` key as an array over the frames
    its mask covers, plus ``frame`` (global spectrogram frame index, for tying a
    point back to a time in the recording).

    ``frames`` is the output of :func:`call_frame_features`; ``dt`` seconds per
    frame; ``frame_offset`` the global index of the call's first column. Columns
    the mask misses are dropped rather than interpolated — a gap in the mask is
    a gap in the evidence. Returns None when fewer than two frames remain, since
    a single point has no path and no rate of change.
    """
    if not frames:
        return None
    keep = np.asarray(frames['has_mask'], dtype=bool)
    if keep.sum() < 2:
        return None
    cols = np.nonzero(keep)[0]
    t_ms = (cols - cols[0]) * float(dt) * 1000.0

    def s(key, scale=1.0):
        return _smooth(np.asarray(frames[key], dtype=np.float64)[keep] * scale,
                       smooth_frames)

    pitch = s('peak_freq_hz', 1e-3)
    return {
        'frame': cols + int(frame_offset),
        'time': t_ms,
        'pitch': pitch,
        # np.gradient on the real spacing, so a gap in the mask is a longer
        # step rather than a false jump in rate.
        'pitch_rate': np.gradient(pitch, t_ms),
        'entropy': s('entropy'),
        'tonality': s('tonality'),
        'bandwidth': s('bandwidth_hz', 1e-3),
        'centroid': s('centroid_hz', 1e-3),
        'power': s('power_db'),
    }


def axis_ranges(
    trajectories: Iterable[Dict], keys: Sequence[str], pad: float = 0.05,
) -> Dict[str, Tuple[float, float]]:
    """Common (lo, hi) per axis across every shown call, padded by ``pad`` of
    the span.

    Deliberately min/max rather than percentiles: a percentile range would push
    real points outside the box, and a trail poking through its own axes reads
    as a drawing bug. A zero-width axis (every call flat in that feature) is
    widened around its value so it still spans the box instead of collapsing.
    """
    trajectories = list(trajectories)
    out: Dict[str, Tuple[float, float]] = {}
    for key in keys:
        vals = [np.asarray(t[key], dtype=np.float64) for t in trajectories]
        vals = [v[np.isfinite(v)] for v in vals]
        vals = [v for v in vals if v.size]
        if not vals:
            out[key] = (0.0, 1.0)
            continue
        allv = np.concatenate(vals)
        lo, hi = float(allv.min()), float(allv.max())
        span = hi - lo
        if span <= 1e-12:
            w = max(abs(lo) * 0.05, 1e-3)
            out[key] = (lo - w, hi + w)
        else:
            out[key] = (lo - pad * span, hi + pad * span)
    return out


def to_unit_cube(
    traj: Dict, keys: Sequence[str], ranges: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """(N, 3) points in [-1, 1]^3 for the three axis keys.

    Each axis is scaled on its own: pitch in kHz, entropy in [0, 1] and pitch
    rate in kHz/ms have nothing in common, and a shared scale would flatten
    two of the three axes to nothing.
    """
    cols = []
    for key in keys:
        lo, hi = ranges[key]
        v = np.asarray(traj[key], dtype=np.float64)
        cols.append(2.0 * (v - lo) / (hi - lo) - 1.0)
    return np.column_stack(cols)


def direction_alpha(n: int, lo: float = 0.25, hi: float = 1.0) -> np.ndarray:
    """Per-point opacity rising from onset to offset, so a static path still
    shows which way the call went."""
    if n <= 1:
        return np.full(max(n, 0), hi)
    return np.linspace(lo, hi, n)


def playback_progress(t_s: np.ndarray, pos_s: float) -> Tuple[int, bool]:
    """How much of a call the playhead has reached: ``(n, active)``.

    ``t_s`` are the times (s) of the call's points, ascending; ``n`` is how many
    of them are at or before ``pos_s`` — the length of trail to draw — and
    ``active`` is whether the playhead is inside the call right now, which is
    what earns it the bright style and the "now" marker rather than the dim
    finished one.
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    if t_s.size == 0:
        return 0, False
    n = int(np.searchsorted(t_s, pos_s, side='right'))
    return n, bool(t_s[0] <= pos_s <= t_s[-1])


def finite_rows(points: np.ndarray) -> np.ndarray:
    """Bool mask of rows whose three coordinates are all finite.

    A GL line with one NaN vertex draws nothing at all. A mask rather than a
    filtered copy, so the caller can drop the same rows from ``frame`` and keep
    each point tied to its moment in the recording."""
    return np.isfinite(points).all(axis=1)


__all__: List[str] = [
    'FEATURES', 'DEFAULT_AXES', 'DEFAULT_SMOOTH_FRAMES', 'label_for',
    'call_trajectory', 'axis_ranges', 'to_unit_cube', 'direction_alpha',
    'finite_rows', 'playback_progress',
]
