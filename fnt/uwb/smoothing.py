"""Track smoothing for UWB fixes: every method by time (Seconds) or by reads (Samples).

Tags flashed at different rates (10/1, 4/1, 1/1 deliver about 7, 2 and 0.7 reads/s) record the same
movement with very different numbers of reads. A window counted in reads therefore smooths a different
stretch of time on each tag: Savitzky-Golay over 31 reads is about 4 s of a fast tag's track and 40 s of a
slow one's, and an EWMA span of 20 reads about 3 s against 28 s. Distance, speed and chases then differ
by tag, not by animal. A window in seconds smooths the same stretch of time on every tag.

Every method here therefore takes ``window`` plus ``units``:

* Savitzky-Golay - Samples: scipy's savgol_filter over ``window`` reads (forced odd, at most the tag's
  length), order 2, exactly as before. Seconds: the same local quadratic least-squares fit, but over the
  reads within +/- window/2 s of each read, on their real times (a Savitzky-Golay filter IS this fit on
  evenly spaced reads; the time form does not need them evenly spaced).
* Rolling Average / Rolling Median - centred window of ``window`` reads or ``window`` seconds.
* Forward-Backward EWMA - Samples: span of ``window`` reads (alpha = 2/(span+1)), forward then backward.
  Seconds: a span of ``window`` seconds. Each read's weight decays with the time to it, half-life
  ln(2) * window / 2 s, which is the half-life a span of ``window`` reads has at one read per second
  (0.347 x span), forward then backward on the real times.

All methods are centred / zero-phase, so smoothing adds no lag. Smoothing is per tag.
"""
import math

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SAVGOL_METHOD = "Savitzky-Golay"
EWMA_METHOD = "Forward-Backward Exponentially Weighted Moving Average"
ROLLING_METHODS = ("Rolling Average", "Rolling Median")
SECONDS, SAMPLES = "Seconds", "Samples"
SAVGOL_POLYORDER = 2

# What each method used before it had a units choice, and so what a config that names no units means.
LEGACY_WINDOW = {SAVGOL_METHOD: (31, SAMPLES), EWMA_METHOD: (20, SAMPLES)}
DEFAULT_WINDOW = {SAVGOL_METHOD: (31, SAMPLES), EWMA_METHOD: (20, SAMPLES),
                  "Rolling Average": (30, SECONDS), "Rolling Median": (30, SECONDS)}


def clean_method(method):
    """The method name without UI hints; the old abbreviated EWMA label maps to the full one."""
    m = (method or "None").replace(" (default)", "").strip()
    return EWMA_METHOD if m == "Forward-Backward EWMA" else m


def has_window(method):
    return clean_method(method) in DEFAULT_WINDOW


def resolve_window(method, window, units, span=None):
    """(window, units) for ``method`` from saved settings, old configs included.

    A config that names its units is taken at its word. One that does not was written before every
    method had units: Savitzky-Golay then always ran over 31 reads (it ignored the window field, which
    may hold another method's leftover), EWMA over a span of reads (``span``, else ``window``), and the
    rolling methods in seconds. Returns (None, None) for a method without a window.
    """
    m = clean_method(method)
    if m not in DEFAULT_WINDOW:
        return None, None
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
    if units in (SECONDS, SAMPLES) and num(window):
        return int(window), units
    if m == SAVGOL_METHOD:
        return LEGACY_WINDOW[m]
    if m == EWMA_METHOD:
        for v in (span, window):
            if num(v):
                return int(v), SAMPLES
        return LEGACY_WINDOW[m]
    return (int(window) if num(window) else DEFAULT_WINDOW[m][0]), SECONDS


def forward_backward_ewma(series, span):
    """Zero-phase EWMA over a span of reads (filtfilt-style cascade).

    The Wiser hardware applies a causal EWMA, which lags the signal. Post hoc each read's future is
    known, so the EWMA runs forward, then backward over that result; the two delays cancel. Reversing
    with [::-1] keeps the index labels, so the result aligns inside groupby(...).transform(...).
    """
    forward = series.ewm(span=span, adjust=False).mean()
    return forward[::-1].ewm(span=span, adjust=False).mean()[::-1]


def ewma_halflife_s(span_s):
    """Half-life (s) of a time-based EWMA whose span is ``span_s`` seconds."""
    return math.log(2) * float(span_s) / 2.0


def forward_backward_ewma_time(t_s, values, span_s):
    """Zero-phase EWMA on real read times: a read dt seconds away weighs 0.5 ** (dt / half-life)."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return v.copy()
    t = np.asarray(t_s, dtype=float)
    t = t - t[0]
    hl = pd.Timedelta(seconds=max(ewma_halflife_s(span_s), 1e-3))
    fwd = pd.Series(v).ewm(halflife=hl, times=pd.to_datetime(t, unit="s")).mean().to_numpy()
    back_t = pd.to_datetime(t[-1] - t[::-1], unit="s")      # mirrored, so still increasing
    return pd.Series(fwd[::-1]).ewm(halflife=hl, times=back_t).mean().to_numpy()[::-1]


def savgol_samples(values, window):
    """scipy Savitzky-Golay over ``window`` reads (odd, at most the length), order 2."""
    v = np.asarray(values, dtype=float)
    wl = min(int(window), v.size)
    if wl % 2 == 0:
        wl -= 1
    order = min(SAVGOL_POLYORDER, wl - 1)
    if wl < 1 or v.size <= order or order < 0:
        return v.copy()
    return savgol_filter(v, window_length=wl, polyorder=order)


def _local_fit_value(M, N, order):
    """Intercept of the local least-squares polynomial from centred moments.

    M[p] = sum (u_j - u_i)^p (p = 0..4), N[p] = sum x_j (u_j - u_i)^p (p = 0..2), u in half-windows.
    Quadratic where the window supports it, else a line, else the mean.
    """
    M0, M1, M2, M3, M4 = M
    N0, N1, N2 = N
    res = np.full(M0.shape, np.nan)
    det3 = M0 * (M2 * M4 - M3 * M3) - M1 * (M1 * M4 - M3 * M2) + M2 * (M1 * M3 - M2 * M2)
    num3 = N0 * (M2 * M4 - M3 * M3) - M1 * (N1 * M4 - M3 * N2) + M2 * (N1 * M3 - M2 * N2)
    tol = 1e-9
    ok3 = (order >= 2) & (M0 >= 3) & (np.abs(det3) > tol * np.maximum(M0, 1) ** 3)
    res[ok3] = num3[ok3] / det3[ok3]
    det2 = M0 * M2 - M1 * M1
    num2 = N0 * M2 - M1 * N1
    ok2 = ~ok3 & (order >= 1) & (M0 >= 2) & (np.abs(det2) > tol * np.maximum(M0, 1) ** 2)
    res[ok2] = num2[ok2] / det2[ok2]
    ok1 = ~ok3 & ~ok2 & (M0 >= 1)
    res[ok1] = N0[ok1] / M0[ok1]
    return res


def savgol_time_columns(t_s, values, window_s, order=SAVGOL_POLYORDER):
    """savgol_time for several value columns sharing one time line (``values`` is n x m)."""
    V = np.asarray(values, dtype=float)
    one = V.ndim == 1
    V = V[:, None] if one else V
    t = np.asarray(t_s, dtype=float)
    n = t.size
    out = V.copy()
    if n < 2:
        return out[:, 0] if one else out
    t = t - t[0]
    half = max(float(window_s) / 2.0, 1e-6)
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    # Window sums from prefix sums, taken in blocks of 20 half-windows with the block's own origin, so
    # the powers stay small and the centred moments do not lose precision to cancellation.
    blk = np.floor(t / (20.0 * half)).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, blk[1:] != blk[:-1]])
    ends = np.r_[starts[1:], n]
    W_all = np.isfinite(V)
    X_all = np.where(W_all, V, 0.0)
    W_all = W_all.astype(float)
    for s, e in zip(starts, ends):
        a0, a1 = lo[s], hi[e - 1]
        u = ((t[a0:a1] - t[s]) / half)[:, None]
        w, x = W_all[a0:a1], X_all[a0:a1]
        up = [np.ones_like(u), u, u * u, u ** 3, u ** 4]
        P = [np.vstack([np.zeros((1, V.shape[1])), np.cumsum(w * up[p], 0)]) for p in range(5)]
        Q = [np.vstack([np.zeros((1, V.shape[1])), np.cumsum(x * up[p], 0)]) for p in range(3)]
        i0, i1 = lo[s:e] - a0, hi[s:e] - a0
        S = [P[p][i1] - P[p][i0] for p in range(5)]
        T = [Q[p][i1] - Q[p][i0] for p in range(3)]
        a = ((t[s:e] - t[s]) / half)[:, None]
        M = [S[0], S[1] - a * S[0], S[2] - 2 * a * S[1] + a * a * S[0],
             S[3] - 3 * a * S[2] + 3 * a * a * S[1] - a ** 3 * S[0],
             S[4] - 4 * a * S[3] + 6 * a * a * S[2] - 4 * a ** 3 * S[1] + a ** 4 * S[0]]
        N = [T[0], T[1] - a * T[0], T[2] - 2 * a * T[1] + a * a * T[0]]
        res = _local_fit_value(M, N, order)
        out[s:e] = np.where(np.isfinite(res), res, V[s:e])
    return out[:, 0] if one else out


def savgol_time(t_s, values, window_s, order=SAVGOL_POLYORDER):
    """Local polynomial least squares over the reads within +/- window_s/2 of each read, on real times.

    Evaluated at the read itself, so it is Savitzky-Golay for unevenly spaced reads (on evenly spaced
    reads it equals scipy's savgol_filter away from the ends). Where a window holds too few reads for the
    quadratic it falls back to a line, then to the window's mean. Reads must be in time order.
    """
    return savgol_time_columns(t_s, values, window_s, order)


def rolling_smooth_xy(data, agg, window_value, time_based):
    """Centred (zero-phase) rolling mean/median of location_x/location_y per tag.

    ``time_based``: the window is ``window_value`` seconds, evaluated on each tag's Timestamp, so the
    same stretch of time is smoothed whatever the read rate. Otherwise it is ``window_value``
    consecutive reads (at least 3). Partial windows at the ends use the reads there (min_periods=1).
    Writes smoothed_x/smoothed_y into ``data`` and returns it.
    """
    def _agg(roller):
        return roller.mean() if agg == 'mean' else roller.median()

    if time_based:
        win = f"{int(window_value)}s"
        data['smoothed_x'] = np.nan
        data['smoothed_y'] = np.nan
        for _, g in data.groupby('shortid'):
            g_sorted = g.sort_values('Timestamp')
            idx = pd.DatetimeIndex(g_sorted['Timestamp'])
            for src, dst in (('location_x', 'smoothed_x'), ('location_y', 'smoothed_y')):
                s = pd.Series(g_sorted[src].to_numpy(), index=idx)
                rolled = _agg(s.rolling(win, center=True, min_periods=1))
                data.loc[g_sorted.index, dst] = rolled.to_numpy()
    else:
        window_size = max(3, int(window_value))
        for src, dst in (('location_x', 'smoothed_x'), ('location_y', 'smoothed_y')):
            data[dst] = data.groupby('shortid')[src].transform(
                lambda x: _agg(x.rolling(window=window_size, center=True, min_periods=1)))
    return data


def _seconds(ts):
    """A Timestamp column (tz-aware or not) as float seconds since the epoch.

    Through as_unit('ns'): pandas keeps the resolution it was built with (to_datetime(..., unit='ms')
    is datetime64[ms] in pandas 3), so asi8 alone is ms or ns depending on where the column came from.
    """
    return pd.DatetimeIndex(ts).as_unit("ns").asi8 / 1e9


def smooth_xy(data, method, window, units):
    """Smooth location_x/location_y per tag into smoothed_x/smoothed_y with ``method``.

    ``window``/``units`` as in the module docstring; Seconds needs a Timestamp column. "None" (or an
    unknown method) leaves ``data`` as it is. Returns ``data``.
    """
    m = clean_method(method)
    if m not in DEFAULT_WINDOW:
        return data
    window = max(1, int(window))
    by_time = units == SECONDS
    if m in ROLLING_METHODS:
        return rolling_smooth_xy(data, 'mean' if m == "Rolling Average" else 'median', window, by_time)
    if not by_time:
        if m == SAVGOL_METHOD:
            f = lambda s: savgol_samples(s.to_numpy(float), window)  # noqa: E731
        else:
            f = lambda s: forward_backward_ewma(s, window)  # noqa: E731
        for src, dst in (('location_x', 'smoothed_x'), ('location_y', 'smoothed_y')):
            data[dst] = data.groupby('shortid')[src].transform(f)
        return data
    data['smoothed_x'] = np.nan
    data['smoothed_y'] = np.nan
    for _, g in data.groupby('shortid'):
        g_sorted = g.sort_values('Timestamp', kind='stable')
        t = _seconds(g_sorted['Timestamp'])
        xy = g_sorted[['location_x', 'location_y']].to_numpy(float)
        if m == SAVGOL_METHOD:
            sm = savgol_time_columns(t, xy, window)
        else:
            sm = np.column_stack([forward_backward_ewma_time(t, xy[:, k], window) for k in (0, 1)])
        data.loc[g_sorted.index, 'smoothed_x'] = sm[:, 0]
        data.loc[g_sorted.index, 'smoothed_y'] = sm[:, 1]
    return data


def describe(method, window, units):
    """One line for logs and configs, e.g. 'Savitzky-Golay, 10 s (order 2)'."""
    m = clean_method(method)
    if m not in DEFAULT_WINDOW:
        return m
    unit = "s" if units == SECONDS else "reads"
    extra = {SAVGOL_METHOD: " (order 2)",
             EWMA_METHOD: (f" span (half-life {ewma_halflife_s(window):.2f} s)" if units == SECONDS
                           else " span")}.get(m, "")
    return f"{m}, {int(window)} {unit}{extra}"
