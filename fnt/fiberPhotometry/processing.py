"""Fiber photometry processing, independent of the acquisition system.

A recording is a signal channel (e.g. 470 nm for a GFP-based sensor such as dLight or GRAB-DA)
and a control channel (405/415 nm, the sensor's isosbestic point), sampled on one time base.
``process`` turns them into dF/F and a z-score with one of three pipelines:

- ``"matlab"`` -- the Warden lab's MATLAB method, as Caleb Vogt used it in his PhD
  (``ReferenceMaterials/fitcaldff.m`` and ``filteredfitcaldff.m``): FIR low-pass, a straight-line fit of
  the control onto the signal (non-negative scaling if the slope is not positive), and
  dF/F = 100 x (signal - fit) / mean(signal). Validated in Caleb's PhD work.
- ``"guppy"`` -- GuPPy's default (Sherathiya et al. 2021): moving-average smoothing, a robust
  (Tukey biweight) fit of the control onto the signal, dF/F = 100 x (signal - fit) / fit.
  See ``guppy_methods.py``. The default (Caleb, 2026-11-02).
- ``"phat"`` -- PhAT's normalisation (Murphy et al. 2023): each channel divided by its own
  biexponential bleaching fit, the reference matched to the signal by medians and interquartile
  ranges, and the signal divided by it. See ``phat_methods.py``.

Wireless systems drop samples: both channels fall to ~0 V for a few to a few hundred
milliseconds (Doric NC500, pilot P001: 0-17% of samples). ``find_dropouts`` marks them; every
pipeline bridges them before filtering (so a filter never rings off a drop to zero), leaves them
out of every fit, and returns NaN there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import nnls
from scipy.signal import filtfilt, firwin

from . import guppy_methods, phat_methods

PIPELINES = ("guppy", "matlab", "phat")      # first = default
PIPELINE_LABELS = {
    "guppy": "Robust fit, dF/F on fitted control (GuPPy)",
    "matlab": "Linear fit, dF/F on mean signal (Warden lab MATLAB)",
    "phat": "Biexponential + robust reference fit (PhAT)",
}


@dataclass
class PhotometryResult:
    """What ``process`` returns; every array is one value per photometry sample."""
    time: np.ndarray
    signal: np.ndarray            # signal channel after bridging dropouts and filtering
    control: np.ndarray           # control channel, the same
    fitted_control: np.ndarray    # the control fitted onto the signal (the dF/F baseline)
    dff: np.ndarray               # percent dF/F; NaN at dropouts
    z: np.ndarray                 # z-score of dF/F; NaN at dropouts
    dropout: np.ndarray           # True where a sample is a dropout (or within pad_s of one)
    pipeline: str
    info: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dropouts
# ---------------------------------------------------------------------------

def find_dropouts(signal: np.ndarray, control: Optional[np.ndarray] = None,
                  time: Optional[np.ndarray] = None, sampling_rate: Optional[float] = None,
                  fraction: float = 0.5, pad_s: float = 0.1, local_fraction: Optional[float] = 0.85,
                  local_window_s: float = 10.0) -> np.ndarray:
    """Samples lost to the wireless link.

    A sample is a dropout when either channel is below ``fraction`` x that channel's median
    (a full drop to ~0 V), or below ``local_fraction`` x its running median over
    ``local_window_s`` (a partial drop: in pilot P001 both channels step down 14-24% for
    0.1-0.5 s, by different amounts, so no control fit cancels it), or when it follows a gap in
    the timestamps longer than three sample intervals. Each dropout is widened by ``pad_s`` on
    both sides, since the edges of a drop are partial values.
    """
    if sampling_rate is None:
        sampling_rate = 1 / np.median(np.diff(time)) if time is not None else 1.0
    channels = [np.asarray(signal, float)] + ([] if control is None else [np.asarray(control, float)])
    bad = np.zeros(len(channels[0]), bool)
    for x in channels:
        bad |= x < fraction * np.nanmedian(x)
    if local_fraction:
        from scipy.ndimage import median_filter
        w = max(3, int(round(local_window_s * sampling_rate)) | 1)
        for x in channels:
            # running median over samples that are not full drops, so a long drop cannot drag it
            bridged = bridge(x, bad)
            bad |= bridged < local_fraction * median_filter(bridged, w, mode="nearest")
    if time is not None and len(time) > 1:
        dt = np.diff(time)
        gap = np.zeros(len(time), bool)
        gap[1:] = dt > 3 * np.median(dt)
        bad |= gap
    if pad_s > 0 and bad.any():
        k = int(round(pad_s * sampling_rate))
        if k > 0:
            bad = np.convolve(bad.astype(float), np.ones(2 * k + 1), mode="same") > 0
    return bad


def bridge(x: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Linear interpolation across ``bad`` samples (constant at the ends)."""
    x = np.asarray(x, float).copy()
    if bad.any() and (~bad).sum() > 1:
        good = np.flatnonzero(~bad)
        x[bad] = np.interp(np.flatnonzero(bad), good, x[good])
    return x


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def fir_lowpass(x: np.ndarray, sampling_rate: float, cutoff: float = 20.0, order: int = 100):
    """Zero-phase FIR low-pass, as MATLAB ``filtfilt(fir1(order, cutoff/Nyquist), 1, x)``."""
    wn = min(max(cutoff / (sampling_rate / 2), 0.01), 0.99)
    b = firwin(order + 1, wn)
    return filtfilt(b, [1.0], x, padlen=min(3 * (order + 1), len(x) - 1))


def notch(x: np.ndarray, sampling_rate: float, freqs_hz: Sequence[float], q: float = 30.0):
    """Remove narrow lines at ``freqs_hz``: one zero-phase IIR notch (scipy ``iirnotch``) per
    frequency, each ``f / q`` Hz wide at -3 dB."""
    from scipy.signal import iirnotch
    y = np.asarray(x, float)
    for f0 in freqs_hz:
        if 0 < f0 < sampling_rate / 2:
            b, a = iirnotch(f0, q, sampling_rate)
            y = filtfilt(b, a, y)
    return y


def moving_average(x: np.ndarray, sampling_rate: float, window_s: float = 0.1):
    """Centred moving average over ``window_s`` seconds (GuPPy smooths this way)."""
    n = max(1, int(round(window_s * sampling_rate)))
    return uniform_filter1d(np.asarray(x, float), size=n, mode="nearest")


# ---------------------------------------------------------------------------
# Control fits
# ---------------------------------------------------------------------------

def fit_control_matlab(control: np.ndarray, signal: np.ndarray, use: Optional[np.ndarray] = None):
    """``fitcaldff.m``: polyfit(control, signal, 1); if the slope is not positive, the
    non-negative scaling lsqnonneg(control, signal) instead. Returns the fitted control."""
    c = np.asarray(control, float)
    s = np.asarray(signal, float)
    sel = np.ones(len(c), bool) if use is None else use
    slope, intercept = np.polyfit(c[sel], s[sel], 1)
    if slope <= 0:
        k = nnls(c[sel].reshape(-1, 1), s[sel])[0][0]
        return k * c, {"slope": float(k), "intercept": 0.0, "fit": "non-negative scaling"}
    return slope * c + intercept, {"slope": float(slope), "intercept": float(intercept),
                                   "fit": "linear"}


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def process(signal: np.ndarray, control: Optional[np.ndarray], time: np.ndarray,
            sampling_rate: Optional[float] = None, pipeline: str = "guppy", *,
            remove_dropouts: bool = True, dropout_fraction: float = 0.5, dropout_pad_s: float = 0.1,
            fir_cutoff: float = 20.0, fir_order: int = 100, apply_filter: bool = True,
            smooth_s: float = 0.1, biexp_r2: float = 0.5, reference_fit: str = "robust",
            phat_bleaching: str = "each channel", notch_hz: Sequence[float] = (),
            notch_q: float = 30.0, zscore_method: str = "standard",
            zscore_baseline: Optional[Tuple[float, float]] = None) -> PhotometryResult:
    """dF/F and z-score for one recording. ``control`` may be None (no isosbestic channel):
    the signal is then expressed against its own mean (``matlab``/``guppy``) or bleaching fit
    (``phat``).

    ``notch_hz``: frequencies removed from both channels (after bridging dropouts, before any
    smoothing) by zero-phase notch filters of quality ``notch_q`` (width = f / Q), e.g. the
    3.32 and 9.94 Hz lines of the Doric NC500 in pilot P001.

    ``phat_bleaching`` (PhAT only): ``"each channel"`` divides each channel by its own
    bleaching fit, as PhAT does, which also removes any slow change the signal has and the
    control does not; ``"control"`` divides both by the control's bleaching curve, so a slow
    change in the signal alone is kept (as the MATLAB and GuPPy pipelines keep it)."""
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}: use one of {PIPELINES}")
    time = np.asarray(time, float)
    n = min(len(signal), len(time), len(control) if control is not None else len(signal))
    sig = np.asarray(signal, float)[:n]
    ctl = None if control is None else np.asarray(control, float)[:n]
    time = time[:n]
    fs = sampling_rate or 1 / np.median(np.diff(time))

    bad = (find_dropouts(sig, ctl, time, fs, dropout_fraction, dropout_pad_s)
           if remove_dropouts else np.zeros(n, bool))
    use = ~bad
    sig_b = bridge(sig, bad)
    ctl_b = None if ctl is None else bridge(ctl, bad)
    info = {"dropout_fraction": float(bad.mean()), "sampling_rate": float(fs)}
    if len(notch_hz):
        sig_b = notch(sig_b, fs, notch_hz, notch_q)
        ctl_b = None if ctl_b is None else notch(ctl_b, fs, notch_hz, notch_q)
        info["notch_hz"] = [float(f) for f in notch_hz]

    if pipeline == "matlab":
        if apply_filter:
            sig_f = fir_lowpass(sig_b, fs, fir_cutoff, fir_order)
            ctl_f = None if ctl_b is None else fir_lowpass(ctl_b, fs, fir_cutoff, fir_order)
        else:
            sig_f, ctl_f = sig_b, ctl_b
        if ctl_f is None:
            fitted = np.full(n, np.mean(sig_f[use]))
        else:
            fitted, fit_info = fit_control_matlab(ctl_f, sig_f, use)
            info.update(fit_info)
        dff = 100 * (sig_f - fitted) / np.mean(sig_f[use])

    elif pipeline == "guppy":
        sig_f = moving_average(sig_b, fs, smooth_s)
        ctl_f = None if ctl_b is None else moving_average(ctl_b, fs, smooth_s)
        if ctl_f is None:
            fitted = np.full(n, np.mean(sig_f[use]))
        else:
            slope, intercept = guppy_methods.fit_control_irwls(ctl_f, sig_f, use)
            fitted = slope * ctl_f + intercept
            info.update(slope=slope, intercept=intercept, fit="robust (Tukey biweight)")
        dff = guppy_methods.delta_ff(sig_f, fitted)

    else:  # phat
        if phat_bleaching not in ("each channel", "control"):
            raise ValueError(f"phat_bleaching must be 'each channel' or 'control', not {phat_bleaching!r}")
        sig_f = moving_average(sig_b, fs, smooth_s)
        ctl_f = None if ctl_b is None else moving_average(ctl_b, fs, smooth_s)
        if ctl_f is not None:
            ctl_exp, r2c, _ = phat_methods.biexponential_fit(time, np.where(use, ctl_f, np.nan),
                                                             biexp_r2)
        if ctl_f is not None and phat_bleaching == "control":
            # the control's bleaching curve, in the signal's units
            sig_exp, r2s = ctl_exp * np.median(sig_f[use]) / np.median(ctl_f[use]), r2c
        else:
            sig_exp, r2s, _ = phat_methods.biexponential_fit(time, np.where(use, sig_f, np.nan),
                                                             biexp_r2)
        normed_sig = sig_f / sig_exp
        info.update(signal_biexp_r2=r2s, bleaching=phat_bleaching)
        if ctl_f is None:
            adjusted = np.ones(n)
            fitted = sig_exp
        else:
            normed_ctl = ctl_f / ctl_exp
            slope, intercept = phat_methods.fit_reference(normed_ctl, normed_sig, reference_fit, use)
            adjusted = slope * normed_ctl + intercept
            fitted = adjusted * sig_exp          # the baseline in the signal's own units
            info.update(control_biexp_r2=r2c, slope=slope, intercept=intercept,
                        fit=f"{reference_fit} reference fit")
        dff = (normed_sig / adjusted - 1) * 100

    dff = np.where(bad, np.nan, dff)
    z = guppy_methods.z_score(dff, zscore_method, time, zscore_baseline)
    return PhotometryResult(time=time, signal=sig_f, control=ctl_f if ctl_f is not None else
                            np.full(n, np.nan), fitted_control=fitted, dff=dff, z=z, dropout=bad,
                            pipeline=pipeline, info=info)


class TraditionalDFFCalculator:
    """The MATLAB pipeline with the older call signature the GUI used.

    ``calculate`` returns (dff, filtered_signal, filtered_control, fitted_control); dropouts are
    NaN in ``dff`` and ``last_dropout_count`` says how many samples were dropped.
    """

    def __init__(self, apply_filter: bool = True, filter_cutoff: float = 20.0,
                 filter_order: int = 100, sampling_rate_override: Optional[float] = None,
                 clean_dropouts: bool = True, dropout_threshold: float = 0.5):
        self.apply_filter = apply_filter
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        self.sampling_rate_override = sampling_rate_override
        self.clean_dropouts = clean_dropouts
        self.dropout_threshold = dropout_threshold
        self.last_dropout_count = 0

    def calculate(self, signal, isosbestic, sampling_rate, time=None):
        fs = self.sampling_rate_override or sampling_rate
        n = min(len(signal), len(isosbestic))
        t = np.arange(n) / fs if time is None else np.asarray(time)[:n]
        r = process(signal[:n], isosbestic[:n], t, fs, "matlab",
                    remove_dropouts=self.clean_dropouts, dropout_fraction=self.dropout_threshold,
                    fir_cutoff=self.filter_cutoff, fir_order=self.filter_order,
                    apply_filter=self.apply_filter)
        self.last_dropout_count = int(r.dropout.sum())
        return r.dff, r.signal, r.control, r.fitted_control
