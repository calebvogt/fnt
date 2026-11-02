# Adapted from GuPPy (Guided Photometry Analysis in Python), https://github.com/LernerLab/GuPPy
# src/guppy/analysis/control_fit.py, z_score.py and transients.py (2026-11-02, main branch).
# GuPPy is BSD-3-Clause licensed; its licence is reproduced below as that licence requires.
# Cite: Sherathiya VN, Schaid MD, Seiler JL, Lopez GC, Lerner TN (2021). GuPPy, a Python toolbox
# for the analysis of fiber photometry data. Scientific Reports 11:24212.
# https://doi.org/10.1038/s41598-021-03626-9
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Talia Lerner and Venus Sherathiya
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""The parts of GuPPy's analysis FNT uses, rewritten for NaN-masked traces.

- ``fit_control_irwls``: GuPPy's default control fit, a robust (Tukey biweight) regression of
  the signal on the control. GuPPy calls statsmodels' RLM; FNT does not depend on statsmodels, so
  this is the same estimator (Tukey biweight, c = 4.685, MAD scale) solved by iteratively
  reweighted least squares in NumPy, started from a Huber fit.
- ``delta_ff``: GuPPy's dF/F, (signal - fitted control) / fitted control x 100.
- ``z_score``: GuPPy's three z-scores (whole-session, baseline-window, modified median/MAD).
- ``detect_transients``: GuPPy's two-threshold MAD transient detection in fixed windows.

Differences from GuPPy are marked "FNT:" in the code. Samples set to NaN (wireless dropouts)
are left out of every estimate instead of being cut out of the trace.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from scipy.signal import argrelextrema

TUKEY_C = 4.685          # statsmodels TukeyBiweight default (95% efficiency under normal errors)
HUBER_K = 1.345          # statsmodels HuberT default


def fit_control_irwls(control: np.ndarray, signal: np.ndarray, use: Optional[np.ndarray] = None,
                      max_iter: int = 50, tol: float = 1e-8) -> Tuple[float, float]:
    """Robust fit ``signal ~ intercept + slope * control``; returns (slope, intercept).

    ``use`` selects the samples the fit may see (FNT: dropouts excluded). Transients sit far
    above the line and get zero weight, so they do not pull the fit up.
    """
    x = np.asarray(control, float)
    y = np.asarray(signal, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if use is not None:
        keep &= use
    x, y = x[keep], y[keep]
    # Start from OLS; polyfit scales the columns (the control's offset dwarfs its fluctuation).
    slope, intercept = np.polyfit(x, y, 1)
    # FNT: a few Huber iterations first. Tukey's weights fall to zero, so from an OLS start that
    # strong transients have pulled up, the whole bulk of the data can land outside the cutoff
    # and the fit never moves; Huber's weights never vanish and bring the line back first.
    for norm, n_iter in (("huber", 20), ("tukey", max_iter)):
        for _ in range(n_iter):
            r = y - (intercept + slope * x)
            scale = np.median(np.abs(r - np.median(r))) / 0.6745
            if scale <= 0:
                break
            if norm == "huber":
                a = np.abs(r) / (HUBER_K * scale)
                w = np.where(a <= 1, 1.0, 1.0 / np.maximum(a, 1e-12))
            else:
                u = r / (TUKEY_C * scale)
                w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
            if (w > 0).sum() < 3:
                break
            sw = np.sqrt(w)
            xm = np.average(x, weights=w)
            ym = np.average(y, weights=w)
            A = np.column_stack([(x - xm) * sw, sw])
            new_slope, new_c = np.linalg.lstsq(A, (y - ym) * sw, rcond=None)[0]
            new_intercept = ym + new_c - new_slope * xm
            done = abs(new_slope - slope) <= tol * max(1.0, abs(slope)) and                 abs(new_intercept - intercept) <= tol * max(1.0, abs(intercept))
            slope, intercept = new_slope, new_intercept
            if done:
                break
    return float(slope), float(intercept)


def delta_ff(signal: np.ndarray, fitted_control: np.ndarray) -> np.ndarray:
    """Percent dF/F against the fitted control: (signal - fit) / fit x 100."""
    return (np.asarray(signal, float) - fitted_control) / fitted_control * 100


def z_score(dff: np.ndarray, method: str = "standard", time: Optional[np.ndarray] = None,
            baseline: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Z-score a dF/F trace.

    method: ``"standard"`` (whole-session mean and SD), ``"baseline"`` (mean and SD of the
    ``baseline`` = (start, end) window in seconds of ``time``) or ``"modified"`` (robust:
    0.6745 x (x - median) / MAD). NaN samples are ignored and stay NaN.
    """
    x = np.asarray(dff, float)
    if method == "standard":
        return (x - np.nanmean(x)) / np.nanstd(x)
    if method == "baseline":
        if time is None or baseline is None:
            raise ValueError("a baseline z-score needs time and a (start, end) baseline window")
        sel = (time > baseline[0]) & (time < baseline[1]) & np.isfinite(x)
        if not sel.any():
            raise ValueError(f"no samples in the baseline window {baseline} s "
                             f"(trace spans {np.nanmin(time):.4g}-{np.nanmax(time):.4g} s)")
        return (x - np.mean(x[sel])) / np.std(x[sel])
    if method == "modified":
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        return 0.6745 * (x - med) / mad
    raise ValueError(f"unknown z-score method {method!r}: use standard, baseline or modified")


def detect_transients(z: np.ndarray, time: np.ndarray, sampling_rate: float, window_s: float = 15.0,
                      high_amp: float = 2.0, threshold: float = 3.0):
    """GuPPy's transient detection: in each ``window_s`` window, drop samples above
    median + ``high_amp`` x MAD, recompute median and MAD from the rest, and call every local
    maximum above median + ``threshold`` x MAD a transient.

    Returns (peak_indices, peak_times, peak_amplitudes, events_per_min). FNT: NaN samples are
    skipped inside each window rather than removed from the trace first, so peak indices refer
    to the original trace.
    """
    z = np.asarray(z, float)
    n_win = max(1, math.ceil(sampling_rate * window_s))
    peaks = []
    for start in range(0, len(z), n_win):
        seg = z[start:start + n_win]
        ok = np.isfinite(seg)
        if ok.sum() < 3:
            continue
        vals = seg[ok]
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        low = vals[vals <= med + high_amp * mad]
        med2 = np.median(low)
        mad2 = np.median(np.abs(low - med2))
        thr = med2 + threshold * mad2
        above = np.where(ok & (seg > thr), seg, 0.0)
        idx = argrelextrema(above, np.greater)[0]
        peaks.extend((idx + start).tolist())
    peaks = np.asarray(peaks, int)
    minutes = np.isfinite(z).sum() / sampling_rate / 60
    rate = len(peaks) / minutes if minutes > 0 else float("nan")
    return peaks, time[peaks], z[peaks], rate
