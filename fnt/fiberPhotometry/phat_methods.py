# Adapted from PhAT (Photometry Analysis Toolkit), Donaldson lab, https://github.com/donaldsonlab/PhAT
# FiberPho_Main/FiberClass.py: fit_exp, fit_lin and normalize_a_signal (2026-11-02, main branch).
# PhAT is MIT licensed; its licence is reproduced below as that licence requires.
# Cite: Murphy KZ, Haile ED, McTigue AD, Pierce AF, Donaldson ZR (2023). PhAT: A flexible
# open-source GUI-driven toolkit for photometry analysis. Current Protocols 3:e763.
# https://doi.org/10.1002/cpz1.763
#
# MIT License
#
# Copyright (c) 2023 Donaldson Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""PhAT's normalisation, for recordings long enough to photobleach.

PhAT fits a biexponential decay, y = A e^(-Bt) + C e^(-Dt) + E, to each channel and divides the
channel by it (rejecting the fit when its R^2 is below a threshold and dividing by the median
instead). It then scales the normalised reference onto the normalised signal, by least squares or
by a robust match of medians and interquartile ranges, and divides: the result is a ratio around 1.
FNT reports it as percent change, (ratio - 1) x 100.

FNT: NaN samples (wireless dropouts) are left out of every fit.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


def biexponential(t, a, b, c, d, e):
    """A e^(-Bt) + C e^(-Dt) + E (PhAT ``fit_exp``)."""
    t = np.asarray(t, float)
    return a * np.exp(-b * t) + c * np.exp(-d * t) + e


def biexponential_fit(time: np.ndarray, x: np.ndarray, r2_threshold: float = 0.5):
    """Fit the biexponential to one channel; returns (fit, r_squared, params).

    As in PhAT the channel is scaled by 1.5 x its maximum for the fit, with A, C >= 0,
    0 <= B <= 10, 0 <= D <= 100 and 0 <= E <= 1 (in the scaled units). A fit with R^2 below
    ``r2_threshold`` is rejected and the median is returned as a flat fit instead.
    """
    t = np.asarray(time, float) - time[0]       # FNT: time from the recording's start
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    scale = 1.5 * np.nanmax(x)
    try:
        popt, _ = curve_fit(biexponential, t[ok], x[ok] / scale, p0=(1.0, 0, 1.0, 0, 0),
                            bounds=([0, 0, 0, 0, 0], [np.inf, 10, np.inf, 100, 1]), maxfev=20000)
        fit = scale * biexponential(t, *popt)
        r2 = float(np.corrcoef(x[ok], fit[ok])[0, 1] ** 2)
    except (RuntimeError, ValueError):
        popt, r2 = None, 0.0
    if popt is None or r2 < r2_threshold:
        return np.full_like(x, np.nanmedian(x)), r2, None
    return fit, r2, popt


def fit_reference(normed_ref: np.ndarray, normed_sig: np.ndarray, method: str = "robust",
                  use: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Scale the normalised reference onto the normalised signal; returns (slope, intercept).

    ``"least squares"``: PhAT's bounded least-squares line (0 <= slope <= 100,
    -1 <= intercept <= 1). ``"robust"``: slope = IQR(signal) / IQR(reference),
    intercept = median(signal) - slope x median(reference).
    """
    r = np.asarray(normed_ref, float)
    s = np.asarray(normed_sig, float)
    ok = np.isfinite(r) & np.isfinite(s)
    if use is not None:
        ok &= use
    r, s = r[ok], s[ok]
    if method == "least squares":
        popt, _ = curve_fit(lambda x, a, b: a * x + b, r, s, bounds=([0, -1], [100, 1]))
        return float(popt[0]), float(popt[1])
    if method == "robust":
        sq75, sq25 = np.percentile(s, [75, 25])
        rq75, rq25 = np.percentile(r, [75, 25])
        slope = (sq75 - sq25) / (rq75 - rq25)
        return float(slope), float(np.median(s) - slope * np.median(r))
    raise ValueError(f"unknown reference fit {method!r}: use 'robust' or 'least squares'")
