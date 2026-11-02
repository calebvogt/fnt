"""Fiber photometry processing on synthetic recordings with known answers.

A synthetic recording: a slowly bleaching baseline, shared motion artefacts in both channels,
known transients in the signal only, and wireless dropouts (full drops to ~0 V and a partial
step). Every pipeline must recover the transients, ignore the artefacts and blank the dropouts.
"""
import numpy as np
import pytest
from scipy.signal import filtfilt, firwin

from fnt.fiberPhotometry import guppy_methods, phat_methods
from fnt.fiberPhotometry.processing import (
    TraditionalDFFCalculator, bridge, find_dropouts, fit_control_matlab, process)

FS = 100.0


def recording(seed=0, minutes=5, dropouts=True):
    rng = np.random.default_rng(seed)
    t = np.arange(int(minutes * 60 * FS)) / FS
    bleach = 1 + 0.15 * np.exp(-t / 120)
    motion = 0.03 * np.sin(2 * np.pi * t / 7) + 0.02 * rng.standard_normal(len(t)).cumsum() / 300
    control = 0.055 * bleach * (1 + motion) + 0.0003 * rng.standard_normal(len(t))
    peaks = np.arange(10, t[-1] - 5, 9.0)
    transient = np.zeros_like(t)
    for p in peaks:
        dt = t - p
        transient += np.where(dt >= 0, 0.08 * np.exp(-np.clip(dt, 0, None) / 0.4), 0.0)  # 8%
    signal = 0.064 * bleach * (1 + motion) * (1 + transient) + 0.0003 * rng.standard_normal(len(t))
    truth = np.zeros(len(t), bool)
    if dropouts:
        for start in rng.choice(len(t) - 50, 20, replace=False):
            signal[start:start + 20] = 0.001
            control[start:start + 20] = 0.001
            truth[start:start + 20] = True
        s = len(t) // 2
        signal[s:s + 30] *= 0.85          # partial step: both channels, by different amounts
        control[s:s + 30] *= 0.75
        truth[s:s + 30] = True
    return t, signal, control, peaks, truth


def transient_amplitude(r, peaks):
    amps = []
    for p in peaks:
        i = int(p * FS)
        pre = np.nanmean(r.dff[i - 150:i - 20])
        amps.append(np.nanmax(r.dff[i:i + 30]) - pre)
    return np.nanmedian(amps)


@pytest.mark.parametrize("pipeline", ["matlab", "guppy", "phat"])
def test_each_pipeline_recovers_the_transients(pipeline):
    t, s, c, peaks, _ = recording()
    r = process(s, c, t, FS, pipeline)
    amp = transient_amplitude(r, peaks)
    assert 5.5 < amp < 9.5, amp                      # 8% transients, smoothed a little
    quiet = np.ones(len(t), bool)
    for p in peaks:
        quiet &= ~((t > p - 0.5) & (t < p + 3))
    assert np.nanstd(r.dff[quiet & ~r.dropout]) < 1.0  # motion cancelled by the control


@pytest.mark.parametrize("pipeline", ["matlab", "guppy", "phat"])
def test_dropouts_are_blank_and_do_not_spike(pipeline):
    t, s, c, peaks, truth = recording()
    r = process(s, c, t, FS, pipeline)
    assert np.isnan(r.dff[truth]).all()
    assert np.nanmax(np.abs(r.dff)) < 20


def test_without_dropout_removal_the_partial_step_is_an_artefact():
    """Both channels step down by different amounts: no control fit cancels it."""
    t, s, c, _, truth = recording()
    r = process(s, c, t, FS, "matlab", remove_dropouts=False)
    step = slice(len(t) // 2 + 5, len(t) // 2 + 25)
    assert np.nanmean(r.dff[step]) > 5           # a fake +5-10% "transient"
    assert np.isnan(process(s, c, t, FS, "matlab").dff[step]).all()


def test_find_dropouts_catches_full_and_partial_drops():
    t, s, c, _, truth = recording()
    found = find_dropouts(s, c, t, FS)
    assert found[truth].all()
    assert found.mean() < truth.mean() + 0.02         # padding only, few false alarms


def test_a_timestamp_gap_is_a_dropout():
    t = np.arange(1000) / FS
    t[500:] += 1.0
    s = np.ones(1000)
    bad = find_dropouts(s, s, t, FS, pad_s=0)
    assert bad[500] and bad.sum() == 1


def test_matlab_pipeline_is_filteredfitcaldff():
    """Without dropout removal the result is filteredfitcaldff.m line by line."""
    t, s, c, _, _ = recording(dropouts=False)
    b = firwin(101, 20 / (FS / 2))
    f473 = filtfilt(b, [1.0], s, padlen=303)
    f405 = filtfilt(b, [1.0], c, padlen=303)
    slope, intercept = np.polyfit(f405, f473, 1)
    expected = 100 * (f473 - (slope * f405 + intercept)) / np.mean(f473)
    r = process(s, c, t, FS, "matlab", remove_dropouts=False)
    np.testing.assert_allclose(r.dff, expected, rtol=0, atol=1e-10)


def test_fitcaldff_without_filter_and_negative_slope_uses_nonnegative_scaling():
    c = np.linspace(1, 2, 200)
    s = 3 - c                                   # negative slope
    fitted, info = fit_control_matlab(c, s)
    assert info["fit"] == "non-negative scaling"
    k = np.dot(c, s) / np.dot(c, c)
    np.testing.assert_allclose(fitted, max(k, 0) * c)


def test_traditional_calculator_keeps_its_interface():
    t, s, c, _, truth = recording()
    calc = TraditionalDFFCalculator()
    dff, fs_, fc, fit = calc.calculate(s, c, FS, t)
    assert len(dff) == len(s) and calc.last_dropout_count >= truth.sum()


def test_robust_fit_ignores_transients():
    rng = np.random.default_rng(1)
    x = 1 + 0.1 * rng.standard_normal(5000)
    y = 2 * x + 0.5 + 0.01 * rng.standard_normal(5000)
    y[::10] += 1.0                               # 10% of samples are big positive transients
    slope, intercept = guppy_methods.fit_control_irwls(x, y)
    assert abs(slope - 2) < 0.02 and abs(intercept - 0.5) < 0.03
    ols = np.polyfit(x, y, 1)
    assert abs(ols[1] - 0.5) > 0.05              # least squares is pulled up


def test_z_scores():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(10000) * 3 + 5
    z = guppy_methods.z_score(x, "standard")
    assert abs(np.mean(z)) < 1e-9 and abs(np.std(z) - 1) < 1e-9
    zm = guppy_methods.z_score(x, "modified")
    assert abs(np.median(zm)) < 1e-9
    t = np.arange(10000) / FS
    zb = guppy_methods.z_score(x, "baseline", t, (0, 10))
    sel = (t > 0) & (t < 10)
    assert abs(np.mean(zb[sel])) < 1e-9
    with pytest.raises(ValueError):
        guppy_methods.z_score(x, "baseline", t, (500, 600))


def test_transient_detection_finds_the_planted_peaks():
    t, s, c, peaks, _ = recording(dropouts=False)
    r = process(s, c, t, FS, "guppy")
    idx, times, amps, rate = guppy_methods.detect_transients(r.z, r.time, FS)
    for p in peaks:
        assert np.min(np.abs(times - p)) < 0.3


def test_biexponential_fit_and_rejection():
    t = np.arange(30000) / FS
    x = 0.06 * (1 + 0.2 * np.exp(-t / 60) + 0.05 * np.exp(-t / 5))
    fit, r2, popt = phat_methods.biexponential_fit(t, x)
    assert r2 > 0.99 and np.max(np.abs(fit - x) / x) < 0.01
    flat = 0.06 + 0.001 * np.random.default_rng(3).standard_normal(len(t))
    fit, r2, popt = phat_methods.biexponential_fit(t, flat, r2_threshold=0.5)
    assert popt is None and np.allclose(fit, np.median(flat))


def test_bridge():
    x = np.array([1.0, 2, 0, 0, 5])
    bad = x == 0
    np.testing.assert_allclose(bridge(x, bad), [1, 2, 3, 4, 5])


def test_notch_removes_a_line_and_keeps_the_rest():
    from fnt.fiberPhotometry.processing import notch
    t = np.arange(60000) / FS
    slow = np.sin(2 * np.pi * 0.5 * t)
    line = 0.2 * np.sin(2 * np.pi * 3.32 * t)
    out = notch(slow + line, FS, [3.32])
    resid = out - slow
    assert np.std(resid[1000:-1000]) < 0.01 * np.std(line)
    assert np.corrcoef(out, slow)[0, 1] > 0.999


def test_phat_bleaching_option():
    """A slow decline in the signal only: PhAT's own-bleaching removes it, 'control' keeps it."""
    t, s, c, _, _ = recording(dropouts=False, minutes=10)
    s = s * (1 - 0.03 * t / t[-1])            # the signal falls 3% more than the control
    own = process(s, c, t, FS, "phat", phat_bleaching="each channel")
    ctl = process(s, c, t, FS, "phat", phat_bleaching="control")
    early, late = t < 60, t > t[-1] - 60
    drop_own = np.nanmean(own.dff[late]) - np.nanmean(own.dff[early])
    drop_ctl = np.nanmean(ctl.dff[late]) - np.nanmean(ctl.dff[early])
    assert abs(drop_own) < 1.0
    assert drop_ctl < -2.0
    with pytest.raises(ValueError):
        process(s, c, t, FS, "phat", phat_bleaching="neither")


def test_guppy_is_the_default():
    from fnt.fiberPhotometry.batch import Settings
    t, s, c, _, _ = recording(dropouts=False, minutes=1)
    assert process(s, c, t, FS).pipeline == "guppy" and Settings().pipeline == "guppy"
