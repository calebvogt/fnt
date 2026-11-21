"""Every smoothing method takes its window in Seconds or in Samples.

Tags at different read rates (VoleCosm: ~7, ~2 and ~0.7 reads/s) smooth very
different stretches of time under a window counted in reads: Savitzky-Golay
over 31 reads is ~4 s of a fast tag and ~40 s of a slow one. In Seconds every
method smooths the same stretch of time on every tag. Old configs, which
carried no units for Savitzky-Golay or EWMA, must still load as what they ran.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from fnt.uwb import smoothing as S  # noqa: E402


def _path(rate_hz, seconds=1800, noise=0.0, seed=0):
    """A smooth walk around the arena read at ``rate_hz`` (irregular times)."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, seconds, int(seconds * rate_hz)))
    x = 2 + np.sin(t / 40.0) + 0.3 * np.sin(t / 7.0)
    y = 1 + np.cos(t / 55.0)
    return t, x + rng.normal(0, noise, t.size), y + rng.normal(0, noise, t.size)


def test_time_savgol_is_savgol_on_even_reads():
    rng = np.random.default_rng(1)
    dt = 0.134
    t = 1.79e9 + np.arange(5000) * dt
    v = np.cumsum(rng.normal(0, 0.05, t.size))
    ref = savgol_filter(v, 31, 2)
    got = S.savgol_time(t, v, 30 * dt + 1e-6)
    np.testing.assert_allclose(got[15:-15], ref[15:-15], atol=1e-6)


def test_time_savgol_keeps_a_quadratic_and_survives_sparse_windows():
    t = np.r_[np.arange(0, 50, 0.5), 200.0, 400.0]          # two isolated reads
    v = 0.01 * t ** 2 - t + 3
    out = S.savgol_time(t, v, 10)
    np.testing.assert_allclose(out[:100], v[:100], atol=1e-6)
    assert np.isfinite(out).all() and out[-1] == v[-1]      # a lone read stays put


@pytest.mark.parametrize("method", [S.SAVGOL_METHOD, S.EWMA_METHOD, "Rolling Average", "Rolling Median"])
def test_seconds_smooth_the_same_time_at_any_read_rate(method):
    """On a noise-free path, a fast and a slow tag smoothed in Seconds agree
    with each other far better than in Samples."""
    def run(rate, window, units):
        t, x, y = _path(rate, seed=int(rate * 10))
        # built from epoch ms as the export does: datetime64[ms] in pandas 3, which once made a
        # 10 s window 10,000 s long
        ms = np.round(t * 1000).astype(np.int64) + 1_792_000_000_000
        df = pd.DataFrame({"shortid": 1, "Timestamp": pd.to_datetime(ms, unit="ms", utc=True),
                           "location_x": x, "location_y": y})
        out = S.smooth_xy(df, method, window, units)
        return t, out["smoothed_x"].to_numpy()

    def gap(window, units):
        tf, xf = run(7.0, window, units)
        ts, xs = run(0.7, window, units)
        keep = (ts > 120) & (ts < ts.max() - 120)
        return np.median(np.abs(np.interp(ts[keep], tf, xf) - xs[keep]))

    by_time = gap(10, S.SECONDS)
    by_reads = gap(31 if method == S.SAVGOL_METHOD else 20, S.SAMPLES)
    assert by_time < 0.02, by_time
    if method in (S.SAVGOL_METHOD, S.EWMA_METHOD):
        assert by_reads > 3 * by_time, (by_reads, by_time)


def test_ewma_seconds_halflife():
    assert S.ewma_halflife_s(10) == pytest.approx(np.log(2) * 5)
    # a step: after one half-life of the forward pass alone the weight is halved
    t = np.arange(0, 200, 0.1)
    v = (t >= 100).astype(float)
    out = S.forward_backward_ewma_time(t, v, 10)
    assert out[t < 50].max() < 1e-3 and out[t > 150].min() > 1 - 1e-3
    assert 0.3 < out[np.searchsorted(t, 100)] < 0.7      # centred: no lag


def test_samples_are_the_old_behaviour():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"shortid": np.repeat([3, 5], 300),
                       "Timestamp": pd.to_datetime(np.tile(np.arange(300) * 0.3, 2), unit="s", utc=True),
                       "location_x": rng.normal(size=600), "location_y": rng.normal(size=600)})
    sg = S.smooth_xy(df.copy(), S.SAVGOL_METHOD, 31, S.SAMPLES)
    np.testing.assert_allclose(sg.smoothed_x[:300], savgol_filter(df.location_x[:300], 31, 2))
    ew = S.smooth_xy(df.copy(), S.EWMA_METHOD, 20, S.SAMPLES)
    np.testing.assert_allclose(ew.smoothed_x[300:],
                               S.forward_backward_ewma(df.location_x[300:], 20))


@pytest.mark.parametrize("method,window,units,span,want", [
    ("Savitzky-Golay", None, None, 20, (31, "Samples")),        # SG ignored the window; span was EWMA's
    ("Savitzky-Golay", 30, None, None, (31, "Samples")),
    ("Savitzky-Golay", 10, "Seconds", None, (10, "Seconds")),
    ("Forward-Backward EWMA", None, None, 15, (15, "Samples")),
    (S.EWMA_METHOD, 30, None, None, (30, "Samples")),           # pre-split: span stored as window
    (S.EWMA_METHOD, 10, "Seconds", None, (10, "Seconds")),
    ("Rolling Average", 45, "Seconds", None, (45, "Seconds")),
    ("Rolling Median (default)", 9, "Samples", None, (9, "Samples")),
    ("None", 30, "Seconds", None, (None, None)),
])
def test_old_configs_read_as_what_they_ran(method, window, units, span, want):
    assert S.resolve_window(method, window, units, span) == want


# --- the window ------------------------------------------------------------ #

_APP = []


@pytest.fixture
def win():
    from PyQt5.QtWidgets import QApplication
    from fnt.uwb.uwb_preprocessing_pyqt import UWBQuickVisualizationWindow
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    w = UWBQuickVisualizationWindow()
    w.db_path = os.path.join("C:", os.sep, "nowhere", "T000_fake.sqlite")
    yield w
    w.close()


def _load(w, cfg):
    w._pending_config_override = dict({"timezone": "US/Mountain"}, **cfg)
    w.load_config_if_exists()
    w._sync_export_from_preview()


def test_a_volecosm_config_still_runs_31_read_savgol(win):
    # as written by FNT before the units choice (VC-T013's export)
    _load(win, {"smoothing_method": "Savitzky-Golay", "rolling_window": None,
                "rolling_window_units": None, "smoothing_span": 20,
                "preview": {"smoothing": "Savitzky-Golay", "window": None,
                            "window_units": None, "smoothing_span": 20}})
    assert (win.spin_preview_window.value(), win.combo_preview_window_units.currentText()) == (31, "Samples")
    assert (win.spin_rolling_window.value(), win.combo_window_units.currentText()) == (31, "Samples")


def test_a_legacy_config_without_a_preview_block(win):
    _load(win, {"smoothing_method": "Forward-Backward EWMA", "rolling_window": None,
                "rolling_window_units": None, "smoothing_span": 15})
    assert win.combo_preview_smoothing.currentText() == S.EWMA_METHOD
    assert (win.spin_rolling_window.value(), win.combo_window_units.currentText()) == (15, "Samples")


def test_seconds_round_trip_through_the_config(win):
    _load(win, {"preview": {"smoothing": S.EWMA_METHOD, "window": 10, "window_units": "Seconds"}})
    cfg = win.get_config_dict()
    assert (cfg["smoothing_method"], cfg["rolling_window"], cfg["rolling_window_units"]) == \
        (S.EWMA_METHOD, 10, "Seconds")
    assert "smoothing_span" not in cfg                       # a span of reads would be a lie here
    assert (cfg["preview"]["window"], cfg["preview"]["window_units"]) == (10, "Seconds")
    _load(win, cfg)
    assert (win.spin_rolling_window.value(), win.combo_window_units.currentText()) == (10, "Seconds")


def test_each_method_keeps_its_own_window(win):
    c, sp, un = win.combo_preview_smoothing, win.spin_preview_window, win.combo_preview_window_units
    c.setCurrentText("Rolling Average")
    assert (sp.value(), un.currentText()) == (30, "Seconds")
    c.setCurrentText("Savitzky-Golay")
    assert (sp.value(), un.currentText()) == (31, "Samples")      # not 30 s left over
    assert un.isVisibleTo(win) or not win.isVisible()
    sp.setValue(10)
    un.setCurrentText("Seconds")
    c.setCurrentText("Rolling Average")
    assert (sp.value(), un.currentText()) == (30, "Seconds")
    c.setCurrentText("Savitzky-Golay")
    assert (sp.value(), un.currentText()) == (10, "Seconds")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
