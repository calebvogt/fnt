"""Random downsampling of a tag's reads before filtering (fnt/uwb/downsampling.py).

Tags flashed with different rate pairs (10/1, 4/1, 1/1) feed the thresholds and
the 31-read Savitzky-Golay smoother very different numbers of reads, so the
fast tags are thinned to a common ceiling first. These tests check the ceiling
holds, that slow tags are left alone, and that the pick is reproducible and
independent of how the reads were chunked - the preview processes time slices
and must keep the same reads the export does.

Runs under pytest, or directly (``python test_downsampling.py``).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fnt.uwb import downsampling as DS  # noqa: E402

T0 = int(pd.Timestamp("2026-09-21 12:00", tz="US/Mountain").value // 10**6)


def _track(shortid, hz, seconds=600, jitter=0.05, seed=0):
    """Reads of one tag at ``hz``, with jittered intervals, as the export has them."""
    rng = np.random.default_rng(seed)
    step = 1000.0 / hz
    n = int(seconds * hz)
    ts = (T0 + np.arange(n) * step + rng.normal(0, step * jitter, n)).astype("int64")
    ts = np.unique(ts)
    return pd.DataFrame({"shortid": shortid, "timestamp": ts,
                         "location_x": rng.normal(0, 1, len(ts)),
                         "location_y": rng.normal(0, 1, len(ts))})


def _reads():
    return pd.concat([_track(7, 10.0, seed=1), _track(11, 4.0, seed=2),
                      _track(32, 1.0, jitter=0.0, seed=3)], ignore_index=True)


def test_at_most_one_read_per_tag_per_bin():
    df = _reads()
    out = DS.downsample_reads(df, 1.0)
    bins = out["timestamp"] // 1000
    assert not out.assign(b=bins).duplicated(["shortid", "b"]).any()
    # A 10/1 or 4/1 tag that reports every second keeps one read per second.
    for tag in (7, 11):
        n_sec = (df.loc[df.shortid == tag, "timestamp"] // 1000).nunique()
        assert (out.shortid == tag).sum() == n_sec


def test_slow_tag_left_alone():
    df = _reads()
    out = DS.downsample_reads(df, 1.0)
    assert (out.shortid == 32).sum() == (df.shortid == 32).sum()


def test_rate_below_one():
    df = _track(7, 10.0, seconds=1000)
    out = DS.downsample_reads(df, 0.5)
    bins = np.floor(out["timestamp"] * 0.5 / 1000)
    assert bins.is_unique
    assert abs(len(out) - 500) <= 2


def test_order_and_index_kept():
    df = _reads().sample(frac=1.0, random_state=0)
    out = DS.downsample_reads(df, 1.0)
    assert list(out.index) == [i for i in df.index if i in set(out.index)]


def test_reproducible_and_seeded():
    df = _reads()
    a = DS.downsample_reads(df, 1.0, seed=0)
    b = DS.downsample_reads(df, 1.0, seed=0)
    c = DS.downsample_reads(df, 1.0, seed=1)
    assert a.index.equals(b.index)
    assert not a.index.equals(c.index)
    assert len(a) == len(c)


def test_random_pick_within_the_second():
    # Not always the first read of the bin (what the old 1 Hz export kept):
    # where in the second the kept read falls is spread across the second.
    out = DS.downsample_reads(_track(7, 10.0, seconds=2000), 1.0)
    offset = out["timestamp"] % 1000
    assert offset.min() < 150 and offset.max() > 850
    assert 350 < offset.mean() < 650


def test_chunking_does_not_change_the_pick():
    df = _reads().sort_values("timestamp")
    whole = DS.downsample_reads(df, 1.0)
    cut = T0 + 123_456          # mid-second, as a preview slice boundary can be
    parts = pd.concat([DS.downsample_reads(df[df.timestamp < cut], 1.0),
                       DS.downsample_reads(df[df.timestamp >= cut], 1.0)])
    # The only possible difference is the one bin the cut splits.
    diff = set(whole.index) ^ set(parts.index)
    assert all(df.loc[i, "timestamp"] // 1000 == cut // 1000 for i in diff)


def test_timestamp_column_fallback():
    df = _reads()
    dt = df.assign(Timestamp=pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                   .dt.tz_convert("US/Mountain")).drop(columns="timestamp")
    assert DS.downsample_reads(dt, 1.0).index.equals(DS.downsample_reads(df, 1.0).index)


def test_off_and_empty():
    df = _reads()
    assert DS.downsample_reads(df, None) is df
    assert DS.downsample_reads(df, 0) is df
    assert len(DS.downsample_reads(df.iloc[:0], 1.0)) == 0


def test_window_pipeline_and_config():
    """The UWB window downsamples before thresholding and records it in the config."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    from fnt.uwb.uwb_preprocessing_pyqt import UWBQuickVisualizationWindow
    app = QApplication.instance() or QApplication([])  # noqa: F841
    w = UWBQuickVisualizationWindow()
    try:
        # A config from before downsampling existed loads as every read.
        w.db_path = os.path.join("C:", os.sep, "nowhere", "T000_fake.sqlite")
        w._pending_config_override = {"timezone": "US/Mountain"}
        w.load_config_if_exists()
        assert w.combo_downsampling.currentText() == DS.DOWNSAMPLING_NONE
        cfg = w.get_config_dict()
        assert cfg["downsampling"] == DS.DOWNSAMPLING_NONE
        assert cfg["downsampling_hz"] is None and cfg["downsampling_seed"] is None

        w._pending_config_override = {"downsampling": DS.DOWNSAMPLING_RANDOM,
                                      "downsampling_hz": 0.75, "downsampling_seed": 3}
        w.load_config_if_exists()
        assert w.combo_preview_downsampling.currentText() == DS.DOWNSAMPLING_RANDOM
        assert w.spin_preview_downsampling_hz.value() == 0.75
        cfg = w.get_config_dict()
        assert (cfg["downsampling"], cfg["downsampling_hz"], cfg["downsampling_seed"]) \
            == (DS.DOWNSAMPLING_RANDOM, 0.75, 3)
        assert cfg["preview"]["downsampling"] == DS.DOWNSAMPLING_RANDOM

        df = _track(7, 10.0, seconds=300)
        df["Timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        w.reset_filter_stats()
        out = w._filter_and_smooth(df, "Savitzky-Golay", velocity=False, jump=False,
                                   outlier=False, adjacency=False)
        assert len(out) <= int(300 * 0.75) + 2
        assert out.index.equals(DS.downsample_reads(df, 0.75, seed=3).index)
        assert w.filter_stats["removed_downsampling"] == len(df) - len(out)
    finally:
        w.close()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
