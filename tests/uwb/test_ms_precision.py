"""Millisecond precision in the UWB exports' time arithmetic.

Wiser timestamps are integer epoch milliseconds, and every exported bout is
meant to carry them unrounded: ``duration_s == bout_stop - bout_start`` in
every file. Three places quietly quantised to whole seconds and are guarded
here:

* the social-overlap gap test compared 1 s pairing-bin LABELS, so whether two
  contacts were bridged depended on where the fixes fell inside their bins;
* behaviour-event durations were the larger of the observed span and the
  classifier's frame count, which the frame count always won - every duration
  came out a whole number of seconds - and the edges were any tag's fix in
  the edge second, not the actor's or target's;
* the Data View's CSV time parser inferred one format from the first row, so
  a file whose first time sat on a whole second lost every fractional row.

Runs under pytest, or directly (``python test_ms_precision.py``).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fnt.uwb import dataview as DV  # noqa: E402
from fnt.uwb.proximity_detection import detect_proximity_bouts  # noqa: E402

TZ = 'US/Mountain'
T0 = 1730899200000          # 2024-11-06 06:20:00 -07:00, epoch ms


def _fixes(rows):
    """[(animal_tag, ms_offset, x, y)] -> tz-aware smoothed-style frame."""
    df = pd.DataFrame(rows, columns=['shortid', 'ms', 'smoothed_x', 'smoothed_y'])
    df['Timestamp'] = pd.to_datetime(T0 + df['ms'], unit='ms',
                                     utc=True).dt.tz_convert(TZ)
    return df.drop(columns='ms')


def test_epoch_seconds_mixed_precision_and_offsets():
    s = pd.Series(["2024-11-06 06:20:00-07:00",          # whole second first
                   "2024-11-06 06:20:01.123000-07:00",
                   "2024-11-03 01:59:59.500-06:00",       # DST: two offsets
                   "2024-11-03 01:00:00.250-07:00",
                   "not a time"])
    got = DV.epoch_seconds(s)
    assert got[0] == T0 / 1000
    assert abs(got[1] - (T0 + 1123) / 1000) < 1e-6
    # 01:59:59.5 MDT and 01:00:00.25 MST are 0.75 s apart in real time.
    assert abs((got[3] - got[2]) - 0.75) < 1e-6
    assert np.isnan(got[4])


def test_epoch_seconds_naive_strings_use_given_tz():
    got = DV.epoch_seconds(pd.Series(["2024-11-06 06:20:00",
                                      "2024-11-06 06:20:00.250"]), tz=TZ)
    assert got[0] == T0 / 1000 and abs(got[1] - got[0] - 0.25) < 1e-6


def test_social_gap_is_measured_between_real_fixes():
    # Both animals together at 0.0 s and again at 5.950 s. The bins are
    # 0 and 5 - five apart, so a bin-label test bridges them - but the fixes
    # are 5.95 s apart (5.05 s from bin 0's LAST fix at 0.9 s): two bouts.
    rows = []
    for ms in (0, 900, 5950):
        rows += [(1, ms, 0.0, 0.0), (2, ms, 0.1, 0.0)]
    _ev, bouts = detect_proximity_bouts(_fixes(rows), threshold=0.5, gap_s=5)
    assert len(bouts) == 2, bouts
    assert list(bouts['duration_s']) == [0.9, 0.0]
    assert bouts['bout_start'].iloc[1].microsecond == 950000

    # 4.9 s apart in real time: one bout, spanning 0.0 -> 5.8 s exactly.
    rows = []
    for ms in (0, 900, 5800):
        rows += [(1, ms, 0.0, 0.0), (2, ms, 0.1, 0.0)]
    _ev, bouts = detect_proximity_bouts(_fixes(rows), threshold=0.5, gap_s=5)
    assert len(bouts) == 1
    assert abs(bouts['duration_s'].iloc[0] - 5.8) < 1e-9
    span = (bouts['bout_stop'] - bouts['bout_start']).dt.total_seconds()
    assert (span == bouts['duration_s']).all()


def _events_frame(data, tags, events, n_frames):
    """Run behavior_events_frame over a canned classification."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv)
    from fnt.uwb.uwb_preprocessing_pyqt import PlotSaverWorker

    t0 = int(data['Timestamp'].dt.as_unit('ns').astype('int64').min())
    grid = t0 + np.arange(n_frames, dtype='int64') * 1_000_000_000
    w = PlotSaverWorker(None, None, tags, False, "None", plot_types={})
    w._behavior_classification = lambda d, t, **k: (grid, None, None,
                                                    events, None)
    return w.behavior_events_frame(data, tags, tz=TZ)


def test_behavior_events_edges_come_from_the_dyad():
    # Tag 3 is a bystander reporting at the very start of every second; the
    # chaser (1) and target (2) report later inside it.
    rows = [(3, s * 1000 + 1, 5.0, 5.0) for s in range(8)]
    rows += [(1, 1_250, 0, 0), (2, 1_600, 1, 0), (1, 3_400, 0, 0),
             (2, 4_730, 1, 0)]
    data = _fixes(rows)
    ev = _events_frame(data, [1, 2, 3], [
        {'behavior': 'chase', 'actor': 0, 'target': 1,
         'start_frame': 1, 'stop_frame': 4},
        # A one-frame bout backed by a single fix: 0 s, not one second.
        {'behavior': 'chase', 'actor': 1, 'target': 0,
         'start_frame': 3, 'stop_frame': 3},
        # No actor/target fix inside: falls back to the 1 Hz slot edges.
        {'behavior': 'chase', 'actor': 0, 'target': 1,
         'start_frame': 6, 'stop_frame': 7},
    ], n_frames=8)
    assert list(ev.columns[-3:]) == ['duration_s', 'n_frames', 'n_fixes']
    first = ev.iloc[0]
    assert first['bout_start'] == pd.Timestamp(T0 + 1_250, unit='ms', tz='UTC')
    assert first['bout_stop'] == pd.Timestamp(T0 + 4_730, unit='ms', tz='UTC')
    assert abs(first['duration_s'] - 3.48) < 1e-9          # not 4 s
    assert (first['n_frames'], first['n_fixes']) == (4, 4)
    lone = ev.iloc[1]
    assert (lone['duration_s'], lone['n_frames'], lone['n_fixes']) == (0.0, 1, 1)
    held = ev.iloc[2]
    assert held['n_fixes'] == 0 and held['duration_s'] == 1.0
    span = (ev['bout_stop'] - ev['bout_start']).dt.total_seconds()
    assert ((span - ev['duration_s']).abs() < 1e-9).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
