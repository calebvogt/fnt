"""A ROI bout never spans a recording gap.

``region_bouts`` used to end a run only when the region changed. In the
2024_LID_MesoUWB trials every tag went silent for 14-16 h at once; where an
animal's last fix before the outage and its first after it fell in the same
zone, the export wrote one 15.8 h "visit". A run now also ends at any silence
longer than ``max_gap_s`` (the preprocessing Time gap setting, default
``MAX_GAP_S``), and the window's export path passes that setting through.

Runs under pytest, or directly (``python test_roi_bouts.py``).
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fnt.uwb import roi_bouts as RB  # noqa: E402

S = 1_000_000_000      # ns per second


def _t(*seconds):
    return np.array([int(s * S) for s in seconds], dtype="int64")


def test_an_outage_inside_one_zone_ends_the_bout():
    # four fixes in zone 0 at 0-3 s, a 15.8 h silence, three more in zone 0
    gap = 15.8 * 3600
    t = _t(0, 1, 2, 3, 3 + gap, 4 + gap, 5 + gap)
    rows = RB.region_bouts(np.zeros(7, int), t, max_gap_s=60)
    assert [(r, n) for r, _a, _b, _s, n in rows] == [(0, 4), (0, 3)]
    assert [round(s, 6) for *_x, s, _n in rows] == [3.0, 2.0]     # the outage is in neither
    assert rows[0][2] == t[3] and rows[1][1] == t[4]


def test_a_dropout_shorter_than_the_gap_does_not_split():
    t = _t(0, 1, 2, 47, 48)                      # a 45 s hole, inside a 60 s gap rule
    rows = RB.region_bouts(np.zeros(5, int), t, max_gap_s=60)
    assert len(rows) == 1 and rows[0][4] == 5 and rows[0][3] == 48.0


def test_the_default_gap_applies_and_none_restores_the_old_rule():
    t = _t(0, 1, 1 + RB.MAX_GAP_S + 1, 2 + RB.MAX_GAP_S + 1)
    assert len(RB.region_bouts(np.zeros(4, int), t)) == 2
    assert len(RB.region_bouts(np.zeros(4, int), t, max_gap_s=None)) == 1


def test_regions_outside_and_lone_reads_are_unchanged():
    # zone 0, outside, zone 1, then a lone zone-1 fix after a gap
    zi = np.array([0, 0, -1, 1, 1, 1])
    t = _t(0, 1, 2, 3, 4, 500)
    rows = RB.region_bouts(zi, t, max_gap_s=60)
    assert [(r, n) for r, _a, _b, _s, n in rows] == [(0, 2), (1, 2), (1, 1)]
    lone = rows[-1]
    assert lone[1] == lone[2] and lone[3] == RB.LONE_READ_S


def test_the_export_path_uses_the_time_gap_setting():
    from fnt.uwb.uwb_preprocessing_pyqt import UWBQuickVisualizationWindow as W
    stub = SimpleNamespace(spin_time_gap=SimpleNamespace(value=lambda: 30))
    assert W.roi_bout_max_gap_s(stub) == 30.0
    assert W.roi_bout_max_gap_s(SimpleNamespace()) == RB.MAX_GAP_S
    # region_bout_rows: one zone, a 2 h hole in the middle -> two exported visits
    ts = pd.to_datetime([0, 1, 2, 7202, 7203], unit="s").tz_localize("UTC").tz_convert("US/Eastern")
    stub.animal_label = lambda tag: "M1"
    stub.roi_bout_max_gap_s = lambda: W.roi_bout_max_gap_s(stub)
    rows = W.region_bout_rows(stub, pd.DataFrame({"Timestamp": ts}), 1,
                              [("ROI", [{"name": "Zone1"}])],
                              idx={"ROI": np.zeros(5, int)})
    assert [(r["ROI"], r["n_reads"], r["duration_s"]) for r in rows] == [("Zone1", 3, 2.0), ("Zone1", 2, 1.0)]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
