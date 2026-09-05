"""Bout detection: the default must not produce bouts that contain their own gaps.

The R pipeline this replaces pairs START rows to STOP rows by position after
dropping the reads in between, which mis-pairs 2.34% of bouts on the 2021_LID
data - they end up spanning a gap at or beyond the threshold, or a zone change.
``segment`` cannot do that by construction, and this pins the difference so a
future change to the default is a deliberate one.
"""
import numpy as np
import pandas as pd
import pytest

from fnt.rfid.core.bout_detector import detect_bouts

THRESH = 50.0


def make_reads(offsets_and_zones, name="A", day=1):
    base = pd.Timestamp("2021-05-07 18:00:00")
    return pd.DataFrame({
        "trial": "T001", "name": name, "sex": "M", "noon_day": day,
        "zone": [z for _o, z in offsets_and_zones],
        "field_time": [base + pd.Timedelta(seconds=o)
                       for o, _z in offsets_and_zones]})


def test_a_run_of_close_reads_in_one_zone_is_one_bout():
    reads = make_reads([(0, 1), (5, 1), (10, 1), (15, 1)])
    bouts = detect_bouts(reads, THRESH, algorithm="segment")
    assert len(bouts) == 1
    assert bouts["duration_s"].iloc[0] == 15
    assert bouts["n_reads"].iloc[0] == 4


def test_a_gap_at_the_threshold_splits_the_bout():
    """The threshold is inclusive: exactly 50 s apart is already two bouts."""
    reads = make_reads([(0, 1), (10, 1), (60, 1), (70, 1)])
    bouts = detect_bouts(reads, THRESH, algorithm="segment")
    assert len(bouts) == 2
    assert list(bouts["duration_s"]) == [10, 10]


def test_a_zone_change_splits_the_bout_even_with_no_gap():
    reads = make_reads([(0, 1), (1, 1), (2, 3), (3, 3)])
    bouts = detect_bouts(reads, THRESH, algorithm="segment")
    assert len(bouts) == 2
    assert list(bouts["zone"]) == [1, 3]


def test_an_isolated_read_becomes_a_minimum_duration_bout():
    reads = make_reads([(0, 1), (500, 4), (1000, 1)])
    bouts = detect_bouts(reads, THRESH, min_duration_s=1.0, algorithm="segment")
    assert len(bouts) == 3
    assert set(bouts["bout_status"]) == {"SINGLE_READ"}
    assert set(bouts["duration_s"]) == {1.0}


def test_segment_bouts_never_contain_a_gap_or_a_zone_change():
    """The property that motivates making `segment` the default."""
    rng = np.random.default_rng(0)
    offsets = np.cumsum(rng.integers(1, 120, size=400))
    zones = rng.integers(1, 9, size=400)
    reads = make_reads(list(zip(offsets.tolist(), zones.tolist())))
    bouts = detect_bouts(reads, THRESH, algorithm="segment")

    t = (reads["field_time"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    z = reads["zone"].to_numpy()
    for start, stop in zip(bouts["field_time"], bouts["field_time_stop"]):
        s = (start - pd.Timestamp("1970-01-01")).total_seconds()
        e = (pd.Timestamp(stop) - pd.Timestamp("1970-01-01")).total_seconds()
        inside = (t >= s) & (t <= e)
        assert (z[inside] == z[inside][0]).all(), "bout spans a zone change"
        if inside.sum() > 1:
            assert (np.diff(t[inside]) < THRESH).all(), "bout spans a gap"


def test_r_compat_still_runs_and_can_produce_incoherent_bouts():
    """Why `segment` is the default, stated as a property rather than a count.

    On clean data the two agree. On realistic data R's positional START/STOP
    pairing produces bouts that contain a gap at or beyond the threshold, or a
    zone change - the animal recorded as continuously present in a zone its own
    reads say it left. `segment` cannot do that; this asserts both halves.
    """
    rng = np.random.default_rng(7)
    offsets = np.cumsum(rng.integers(1, 200, size=600))
    zones = rng.integers(1, 5, size=600)
    reads = make_reads(list(zip(offsets.tolist(), zones.tolist())))

    t = (reads["field_time"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    z = reads["zone"].to_numpy()

    def incoherent(bouts):
        bad = 0
        for start, stop in zip(bouts["field_time"], bouts["field_time_stop"]):
            s = (start - pd.Timestamp("1970-01-01")).total_seconds()
            e = (pd.Timestamp(stop) - pd.Timestamp("1970-01-01")).total_seconds()
            inside = (t >= s) & (t <= e)
            if inside.sum() < 2:
                continue
            if (np.diff(t[inside]) >= THRESH).any() or (z[inside] != z[inside][0]).any():
                bad += 1
        return bad

    assert incoherent(detect_bouts(reads, THRESH, algorithm="r_compat")) > 0
    assert incoherent(detect_bouts(reads, THRESH, algorithm="segment")) == 0


def test_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError, match="Unknown bout algorithm"):
        detect_bouts(make_reads([(0, 1)]), THRESH, algorithm="nonsense")


def test_microsecond_datetimes_do_not_rescale_gaps():
    """A parquet round-trip can hand back us-resolution datetimes.

    Converting those with astype("int64")/1e9 scales every gap by 1/1000, so
    nothing ever exceeds the threshold and a whole day collapses into one bout.
    """
    reads = make_reads([(0, 1), (10, 1), (200, 1), (210, 1)])
    reads["field_time"] = reads["field_time"].astype("datetime64[us]")
    bouts = detect_bouts(reads, THRESH, algorithm="segment")
    assert len(bouts) == 2
