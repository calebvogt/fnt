"""The same times stored in any datetime64 unit give the same answers.

pandas 3 parses times to microseconds where pandas 2 used nanoseconds, and
code that read the raw integers off a DatetimeIndex (``asi8``) and divided by
1e9 came out 1000x off: gaps, bout durations and co-occupancy seconds. These
run the unit-sensitive steps on one set of times stored at s, ms, us and ns,
so a regression shows up under either pandas version.
"""
import pandas as pd
import pytest

from fnt.rfid.core.bout_detector import detect_bouts
from fnt.rfid.core.cleaning import elapsed_seconds
from fnt.rfid.core.summaries import overlapping_bouts

UNITS = ["s", "ms", "us", "ns"]
T0 = pd.Timestamp("2021-05-08 18:00:00")


def times(seconds, unit):
    return pd.Series(T0 + pd.to_timedelta(seconds, unit="s")).astype(f"datetime64[{unit}]")


@pytest.mark.parametrize("tz", ["", "America/New_York"])
@pytest.mark.parametrize("unit", UNITS)
def test_elapsed_seconds(unit, tz):
    secs, ambiguous = elapsed_seconds(times([0, 30, 95, 3600], unit), tz)
    assert secs.tolist() == [0.0, 30.0, 95.0, 3600.0] and ambiguous == 0


@pytest.mark.parametrize("unit", UNITS)
def test_bout_durations(unit):
    """0, 20, 40 s are one 40 s bout; the read at 200 s is a lone 1 s bout."""
    reads = pd.DataFrame({"animal": "Mars", "t": times([0, 20, 40, 200], unit),
                          "zone_id": "A3", "paddock": "A", "zone": "3",
                          "away": False, "day": 2})
    b = detect_bouts(reads, 50.0)
    assert b["duration_s"].tolist() == [40.0, 1.0]
    assert b["n_reads"].tolist() == [3, 1]


@pytest.mark.parametrize("unit", UNITS)
def test_overlap_seconds(unit):
    """X 0-600 s and Y 300-900 s in one zone overlap for 300 s."""
    bouts = pd.DataFrame({"animal": ["X", "Y"], "zone_id": "A1", "day": 2,
                          "start": times([0, 300], unit), "stop": times([600, 900], unit)})
    assert overlapping_bouts(bouts)["seconds"].tolist() == [300.0]
