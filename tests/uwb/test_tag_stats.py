"""Measuring each tag's ping rate from the recording (fnt/uwb/tag_stats.py).

A tag's active/rest rate pair lives on its label and nowhere in the data, so a
mis-flashed tag looks like a well-behaved animal that is simply sampled less.
In VT-P002 two tags ran at 22 Hz with no rest rate and died in six days; in
T012 two 1 Hz tags were deployed where fast ones were meant. No profile is
named here - rate pairs are the user's to choose - so these tests build
recordings at known rates and check the reported numbers describe them.

Runs under pytest, or directly (``python test_tag_stats.py``).
"""
import os
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fnt.uwb import tag_stats as TS  # noqa: E402

T0 = int(pd.Timestamp("2026-09-21 12:00", tz="US/Mountain").value // 10**6)


def _pings(active_hz, rest_hz, minutes=90, active_share=0.6, volts=(3.10, 2.95),
           rng=None):
    """Timestamps for a tag that alternates active and rest in 5-min blocks."""
    rng = rng or np.random.default_rng(0)
    t, out = T0, []
    block = 0
    while t < T0 + minutes * 60_000:
        hz = active_hz if (rest_hz is None or (block % 5) / 5.0 < active_share) else rest_hz
        step = 1000.0 / hz
        end = min(t + 5 * 60_000, T0 + minutes * 60_000)
        n = int((end - t) / step)
        out.extend((t + np.arange(n) * step + rng.normal(0, step * 0.05, n)).astype("int64"))
        t = end
        block += 1
    out = np.sort(np.array(out))
    v = np.linspace(volts[0], volts[1], len(out))
    return out, v


def _db(tags):
    """tags: {shortid: (timestamps, volts)} -> path to a throwaway database."""
    path = os.path.join(tempfile.mkdtemp(), "rates.sqlite")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE data (shortid INT, timestamp INT, battery_voltage REAL)")
    for tag, (ts, v) in tags.items():
        con.executemany("INSERT INTO data VALUES (?,?,?)",
                        [(tag, int(a), float(b)) for a, b in zip(ts, v)])
    con.execute("CREATE INDEX idx ON data (shortid, timestamp)")
    con.commit()
    con.close()
    return path


def _stats(path, **kw):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return TS.tag_rate_stats(con, "data", **kw)
    finally:
        con.close()


def test_the_numbers_describe_the_rates_that_were_recorded():
    path = _db({
        0x11: _pings(10, 1),            # switches 10 <-> 1
        0x2A: _pings(4, 1),             # switches 4 <-> 1
        0x0C: _pings(1, None),          # one rate, always
        0x06: _pings(22, None),         # one rate, very fast
    })
    st = _stats(path, has_index=True).set_index("HexID")
    # Peak reflects the active rate, quiet the rest rate.
    assert 8 <= st.loc["0011", "peak_hz"] <= 12 and st.loc["0011", "slow_hz"] < 2
    assert 3 <= st.loc["002A", "peak_hz"] <= 5 and st.loc["002A", "slow_hz"] < 2
    assert 18 <= st.loc["0006", "peak_hz"] <= 24
    # A tag holding one rate has a peak close to its quiet figure; a switching
    # tag does not. That contrast is what the table is read for.
    assert st.loc["000C", "peak_hz"] / max(st.loc["000C", "slow_hz"], .01) < 2
    assert st.loc["0011", "peak_hz"] / max(st.loc["0011", "slow_hz"], .01) > 4
    # Gap columns show the same thing per ping.
    assert st.loc["000C", "0.7-1.4s"] > 60
    assert st.loc["0011", "0.05-0.08s"] + st.loc["0011", "0.08-0.13s"] > 30


def test_rates_and_voltage_are_measured():
    path = _db({0x13: _pings(10, 1, minutes=120, volts=(3.20, 3.00))})
    st = _stats(path, has_index=True).iloc[0]
    assert st["HexID"] == "0013" and st["shortid"] == 0x13
    assert 4 < st["median_hz"] < 11 and 8 <= st["peak_hz"] <= 12
    assert st["v_first"] == 3.20 and st["v_last"] == 3.00
    assert st["hours"] == 2.0
    assert st["pings"] > 10_000
    assert 60 <= st["modal_gap_ms"] <= 160        # the fast mode
    assert 60 <= st["median_gap_ms"] <= 1100


def test_the_scan_path_matches_the_indexed_path():
    path = _db({0x11: _pings(10, 1), 0x0C: _pings(1, None)})
    fast = _stats(path, has_index=True).set_index("HexID")
    slow = _stats(path, has_index=False).set_index("HexID")
    for hexid in ("0011", "000C"):
        assert fast.loc[hexid, "pings"] == slow.loc[hexid, "pings"]
        assert abs(fast.loc[hexid, "median_hz"] - slow.loc[hexid, "median_hz"]) < 0.2
        assert fast.loc[hexid, "modal_gap_ms"] == slow.loc[hexid, "modal_gap_ms"]


def test_no_battery_forecast_is_offered():
    """The fitted days-left estimate is gone, and must not creep back.

    On VT-P002 it called 18 days two days before tag 000B died and 3.4 days
    six hours before: the cells hold ~2.9 V for most of their life and then
    fall off a cliff that no slope fitted to the plateau can see. The table
    reports the voltage and the ping count as measured instead.
    """
    ts, v = _pings(22, None, minutes=90, volts=(3.14, 2.36))
    st = _stats(_db({0x06: (ts, v)}), has_index=True).iloc[0]
    assert st["v_last"] == 2.36 and st["pings"] > 0
    for gone in ("days_left", "days_left_lo", "days_left_hi", "v_per_day", "note"):
        assert gone not in st.index, gone
    assert not hasattr(TS, "battery_estimate")


def test_a_short_recording_still_reports_its_numbers():
    # Four minutes is too little to conclude anything, but the row should say
    # so through pings and hours rather than by being withheld.
    ts, v = _pings(10, 1, minutes=4, volts=(2.56, 2.36))
    st = _stats(_db({0x1D: (ts, v)}), has_index=True).iloc[0]
    assert st["hours"] <= 0.1 and st["pings"] > 0 and st["peak_hz"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
