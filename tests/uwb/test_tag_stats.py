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


def test_rates_and_battery_are_measured():
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


def test_a_flat_battery_is_called_out():
    # Tags that reach 2.36 V stop reporting; VT-P002's 0006/0007 and T012's
    # 001D all ended there.
    ts, v = _pings(22, None, minutes=90, volts=(3.14, 2.36))
    st = _stats(_db({0x06: (ts, v)}), has_index=True).iloc[0]
    assert st["days_left"] == 0.0 and "stop reporting" in st["note"]


def test_a_short_recording_still_reports_its_numbers():
    # Four minutes is too little to conclude anything, but the row should say
    # so through pings and hours rather than by being withheld.
    ts, v = _pings(10, 1, minutes=4, volts=(2.56, 2.36))
    st = _stats(_db({0x1D: (ts, v)}), has_index=True).iloc[0]
    assert st["hours"] <= 0.1 and st["pings"] > 0 and st["peak_hz"] > 0


def test_battery_drain_needs_more_than_a_day():
    # A cell settles steeply for hours after it goes in; a slope fitted to
    # that reads as a tag about to die.
    short = _stats(_db({0x11: _pings(10, 1, minutes=120, volts=(3.20, 2.95))}),
                   has_index=True).iloc[0]
    assert np.isnan(short["v_per_day"]) and np.isnan(short["days_left"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)


def _hourly(volts_by_hour, start="2026-09-15 12:00"):
    """(hour timestamps in ns, volts) for battery_estimate."""
    t0 = pd.Timestamp(start, tz="US/Mountain").value
    hrs = np.arange(len(volts_by_hour), dtype="float64")
    return t0 + hrs * 3_600e9, np.asarray(volts_by_hour, dtype="float64")


def test_the_settle_is_excluded_not_averaged_in():
    # A fresh cell: 3.22 -> 2.95 V in four hours (tag 0016 did exactly this),
    # then a steady 0.02 V/day for five days. End-to-end that reads as ~0.07
    # V/day; only the plateau is the tag's real drain.
    settle = list(np.linspace(3.22, 2.95, 5))
    plateau = list(2.95 - 0.02 * np.arange(1, 5 * 24) / 24.0)
    est = TS.battery_estimate(*_hourly(settle + plateau))
    assert abs(est["v_per_day"] - 0.02) < 0.004, est["v_per_day"]
    naive = (3.22 - plateau[-1]) / ((len(settle) + len(plateau)) / 24.0)
    assert naive > 2 * est["v_per_day"]        # what the old two-point slope gave
    # (2.93 - 2.36) / 0.02 is about 28 days, and the range brackets it.
    assert 20 < est["days_left"] < 40
    assert est["days_left_lo"] <= est["days_left"] <= est["days_left_hi"]


def test_below_the_plateau_the_estimate_is_flagged():
    volts = list(np.linspace(2.62, 2.48, 4 * 24))      # past the knee
    est = TS.battery_estimate(*_hourly(volts))
    assert est["days_left"] > 0 and "expect sooner" in est["note"]


def test_a_dead_tag_reads_zero():
    est = TS.battery_estimate(*_hourly(list(np.linspace(2.50, 2.36, 3 * 24))),
                              v_now=2.36)
    assert est["days_left"] == 0.0 and "stop reporting" in est["note"]


def test_too_little_data_gives_no_number():
    est = TS.battery_estimate(*_hourly(list(np.linspace(3.20, 3.05, 20))))
    assert np.isnan(est["days_left"]) and np.isnan(est["v_per_day"])
    assert "needs about" in est["note"]


def test_a_flat_battery_reports_no_measurable_drain():
    est = TS.battery_estimate(*_hourly([3.00] * (4 * 24)))
    assert est["v_per_day"] < TS.MIN_DRAIN_V_DAY
    assert np.isnan(est["days_left"]) and est["note"] == "no measurable drain yet"


def test_one_noisy_hour_does_not_move_the_answer():
    volts = list(2.95 - 0.03 * np.arange(4 * 24) / 24.0)
    clean = TS.battery_estimate(*_hourly(volts))
    volts[-8] = 2.40                                   # a single bad reading
    noisy = TS.battery_estimate(*_hourly(volts))
    assert abs(clean["days_left"] - noisy["days_left"]) < 1.0
