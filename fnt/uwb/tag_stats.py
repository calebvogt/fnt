"""How often is each tag actually pinging?

A UWB tag is flashed with an active/rest rate pair - 10 Hz moving and 1 Hz
after 30 s still, or 4/1, or 1/1, or whatever it was configured with - and
the label on the tag is the only place that is written down. When a tag is
flashed wrong, or the wrong one goes on an animal, nothing in the recording
says so: the data look fine, one animal is simply sampled several times more
often than another, and the first sign is a battery dead in six days or a
contact rate that is really a sampling difference.

This reports the rates as measured, and names no profiles: rate pairs are the
user's to choose, so the numbers are given plainly - the busiest minute, the
quietest, the typical one, and how the gaps between pings are distributed -
and the reader decides what the tag is.

Nothing here imports Qt, so it is testable without a GUI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Gap boundaries (seconds) around the rate modes tags are flashed with.
#: 20 Hz | 10 Hz | 6-7 Hz | 4 Hz | 2 Hz | 1 Hz | slower
GAP_EDGES = np.array([0.0, 0.05, 0.08, 0.13, 0.18, 0.45, 0.70, 1.40, 3.0, 1e9])
GAP_LABELS = ("<0.05s", "0.05-0.08s", "0.08-0.13s", "0.13-0.18s", "0.18-0.45s",
              "0.45-0.7s", "0.7-1.4s", "1.4-3s", ">3s")

#: Voltage the tags in this system have stopped reporting at.
DEAD_V = 2.36
#: Hours after a tag's first ping that are ignored when fitting the drain. A
#: fresh cell settles steeply for several hours - tag 0016 fell 3.22 -> 2.95 V
#: in four - and a slope fitted through that reads as a tag about to die.
SETTLE_H = 12.0
#: How much of the recent past the drain is fitted over.
FIT_WINDOW_H = 72.0
#: Below this much post-settle data, no estimate is offered at all.
MIN_FIT_H = 24.0
#: These cells hold a plateau and then fall off a knee: VT-P002's 0006 and
#: 0007 dropped ~0.05 V per HOUR in their last eight, against a 0.14 V/day
#: trial average. Below this voltage a plateau slope flatters the tag.
KNEE_V = 2.55
#: A drain slower than this is not distinguishable from measurement noise.
MIN_DRAIN_V_DAY = 0.002


def battery_estimate(hours_ns, volts, dead_v=DEAD_V, v_now=None, first_ns=None):
    """Drain and time left from an hourly voltage series. Returns a dict.

    Fitted rather than taken end-to-end: the first ``SETTLE_H`` are dropped so
    the post-insertion settle is excluded instead of merely diluted, and a
    Theil-Sen line over the most recent ``FIT_WINDOW_H`` gives a slope that one
    noisy hour cannot move, plus a range from its confidence band.

    The estimate assumes the plateau continues, which it does not forever -
    hence the note once a tag is below ``KNEE_V``, where these cells start
    falling much faster than any plateau fit predicts.

    ``v_now`` is the tag's latest single reading. Hourly means lag it, and a
    tag that has just reached the cutoff should say so rather than be averaged
    back above it. ``first_ns`` is the tag's first ping, so the settle window
    is measured from when the cell went in even when only the recent hours
    were read back.
    """
    from scipy import stats as _stats

    out = {"v_per_day": np.nan, "days_left": np.nan,
           "days_left_lo": np.nan, "days_left_hi": np.nan, "note": ""}
    t = np.asarray(hours_ns, dtype="float64")
    v = np.asarray(volts, dtype="float64")
    ok = np.isfinite(t) & np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(t) < 2:
        return out
    order = np.argsort(t)
    t, v = t[order], v[order]
    v_last = float(v_now) if v_now is not None and np.isfinite(v_now) \
        else float(np.median(v[-3:]))
    origin = float(first_ns) if first_ns is not None else t[0]
    days = (t - origin) / 86_400e9
    settled = days >= SETTLE_H / 24.0
    t_s, v_s = days[settled], v[settled]
    if len(t_s) >= 2:
        recent = t_s >= t_s[-1] - FIT_WINDOW_H / 24.0
        t_s, v_s = t_s[recent], v_s[recent]
    span_h = (t_s[-1] - t_s[0]) * 24 if len(t_s) >= 2 else 0.0

    if v_last <= dead_v + 0.02:
        out.update(days_left=0.0,
                   note=f"at {dead_v:.2f} V, where these tags stop reporting")
        return out
    if span_h < MIN_FIT_H or len(t_s) < 6:
        out["note"] = (f"needs about {SETTLE_H + MIN_FIT_H:.0f} h of recording "
                       f"to estimate (the first {SETTLE_H:.0f} h are the cell "
                       f"settling)")
        return out

    fit = _stats.theilslopes(v_s, t_s, alpha=0.90)
    drain = -float(fit.slope)
    lo_drain, hi_drain = -float(fit.high_slope), -float(fit.low_slope)
    out["v_per_day"] = drain
    if drain < MIN_DRAIN_V_DAY:
        out["note"] = "no measurable drain yet"
        return out
    headroom = v_last - dead_v
    out["days_left"] = headroom / drain
    if hi_drain >= MIN_DRAIN_V_DAY:
        out["days_left_lo"] = headroom / hi_drain
    if lo_drain >= MIN_DRAIN_V_DAY:
        out["days_left_hi"] = headroom / lo_drain
    if v_last < KNEE_V:
        out["note"] = ("below the plateau - these cells fall much faster from "
                       "here, so expect sooner than this")
    return out


def _hexid(shortid):
    return hex(int(shortid)).upper().replace("0X", "").zfill(4)


def tag_rate_stats(conn, table, tags=None, sample_rows=250_000,
                   has_index=False, progress=None):
    """One row per tag: how often it pings, and what its battery is doing.

    ``has_index`` says the connection is the indexed copy, where each tag can
    be read on its own; otherwise the table is scanned once and split up here.
    ``sample_rows`` caps how many of a tag's most recent pings the interval
    statistics are measured on - the profile is a property of the tag, so the
    recent past describes it as well as the whole trial and costs far less.
    """
    say = progress or (lambda _m: None)
    if tags is None:
        tags = [r[0] for r in conn.execute(
            f"SELECT DISTINCT shortid FROM {table} ORDER BY shortid")]
    rows = []
    if has_index:
        for i, tag in enumerate(tags, 1):
            say(f"Measuring tag {_hexid(tag)} ({i} of {len(tags)})…")
            n, lo, hi = conn.execute(
                f"SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM {table} "
                f"WHERE shortid = ?", (tag,)).fetchone()
            if not n:
                continue
            df = pd.read_sql_query(
                f"SELECT timestamp, battery_voltage FROM {table} "
                f"WHERE shortid = ? ORDER BY timestamp DESC LIMIT {int(sample_rows)}",
                conn, params=(tag,))
            first_v = conn.execute(
                f"SELECT battery_voltage FROM {table} WHERE shortid = ? "
                f"ORDER BY timestamp LIMIT 1", (tag,)).fetchone()
            # One mean voltage per hour, but only over the window the drain
            # is actually fitted on. Reading every hour of a week-long trial
            # cost four times as much for numbers the fit then discards.
            since = max(int(lo), int(hi) - int((FIT_WINDOW_H + 2) * 3_600_000))
            volt_hours = pd.read_sql_query(
                f"SELECT timestamp / 3600000 AS hr, AVG(battery_voltage) AS v "
                f"FROM {table} WHERE shortid = ? AND timestamp >= ? "
                f"GROUP BY hr ORDER BY hr", conn, params=(tag, since))
            rows.append(_one_tag(tag, n, lo, hi, df, first_v, volt_hours))
    else:
        say("Reading the table (no fast index yet)…")
        keep = {t: [] for t in tags}
        total = {t: [0, None, None] for t in tags}
        first_v = {}
        hourly = {}        # (tag, hour) -> [volt sum, count]
        q = f"SELECT shortid, timestamp, battery_voltage FROM {table}"
        for chunk in pd.read_sql_query(q, conn, chunksize=2_000_000):
            for tag, sub in chunk.groupby("shortid"):
                if tag not in keep:
                    continue
                e = total[tag]
                e[0] += len(sub)
                lo, hi = sub["timestamp"].min(), sub["timestamp"].max()
                e[1] = lo if e[1] is None else min(e[1], lo)
                e[2] = hi if e[2] is None else max(e[2], hi)
                first_v.setdefault(tag, (sub["battery_voltage"].iloc[0],))
                agg = sub.groupby(sub["timestamp"] // 3_600_000)[
                    "battery_voltage"].agg(["sum", "count"])
                for hr, r in agg.iterrows():
                    e = hourly.setdefault((tag, int(hr)), [0.0, 0])
                    e[0] += float(r["sum"] or 0.0)
                    e[1] += int(r["count"])
                keep[tag].append(sub[["timestamp", "battery_voltage"]])
                if sum(len(p) for p in keep[tag]) > sample_rows * 1.5:
                    keep[tag] = [pd.concat(keep[tag]).tail(sample_rows)]
            say(f"Read {sum(v[0] for v in total.values()):,} rows…")
        for tag in tags:
            if not total[tag][0]:
                continue
            df = pd.concat(keep[tag]).tail(sample_rows)
            hrs = sorted(h for (tg, h) in hourly if tg == tag)
            volt_hours = pd.DataFrame({
                "hr": hrs,
                "v": [hourly[(tag, h)][0] / max(hourly[(tag, h)][1], 1) for h in hrs]})
            rows.append(_one_tag(tag, *total[tag], df, first_v.get(tag), volt_hours))
    out = pd.DataFrame(rows)
    return out.sort_values("shortid").reset_index(drop=True) if len(out) else out


def _one_tag(tag, n, lo, hi, df, first_v, volt_hours=None):
    ts = np.sort(df["timestamp"].to_numpy(dtype="int64"))
    uniq = np.unique(ts)
    gaps = np.diff(uniq) / 1000.0
    gaps = gaps[gaps > 0]
    counts, _ = np.histogram(gaps, bins=GAP_EDGES)
    pct = 100.0 * counts / max(counts.sum(), 1)
    # Rates are measured per minute: a median over minutes describes the tag's
    # usual behaviour, and the busiest minute shows what it is capable of -
    # which is what separates a 1 Hz tag from a 10/1 tag whose animal slept.
    per_min = pd.Series(1, index=pd.to_datetime(ts, unit="ms")).resample("1min").size()
    per_min = per_min[per_min > 0] / 60.0
    span_s = max((hi - lo) / 1000.0, 1.0)
    volts = pd.to_numeric(df.get("battery_voltage"), errors="coerce").dropna()
    v_last = float(volts.iloc[0]) if len(volts) else np.nan   # DESC order
    v_first = float(first_v[0]) if first_v and pd.notna(first_v[0]) else np.nan
    median_hz = float(per_min.median()) if len(per_min) else 0.0
    batt = {"v_per_day": np.nan, "days_left": np.nan, "days_left_lo": np.nan,
            "days_left_hi": np.nan, "note": ""}
    if volt_hours is not None and len(volt_hours):
        batt = battery_estimate(volt_hours["hr"].to_numpy(dtype="float64")
                                * 3_600_000 * 1_000_000,
                                volt_hours["v"].to_numpy(dtype="float64"),
                                v_now=v_last, first_ns=int(lo) * 1_000_000)
    note = batt["note"]
    return {
        "shortid": int(tag), "HexID": _hexid(tag), "pings": int(n),
        "first_ns": int(lo) * 1_000_000, "last_ns": int(hi) * 1_000_000,
        "hours": round(span_s / 3600.0, 1),
        "median_hz": round(median_hz, 2),
        "peak_hz": round(float(per_min.max()), 1) if len(per_min) else 0.0,
        # The quietest minute is where a rest rate shows itself, whatever
        # rate pair the tag was configured with.
        "slow_hz": round(float(np.percentile(per_min, 5)), 2) if len(per_min) else 0.0,
        "median_gap_ms": int(round(1000 * float(np.median(gaps)))) if len(gaps) else 0,
        "modal_gap_ms": int(round(1000 * (GAP_EDGES[int(np.argmax(counts))]
                                          + GAP_EDGES[int(np.argmax(counts)) + 1]) / 2))
        if counts.sum() else 0,
        "note": note,
        "v_first": round(v_first, 2) if np.isfinite(v_first) else np.nan,
        "v_last": round(v_last, 2) if np.isfinite(v_last) else np.nan,
        "v_per_day": (round(batt["v_per_day"], 3)
                      if np.isfinite(batt["v_per_day"]) else np.nan),
        "days_left": (round(batt["days_left"], 1)
                      if np.isfinite(batt["days_left"]) else np.nan),
        "days_left_lo": (round(batt["days_left_lo"], 1)
                         if np.isfinite(batt["days_left_lo"]) else np.nan),
        "days_left_hi": (round(batt["days_left_hi"], 1)
                         if np.isfinite(batt["days_left_hi"]) else np.nan),
        **{GAP_LABELS[i]: round(float(pct[i]), 1) for i in range(len(GAP_LABELS))},
    }
