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

There is deliberately no battery forecast. One was tried - a robust fit of
the recent voltage drain, extrapolated to the cutoff - and on VT-P002 it
predicted 18 days two days before tag 000B died, and 3.4 days six hours
before. These cells hold a near-flat voltage for most of their life and then
fall off a cliff, so no slope fitted to the flat part can see the end coming.
What did track battery death was the number of pings sent: five battery lives
on that trial ended at 5.57-5.76 million recorded pings, at both ~22 Hz and
~7 Hz. So the table reports the raw ping count and the latest voltage, and
leaves the judgement to the reader.

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

def _hexid(shortid):
    return hex(int(shortid)).upper().replace("0X", "").zfill(4)


def tag_rate_stats(conn, table, tags=None, sample_rows=250_000,
                   has_index=False, progress=None):
    """One row per tag: how often it pings, how many times, and its voltage.

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
            rows.append(_one_tag(tag, n, lo, hi, df, first_v))
    else:
        say("Reading the table (no fast index yet)…")
        keep = {t: [] for t in tags}
        total = {t: [0, None, None] for t in tags}
        first_v = {}
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
                keep[tag].append(sub[["timestamp", "battery_voltage"]])
                if sum(len(p) for p in keep[tag]) > sample_rows * 1.5:
                    keep[tag] = [pd.concat(keep[tag]).tail(sample_rows)]
            say(f"Read {sum(v[0] for v in total.values()):,} rows…")
        for tag in tags:
            if not total[tag][0]:
                continue
            df = pd.concat(keep[tag]).tail(sample_rows)
            rows.append(_one_tag(tag, *total[tag], df, first_v.get(tag)))
    out = pd.DataFrame(rows)
    return out.sort_values("shortid").reset_index(drop=True) if len(out) else out


def _one_tag(tag, n, lo, hi, df, first_v):
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
        "v_first": round(v_first, 2) if np.isfinite(v_first) else np.nan,
        "v_last": round(v_last, 2) if np.isfinite(v_last) else np.nan,
        **{GAP_LABELS[i]: round(float(pct[i]), 1) for i in range(len(GAP_LABELS))},
    }
