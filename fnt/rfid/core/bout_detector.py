"""Stage 2: reads -> movement bouts.

A bout is a continuous stay in one zone. Reads inside a bout arrive about once
a second (a passive tag on an antenna is re-read constantly), so a bout ends
when the animal leaves - seen as either a gap of at least the threshold or a
read from a different zone.

Two algorithms, because the honest default and the reproducible one differ:

``segment`` (default)
    Start a new bout whenever the gap reaches the threshold OR the zone
    changes. Every bout is then, by construction, a run of reads in one zone
    with no gap in it.

``r_compat``
    A literal transcription of ``2_create_ALLTRIAL_MOVEBOUT.R``, validated
    row-for-row against that script's published output. It is available so old
    results reproduce exactly, but it is not the default: R pairs START rows to
    STOP rows by POSITION after dropping the reads in between, and its loop
    skips the transition check at ``i+1`` whenever it has just marked ``i+1`` as
    a START. On the 2021_LID data that mis-pairs 2.34% of bouts, which then
    span a gap or even a zone change internally.

Note also ``time_resolution`` on the config: R lost millisecond precision by
round-tripping through CSV between scripts, so its 50 s threshold was applied
to whole-second timestamps. Reproducing that needs ``time_resolution="s"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.defaults import TrialConfig

BOUT_SCHEMA = ["trial", "name", "code", "sex", "phase", "group", "strain",
               "zone", "zone_x", "zone_y", "noon_day", "field_time",
               "field_time_stop", "duration_s", "n_reads", "bout_status"]

EPOCH = pd.Timestamp("1970-01-01")


def _seconds(series: pd.Series) -> np.ndarray:
    """Datetimes as float seconds, independent of the column's time unit.

    ``astype("int64") / 1e9`` is the obvious way to do this and is a trap: a
    parquet round-trip can hand back microsecond-resolution datetimes, and the
    division then scales every gap by 1/1000 so no bout ever breaks.
    """
    return (series - EPOCH).dt.total_seconds().to_numpy()


def _classify_r(t: np.ndarray, z: np.ndarray, thresh: float) -> list:
    """R's ``bout_status`` pass for one animal-day: 'SINGLE_READ' or None."""
    n = len(t)
    status: list = [None] * n
    if n == 1:
        status[0] = "SINGLE_READ"
        return status
    if n == 2:
        # R evaluates both conditions in order; the second overwrites the first
        if z[0] == z[1] and (t[1] - t[0]) <= thresh:
            status[0], status[1] = "START", "STOP"
        if (t[1] - t[0]) >= thresh:
            status[0] = status[1] = "SINGLE_READ"
        return status
    if (t[1] - t[0]) >= thresh or z[1] != z[0]:
        status[0] = "SINGLE_READ"
    if (t[-1] - t[-2]) >= thresh or z[-1] != z[-2]:
        status[-1] = "SINGLE_READ"
    gaps = np.diff(t)
    mid = np.arange(1, n - 1)
    isolated = (gaps[mid] >= thresh) & (gaps[mid - 1] >= thresh)
    isolated |= (z[mid - 1] != z[mid]) & (z[mid + 1] != z[mid])
    for i in mid[isolated]:
        status[i] = "SINGLE_READ"
    return status


def _bouts_r_compat(group: pd.DataFrame, thresh: float) -> pd.DataFrame | None:
    t, z = _seconds(group["field_time"]), group["zone"].to_numpy()
    status = _classify_r(t, z, thresh)
    single = np.array([s == "SINGLE_READ" for s in status])

    pieces = []
    if single.any():
        s = group[single].copy()
        s["duration_s"] = 1.0
        s["field_time_stop"] = s["field_time"] + pd.Timedelta(seconds=1)
        s["bout_status"] = "SINGLE_READ"
        s["n_reads"] = 1
        pieces.append(s)

    rest = group[~single].reset_index(drop=True)
    if rest.empty:
        return pd.concat(pieces) if pieces else None

    marks: list = [None] * len(rest)
    marks[0], marks[-1] = "START", "STOP"
    rt, rz = _seconds(rest["field_time"]), rest["zone"].to_numpy()
    for i in range(len(rest)):
        if marks[i] is not None:
            continue
        if rz[i + 1] == rz[i] and (rt[i + 1] - rt[i]) >= thresh:
            marks[i], marks[i + 1] = "STOP", "START"
        if rz[i + 1] != rz[i]:
            marks[i], marks[i + 1] = "STOP", "START"
    rest["bout_status"] = marks
    kept = rest[rest["bout_status"].notna()].reset_index(drop=True)

    starts = kept[kept["bout_status"] == "START"].reset_index(drop=True)
    stops = kept[kept["bout_status"] == "STOP"].reset_index(drop=True)
    if len(starts) == len(stops) + 1:
        starts = starts.iloc[:-1]
    if len(stops) == len(starts) + 1:
        stops = stops.iloc[:-1]
    if len(starts) != len(stops) or starts.empty:
        return pd.concat(pieces) if pieces else None

    out = starts.copy()
    out["field_time_stop"] = stops["field_time"].to_numpy()
    duration = _seconds(stops["field_time"]) - _seconds(starts["field_time"])
    duration[duration == 0] = 1.0
    out["duration_s"] = duration
    out["n_reads"] = np.nan               # not recoverable from R's pairing
    pieces.append(out)
    return pd.concat(pieces)


def _bouts_segment(group: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """New bout whenever the gap reaches the threshold or the zone changes."""
    t, z = _seconds(group["field_time"]), group["zone"].to_numpy()
    breaks = np.ones(len(group), bool)
    if len(group) > 1:
        breaks[1:] = (np.diff(t) >= thresh) | (z[1:] != z[:-1])
    bout_id = np.cumsum(breaks)

    first = group.groupby(bout_id, sort=True).head(1).reset_index(drop=True)
    agg = (pd.DataFrame({"bout": bout_id, "t": t})
           .groupby("bout", sort=True)
           .agg(last=("t", "last"), first=("t", "first"), n=("t", "size")))
    out = first.copy()
    out["field_time_stop"] = pd.to_datetime(
        agg["last"].to_numpy(), unit="s").round("ms")
    duration = (agg["last"] - agg["first"]).to_numpy(float)
    out["n_reads"] = agg["n"].to_numpy()
    out["bout_status"] = np.where(out["n_reads"] == 1, "SINGLE_READ", "BOUT")
    out["duration_s"] = duration
    return out


def detect_bouts(reads: pd.DataFrame, threshold_s: float = 50.0,
                 min_duration_s: float = 1.0, algorithm: str = "segment",
                 progress=None) -> pd.DataFrame:
    """Turn a reads table into bouts, one row per continuous stay."""
    if reads.empty:
        return pd.DataFrame(columns=BOUT_SCHEMA)
    if algorithm not in ("segment", "r_compat"):
        raise ValueError(f"Unknown bout algorithm {algorithm!r}")
    say = progress or (lambda _m: None)

    build = _bouts_segment if algorithm == "segment" else _bouts_r_compat
    groups = list(reads.groupby(["name", "noon_day"], sort=False))
    out = []
    for i, ((name, _day), group) in enumerate(groups):
        if i % 200 == 0:
            say(f"Detecting bouts... {i:,}/{len(groups):,} animal-days")
        piece = build(group.sort_values("field_time", kind="stable"), threshold_s)
        if piece is not None and len(piece):
            out.append(piece)
    if not out:
        return pd.DataFrame(columns=BOUT_SCHEMA)

    bouts = pd.concat(out, ignore_index=True)
    # A zero-length bout is a single instant, not an absence of one; give it the
    # minimum duration so it still contributes to occupancy.
    bouts.loc[bouts["duration_s"] < min_duration_s, "duration_s"] = min_duration_s
    bouts = bouts.sort_values(["name", "field_time"], kind="stable").reset_index(drop=True)
    say(f"{len(bouts):,} bouts")
    return bouts[[c for c in BOUT_SCHEMA if c in bouts.columns]]


class BoutDetector:
    """Config-driven wrapper around :func:`detect_bouts`."""

    def __init__(self, config: TrialConfig, algorithm: str = "segment"):
        self.config = config
        self.algorithm = algorithm

    def run(self, reads: pd.DataFrame, progress=None) -> pd.DataFrame:
        return detect_bouts(reads, self.config.bout_threshold_s,
                            self.config.min_duration_s, self.algorithm, progress)
