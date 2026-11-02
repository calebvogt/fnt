"""Cleaned reads -> bouts: continuous stays of one animal in one zone.

A tag sitting on an antenna is re-read about once a second, so an animal in a
zone produces a dense run of reads and a bout is that run. The default rule
(``gap_zone_day``):

* per animal, reads in time order (both of its tags together);
* consecutive reads in the same ``zone_id`` no more than ``gap`` seconds apart
  (default 50 s) belong to the same bout;
* a zone change, a longer gap, or the noon day boundary ends it;
* duration = max(stop - start, 1 s), so a lone read is a 1 s bout and still
  counts as a visit; ``stop`` is moved out to ``start + duration`` to match.

The day split matters for per-day tables - a bout that ran across noon would
otherwise be credited entirely to the day it started in.

``legacy_r`` reproduces ``2_create_ALLTRIAL_MOVEBOUT.R`` (the original
pipeline), for regenerating published numbers only. It classifies lone reads,
then pairs the remaining START and STOP marks by POSITION within each
animal-day, which mis-pairs about 2% of bouts on real data: they end up
spanning a gap or even a zone change. Its loop skips the check at ``i+1``
whenever it has just marked ``i+1``; that makes the marks alternate within a
run of consecutive transitions, which is what lets it be computed here with
array operations rather than a loop over reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BOUT_COLUMNS = ["animal", "name", "trial", "sex", "phase", "code", "paddock",
                "zone", "zone_id", "away", "day", "start", "stop", "duration_s",
                "n_reads"]


def _codes(series: pd.Series) -> np.ndarray:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.to_numpy()
    return pd.factorize(series)[0]


def gaps(values: np.ndarray) -> np.ndarray:
    """Differences of elapsed seconds, rounded to the microsecond.

    Reader times are whole milliseconds, so a gap of exactly 50 s is exactly
    50 s - but as a difference of two floats it can come out 50.0000000001
    and fall on the wrong side of the threshold. Rounding makes the
    comparison mean what it says.
    """
    return np.round(np.diff(values), 6)


def _elapsed(reads: pd.DataFrame) -> np.ndarray:
    if "elapsed" in reads.columns:
        return reads["elapsed"].to_numpy(float)
    # asi8 is in the index's own unit (microseconds under pandas 3): pin ns
    return pd.DatetimeIndex(reads["t"]).as_unit("ns").asi8 / 1e9


def _attach_animal_info(bouts: pd.DataFrame, animals: pd.DataFrame | None) -> pd.DataFrame:
    if animals is None or animals.empty:
        for col in ("name", "trial", "sex", "phase", "code"):
            if col not in bouts.columns:
                bouts[col] = ""
        return bouts
    info = animals.set_index("animal")
    for col in ("name", "trial", "sex", "phase", "code"):
        bouts[col] = bouts["animal"].astype(str).map(info[col]).fillna("")
    return bouts


def detect_bouts(reads: pd.DataFrame, gap_s: float = 50.0, min_bout_s: float = 1.0,
                 rule: str = "gap_zone_day", animals: pd.DataFrame | None = None,
                 progress=None) -> pd.DataFrame:
    """Bouts from cleaned reads sorted by (animal, t). See the module docstring."""
    say = progress or (lambda _m: None)
    if rule not in ("gap_zone_day", "legacy_r"):
        raise ValueError(f"Unknown bout rule {rule!r}")
    if reads.empty:
        return pd.DataFrame(columns=BOUT_COLUMNS)
    if rule == "legacy_r":
        bouts = _legacy_r(reads, gap_s)
    else:
        bouts = _gap_zone_day(reads, gap_s)
    short = bouts["duration_s"] < min_bout_s
    bouts.loc[short, "duration_s"] = float(min_bout_s)
    bouts["stop"] = bouts["stop"].where(
        ~short, bouts["start"] + pd.to_timedelta(bouts["duration_s"], unit="s"))
    bouts = _attach_animal_info(bouts, animals)
    say(f"{len(bouts):,} bouts ({rule}), "
        f"{bouts['duration_s'].sum() / 3600:,.0f} animal-hours in zones")
    return bouts[BOUT_COLUMNS].sort_values(["animal", "start"], kind="mergesort") \
                              .reset_index(drop=True)


def _gap_zone_day(reads: pd.DataFrame, gap_s: float) -> pd.DataFrame:
    n = len(reads)
    animal = _codes(reads["animal"])
    zone = _codes(reads["zone_id"])
    day = reads["day"].to_numpy()
    el = _elapsed(reads)
    new = np.ones(n, bool)
    if n > 1:
        new[1:] = ((animal[1:] != animal[:-1]) | (zone[1:] != zone[:-1])
                   | (day[1:] != day[:-1]) | (gaps(el) > gap_s))
    first = np.flatnonzero(new)
    last = np.append(first[1:] - 1, n - 1)
    t = reads["t"].to_numpy()
    out = reads.iloc[first][["animal", "paddock", "zone", "zone_id", "away", "day"]] \
               .reset_index(drop=True)
    out["start"] = t[first]
    out["stop"] = t[last]
    # the reader stamps milliseconds; anything finer is float noise
    out["duration_s"] = np.round(el[last] - el[first], 3)
    out["n_reads"] = last - first + 1
    return out


# --------------------------------------------------------------------------- #
# The legacy R rule
# --------------------------------------------------------------------------- #

def _group_positions(group: np.ndarray):
    """Position within each run of equal group ids, and each run's size."""
    n = len(group)
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]]) if n else np.array([], int)
    sizes = np.diff(np.append(starts, n))
    pos = np.arange(n) - np.repeat(starts, sizes)
    size = np.repeat(sizes, sizes)
    return pos, size


def legacy_single_reads(t: np.ndarray, z: np.ndarray, group: np.ndarray,
                        thresh: float) -> np.ndarray:
    """R's SINGLE_READ classification, per animal-day group."""
    n = len(t)
    pos, size = _group_positions(group)
    same_prev = np.r_[False, group[1:] == group[:-1]]
    same_next = np.r_[group[:-1] == group[1:], False]
    gap_prev = np.where(same_prev, np.r_[np.inf, gaps(t)], np.inf)
    gap_next = np.where(same_next, np.r_[gaps(t), np.inf], np.inf)
    zdiff_prev = same_prev & np.r_[False, z[1:] != z[:-1]]
    zdiff_next = same_next & np.r_[z[:-1] != z[1:], False]

    single = np.zeros(n, bool)
    single |= size == 1
    two = size == 2
    # R's two-read case: the gap test overwrites the same-zone test
    pair_gap = np.where(pos == 0, gap_next, gap_prev)
    single |= two & (pair_gap >= thresh)
    many = size >= 3
    first, lastpos = pos == 0, pos == size - 1
    middle = ~first & ~lastpos
    single |= many & first & ((gap_next >= thresh) | zdiff_next)
    single |= many & lastpos & ((gap_prev >= thresh) | zdiff_prev)
    single |= many & middle & (((gap_prev >= thresh) & (gap_next >= thresh))
                               | (zdiff_prev & zdiff_next))
    return single


def legacy_marks(t: np.ndarray, z: np.ndarray, group: np.ndarray,
                 thresh: float) -> np.ndarray:
    """START (1) / STOP (2) / none (0) marks on the non-single reads.

    R's loop, for i in order: skip if already marked; else if the step to
    i+1 is a transition (same zone and gap >= thresh, or a zone change) mark
    i STOP and i+1 START. Index 0 and the last index start out marked. So in
    a run of consecutive transitions only every other one is applied - the
    first, third, fifth... (counting from index 1 when the run touches 0).
    """
    n = len(t)
    marks = np.zeros(n, np.int8)
    if n == 0:
        return marks
    pos, size = _group_positions(group)
    same_next = np.r_[group[:-1] == group[1:], False]
    gap_next = np.r_[gaps(t), np.inf]
    zchange = np.r_[z[:-1] != z[1:], False]
    trans = same_next & (zchange | (gap_next >= thresh))
    candidate = trans & (pos >= 1) & (pos <= size - 2)
    # alternate within runs of consecutive candidates
    run_start = candidate & ~np.r_[False, candidate[:-1] & same_next[:-1]]
    run_id = np.cumsum(run_start)
    first_in_run = np.zeros(n, int)
    starts_idx = np.flatnonzero(run_start)
    first_in_run[candidate] = starts_idx[run_id[candidate] - 1]
    applied = candidate & (((np.arange(n) - first_in_run) % 2) == 0)

    marks[pos == 0] = 1
    marks[pos == size - 1] = 2
    marks[applied] = 2
    marks[np.flatnonzero(applied) + 1] = 1
    return marks


def _legacy_r(reads: pd.DataFrame, thresh: float) -> pd.DataFrame:
    animal = _codes(reads["animal"])
    day = reads["day"].to_numpy()
    group = pd.factorize(pd.MultiIndex.from_arrays([animal, day]))[0]
    t = _elapsed(reads)
    z = _codes(reads["zone_id"])
    cols = ["animal", "paddock", "zone", "zone_id", "away", "day"]

    single = legacy_single_reads(t, z, group, thresh)
    singles = reads.loc[single, cols].reset_index(drop=True)
    singles["start"] = reads["t"].to_numpy()[single]
    singles["stop"] = singles["start"] + pd.Timedelta(seconds=1)
    singles["duration_s"] = 1.0
    singles["n_reads"] = 1

    rest = np.flatnonzero(~single)
    marks = legacy_marks(t[rest], z[rest], group[rest], thresh)
    kept = rest[marks > 0]
    kmarks = marks[marks > 0]
    kgroup = group[kept]
    frame = pd.DataFrame({"row": kept, "mark": kmarks, "group": kgroup})
    frame["k"] = frame.groupby(["group", "mark"]).cumcount()
    counts = frame.groupby(["group", "mark"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[1, 2], fill_value=0)
    ns, ne = counts[1], counts[2]
    # drop the unmatched trailing mark when the counts differ by one; a group
    # whose counts differ by more has no paired bouts at all (as in R)
    n_pairs = np.where((ns - ne).abs() <= 1, np.minimum(ns, ne), 0)
    n_pairs = pd.Series(n_pairs, index=counts.index)
    frame = frame[frame["k"] < frame["group"].map(n_pairs).to_numpy()]
    starts = frame[frame["mark"] == 1].sort_values(["group", "k"])
    stops = frame[frame["mark"] == 2].sort_values(["group", "k"])
    s_rows, e_rows = starts["row"].to_numpy(), stops["row"].to_numpy()
    paired = reads.iloc[s_rows][cols].reset_index(drop=True)
    paired["start"] = reads["t"].to_numpy()[s_rows]
    paired["stop"] = reads["t"].to_numpy()[e_rows]
    paired["duration_s"] = np.round(t[e_rows] - t[s_rows], 3)
    paired.loc[paired["duration_s"] == 0, "duration_s"] = 1.0
    paired["n_reads"] = np.nan               # not recoverable from R's pairing
    return pd.concat([singles, paired], ignore_index=True)
