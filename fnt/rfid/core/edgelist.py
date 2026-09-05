"""Dyadic co-presence: how long each pair of animals actually spent together.

Written fresh rather than ported. The R script for this (``3b``) never ran to
completion - it loops over one variable while reading another, and its method
for stitching adjacent intervals back together (dropping timestamps that appear
more than once) breaks whenever three intervals share an endpoint. No output
file was ever produced, so there is nothing to reproduce.

The definition here: for a pair, walk the GBI intervals of one zone in time
order and merge every run of CONTIGUOUS intervals in which both animals are
present. Contiguity is required - one interval's stop being the next one's
start - so a pair that leaves and returns records two encounters rather than
one long one. Other animals coming and going do not split the encounter, which
is the point: this measures the pair's association, not the group's.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .gbi_generator import GBI_META

BOUT_COLUMNS = ["trial", "day", "zone", "id1", "id2",
                "field_time_start", "field_time_stop", "duration_s"]
EDGE_COLUMNS = ["trial", "day", "id1", "id2", "n_bouts", "total_duration_s",
                "mean_duration_s", "n_zones"]


def co_presence_bouts(gbi: pd.DataFrame, animals: list[str] | None = None,
                      progress=None) -> pd.DataFrame:
    """One row per continuous encounter between a pair of animals."""
    if gbi.empty:
        return pd.DataFrame(columns=BOUT_COLUMNS)
    say = progress or (lambda _m: None)

    animals = animals or [c for c in gbi.columns if c not in GBI_META]
    trial = gbi["trial"].iloc[0] if "trial" in gbi.columns else ""
    gbi = gbi.sort_values(["zone", "field_time_start"], kind="stable")

    zone = gbi["zone"].to_numpy()
    day = gbi["day"].to_numpy()
    start = gbi["field_time_start"].to_numpy()
    stop = gbi["field_time_stop"].to_numpy()
    present = {a: gbi[a].to_numpy().astype(bool) for a in animals}

    rows = []
    pairs = list(itertools.combinations(animals, 2))
    for n, (a, b) in enumerate(pairs):
        if n % 100 == 0:
            say(f"Co-presence... pair {n:,}/{len(pairs):,}")
        together = present[a] & present[b]
        if not together.any():
            continue
        idx = np.flatnonzero(together)
        # split the run wherever the zone changes or the intervals are not
        # butted up against each other
        breaks = np.ones(len(idx), bool)
        if len(idx) > 1:
            prev, cur = idx[:-1], idx[1:]
            breaks[1:] = (zone[cur] != zone[prev]) | (start[cur] != stop[prev])
        group = np.cumsum(breaks)
        frame = pd.DataFrame({"g": group, "i": idx})
        for g, part in frame.groupby("g", sort=True):
            lo, hi = part["i"].iloc[0], part["i"].iloc[-1]
            rows.append((trial, int(day[lo]), int(zone[lo]), a, b,
                         start[lo], stop[hi],
                         (stop[hi] - start[lo]) / np.timedelta64(1, "s")))

    out = pd.DataFrame(rows, columns=BOUT_COLUMNS)
    say(f"{len(out):,} co-presence bouts")
    return out.sort_values(["id1", "id2", "field_time_start"],
                           kind="stable").reset_index(drop=True)


def edgelist(bouts: pd.DataFrame, by_day: bool = True) -> pd.DataFrame:
    """Aggregate co-presence bouts into a weighted edge list."""
    if bouts.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    keys = ["trial", "id1", "id2"] + (["day"] if by_day else [])
    out = (bouts.groupby(keys, observed=True)
                .agg(n_bouts=("duration_s", "size"),
                     total_duration_s=("duration_s", "sum"),
                     mean_duration_s=("duration_s", "mean"),
                     n_zones=("zone", "nunique"))
                .reset_index())
    if not by_day:
        out["day"] = np.nan
    return out[EDGE_COLUMNS].sort_values(
        ["day", "id1", "id2"], kind="stable").reset_index(drop=True)


class EdgelistGenerator:
    """Config-driven wrapper producing bouts and their aggregation."""

    def __init__(self, config=None):
        self.config = config

    def run(self, gbi: pd.DataFrame, animals=None,
            progress=None) -> dict[str, pd.DataFrame]:
        bouts = co_presence_bouts(gbi, animals, progress)
        return {"co_presence_bouts": bouts, "edgelist": edgelist(bouts)}
