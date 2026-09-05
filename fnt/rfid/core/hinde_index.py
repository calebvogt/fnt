"""Hinde contact indices: who closes the distance and who breaks it.

For a pair of animals, look through the GBI intervals for a run where exactly
one of them is present, then both, then exactly one again. The animal that
arrives to make the pair *made contact*; the one that is gone afterwards
*broke contact*. Summed over a trial this is Hinde's index of who maintains
proximity.

Two scopes:

``broad``
    Every such run, regardless of who else was in the zone. If A and B are
    together and C joins, that still registers as a contact between the pair
    being examined.

``narrow``
    Only runs where the pair are the sole occupants (``mf_sum == pair_sum``),
    so the contact is unambiguously between those two animals.

The three intervals are consecutive *among the rows that survive filtering*,
which is not the same as consecutive in time - a row where neither animal is
present is skipped rather than breaking the run. That is how the R original
behaved and it is the right call: an interval in which neither animal is in the
zone does not interrupt a contact between them, it just is not about them.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .gbi_generator import GBI_META

HINDE_SCHEMA = ["trial", "day", "zone", "field_time_start", "field_time_stop",
                "duration_s", "mouse1", "mouse2", "made_contact",
                "broke_contact"]


def hinde_index(gbi: pd.DataFrame, scope: str = "broad",
                animals: list[str] | None = None, progress=None) -> pd.DataFrame:
    """Contact events for every pair of animals in ``gbi``."""
    if scope not in ("broad", "narrow"):
        raise ValueError(f"scope must be 'broad' or 'narrow', not {scope!r}")
    if gbi.empty:
        return pd.DataFrame(columns=HINDE_SCHEMA)
    say = progress or (lambda _m: None)

    animals = animals or [c for c in gbi.columns if c not in GBI_META]
    trial = gbi["trial"].iloc[0] if "trial" in gbi.columns else ""
    zone = gbi["zone"].to_numpy()
    day = gbi["day"].to_numpy()
    mf = gbi["mf_sum"].to_numpy()
    start = gbi["field_time_start"].to_numpy()
    stop = gbi["field_time_stop"].to_numpy()
    duration = gbi["duration_s"].to_numpy()
    presence = {a: gbi[a].to_numpy() for a in animals}
    zones = np.unique(zone)

    rows = []
    pairs = list(itertools.combinations(animals, 2))
    for n, (a, b) in enumerate(pairs):
        if n % 100 == 0:
            say(f"Hinde ({scope})... pair {n:,}/{len(pairs):,}")
        pair_sum = presence[a] + presence[b]
        eligible = (mf == pair_sum) if scope == "narrow" else np.ones(len(mf), bool)
        for z in zones:
            idx = np.flatnonzero(eligible & (zone == z) & (pair_sum > 0))
            if len(idx) < 3:
                continue
            v = pair_sum[idx]
            hits = np.flatnonzero((v[:-2] == 1) & (v[1:-1] > 1) & (v[2:] == 1))
            for h in hits:
                before, during, after = idx[h], idx[h + 1], idx[h + 2]
                first = a if presence[a][before] == 1 else b
                second = b if first == a else a
                stayed = a if presence[a][after] == 1 else b
                broke = b if stayed == a else a
                rows.append((trial, int(day[during]), int(z), start[during],
                             stop[during], float(duration[during]),
                             first, second, second, broke))

    out = pd.DataFrame(rows, columns=HINDE_SCHEMA)
    say(f"{len(out):,} {scope} contact events")
    return out.sort_values(["field_time_start", "mouse1", "mouse2"],
                           kind="stable").reset_index(drop=True)


def hinde_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Per animal: contacts made, contacts broken, and Hinde's index.

    The index is ``(made - broke) / (made + broke)``: +1 means the animal does
    all the approaching and none of the leaving, -1 the reverse, 0 balanced.
    """
    if events.empty:
        return pd.DataFrame(columns=["name", "made", "broke", "total",
                                     "hinde_index"])
    made = events["made_contact"].value_counts()
    broke = events["broke_contact"].value_counts()
    names = sorted(set(made.index) | set(broke.index))
    out = pd.DataFrame({
        "name": names,
        "made": [int(made.get(n, 0)) for n in names],
        "broke": [int(broke.get(n, 0)) for n in names]})
    out["total"] = out["made"] + out["broke"]
    out["hinde_index"] = np.where(
        out["total"] > 0, (out["made"] - out["broke"]) / out["total"], np.nan)
    return out


class HindeIndexCalculator:
    """Convenience wrapper producing both scopes."""

    def __init__(self, config=None):
        self.config = config

    def run(self, gbi: pd.DataFrame, progress=None) -> dict[str, pd.DataFrame]:
        return {scope: hinde_index(gbi, scope, progress=progress)
                for scope in ("broad", "narrow")}
