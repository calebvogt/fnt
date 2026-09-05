"""Displacement events: one male arrives, and the resident leaves.

Within a zone, a run of male-count 1 -> more than 1 -> 1 is read as a contest.
The male present both before and after held the zone; the other one left. This
is a coarse but reliable signal precisely because it needs no thresholds - it
falls out of who is in the zone and when.

What it cannot distinguish is a contest from a coincidence: two males swapping
places within one interval look the same as one displacing the other. Treat the
counts as an upper bound on displacements, not a scored behaviour.

Zone ownership supplies home/away, which is what makes the events interpretable
- a resident repelling an intruder and an intruder evicting a resident are very
different results with the same shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .gbi_generator import GBI_META
from .zone_ownership import zones_owned_per_day

DISPLACE_SCHEMA = [
    "trial", "day", "zone", "field_time_start", "field_time_stop", "duration_s",
    "zone_owner", "zone_owner_perc_reads", "mouse1", "mouse2", "winner_order",
    "winner", "winner_loc", "winner_zones_owned", "loser", "loser_loc",
    "loser_zones_owned", "pre_dispute_type", "dispute_type",
    "post_dispute_type", "interaction_type"]


def detect_displacements(gbi: pd.DataFrame, males: list[str],
                         progress=None) -> pd.DataFrame:
    """Raw 1 -> n -> 1 male displacement events, before ownership is joined."""
    if gbi.empty or not males:
        return pd.DataFrame(columns=DISPLACE_SCHEMA)
    say = progress or (lambda _m: None)

    present = {m: gbi[m].to_numpy() for m in males if m in gbi.columns}
    males = list(present)
    matrix = np.vstack([present[m] for m in males])
    zone = gbi["zone"].to_numpy()
    m_sum = gbi["m_sum"].to_numpy()
    trial = gbi["trial"].iloc[0] if "trial" in gbi.columns else ""

    rows = []
    for z in np.unique(zone):
        idx = np.flatnonzero((zone == z) & (m_sum > 0))
        if len(idx) < 3:
            continue
        counts = m_sum[idx]
        hits = np.flatnonzero((counts[:-2] == 1) & (counts[1:-1] > 1)
                              & (counts[2:] == 1))
        for h in hits:
            before, during, after = idx[h], idx[h + 1], idx[h + 2]
            first = [males[k] for k in np.flatnonzero(matrix[:, before] == 1)]
            both = [males[k] for k in np.flatnonzero(matrix[:, during] == 1)]
            last = [males[k] for k in np.flatnonzero(matrix[:, after] == 1)]
            second = [m for m in both if m not in first]
            loser = [m for m in both if m not in last]
            # More than two males in the middle interval, or an ambiguous
            # winner, is not a clean dyadic displacement - skip it.
            if len(first) != 1 or len(second) != 1 or len(last) != 1 or len(loser) != 1:
                continue
            rows.append({
                "trial": trial, "day": int(gbi["day"].to_numpy()[during]),
                "zone": int(z),
                "field_time_start": gbi["field_time_start"].to_numpy()[during],
                "field_time_stop": gbi["field_time_stop"].to_numpy()[during],
                "duration_s": float(gbi["duration_s"].to_numpy()[during]),
                "mouse1": first[0], "mouse2": second[0],
                "winner": last[0], "loser": loser[0],
                "winner_order": "mouse1" if last[0] == first[0] else "mouse2"})
    say(f"{len(rows):,} displacement events")
    return pd.DataFrame(rows)


def annotate_ownership(events: pd.DataFrame, ownership: pd.DataFrame,
                       min_percent: float = 50.0,
                       days: tuple[int, int] | None = None) -> pd.DataFrame:
    """Add zone owner, home/away, and the R/I interaction typing.

    ``pre``/``post`` are the status of the zone's holder before and after:
    R for the resident owner, I for an intruder. ``dispute_type`` is RI when the
    owner was one of the two, II when neither was.
    """
    if events.empty:
        return pd.DataFrame(columns=DISPLACE_SCHEMA)
    out = events.copy()
    out["trial_zone_day"] = (out["trial"].astype(str) + "_"
                             + out["zone"].astype(str) + "_"
                             + out["day"].astype(str))

    top = ownership[ownership["rank_order"] == 1][
        ["trial_zone_day", "name", "mus_perc_zone_reads"]]
    out = out.merge(top, on="trial_zone_day", how="left").rename(
        columns={"name": "zone_owner", "mus_perc_zone_reads": "zone_owner_perc_reads"})

    owner = out["zone_owner"]
    out["pre_dispute_type"] = np.where(out["mouse1"] == owner, "R", "I")
    out["dispute_type"] = np.where((out["mouse1"] == owner)
                                   | (out["mouse2"] == owner), "RI", "II")
    out["post_dispute_type"] = np.where(out["winner"] == owner, "R", "I")
    out["interaction_type"] = (out["pre_dispute_type"] + "_"
                               + out["dispute_type"] + "_"
                               + out["post_dispute_type"])
    out["winner_loc"] = np.where(out["winner"] == owner, "home", "away")
    out["loser_loc"] = np.where(out["loser"] == owner, "home", "away")

    owned = zones_owned_per_day(ownership, min_percent, days)
    for who in ("winner", "loser"):
        out = out.merge(
            owned.rename(columns={"name": who, "noon_day": "day",
                                  "zones_owned": f"{who}_zones_owned"}),
            on=[who, "day"], how="left")
        out[f"{who}_zones_owned"] = out[f"{who}_zones_owned"].fillna(0).astype(int)

    for col in DISPLACE_SCHEMA:
        if col not in out.columns:
            out[col] = np.nan
    return out[DISPLACE_SCHEMA].reset_index(drop=True)


class DisplacementDetector:
    """Config-driven wrapper: detect, then annotate with ownership."""

    def __init__(self, config=None):
        self.config = config

    def run(self, gbi: pd.DataFrame, males: list[str], ownership: pd.DataFrame,
            progress=None) -> pd.DataFrame:
        days = getattr(self.config, "analysis_days", None) if self.config else None
        return annotate_ownership(detect_displacements(gbi, males, progress),
                                  ownership, days=days)
