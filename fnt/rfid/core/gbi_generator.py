"""Stage 3: bouts -> a group-by-individual (GBI) matrix.

Within a zone, take every bout start and stop as an endpoint and sort them.
Consecutive endpoints bound an interval over which the set of animals present
cannot change, because the only thing that changes it is an endpoint. Each
interval becomes a row, with a 1 for every animal in the zone during it.

The R original tested every interval against every bout with lubridate's
``%within%``, which is O(intervals x bouts) per zone and took hours. The sweep
here adds a bout when its start is reached and removes it when its stop is
passed, so the active set IS the answer - same result, seconds instead.

One inherited subtlety, kept deliberately: when two endpoints coincide (one
animal leaves exactly as another arrives) the interval has zero length, and the
probe point sits exactly on the boundary. Because intervals are closed at both
ends, BOTH the departing and the arriving animal count as present. Those rows
are the clearest hand-off events in the data, so they are worth keeping; the
duration is reported as ``min_duration_s`` rather than zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config.defaults import TrialConfig

EPOCH = pd.Timestamp("1970-01-01")
GBI_META = ["trial", "day", "zone", "field_time_start", "field_time_stop",
            "duration_s", "m_sum", "f_sum", "mf_sum"]


def _seconds(series: pd.Series) -> np.ndarray:
    return (series - EPOCH).dt.total_seconds().to_numpy()


def _zone_intervals(starts, stops, days, who, min_duration_s):
    """Sweep one zone's bouts into (start, stop, duration, day, present)."""
    endpoints = np.sort(np.concatenate([starts, stops]))
    left, right = endpoints[:-1], endpoints[1:]
    raw = right - left
    duration = np.where(raw == 0, min_duration_s, raw)
    # probe from the RAW span, so a zero-length interval is probed on the
    # boundary itself and catches both sides of a hand-off
    probe = left + raw / 2.0

    entering = np.argsort(starts, kind="stable")
    leaving = np.argsort(stops, kind="stable")
    i = j = 0
    active: set[int] = set()
    rows = []
    for k, point in enumerate(probe):
        while i < len(entering) and starts[entering[i]] <= point:
            active.add(int(entering[i]))
            i += 1
        while j < len(leaving) and stops[leaving[j]] < point:
            active.discard(int(leaving[j]))
            j += 1
        if not active:
            continue
        rows.append((left[k], right[k], duration[k],
                     int(days[max(active)]), [who[m] for m in active]))
    return rows


def create_gbi(bouts: pd.DataFrame, animals: list[str] | None = None,
               sex_by_name: dict[str, str] | None = None,
               min_duration_s: float = 1.0, progress=None) -> pd.DataFrame:
    """Build the GBI matrix for one trial's bouts."""
    if bouts.empty:
        return pd.DataFrame(columns=GBI_META)
    say = progress or (lambda _m: None)

    bouts = bouts.copy()
    bouts["field_time_stop"] = pd.to_datetime(bouts["field_time_stop"])
    animals = animals or sorted(bouts["name"].unique())
    if sex_by_name is None:
        sex_by_name = (bouts.drop_duplicates("name")
                            .set_index("name")["sex"].to_dict())
    trial = bouts["trial"].iloc[0] if "trial" in bouts.columns else ""

    records = []
    zones = sorted(bouts["zone"].dropna().unique())
    for zone in zones:
        say(f"Building GBI for zone {zone}...")
        z = bouts[bouts["zone"] == zone].sort_values("field_time", kind="stable")
        if z.empty:
            continue
        for start, stop, duration, day, present in _zone_intervals(
                _seconds(z["field_time"]), _seconds(z["field_time_stop"]),
                z["noon_day"].to_numpy(), z["name"].to_numpy(), min_duration_s):
            row = {"trial": trial, "day": day, "zone": int(zone),
                   "field_time_start": start, "field_time_stop": stop,
                   "duration_s": duration}
            row.update(dict.fromkeys(animals, 0))
            for animal in present:
                row[animal] = 1
            records.append(row)

    if not records:
        return pd.DataFrame(columns=GBI_META + animals)

    gbi = pd.DataFrame.from_records(records)
    for col in ("field_time_start", "field_time_stop"):
        gbi[col] = pd.to_datetime(gbi[col], unit="s").dt.round("ms")

    males = [a for a in animals if sex_by_name.get(a) == "M"]
    females = [a for a in animals if sex_by_name.get(a) == "F"]
    gbi["m_sum"] = gbi[males].sum(axis=1) if males else 0
    gbi["f_sum"] = gbi[females].sum(axis=1) if females else 0
    gbi["mf_sum"] = gbi["m_sum"] + gbi["f_sum"]

    gbi = gbi.sort_values(["zone", "field_time_start"], kind="stable")
    say(f"{len(gbi):,} GBI intervals across {len(zones)} zones")
    return gbi[GBI_META + animals].reset_index(drop=True)


def melt_gbi(gbi: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long form: one row per animal per interval it was present for.

    This is the per-animal occupancy table. It is derived from the GBI rather
    than from the bouts, so an animal's row carries who else was in the zone at
    the time - which is the whole point of having it.
    """
    animals = [c for c in gbi.columns if c not in GBI_META]
    pieces = []
    for animal in animals:
        part = gbi.loc[gbi[animal] == 1, GBI_META].copy()
        if part.empty:
            continue
        part.insert(3, "name", animal)
        pieces.append(part)
    if not pieces:
        return pd.DataFrame(columns=GBI_META[:3] + ["name"] + GBI_META[3:])
    out = pd.concat(pieces, ignore_index=True)
    if meta is not None and not meta.empty:
        cols = [c for c in ("name", "code", "sex", "phase", "group", "strain")
                if c in meta.columns]
        out = out.merge(meta[cols].drop_duplicates("name"), on="name", how="left")
    return out.sort_values(["name", "field_time_start"],
                           kind="stable").reset_index(drop=True)


class GBIGenerator:
    """Config-driven wrapper around :func:`create_gbi`."""

    def __init__(self, config: TrialConfig):
        self.config = config

    def run(self, bouts: pd.DataFrame, animals=None, sex_by_name=None,
            progress=None) -> pd.DataFrame:
        return create_gbi(bouts, animals, sex_by_name,
                          self.config.min_duration_s, progress)
