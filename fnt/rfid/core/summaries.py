"""Analysis-ready summaries built from cleaned reads and bouts.

Three tables, each answering a question a reader of the data asks first:

``animal_days``
    Per animal and day: reads, bouts, zones entered (distinct zone_ids), the
    minimum distance travelled, and time in zones. Days between an animal's
    first and last day with no reads are filled in as zeros (``detected`` is
    False) - a day with no reads is a result, and dropping it biases every
    per-day mean upwards.

``zone_holders``
    Per paddock-zone and day: the animal of the holder sex (males, by default)
    with the largest share of that zone's reads from that sex, and whether the
    share clears the threshold (> 50%). Below it the zone is 'contested'.
    Visitors from the other paddock count where they were read.

``co_occupancy``
    Per day and pair of animals: seconds their bouts overlapped in the same
    zone_id. Built by an interval sweep per zone - sort bouts by start, and
    each bout's overlapping partners are exactly the later-starting bouts that
    start before it stops - so the work grows with the number of overlaps,
    not with the square of the number of bouts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _info(animals: pd.DataFrame | None) -> pd.DataFrame:
    if animals is None or animals.empty:
        return pd.DataFrame(columns=["animal", "name", "trial", "sex", "phase", "code"])
    return animals[["animal", "name", "trial", "sex", "phase", "code"]]


# --------------------------------------------------------------------------- #
# Per animal-day
# --------------------------------------------------------------------------- #

ANIMAL_DAY_COLUMNS = [
    "animal", "name", "trial", "home_paddock", "sex", "phase", "code", "day",
    "detected", "reads", "reads_away", "bouts", "zones_entered", "zone_ids",
    "min_distance_m", "time_in_zones_s", "time_away_s", "first_read", "last_read"]


def min_distance(bouts: pd.DataFrame, zone_xy: dict) -> pd.Series:
    """Minimum distance per (animal, day): straight lines between consecutive
    distinct zones, within a paddock only.

    A move between paddocks adds nothing: the route through the wall is
    unknown, and the two paddocks' coordinates are separate frames anyway.
    A zone without coordinates makes that animal-day's distance NaN rather
    than silently short.
    """
    if bouts.empty:
        return pd.Series(dtype=float)
    b = bouts.sort_values(["animal", "day", "start"], kind="mergesort")
    animal = b["animal"].astype(str).to_numpy()
    day = b["day"].to_numpy()
    zone = b["zone_id"].astype(str).to_numpy()
    paddock = b["paddock"].astype(str).to_numpy()
    same_group = np.r_[False, (animal[1:] == animal[:-1]) & (day[1:] == day[:-1])]
    change = ~same_group | np.r_[False, zone[1:] != zone[:-1]]
    k = np.flatnonzero(change)
    ka, kd, kz, kp = animal[k], day[k], zone[k], paddock[k]
    def coord(i):
        return {z: (np.nan if v[i] is None else v[i]) for z, v in zone_xy.items()}
    xs = pd.Series(kz).map(coord(0)).to_numpy(float)
    ys = pd.Series(kz).map(coord(1)).to_numpy(float)
    step_ok = np.r_[False, (ka[1:] == ka[:-1]) & (kd[1:] == kd[:-1]) & (kp[1:] == kp[:-1])]
    step = np.r_[0.0, np.hypot(np.diff(xs), np.diff(ys))]
    missing = step_ok & np.isnan(step)
    step = np.where(step_ok & ~missing, step, 0.0)
    frame = pd.DataFrame({"animal": ka, "day": kd, "step": step, "missing": missing})
    agg = frame.groupby(["animal", "day"]).agg(dist=("step", "sum"),
                                               missing=("missing", "any"))
    return agg["dist"].where(~agg["missing"], np.nan)


def animal_days(reads: pd.DataFrame, bouts: pd.DataFrame, animals: pd.DataFrame,
                zone_xy: dict, home: dict | None = None,
                complete: bool = True) -> pd.DataFrame:
    """One row per animal per day. See the module docstring."""
    if reads.empty:
        return pd.DataFrame(columns=ANIMAL_DAY_COLUMNS)
    r = pd.DataFrame({"animal": reads["animal"].astype(str), "day": reads["day"],
                      "away": reads["away"].astype(bool), "t": reads["t"]})
    per_read = r.groupby(["animal", "day"]).agg(
        reads=("t", "size"), reads_away=("away", "sum"),
        first_read=("t", "min"), last_read=("t", "max"))
    b = pd.DataFrame({"animal": bouts["animal"].astype(str), "day": bouts["day"],
                      "zone_id": bouts["zone_id"].astype(str),
                      "duration_s": bouts["duration_s"],
                      "away_s": np.where(bouts["away"].astype(bool), bouts["duration_s"], 0.0)})
    per_bout = b.groupby(["animal", "day"]).agg(
        bouts=("duration_s", "size"), zones_entered=("zone_id", "nunique"),
        zone_ids=("zone_id", lambda s: ";".join(sorted(set(s)))),
        time_in_zones_s=("duration_s", "sum"), time_away_s=("away_s", "sum"))
    out = per_read.join(per_bout, how="outer")
    out["min_distance_m"] = min_distance(bouts, zone_xy)

    if complete:
        span = out.reset_index().groupby("animal")["day"].agg(["min", "max"])
        index = pd.MultiIndex.from_tuples(
            [(a, d) for a, (lo, hi) in span.iterrows() for d in range(int(lo), int(hi) + 1)],
            names=["animal", "day"])
        out = out.reindex(index)
    out = out.reset_index()
    for col in ("reads", "reads_away", "bouts", "zones_entered"):
        out[col] = out[col].fillna(0).astype(int)
    for col in ("time_in_zones_s", "time_away_s"):
        out[col] = out[col].fillna(0.0).round(3)
    out["min_distance_m"] = out["min_distance_m"].round(3)
    # a day with no bouts travelled nowhere; NaN stays only where coordinates are missing
    out.loc[out["bouts"] == 0, "min_distance_m"] = 0.0
    out["zone_ids"] = out["zone_ids"].fillna("")
    out["detected"] = out["reads"] > 0
    out = out.merge(_info(animals), on="animal", how="left")
    out["home_paddock"] = out["animal"].map(home or {}).fillna("")
    return out[ANIMAL_DAY_COLUMNS].sort_values(["animal", "day"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Zone holders
# --------------------------------------------------------------------------- #

HOLDER_COLUMNS = ["paddock", "zone", "zone_id", "day", "sex_reads", "n_animals",
                  "top_animal", "top_code", "top_phase", "top_home", "top_reads",
                  "top_share", "runner_up", "runner_up_share", "holder", "status"]


def zone_holders(reads: pd.DataFrame, animals: pd.DataFrame, share: float = 0.5,
                 sex: str = "M", home: dict | None = None) -> pd.DataFrame:
    """Who held each zone on each day, by share of that sex's reads there.

    Scored from READS rather than bouts: an animal sitting on a zone's antenna
    generates reads continuously, so a share of reads is a share of presence.
    The threshold is strict (> ``share``), so two animals level at 50% leave
    the zone contested rather than inventing a holder.
    """
    if reads.empty:
        return pd.DataFrame(columns=HOLDER_COLUMNS)
    info = _info(animals).set_index("animal")
    sex_of = info["sex"].to_dict()
    animal = reads["animal"].astype(str)
    mask = animal.map(sex_of).fillna("").astype(str).str.upper() == sex.upper() \
        if sex else pd.Series(True, index=reads.index)
    r = pd.DataFrame({"paddock": reads["paddock"].astype(str)[mask],
                      "zone": reads["zone"].astype(str)[mask],
                      "zone_id": reads["zone_id"].astype(str)[mask],
                      "day": reads["day"][mask], "animal": animal[mask]})
    if r.empty:
        return pd.DataFrame(columns=HOLDER_COLUMNS)
    keys = ["paddock", "zone", "zone_id", "day"]
    counts = r.groupby(keys + ["animal"]).size().rename("n").reset_index()
    counts["total"] = counts.groupby(keys)["n"].transform("sum")
    counts["share"] = counts["n"] / counts["total"]
    counts = counts.sort_values(keys + ["n", "animal"], ascending=[True] * 4 + [False, True],
                                kind="mergesort")
    counts["rank"] = counts.groupby(keys).cumcount()
    top = counts[counts["rank"] == 0].set_index(keys)
    second = counts[counts["rank"] == 1].set_index(keys)
    out = pd.DataFrame({
        "sex_reads": top["total"],
        "n_animals": counts.groupby(keys)["animal"].size(),
        "top_animal": top["animal"],
        "top_reads": top["n"],
        "top_share": top["share"],
        "runner_up": second["animal"].reindex(top.index),
        "runner_up_share": second["share"].reindex(top.index)}).reset_index()
    out["top_code"] = out["top_animal"].map(info["code"]).fillna("")
    out["top_phase"] = out["top_animal"].map(info["phase"]).fillna("")
    out["top_home"] = out["top_animal"].map(home or {}).fillna("")
    held = out["top_share"] > share
    out["holder"] = np.where(held, out["top_animal"], "")
    out["status"] = np.where(held, "held", "contested")
    out["runner_up"] = out["runner_up"].fillna("")
    return out[HOLDER_COLUMNS].sort_values(["paddock", "day", "zone_id"]) \
                              .reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Co-occupancy
# --------------------------------------------------------------------------- #

CO_COLUMNS = ["day", "animal_a", "animal_b", "sex_a", "sex_b", "seconds",
              "n_overlaps", "zone_ids"]


def overlapping_bouts(bouts: pd.DataFrame) -> pd.DataFrame:
    """Every pair of bouts by different animals that overlap in the same zone_id.

    Returns one row per overlapping pair with the overlap in seconds. Per zone:
    sort by start; bout i's partners are the bouts j > i with
    ``start_j < stop_i`` (``searchsorted`` finds where they end). Since
    ``start_j >= start_i``, the overlap is ``min(stop_i, stop_j) - start_j``,
    always positive for those j.
    """
    cols = ["zone_id", "day", "i", "j", "seconds"]
    if bouts.empty:
        return pd.DataFrame(columns=cols)
    b = bouts.reset_index(drop=True)
    animal = pd.factorize(b["animal"].astype(str))[0]
    start = pd.DatetimeIndex(b["start"]).asi8
    stop = pd.DatetimeIndex(b["stop"]).asi8
    zone = b["zone_id"].astype(str).to_numpy()
    day = b["day"].to_numpy()
    pieces = []
    for z in pd.unique(zone):
        idx = np.flatnonzero(zone == z)
        idx = idx[np.argsort(start[idx], kind="mergesort")]
        s, e = start[idx], stop[idx]
        end = np.searchsorted(s, e, side="left")
        counts = np.maximum(end - np.arange(len(idx)) - 1, 0)
        total = int(counts.sum())
        if not total:
            continue
        ii = np.repeat(np.arange(len(idx)), counts)
        offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        jj = ii + 1 + offsets
        overlap = (np.minimum(e[ii], e[jj]) - s[jj]) / 1e9
        keep = (overlap > 0) & (animal[idx[ii]] != animal[idx[jj]])
        gi, gj = idx[ii[keep]], idx[jj[keep]]
        pieces.append(pd.DataFrame({"zone_id": z, "day": day[gj], "i": gi, "j": gj,
                                    "seconds": overlap[keep]}))
    if not pieces:
        return pd.DataFrame(columns=cols)
    return pd.concat(pieces, ignore_index=True)


def co_occupancy(bouts: pd.DataFrame, animals: pd.DataFrame | None = None) -> pd.DataFrame:
    """Seconds each pair of animals spent in the same zone at the same time, per day.

    One row per unordered pair (``animal_a`` < ``animal_b``); mirror it to get
    each animal's partners. The day is the day of the later-starting bout,
    i.e. of the moment the overlap began.
    """
    pairs = overlapping_bouts(bouts)
    if pairs.empty:
        return pd.DataFrame(columns=CO_COLUMNS)
    names = bouts.reset_index(drop=True)["animal"].astype(str).to_numpy()
    a, b = names[pairs["i"].to_numpy()], names[pairs["j"].to_numpy()]
    pairs["animal_a"] = np.where(a < b, a, b)
    pairs["animal_b"] = np.where(a < b, b, a)
    out = (pairs.groupby(["day", "animal_a", "animal_b"])
                .agg(seconds=("seconds", "sum"), n_overlaps=("seconds", "size"),
                     zone_ids=("zone_id", lambda s: ";".join(sorted(set(s)))))
                .reset_index())
    sex = _info(animals).set_index("animal")["sex"].to_dict()
    out["sex_a"] = out["animal_a"].map(sex).fillna("")
    out["sex_b"] = out["animal_b"].map(sex).fillna("")
    out["seconds"] = out["seconds"].round(3)
    return out[CO_COLUMNS].sort_values(["day", "animal_a", "animal_b"]) \
                          .reset_index(drop=True)
