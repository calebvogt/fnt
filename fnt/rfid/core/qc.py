"""Quality-control reports: what a person should look at before trusting a run.

Each report is a table, shown in the GUI and written as ``<name>_qc_*.csv``.
Two of them look for problems the pipeline cannot decide on its own and so
run on the IDENTIFIED reads (before lost-tag and found-dead cuts), marking
what a configured cut already covers:

``two_places``
    Animal-days on which two of an animal's tags are each read mostly
    (> 80%) in different zones. A mouse is in one place; one of those tags is
    probably not on it. This is how Calypso's lost tag in the B3 entrance tube
    was found in LID.

``stationary_tags``
    Tag-days on which a tag is read by a single antenna in most hours of the
    day. With no entries that looks like a body or a dropped tag under a floor
    antenna; with entries on the one antenna, like a tag lying at an
    entrance. Resting animals can trip it too, which is why it is a
    candidates list and not a rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .bout_detector import gaps
from .cleaning import DROP_RULES

CUT_RULES = ("lost_tag", "found_dead", "after_stop", "before_start")


def _animal_info(animals: pd.DataFrame) -> pd.DataFrame:
    return animals.set_index("animal")


def cross_paddock(identified: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Reads made by another paddock's reader, per animal."""
    cols = ["animal", "home_paddock", "reads", "zone_ids", "days", "first", "last",
            "policy"]
    if identified.empty:
        return pd.DataFrame(columns=cols)
    away = identified[identified["away"].astype(bool)]
    if away.empty:
        return pd.DataFrame(columns=cols)
    out = (away.assign(animal=away["animal"].astype(str),
                       zone_id=away["zone_id"].astype(str), home=away["home"].astype(str))
               .groupby("animal")
               .agg(home_paddock=("home", "first"), reads=("t", "size"),
                    zone_ids=("zone_id", lambda s: ", ".join(sorted(set(s)))),
                    days=("day", "nunique"), first=("t", "min"), last=("t", "max"))
               .reset_index())
    out["policy"] = policy
    return out[cols].sort_values("reads", ascending=False).reset_index(drop=True)


def animal_roster(animals: pd.DataFrame, identified: pd.DataFrame,
                  reads: pd.DataFrame, last_entries: pd.DataFrame,
                  home: dict) -> pd.DataFrame:
    """Every animal in the metadata: reads, days, first/last vs release/trap,
    and what each rule removed. Animals never read have 0 reads."""
    a = animals.copy()
    a["home_paddock"] = a["animal"].map(home).fillna("")
    ident = pd.DataFrame({"animal": identified["animal"].astype(str),
                          "t": identified["t"], "drop": identified["drop"].astype(str)})
    a["identified_reads"] = a["animal"].map(ident.groupby("animal").size()).fillna(0).astype(int)
    per_rule = ident[ident["drop"] != ""].groupby(["animal", "drop"]).size().unstack(fill_value=0)
    for rule in ("cross_paddock", "before_start", "after_stop", "lost_tag", "found_dead"):
        col = per_rule[rule] if rule in per_rule.columns else pd.Series(dtype=int)
        a[f"dropped_{rule}"] = a["animal"].map(col).fillna(0).astype(int)

    r = pd.DataFrame({"animal": reads["animal"].astype(str), "t": reads["t"],
                      "day": reads["day"], "away": reads["away"].astype(bool)})
    g = r.groupby("animal").agg(reads=("t", "size"), away_reads=("away", "sum"),
                                days=("day", "nunique"), first=("t", "min"),
                                last=("t", "max"))
    a = a.merge(g, left_on="animal", right_index=True, how="left")
    a["reads"] = a["reads"].fillna(0).astype(int)
    a["away_reads"] = a["away_reads"].fillna(0).astype(int)
    a["days"] = a["days"].fillna(0).astype(int)
    a["never_read"] = a["identified_reads"] == 0

    # reads outside release..trap, whether or not a trim removed them
    rel = pd.to_datetime(ident["animal"].map(dict(zip(a["animal"], a["release"]))))
    trp = pd.to_datetime(ident["animal"].map(dict(zip(a["animal"], a["trap"]))))
    before = ident[rel.notna() & (ident["t"] < rel)].groupby("animal").size()
    after = ident[trp.notna() & (ident["t"] > trp)].groupby("animal").size()
    a["reads_before_release"] = a["animal"].map(before).fillna(0).astype(int)
    a["reads_after_trap"] = a["animal"].map(after).fillna(0).astype(int)
    a["first_minus_release_h"] = (a["first"] - a["release"]).dt.total_seconds() / 3600
    a["last_minus_trap_h"] = (a["last"] - a["trap"]).dt.total_seconds() / 3600
    if not last_entries.empty:
        le = last_entries.set_index("animal")
        a["found_dead_rule"] = a["animal"].map(le["rule"]).fillna("")
        a["last_entry"] = a["animal"].map(le["last_entry"])
        a["last_entry_zone"] = a["animal"].map(le["last_entry_zone"]).fillna("")
    else:
        a["found_dead_rule"] = ""
        a["last_entry"] = pd.NaT
        a["last_entry_zone"] = ""
    cols = ["animal", "name", "trial", "home_paddock", "sex", "phase", "code", "tags",
            "release", "trap", "identified_reads", "reads", "away_reads", "days",
            "first", "last", "first_minus_release_h", "last_minus_trap_h",
            "reads_before_release", "reads_after_trap", "never_read",
            "dropped_cross_paddock", "dropped_before_start", "dropped_after_stop",
            "dropped_lost_tag", "dropped_found_dead", "found_dead_rule",
            "last_entry", "last_entry_zone"]
    return a[cols].sort_values(["trial", "phase", "sex", "name"]).reset_index(drop=True)


def two_places(identified: pd.DataFrame, share: float = 0.8,
               first_day: int = 1) -> pd.DataFrame:
    """Animal-days whose tags each live mostly in a different zone."""
    cols = ["animal", "day", "tag_a", "zone_a", "share_a", "reads_a", "tag_b",
            "zone_b", "share_b", "reads_b", "cut_reads"]
    if identified.empty:
        return pd.DataFrame(columns=cols)
    d = identified[(identified["day"] >= first_day)
                   & (identified["drop"].astype(str) != "cross_paddock")]
    f = pd.DataFrame({"animal": d["animal"].astype(str), "tag": d["tag"].astype(str),
                      "day": d["day"], "zone_id": d["zone_id"].astype(str),
                      "cut": d["drop"].astype(str).isin(CUT_RULES)})
    n = f.groupby(["animal", "tag", "day", "zone_id"]).agg(
        n=("cut", "size"), cut=("cut", "sum")).reset_index()
    n["tag_total"] = n.groupby(["animal", "tag", "day"])["n"].transform("sum")
    n["share"] = n["n"] / n["tag_total"]
    n = n.sort_values(["animal", "tag", "day", "n"], ascending=[True, True, True, False],
                      kind="mergesort")
    top = n.groupby(["animal", "tag", "day"]).head(1)
    top = top[top["share"] > share]
    zones = top.groupby(["animal", "day"])["zone_id"].nunique()
    flagged = zones[zones >= 2].index
    if not len(flagged):
        return pd.DataFrame(columns=cols)
    top = top.set_index(["animal", "day"]).loc[flagged].reset_index()
    top = top.sort_values(["animal", "day", "tag_total"], ascending=[True, True, False])
    cut = f[f["cut"]].groupby(["animal", "day"]).size()
    rows = []
    for (animal, day), g in top.groupby(["animal", "day"], sort=True):
        first = g.iloc[0]
        other = g[g["zone_id"] != first["zone_id"]].iloc[0]
        rows.append({"animal": animal, "day": int(day),
                     "tag_a": first["tag"], "zone_a": first["zone_id"],
                     "share_a": first["share"], "reads_a": int(first["tag_total"]),
                     "tag_b": other["tag"], "zone_b": other["zone_id"],
                     "share_b": other["share"], "reads_b": int(other["tag_total"]),
                     "cut_reads": int(cut.get((animal, day), 0))})
    return pd.DataFrame(rows, columns=cols)


def stationary_tags(identified: pd.DataFrame, entrance: list[str], gap_s: float,
                    hour_fraction: float = 0.5, first_day: int = 1) -> pd.DataFrame:
    """Tag-days read by one antenna in most hours of the day."""
    cols = ["animal", "tag", "day", "zone_id", "reader", "antenna", "location",
            "reads", "hours_read", "entries", "cut_reads", "status"]
    if identified.empty:
        return pd.DataFrame(columns=cols)
    d = identified[identified["drop"].astype(str) != "cross_paddock"]
    tag_codes = d["tag"].astype(str).to_numpy()
    order = np.lexsort((d["elapsed"].to_numpy(), tag_codes))
    s = d.iloc[order]
    tag = s["tag"].astype(str).to_numpy()
    el = s["elapsed"].to_numpy()
    new_tag = np.r_[True, tag[1:] != tag[:-1]]
    gap = np.where(new_tag, np.inf, np.r_[np.inf, gaps(el)])
    is_entrance = s["location"].astype(str).str.lower().isin(
        {x.strip().lower() for x in entrance}).to_numpy()
    f = pd.DataFrame({"animal": s["animal"].astype(str).to_numpy(), "tag": tag,
                      "day": s["day"].to_numpy(),
                      "zone_id": s["zone_id"].astype(str).to_numpy(),
                      "reader": s["reader"].astype(str).to_numpy(),
                      "antenna": s["antenna"].astype(str).to_numpy(),
                      "location": s["location"].astype(str).to_numpy(),
                      "hour": s["t"].dt.floor("h").to_numpy(),
                      "entry": is_entrance & (gap > gap_s),
                      "cut": s["drop"].astype(str).isin(CUT_RULES).to_numpy()})
    f = f[f["day"] >= first_day]
    f["ant_key"] = f["reader"] + ":" + f["antenna"]
    g = f.groupby(["animal", "tag", "day"]).agg(
        n_antennas=("ant_key", "nunique"), reads=("hour", "size"),
        hours_read=("hour", "nunique"), entries=("entry", "sum"),
        cut_reads=("cut", "sum"), zone_id=("zone_id", "first"),
        reader=("reader", "first"), antenna=("antenna", "first"),
        location=("location", "first")).reset_index()
    cand = g[(g["n_antennas"] == 1) & (g["hours_read"] > hour_fraction * 24)].copy()
    cand["status"] = np.where(cand["entries"] == 0,
                              "no entries: body or dropped tag?",
                              "entries on its only antenna: tag lying at an entrance?")
    cand["entries"] = cand["entries"].astype(int)
    cand["cut_reads"] = cand["cut_reads"].astype(int)
    return cand[cols].sort_values(["animal", "tag", "day"]).reset_index(drop=True)


def dropped_summary(dropped: dict, identified: pd.DataFrame) -> pd.DataFrame:
    """Reads removed by each rule, with the animals most affected."""
    per_animal = (identified[identified["drop"].astype(str) != ""]
                  .assign(animal=lambda x: x["animal"].astype(str),
                          drop=lambda x: x["drop"].astype(str))
                  .groupby(["drop", "animal"]).size())
    rows = []
    for rule in DROP_RULES:
        n = int(dropped.get(rule, 0))
        detail = ""
        if rule in per_animal.index.get_level_values(0):
            top = per_animal.loc[rule].sort_values(ascending=False)
            detail = ", ".join(f"{a} {c:,}" for a, c in top.head(8).items())
            if len(top) > 8:
                detail += f", ... ({len(top)} animals)"
        rows.append({"rule": rule, "reads": n, "detail": detail})
    return pd.DataFrame(rows)
