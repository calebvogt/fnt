"""Who holds each zone, per day.

Ownership is scored from READS rather than bouts: a male sitting on a zone's
antenna generates reads continuously, so his share of a zone-day's reads is a
direct measure of how much of that zone's time he took up. Using bouts instead
would weight a brief visit the same as an all-night occupation once they were
both collapsed to one row.

Males only, by default. The territory question this answers is about male
competition for resource zones; including females makes the denominator a
different quantity and every percentage means something else.
"""

from __future__ import annotations

import pandas as pd

OWNERSHIP_SCHEMA = ["trial", "strain", "name", "code", "zone", "noon_day",
                    "total_zone_reads", "mus_zone_reads", "mus_perc_zone_reads",
                    "rank_order", "trial_zone_day"]


def zone_ownership(reads: pd.DataFrame, sexes: tuple[str, ...] = ("M",),
                   ) -> pd.DataFrame:
    """Per trial x zone x day, each animal's share of that zone's reads.

    ``rank_order`` uses average ranking for ties, so two animals level at the
    top both score 1.5 and neither is reported as the owner. Breaking the tie
    arbitrarily would invent an owner where the data does not support one.
    """
    if reads.empty:
        return pd.DataFrame(columns=OWNERSHIP_SCHEMA)

    df = reads[reads["sex"].isin(sexes)] if sexes else reads
    if df.empty:
        return pd.DataFrame(columns=OWNERSHIP_SCHEMA)

    keys = ["trial", "zone", "noon_day"]
    per_animal = (df.groupby(keys + ["name"], observed=True)
                    .size().rename("mus_zone_reads").reset_index())
    per_zone = (df.groupby(keys, observed=True)
                  .size().rename("total_zone_reads").reset_index())
    out = per_animal.merge(per_zone, on=keys)
    out["mus_perc_zone_reads"] = (out["mus_zone_reads"]
                                  / out["total_zone_reads"] * 100)
    out["rank_order"] = (out.groupby(keys)["mus_zone_reads"]
                            .rank(method="average", ascending=False))
    out["trial_zone_day"] = (out["trial"].astype(str) + "_"
                             + out["zone"].astype(str) + "_"
                             + out["noon_day"].astype(str))

    extra = [c for c in ("strain", "code") if c in reads.columns]
    if extra:
        info = reads.drop_duplicates("name")[["name"] + extra]
        out = out.merge(info, on="name", how="left")
    for col in OWNERSHIP_SCHEMA:
        if col not in out.columns:
            out[col] = ""
    return (out[OWNERSHIP_SCHEMA]
            .sort_values(["trial", "zone", "noon_day", "rank_order"])
            .reset_index(drop=True))


def daily_owners(ownership: pd.DataFrame, min_percent: float = 50.0
                 ) -> pd.DataFrame:
    """The single owner of each zone-day, where one animal clears the bar.

    ``min_percent`` is what separates an owner from the busiest of several
    visitors. A zone whose top male holds 30% of its reads has no owner.
    """
    if ownership.empty:
        return ownership
    top = ownership[(ownership["rank_order"] == 1)
                    & (ownership["mus_perc_zone_reads"] > min_percent)]
    return top.reset_index(drop=True)


def zones_owned_per_day(ownership: pd.DataFrame, min_percent: float = 50.0,
                        days: tuple[int, int] | None = None) -> pd.DataFrame:
    """How many zones each animal owned on each day."""
    top = daily_owners(ownership, min_percent)
    if days is not None:
        top = top[top["noon_day"].between(days[0], days[1])]
    if top.empty:
        return pd.DataFrame(columns=["name", "noon_day", "zones_owned"])
    return (top.groupby(["name", "noon_day"]).size()
               .rename("zones_owned").reset_index()
               .sort_values(["noon_day", "name"]).reset_index(drop=True))
