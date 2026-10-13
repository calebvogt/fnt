"""Raw reads -> cleaned reads: who, where, when, and what to throw away.

Every later table is a reshaping of this one, so what goes wrong here goes
wrong everywhere. The rules, in the order they are applied:

1. **Identity.** A read's tag is matched to an animal (DEC tag first, HEX as a
   fallback). Reads of tags no animal wears - test tags, strays, a
   neighbouring study - are dropped and listed in the unknown-tags report.
2. **Place.** ``(reader, antenna)`` -> paddock, zone, zone_id, location. A
   pair the arena does not describe is dropped and listed; it usually means
   the antenna table is incomplete, not that the read is wrong.
3. **Cross-paddock reads.** A read by another paddock's reader is a real visit
   (LID had a hole in the shared wall). ``keep`` (default) keeps it in the
   paddock that read it, flagged ``away``; ``drop`` discards it; ``fold``
   reproduces the original pipeline, which relabelled it as the same zone
   number in the animal's own paddock.
4. **Per-animal windows.** Optional trims to release / trap (metadata) or to
   manual times.
5. **Lost tags.** A tag known to be off its animal from a time (to a time):
   its reads inside are dropped. A tag lying in an entrance tube keeps being
   read every hour and otherwise places the animal there.
6. **Found dead.** A body lying in or under a zone is read like a resident
   that never leaves - in LID, Isac's body under B5 produced 75,693 reads
   over a week and would have "held" B5 for days. The ``last_entry`` rule cuts
   the animal at its last entry plus one bout gap, an entry being a read on
   an entrance-type antenna (``wall``) after a gap longer than the bout gap.
   A body cannot make an entry; a mouse cannot help making them.

Nothing is dropped silently: every discarded read carries a reason, and the
counts per rule and per animal go to the QC reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config.defaults import RFIDConfig
from .bout_detector import gaps
from .file_readers import normalise_tag_value
from .metadata import AnimalTable

READ_COLUMNS = ["animal", "tag", "reader", "paddock", "antenna", "location",
                "zone", "zone_id", "away", "day", "t"]

# drop reasons, in the order the rules run
DROP_RULES = ["unknown_tag", "unmapped_antenna", "bad_timestamp", "cross_paddock",
              "before_start", "after_stop", "lost_tag", "found_dead"]

DAY = np.timedelta64(86_400_000_000_000, "ns")


@dataclass
class CleanResult:
    """Cleaned reads plus the record of everything that was removed.

    ``identified`` holds every read that could be tied to an animal and a zone,
    each with its ``drop`` reason ('' if kept). The QC reports that look for
    problems a person should decide about (two tags in two places, a tag lying
    still) run on it, so they still see what a configured cut removed.
    """
    reads: pd.DataFrame
    identified: pd.DataFrame
    unknown_tags: pd.DataFrame
    unmapped: pd.DataFrame
    last_entries: pd.DataFrame
    origin: pd.Timestamp
    dropped: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #

def elapsed_seconds(t: pd.Series, tz: str = "") -> tuple[np.ndarray, int]:
    """Seconds on a clock that does not jump, and how many reads were ambiguous.

    Only differences of these values mean anything (gaps, durations); zero is
    the earliest read.

    Readers log local wall-clock time. Across a DST change that clock jumps an
    hour, so a gap measured on it is wrong by an hour; with a timezone set, the
    times are localised and measured in UTC instead. Reads in the repeated
    autumn hour cannot be placed from the clock alone and are taken as
    daylight time (and counted).
    """
    if not len(t):
        return np.array([], float), 0
    # asi8 counts in the index's own unit, which pandas 3 makes microseconds
    # for parsed times; pin nanoseconds so the /1e9 below gives seconds
    values = pd.DatetimeIndex(t).as_unit("ns")
    if not tz:
        ns, ambiguous = values.asi8, 0
    else:
        as_dst = values.tz_localize(tz, ambiguous=np.ones(len(values), bool),
                                    nonexistent="shift_forward")
        as_std = values.tz_localize(tz, ambiguous=np.zeros(len(values), bool),
                                    nonexistent="shift_forward")
        ambiguous = int((as_dst.asi8 != as_std.asi8).sum())
        ns = as_dst.asi8
    # relative to the earliest read: at epoch magnitudes (1.6e9 s) a float64
    # carries ~1e-7 s of noise into every difference; from zero it does not
    return (ns - ns.min()) / 1e9, ambiguous


def resolve_origin(config: RFIDConfig, animals: AnimalTable | None,
                   first_read) -> pd.Timestamp:
    """Start of day 1 (local wall clock).

    Explicit origin if set; otherwise ``day_start_time`` (noon) on the date of
    the earliest release in the metadata, else of the first read. The first
    read is a poor default on its own: readers are tested with tags before
    animals go in (LID's first read is the day before release).
    """
    if config.day_origin:
        return pd.Timestamp(config.day_origin)
    anchor = None
    if animals is not None and animals.animals["release"].notna().any():
        anchor = animals.animals["release"].min()
    elif first_read is not None and not pd.isna(first_read):
        anchor = pd.Timestamp(first_read)
    if anchor is None:
        return pd.NaT
    return pd.Timestamp(f"{anchor.date()} {config.day_start_time or '12:00:00'}")


def day_index(t: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    """Noon-to-noon day number: day 1 is (origin, origin + 24 h].

    ``ceiling`` of elapsed days, as the original pipeline and the R reference
    do, so a read exactly at the origin is the last moment of day 0. Measured
    on the wall clock, so "noon" stays local noon across a DST change.
    """
    if pd.isna(origin):
        return np.zeros(len(t), dtype=int)
    delta = (t.to_numpy() - origin.to_datetime64()) / DAY
    return np.ceil(delta).astype(int)


def parse_when(text) -> pd.Timestamp:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return pd.NaT
    if isinstance(text, pd.Timestamp):
        return text
    s = str(text).strip()
    return pd.to_datetime(s, errors="coerce") if s else pd.NaT


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #

def _numeric_order(values: pd.Series) -> pd.Series:
    """Sort key for antenna IDs: numeric where possible, so 9 < 10."""
    cats = pd.Series(pd.unique(values.astype(str)))
    as_num = pd.to_numeric(cats, errors="coerce")
    order = {c: (n if not np.isnan(n) else 1e12 + i)
             for i, (c, n) in enumerate(zip(cats, as_num))}
    return values.astype(str).map(order).astype(float)


def clean_reads(raw: pd.DataFrame, animals: AnimalTable, config: RFIDConfig,
                progress=None) -> CleanResult:
    """Apply identity, place and every cleaning rule. See the module docstring."""
    say = progress or (lambda _m: None)
    warnings: list[str] = []
    dropped: dict[str, int] = {r: 0 for r in DROP_RULES}
    arena = config.arena

    # --- 1. identity -----------------------------------------------------
    say("Matching tags to animals...")
    tag_map = animals.tag_to_animal()
    tag = raw["tag"].astype(str)
    animal = tag.map(tag_map)
    via_hex = animal.isna() & raw["tag_hex"].astype(str).isin(tag_map.keys())
    if via_hex.any():
        animal = animal.where(~via_hex, raw["tag_hex"].astype(str).map(tag_map))
        tag = tag.where(~via_hex, raw["tag_hex"].astype(str))
    unknown_mask = animal.isna().to_numpy()
    unknown = raw.loc[unknown_mask]
    if len(unknown):
        u = unknown.assign(tag=unknown["tag"].astype(str).where(
            unknown["tag"].astype(str) != "", unknown["tag_hex"].astype(str)))
        unknown_tags = (u.groupby("tag", observed=True)
                          .agg(reads=("t", "size"), first=("t", "min"), last=("t", "max"),
                               readers=("reader", lambda s: ",".join(sorted(set(map(str, s))))),
                               antennas=("antenna", "nunique"))
                          .reset_index().sort_values("reads", ascending=False)
                          .reset_index(drop=True))
    else:
        unknown_tags = pd.DataFrame(columns=["tag", "reads", "first", "last",
                                             "readers", "antennas"])
    dropped["unknown_tag"] = int(unknown_mask.sum())
    say(f"{dropped['unknown_tag']:,} reads of {len(unknown_tags)} unknown tag(s) dropped")

    df = pd.DataFrame({"animal": animal[~unknown_mask].to_numpy(),
                       "tag": tag[~unknown_mask].to_numpy(),
                       "reader": raw["reader"].astype(str)[~unknown_mask].to_numpy(),
                       "antenna": raw["antenna"].astype(str)[~unknown_mask].to_numpy(),
                       "t": raw["t"][~unknown_mask].to_numpy()})

    # --- 2. place ----------------------------------------------------------
    say("Mapping (reader, antenna) to zones...")
    ant = pd.DataFrame(arena.antenna_rows())
    if ant.empty:
        ant = pd.DataFrame(columns=["reader", "antenna", "paddock", "zone", "zone_id",
                                    "location", "zone_x", "zone_y"])
    ant = ant.drop_duplicates(["reader", "antenna"])[
        ["reader", "antenna", "paddock", "zone", "zone_id", "location"]]
    df = df.merge(ant, on=["reader", "antenna"], how="left")
    unmapped_mask = (df["zone_id"].isna() | (df["zone_id"] == "")).to_numpy()
    if unmapped_mask.any():
        um = df[unmapped_mask]
        unmapped = (um.groupby(["reader", "antenna"])
                      .agg(reads=("t", "size"), first=("t", "min"), last=("t", "max"),
                           animals=("animal", "nunique"))
                      .reset_index())
        warnings.append(f"{int(unmapped_mask.sum()):,} reads on {len(unmapped)} "
                        f"(reader, antenna) pair(s) the arena does not describe were "
                        f"dropped: " + ", ".join(f"reader {r} antenna {a}"
                                                 for r, a in zip(unmapped["reader"],
                                                                 unmapped["antenna"])))
    else:
        unmapped = pd.DataFrame(columns=["reader", "antenna", "reads", "first",
                                         "last", "animals"])
    dropped["unmapped_antenna"] = int(unmapped_mask.sum())
    df = df[~unmapped_mask].reset_index(drop=True)

    bad_time = df["t"].isna().to_numpy()
    dropped["bad_timestamp"] = int(bad_time.sum())
    df = df[~bad_time].reset_index(drop=True)

    # --- 3. home paddock and cross-paddock reads -----------------------------
    home = {a: arena.home_paddock(tr) for a, tr in
            zip(animals.animals["animal"], animals.animals["trial"])}
    no_home = sorted(a for a, h in home.items() if not h)
    if no_home and len(arena.paddocks) > 1:
        warnings.append(f"{len(no_home)} animal(s) match no paddock by their trial "
                        f"value, so their reads are never flagged away: "
                        + ", ".join(no_home[:8]) + ("..." if len(no_home) > 8 else ""))
    df["home"] = df["animal"].map(home).fillna("")
    df["away"] = (df["home"] != "") & (df["paddock"] != df["home"])
    df["drop"] = ""
    if config.cross_paddock_policy == "drop":
        df.loc[df["away"], "drop"] = "cross_paddock"
    elif config.cross_paddock_policy == "fold":
        # the legacy relabelling: same zone label, the animal's own paddock
        fold = df["away"].to_numpy()
        df.loc[fold, "paddock"] = df.loc[fold, "home"]
        df.loc[fold, "zone_id"] = df.loc[fold, "home"] + df.loc[fold, "zone"]

    # --- time --------------------------------------------------------------
    say("Sorting reads...")
    df["_ant"] = _numeric_order(df["antenna"])
    # (animal, t) is the order bouts are built in; ties (two antennas at the
    # same millisecond) break by antenna then tag, as the R reference does.
    df = df.sort_values(["animal", "t", "_ant", "tag"], kind="mergesort").reset_index(drop=True)
    df = df.drop(columns="_ant")
    df["elapsed"], ambiguous = elapsed_seconds(df["t"], config.timezone)
    if ambiguous:
        warnings.append(f"{ambiguous:,} reads fall in the repeated hour of an autumn "
                        f"DST change and were taken as daylight time")
    origin = resolve_origin(config, animals, df["t"].min() if len(df) else None)
    df["day"] = day_index(df["t"], origin)

    # --- 4. per-animal windows ---------------------------------------------
    release = animals.info("release")
    trap = animals.info("trap")
    starts, stops = {}, {}
    for a in animals.animals["animal"]:
        s = config.animal(a)
        if s["start_mode"] == "manual" and s["start_time"]:
            starts[a] = parse_when(s["start_time"])
        elif config.trim_to_release and not pd.isna(release.get(a)):
            starts[a] = release[a]
        if s["stop_mode"] == "manual" and s["stop_time"]:
            stops[a] = parse_when(s["stop_time"])
        elif config.trim_to_trap and not pd.isna(trap.get(a)):
            stops[a] = trap[a]
    keep = df["drop"] == ""
    if starts:
        start_t = pd.to_datetime(df["animal"].map(starts))
        mask = keep & start_t.notna() & (df["t"] < start_t)
        df.loc[mask, "drop"] = "before_start"
        keep &= ~mask
    if stops:
        stop_t = pd.to_datetime(df["animal"].map(stops))
        mask = keep & stop_t.notna() & (df["t"] > stop_t)
        df.loc[mask, "drop"] = "after_stop"
        keep &= ~mask

    # --- 5. lost tags ----------------------------------------------------------
    for a in animals.animals["animal"]:
        for lost in config.animal(a)["lost_tags"]:
            lt = normalise_tag_value(lost.get("tag"))
            start = parse_when(lost.get("start"))
            stop = parse_when(lost.get("stop"))
            if not lt or pd.isna(start):
                warnings.append(f"{a}: lost-tag entry without a tag or start ignored")
                continue
            mask = keep & (df["tag"] == lt) & (df["t"] >= start)
            if not pd.isna(stop):
                mask &= df["t"] < stop
            df.loc[mask, "drop"] = "lost_tag"
            keep &= ~mask
            say(f"{a}: tag {lt} off from {start}"
                + (f" to {stop}" if not pd.isna(stop) else "")
                + f" - {int(mask.sum()):,} reads dropped")

    # --- 6. found dead ---------------------------------------------------------
    entrance = {x.strip().lower() for x in config.entrance_locations if x.strip()}
    entry_rows = []
    for a in animals.animals["animal"]:
        s = config.animal(a)
        mode = s["dead_mode"]
        if mode not in ("manual", "last_entry"):
            continue
        found = parse_when(s["dead_time"])
        rows = np.flatnonzero((df["animal"] == a).to_numpy() & keep.to_numpy())
        if not len(rows):
            continue
        if mode == "manual":
            if pd.isna(found):
                warnings.append(f"{a}: found-dead time missing; nothing cut")
                continue
            cut_local = found
            last_entry, last_zone = pd.NaT, ""
            mask_idx = rows[(df["t"].to_numpy()[rows] > cut_local.to_datetime64())]
        else:
            el = df["elapsed"].to_numpy()[rows]
            gap = np.r_[np.inf, gaps(el)]
            loc = df["location"].iloc[rows].astype(str).str.lower()
            entry = loc.isin(entrance).to_numpy() & (gap > config.bout_gap_s)
            if not pd.isna(found):
                entry &= df["t"].to_numpy()[rows] <= found.to_datetime64()
            if not entry.any():
                warnings.append(f"{a}: found-dead 'last entry' rule found no entry on "
                                f"an entrance antenna ({', '.join(sorted(entrance))}); "
                                f"nothing cut")
                continue
            k = np.flatnonzero(entry)[-1]
            last_entry = df["t"].iloc[rows[k]]
            last_zone = df["zone_id"].iloc[rows[k]]
            cut_el = el[k] + config.bout_gap_s
            cut_local = last_entry + pd.Timedelta(seconds=config.bout_gap_s)
            mask_idx = rows[el > cut_el]
        df.loc[mask_idx, "drop"] = "found_dead"
        keep.iloc[mask_idx] = False
        cut_zones = sorted(set(df["zone_id"].iloc[mask_idx])) if len(mask_idx) else []
        entry_rows.append({
            "animal": a, "rule": mode, "found_time": found, "last_entry": last_entry,
            "last_entry_zone": last_zone, "cut_at": cut_local,
            "reads_dropped": int(len(mask_idx)),
            "zones_after_cut": ", ".join(cut_zones),
            "last_dropped_read": df["t"].iloc[mask_idx].max() if len(mask_idx) else pd.NaT})
        say(f"{a}: found dead ({mode}) - cut at {cut_local}, "
            f"{len(mask_idx):,} reads dropped")
    last_entries = pd.DataFrame(entry_rows, columns=[
        "animal", "rule", "found_time", "last_entry", "last_entry_zone", "cut_at",
        "reads_dropped", "zones_after_cut", "last_dropped_read"])

    for rule in ("cross_paddock", "before_start", "after_stop", "lost_tag", "found_dead"):
        dropped[rule] = int((df["drop"] == rule).sum())

    for col in ("animal", "tag", "reader", "paddock", "antenna", "location", "zone",
                "zone_id", "home", "drop"):
        df[col] = df[col].astype("category")
    reads = df.loc[keep.to_numpy(), READ_COLUMNS + ["elapsed"]].reset_index(drop=True)
    for col in ("animal", "tag", "reader", "paddock", "antenna", "location", "zone",
                "zone_id"):
        reads[col] = reads[col].cat.remove_unused_categories()
    say(f"{len(reads):,} cleaned reads for {reads['animal'].nunique()} animals")
    return CleanResult(reads=reads, identified=df, unknown_tags=unknown_tags,
                       unmapped=unmapped, last_entries=last_entries, origin=origin,
                       dropped=dropped, warnings=warnings)
