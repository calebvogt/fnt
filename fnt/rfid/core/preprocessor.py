"""Stage 1: raw reader exports -> a reads table with identity, place and time.

Every later stage is a reshaping of this table, so the things that go wrong
here go wrong everywhere: a tag that matches no animal, an antenna the arena
does not describe, a read from the wrong paddock's reader. The R pipeline
dropped all three silently through an inner join. Here they are counted and
handed back in :class:`ReadsResult` so the GUI can show them.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config.defaults import TrialConfig
from .file_readers import read_download_dir

# Columns of the reads table, in export order.
READS_SCHEMA = ["trial", "name", "code", "sex", "phase", "group", "strain",
                "reader_id", "antenna_id", "ant_loc", "zone", "zone_x", "zone_y",
                "scan_date", "scan_time", "tag_id", "field_time", "noon_day",
                "time_sec"]


@dataclass
class ReadsResult:
    """The reads table plus what had to be discarded to build it."""
    reads: pd.DataFrame
    file_report: list[dict] = field(default_factory=list)
    raw_read_count: int = 0
    unique_read_count: int = 0
    unmatched_tags: dict[str, int] = field(default_factory=dict)
    unread_tags: list[str] = field(default_factory=list)
    animals_without_reads: list[str] = field(default_factory=list)
    unmapped_antennas: list[int] = field(default_factory=list)
    silent_antennas: list[int] = field(default_factory=list)
    foreign_reader_counts: dict[int, int] = field(default_factory=dict)
    foreign_reads_dropped: int = 0

    def warnings(self) -> list[str]:
        """Human-readable problems worth surfacing before an export."""
        out = []
        if self.unmatched_tags:
            n = sum(self.unmatched_tags.values())
            out.append(f"{len(self.unmatched_tags)} tag(s) read that are not in "
                       f"metadata ({n:,} reads discarded)")
        if self.animals_without_reads:
            out.append("animals with zero reads: "
                       + ", ".join(self.animals_without_reads))
        if self.unmapped_antennas:
            out.append(f"antenna ID(s) not described by the arena: "
                       f"{self.unmapped_antennas} - their reads were discarded")
        if self.silent_antennas:
            out.append(f"antenna ID(s) in the arena with no reads at all: "
                       f"{self.silent_antennas}")
        if self.foreign_reader_counts:
            verb = "dropped" if self.foreign_reads_dropped else "kept"
            out.append(f"reads from other trials' readers ({verb}): "
                       + ", ".join(f"reader {k}: {v:,}"
                                   for k, v in sorted(self.foreign_reader_counts.items())))
        return out


def pad_tag(value, digits: int = 15) -> str | float:
    """Restore a tag ID that was stored as a number.

    ``985.113004548310`` written by anything that treats it as a float comes
    back as ``985.11300454831``: the trailing zero is not significant to a
    float but it IS significant to a tag. Pad back out to ``digits``
    significant figures rather than trusting whatever wrote the file.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in ("nan", "na", "none"):
        return np.nan
    if "." not in text:
        return text
    whole, frac = text.split(".", 1)
    want = digits - len(whole)
    return f"{whole}.{frac.ljust(want, '0')}" if want > len(frac) else text


def load_metadata(path: str, tag_columns: list[str], digits: int = 15) -> pd.DataFrame:
    """Read the animal metadata table.

    Everything arrives as text (see :func:`pad_tag`), and header whitespace is
    stripped because these files are usually exported from a spreadsheet whose
    first column header has drifted.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        meta = pd.read_excel(path, dtype=str)
    else:
        meta = pd.read_csv(path, dtype=str, skipinitialspace=True)
    meta.columns = [str(c).strip() for c in meta.columns]

    # 'genotype' is this project's word for what the pipeline calls 'strain'.
    if "strain" not in meta.columns and "genotype" in meta.columns:
        meta["strain"] = meta["genotype"]

    required = ["name", "sex"] + [c for c in tag_columns if c in meta.columns]
    missing = [c for c in ("name", "sex") if c not in meta.columns]
    if missing:
        raise ValueError(f"Metadata is missing required column(s): {missing}")
    if not any(c in meta.columns for c in tag_columns):
        raise ValueError(f"Metadata has none of the tag columns {tag_columns}")

    for col in tag_columns:
        if col in meta.columns:
            meta[col] = meta[col].map(lambda v: pad_tag(v, digits))
    for col in ("name", "code", "sex", "phase", "group", "strain", "trial"):
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.strip()
    return meta


def tag_lookup(meta: pd.DataFrame, tag_columns: list[str]) -> pd.DataFrame:
    """Long table of one row per (tag, animal).

    Built by stacking the tag columns rather than merging on each in turn, so
    an animal wearing two tags contributes two rows and a read matches exactly
    one of them.
    """
    keep = [c for c in ("trial", "name", "code", "sex", "phase", "group", "strain")
            if c in meta.columns]
    frames = []
    for col in tag_columns:
        if col not in meta.columns:
            continue
        part = meta.loc[meta[col].notna(), keep + [col]].copy()
        part = part.rename(columns={col: "tag_id"})
        part["tag_column"] = col
        frames.append(part)
    if not frames:
        raise ValueError("No usable tag columns in metadata")
    out = pd.concat(frames, ignore_index=True)
    dupes = out[out.duplicated("tag_id", keep=False)]
    if not dupes.empty:
        pairs = dupes.groupby("tag_id")["name"].apply(lambda s: sorted(set(s)))
        clashes = {t: n for t, n in pairs.items() if len(n) > 1}
        if clashes:
            raise ValueError(f"Tag(s) assigned to more than one animal: {clashes}")
    return out.drop_duplicates("tag_id").reset_index(drop=True)


class RFIDPreprocessor:
    """Builds a trial's reads table from its raw exports."""

    def __init__(self, config: TrialConfig):
        self.config = config
        self.metadata: pd.DataFrame | None = None

    def load_metadata(self) -> pd.DataFrame:
        self.metadata = load_metadata(self.config.metadata_path,
                                      self.config.tag_columns,
                                      self.config.tag_digits)
        return self.metadata

    def run(self, progress=None) -> ReadsResult:
        cfg = self.config
        say = progress or (lambda _m: None)

        if self.metadata is None:
            say("Loading metadata...")
            self.load_metadata()
        meta = self.metadata
        if cfg.trial_id and "trial" in meta.columns:
            trial_meta = meta[meta["trial"] == cfg.trial_id]
            if not trial_meta.empty:
                meta = trial_meta

        say(f"Reading exports from {cfg.raw_dir}...")
        raw, file_report = read_download_dir(cfg.raw_dir)
        result = ReadsResult(reads=pd.DataFrame(), file_report=file_report,
                             raw_read_count=sum(e["rows"] for e in file_report),
                             unique_read_count=len(raw))
        say(f"{result.raw_read_count:,} rows across {len(file_report)} file(s) "
            f"-> {len(raw):,} unique reads")

        # --- identity -------------------------------------------------------
        lookup = tag_lookup(meta, cfg.tag_columns)
        merged = raw.merge(lookup.drop(columns=["tag_column"]), on="tag_id",
                           how="left")
        unmatched = merged[merged["name"].isna()]
        if not unmatched.empty:
            result.unmatched_tags = (unmatched["tag_id"].value_counts()
                                     .astype(int).to_dict())
        df = merged[merged["name"].notna()].copy()

        seen_tags = set(raw["tag_id"])
        result.unread_tags = sorted(set(lookup["tag_id"]) - seen_tags)
        result.animals_without_reads = sorted(set(meta["name"]) - set(df["name"]))

        # --- place ----------------------------------------------------------
        df["antenna_id"] = pd.to_numeric(df["antenna_id"], errors="coerce").astype("Int64")
        df["reader_id"] = pd.to_numeric(df["reader_id"], errors="coerce").astype("Int64")
        seen_antennas = df["antenna_id"].dropna().astype(int).unique().tolist()
        result.unmapped_antennas = cfg.arena.unmapped_antennas(seen_antennas)
        result.silent_antennas = cfg.arena.silent_antennas(seen_antennas)

        zone_of = cfg.arena.antenna_zone_map()
        loc_of = cfg.arena.antenna_location_map()
        xy = cfg.arena.zone_xy()
        df["zone"] = df["antenna_id"].map(zone_of).astype("Int64")
        df = df[df["zone"].notna()].copy()
        df["ant_loc"] = df["antenna_id"].map(loc_of).fillna("")
        df["zone_x"] = df["zone"].map(lambda z: xy.get(int(z), (np.nan, np.nan))[0])
        df["zone_y"] = df["zone"].map(lambda z: xy.get(int(z), (np.nan, np.nan))[1])

        # --- foreign readers -------------------------------------------------
        if cfg.reader_ids:
            foreign = df[~df["reader_id"].isin(cfg.reader_ids)]
            if not foreign.empty:
                result.foreign_reader_counts = (foreign["reader_id"].value_counts()
                                                .astype(int).to_dict())
                if cfg.foreign_reader_policy == "drop":
                    result.foreign_reads_dropped = len(foreign)
                    df = df[df["reader_id"].isin(cfg.reader_ids)].copy()

        # --- time ------------------------------------------------------------
        df["field_time"] = pd.to_datetime(
            df["scan_date"] + " " + df["scan_time"], format="mixed", errors="coerce")
        bad = int(df["field_time"].isna().sum())
        if bad:
            say(f"WARNING: {bad:,} read(s) had an unparseable timestamp; dropped")
            df = df[df["field_time"].notna()].copy()
        if cfg.time_resolution == "s":
            df["field_time"] = df["field_time"].dt.floor("s")

        df = df.sort_values("field_time", kind="stable").reset_index(drop=True)
        if df.empty:
            result.reads = df
            return result

        origin = pd.Timestamp(f"{df['field_time'].iloc[0].date()} "
                              f"{cfg.day_origin_time}")
        elapsed = (df["field_time"] - origin).dt.total_seconds() / 86400.0
        df["noon_day"] = np.ceil(elapsed).astype(int)
        df["time_sec"] = ((df["field_time"] - df["field_time"].min())
                          .dt.total_seconds() + 1)

        if cfg.trial_id:
            df["trial"] = cfg.trial_id
        for col in ("code", "phase", "group", "strain"):
            if col not in df.columns:
                df[col] = ""
        result.reads = df[[c for c in READS_SCHEMA if c in df.columns]]
        say(f"{len(result.reads):,} reads for {result.reads['name'].nunique()} animals")
        return result
