"""Run a dataset end to end and write its analysis folder.

    downloads -> reads -> cleaned reads -> bouts -> summaries (+ QC, + layers)

Nothing here is interactive. The GUI drives :func:`run` from a worker thread
and receives progress through the callback; a script or a test calls the same
function, so there is one code path whichever way a run is started.

The analysis folder is written only after every stage has succeeded, so a
failed run never leaves half a set of tables looking like a finished one.
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .config.config_manager import (ConfigManager, check_dir_usable, csv_dir,
                                    default_name, ensure_dir, output_inside_downloads,
                                    provenance, resolve_output_dir)
from .config.defaults import RFIDConfig
from .core import qc as QC
from .core.bout_detector import detect_bouts
from .core.cleaning import CleanResult, clean_reads
from .core.file_readers import DownloadResult, list_download_files, read_downloads
from .core.metadata import AnimalTable, load_animals
from .core.summaries import animal_days, co_occupancy, zone_holders

# Output tables, in the order they are written. QC tables carry a qc_ prefix.
MAIN_TABLES = ["reads", "bouts", "animal_days", "zone_holders", "co_occupancy"]
QC_TABLES = ["qc_files", "qc_unknown_tags", "qc_unmapped_antennas", "qc_animals",
             "qc_cross_paddock", "qc_two_places", "qc_stationary_tags",
             "qc_found_dead", "qc_dropped_reads"]


@dataclass
class RunResult:
    """Everything a run produced, in memory, plus where it was written."""
    config: RFIDConfig
    output_dir: str = ""
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    written: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    error: str | None = None
    cancelled: bool = False
    origin: pd.Timestamp | None = None
    animals: AnimalTable | None = None
    input_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.cancelled


def _format_times(frame: pd.DataFrame) -> pd.DataFrame:
    """Datetime columns as 'YYYY-MM-DD HH:MM:SS.mmm' text for the CSVs.

    Done with numpy's C formatter: ``strftime`` on a million-row column takes
    seconds, this takes a fraction of one.
    """
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            values = out[col].to_numpy(dtype="datetime64[ms]")
            text = np.datetime_as_string(values, unit="ms")
            text = np.char.replace(text, "T", " ")
            text[np.isnat(values)] = ""
            out[col] = text
        elif pd.api.types.is_float_dtype(out[col]):
            # float noise (15.260000000000002) is not precision anyone measured
            out[col] = out[col].round(6)
    return out


def write_table(frame: pd.DataFrame, output_dir: str, name: str, table: str) -> str:
    path = os.path.join(csv_dir(output_dir, create=True), f"{name}_{table}.csv")
    _format_times(frame).to_csv(path, index=False)
    return path


def home_paddocks(config: RFIDConfig, animals: AnimalTable) -> dict:
    return {a: config.arena.home_paddock(t)
            for a, t in zip(animals.animals["animal"], animals.animals["trial"])}


def legacy_layers(config: RFIDConfig, reads: pd.DataFrame, bouts: pd.DataFrame,
                  animals: AnimalTable, say) -> dict[str, pd.DataFrame]:
    """The optional layers carried over from the earlier module.

    They were written against ``name / zone / noon_day / field_time`` columns;
    this adapts the new tables to that schema (zone = zone_id) rather than
    rewriting layers that are opt-in.
    """
    from .core.gbi_generator import create_gbi
    tables: dict[str, pd.DataFrame] = {}
    layers = config.layers
    if not any(layers.values()) or bouts.empty:
        return tables
    trial = config.name or "rfid"
    info = animals.animals.set_index("animal")
    names = sorted(bouts["animal"].astype(str).unique())
    sex_by_name = {a: info["sex"].get(a, "") for a in names}
    lb = pd.DataFrame({
        "trial": trial, "name": bouts["animal"].astype(str),
        "sex": bouts["sex"], "zone": bouts["zone_id"].astype(str),
        "noon_day": bouts["day"], "field_time": bouts["start"],
        "field_time_stop": bouts["stop"], "duration_s": bouts["duration_s"]})
    say("Optional layers: building the GBI...")
    gbi = create_gbi(lb, names, sex_by_name, config.min_bout_s, say)
    if layers.get("gbi"):
        tables["gbi"] = gbi
    males = [a for a in names if sex_by_name.get(a) == "M"]
    if layers.get("edgelist"):
        from .core.edgelist import co_presence_bouts, edgelist
        pair_bouts = co_presence_bouts(gbi, names, say)
        tables["co_presence_bouts"] = pair_bouts
        tables["edgelist"] = edgelist(pair_bouts)
    if layers.get("displacement"):
        from .core.displacement import annotate_ownership, detect_displacements
        from .core.zone_ownership import zone_ownership
        lr = pd.DataFrame({"trial": trial, "name": reads["animal"].astype(str),
                           "sex": reads["animal"].astype(str).map(info["sex"]),
                           "code": reads["animal"].astype(str).map(info["code"]),
                           "zone": reads["zone_id"].astype(str),
                           "noon_day": reads["day"]})
        ownership = zone_ownership(lr)
        tables["zone_ownership"] = ownership
        tables["displacement"] = annotate_ownership(
            detect_displacements(gbi, males, say), ownership)
    if layers.get("hinde"):
        from .core.hinde_index import hinde_index, hinde_summary
        for scope in ("broad", "narrow"):
            events = hinde_index(gbi, scope, names, say)
            tables[f"hinde_{scope}"] = events
            tables[f"hinde_{scope}_summary"] = hinde_summary(events)
    if layers.get("sna"):
        try:
            from .core.social_network import social_networks
            # the GBI's animal columns are animal keys, so key the metadata on them
            meta = animals.animals.assign(name=animals.animals["animal"])
            nets = social_networks(gbi, meta, names, None, say)
            tables["sna_node_stats"] = nets["node_stats"]
            tables["sna_net_stats"] = nets["net_stats"]
        except ImportError as exc:
            say(f"WARNING: social networks skipped - {exc}")
    return tables


def run(config: RFIDConfig, progress=None, write: bool = True,
        cancelled=None) -> RunResult:
    """Process one dataset. Returns its tables whether or not they were written.

    ``progress(percent, message)`` is called as the run goes; ``cancelled()``
    is polled between stages (and between files while reading) and stops the
    run cleanly when it returns True.
    """
    result = RunResult(config=config)
    started = time.time()
    pct = [0]

    def say(message: str, percent: int | None = None) -> None:
        if percent is not None:
            pct[0] = percent
        result.log.append(f"[{datetime.now():%H:%M:%S}] {message}")
        if progress:
            progress(pct[0], message)

    def check():
        if cancelled and cancelled():
            raise InterruptedError("cancelled")

    blocking, warnings = config.problems()
    if blocking:
        result.error = "; ".join(blocking)
        for p in blocking:
            say(f"Configuration is not runnable: {p}")
        return result
    for w in warnings:
        say(f"WARNING: {w}")
    result.warnings.extend(warnings)
    if not config.name:
        config.name = default_name(config.download_dirs)
    output_dir = resolve_output_dir(config)
    result.output_dir = output_dir
    if write:
        inside = output_inside_downloads(output_dir, config.download_dirs)
        if inside:
            result.error = (f"The analysis folder {output_dir} is inside the download "
                            f"folder {inside}. Outputs never go into raw-data folders; "
                            f"choose a folder beside it.")
            say(result.error)
            return result
        unusable = check_dir_usable(output_dir)
        if unusable:
            result.error = unusable
            say(unusable)
            return result

    try:
        say(f"=== {config.name} ===", 0)
        say("Loading metadata...", 2)
        animals = load_animals(config.metadata)
        result.animals = animals
        for w in animals.warnings:
            say(f"WARNING: {w}")
        result.warnings.extend(animals.warnings)
        say(f"{len(animals.animals)} animals, {len(animals.tags)} tags "
            f"(header on row {animals.header_row})")
        check()

        paths = list_download_files(config.download_dirs, config.include_subfolders)
        result.input_files = paths
        if not paths:
            raise FileNotFoundError("No download files (.xlsx, .xls, .txt, .csv) in "
                                    + ", ".join(config.download_dirs))
        say(f"Reading {len(paths)} file(s)...", 5)

        def file_progress(message):
            say(message)
        downloads: DownloadResult = read_downloads(
            paths, config.date_order, config.time_resolution, file_progress, cancelled)
        for w in downloads.warnings:
            say(f"WARNING: {w}")
        result.warnings.extend(downloads.warnings)
        used = int((downloads.files["status"] == "used").sum())
        say(f"{downloads.raw_rows:,} rows in {used} file(s) -> "
            f"{len(downloads.reads):,} unique reads", 45)
        skipped = downloads.files[downloads.files["status"].isin(["not a download", "error"])]
        for _, row in skipped.iterrows():
            say(f"Skipped {row['file']} ({row['status']}: {row['note']})")
        if downloads.reads.empty:
            raise ValueError("No readable reads in the download folder(s)")
        check()

        say("Cleaning reads...", 50)
        clean: CleanResult = clean_reads(downloads.reads, animals, config,
                                         lambda m: say(m))
        for w in clean.warnings:
            say(f"WARNING: {w}")
        result.warnings.extend(clean.warnings)
        result.origin = clean.origin
        say(f"Day 1 starts {clean.origin} (local wall clock)")
        reads = clean.reads
        if reads.empty:
            raise ValueError("No reads survived cleaning - check the antenna table "
                             "and the metadata tag columns")
        check()

        say("Detecting bouts...", 65)
        bouts = detect_bouts(reads, config.bout_gap_s, config.min_bout_s,
                             config.bout_rule, animals.animals, lambda m: say(m))
        check()

        say("Summaries...", 72)
        home = home_paddocks(config, animals)
        tables = result.tables
        tables["reads"] = reads[["animal", "tag", "reader", "paddock", "antenna",
                                 "location", "zone", "zone_id", "away", "day", "t"]]
        tables["bouts"] = bouts
        tables["animal_days"] = animal_days(reads, bouts, animals.animals,
                                            config.arena.zone_xy(), home)
        tables["zone_holders"] = zone_holders(reads, animals.animals,
                                              config.holder_share, config.holder_sex, home)
        tables["co_occupancy"] = co_occupancy(bouts, animals.animals)
        say(f"{len(tables['animal_days']):,} animal-days, "
            f"{len(tables['zone_holders']):,} zone-days, "
            f"{len(tables['co_occupancy']):,} pair-days of co-occupancy")
        check()

        say("Quality control...", 80)
        ident = clean.identified
        tables["qc_files"] = downloads.files
        tables["qc_unknown_tags"] = clean.unknown_tags
        tables["qc_unmapped_antennas"] = clean.unmapped
        tables["qc_animals"] = QC.animal_roster(animals.animals, ident, reads,
                                                clean.last_entries, home)
        tables["qc_cross_paddock"] = QC.cross_paddock(ident, config.cross_paddock_policy)
        tables["qc_two_places"] = QC.two_places(ident, config.two_place_share)
        tables["qc_stationary_tags"] = QC.stationary_tags(
            ident, config.entrance_locations, config.bout_gap_s,
            config.stationary_hour_fraction)
        tables["qc_found_dead"] = clean.last_entries
        tables["qc_dropped_reads"] = QC.dropped_summary(clean.dropped, ident)
        n_two = len(tables["qc_two_places"])
        n_still = len(tables["qc_stationary_tags"])
        if n_two:
            say(f"QC: {n_two} animal-day(s) with two tags in two places")
        if n_still:
            say(f"QC: {n_still} tag-day(s) read by one antenna all day")
        check()

        if any(config.layers.values()):
            say("Optional layers...", 85)
            tables.update(legacy_layers(config, reads, bouts, animals, lambda m: say(m)))
            check()

        result.stats = {
            "files_used": used,
            "raw_rows": downloads.raw_rows,
            "unique_reads": len(downloads.reads),
            "animals_in_metadata": len(animals.animals),
            "tags_in_metadata": len(animals.tags),
            "dropped": clean.dropped,
            "unknown_tags": len(clean.unknown_tags),
            "cleaned_reads": len(reads),
            "animals_read": int(reads["animal"].nunique()),
            "away_reads": int(reads["away"].sum()),
            "bouts": len(bouts),
            "animal_hours_in_zones": round(float(bouts["duration_s"].sum()) / 3600, 2),
            "day_origin": str(clean.origin),
            "first_day": int(reads["day"].min()), "last_day": int(reads["day"].max()),
        }

        if write:
            say("Writing tables...", 90)
            ensure_dir(output_dir)
            names = MAIN_TABLES + QC_TABLES + [k for k in tables
                                               if k not in MAIN_TABLES + QC_TABLES]
            for key in names:
                if key not in tables:
                    continue
                if key == "reads" and not config.write_reads:
                    continue
                result.written.append(write_table(tables[key], output_dir,
                                                  config.name, key))
                say(f"Wrote {os.path.basename(result.written[-1])}")
            result.stats["run_seconds"] = round(time.time() - started, 1)
            ConfigManager.save(config, output_dir, extra={
                "provenance": provenance(paths, config.metadata.path),
                "run_stats": result.stats,
                "warnings": result.warnings,
                "outputs": [os.path.relpath(p, output_dir) for p in result.written]})
            say(f"Wrote {len(result.written)} table(s) and fnt_config.json to {output_dir}")
        result.stats["run_seconds"] = round(time.time() - started, 1)
        say(f"Done in {result.stats['run_seconds']:.1f} s", 100)

    except InterruptedError:
        result.cancelled = True
        say("Cancelled - nothing was written")
    except Exception as exc:                              # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        say(f"FAILED: {result.error}")
        result.log.append(traceback.format_exc())

    if write and result.output_dir and os.path.isdir(result.output_dir):
        try:
            log_path = os.path.join(result.output_dir, f"{config.name}_messageLog.txt")
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(result.log) + "\n")
        except OSError:
            pass
    return result
