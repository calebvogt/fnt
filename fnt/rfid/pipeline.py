"""Run one trial end to end and write its analysis folder.

The stage order is fixed because each stage consumes the previous one's table.
What varies is which of the analysis layers below the GBI get written, which
``TrialConfig.exports`` decides.

Nothing here is interactive: the GUI drives this through ``run_trial`` and
receives progress through the callback, so the same code path serves a batch
run over many trials.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .config.config_manager import (ConfigManager, LOG_NAME, analysis_dir,
                                    check_dir_usable, csv_dir, ensure_dir)
from .config.defaults import TrialConfig
from .core.bout_detector import detect_bouts
from .core.displacement import annotate_ownership, detect_displacements
from .core.edgelist import co_presence_bouts, edgelist
from .core.gbi_generator import create_gbi, melt_gbi
from .core.hinde_index import hinde_index, hinde_summary
from .core.preprocessor import RFIDPreprocessor, ReadsResult
from .core.social_network import social_networks
from .core.zone_ownership import zone_ownership


@dataclass
class TrialResult:
    """Everything one run produced, in memory, plus where it was written."""
    config: TrialConfig
    output_dir: str = ""
    reads_result: ReadsResult | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    written: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def reads(self) -> pd.DataFrame:
        return self.tables.get("reads", pd.DataFrame())

    @property
    def bouts(self) -> pd.DataFrame:
        return self.tables.get("bouts", pd.DataFrame())

    @property
    def gbi(self) -> pd.DataFrame:
        return self.tables.get("gbi", pd.DataFrame())


def _write_csv(table: pd.DataFrame, output_dir: str, trial: str, name: str,
               written: list[str]) -> None:
    if table is None or table.empty:
        return
    path = os.path.join(csv_dir(output_dir, create=True), f"{trial}_{name}.csv")
    table.to_csv(path, index=False)
    written.append(path)


def run_trial(config: TrialConfig, progress=None, write: bool = True,
              bout_algorithm: str = "segment") -> TrialResult:
    """Process one trial. Returns its tables whether or not they were written."""
    result = TrialResult(config=config)
    lines: list[str] = []

    def say(message: str) -> None:
        stamped = f"[{datetime.now():%H:%M:%S}] {message}"
        lines.append(stamped)
        if progress:
            progress(message)

    ok, problems = ConfigManager.validate(config)
    if not ok:
        result.error = "; ".join(problems)
        result.log = [f"Configuration is not runnable: {p}" for p in problems]
        return result

    trial = config.trial_id or os.path.basename(os.path.normpath(config.raw_dir))
    output_dir = config.output_dir or analysis_dir(config.raw_dir, config.trial_id)
    result.output_dir = output_dir
    if write:
        unusable = check_dir_usable(output_dir)
        if unusable:
            result.error = unusable
            result.log = [unusable]
            return result
        ensure_dir(output_dir)

    try:
        say(f"=== {trial} ===")

        # --- stage 1: reads ------------------------------------------------
        reads_result = RFIDPreprocessor(config).run(say)
        result.reads_result = reads_result
        reads = reads_result.reads
        result.tables["reads"] = reads
        for warning in reads_result.warnings():
            say(f"WARNING: {warning}")
        if reads.empty:
            result.error = "No reads survived preprocessing"
            say(result.error)
            return result

        meta = reads.drop_duplicates("name")[
            [c for c in ("name", "code", "sex", "phase", "group", "strain")
             if c in reads.columns]]
        animals = sorted(reads["name"].unique())
        sex_by_name = meta.set_index("name")["sex"].to_dict()
        males = [a for a in animals if sex_by_name.get(a) == "M"]

        # --- stage 2: bouts -------------------------------------------------
        bouts = detect_bouts(reads, config.bout_threshold_s,
                             config.min_duration_s, bout_algorithm, say)
        result.tables["bouts"] = bouts

        # --- stage 3: GBI ---------------------------------------------------
        gbi = create_gbi(bouts, animals, sex_by_name, config.min_duration_s, say)
        result.tables["gbi"] = gbi

        exports = config.exports
        if exports.get("bout_summary"):
            result.tables["bout_summary"] = melt_gbi(gbi, meta)
        if exports.get("zone_ownership") or exports.get("displacement"):
            ownership = zone_ownership(reads)
            result.tables["zone_ownership"] = ownership
        else:
            ownership = pd.DataFrame()

        if exports.get("edgelist"):
            pair_bouts = co_presence_bouts(gbi, animals, say)
            result.tables["co_presence_bouts"] = pair_bouts
            result.tables["edgelist"] = edgelist(pair_bouts)

        if exports.get("displacement"):
            events = detect_displacements(gbi, males, say)
            result.tables["displacement"] = annotate_ownership(
                events, ownership, days=config.analysis_days)

        for scope in ("broad", "narrow"):
            if exports.get(f"hinde_{scope}"):
                events = hinde_index(gbi, scope, animals, say)
                result.tables[f"hinde_{scope}"] = events
                result.tables[f"hinde_{scope}_summary"] = hinde_summary(events)

        if exports.get("sna"):
            try:
                nets = social_networks(gbi, meta, animals,
                                       config.analysis_days, say)
                result.tables.update({"sna_node_stats": nets["node_stats"],
                                      "sna_net_stats": nets["net_stats"]})
            except ImportError as exc:
                say(f"WARNING: social networks skipped - {exc}")

        # --- write -----------------------------------------------------------
        if write:
            for name, table in result.tables.items():
                if exports.get(name, True):
                    _write_csv(table, output_dir, trial, name, result.written)
            ConfigManager.save(config, output_dir, extra={
                "bout_algorithm": bout_algorithm,
                "run_stats": {
                    "raw_rows": reads_result.raw_read_count,
                    "unique_reads": reads_result.unique_read_count,
                    "reads": len(reads),
                    "bouts": len(bouts),
                    "gbi_intervals": len(gbi),
                    "animals": len(animals),
                },
                "warnings": reads_result.warnings(),
                "files": [os.path.basename(p) for p in result.written]})
            say(f"Wrote {len(result.written)} table(s) to {csv_dir(output_dir)}")

    except Exception as exc:                              # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        say(f"FAILED: {result.error}")
        lines.append(traceback.format_exc())

    result.log = lines
    if write and result.output_dir:
        try:
            with open(os.path.join(result.output_dir, LOG_NAME), "w",
                      encoding="utf-8") as fh:
                fh.write("\n".join(lines))
        except OSError:
            pass
    return result
