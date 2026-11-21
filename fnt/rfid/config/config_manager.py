"""The analysis folder: where outputs go, and how a config persists.

Layout, mirroring the UWB tool so an analysis folder means the same thing
whichever modality produced it::

    <parent of the downloads>/
        T001_Alpha/                         raw downloads, never written to
        T002_Bravo/
        LID_2021_FNT_analysis/
            fnt_config.json                 every setting + provenance
            LID_2021_messageLog.txt
            csvs/                           every table the run produced

The analysis folder sits BESIDE the download folders, never inside one: a
download folder is raw data, and a run that wrote into it would leave files the
next run's folder scan has to know to ignore.

``fnt_config.json`` is the record of how the folder's contents were produced:
the settings, the FNT version, the time of the run, and the size and
modification time of every input file. Losing it does not lose data, but it
loses the ability to say what the data means.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from datetime import datetime

from .defaults import (ANTENNA_CSV_COLUMNS, Antenna, RFIDConfig, Zone,
                       get_fnt_version, normalise_id)

CONFIG_NAME = "fnt_config.json"
CSV_SUBDIR = "csvs"
ANALYSIS_SUFFIX = "_FNT_analysis"


def default_name(download_dirs: list[str]) -> str:
    """A dataset name from its download folders.

    One folder: its own name. Several: their names joined (``T001_Alpha+
    T002_Bravo``), which says exactly what went in; past three, the parent
    folder's name, since a list that long stops being readable.
    """
    names = [os.path.basename(os.path.normpath(d)) for d in download_dirs if d]
    if not names:
        return "rfid"
    if len(names) == 1:
        return names[0]
    if len(names) <= 3:
        return "+".join(names)
    parent = os.path.basename(os.path.dirname(os.path.normpath(download_dirs[0])))
    return f"{parent or 'rfid'}_combined"


def default_output_dir(config: RFIDConfig) -> str:
    """``<name>_FNT_analysis`` in the folder that holds the download folders."""
    dirs = [os.path.abspath(d) for d in config.download_dirs if d]
    if not dirs:
        return ""
    parents = [os.path.dirname(os.path.normpath(d)) for d in dirs]
    try:
        parent = os.path.commonpath(parents)
    except ValueError:                        # different drives on Windows
        parent = parents[0]
    name = config.name or default_name(config.download_dirs)
    return os.path.join(parent, f"{name}{ANALYSIS_SUFFIX}")


def resolve_output_dir(config: RFIDConfig) -> str:
    return config.output_dir or default_output_dir(config)


def output_inside_downloads(output_dir: str, download_dirs: list[str]) -> str | None:
    """The download folder ``output_dir`` would sit inside, or None."""
    out = os.path.abspath(output_dir)
    for d in download_dirs:
        d = os.path.abspath(d)
        try:
            if os.path.commonpath([out, d]) == d:
                return d
        except ValueError:
            continue
    return None


def ensure_dir(path: str, attempts: int = 12, delay: float = 0.25) -> str:
    """Create a directory, tolerating a delete that has not landed yet.

    On a network share or a OneDrive-synced folder a removed directory lingers
    in a pending-delete state until every handle closes; ``makedirs`` then
    raises even though ``isdir`` says it is gone. Retry briefly rather than
    failing a run over a race that resolves itself in under a second.
    """
    last = None
    for i in range(attempts):
        try:
            os.makedirs(path, exist_ok=True)
            if os.path.isdir(path):
                return path
        except OSError as exc:
            last = exc
        time.sleep(delay * (i + 1))
        if os.path.isdir(path):
            return path
    if last is not None:
        raise last
    raise OSError(f"could not create {path}")


def csv_dir(output_dir: str, create: bool = False) -> str:
    path = os.path.join(output_dir, CSV_SUBDIR)
    return ensure_dir(path) if create else path


def check_dir_usable(path: str) -> str | None:
    """Why ``path`` cannot serve as an output folder, or None if it can.

    A directory can accept files while refusing to be listed - a state a
    network share or a half-synced OneDrive folder can leave one in. Exports
    would appear to work, but old products could never be told apart from
    new ones.
    """
    if not os.path.isdir(path):
        return None
    try:
        os.listdir(path)
    except OSError as exc:
        return (f"{path} cannot be listed ({exc.strerror or exc}). Delete it "
                f"and run again; the run recreates it.")
    return None


def file_stamp(path: str) -> dict:
    """Size and modification time of an input, for the provenance record."""
    try:
        st = os.stat(path)
        return {"path": os.path.abspath(path), "size_bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S")}
    except OSError as exc:
        return {"path": os.path.abspath(path), "error": str(exc)}


def provenance(input_files: list[str], metadata_path: str = "") -> dict:
    """What an output folder was made from, and by what."""
    try:
        import numpy
        import pandas
        libs = {"pandas": pandas.__version__, "numpy": numpy.__version__}
    except Exception:                                  # pragma: no cover
        libs = {}
    return {
        "fnt_version": get_fnt_version(),
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": libs,
        "metadata_file": file_stamp(metadata_path) if metadata_path else None,
        "input_files": [file_stamp(p) for p in input_files],
    }


class ConfigManager:
    """Read and write ``fnt_config.json`` and shareable config files."""

    @staticmethod
    def config_path(output_dir: str) -> str:
        return os.path.join(output_dir, CONFIG_NAME)

    @staticmethod
    def save(config: RFIDConfig, output_dir: str, extra: dict | None = None) -> str:
        """Write the run record: settings under ``config``, provenance beside."""
        ensure_dir(output_dir)
        payload = {"tool": "FNT RFID PreProcessing",
                   "config": config.to_dict()}
        if extra:
            payload.update(extra)
        path = ConfigManager.config_path(output_dir)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path

    @staticmethod
    def load(output_dir: str) -> RFIDConfig | None:
        path = ConfigManager.config_path(output_dir)
        if not os.path.exists(path):
            return None
        return ConfigManager.load_from_file(path)

    @staticmethod
    def load_raw(output_dir: str) -> dict | None:
        path = ConfigManager.config_path(output_dir)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def save_to_file(config: RFIDConfig, filepath: str) -> None:
        """Write a config anywhere - a preset to reuse for the next dataset."""
        folder = os.path.dirname(os.path.abspath(filepath))
        ensure_dir(folder)
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump({"tool": "FNT RFID PreProcessing",
                       "fnt_version": get_fnt_version(),
                       "config": config.to_dict()}, fh, indent=2, default=str)

    @staticmethod
    def load_from_file(filepath: str) -> RFIDConfig:
        """Read a saved config, or a previous run's ``fnt_config.json``."""
        with open(filepath, "r", encoding="utf-8") as fh:
            return RFIDConfig.from_dict(json.load(fh))


# --------------------------------------------------------------------------- #
# Antenna table as CSV
# --------------------------------------------------------------------------- #

def write_antenna_csv(arena, path: str) -> None:
    """The antenna table as a CSV a person can edit in a spreadsheet."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(ANTENNA_CSV_COLUMNS)
        for a in arena.antennas:
            w.writerow([a.reader, a.antenna, a.zone, a.location, a.paddock_id,
                        "" if a.x is None else a.x, "" if a.y is None else a.y])


def read_antenna_csv(path: str) -> list[Antenna]:
    """Antennas from a CSV with at least ``reader``, ``antenna`` and ``zone``.

    Header names are matched loosely (``Reader ID``, ``antenna_id``,
    ``Zone``...) because this file is usually typed by hand.
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{os.path.basename(path)} is empty")

    def key(text):
        return "".join(ch for ch in str(text).lower() if ch.isalnum())

    header = [key(h) for h in rows[0]]
    aliases = {"reader": ("reader", "readerid"),
               "antenna": ("antenna", "antennaid", "ant"),
               "zone": ("zone", "zoneid"),
               "location": ("location", "loc", "antloc", "type"),
               "paddock": ("paddock", "paddockid", "enclosure"),
               "x": ("x", "antennax"), "y": ("y", "antennay")}
    index = {}
    for field_name, names in aliases.items():
        for n in names:
            if n in header:
                index[field_name] = header.index(n)
                break
    missing = [f for f in ("reader", "antenna", "zone") if f not in index]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: missing column(s) {missing}; "
                         f"found {rows[0]}")

    def cell(row, name):
        i = index.get(name)
        return row[i] if i is not None and i < len(row) else ""

    out = []
    for row in rows[1:]:
        if not any(str(c).strip() for c in row):
            continue
        out.append(Antenna(reader=cell(row, "reader"), antenna=cell(row, "antenna"),
                           zone=cell(row, "zone"), location=cell(row, "location"),
                           paddock_id=cell(row, "paddock"),
                           x=cell(row, "x"), y=cell(row, "y")))
    return out


def zones_for_antennas(arena, antennas: list[Antenna]) -> list[Zone]:
    """Zones the given antennas point at that the arena does not have yet."""
    have = set(arena.zone_ids())
    reader_paddock = arena.reader_paddock()
    new, seen = [], set()
    for a in antennas:
        paddock = a.paddock_id or reader_paddock.get(normalise_id(a.reader), "")
        zid = f"{paddock}{a.zone}"
        if paddock and zid not in have and zid not in seen:
            seen.add(zid)
            new.append(Zone(paddock, a.zone))
    return new
