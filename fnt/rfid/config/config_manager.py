"""The trial's analysis folder: where things go, and how the config persists.

Layout, mirroring the UWB tool so a trial folder means the same thing whichever
modality produced it:

    <trial raw folder>/
        LID_2021_ALPHA_RFID_DL_1.txt        raw exports, never touched
        ...
        <trial>_FNT/
            fnt_config.json                 settings + provenance
            _messageLog.txt
            csvs/                           every table this run produced
            plots/
            animations/

``fnt_config.json`` is the record of how a folder's contents were produced, so
a clear-and-re-export keeps it while removing everything else. Losing it does
not lose data, but it loses the ability to say what the data means.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

from .defaults import TrialConfig, get_fnt_version

CONFIG_NAME = "fnt_config.json"
LOG_NAME = "_messageLog.txt"
CSV_SUBDIR = "csvs"
PLOT_SUBDIR = "plots"
ANIM_SUBDIR = "animations"
ANALYSIS_SUFFIX = "_FNT"

# Everything in an analysis folder is an export EXCEPT these. A keep-list, not
# a delete-list: a product that gets renamed or retired leaves behind a file the
# tool no longer writes, and only a keep-list can see that it is an orphan.
CLEAR_KEEP_NAMES = (CONFIG_NAME,)


def analysis_dir(raw_dir: str, trial_id: str = "") -> str:
    """Where a trial's outputs live, given the folder its raw exports are in."""
    name = (trial_id or os.path.basename(os.path.normpath(raw_dir))) + ANALYSIS_SUFFIX
    return os.path.join(raw_dir, name)


def ensure_dir(path: str, attempts: int = 12, delay: float = 0.25) -> str:
    """Create a directory, tolerating a delete that has not landed yet.

    On a network share or a OneDrive-synced folder a removed directory lingers
    in a pending-delete state until every handle closes. ``makedirs`` then
    raises even though ``isdir`` reports False - and clear-then-re-export walks
    straight into it. Retry briefly rather than failing a run over a race that
    resolves itself in well under a second.
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


def plot_dir(output_dir: str, create: bool = False) -> str:
    path = os.path.join(output_dir, PLOT_SUBDIR)
    return ensure_dir(path) if create else path


def animation_dir(output_dir: str, create: bool = False) -> str:
    path = os.path.join(output_dir, ANIM_SUBDIR)
    return ensure_dir(path) if create else path


def check_dir_usable(path: str) -> str | None:
    """Why ``path`` cannot serve as an output folder, or None if it can.

    Being able to write is not enough: a directory can accept files while
    refusing to be ENUMERATED, which is a state a network share or a
    half-synced OneDrive folder can leave one in. Exports would appear to work
    but the folder could never be cleared, so each run's output would pile up
    on the last one's, indistinguishable.
    """
    if not os.path.isdir(path):
        return None
    try:
        os.listdir(path)
    except OSError as exc:
        return (f"{path} cannot be listed ({exc.strerror or exc}). It still "
                f"accepts files, so an export would look like it worked - but "
                f"the folder could never be cleared, and old products would "
                f"survive beside the new ones looking identical. Delete it in "
                f"Explorer; the next export recreates it.")
    return None


class ConfigManager:
    """Read and write a trial's ``fnt_config.json``."""

    @staticmethod
    def config_path(output_dir: str) -> str:
        return os.path.join(output_dir, CONFIG_NAME)

    @staticmethod
    def save(config: TrialConfig, output_dir: str, extra: dict | None = None) -> str:
        """Write the config, stamped with the version and time that made it."""
        ensure_dir(output_dir)
        payload = config.to_dict()
        payload["fnt_version"] = get_fnt_version()
        payload["run_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if extra:
            payload.update(extra)
        path = ConfigManager.config_path(output_dir)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path

    @staticmethod
    def load(output_dir: str) -> TrialConfig | None:
        """Read a previous run's config, or None if there is not one."""
        path = ConfigManager.config_path(output_dir)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return TrialConfig.from_dict(json.load(fh))

    @staticmethod
    def load_raw(output_dir: str) -> dict | None:
        """The config file as written, including the provenance stamps."""
        path = ConfigManager.config_path(output_dir)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def save_to_file(config: TrialConfig, filepath: str) -> None:
        """Write a config anywhere - used for shareable presets."""
        ensure_dir(os.path.dirname(os.path.abspath(filepath)))
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(config.to_dict(), fh, indent=2, default=str)

    @staticmethod
    def load_from_file(filepath: str) -> TrialConfig:
        with open(filepath, "r", encoding="utf-8") as fh:
            return TrialConfig.from_dict(json.load(fh))

    @staticmethod
    def validate(config: TrialConfig) -> tuple[bool, list[str]]:
        """Problems that would stop a run, as a list of messages."""
        problems: list[str] = []
        if not config.raw_dir or not os.path.isdir(config.raw_dir):
            problems.append(f"Raw data folder not found: {config.raw_dir!r}")
        if not config.metadata_path or not os.path.isfile(config.metadata_path):
            problems.append(f"Metadata file not found: {config.metadata_path!r}")
        if not config.arena.zones:
            problems.append("No zones defined - set up the arena first")
        if not config.arena.antennas:
            problems.append("No antennas defined - set up the arena first")

        zone_ids = set(config.arena.zone_ids())
        orphans = sorted({a.antenna_id for a in config.arena.antennas
                          if a.zone not in zone_ids})
        if orphans:
            problems.append(f"Antenna(s) assigned to a zone that does not "
                            f"exist: {orphans}")
        covered = {a.zone for a in config.arena.antennas}
        bare = sorted(zone_ids - covered)
        if bare:
            problems.append(f"Zone(s) with no antenna, so they can never record "
                            f"an animal: {bare}")

        seen: dict[int, int] = {}
        for antenna in config.arena.antennas:
            seen[antenna.antenna_id] = seen.get(antenna.antenna_id, 0) + 1
        repeats = sorted(a for a, n in seen.items() if n > 1)
        if repeats:
            problems.append(f"Antenna ID(s) defined more than once: {repeats}")

        if config.bout_threshold_s <= 0:
            problems.append("Bout threshold must be greater than zero")
        if config.time_resolution not in ("ms", "s"):
            problems.append(f"time_resolution must be 'ms' or 's', "
                            f"not {config.time_resolution!r}")
        if config.foreign_reader_policy not in ("drop", "keep"):
            problems.append(f"foreign_reader_policy must be 'drop' or 'keep', "
                            f"not {config.foreign_reader_policy!r}")
        return (not problems), problems

    # Kept so existing callers keep working.
    save_config = save_to_file
    load_config = load_from_file
    validate_config = validate
