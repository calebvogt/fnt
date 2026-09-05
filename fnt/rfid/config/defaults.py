"""Configuration for a single RFID trial.

One trial, one config, one output folder - the same shape the UWB tool uses.
The previous version of this module described a whole multi-trial run at once
(``trial_ids``, ``trial_reader_map``, one ``input_dir`` above them all), which
is what produced the single-giant-ALLTRIAL-file workflow this replaces.

The arena is a first-class part of the config rather than a flat
``{antenna: zone}`` dictionary, because assigning several antennas to one zone
is the normal case here (a wall and a floor antenna per resource zone), and
because the preview has to draw the thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


def get_fnt_version() -> str:
    """Installed FNT version, for stamping into a trial's config record."""
    try:
        import tomllib
        from pathlib import Path
        toml_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
        if toml_path.exists():
            with open(toml_path, "rb") as fh:
                version = tomllib.load(fh).get("project", {}).get("version")
            if version:
                return version
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("fnt")
    except Exception:
        return "unknown"


@dataclass
class Zone:
    """A resource zone: the spatial unit an animal is 'in'."""
    zone_id: int
    x: float
    y: float
    label: str = ""


@dataclass
class Antenna:
    """One physical antenna, and the zone a read on it places the animal in.

    ``location`` is free text (``wall``, ``floor``, ``tube``...) and is carried
    through to the reads table. It is descriptive only - two antennas in the
    same zone are interchangeable for occupancy - but it is what makes a dead
    antenna diagnosable, so it is worth keeping.
    """
    antenna_id: int
    zone: int
    x: Optional[float] = None
    y: Optional[float] = None
    location: str = ""


@dataclass
class Arena:
    """Zone geometry plus the antenna-to-zone wiring."""
    zones: list[Zone] = field(default_factory=list)
    antennas: list[Antenna] = field(default_factory=list)
    units: str = "m"

    def antenna_zone_map(self) -> dict[int, int]:
        return {a.antenna_id: a.zone for a in self.antennas}

    def antenna_location_map(self) -> dict[int, str]:
        return {a.antenna_id: a.location for a in self.antennas}

    def zone_xy(self) -> dict[int, tuple[float, float]]:
        return {z.zone_id: (z.x, z.y) for z in self.zones}

    def zone_ids(self) -> list[int]:
        return [z.zone_id for z in self.zones]

    def unmapped_antennas(self, seen: list[int]) -> list[int]:
        """Antenna IDs present in the data that the arena does not describe."""
        known = self.antenna_zone_map()
        return sorted({int(a) for a in seen if int(a) not in known})

    def silent_antennas(self, seen: list[int]) -> list[int]:
        """Antenna IDs the arena describes that produced no reads.

        A zone whose only antenna is silent looks like an unvisited zone, so
        this is the check that separates 'nobody went there' from 'the antenna
        was unplugged'.
        """
        seen_set = {int(a) for a in seen}
        return sorted(a.antenna_id for a in self.antennas
                      if a.antenna_id not in seen_set)

    @classmethod
    def grid(cls, cols: int, rows: int, dx: float, dy: float,
             x0: float = 0.0, y0: float = 0.0,
             antennas_per_zone: int = 1,
             locations: tuple[str, ...] = ("wall", "floor")) -> "Arena":
        """Lay zones out on a regular grid, numbering left-to-right then up.

        Antennas are numbered so that the Nth block of ``len(zones)`` IDs is the
        Nth antenna of every zone - antennas 1-8 wall, 9-16 floor for an 8-zone
        paddock with two each. That is the convention the 2021_LID readers were
        wired with, and it keeps ``antenna_id -> zone`` a simple modulo.
        """
        zones = [Zone(zone_id=r * cols + c + 1, x=x0 + c * dx, y=y0 + r * dy)
                 for r in range(rows) for c in range(cols)]
        n = len(zones)
        antennas = []
        for block in range(antennas_per_zone):
            loc = locations[block] if block < len(locations) else f"a{block + 1}"
            for z in zones:
                antennas.append(Antenna(antenna_id=block * n + z.zone_id,
                                        zone=z.zone_id, x=z.x, y=z.y,
                                        location=loc))
        return cls(zones=zones, antennas=antennas)

    def to_dict(self) -> dict:
        return {"units": self.units,
                "zones": [asdict(z) for z in self.zones],
                "antennas": [asdict(a) for a in self.antennas]}

    @classmethod
    def from_dict(cls, data: dict) -> "Arena":
        return cls(units=data.get("units", "m"),
                   zones=[Zone(**z) for z in data.get("zones", [])],
                   antennas=[Antenna(**a) for a in data.get("antennas", [])])


# Which derived tables an export writes. Reads, bouts and the GBI are the
# preprocessing itself and are always produced; everything below them is an
# analysis layer the user opts into.
DEFAULT_EXPORTS = {
    "reads": True,
    "bouts": True,
    "gbi": True,
    "bout_summary": True,
    "zone_ownership": True,
    "edgelist": False,
    "displacement": False,
    "hinde_broad": False,
    "hinde_narrow": False,
    "sna": False,
}


@dataclass
class TrialConfig:
    """Everything needed to turn one trial's raw exports into its outputs."""

    trial_id: str = ""
    raw_dir: str = ""
    output_dir: str = ""
    metadata_path: str = ""

    # Readers physically installed in THIS trial's enclosure. A read from any
    # other reader is an animal that crossed into another paddock (or a stray
    # tag), which is a real event but not a valid observation of this arena.
    reader_ids: list[int] = field(default_factory=list)
    foreign_reader_policy: str = "drop"      # drop | keep

    bout_threshold_s: float = 50.0
    min_duration_s: float = 1.0
    day_origin_time: str = "12:00:00"

    # "ms" keeps the reader's millisecond timestamps. "s" truncates to whole
    # seconds, which is what the R pipeline effectively did by round-tripping
    # through CSV between scripts - kept so old results can be reproduced.
    time_resolution: str = "ms"

    analysis_days: tuple[int, int] = (1, 12)

    tag_columns: list[str] = field(default_factory=lambda: ["tag_1", "tag_2"])
    tag_digits: int = 15

    arena: Arena = field(default_factory=Arena)
    exports: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_EXPORTS))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["arena"] = self.arena.to_dict()
        data["analysis_days"] = list(self.analysis_days)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrialConfig":
        data = dict(data)
        data.pop("fnt_version", None)
        data.pop("run_timestamp", None)
        if isinstance(data.get("arena"), dict):
            data["arena"] = Arena.from_dict(data["arena"])
        if isinstance(data.get("analysis_days"), list):
            data["analysis_days"] = tuple(data["analysis_days"])
        exports = dict(DEFAULT_EXPORTS)
        exports.update(data.get("exports") or {})
        data["exports"] = exports
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def eight_zone_paddock() -> TrialConfig:
    """The 2021_LID enclosure: 8 zones in 2 columns x 4 rows, 16 antennas.

    Coordinates are the ones the R pipeline hard-coded, in metres.
    """
    arena = Arena.grid(cols=2, rows=4, dx=7.5, dy=7.6, x0=3.75, y0=7.6,
                       antennas_per_zone=2, locations=("wall", "floor"))
    return TrialConfig(arena=arena, reader_ids=[])


def empty_trial() -> TrialConfig:
    """A blank config, for an enclosure that is not the 8-zone paddock."""
    return TrialConfig()


TEMPLATES = {
    "8_zone_paddock": eight_zone_paddock,
    "custom": empty_trial,
}


def get_default_config(template: str = "8_zone_paddock") -> TrialConfig:
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template {template!r}. "
                         f"Available: {sorted(TEMPLATES)}")
    return TEMPLATES[template]()


def get_available_templates() -> list[str]:
    return sorted(TEMPLATES)


# Back-compatible alias: the previous class name, used by the old GUI.
RFIDConfig = TrialConfig
