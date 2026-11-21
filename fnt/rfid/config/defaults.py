"""Configuration for an RFID preprocessing run.

One config describes one dataset: the download folders, the animal metadata and
how to read it, the arena (paddocks, readers, zones, antennas), and every
cleaning and bout rule. It is saved beside the outputs as ``fnt_config.json``,
so the folder a run produced always carries the settings that produced it.

The arena is the part that needed rethinking. The previous version keyed
everything on a bare antenna number and assumed one reader per trial, which is
what made the 2021 LID dataset awkward: two paddocks run side by side, reader 1
in Alpha and reader 2 in Bravo, both numbering their antennas 1-16. An antenna
number alone does not say where a read happened; ``(reader, antenna)`` does.
Zones are therefore identified by paddock + zone (``A1`` .. ``A8``,
``B1`` .. ``B8``), and a read by the other paddock's reader lands in the other
paddock's zone rather than being folded into the animal's own.

No site coordinates live here - only arena geometry in metres, which is the
same whichever field station it is built at.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


def get_fnt_version() -> str:
    """Installed FNT version, for stamping into a run's config record.

    Same resolution order as the main GUI: pyproject.toml when running from a
    source checkout, then package metadata (the PyInstaller build), then
    'unknown' so stamping can never break an export.
    """
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


def normalise_id(value) -> str:
    """Reader / antenna IDs as canonical text: '001' -> '1', 1.0 -> '1'.

    Biomark writes zero-padded IDs ('001'), a spreadsheet that touched the file
    turns them into numbers, and a person typing the arena table writes '1'.
    All three have to be the same key, or every read falls through the
    antenna map as 'unmapped'.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "na"):
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _clean_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.lower() in ("nan", "none", "na"):
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Arena
# --------------------------------------------------------------------------- #

@dataclass
class Paddock:
    """One enclosure: its ID (the zone-ID prefix), readers, and residents.

    ``trials`` are the values of the metadata's trial / home column that mean
    "lives in this paddock" - an animal whose trial is T001 has paddock A as
    its home, so a read of it by reader 2 is a visit to B (``away``).
    """
    paddock_id: str
    name: str = ""
    readers: list[str] = field(default_factory=list)
    trials: list[str] = field(default_factory=list)
    width_m: Optional[float] = None
    length_m: Optional[float] = None

    def __post_init__(self):
        self.paddock_id = str(self.paddock_id).strip()
        self.readers = [normalise_id(r) for r in (self.readers or []) if normalise_id(r)]
        self.trials = [str(t).strip() for t in (self.trials or []) if str(t).strip()]
        self.width_m = _clean_float(self.width_m)
        self.length_m = _clean_float(self.length_m)


@dataclass
class Zone:
    """A resource zone, identified by paddock + zone label (``A`` + ``3``)."""
    paddock_id: str
    zone: str
    x: Optional[float] = None
    y: Optional[float] = None
    label: str = ""

    def __post_init__(self):
        self.paddock_id = str(self.paddock_id).strip()
        self.zone = normalise_id(self.zone)
        self.x = _clean_float(self.x)
        self.y = _clean_float(self.y)

    @property
    def zone_id(self) -> str:
        return f"{self.paddock_id}{self.zone}"


@dataclass
class Antenna:
    """One physical antenna, keyed by ``(reader, antenna)``.

    ``paddock_id`` is normally blank: the antenna is in its reader's paddock.
    It is there for the rare rig where one reader's multiplexer serves two
    enclosures.

    ``location`` is free text (``wall``, ``floor``, ``tube``...). It is
    descriptive for occupancy - two antennas in one zone are interchangeable -
    but not for everything: an *entry* is a read on an entrance-type antenna
    after a gap, which is what separates a mouse from a body lying under a
    floor antenna (see the found-dead rule in the cleaning stage).
    """
    reader: str
    antenna: str
    zone: str
    location: str = ""
    paddock_id: str = ""
    x: Optional[float] = None
    y: Optional[float] = None

    def __post_init__(self):
        self.reader = normalise_id(self.reader)
        self.antenna = normalise_id(self.antenna)
        self.zone = normalise_id(self.zone)
        self.location = str(self.location or "").strip()
        self.paddock_id = str(self.paddock_id or "").strip()
        self.x = _clean_float(self.x)
        self.y = _clean_float(self.y)


ANTENNA_CSV_COLUMNS = ["reader", "antenna", "zone", "location", "paddock",
                       "x", "y"]


@dataclass
class Arena:
    """Paddocks, zones and the ``(reader, antenna) -> zone`` wiring."""
    paddocks: list[Paddock] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    antennas: list[Antenna] = field(default_factory=list)
    units: str = "m"

    # --- lookups ---------------------------------------------------------
    def reader_paddock(self) -> dict[str, str]:
        out = {}
        for p in self.paddocks:
            for r in p.readers:
                out.setdefault(r, p.paddock_id)
        return out

    def paddock_ids(self) -> list[str]:
        return [p.paddock_id for p in self.paddocks]

    def paddock_names(self) -> dict[str, str]:
        return {p.paddock_id: (p.name or p.paddock_id) for p in self.paddocks}

    def antenna_paddock(self, antenna: Antenna) -> str:
        return antenna.paddock_id or self.reader_paddock().get(antenna.reader, "")

    def zone_xy(self) -> dict[str, tuple]:
        return {z.zone_id: (z.x, z.y) for z in self.zones}

    def zone_ids(self) -> list[str]:
        return [z.zone_id for z in self.zones]

    def home_paddock(self, trial_value) -> str:
        """The paddock an animal lives in, from its metadata trial/home value.

        Matched against each paddock's ``trials`` first, then its ID, then its
        name, all case-insensitively - so a metadata column that simply says
        'A' or 'Alpha' works as well as one that says 'T001'.
        """
        text = str(trial_value or "").strip().lower()
        if not text or text in ("nan", "none"):
            return ""
        for p in self.paddocks:
            if text in (t.lower() for t in p.trials):
                return p.paddock_id
        for p in self.paddocks:
            if text == p.paddock_id.lower():
                return p.paddock_id
        for p in self.paddocks:
            if p.name and text == p.name.lower():
                return p.paddock_id
        return ""

    def antenna_rows(self) -> list[dict]:
        """One dict per antenna with its paddock, zone ID and coordinates."""
        xy = self.zone_xy()
        rows = []
        for a in self.antennas:
            paddock = self.antenna_paddock(a)
            zone_id = f"{paddock}{a.zone}" if paddock else ""
            zx, zy = xy.get(zone_id, (None, None))
            rows.append({"reader": a.reader, "antenna": a.antenna,
                         "paddock": paddock, "zone": a.zone,
                         "zone_id": zone_id, "location": a.location,
                         "zone_x": zx, "zone_y": zy,
                         "antenna_x": a.x if a.x is not None else zx,
                         "antenna_y": a.y if a.y is not None else zy})
        return rows

    # --- checks ----------------------------------------------------------
    def problems(self) -> tuple[list[str], list[str]]:
        """``(blocking, warnings)``: what would stop a run, what merely looks odd."""
        blocking, warnings = [], []
        if not self.paddocks:
            blocking.append("No paddocks defined - set up the arena first")
        if not self.antennas:
            blocking.append("No antennas defined - set up the arena first")

        ids = [p.paddock_id for p in self.paddocks]
        if any(not i for i in ids):
            blocking.append("A paddock has no ID (the ID is the zone prefix, e.g. 'A')")
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            blocking.append(f"Paddock ID(s) used twice: {dupes}")

        seen: dict[str, str] = {}
        for p in self.paddocks:
            for r in p.readers:
                if r in seen and seen[r] != p.paddock_id:
                    blocking.append(f"Reader {r} is listed in paddocks "
                                    f"{seen[r]} and {p.paddock_id}")
                seen[r] = p.paddock_id

        keys = [(a.reader, a.antenna) for a in self.antennas]
        repeats = sorted({k for k in keys if keys.count(k) > 1})
        if repeats:
            blocking.append("Antenna(s) defined more than once: "
                            + ", ".join(f"reader {r} antenna {a}" for r, a in repeats))

        zone_ids = set(self.zone_ids())
        homeless, missing_zone = set(), set()
        for a in self.antennas:
            paddock = self.antenna_paddock(a)
            if not paddock:
                homeless.add(a.reader)
            elif f"{paddock}{a.zone}" not in zone_ids:
                missing_zone.add(f"{paddock}{a.zone}")
        if homeless:
            blocking.append(f"Reader(s) {sorted(homeless)} belong to no paddock - "
                            f"add them to a paddock's reader list")
        if missing_zone:
            blocking.append(f"Antennas point at zone(s) that are not in the zones "
                            f"table: {sorted(missing_zone)}")

        covered = {f"{self.antenna_paddock(a)}{a.zone}" for a in self.antennas}
        bare = sorted(zone_ids - covered)
        if bare:
            warnings.append(f"Zone(s) with no antenna, which can never record an "
                            f"animal: {bare}")
        no_xy = sorted(z.zone_id for z in self.zones if z.x is None or z.y is None)
        if no_xy:
            warnings.append(f"Zone(s) without coordinates - minimum distance "
                            f"cannot be computed for moves involving them: {no_xy}")
        return blocking, warnings

    # --- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return {"units": self.units,
                "paddocks": [asdict(p) for p in self.paddocks],
                "zones": [asdict(z) for z in self.zones],
                "antennas": [asdict(a) for a in self.antennas]}

    @classmethod
    def from_dict(cls, data: dict | None) -> "Arena":
        data = data or {}
        return cls(units=data.get("units", "m"),
                   paddocks=[Paddock(**_known(Paddock, p)) for p in data.get("paddocks", [])],
                   zones=[Zone(**_known(Zone, z)) for z in data.get("zones", [])],
                   antennas=[Antenna(**_known(Antenna, a)) for a in data.get("antennas", [])])


def _known(cls, data: dict) -> dict:
    fields = set(cls.__dataclass_fields__)
    return {k: v for k, v in (data or {}).items() if k in fields}


# --------------------------------------------------------------------------- #
# Arena presets
# --------------------------------------------------------------------------- #

# Zone centres of the Liddell paddocks (metres, in a 15 x 38 m paddock): two
# columns of four resource zones. Geometry, not a location.
LIDDELL_X = (3.75, 11.25)
LIDDELL_Y = (7.6, 15.2, 22.8, 30.4)


def _liddell_paddock(paddock_id: str, name: str, reader: str,
                     trials: list[str]) -> tuple[Paddock, list[Zone], list[Antenna]]:
    paddock = Paddock(paddock_id, name, [reader], trials, 15.0, 38.0)
    zones, antennas = [], []
    n = 0
    for y in LIDDELL_Y:
        for x in LIDDELL_X:
            n += 1
            zones.append(Zone(paddock_id, str(n), x, y))
    # Antennas 1-8 ring the entrance tubes ("wall"), 9-16 sit under the floor
    # of the same zones: zone = antenna, or antenna - 8.
    for block, location in enumerate(("wall", "floor")):
        for z in range(1, 9):
            antennas.append(Antenna(reader, str(block * 8 + z), str(z), location))
    return paddock, zones, antennas


def liddell_8_zone(paddocks: int = 1) -> Arena:
    """The Liddell 8-zone paddock: 2 x 4 zones, a wall and a floor antenna each.

    ``paddocks=2`` builds the side-by-side pair (A on reader 1, B on reader 2)
    the 2021 LID trials used. Trial lists are left empty: which metadata trial
    lives in which paddock is the user's to fill in.
    """
    arena = Arena()
    for i in range(max(1, int(paddocks))):
        pid = chr(ord("A") + i)
        p, z, a = _liddell_paddock(pid, "", str(i + 1), [])
        arena.paddocks.append(p)
        arena.zones.extend(z)
        arena.antennas.extend(a)
    return arena


def one_antenna_per_zone(n_zones: int = 8, cols: int = 2, width_m: float = 2.44,
                         length_m: float = 2.44, paddock_id: str = "A",
                         reader: str = "1", location: str = "tube") -> Arena:
    """A single enclosure with one antenna per zone, zones on an even grid.

    Antenna N watches zone N. Defaults describe an 8 x 8 ft (2.44 m) lab
    enclosure; the zone centres sit in the middle of equal grid cells, which
    is only a starting point - edit the zones table to the real positions.
    """
    n_zones, cols = max(1, int(n_zones)), max(1, int(cols))
    rows = -(-n_zones // cols)
    dx, dy = width_m / cols, length_m / rows
    arena = Arena(paddocks=[Paddock(paddock_id, "", [reader], [], width_m, length_m)])
    for i in range(n_zones):
        r, c = divmod(i, cols)
        arena.zones.append(Zone(paddock_id, str(i + 1), round(dx * (c + 0.5), 4),
                                round(dy * (r + 0.5), 4)))
        arena.antennas.append(Antenna(reader, str(i + 1), str(i + 1), location))
    return arena


ARENA_PRESETS = {
    "Liddell 8-zone paddock (2x4, wall + floor antennas)":
        lambda: liddell_8_zone(1),
    "Two Liddell 8-zone paddocks (A = reader 1, B = reader 2)":
        lambda: liddell_8_zone(2),
    "One antenna per zone (8 zones, 8x8 ft enclosure)":
        lambda: one_antenna_per_zone(8, 2),
    "Empty": lambda: Arena(),
}


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #

# The animal fields the pipeline understands. Anything else in the metadata is
# ignored on the way in (the metadata file stays the place to look it up).
METADATA_FIELDS = ["name", "sex", "trial", "phase", "code",
                   "release_date", "release_time", "trap_date", "trap_time"]


@dataclass
class MetadataSpec:
    """Where the animal table is and which of its columns mean what.

    ``header_row`` is 1-based as a spreadsheet shows it; 0 means detect it.
    The LID workbook keeps storage notes on row 1 and its header on row 2,
    which is common enough in field metadata that a fixed "row 1" would be
    wrong more often than not.
    """
    path: str = ""
    sheet: str = ""
    header_row: int = 0
    columns: dict[str, str] = field(default_factory=dict)
    tag_columns: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Per-animal settings
# --------------------------------------------------------------------------- #

def animal_settings(raw: dict | None) -> dict:
    """One animal's overrides with every key present.

    Stored as plain dicts in the config (JSON-friendly), keyed by animal:

    ``start_mode`` / ``stop_mode``
        ``auto`` follows the run-wide rule (trim to the metadata release / trap
        time, or not at all); ``manual`` trims at ``start_time`` / ``stop_time``.
    ``dead_mode``
        ``none``; ``manual`` - drop everything after ``dead_time``; or
        ``last_entry`` - cut at the animal's last entry (``dead_time``, if
        given, is when the body was found: entries after it are ignored).
    ``lost_tags``
        ``[{tag, start, stop, note}]``: a tag off the animal from ``start``
        (to ``stop`` if it went back on). Reads of that tag inside are dropped.
    """
    out = {"start_mode": "auto", "start_time": "", "stop_mode": "auto",
           "stop_time": "", "dead_mode": "none", "dead_time": "",
           "lost_tags": [], "note": ""}
    if raw:
        out.update({k: v for k, v in raw.items() if k in out})
    out["lost_tags"] = [dict(t) for t in (out.get("lost_tags") or [])]
    return out


def is_default_animal(settings: dict) -> bool:
    return animal_settings(settings) == animal_settings(None)


# --------------------------------------------------------------------------- #
# The run config
# --------------------------------------------------------------------------- #

BOUT_RULES = ("gap_zone_day", "legacy_r")
CROSS_PADDOCK_POLICIES = ("keep", "drop", "fold")

# Analysis layers carried over from the earlier module. They work on zone_id
# keys and are tested, but they are interpretive (who displaced whom, who
# approached whom) rather than preprocessing, so they are off unless asked for.
OPTIONAL_LAYERS = {
    "gbi": False,
    "edgelist": False,
    "displacement": False,
    "hinde": False,
    "sna": False,
}


@dataclass
class RFIDConfig:
    """Everything needed to turn a dataset's downloads into its outputs."""

    name: str = ""
    download_dirs: list[str] = field(default_factory=list)
    include_subfolders: bool = False
    output_dir: str = ""                  # "" = <name>_FNT_analysis beside the downloads

    metadata: MetadataSpec = field(default_factory=MetadataSpec)
    arena: Arena = field(default_factory=Arena)

    # Readers log local wall-clock time. The timezone is used to measure real
    # elapsed seconds across a DST change; "" treats the clock as never
    # changing (right for a reader left on standard time all year).
    timezone: str = ""
    date_order: str = "MDY"               # how to read '05/06/2021'
    # Start of day 1, local wall clock. "" = auto: noon (day_start_time) on the
    # date of the earliest metadata release, else of the first read.
    day_origin: str = ""
    day_start_time: str = "12:00:00"

    bout_gap_s: float = 50.0
    min_bout_s: float = 1.0
    bout_rule: str = "gap_zone_day"
    # "ms" keeps the reader's milliseconds; "s" truncates to whole seconds, as
    # the original R pipeline effectively did by round-tripping through CSV.
    time_resolution: str = "ms"

    cross_paddock_policy: str = "keep"
    entrance_locations: list[str] = field(default_factory=lambda: ["wall"])
    trim_to_release: bool = False
    trim_to_trap: bool = False

    holder_share: float = 0.5
    holder_sex: str = "M"
    two_place_share: float = 0.8
    stationary_hour_fraction: float = 0.5

    animals: dict[str, dict] = field(default_factory=dict)
    layers: dict[str, bool] = field(default_factory=lambda: dict(OPTIONAL_LAYERS))
    write_reads: bool = True

    # --- persistence -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["arena"] = self.arena.to_dict()
        data["metadata"] = asdict(self.metadata)
        data["animals"] = {k: animal_settings(v) for k, v in self.animals.items()
                           if not is_default_animal(v)}
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RFIDConfig":
        data = dict(data or {})
        # a saved fnt_config.json wraps the settings with provenance
        if isinstance(data.get("config"), dict):
            data = dict(data["config"])
        if isinstance(data.get("arena"), dict):
            data["arena"] = Arena.from_dict(data["arena"])
        if isinstance(data.get("metadata"), dict):
            data["metadata"] = MetadataSpec(**_known(MetadataSpec, data["metadata"]))
        layers = dict(OPTIONAL_LAYERS)
        layers.update({k: bool(v) for k, v in (data.get("layers") or {}).items()
                       if k in OPTIONAL_LAYERS})
        data["layers"] = layers
        data["animals"] = {str(k): animal_settings(v)
                           for k, v in (data.get("animals") or {}).items()}
        return cls(**_known(cls, data))

    def animal(self, key: str) -> dict:
        return animal_settings(self.animals.get(key))

    def problems(self) -> tuple[list[str], list[str]]:
        """``(blocking, warnings)`` for the whole config."""
        import os
        blocking, warnings = self.arena.problems()
        if not self.download_dirs:
            blocking.append("No download folder chosen")
        for d in self.download_dirs:
            if not os.path.isdir(d):
                blocking.append(f"Download folder not found: {d}")
        if not self.metadata.path:
            blocking.append("No metadata file chosen")
        elif not os.path.isfile(self.metadata.path):
            blocking.append(f"Metadata file not found: {self.metadata.path}")
        elif not self.metadata.columns.get("name"):
            blocking.append("Metadata column mapping has no 'name' column")
        elif not self.metadata.tag_columns:
            blocking.append("Metadata column mapping has no tag column")
        if self.bout_gap_s <= 0:
            blocking.append("Bout gap must be greater than zero")
        if self.bout_rule not in BOUT_RULES:
            blocking.append(f"Unknown bout rule {self.bout_rule!r}")
        if self.cross_paddock_policy not in CROSS_PADDOCK_POLICIES:
            blocking.append(f"Unknown cross-paddock policy {self.cross_paddock_policy!r}")
        if self.time_resolution not in ("ms", "s"):
            blocking.append("time_resolution must be 'ms' or 's'")
        if self.timezone:
            try:
                import pytz
                pytz.timezone(self.timezone)
            except Exception:
                blocking.append(f"Unknown timezone {self.timezone!r}")
        if self.day_origin:
            import pandas as pd
            if pd.isna(pd.to_datetime(self.day_origin, errors="coerce")):
                blocking.append(f"Day origin is not a date/time: {self.day_origin!r}")
        if not 0 < self.holder_share < 1:
            blocking.append("Holder share must be between 0 and 1")
        return blocking, warnings


def get_default_config(preset: str | None = None) -> RFIDConfig:
    """A fresh config, optionally starting from one of the arena presets."""
    cfg = RFIDConfig()
    if preset:
        if preset not in ARENA_PRESETS:
            raise ValueError(f"Unknown arena preset {preset!r}. "
                             f"Available: {list(ARENA_PRESETS)}")
        cfg.arena = ARENA_PRESETS[preset]()
    return cfg


def get_available_presets() -> list[str]:
    return list(ARENA_PRESETS)
