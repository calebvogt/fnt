"""ABMA experiment configuration schema.

These dataclasses are the single source of truth for an experiment. They are
shared by the headless engine and the PyQt GUI, and round-trip to/from JSON so a
whole experiment can be saved, version-controlled, and re-run.

Design principle: the simulator emits data in the *same* schema FNT produces for
real Ultra-Wideband tracking (uwb_<trial>_processed.csv), so every downstream
FNT / R analysis runs unchanged on simulated animals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------- #
# Arena
# --------------------------------------------------------------------------- #
@dataclass
class ResourceObject:
    """A point resource / structure placed in the arena.

    kind: 'nest' | 'food' | 'water'
    x, y: centre position in arena units (metres)
    radius: interaction radius (metres)
    """
    kind: str = "nest"
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.15
    capacity: float = 1.0
    label: str = ""


@dataclass
class Zone:
    """A named rectangular region of interest.

    Used for visualization and zone-occupancy analysis (e.g. the Open Field Test
    centre zone). ``x, y`` is the lower-left corner; ``w, h`` the size (metres).
    ``role`` tags it for analysis/colour ('center' | 'periphery' | 'roi').
    """
    name: str = "zone"
    x: float = 0.0
    y: float = 0.0
    w: float = 0.1
    h: float = 0.1
    role: str = "roi"


@dataclass
class ArenaConfig:
    width: float = 2.2
    height: float = 2.2
    units: str = "m"
    boundary: str = "reflective"  # 'reflective' | 'absorbing' | 'wrap'
    objects: list[ResourceObject] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    wall_height: float = 0.0       # 3D wall height (m); 0 = flat/open arena
    wall_thickness: float = 0.005  # 3D wall thickness (m)

    def objects_of(self, kind: str) -> list[ResourceObject]:
        return [o for o in self.objects if o.kind == kind]


# --------------------------------------------------------------------------- #
# Agent biology
# --------------------------------------------------------------------------- #
@dataclass
class TraitProfile:
    """Baseline behavioural traits. Most are 0..1 dimensionless dials.

    smell_ability   : olfactory sensitivity. 0 = fully anosmic.
    identity_signal : how identifiable this animal's scent is to others.
                      0 = no individual signature (e.g. MUP knockout).
    base_speed      : locomotor speed (m/s) when active.
    home_range_r    : preferred home-range radius (m); sets territory size.
    """
    aggression: float = 0.5
    boldness: float = 0.5
    sociability: float = 0.5
    exploration: float = 0.5
    smell_ability: float = 1.0
    identity_signal: float = 1.0
    base_speed: float = 0.12
    home_range_r: float = 0.55
    mass: float = 40.0        # body mass (g); initial value, drifts over the run
    metabolism: float = 1.0   # metabolic-rate multiplier
    turn_rate: float = 0.5    # heading jitter per step (rad) — path tortuosity
    wander: float = 0.5       # random-walk drive gain — exploratory movement


@dataclass
class Genotype:
    """Mapping of gene name -> status: 'WT' | 'HET' | 'KO'."""
    genes: dict[str, str] = field(default_factory=dict)


@dataclass
class Treatment:
    """A pharmacological treatment delivered relative to arena release.

    drug       : 'none' | 'saline' | 'methimazole'
    dose       : 0..1 normalized severity (methimazole: 1.0 -> full anosmia).
    day_offset : days relative to release when delivered (negative = before).
    """
    drug: str = "none"
    dose: float = 0.0
    day_offset: float = -5.0


@dataclass
class Appearance:
    """How an agent type is drawn in the God's-eye view."""
    shape: str = "rodent"   # 'rodent' | 'blob' | 'bird'
    color: str = ""         # hex '#4a90d9'; '' = auto (blue male / pink female)
    size: float = 1.0       # size multiplier


@dataclass
class AgentGroup:
    """An agent *type* — its look, biology, movement, and how many to add.

    ``dists`` maps a trait name to a distribution spec string (e.g. "N(33,3)")
    that each founder samples individually; traits absent from ``dists`` use the
    scalar in ``traits``. This is how a cohort is seeded from domain knowledge —
    e.g. 8 males with mass ~ N(33, 3) g.
    """
    label: str = "group"
    species: str = "prairie"
    sex: str = "F"  # 'F' | 'M'
    count: int = 4
    genotype: Genotype = field(default_factory=Genotype)
    treatment: Treatment = field(default_factory=Treatment)
    traits: TraitProfile = field(default_factory=TraitProfile)
    appearance: Appearance = field(default_factory=Appearance)
    dists: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
@dataclass
class Intervention:
    """A scheduled change to a target's attribute at a given day.

    Generalises "deliver a treatment": e.g. induce anosmia on day 3 is
    ``Intervention(at_day=3, target="all", attribute="smell_ability",
    op="scale", value=0.0)``. ``target`` is a group label, a sexid, or "all".
    """
    at_day: float = 3.0
    target: str = "all"
    attribute: str = "smell_ability"
    op: str = "scale"          # 'set' | 'scale' | 'add'
    value: float = 0.0
    label: str = ""


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    groups: list[AgentGroup] = field(default_factory=list)
    interventions: list[Intervention] = field(default_factory=list)

    days: float = 10.0                 # simulated duration (days)
    dt: float = 2.0                    # integration step (simulated seconds)
    record_interval: float = 10.0      # seconds between recorded position samples
    n_trials: int = 3
    seed: int = 0

    # Circadian activity
    day_start_hour: float = 6.0        # local hour when 'day' begins
    day_activity: float = 0.7          # activity multiplier during day
    night_activity: float = 1.3        # activity multiplier during night

    # Biology options
    individual_variation: float = 0.0  # per-agent trait jitter SD (0 = clones)
    enable_mortality: bool = False     # allow starvation death

    # Output / execution
    trial_prefix: str = "S"            # trial id prefix, e.g. S001, S002 ...
    start_datetime: str = "2025-11-07T18:00:00"  # release timestamp (local)
    parallel: bool = False
    n_workers: int = 2

    # ---- serialization ---------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ExperimentConfig":
        arena_d = d.get("arena", {})
        arena = ArenaConfig(
            width=arena_d.get("width", 2.2),
            height=arena_d.get("height", 2.2),
            units=arena_d.get("units", "m"),
            boundary=arena_d.get("boundary", "reflective"),
            objects=[ResourceObject(**o) for o in arena_d.get("objects", [])],
            zones=[Zone(**z) for z in arena_d.get("zones", [])],
            wall_height=arena_d.get("wall_height", 0.0),
            wall_thickness=arena_d.get("wall_thickness", 0.005),
        )
        groups = []
        for g in d.get("groups", []):
            groups.append(
                AgentGroup(
                    label=g.get("label", "group"),
                    species=g.get("species", "prairie"),
                    sex=g.get("sex", "F"),
                    count=g.get("count", 4),
                    genotype=Genotype(**g.get("genotype", {"genes": {}})),
                    treatment=Treatment(**g.get("treatment", {})),
                    traits=TraitProfile(**g.get("traits", {})),
                    appearance=Appearance(**g.get("appearance", {})),
                    dists=dict(g.get("dists", {})),
                )
            )
        known = {
            "name", "days", "dt", "record_interval", "n_trials", "seed",
            "day_start_hour", "day_activity", "night_activity", "trial_prefix",
            "start_datetime", "parallel", "n_workers",
            "individual_variation", "enable_mortality",
        }
        scalars = {k: d[k] for k in known if k in d}
        interventions = [Intervention(**iv) for iv in d.get("interventions", [])]
        return ExperimentConfig(arena=arena, groups=groups,
                                interventions=interventions, **scalars)

    @staticmethod
    def from_json(path: str) -> "ExperimentConfig":
        with open(path) as fh:
            return ExperimentConfig.from_dict(json.load(fh))

    def total_agents(self) -> int:
        return sum(g.count for g in self.groups)


def blank_experiment() -> ExperimentConfig:
    """A neutral starting point — an empty arena and one generic agent type.

    ABMA is a general-purpose ABM sandbox; this is what it opens with. The user
    shapes the arena, defines their own agent type(s), and scripts the trial.
    """
    arena = ArenaConfig(width=2.0, height=2.0, boundary="reflective", objects=[])
    groups = [AgentGroup(label="agent_type_1", species="agent", sex="M", count=8,
                         genotype=Genotype({}), treatment=Treatment("none", 0.0, 0.0),
                         traits=TraitProfile())]
    return ExperimentConfig(name="experiment", arena=arena, groups=groups,
                            days=2.0, n_trials=1,
                            start_datetime="2025-01-01T12:00:00")


def default_vole_experiment() -> ExperimentConfig:
    """The canonical 2.2x2.2 m, 4M/4F, saline-vs-methimazole vole design.

    Mirrors the user's real prairie-vole enclosure experiment so a first run
    reproduces a familiar setup out of the box.
    """
    w = h = 2.2
    arena = ArenaConfig(
        width=w, height=h, boundary="reflective",
        objects=[
            ResourceObject("nest", 0.4, 0.4, 0.15, label="nest_A"),
            ResourceObject("nest", 1.8, 0.4, 0.15, label="nest_B"),
            ResourceObject("nest", 0.4, 1.8, 0.15, label="nest_C"),
            ResourceObject("nest", 1.8, 1.8, 0.15, label="nest_D"),
            ResourceObject("food", 1.1, 1.1, 0.18, label="food_1"),
            ResourceObject("water", 1.1, 0.3, 0.15, label="water_1"),
            ResourceObject("water", 1.1, 1.9, 0.15, label="water_1b"),
        ],
    )
    saline = Treatment(drug="saline", dose=0.0, day_offset=-5.0)
    methim = Treatment(drug="methimazole", dose=1.0, day_offset=-5.0)
    groups = [
        AgentGroup("saline_F", "prairie", "F", 4,
                   Genotype({}), saline,
                   TraitProfile(aggression=0.3, sociability=0.6, mass=38.0),
                   dists={"mass": "N(38,3)"}),
        AgentGroup("saline_M", "prairie", "M", 4,
                   Genotype({}), saline,
                   TraitProfile(aggression=0.6, home_range_r=0.7, mass=45.0),
                   dists={"mass": "N(45,4)"}),
    ]
    return ExperimentConfig(
        name="vole_saline_vs_methimazole",
        arena=arena, groups=groups,
        days=10.0, dt=2.0, record_interval=10.0, n_trials=3,
        start_datetime="2025-11-07T18:00:00",
    )
