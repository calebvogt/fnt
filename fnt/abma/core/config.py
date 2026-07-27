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
class Pole:
    """A vertical cylindrical structure (post/pillar) in the arena.

    x, y   : centre position (metres)
    radius : cylinder radius (metres)
    height : cylinder height (metres) — may exceed the wall height
    """
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0762   # 6" diameter
    height: float = 2.1336   # 7 ft
    label: str = ""


@dataclass
class AntennaBox:
    """A UWB antenna enclosure mounted on a pole (weatherproof box).

    x, y : centre position (metres, usually a pole's position)
    z    : centre height above ground (metres)
    w,d,h: width (x), depth (y), height (z) in metres
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 1.8288    # 6 ft mount height (box centre)
    w: float = 0.2032    # 8"
    d: float = 0.1016    # 4"
    h: float = 0.2032    # 8"
    style: str = "box"   # 'box' (weatherproof enclosure) | 'bare' (open antenna)
    label: str = ""


@dataclass
class Cable:
    """A PoE daisy-chain run from a gateway through its antennas.

    gateway : the gateway antenna's label (e.g. "1 (G)")
    arm     : "A" | "B" (a gateway has two arms)
    nodes   : ordered [[x, y, z], …] box centres, starting at the gateway.
    """
    gateway: str = ""
    arm: str = "A"
    nodes: list[list[float]] = field(default_factory=list)


@dataclass
class AntennaLayout:
    """A named set of antenna boxes (a selectable UWB deployment)."""
    name: str = "antennas"
    antennas: list[AntennaBox] = field(default_factory=list)
    cables: list[Cable] = field(default_factory=list)


@dataclass
class WaterTower:
    """A free-standing water tower (small cylinder). Default 1 qt: 6" dia, 8" tall."""
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0762   # 6" diameter
    height: float = 0.2032   # 8"
    label: str = ""


@dataclass
class GrassSpec:
    """Ground-cover appearance for outdoor sites.

    This is a *visual* model, not a botanical stem count: ``density`` is how
    many blades we draw per m², chosen to reproduce the fractional cover seen
    from above. ``patchiness`` clumps them (0 = uniform, 1 = strongly clumped,
    leaving bare ground between tufts) and ``dry_fraction`` is the share drawn
    straw-coloured rather than green.

    ``cover_map`` optionally carries the site's large-scale cover pattern as a
    coarse grid of relative density (0-1). It is stored in ARENA orientation:
    row 0 = South, last row = North; column 0 = West, last column = East
    (matching +y = N, +x = E), so a map traced off a north-up aerial photo must
    be flipped vertically before being stored here. Empty = uniform.
    """
    density: float = 44.0        # blades drawn per m²
    h_min: float = 0.0508        # 2"
    h_max: float = 0.1016        # 4"
    dry_fraction: float = 0.0    # 0 = all green, 1 = all straw
    patchiness: float = 0.0      # 0 = uniform, 1 = strongly clumped
    cover_map: list = field(default_factory=list)   # rows S->N, cols W->E


@dataclass
class Hut:
    """A small acrylic shelter set on the floor.

    kind  : 'tube' (hollow rectangular tube, open both ends) | 'dome'
            (half dome with entrance arches)
    w,d,h : tube = length × width × height; dome = diameter × – × height
    angle : rotation about z, degrees (tube long axis)
    """
    kind: str = "tube"
    x: float = 0.0
    y: float = 0.0
    w: float = 0.1524    # 6" tube length / 5" dome diameter
    d: float = 0.1016    # 4" tube width
    h: float = 0.1016    # 4" tube height / dome height
    thickness: float = 0.00635   # 1/4" acrylic
    angle: float = 0.0
    label: str = ""


@dataclass
class ResourceZone:
    """A ground resource station (box) holding nesting material + food.

    entrance : "N" | "S" | "E" | "W" — side carrying the doorway.
    """
    x: float = 0.0
    y: float = 0.0
    w: float = 0.762    # 30" (east-west)
    d: float = 0.508    # 20" (north-south)
    h: float = 0.4318   # 17" tall
    entrance: str = "E"
    hole: float = 0.0762  # 3" entrance hole (diameter, vole-passable)
    label: str = ""


@dataclass
class PolicyParams:
    """Free parameters of the rule-based movement policy.

    These are *free* in the provenance sense — not measured from any animal,
    just weights that trade off competing drives. Keeping them in the config
    (rather than as hidden constants) is what makes them sweepable in a study
    and visible as the tuned quantities they are.
    """
    k_home: float = 1.0         # home-range spring gain (when satiated)
    # A hungry animal leaves its home range to forage and returns after: home
    # fidelity is relaxed in proportion to need. Without this, fidelity has to
    # be traded off globally against foraging, and no single value works —
    # measured: a large sparse arena needs weak fidelity to reach dispersed
    # resources, while a small one needs strong fidelity to stay on nearby
    # ones, and each setting starves the other case.
    forage_releases_home: float = 0.85   # 0 = never relax, 1 = fully relax
    k_resource: float = 1.6     # resource-seeking gain
    k_social: float = 1.0       # social attraction/repulsion gain
    k_territory: float = 2.0    # scent-marked territory avoidance gain
    k_random: float = 0.5       # exploratory noise gain
    perception_r: float = 0.6   # neighbour perception radius (m)
    forage_threshold: float = 0.5   # hunger/thirst at which seeking switches on


@dataclass
class ArenaConfig:
    width: float = 2.2
    height: float = 2.2
    units: str = "m"
    boundary: str = "reflective"  # 'reflective' | 'absorbing' | 'wrap'
    objects: list[ResourceObject] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    poles: list[Pole] = field(default_factory=list)   # vertical posts/pillars
    water_towers: list[WaterTower] = field(default_factory=list)
    resource_zones: list[ResourceZone] = field(default_factory=list)
    huts: list[Hut] = field(default_factory=list)      # acrylic tube / dome huts
    antennas: list[AntennaBox] = field(default_factory=list)  # primary UWB set
    antenna_layouts: list[AntennaLayout] = field(default_factory=list)  # cyclable
    wall_height: float = 0.0       # 3D wall height (m); 0 = flat/open arena
    wall_thickness: float = 0.005  # 3D wall thickness (m)
    ground: str = "floor"          # 'floor' | 'grass' (outdoor field site)
    grass: GrassSpec = field(default_factory=GrassSpec)   # ground-cover look
    oriented: bool = False         # geographically aligned (+y=N,+x=E) -> compass

    def objects_of(self, kind: str) -> list[ResourceObject]:
        return [o for o in self.objects if o.kind == kind]

    def antenna_sets(self):
        """Selectable UWB layouts as (name, boxes, cables). Falls back to ``antennas``."""
        if self.antenna_layouts:
            return [(l.name, l.antennas, l.cables) for l in self.antenna_layouts]
        if self.antennas:
            return [("antennas", self.antennas, [])]
        return []


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
class Coupling:
    """One row of the attribute-interaction table (condition dynamics).

    ``target += gain × source × (dt / hour)`` each step, or with ``effect='set'``
    the target is set to ``gain`` wherever the source is active. Gains are per-hour.
      source   : time | movement | activity | crowding | on_food | on_water |
                 mass | metabolism | fed | hydrated | rested |
                 energy | hunger | thirst | stress | health
      target   : energy | hunger | thirst | stress | health
      scale_by : none | mass | activity | metabolism
      only_when: always | source_high | source_low   (gated by ``threshold``)
    """
    source: str = "time"
    target: str = "energy"
    effect: str = "rate"         # 'rate' | 'set'
    gain: float = 0.0
    scale_by: str = "none"
    only_when: str = "always"
    threshold: float = 0.5
    note: str = ""


def default_dynamics() -> list["Coupling"]:
    """The built-in metabolic model as editable interaction rules.

    Reproduces sensible physiology: needs rise (faster when moving), moving and
    metabolism spend energy, being well-fed/hydrated restores it, deprivation
    drains energy and eventually erodes health; crowding raises stress.
    """
    C = Coupling
    return [
        C("time", "hunger", "rate", 0.15, note="hunger rises over time"),
        C("movement", "hunger", "rate", 1.5, note="moving builds hunger"),
        C("on_food", "hunger", "set", 0.0, note="eating resets hunger"),
        C("time", "thirst", "rate", 0.18, note="thirst rises over time"),
        C("movement", "thirst", "rate", 1.8, note="moving builds thirst"),
        C("on_water", "thirst", "set", 0.0, note="drinking resets thirst"),
        C("fed", "energy", "rate", 0.5, note="recover energy when well-fed"),
        C("hydrated", "energy", "rate", 0.25, note="recover energy when hydrated"),
        C("time", "energy", "rate", -0.10, scale_by="metabolism",
          note="baseline metabolism"),
        C("movement", "energy", "rate", -1.2, scale_by="mass",
          note="locomotion cost ∝ mass × speed"),
        C("hunger", "energy", "rate", -0.4, only_when="source_high", threshold=0.5,
          note="hunger drains energy"),
        C("thirst", "energy", "rate", -0.4, only_when="source_high", threshold=0.5,
          note="thirst drains energy"),
        C("energy", "health", "rate", 0.05, only_when="source_high", threshold=0.3,
          note="heal when energy is high"),
        C("hunger", "health", "rate", -0.15, only_when="source_high",
          threshold=0.9, note="starvation erodes health"),
        C("thirst", "health", "rate", -0.15, only_when="source_high",
          threshold=0.9, note="dehydration erodes health"),
        C("stress", "health", "rate", -0.03, note="chronic stress erodes health"),
        C("crowding", "stress", "rate", 0.006, note="crowding raises stress"),
        C("time", "stress", "rate", -0.05, note="stress decays when calm"),
    ]


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    groups: list[AgentGroup] = field(default_factory=list)
    interventions: list[Intervention] = field(default_factory=list)
    dynamics: list[Coupling] = field(default_factory=default_dynamics)

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
    # movement couplings (kept out of the condition-dynamics table since they
    # affect speed, not a condition variable)
    energy_speed_coupling: float = 0.6  # how much low energy slows movement (0=none)
    rest_speed_factor: float = 0.15     # speed × this when satiated near home (1=off)
    policy: PolicyParams = field(default_factory=PolicyParams)

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
            poles=[Pole(**p) for p in arena_d.get("poles", [])],
            water_towers=[WaterTower(**w)
                          for w in arena_d.get("water_towers", [])],
            resource_zones=[ResourceZone(**z)
                            for z in arena_d.get("resource_zones", [])],
            huts=[Hut(**hh) for hh in arena_d.get("huts", [])],
            antennas=[AntennaBox(**a) for a in arena_d.get("antennas", [])],
            antenna_layouts=[
                AntennaLayout(
                    name=l.get("name", "antennas"),
                    antennas=[AntennaBox(**a) for a in l.get("antennas", [])],
                    cables=[Cable(gateway=c.get("gateway", ""),
                                  arm=c.get("arm", "A"),
                                  nodes=[list(n) for n in c.get("nodes", [])])
                            for c in l.get("cables", [])])
                for l in arena_d.get("antenna_layouts", [])],
            wall_height=arena_d.get("wall_height", 0.0),
            wall_thickness=arena_d.get("wall_thickness", 0.005),
            ground=arena_d.get("ground", "floor"),
            grass=GrassSpec(**arena_d.get("grass", {})),
            oriented=arena_d.get("oriented", False),
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
            "energy_speed_coupling", "rest_speed_factor",
        }
        scalars = {k: d[k] for k in known if k in d}
        interventions = [Intervention(**iv) for iv in d.get("interventions", [])]
        dynamics = ([Coupling(**c) for c in d["dynamics"]]
                    if "dynamics" in d else default_dynamics())
        return ExperimentConfig(arena=arena, groups=groups,
                                interventions=interventions, dynamics=dynamics,
                                policy=PolicyParams(**d.get("policy", {})),
                                **scalars)

    @staticmethod
    def from_json(path: str) -> "ExperimentConfig":
        with open(path) as fh:
            return ExperimentConfig.from_dict(json.load(fh))

    def total_agents(self) -> int:
        return sum(g.count for g in self.groups)


def blank_experiment() -> ExperimentConfig:
    """A neutral starting point — an empty arena and NO agents.

    ABMA is a general-purpose ABM sandbox; this is what it opens with. The user
    shapes the arena, defines their own agent type(s) in Build & Add Agents, and
    only then do agents appear (and animate) in the preview.
    """
    arena = ArenaConfig(width=2.0, height=2.0, boundary="reflective", objects=[])
    return ExperimentConfig(name="experiment", arena=arena, groups=[],
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
