"""Species library — the *body* an agent is issued, not the behaviour it shows.

A species card carries biophysical facts: how heavy the animal is, how long,
how fast it moves, how much it costs to run it, how good its nose is, and how
much scent it can produce. Those are things you could go and measure on a real
animal, and they are the honest content of "this is a prairie vole".

What a species card deliberately does NOT set
---------------------------------------------
Home-range size, territory radius, dominance rank, group structure. Those are
the *results* an ABMA experiment exists to produce — prescribing them would
answer the question before the run starts. Space use emerges from body size,
olfactory acuity, scent production, mark persistence (:class:`ScentParams`) and
competition. See :mod:`fnt.abma.core.scent`.

Personality (aggression, boldness, sociability, exploration) sits between the
two: it is a disposition, not a body fact and not an outcome. Cards ship a
*suggested distribution* per sex as a starting point, clearly editable — an
experimenter who has measured attack rates in their own colony should overwrite
them.

Values are typical adult figures for the literature-reported range, meant as a
defensible starting point rather than a citation. Every one is editable in the
Agent Builder, and a project records where its numbers came from (see
``project.SOURCES``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import AgentGroup, TraitProfile, Appearance

#: the 3D/2D views draw a body this long (cm) at size ×1
_REF_BODY_CM = 9.0


@dataclass
class Species:
    """One animal's biophysical card, plus suggested personality spread."""
    key: str
    name: str
    latin: str
    summary: str
    activity: str                     # nocturnal | diurnal | crepuscular
    body_length_cm: float
    base_speed: float                 # m/s when active
    metabolism: float                 # metabolic-rate multiplier
    smell_ability: float              # olfactory acuity, 0..1
    identity_signal: float            # individuality of the scent signature
    scent_rate: float                 # urine marks per hour it can produce
    mass_g: dict[str, str]            # sex -> distribution spec
    personality: dict[str, str] = field(default_factory=dict)
    genes: list[str] = field(default_factory=list)
    notes: str = ""

    # ---- body facts, for display next to the dials -------------------- #
    def body_rows(self, sex: str = "M") -> list[tuple[str, str]]:
        return [
            ("Body mass", f"{self.mass_g.get(sex, self.mass_g['M'])} g"),
            ("Body length", f"{self.body_length_cm:g} cm"),
            ("Travel speed", f"{self.base_speed:g} m/s"),
            ("Metabolic rate", f"×{self.metabolism:g}"),
            ("Olfactory acuity", f"{self.smell_ability:g}"),
            ("Scent signature", f"{self.identity_signal:g}"),
            ("Marking capacity", f"{self.scent_rate:g} marks/h"),
            ("Active phase", self.activity),
        ]

    def size_multiplier(self) -> float:
        return round(self.body_length_cm / _REF_BODY_CM, 3)


SPECIES: list[Species] = [
    Species(
        key="prairie_vole", name="Prairie vole", latin="Microtus ochrogaster",
        summary="Socially monogamous grassland vole; forms pair bonds and "
                "communal nests. The workhorse of social-neuroscience voles.",
        activity="crepuscular",
        body_length_cm=13.0, base_speed=0.14, metabolism=1.0,
        smell_ability=1.0, identity_signal=1.0, scent_rate=18.0,
        mass_g={"M": "N(45,4)", "F": "N(38,3)"},
        personality={"aggression": "U(0.15,0.40)", "boldness": "U(0.3,0.7)",
                     "sociability": "U(0.55,0.90)",
                     "exploration": "U(0.3,0.7)"},
        genes=["OXTR", "AVPR1A"],
        notes="Low baseline aggression and high sociability relative to "
              "meadow voles — expect tolerant, overlapping space use.",
    ),
    Species(
        key="meadow_vole", name="Meadow vole", latin="Microtus pennsylvanicus",
        summary="Promiscuous, largely solitary vole. Females defend space "
                "against other females; males range widely in the breeding "
                "season.",
        activity="crepuscular",
        body_length_cm=14.0, base_speed=0.15, metabolism=1.05,
        smell_ability=1.0, identity_signal=1.0, scent_rate=22.0,
        mass_g={"M": "N(44,5)", "F": "N(40,4)"},
        personality={"aggression": "U(0.35,0.70)", "boldness": "U(0.4,0.8)",
                     "sociability": "U(0.10,0.40)",
                     "exploration": "U(0.4,0.9)"},
        genes=["OXTR", "AVPR1A"],
        notes="The contrast species to prairie voles: higher aggression, "
              "lower sociability, so exclusive space use is more likely.",
    ),
    Species(
        key="house_mouse", name="House mouse", latin="Mus musculus",
        summary="Commensal mouse with strong, individually distinctive urine "
                "marking (MUPs). Males counter-mark heavily.",
        activity="nocturnal",
        body_length_cm=8.5, base_speed=0.16, metabolism=1.2,
        smell_ability=1.0, identity_signal=1.0, scent_rate=35.0,
        mass_g={"M": "N(22,3)", "F": "N(18,2.5)"},
        personality={"aggression": "U(0.35,0.75)", "boldness": "U(0.3,0.8)",
                     "sociability": "U(0.25,0.60)",
                     "exploration": "U(0.5,0.9)"},
        genes=["MUP"],
        notes="Highest marking capacity in the library — the species where "
              "mark persistence should matter most.",
    ),
    Species(
        key="deer_mouse", name="Deer mouse", latin="Peromyscus maniculatus",
        summary="Widespread wild mouse; agile, highly exploratory, and less "
                "reliant on scent marks for spacing than Mus.",
        activity="nocturnal",
        body_length_cm=9.0, base_speed=0.18, metabolism=1.15,
        smell_ability=0.9, identity_signal=0.8, scent_rate=14.0,
        mass_g={"M": "N(20,3)", "F": "N(19,3)"},
        personality={"aggression": "U(0.20,0.50)", "boldness": "U(0.4,0.9)",
                     "sociability": "U(0.20,0.50)",
                     "exploration": "U(0.6,1.0)"},
        notes="Use as a contrast to house mice when asking how much of "
              "spacing is scent-driven.",
    ),
    Species(
        key="generic_rodent", name="Generic rodent", latin="—",
        summary="Neutral mid-sized rodent with every dial at its midpoint. "
                "Start here when you want to tune everything yourself.",
        activity="nocturnal",
        body_length_cm=9.0, base_speed=0.12, metabolism=1.0,
        smell_ability=1.0, identity_signal=1.0, scent_rate=20.0,
        mass_g={"M": "N(40,4)", "F": "N(36,4)"},
        personality={"aggression": "U(0.3,0.6)", "boldness": "U(0.3,0.7)",
                     "sociability": "U(0.3,0.7)", "exploration": "U(0.3,0.7)"},
        notes="No species assumptions baked in.",
    ),
]

_BY_KEY = {s.key: s for s in SPECIES}


def get_species(key: str) -> Species | None:
    return _BY_KEY.get(key)


def species_names() -> list[str]:
    return [s.name for s in SPECIES]


def by_name(name: str) -> Species | None:
    return next((s for s in SPECIES if s.name == name), None)


def apply_species(group: AgentGroup, sp: Species,
                  personality: bool = True) -> AgentGroup:
    """Stamp a species' BODY onto ``group`` (in place) and return it.

    Body facts overwrite unconditionally — they are what the species *is*.
    Personality distributions are only suggested, and are skipped when
    ``personality`` is False so an experimenter who has already tuned
    dispositions can swap the body underneath them without losing that work.

    Home-range size is never set: see the module docstring.
    """
    t: TraitProfile = group.traits
    t.body_length_cm = sp.body_length_cm
    t.base_speed = sp.base_speed
    t.metabolism = sp.metabolism
    t.smell_ability = sp.smell_ability
    t.identity_signal = sp.identity_signal
    t.scent_rate = sp.scent_rate

    dists = dict(group.dists or {})
    mass_spec = sp.mass_g.get(group.sex, sp.mass_g.get("M"))
    if mass_spec:
        dists["mass"] = mass_spec
    if personality:
        dists.update(sp.personality)
    # a prescribed home-range radius is meaningless once marks drive spacing
    dists.pop("home_range_r", None)
    group.dists = dists

    group.species = sp.name
    ap = group.appearance or Appearance()
    ap.size = sp.size_multiplier()
    group.appearance = ap
    return group


def build_group(sp: Species, label: str, sex: str = "M",
                count: int = 4) -> AgentGroup:
    """A ready-to-add agent type for ``sp`` — what a species card produces."""
    g = AgentGroup(label=label, species=sp.name, sex=sex, count=count)
    return apply_species(g, sp)
