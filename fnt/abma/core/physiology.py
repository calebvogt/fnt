"""Energy in, energy out; water in, urine out.

Condition bars in ABMA are not free-floating dials — they are *readouts of
stores*. An animal has a somatic energy store, a body-water pool and a bladder,
each sized by its body mass. Eating puts kilojoules in; running and fighting
take them out. Drinking puts millilitres in; evaporative loss and urine take
them out. The 0-100 bars the user sees are those stores expressed as a
percentage of capacity.

Why bother with real units
--------------------------
Because it makes the interesting couplings automatic instead of hand-tuned:

* Scent marking is limited by **bladder volume**, which is filled by drinking.
  An animal that cannot reach water cannot mark, so removing a water tower is a
  territorial manipulation, not just a thirst manipulation.
* A heavier animal carries a bigger store *and* burns more moving it, so body
  mass trades off against range size without anyone specifying a range.
* Food is worth what it contains: swapping standard chow for a high-fat diet
  changes how long a foraging trip has to be, through one number.

Hunger and thirst are therefore *derived*: hunger is the energy the animal is
missing, thirst the water it is missing. They are displayed, logged, and used
to drive foraging, but nothing writes to them directly.

Every rate here is editable (see :class:`PhysiologyParams`), and the shipped
defaults are the "Standard" preset — deliberately flat, round numbers rather
than a fit to any one species.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: body mass (g) the basal rate is quoted for; scaling is Kleiber (M^0.75)
REF_MASS_G = 40.0


# --------------------------------------------------------------------------- #
# Food
# --------------------------------------------------------------------------- #
@dataclass
class FoodType:
    """What a gram of a given food is worth.

    ``energy_density`` is kJ per gram of food as eaten; ``water_fraction`` is
    the share of that gram which is water (fresh greens hydrate, dry chow does
    not). ``palatability`` scales how fast an animal will eat it.
    """
    key: str
    name: str
    energy_density: float        # kJ per gram
    water_fraction: float = 0.10  # 0..1 of mass that is water
    palatability: float = 1.0     # intake-rate multiplier
    description: str = ""


FOOD_TYPES: list[FoodType] = [
    FoodType("standard_chow", "Standard chow", 15.0, 0.10, 1.0,
             "Dry laboratory pellet — the default provisioning."),
    FoodType("high_fat", "High-fat chow", 22.0, 0.08, 1.15,
             "Energy-dense diet: shorter foraging trips sustain the same animal."),
    FoodType("low_energy", "Low-energy chow", 8.0, 0.12, 0.9,
             "Dilute diet: animals must spend more time at the food source."),
    FoodType("seeds", "Seeds", 24.0, 0.06, 1.2,
             "Natural high-energy forage, very little water."),
    FoodType("greens", "Fresh greens", 2.5, 0.85, 1.0,
             "Low energy but strongly hydrating — can substitute for water."),
]

_FOOD_BY_KEY = {f.key: f for f in FOOD_TYPES}
DEFAULT_FOOD = "standard_chow"


def get_food(key: str) -> FoodType:
    """Look up a food type, falling back to standard chow."""
    return _FOOD_BY_KEY.get(key or "", _FOOD_BY_KEY[DEFAULT_FOOD])


def food_names() -> list[str]:
    return [f.name for f in FOOD_TYPES]


def food_by_name(name: str) -> FoodType | None:
    return next((f for f in FOOD_TYPES if f.name == name), None)


# --------------------------------------------------------------------------- #
# Rates and capacities
# --------------------------------------------------------------------------- #
@dataclass
class PhysiologyParams:
    """Tunable energy/water budget. Defaults are the "Standard" preset.

    Capacities are per gram of body mass, so a 20 g mouse and a 45 g vole get
    proportionate stores without either being configured separately.
    """
    enabled: bool = False        # off unless asked for, so old runs reproduce

    # ---- capacities (per gram of body mass) ---------------------------- #
    # A small rodent carries roughly a day of usable reserve, which is what
    # makes hunger bite on the timescale of an enclosure experiment: too large
    # a store and animals coast for days without ever needing to forage.
    energy_store_kj_per_g: float = 2.0    # usable somatic energy per gram
    # Readily-exchangeable water, ~a quarter of body mass. Sized so a missed
    # drinking opportunity is a setback rather than a death sentence — with a
    # pool this small relative to daily loss, animals in a large enclosure die
    # of thirst before they can walk to the next water tower.
    water_ml_per_g: float = 0.25
    bladder_ml_per_g: float = 0.012       # bladder volume per gram

    # ---- intake --------------------------------------------------------- #
    feed_rate_g_min: float = 0.15         # grams eaten per minute at food
    drink_rate_ml_min: float = 0.35       # mL drunk per minute at water

    # ---- energy expenditure --------------------------------------------- #
    # Balanced so a fed animal holds station on ~15-20 min of feeding a day,
    # locomotion is a visible slice of the budget (not a rounding error), and
    # a cut-off animal empties its store in ~2 days before health goes.
    basal_kj_h: float = 1.5               # kJ/h for a REF_MASS_G animal at rest
    # Charged per metre travelled. Agents cover a lot of ground in this
    # movement model, so this is calibrated to make locomotion roughly a
    # quarter of the daily budget rather than to a treadmill measurement.
    locomotion_kj_per_kg_m: float = 0.020  # kJ per kg of body mass per metre
    fight_kj: float = 2.5                 # kJ spent by a contestant per contest
    mate_kj: float = 0.8                  # kJ spent per mating

    # ---- water balance --------------------------------------------------- #
    water_loss_ml_h: float = 0.12         # evaporative/faecal loss per hour
    urine_fraction: float = 0.45          # share of drunk water routed to bladder
    mark_volume_ul: float = 22.0          # urine spent per scent mark (µL)

    # ---- consequences of an empty store ---------------------------------- #
    starvation_health_h: float = 6.0      # health lost per hour at zero energy
    # Dehydration is faster than starvation but not instant: a fully parched
    # animal has about a day before it is gone, so a long trip to water is
    # survivable and thirst drives behaviour rather than just killing.
    dehydration_health_h: float = 4.0

    # ---- capacities, resolved for an actual body mass -------------------- #
    def energy_capacity(self, mass_g):
        return mass_g * self.energy_store_kj_per_g

    def water_capacity(self, mass_g):
        return mass_g * self.water_ml_per_g

    def bladder_capacity(self, mass_g):
        return mass_g * self.bladder_ml_per_g

    def basal_kj(self, mass_g, metabolism, dt_s):
        """Resting cost over ``dt_s``, Kleiber-scaled (M^0.75) from REF_MASS_G."""
        return (self.basal_kj_h * (mass_g / REF_MASS_G) ** 0.75
                * metabolism * dt_s / 3600.0)

    def locomotion_kj(self, mass_g, metres):
        return self.locomotion_kj_per_kg_m * (mass_g / 1000.0) * metres

    def marks_per_full_bladder(self, mass_g):
        """How many marks a full bladder buys — the emergent marking budget."""
        cap_ul = self.bladder_capacity(mass_g) * 1000.0
        return cap_ul / max(1e-9, self.mark_volume_ul)


#: Named starting points. "Standard" is the flat, neutral setting; the others
#: exist so the shape of the trade-off can be explored without hand-editing
#: every rate.
PHYSIOLOGY_PRESETS: dict[str, dict] = {
    "Standard": {},
    "Lean / fast burn": {
        "basal_kj_h": 4.6, "locomotion_kj_per_kg_m": 0.030,
        "feed_rate_g_min": 0.11,
    },
    "Thrifty / slow burn": {
        "basal_kj_h": 2.2, "locomotion_kj_per_kg_m": 0.013,
        "feed_rate_g_min": 0.07,
    },
    "Arid (water-limited)": {
        "water_loss_ml_h": 0.34, "drink_rate_ml_min": 0.28,
        "urine_fraction": 0.30,
    },
}


def preset_params(name: str) -> PhysiologyParams:
    """Build :class:`PhysiologyParams` for a named preset."""
    p = PhysiologyParams(enabled=True)
    for k, v in PHYSIOLOGY_PRESETS.get(name, {}).items():
        setattr(p, k, v)
    return p


def preset_name_for(p: PhysiologyParams) -> str:
    """Which preset ``p`` matches, or "Custom"."""
    for name in PHYSIOLOGY_PRESETS:
        ref = preset_params(name)
        if all(getattr(ref, f) == getattr(p, f)
               for f in PhysiologyParams.__dataclass_fields__
               if f != "enabled"):
            return name
    return "Custom"
