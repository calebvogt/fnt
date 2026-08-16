"""Energy and water as conserved quantities, and the 0-100 bar scale."""
import numpy as np
import pandas as pd
import pytest

from fnt.abma.core.config import (
    ExperimentConfig, ArenaConfig, ResourceObject, ScentParams, Coupling,
    AgentGroup, Genotype, Treatment, TraitProfile, default_dynamics,
)
from fnt.abma.core.physiology import (
    PhysiologyParams, FOOD_TYPES, get_food, food_by_name, preset_params,
    preset_name_for, PHYSIOLOGY_PRESETS, REF_MASS_G,
)
from fnt.abma.core.simulation import Simulation

DAY = 86400.0
BARS = ("energy", "hunger", "thirst", "stress", "health", "bladder")


def _cfg(food=True, water=True, food_type="standard_chow", n=4, mass=45.0,
         amount_g=0.0, mortality=False, scent=False, days=5.0, seed=2):
    objs = []
    if food:
        objs.append(ResourceObject("food", 1.1, 1.1, 0.18, label="chow",
                                   food_type=food_type, amount_g=amount_g))
    if water:
        objs.append(ResourceObject("water", 1.1, 0.3, 0.15, label="w1"))
        objs.append(ResourceObject("water", 1.1, 1.9, 0.15, label="w2"))
    arena = ArenaConfig(width=2.2, height=2.2, objects=objs)
    g = [AgentGroup("M", "prairie", "M", n, Genotype({}), Treatment(),
                    TraitProfile(mass=mass))]
    return ExperimentConfig(
        arena=arena, groups=g, seed=seed, days=days,
        enable_mortality=mortality,
        physiology=preset_params("Standard"),
        scent=ScentParams(enabled=scent))


def _run(cfg, days=None, dt=8.0):
    sim = Simulation(cfg, 0)
    days = cfg.days if days is None else days
    for k in range(int(days * DAY / dt)):
        sim.step(k * dt, dt)
    return sim


# --------------------------------------------------------------------------- #
# Food types
# --------------------------------------------------------------------------- #
def test_food_library_is_sane():
    assert len(FOOD_TYPES) >= 4
    for f in FOOD_TYPES:
        assert f.energy_density > 0
        assert 0.0 <= f.water_fraction <= 1.0
        assert f.palatability > 0
        assert f.name and f.description
    assert get_food("nope").key == "standard_chow", "unknown food falls back"
    assert food_by_name("Seeds").key == "seeds"


def test_energy_density_changes_what_a_gram_is_worth():
    """Same grams eaten, more energy — that is what a rich diet means."""
    lean = _run(_cfg(food_type="low_energy"))
    rich = _run(_cfg(food_type="high_fat"))
    assert rich.energy.mean() > lean.energy.mean() + 15, (
        f"high-fat should leave animals better fed "
        f"({rich.energy.mean():.0f} vs {lean.energy.mean():.0f})")


# --------------------------------------------------------------------------- #
# Energy budget
# --------------------------------------------------------------------------- #
def test_eating_raises_energy_and_is_recorded():
    sim = _run(_cfg())
    assert (sim.food_eaten_g > 0).all(), "every animal should find the food"
    assert sim.health.min() == 100.0, "a provisioned population should not suffer"
    assert sim.energy.min() > 0.0


def test_no_food_starves_then_kills():
    sim = _cfg(food=False, mortality=True, days=4.0)
    s = _run(sim)
    assert (s.food_eaten_g == 0).all()
    assert not s.alive.any(), "an unprovisioned population must die"


def test_starvation_takes_about_the_store_duration():
    """~2 days of reserve: they should not die on day 1, nor last a week."""
    cfg = _cfg(food=False, mortality=True, days=5.0)
    sim = Simulation(cfg, 0)
    dt, death_day = 8.0, None
    for k in range(int(5 * DAY / dt)):
        sim.step(k * dt, dt)
        if death_day is None and not sim.alive.all():
            death_day = k * dt / DAY
    assert 1.0 < death_day < 4.0, \
        f"first death at day {death_day:.2f} — expected the store to last ~2 d"


def test_satiated_animal_stops_eating():
    """A full animal must not keep stripping the pile."""
    sim = _run(_cfg(), days=2.0)
    # 45 g animals eating a few grams a day, not tens
    assert sim.food_eaten_g.max() < 30.0, \
        f"runaway intake: {sim.food_eaten_g.max():.1f} g in 2 days"


def test_finite_pile_depletes_and_stops():
    cfg = _cfg(amount_g=1.5, days=3.0)
    sim = _run(cfg)
    pile = [o for o in sim._res_objects if o.kind == "food"][0]
    assert pile.amount_g == pytest.approx(0.0, abs=1e-6), "pile should empty"
    assert sim.food_eaten_g.sum() <= 1.5 + 1e-6, \
        "animals cannot eat more than the pile held"


def test_small_animals_run_out_first():
    """Kleiber: cost scales as M^0.75 but the store scales as M.

    So a bigger animal carries proportionally more fuel than it burns and
    survives a fast longer — which is why small rodents must feed so often.
    """
    light = _run(_cfg(food=False, mass=20.0, days=1.0))
    heavy = _run(_cfg(food=False, mass=60.0, days=1.0))
    assert light.energy.mean() < heavy.energy.mean(), \
        "the smaller animal should deplete its proportionate store faster"


def test_absolute_basal_cost_still_rises_with_mass():
    p = preset_params("Standard")
    assert p.basal_kj(60.0, 1.0, 3600.0) > p.basal_kj(20.0, 1.0, 3600.0)


def test_fighting_costs_energy():
    p = preset_params("Standard")
    cfg = _cfg(n=2)
    sim = Simulation(cfg, 0)
    before = sim.energy.copy()
    sim._spend_energy_kj((0,), p.fight_kj)
    assert sim.energy[0] < before[0]
    assert sim.energy[1] == before[1]
    assert sim.hunger[0] == pytest.approx(100.0 - sim.energy[0])


# --------------------------------------------------------------------------- #
# Water, bladder and marking
# --------------------------------------------------------------------------- #
def test_drinking_fills_the_bladder():
    watered = _run(_cfg(water=True, scent=True))
    assert watered.water_drunk_ml.sum() > 0
    assert watered.bladder.max() > 0


def test_marking_is_limited_by_water_supply():
    """Take the water away and territorial marking collapses."""
    watered = _run(_cfg(water=True, scent=True))
    dry = _run(_cfg(water=False, scent=True))
    assert dry.thirst.min() > 90, "no water source should leave them parched"
    assert watered.marks_made.sum() > dry.marks_made.sum() * 3, (
        f"marking should track water availability "
        f"({watered.marks_made.sum()} watered vs {dry.marks_made.sum()} dry)")


def test_bladder_capacity_scales_with_body_size():
    p = preset_params("Standard")
    assert p.bladder_capacity(60.0) > p.bladder_capacity(20.0)
    assert p.marks_per_full_bladder(60.0) > p.marks_per_full_bladder(20.0)


def test_basal_cost_uses_kleiber_scaling():
    p = preset_params("Standard")
    ref = p.basal_kj(REF_MASS_G, 1.0, 3600.0)
    assert ref == pytest.approx(p.basal_kj_h)
    big = p.basal_kj(REF_MASS_G * 2, 1.0, 3600.0)
    # M^0.75: doubling mass raises cost by 2^0.75 ~= 1.68, not 2
    assert big / ref == pytest.approx(2 ** 0.75, rel=1e-6)


# --------------------------------------------------------------------------- #
# The 0-100 scale
# --------------------------------------------------------------------------- #
def test_all_bars_stay_within_0_100():
    sim = _run(_cfg(scent=True), days=3.0)
    for name in BARS:
        v = getattr(sim, name)
        assert v.min() >= 0.0 and v.max() <= 100.0, f"{name} left 0-100: {v}"


def test_hunger_is_the_energy_deficit():
    sim = _run(_cfg(), days=2.0)
    assert np.allclose(sim.hunger, 100.0 - sim.energy)


def test_condition_csv_is_written_on_the_same_scale(tmp_path):
    cfg = _cfg(scent=True, days=1.0)
    cfg.dt = 60.0
    cfg.record_interval = 600.0
    res = Simulation(cfg, 0).run(str(tmp_path))
    cd = pd.read_csv(res["condition"])
    for col in ("health", "energy", "hunger", "thirst", "stress", "bladder"):
        assert cd[col].min() >= 0 and cd[col].max() <= 100, f"{col} out of range"
    # a full bar must read 100, not 1 — the CSV is no longer rescaled
    assert cd["health"].max() == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# Navigation: an animal must be able to reach what it needs
# --------------------------------------------------------------------------- #
def test_animals_can_leave_a_walled_resource_zone():
    """Belly full inside a zone, thirsty: it must find the doorway and drink.

    A walled zone has one small gap. Without steering for it, an animal inside
    aims straight at the water, walks into a wall, and dehydrates with a full
    stomach — which is exactly what the enclosure presets used to do.
    """
    from fnt.abma.core.config import ResourceZone, WaterTower

    arena = ArenaConfig(width=3.0, height=3.0)
    arena.resource_zones = [ResourceZone(x=1.0, y=1.5, w=0.76, d=0.51,
                                         entrance="E", label="rz")]
    arena.water_towers = [WaterTower(x=2.6, y=1.5, radius=0.076, label="wt")]
    cfg = ExperimentConfig(
        arena=arena,
        groups=[AgentGroup("M", "prairie", "M", 4, Genotype({}), Treatment(),
                           TraitProfile(mass=45.0))],
        seed=5, physiology=preset_params("Standard"))
    sim = Simulation(cfg, 0)
    # start them inside the zone, fed but parched
    sim.P[:] = [1.0, 1.5]
    sim.home[:] = [1.0, 1.5]
    dt, reached = 8.0, np.zeros(sim.n, bool)
    for k in range(int(0.5 * DAY / dt)):
        sim.step(k * dt, dt)
        sim.energy[:] = 100.0                 # hold the belly full
        sim.hunger[:] = 0.0
        sim.thirst[:] = 100.0                 # and the thirst maximal
        reached |= sim._on_resource(sim.P, sim.water, sim.water_r)
    assert reached.all(), \
        f"{(~reached).sum()} of {sim.n} animals never escaped the zone"


def test_wall_sliding_does_not_let_agents_through_solids():
    """Sliding must skirt obstacles, never tunnel into them."""
    from fnt.abma.core.config import Pole

    arena = ArenaConfig(width=1.0, height=1.0)
    arena.poles = [Pole(0.5, 0.5, 0.12, 2.0)]
    cfg = ExperimentConfig(
        arena=arena, seed=11,
        groups=[AgentGroup("g", "prairie", "M", 10, Genotype({}), Treatment(),
                           TraitProfile(base_speed=0.35, wander=2.0))])
    sim = Simulation(cfg, 0)
    for k in range(600):
        sim.step(k * 2.0, 2.0)
        d = np.hypot(sim.P[:, 0] - 0.5, sim.P[:, 1] - 0.5)
        assert (d >= 0.12 - 1e-6).all(), f"slid into the pole: {d.min():.4f}"


# --------------------------------------------------------------------------- #
# Config plumbing and migration
# --------------------------------------------------------------------------- #
def test_physiology_roundtrip():
    cfg = _cfg()
    cfg.physiology.feed_rate_g_min = 0.22
    d = cfg.to_dict()
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2.to_dict() == d
    assert cfg2.physiology.feed_rate_g_min == 0.22
    assert cfg2.schema_version == 2


def test_configs_without_physiology_stay_legacy():
    d = _cfg().to_dict()
    d.pop("physiology")
    assert ExperimentConfig.from_dict(d).physiology.enabled is False


def test_presets_are_distinct_and_identifiable():
    for name in PHYSIOLOGY_PRESETS:
        p = preset_params(name)
        assert p.enabled is True
        assert preset_name_for(p) == name
    custom = preset_params("Standard")
    custom.basal_kj_h += 1.0
    assert preset_name_for(custom) == "Custom"


def test_v1_dynamics_are_migrated_to_the_0_100_scale():
    """An old project's rules must mean the same thing after the rescale."""
    cfg = _cfg()
    d = cfg.to_dict()
    # a v1 table: bars were 0..1 back then
    d["dynamics"] = [
        Coupling("time", "hunger", "rate", 0.15).__dict__,
        Coupling("hunger", "energy", "rate", -0.4, only_when="source_high",
                 threshold=0.5).__dict__,
        Coupling("on_food", "hunger", "set", 0.0).__dict__,
    ]
    d.pop("schema_version")
    out = ExperimentConfig.from_dict(d).dynamics
    # dimensionless source -> gain takes the full x100
    assert out[0].gain == pytest.approx(15.0)
    # bar source against a bar target -> gain unchanged, threshold rescaled
    assert out[1].gain == pytest.approx(-0.4)
    assert out[1].threshold == pytest.approx(50.0)
    # 'set' writes a target value, so it always rescales
    assert out[2].gain == pytest.approx(0.0)


def test_current_configs_are_not_migrated_twice():
    cfg = _cfg()
    gains = [c.gain for c in cfg.dynamics]
    out = ExperimentConfig.from_dict(cfg.to_dict()).dynamics
    assert [c.gain for c in out] == gains


def test_default_dynamics_are_on_the_0_100_scale():
    rules = {(c.source, c.target): c for c in default_dynamics()}
    assert rules[("time", "hunger")].gain == pytest.approx(15.0)
    assert rules[("hunger", "health")].threshold == pytest.approx(90.0)
