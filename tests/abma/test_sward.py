"""The sward: an environment the animals wear down and share.

The claims under test are the couplings, not the numbers. Deep grass must be
slow and expensive and worn grass fast and cheap; walking must wear it a little
and clipping a lot; clipping must cost time and yield no food; and regrowth
must be what turns a trail into something that needs maintaining. Whether a
pass removes 0.05 cm is a free parameter — that it removes *some*, and less
than a clipping bout does, is the model.
"""
from __future__ import annotations

import numpy as np
import pytest

from fnt.abma.core.compose import design
from fnt.abma.core.config import (
    ExperimentConfig, ArenaConfig, AgentGroup, Genotype, Treatment,
    TraitProfile, SwardParams, ScentParams,
)
from fnt.abma.core.simulation import Simulation
from fnt.abma.core.sward import SwardField


@pytest.fixture
def field():
    p = SwardParams(enabled=True, cell_size=0.25, initial_min_cm=6.0,
                    initial_max_cm=10.0, patchiness=0.0)
    return SwardField(5.0, 5.0, p, rng=np.random.default_rng(3))


# --------------------------------------------------------------------------- #
# The field itself
# --------------------------------------------------------------------------- #
def test_starts_at_a_field_realistic_height(field):
    assert 6.0 <= field.height.min() <= field.height.max() <= 10.0
    assert 7.0 < field.mean_height() < 9.0


def test_taller_grass_is_slower_and_a_trail_is_faster(field):
    ref = field.p.speed_ref_cm
    assert field.speed_factor(np.array([ref]))[0] == pytest.approx(1.0)
    assert field.speed_factor(np.array([0.0]))[0] > 1.0        # bare trail
    assert field.speed_factor(np.array([field.p.max_cm]))[0] < 1.0
    heights = np.linspace(0, field.p.max_cm, 20)
    factors = field.speed_factor(heights)
    assert np.all(np.diff(factors) <= 1e-9), "speed must fall monotonically"


def test_taller_grass_costs_more_to_cross(field):
    cheap = field.push_kj(np.array([45.0]), np.array([1.0]), np.array([1.0]))
    dear = field.push_kj(np.array([45.0]), np.array([1.0]), np.array([10.0]))
    assert dear[0] > cheap[0] > 0
    # and standing still costs nothing extra however deep the grass
    assert field.push_kj(np.array([45.0]), np.array([0.0]),
                         np.array([10.0]))[0] == 0.0


def test_walking_wears_the_sward_down(field):
    before = field.height.copy()
    P = np.array([[2.5, 2.5]])
    field.trample(P, np.array([3.0]))
    r, c = field.cells_of(P)
    assert field.height[r[0], c[0]] < before[r[0], c[0]]
    assert field.trampled_cm > 0


def test_clipping_is_several_times_faster_than_walking(field):
    P = np.array([[1.0, 1.0]])
    walk = SwardField(5.0, 5.0, field.p, rng=np.random.default_rng(3))
    chew = SwardField(5.0, 5.0, field.p, rng=np.random.default_rng(3))
    # the same second of effort, spent two ways
    metres = 1.0
    walk.trample(P, np.array([metres]))
    chew.chew(P, 1.0, np.array([True]))
    r, c = walk.cells_of(P)
    worn = walk.initial_mean - float(walk.height[r[0], c[0]])
    clipped = chew.initial_mean - float(chew.height[r[0], c[0]])
    assert clipped > worn


def test_a_clipping_bout_is_longer_in_taller_grass(field):
    short, tall = field.chew_bout_s(np.array([2.0, 10.0]))
    assert tall > short > 0


def test_the_sward_cannot_be_cut_below_its_floor(field):
    P = np.array([[1.0, 1.0]])
    for _ in range(500):
        field.chew(P, 10.0, np.array([True]))
    r, c = field.cells_of(P)
    assert field.height[r[0], c[0]] == pytest.approx(field.p.chew_floor_cm)


def test_regrowth_closes_a_trail_but_never_overshoots(field):
    field.height[:] = 1.0
    field.grow(86400.0 * 5)
    assert field.mean_height() > 1.0
    field.grow(86400.0 * 500)
    assert field.height.max() <= field.p.max_cm + 1e-6


def test_growth_can_be_switched_off_by_the_season(field):
    field.height[:] = 2.0
    field.grow(86400.0, season_factor=0.0)
    assert field.mean_height() == pytest.approx(2.0)


def test_trail_fraction_is_zero_before_anyone_has_walked(field):
    """It must measure the animals' work, not the initial 6-10 cm spread."""
    assert field.trail_fraction() == 0.0


def test_trail_fraction_rises_once_ground_is_worn(field):
    field.height[:4, :4] = 1.0
    assert field.trail_fraction() > 0.0


def test_image_is_drawable_and_matches_its_extent(field):
    rgba, extent = field.image(max_side=1000)
    assert rgba.dtype == np.uint8 and rgba.shape[2] == 4
    assert extent[1] == pytest.approx(rgba.shape[1] * field.cell)


# --------------------------------------------------------------------------- #
# In a running simulation
# --------------------------------------------------------------------------- #
def _sward_cfg(days=0.25, **sward) -> ExperimentConfig:
    params = dict(enabled=True, cell_size=0.25)
    params.update(sward)
    arena = ArenaConfig(width=8.0, height=8.0, ground="grass")
    group = AgentGroup(label="m", species="prairie", sex="M", count=4,
                       genotype=Genotype({}), treatment=Treatment(),
                       traits=TraitProfile(mass=42.0))
    return ExperimentConfig(
        arena=arena, groups=[group], days=days, dt=2.0, n_trials=1, seed=5,
        scent=ScentParams(enabled=True), sward=SwardParams(**params))


def _run(cfg) -> Simulation:
    sim = Simulation(cfg, 0)
    elapsed = 0.0
    for _ in range(int(cfg.days * 86400 / cfg.dt)):
        sim.step(elapsed, cfg.dt)
        elapsed += cfg.dt
    return sim


def test_animals_wear_paths_into_the_sward():
    sim = _run(_sward_cfg())
    assert sim.sward.trampled_cm > 0
    assert sim.sward.height.min() < sim.sward.initial_mean


def test_animals_end_up_standing_in_shorter_grass_than_average():
    """The point of a trail: they use the ground they have already worn."""
    sim = _run(_sward_cfg(days=0.5))
    assert sim.grass_cm.mean() < sim.sward.mean_height()


def test_clipping_happens_stops_the_animal_and_yields_no_food():
    sim = _run(_sward_cfg(days=0.5))
    assert sim.chew_seconds.sum() > 0, "nobody ever clipped"
    assert sim.sward.chewed_cm > 0
    # clipping is not eating: the food counter only moves at a food station,
    # and this arena has none
    assert sim.food_eaten_g.sum() == 0.0


def test_a_clipping_animal_does_not_move():
    cfg = _sward_cfg(days=0.05)
    sim = Simulation(cfg, 0)
    elapsed, caught = 0.0, False
    for _ in range(int(cfg.days * 86400 / cfg.dt)):
        before = sim.P.copy()
        sim.step(elapsed, cfg.dt)
        elapsed += cfg.dt
        chewing = sim.chew_left > 0
        if chewing.any():
            caught = True
            assert np.allclose(sim.P[chewing], before[chewing], atol=1e-9)
    assert caught, "no clipping bout occurred, so the test proved nothing"


def test_activity_reports_clipping_as_its_own_state():
    sim = _run(_sward_cfg(days=0.5))
    # code 6 is clipping; it must be reachable and distinct from resting
    seen = set()
    cfg = _sward_cfg(days=0.2)
    s2 = Simulation(cfg, 0)
    elapsed = 0.0
    for _ in range(int(cfg.days * 86400 / cfg.dt)):
        s2.step(elapsed, cfg.dt)
        elapsed += cfg.dt
        seen.update(np.unique(s2.activity).tolist())
    assert 6 in seen


def test_the_sward_is_shared_not_owned():
    """A trail one animal wore is cheaper for the next animal to use."""
    sim = _run(_sward_cfg(days=0.3))
    worn = sim.sward.height < 0.6 * sim.sward.initial_mean
    assert worn.any(), "nothing was worn"
    # the speed advantage is a property of the ground, not of who made it
    fast = sim.sward.speed_factor(sim.sward.height[worn])
    assert np.all(fast > 1.0)


def test_switching_the_sward_off_leaves_the_engine_untouched():
    cfg = _sward_cfg()
    cfg.sward = SwardParams(enabled=False)
    sim = _run(cfg)
    assert sim.sward is None
    assert np.allclose(sim.grass_cm, 0.0)
    assert sim.chew_seconds.sum() == 0.0


def test_voleterra_ships_with_a_living_sward():
    cfg = design(preset="voleterra", males=2, females=2, days=0.05)
    assert cfg.sward.enabled
    assert 6.0 <= cfg.sward.initial_min_cm <= cfg.sward.initial_max_cm <= 12.0
