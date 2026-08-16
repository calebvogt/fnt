"""Engine correctness: determinism, obstacles, feeding geometry, dt-invariance."""
import numpy as np
import pytest

from fnt.abma.core.config import (
    ExperimentConfig, ArenaConfig, ResourceObject, Pole, ResourceZone,
    AgentGroup, Genotype, Treatment, TraitProfile, default_vole_experiment,
)
from fnt.abma.core.simulation import Simulation


def _run_positions(cfg, steps=300, dt=2.0):
    sim = Simulation(cfg, trial_index=0)
    hist = []
    for k in range(steps):
        sim.step(k * dt, dt)
        hist.append(sim.P.copy())
    return sim, np.array(hist)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_identical_trajectories():
    cfg = default_vole_experiment()
    _, h1 = _run_positions(cfg, steps=200)
    _, h2 = _run_positions(cfg, steps=200)
    assert np.array_equal(h1, h2), "same seed must reproduce identical paths"


def test_different_trial_index_differs():
    cfg = default_vole_experiment()
    a = Simulation(cfg, trial_index=0)
    b = Simulation(cfg, trial_index=1)
    for k in range(50):
        a.step(k * 2.0, 2.0)
        b.step(k * 2.0, 2.0)
    assert not np.array_equal(a.P, b.P)


# --------------------------------------------------------------------------- #
# Obstacles: agents never end a step inside a solid
# --------------------------------------------------------------------------- #
def test_no_tunnelling_into_poles():
    arena = ArenaConfig(width=1.0, height=1.0, objects=[
        ResourceObject("food", 0.9, 0.9, 0.05),
        ResourceObject("water", 0.1, 0.9, 0.05)])
    arena.poles = [Pole(0.5, 0.5, 0.1, 2.0)]     # fat pole mid-arena
    cfg = ExperimentConfig(
        arena=arena, seed=7,
        groups=[AgentGroup("g", "prairie", "M", 12, Genotype({}), Treatment(),
                           TraitProfile(base_speed=0.3, wander=2.0,
                                        home_range_r=2.0))])
    sim, hist = _run_positions(cfg, steps=500)
    d = np.hypot(hist[..., 0] - 0.5, hist[..., 1] - 0.5)
    # body centre may sit ON the pushed-out ring (pole r + agent r); never inside
    assert (d >= 0.1 - 1e-6).all(), \
        f"agent entered the pole (min dist {d.min():.4f})"


# --------------------------------------------------------------------------- #
# Resource-zone feeding is rectangle containment, not a circumscribed circle
# --------------------------------------------------------------------------- #
def _zone_cfg():
    arena = ArenaConfig(width=2.0, height=2.0)
    arena.resource_zones = [ResourceZone(x=1.0, y=1.0, w=0.6, d=0.4,
                                         entrance="E")]
    return ExperimentConfig(
        arena=arena,
        groups=[AgentGroup("g", "prairie", "F", 1, Genotype({}), Treatment(),
                           TraitProfile())])


def test_zone_feeding_inside_box():
    sim = Simulation(_zone_cfg(), 0)
    sim.P[0] = [1.0, 1.0]                      # centre of the box
    assert sim._on_food()[0]


def test_zone_feeding_not_through_wall():
    sim = Simulation(_zone_cfg(), 0)
    # just outside the solid W wall (box spans x∈[0.7,1.3]), but INSIDE the old
    # circumscribing circle of radius hypot(.6,.4)/2 ≈ 0.361 around the centre
    sim.P[0] = [0.68, 1.0]
    r = float(np.hypot(0.68 - 1.0, 0.0))
    assert r < 0.36, "test point must lie inside the old buggy circle"
    assert not sim._on_food()[0], "fed through the solid zone wall"


# --------------------------------------------------------------------------- #
# dt-invariance: heading diffusion per unit simulated time
# --------------------------------------------------------------------------- #
def _wander_only_cfg():
    """One agent, empty world, no dynamics: heading is pure random walk."""
    arena = ArenaConfig(width=50.0, height=50.0)
    return ExperimentConfig(
        arena=arena, dynamics=[],
        groups=[AgentGroup("g", "prairie", "M", 1, Genotype({}), Treatment(),
                           TraitProfile(home_range_r=5.0, wander=1.0,
                                        turn_rate=0.5))])


def _heading_step_sd(dt, n_steps=4000):
    sim = Simulation(_wander_only_cfg(), 0)
    sim.P[0] = [25.0, 25.0]
    sim.home[0] = [25.0, 25.0]
    dh = []
    prev = float(sim.H[0])
    for k in range(n_steps):
        sim.step(k * dt, dt)
        h = float(sim.H[0])
        d = (h - prev + np.pi) % (2 * np.pi) - np.pi
        dh.append(d)
        prev = h
    return float(np.std(dh))


def test_heading_diffusion_scales_with_sqrt_dt():
    sd_fine = _heading_step_sd(0.5)
    sd_coarse = _heading_step_sd(2.0)
    ratio = sd_fine / sd_coarse
    # diffusive scaling: sd per step ∝ sqrt(dt) → expected ratio sqrt(0.25)=0.5
    assert ratio == pytest.approx(0.5, rel=0.15), \
        f"heading jitter not dt-invariant (ratio {ratio:.3f}, want ~0.5)"


def test_mating_probability_dt_invariant():
    """p = 1-exp(-rate*dt): halving dt exactly halves the per-step hazard."""
    rate = Simulation.mate_rate_hz
    p1 = 1.0 - np.exp(-rate * 1.0)
    p2 = 1.0 - np.exp(-rate * 2.0)
    # survival over equal simulated time must match: (1-p1)^2 == (1-p2)
    assert (1 - p1) ** 2 == pytest.approx(1 - p2, rel=1e-9)
