"""Per-agent random streams: one animal's noise is its own.

The property under test is what makes a paired study paired. With a single
shared generator, every draw depends on how many draws came before it, so
adding an animal, removing one, or ablating one silently reseeds the whole
cohort from that moment on — and the contrast between arms then mixes the
manipulation with a different random world. See :mod:`fnt.abma.core.rng`.
"""
from __future__ import annotations

import numpy as np
import pytest

from fnt.abma.core.config import (
    ExperimentConfig, ArenaConfig, AgentGroup, Genotype, Treatment,
    TraitProfile, PolicyParams, Intervention, ProtocolEvent,
)
from fnt.abma.core.rng import AgentRandom, CH_HEADING, CH_MARK
from fnt.abma.core.simulation import Simulation


# --------------------------------------------------------------------------- #
# The generator itself
# --------------------------------------------------------------------------- #
def test_draw_depends_only_on_identity_not_on_cohort():
    r = AgentRandom(7)
    everyone = r.normal(np.array([0, 1, 2, 3, 4]), 900, CH_HEADING)
    survivors = r.normal(np.array([0, 2, 4]), 900, CH_HEADING)
    assert np.allclose(everyone[[0, 2, 4]], survivors), (
        "an agent's draw changed when other agents were removed")


def test_draws_are_pure_functions():
    r = AgentRandom(7)
    uids = np.arange(6)
    assert np.array_equal(r.normal(uids, 5, CH_HEADING),
                          r.normal(uids, 5, CH_HEADING))


@pytest.mark.parametrize("channel", [CH_HEADING, CH_MARK])
def test_uniform_is_in_range_and_flat(channel):
    u = AgentRandom(3).uniform(np.arange(40000), 11, channel)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.01


def test_normal_is_standard():
    x = AgentRandom(3).normal(np.arange(40000), 11, CH_HEADING)
    assert abs(x.mean()) < 0.02
    assert abs(x.std() - 1.0) < 0.02


def test_neighbouring_coordinates_are_decorrelated():
    """Adjacent uids, steps and channels must not produce related numbers."""
    r = AgentRandom(3)
    a = r.normal(np.arange(5000), 1, 0)
    for other in (r.normal(np.arange(5000), 2, 0),     # next step
                  r.normal(np.arange(5000), 1, 1),     # next channel
                  r.normal(np.arange(5000) + 1, 1, 0)):  # shifted uids
        assert abs(np.corrcoef(a, other)[0, 1]) < 0.05


def test_pair_draws_are_specific_to_the_pair():
    r = AgentRandom(5)
    a, b, c = np.array([0]), np.array([1]), np.array([2])
    assert r.pair_uniform(a, b, 4, 0) != r.pair_uniform(a, c, 4, 0)
    assert r.pair_uniform(a, b, 4, 0) == r.pair_uniform(a, b, 4, 0)


# --------------------------------------------------------------------------- #
# The property that matters at the engine level
# --------------------------------------------------------------------------- #
def _isolated_config(intervene: bool) -> ExperimentConfig:
    """Four animals that cannot perceive each other, in a large arena.

    Social and territorial coupling are switched off and the arena is big
    enough that nobody makes contact, so the *only* route by which one animal
    could influence another's path is the random stream. That isolates the
    property under test from real physical interaction.
    """
    arena = ArenaConfig(width=20.0, height=20.0, objects=[])
    group = AgentGroup(
        label="a", species="generic", sex="M", count=4,
        genotype=Genotype({}), treatment=Treatment("none", 0.0, 0.0),
        traits=TraitProfile(mass=30.0, home_range_r=2.0))
    cfg = ExperimentConfig(
        name="isolated", arena=arena, groups=[group],
        days=0.02, dt=2.0, record_interval=10.0, n_trials=1, seed=4,
        policy=PolicyParams(k_social=0.0, k_territory=0.0, k_resource=0.0))
    if intervene:
        # Manipulate exactly ONE animal (sexid M9001 is row 0 of trial 0), and
        # pick a trait that changes how *it* moves — with social coupling off,
        # ablating its nose would be behaviourally silent and the test would
        # pass for the wrong reason.
        cfg.interventions = [Intervention(at_day=0.005, target="M9001",
                                          attribute="wander",
                                          op="scale", value=0.2)]
    return cfg


def _paths(cfg: ExperimentConfig, steps: int = 400,
           keep: int | None = None) -> np.ndarray:
    """(steps, keep, 2) positions. ``keep`` bounds the rows so a run whose
    roster grows mid-way can still be stacked against one whose roster did
    not."""
    sim = Simulation(cfg, trial_index=0)
    out = []
    elapsed = 0.0
    for _ in range(steps):
        sim.step(elapsed, cfg.dt)
        elapsed += cfg.dt
        out.append(sim.P[:keep].copy())
    return np.array(out)


def test_ablating_one_agent_leaves_the_others_bit_identical():
    """An intervention on agent 0 must not move agents 1-3 at all.

    Under a shared sequential generator the intervention's own draws (and the
    ablated animal's changed marking behaviour) shift every subsequent draw,
    so the untouched animals wander off onto different paths. That is the bug
    this design exists to prevent.
    """
    base = _paths(_isolated_config(intervene=False))
    lesioned = _paths(_isolated_config(intervene=True))
    others = slice(1, None)
    assert np.array_equal(base[:, others], lesioned[:, others]), (
        "ablating one animal perturbed the others' trajectories")


def test_the_targeted_agent_itself_does_change():
    """Guard against the test above passing because nothing happened at all."""
    base = _paths(_isolated_config(intervene=False))
    lesioned = _paths(_isolated_config(intervene=True))
    assert not np.array_equal(base[:, 0], lesioned[:, 0]), (
        "the intervention had no effect, so the isolation test proves nothing")


def _roster_config(add_newcomer: bool) -> ExperimentConfig:
    """The same four isolated animals, optionally joined by a fifth mid-run."""
    cfg = _isolated_config(intervene=False)
    if add_newcomer:
        newcomer = AgentGroup(
            label="b", species="generic", sex="M", count=1,
            genotype=Genotype({}), treatment=Treatment("none", 0.0, 0.0),
            traits=TraitProfile(mass=30.0, home_range_r=2.0))
        cfg.protocol = [ProtocolEvent(at_day=0.005, kind="add_agents",
                                      group=newcomer)]
    return cfg


def test_adding_an_animal_mid_run_does_not_reroll_the_residents():
    """A newcomer is constructed from setup draws; residents must not notice.

    This is the case a single shared generator cannot survive: building the
    newcomer's traits, release position and starting condition consumes draws,
    so every per-step draw after it shifts and all four residents wander onto
    different paths. Keyed per-agent streams make the residents' paths a
    function of their own uids alone.
    """
    alone = _paths(_roster_config(add_newcomer=False), keep=4)
    joined = _paths(_roster_config(add_newcomer=True), keep=4)
    assert np.array_equal(alone, joined), (
        "adding an animal changed the residents' trajectories")


def test_uid_is_assigned_and_stable():
    sim = Simulation(_isolated_config(intervene=False))
    assert [a.uid for a in sim.agents] == list(range(sim.n))
    assert np.array_equal(sim.uid, np.arange(sim.n))
