"""The mechanistic nose: what it must reproduce, and what it adds.

Two obligations, and the tests are split along them.

*Reproduce.* Switched off, nothing changes at all. Switched on, the cohort-mean
recognition still tracks the scalar gate ``smell_ability x identity_signal`` it
replaces, so the previously validated methimazole dose-response survives.

*Add.* Recognition now fails **selectively** — at one uniform dose, animals
differ in which individuals they can still tell apart — and detection fails
separately from identification, so a MUP knockout is smelled but not
identified.
"""
from __future__ import annotations

import numpy as np
import pytest

from fnt.abma.core.config import OlfactionParams
from fnt.abma.core.olfaction import (
    OlfactorySystem, emitted_profiles, receptor_gains,
)
from fnt.abma.core.presets import vole_anosmia
from fnt.abma.core.rng import AgentRandom
from fnt.abma.core.simulation import Simulation

N = 8
OFF_DIAGONAL = ~np.eye(N, dtype=bool)


def _system(seed: int = 11, **kw) -> OlfactorySystem:
    params = OlfactionParams(enabled=True, **kw)
    return OlfactorySystem(params, AgentRandom(seed), np.arange(N))


def _mean_recognition(system: OlfactorySystem, smell: float,
                      identity: float) -> float:
    system.rebuild(np.full(N, smell), np.full(N, identity))
    return float(system.recognition[OFF_DIAGONAL].mean())


# --------------------------------------------------------------------------- #
# Obligation 1: reproduce the model it replaces
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("smell,identity", [
    (1.0, 1.0), (0.75, 1.0), (0.5, 1.0), (0.25, 1.0), (0.0, 1.0),
    (1.0, 0.5), (1.0, 0.0),
])
def test_cohort_mean_tracks_the_scalar_gate(smell, identity):
    """Recognition must stay close to ``smell x identity`` on average.

    The tolerance is loose on purpose: exact agreement is not the goal, and the
    residual is a real prediction of the model (losing half your receptors
    costs more than half your discrimination, because discrimination needs the
    channels that carry the distinguishing information). What must not happen
    is a wholesale change of scale that silently invalidates the tuning of
    every scent-driven parameter around it.
    """
    got = _mean_recognition(_system(), smell, identity)
    assert abs(got - smell * identity) < 0.12


@pytest.mark.parametrize("endpoint", [(1.0, 1.0, 1.0), (0.0, 1.0, 0.0),
                                      (1.0, 0.0, 0.0)])
def test_endpoints_are_exact(endpoint):
    """Intact, fully anosmic and fully anonymous must be exactly 1, 0 and 0."""
    smell, identity, expected = endpoint
    assert _mean_recognition(_system(), smell, identity) == pytest.approx(
        expected, abs=1e-6)


def test_recognition_is_monotone_in_dose():
    system = _system()
    doses = [0.0, 0.25, 0.5, 0.75, 1.0]
    got = [_mean_recognition(system, 1.0 - d, 1.0) for d in doses]
    assert got == sorted(got, reverse=True)


def test_receptor_mass_equals_smell_ability_exactly():
    """The construction is exact, not sampled — that is what preserves the
    dose-response instead of adding binomial noise to it."""
    system = _system()
    for smell in (0.0, 0.13, 0.5, 0.87, 1.0):
        system.rebuild(np.full(N, smell), np.ones(N))
        assert np.allclose(system.receptors.mean(axis=1), smell, atol=1e-12)


def test_uniform_ablation_recovers_the_old_behaviour():
    """``ablation_selectivity=0`` is the scalar model: no per-animal structure."""
    system = _system(ablation_selectivity=0.0)
    system.rebuild(np.full(N, 0.5), np.ones(N))
    spread = system.recognition[OFF_DIAGONAL].reshape(N, N - 1).std(axis=1)
    assert np.allclose(spread, 0.0, atol=1e-9)


def test_disabled_by_default_and_leaves_the_engine_on_the_scalar_gate():
    cfg = vole_anosmia()
    assert cfg.olfaction.enabled is False
    sim = Simulation(cfg, trial_index=0)
    assert sim.olf is None
    assert np.allclose(sim.recognition_matrix(),
                       np.outer(sim.smell, sim.identity))


# --------------------------------------------------------------------------- #
# Obligation 2: express what the scalar gate could not
# --------------------------------------------------------------------------- #
def test_partial_anosmia_is_selective():
    """One dose, one cohort — but animals disagree about who is who.

    Under the scalar gate every partially anosmic animal is equally confused
    about everybody. Here each animal loses its own channels, so the recognition
    it retains is target-specific. That is the phenomenon a
    habituation-dishabituation assay measures.
    """
    system = _system()
    system.rebuild(np.full(N, 0.5), np.ones(N))
    spread = system.recognition[OFF_DIAGONAL].reshape(N, N - 1).std(axis=1)
    assert (spread > 0.01).all(), (
        "every animal is confused about every target to the same degree")


def test_two_animals_at_the_same_dose_lose_different_channels():
    system = _system()
    system.rebuild(np.full(N, 0.5), np.ones(N))
    alive = system.receptors > 0.5
    assert not np.array_equal(alive[0], alive[1])


def test_knockout_is_smelled_but_not_identified():
    """Presence and identity are separate readouts that fail separately."""
    system = _system()
    system.rebuild(np.ones(N), np.zeros(N))      # MUP-KO cohort, intact noses
    assert system.detection[OFF_DIAGONAL].mean() > 0.9   # marks are detected
    assert system.recognition[OFF_DIAGONAL].mean() < 1e-6  # nobody is identified


def test_anosmia_removes_detection_as_well():
    system = _system()
    system.rebuild(np.zeros(N), np.ones(N))
    assert system.detection.max() == pytest.approx(0.0, abs=1e-12)


def test_heterozygote_sits_between_wildtype_and_knockout():
    system = _system()
    wt = _mean_recognition(system, 1.0, 1.0)
    het = _mean_recognition(system, 1.0, 0.5)
    ko = _mean_recognition(system, 1.0, 0.0)
    assert ko < het < wt


# --------------------------------------------------------------------------- #
# Structure of the pieces
# --------------------------------------------------------------------------- #
def test_anonymous_emitters_all_smell_identical():
    private = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    flat = emitted_profiles(private, np.zeros(2))
    assert np.allclose(flat[0], flat[1])
    assert np.allclose(flat.sum(axis=1), 1.0)


def test_distinctive_emitters_keep_their_own_profile():
    private = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert np.allclose(emitted_profiles(private, np.ones(2)), private)


def test_selectivity_blends_between_channel_loss_and_gain_loss():
    key = AgentRandom(2).vector(np.arange(3), 0, 5, 8)
    smell = np.full(3, 0.5)
    whole = receptor_gains(key, smell, selectivity=1.0)
    uniform = receptor_gains(key, smell, selectivity=0.0)
    assert np.allclose(uniform, 0.5)                 # every channel halved
    assert set(np.unique(np.round(whole, 6))) <= {0.0, 1.0}  # channels die
    assert np.allclose(whole.mean(axis=1), 0.5)      # same total either way


def test_a_signature_belongs_to_the_animal_not_its_row():
    """Signatures are keyed by uid, so a cohort's members keep their own smell
    when the roster changes around them."""
    system = _system()
    first = system.private[3].copy()
    system.set_roster(np.array([3, 7, 11]))
    assert np.allclose(system.private[0], first)
