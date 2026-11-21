"""The brain that ships with ABMA must behave like an animal.

These are behavioural acceptance tests, not unit tests: nothing here was
written into the brain, so each one is a statement about what training
produced. If a retrained brain fails one, the brain is what is wrong.
"""
import numpy as np
import pytest

from fnt.abma.core import assays, senses
from fnt.abma.core.brain import Brain, find_brain, list_brains
from fnt.abma.core.config import BrainParams
from fnt.abma.core.neural import NeuralPolicy
from fnt.abma.core.presets import vole_anosmia
from fnt.abma.core.simulation import Simulation

NAME = BrainParams().name


@pytest.fixture(scope="module")
def brain():
    return Brain.load(find_brain(NAME))


@pytest.fixture(scope="module")
def make(brain):
    return lambda: NeuralPolicy(brain)


def test_the_default_brain_ships_and_fits_this_body(brain):
    assert NAME in [n for n, _ in list_brains()]
    assert brain.meta.get("trained") is True
    assert set(brain.sensors) <= set(senses.SENSORS), \
        "the shipped brain needs a sense the body does not have"
    assert "trained on" in brain.describe()


def test_it_finds_food_by_smell(make):
    """Hungry, it reaches food nearly every time; without a nose it mostly
    does not, and when it does it took a wandering path to get there."""
    intact = assays.approach(make, hunger=80, smell=1.0, trials=12, max_s=400)
    anosmic = assays.approach(make, hunger=80, smell=0.0, trials=12, max_s=400)
    assert intact["found"] >= 0.9
    assert intact["directness"] > 0.5
    assert anosmic["found"] <= intact["found"] - 0.3
    assert anosmic["directness"] < 0.5 * intact["directness"] \
        or np.isnan(anosmic["directness"])


def test_the_stronger_the_need_the_surer_the_choice(make):
    """Motivation is graded, not a switch: a slightly thirsty animal usually
    goes to water, a very thirsty one always does, and one with no need at
    all splits between the two."""
    none = assays.choice(make, "thirst", level=5, trials=16, max_s=300)
    mild = assays.choice(make, "thirst", level=20, trials=16, max_s=300)
    strong = assays.choice(make, "thirst", level=80, trials=16, max_s=300)
    assert none["first_water"] < mild["first_water"] <= strong["first_water"]
    assert strong["first_water"] >= 0.9
    assert 0.1 < none["first_water"] < 0.9, "with no need it has no rule"


def test_a_need_picks_its_own_goal(make):
    """Food one side, water the other: nothing told it which is for what."""
    r = assays.choice_battery(make, trials=12, max_s=400)
    assert r["hungry"]["first_food"] >= 0.8
    assert r["thirsty"]["first_water"] >= 0.8
    assert r["thirsty"]["time_at_water"] > 5 * r["thirsty"]["time_at_food"]


def test_it_searches_where_food_was(make):
    r = assays.search_after_loss(make, trials=10)
    assert r["dist_after_loss_m"] < 0.75 * r["dist_never_fed_m"]


def test_a_cohort_runs_in_an_enclosure_it_was_not_raised_in(brain):
    cfg = vole_anosmia()
    cfg.brain = BrainParams(enabled=True, name=NAME)
    sim = Simulation(cfg, 0)
    assert isinstance(sim.policy, NeuralPolicy)
    for k in range(900):                        # half an hour
        sim.step(k * 2.0, 2.0)
    assert np.isfinite(sim.P).all() and sim.alive.all()
    assert (sim.dist_total > 1.0).all(), "every animal should have moved"
