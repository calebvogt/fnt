"""Bench assays: single-animal tests that work on any policy."""
import numpy as np
import pytest

from fnt.abma.core import assays
from fnt.abma.core.brain import Brain
from fnt.abma.core.neural import NeuralPolicy
from fnt.abma.core.presets import vole_anosmia
from fnt.abma.run_headless import main as headless


def _random_brain():
    return NeuralPolicy(Brain.random(5))


def test_rule_based_animal_finds_food_when_hungry_and_not_when_sated():
    hungry = assays.approach(None, hunger=80, trials=6, max_s=240)
    sated = assays.approach(None, hunger=5, trials=6, max_s=240)
    assert hungry["found"] == 1.0
    assert hungry["directness"] > 0.7, "it is steered straight there"
    assert sated["found"] < hungry["found"]


def test_rule_based_foraging_does_not_use_the_nose():
    """The reason a learned brain is worth having: take the nose from the
    rule-based animal and it finds food exactly as before."""
    intact = assays.approach(None, hunger=80, smell=1.0, trials=6, max_s=240)
    anosmic = assays.approach(None, hunger=80, smell=0.0, trials=6, max_s=240)
    assert anosmic["found"] == intact["found"] == 1.0
    assert anosmic["latency_s"] == pytest.approx(intact["latency_s"], rel=0.2)


def test_choice_follows_the_need_for_the_rule_based_animal():
    r = assays.choice_battery(None, trials=6, max_s=300)
    assert r["hungry"]["first_food"] == 1.0
    assert r["thirsty"]["first_water"] == 1.0
    assert r["hungry"]["first_correct"] == r["thirsty"]["first_correct"] == 1.0


def test_choice_places_the_goals_symmetrically():
    """Neither goal is nearer or always on one side, so a bias in the answer
    is the animal's and not the arena's."""
    r = assays.choice(_random_brain, "hunger", trials=4, max_s=20)
    assert r["first_food"] + r["first_water"] + r["first_neither"] \
        == pytest.approx(1.0)
    with pytest.raises(ValueError, match="hunger"):
        assays.choice(None, need="sleep")


def test_assays_run_on_a_learned_brain():
    r = assays.approach(_random_brain, trials=3, max_s=60)
    assert 0.0 <= r["found"] <= 1.0 and r["speed_m_s"] >= 0
    s = assays.search_after_loss(_random_brain, trials=2, eat_s=10,
                                 watch_s=30)
    assert np.isfinite([s["dist_after_loss_m"], s["dist_never_fed_m"],
                        s["turn_after_loss_rad"]]).all()


def test_search_after_loss_takes_the_food_away():
    """After the food is removed the animal must not still be eating."""
    seen = {}

    class Spy(NeuralPolicy):
        def decide(self, sim, elapsed_s, dt):
            seen[elapsed_s] = len(sim._res_objects)
            return super().decide(sim, elapsed_s, dt)

    assays.search_after_loss(lambda: Spy(Brain.random(5)), trials=1,
                             eat_s=10, watch_s=10)
    assert seen[0.0] in (0, 1) and seen[max(seen)] == 0


def test_vitals_describe_a_cohort():
    cfg = vole_anosmia()
    v = assays.vitals(cfg, days=0.01)
    assert v["animals"] == v["alive"] > 0
    assert 0 <= v["time_resting"] <= 1 and v["metres_per_day"] > 0
    text = assays.format_vitals(v)
    assert "energy" in text and "social" in text


def test_bench_report_and_unknown_assay():
    results, text = assays.bench(None, "choice", trials=3)
    assert set(results) == {"choice"} and "FOOD OR WATER" in text
    with pytest.raises(ValueError, match="approach, choice, search"):
        assays.bench(None, "maze")


def test_headless_assay_needs_no_config(capsys):
    assert headless(["--assay", "choice", "--assay-trials", "3"]) == 0
    out = capsys.readouterr().out
    assert "rule-based" in out and "FOOD OR WATER" in out
