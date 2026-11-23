"""Counterfactual twins: one life lived twice, with one thing changed."""
import numpy as np
import pytest

from fnt.abma.core.brain import Brain
from fnt.abma.core.config import BrainParams
from fnt.abma.core.neural import NeuralPolicy
from fnt.abma.core.presets import vole_anosmia
from fnt.abma.core.simulation import Simulation
from fnt.abma.core.twin import (Tweak, TwinResult, apply_tweak, fork,
                                parse_tweak, rows_of, run_twin)

H = 3600.0


def _brain_cfg(tmp_path):
    cfg = vole_anosmia()
    cfg.brain = BrainParams(enabled=True,
                            name=Brain.random(5).save(str(tmp_path / "b.npz")))
    return cfg


# --------------------------------------------------------------------------- #
# The property everything rests on
# --------------------------------------------------------------------------- #
def test_an_unchanged_twin_retraces_the_original_exactly():
    """If a copy drifted on its own, no difference could be blamed on a tweak."""
    res = run_twin(vole_anosmia(),
                   [Tweak(target="0", name="hunger", op="add", value=0.0)],
                   fork_at_s=0.5 * H, horizon_s=1.0 * H)
    assert res.divergence().max() == 0.0
    assert np.isnan(res.split_time()).all()
    assert "made no difference" in res.sentence()


def test_an_unchanged_neural_twin_also_retraces(tmp_path):
    res = run_twin(_brain_cfg(tmp_path),
                   [Tweak(target="0", name="hunger", op="add", value=0.0)],
                   fork_at_s=0.2 * H, horizon_s=0.5 * H)
    assert res.divergence().max() == 0.0
    assert res.units_a is not None and np.array_equal(res.units_a, res.units_b)


def test_fork_is_independent_of_the_original():
    sim = Simulation(vole_anosmia(), 0)
    for k in range(50):
        sim.step(k * 2.0, 2.0)
    twin = fork(sim)
    twin.energy[0] = 1.0
    twin.P[0] = (0.1, 0.1)
    assert sim.energy[0] != 1.0 and not np.allclose(sim.P[0], (0.1, 0.1))


# --------------------------------------------------------------------------- #
# A change makes a difference, and the difference spreads
# --------------------------------------------------------------------------- #
def test_a_small_change_to_one_animal_changes_its_life():
    res = run_twin(vole_anosmia(), "hunger-30@0", fork_at_s=1.0 * H,
                   horizon_s=3.0 * H)
    s = res.summary()
    assert s["target_split_after_s"][0] is not None
    assert s["target_max_gap_m"] > 0.2
    assert res.divergence()[0].max() == 0.0, "they start from the same place"
    # the change was to her body, and shows there first
    assert res.bars_b["hunger"][0, 0] < res.bars_a["hunger"][0, 0]
    # ...and reaches animals nothing was done to
    assert s["others_changed"] >= 1
    assert "parted" in res.sentence()


def test_result_round_trips_to_disk(tmp_path):
    res = run_twin(vole_anosmia(), "thirst+40@0", fork_at_s=0.2 * H,
                   horizon_s=0.5 * H)
    path = res.save(str(tmp_path / "twin.npz"))
    with np.load(path) as z:
        assert z["xy_a"].shape == res.xy_a.shape
        assert "summary" in str(z["meta"])


# --------------------------------------------------------------------------- #
# Tweaks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,kind,name,op,value,target,dur", [
    ("hunger-30@0", "state", "hunger", "add", -30.0, "0", 0.0),
    ("thirst+25", "state", "thirst", "add", 25.0, "0", 0.0),
    ("smell_ability=0@saline_M", "trait", "smell_ability", "set", 0.0,
     "saline_M", 0.0),
    ("aggression*2@all", "trait", "aggression", "scale", 2.0, "all", 0.0),
    ("rnn:12=1@3~10", "unit", "rnn:12", "set", 1.0, "3", 10.0),
    ("olfaction*0@0~600", "sense", "olfaction", "scale", 0.0, "0", 600.0),
])
def test_tweak_grammar(text, kind, name, op, value, target, dur):
    tw = parse_tweak(text)
    assert (tw.kind, tw.name, tw.op, tw.value, tw.target, tw.duration_s) == \
        (kind, name, op, value, target, dur)


def test_unreadable_tweaks_say_what_was_expected():
    with pytest.raises(ValueError, match="condition bar"):
        parse_tweak("nonsense=1")
    with pytest.raises(ValueError, match="cannot read"):
        parse_tweak("hunger")


def test_hunger_tweak_moves_the_energy_store():
    sim = Simulation(vole_anosmia(), 0)
    sim.energy[:] = 60.0
    sim.hunger[:] = 40.0
    apply_tweak(sim, parse_tweak("hunger-30@0"), 0.0)
    assert sim.hunger[0] == pytest.approx(10.0)
    assert sim.energy[0] == pytest.approx(90.0)
    assert sim.hunger[1] == pytest.approx(40.0), "only the target changes"


def test_targets_resolve_by_row_sexid_and_group():
    sim = Simulation(vole_anosmia(), 0)
    assert rows_of(sim, "2").tolist() == [2]
    assert rows_of(sim, sim.agents[3].sexid).tolist() == [3]
    assert len(rows_of(sim, "all")) == sim.n
    with pytest.raises(ValueError, match="matches no animal"):
        rows_of(sim, "nobody")


def test_unit_tweaks_need_a_brain():
    with pytest.raises(ValueError, match="needs a learned brain"):
        run_twin(vole_anosmia(), "rnn:3=1@0~10", fork_at_s=60.0,
                 horizon_s=120.0)


def test_a_timed_sense_tweak_is_undone_when_it_ends(tmp_path):
    cfg = _brain_cfg(tmp_path)
    sim = Simulation(cfg, 0)
    sim.step(0.0, 2.0)
    undo = apply_tweak(sim, parse_tweak("olfaction*0@0~60"), 2.0)
    col = __import__("fnt.abma.core.senses", fromlist=["x"]).index("olf_food")
    assert sim.policy.sensor_gain[0, col] == 0.0
    assert sim.policy.sensor_gain[1, col] == 1.0, "only the target is ablated"
    (end, restore), = undo
    assert end == pytest.approx(62.0)
    restore()
    assert sim.policy.sensor_gain[0, col] == 1.0


def test_driving_a_unit_changes_a_neural_life(tmp_path):
    res = run_twin(_brain_cfg(tmp_path), "rnn:9=1@0~20", fork_at_s=0.1 * H,
                   horizon_s=0.5 * H)
    assert res.divergence()[:, 0].max() > 0.0
    assert not np.array_equal(res.units_a[1:, 0], res.units_b[1:, 0])
