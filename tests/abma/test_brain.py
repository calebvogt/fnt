"""Senses, the brain, and the policy that puts a brain in charge of an animal."""
import numpy as np
import pytest

from fnt.abma.core import senses
from fnt.abma.core.brain import Brain, find_brain
from fnt.abma.core.config import (
    ArenaConfig, BrainParams, ExperimentConfig, ProtocolEvent, ResourceObject,
    ScentParams, Stimulation,
)
from fnt.abma.core.compose import cohort
from fnt.abma.core.neural import NeuralPolicy, discomfort, make_policy
from fnt.abma.core.physiology import preset_params
from fnt.abma.core.policy import RuleBasedPolicy
from fnt.abma.core.presets import vole_anosmia
from fnt.abma.core.simulation import Simulation


def _one_animal(food=(2.0, 1.5), start=(1.0, 1.5), heading=0.0, side=3.0):
    """One vole in an empty square, facing ``heading``, food at ``food``."""
    cfg = ExperimentConfig(
        arena=ArenaConfig(width=side, height=side, objects=[
            ResourceObject("food", food[0], food[1], 0.18, label="food")]),
        groups=[cohort("Prairie vole", "M", 1)], dt=2.0, seed=4)
    cfg.physiology = preset_params("Standard")
    cfg.scent = ScentParams(enabled=True)
    sim = Simulation(cfg, 0, policy=NeuralPolicy(Brain.random(1)))
    sim.P[0] = start
    sim.H[0] = heading
    return sim


def _obs(sim):
    pol = sim.policy
    pol._ensure(sim.n)
    obs, _ = senses.sense(sim, 0.0, pol.memory, pol.params)
    return {name: float(obs[0, i]) for i, name in enumerate(senses.SENSORS)}


# --------------------------------------------------------------------------- #
# Senses
# --------------------------------------------------------------------------- #
def test_sensor_list_is_well_formed():
    assert len(set(senses.SENSORS)) == len(senses.SENSORS), "duplicate sensor"
    grouped = [s for g in senses.GROUPS.values() for s in g]
    assert sorted(grouped) == sorted(senses.SENSORS), \
        "every sensor belongs to exactly one group"
    assert set(senses.MODULATORS) <= set(senses.SENSORS)


def test_smell_is_egocentric():
    """The same food is 'ahead' or 'behind' depending only on which way the
    animal faces — never on where north is."""
    facing = _obs(_one_animal(heading=0.0))            # food is due east
    away = _obs(_one_animal(heading=np.pi))
    left = _obs(_one_animal(heading=-np.pi / 2))       # facing south
    assert facing["olf_food_fwd"] > 0.3 and abs(facing["olf_food_lat"]) < 0.05
    assert away["olf_food_fwd"] < -0.3
    assert left["olf_food_lat"] > 0.3, "food to its left reads as left"
    assert facing["olf_food"] == pytest.approx(away["olf_food"])


def test_smell_fades_with_distance():
    near = _obs(_one_animal(start=(1.6, 1.5)))
    far = _obs(_one_animal(start=(0.4, 1.5)))
    assert near["olf_food"] > far["olf_food"] > 0


def test_anosmia_removes_the_olfactory_input_and_nothing_else():
    sim = _one_animal()
    intact = _obs(sim)
    sim.smell[0] = 0.0
    anosmic = _obs(sim)
    for name in senses.GROUPS["olfaction"]:
        assert anosmic[name] == 0.0, f"{name} survived anosmia"
    for name in ("wall_ahead", "hunger", "thirst", "home", "clock_sin"):
        assert anosmic[name] == pytest.approx(intact[name]), \
            f"anosmia changed a non-olfactory sense ({name})"


def test_memory_is_silent_until_the_place_has_been_found():
    sim = _one_animal()
    assert _obs(sim)["mem_food"] == 0.0, "remembers food it never found"
    sim.P[0] = (2.0, 1.5)                    # stand on the food
    _obs(sim)                                # ... and notice
    sim.P[0] = (1.0, 1.5)
    sim.H[0] = 0.0
    after = _obs(sim)
    assert after["mem_food"] > 0.3
    assert after["mem_food_fwd"] > 0.5, "the remembered food is ahead"


def test_whiskers_feel_the_wall_they_face():
    toward = _obs(_one_animal(start=(2.8, 1.5), heading=0.0))
    clear = _obs(_one_animal(start=(1.5, 1.5), heading=0.0))
    assert toward["wall_ahead"] > 0.5
    assert clear["wall_ahead"] == 0.0


def test_taste_reports_standing_on_food():
    assert _obs(_one_animal(start=(2.0, 1.5)))["at_food"] == 1.0
    assert _obs(_one_animal(start=(1.0, 1.5)))["at_food"] == 0.0


# --------------------------------------------------------------------------- #
# The brain
# --------------------------------------------------------------------------- #
def test_brain_file_round_trip(tmp_path):
    b = Brain.random(3)
    b2 = Brain.load(b.save(str(tmp_path / "b.npz")))
    obs = np.random.default_rng(0).standard_normal(
        (5, len(senses.SENSORS))).astype(np.float32)
    m1, h1, _ = b.forward(obs, b.initial_state(5))
    m2, h2, _ = b2.forward(obs, b2.initial_state(5))
    assert np.array_equal(m1, m2) and np.array_equal(h1, h2)
    assert b2.tick_s == b.tick_s and b2.sensors == b.sensors


def test_brain_is_matched_to_the_body_by_sensor_name(tmp_path):
    """A brain trained before a sensor existed still loads; one that needs a
    sensor the body has lost refuses rather than reading zeros."""
    b = Brain.random(3)
    drop = [s for s in senses.SENSORS if s not in senses.MODULATORS][-1]   # a sense that is not also a neuromodulator
    keep = [i for i, s in enumerate(b.sensors) if s != drop]
    older = Brain({**b.w, "W1": b.w["W1"][:, keep]},
                  {**b.meta, "sensors": [b.sensors[i] for i in keep]})
    obs = np.ones((2, len(senses.SENSORS)), np.float32)
    older.forward(obs, older.initial_state(2))           # runs
    with pytest.raises(ValueError, match="no longer has"):
        Brain(b.w, {**b.meta, "sensors": b.sensors[:-1] + ["a_lost_sense"]})


def test_stimulation_reaches_one_unit_of_one_animal():
    b = Brain.random(3)
    obs = np.random.default_rng(1).standard_normal(
        (4, len(senses.SENSORS))).astype(np.float32)
    h0 = b.initial_state(4)
    _, h, _ = b.forward(obs, h0)
    _, hs, _ = b.forward(obs, h0, stim=[("rnn", 7, "set", 1.0,
                                         np.array([2]))])
    assert hs[2, 7] == 1.0
    changed = np.argwhere(hs != h)
    assert changed.tolist() == [[2, 7]], "the clamp leaked to another unit"
    _, hz, _ = b.forward(obs, h0, stim=[("rnn", 7, "scale", 0.0,
                                         np.arange(4))])
    assert (hz[:, 7] == 0).all()


def test_find_brain_says_what_exists():
    with pytest.raises(FileNotFoundError, match="python -m fnt.abma.learn"):
        find_brain("no_such_brain_xyz")


# --------------------------------------------------------------------------- #
# The policy in the engine
# --------------------------------------------------------------------------- #
def test_config_chooses_the_policy(tmp_path):
    cfg = vole_anosmia()
    assert isinstance(make_policy(cfg), RuleBasedPolicy), \
        "rules must stay the default — old configs depend on it"
    cfg.brain = BrainParams(enabled=True,
                            name=Brain.random(1).save(str(tmp_path / "b.npz")))
    assert isinstance(Simulation(cfg, 0).policy, NeuralPolicy)


def test_brain_config_round_trip():
    cfg = vole_anosmia()
    cfg.brain = BrainParams(enabled=True, name="x", stochastic=False,
                            noise_scale=0.5)
    cfg.stimulations = [Stimulation(at_day=0.5, duration_s=20, target="all",
                                    layer="rnn", unit=5, op="scale", value=0)]
    d = cfg.to_dict()
    again = ExperimentConfig.from_dict(d)
    assert again.to_dict() == d
    assert again.stimulations[0].unit == 5


def test_a_neural_run_is_deterministic():
    def run():
        sim = Simulation(vole_anosmia(), 0, policy=NeuralPolicy(Brain.random(2)))
        for k in range(150):
            sim.step(k * 2.0, 2.0)
        return sim.P.copy(), sim.policy.h.copy()
    (p1, h1), (p2, h2) = run(), run()
    assert np.array_equal(p1, p2) and np.array_equal(h1, h2)


def test_the_brain_owns_speed():
    """A brain that commands zero speed leaves the animal where it is — the
    engine no longer decides when an animal rests."""
    b = Brain.random(2)
    b.w["Wo"][:] = 0.0
    b.w["bo"][:] = (0.0, -30.0)                # speed -> sigmoid(-30) ~ 0
    sim = Simulation(vole_anosmia(), 0,
                     policy=NeuralPolicy(b, stochastic=False))
    start = sim.P.copy()
    for k in range(50):
        sim.step(k * 2.0, 2.0)
    assert np.allclose(sim.P, start, atol=1e-6)
    assert (sim.activity == 0).all(), "a stationary animal is resting"


def test_brain_holds_its_command_between_ticks():
    """At a step finer than the brain's tick, the brain is consulted once per
    tick, not once per step."""
    sim = Simulation(vole_anosmia(), 0, policy=NeuralPolicy(Brain.random(2)))
    ticks = 0
    for k in range(40):                          # 40 x 0.5 s = 20 s = 10 ticks
        sim.step(k * 0.5, 0.5)
        ticks += sim.policy.ticked
    assert ticks == 10


def test_animals_added_mid_run_get_a_brain_state():
    cfg = vole_anosmia()
    cfg.protocol = [ProtocolEvent(at_day=100.0 / 86400.0, kind="add_agents",
                                  group=cohort("Prairie vole", "M", 3))]
    sim = Simulation(cfg, 0, policy=NeuralPolicy(Brain.random(2)))
    n0 = sim.n
    for k in range(100):
        sim.step(k * 2.0, 2.0)
    assert sim.n == n0 + 3
    assert len(sim.policy.h) == sim.n and len(sim.policy.memory.food) == sim.n
    assert np.isfinite(sim.P).all()


def test_scheduled_stimulation_is_applied_then_released():
    cfg = vole_anosmia()
    cfg.stimulations = [Stimulation(at_day=20.0 / 86400.0, duration_s=10.0,
                                    target="all", layer="rnn", unit=4,
                                    op="set", value=1.0)]
    sim = Simulation(cfg, 0, policy=NeuralPolicy(Brain.random(2)))
    seen = []
    for k in range(25):
        sim.step(k * 2.0, 2.0)
        seen.append(bool((sim.policy.h[:, 4] == 1.0).all()))
    assert seen[10:15] == [True] * 5, "unit should be clamped for 10 s"
    assert not any(seen[:10]) and not any(seen[15:])


def test_frames_carry_the_brain():
    sim = Simulation(vole_anosmia(), 0, policy=NeuralPolicy(Brain.random(2)))
    sim.step(0.0, 2.0)
    fr = sim._frame(2.0)
    assert fr["brain_rnn"].shape == (sim.n, 64)
    assert fr["brain_obs"].shape == (sim.n, len(senses.SENSORS))
    assert "brain_rnn" not in Simulation(vole_anosmia(), 0)._frame(0.0)


def test_discomfort_is_convex_in_need():
    sim = Simulation(vole_anosmia(), 0)
    sim.thirst[:] = 0
    sim.stress[:] = 0
    sim.hunger[:] = 80
    starving = discomfort(sim)[0]
    sim.hunger[:] = 40
    peckish = discomfort(sim)[0]
    assert starving > 3.5 * peckish, "twice as hungry should hurt far more"


def test_command_units_are_the_strongest_motor_lines():
    b = Brain.random(3)
    b.w["Wo"][:] = 0.0
    b.w["Wo"][0, 7] = 2.0
    b.w["Wo"][0, 9] = -1.5
    b.w["Wo"][1, 4] = -3.0
    cu = b.command_units(2)
    assert [u for u, _ in cu["turn"]] == [7, 9]
    assert cu["speed"][0] == (4, -3.0)
    labels = b.unit_labels()
    assert labels[7] == "turns it left" and labels[9] == "turns it right"
    assert labels[4] == "slows it down" and labels[0] == ""


def test_driving_a_turn_unit_turns_the_animal():
    """The label is a promise: clamp a 'turns it left' unit on and the
    animal's heading must swing left (counter-clockwise), and the other way
    for 'right'."""
    b = Brain.random(3)
    b.w["Wo"][:] = 0.0
    b.w["Wo"][0, 7] = 4.0                       # unit 7 -> turn left
    b.w["bo"][:] = (0.0, 0.0)

    def swing(value):
        sim = _one_animal(food=(0.3, 0.3), start=(1.5, 1.5))
        sim.policy = NeuralPolicy(b, stochastic=False)
        sim.policy.stim = [("rnn", 7, "set", value, np.array([0]))]
        total, prev = 0.0, float(sim.H[0])
        for k in range(4):
            sim.step(k * 2.0, 2.0)
            total += (float(sim.H[0]) - prev + np.pi) % (2 * np.pi) - np.pi
            prev = float(sim.H[0])
        return total

    assert swing(1.0) > 1.0, "driving a left-turn unit turned it left"
    assert swing(-1.0) < -1.0
