"""The trainer: the numpy brain and its PyTorch twin, and one PPO step."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fnt.abma.core.brain import Brain                      # noqa: E402
from fnt.abma.core.senses import SENSORS                   # noqa: E402
from fnt.abma.learn import nursery                         # noqa: E402
from fnt.abma.learn.nursery import Curriculum              # noqa: E402
from fnt.abma.learn.ppo import (Hyper, TorchBrain, Trainer,  # noqa: E402
                                _brain_meta)

SMALL = Curriculum(steps=64, animals=(2, 4))


def _random_net(seed=0):
    torch.manual_seed(seed)
    net = TorchBrain()
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.3 * torch.randn_like(p))
    return net


def test_the_numpy_brain_and_the_torch_brain_are_the_same_network():
    """Training happens in one, living in the other; they must agree."""
    net = _random_net()
    brain = net.export(_brain_meta(Hyper(), Curriculum(), "t", {}))
    obs = np.random.default_rng(0).standard_normal(
        (6, 5, len(SENSORS))).astype(np.float32)
    h = brain.initial_state(5)
    out = []
    for t in range(6):
        m, h, _ = brain.forward(obs[t], h)
        out.append(m)
    mu, _, hT = net(torch.as_tensor(obs), torch.zeros(1, 5, 64))
    assert np.abs(np.stack(out) - mu.detach().numpy()).max() < 1e-4
    assert np.abs(h - hT[0].detach().numpy()).max() < 1e-4


def test_a_newborn_brain_already_listens_to_its_body():
    """Needs must be able to change what a sense does from the first tick:
    sensory gains differ between units and move when hunger moves."""
    torch.manual_seed(0)
    brain = TorchBrain().export(_brain_meta(Hyper(), Curriculum(), "t", {}))
    obs = np.full((2, len(SENSORS)), 0.5, np.float32)
    obs[1, SENSORS.index("hunger")] = 0.95
    mod = [SENSORS.index(m) for m in brain.meta["modulators"]]
    gain = 2 / (1 + np.exp(-(obs[:, mod] @ brain.w["Wg"].T + brain.w["bg"])))
    assert np.allclose(gain[0], 1.0, atol=1e-5), \
        "a middling animal starts with every gain at 1"
    assert np.abs(gain[1] - 1.0).mean() > 0.15, "hunger changes the gains"
    assert (gain[1] > 1.05).any() and (gain[1] < 0.95).any(), \
        "some units are turned up by hunger and some down"


def test_weights_survive_export_and_reload():
    net = _random_net(1)
    brain = net.export(_brain_meta(Hyper(), Curriculum(), "t", {}))
    again = TorchBrain()
    again.load_brain(brain)
    for a, b in zip(net.parameters(), again.parameters()):
        assert torch.equal(a, b)


def test_worlds_are_reproducible_and_varied():
    a, b = nursery.make_world(11, SMALL), nursery.make_world(11, SMALL)
    assert a.to_dict() == b.to_dict()
    sides = {round(nursery.make_world(s, SMALL).arena.width, 3)
             for s in range(8)}
    assert len(sides) > 4, "the nursery should not hand out one world"
    kinds = [frozenset(o.kind for o in nursery.make_world(s, SMALL).arena.objects)
             for s in range(60)]
    assert set(kinds) == {frozenset({"food"}), frozenset({"water"}),
                          frozenset({"food", "water"})}, \
        "some worlds hold one resource only, some both — never neither"
    both = Curriculum(steps=64, one_resource=0.0)
    assert all({o.kind for o in nursery.make_world(s, both).arena.objects}
               == {"food", "water"} for s in range(20))


def test_summary_counts_a_resource_only_where_it_exists():
    """Time at water is averaged over animals whose world had water, and
    'right choice' is the share of resource time spent at the needed one."""
    f = nursery.STAT_FIELDS
    rows = np.zeros((3, len(f)))
    col = {name: i for i, name in enumerate(f)}
    # thirsty, both resources, all its resource time at water
    rows[0, [col["thirst0"], col["has_food"], col["has_water"],
             col["frac_at_water"]]] = (90, 1, 1, 0.4)
    # thirsty, in a world with no water at all
    rows[1, [col["thirst0"], col["has_food"]]] = (90, 1)
    # hungry, both resources, three quarters of its resource time at food
    rows[2, [col["hunger0"], col["has_food"], col["has_water"],
             col["frac_at_food"], col["frac_at_water"]]] = (90, 1, 1, 0.3, 0.1)
    out = nursery.summarise(rows)
    assert out["water_time_thirsty"] == pytest.approx(0.4), \
        "the animal with no water to find must not count against the others"
    assert out["right_choice"] == pytest.approx(0.7 / 0.8)


def test_animals_arrive_in_every_state_of_need():
    brain = Brain.random(0)
    hunger, thirst = [], []
    for s in range(30):
        sim = nursery.spawn(s, SMALL, brain)
        hunger += list(sim.hunger)
        thirst += list(sim.thirst)
        assert np.allclose(sim.energy, 100.0 - sim.hunger)
    hunger, thirst = np.array(hunger), np.array(thirst)
    assert hunger.min() < 15 and hunger.max() > 80
    # and one need at a time, not only both together
    assert ((hunger > 60) & (thirst < 25)).any()
    assert ((thirst > 60) & (hunger < 25)).any()


def test_rollout_shapes_and_reward_definition():
    brain = Brain.random(0)
    data = nursery.rollout(brain.w, brain.meta, [1, 2, 3], SMALL.to_dict())
    T, A = data["rew"].shape
    assert T == 64 and data["obs"].shape == (T, A, len(SENSORS))
    assert data["u"].shape == (T, A, 2)
    assert np.isfinite(data["rew"]).all()
    assert data["stats"].shape == (A, len(nursery.STAT_FIELDS))
    summary = nursery.summarise(data["stats"])
    assert set(summary) >= {"food_time_hungry", "water_time_thirsty"}


def test_one_ppo_iteration_runs_and_moves_the_weights(tmp_path):
    hp = Hyper(worlds=4, workers=1, epochs=1, minibatches=2, chunk=16)
    tr = Trainer(hp, SMALL, str(tmp_path), name="t")
    before = [p.detach().clone() for p in tr.net.parameters()]
    info = tr.step()
    tr.close()
    assert info["agent_steps"] > 0 and np.isfinite(info["return"])
    assert any(not torch.equal(a, b)
               for a, b in zip(before, tr.net.parameters()))
    path = tr.save()
    brain = Brain.load(path)
    assert brain.meta["trained"] and brain.meta["agent_steps"] > 0
    assert (tmp_path / "training_log.jsonl").exists()


def test_training_resumes_from_a_brain_file(tmp_path):
    hp = Hyper(worlds=2, workers=1, epochs=1, minibatches=1, chunk=16)
    tr = Trainer(hp, SMALL, str(tmp_path), name="t")
    tr.step()
    tr.close()
    saved = Brain.load(tr.save())
    tr2 = Trainer(hp, SMALL, str(tmp_path), name="t", resume=saved)
    tr2.close()
    assert tr2.agent_steps == saved.meta["agent_steps"]
    for a, b in zip(tr.net.parameters(), tr2.net.parameters()):
        assert torch.equal(a, b)
