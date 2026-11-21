"""The trainer: the numpy brain and its PyTorch twin, and one PPO step."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fnt.abma.core.brain import Brain                      # noqa: E402
from fnt.abma.core.config import zone_doors                # noqa: E402
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


def _has(cfg):
    """Which resources a nursery world offers, wherever they are kept."""
    zones = cfg.arena.resource_zones
    kinds = {o.kind for o in cfg.arena.objects}
    if cfg.arena.water_towers:
        kinds.add("water")
    if zones:
        kinds.add("food")
    if any(z.water for z in zones):
        kinds.add("water")
    return frozenset(kinds)


def test_worlds_are_reproducible_and_varied():
    a, b = nursery.make_world(11, SMALL), nursery.make_world(11, SMALL)
    assert a.to_dict() == b.to_dict()
    sides = {round(nursery.make_world(s, SMALL).arena.width, 3)
             for s in range(8)}
    assert len(sides) > 4, "the nursery should not hand out one world"
    kinds = [_has(nursery.make_world(s, SMALL)) for s in range(80)]
    assert set(kinds) == {frozenset({"food"}), frozenset({"water"}),
                          frozenset({"food", "water"})}, \
        "some worlds hold one resource only, some both — never neither"
    both = Curriculum(steps=64, one_resource=0.0)
    assert all(_has(nursery.make_world(s, both)) == {"food", "water"}
               for s in range(30))


def test_some_worlds_keep_their_food_in_walled_boxes():
    worlds = [nursery.make_world(s, SMALL) for s in range(120)]
    boxed = [w for w in worlds if w.arena.resource_zones]
    assert 0.2 < len(boxed) / len(worlds) < 0.6
    seen_doors, seen_water = set(), set()
    for w in boxed:
        side = w.arena.width
        zones = w.arena.resource_zones
        assert not any(o.kind == "food" for o in w.arena.objects), \
            "in a box world the boxes are the food"
        for i, z in enumerate(zones):
            h = z.w / 2
            assert -1e-9 <= z.x - h and z.x + h <= side + 1e-9
            assert -1e-9 <= z.y - h and z.y + h <= side + 1e-9
            for x, y in zone_doors(z):
                assert 1e-6 < x < side - 1e-6 and 1e-6 < y < side - 1e-6, \
                    "a doorway in a wall that is against the room"
            for other in zones[i + 1:]:
                apart = max(abs(z.x - other.x), abs(z.y - other.y))
                assert apart >= h + other.w / 2 + 0.2, "boxes too close to pass"
            seen_doors.add(len(zone_doors(z)))
            seen_water.add(z.water)
        for o in w.arena.objects:            # nothing reachable through a wall
            assert all(abs(o.x - z.x) > z.w / 2 + o.radius
                       or abs(o.y - z.y) > z.d / 2 + o.radius for z in zones)
    assert seen_doors == {1, 2} and seen_water == {True, False}
    open_only = Curriculum(steps=64, boxes=0.0)
    assert not any(nursery.make_world(s, open_only).arena.resource_zones
                   for s in range(40))


def test_some_open_water_is_a_solid_tower():
    """A tower is a wall to the whiskers; a brain has to have met one."""
    worlds = [nursery.make_world(s, SMALL) for s in range(120)]
    with_tower = [w for w in worlds if w.arena.water_towers]
    with_patch = [w for w in worlds
                  if any(o.kind == "water" for o in w.arena.objects)]
    assert len(with_tower) > 15 and len(with_patch) > 15
    reach = 0.18
    for w in with_tower:
        for t in w.arena.water_towers:
            assert all(abs(t.x - z.x) > z.w / 2 + t.radius + reach
                       or abs(t.y - z.y) > z.d / 2 + t.radius + reach
                       for z in w.arena.resource_zones), \
                "a tower that can be drunk from through a box wall"
    # an animal born at a tower stands beside it, not inside it
    brain = Brain.random(1)
    cur = Curriculum(steps=64, animals=(4,), boxes=0.0, towers=1.0,
                     one_resource=1.0, born_at_resource=1.0, born_at_door=0.0)
    seen = 0
    for seed in range(30):
        sim = nursery.spawn(seed, cur, brain)
        for t in sim.cfg.arena.water_towers:
            d = np.linalg.norm(sim.P - (t.x, t.y), axis=1)
            assert (d > t.radius + 0.02).all()
            seen += int((d < t.radius + 0.2).sum())
    assert seen > 10
    assert not any(nursery.make_world(s, Curriculum(steps=64, towers=0.0))
                   .arena.water_towers for s in range(40))


def test_some_animals_are_raised_without_a_nose():
    brain = Brain.random(1)
    smell = np.concatenate([nursery.spawn(s, SMALL, brain).smell
                            for s in range(80)])
    assert 0.15 < (smell == 0).mean() < 0.35
    assert 0.03 < ((smell > 0) & (smell < 1)).mean() < 0.2
    all_nosed = Curriculum(steps=64, animals=(2, 4), anosmic=0.0, hyposmic=0.0)
    assert (np.concatenate([nursery.spawn(s, all_nosed, brain).smell
                            for s in range(20)]) == 1).all()
    sim = nursery.spawn(3, Curriculum(steps=64, animals=(4,), anosmic=1.0),
                        brain)
    assert all(a.traits.smell_ability == 0 for a in sim.agents), \
        "the trait must agree with the state, or biology would restore it"


def test_some_animals_start_at_a_doorway():
    """Put down within reach of a doorway, on the room side of it."""
    brain = Brain.random(1)
    cur = Curriculum(steps=64, animals=(4,), boxes=1.0, one_resource=0.0,
                     born_at_door=1.0, born_at_resource=0.0)
    near = []
    for seed in range(30):
        sim = nursery.spawn(seed, cur, brain)
        zones = sim.cfg.arena.resource_zones
        doors = np.array([d for z in zones for d in zone_doors(z)])
        for p in sim.P:
            assert not any(abs(p[0] - z.x) < z.w / 2 - 1e-9
                           and abs(p[1] - z.y) < z.d / 2 - 1e-9
                           for z in zones), "it was put down inside a box"
            near.append(np.linalg.norm(doors - p, axis=1).min())
    assert np.mean(np.array(near) < 0.36) > 0.9
    nobody = Curriculum(steps=64, animals=(4,), boxes=1.0, born_at_door=0.0,
                        born_at_resource=0.0)
    far = [np.linalg.norm(np.array([d for z in s.cfg.arena.resource_zones
                                    for d in zone_doors(z)])[None] -
                          s.P[:, None], axis=2).min(axis=1)
           for s in (nursery.spawn(k, nobody, brain) for k in range(30))
           if s.cfg.arena.resource_zones]
    assert np.mean(np.concatenate(far) < 0.36) < 0.5


def test_summary_tells_nosed_from_noseless_in_box_worlds():
    f = nursery.STAT_FIELDS
    col = {name: i for i, name in enumerate(f)}
    rows = np.zeros((4, len(f)))
    rows[:, col["hunger0"]] = 80
    rows[:, col["has_food"]] = 1
    rows[:, col["has_boxes"]] = (1, 1, 1, 0)
    rows[:, col["smell"]] = (1, 0, 0, 0)
    rows[:, col["frac_at_food"]] = (0.5, 0.4, 0.0, 0.9)
    out = nursery.summarise(rows)
    assert out["box_found_intact"] == 1.0
    assert out["box_found_anosmic"] == 0.5, "the open-world animal is not counted"


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
