"""Contest physics: a bite as a motor act, sprint and fatigue, huts as cover.

These test the body, not behaviour: a scripted policy issues the commands, and
the engine has to resolve them the way a body would.
"""
import numpy as np
import pytest

from fnt.abma.core.brain import Brain, find_brain
from fnt.abma.core.config import (ArenaConfig, ContestParams, ExperimentConfig,
                                  Hut, Pole)
from fnt.abma.core.neural import NeuralPolicy, discomfort
from fnt.abma.core.policy import Policy
from fnt.abma.core.senses import PlaceMemory, SenseParams, sense, index
from fnt.abma.core.simulation import Simulation
from fnt.abma.core.species import build_group, by_name


class Script(Policy):
    """Faces, moves and bites exactly as told. ``facing`` in radians, per animal."""

    def __init__(self, facing, speed=0.0, bite=False):
        self.facing = np.asarray(facing, float)
        self.speed = speed
        self.bite = bite
        self.memory = None
        self.last_obs = None

    def decide(self, sim, elapsed_s, dt):
        n = sim.n
        if self.memory is None:
            self.memory = PlaceMemory(n)
        obs, perc = sense(sim, elapsed_s, self.memory, SenseParams())
        self.last_obs = obs
        speed = np.broadcast_to(np.asarray(self.speed, float), (n,)).copy()
        perc.update(speed_frac=speed, resting=speed < 0.12, drives={},
                    want_chew=np.zeros(n, bool),
                    bite=np.broadcast_to(np.asarray(self.bite, bool), (n,)).copy())
        return np.stack([np.cos(self.facing), np.sin(self.facing)], axis=1), perc


class Events:
    def __init__(self):
        self.rows = []

    def record(self, t, kind, actor, target, x, y, value=None):
        self.rows.append((t, kind, actor.index, None if target is None else target.index, value))

    def of(self, kind):
        return [r for r in self.rows if r[1] == kind]


def make(n=2, policy=None, huts=(), poles=(), contest=True, **kw):
    mouse = by_name("House mouse")
    cfg = ExperimentConfig(
        name="contest_test", arena=ArenaConfig(width=3.0, height=3.0, huts=list(huts), poles=list(poles)),
        groups=[build_group(mouse, "males", sex="M", count=n)], days=1.0, dt=1.0, n_trials=1)
    cfg.contest = ContestParams(enabled=contest, **kw)
    return Simulation(cfg, policy=policy)


def place(sim, xy):
    sim.P[:] = np.asarray(xy, float)


def run(sim, steps, ev=None, t0=0.0):
    ev = ev if ev is not None else Events()
    for k in range(steps):
        sim.step(t0 + k * sim.cfg.dt, sim.cfg.dt, events=ev)
    return ev


def test_off_by_default_and_round_trips():
    cfg = ExperimentConfig()
    assert cfg.contest.enabled is False
    cfg.contest = ContestParams(enabled=True, sprint_factor=3.0)
    back = ExperimentConfig.from_dict(cfg.to_dict())
    assert back.contest.enabled is True and back.contest.sprint_factor == 3.0


def test_a_bite_lands_ahead_and_within_reach():
    pol = Script(facing=[0.0, np.pi], bite=[True, False])       # animal 0 faces animal 1 and bites
    sim = make(policy=pol)
    place(sim, [(1.0, 1.0), (1.1, 1.0)])
    ev = run(sim, 1)
    assert sim.health[1] < 100.0 and sim.health[0] == 100.0
    assert sim.bites_given[0] == 1 and sim.bites_taken[1] == 1
    assert sim.hurt[1] > 0.0
    assert sim.hurt_dir[1] == pytest.approx([-1.0, 0.0], abs=1e-6)      # it came from animal 0
    bites = ev.of("bite")
    assert len(bites) == 1 and bites[0][2] == 0 and bites[0][3] == 1
    # the bitten animal feels it, and from the right side (it faces -x, so the bite is ahead)
    pol.decide(sim, 1.0, 1.0)
    assert pol.last_obs[1, index("hurt")] > 0.0
    assert pol.last_obs[1, index("hurt_fwd")] > 0.0
    assert pol.last_obs[0, index("hurt")] == 0.0


def test_bites_have_a_refractory_period():
    sim = make(policy=Script(facing=[0.0, np.pi], bite=[True, False]), bite_refractory_s=2.0)
    place(sim, [(1.0, 1.0), (1.1, 1.0)])
    run(sim, 1)
    assert sim.bites_given[0] == 1
    run(sim, 1, t0=1.0)
    assert sim.bites_given[0] == 1               # still recovering
    run(sim, 2, t0=2.0)
    assert sim.bites_given[0] == 2


def test_no_bite_behind_or_out_of_reach():
    sim = make(policy=Script(facing=[np.pi, np.pi], bite=[True, False]))      # animal 0 faces away
    place(sim, [(1.0, 1.0), (1.1, 1.0)])
    run(sim, 3)
    assert sim.health[1] == 100.0 and sim.bites_given[0] == 0
    sim = make(policy=Script(facing=[0.0, np.pi], bite=[True, False]))
    place(sim, [(1.0, 1.0), (1.6, 1.0)])                                       # half a metre off
    run(sim, 3)
    assert sim.health[1] == 100.0
    assert sim.energy[0] < sim.energy[1]         # snapping at air still costs something


def test_a_hut_shields_its_occupant():
    hut = Hut(kind="tube", x=1.1, y=1.0)                                       # radius about 7.6 cm
    sim = make(policy=Script(facing=[0.0, np.pi], bite=[True, False]), huts=[hut])
    place(sim, [(1.0, 1.0), (1.1, 1.0)])                                       # 0 outside, 1 under it
    assert list(sim.cover_of()) == [-1, 0]
    run(sim, 3)
    assert sim.health[1] == 100.0
    # both under it (a hut with room for two): no shelter from each other
    sim = make(policy=Script(facing=[0.0, np.pi], bite=[True, False]), huts=[Hut(kind="dome", x=1.05, y=1.0, w=0.3)],
               refuge_capacity=2)
    place(sim, [(1.0, 1.0), (1.1, 1.0)])
    run(sim, 1)
    assert sim.health[1] < 100.0
    # and a hut is not cover when the setting says so
    sim = make(policy=Script(facing=[0.0, np.pi], bite=[True, False]), huts=[hut], refuge_huts=False)
    place(sim, [(1.0, 1.0), (1.1, 1.0)])
    run(sim, 1)
    assert sim.health[1] < 100.0


def test_a_full_hut_cannot_be_entered():
    hut = Hut(kind="dome", x=1.5, y=1.0, w=0.2)                               # radius 10 cm
    # animal 1 sits under it; animal 0 walks straight at it
    pol = Script(facing=[0.0, 0.0], speed=[0.25, 0.0], bite=[True, False])
    sim = make(policy=pol, huts=[hut])
    place(sim, [(1.2, 1.0), (1.5, 1.0)])
    run(sim, 40)
    assert list(sim.cover_of()) == [-1, 0]            # still outside, still inside
    assert sim.health[1] == 100.0                     # and it could not bite in from there
    assert np.linalg.norm(sim.P[0] - [1.5, 1.0]) > 0.10
    # with room for two it walks in, and then they can bite each other
    sim = make(policy=Script(facing=[0.0, 0.0], speed=[0.25, 0.0], bite=[True, False]), huts=[hut],
               refuge_capacity=2)
    place(sim, [(1.2, 1.0), (1.5, 1.0)])
    run(sim, 40)
    assert sim.cover_of()[0] == 0 or sim.health[1] < 100.0
    assert sim.health[1] < 100.0


def test_a_bite_is_resolved_along_the_path_not_at_the_end_of_the_step():
    # animal 1 sprints across in front of animal 0; at a 2 s step it starts and
    # ends the step out of reach, and passes 8 cm from its nose in between
    pol = Script(facing=[0.0, np.pi / 2], speed=[0.0, 1.0], bite=[True, False])
    sim = make(policy=pol)
    sim.cfg.dt = 2.0
    place(sim, [(1.0, 1.0), (1.08, 0.5)])
    run(sim, 1)
    assert np.linalg.norm(sim.P[1] - sim.P[0]) > 0.3                           # it ended up well clear
    assert sim.bites_taken[1] == 1


def test_a_wall_stops_a_bite():
    pole = Pole(x=1.07, y=1.0, radius=0.02, height=0.5)
    sim = make(policy=Script(facing=[0.0, np.pi], bite=[True, False]), poles=[pole])
    place(sim, [(1.0, 1.0), (1.14, 1.0)])
    run(sim, 3)
    assert sim.health[1] == 100.0


def test_sprint_tires_and_rest_recovers():
    easy = make(n=1, policy=Script(facing=[0.0], speed=1.0), contest=False)
    place(easy, [(0.2, 1.5)])
    run(easy, 1)
    cruise_step = easy.P[0, 0] - 0.2
    pol = Script(facing=[0.0], speed=1.0)
    sim = make(n=1, policy=pol, sprint_factor=4.0, fatigue_per_s=2.0, recover_per_s=0.5)
    place(sim, [(0.2, 1.5)])
    run(sim, 1)
    first = sim.P[0, 0] - 0.2
    assert first == pytest.approx(4.0 * cruise_step, rel=1e-6)          # fresh: a full sprint
    assert sim.fatigue[0] == pytest.approx(2.0)
    x = sim.P[0, 0]
    # the arena is 3 m wide: turn round at the far end rather than run into the wall
    for k in range(1, 30):
        pol.facing = np.array([0.0 if (k // 5) % 2 == 0 else np.pi])
        run(sim, 1, t0=float(k))
    assert 40.0 < sim.fatigue[0] < 60.0
    before = sim.P[0].copy()
    run(sim, 1, t0=30.0)
    later = np.linalg.norm(sim.P[0] - before)
    assert cruise_step < later < first                                   # slower than fresh, still faster than easy
    assert discomfort(sim)[0] > discomfort(easy)[0] - 1e-9
    tired = sim.fatigue[0]
    pol.speed = 0.0
    run(sim, 20, t0=31.0)
    assert sim.fatigue[0] == pytest.approx(tired - 10.0)                 # 0.5 a second, standing still
    assert x > 0.2


def _dice_roll_fights(contest: bool) -> int:
    """Legacy contests resolved for one touching, roaming same-sex pair."""
    mouse = by_name("House mouse")
    cfg = ExperimentConfig(name="rb", arena=ArenaConfig(width=3.0, height=3.0),
                           groups=[build_group(mouse, "males", sex="M", count=4)], days=1.0, dt=2.0)
    cfg.contest = ContestParams(enabled=contest)
    sim = Simulation(cfg)
    ev = Events()
    dist = np.full((sim.n, sim.n), np.inf)
    dist[0, 1] = dist[1, 0] = 0.02
    for k in range(60):
        sim.activity[:] = 2                                  # roaming
        sim._step_k += 1
        sim._resolve_events(k * 4000.0, 2.0, dist, np.zeros(sim.n), ev)   # past the dyad's cooldown each time
    return len(ev.of("fight"))


def test_rule_based_animals_do_not_fight_under_contest_physics():
    assert _dice_roll_fights(contest=False) > 0   # the engine's dice roll
    assert _dice_roll_fights(contest=True) == 0   # gone: a contest is bites the animals choose
    # and the rules never bite or ask for more than the easy pace
    mouse = by_name("House mouse")
    cfg = ExperimentConfig(name="rb", arena=ArenaConfig(width=1.0, height=1.0),
                           groups=[build_group(mouse, "males", sex="M", count=6)], days=1.0, dt=2.0)
    cfg.contest = ContestParams(enabled=True)
    sim = Simulation(cfg)
    ev = run(sim, 600)
    assert len(ev.of("fight")) == 0 and len(ev.of("bite")) == 0
    assert sim.fatigue.max() == 0.0


def test_repeated_bites_can_kill():
    sim = make(policy=Script(facing=[0.0, np.pi], bite=[True, False]), bite_damage=40.0, bite_refractory_s=1.0)
    sim.cfg.enable_mortality = True
    place(sim, [(1.0, 1.0), (1.1, 1.0)])
    ev = run(sim, 6)
    assert not sim.alive[1]
    assert len(ev.of("death")) == 1
    n = sim.bites_given[0]
    run(sim, 3, ev, t0=6.0)
    assert sim.bites_given[0] == n                # nothing left to bite


def test_shipped_brain_has_no_bite_and_a_three_output_brain_can():
    old = Brain.load(find_brain("vole_forager"))
    assert old.n_out == 2
    assert not old.bite(np.ones((3, 2), np.float32)).any()
    new = Brain.random(seed=1, n_out=3)
    assert new.n_out == 3 and "bite" in new.describe()
    sim = make(n=4, policy=NeuralPolicy(new))
    ev = run(sim, 300)
    assert "bite" in sim._last_perc
    assert sim._last_perc["bite"].dtype == bool
    assert np.isfinite(sim.P).all() and np.isfinite(sim.fatigue).all()
    # the old brain still runs in a world with contest physics, and never bites
    sim = make(n=4, policy=NeuralPolicy(old))
    ev = run(sim, 200)
    assert len(ev.of("bite")) == 0


def test_senses_of_a_closing_animal_and_of_cover():
    hut = Hut(kind="dome", x=2.0, y=1.0, w=0.2)
    pol = Script(facing=[0.0, np.pi], speed=[0.0, 0.25])           # animal 1 walks at animal 0
    sim = make(policy=pol, huts=[hut])
    place(sim, [(1.0, 1.0), (1.5, 1.0)])
    sim.mass[:] = [20.0, 30.0]
    run(sim, 2)
    pol.decide(sim, 2.0, 1.0)
    o = pol.last_obs
    assert o[0, index("other_closing")] > 0.1        # coming at it, at an easy pace
    assert o[1, index("other_closing")] == pytest.approx(0.0, abs=1e-6)     # animal 0 is standing still
    assert o[0, index("other_size")] > 0.0 > o[1, index("other_size")]
    assert o[0, index("cover")] > 0.0 and o[0, index("cover_fwd")] > 0.0    # the hut is ahead of animal 0
    assert o[0, index("in_cover")] == 0.0


# --------------------------------------------------------------------------- #
# Raising a brain that can bite (needs PyTorch)
# --------------------------------------------------------------------------- #
def test_a_brain_can_be_raised_with_a_bite(tmp_path):
    pytest.importorskip("torch")
    from fnt.abma.learn import nursery
    from fnt.abma.learn.nursery import Curriculum
    from fnt.abma.learn.ppo import Hyper, Trainer

    cur = Curriculum(steps=64, animals=(2, 4), contest=True, huts=1.0)
    cfg = nursery.make_world(5, cur)
    assert cfg.contest.enabled and cfg.enable_mortality and len(cfg.arena.huts) >= 1
    assert not nursery.make_world(5, Curriculum(steps=64)).contest.enabled
    hp = Hyper(worlds=4, workers=1, epochs=1, minibatches=2, chunk=16)
    tr = Trainer(hp, cur, str(tmp_path), name="biter")
    try:
        info = tr.step()
        brain = tr.brain()
    finally:
        tr.close()
    assert brain.n_out == 3
    for k in ("bites", "bites_male", "bites_female", "bitten", "died"):
        assert k in info
    assert Brain.load(brain.save(str(tmp_path / "biter.npz"))).n_out == 3


def test_an_older_brain_resumes_into_contest_worlds_unchanged(tmp_path):
    torch = pytest.importorskip("torch")
    from fnt.abma.learn.nursery import Curriculum
    from fnt.abma.learn.ppo import Hyper, Trainer
    from fnt.abma.core.senses import SENSORS

    old = Brain.load(find_brain("vole_forager"))
    assert len(old.sensors) < len(SENSORS)            # raised before the contest senses existed
    hp = Hyper(worlds=4, workers=1, epochs=1, minibatches=2, chunk=16)
    tr = Trainer(hp, Curriculum(steps=64, animals=(2,), contest=True), str(tmp_path), name="r", resume=old)
    try:
        new = tr.brain()
    finally:
        tr.close()
    assert new.n_out == 3 and len(new.sensors) == len(SENSORS)
    # the same senses give the same turn and speed, whatever the new channels carry
    rng = np.random.default_rng(0)
    obs = rng.uniform(0, 1, (5, len(SENSORS))).astype(np.float32)
    h = rng.normal(0, 0.3, (5, old.n_rnn)).astype(np.float32)
    m_old, _, _ = old.forward(obs, h)
    m_new, _, _ = new.forward(obs, h)
    assert m_new[:, :2] == pytest.approx(m_old, abs=1e-5)
    assert (m_new[:, 2] < 0).all()                    # and it starts out not biting
