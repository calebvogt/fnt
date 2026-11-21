"""The nursery: the worlds a brain grows up in, and what happens to it there.

A brain is trained on the *real* engine — the same ``Simulation`` that runs an
experiment, with the same physiology, scent marks, contests and walls — not on
a simplified copy. Whatever it learns, it learned in the world it will be
tested in. What the nursery varies is the things an animal should not be able
to rely on: how big the enclosure is, where the food and water are, how many
others it shares them with, and how hungry and thirsty it is when it arrives.

Each *episode* is one such world run for a fixed number of brain ticks. The
only thing written down as good or bad is the change in the body's own
:func:`~fnt.abma.core.neural.discomfort`: eating when hungry lowers it, a lost
fight raises it. Nothing rewards approaching food, or avoiding a rival.

This module is numpy-only on purpose. :func:`rollout` is a pure function of
``(weights, seeds)`` so it can run in worker processes that never import
PyTorch, using the very :class:`~fnt.abma.core.brain.Brain` forward pass that
runs at experiment time.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

from ..core.brain import Brain
from ..core.compose import cohort
from ..core.config import (ArenaConfig, ExperimentConfig, ResourceObject,
                           ScentParams)
from ..core.neural import NeuralPolicy, discomfort
from ..core.physiology import preset_params
from ..core.simulation import Simulation


@dataclass
class Curriculum:
    """What the worlds a brain is raised in look like."""
    species: str = "Prairie vole"
    #: enclosure side, metres (uniform)
    arena_m: tuple = (1.6, 4.0)
    #: how many animals share a world (one value is drawn per world)
    animals: tuple = (2, 4, 8)
    food_sites: tuple = (1, 2)        # inclusive range
    water_sites: tuple = (1, 2)
    #: radius of a food or water source, metres (uniform). Food and water are
    #: deliberately made equally easy to stumble on: when food was the bigger
    #: target, animals learned to eat first, parked on the food, and then
    #: never wandered far enough to learn that water exists.
    site_radius_m: tuple = (0.15, 0.20)
    #: fraction of worlds that hold only food, or only water (half each). In
    #: worlds that always had both, animals learned to find food, sat on it,
    #: and from there never met water often enough to learn what it is for.
    #: A world with nothing but water teaches that at full strength; the
    #: worlds with both then only have to teach *which* one a need calls for.
    one_resource: float = 0.5
    #: hunger and thirst on arrival, 0-100. An animal has to meet every state
    #: of need, not only the comfortable ones a short episode would otherwise
    #: confine it to — and has to meet them one at a time as well as together,
    #: or it cannot learn which need a given smell answers.
    need_on_arrival: tuple = (0.0, 95.0)
    #: fraction of animals put down already standing at food or water, so
    #: "I am thirsty and I am at water" is experienced from the first episode
    born_at_resource: float = 0.2
    #: contests and matings happen (False = animals pass through each other
    #: socially, which isolates foraging)
    social: bool = True
    scent: bool = True
    tick_s: float = 2.0
    steps: int = 1024
    #: reward per unit drop in discomfort, and per mating
    reward_scale: float = 10.0
    mate_bonus: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Curriculum":
        known = {f for f in Curriculum.__dataclass_fields__}
        return Curriculum(**{k: (tuple(v) if isinstance(v, list) else v)
                             for k, v in d.items() if k in known})


class _NoEvents:
    """An event sink that keeps nothing — contests still happen to bodies."""

    def record(self, *a, **k):
        pass


def make_world(seed: int, cur: Curriculum) -> ExperimentConfig:
    """One randomised world, as an ordinary experiment config."""
    rng = np.random.default_rng(seed)
    side = float(rng.uniform(*cur.arena_m))
    margin = 0.25

    def spot():
        return rng.uniform(margin, side - margin, 2)

    objs = []
    only = None
    if rng.random() < cur.one_resource:
        only = "food" if rng.random() < 0.5 else "water"
    for kind, (lo, hi) in (("food", cur.food_sites),
                           ("water", cur.water_sites)):
        if only is not None and kind != only:
            continue
        for k in range(max(int(rng.integers(lo, hi + 1)),
                           1 if only else 0)):
            x, y = spot()
            objs.append(ResourceObject(
                kind, float(x), float(y),
                float(rng.uniform(*cur.site_radius_m)), label=f"{kind}_{k}"))
    n = int(rng.choice(cur.animals))
    males = int(rng.integers(0, n + 1))
    groups = [g for g in (
        cohort(cur.species, "M", males) if males else None,
        cohort(cur.species, "F", n - males) if n - males else None)
        if g is not None]
    cfg = ExperimentConfig(
        name="nursery",
        arena=ArenaConfig(width=side, height=side, objects=objs),
        groups=groups, dt=cur.tick_s, seed=int(seed),
        days=cur.steps * cur.tick_s / 86400.0, n_trials=1,
        release_mode="scatter",
        # a random hour, so the clock and the light are inputs it has to cope
        # with rather than constants it can ignore
        start_datetime=f"2026-06-{int(rng.integers(1, 28)):02d}"
                       f"T{int(rng.integers(0, 24)):02d}:00:00",
    )
    cfg.physiology = preset_params("Standard")
    cfg.scent = ScentParams(enabled=cur.scent)
    return cfg


def spawn(seed: int, cur: Curriculum, brain: Brain,
          stochastic: bool = True) -> Simulation:
    """A nursery world with a brain already in every animal."""
    cfg = make_world(seed, cur)
    sim = Simulation(cfg, 0, policy=NeuralPolicy(brain, stochastic=stochastic))
    rng = np.random.default_rng(seed + 7919)
    lo, hi = cur.need_on_arrival
    n = sim.n
    hunger = rng.uniform(lo, hi, n)
    thirst = rng.uniform(lo, hi, n)
    # a third arrive with one need only, a third with the other, a third with
    # whatever the draw gave them
    kind = rng.integers(0, 3, n)
    calm = rng.uniform(lo, min(hi, 25.0), n)
    thirst = np.where(kind == 0, calm, thirst)        # hungry, not thirsty
    hunger = np.where(kind == 1, calm, hunger)        # thirsty, not hungry
    sim.energy = 100.0 - hunger
    sim.hunger = hunger
    sim.thirst = thirst
    objs = [o for o in cfg.arena.objects if o.kind in ("food", "water")]
    if objs and cur.born_at_resource > 0:
        for i in np.nonzero(rng.random(n) < cur.born_at_resource)[0]:
            o = objs[int(rng.integers(len(objs)))]
            sim.P[i] = (o.x, o.y) + rng.normal(0, 0.03, 2)
            sim.home[i] = sim.P[i]
    return sim


def rollout(weights: dict, meta: dict, seeds: list, cur_d: dict) -> dict:
    """Run one episode in each seeded world; return what every animal lived.

    Arrays come back time-major with the worlds' animals side by side:
    ``obs`` is ``(T, A, n_sensors)``, ``u`` the noisy pre-squash motor command
    it actually issued ``(T, A, 2)``, ``rew`` ``(T, A)``.
    """
    cur = Curriculum.from_dict(cur_d)
    brain = Brain(weights, meta)
    T, dt = int(cur.steps), float(cur.tick_s)
    obs_l, u_l, rew_l, stat_l = [], [], [], []
    sink = _NoEvents() if cur.social else None
    for seed in seeds:
        sim = spawn(int(seed), cur, brain)
        n = sim.n
        obs = np.zeros((T, n, brain.n_in), np.float32)
        u = np.zeros((T, n, 2), np.float32)
        rew = np.zeros((T, n), np.float32)
        hunger0, thirst0 = sim.hunger.copy(), sim.thirst.copy()
        at_food = np.zeros(n)
        at_water = np.zeros(n)
        speed = np.zeros(n)
        d_prev = discomfort(sim)
        mat_prev = sim.matings.copy()
        fights0 = (sim.fights_won + sim.fights_lost).copy()
        for t in range(T):
            sim.step(t * dt, dt, events=sink)
            pol = sim.policy
            obs[t] = pol.last_obs[:, brain._take]
            u[t] = pol.u
            d_now = discomfort(sim)
            rew[t] = (cur.reward_scale * (d_prev - d_now)
                      + cur.mate_bonus * (sim.matings - mat_prev))
            d_prev, mat_prev = d_now, sim.matings.copy()
            perc = sim._last_perc
            at_food += perc["at_food"]
            at_water += perc["at_water"]
            speed += pol.last_speed
        obs_l.append(obs)
        u_l.append(u)
        rew_l.append(rew)
        stat_l.append(np.stack([
            hunger0, sim.hunger, thirst0, sim.thirst, at_food / T,
            at_water / T, speed / T, sim.dist_total, sim.health,
            (sim.fights_won + sim.fights_lost) - fights0, sim.matings,
            sim.stress, np.full(n, n, float),
            np.full(n, float(any(o.kind == "food"
                                 for o in sim.cfg.arena.objects))),
            np.full(n, float(any(o.kind == "water"
                                 for o in sim.cfg.arena.objects))),
        ], axis=1))
    return {"obs": np.concatenate(obs_l, axis=1),
            "u": np.concatenate(u_l, axis=1),
            "rew": np.concatenate(rew_l, axis=1),
            "stats": np.concatenate(stat_l, axis=0)}


#: columns of ``rollout()["stats"]``
STAT_FIELDS = ["hunger0", "hunger1", "thirst0", "thirst1", "frac_at_food",
               "frac_at_water", "mean_speed", "dist_m", "health", "fights",
               "matings", "stress", "group_size", "has_food", "has_water"]


def summarise(stats: np.ndarray) -> dict:
    """The handful of numbers that say whether animals are learning to live.

    Time at food is counted only over animals whose world had food (likewise
    water). ``right_choice`` is the one that shows a need being *used*: among
    animals with one need much stronger than the other, in a world offering
    both, the share of their time-at-a-resource spent at the one they needed.
    Chance is 0.5.
    """
    s = {name: stats[:, i] for i, name in enumerate(STAT_FIELDS)}
    food, water = s["has_food"] > 0, s["has_water"] > 0
    hungry = s["hunger0"] > 50
    thirsty = s["thirst0"] > 50

    def mean(x, m):
        return float(x[m].mean()) if m.any() else 0.0

    gap = s["hunger0"] - s["thirst0"]
    both = food & water & (np.abs(gap) > 30)
    right = np.where(gap > 0, s["frac_at_food"], s["frac_at_water"])
    total = s["frac_at_food"] + s["frac_at_water"]
    used = both & (total > 0.02)
    return {
        "d_hunger": mean(s["hunger1"] - s["hunger0"], food),
        "d_thirst": mean(s["thirst1"] - s["thirst0"], water),
        "food_time_hungry": mean(s["frac_at_food"], hungry & food),
        "food_time_sated": mean(s["frac_at_food"], ~hungry & food),
        "water_time_thirsty": mean(s["frac_at_water"], thirsty & water),
        "water_time_sated": mean(s["frac_at_water"], ~thirsty & water),
        "right_choice": float(right[used].sum() / total[used].sum())
        if used.any() else 0.5,
        "speed": float(s["mean_speed"].mean()),
        "fights": float(s["fights"].mean()),
        "matings": float(s["matings"].mean()),
        "health": float(s["health"].mean()),
        "stress": float(s["stress"].mean()),
    }
