"""Behavioural assays: put one animal in a simple world and measure what it does.

A learned brain is only worth experimenting on if it behaves like an animal in
situations it was never scored on. These are the checks — the in-silico
versions of the bench assays you would run on a new line before trusting it in
an enclosure.

``approach``
    One animal, one food source out of reach, nothing else. How often does it
    find the food, how long does it take, and how direct is the path? Run
    hungry, sated, and hungry-but-anosmic, it answers three questions at once:
    does motivation gate the behaviour (a sated animal should not bother), does
    the behaviour depend on the nose (take the nose away and see), and is the
    path a search or a beeline.

``search_after_loss``
    The animal reaches food, eats briefly, and the food is taken away. An
    animal doing *area-restricted search* stays and turns — it has just learned
    food is here — where one that was merely passing through carries on. The
    measure is how far it gets from the spot in the following minutes, against
    the same animal at the same place that never found food there.

``choice``
    Food on one side, water on the other, the animal half way between. Run
    hungry and run thirsty, it asks whether a need selects its own goal — the
    difference between an animal that wants *something* and one that knows
    what it wants.

``doorway``
    The food is inside a walled box in a corner of the room, reached through
    two small doorways. Hungry, with and without a nose: does it get in, and
    how long does that take? Smell leads straight to a doorway; without it the
    animal has only its whiskers and the walls.

``partner``
    The partner preference test, the standard assay of a prairie vole's pair
    bond: the animal is free in a long box with a tethered animal at each
    end, one it is attached to and one it is not, and the measure is whose
    side it huddles on. Run with a nose, without one, and unattached. Needs
    a brain raised with ``--life`` (``SocialParams``); the rule-based animal
    has no attachments.

``chase``
    Not a test of a brain at all, but of the body it lives in (contest
    physics): a scripted chaser and a scripted fleer, identical bodies, in
    arenas of different sizes, with and without a hut. It asks what the
    physics lets space and cover do, with behaviour held fixed.

The first four work on any policy, so the same assay can be run on the rule-based animal
for comparison. That comparison is informative in itself: the rule-based
forager is told where the food is and steers for it, so anosmia does not touch
its foraging at all. A learned brain has to smell its way there.
"""
from __future__ import annotations

import numpy as np

from .compose import cohort
from .config import (ArenaConfig, ContestParams, ExperimentConfig, Hut,
                     ResourceObject, ResourceZone, ScentParams, SocialParams)
from .physiology import preset_params
from .simulation import Simulation
from .species import build_group, by_name


def _world(seed: int, side: float, min_dist: float, species: str,
           sex: str = "M"):
    """An empty square with one food source and one animal ``min_dist`` away."""
    rng = np.random.default_rng(seed)
    margin = 0.3
    for _ in range(200):
        food = rng.uniform(margin, side - margin, 2)
        start = rng.uniform(margin, side - margin, 2)
        if np.linalg.norm(food - start) >= min_dist:
            break
    cfg = ExperimentConfig(
        name="assay",
        arena=ArenaConfig(width=side, height=side, objects=[
            ResourceObject("food", float(food[0]), float(food[1]), 0.18,
                           label="food")]),
        groups=[cohort(species, sex, 1)], dt=2.0, seed=int(seed),
        days=1.0, n_trials=1, start_datetime="2026-06-15T22:00:00")
    cfg.physiology = preset_params("Standard")
    cfg.scent = ScentParams(enabled=True)
    return cfg, food, start


def _place(sim: Simulation, start, hunger: float, smell: float, rng) -> None:
    sim.P[0] = start
    sim.home[0] = start
    sim.H[0] = rng.uniform(0, 2 * np.pi)
    sim.energy[0] = 100.0 - hunger
    sim.hunger[0] = hunger
    sim.thirst[0] = 5.0
    if smell != 1.0:
        sim.smell[0] = smell
        sim.agents[0].traits.smell_ability = smell
        sim.mark_biology_dirty()


def approach(make_policy, hunger: float = 80.0, smell: float = 1.0,
             trials: int = 40, side: float = 3.0, min_dist: float = 1.2,
             max_s: float = 600.0, species: str = "Prairie vole",
             seed: int = 0) -> dict:
    """Find-the-food assay. ``make_policy()`` returns a fresh policy, or
    ``None`` for the rule-based one."""
    dt = 2.0
    steps = int(max_s / dt)
    found, latency, direct, stay, speed = [], [], [], [], []
    for k in range(trials):
        cfg, food, start = _world(seed * 100_000 + k, side, min_dist, species)
        sim = Simulation(cfg, 0, policy=make_policy() if make_policy else None)
        _place(sim, start, hunger, smell,
               np.random.default_rng(seed * 100_000 + k + 1))
        d0 = float(np.linalg.norm(food - start)) - 0.18
        path = 0.0
        hit = None
        on = 0
        prev = sim.P[0].copy()
        for t in range(steps):
            sim.step(t * dt, dt)
            path += float(np.linalg.norm(sim.P[0] - prev))
            prev = sim.P[0].copy()
            at = bool(sim._food_index()[0] >= 0)
            if at and hit is None:
                hit = (t + 1) * dt
                path_at_hit = path
            if hit is not None:
                on += at
        found.append(hit is not None)
        if hit is not None:
            latency.append(hit)
            direct.append(d0 / max(path_at_hit, d0, 1e-6))
            stay.append(on / max(1, steps - int(hit / dt)))
        speed.append(path / max_s)
    return {
        "trials": trials, "hunger": hunger, "smell": smell,
        "found": float(np.mean(found)),
        "latency_s": float(np.median(latency)) if latency else float("nan"),
        "directness": float(np.mean(direct)) if direct else float("nan"),
        "stay_after": float(np.mean(stay)) if stay else float("nan"),
        "speed_m_s": float(np.mean(speed)),
    }


def approach_battery(make_policy, trials: int = 40, **kw) -> dict:
    """The three conditions that make ``approach`` an experiment."""
    return {
        "hungry": approach(make_policy, hunger=80, smell=1.0, trials=trials,
                           **kw),
        "sated": approach(make_policy, hunger=5, smell=1.0, trials=trials,
                          **kw),
        "hungry, anosmic": approach(make_policy, hunger=80, smell=0.0,
                                    trials=trials, **kw),
    }


def choice(make_policy, need: str = "hunger", level: float = 80.0,
           trials: int = 40, side: float = 3.0, apart: float = 1.6,
           max_s: float = 600.0, species: str = "Prairie vole",
           seed: int = 0) -> dict:
    """Two-choice test: food and water ``apart`` metres from each other, the
    animal midway, facing a random direction, with one need at ``level`` and
    the other near zero. Returns which it reached first and where it spent
    its time. ``need`` is ``"hunger"`` or ``"thirst"``."""
    if need not in ("hunger", "thirst"):
        raise ValueError("need must be 'hunger' or 'thirst'")
    dt = 2.0
    steps = int(max_s / dt)
    first = {"food": 0, "water": 0, "neither": 0}
    t_food = t_water = 0.0
    for k in range(trials):
        rng = np.random.default_rng(seed * 100_000 + k)
        # a random axis through the centre, so neither goal has a fixed side
        ang = rng.uniform(0, 2 * np.pi)
        mid = np.array([side / 2, side / 2])
        arm = 0.5 * apart * np.array([np.cos(ang), np.sin(ang)])
        food, water = mid + arm, mid - arm
        cfg = ExperimentConfig(
            name="assay",
            arena=ArenaConfig(width=side, height=side, objects=[
                ResourceObject("food", float(food[0]), float(food[1]), 0.18,
                               label="food"),
                ResourceObject("water", float(water[0]), float(water[1]),
                               0.18, label="water")]),
            groups=[cohort(species, "M", 1)], dt=dt,
            seed=int(seed * 100_000 + k), days=1.0, n_trials=1,
            start_datetime="2026-06-15T22:00:00")
        cfg.physiology = preset_params("Standard")
        cfg.scent = ScentParams(enabled=True)
        sim = Simulation(cfg, 0, policy=make_policy() if make_policy else None)
        _place(sim, mid, level if need == "hunger" else 5.0, 1.0, rng)
        sim.thirst[0] = level if need == "thirst" else 5.0
        got = None
        for t in range(steps):
            sim.step(t * dt, dt)
            on_food = bool(sim._food_index()[0] >= 0)
            on_water = bool(sim._on_water()[0])
            t_food += on_food
            t_water += on_water
            if got is None and (on_food or on_water):
                got = "food" if on_food else "water"
        first[got or "neither"] += 1
    wanted = "food" if need == "hunger" else "water"
    chose = first["food"] + first["water"]
    return {
        "trials": trials, "need": need, "level": level,
        "first_food": first["food"] / trials,
        "first_water": first["water"] / trials,
        "first_neither": first["neither"] / trials,
        "first_correct": first[wanted] / chose if chose else float("nan"),
        "time_at_food": t_food / (trials * steps),
        "time_at_water": t_water / (trials * steps),
    }


def choice_battery(make_policy, trials: int = 40, **kw) -> dict:
    return {"hungry": choice(make_policy, "hunger", trials=trials, **kw),
            "thirsty": choice(make_policy, "thirst", trials=trials, **kw)}


def doorway(make_policy, smell: float = 1.0, hunger: float = 80.0,
            trials: int = 40, side: float = 2.6, box: float = 0.51,
            hole: float = 0.0762, min_dist: float = 1.2, max_s: float = 900.0,
            species: str = "Prairie vole", seed: int = 0) -> dict:
    """Find-the-box assay: chow inside a ``box`` m walled box in one corner
    of a ``side`` m room, a doorway in each of the two walls that face the
    room. One hungry animal starts at least ``min_dist`` from the box.
    Returns how often it got inside within ``max_s``, how long that took,
    and how much of the remaining time it then spent inside."""
    dt = 2.0
    steps = int(max_s / dt)
    h = box / 2
    entered, latency, stay = [], [], []
    for k in range(trials):
        rng = np.random.default_rng(seed * 100_000 + k)
        left, low = rng.random() < 0.5, rng.random() < 0.5
        cx = h if left else side - h
        cy = h if low else side - h
        doors = ("E" if left else "W") + ("N" if low else "S")
        for _ in range(200):
            start = rng.uniform(0.3, side - 0.3, 2)
            if np.linalg.norm(start - (cx, cy)) >= min_dist:
                break
        cfg = ExperimentConfig(
            name="assay",
            arena=ArenaConfig(width=side, height=side, objects=[],
                              resource_zones=[ResourceZone(
                                  x=cx, y=cy, w=box, d=box, entrance=doors,
                                  hole=hole, label="box")]),
            groups=[cohort(species, "M", 1)], dt=dt,
            seed=int(seed * 100_000 + k), days=1.0, n_trials=1,
            start_datetime="2026-06-15T22:00:00")
        cfg.physiology = preset_params("Standard")
        cfg.scent = ScentParams(enabled=True)
        sim = Simulation(cfg, 0, policy=make_policy() if make_policy else None)
        _place(sim, start, hunger, smell, rng)
        hit, inside = None, 0
        for t in range(steps):
            sim.step(t * dt, dt)
            at = bool(sim._food_index()[0] >= 0)
            if at and hit is None:
                hit = (t + 1) * dt
            if hit is not None:
                inside += at
        entered.append(hit is not None)
        if hit is not None:
            latency.append(hit)
            stay.append(inside / max(1, steps - int(hit / dt)))
    return {
        "trials": trials, "smell": smell, "hunger": hunger,
        "entered": float(np.mean(entered)),
        "latency_s": float(np.median(latency)) if latency else float("nan"),
        "stay_after": float(np.mean(stay)) if stay else float("nan"),
    }


def doorway_battery(make_policy, trials: int = 40, **kw) -> dict:
    return {"hungry": doorway(make_policy, smell=1.0, trials=trials, **kw),
            "hungry, anosmic": doorway(make_policy, smell=0.0, trials=trials,
                                       **kw)}


def format_doorway(results: dict) -> str:
    rows = [f"{'condition':18s} {'got in':>7s} {'latency':>9s} {'stays':>6s}"]
    for name, r in results.items():
        rows.append(f"{name:18s} {r['entered'] * 100:6.0f}% "
                    f"{r['latency_s']:8.0f}s {r['stay_after'] * 100:5.0f}%")
    return "\n".join(rows)


def search_after_loss(make_policy, trials: int = 40, eat_s: float = 40.0,
                      watch_s: float = 180.0, hunger: float = 80.0,
                      side: float = 4.0, species: str = "Prairie vole",
                      seed: int = 0) -> dict:
    """Area-restricted search: does finding food, then losing it, hold it there?

    The animal is put on a food source and eats for ``eat_s``. The food is then
    removed. In the control it is put on the same spot with no food ever
    there. Returns the distance from the spot ``watch_s`` later and the mean
    turn size in each case; a searcher stays closer and turns more.
    """
    dt = 2.0
    out = {"fed_then_lost": [], "never_fed": []}
    turn = {"fed_then_lost": [], "never_fed": []}
    for k in range(trials):
        for cond in ("fed_then_lost", "never_fed"):
            cfg, food, _ = _world(seed * 100_000 + k, side, 0.0, species)
            # centre it, so walls do not decide how far it can get
            food = np.array([side / 2, side / 2])
            cfg.arena.objects[0].x = cfg.arena.objects[0].y = side / 2
            if cond == "never_fed":
                cfg.arena.objects = []
            sim = Simulation(cfg, 0,
                             policy=make_policy() if make_policy else None)
            _place(sim, food.copy(), hunger, 1.0,
                   np.random.default_rng(seed * 100_000 + k + 1))
            t = 0
            for _ in range(int(eat_s / dt)):
                sim.step(t * dt, dt)
                t += 1
            if cond == "fed_then_lost":
                sim._res_objects = []
                sim._rebuild_resources()
            turns = []
            h_prev = float(sim.H[0])
            for _ in range(int(watch_s / dt)):
                sim.step(t * dt, dt)
                t += 1
                dh = (float(sim.H[0]) - h_prev + np.pi) % (2 * np.pi) - np.pi
                turns.append(abs(dh))
                h_prev = float(sim.H[0])
            out[cond].append(float(np.linalg.norm(sim.P[0] - food)))
            turn[cond].append(float(np.mean(turns)))
    return {
        "trials": trials,
        "dist_after_loss_m": float(np.mean(out["fed_then_lost"])),
        "dist_never_fed_m": float(np.mean(out["never_fed"])),
        "turn_after_loss_rad": float(np.mean(turn["fed_then_lost"])),
        "turn_never_fed_rad": float(np.mean(turn["never_fed"])),
    }


def vitals(cfg: ExperimentConfig, days: float = 2.0, policy=None,
           trial_index: int = 0) -> dict:
    """Can this cohort keep itself alive in this enclosure?

    Runs ``cfg`` for ``days`` and reports what a keeper would check: whether
    anyone is starving or parched, how much they ate and drank, how much time
    went on each, how far they moved, and what they did to each other. It is
    the first thing to run on a brain in an enclosure it was not raised in —
    an experiment on animals that cannot feed themselves measures only that.

    ``policy`` overrides the one the config asks for (``None`` = as configured).
    """
    dt = float(cfg.dt)
    sim = Simulation(cfg, trial_index=trial_index, policy=policy)

    class _Sink:
        def record(self, *a, **k):
            pass

    n_steps = int(days * 86400.0 / dt)
    at_food = np.zeros(sim.n)
    at_water = np.zeros(sim.n)
    resting = np.zeros(sim.n)
    worst_energy = np.full(sim.n, 100.0)
    worst_thirst = np.zeros(sim.n)
    for k in range(n_steps):
        sim.step(k * dt, dt, events=_Sink())
        n = len(at_food)
        if sim.n != n:                      # a protocol event added animals
            pad = sim.n - n
            at_food, at_water, resting = (np.concatenate([a, np.zeros(pad)])
                                          for a in (at_food, at_water, resting))
            worst_energy = np.concatenate([worst_energy, np.full(pad, 100.0)])
            worst_thirst = np.concatenate([worst_thirst, np.zeros(pad)])
        at_food += sim._food_index() >= 0
        at_water += sim._on_water()
        resting += sim.activity == 0
        worst_energy = np.minimum(worst_energy, sim.energy)
        worst_thirst = np.maximum(worst_thirst, sim.thirst)
    area = sim.territory_area()
    return {
        "animals": int(sim.n), "days": float(days),
        "alive": int(sim.alive.sum()),
        "energy_mean": float(sim.energy.mean()),
        "energy_min": float(sim.energy.min()),
        "energy_worst_ever": float(worst_energy.min()),
        "thirst_mean": float(sim.thirst.mean()),
        "thirst_max": float(sim.thirst.max()),
        "thirst_worst_ever": float(worst_thirst.max()),
        "health_min": float(sim.health.min()),
        "stress_mean": float(sim.stress.mean()),
        "food_g_per_day": float(sim.food_eaten_g.mean() / days),
        "water_ml_per_day": float(sim.water_drunk_ml.mean() / days),
        "time_at_food": float(at_food.mean() / n_steps),
        "time_at_water": float(at_water.mean() / n_steps),
        "time_resting": float(resting.mean() / n_steps),
        "metres_per_day": float(sim.dist_total.mean() / days),
        "fights_per_animal_day": float(
            (sim.fights_won + sim.fights_lost).mean() / days),
        "matings_per_animal_day": float(sim.matings.mean() / days),
        "territory_m2": float(area.mean()) if len(area) else 0.0,
        "fed_everyone": bool((sim.food_eaten_g > 0).all()),
        "watered_everyone": bool((sim.water_drunk_ml > 0).all()),
    }


def format_vitals(v: dict) -> str:
    # two separate questions: can they feed and water themselves, and what do
    # they do to each other. An enclosure where everyone eats well and fights
    # constantly is a different finding from one where someone goes hungry.
    fed = (v["energy_worst_ever"] > 20 and v["thirst_worst_ever"] < 80
           and v["fed_everyone"] and v["watered_everyone"])
    hurt = v["alive"] < v["animals"] or v["health_min"] < 95
    verdict = ("they keep themselves fed and watered" if fed else
               "NOT feeding or watering themselves here")
    if hurt:
        verdict += "; injuries from fighting"
    lines = [
        f"{v['animals']} animals, {v['days']:g} days — "
        + verdict,
        f"  energy  mean {v['energy_mean']:5.1f}  lowest now "
        f"{v['energy_min']:5.1f}  lowest ever {v['energy_worst_ever']:5.1f}",
        f"  thirst  mean {v['thirst_mean']:5.1f}  highest now "
        f"{v['thirst_max']:5.1f}  highest ever {v['thirst_worst_ever']:5.1f}",
        f"  health  lowest {v['health_min']:5.1f}   stress mean "
        f"{v['stress_mean']:5.1f}   alive {v['alive']}/{v['animals']}",
        f"  intake  {v['food_g_per_day']:.1f} g food/day, "
        f"{v['water_ml_per_day']:.1f} mL water/day",
        f"  time    {v['time_at_food'] * 100:.0f}% at food, "
        f"{v['time_at_water'] * 100:.0f}% at water, "
        f"{v['time_resting'] * 100:.0f}% resting; "
        f"{v['metres_per_day']:.0f} m/day",
        f"  social  {v['fights_per_animal_day']:.1f} fights and "
        f"{v['matings_per_animal_day']:.1f} matings per animal per day; "
        f"territory {v['territory_m2']:.2f} m²",
    ]
    return "\n".join(lines)


def format_choice(results: dict) -> str:
    rows = [f"{'condition':10s} {'food first':>10s} {'water first':>11s} "
            f"{'neither':>8s} {'at food':>8s} {'at water':>9s}"]
    for name, r in results.items():
        rows.append(f"{name:10s} {r['first_food'] * 100:9.0f}% "
                    f"{r['first_water'] * 100:10.0f}% "
                    f"{r['first_neither'] * 100:7.0f}% "
                    f"{r['time_at_food'] * 100:7.0f}% "
                    f"{r['time_at_water'] * 100:8.0f}%")
    return "\n".join(rows)


def format_battery(results: dict) -> str:
    rows = [f"{'condition':18s} {'found':>6s} {'latency':>9s} "
            f"{'directness':>10s} {'stays':>6s}"]
    for name, r in results.items():
        rows.append(f"{name:18s} {r['found'] * 100:5.0f}% "
                    f"{r['latency_s']:8.0f}s {r['directness']:10.2f} "
                    f"{r['stay_after'] * 100:5.0f}%")
    return "\n".join(rows)


#: the bench assays ``bench`` knows, in the order it runs them
BENCH = ("approach", "choice", "search", "doorway")


# --------------------------------------------------------------------------- #
# A check of the body, not of a brain: can a chased animal get away?
# --------------------------------------------------------------------------- #
class _ChaseAndFlee:
    """Animal 0 runs at animal 1 and bites; animal 1 runs for the reachable
    spot farthest from animal 0, or for the hut when there is one. Scripted on
    purpose: the question is what the contest physics allows, with behaviour
    held fixed."""

    def __init__(self, side: float, hut=None):
        self.side, self.hut, self.memory = side, hut, None
        a = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
        self.cand = np.stack([np.cos(a), np.sin(a)], axis=1)

    def decide(self, sim, elapsed_s: float, dt: float):
        from .senses import PlaceMemory, SenseParams, sense
        if self.memory is None:
            self.memory = PlaceMemory(sim.n)
        _, perc = sense(sim, elapsed_s, self.memory, SenseParams())
        P = sim.P
        to = P[1] - P[0]
        chase = to / (np.linalg.norm(to) + 1e-9)
        if self.hut is not None:
            th = np.asarray(self.hut) - P[1]
            dh = np.linalg.norm(th)
            flee = th / (dh + 1e-9)
            fspeed = 1.0 if dh > 0.02 else 0.0
        else:
            ahead = np.clip(P[1] + self.cand * 0.4, 0.02, self.side - 0.02)
            flee = self.cand[np.argmax(np.linalg.norm(ahead - P[0], axis=1))]
            fspeed = 1.0
        perc.update(speed_frac=np.array([1.0, fspeed]),
                    resting=np.array([False, fspeed < 0.12]), drives={},
                    want_chew=np.zeros(2, bool), bite=np.array([True, False]))
        return np.stack([chase, flee]), perc


class _BiteCount:
    def __init__(self):
        self.n = 0

    def record(self, t, kind, *a, **k):
        self.n += int(kind == "bite")


def chase(side: float, hut: bool = False, minutes: float = 5.0, seed: int = 0,
          species: str = "House mouse", dt: float = 0.25,
          contest: ContestParams | None = None) -> dict:
    """One scripted chase in a square arena: bites landed and the fled
    animal's health. Two identical bodies; the chaser starts 40 cm away."""
    sp = by_name(species)
    huts = [Hut(kind="dome", x=side * 0.8, y=side * 0.8, w=0.127)] if hut else []
    cfg = ExperimentConfig(
        name="chase", arena=ArenaConfig(width=side, height=side, huts=huts),
        groups=[build_group(sp, "m", sex="M", count=2)], days=1.0, dt=dt,
        n_trials=1, seed=int(seed))
    cfg.contest = contest or ContestParams()
    cfg.contest.enabled = True
    sim = Simulation(cfg, policy=_ChaseAndFlee(
        side, (side * 0.8, side * 0.8) if hut else None))
    sim.mass[:] = sim.mass.mean()
    sim.speed[:] = sim.speed.mean()
    rng = np.random.default_rng(seed)
    c = np.array([side / 2.0, side / 2.0])
    a = rng.uniform(0.0, 2.0 * np.pi)
    sim.P[1] = c
    sim.P[0] = np.clip(c + 0.4 * np.array([np.cos(a), np.sin(a)]),
                       0.05, side - 0.05)
    ev = _BiteCount()
    for k in range(int(minutes * 60.0 / dt)):
        sim.step(k * dt, dt, events=ev)
    return {"side_m": side, "hut": bool(hut), "bites_per_min": ev.n / minutes,
            "health": float(sim.health[1])}


def chase_battery(sides=(0.6, 1.2, 2.4, 5.0, 10.0, 20.0, 38.0), starts: int = 3,
                  minutes: float = 5.0, species: str = "House mouse",
                  seed: int = 0, **kw) -> list:
    """The chase across arena sizes, on an open floor and with one hut to
    reach. One row per size and condition, averaged over ``starts``."""
    rows = []
    for hut in (False, True):
        for side in sides:
            runs = [chase(side, hut, minutes, seed + s, species, **kw)
                    for s in range(starts)]
            b = [r["bites_per_min"] for r in runs]
            rows.append({"side_m": side, "hut": hut, "starts": starts,
                         "minutes": minutes,
                         "bites_per_min": float(np.mean(b)),
                         "bites_min": float(np.min(b)),
                         "bites_max": float(np.max(b)),
                         "health": float(np.mean([r["health"] for r in runs]))})
    return rows


def format_chase(rows: list) -> str:
    out = ["  arena side     open floor                  one hut to reach",
           "                 bites/min   health after    bites/min   health after"]
    by = {(r["side_m"], r["hut"]): r for r in rows}
    for side in sorted({r["side_m"] for r in rows}):
        o, h = by[(side, False)], by[(side, True)]
        out.append(f"  {side:6.1f} m      {o['bites_per_min']:7.2f}      "
                   f"{o['health']:6.1f}        {h['bites_per_min']:7.2f}      "
                   f"{h['health']:6.1f}")
    return "\n".join(out)


def partner_preference(make_policy, smell: float = 1.0, vno: float = 1.0,
                       attached: float = 1.0, trials: int = 40,
                       minutes: float = 30.0, species: str = "Prairie vole",
                       focal_sex: str = "F", seed: int = 0,
                       social: dict | None = None, in_brain: bool = False,
                       pairing_s: float = 600.0, smell_after: float | None = None) -> dict:
    """One partner preference test per trial; returns the means.

    A box 1.2 m long and 0.4 m wide. A stimulus animal of the other sex is
    held at each end: the *partner* (the focal animal is ``attached`` to it,
    0-1) and a *stranger* (it is not). The focal animal starts in the middle,
    fed and watered, tired, cold and missing its partner, and is free for
    ``minutes``. Which end the partner is at alternates between trials.

    ``partner_s`` and ``stranger_s`` are the seconds it spent against each
    (within the huddling distance); ``preference`` is partner time over the
    sum (0.5 = none), over trials in which it huddled at all; ``huddled`` is
    the share of trials in which it did. ``smell`` and ``vno`` are the focal
    animal's senses; ``social`` overrides ``SocialParams`` values (for one,
    which nose an attachment needs).

    ``in_brain``: the attachment is not dealt but has to be written by the
    brain (``SocialParams.bond_in_brain``, an anatomy spec with plastic
    connections). The trial then begins with ``pairing_s`` seconds beside the
    partner-to-be, the body reporting a mating if ``attached`` > 0 (0 = they
    only sat together). ``smell_after`` is the focal animal's sense of smell
    from the end of the pairing on (methimazole given to a paired animal).
    """
    L, Wd, dt = 1.2, 0.4, 2.0
    steps = int(minutes * 60.0 / dt)
    other = "M" if focal_sex == "F" else "F"
    t_partner, t_stranger, pref, moved = [], [], [], []
    for k in range(trials):
        cfg = ExperimentConfig(
            name="assay", arena=ArenaConfig(width=L, height=Wd, objects=[]),
            groups=[cohort(species, focal_sex, 1), cohort(species, other, 2)],
            dt=dt, seed=int(seed * 100_000 + k), days=1.0, n_trials=1,
            start_datetime="2026-06-15T12:00:00")
        cfg.physiology = preset_params("Standard")
        cfg.scent = ScentParams(enabled=True)
        cfg.social = SocialParams(enabled=True, **dict(social or {}))
        cfg.social.bond_in_brain = bool(in_brain)
        sim = Simulation(cfg, 0, policy=make_policy() if make_policy else None)
        rng = np.random.default_rng(seed * 100_000 + k + 1)
        ends = np.array([[0.12, Wd / 2], [L - 0.12, Wd / 2]])
        if k % 2:
            ends = ends[::-1]
        held = np.array([1, 2])                       # partner, stranger
        sim.P[0] = (L / 2, Wd / 2)
        sim.H[0] = rng.uniform(0, 2 * np.pi)
        sim.P[held] = ends
        sim.home[:] = sim.P
        sim.energy[:], sim.hunger[:], sim.thirst[:] = 95.0, 5.0, 5.0
        sim.smell[0] = smell
        sim.agents[0].traits.smell_ability = float(smell)
        sim.vno[0] = vno
        sim.mark_biology_dirty()
        soc = sim.soc
        t_start = 0.0
        if in_brain:                                  # the pairing, in the partner's end of the box
            beside = ends[0] + np.array([0.03 if ends[0][0] < L / 2 else -0.03, 0.0])
            for t in range(int(pairing_s / dt)):
                sim.P[0] = beside
                sim.P[held] = ends
                soc.mated[0] = 1.0 if attached > 0 else 0.0
                sim.step(t * dt, dt)
            t_start = int(pairing_s / dt) * dt
            sim.P[0] = (L / 2, Wd / 2)
            sim.H[0] = rng.uniform(0, 2 * np.pi)
            sim.energy[:], sim.hunger[:], sim.thirst[:] = 95.0, 5.0, 5.0
            if smell_after is not None:
                sim.smell[0] = smell_after
                sim.agents[0].traits.smell_ability = float(smell_after)
                sim.mark_biology_dirty()
        soc.A[0, 1] = soc.A[1, 0] = attached
        soc.tired[0], soc.cold[0], soc.lonely[0] = 70.0, 40.0, 60.0 * attached
        reach = cfg.social.huddle_r_m
        with_p = with_s = 0
        path, prev = 0.0, sim.P[0].copy()
        for t in range(steps):
            sim.step(t_start + t * dt, dt)
            sim.P[held] = ends                        # tethered
            d = np.linalg.norm(sim.P[held] - sim.P[0], axis=1)
            with_p += d[0] < reach
            with_s += d[1] < reach
            path += float(np.linalg.norm(sim.P[0] - prev))
            prev = sim.P[0].copy()
        t_partner.append(with_p * dt)
        t_stranger.append(with_s * dt)
        if with_p + with_s > 0:
            pref.append(with_p / (with_p + with_s))
        moved.append(path)
    return {
        "trials": trials, "smell": smell, "vno": vno, "attached": attached,
        "minutes": minutes,
        "partner_s": float(np.mean(t_partner)),
        "stranger_s": float(np.mean(t_stranger)),
        "preference": float(np.mean(pref)) if pref else float("nan"),
        "huddled": float(len(pref) / max(1, trials)),
        "path_m": float(np.mean(moved)),
    }


def partner_battery(make_policy, trials: int = 40, **kw) -> dict:
    """The conditions that make it an experiment: attached with a nose,
    attached without one, without a vomeronasal organ, and unattached."""
    if "in_brain" not in kw:                          # a brain with plastic connections keeps its own
        probe = make_policy() if make_policy else None
        kw["in_brain"] = bool(getattr(getattr(probe, "brain", None), "plastic", None))
    out = {
        "attached": partner_preference(make_policy, trials=trials, **kw),
        "attached, no smell": partner_preference(make_policy, smell=0.0,
                                                 trials=trials, **kw),
    }
    if kw["in_brain"]:
        out["attached, smell lost afterwards"] = partner_preference(
            make_policy, smell_after=0.0, trials=trials, **kw)
    out["attached, no vomeronasal organ"] = partner_preference(
        make_policy, vno=0.0, trials=trials, **kw)
    out["unattached"] = partner_preference(make_policy, attached=0.0,
                                           trials=trials, **kw)
    return out


def format_partner(results: dict) -> str:
    rows = ["  condition                         with partner  with stranger  preference  huddled  path"]
    for name, r in results.items():
        rows.append(f"  {name:<33s} {r['partner_s']:8.0f} s   {r['stranger_s']:8.0f} s     "
                    f"{r['preference']:5.2f}     {r['huddled']:4.0%}   {r['path_m']:5.1f} m")
    return "\n".join(rows)


def policy_factory(brain_name: str | None, stochastic: bool = True,
                   anatomy: str = "", weights: str = ""):
    """``make_policy`` for the assays: a fresh learned-brain policy each call,
    or ``None`` (the rule-based animal) when no brain is named. ``anatomy``
    (with trained ``weights``) asks for a brain wired from an anatomy spec."""
    if anatomy:
        from . import anatomy as an
        spec = an.load_spec(anatomy)
        w, seed = None, 0
        if weights:
            w, meta = an.load_weights(weights)
            seed = int(meta.get("wiring_seed", 0))
        counter = iter(range(1_000_000))
        return lambda: an.AnatomicalPolicy(
            an.AnatomicalBrain(spec, seed=seed, weights=w), noise_seed=next(counter))
    if not brain_name:
        return None
    from .brain import Brain, find_brain
    from .neural import NeuralPolicy
    brain = Brain.load(find_brain(brain_name))
    return lambda: NeuralPolicy(brain, stochastic=stochastic)


def bench(make_policy, which: str = "all", trials: int = 40,
          species: str = "Prairie vole", seed: int = 0) -> tuple[dict, str]:
    """Run the single-animal assays; return ``(results, printable report)``."""
    names = BENCH if which == "all" else (which,)
    unknown = [n for n in names if n not in BENCH]
    if unknown:
        raise ValueError(f"unknown assay {unknown[0]!r}; choose from "
                         f"{', '.join(BENCH)}, all, or vitals")
    results, text = {}, []
    for name in names:
        if name == "approach":
            r = approach_battery(make_policy, trials=trials, species=species,
                                 seed=seed)
            text += ["FIND THE FOOD  (one animal, one food source, "
                     "10 minutes)", format_battery(r), ""]
        elif name == "choice":
            r = choice_battery(make_policy, trials=trials, species=species,
                               seed=seed)
            text += ["FOOD OR WATER  (one each side, the animal between "
                     "them)", format_choice(r), ""]
        elif name == "doorway":
            r = doorway_battery(make_policy, trials=trials, species=species,
                                seed=seed)
            text += ["FIND THE BOX  (chow in a walled box in the corner, two "
                     "doorways, 15 minutes)", format_doorway(r), ""]
        else:
            r = search_after_loss(make_policy, trials=trials, species=species,
                                  seed=seed)
            text += ["SEARCH AFTER LOSS  (eats for 40 s, the food is taken "
                     "away, watched for 3 minutes)",
                     f"  distance from the spot   after losing food "
                     f"{r['dist_after_loss_m']:.2f} m   never fed there "
                     f"{r['dist_never_fed_m']:.2f} m",
                     f"  mean turn per 2 s        after losing food "
                     f"{r['turn_after_loss_rad']:.2f} rad  never fed there "
                     f"{r['turn_never_fed_rad']:.2f} rad", ""]
        results[name] = r
    return results, "\n".join(text).rstrip()
