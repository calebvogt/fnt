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
                           ResourceZone, ScentParams, WaterTower, zone_doors,
                           zone_sides, ContestParams, Hut, SocialParams)
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
    #: fraction of worlds whose food is kept in walled boxes with doorways
    #: (resource zones, as in a mesocosm) rather than lying in the open. A
    #: brain raised only in open worlds meets its first wall in an experiment.
    boxes: float = 0.5
    #: how many boxes such a world has (inclusive range; fewer if they do not
    #: fit), their side in metres, and the width of a doorway
    box_count: tuple = (1, 4)
    box_side_m: tuple = (0.40, 0.55)
    box_door_m: tuple = (0.07, 0.11)
    #: in a world with both resources, the chance that a box holds water too
    box_water: float = 0.5
    #: chance that a water source in the open is a tower — a solid cylinder
    #: to drink beside — rather than a patch to walk over. To an animal's
    #: whiskers a tower is a wall; one that has learned to follow walls and
    #: has never met a tower circles it indefinitely.
    towers: float = 0.5
    tower_radius_m: tuple = (0.06, 0.12)
    #: fraction of animals raised with no sense of smell, and with a weak one.
    #: While every animal can smell, the smell of food always leads to the
    #: doorway and nothing else about finding one is ever learned — so a brain
    #: that loses its nose in an experiment is left with chance. These animals
    #: have to learn what walls and whiskers are for.
    anosmic: float = 0.25
    hyposmic: float = 0.10
    #: hunger and thirst on arrival, 0-100. An animal has to meet every state
    #: of need, not only the comfortable ones a short episode would otherwise
    #: confine it to — and has to meet them one at a time as well as together,
    #: or it cannot learn which need a given smell answers.
    need_on_arrival: tuple = (0.0, 95.0)
    #: fraction of animals put down already standing at food or water, so
    #: "I am thirsty and I am at water" is experienced from the first episode
    born_at_resource: float = 0.2
    #: fraction put down just outside a doorway of a box (where there are
    #: boxes). An animal with no nose that wanders the open floor almost never
    #: comes across a 7 cm doorway, so it never learns that an opening in a
    #: wall is worth going through; starting some of them at one makes that a
    #: common experience, and what is learned there spreads outward to "walls
    #: are worth following".
    born_at_door: float = 0.15
    #: contests and matings happen (False = animals pass through each other
    #: socially, which isolates foraging)
    social: bool = True
    scent: bool = True
    #: animals can bite, sprint, tire and hide (``ContestParams``), and die of
    #: it; the brain gets a third output, the bite. Nothing rewards a bite or
    #: a retreat: a bite costs energy, being bitten costs health, and what an
    #: animal makes of that is what is being learned.
    contest: bool = False
    #: fraction of contest worlds with huts in them, and how many (inclusive)
    huts: float = 0.6
    hut_count: tuple = (1, 4)
    #: animals tire, get cold and become attached to each other
    #: (``SocialParams``). Nothing rewards resting, huddling or staying with a
    #: partner: being tired, cold or apart is uncomfortable, and that is all.
    life: bool = False
    #: fraction of animals that arrive already attached to another animal of
    #: the world (mutually; mostly of the other sex). A life of half an hour is
    #: too short to form an attachment in, so without these nothing about
    #: having one would ever be experienced.
    attached: float = 0.5
    #: among attached animals, the share whose partner is of the same sex
    attached_same_sex: float = 0.2
    #: tiredness, cold and (for the attached) separation distress on arrival
    tired_on_arrival: tuple = (0.0, 95.0)
    lonely_on_arrival: tuple = (0.0, 90.0)
    #: overrides of single ``SocialParams`` values, e.g. {"cold_alone": 60}
    social_params: dict = field(default_factory=dict)
    #: with ``life``: attachments are kept by the brain (an anatomy spec's
    #: plastic connections), not dealt from the engine's table. Each dealt
    #: pair then spends the first ``wedding_s`` seconds of its life side by
    #: side with the body reporting a mating, and is put apart afterwards.
    bond_in_brain: bool = False
    wedding_s: float = 180.0
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


def _place_boxes(rng, cur: Curriculum, side: float, water: bool) -> list:
    """Walled boxes for one world: in a corner, against a wall, or standing
    free; one or two doorways each, never in a wall that is against the room.
    """
    want = int(rng.integers(cur.box_count[0], cur.box_count[1] + 1))
    gap = 0.22                        # room for an animal to pass between boxes
    boxes, rects = [], []
    for k in range(want):
        for _ in range(30):
            s = float(rng.uniform(*cur.box_side_m))
            h = s / 2
            style = rng.choice(["corner", "wall", "free"], p=[0.45, 0.25, 0.30])
            if style == "corner":
                x = h if rng.random() < 0.5 else side - h
                y = h if rng.random() < 0.5 else side - h
            elif style == "wall":
                along = float(rng.uniform(h + gap, side - h - gap))
                x, y = [(along, h), (along, side - h), (h, along),
                        (side - h, along)][int(rng.integers(4))]
            else:
                lo, hi = h + gap, side - h - gap
                if hi <= lo:
                    continue
                x, y = (float(v) for v in rng.uniform(lo, hi, 2))
            if any(abs(x - bx) < h + bh + gap and abs(y - by) < h + bh + gap
                   for bx, by, bh in rects):
                continue
            # a doorway only in a wall that faces the room
            open_walls = [w for w, flush in (("W", x - h < 1e-6),
                                             ("E", x + h > side - 1e-6),
                                             ("S", y - h < 1e-6),
                                             ("N", y + h > side - 1e-6))
                          if not flush]
            n_doors = min(len(open_walls), int(rng.integers(1, 3)))
            doors = "".join(rng.choice(open_walls, n_doors, replace=False))
            boxes.append(ResourceZone(
                x=float(x), y=float(y), w=s, d=s, entrance=doors,
                hole=float(rng.uniform(*cur.box_door_m)), label=f"box_{k}",
                water=bool(water and rng.random() < cur.box_water)))
            rects.append((x, y, h))
            break
    return boxes


def make_world(seed: int, cur: Curriculum) -> ExperimentConfig:
    """One randomised world, as an ordinary experiment config."""
    rng = np.random.default_rng(seed)
    side = float(rng.uniform(*cur.arena_m))
    margin = 0.25

    only = None
    if rng.random() < cur.one_resource:
        only = "food" if rng.random() < 0.5 else "water"
    # food in boxes: any world that has food, part of the time
    boxes = []
    if only != "water" and rng.random() < cur.boxes:
        boxes = _place_boxes(rng, cur, side, water=only is None)

    def spot(radius):
        """A place in the open: clear of the walls and of every box, by more
        than the source's own radius — it must not be reachable through a
        box wall."""
        for _ in range(200):
            x, y = rng.uniform(margin, side - margin, 2)
            if all(abs(x - b.x) > b.w / 2 + radius + 0.05
                   or abs(y - b.y) > b.d / 2 + radius + 0.05 for b in boxes):
                break
        return float(x), float(y)

    objs, towers = [], []
    for kind, (lo, hi) in (("food", cur.food_sites),
                           ("water", cur.water_sites)):
        if only is not None and kind != only:
            continue
        if kind == "food" and boxes:
            continue                      # the boxes are this world's food
        count = max(int(rng.integers(lo, hi + 1)), 1 if only else 0)
        if kind == "water" and boxes:
            # boxes that hold water may be all the water there is, or there
            # may be a source in the open as well (a tower in the middle)
            if any(b.water for b in boxes):
                count = int(rng.integers(0, 2))
            else:
                count = max(count, 1)
        for k in range(count):
            if kind == "water" and rng.random() < cur.towers:
                r = float(rng.uniform(*cur.tower_radius_m))
                # clear of the boxes by its whole drinking reach
                x, y = spot(r + Simulation._REACH)
                towers.append(WaterTower(x=x, y=y, radius=r,
                                         label=f"tower_{k}"))
                continue
            r = float(rng.uniform(*cur.site_radius_m))
            x, y = spot(r)
            objs.append(ResourceObject(kind, x, y, r, label=f"{kind}_{k}"))
    n = int(rng.choice(cur.animals))
    males = int(rng.integers(0, n + 1))
    groups = [g for g in (
        cohort(cur.species, "M", males) if males else None,
        cohort(cur.species, "F", n - males) if n - males else None)
        if g is not None]
    huts = []
    if (cur.contest or cur.life) and rng.random() < cur.huts:
        for k in range(int(rng.integers(cur.hut_count[0], cur.hut_count[1] + 1))):
            x, y = spot(0.10)
            huts.append(Hut(kind="dome", x=x, y=y, w=0.127, label=f"hut_{k}"))
    cfg = ExperimentConfig(
        name="nursery",
        arena=ArenaConfig(width=side, height=side, objects=objs,
                          resource_zones=boxes, water_towers=towers,
                          huts=huts),
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
    if cur.contest:
        cfg.contest = ContestParams(enabled=True)
        cfg.enable_mortality = True       # an injury can be the end of it
    if cur.life:
        cfg.social = SocialParams(enabled=True, **dict(cur.social_params or {}))
        cfg.social.bond_in_brain = bool(cur.bond_in_brain)
    return cfg


def spawn(seed: int, cur: Curriculum, brain: Brain | None,
          stochastic: bool = True, policy=None) -> Simulation:
    """A nursery world with a brain already in every animal (or, given a
    ``policy``, with that driving them)."""
    cfg = make_world(seed, cur)
    sim = Simulation(cfg, 0, policy=policy or NeuralPolicy(brain, stochastic=stochastic))
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
    objs = ([o for o in cfg.arena.objects if o.kind in ("food", "water")]
            + list(cfg.arena.resource_zones) + list(cfg.arena.water_towers))
    if objs and cur.born_at_resource > 0:
        for i in np.nonzero(rng.random(n) < cur.born_at_resource)[0]:
            o = objs[int(rng.integers(len(objs)))]
            # Anywhere within reach of it, out to the very edge — not always
            # at its centre. An animal that has no use for the resource it is
            # standing at has to learn to walk away from it, and one that
            # starts a step from the edge finds out what lies beyond; from the
            # centre, with no smell of anything else, it never did.
            ang = rng.uniform(0, 2 * np.pi)
            if isinstance(o, WaterTower):          # beside the cylinder
                dist = o.radius + rng.uniform(0.04, Simulation._REACH)
            elif isinstance(o, ResourceZone):      # inside the box
                dist = abs(rng.normal(0, 0.03))
            else:                                  # anywhere on the patch
                dist = o.radius * np.sqrt(rng.random())
            off = dist * np.array([np.cos(ang), np.sin(ang)])
            sim.P[i] = np.clip((o.x, o.y) + off, 0.02, cfg.arena.width - 0.02)
            sim.home[i] = sim.P[i]
    zones = list(cfg.arena.resource_zones)
    if zones and cur.born_at_door > 0:
        out = {"N": (0.0, 1.0), "S": (0.0, -1.0), "E": (1.0, 0.0),
               "W": (-1.0, 0.0)}
        side = cfg.arena.width
        for i in np.nonzero(rng.random(n) < cur.born_at_door)[0]:
            z = zones[int(rng.integers(len(zones)))]
            k = int(rng.integers(len(zone_sides(z))))
            ox, oy = out[zone_sides(z)[k]]
            dx, dy = zone_doors(z)[k]
            away, along = rng.uniform(0.04, 0.30), rng.uniform(-0.15, 0.15)
            pos = np.clip((dx + ox * away + oy * along,
                           dy + oy * away + ox * along), 0.02, side - 0.02)
            if any(abs(pos[0] - b.x) <= b.w / 2 and abs(pos[1] - b.y) <= b.d / 2
                   for b in zones):
                continue                  # that spot is inside another box
            sim.P[i] = pos
            sim.home[i] = pos
            sim.H[i] = rng.uniform(0, 2 * np.pi)
    # some are raised without a nose, some with a weak one
    draw = rng.random(n)
    smell = np.where(draw < cur.anosmic, 0.0,
                     np.where(draw < cur.anosmic + cur.hyposmic,
                              rng.uniform(0.2, 0.8, n), 1.0))
    for i in np.nonzero(smell < 1.0)[0]:
        sim.smell[i] = smell[i]
        sim.agents[i].traits.smell_ability = float(smell[i])
    if (smell < 1.0).any():
        sim.mark_biology_dirty()
    if sim.soc is not None:
        _arrive_social(sim, cur, rng)
    return sim


def _arrive_social(sim, cur: Curriculum, rng) -> None:
    """Tiredness, cold and attachments on arrival.

    Attachments are dealt in mutual pairs. An animal with no nose can have
    one: it formed before the nose was lost, and it now cannot be read. That
    animal's separation distress has no remedy, and what it does about that
    is learned like everything else.
    """
    soc, n = sim.soc, sim.n
    #: pairs to hold together at the start of the life: (i, j, where each
    #: arrived, the separation distress each is left with afterwards)
    sim._wedding = []
    soc.tired[:] = rng.uniform(*cur.tired_on_arrival, n)
    soc.cold[:] = rng.uniform(0.0, sim.cfg.social.cold_alone, n)
    free = list(rng.permutation(n))
    males = [i for i in free if sim.sex_m[i] > 0.5]
    females = [i for i in free if sim.sex_m[i] <= 0.5]
    while len(free) >= 2:
        i = free.pop()
        if rng.random() >= cur.attached:
            continue
        same = rng.random() < cur.attached_same_sex
        mine = males if sim.sex_m[i] > 0.5 else females
        other = females if sim.sex_m[i] > 0.5 else males
        pool = [j for j in (mine if same else other) if j in free]
        if not pool:
            continue
        j = pool[int(rng.integers(len(pool)))]
        free.remove(j)
        a = rng.uniform(0.5, 1.0)
        soc.A[i, j] = soc.A[j, i] = a
        lonely = (a * rng.uniform(*cur.lonely_on_arrival), a * rng.uniform(*cur.lonely_on_arrival))
        if cur.bond_in_brain:
            sim._wedding.append((int(i), int(j), sim.P[i].copy(), sim.P[j].copy(), lonely))
        else:
            soc.lonely[i], soc.lonely[j] = lonely


def _wedding(sim, cur: Curriculum, t_s: float) -> None:
    """Hold each dealt pair side by side while the pairing lasts, with the
    body reporting a mating; when it ends, put them apart."""
    pairs = getattr(sim, "_wedding", None)
    if not pairs:
        return
    if t_s < cur.wedding_s:
        for i, j, at_i, _, _ in pairs:
            sim.P[i] = at_i
            sim.P[j] = at_i + np.array([0.02, 0.0])
            sim.soc.mated[[i, j]] = 1.0
        return
    for i, j, at_i, at_j, lonely in pairs:
        sim.P[i], sim.P[j] = at_i, at_j
        sim.soc.lonely[i], sim.soc.lonely[j] = lonely
    sim.soc.touch_s[:] = 0.0                  # what they did while held together is not their doing
    sim.soc.partner_touch_s[:] = 0.0
    sim._wedding = []


def rollout(weights: dict, meta: dict, seeds: list, cur_d: dict) -> dict:
    """Run one episode in each seeded world; return what every animal lived.

    Arrays come back time-major with the worlds' animals side by side:
    ``obs`` is ``(T, A, n_sensors)``, ``u`` the noisy pre-squash motor command
    it actually issued ``(T, A, 2)``, ``rew`` ``(T, A)``.
    """
    cur = Curriculum.from_dict(cur_d)
    anat = None
    if meta.get("kind") == "anatomy":          # a brain wired from an anatomy spec, being trained
        from ..core import anatomy
        from ..core.senses import SENSORS
        anat = anatomy.AnatomicalBrain(anatomy.load_spec(meta["anatomy"]), seed=int(meta["wiring_seed"]),
                                       tick_s=cur.tick_s, weights=weights)
        n_in, n_out, take = len(SENSORS), len(anat.out_units), slice(None)
        brain = None
    else:
        brain = Brain(weights, meta)
        n_in, n_out, take = brain.n_in, brain.n_out, brain._take
    T, dt = int(cur.steps), float(cur.tick_s)
    obs_l, u_l, rew_l, stat_l, alive_l = [], [], [], [], []
    sink = _NoEvents() if cur.social else None
    for seed in seeds:
        pol0 = None
        if anat is not None:
            from ..core.anatomy import AnatomicalPolicy
            pol0 = AnatomicalPolicy(anat, noise_seed=int(seed), train_std=np.exp(weights["log_std"]))
        sim = spawn(int(seed), cur, brain, policy=pol0)
        n = sim.n
        obs = np.zeros((T, n, n_in), np.float32)
        u = np.zeros((T, n, n_out), np.float32)
        rew = np.zeros((T, n), np.float32)
        hunger0, thirst0 = sim.hunger.copy(), sim.thirst.copy()
        at_food = np.zeros(n)
        at_water = np.zeros(n)
        speed = np.zeros(n)
        d_prev = discomfort(sim)
        mat_prev = sim.matings.copy()
        fights0 = (sim.fights_won + sim.fights_lost).copy()
        bond0 = sim.soc.bond().copy() if sim.soc is not None else np.zeros(n)
        tired0 = sim.soc.tired.copy() if sim.soc is not None else np.zeros(n)
        for t in range(T):
            if cur.life and cur.bond_in_brain:
                _wedding(sim, cur, t * dt)
            sim.step(t * dt, dt, events=sink)
            pol = sim.policy
            obs[t] = pol.last_obs[:, take]
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
        if anat is not None:              # which cells each animal had (a lesioned sense organ)
            cells = getattr(sim.policy, "alive", None)
            alive_l.append(np.ones((n, anat.n), np.float32) if cells is None else cells.astype(np.float32))
        stat_l.append(np.stack([
            hunger0, sim.hunger, thirst0, sim.thirst, at_food / T,
            at_water / T, speed / T, sim.dist_total, sim.health,
            (sim.fights_won + sim.fights_lost) - fights0, sim.matings,
            sim.stress, np.full(n, n, float),
            np.full(n, float(len(sim.food) + len(sim.food_rects) > 0)),
            np.full(n, float(len(sim.water) + len(sim.water_rects) > 0)),
            sim.smell.astype(float),
            np.full(n, float(len(sim.food_rects) > 0)),
            sim.bites_given.astype(float), sim.bites_taken.astype(float),
            sim.alive.astype(float), sim.sex_m.astype(float),
            *_social_stats(sim, T * dt, bond0, tired0),
        ], axis=1))
    extra = {"alive": np.concatenate(alive_l, axis=0)} if alive_l else {}
    return {**extra,
            "obs": np.concatenate(obs_l, axis=1),
            "u": np.concatenate(u_l, axis=1),
            "rew": np.concatenate(rew_l, axis=1),
            "stats": np.concatenate(stat_l, axis=0)}


def _social_stats(sim, seconds: float, bond0, tired0) -> list:
    """Per-animal: share of the life spent resting, touching anyone, touching
    its partner; attachment on arrival; tiredness on arrival and at the end;
    cold and separation distress at the end."""
    n = sim.n
    soc = sim.soc
    if soc is None:
        return [np.zeros(n)] * 8
    return [soc.rest_s / seconds, soc.touch_s / seconds, soc.partner_touch_s / seconds,
            np.asarray(bond0, float), np.asarray(tired0, float), soc.tired.copy(),
            soc.cold.copy(), soc.lonely.copy()]


#: columns of ``rollout()["stats"]``
STAT_FIELDS = ["hunger0", "hunger1", "thirst0", "thirst1", "frac_at_food",
               "frac_at_water", "mean_speed", "dist_m", "health", "fights",
               "matings", "stress", "group_size", "has_food", "has_water",
               "smell", "has_boxes", "bites_given", "bites_taken", "alive",
               "male", "rest", "touch", "partner_touch", "bond0", "tired0",
               "tired1", "cold1", "lonely1"]


def summarise(stats: np.ndarray) -> dict:
    """The handful of numbers that say whether animals are learning to live.

    Time at food is counted only over animals whose world had food (likewise
    water). ``right_choice`` is the one that shows a need being *used*: among
    animals with one need much stronger than the other, in a world offering
    both, the share of their time-at-a-resource spent at the one they needed.
    Chance is 0.5. ``box_found_*`` is the share of hungry animals, in worlds
    whose food is in boxes, that got into one and ate — for animals with a
    nose and for animals without.
    """
    s = {name: stats[:, i] for i, name in enumerate(STAT_FIELDS)}
    food, water = s["has_food"] > 0, s["has_water"] > 0
    hungry = s["hunger0"] > 50
    thirsty = s["thirst0"] > 50

    def mean(x, m):
        return float(x[m].mean()) if m.any() else 0.0

    gap = s["hunger0"] - s["thirst0"]
    # (not in box worlds: a box may hold both, and then time in it counts as
    # time at each, which would read as no preference)
    both = food & water & (np.abs(gap) > 30) & (s["has_boxes"] == 0)
    right = np.where(gap > 0, s["frac_at_food"], s["frac_at_water"])
    total = s["frac_at_food"] + s["frac_at_water"]
    used = both & (total > 0.02)
    boxed = hungry & (s["has_boxes"] > 0)
    ate = s["frac_at_food"] > 0.05
    return {
        "box_found_intact": mean(ate, boxed & (s["smell"] >= 1.0)),
        "box_found_anosmic": mean(ate, boxed & (s["smell"] <= 0.0)),
        "food_time_anosmic": mean(s["frac_at_food"],
                                  hungry & food & (s["smell"] <= 0.0)),
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
        # contest worlds only (zero elsewhere): bites per animal per episode,
        # by sex of the biter, the share of animals bitten at all, and deaths
        "bites": float(s["bites_given"].mean()),
        "bites_male": mean(s["bites_given"], s["male"] > 0),
        "bites_female": mean(s["bites_given"], s["male"] <= 0),
        "bitten": float((s["bites_taken"] > 0).mean()),
        "died": float(1.0 - s["alive"].mean()),
        # life worlds only (zero elsewhere). ``rest_tired`` and ``rest_fresh``:
        # share of the life spent resting, for animals that arrived tired and
        # for animals that arrived rested. ``touch``: share spent against
        # another body (not for animals alone in their world).
        # ``partner_share_*``: of an attached animal's time against a body,
        # the share against its partner, with a nose and without.
        "rest_tired": mean(s["rest"], s["tired0"] > 60),
        "rest_fresh": mean(s["rest"], s["tired0"] < 30),
        "touch": mean(s["touch"], s["group_size"] > 1),
        "touch_anosmic": mean(s["touch"], (s["group_size"] > 1) & (s["smell"] <= 0.0)),
        "partner_share_intact": _share(s, (s["bond0"] > 0.4) & (s["smell"] >= 1.0) & (s["group_size"] > 2)),
        "partner_share_anosmic": _share(s, (s["bond0"] > 0.4) & (s["smell"] <= 0.0) & (s["group_size"] > 2)),
        "cold": float(s["cold1"].mean()),
        "lonely_attached": mean(s["lonely1"], s["bond0"] > 0.4),
    }


def _share(s: dict, m) -> float:
    tot = s["touch"][m].sum()
    return float(s["partner_touch"][m].sum() / tot) if tot > 0 else 0.0
