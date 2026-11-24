"""What an animal's body tells its brain.

The rule-based policy reads the world directly: it knows where the nearest food
is, however far away, and steers for it. A nervous system cannot do that. It
gets what the body's sensors deliver, in the body's own frame of reference, and
has to work out the rest.

This module is that delivery. :func:`sense` turns the simulation's state into
one row of numbers per animal — the input layer of a learned brain
(:mod:`fnt.abma.core.brain`). Everything is **egocentric**: a direction is
"ahead and to the left", never "north-east", so the same brain works wherever
the animal is standing and whichever way it faces.

The channels, and why each is there
-----------------------------------
*Olfaction.* Each odorant arrives as a level and a direction (ahead, left).
An animal gets the direction from comparing its nostrils and from how the smell
changes as it moves; this module hands over the result of that comparison
rather than modelling the sniffing. Reach is short — a plume fades with a
length scale of about half a metre — so beyond a couple of metres there is
nothing to follow and the animal has to search or remember. Every olfactory
channel is multiplied by the animal's ``smell_ability``: methimazole does not
change a rule, it removes the input.

*Touch and sight.* Three whisker rays report how close a wall is ahead and to
either side. One channel reports the nearest other animal, by any sense.

*Taste.* Whether it is at food, or at water, right now.

*Place memory.* Where home is (the running centre of where it has been living),
and where it last ate and last drank. These are the only things in the list
that are not sensed in the moment, and they are what let an animal cross an
enclosure too big to smell across. An animal that has not yet found water has
no memory of it; the channel is silent until it does.

*Interoception.* Hunger, thirst, bladder, stress, pain, oestrus, the light and
the clock, and the animal's standing dispositions. The same signals also act as
**neuromodulators** — they set the gain on the brain's sensory layer (see
:mod:`fnt.abma.core.brain`) — which is how one network can follow a food smell
when it is hungry and walk straight past it when it is not.

Nothing here says what to do about any of it.
"""
from __future__ import annotations

import numpy as np

#: Ordered input channels. A brain file records the list it was trained on and
#: is matched to this one **by name**, so channels may be appended later
#: without invalidating an existing brain. Never rename or reuse a name.
SENSORS: list[str] = [
    # --- olfaction: level, ahead, left (all scaled by smell_ability) --------
    "olf_food", "olf_food_fwd", "olf_food_lat",
    "olf_water", "olf_water_fwd", "olf_water_lat",
    "olf_own", "olf_own_fwd", "olf_own_lat",                # own scent marks
    "olf_foreign", "olf_foreign_fwd", "olf_foreign_lat",    # others' marks
    "olf_male", "olf_male_fwd", "olf_male_lat",             # male body odour
    "olf_female", "olf_female_fwd", "olf_female_lat",
    "olf_estrous", "olf_estrous_fwd", "olf_estrous_lat",    # receptive female
    # --- whiskers / sight ----------------------------------------------------
    "wall_ahead", "wall_left", "wall_right",
    "other", "other_fwd", "other_lat",
    # --- taste / contact -----------------------------------------------------
    "at_food", "at_water", "grass",
    # --- place memory --------------------------------------------------------
    "home", "home_fwd", "home_lat",
    "mem_food", "mem_food_fwd", "mem_food_lat",
    "mem_water", "mem_water_fwd", "mem_water_lat",
    # --- interoception -------------------------------------------------------
    "hunger", "thirst", "bladder", "stress", "pain",
    "receptive", "male", "light", "clock_sin", "clock_cos",
    "aggression", "boldness", "sociability", "exploration",
    # --- efference copy: what it just did ------------------------------------
    "last_speed", "last_turn",
    # --- contest (appended; silent unless ContestParams is on) ---------------
    "hurt", "hurt_fwd", "hurt_lat",          # the sting of a bite, and from where
    "fatigue",
    "in_cover", "cover", "cover_fwd", "cover_lat",   # under a hut; the nearest hut
    "other_closing",                         # the nearest animal is coming at it
    "other_size",                            # ...and is bigger (+) or smaller (-)
]

#: The interoceptive channels that also act as neuromodulators — they gate the
#: brain's sensory layer rather than only feeding it.
MODULATORS: list[str] = [
    "hunger", "thirst", "stress", "pain", "receptive", "male", "light",
    "aggression", "boldness", "sociability", "exploration",
    "hurt", "fatigue",
]

#: Channel groups, for display and for ablating a whole sense at once.
GROUPS: dict[str, list[str]] = {
    "olfaction": [s for s in SENSORS if s.startswith("olf_")],
    "touch": ["wall_ahead", "wall_left", "wall_right",
              "other", "other_fwd", "other_lat"],
    "taste": ["at_food", "at_water", "grass"],
    "memory": [s for s in SENSORS
               if s.startswith(("home", "mem_"))],
    "interoception": ["hunger", "thirst", "bladder", "stress", "pain",
                      "receptive", "male", "light", "clock_sin", "clock_cos"],
    "disposition": ["aggression", "boldness", "sociability", "exploration"],
    "efference": ["last_speed", "last_turn"],
    "contest": ["hurt", "hurt_fwd", "hurt_lat", "fatigue", "in_cover", "cover",
                "cover_fwd", "cover_lat", "other_closing", "other_size"],
}

_IDX = {name: i for i, name in enumerate(SENSORS)}


class SenseParams:
    """Reach of each sense. Stored with a brain: it was trained on these."""

    def __init__(self, plume_m: float = 0.5, whisker_m: float = 0.6,
                 half_level: float = 0.05, sight_m: float = 1.5):
        #: length scale over which a food or water smell fades (m)
        self.plume_m = float(plume_m)
        #: how far ahead a wall is felt / seen (m)
        self.whisker_m = float(whisker_m)
        #: odour level at which the direction is half as reliable as at source
        self.half_level = float(half_level)
        #: how far away a hut can be seen (m)
        self.sight_m = float(sight_m)

    def to_dict(self) -> dict:
        return {"plume_m": self.plume_m, "whisker_m": self.whisker_m,
                "half_level": self.half_level, "sight_m": self.sight_m}

    @staticmethod
    def from_dict(d: dict | None) -> "SenseParams":
        return SenseParams(**(d or {}))


class PlaceMemory:
    """Where each animal last ate and last drank.

    Not learned and not sensed: it is bookkeeping the body does, the way a
    rodent that has fed somewhere can go back there. Arrays are parallel to the
    simulation's agent rows and grow with the roster.
    """

    def __init__(self, n: int):
        self.food = np.zeros((n, 2))
        self.water = np.zeros((n, 2))
        self.food_known = np.zeros(n, bool)
        self.water_known = np.zeros(n, bool)

    def resize(self, n: int) -> None:
        have = len(self.food_known)
        if n <= have:
            return
        pad = n - have
        self.food = np.vstack([self.food, np.zeros((pad, 2))])
        self.water = np.vstack([self.water, np.zeros((pad, 2))])
        self.food_known = np.concatenate([self.food_known, np.zeros(pad, bool)])
        self.water_known = np.concatenate([self.water_known,
                                           np.zeros(pad, bool)])

    def update(self, P, at_food, at_water) -> None:
        if at_food.any():
            self.food[at_food] = P[at_food]
            self.food_known |= at_food
        if at_water.any():
            self.water[at_water] = P[at_water]
            self.water_known |= at_water


def _ego(vec: np.ndarray, H: np.ndarray):
    """World-frame vectors (n, 2) as (ahead, left) components for heading H."""
    c, s = np.cos(H), np.sin(H)
    return vec[:, 0] * c + vec[:, 1] * s, -vec[:, 0] * s + vec[:, 1] * c


def _plume(P: np.ndarray, sources: np.ndarray, lam: float, half: float):
    """Level and steering direction of a smell with point sources.

    Each source contributes ``exp(-d / lam)``. The direction is the
    level-weighted mean of the bearings, scaled by how far above the detection
    floor the smell is: at the source it is a unit vector, far away it fades to
    nothing, so a faint smell is a weak hint rather than a compass.
    """
    n = len(P)
    if len(sources) == 0:
        return np.zeros(n), np.zeros((n, 2))
    d = sources[None, :, :] - P[:, None, :]              # (n, m, 2)
    dist = np.linalg.norm(d, axis=2) + 1e-9
    w = np.exp(-dist / lam)                              # (n, m)
    level = np.clip(w.sum(axis=1), 0.0, 1.0)
    g = (w[:, :, None] * d / dist[:, :, None]).sum(axis=1)
    gm = np.linalg.norm(g, axis=1) + 1e-12
    conf = level / (level + half)
    return level, g / gm[:, None] * conf[:, None]


def _toward(P: np.ndarray, target: np.ndarray, known=None):
    """Closeness (1 at the spot, falling with distance) and bearing to a place."""
    v = target - P
    d = np.linalg.norm(v, axis=1)
    near = 1.0 / (1.0 + d)
    # the bearing fades out on top of the target, where it means nothing
    unit = v / (d[:, None] + 1e-9) * np.minimum(1.0, d / 0.10)[:, None]
    if known is not None:
        k = known.astype(float)
        near = near * k
        unit = unit * k[:, None]
    return near, unit


def _wall_rays(sim, P: np.ndarray, H: np.ndarray, reach: float) -> np.ndarray:
    """Proximity (0 clear .. 1 touching) of the nearest surface along three
    rays: ahead, 60° left and 60° right. Counts the arena boundary and every
    solid the engine blocks movement with."""
    n = len(P)
    out = np.zeros((n, 3))
    w, h = sim.cfg.arena.width, sim.cfg.arena.height
    bounded = sim.cfg.arena.boundary != "wrap"
    for k, off in enumerate((0.0, np.pi / 3, -np.pi / 3)):
        ang = H + off
        dx, dy = np.cos(ang) * reach, np.sin(ang) * reach
        B = P + np.stack([dx, dy], axis=1)
        t = sim._first_hit(P, B)
        if bounded:
            with np.errstate(divide="ignore", invalid="ignore"):
                tx = np.where(dx > 0, (w - P[:, 0]) / dx,
                              np.where(dx < 0, -P[:, 0] / dx, np.inf))
                ty = np.where(dy > 0, (h - P[:, 1]) / dy,
                              np.where(dy < 0, -P[:, 1] / dy, np.inf))
            t = np.minimum(t, np.clip(np.minimum(tx, ty), 0.0, None))
        out[:, k] = 1.0 - np.clip(t, 0.0, 1.0)
    return out


def sense(sim, elapsed_s: float, memory: PlaceMemory, params: SenseParams,
          last_speed=None, last_turn=None):
    """One row of sensory input per animal, plus what the engine needs.

    Returns ``(obs, perception)``. ``obs`` is ``(n, len(SENSORS))`` float32.
    ``perception`` carries the same intermediates the rule-based policy hands
    the engine (pairwise distances, who is within reach, need for food and
    water, scent levels), so the engine does not care which kind of policy is
    driving.
    """
    P, H, n = sim.P, sim.H, sim.n
    alive = sim.alive
    alive_f = alive.astype(float)
    smell = sim.smell
    pp = sim.cfg.policy
    obs = np.zeros((n, len(SENSORS)))

    def put(name, value):
        obs[:, _IDX[name]] = value

    def put_vec(prefix, level, vec, gain=1.0):
        fwd, lat = _ego(vec, H)
        put(prefix, level * gain)
        put(prefix + "_fwd", fwd * gain)
        put(prefix + "_lat", lat * gain)

    # ---- where the animal is standing ---------------------------------- #
    food_idx = sim._food_index()
    at_food = (food_idx >= 0) & alive
    at_water = sim._on_water(P) & alive
    rects = getattr(sim, "food_rects", np.zeros((0, 4)))
    doors = getattr(sim, "zone_doors", np.zeros((0, 2)))
    in_zone = np.zeros(n, bool)
    zone_of = np.zeros(n, int)
    if len(rects) and len(doors):
        cx, cy, hw, hd = rects.T
        inz = ((np.abs(P[:, 0][:, None] - cx[None, :]) <= hw[None, :])
               & (np.abs(P[:, 1][:, None] - cy[None, :]) <= hd[None, :]))
        in_zone = inz.any(axis=1)
        zone_of = np.argmax(inz, axis=1)

    def through_door(vec):
        """Inside a walled zone, every smell from outside comes in through the
        doorway — so that is the direction it has."""
        if not in_zone.any():
            return vec
        to_door = sim.exit_door(P, zone_of) - P
        dd = np.linalg.norm(to_door, axis=1)[:, None] + 1e-9
        mag = np.linalg.norm(vec, axis=1)[:, None]
        return np.where(in_zone[:, None], to_door / dd * mag, vec)

    # ---- olfaction ------------------------------------------------------ #
    lam, half = params.plume_m, params.half_level
    f_lvl, f_vec = _plume(P, getattr(sim, "food_seek", sim.food), lam, half)
    # standing in the chow there is no gradient to follow: it is everywhere
    f_lvl = np.where(at_food, 1.0, f_lvl)
    f_vec = np.where(at_food[:, None], 0.0, f_vec)
    put_vec("olf_food", f_lvl, f_vec, smell)
    w_lvl, w_vec = _plume(P, getattr(sim, "water_seek", sim.water), lam, half)
    # ...and likewise at the water. Leaving a direction on here made a small
    # source a moving target: the bearing swings as the animal jitters across
    # the centre, and it overshoots instead of stopping to drink.
    w_lvl = np.where(at_water, 1.0, w_lvl)
    w_vec = np.where(at_water[:, None], 0.0, through_door(w_vec))
    put_vec("olf_water", w_lvl, w_vec, smell)

    own_lvl = foreign_lvl = None
    if getattr(sim, "scent", None) is not None:
        disc = (sim.discrimination_matrix()
                if getattr(sim, "olf", None) is not None else None)
        own_vec, foreign_vec, own_lvl, foreign_lvl = sim.scent.sample(
            P, np.arange(n), recog=disc)
        put_vec("olf_own", own_lvl, own_vec, smell)
        # `sample` returns the direction AWAY from foreign marks (it was
        # written for avoidance); the nose reports where the smell IS and
        # leaves what to do about it to the brain
        put_vec("olf_foreign", foreign_lvl, -foreign_vec, smell)

    # ---- the other animals --------------------------------------------- #
    diff = P[None, :, :] - P[:, None, :]                 # diff[i, j] = Pj - Pi
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    unit = diff / (dist[:, :, None] + 1e-9)
    reach = max(1e-6, float(pp.perception_r))
    within = (dist < reach) * alive_f[:, None] * alive_f[None, :]
    rec_j = sim._receptivity(elapsed_s)
    wj = np.exp(-dist / reach) * alive_f[None, :]        # diagonal -> 0
    for name, cls in (("olf_male", sim.sex_m), ("olf_female", 1.0 - sim.sex_m),
                      ("olf_estrous", rec_j)):
        wc = wj * cls[None, :]
        lvl = np.clip(wc.sum(axis=1), 0.0, 1.0)
        g = np.einsum("ij,ijk->ik", wc, unit)
        gm = np.linalg.norm(g, axis=1) + 1e-12
        vec = g / gm[:, None] * (lvl / (lvl + half))[:, None]
        put_vec(name, lvl, vec, smell)

    # nearest other animal, by sight / sound / touch — not gated by the nose
    if n > 1:
        dm = np.where(alive[None, :], dist, np.inf)
        j = np.argmin(dm, axis=1)
        dj = dm[np.arange(n), j]
        near = np.clip(1.0 - dj / reach, 0.0, 1.0)
        near = np.where(np.isfinite(dj), near, 0.0)
        put_vec("other", near, unit[np.arange(n), j] * (near > 0)[:, None])
        # how fast that animal is coming at this one (its own velocity along
        # the line between them; this one's own running is not counted), and
        # how its size compares. Both are things an eye and an ear deliver.
        V = getattr(sim, "V", None)
        if V is not None and len(V) == n:
            seen = near > 0
            toward = -np.einsum("ik,ik->i", V[j], unit[np.arange(n), j])
            top = np.maximum(1e-6, sim.speed * float(sim.cfg.contest.sprint_factor))
            put("other_closing", np.where(seen, np.clip(toward / top, -1.0, 1.0), 0.0))
            put("other_size", np.where(
                seen, np.tanh(2.0 * (sim.mass[j] - sim.mass) / np.maximum(1e-6, sim.mass)), 0.0))

    # ---- contest: the sting of a bite, tiredness, cover ----------------- #
    hurt = getattr(sim, "hurt", None)
    if hurt is not None and len(hurt) == n:
        put_vec("hurt", hurt, sim.hurt_dir * hurt[:, None])
        put("fatigue", sim.fatigue / 100.0)
    cover = getattr(sim, "_cover", None)
    if cover is not None and len(cover):
        dc = np.linalg.norm(P[:, None, :] - cover[None, :, :2], axis=2)
        k = np.argmin(dc, axis=1)
        dk = dc[np.arange(n), k]
        inside = dk <= cover[k, 2]
        put("in_cover", inside.astype(float))
        vc = cover[k, :2] - P
        seen = np.clip(1.0 - dk / max(1e-6, params.sight_m), 0.0, 1.0)
        put_vec("cover", np.where(inside, 1.0, seen),
                vc / (dk[:, None] + 1e-9) * ((seen > 0) & ~inside)[:, None])

    # ---- walls, taste, grass ------------------------------------------- #
    rays = _wall_rays(sim, P, H, params.whisker_m)
    put("wall_ahead", rays[:, 0])
    put("wall_left", rays[:, 1])
    put("wall_right", rays[:, 2])
    put("at_food", at_food.astype(float))
    put("at_water", at_water.astype(float))
    # only refreshed while a sward exists, and not grown when a protocol event
    # adds animals — so trust it only when it matches the roster
    grass = getattr(sim, "grass_cm", None)
    if grass is not None and len(grass) == n:
        put("grass", np.clip(grass / 15.0, 0, 1))

    # ---- place memory --------------------------------------------------- #
    memory.resize(n)
    memory.update(P, at_food, at_water)
    hv = sim.home - P
    dist_home = np.linalg.norm(hv, axis=1) + 1e-9
    near, vec = _toward(P, sim.home)
    put_vec("home", near, vec)
    near, vec = _toward(P, memory.food, memory.food_known)
    put_vec("mem_food", near, through_door(vec))
    near, vec = _toward(P, memory.water, memory.water_known)
    put_vec("mem_water", near, through_door(vec))

    # ---- interoception -------------------------------------------------- #
    put("hunger", sim.hunger / 100.0)
    put("thirst", sim.thirst / 100.0)
    put("bladder", sim.bladder / 100.0)
    put("stress", sim.stress / 100.0)
    put("pain", (100.0 - sim.health) / 100.0)
    put("receptive", rec_j)
    put("male", sim.sex_m)
    sky = sim.sky_at(elapsed_s)
    put("light", sky.daylight if sky is not None
        else float(sim._is_day(elapsed_s)))
    phase = 2.0 * np.pi * sim._hour(elapsed_s) / 24.0
    put("clock_sin", np.sin(phase))
    put("clock_cos", np.cos(phase))
    put("aggression", np.clip(sim.aggr, 0, 1))
    put("boldness", np.clip(sim.bold / 2.0, 0, 1))
    put("sociability", np.clip(sim.social / 2.0, 0, 1))
    put("exploration", np.clip(sim.explore / 2.0, 0, 1))
    if last_speed is not None:
        put("last_speed", last_speed)
    if last_turn is not None:
        put("last_turn", last_turn)

    obs[~alive] = 0.0

    thr = float(pp.forage_threshold)
    span = max(1e-6, 100.0 - thr)
    perception = {
        "dist": dist, "dist_home": dist_home, "within": within,
        "rec_j": rec_j,
        "need_food": np.clip((sim.hunger - thr) / span, 0, 1),
        "need_water": np.clip((sim.thirst - thr) / span, 0, 1),
        "scent_own": own_lvl, "scent_foreign": foreign_lvl,
        "recognition": sim.recognition_matrix(),
        "at_food": at_food, "at_water": at_water,
    }
    return obs.astype(np.float32), perception


def index(name: str) -> int:
    """Column of a named sensor channel."""
    return _IDX[name]
