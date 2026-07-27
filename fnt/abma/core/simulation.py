"""ABMA simulation engine (headless, vectorised).

One :class:`Simulation` runs a single trial: it builds a population from an
:class:`ExperimentConfig`, integrates agent movement and physiology on a fixed
timestep, resolves social events, and streams output in FNT's canonical schema.

Behavioural model (v1)
----------------------
Each agent's desired heading each step is a weighted blend of:
  * **home-range attraction** — a spring toward an assigned nest, sized by the
    agent's ``home_range_r`` (males larger). Produces residency & territories.
  * **resource seeking** — toward the nearest food/water when hungry/thirsty.
  * **social & territorial forces** — pairwise, *gated by olfaction*:
      - opposite-sex attraction, amplified by female estrus (mate seeking);
      - female–female affiliation scaled by sociability;
      - male–male territorial avoidance scaled by aggression.
    Recognition strength ``= smell_i * identity_signal_j``. Anosmia (methimazole)
    or loss of identity signal (MUP-KO) collapses recognition, degrading clean
    territorial spacing and reshaping the social network — the core manipulation.
  * **correlated random walk** — exploratory noise.

The magnitudes are deliberately transparent constants (``self.k_*``) so they can
be tuned from the GUI or a config extension.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from copy import deepcopy

from .config import ExperimentConfig, AgentGroup
from .biology import (
    resolve_traits, apply_drug, TRAIT_TO_ARRAY, _TRAIT_RANGES,
)
from .sampling import parse_spec
from .policy import RuleBasedPolicy
from .recorder import (
    TrajectoryRecorder, EventRecorder, ConditionRecorder, write_agents_table,
    parse_start,
)


_ARRAY_TO_TRAIT = {v: k for k, v in TRAIT_TO_ARRAY.items()}
_SHAPE_CODE = {"rodent": 0, "blob": 1, "bird": 2}
_MALE_RGBA = (0.29, 0.56, 0.85, 1.0)
_FEMALE_RGBA = (0.88, 0.33, 0.60, 1.0)


def _zone_door(z):
    """Outside-facing centre of a resource zone's doorway."""
    side = getattr(z, "entrance", "E")
    if side == "E":
        return (z.x + z.w / 2, z.y)
    if side == "W":
        return (z.x - z.w / 2, z.y)
    if side == "N":
        return (z.x, z.y + z.d / 2)
    return (z.x, z.y - z.d / 2)


def _appearance_rgba(appearance, sex):
    """Resolve an agent's base colour: explicit hex, else auto by sex."""
    hex_col = getattr(appearance, "color", "") or ""
    if hex_col.startswith("#") and len(hex_col) == 7:
        try:
            r = int(hex_col[1:3], 16) / 255.0
            g = int(hex_col[3:5], 16) / 255.0
            b = int(hex_col[5:7], 16) / 255.0
            return (r, g, b, 1.0)
        except ValueError:
            pass
    return _MALE_RGBA if sex == "M" else _FEMALE_RGBA


@dataclass
class AgentMeta:
    """Static, per-agent identity and resolved biology."""
    index: int
    sexid: str
    shortid: int
    species: str
    sex: str
    group: str
    genotype: object
    treatment: object
    traits: object
    appearance: object = None
    home: np.ndarray = field(default_factory=lambda: np.zeros(2))
    alive: bool = True


class Simulation:
    # Movement-decision weights now live on the Policy (see policy.py). The
    # engine keeps the physics/physiology/combat constants below.
    contact_r = 0.12      # interaction/contact radius (m)
    # how far an animal can feed/drink from a structure's surface: body reach
    # plus a step of slack, so contact is not knife-edge at coarse dt.
    _REACH = 0.18
    estrus_period_days = 4.0
    mate_rate = 0.02      # per-tick mating probability when in contact & receptive
    aggr_rate = 0.6       # per-encounter contest probability scaler (× aggr_i × aggr_j)
    fight_cooldown_s = 3600.0   # min interval between contests for a dyad
    mate_cooldown_s = 3600.0    # min interval between matings for a dyad

    def __init__(self, config: ExperimentConfig, trial_index: int = 0,
                 seed: int | None = None, policy=None):
        self.cfg = config
        self.trial_index = trial_index
        self.trial_id = f"{config.trial_prefix}{trial_index + 1:03d}"
        self.rng = np.random.default_rng(
            config.seed + trial_index if seed is None else seed)
        self.start_dt = parse_start(config.start_datetime)
        self.policy = policy if policy is not None else RuleBasedPolicy()
        self._build_population()

    # ------------------------------------------------------------------ #
    # Population construction
    # ------------------------------------------------------------------ #
    def _build_population(self) -> None:
        cfg = self.cfg
        nests = cfg.arena.objects_of("nest")
        self.agents: list[AgentMeta] = []
        idx = 0
        base_id = 9000 + self.trial_index * 100
        # Animals are released together near the arena centre (as in a real
        # enclosure release) and then self-organise; home ranges are emergent.
        release = np.array([cfg.arena.width / 2, cfg.arena.height / 2])
        self._schedule: list[tuple[float, int, str, float]] = []  # onset,idx,attr,val
        self._last_fight: dict[tuple[int, int], float] = {}  # dyad -> last fight time
        self._last_mate: dict[tuple[int, int], float] = {}   # dyad -> last mating
        sd = cfg.individual_variation
        for g in cfg.groups:
            delayed = (g.treatment.drug not in ("none", None)
                       and g.treatment.day_offset > 0.0)
            specs = {k: parse_spec(v) for k, v in (g.dists or {}).items()}
            for _ in range(g.count):
                # seed each founder's innate attributes from the group's specs
                base = deepcopy(g.traits)
                for tname, spec in specs.items():
                    if hasattr(base, tname):
                        setattr(base, tname, spec.sample(self.rng))
                # pre-onset profile: drug excluded if delivered after release
                traits = resolve_traits(base, g.genotype, g.treatment,
                                        drug_active=(False if delayed else None))
                if sd > 0:  # global jitter only for traits without an explicit spec
                    self._jitter_traits(traits, sd, skip=set(specs))
                shortid = base_id + idx + 1
                sexid = f"{g.sex}{shortid}"
                start = release + self.rng.normal(0, 0.15, 2)
                self.agents.append(AgentMeta(
                    index=idx, sexid=sexid, shortid=shortid,
                    species=g.species, sex=g.sex, group=g.label,
                    genotype=g.genotype, treatment=g.treatment,
                    traits=traits, appearance=getattr(g, "appearance", None),
                    home=start.copy(),
                ))
                if delayed:  # schedule the drug to take effect mid-experiment
                    post = deepcopy(traits)
                    apply_drug(post, g.treatment)
                    onset_s = g.treatment.day_offset * 86400.0
                    for tname, arr in TRAIT_TO_ARRAY.items():
                        if abs(getattr(post, tname) - getattr(traits, tname)) > 1e-9:
                            self._schedule.append(
                                (onset_s, idx, arr, getattr(post, tname)))
                idx += 1

        n = len(self.agents)
        self.n = n
        # ---- state arrays ----
        self.home = np.array([a.home for a in self.agents], float)
        self.P = np.clip(self.home.copy(), 0.02,
                         [cfg.arena.width - 0.02, cfg.arena.height - 0.02])
        # home-range adaptation rate: home tracks a slow average of position so
        # territories emerge over ~half a day rather than being prescribed.
        self._settle_tau_s = 0.5 * 86400.0
        self.H = self.rng.uniform(0, 2 * np.pi, n)
        self._build_obstacles()
        # ---- condition (dynamic 0..1 bars; presented ×100) ----
        self.hunger = self.rng.uniform(0, 0.3, n)
        self.thirst = self.rng.uniform(0, 0.3, n)
        self.energy = np.ones(n)
        self.health = np.ones(n)
        self.stress = np.full(n, 0.1)
        self.alive = np.ones(n, bool)
        self.estrus_phase = self.rng.uniform(0, 1, n)

        # ---- behaviour counters / bookkeeping ----
        self.fights_won = np.zeros(n, int)
        self.fights_lost = np.zeros(n, int)
        self.matings = np.zeros(n, int)
        self.dist_today = np.zeros(n)
        self.dist_total = np.zeros(n)
        self.activity = np.zeros(n, int)   # 0 rest,1 forage,2 roam,3 flee,4 mate,5 dead
        self._cur_day = 1

        # ---- trait vectors ----
        self.sex_m = np.array([1.0 if a.sex == "M" else 0.0
                               for a in self.agents])
        self.aggr = np.array([a.traits.aggression for a in self.agents])
        self.bold = np.array([a.traits.boldness for a in self.agents])
        self.social = np.array([a.traits.sociability for a in self.agents])
        self.explore = np.array([a.traits.exploration for a in self.agents])
        self.smell = np.array([a.traits.smell_ability for a in self.agents])
        self.identity = np.array([a.traits.identity_signal for a in self.agents])
        self.speed = np.array([a.traits.base_speed for a in self.agents])
        self.home_r = np.array([a.traits.home_range_r for a in self.agents])
        self.mass = np.array([a.traits.mass for a in self.agents])
        self.mass0 = self.mass.copy()      # release mass, for drift reporting
        self.metabolism = np.array([a.traits.metabolism for a in self.agents])
        self.turn_rate = np.array([a.traits.turn_rate for a in self.agents])
        self.wander = np.array([a.traits.wander for a in self.agents])

        # ---- appearance (per-agent colour / size / shape for the views) ----
        self.agent_rgba = np.array(
            [_appearance_rgba(a.appearance, a.sex) for a in self.agents])
        self.agent_size = np.array(
            [getattr(a.appearance, "size", 1.0) or 1.0 for a in self.agents])
        self.agent_shape = np.array(
            [_SHAPE_CODE.get(getattr(a.appearance, "shape", "rodent"), 0)
             for a in self.agents])

        # ---- condition-dynamics ruleset (editable interaction table) ----
        self.dynamics = list(getattr(cfg, "dynamics", []) or [])

        # ---- scheduled interventions (target, attribute, op, value, at time) ----
        self._iv_schedule = []
        for iv in getattr(cfg, "interventions", []):
            arr = TRAIT_TO_ARRAY.get(iv.attribute)
            if arr is None:
                continue
            idxs = [a.index for a in self.agents if self._match_target(a, iv.target)]
            if idxs:
                self._iv_schedule.append(
                    (iv.at_day * 86400.0, idxs, arr, iv.op, float(iv.value)))

        # ---- resources ----
        # Built structures count as resources, not just decoration: a resource
        # zone holds the chow pile, a water tower holds water. Otherwise an
        # enclosure that visibly contains food would starve its animals.
        # A walled resource zone is only reachable through its doorway, so the
        # animal steers for the *entrance* while feeding happens anywhere
        # inside. Seek target and feeding region are therefore separate: aiming
        # at the centre would just press the animal against the outside wall.
        food, food_seek = [], []
        for o in cfg.arena.objects_of("food"):
            food.append((o.x, o.y, o.radius))
            food_seek.append((o.x, o.y))
        for z in getattr(cfg.arena, "resource_zones", []):
            # anywhere inside the box counts as being at the chow pile
            food.append((z.x, z.y, float(np.hypot(z.w, z.d) / 2.0)))
            food_seek.append(_zone_door(z))
        water, water_seek = [], []
        for o in cfg.arena.objects_of("water"):
            water.append((o.x, o.y, o.radius))
            water_seek.append((o.x, o.y))
        for t in getattr(cfg.arena, "water_towers", []):
            # a tower is also a solid: collision holds the animal at
            # radius + body, so the drinkable band must clear that and add a
            # reach margin, or there is no reachable annulus at all.
            water.append((t.x, t.y, t.radius + self._REACH))
            water_seek.append((t.x, t.y))
        self.food = np.array([[x, y] for x, y, _ in food], float).reshape(-1, 2)
        self.water = np.array([[x, y] for x, y, _ in water], float).reshape(-1, 2)
        self.food_r = np.array([r for _, _, r in food], float)
        self.water_r = np.array([r for _, _, r in water], float)
        self.food_seek = np.array(food_seek, float).reshape(-1, 2)
        self.water_seek = np.array(water_seek, float).reshape(-1, 2)

    # ------------------------------------------------------------------ #
    # Circadian activity
    # ------------------------------------------------------------------ #
    def _hour(self, elapsed_s: float) -> float:
        return (self.start_dt.hour + elapsed_s / 3600.0) % 24.0

    def _is_day(self, elapsed_s: float) -> bool:
        # day window is [day_start, day_start + 12); night otherwise
        return (self._hour(elapsed_s) - self.cfg.day_start_hour) % 24 < 12

    def _activity(self, elapsed_s: float) -> float:
        return self.cfg.day_activity if self._is_day(elapsed_s) \
            else self.cfg.night_activity

    def _match_target(self, agent, target: str) -> bool:
        t = (target or "all").strip()
        return t in ("all", "*", "") or t == agent.group or t == agent.sexid

    def _jitter_traits(self, traits, sd: float, skip=None) -> None:
        """Apply multiplicative Gaussian jitter for between-individual variation."""
        skip = skip or set()
        for tname in TRAIT_TO_ARRAY:
            if tname in skip:
                continue
            v = getattr(traits, tname) * (1.0 + self.rng.normal(0, sd))
            lo, hi = _TRAIT_RANGES.get(tname, (0.0, 1.0))
            setattr(traits, tname, min(hi, max(lo, v)))

    # ------------------------------------------------------------------ #
    # One integration step
    # ------------------------------------------------------------------ #
    def step(self, elapsed_s: float, dt: float, events=None) -> None:
        cfg = self.cfg
        n = self.n
        P = self.P

        # --- apply any treatments whose onset time has arrived ---
        if self._schedule:
            still = []
            for onset_s, i, attr, val in self._schedule:
                if elapsed_s >= onset_s:
                    getattr(self, attr)[i] = val
                    setattr(self.agents[i].traits, _ARRAY_TO_TRAIT[attr], val)
                else:
                    still.append((onset_s, i, attr, val))
            self._schedule = still

        # --- apply any scheduled interventions whose time has arrived ---
        if self._iv_schedule:
            still = []
            for at_s, idxs, arr, op, val in self._iv_schedule:
                if elapsed_s >= at_s:
                    a = getattr(self, arr)
                    for i in idxs:
                        if op == "set":
                            a[i] = val
                        elif op == "scale":
                            a[i] *= val
                        elif op == "add":
                            a[i] += val
                        setattr(self.agents[i].traits, _ARRAY_TO_TRAIT[arr], a[i])
                    if events is not None and idxs:
                        events.record(elapsed_s, "intervention",
                                      self.agents[idxs[0]], None,
                                      self.P[idxs[0], 0], self.P[idxs[0], 1],
                                      val)
                else:
                    still.append((at_s, idxs, arr, op, val))
            self._iv_schedule = still

        # --- decision: the policy turns state into a desired heading ---
        desired, perc = self.policy.decide(self, elapsed_s)
        dist = perc["dist"]
        dist_home = perc["dist_home"]
        within = perc["within"]
        rec_j = perc["rec_j"]
        need_food = perc["need_food"]
        need_water = perc["need_water"]

        # --- resolve to movement ---
        dmag = np.linalg.norm(desired, axis=1) + 1e-9
        move_dir = desired / dmag[:, None]
        activity = self._activity(elapsed_s)
        # energy -> speed coupling (low energy is sluggish); config-driven
        f = cfg.energy_speed_coupling
        spd = self.speed * activity * (1.0 - f + f * self.energy)
        satiated = (self.hunger < 0.5) & (self.thirst < 0.5) & \
                   (dist_home < self.home_r)
        spd = np.where(satiated, spd * cfg.rest_speed_factor, spd)
        self.P = self._resolve_obstacles(P, P + move_dir * spd[:, None] * dt,
                                         fwd=move_dir)
        self.H = np.arctan2(move_dir[:, 1], move_dir[:, 0])
        self._apply_boundary()
        if not self.alive.all():          # dead animals do not move
            self.P[~self.alive] = P[~self.alive]

        # distance bookkeeping (per-day resets at midnight)
        step_dist = np.linalg.norm(self.P - P, axis=1)
        self.dist_today += step_dist
        self.dist_total += step_dist
        day = int(elapsed_s // 86400) + 1
        if day != self._cur_day:
            self.dist_today[:] = 0.0
            self._cur_day = day

        # emergent home range: home slowly tracks occupied position (living only)
        self.home[self.alive] += ((self.P[self.alive] - self.home[self.alive])
                                  * (dt / self._settle_tau_s))

        # --- condition dynamics (data-driven interaction rules) ---
        on_food = (self._on_resource(self.P, self.food, self.food_r)
                   if self.food.shape[0] else np.zeros(n, bool))
        on_water = (self._on_resource(self.P, self.water, self.water_r)
                    if self.water.shape[0] else np.zeros(n, bool))
        self._apply_dynamics(dt, step_dist, activity, within.sum(axis=1),
                             on_food, on_water)

        # --- body mass drifts with energy balance, within a physiological band ---
        self.mass = np.clip(self.mass + dt * 1.5e-5 * (self.energy - 0.7),
                            0.7 * self.mass0, 1.3 * self.mass0)

        # --- baseline activity label (events may override to flee/mate) ---
        self.activity = np.where(
            satiated, 0, np.where((need_food > 0) | (need_water > 0), 1, 2))
        self.activity[~self.alive] = 5

        # --- mortality (starvation or fatal injury drive health to 0) ---
        if cfg.enable_mortality:
            for i in np.nonzero((self.health <= 0.0) & self.alive)[0]:
                self.alive[i] = False
                self.agents[i].alive = False
                self.activity[i] = 5
                if events is not None:
                    events.record(elapsed_s, "death", self.agents[i], None,
                                  self.P[i, 0], self.P[i, 1])

        # --- social events (mating, combat) ---
        if events is not None:
            self._resolve_events(elapsed_s, dist, rec_j, events)

    # ------------------------------------------------------------------ #
    def _seek(self, P, targets, need):
        d = np.linalg.norm(P[:, None, :] - targets[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        tv = targets[nearest] - P
        tdir = tv / (np.linalg.norm(tv, axis=1)[:, None] + 1e-9)
        return need[:, None] * tdir

    def _on_resource(self, P, targets, radii):
        d = np.linalg.norm(P[:, None, :] - targets[None, :, :], axis=2)
        return np.any(d < radii[None, :], axis=1)

    def _apply_dynamics(self, dt, step_dist, activity, n_near, on_food, on_water):
        """Apply the editable interaction rules to the condition variables.

        Each rule: target += gain × source × (dt/hour), or (effect='set')
        target = gain where source is active. Sources are read from a snapshot so
        rules act simultaneously (forward Euler), and gains are per-hour.
        """
        if not self.dynamics:
            return
        n = self.n
        dth = dt / 3600.0
        snap = {k: getattr(self, k).copy()
                for k in ("energy", "hunger", "thirst", "stress", "health")}
        speed = step_dist / dt
        mass_rel = self.mass / 40.0
        cache = {}                       # each distinct source is computed once

        def src(name):
            v = cache.get(name)
            if v is not None:
                return v
            if name == "time":
                v = np.ones(n)
            elif name == "movement":
                v = speed
            elif name == "activity":
                v = np.full(n, activity)
            elif name == "crowding":
                v = n_near.astype(float)
            elif name == "on_food":
                v = on_food.astype(float)
            elif name == "on_water":
                v = on_water.astype(float)
            elif name == "mass":
                v = mass_rel
            elif name == "metabolism":
                v = self.metabolism
            elif name == "fed":
                v = 1.0 - snap["hunger"]
            elif name == "hydrated":
                v = 1.0 - snap["thirst"]
            elif name == "rested":
                v = 1.0 - snap["stress"]
            elif name in snap:
                v = snap[name]
            else:
                v = np.zeros(n)
            cache[name] = v
            return v

        for c in self.dynamics:
            tgt = getattr(self, c.target, None)
            if tgt is None:
                continue
            s = src(c.source)
            if c.effect == "set":
                tgt[s > 0.5] = c.gain
                continue
            if c.scale_by == "mass":
                f = s * mass_rel
            elif c.scale_by == "activity":
                f = s * activity
            elif c.scale_by == "metabolism":
                f = s * self.metabolism
            else:
                f = s                    # no copy — never mutated below
            if c.only_when == "source_high":
                f = f * (s > c.threshold)
            elif c.only_when == "source_low":
                f = f * (s < c.threshold)
            tgt += (c.gain * dth) * f

        for k in snap:
            arr = getattr(self, k)
            np.clip(arr, 0.0, 1.0, out=arr)

    def _receptivity(self, elapsed_s):
        days = elapsed_s / 86400.0
        r = 0.5 * (1 + np.sin(2 * np.pi * (days / self.estrus_period_days
                                           + self.estrus_phase)))
        r = np.clip((r - 0.6) / 0.4, 0, 1)  # only high near peak
        return r * (1 - self.sex_m)          # males not receptive

    # ------------------------------------------------------------------ #
    # Physical obstacles: poles + water towers (circles), resource-zone
    # walls with a doorway gap (segments). Agents cannot pass through them.
    # ------------------------------------------------------------------ #
    def _build_obstacles(self):
        self._agent_r = 0.02      # ~half a body width (m)
        # the drawn body is a box + head sphere reaching ~8 cm ahead of centre;
        # the nose is tested too so heads cannot poke through solids.
        self._nose_r = 0.08
        a = self.cfg.arena
        circ = []                                  # (cx, cy, r_eff)
        for p in getattr(a, "poles", []):
            circ.append((p.x, p.y, p.radius + self._agent_r))
        for wt in getattr(a, "water_towers", []):
            circ.append((wt.x, wt.y, wt.radius + self._agent_r))
        segs = []                                  # (x1, y1, x2, y2) wall panels
        for z in getattr(a, "resource_zones", []):
            hw = getattr(z, "hole", 0.0762)
            x0, x1 = z.x - z.w / 2, z.x + z.w / 2
            y0, y1 = z.y - z.d / 2, z.y + z.d / 2
            side = getattr(z, "entrance", "E")
            for yw, name in ((y1, "N"), (y0, "S")):        # walls running E-W
                if name == side:                    # split around the doorway
                    segs += [(x0, yw, z.x - hw / 2, yw),
                             (z.x + hw / 2, yw, x1, yw)]
                else:
                    segs += [(x0, yw, x1, yw)]
            for xw, name in ((x1, "E"), (x0, "W")):        # walls running N-S
                if name == side:
                    segs += [(xw, y0, xw, z.y - hw / 2),
                             (xw, z.y + hw / 2, xw, y1)]
                else:
                    segs += [(xw, y0, xw, y1)]
        self._obs_circles = np.array(circ, float) if circ else np.zeros((0, 3))
        self._obs_segs = np.array(segs, float) if segs else np.zeros((0, 4))

    def _resolve_obstacles(self, P_old, P_new, fwd=None):
        """Block moves that cross a wall or enter a solid; no tunnelling.

        Both the body centre and the nose (``fwd`` × nose radius ahead of it)
        are swept, so the drawn body never overlaps a solid — while a doorway
        wide enough for the animal still lets it through.
        """
        n = len(P_old)
        tmin = np.ones(n)
        offsets = [None] if fwd is None else [None, fwd * self._nose_r]
        for off in offsets:
            A = P_old if off is None else P_old + off
            B = P_new if off is None else P_new + off
            tmin = np.minimum(tmin, self._first_hit(A, B))
        moved = tmin < 1.0
        if moved.any():
            scale = np.where(moved, np.maximum(tmin - 0.02, 0.0), 1.0)
            P_new = P_old + (P_new - P_old) * scale[:, None]
        # push out anything that began the step already inside a solid
        C = self._obs_circles
        if len(C):
            cxa, cya, cra = C.T
            for _ in range(2):
                dx = P_new[:, 0][:, None] - cxa[None, :]
                dy = P_new[:, 1][:, None] - cya[None, :]
                dist = np.hypot(dx, dy)
                inside = dist < cra[None, :]
                if not inside.any():
                    break
                dm = np.where(inside, dist, np.inf)
                j = np.argmin(dm, axis=1)
                for i in np.where(np.isfinite(dm[np.arange(n), j]))[0]:
                    k = j[i]
                    d = max(dist[i, k], 1e-9)
                    P_new[i] = [cxa[k] + dx[i, k] / d * cra[k],
                                cya[k] + dy[i, k] / d * cra[k]]
        return P_new

    def _first_hit(self, A, B):
        """Earliest fraction along A->B that crosses a wall or enters a solid."""
        n = len(A)
        tmin = np.ones(n)
        dxp = (B[:, 0] - A[:, 0])[:, None]
        dyp = (B[:, 1] - A[:, 1])[:, None]
        # --- walls (path vs segment) ---
        S = self._obs_segs
        if len(S):
            sx = (S[:, 2] - S[:, 0])[None, :]
            sy = (S[:, 3] - S[:, 1])[None, :]
            denom = dxp * sy - dyp * sx
            cax = S[:, 0][None, :] - A[:, 0][:, None]
            cay = S[:, 1][None, :] - A[:, 1][:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (cax * sy - cay * sx) / denom
                u = (cax * dyp - cay * dxp) / denom
            hit = (np.abs(denom) > 1e-12) & (t >= 0) & (t <= 1) & \
                  (u >= 0) & (u <= 1)
            tmin = np.minimum(tmin, np.where(hit, t, np.inf).min(axis=1))
        # --- solids (swept path vs circle: poles + water towers) ---
        C = self._obs_circles
        if len(C):
            cx, cy, cr = C[:, 0][None, :], C[:, 1][None, :], C[:, 2][None, :]
            fx = A[:, 0][:, None] - cx
            fy = A[:, 1][:, None] - cy
            aa = dxp * dxp + dyp * dyp
            bb = 2 * (fx * dxp + fy * dyp)
            cc = fx * fx + fy * fy - cr * cr
            disc = bb * bb - 4 * aa * cc
            sq = np.sqrt(np.maximum(disc, 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                tc = (-bb - sq) / (2 * aa)
            hitc = (disc >= 0) & (tc >= 0) & (tc <= 1)
            tmin = np.minimum(tmin, np.where(hitc, tc, np.inf).min(axis=1))
        return tmin

    def _apply_boundary(self):
        w, h = self.cfg.arena.width, self.cfg.arena.height
        b = self.cfg.arena.boundary
        if b == "wrap":
            self.P[:, 0] %= w
            self.P[:, 1] %= h
        else:  # reflective (absorbing treated as reflective for v1)
            for ax, lim in ((0, w), (1, h)):
                lo = self.P[:, ax] < 0
                hi = self.P[:, ax] > lim
                self.P[lo, ax] = -self.P[lo, ax]
                self.P[hi, ax] = 2 * lim - self.P[hi, ax]
            self.P[:, 0] = np.clip(self.P[:, 0], 0, w)
            self.P[:, 1] = np.clip(self.P[:, 1], 0, h)

    def _fight_power(self, i) -> float:
        """Resource-holding potential: bigger, healthier, bolder, calmer wins."""
        return float(self.aggr[i] * np.sqrt(self.mass[i]) * (0.3 + self.health[i])
                     * (1.0 + self.bold[i]) * (1.0 - 0.3 * self.stress[i]))

    def _resolve_events(self, elapsed_s, dist, rec_j, events):
        contact = np.argwhere((dist < self.contact_r) & np.isfinite(dist))
        for i, j in contact:
            if i >= j or not (self.alive[i] and self.alive[j]):
                continue
            ai, aj = self.agents[i], self.agents[j]
            if ai.sex != aj.sex:
                # --- mating: opposite sex, female receptive ---
                fem = i if ai.sex == "F" else j
                if (rec_j[fem] > 0.3 and self.rng.random() < self.mate_rate
                        and elapsed_s - self._last_mate.get((i, j), -1e9)
                        >= self.mate_cooldown_s):
                    self._last_mate[(i, j)] = elapsed_s
                    male, female = (aj, ai) if ai.sex == "F" else (ai, aj)
                    events.record(elapsed_s, "mating", male, female,
                                  self.P[i, 0], self.P[i, 1], rec_j[fem])
                    self.matings[i] += 1
                    self.matings[j] += 1
                    self.activity[i] = self.activity[j] = 4
                    self.energy[[i, j]] = np.clip(self.energy[[i, j]] - 0.01, 0, 1)
            else:
                # --- same-sex contest -> winner/loser, damage, dominance ---
                # Contests happen during active patrol — not while nesting/huddling
                # (rest) or feeding. At least one animal must be roaming, so
                # affiliative huddles and shared foraging don't register as fights.
                if self.activity[i] != 2 and self.activity[j] != 2:
                    continue
                # one contest per dyad per cooldown; provocation scales with the
                # aggression of BOTH animals, so affiliative (low-aggression)
                # huddling does not register as fighting.
                if elapsed_s - self._last_fight.get((i, j), -1e9) \
                        < self.fight_cooldown_s:
                    continue
                # one evaluation per encounter window, so the outcome probability
                # reflects aggression rather than saturating over many ticks.
                self._last_fight[(i, j)] = elapsed_s
                prov = (self.aggr_rate * self.aggr[i] * self.aggr[j]
                        * (0.5 + 0.5 * self.bold[i]))
                if self.rng.random() >= prov:
                    continue
                fi, fj = self._fight_power(i), self._fight_power(j)
                if self.rng.random() < fi / (fi + fj + 1e-9):
                    w, l = i, j
                else:
                    w, l = j, i
                dmg = 0.05 * self.aggr[w] * (self.mass[w] / 40.0)
                self.health[l] = max(0.0, self.health[l] - dmg)
                self.energy[w] = max(0.0, self.energy[w] - 0.02)
                self.energy[l] = max(0.0, self.energy[l] - 0.04)
                self.stress[l] = min(1.0, self.stress[l] + 0.25)
                self.stress[w] = max(0.0, self.stress[w] - 0.05)
                self.fights_won[w] += 1
                self.fights_lost[l] += 1
                self.activity[l] = 3  # flee
                events.record(elapsed_s, "fight", self.agents[w], self.agents[l],
                              self.P[w, 0], self.P[w, 1], round(dmg, 4))
                # loser recoils toward its home
                away = self.home[l] - self.P[l]
                away /= (np.linalg.norm(away) + 1e-9)
                self.P[l] += away * 0.12

    def agent_static(self) -> list[dict]:
        """Static per-agent stat block (identity + innate attributes), ordered."""
        out = []
        for a in self.agents:
            t = a.traits
            geno = ";".join(f"{k}:{v}" for k, v in
                            (a.genotype.genes or {}).items()) or "WT"
            out.append({
                "index": a.index, "sexid": a.sexid, "shortid": a.shortid,
                "species": a.species, "sex": a.sex, "group": a.group,
                "genotype": geno, "drug": a.treatment.drug,
                "dose": a.treatment.dose, "onset": a.treatment.day_offset,
                "mass0": round(float(self.mass0[a.index]), 1),
                "aggression": round(t.aggression, 2),
                "boldness": round(t.boldness, 2),
                "sociability": round(t.sociability, 2),
                "exploration": round(t.exploration, 2),
                "smell_ability": round(t.smell_ability, 2),
                "identity_signal": round(t.identity_signal, 2),
                "base_speed": round(t.base_speed, 3),
                "metabolism": round(t.metabolism, 2),
            })
        return out

    # ------------------------------------------------------------------ #
    # Full trial
    # ------------------------------------------------------------------ #
    def run(self, output_dir: str, progress_cb=None, frame_cb=None,
            frame_interval_s: float = 300.0, meta_cb=None) -> dict:
        cfg = self.cfg
        os.makedirs(output_dir, exist_ok=True)
        traj_path = os.path.join(output_dir, f"uwb_{self.trial_id}_processed.csv")
        evt_path = os.path.join(output_dir, f"events_{self.trial_id}.csv")
        cond_path = os.path.join(output_dir, f"condition_{self.trial_id}.csv")
        agents_path = os.path.join(output_dir, f"agents_{self.trial_id}.csv")

        rec = TrajectoryRecorder(traj_path, self.trial_id, self.start_dt,
                                 self.agents)
        evt = EventRecorder(evt_path, self.trial_id, self.start_dt)
        cond = ConditionRecorder(cond_path, self.trial_id, self.start_dt,
                                 self.agents)
        write_agents_table(agents_path, self.agents)
        if meta_cb is not None:
            meta_cb(self.agent_static())

        total_s = cfg.days * 86400.0
        dt = cfg.dt
        n_steps = int(total_s / dt)
        rec_every = max(1, int(round(cfg.record_interval / dt)))
        cond_every = max(1, int(round(max(300.0, cfg.record_interval) / dt)))
        frame_every = max(1, int(round(frame_interval_s / dt)))
        report_every = max(1, n_steps // 100)

        elapsed = 0.0
        try:
            for k in range(n_steps):
                self.step(elapsed, dt, events=evt)
                elapsed += dt
                if k % rec_every == 0:
                    rec.record(elapsed, self.P[:, 0], self.P[:, 1])
                if k % cond_every == 0:
                    cond.record(elapsed, self.health, self.energy, self.hunger,
                                self.thirst, self.stress, self.mass,
                                self.smell < 0.5)
                if frame_cb is not None and k % frame_every == 0:
                    frame_cb(self._frame(elapsed))
                if progress_cb is not None and k % report_every == 0:
                    progress_cb(k / n_steps)
        finally:
            rec.close()
            evt.close()
            cond.close()
        if progress_cb is not None:
            progress_cb(1.0)
        return {"trajectory": traj_path, "events": evt_path,
                "condition": cond_path, "agents": agents_path,
                "trial_id": self.trial_id}

    def _frame(self, elapsed: float) -> dict:
        """Snapshot streamed to the live view / inspector (all arrays copied)."""
        return {
            "trial": self.trial_id, "elapsed": elapsed,
            "day": int(elapsed // 86400) + 1,
            "hour": self._hour(elapsed), "is_day": self._is_day(elapsed),
            "x": self.P[:, 0].copy(), "y": self.P[:, 1].copy(),
            "heading": self.H.copy(),
            "sex_m": self.sex_m, "alive": self.alive.copy(),
            "color": self.agent_rgba, "size": self.agent_size,
            "shape": self.agent_shape,
            "health": self.health.copy(), "energy": self.energy.copy(),
            "hunger": self.hunger.copy(), "thirst": self.thirst.copy(),
            "stress": self.stress.copy(), "mass": self.mass.copy(),
            "anosmic": (self.smell < 0.5).copy(),
            "estrus": (self._receptivity(elapsed) > 0.3),
            "activity": self.activity.copy(),
            "fights_won": self.fights_won.copy(),
            "fights_lost": self.fights_lost.copy(),
            "matings": self.matings.copy(),
            "dist_today": self.dist_today.copy(),
        }


def run_trial(args) -> dict:
    """Top-level helper for multiprocessing: args = (config_dict, trial_index, out_dir)."""
    config_dict, trial_index, out_dir = args
    cfg = ExperimentConfig.from_dict(config_dict)
    sim = Simulation(cfg, trial_index=trial_index)
    return sim.run(out_dir)
