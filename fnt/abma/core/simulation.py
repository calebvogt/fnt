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
from .scent import ScentField
from .physiology import get_food, DEFAULT_FOOD
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
    removed: bool = False    # trapped out by a protocol event (vs died)


class Simulation:
    # Movement-decision weights now live on the Policy (see policy.py). The
    # engine keeps the physics/physiology/combat constants below.
    contact_r = 0.12      # interaction/contact radius (m)
    # how far an animal can feed/drink from a structure's surface: body reach
    # plus a step of slack, so contact is not knife-edge at coarse dt.
    _REACH = 0.18
    estrus_period_days = 4.0
    # Mating hazard while in contact & receptive, per SECOND — converted to a
    # per-step probability via p = 1 - exp(-rate*dt) so results don't depend on
    # the integration timestep. 0.01/s matches the old 0.02-per-tick at dt=2.
    mate_rate_hz = 0.01
    #: fraction of its own marks around it that counts as "in my patch"
    at_home_scent = 0.05
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
        self.agents: list[AgentMeta] = []
        self._next_index = 0
        self._next_shortid = 9000 + self.trial_index * 100 + 1
        self._schedule: list[tuple[float, int, str, float]] = []  # onset,idx,attr,val
        self._last_fight: dict[tuple[int, int], float] = {}  # dyad -> last fight time
        self._last_mate: dict[tuple[int, int], float] = {}   # dyad -> last mating
        self._pop_dirty = False   # set when protocol events change the roster
        # Animals are released together near the arena centre (as in a real
        # enclosure release) and then self-organise; home ranges are emergent.
        for g in cfg.groups:
            self.agents.extend(self._spawn_group(g))

        n = len(self.agents)
        self.n = n
        # ---- state arrays ----
        # home-range adaptation rate: home tracks a slow average of position so
        # territories emerge over ~half a day rather than being prescribed.
        self._settle_tau_s = 0.5 * 86400.0
        for k, v in self._init_state_for(self.agents).items():
            setattr(self, k, v)
        self.P = np.clip(self.home.copy(), 0.02,
                         [cfg.arena.width - 0.02, cfg.arena.height - 0.02])
        self.mass0 = self.mass.copy()      # release mass, for drift reporting
        self._build_obstacles()
        self._cur_day = 1

        # ---- scent-mark field (territoriality emerges from this) ----
        sp = getattr(cfg, "scent", None)
        self.scent = (ScentField(cfg.arena.width, cfg.arena.height, sp)
                      if sp is not None and sp.enabled else None)
        # ---- mechanistic energy/water budget ----
        self._physio_on = bool(getattr(getattr(cfg, "physiology", None),
                                       "enabled", False))

        # ---- protocol events (timed add/remove of animals and resources) ----
        self._proto_schedule = sorted(
            [(p.at_day * 86400.0, p) for p in getattr(cfg, "protocol", [])],
            key=lambda t: t[0])

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
        # Sim-local copies so protocol events can add/remove resources mid-run
        # without mutating the config (which is shared across lockstep trials).
        self._res_objects = deepcopy(list(cfg.arena.objects))
        self._res_zones = deepcopy(list(getattr(cfg.arena, "resource_zones", [])))
        self._res_towers = deepcopy(list(getattr(cfg.arena, "water_towers", [])))
        self._rebuild_resources()

    def _rebuild_resources(self) -> None:
        """(Re)build the food/water target arrays from the sim-local lists.

        A walled resource zone is only reachable through its doorway, so the
        animal steers for the *entrance* while feeding happens anywhere
        inside. Seek target and feeding region are therefore separate: aiming
        at the centre would just press the animal against the outside wall.
        """
        food, food_seek, frects = [], [], []
        # what each food entry is made of, in the same order: circles first,
        # then zone rectangles. Drives energy yield and depletion.
        self._food_meta = []
        for o in self._res_objects:
            if o.kind == "food":
                food.append((o.x, o.y, o.radius))
                food_seek.append((o.x, o.y))
                self._food_meta.append((get_food(getattr(o, "food_type", "")), o))
        for z in self._res_zones:
            # anywhere inside the box counts as being at the chow pile — a
            # rectangle containment test (the old circumscribing circle let
            # animals "feed" through the corners from outside the walls)
            frects.append((z.x, z.y, z.w / 2.0, z.d / 2.0))
            food_seek.append(_zone_door(z))
            # a resource zone holds the standard provisioned diet, unlimited
            self._food_meta.append((get_food(DEFAULT_FOOD), None))
        water, water_seek = [], []
        for o in self._res_objects:
            if o.kind == "water":
                water.append((o.x, o.y, o.radius))
                water_seek.append((o.x, o.y))
        for t in self._res_towers:
            # a tower is also a solid: collision holds the animal at
            # radius + body, so the drinkable band must clear that and add a
            # reach margin, or there is no reachable annulus at all.
            water.append((t.x, t.y, t.radius + self._REACH))
            water_seek.append((t.x, t.y))
        self.food = np.array([[x, y] for x, y, _ in food], float).reshape(-1, 2)
        self.water = np.array([[x, y] for x, y, _ in water], float).reshape(-1, 2)
        self.food_r = np.array([r for _, _, r in food], float)
        self.water_r = np.array([r for _, _, r in water], float)
        self.food_rects = np.array(frects, float).reshape(-1, 4)  # cx,cy,hw,hd
        self.food_seek = np.array(food_seek, float).reshape(-1, 2)
        self.water_seek = np.array(water_seek, float).reshape(-1, 2)

    # ------------------------------------------------------------------ #
    # Founder construction (initial release AND protocol additions)
    # ------------------------------------------------------------------ #
    def _spawn_group(self, g: AgentGroup) -> list[AgentMeta]:
        """Construct founders for group ``g``, released near the arena centre."""
        cfg = self.cfg
        sd = cfg.individual_variation
        release = np.array([cfg.arena.width / 2, cfg.arena.height / 2])
        delayed = (g.treatment.drug not in ("none", None)
                   and g.treatment.day_offset > 0.0)
        specs = {k: parse_spec(v) for k, v in (g.dists or {}).items()}
        metas = []
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
            idx = self._next_index
            shortid = self._next_shortid
            self._next_index += 1
            self._next_shortid += 1
            start = release + self.rng.normal(0, 0.15, 2)
            metas.append(AgentMeta(
                index=idx, sexid=f"{g.sex}{shortid}", shortid=shortid,
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
        return metas

    def _init_state_for(self, metas: list[AgentMeta]) -> dict:
        """Fresh per-agent state arrays for ``metas`` (attr name -> array)."""
        m = len(metas)
        rng = self.rng
        return {
            "home": np.array([a.home for a in metas], float).reshape(-1, 2),
            "H": rng.uniform(0, 2 * np.pi, m),
            # ---- condition: every bar is 0-100, here and in the CSVs ----
            "hunger": rng.uniform(0, 30, m),
            "thirst": rng.uniform(0, 30, m),
            "energy": np.full(m, 100.0),
            "health": np.full(m, 100.0),
            "stress": np.full(m, 10.0),
            # bladder fills by drinking and is spent on scent marks
            "bladder": np.full(m, 50.0),
            "alive": np.ones(m, bool),
            "estrus_phase": rng.uniform(0, 1, m),
            # ---- behaviour counters / bookkeeping ----
            "fights_won": np.zeros(m, int),
            "fights_lost": np.zeros(m, int),
            "matings": np.zeros(m, int),
            "dist_today": np.zeros(m),
            "dist_total": np.zeros(m),
            # 0 rest, 1 forage, 2 roam, 3 flee, 4 mate, 5 dead
            "activity": np.zeros(m, int),
            # ---- trait vectors ----
            "sex_m": np.array([1.0 if a.sex == "M" else 0.0 for a in metas]),
            "aggr": np.array([a.traits.aggression for a in metas]),
            "bold": np.array([a.traits.boldness for a in metas]),
            "social": np.array([a.traits.sociability for a in metas]),
            "explore": np.array([a.traits.exploration for a in metas]),
            "smell": np.array([a.traits.smell_ability for a in metas]),
            "identity": np.array([a.traits.identity_signal for a in metas]),
            # how often it *wants* to mark; whether it *can* is the bladder
            "scent_rate": np.array([getattr(a.traits, "scent_rate", 20.0)
                                    for a in metas]),
            "marks_made": np.zeros(m, int),
            "food_eaten_g": np.zeros(m),
            "water_drunk_ml": np.zeros(m),
            "speed": np.array([a.traits.base_speed for a in metas]),
            "body_len": np.array([getattr(a.traits, "body_length_cm", 9.0)
                                  for a in metas]),
            "home_r": np.array([a.traits.home_range_r for a in metas]),
            "mass": np.array([a.traits.mass for a in metas]),
            "metabolism": np.array([a.traits.metabolism for a in metas]),
            "turn_rate": np.array([a.traits.turn_rate for a in metas]),
            "wander": np.array([a.traits.wander for a in metas]),
            # ---- appearance (per-agent colour / size / shape for the views) --
            "agent_rgba": np.array(
                [_appearance_rgba(a.appearance, a.sex)
                 for a in metas]).reshape(-1, 4),
            "agent_size": np.array(
                [getattr(a.appearance, "size", 1.0) or 1.0 for a in metas]),
            "agent_shape": np.array(
                [_SHAPE_CODE.get(getattr(a.appearance, "shape", "rodent"), 0)
                 for a in metas], int),
        }

    def _append_state(self, metas: list[AgentMeta]) -> None:
        """Grow every per-agent array for newly introduced founders."""
        if not metas:
            return
        st = self._init_state_for(metas)
        for k, v in st.items():
            cur = getattr(self, k)
            setattr(self, k, np.concatenate([cur, v]) if v.ndim == 1
                    else np.vstack([cur, v]))
        cfg = self.cfg
        newP = np.clip(st["home"].copy(), 0.02,
                       [cfg.arena.width - 0.02, cfg.arena.height - 0.02])
        self.P = np.vstack([self.P, newP])
        self.mass0 = np.concatenate([self.mass0, st["mass"].copy()])
        self.agents.extend(metas)
        self.n = len(self.agents)

    # ------------------------------------------------------------------ #
    # Protocol events — timed add/remove of animals and resources
    # ------------------------------------------------------------------ #
    def _apply_protocol(self, elapsed_s: float, events) -> None:
        due = [p for t, p in self._proto_schedule if elapsed_s >= t]
        self._proto_schedule = [(t, p) for t, p in self._proto_schedule
                                if elapsed_s < t]
        for p in due:
            if p.kind == "add_agents" and p.group is not None:
                self._add_agents(p.group, elapsed_s, events)
            elif p.kind == "remove_agents":
                self._remove_agents(p.target or "all", p.count,
                                    elapsed_s, events)
            elif p.kind == "add_resource" and p.object is not None:
                self._res_objects.append(deepcopy(p.object))
                self._rebuild_resources()
            elif p.kind == "remove_resource":
                t = (p.target or "").strip()
                keep = [o for o in self._res_objects
                        if not (o.label == t or o.kind == t)]
                if len(keep) != len(self._res_objects):
                    self._res_objects = keep
                    self._rebuild_resources()

    def _add_agents(self, group: AgentGroup, elapsed_s: float, events) -> None:
        metas = self._spawn_group(group)
        self._append_state(metas)
        self._pop_dirty = True
        if events is not None:
            for a in metas:
                events.record(elapsed_s, "release", a, None,
                              a.home[0], a.home[1])

    def _remove_agents(self, target: str, count: int, elapsed_s: float,
                       events) -> None:
        """Trap out living agents matching ``target`` (count 0 = all matching).

        Removed animals keep their identity and history, but stop moving,
        interacting and appearing in the trajectory — like a real removal.
        """
        idxs = [a.index for a in self.agents
                if self.alive[a.index] and self._match_target(a, target)]
        if count > 0:
            idxs = idxs[:count]
        for i in idxs:
            self.alive[i] = False
            self.agents[i].alive = False
            self.agents[i].removed = True
            self.activity[i] = 5
            if events is not None:
                events.record(elapsed_s, "removal", self.agents[i], None,
                              self.P[i, 0], self.P[i, 1])
        if idxs:
            self._pop_dirty = True

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

        # --- apply any protocol events whose time has arrived ---
        if self._proto_schedule and elapsed_s >= self._proto_schedule[0][0]:
            self._apply_protocol(elapsed_s, events)
            n = self.n                      # roster may have grown
            P = self.P

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
                    if events is not None:
                        for i in idxs:   # one event row per affected agent
                            events.record(elapsed_s, "intervention",
                                          self.agents[i], None,
                                          self.P[i, 0], self.P[i, 1], val)
                else:
                    still.append((at_s, idxs, arr, op, val))
            self._iv_schedule = still

        # --- decision: the policy turns state into a desired heading ---
        desired, perc = self.policy.decide(self, elapsed_s, dt)
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
        spd = self.speed * activity * (1.0 - f + f * self.energy / 100.0)
        # "settled" = fed, watered, and standing on familiar ground. With scent
        # marking that means inside its own marked patch (emergent); without
        # it, inside the prescribed home-range radius (legacy).
        own_lvl = perc.get("scent_own")
        at_home = (own_lvl > self.at_home_scent if own_lvl is not None
                   else dist_home < self.home_r)
        satiated = (self.hunger < 50.0) & (self.thirst < 50.0) & at_home
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

        # --- scent marking (marks fade, then animals lay new ones) ---
        if self.scent is not None:
            self._update_scent(dt, perc.get("scent_foreign"))

        # --- condition dynamics ---
        on_food = self._on_food()
        on_water = (self._on_resource(self.P, self.water, self.water_r)
                    if self.water.shape[0] else np.zeros(n, bool))
        if self._physio_on:
            # a real energy/water budget owns energy, hunger, thirst and the
            # bladder; the rules table is left to stress and health
            self._apply_physiology(dt, step_dist, on_water)
            self._apply_dynamics(dt, step_dist, activity, within.sum(axis=1),
                                 on_food, on_water,
                                 only_targets=("stress", "health"))
        else:
            self._apply_dynamics(dt, step_dist, activity, within.sum(axis=1),
                                 on_food, on_water)

        # --- body mass drifts with energy balance, within a physiological band ---
        self.mass = np.clip(
            self.mass + dt * 1.5e-5 * (self.energy / 100.0 - 0.7),
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
            self._resolve_events(elapsed_s, dt, dist, rec_j, events)

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

    def _on_food(self):
        """At a chow pile: inside a food object's radius or a resource zone's box."""
        return self._food_index() >= 0

    def _food_index(self):
        """Which food entry each agent is feeding at (-1 = none).

        Indexes ``self._food_meta``: point sources first, then resource zones,
        matching the order ``_rebuild_resources`` builds them in.
        """
        n = self.n
        idx = np.full(n, -1, int)
        n_circ = self.food.shape[0]
        if n_circ:
            d = np.linalg.norm(self.P[:, None, :] - self.food[None, :, :],
                               axis=2)
            inside = d < self.food_r[None, :]
            near = np.where(inside, d, np.inf)
            best = np.argmin(near, axis=1)
            got = np.isfinite(near[np.arange(n), best])
            idx[got] = best[got]
        if len(self.food_rects):
            cx, cy, hw, hd = self.food_rects.T
            inx = np.abs(self.P[:, 0][:, None] - cx[None, :]) <= hw[None, :]
            iny = np.abs(self.P[:, 1][:, None] - cy[None, :]) <= hd[None, :]
            inz = inx & iny
            any_z = inz.any(axis=1)
            first = np.argmax(inz, axis=1) + n_circ
            take = any_z & (idx < 0)          # a point source wins if both
            idx[take] = first[take]
        return idx

    def _apply_dynamics(self, dt, step_dist, activity, n_near, on_food, on_water,
                        only_targets=None):
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
                v = 100.0 - snap["hunger"]
            elif name == "hydrated":
                v = 100.0 - snap["thirst"]
            elif name == "rested":
                v = 100.0 - snap["stress"]
            elif name in snap:
                v = snap[name]
            else:
                v = np.zeros(n)
            cache[name] = v
            return v

        for c in self.dynamics:
            if only_targets is not None and c.target not in only_targets:
                continue
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
            np.clip(arr, 0.0, 100.0, out=arr)

    # ------------------------------------------------------------------ #
    # Mechanistic energy & water budget
    # ------------------------------------------------------------------ #
    def _apply_physiology(self, dt: float, step_dist, on_water) -> None:
        """Move real kilojoules and millilitres, then read the bars off them.

        Energy in comes only from eating (grams × the diet's energy density);
        energy out is basal metabolism + locomotion (charged in
        ``_resolve_events`` for fights and matings). Water in comes from
        drinking and from moisture in food; a share of it is routed to the
        bladder, which is what pays for scent marks. Hunger and thirst are not
        integrated — they are what the stores are missing.
        """
        p = self.cfg.physiology
        alive = self.alive
        mass = self.mass
        e_cap = np.maximum(1e-6, p.energy_capacity(mass))     # kJ
        w_cap = np.maximum(1e-6, p.water_capacity(mass))      # mL
        b_cap = np.maximum(1e-6, p.bladder_capacity(mass))    # mL

        # ---- eating ---------------------------------------------------- #
        fidx = self._food_index()
        eating = (fidx >= 0) & alive
        kj_in = np.zeros(self.n)
        ml_food = np.zeros(self.n)
        if eating.any() and self._food_meta:
            dens = np.array([m[0].energy_density for m in self._food_meta])
            wfrac = np.array([m[0].water_fraction for m in self._food_meta])
            pal = np.array([m[0].palatability for m in self._food_meta])
            who = fidx[eating]
            grams = p.feed_rate_g_min * (dt / 60.0) * pal[who]
            # an animal stops when it is full: cap intake at the room left in
            # the energy store, so a satiated animal does not strip a pile
            room_kj = np.maximum(0.0, (100.0 - self.energy[eating]) / 100.0
                                 * e_cap[eating])
            grams = np.minimum(grams, room_kj / np.maximum(1e-9, dens[who]))
            # a finite pile can run out; share what is left among the feeders
            remaining = np.array(
                [(m[1].amount_g if (m[1] is not None and m[1].amount_g > 0)
                  else np.inf) for m in self._food_meta])
            if np.isfinite(remaining).any():
                want = np.zeros(len(self._food_meta))
                np.add.at(want, who, grams)
                scale = np.ones(len(self._food_meta))
                lim = np.isfinite(remaining) & (want > 0)
                scale[lim] = np.minimum(1.0, remaining[lim] / want[lim])
                grams = grams * scale[who]
                taken = np.zeros(len(self._food_meta))
                np.add.at(taken, who, grams)
                for k, (_, obj) in enumerate(self._food_meta):
                    if obj is not None and obj.amount_g > 0:
                        obj.amount_g = max(0.0, obj.amount_g - taken[k])
            kj_in[eating] = grams * dens[who]
            ml_food[eating] = grams * wfrac[who]
            self.food_eaten_g[eating] += grams

        # ---- energy out ------------------------------------------------- #
        kj_out = p.basal_kj(mass, self.metabolism, dt)
        kj_out = kj_out + p.locomotion_kj(mass, step_dist)
        kj_out[~alive] = 0.0

        self.energy = np.clip(
            self.energy + (kj_in - kj_out) / e_cap * 100.0, 0.0, 100.0)
        # hunger IS the energy deficit — nothing writes to it directly
        self.hunger = 100.0 - self.energy

        # ---- water ------------------------------------------------------ #
        drinking = on_water & alive
        ml_in = np.where(drinking, p.drink_rate_ml_min * (dt / 60.0), 0.0)
        ml_in = ml_in + ml_food
        self.water_drunk_ml += ml_in
        ml_out = np.where(alive, p.water_loss_ml_h * (dt / 3600.0), 0.0)
        self.thirst = np.clip(
            self.thirst + (ml_out - ml_in) / w_cap * 100.0, 0.0, 100.0)

        # ---- bladder: what marking is actually paid for with ------------ #
        self.bladder = np.clip(
            self.bladder + (ml_in * p.urine_fraction) / b_cap * 100.0,
            0.0, 100.0)

        # ---- an empty store costs health -------------------------------- #
        dth = dt / 3600.0
        starving = alive & (self.energy <= 0.0)
        parched = alive & (self.thirst >= 100.0)
        if starving.any():
            self.health[starving] -= p.starvation_health_h * dth
        if parched.any():
            self.health[parched] -= p.dehydration_health_h * dth
        np.clip(self.health, 0.0, 100.0, out=self.health)

    def _spend_energy_kj(self, idxs, kj: float) -> None:
        """Charge an action (a fight, a mating) against the energy store."""
        p = self.cfg.physiology
        if not getattr(p, "enabled", False):
            return
        for i in idxs:
            cap = max(1e-6, p.energy_capacity(self.mass[i]))
            self.energy[i] = max(0.0, self.energy[i] - kj / cap * 100.0)
            self.hunger[i] = 100.0 - self.energy[i]

    def _update_scent(self, dt: float, foreign_level=None) -> None:
        """Fade existing marks, refill bladders, and lay new marks.

        Marking is deliberately a *limited resource*: the reserve refills at the
        animal's ``scent_rate`` (marks/hour) and each mark spends
        ``deposit_cost``, so an animal cannot saturate an arena. Animals also
        counter-mark — marking rate is multiplied where foreign scent is
        detected, which is what turns a chance encounter into a contested
        boundary rather than a uniform smear.
        """
        sp = self.cfg.scent
        self.scent.decay(dt)

        p = self.cfg.physiology
        if self._physio_on:
            # marking is paid for out of the bladder, which is filled by
            # drinking — so water availability is a territorial constraint
            b_cap_ml = np.maximum(1e-6, p.bladder_capacity(self.mass))
            cost = (p.mark_volume_ul / 1000.0) / b_cap_ml * 100.0
            reserve = self.bladder
        else:
            # legacy: an abstract reserve refilling at the animal's scent_rate
            self.bladder = np.clip(
                self.bladder
                + self.scent_rate * sp.deposit_cost * (dt / 36.0), 0.0, 100.0)
            cost = np.full(self.n, sp.deposit_cost * 100.0)
            reserve = self.bladder

        rate_h = float(self.cfg.policy.mark_rate_h)
        boost = np.ones(self.n)
        if foreign_level is not None and len(foreign_level) == self.n:
            boost = 1.0 + (sp.counter_mark - 1.0) * np.clip(foreign_level, 0, 1)
        # dt-invariant hazard, as elsewhere in the engine
        p_mark = 1.0 - np.exp(-(rate_h / 3600.0) * boost * dt)
        can = self.alive & (reserve >= cost)
        fire = can & (self.rng.random(self.n) < p_mark)
        idxs = np.nonzero(fire)[0]
        if len(idxs) == 0:
            return
        self.scent.deposit(idxs, self.P, sp.mark_strength,
                           self.identity[idxs])
        self.bladder[idxs] = np.maximum(0.0, self.bladder[idxs] - cost[idxs])
        self.marks_made[idxs] += 1

    def territory_area(self) -> np.ndarray:
        """Emergent territory area per agent (m²), or zeros without marking."""
        if self.scent is None:
            return np.zeros(self.n)
        return self.scent.area_by_owner(self.n)

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
        return float(self.aggr[i] * np.sqrt(self.mass[i])
                     * (0.3 + self.health[i] / 100.0)
                     * (1.0 + self.bold[i])
                     * (1.0 - 0.3 * self.stress[i] / 100.0))

    def _resolve_events(self, elapsed_s, dt, dist, rec_j, events):
        # per-step mating probability from the per-second hazard (dt-invariant)
        p_mate = 1.0 - np.exp(-self.mate_rate_hz * dt)
        contact = np.argwhere((dist < self.contact_r) & np.isfinite(dist))
        for i, j in contact:
            if i >= j or not (self.alive[i] and self.alive[j]):
                continue
            ai, aj = self.agents[i], self.agents[j]
            if ai.sex != aj.sex:
                # --- mating: opposite sex, female receptive ---
                fem = i if ai.sex == "F" else j
                if (rec_j[fem] > 0.3 and self.rng.random() < p_mate
                        and elapsed_s - self._last_mate.get((i, j), -1e9)
                        >= self.mate_cooldown_s):
                    self._last_mate[(i, j)] = elapsed_s
                    male, female = (aj, ai) if ai.sex == "F" else (ai, aj)
                    events.record(elapsed_s, "mating", male, female,
                                  self.P[i, 0], self.P[i, 1], rec_j[fem])
                    self.matings[i] += 1
                    self.matings[j] += 1
                    self.activity[i] = self.activity[j] = 4
                    if self._physio_on:
                        self._spend_energy_kj((i, j), self.cfg.physiology.mate_kj)
                    else:
                        self.energy[[i, j]] = np.clip(
                            self.energy[[i, j]] - 1.0, 0, 100)
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
                # aggression IS the attack probability on a same-sex encounter
                # (0 = never attacks, 1 = always), modulated by willingness to
                # escalate. Keeping this literal is what makes the personality
                # dial mean something an experimenter can reason about.
                p_attack = self.aggr[i] * (0.5 + 0.5 * self.bold[i])
                if self.rng.random() >= p_attack:
                    continue
                fi, fj = self._fight_power(i), self._fight_power(j)
                if self.rng.random() < fi / (fi + fj + 1e-9):
                    w, l = i, j
                else:
                    w, l = j, i
                dmg = 5.0 * self.aggr[w] * (self.mass[w] / 40.0)
                self.health[l] = max(0.0, self.health[l] - dmg)
                if self._physio_on:
                    # a contest is metabolically expensive; the loser more so
                    fkj = self.cfg.physiology.fight_kj
                    self._spend_energy_kj((w,), fkj)
                    self._spend_energy_kj((l,), fkj * 2.0)
                else:
                    self.energy[w] = max(0.0, self.energy[w] - 2.0)
                    self.energy[l] = max(0.0, self.energy[l] - 4.0)
                self.stress[l] = min(100.0, self.stress[l] + 25.0)
                self.stress[w] = max(0.0, self.stress[w] - 5.0)
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
                if self._pop_dirty:      # protocol changed the roster
                    write_agents_table(agents_path, self.agents)
                    if meta_cb is not None:
                        meta_cb(self.agent_static())
                    self._pop_dirty = False
                if k % rec_every == 0:
                    rec.record(elapsed, self.P[:, 0], self.P[:, 1])
                if k % cond_every == 0:
                    cond.record(elapsed, self.health, self.energy, self.hunger,
                                self.thirst, self.stress, self.mass,
                                self.smell < 0.5, self.bladder)
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
            "bladder": self.bladder.copy(),
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
