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

import math
import os
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from copy import deepcopy

from .config import ExperimentConfig, AgentGroup, zone_doors, zone_sides
from .scent import ScentField
from .sward import SwardField
from .sky import sky_state, growth_factor
from .physiology import get_food, DEFAULT_FOOD
from .biology import (
    resolve_traits, apply_drug, TRAIT_TO_ARRAY, _TRAIT_RANGES,
)
from .sampling import parse_spec
from .policy import RuleBasedPolicy
from .neural import make_policy, discomfort
from .olfaction import OlfactorySystem
from .rng import (
    AgentRandom, CH_HEADING, CH_MARK, CH_FIGHT, CH_FIGHT_OUTCOME, CH_MATE,
)
from .recorder import (
    TrajectoryRecorder, EventRecorder, ConditionRecorder, write_agents_table,
    parse_start,
)
from .record import RunRecord


_ARRAY_TO_TRAIT = {v: k for k, v in TRAIT_TO_ARRAY.items()}
_SHAPE_CODE = {"rodent": 0, "blob": 1, "bird": 2}
_MALE_RGBA = (0.29, 0.56, 0.85, 1.0)
_FEMALE_RGBA = (0.88, 0.33, 0.60, 1.0)


def _zone_door(z):
    """Outside-facing centre of a resource zone's (first) doorway."""
    return zone_doors(z)[0]


def _run_wall_past_boundary(seg, w, h, far=1.0e3, tol=1e-6):
    """Carry a wall that ends at the room's boundary on beyond it.

    A step can overshoot a reflective boundary before it is bounced back in.
    A wall that stopped exactly at the boundary could therefore be walked
    around on the outside: out past the end of the wall, and back in on the
    other side of it. That let animals into (and out of) any box set against
    the room's wall without using a doorway. Only an end that meets the
    boundary head-on is extended; a wall lying *along* the boundary is left
    alone, so it does not turn the room's own wall into something to slide on.
    """
    x1, y1, x2, y2 = (float(v) for v in seg)
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0.0:
        return (x1, y1, x2, y2)
    dx, dy = (x2 - x1) / length, (y2 - y1) / length

    def end(x, y, sx, sy):              # (sx, sy) points away from the wall
        head_on = ((x <= tol and sx < -0.5) or (x >= w - tol and sx > 0.5)
                   or (y <= tol and sy < -0.5) or (y >= h - tol and sy > 0.5))
        return (x + sx * far, y + sy * far) if head_on else (x, y)

    ax, ay = end(x1, y1, -dx, -dy)
    bx, by = end(x2, y2, dx, dy)
    return (ax, ay, bx, by)


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
    #: Stable identity for the agent's random stream. ``index`` is a row
    #: position and would shift if the roster were ever compacted; ``uid`` is
    #: assigned once at spawn and never reused, so an animal's noise, odour
    #: signature and receptor profile belong to the *animal* rather than to
    #: where it happens to sit in the state arrays. See :mod:`.rng`.
    uid: int
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
        root = config.seed + trial_index if seed is None else seed
        self._root_seed = int(root)
        #: urine spots: where each void landed, whose it was and how much.
        #: Finer than the scent grid, so a nose can be on one or beside it.
        self.urine_spots = {"xy": np.zeros((0, 2)), "owner": np.zeros(0, int),
                            "vol": np.zeros(0), "t": np.zeros(0)}
        #: voids of the current step, as (agent, x, y, volume): for a viewer
        self.last_voids: list = []
        self._voiding = np.zeros(0, bool)
        self._mup = None
        #: Setup-time draws (founder traits, release scatter, initial condition).
        #: Sequential, and only consumed when the roster is built.
        self.rng = np.random.default_rng(root)
        #: Per-step draws. Counter-based and keyed by stable agent uid, so an
        #: ablated or added animal perturbs nobody else's trajectory — which is
        #: what makes a paired study a paired study. See :mod:`.rng`.
        self.arand = AgentRandom(root)
        self._step_k = 0
        self.start_dt = parse_start(config.start_datetime)
        # A learned brain when the config asks for one (`brain.enabled`),
        # otherwise the hand-written rules. An explicit `policy` wins.
        self.policy = policy if policy is not None else make_policy(config)
        self._build_population()

    # ------------------------------------------------------------------ #
    # Population construction
    # ------------------------------------------------------------------ #
    def _build_population(self) -> None:
        cfg = self.cfg
        self.agents: list[AgentMeta] = []
        self._next_index = 0
        self._next_uid = 0
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
        self._build_cover()
        self._cur_day = 1

        # ---- scent-mark field (territoriality emerges from this) ----
        sp = getattr(cfg, "scent", None)
        self.scent = (ScentField(cfg.arena.width, cfg.arena.height, sp)
                      if sp is not None and sp.enabled else None)
        # ---- mechanistic energy/water budget ----
        self._physio_on = bool(getattr(getattr(cfg, "physiology", None),
                                       "enabled", False))

        # ---- living grass layer (trails emerge from movement) ----
        sw = getattr(cfg, "sward", None)
        self.sward = (SwardField(cfg.arena.width, cfg.arena.height, sw,
                                 rng=self.rng)
                      if sw is not None and sw.enabled else None)
        #: grass height (cm) under each animal, refreshed every step
        self.grass_cm = np.zeros(n)
        #: seconds left in the current clipping bout; >0 means "chewing"
        self.chew_left = np.zeros(n)
        self.chew_seconds = np.zeros(n)     # cumulative, per animal
        self.grass_cut_cm = np.zeros(n)     # cumulative cm clipped

        # ---- sky: a real sun and moon for a real place ----
        sk = getattr(cfg, "sky", None)
        self.sky_p = sk if sk is not None and sk.enabled else None
        self._sky = None
        self._sky_t = -1e18

        # ---- rest, warmth and attachment (off unless SocialParams is on) ----
        # Built lazily at the end of construction, once the roster's arrays
        # exist (see ``_build_population``'s last lines).
        self.soc = None

        # ---- olfactory system (receptors + odour signatures) ----
        # Off by default, in which case recognition stays the scalar product
        # `smell_ability x identity_signal` and old configs reproduce exactly.
        op = getattr(cfg, "olfaction", None)
        self.olf = (OlfactorySystem(op, self.arand, self.uid)
                    if op is not None and op.enabled else None)
        self._recog = None
        self._olf_dirty = True

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
        sc = getattr(cfg, "social", None)
        if sc is not None and sc.enabled:
            from .social import SocialState
            self.soc = SocialState(self)
        #: each animal's speed last step as a fraction of its easy pace
        self.effort = np.zeros(n)

    def _rebuild_resources(self) -> None:
        """(Re)build the food/water target arrays from the sim-local lists.

        A walled resource zone is only reachable through its doorway, so the
        animal steers for the *entrance* while feeding happens anywhere
        inside. Seek target and feeding region are therefore separate: aiming
        at the centre would just press the animal against the outside wall.
        """
        food, food_seek, frects, fdoors, door_sets = [], [], [], [], []
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
            doors = zone_doors(z)
            food_seek.extend(doors)            # any doorway is a way in
            fdoors.append(doors[0])
            door_sets.append(np.array(doors, float))
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
        # A zone that holds a water tower: drinking, like feeding, happens
        # anywhere inside the box, and from outside the way to it is a
        # doorway. (A tower object placed inside the walls would be drinkable
        # through them — its reach is wider than the gap to the wall.)
        wrects = []
        for z in self._res_zones:
            if getattr(z, "water", False):
                wrects.append((z.x, z.y, z.w / 2.0, z.d / 2.0))
                water_seek.extend(zone_doors(z))
        self.water_rects = np.array(wrects, float).reshape(-1, 4)
        self.zone_has_water = np.array(
            [bool(getattr(z, "water", False)) for z in self._res_zones], bool)
        self._zone_door_sets = door_sets
        self.food = np.array([[x, y] for x, y, _ in food], float).reshape(-1, 2)
        self.water = np.array([[x, y] for x, y, _ in water], float).reshape(-1, 2)
        self.food_r = np.array([r for _, _, r in food], float)
        self.water_r = np.array([r for _, _, r in water], float)
        self.food_rects = np.array(frects, float).reshape(-1, 4)  # cx,cy,hw,hd
        # the one gap in each zone's wall — an animal inside can only leave
        # through it, so it has to be steered for explicitly
        # (the first one, where a zone has several — see ``exit_door``)
        self.zone_doors = np.array(fdoors, float).reshape(-1, 2)
        self.food_seek = np.array(food_seek, float).reshape(-1, 2)
        self.water_seek = np.array(water_seek, float).reshape(-1, 2)
        # Grams left at each food entry, inf = never runs out. This is the
        # authority at run time: `amount_g == 0` means "unlimited" in a config
        # but "empty" once eaten, so the two cannot share one number.
        self._food_left = np.array(
            [(obj.amount_g if (obj is not None and obj.amount_g > 0)
              else np.inf) for _, obj in self._food_meta], float)

    # ------------------------------------------------------------------ #
    # Founder construction (initial release AND protocol additions)
    # ------------------------------------------------------------------ #
    def _spawn_group(self, g: AgentGroup) -> list[AgentMeta]:
        """Construct founders for group ``g``, released near the arena centre."""
        cfg = self.cfg
        sd = cfg.individual_variation
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
            uid = self._next_uid
            shortid = self._next_shortid
            self._next_index += 1
            self._next_uid += 1
            self._next_shortid += 1
            start = self._release_point()
            metas.append(AgentMeta(
                index=idx, uid=uid, sexid=f"{g.sex}{shortid}", shortid=shortid,
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

    #: Arenas narrower than this keep the historical point release under
    #: 'auto'. A cage is small enough that where you put an animal down does
    #: not decide the experiment; an enclosure is not.
    _POINT_RELEASE_MAX_M = 3.0

    def _release_point(self) -> np.ndarray:
        """Where one founder is put down at t=0.

        A release is part of the design, not an implementation detail. Twelve
        animals set down inside one body-length of each other in a 523 m²
        enclosure do not disperse — social attraction holds the clump together
        and the run measures a scrum rather than a population. See
        ``ExperimentConfig.release_mode``.
        """
        cfg = self.cfg
        w, h = cfg.arena.width, cfg.arena.height
        mode = (cfg.release_mode or "auto").lower()
        margin = 0.05 * min(w, h)
        if mode == "auto":
            mode = ("point" if min(w, h) <= self._POINT_RELEASE_MAX_M
                    else "scatter")

        if mode == "nests":
            sites = [(o.x, o.y) for o in cfg.arena.objects if o.kind == "nest"]
            sites += [(z.x, z.y)
                      for z in getattr(cfg.arena, "resource_zones", [])]
            if sites:
                x, y = sites[self._next_uid % len(sites)]
                spread = cfg.release_scatter_m or 0.25
                return np.array([x, y]) + self.rng.normal(0, spread, 2)
            mode = "scatter"          # nowhere to nest; fall back

        if mode == "scatter":
            lo, hi = margin, np.array([w, h]) - margin
            return self.rng.uniform(lo, hi, 2)

        centre = np.array([w / 2, h / 2])
        return centre + self.rng.normal(0, cfg.release_scatter_m or 0.15, 2)

    def _init_state_for(self, metas: list[AgentMeta]) -> dict:
        """Fresh per-agent state arrays for ``metas`` (attr name -> array)."""
        m = len(metas)
        rng = self.rng
        return {
            "home": np.array([a.home for a in metas], float).reshape(-1, 2),
            # stable random-stream identity, parallel to the state arrays
            "uid": np.array([a.uid for a in metas], np.int64),
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
            # ---- contest physics (these only move under ContestParams) ----
            "fatigue": np.zeros(m),             # 0-100; sprinting fills it
            "hurt": np.zeros(m),                # the sting of a bite, 0-1
            "hurt_dir": np.zeros((m, 2)),       # where the last bite came from
            "bite_wait": np.zeros(m),           # seconds until it can bite again
            "bites_given": np.zeros(m, int),
            "bites_taken": np.zeros(m, int),
            "V": np.zeros((m, 2)),              # last step's velocity (m/s)
            # ---- trait vectors ----
            "sex_m": np.array([1.0 if a.sex == "M" else 0.0 for a in metas]),
            "aggr": np.array([a.traits.aggression for a in metas]),
            "bold": np.array([a.traits.boldness for a in metas]),
            "social": np.array([a.traits.sociability for a in metas]),
            "explore": np.array([a.traits.exploration for a in metas]),
            "smell": np.array([a.traits.smell_ability for a in metas]),
            "identity": np.array([a.traits.identity_signal for a in metas]),
            # the vomeronasal organ (contact chemosignals), lost separately
            "vno": np.array([getattr(a.traits, "vno_ability", 1.0)
                             for a in metas]),
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
        self.mark_biology_dirty()   # the cohort changed: recognition restales

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

    def when(self, elapsed_s: float):
        """Wall-clock timestamp at ``elapsed_s`` into the run."""
        return self.start_dt + timedelta(seconds=float(elapsed_s))

    #: The sun moves ~0.25°/min, so recomputing it more often than this buys
    #: nothing at any timestep an experiment runs at.
    _SKY_REFRESH_S = 60.0

    def sky_at(self, elapsed_s: float):
        """Cached sky for this instant, or ``None`` when the sky is off."""
        if self.sky_p is None:
            return None
        if (self._sky is None
                or abs(elapsed_s - self._sky_t) >= self._SKY_REFRESH_S):
            self._sky = sky_state(self.when(elapsed_s), self.sky_p)
            self._sky_t = elapsed_s
        return self._sky

    def _is_day(self, elapsed_s: float) -> bool:
        """Daytime — from the real sun outdoors, from the clock indoors."""
        sky = self.sky_at(elapsed_s)
        if sky is not None:
            return sky.is_day
        # day window is [day_start, day_start + day length); night otherwise
        return ((self._hour(elapsed_s) - self.cfg.day_start_hour) % 24
                < float(getattr(self.cfg, "day_length_hours", 12.0)))

    def _activity(self, elapsed_s: float) -> float:
        """How hard the cohort is working right now.

        With a sky, day and night are interpolated through twilight rather
        than switched, so a crepuscular animal gets a dawn and a dusk instead
        of a step change — and a bright moon suppresses activity, which is one
        of the better-described behaviours in nocturnal small mammals.
        """
        cfg = self.cfg
        sky = self.sky_at(elapsed_s)
        if sky is None:
            return cfg.day_activity if self._is_day(elapsed_s) \
                else cfg.night_activity
        lit = sky.daylight
        base = cfg.night_activity + (cfg.day_activity - cfg.night_activity) * lit
        if sky.night_light > 0.0:
            base *= (1.0 - self.sky_p.moonlight_suppression * sky.night_light)
        return float(base)

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
        # counter for the per-agent random streams; every stochastic draw this
        # step is a pure function of (seed, agent uid, this counter, channel)
        self._step_k += 1

        # --- apply any treatments whose onset time has arrived ---
        if self._schedule:
            still = []
            for onset_s, i, attr, val in self._schedule:
                if elapsed_s >= onset_s:
                    getattr(self, attr)[i] = val
                    setattr(self.agents[i].traits, _ARRAY_TO_TRAIT[attr], val)
                    if attr in ("smell", "identity"):
                        self.mark_biology_dirty()
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
                    if arr in ("smell", "identity"):
                        self.mark_biology_dirty()
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
        # Stashed for the frame emitter: the named drives whose sum produced
        # this step's heading, plus the perception they were computed from.
        # Summing them and discarding the parts would throw away the only
        # record of *why* the animal moved where it did (see policy.py).
        self._last_perc = perc
        self._last_desired = desired

        # --- resolve to movement ---
        dmag = np.linalg.norm(desired, axis=1) + 1e-9
        move_dir = desired / dmag[:, None]
        activity = self._activity(elapsed_s)
        # energy -> speed coupling (low energy is sluggish); config-driven
        f = cfg.energy_speed_coupling
        spd = self.speed * activity * (1.0 - f + f * self.energy / 100.0)

        # --- the sward underfoot: how fast, and how expensive, this is ---
        if self.sward is not None:
            self.grass_cm = self.sward.sample(P)
            # deep grass drags; a worn trail is faster than open ground ever
            # was, which is the payoff that makes clipping one worth the time
            spd = spd * self.sward.speed_factor(self.grass_cm)

        # --- clipping bouts: an animal that is cutting a trail stands still --
        chewing = self._update_chewing(perc.get("want_chew"), dt)
        if chewing.any():
            spd = np.where(chewing, 0.0, spd)
        # "settled" = fed, watered, and standing on familiar ground. With scent
        # marking that means inside its own marked patch (emergent); without
        # it, inside the prescribed home-range radius (legacy).
        own_lvl = perc.get("scent_own")
        at_home = (own_lvl > self.at_home_scent if own_lvl is not None
                   else dist_home < self.home_r)
        satiated = (self.hunger < 50.0) & (self.thirst < 50.0) & at_home
        ct = cfg.contest
        cruise = spd                       # this body's easy pace right now
        speed_frac = perc.get("speed_frac")
        if speed_frac is not None:
            # A learned brain emits its own speed, so slowing to a rest is
            # something the animal decides rather than a rule applied to it.
            # `spd` is still the ceiling: what this body can do right now.
            if ct.enabled:
                # ...and with contests on the ceiling is a sprint, as much of
                # one as the animal has left in it
                spd = spd * (1.0 + (ct.sprint_factor - 1.0)
                             * (1.0 - self.fatigue / 100.0))
            spd = spd * np.clip(speed_frac, 0.0, 1.0)
            satiated = np.asarray(perc.get("resting", speed_frac < 0.12), bool)
        else:
            spd = np.where(satiated, spd * cfg.rest_speed_factor, spd)
        self.effort = np.where(self.alive & (cruise > 1e-9),
                               spd / np.maximum(cruise, 1e-9), 0.0)
        self.P = self._resolve_obstacles(P, P + move_dir * spd[:, None] * dt,
                                         fwd=move_dir)
        self.H = np.arctan2(move_dir[:, 1], move_dir[:, 0])
        self._apply_boundary()
        if ct.enabled and len(self._cover):
            self._keep_out_of_full_huts(P)
        if not self.alive.all():          # dead animals do not move
            self.P[~self.alive] = P[~self.alive]

        # distance bookkeeping (per-day resets at midnight)
        step_dist = np.linalg.norm(self.P - P, axis=1)
        self.V = (self.P - P) / dt
        if ct.enabled:
            self._tire(cruise, spd, dt)
        self.dist_today += step_dist
        self.dist_total += step_dist
        day = int(elapsed_s // 86400) + 1
        if day != self._cur_day:
            self.dist_today[:] = 0.0
            self._cur_day = day

        # emergent home range: home slowly tracks occupied position (living only)
        self.home[self.alive] += ((self.P[self.alive] - self.home[self.alive])
                                  * (dt / self._settle_tau_s))

        # --- the sward changes because they were here -------------------- #
        # Walking wears it down a little, clipping a lot; both leave a route
        # that is cheaper for EVERY animal, not just the one that made it.
        if self.sward is not None:
            self.sward.trample(self.P, np.where(self.alive, step_dist, 0.0))
            cut = self.sward.chew(self.P, dt, chewing & self.alive)
            self.grass_cut_cm += cut
            self.chew_seconds += np.where(chewing, dt, 0.0)
            season = (growth_factor(self.when(elapsed_s))
                      if self.sky_p is not None and self.sky_p.seasonal_growth
                      else 1.0)
            self.sward.grow(dt, season)
            self.grass_cm = self.sward.sample(self.P)

        # --- scent marking (marks fade, then animals lay new ones) ---
        if self.scent is not None:
            self._elapsed = float(elapsed_s)
            self._update_scent(dt, perc.get("scent_foreign"))

        # --- condition dynamics ---
        on_food = self._on_food()
        on_water = self._on_water()
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

        # --- rest, warmth and attachment ---
        if self.soc is not None:
            self.soc.step(self, dt, dist, self.effort)

        # --- body mass drifts with energy balance, within a physiological band ---
        self.mass = np.clip(
            self.mass + dt * 1.5e-5 * (self.energy / 100.0 - 0.7),
            0.7 * self.mass0, 1.3 * self.mass0)

        # --- baseline activity label (events may override to flee/mate) ---
        self.activity = np.where(
            satiated, 0, np.where((need_food > 0) | (need_water > 0), 1, 2))
        if chewing.any():
            self.activity = np.where(chewing, 6, self.activity)
        self.activity[~self.alive] = 5

        # --- bites: the animals' own doing, resolved by the body's physics ---
        if ct.enabled:
            self._resolve_bites(elapsed_s, dt, perc.get("bite"), events)

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
    def _update_chewing(self, want_chew, dt: float) -> np.ndarray:
        """Advance the clipping state machine; returns who is chewing now.

        A bout is committed once it starts: the animal stops, and stays stopped
        for a time proportional to how tall the grass is
        (``chew_seconds_per_cm``). That commitment is what makes clipping a
        real decision rather than a free action taken every step — the cost is
        the foraging time it displaces, and a tall sward costs more of it.
        """
        n = self.n
        if self.sward is None:
            return np.zeros(n, bool)
        if len(self.chew_left) != n:            # the roster grew mid-run
            pad = n - len(self.chew_left)
            self.chew_left = np.concatenate([self.chew_left, np.zeros(pad)])
            self.chew_seconds = np.concatenate([self.chew_seconds,
                                                np.zeros(pad)])
            self.grass_cut_cm = np.concatenate([self.grass_cut_cm,
                                                np.zeros(pad)])
        self.chew_left = np.maximum(0.0, self.chew_left - dt)
        if want_chew is not None and np.any(want_chew):
            starting = np.asarray(want_chew, bool) & (self.chew_left <= 0.0)
            if starting.any():
                bout = self.sward.chew_bout_s(self.grass_cm)
                self.chew_left = np.where(starting, bout, self.chew_left)
        # dead animals do not clip
        self.chew_left = np.where(self.alive, self.chew_left, 0.0)
        return self.chew_left > 0.0

    def _seek(self, P, targets, need):
        d = np.linalg.norm(P[:, None, :] - targets[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        tv = targets[nearest] - P
        tdir = tv / (np.linalg.norm(tv, axis=1)[:, None] + 1e-9)
        return need[:, None] * tdir

    def _on_resource(self, P, targets, radii):
        d = np.linalg.norm(P[:, None, :] - targets[None, :, :], axis=2)
        return np.any(d < radii[None, :], axis=1)

    def _on_water(self, P=None):
        """At water: within reach of a water source or tower, or inside a
        resource zone that holds one."""
        P = self.P if P is None else P
        on = (self._on_resource(P, self.water, self.water_r)
              if self.water.shape[0] else np.zeros(len(P), bool))
        if len(self.water_rects):
            cx, cy, hw, hd = self.water_rects.T
            on = on | ((np.abs(P[:, 0][:, None] - cx[None, :]) <= hw[None, :])
                       & (np.abs(P[:, 1][:, None] - cy[None, :])
                          <= hd[None, :])).any(axis=1)
        return on

    def exit_door(self, P, zone_of):
        """The doorway each animal should make for to leave zone
        ``zone_of[i]`` — the nearest one, where the zone has several."""
        out = self.zone_doors[zone_of].astype(float).copy()
        for j, doors in enumerate(self._zone_door_sets):
            if len(doors) > 1:
                m = zone_of == j
                if m.any():
                    d = np.linalg.norm(P[m][:, None, :] - doors[None, :, :],
                                       axis=2)
                    out[m] = doors[np.argmin(d, axis=1)]
        return out

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
            remaining = self._food_left
            if np.isfinite(remaining).any():
                want = np.zeros(len(self._food_meta))
                np.add.at(want, who, grams)
                scale = np.ones(len(self._food_meta))
                lim = np.isfinite(remaining) & (want > 0)
                scale[lim] = np.minimum(1.0, remaining[lim] / want[lim])
                grams = grams * scale[who]
                taken = np.zeros(len(self._food_meta))
                np.add.at(taken, who, grams)
                fin = np.isfinite(self._food_left)
                self._food_left[fin] = np.maximum(
                    0.0, self._food_left[fin] - taken[fin])
                # mirror onto the resource so the remaining amount is visible
                for k, (_, obj) in enumerate(self._food_meta):
                    if obj is not None and np.isfinite(self._food_left[k]):
                        obj.amount_g = float(self._food_left[k])
            kj_in[eating] = grams * dens[who]
            ml_food[eating] = grams * wfrac[who]
            self.food_eaten_g[eating] += grams

        # ---- energy out ------------------------------------------------- #
        kj_out = p.basal_kj(mass, self.metabolism, dt)
        if self.soc is not None:
            # keeping warm is paid for: a cold animal burns more at rest, so a
            # huddle saves energy as well as discomfort
            self.soc.resize(self.n)
            kj_out = kj_out * (1.0 + self.cfg.social.cold_kj_factor
                               * self.soc.cold / 100.0)
        kj_out = kj_out + p.locomotion_kj(mass, step_dist)
        if self.sward is not None:
            # Forcing a body through standing grass costs on top of the
            # flat-ground price, per metre and per centimetre of sward. This is
            # what makes a worn trail *cheaper* as well as quicker, and so what
            # makes maintaining one pay for itself.
            kj_out = kj_out + self.sward.push_kj(mass, step_dist, self.grass_cm)
        kj_out[~alive] = 0.0

        self.energy = np.clip(
            self.energy + (kj_in - kj_out) / e_cap * 100.0, 0.0, 100.0)
        # hunger IS the energy deficit — nothing writes to it directly
        self.hunger = 100.0 - self.energy

        # ---- water ------------------------------------------------------ #
        drinking = on_water & alive
        ml_out = np.where(alive, p.water_loss_ml_h * (dt / 3600.0), 0.0)
        # an animal stops when it is no longer thirsty, exactly as it stops
        # eating when full: intake is capped at what the body is short of
        # (plus what it loses this step). Without the cap an animal resting
        # beside the water drank at the full rate for as long as it sat there
        # — ten times a vole's daily intake — and refilled its bladder, and so
        # its supply of scent marks, for nothing.
        room_ml = self.thirst / 100.0 * w_cap + ml_out
        ml_in = np.where(drinking, np.minimum(
            p.drink_rate_ml_min * (dt / 60.0), room_ml), 0.0)
        ml_in = ml_in + ml_food
        self.water_drunk_ml += ml_in
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
        self.last_voids = []
        perc = getattr(self, "_last_perc", None) or {}
        wall, sph = perc.get("bladder_wall"), perc.get("sphincter")
        if wall is not None and sph is not None and len(wall) == self.n:
            self._void_by_muscle(dt, np.asarray(wall, float), np.asarray(sph, float), cost)
            return
        can = self.alive & (reserve >= cost)
        # per-agent stream: whether one animal marks cannot depend on how many
        # other animals happened to draw before it this step
        fire = can & (self.arand.uniform(self.uid, self._step_k, CH_MARK)
                      < p_mark)
        idxs = np.nonzero(fire)[0]
        if len(idxs) == 0:
            return
        self.scent.deposit(idxs, self.P, sp.mark_strength,
                           self.identity[idxs])
        self._add_spots(idxs, np.minimum(self.bladder[idxs], cost[idxs]) / 100.0)
        self.bladder[idxs] = np.maximum(0.0, self.bladder[idxs] - cost[idxs])
        self.marks_made[idxs] += 1

    def _void_by_muscle(self, dt: float, wall, sph, cost) -> None:
        """Urination decided by two muscles rather than by chance (see
        ``ScentParams``): pressure against closure, and a void that runs on
        once it has started."""
        sp = self.cfg.scent
        if len(self._voiding) != self.n:
            self._voiding = np.concatenate(
                [self._voiding[:self.n], np.zeros(max(0, self.n - len(self._voiding)), bool)])
        push = wall + sp.void_stretch * self.bladder / 100.0 - sph - sp.void_resistance
        start = ~self._voiding & (push > 0.0) & (self.bladder >= cost)
        self._voiding = self.alive & (self._voiding | start) & (self.bladder > 0.5)
        out = np.where(self._voiding, np.minimum(self.bladder, sp.void_flow_s * 100.0 * dt), 0.0)
        idxs = np.nonzero(out > 0.0)[0]
        if len(idxs) == 0:
            return
        amount = sp.mark_strength * np.clip(out[idxs] / np.maximum(cost[idxs], 1e-9), 0.1, 1.0)
        for k, i in enumerate(idxs):                 # one cell each; amounts differ per animal
            self.scent.deposit(np.array([i]), self.P, float(amount[k]), self.identity[[i]])
        self._add_spots(idxs, out[idxs] / 100.0)
        self.bladder[idxs] -= out[idxs]
        self.marks_made[idxs[start[idxs]]] += 1      # a void is counted once, when it starts

    def _add_spots(self, idxs, vol) -> None:
        """Remember where urine landed. A void that carries on within 2.5 cm
        of the animal's last spot adds to that spot instead of making another."""
        s, t = self.urine_spots, getattr(self, "_elapsed", 0.0)
        for i, v in zip(np.atleast_1d(idxs), np.atleast_1d(vol)):
            x, y = float(self.P[i, 0]), float(self.P[i, 1])
            mine = np.nonzero(s["owner"] == i)[0]
            if len(mine) and np.hypot(*(s["xy"][mine[-1]] - (x, y))) < 0.025 and t - s["t"][mine[-1]] < 5.0:
                s["vol"][mine[-1]] += v
                s["t"][mine[-1]] = t
            else:
                s["xy"] = np.vstack([s["xy"], [x, y]])
                s["owner"] = np.append(s["owner"], int(i))
                s["vol"] = np.append(s["vol"], float(v))
                s["t"] = np.append(s["t"], t)
            self.last_voids.append((int(i), x, y, float(v)))
        if len(s["t"]) > 4000:                       # the oldest go first
            for key in s:
                s[key] = s[key][-3000:]

    def mup_barcodes(self) -> np.ndarray:
        """Each animal's urine barcode, ``(n, MUP_LOCI)`` (see
        :func:`fnt.abma.core.biology.mup_barcode`)."""
        from .biology import mup_barcode
        if self._mup is None or len(self._mup) != self.n:
            self._mup = np.stack([
                mup_barcode((a.genotype.genes or {}).get("MUP", "WT"),
                            int(self.uid[a.index]), self._root_seed)
                for a in self.agents])
        return self._mup

    def nose_contact(self, elapsed_s: float):
        """What each animal's nose is on: ``(level, barcode)``.

        ``level`` is 0 unless the nose (2 cm ahead of the body's centre) is
        within reach of a urine spot or of another animal's body; a spot's
        level fades with the scent half-life. ``barcode`` is that spot's or
        animal's MUP barcode. Nothing here says whose it is or whether it is
        the animal's own: that is for a brain to work out.
        """
        from .biology import MUP_LOCI
        n = self.n
        level, code = np.zeros(n), np.zeros((n, MUP_LOCI))
        if n == 0:
            return level, code
        sp, codes = self.cfg.scent, self.mup_barcodes()
        nose = self.P[:n] + 0.02 * np.stack([np.cos(self.H[:n]), np.sin(self.H[:n])], axis=1)
        s = self.urine_spots
        if len(s["t"]):
            fresh = 0.5 ** (np.maximum(0.0, elapsed_s - s["t"]) / max(1.0, sp.half_life_h * 3600.0))
            d = np.linalg.norm(nose[:, None, :] - s["xy"][None, :, :], axis=2)
            w = np.where(d < sp.spot_reach_m, fresh[None, :], 0.0)
            j = w.argmax(axis=1)
            level = w[np.arange(n), j]
            code = np.where(level[:, None] > 0.0, codes[np.clip(s["owner"][j], 0, n - 1)], 0.0)
        if n > 1:                                    # nose on another animal's body
            d = np.linalg.norm(nose[:, None, :] - self.P[None, :n, :], axis=2)
            d[np.arange(n), np.arange(n)] = np.inf
            d[:, ~self.alive[:n]] = np.inf
            j = d.argmin(axis=1)
            touch = d[np.arange(n), j] < 0.03
            level = np.where(touch, 1.0, level)
            code = np.where(touch[:, None], codes[j], code)
        level = np.where(self.alive[:n], level, 0.0)
        return level, code

    # ------------------------------------------------------------------ #
    # Olfactory recognition
    # ------------------------------------------------------------------ #
    def mark_biology_dirty(self) -> None:
        """Flag that ``smell`` or ``identity`` changed, staling the recognition.

        Called whenever a drug takes effect, an intervention fires, or the
        roster grows. Recognition is expensive relative to a step but changes
        only at those moments, so it is rebuilt on demand rather than per step.
        """
        self._olf_dirty = True

    def recognition_matrix(self) -> np.ndarray:
        """``R[i, j]`` — how well animal *i* recognises animal *j*'s scent.

        Two implementations behind one call:

        * **legacy** (default) — the scalar gate ``smell_i x identity_j``.
        * **mechanistic** (``olfaction.enabled``) — a receptor/signature model
          whose cohort mean tracks the scalar gate but which degrades
          *selectively*, so a partially anosmic animal is confused about some
          individuals and not others. See :mod:`fnt.abma.core.olfaction`.

        Cached; invalidated by :meth:`mark_biology_dirty`.
        """
        if self.olf is None:
            return np.outer(self.smell, self.identity)
        if self._olf_dirty or self._recog is None:
            if len(self.olf.uids) != self.n:
                self.olf.set_roster(self.uid)
            self.olf.rebuild(self.smell, self.identity)
            self._recog = self.olf.recognition
            self._olf_dirty = False
        return self._recog

    def discrimination_matrix(self) -> np.ndarray:
        """``C[i, j]`` — identity legibility with detection factored *out*.

        The scent field needs to know how legible a mark's signature is, but
        not how well the reader can smell at all: the policy already gates
        marks by the reader's acuity (``k_scent_avoid * smell``), so folding
        detection in here too would count anosmia twice.

        Legacy path: ``identity_j`` broadcast over readers — every animal reads
        a given mark equally well, which is what the stored per-cell ``ident``
        already encodes. Mechanistic path: ``recognition / detection``, i.e. the
        purely perceptual "can I tell whose this is" term.
        """
        if self.olf is None:
            return np.broadcast_to(self.identity[None, :], (self.n, self.n))
        recog = self.recognition_matrix()
        det = self.olf.detection
        return np.where(det > 1e-9, recog / np.clip(det, 1e-9, None), 0.0)

    #: set by a live view that wants the scent map drawn under the arena.
    #: Off by default: headless runs should not pay to rasterise a picture.
    emit_scent_map = False

    def territory_image(self, max_side: int = 160):
        """The scent field as an RGBA image: who owns the ground, how strongly.

        Territory is the thing this whole model exists to produce, and until
        now it was invisible — an emergent mosaic you could only see by opening
        a CSV after the fact. Each cell is tinted with its owner's colour and
        made more opaque by mark strength, so the map that the animals are
        actually navigating is the map on screen.

        Returns ``(rgba, extent)`` with extent ``(x0, x1, y0, y1)`` in metres,
        or ``None`` when marking is off. Large arenas are block-averaged down
        to ``max_side`` so a 75-foot enclosure costs the same as a cage.
        """
        if self.scent is None:
            return None
        owner, strength = self.scent.occupancy()
        ny, nx = owner.shape
        rgba = np.zeros((ny, nx, 4), np.float32)
        live = owner >= 0
        if live.any():
            idx = np.clip(owner[live], 0, max(0, len(self.agent_rgba) - 1))
            rgba[live, :3] = self.agent_rgba[idx][:, :3]
            # alpha carries mark strength, so a fresh boundary reads darker
            # than ground someone crossed once and left
            rgba[live, 3] = np.clip(strength[live], 0.0, 1.0) * 0.75
        cell = self.scent.cell
        step = max(1, int(np.ceil(max(ny, nx) / max(8, max_side))))
        if step > 1:
            # Block-average. The trailing partial block is dropped, so the
            # extent has to shrink with it — reporting the full arena would
            # stretch the image and slide the territory boundaries off the
            # positions the animals are actually at.
            ty, tx = (ny // step) * step, (nx // step) * step
            rgba = (rgba[:ty, :tx]
                    .reshape(ty // step, step, tx // step, step, 4)
                    .mean(axis=(1, 3)))
            ny, nx = ty, tx
        out = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)
        return out, (0.0, nx * cell, 0.0, ny * cell)

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
            sides = zone_sides(z)
            for yw, name in ((y1, "N"), (y0, "S")):        # walls running E-W
                if name in sides:                   # split around the doorway
                    segs += [(x0, yw, z.x - hw / 2, yw),
                             (z.x + hw / 2, yw, x1, yw)]
                else:
                    segs += [(x0, yw, x1, yw)]
            for xw, name in ((x1, "E"), (x0, "W")):        # walls running N-S
                if name in sides:
                    segs += [(xw, y0, xw, z.y - hw / 2),
                             (xw, z.y + hw / 2, xw, y1)]
                else:
                    segs += [(xw, y0, xw, y1)]
        if getattr(a, "boundary", "reflective") != "wrap":
            segs = [_run_wall_past_boundary(s, a.width, a.height) for s in segs]
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
            step = P_new - P_old
            P_new = P_old + step * scale[:, None]
            # Slide along the obstacle instead of stopping dead against it.
            # Without this an animal whose target is on the far side of a wall
            # presses into that wall indefinitely — which is how agents end up
            # pinned beside a resource zone, or trapped inside one, until they
            # starve. Retrying the blocked remainder one axis at a time lets
            # them skirt the wall and find the doorway.
            resid = step * (1.0 - scale)[:, None]
            blocked = np.nonzero(moved)[0]
            for axis in (0, 1):
                if not len(blocked):
                    break
                cand = P_new[blocked].copy()
                cand[:, axis] += resid[blocked, axis]
                free = self._first_hit(P_new[blocked], cand) >= 1.0
                if free.any():
                    idx = blocked[free]
                    P_new[idx] = cand[free]
        self._push_out_solids(P_new)
        return P_new

    def _push_out_solids(self, P) -> None:
        """Eject anything sitting inside a solid, in place.

        Called after obstacle resolution *and* again after boundary handling:
        a reflective wall can bounce an animal straight into a pole, which the
        sweep never sees because the reflected position is not on the path.
        """
        C = self._obs_circles
        if not len(C):
            return
        n = len(P)
        cxa, cya, cra = C.T
        for _ in range(2):
            dx = P[:, 0][:, None] - cxa[None, :]
            dy = P[:, 1][:, None] - cya[None, :]
            dist = np.hypot(dx, dy)
            inside = dist < cra[None, :]
            if not inside.any():
                break
            dm = np.where(inside, dist, np.inf)
            j = np.argmin(dm, axis=1)
            for i in np.where(np.isfinite(dm[np.arange(n), j]))[0]:
                k = j[i]
                d = max(dist[i, k], 1e-9)
                P[i] = [cxa[k] + dx[i, k] / d * cra[k],
                        cya[k] + dy[i, k] / d * cra[k]]

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
        # a bounce can land an animal inside a pole or tower — undo that
        self._push_out_solids(self.P)

    # ------------------------------------------------------------------ #
    # Contest physics (ContestParams): tiring, cover, bites
    # ------------------------------------------------------------------ #
    def _build_cover(self) -> None:
        """Huts as cover: ``(x, y, radius)`` per hut, or none."""
        ct = self.cfg.contest
        huts = self.cfg.arena.huts if (ct.enabled and ct.refuge_huts) else []
        # a tube is a rectangle; a circle of its half-length is close enough
        # for "is this animal under it"
        self._cover = np.array(
            [(h.x, h.y, 0.5 * max(h.w, h.d if h.kind == "tube" else h.w))
             for h in huts], float).reshape(-1, 3)

    def cover_of(self, P=None) -> np.ndarray:
        """Index of the hut each animal is under, or -1."""
        P = self.P if P is None else P
        out = np.full(len(P), -1, int)
        if len(self._cover):
            d = np.linalg.norm(P[:, None, :] - self._cover[None, :, :2], axis=2)
            k = np.argmin(d, axis=1)
            inside = d[np.arange(len(P)), k] <= self._cover[k, 2]
            out[inside] = k[inside]
        return out

    def _keep_out_of_full_huts(self, P_old) -> None:
        """A hut holds ``refuge_capacity`` animals; latecomers stay outside.

        Whoever was already under it keeps its place. Among animals arriving
        in the same step the ones that end nearest its middle get in. An
        animal turned away goes back to where it started the step, or to the
        hut's edge if that was inside too.
        """
        cap = int(self.cfg.contest.refuge_capacity)
        if cap <= 0:
            return
        now = self.cover_of(self.P)
        now[~self.alive] = -1
        before = self.cover_of(P_old)
        for k in np.unique(now[now >= 0]):
            inside = np.nonzero(now == k)[0]
            if len(inside) <= cap:
                continue
            c = self._cover[k]
            d = np.linalg.norm(self.P[inside] - c[:2], axis=1)
            # residents first, then by closeness to the middle
            order = inside[np.lexsort((d, before[inside] != k))]
            for i in order[cap:]:
                if before[i] != k:
                    self.P[i] = P_old[i]
                else:
                    v = self.P[i] - c[:2]
                    nv = np.linalg.norm(v)
                    v = v / nv if nv > 1e-9 else np.array([1.0, 0.0])
                    self.P[i] = c[:2] + v * (c[2] + 0.005)

    def _tire(self, cruise, spd, dt: float) -> None:
        """Fatigue fills above the easy pace and drains below it."""
        ct = self.cfg.contest
        effort = np.where(cruise > 1e-9, spd / np.maximum(cruise, 1e-9), 0.0)
        over = np.clip((effort - 1.0) / max(1e-6, ct.sprint_factor - 1.0), 0, 1)
        rest = np.clip(1.0 - effort, 0.0, 1.0)
        self.fatigue = np.clip(
            self.fatigue + dt * (ct.fatigue_per_s * over
                                 - ct.recover_per_s * rest), 0.0, 100.0)
        self.fatigue[~self.alive] = 0.0

    def _resolve_bites(self, elapsed_s, dt: float, want, events) -> None:
        """Land the bites the animals asked for.

        A bite goes to the living animal that came nearest during the step,
        if that was within reach and inside the cone ahead of the biter. It
        does nothing if a wall stood between them, or if the target is under a
        hut the biter is not under. Bites issued in the same step all land on
        the state as it was, so two animals can hurt each other at once.

        "Came nearest" is measured along the paths of the step, not from where
        the two ended up: at a coarse step a sprinting animal covers several
        times the reach, and judged by end points alone it could run straight
        through another without either being in range.
        """
        ct = self.cfg.contest
        self.bite_wait = np.maximum(0.0, self.bite_wait - dt)
        self.hurt = self.hurt * np.exp(-dt / max(1e-6, ct.hurt_tau_s))
        if want is None:
            return
        who = np.nonzero(np.asarray(want, bool) & self.alive
                         & (self.bite_wait <= 0.0))[0]
        if not len(who):
            return
        P, H = self.P, self.H
        D = self.V * dt                       # this step's displacement
        P0 = P - D
        reach = self.contact_r + ct.bite_reach_m
        half = np.deg2rad(ct.bite_arc_deg) / 2.0
        cover = self.cover_of(P)
        self.bite_wait[who] = ct.bite_refractory_s
        if self._physio_on:
            self._spend_energy_kj(tuple(who), ct.bite_kj)
        else:
            self.energy[who] = np.clip(self.energy[who] - 0.2, 0, 100)
        hits = []
        for i in who:
            # closest approach of every other animal to this one during the
            # step: r(t) = r0 + t * dr, t in [0, 1]
            r0 = P0 - P0[i]
            dr = D - D[i]
            t = np.clip(-(r0 * dr).sum(axis=1)
                        / np.maximum((dr * dr).sum(axis=1), 1e-12), 0.0, 1.0)
            d = r0 + t[:, None] * dr
            dist = np.linalg.norm(d, axis=1)
            dist[i] = np.inf
            dist[~self.alive] = np.inf
            ang = np.arctan2(d[:, 1], d[:, 0]) - H[i]
            ang = np.abs((ang + np.pi) % (2.0 * np.pi) - np.pi)
            dist[ang > half] = np.inf
            j = int(np.argmin(dist))
            if not dist[j] < reach:
                continue
            if cover[j] >= 0 and cover[i] != cover[j]:
                continue                               # the target is under cover
            at_i = (P0[i] + t[j] * D[i])[None, :]
            at_j = (P0[j] + t[j] * D[j])[None, :]
            if self._first_hit(at_i, at_j)[0] < 1.0:
                continue                               # a wall is in the way
            dmg = (ct.bite_damage * (self.mass[i] / 25.0)
                   * (0.5 + 0.5 * self.health[i] / 100.0))
            hits.append((int(i), j, float(dmg)))
        for i, j, dmg in hits:
            self.health[j] = max(0.0, self.health[j] - dmg)
            self.stress[j] = min(100.0, self.stress[j] + ct.bite_stress)
            self.hurt[j] = min(1.0, self.hurt[j] + dmg / 10.0)
            back = P[i] - P[j]
            self.hurt_dir[j] = back / (np.linalg.norm(back) + 1e-9)
            self.bites_given[i] += 1
            self.bites_taken[j] += 1
            if events is not None:
                events.record(elapsed_s, "bite", self.agents[i], self.agents[j],
                              P[j, 0], P[j, 1], round(dmg, 4))

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
        # Dyadic draws, batched: a contest or a mating belongs to the pair, so
        # it gets a per-pair stream keyed by both uids. Drawn for every contact
        # up front rather than inside the loop — vectorised, and (the point)
        # independent of how many other pairs were resolved first, so removing
        # or ablating one animal cannot reshuffle another pair's outcome.
        if len(contact):
            ua = self.uid[contact[:, 0]]
            ub = self.uid[contact[:, 1]]
            k = self._step_k
            u_mate = self.arand.pair_uniform(ua, ub, k, CH_MATE)
            u_fight = self.arand.pair_uniform(ua, ub, k, CH_FIGHT)
            u_win = self.arand.pair_uniform(ua, ub, k, CH_FIGHT_OUTCOME)
        for pair_k, (i, j) in enumerate(contact):
            if i >= j or not (self.alive[i] and self.alive[j]):
                continue
            ai, aj = self.agents[i], self.agents[j]
            if ai.sex != aj.sex:
                # --- mating: opposite sex, female receptive ---
                fem = i if ai.sex == "F" else j
                if (rec_j[fem] > 0.3 and u_mate[pair_k] < p_mate
                        and elapsed_s - self._last_mate.get((i, j), -1e9)
                        >= self.mate_cooldown_s):
                    self._last_mate[(i, j)] = elapsed_s
                    male, female = (aj, ai) if ai.sex == "F" else (ai, aj)
                    events.record(elapsed_s, "mating", male, female,
                                  self.P[i, 0], self.P[i, 1], rec_j[fem])
                    self.matings[i] += 1
                    self.matings[j] += 1
                    if self.soc is not None:
                        self.soc.on_mating(self, int(i), int(j))
                    self.activity[i] = self.activity[j] = 4
                    if self._physio_on:
                        self._spend_energy_kj((i, j), self.cfg.physiology.mate_kj)
                    else:
                        self.energy[[i, j]] = np.clip(
                            self.energy[[i, j]] - 1.0, 0, 100)
            else:
                # --- same-sex contest -> winner/loser, damage, dominance ---
                if self.cfg.contest.enabled:
                    continue     # a contest is then bites the animals choose
                # A contest is about ground, not about walking. The thing that
                # must NOT register as a fight is an affiliative huddle — and a
                # huddle is *both* animals settled together, so that is what
                # this excludes.
                #
                # The rule used to require at least one animal to be ROAMING.
                # That was wrong in exactly the case the enclosure model is
                # for: once animals successfully settle they rest ~84% of the
                # time, only ~2% of same-sex contacts had anyone roaming, and a
                # resident was structurally incapable of defending its patch
                # against an intruder walking through it. Dominance could not
                # form, so `dominance_<trial>.csv` was empty of content.
                if self.activity[i] == 0 and self.activity[j] == 0:
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
                if u_fight[pair_k] >= p_attack:
                    continue
                fi, fj = self._fight_power(i), self._fight_power(j)
                if u_win[pair_k] < fi / (fi + fj + 1e-9):
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
            frame_interval_s: float = 300.0, meta_cb=None,
            record_frames: bool = True) -> dict:
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
        # The replayable archive: same frames the live view gets, kept so a
        # finished run can be reopened and inspected animal by animal rather
        # than only read back as a table of positions. See core/record.py.
        rec_path = os.path.join(output_dir, f"record_{self.trial_id}.npz")
        archive = (RunRecord(trial_id=self.trial_id, n_agents=self.n,
                             frame_interval_s=frame_interval_s,
                             agents=self.agent_static())
                   if record_frames else None)

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
                if k % frame_every == 0 and (frame_cb is not None
                                             or archive is not None):
                    fr = self._frame(elapsed)
                    if archive is not None:
                        archive.append(fr)
                    if frame_cb is not None:
                        frame_cb(fr)
                if progress_cb is not None and k % report_every == 0:
                    progress_cb(k / n_steps)
        finally:
            rec.close()
            evt.close()
            cond.close()
        if progress_cb is not None:
            progress_cb(1.0)
        out = {"trajectory": traj_path, "events": evt_path,
               "condition": cond_path, "agents": agents_path,
               "trial_id": self.trial_id}
        if archive is not None and len(archive):
            archive.agents = self.agent_static()   # final roster + treatments
            archive.save(rec_path)
            out["record"] = rec_path
        return out

    def _drive_magnitudes(self) -> dict:
        """Per-agent magnitude of each named drive from the last step.

        Zeros for a drive this configuration does not use (``home`` only exists
        without scent marking, ``scent_home``/``memory`` only with it), so the
        record has a fixed set of columns either way.
        """
        n = self.n
        perc = getattr(self, "_last_perc", None) or {}
        drives = perc.get("drives", {})
        out = {}
        for name in ("scent_home", "memory", "home", "resource", "social",
                     "territory", "wander"):
            v = drives.get(name)
            out[f"drive_{name}"] = (np.linalg.norm(v, axis=1) if v is not None
                                    else np.zeros(n))
        desired = getattr(self, "_last_desired", None)
        out["desired_x"] = (desired[:, 0].copy() if desired is not None
                            else np.zeros(n))
        out["desired_y"] = (desired[:, 1].copy() if desired is not None
                            else np.zeros(n))
        return out

    def _environment_summary(self, elapsed: float) -> dict:
        """Per-agent environment state: the sward underfoot, and clipping."""
        n = self.n
        if self.sward is None:
            return {"grass_cm": np.zeros(n), "grass_speed_factor": np.ones(n),
                    "chewing": np.zeros(n), "grass_cut_cm": np.zeros(n)}
        return {
            "grass_cm": self.grass_cm.copy(),
            "grass_speed_factor": self.sward.speed_factor(self.grass_cm),
            "chewing": (self.chew_left > 0).astype(float),
            "grass_cut_cm": self.grass_cut_cm.copy(),
        }

    def _perception_summary(self) -> dict:
        """Per-agent summary of what each animal was sensing last step."""
        n = self.n
        perc = getattr(self, "_last_perc", None) or {}
        own = perc.get("scent_own")
        foreign = perc.get("scent_foreign")
        within = perc.get("within")
        recog = perc.get("recognition")
        # mean over *other* animals: what this nose makes of its cohort
        if recog is not None and n > 1:
            rec_mean = (recog.sum(axis=1) - np.diag(recog)) / (n - 1)
        else:
            rec_mean = np.zeros(n)
        if self.olf is not None and n > 1:
            det = self.olf.detection
            det_mean = (det.sum(axis=1) - np.diag(det)) / (n - 1)
        else:
            det_mean = self.smell.copy()
        return {
            "scent_own": own if own is not None else np.zeros(n),
            "scent_foreign": foreign if foreign is not None else np.zeros(n),
            "recognition_mean": rec_mean,
            "detection_mean": det_mean,
            "need_food": perc.get("need_food", np.zeros(n)),
            "need_water": perc.get("need_water", np.zeros(n)),
            "neighbours": (within.sum(axis=1) if within is not None
                           else np.zeros(n)),
            "territory_m2": self.territory_area(),
        }

    def _frame(self, elapsed: float) -> dict:
        """Snapshot streamed to the live view / inspector (all arrays copied).

        Carries three things: where every animal is, how it is doing, and — the
        part a trajectory alone cannot recover — what it was trying to do and
        what it was sensing when it decided.
        """
        extra = {}
        sky = self.sky_at(elapsed)
        if sky is not None:
            # the sun and moon are part of the world, so the views can draw
            # them and an analysis can ask what the light was doing
            extra.update(
                sun_elevation=sky.sun_elevation, sun_azimuth=sky.sun_azimuth,
                moon_elevation=sky.moon_elevation,
                moon_azimuth=sky.moon_azimuth,
                moon_phase=sky.moon_phase,
                moon_illumination=sky.moon_illumination,
                daylight=sky.daylight, night_light=sky.night_light)
        if self.sward is not None and self.emit_scent_map:
            got = self.sward.image()
            extra["grass_rgba"], extra["grass_extent"] = got
            extra["grass_mean_cm"] = self.sward.mean_height()
            extra["trail_fraction"] = self.sward.trail_fraction()
        if self.emit_scent_map:
            got = self.territory_image()
            if got is not None:
                extra["scent_rgba"], extra["scent_extent"] = got
        frame_extra = getattr(self.policy, "frame_extra", None)
        if frame_extra is not None:
            # a learned brain's activity: every unit, every animal, this tick
            extra.update(frame_extra(self.n))
        return {
            **self._drive_magnitudes(),
            **self._perception_summary(),
            **self._environment_summary(elapsed),
            **extra,
            "marks_made": self.marks_made.copy(),
            "discomfort": discomfort(self),
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
