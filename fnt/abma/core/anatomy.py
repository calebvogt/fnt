"""A brain wired like the animal's: one block of units per anatomical area.

The learned brain in :mod:`fnt.abma.core.brain` is one sensory layer and one
recurrent layer; nothing in it corresponds to a part of a real brain. This
module builds a network from an *anatomy spec* instead: a JSON file that lists
the areas, the projections between them (with a sign and a strength), the
neuromodulators (where each is released and how many receptors each area has)
and where every sense enters.

    spec = load_spec("c57_male")
    cfg.brain.enabled, cfg.brain.anatomy = True, "c57_male"

A unit of area A can talk to a unit of area B only where the spec lists a
projection A -> B, and only with the sign the spec gives it. **The weights
themselves are random draws.** Nothing here has been trained or selected, so
what such an animal does means nothing yet; what is real is the route activity
takes through the wiring, and what changes when a projection or a receptor
level is changed. ``anatomy_seed`` picks the draw.

How the nervous system meets the body:

*In.* The engine's sensor channels (:data:`fnt.abma.core.senses.SENSORS`) enter
only at the sense organ, or the area, that the spec's ``entry`` table names. A
channel the spec does not place is not wired at all (the four hand-set
"disposition" channels are left out on purpose: this brain is meant to replace
them). The vomeronasal organ has no channel yet, because the engine has no
sense of a urine mark's identity on nose contact.

*Out.* The units of class ``output`` are muscles. ``TURN`` is a left and a
right unit, read as their difference; ``SPEED`` and ``BITE`` are read as the
mean of their units. Urination is two muscles, as in the animal: the bladder
wall (``DET``) and the urethral sphincter (``EUS``, tonically active). The
push ``DET * (1 - EUS)`` is recorded but the engine does not obey it yet.

*Slow chemistry.* A neuromodulator's level multiplies the gain of every area
by that area's receptor level. The level is the mean activity of the area that
releases it, or a state of the body where the source is outside the brain
(:data:`BODY_MODULATORS`).

Every unit's activity is a firing rate as a fraction of its maximum, 0 to 1.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .brain import package_dir as _brain_package_dir
from .brain import user_dir as _brain_user_dir
from .policy import Policy
from .senses import SENSORS, PlaceMemory, SenseParams, sense

#: modulators whose level is a state of the body, not the activity of an area
BODY_MODULATORS = {"CRF": "stress", "T": "male"}
#: the muscles the engine (or the record) reads; a spec must have all of them
MUSCLES = ("TURN", "SPEED", "BITE", "DET", "EUS")
#: channels that carry a direction or a signed quantity (the rest are levels)
_SIGNED = ("_fwd", "_lat", "last_turn", "clock_sin", "clock_cos", "other_size")
_SIGNS = ("+", "-", "±")


# --------------------------------------------------------------------------- #
# Specs on disk
# --------------------------------------------------------------------------- #
def package_dir() -> str:
    return os.path.join(_brain_package_dir(), "anatomy")


def user_dir() -> str:
    return os.path.join(_brain_user_dir(), "anatomy")


def list_anatomies() -> list[tuple[str, str]]:
    """Every anatomy spec that can be asked for by name, as ``(name, path)``;
    the user's own first, so an edited copy shadows the shipped one."""
    out, seen = [], set()
    for d in (user_dir(), package_dir()):
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            name = os.path.splitext(os.path.basename(p))[0]
            if name not in seen:
                seen.add(name)
                out.append((name, p))
    return out


def find_anatomy(name: str) -> str:
    if name and os.path.isfile(name):
        return name
    for n, p in list_anatomies():
        if n == name:
            return p
    have = ", ".join(n for n, _ in list_anatomies()) or "none"
    raise FileNotFoundError(f"no anatomy named {name!r} (available: {have}); "
                            "or give a path to a .json spec")


def check_spec(spec: dict) -> None:
    """Raise ``ValueError`` naming the first thing wrong with a spec."""
    def need(ok, msg):
        if not ok:
            raise ValueError(f"anatomy spec: {msg}")

    for key in ("classes", "regions", "projections", "modulators", "entry"):
        need(key in spec, f"missing '{key}'")
    ids = [r["id"] for r in spec["regions"]]
    need(len(ids) == len(set(ids)), "an area id is used twice")
    classes = {c["id"] for c in spec["classes"]}
    for r in spec["regions"]:
        need(r["class"] in classes, f"{r['id']} has an unknown class")
        need(int(r["n"]) >= 2, f"{r['id']} needs at least two units")
    for m in MUSCLES:
        need(m in ids, f"the muscle {m} is missing")
    seen = set()
    for p in spec["projections"]:
        a, b, sign, w = p[:4]
        need(a in ids and b in ids, f"projection {a}>{b} names an unknown area")
        need(a != b and (a, b) not in seen, f"projection {a}>{b} is a repeat or a loop onto itself")
        need(sign in _SIGNS and w in (1, 2, 3), f"projection {a}>{b}: sign must be + - or ±, strength 1-3")
        seen.add((a, b))
    for rid in spec["entry"]:
        need(rid in ids, f"entry names an unknown area {rid}")
    for m in spec["modulators"]:
        need(m["source"] in ids or m["source"] == "", f"modulator {m['id']} has an unknown source")
        for rid, lev in m["receptors"].items():
            need(rid in ids and lev in (1, 2, 3), f"modulator {m['id']}: receptor level for {rid} must be 1-3")
    # an area that reaches no muscle and releases nothing is decoration
    out = {a: set() for a in ids}
    for p in spec["projections"]:
        out[p[0]].add(p[1])
    sources = {m["source"] for m in spec["modulators"]}
    outputs = {r["id"] for r in spec["regions"] if r["class"] == "output"}
    for rid in ids:
        if rid in outputs:
            continue
        todo, done = [rid], set()
        while todo:
            x = todo.pop()
            if x not in done:
                done.add(x)
                todo.extend(out[x])
        need((done & outputs) or (done & sources), f"{rid} reaches no muscle and no area that releases a signal")


def load_spec(name_or_path: str) -> dict:
    path = find_anatomy(name_or_path)
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    check_spec(spec)
    return spec


def describe(spec: dict) -> str:
    n = sum(r["n"] for r in spec["regions"])
    tied = sum(1 for p in spec["projections"] if len(p) > 4 and p[4])
    return (f"{spec.get('name', '?')} · {len(spec['regions'])} areas, {n} units, "
            f"{len(spec['projections'])} projections ({tied} tied to evidence), "
            f"{len(spec['modulators'])} neuromodulators · untrained")


def parse_strength(text: str) -> tuple[str, float]:
    """``'mPFC>DRN=1.5'`` -> ``('mPFC>DRN', 1.5)``: multiply that projection."""
    key, sep, val = text.partition("=")
    if not sep or ">" not in key:
        raise ValueError(f"expected A>B=X (e.g. mPFC>DRN=1.5), got {text!r}")
    x = float(val)
    if not 0.0 <= x <= 3.0:
        raise ValueError(f"a strength multiplier is 0 to 3, got {x}")
    return key.strip(), x


def parse_receptor(text: str) -> tuple[str, int]:
    """``'OT@LS=3'`` -> ``('OT@LS', 3)``: set that area's receptor level."""
    key, sep, val = text.partition("=")
    if not sep or "@" not in key:
        raise ValueError(f"expected MODULATOR@AREA=N (e.g. OT@LS=3), got {text!r}")
    n = int(val)
    if n not in (0, 1, 2, 3):
        raise ValueError(f"a receptor level is 0 to 3, got {n}")
    return key.strip(), n


# --------------------------------------------------------------------------- #
# The network
# --------------------------------------------------------------------------- #
class AnatomicalBrain:
    """Rate units in anatomical blocks. ``step`` returns ``r``-derived
    commands; ``r`` itself is ``(animals, units)``, each in 0..1.

    ``strengths`` multiplies single projections (``{"mPFC>DRN": 1.5}``; 0 cuts
    one), ``receptors`` overrides single receptor levels (``{"OT@LS": 3}``).
    Both are applied on top of the same random draw, so two brains that differ
    only in an edit are otherwise identical cell for cell.
    """

    def __init__(self, spec: dict, seed: int = 0, tick_s: float = 0.25,
                 tau_s: float = 0.6, g_proj: float = 1.6, g_local: float = 0.5,
                 bias: float = 0.25, noise: float = 0.15, k_inh: float = 3.0,
                 mod_gain: float = 0.5, max_turn_rad_s: float = 2.5,
                 strengths: dict | None = None, receptors: dict | None = None):
        check_spec(spec)
        rng = np.random.default_rng(seed)
        self.spec, self.seed, self.tick_s = spec, int(seed), float(tick_s)
        self.alpha = min(1.0, tick_s / tau_s)
        self.noise, self.max_turn = float(noise), float(max_turn_rad_s)
        self.ids = [r["id"] for r in spec["regions"]]
        self.sl, k = {}, 0
        for r in spec["regions"]:
            self.sl[r["id"]] = slice(k, k + r["n"])
            k += r["n"]
        self.n = k
        self.region_of = np.concatenate([np.full(r["n"], i) for i, r in enumerate(spec["regions"])])
        #: a cell's resting drive; an area marked "tonic" is active unless
        #: something holds it back (the urethral sphincter)
        self.bias = np.concatenate([np.full(r["n"], bias + r.get("tonic", 0.0)) for r in spec["regions"]])
        # every area's own interneurons: each cell is held back by the mean
        # activity of its area, so an area cannot switch fully on and stay on
        same = self.region_of[:, None] == self.region_of[None, :]
        self.pool = same / np.bincount(self.region_of)[self.region_of][None, :]
        self.k_inh = float(k_inh)

        strengths = dict(strengths or {})
        known = {f"{p[0]}>{p[1]}" for p in spec["projections"]}
        bad = sorted(set(strengths) - known)
        if bad:
            raise ValueError(f"no such projection: {', '.join(bad)}")
        self.strengths = {k_: float(v) for k_, v in strengths.items()}

        W = np.zeros((self.n, self.n))
        for r in spec["regions"]:        # local wiring: mixed, weak, no self-connection
            s = self.sl[r["id"]]
            w = rng.standard_normal((r["n"], r["n"])) * g_local / np.sqrt(r["n"])
            np.fill_diagonal(w, 0.0)
            W[s, s] = w
        for a, b, sign, strength, *_ in spec["projections"]:
            sa, sb = self.sl[a], self.sl[b]
            na, nb = sa.stop - sa.start, sb.stop - sb.start
            w = np.abs(rng.standard_normal((nb, na))) * (rng.random((nb, na)) < 0.6)
            w *= g_proj * strength / 3.0 / np.sqrt(0.6 * na)
            if sign == "-":
                w = -w
            elif sign != "+":            # mixed cell types: each source cell is one or the other
                w *= np.where(rng.random(na) < 0.5, 1.0, -1.0)[None, :]
            W[sb, sa] = w * self.strengths.get(f"{a}>{b}", 1.0)
        self.W = W

        col = {s: i for i, s in enumerate(SENSORS)}
        Win = np.zeros((self.n, len(SENSORS)))
        #: channels the spec places that this engine does not have
        self.unwired = []
        for rid, chans in spec["entry"].items():
            s = self.sl[rid]
            width = s.stop - s.start
            for j in range(width):       # each cell has one channel it mainly answers to
                for c, name in enumerate(chans):
                    w = (1.6 if (c % width) == j else 0.15) * abs(rng.standard_normal())
                    if name.endswith(_SIGNED) or name in _SIGNED:
                        w *= rng.choice([-1.0, 1.0])
                    if name in col:
                        Win[s.start + j, col[name]] = w
                    elif name not in self.unwired:
                        self.unwired.append(name)
        self.Win = Win

        receptors = dict(receptors or {})
        mod_ids = {m["id"] for m in spec["modulators"]}
        for key in receptors:
            mid, _, rid = key.partition("@")
            if mid not in mod_ids or rid not in self.sl:
                raise ValueError(f"no such modulator and area: {key}")
        self.receptors = {k_: int(v) for k_, v in receptors.items()}
        self.mods = []
        for m in spec["modulators"]:
            dens = np.zeros(self.n)
            levels = dict(m["receptors"])
            for key, lev in self.receptors.items():
                mid, _, rid = key.partition("@")
                if mid == m["id"]:
                    levels[rid] = lev
            for rid, lev in levels.items():
                dens[self.sl[rid]] = lev / 3.0
            self.mods.append((m["id"], m["source"], dens * mod_gain))
        self._body_col = {mid: col[name] for mid, name in BODY_MODULATORS.items() if name in col}
        self.sense_units = np.concatenate([np.arange(self.sl[r["id"]].start, self.sl[r["id"]].stop)
                                           for r in spec["regions"] if r["class"] == "sense"])

    def unit_names(self) -> list[str]:
        return [f"{r['id']} {j + 1}" for r in self.spec["regions"] for j in range(r["n"])]

    def area_means(self, r: np.ndarray) -> np.ndarray:
        """``(..., units)`` -> ``(..., areas)``."""
        return np.stack([r[..., self.sl[i]].mean(axis=-1) for i in self.ids], axis=-1)

    def initial_state(self, n: int) -> np.ndarray:
        return np.zeros((n, self.n))

    def step(self, r: np.ndarray, obs: np.ndarray, rng: np.random.Generator):
        """One tick. Returns the new ``r`` (the old one is not modified)."""
        obs = np.asarray(obs, float)
        gain = np.ones((len(obs), self.n))
        for mid, source, dens in self.mods:
            if mid in self._body_col:
                level = np.clip(obs[:, self._body_col[mid]], 0.0, 1.0)
            elif source:
                level = r[:, self.sl[source]].mean(axis=1)
            else:
                continue
            gain += level[:, None] * dens[None, :]
        x = r @ self.W.T + obs @ self.Win.T + self.bias - self.k_inh * (r @ self.pool.T)
        x = gain * x + self.noise * rng.standard_normal(x.shape)
        return r + self.alpha * (np.tanh(np.maximum(x, 0.0)) - r)

    def commands(self, r: np.ndarray):
        """Turn (rad/s), speed (0..1), bite (bool), urine push (0..1)."""
        t = r[:, self.sl["TURN"]]
        turn = self.max_turn * (t[:, 1] - t[:, 0])
        speed = r[:, self.sl["SPEED"]].mean(axis=1)
        bite = r[:, self.sl["BITE"]].mean(axis=1) > 0.5
        push = r[:, self.sl["DET"]].mean(axis=1) * (1.0 - r[:, self.sl["EUS"]].mean(axis=1))
        return turn, speed, bite, push


class AnatomicalPolicy(Policy):
    """Lets an :class:`AnatomicalBrain` drive the animals of a simulation.

    The network runs on its own clock (``brain.tick_s``): an engine step longer
    than that runs several network ticks on the same sensory input, a shorter
    one holds the last command until the next tick is due.
    """

    REST_FRAC = 0.12
    MAX_SUBSTEPS = 16

    def __init__(self, brain: AnatomicalBrain, noise_seed: int = 0):
        self.brain, self.params = brain, SenseParams()
        self.rng = np.random.default_rng([int(noise_seed), brain.seed, 7919])
        self.memory: PlaceMemory | None = None
        self.r = None
        self.last_speed = self.last_turn = None
        self.last_obs = None
        self.turn = self.speed = self.bite = self.push = None
        self._clock = np.inf

    @staticmethod
    def from_config(cfg, trial_index: int = 0) -> "AnatomicalPolicy":
        bp = cfg.brain
        brain = AnatomicalBrain(load_spec(bp.anatomy), seed=bp.anatomy_seed,
                                strengths=bp.strengths, receptors=bp.receptors)
        return AnatomicalPolicy(brain, noise_seed=int(cfg.seed) + int(trial_index))

    def _ensure(self, n: int) -> None:
        if self.r is None:
            self.r = self.brain.initial_state(n)
            self.memory = PlaceMemory(n)
            self.last_speed, self.last_turn = np.zeros(n), np.zeros(n)
            self.turn, self.speed = np.zeros(n), np.zeros(n)
            self.bite, self.push = np.zeros(n, bool), np.zeros(n)
            return
        have = len(self.r)
        if n > have:                              # the roster grew mid-run
            pad = n - have
            self.r = np.vstack([self.r, self.brain.initial_state(pad)])
            self.memory.resize(n)
            for name in ("last_speed", "last_turn", "turn", "speed", "push"):
                setattr(self, name, np.concatenate([getattr(self, name), np.zeros(pad)]))
            self.bite = np.concatenate([self.bite, np.zeros(pad, bool)])

    def decide(self, sim, elapsed_s: float, dt: float):
        n = sim.n
        self._ensure(n)
        obs, perc = sense(sim, elapsed_s, self.memory, self.params, self.last_speed, self.last_turn)
        self.last_obs = obs
        # the first call ticks at once; after that the clock runs with the engine
        self._clock = self._clock + dt if np.isfinite(self._clock) else self.brain.tick_s
        ticks = int(self._clock / self.brain.tick_s + 1e-9)
        if ticks > self.MAX_SUBSTEPS:             # a very coarse engine step: do not try to catch up
            ticks, self._clock = self.MAX_SUBSTEPS, self.MAX_SUBSTEPS * self.brain.tick_s
        bit = np.zeros(n, bool)
        for _ in range(ticks):
            self.r = self.brain.step(self.r, obs, self.rng)
            self.turn, self.speed, b, self.push = self.brain.commands(self.r)
            bit |= b
        if ticks:
            self._clock -= ticks * self.brain.tick_s
            self.bite = bit
        heading = sim.H + self.turn * dt
        self.last_speed, self.last_turn = self.speed, self.turn / self.brain.max_turn
        perc.update(speed_frac=np.where(sim.alive, self.speed, 0.0), resting=self.speed < self.REST_FRAC,
                    drives={}, want_chew=np.zeros(n, bool),
                    bite=(self.bite & sim.alive) if ticks else np.zeros(n, bool))
        return np.stack([np.cos(heading), np.sin(heading)], axis=1), perc

    def frame_extra(self, n: int) -> dict:
        """Activity for the live view and the run record, under the names the
        learned brain uses: every unit, the sense organs, and the senses."""
        if self.r is None or self.last_obs is None:
            return {}
        return {
            "brain_rnn": self.r[:n].astype(np.float32),
            "brain_sensory": self.r[:n][:, self.brain.sense_units].astype(np.float32),
            "brain_obs": np.asarray(self.last_obs[:n], np.float32),
            "brain_turn": self.last_turn[:n].copy(),
            "brain_speed": self.speed[:n].copy(),
        }


# --------------------------------------------------------------------------- #
# Recording a run at the grain of the network
# --------------------------------------------------------------------------- #
class _Count:
    def __init__(self):
        self.kinds = {}

    def record(self, t, kind, *a, **k):
        self.kinds[kind] = self.kinds.get(kind, 0) + 1


def record(cfg, seconds: float, trial_index: int = 0) -> dict:
    """Run one trial with the config's anatomical brain and keep everything
    at every engine step: where each animal was, what its muscles were told,
    and the activity of every unit.

    A run record (:mod:`fnt.abma.core.record`) samples every few minutes,
    which is right for a ten-day trial and useless for watching a network. This
    is for short runs: minutes, at the engine's own step.
    """
    from .neural import make_policy
    from .simulation import Simulation

    pol = make_policy(cfg)
    if not isinstance(pol, AnatomicalPolicy):
        raise ValueError("this config has no anatomical brain (set brain.anatomy)")
    sim = Simulation(cfg, trial_index, policy=pol)
    dt = float(cfg.dt)
    T = max(1, int(round(seconds / dt)))
    n, N = sim.n, pol.brain.n
    out = {"t": np.zeros(T), "xy": np.zeros((T, n, 2), np.float32), "heading": np.zeros((T, n), np.float32),
           "speed": np.zeros((T, n), np.float32), "turn": np.zeros((T, n), np.float32),
           "push": np.zeros((T, n), np.float32), "rates": np.zeros((T, n, N), np.uint8)}
    events = _Count()
    for k in range(T):
        sim.step(k * dt, dt, events=events)
        out["t"][k] = (k + 1) * dt
        out["xy"][k], out["heading"][k] = sim.P[:n], sim.H[:n]
        out["speed"][k], out["turn"][k], out["push"][k] = pol.speed, pol.turn, pol.push
        out["rates"][k] = np.clip(pol.r * 255.0, 0, 255).astype(np.uint8)
    out["meta"] = {
        "anatomy": cfg.brain.anatomy, "anatomy_seed": cfg.brain.anatomy_seed,
        "strengths": dict(cfg.brain.strengths), "receptors": dict(cfg.brain.receptors),
        "dt": dt, "seconds": T * dt, "arena": [cfg.arena.width, cfg.arena.height],
        "units": pol.brain.unit_names(), "areas": pol.brain.ids, "events": events.kinds,
        "unwired": pol.brain.unwired, "trained": False,
    }
    return out


def save_recording(rec: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(rec["meta"])),
                        **{k: v for k, v in rec.items() if k != "meta"})
    return path


def load_recording(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta"}
        out["meta"] = json.loads(str(z["meta"]))
    return out


def area_table(rec: dict, spec: dict) -> list[dict]:
    """Mean, spread and peak firing of every area over a recording (percent
    of maximum, pooled over animals and the area's units)."""
    rates = rec["rates"].astype(np.float32) / 255.0 * 100.0
    rows, k = [], 0
    for r in spec["regions"]:
        blk = rates[:, :, k:k + r["n"]].mean(axis=2)      # (T, animals): the area's mean
        k += r["n"]
        rows.append({"area": r["id"], "name": r["name"], "class": r["class"], "units": r["n"],
                     "mean_pct": float(blk.mean()), "sd_pct": float(blk.std()), "peak_pct": float(blk.max())})
    return rows


def movement_summary(rec: dict) -> dict:
    """Path length and time moving, per animal averaged."""
    xy, dt = rec["xy"].astype(float), rec["meta"]["dt"]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=2)
    w, h = rec["meta"]["arena"]
    cx = (xy[:, :, 0] > 0.25 * w) & (xy[:, :, 0] < 0.75 * w)
    cy = (xy[:, :, 1] > 0.25 * h) & (xy[:, :, 1] < 0.75 * h)
    net = np.unwrap(rec["heading"].astype(float), axis=0)
    return {"path_m": float(step.sum(axis=0).mean()),
            "moving_frac": float((step / dt > 0.01).mean()),
            "centre_frac": float((cx & cy).mean()),
            "turns_left_net": float(((net[-1] - net[0]) / (2 * np.pi)).mean()),
            "events": rec["meta"]["events"]}
