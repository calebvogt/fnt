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
them). The vomeronasal organ gets the ``vno_*`` channels: nothing unless the
nose is on a urine spot or on another animal, and then that spot's or animal's
MUP barcode.

*Out.* The units of class ``output`` are muscles. ``TURN`` is a left and a
right unit, read as their difference; ``SPEED`` and ``BITE`` are read as the
mean of their units. Urination is two muscles, as in the animal: the bladder
wall (``DET``) and the urethral sphincter (``EUS``, tonically active). The
engine obeys them when scent marking is on (``ScentParams``): urine leaves when
pressure in the bladder beats the closure of the urethra, and lands as a spot
that carries the animal's MUP barcode.

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
#: A spec may say that the cells of a sense organ die with one of the body's
#: senses (``"lesion": {"MOE": "smell_ability"}``): an animal with half its
#: sense of smell then has half of that organ's units, permanently silent.
#: These are the body's arrays those names refer to.
LESION_SOURCES = {"smell_ability": "smell", "vno_ability": "vno"}
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


def find_weights(name: str) -> str:
    """Trained weights by name (``~/ABMA_brains/anatomy/<name>.npz``) or path."""
    if name and os.path.isfile(name):
        return name
    for d in (user_dir(), package_dir()):
        p = os.path.join(d, name + ".npz")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"no trained anatomy weights named {name!r} in {user_dir()}")


def save_weights(path: str, arrays: dict, meta: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)
    return path


def load_weights(name_or_path: str) -> tuple[dict, dict]:
    """``(arrays, meta)``. Sensor columns are matched to the body's current
    sensor list by name, so weights raised before a sense was added still
    load (the new sense starts unconnected)."""
    with np.load(find_weights(name_or_path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        w = {k: z[k] for k in z.files if k != "meta"}
    had = meta.get("sensors") or list(SENSORS)
    if had != list(SENSORS):
        now = {s: i for i, s in enumerate(SENSORS)}
        gone = [s for s in had if s not in now]
        if gone:
            raise ValueError(f"these weights use senses the body no longer has: {gone}")
        Win = np.zeros((w["Win"].shape[0], len(SENSORS)), w["Win"].dtype)
        Win[:, [now[s] for s in had]] = w["Win"]
        w["Win"] = Win
        w.pop("c0w", None)                 # the critic no longer fits; it is rebuilt
    return w, meta


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

    def __init__(self, spec: dict, seed: int = 0, tick_s: float | None = None,
                 tau_s: float | None = None, g_proj: float = 2.6, g_local: float = 0.5,
                 bias: float = 0.25, noise: float = 0.15, k_inh: float = 3.0,
                 mod_gain: float = 0.5, max_turn_rad_s: float = 2.5,
                 strengths: dict | None = None, receptors: dict | None = None,
                 weights: dict | None = None):
        check_spec(spec)
        rng = np.random.default_rng(seed)
        # how often the network ticks and how fast a unit follows its drive: an
        # argument wins, then the spec's own ``dynamics``, then the defaults
        dyn = spec.get("dynamics", {})
        tick_s = float(dyn.get("tick_s", 0.25) if tick_s is None else tick_s)
        tau_s = float(dyn.get("tau_s", 0.6) if tau_s is None else tau_s)
        self.spec, self.seed, self.tick_s = spec, int(seed), tick_s
        self.alpha = min(1.0, tick_s / tau_s)
        #: update the areas in order from the senses inward within a tick,
        #: each area's own inhibition balanced inside the tick (see ``_sweep``)
        self.sweep = bool(dyn.get("sweep", False))
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
        #: for ``sweep``: the units of each level, and that level's own pooling
        self.levels = []
        if self.sweep:
            order = {rid: i for i, rid in enumerate(self.ids)}
            depth = {}
            for rid in self.ids:
                srcs = [p[0] for p in spec["projections"] if p[1] == rid and order[p[0]] < order[rid]]
                depth[rid] = 1 + max([depth[a] for a in srcs], default=-1)
            for d in sorted(set(depth.values())):
                idx = np.concatenate([np.arange(self.sl[rid].start, self.sl[rid].stop)
                                      for rid in self.ids if depth[rid] == d])
                self.levels.append({"idx": idx, "pool": self.pool[np.ix_(idx, idx)],
                                    "areas": [rid for rid in self.ids if depth[rid] == d]})

        strengths = dict(strengths or {})
        known = {f"{p[0]}>{p[1]}" for p in spec["projections"]}
        bad = sorted(set(strengths) - known)
        if bad:
            raise ValueError(f"no such projection: {', '.join(bad)}")
        self.strengths = {k_: float(v) for k_, v in strengths.items()}

        W = np.zeros((self.n, self.n))
        #: the sign a cell-to-cell weight may have between areas (+1, -1), and
        #: 0 where the spec wires nothing; what training has to respect
        S = np.zeros((self.n, self.n))
        local = np.zeros((self.n, self.n), bool)
        mult = np.ones((self.n, self.n))
        for r in spec["regions"]:        # local wiring: mixed, weak, no self-connection
            s = self.sl[r["id"]]
            w = rng.standard_normal((r["n"], r["n"])) * g_local / np.sqrt(r["n"])
            np.fill_diagonal(w, 0.0)
            W[s, s] = w
            local[s, s] = ~np.eye(r["n"], dtype=bool)
        # An area's inputs share one budget: a projection's weight is its strength as a
        # share of all the strengths arriving at its target. Listing another input to an
        # area therefore redistributes that area's drive instead of adding to it, so a
        # spec with many projections does not saturate where one with few does not.
        budget = {}
        for a, b, sign, strength, *_ in spec["projections"]:
            budget[b] = budget.get(b, 0.0) + strength
        for a, b, sign, strength, *_ in spec["projections"]:
            sa, sb = self.sl[a], self.sl[b]
            na, nb = sa.stop - sa.start, sb.stop - sb.start
            w = np.abs(rng.standard_normal((nb, na))) * (rng.random((nb, na)) < 0.6)
            w *= g_proj * strength / budget[b] / np.sqrt(0.6 * na)
            sgn = np.ones(na)
            if sign == "-":
                sgn = -sgn
            elif sign != "+":            # mixed cell types: each source cell is one or the other
                sgn = np.where(rng.random(na) < 0.5, 1.0, -1.0)
            W[sb, sa] = w * sgn[None, :]
            S[sb, sa] = np.broadcast_to(sgn[None, :], (nb, na))
            mult[sb, sa] = self.strengths.get(f"{a}>{b}", 1.0)
        self.S, self.local_mask, self._mult = S, local, mult
        #: the weights before any strength edit; ``W`` is what runs
        self.W0 = W
        self.W = W * mult

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
        for rid, gains in (spec.get("entry_gain") or {}).items():
            if rid not in self.sl:
                raise ValueError(f"anatomy spec: entry_gain names an unknown area {rid}")
            for name, g in gains.items():      # every cell of the area, the same weight
                if name in col:
                    Win[self.sl[rid], col[name]] = float(g)
        self.Win = Win
        self.in_mask = Win != 0.0

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
        #: sense organs whose cells die with a sense of the body: (first unit,
        #: last unit + 1, the body's array, the sensor columns that organ takes)
        self.lesion = []
        for rid, source in (spec.get("lesion") or {}).items():
            if rid not in self.sl or source not in LESION_SOURCES:
                raise ValueError(f"anatomy spec: lesion {rid}: {source} is not an area and one of {sorted(LESION_SOURCES)}")
            cols = np.array([col[c] for c in spec["entry"].get(rid, []) if c in col], int)
            self.lesion.append((self.sl[rid].start, self.sl[rid].stop, LESION_SOURCES[source], cols))
        #: projections whose strength an animal's own experience changes
        #: (``spec["plasticity"]``): each animal carries extra weights on them,
        #: zero at birth, written while the gate modulators are high
        self.plastic = []
        signs = {f"{p[0]}>{p[1]}": p[2] for p in spec["projections"]}
        for p in spec.get("plasticity") or []:
            a, _, b = str(p.get("projection", "")).partition(">")
            if signs.get(f"{a}>{b}") != "+":
                raise ValueError(f"anatomy spec: plasticity {a}>{b}: not an excitatory projection of this spec")
            gates = []
            for mid in p.get("gate", []):
                m = next((m_ for m_ in spec["modulators"] if m_["id"] == mid), None)
                if m is None:
                    raise ValueError(f"anatomy spec: plasticity {a}>{b}: no modulator {mid}")
                if mid in self._body_col:
                    gates.append(("body", self._body_col[mid]))
                elif m.get("source"):
                    gates.append(("area", self.sl[m["source"]]))
                else:
                    raise ValueError(f"anatomy spec: plasticity {a}>{b}: modulator {mid} has no level")
            # which source cells are written on: all of them, or only those whose own channel
            # (the one a sense cell mainly answers to) starts with "pre_channels"
            width = self.sl[a].stop - self.sl[a].start
            mask = np.ones(width)
            prefix = p.get("pre_channels")
            if prefix:
                chans = spec["entry"].get(a, [])
                if len(chans) > width:
                    raise ValueError(f"anatomy spec: plasticity {a}>{b}: pre_channels needs a cell per channel "
                                     f"({a} has {width} cells for {len(chans)} channels)")
                mask = np.array([1.0 if j < len(chans) and chans[j].startswith(prefix) else 0.0
                                 for j in range(width)])
            self.plastic.append({"name": f"{a}>{b}", "post": self.sl[b], "pre": self.sl[a], "gates": gates,
                                 "pre_mask": mask,
                                 "rate": float(p.get("rate_per_s", 0.02)),
                                 "gate_threshold": float(p.get("gate_threshold", 0.4)),
                                 "pre_threshold": float(p.get("pre_threshold", 0.35)),
                                 "cap": float(p.get("cap", 1.0)),
                                 "decay": 1.0 / (86400.0 * float(p.get("decay_days", 30.0)))})
        self.sense_units = np.concatenate([np.arange(self.sl[r["id"]].start, self.sl[r["id"]].stop)
                                           for r in spec["regions"] if r["class"] == "sense"])
        #: every muscle unit, in spec order: what a learner treats as the action
        self.out_units = np.concatenate([np.arange(self.sl[r["id"]].start, self.sl[r["id"]].stop)
                                         for r in spec["regions"] if r["class"] == "output"])
        self.trained = weights is not None
        if weights is not None:
            W0, Win_t, b = (np.asarray(weights[k], float) for k in ("W", "Win", "bias"))
            if W0.shape != self.W0.shape or Win_t.shape != self.Win.shape or b.shape != self.bias.shape:
                raise ValueError("these trained weights were made for a different spec")
            if (W0[(S == 0) & ~local] != 0).any() or (W0 * S < 0).any():
                raise ValueError("these trained weights wire cells the spec does not, or flip a projection's sign")
            self.W0, self.Win, self.bias = W0, Win_t * self.in_mask, b
            self.W = W0 * mult

    def unit_names(self) -> list[str]:
        return [f"{r['id']} {j + 1}" for r in self.spec["regions"] for j in range(r["n"])]

    def area_means(self, r: np.ndarray) -> np.ndarray:
        """``(..., units)`` -> ``(..., areas)``."""
        return np.stack([r[..., self.sl[i]].mean(axis=-1) for i in self.ids], axis=-1)

    def initial_state(self, n: int) -> np.ndarray:
        return np.zeros((n, self.n))

    # ---- connections that experience changes ---------------------------- #
    def initial_fast(self, n: int) -> list:
        """The experience-written weights of ``n`` newborn animals: one
        ``(animals, target cells, source cells)`` array of zeros per plastic
        projection."""
        return [np.zeros((n, p["post"].stop - p["post"].start, p["pre"].stop - p["pre"].start))
                for p in self.plastic]

    @staticmethod
    def _above(rate, threshold):
        """How far a rate is above a threshold, as 0-1 of the room left."""
        return np.clip((rate - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)

    def _sends(self, p: dict, r: np.ndarray) -> np.ndarray:
        """What the source cells of a plastic projection send through their
        experience-written synapses: how far each is above rest, and nothing
        from a cell the projection is not written on."""
        return self._above(r[:, p["pre"]], p["pre_threshold"]) * p["pre_mask"]

    def gate(self, p: dict, r: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """How open a plastic projection is to change, per animal (0-1):
        the product of how far each gate modulator is above threshold."""
        g = np.ones(len(r))
        for kind, ref in p["gates"]:
            level = np.clip(obs[:, ref], 0.0, 1.0) if kind == "body" else r[:, ref].mean(axis=1)
            g = g * self._above(level, p["gate_threshold"])
        return g

    def learn(self, fast: list, r: np.ndarray, new: np.ndarray, obs: np.ndarray,
              gate_r: np.ndarray | None = None) -> None:
        """One tick of experience, written into ``fast`` in place. ``r`` is the
        activity the source cells had when they drove their targets; the gate
        modulators are read from ``gate_r`` (``r`` if not given)."""
        for p, Fp in zip(self.plastic, fast):
            src = self._sends(p, r)
            Fp += self.tick_s * (p["rate"] * self.gate(p, r if gate_r is None else gate_r, obs)[:, None, None]
                                 * new[:, p["post"], None] * src[:, None, :] - p["decay"] * Fp)
            np.maximum(Fp, 0.0, out=Fp)
            total = Fp.sum(axis=2, keepdims=True)                 # a target cell's budget
            Fp *= np.minimum(1.0, p["cap"] / np.maximum(total, 1e-9))

    def memory(self, r: np.ndarray, fast: list):
        """What the first plastic projection holds, per animal: ``(stored,
        match)``. ``stored`` is how much of its budget is written (0-1);
        ``match`` is how well what is arriving now fits what is written
        (0-1: 1 when every source cell that was written is firing fully)."""
        if not self.plastic:
            return np.zeros(len(r)), np.zeros(len(r))
        p, Fp = self.plastic[0], fast[0]
        src = self._sends(p, r)
        stored = Fp.sum(axis=2).mean(axis=1) / p["cap"]
        drive = np.einsum("abj,aj->ab", Fp, src).mean(axis=1) / p["cap"]
        return np.clip(stored, 0.0, 1.0), np.clip(drive / np.maximum(stored, 1e-6), 0.0, 1.0) * (stored > 1e-4)

    def step(self, r: np.ndarray, obs: np.ndarray, rng: np.random.Generator,
             noise: np.ndarray | None = None, pre: bool = False, alive: np.ndarray | None = None,
             fast: list | None = None):
        """One tick. Returns the new ``r`` (the old one is not modified).
        ``fast`` (from :meth:`initial_fast`) is each animal's experience-written
        weights: they add to the drive, and this tick is written into them.

        ``noise`` replaces the usual small noise in every cell (a learner
        passes noise on the muscle units only); ``pre=True`` also returns the
        drive every cell had, noise included. ``alive`` is ``(animals, units)``
        of 1 and 0: a dead cell stays at zero whatever drives it."""
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
        if self.sweep:
            return self._sweep(r, obs, gain, rng, noise, pre, alive, fast)
        x = r @ self.W.T + obs @ self.Win.T + self.bias - self.k_inh * (r @ self.pool.T)
        if fast:
            for p, Fp in zip(self.plastic, fast):
                x[:, p["post"]] += np.einsum("abj,aj->ab", Fp, self._sends(p, r))
        x = gain * x + (self.noise * rng.standard_normal(x.shape) if noise is None else noise)
        new = r + self.alpha * (np.tanh(np.maximum(x, 0.0)) - r)
        if alive is not None:
            new = new * alive
        if fast:
            self.learn(fast, r, new, obs)
        return (new, x) if pre else new

    BALANCE_STEPS = 14

    def _sweep(self, r, obs, gain, rng, noise, pre, alive, fast):
        """One tick with the areas taken in levels (``dynamics.sweep``)."""
        drive = obs @ self.Win.T + self.bias
        nz = self.noise * rng.standard_normal(r.shape) if noise is None else noise
        new, X = r.copy(), np.zeros_like(r)
        for lv in self.levels:
            idx = lv["idx"]
            e = new @ self.W[idx].T + drive[:, idx]
            if fast:
                for p, Fp in zip(self.plastic, fast):
                    if p["post"].start in idx:
                        loc = np.searchsorted(idx, np.arange(p["post"].start, p["post"].stop))
                        e[:, loc] += np.einsum("abj,aj->ab", Fp, self._sends(p, new))
            g, n_ = gain[:, idx], nz[:, idx]
            lo, hi = np.zeros_like(e), np.ones_like(e)
            for _ in range(self.BALANCE_STEPS):       # each area's mean rate, balanced against its own inhibition
                mid = 0.5 * (lo + hi)
                up = np.tanh(np.maximum(g * (e - self.k_inh * mid) + n_, 0.0)) @ lv["pool"].T > mid
                lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
            x = g * (e - self.k_inh * 0.5 * (lo + hi)) + n_
            val = r[:, idx] + self.alpha * (np.tanh(np.maximum(x, 0.0)) - r[:, idx])
            if alive is not None:
                val = val * alive[:, idx]
            new[:, idx], X[:, idx] = val, x
        if fast:
            self.learn(fast, new, new, obs, gate_r=r)
        return (new, X) if pre else new

    def cells_alive(self, uid, senses: dict) -> np.ndarray | None:
        """``(animals, units)`` of 1 and 0 for the spec's ``lesion`` table,
        or ``None`` when the spec has none.

        ``senses`` maps a body array's name to its per-animal values (0-1):
        the share of that organ's cells an animal still has. Which cells go
        first is fixed for each animal (drawn from its ``uid``), so a deeper
        lesion always includes a shallower one and regrowth undoes it in
        reverse order."""
        if not self.lesion:
            return None
        uid = np.asarray(uid)
        alive = np.ones((len(uid), self.n))
        for a, b, source, _ in self.lesion:
            keep = np.clip(np.asarray(senses[source], float), 0.0, 1.0)
            width = b - a
            for i, u in enumerate(uid):
                order = np.random.default_rng([int(u) & 0x7FFFFFFF, a, self.seed & 0x7FFFFFFF, 0x4C45]).permutation(width)
                dead = order[:int(round((1.0 - keep[i]) * width))]
                alive[i, a + dead] = 0.0
        return alive

    def whole_senses(self, obs: np.ndarray, senses: dict) -> np.ndarray:
        """The input as a surviving cell of a lesioned organ gets it.

        The engine scales every olfactory channel by ``smell_ability``, which
        is the right input for a brain with no sense organs of its own. Here
        the loss is already in the cells that are gone, so the ones that are
        left must get the odour undiminished, or the loss is counted twice."""
        if not self.lesion:
            return obs
        out = np.array(obs, float)
        for _, _, source, cols in self.lesion:
            if source == "smell" and len(cols):
                s = np.asarray(senses[source], float)
                out[:, cols] = out[:, cols] / np.where(s > 1e-6, s, 1.0)[:, None]
        return out

    def commands(self, r: np.ndarray):
        """Turn (rad/s), speed (0..1), bite (bool), and the two urination
        muscles as ``(bladder wall, sphincter)``, each 0..1."""
        t = r[:, self.sl["TURN"]]
        turn = self.max_turn * (t[:, 1] - t[:, 0])
        speed = r[:, self.sl["SPEED"]].mean(axis=1)
        bite = r[:, self.sl["BITE"]].mean(axis=1) > 0.5
        return turn, speed, bite, (r[:, self.sl["DET"]].mean(axis=1), r[:, self.sl["EUS"]].mean(axis=1))


class AnatomicalPolicy(Policy):
    """Lets an :class:`AnatomicalBrain` drive the animals of a simulation.

    The network runs on its own clock (``brain.tick_s``): an engine step longer
    than that runs several network ticks on the same sensory input, a shorter
    one holds the last command until the next tick is due.
    """

    REST_FRAC = 0.12
    MAX_SUBSTEPS = 16

    def __init__(self, brain: AnatomicalBrain, noise_seed: int = 0, train_std=None):
        #: while being trained: the spread of the noise on each muscle unit's
        #: drive (the inside of the network is then quiet), and ``u`` holds the
        #: drive those units actually had on the last tick
        self.train_std = None if train_std is None else np.asarray(train_std, float)
        self.u = None
        self.brain, self.params = brain, SenseParams()
        self.rng = np.random.default_rng([int(noise_seed), brain.seed, 7919])
        self.memory: PlaceMemory | None = None
        self.r = None
        #: (animals, units) of 1 and 0 where the spec ties a sense organ's
        #: cells to a sense of the body; rebuilt when a sense changes
        self.alive = None
        self._alive_key = None
        #: each animal's experience-written weights (``brain.initial_fast``),
        #: and what the first plastic projection holds: how much is stored
        #: and how well the present input matches it (see ``brain.memory``)
        self.fast = None
        self.stored = self.match = None
        self.last_speed = self.last_turn = None
        self.last_obs = None
        self.turn = self.speed = self.bite = self.wall = self.sphincter = None
        self._clock = np.inf

    @staticmethod
    def from_config(cfg, trial_index: int = 0) -> "AnatomicalPolicy":
        bp = cfg.brain
        weights, seed = None, bp.anatomy_seed
        if getattr(bp, "anatomy_weights", ""):
            weights, meta = load_weights(bp.anatomy_weights)
            if meta.get("anatomy") not in (bp.anatomy, os.path.splitext(os.path.basename(bp.anatomy))[0]):
                raise ValueError(f"{bp.anatomy_weights} was raised for {meta.get('anatomy')!r}, not {bp.anatomy!r}")
            seed = int(meta.get("wiring_seed", seed))        # mixed projections keep the signs they drew
        brain = AnatomicalBrain(load_spec(bp.anatomy), seed=seed, weights=weights,
                                strengths=bp.strengths, receptors=bp.receptors)
        return AnatomicalPolicy(brain, noise_seed=int(cfg.seed) + int(trial_index))

    def _ensure(self, n: int) -> None:
        if self.r is None:
            self.r = self.brain.initial_state(n)
            self.fast = self.brain.initial_fast(n)
            self.memory = PlaceMemory(n)
            self.last_speed, self.last_turn = np.zeros(n), np.zeros(n)
            self.turn, self.speed = np.zeros(n), np.zeros(n)
            self.bite, self.wall, self.sphincter = np.zeros(n, bool), np.zeros(n), np.zeros(n)
            return
        have = len(self.r)
        if n > have:                              # the roster grew mid-run
            pad = n - have
            self.r = np.vstack([self.r, self.brain.initial_state(pad)])
            self.fast = [np.concatenate([f, g]) for f, g in zip(self.fast, self.brain.initial_fast(pad))]
            self.memory.resize(n)
            for name in ("last_speed", "last_turn", "turn", "speed", "wall", "sphincter"):
                setattr(self, name, np.concatenate([getattr(self, name), np.zeros(pad)]))
            self.bite = np.concatenate([self.bite, np.zeros(pad, bool)])

    def decide(self, sim, elapsed_s: float, dt: float):
        n = sim.n
        self._ensure(n)
        obs, perc = sense(sim, elapsed_s, self.memory, self.params, self.last_speed, self.last_turn)
        if self.brain.lesion:
            senses = {"smell": sim.smell[:n], "vno": getattr(sim, "vno", np.ones(n))[:n]}
            key = (n, senses["smell"].tobytes(), senses["vno"].tobytes())
            if key != self._alive_key:               # methimazole took effect, or the roster grew
                self.alive, self._alive_key = self.brain.cells_alive(sim.uid[:n], senses), key
                for p, Fp in zip(self.brain.plastic, self.fast):   # a dead cell's synapses go with it
                    Fp *= self.alive[:, None, p["pre"]] * self.alive[:, p["post"], None]
            obs = self.brain.whole_senses(obs, senses).astype(np.float32)
            self.r = self.r * self.alive
        self.last_obs = obs
        # the first call ticks at once; after that the clock runs with the engine
        self._clock = self._clock + dt if np.isfinite(self._clock) else self.brain.tick_s
        ticks = int(self._clock / self.brain.tick_s + 1e-9)
        if ticks > self.MAX_SUBSTEPS:             # a very coarse engine step: do not try to catch up
            ticks, self._clock = self.MAX_SUBSTEPS, self.MAX_SUBSTEPS * self.brain.tick_s
        bit = np.zeros(n, bool)
        for _ in range(ticks):
            if self.train_std is None:
                self.r = self.brain.step(self.r, obs, self.rng, alive=self.alive, fast=self.fast)
            else:
                out = self.brain.out_units
                noise = np.zeros((n, self.brain.n))
                noise[:, out] = self.train_std * self.rng.standard_normal((n, len(out)))
                self.r, x = self.brain.step(self.r, obs, self.rng, noise=noise, pre=True, alive=self.alive,
                                            fast=self.fast)
                self.u = x[:, out].astype(np.float32)
            self.turn, self.speed, b, (self.wall, self.sphincter) = self.brain.commands(self.r)
            bit |= b
        if ticks:
            self._clock -= ticks * self.brain.tick_s
            self.bite = bit
            if self.brain.plastic:
                self.stored, self.match = self.brain.memory(self.r, self.fast)
        heading = sim.H + self.turn * dt
        self.last_speed, self.last_turn = self.speed, self.turn / self.brain.max_turn
        perc.update(speed_frac=np.where(sim.alive, self.speed, 0.0), resting=self.speed < self.REST_FRAC,
                    drives={}, want_chew=np.zeros(n, bool),
                    bite=(self.bite & sim.alive) if ticks else np.zeros(n, bool),
                    bladder_wall=self.wall, sphincter=self.sphincter)
        return np.stack([np.cos(heading), np.sin(heading)], axis=1), perc

    def partner_memory(self, n: int):
        """``(stored, match)`` for the first ``n`` animals, or ``None`` when
        this brain has no plastic projection: what the engine reads when an
        attachment is kept in the brain (``SocialParams.bond_in_brain``)."""
        if not self.brain.plastic or self.stored is None or len(self.stored) < n:
            return None
        return self.stored[:n], self.match[:n]

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
           "bladder": np.zeros((T, n), np.float32), "contact": np.zeros((T, n), np.float32),
           "rates": np.zeros((T, n, N), np.uint8)}
    vno = SENSORS.index("vno_contact")
    voids = []
    events = _Count()
    for k in range(T):
        sim.step(k * dt, dt, events=events)
        out["t"][k] = (k + 1) * dt
        out["xy"][k], out["heading"][k] = sim.P[:n], sim.H[:n]
        out["speed"][k], out["turn"][k], out["bladder"][k] = pol.speed, pol.turn, sim.bladder[:n] / 100.0
        out["contact"][k] = pol.last_obs[:n, vno]
        voids += [(k, a, x, y, v) for a, x, y, v in sim.last_voids]
        out["rates"][k] = np.clip(pol.r * 255.0, 0, 255).astype(np.uint8)
    out["voids"] = np.array(voids, float).reshape(-1, 5)     # step, animal, x, y, fraction of a bladder
    out["meta"] = {
        "anatomy": cfg.brain.anatomy, "anatomy_seed": cfg.brain.anatomy_seed,
        "strengths": dict(cfg.brain.strengths), "receptors": dict(cfg.brain.receptors),
        "dt": dt, "seconds": T * dt, "arena": [cfg.arena.width, cfg.arena.height],
        "name": cfg.name, "units": pol.brain.unit_names(), "areas": pol.brain.ids, "events": events.kinds,
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
