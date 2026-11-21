"""Counterfactual twins: the same life, lived twice, with one thing changed.

The question
------------
An experiment compares groups. But the most direct causal question you can ask
of an animal is not about a group: *what would this one have done if, at that
moment, it had been a little less hungry?* No real animal can be asked. A
simulated one can — provided the two versions of its life differ in exactly one
respect and nothing else, including luck.

That proviso is the whole difficulty, and ABMA's random numbers were rebuilt to
meet it (:mod:`fnt.abma.core.rng`): every draw is a pure function of
``(seed, animal, step, channel)``, so a copy of a simulation, stepped forward,
retraces the original exactly. Change one thing in the copy and every
difference that follows is a consequence of that change. Nothing is re-rolled.

What this module does
---------------------
:func:`run_twin` runs a world up to a chosen moment, :func:`fork` copies it,
applies one or more :class:`Tweak` to the copy, and steps both forward side by
side. The result records where each animal went in each life, how its body
fared, and what happened to it — and how far the two lives drifted apart, for
the animal that was changed and for the others, who were changed only by
living alongside it.

Tweaks
------
``state``   nudge a condition bar now: hunger, thirst, energy, stress, health,
            bladder.
``trait``   change a standing attribute: smell_ability, aggression, ...
``unit``    drive, clamp or silence one unit of a learned brain for a while —
            "a neuron gets really active for a second".
``sense``   scale a sensor, or a whole sense, for a while (0 removes it).

The first two work on any animal. The last two need a learned brain
(``cfg.brain.enabled``), because a rule-based animal has no units to reach.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from .biology import TRAIT_TO_ARRAY
from .config import ExperimentConfig
from .senses import GROUPS, SENSORS
from .simulation import Simulation

#: condition bars a ``state`` tweak may touch
STATE_BARS = ("hunger", "thirst", "energy", "stress", "health", "bladder")
#: what is sampled from each life, per animal, per sample
TRACKED = ("energy", "hunger", "thirst", "stress", "health")


@dataclass
class Tweak:
    """One change made to the twin at the moment of the fork."""
    target: str = "0"          # row index, sexid, group label, or "all"
    kind: str = "state"        # state | trait | unit | sense
    name: str = "hunger"       # bar / trait / "rnn:12" / sensor or sense name
    op: str = "add"            # set | add | scale
    value: float = -30.0
    duration_s: float = 0.0    # unit / sense: how long it lasts (0 = for good)
    label: str = ""

    def describe(self) -> str:
        if self.label:
            return self.label
        verb = {"set": "set to", "add": "changed by", "scale": "scaled by"}
        what = {"state": self.name, "trait": self.name,
                "unit": f"unit {self.name}", "sense": f"{self.name} sense"}
        tail = (f" for {self.duration_s:g} s"
                if self.kind in ("unit", "sense") and self.duration_s else "")
        return (f"{what.get(self.kind, self.name)} {verb.get(self.op, self.op)}"
                f" {self.value:+g}{tail}" if self.op == "add" else
                f"{what.get(self.kind, self.name)} {verb.get(self.op, self.op)}"
                f" {self.value:g}{tail}")


def parse_tweak(text: str) -> Tweak:
    """``"hunger-30@0"``, ``"smell_ability=0@males"``, ``"rnn:12=1@0~10"``.

    Grammar: ``<name><op><value>[@target][~seconds]`` where ``op`` is ``+`` or
    ``-`` (add), ``=`` (set) or ``*`` (scale). The kind is inferred from the
    name: a condition bar, a trait, ``rnn:N`` / ``sensory:N`` for a unit, or a
    sensor / sense name.
    """
    s = text.strip()
    dur = 0.0
    if "~" in s:
        s, d = s.rsplit("~", 1)
        dur = float(d)
    target = "0"
    if "@" in s:
        s, target = s.rsplit("@", 1)
    for sym, op in (("=", "set"), ("*", "scale"), ("+", "add"), ("-", "add")):
        k = s.find(sym, 1)
        if k > 0:
            name, raw = s[:k].strip(), s[k + 1:].strip()
            value = float(raw) * (-1.0 if sym == "-" else 1.0)
            break
    else:
        raise ValueError(f"cannot read tweak {text!r}; expected e.g. "
                         f"'hunger-30@0' or 'smell_ability=0@males'")
    if name in STATE_BARS:
        kind = "state"
    elif name in TRAIT_TO_ARRAY:
        kind = "trait"
    elif name.split(":")[0] in ("rnn", "sensory"):
        kind = "unit"
    elif name in SENSORS or name in GROUPS:
        kind = "sense"
    else:
        raise ValueError(
            f"{name!r} is not a condition bar {STATE_BARS}, a trait, a unit "
            f"('rnn:12'), a sensor or a sense {tuple(GROUPS)}")
    return Tweak(target=target.strip(), kind=kind, name=name, op=op,
                 value=value, duration_s=dur)


def fork(sim: Simulation) -> Simulation:
    """An independent copy of a simulation that will retrace it exactly."""
    return copy.deepcopy(sim)


def rows_of(sim: Simulation, target) -> np.ndarray:
    """Agent rows a target names: an index, a sexid, a group, or ``all``."""
    t = str(target).strip()
    if t.lstrip("-").isdigit():
        i = int(t)
        if not (0 <= i < sim.n):
            raise ValueError(f"no animal at row {i} (cohort of {sim.n})")
        return np.array([i], int)
    rows = np.array([a.index for a in sim.agents
                     if sim._match_target(a, t)], int)
    if not len(rows):
        raise ValueError(f"tweak target {target!r} matches no animal")
    return rows


def _op(cur, op: str, value: float):
    if op == "set":
        return np.full_like(cur, value, dtype=float)
    if op == "scale":
        return cur * value
    return cur + value


def apply_tweak(sim: Simulation, tw: Tweak, elapsed_s: float) -> list:
    """Apply one tweak to ``sim`` now. Returns timed things to undo later,
    as ``(end_s, undo)`` pairs."""
    rows = rows_of(sim, tw.target)
    undo = []
    if tw.kind == "state":
        if tw.name == "hunger":
            # hunger is the energy the animal is missing; move the store
            sim.energy[rows] = np.clip(
                100.0 - _op(sim.hunger[rows], tw.op, tw.value), 0, 100)
            sim.hunger[rows] = 100.0 - sim.energy[rows]
        else:
            arr = getattr(sim, tw.name)
            arr[rows] = np.clip(_op(arr[rows], tw.op, tw.value), 0, 100)
            if tw.name == "energy":
                sim.hunger[rows] = 100.0 - sim.energy[rows]
    elif tw.kind == "trait":
        arr = getattr(sim, TRAIT_TO_ARRAY[tw.name])
        arr[rows] = _op(arr[rows], tw.op, tw.value)
        for i in rows:
            setattr(sim.agents[i].traits, tw.name, float(arr[i]))
        sim.mark_biology_dirty()
    elif tw.kind in ("unit", "sense"):
        pol = sim.policy
        if not hasattr(pol, "brain"):
            raise ValueError(
                f"a '{tw.kind}' tweak needs a learned brain; this run is "
                f"rule-based (set brain.enabled, or use --brain)")
        end = elapsed_s + tw.duration_s if tw.duration_s > 0 else np.inf
        if tw.kind == "unit":
            layer, unit = tw.name.split(":")
            entry = (layer, int(unit), tw.op, float(tw.value), rows)
            pol.stim.append(entry)
            undo.append((end, lambda p=pol, e=entry: p.stim.remove(e)))
        else:
            cols = [SENSORS.index(s) for s in GROUPS.get(tw.name, [tw.name])]
            pol._ensure(sim.n)
            if pol.sensor_gain is None:
                pol.sensor_gain = np.ones((sim.n, len(SENSORS)), np.float32)
            before = pol.sensor_gain[np.ix_(rows, cols)].copy()
            pol.sensor_gain[np.ix_(rows, cols)] = _op(before, tw.op, tw.value)

            def restore(p=pol, r=rows, c=cols, b=before):
                p.sensor_gain[np.ix_(r, c)] = b
            undo.append((end, restore))
    else:
        raise ValueError(f"unknown tweak kind {tw.kind!r}")
    return undo


class _Tally:
    """Counts events per kind per animal; keeps the rows for the record."""

    def __init__(self):
        self.rows = []

    def record(self, elapsed_s, event, actor, target, x, y, value=1.0):
        self.rows.append((float(elapsed_s), str(event),
                          getattr(actor, "sexid", ""),
                          getattr(target, "sexid", "") if target else ""))

    def count(self, kind: str, since: float = 0.0) -> int:
        return sum(1 for t, e, _, _ in self.rows if e == kind and t >= since)


@dataclass
class TwinResult:
    """Two lives, sampled together."""
    t: np.ndarray                    # (K,) seconds since the fork
    xy_a: np.ndarray                 # (K, n, 2) the life as it was
    xy_b: np.ndarray                 # (K, n, 2) the life with the tweak
    bars_a: dict                     # name -> (K, n)
    bars_b: dict
    rows: np.ndarray                 # animals the tweak was applied to
    tweaks: list
    fork_at_s: float
    arena: tuple                     # (width, height)
    sexids: list
    events_a: dict = field(default_factory=dict)
    events_b: dict = field(default_factory=dict)
    units_a: np.ndarray | None = None    # (K, n, n_rnn) when a brain drove it
    units_b: np.ndarray | None = None

    #: how far apart two versions of an animal must be to count as different
    #: lives rather than the same one a hair's breadth off (about a body length)
    SPLIT_M = 0.10

    def divergence(self) -> np.ndarray:
        """(K, n) distance between each animal and its other self."""
        return np.linalg.norm(self.xy_a - self.xy_b, axis=2)

    def split_time(self) -> np.ndarray:
        """Seconds after the fork at which each animal's lives parted, or NaN
        if they never did."""
        d = self.divergence() > self.SPLIT_M
        first = np.argmax(d, axis=0)
        out = self.t[first].astype(float)
        out[~d.any(axis=0)] = np.nan
        return out

    def summary(self) -> dict:
        d = self.divergence()
        split = self.split_time()
        others = np.setdiff1d(np.arange(d.shape[1]), self.rows)
        tail = max(1, len(self.t) // 6)           # the last sixth of the run
        out = {
            "tweak": "; ".join(t.describe() for t in self.tweaks),
            "target": [self.sexids[i] for i in self.rows],
            "fork_at_h": round(self.fork_at_s / 3600.0, 2),
            "horizon_h": round(float(self.t[-1]) / 3600.0, 2),
            "target_split_after_s": [None if np.isnan(split[i])
                                     else round(float(split[i]), 1)
                                     for i in self.rows],
            "target_final_gap_m": round(float(d[-1, self.rows].mean()), 3),
            "target_mean_late_gap_m": round(
                float(d[-tail:, self.rows].mean()), 3),
            "target_max_gap_m": round(float(d[:, self.rows].max()), 3),
            "others_changed": int(np.sum(~np.isnan(split[others])))
            if len(others) else 0,
            "others_total": int(len(others)),
            "others_mean_late_gap_m": round(
                float(d[-tail:, others].mean()), 3) if len(others) else 0.0,
        }
        for bar in TRACKED:
            a = self.bars_a[bar][-1, self.rows].mean()
            b = self.bars_b[bar][-1, self.rows].mean()
            out[f"target_{bar}_final"] = [round(float(a), 1),
                                          round(float(b), 1)]
        for kind in sorted(set(self.events_a) | set(self.events_b)):
            out[f"n_{kind}"] = [int(self.events_a.get(kind, 0)),
                                int(self.events_b.get(kind, 0))]
        return out

    def sentence(self) -> str:
        """The result in plain words, for someone who has not read this file."""
        s = self.summary()
        who = " and ".join(s["target"])
        split = [x for x in s["target_split_after_s"] if x is not None]
        if not split:
            return (f"{who}: the change made no difference — both lives were "
                    f"identical for the whole {s['horizon_h']:g} h.")

        def span(sec):
            return (f"{sec:.0f} s" if sec < 120 else
                    f"{sec / 60:.0f} min" if sec < 7200 else
                    f"{sec / 3600:.1f} h")

        out = [f"{who}: the two lives parted {span(min(split))} after the "
               f"change. {s['horizon_h']:g} h on, the two versions are "
               f"{s['target_final_gap_m']:.2f} m apart"
               f" (as far as {s['target_max_gap_m']:.2f} m at one point)."]
        if s["others_total"]:
            out.append(f"{s['others_changed']} of {s['others_total']} other "
                       f"animals ended up living differently too, though "
                       f"nothing was done to them.")
        return " ".join(out)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        extra = {}
        if self.units_a is not None:
            extra = {"units_a": self.units_a.astype(np.float16),
                     "units_b": self.units_b.astype(np.float16)}
        np.savez_compressed(
            path, t=self.t, xy_a=self.xy_a.astype(np.float32),
            xy_b=self.xy_b.astype(np.float32), rows=self.rows,
            **{f"a_{k}": v.astype(np.float32) for k, v in self.bars_a.items()},
            **{f"b_{k}": v.astype(np.float32) for k, v in self.bars_b.items()},
            **extra,
            meta=np.array(json.dumps({
                "tweaks": [asdict(t) for t in self.tweaks],
                "fork_at_s": self.fork_at_s, "arena": list(self.arena),
                "sexids": self.sexids, "events_a": self.events_a,
                "events_b": self.events_b, "summary": self.summary()})))
        return path


def run_twin(cfg: ExperimentConfig, tweaks, fork_at_s: float,
             horizon_s: float, sample_s: float | None = None,
             trial_index: int = 0, progress_cb=None) -> TwinResult:
    """Live one life twice from ``fork_at_s``, the second with ``tweaks``.

    The world is run normally up to the fork (so both lives share a real
    history), then copied. ``sample_s`` is how often both are sampled; it
    defaults to about 600 samples across the horizon.
    """
    tweaks = [parse_tweak(t) if isinstance(t, str) else t
              for t in (tweaks if isinstance(tweaks, (list, tuple))
                        else [tweaks])]
    dt = float(cfg.dt)
    sim_a = Simulation(cfg, trial_index=trial_index)
    tally_a = _Tally()
    n_pre = int(round(fork_at_s / dt))
    n_post = max(1, int(round(horizon_s / dt)))
    total = n_pre + n_post
    for k in range(n_pre):
        sim_a.step(k * dt, dt, events=tally_a)
        if progress_cb is not None and k % 2000 == 0:
            progress_cb(k / total)
    t_fork = n_pre * dt

    sim_b = fork(sim_a)
    tally_b = _Tally()
    undo = []
    rows = []
    for tw in tweaks:
        undo += apply_tweak(sim_b, tw, t_fork)
        rows.append(rows_of(sim_b, tw.target))
    rows = np.unique(np.concatenate(rows))

    every = max(1, int(round((sample_s or horizon_s / 600.0) / dt)))
    K = n_post // every + 1
    n = sim_a.n
    t = np.zeros(K)
    xy_a = np.zeros((K, n, 2))
    xy_b = np.zeros((K, n, 2))
    bars_a = {b: np.zeros((K, n)) for b in TRACKED}
    bars_b = {b: np.zeros((K, n)) for b in TRACKED}
    neural = hasattr(sim_a.policy, "brain")
    units_a = units_b = None
    if neural:
        R = sim_a.policy.brain.n_rnn
        units_a = np.zeros((K, n, R), np.float32)
        units_b = np.zeros((K, n, R), np.float32)

    def take(i, since):
        t[i] = since
        xy_a[i], xy_b[i] = sim_a.P[:n], sim_b.P[:n]
        for b in TRACKED:
            bars_a[b][i] = getattr(sim_a, b)[:n]
            bars_b[b][i] = getattr(sim_b, b)[:n]
        if neural and sim_a.policy.h is not None:
            units_a[i] = sim_a.policy.h[:n]
            units_b[i] = sim_b.policy.h[:n]

    take(0, 0.0)
    i = 1
    for k in range(n_post):
        now = t_fork + k * dt
        if undo:
            due = [u for u in undo if now >= u[0]]
            for end, fn in due:
                fn()
            undo = [u for u in undo if now < u[0]]
        sim_a.step(now, dt, events=tally_a)
        sim_b.step(now, dt, events=tally_b)
        if (k + 1) % every == 0 and i < K:
            take(i, (k + 1) * dt)
            i += 1
        if progress_cb is not None and k % 2000 == 0:
            progress_cb((n_pre + k) / total)
    if progress_cb is not None:
        progress_cb(1.0)

    kinds = ("fight", "mating", "death")
    return TwinResult(
        t=t[:i], xy_a=xy_a[:i], xy_b=xy_b[:i],
        bars_a={b: v[:i] for b, v in bars_a.items()},
        bars_b={b: v[:i] for b, v in bars_b.items()},
        rows=rows, tweaks=tweaks, fork_at_s=t_fork,
        arena=(cfg.arena.width, cfg.arena.height),
        sexids=[a.sexid for a in sim_a.agents[:n]],
        events_a={k: tally_a.count(k, t_fork) for k in kinds},
        events_b={k: tally_b.count(k, t_fork) for k in kinds},
        units_a=None if units_a is None else units_a[:i],
        units_b=None if units_b is None else units_b[:i])
