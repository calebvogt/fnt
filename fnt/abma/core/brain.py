"""A small nervous system: senses in, movement out, nothing prescribed between.

The rule-based policy in :mod:`fnt.abma.core.policy` is a list of drives someone
wrote down — "when hungry, steer for food" — with a weight on each. It is
legible and it is fast, but every behaviour it shows is one it was told to
have. A :class:`Brain` is the alternative: a recurrent network that receives
what the body senses (:mod:`fnt.abma.core.senses`) and emits a turn and a
speed. What it does with a food smell when it is hungry is not written
anywhere; it is learned, by reinforcement, from what happens to the body
(:mod:`fnt.abma.learn`). So a behaviour seen in a neural animal is a result,
and an experiment on one — take away its nose, clamp a unit, make it a little
less hungry — is an experiment rather than a parameter change.

Architecture
------------
::

    senses ──► sensory layer ──► recurrent layer ──► turn, speed
                    ▲
       neuromodulators (hunger, thirst, stress, ...) set its gain

*Sensory layer.* ``tanh(W·senses)``, each unit multiplied by a gain between 0
and 2 that is a function of the interoceptive signals alone. This is the
"organs send neuromodulators" part: the body does not tell the brain what to
do, it changes how loudly each sensory pathway speaks.

*Recurrent layer.* A GRU. It is what gives the animal a present that depends on
its past — it can keep turning after the smell that started the turn has gone.

*Motor output.* Two numbers. Turn is squashed to ±``max_turn`` radians per
tick; speed to a 0–1 fraction of what the body can currently manage (the engine
still owns top speed, fatigue and the drag of long grass). Motor noise is added
before the squash and comes from the animal's own counter-based random stream,
so two runs of the same animal from the same moment are identical until
something is changed — which is what makes a counterfactual a counterfactual
(:mod:`fnt.abma.core.twin`).

Inference is plain numpy and needs nothing but this file: a brain is a few
hundred kilobytes of weights in an ``.npz`` and runs inside the ordinary
simulation loop. PyTorch is only needed to *train* one.

Every unit can be recorded, clamped, silenced or driven (see
:meth:`Brain.forward`'s ``stim``), per animal and per moment.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from .senses import SENSORS, MODULATORS, SenseParams

#: bumped when the meaning of the weights changes (not when sensors are added)
BRAIN_FORMAT = 1


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def package_dir() -> str:
    """Brains that ship with ABMA."""
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "brains")


def user_dir() -> str:
    """Brains the user has trained."""
    return os.path.join(os.path.expanduser("~"), "ABMA_brains")


def list_brains() -> list[tuple[str, str]]:
    """Every brain that can be asked for by name, as ``(name, path)``.

    The user's own come first, so a retrained brain shadows the shipped one of
    the same name.
    """
    out, seen = [], set()
    for d in (user_dir(), package_dir()):
        for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
            name = os.path.splitext(os.path.basename(p))[0]
            if name not in seen:
                seen.add(name)
                out.append((name, p))
    return out


def find_brain(name: str) -> str:
    """Resolve a brain name or path to a file, or raise with what exists."""
    if name and os.path.isfile(name):
        return name
    for n, p in list_brains():
        if n == name:
            return p
    have = ", ".join(n for n, _ in list_brains()) or "none"
    raise FileNotFoundError(
        f"no brain named {name!r} (available: {have}). Train one with "
        f"`python -m fnt.abma.learn`, or give a path to a .npz file.")


class Brain:
    """Weights plus the description of the body they were trained in."""

    def __init__(self, weights: dict, meta: dict):
        self.w = {k: np.asarray(v, np.float32) for k, v in weights.items()}
        self.meta = dict(meta)
        self.sensors: list[str] = list(self.meta.get("sensors", SENSORS))
        self.modulators: list[str] = list(
            self.meta.get("modulators", MODULATORS))
        self.tick_s = float(self.meta.get("tick_s", 2.0))
        self.max_turn = float(self.meta.get("max_turn", 1.5))
        self.n_in = len(self.sensors)
        self.n_sensory = int(self.w["W1"].shape[0])
        self.n_rnn = int(self.w["W_hh"].shape[1])
        self.senses = SenseParams.from_dict(self.meta.get("senses"))
        # Match this brain's inputs to the body's current sensor list BY NAME:
        # a sensor added to the body after this brain was trained is simply not
        # wired to it; a sensor it needs that the body no longer has is an
        # error rather than a silent zero.
        cur = {name: i for i, name in enumerate(SENSORS)}
        missing = [s for s in self.sensors if s not in cur]
        if missing:
            raise ValueError(
                f"this brain needs sensors the body no longer has: {missing}")
        self._take = np.array([cur[s] for s in self.sensors], int)
        own = {name: i for i, name in enumerate(self.sensors)}
        self._mod = np.array([own[m] for m in self.modulators], int)

    # ---- construction --------------------------------------------------- #
    @staticmethod
    def random(seed: int = 0, n_sensory: int = 64, n_rnn: int = 64,
               tick_s: float = 2.0, max_turn: float = 1.5,
               senses: SenseParams | None = None) -> "Brain":
        """An untrained brain. It moves, and that is all that can be said."""
        rng = np.random.default_rng(seed)
        n_in, n_mod = len(SENSORS), len(MODULATORS)

        def init(rows, cols, gain=1.0):
            return (rng.standard_normal((rows, cols))
                    * gain / np.sqrt(max(1, cols))).astype(np.float32)

        w = {
            "W1": init(n_sensory, n_in), "b1": np.zeros(n_sensory, np.float32),
            "Wg": init(n_sensory, n_mod, 0.1),
            "bg": np.zeros(n_sensory, np.float32),
            "W_ih": init(3 * n_rnn, n_sensory),
            "W_hh": init(3 * n_rnn, n_rnn),
            "b_ih": np.zeros(3 * n_rnn, np.float32),
            "b_hh": np.zeros(3 * n_rnn, np.float32),
            "Wo": init(2, n_rnn, 0.1), "bo": np.zeros(2, np.float32),
            "log_std": np.full(2, -0.5, np.float32),
        }
        meta = {
            "format": BRAIN_FORMAT, "name": "random", "sensors": list(SENSORS),
            "modulators": list(MODULATORS), "tick_s": tick_s,
            "max_turn": max_turn, "senses": (senses or SenseParams()).to_dict(),
            "trained": False,
        }
        return Brain(w, meta)

    # ---- running -------------------------------------------------------- #
    def initial_state(self, n: int) -> np.ndarray:
        return np.zeros((n, self.n_rnn), np.float32)

    def forward(self, obs: np.ndarray, h: np.ndarray, stim=None):
        """One tick for every animal.

        ``obs`` is ``(n, len(senses.SENSORS))``; ``h`` the recurrent state from
        the previous tick. Returns ``(motor, h_new, sensory)`` where ``motor``
        is ``(n, 2)`` pre-squash turn and speed, and ``sensory`` the gated
        sensory-layer activity (kept so it can be recorded).

        ``stim`` is a list of ``(layer, unit, op, value, rows)``: ``layer`` is
        ``"sensory"`` or ``"rnn"``, ``op`` is ``set`` / ``add`` / ``scale``,
        ``rows`` the animals it applies to. That is the whole of in-silico
        optogenetics: clamp a unit on, add drive to it, or scale it to zero.
        """
        w = self.w
        x = np.asarray(obs, np.float32)[:, self._take]
        gain = 2.0 * _sigmoid(x[:, self._mod] @ w["Wg"].T + w["bg"])
        s = np.tanh(x @ w["W1"].T + w["b1"]) * gain
        if stim:
            s = self._apply(s, stim, "sensory")
        R = self.n_rnn
        gi = s @ w["W_ih"].T + w["b_ih"]
        gh = h @ w["W_hh"].T + w["b_hh"]
        r = _sigmoid(gi[:, :R] + gh[:, :R])
        z = _sigmoid(gi[:, R:2 * R] + gh[:, R:2 * R])
        cand = np.tanh(gi[:, 2 * R:] + r * gh[:, 2 * R:])
        h_new = (1.0 - z) * cand + z * h
        if stim:
            h_new = self._apply(h_new, stim, "rnn")
        motor = h_new @ w["Wo"].T + w["bo"]
        return motor, h_new.astype(np.float32), s.astype(np.float32)

    @staticmethod
    def _apply(act: np.ndarray, stim, layer: str) -> np.ndarray:
        act = act.copy()
        for lay, unit, op, value, rows in stim:
            if lay != layer or not (0 <= unit < act.shape[1]):
                continue
            if op == "set":
                act[rows, unit] = value
            elif op == "add":
                act[rows, unit] += value
            elif op == "scale":
                act[rows, unit] *= value
        return act

    def act(self, motor: np.ndarray, noise=None, noise_scale: float = 1.0):
        """Pre-squash motor command -> ``(turn_rad, speed_frac, u)``.

        ``u`` is the noisy pre-squash command; it is what a learner needs to
        score the action it took.
        """
        u = motor
        if noise is not None and noise_scale > 0.0:
            u = motor + np.exp(self.w["log_std"]) * noise_scale * noise
        return self.squash(u) + (u,)

    def squash(self, u: np.ndarray):
        return np.tanh(u[:, 0]) * self.max_turn, _sigmoid(u[:, 1])

    # ---- persistence ---------------------------------------------------- #
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)),
                            **self.w)
        return path

    @staticmethod
    def load(path: str) -> "Brain":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            weights = {k: z[k] for k in z.files if k != "meta"}
        fmt = int(meta.get("format", 0))
        if fmt > BRAIN_FORMAT:
            raise ValueError(f"brain format v{fmt} is newer than this build "
                             f"(v{BRAIN_FORMAT})")
        b = Brain(weights, meta)
        b.path = path
        return b

    # ---- description ---------------------------------------------------- #
    @property
    def name(self) -> str:
        return str(self.meta.get("name", "brain"))

    def describe(self) -> str:
        m = self.meta
        bits = [f"{self.n_in} senses → {self.n_sensory} sensory → "
                f"{self.n_rnn} recurrent → turn, speed"]
        if m.get("trained"):
            steps = m.get("agent_steps")
            if steps:
                bits.append(f"trained on {steps / 1e6:.0f}M animal-ticks")
            if m.get("trained_on"):
                bits.append(str(m["trained_on"]))
        else:
            bits.append("untrained")
        return " · ".join(bits)
