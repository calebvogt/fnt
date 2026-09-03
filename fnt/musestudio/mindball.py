"""Mindball: push a ball into the opponent's end by being the calmer one.

THE ORIGINAL
------------
Interactive Productline's Mindball, a fixture of science museums. Two players
face each other with a ball on a rail between them; the ball rolls away from
whoever is more relaxed. The metric is relaxation — alpha and theta power — not
concentration, which is what makes it such a good demonstration: visibly trying
to win makes you lose, and the crowd can see it happen.

WHAT DRIVES THE BALL HERE
-------------------------
    velocity  =  gain x (my_calm - their_calm)

A DIFFERENCE, not an absolute. That choice does most of the work:

* It cancels anything common to both traces — drift, temperature, the slow
  session-long wander that made a fixed threshold uncomparable in the flight
  pipeline. Only the gap matters.
* It makes the game self-balancing. Two equally calm players sit at a standstill
  regardless of their absolute levels, so the game does not need calibration to
  agree between people.

Calm comes from :mod:`fnt.musestudio.meditation` — the same theta+alpha
conjunction, with the same delta veto for sleep onset. Dozing off must not be a
winning strategy.

PLAYING WITH ONE HEADBAND
-------------------------
Two opponent types, because they answer different questions:

* ``GhostOpponent`` replays a recorded calm trace from a previous session — you
  against your past self. Authentic, and the honest way to see whether you have
  improved. Its problem is length: a match lasts until someone scores, so it can
  outlast its recording. Rather than freeze or end the match arbitrarily, the
  trace WRAPS, and every wrap is counted and reported, because a match that
  looped four times is not the same evidence as one that did not.
* ``SyntheticOpponent`` samples from the statistical shape of your past
  sessions rather than replaying one. It never runs out and its difficulty is a
  single number, which makes it the better practice partner — but it is not a
  record of anything, and the report says so.
"""

from dataclasses import dataclass, field

import numpy as np

TRACK_HALF = 1.0          # ball position runs -1 (my end) .. +1 (their end)
# Tuned for a match of roughly a minute, which is what the original runs.
# Swept against a constant calm gap: at the first guess of 0.55 a 0.25 gap
# scored in FOUR seconds, which is not a game — the ball simply left. 0.012
# gives ~70 s at a 0.15 gap and ~45 s at 0.25, so a lead has to be held rather
# than momentarily achieved, which is the whole point of the original.
DEFAULT_GAIN = 0.012      # track-units per second per unit of calm difference
DEFAULT_FRICTION = 0.82   # velocity retained per second; keeps it from skating
WIN_MARGIN = 0.98         # |position| at which the point is scored


@dataclass
class BallState:
    t: float = 0.0
    position: float = 0.0      # -1 = my end (I am losing), +1 = their end
    velocity: float = 0.0
    my_calm: float = 0.0
    their_calm: float = 0.0
    winner: str = ""           # "" | "player" | "opponent"
    wraps: int = 0             # times the ghost trace looped

    def as_row(self):
        return {"t": round(self.t, 3), "position": round(self.position, 4),
                "velocity": round(self.velocity, 4),
                "my_calm": round(self.my_calm, 4),
                "their_calm": round(self.their_calm, 4),
                "winner": self.winner, "wraps": self.wraps}


class GhostOpponent:
    """Replays a recorded calm trace. Wraps when it runs out, and counts it."""

    kind = "ghost"

    def __init__(self, trace, hz=2.0, label="past self"):
        self.trace = np.asarray(list(trace), dtype=float)
        self.hz = float(hz)
        self.label = label
        self.wraps = 0
        self._t = 0.0

    def duration(self):
        return len(self.trace) / self.hz if self.hz else 0.0

    def calm(self, dt):
        """Sample the trace by ELAPSED TIME, not by call count.

        Advancing one sample per call ties playback speed to whatever rate the
        caller happens to run at. When the physics moved from 2 Hz to 30 Hz the
        ghost silently began playing 15x too fast: a 154-second recording was
        consumed in about ten seconds and looped seven times inside a 79-second
        match, so the "opponent" was a brief fragment on repeat.
        """
        if len(self.trace) == 0:
            return 0.0
        self._t += max(0.0, float(dt))
        dur = self.duration()
        # Small tolerance: floating-point accumulation of dt lands a hair past
        # the end after exactly one pass, which would report a spurious wrap —
        # and the wrap count is shown to the player as a measure of how much of
        # the match was against real recorded behaviour.
        if dur > 0 and self._t >= dur + 1e-6:
            # A match runs until someone scores, so it can outlast the
            # recording. Wrapping keeps the opponent alive without inventing
            # behaviour; the count is surfaced because a looped opponent is
            # weaker evidence than one that played through.
            self._t -= dur
            self.wraps += 1
        i = int(self._t * self.hz)
        return float(self.trace[min(i, len(self.trace) - 1)])


class SyntheticOpponent:
    """Samples from the shape of past sessions. Never runs out; tunable."""

    kind = "synthetic"

    def __init__(self, mean=0.35, spread=0.18, drift_s=12.0, difficulty=1.0,
                 seed=7, label="practice partner"):
        self.label = label
        self.mean = float(mean) * float(difficulty)
        self.spread = float(spread)
        self.wraps = 0
        self._rng = np.random.default_rng(seed)
        self._v = self.mean
        # Ornstein-Uhlenbeck-ish drift: calm wanders on a timescale of seconds
        # rather than jittering per sample, which is how a real opponent behaves.
        self._tau = max(1e-3, float(drift_s))

    def calm(self, dt):
        pull = (self.mean - self._v) * (dt / self._tau)
        noise = self._rng.normal(0.0, self.spread) * np.sqrt(dt / self._tau)
        self._v = float(np.clip(self._v + pull + noise, 0.0, 1.0))
        return self._v


class MindballGame:
    """Pure game state. Feed calm scores, get a ball position."""

    def __init__(self, opponent, gain=DEFAULT_GAIN, friction=DEFAULT_FRICTION,
                 win_margin=WIN_MARGIN, time_limit_s=240.0):
        self.opponent = opponent
        self.gain = float(gain)
        self.friction = float(friction)
        self.win_margin = float(win_margin)
        # Two evenly matched players can hold the ball near centre indefinitely.
        # A draw is a legitimate outcome and must be reachable, or the session
        # never ends.
        self.time_limit_s = float(time_limit_s)
        self.state = BallState()
        self.history = []

    def reset(self):
        self.state = BallState()
        self.history = []

    def step(self, my_calm, dt, t=None):
        s = self.state
        if s.winner:
            return s
        s.t = t if t is not None else s.t + dt
        s.my_calm = float(np.clip(my_calm, 0.0, 1.0))
        s.their_calm = float(np.clip(self.opponent.calm(dt), 0.0, 1.0))

        # The ball accelerates toward whoever is LESS calm.
        accel = self.gain * (s.my_calm - s.their_calm)
        s.velocity = s.velocity * (self.friction ** dt) + accel * dt
        s.position = float(np.clip(s.position + s.velocity * dt,
                                   -TRACK_HALF, TRACK_HALF))
        s.wraps = getattr(self.opponent, "wraps", 0)

        if s.position >= self.win_margin:
            s.winner = "player"
        elif s.position <= -self.win_margin:
            s.winner = "opponent"
        elif self.time_limit_s and s.t >= self.time_limit_s:
            s.winner = "draw"
        self.history.append(s.as_row())
        return s

    def summary(self):
        if not self.history:
            return {}
        h = self.history
        mine = np.array([r["my_calm"] for r in h])
        theirs = np.array([r["their_calm"] for r in h])
        return {
            "duration_s": round(h[-1]["t"] - h[0]["t"], 1),
            "winner": h[-1]["winner"] or "none",
            "my_calm_mean": round(float(mine.mean()), 3),
            "their_calm_mean": round(float(theirs.mean()), 3),
            "lead_fraction": round(float((mine > theirs).mean()), 3),
            "opponent": getattr(self.opponent, "label", "?"),
            "opponent_kind": getattr(self.opponent, "kind", "?"),
            "ghost_wraps": getattr(self.opponent, "wraps", 0),
        }
