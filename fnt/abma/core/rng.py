"""Per-agent random streams — reproducibility that survives a changing roster.

Why not one generator
---------------------
A single ``np.random.Generator`` for the whole simulation makes every draw
depend on how many draws came before it. That is fine until an experiment does
any of the things ABMA exists to do:

  * a protocol event adds or removes an animal mid-run,
  * an intervention silences one agent's behaviour,
  * a study compares a lesioned arm against its paired control.

In all three cases the manipulated run consumes a different number of draws, so
*every other animal* gets different noise from that moment on. The contrast
then mixes the manipulation with a wholesale reseeding of the cohort, which is
exactly what ``seed_policy="paired"`` in :mod:`fnt.abma.core.study` is trying to
avoid.

Counter-based streams
---------------------
This module draws from a *counter-based* PRNG instead: a value is a pure
function of ``(root_seed, agent_uid, step, channel)``. Nothing is consumed, so

  * agent 5's noise at step 900 is identical whether or not agent 3 exists,
  * ablating an animal perturbs only that animal,
  * a run can be resumed or a single step re-derived without replaying history.

The mixer is SplitMix64 and normals come from Box-Muller, matching the
``splitmix64-box-muller-v1`` scheme used by other connectome-driven simulators;
it is fast, vectorises over the whole cohort in numpy, and has no state to
carry. It is not cryptographic and is not meant to be.

``uid`` is a *stable* agent identifier assigned once at spawn — unlike ``index``,
which is a row position that shifts when the roster changes.
"""
from __future__ import annotations

import numpy as np

_U = np.uint64
_GOLDEN = _U(0x9E3779B97F4A7C15)
_MIX1 = _U(0xBF58476D1CE4E5B9)
_MIX2 = _U(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = _U(30), _U(27), _U(31), _U(11)
#: 2**53, so a 53-bit mantissa lands in [0, 1) with uniform spacing
_TWO53 = float(1 << 53)

#: Named draw channels. Distinct channels are independent streams for the same
#: agent and step, so adding a new stochastic mechanism never disturbs an
#: existing one. Add to the end; never renumber (that would change every run).
CH_HEADING = 0        # correlated random-walk heading drift
CH_MARK = 1           # whether an animal lays a scent mark this step
CH_FIGHT = 2          # contest initiation
CH_FIGHT_OUTCOME = 3  # who wins a contest
CH_MATE = 4           # mating hazard
CH_OLFACTION = 5      # receptor / perceptual noise in the olfactory model
CH_SPARE = 6          # reserved


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised SplitMix64 finaliser. ``x`` is a uint64 array.

    Wrap-around on add and multiply *is* the algorithm, so the overflow
    warnings numpy raises for uint64 are silenced here rather than left to
    pollute every run's stderr.
    """
    with np.errstate(over="ignore"):
        z = x + _GOLDEN
        z = (z ^ (z >> _S30)) * _MIX1
        z = (z ^ (z >> _S27)) * _MIX2
        return z ^ (z >> _S31)


def _key(root: int, uids: np.ndarray, step: int, channel: int) -> np.ndarray:
    """Mix the four coordinates into one uint64 per agent.

    Each coordinate is folded through the mixer before being combined so that
    small, highly correlated inputs (uid 0..20, step 0..1e6, channel 0..6) do
    not produce correlated outputs — the failure mode of naively XOR-ing raw
    counters together.
    """
    u = np.ascontiguousarray(uids, dtype=np.uint64)
    # kept as 1-element arrays so every step stays on numpy's (silent,
    # wrapping) array path rather than its warning-emitting scalar path
    with np.errstate(over="ignore"):
        base = _splitmix64(np.array([root], np.uint64)
                           ^ _splitmix64(np.array([step], np.uint64)))
        base = _splitmix64(base ^ (np.array([channel], np.uint64) * _GOLDEN))
        return _splitmix64(base ^ _splitmix64(u * _GOLDEN))


class AgentRandom:
    """Counter-based random draws keyed by stable agent identity.

    Every method is a pure function of ``(root, uids, step, channel)``: calling
    it twice returns the same numbers, and the values an agent receives do not
    depend on which other agents exist.
    """

    #: bumped if the mixing scheme ever changes, so old runs are identifiable
    SCHEME = "splitmix64-box-muller-v1"

    def __init__(self, root: int):
        # numpy's uint64 arithmetic wraps, which is what the mixer wants; the
        # root is masked into range so a negative or oversized seed is legal.
        self.root = int(root) & 0xFFFFFFFFFFFFFFFF

    # ---- primitives ---------------------------------------------------- #
    def uniform(self, uids, step: int, channel: int) -> np.ndarray:
        """One draw in [0, 1) per uid."""
        bits = _key(self.root, uids, step, channel)
        return ((bits >> _S11).astype(np.float64) + 0.5) / _TWO53

    def normal(self, uids, step: int, channel: int,
               scale: float = 1.0) -> np.ndarray:
        """One standard-normal draw per uid, scaled by ``scale``.

        Box-Muller over two independent sub-streams of the same channel. The
        second sub-stream is offset by a large odd constant rather than by
        ``channel + 1`` so it can never collide with the next named channel.
        """
        u1 = self.uniform(uids, step, channel)
        u2 = self.uniform(uids, step, channel + 0x5F00)
        r = np.sqrt(-2.0 * np.log(u1))
        return scale * r * np.cos(2.0 * np.pi * u2)

    def normal_2d(self, uids, step: int, channel: int,
                  scale: float = 1.0) -> np.ndarray:
        """(n, 2) independent normals — Box-Muller's two outputs, both used."""
        u1 = self.uniform(uids, step, channel)
        u2 = self.uniform(uids, step, channel + 0x5F00)
        r = scale * np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    def vector(self, uids, step: int, channel: int, dim: int) -> np.ndarray:
        """(n, dim) independent standard normals — one signature per agent.

        Used for fixed per-agent quantities (an odour signature, a receptor
        sensitivity profile) where ``step`` is a constant tag rather than time.
        """
        cols = [self.normal(uids, step, channel + 0x100 * (d + 1))
                for d in range(dim)]
        return np.stack(cols, axis=1) if cols else np.zeros((len(uids), 0))

    def pair_uniform(self, uids_a, uids_b, step: int,
                     channel: int) -> np.ndarray:
        """One draw in [0, 1) per (a, b) pair — for dyadic events.

        A contest or a mating belongs to a *pair*, not to either animal, so it
        needs its own stream. The two uids are mixed asymmetrically, so callers
        must fix an order (ABMA always passes the lower row index first); that
        keeps ``(i, j)`` and ``(j, i)`` from being the same draw by accident
        while still being reproducible.
        """
        a = np.ascontiguousarray(uids_a, dtype=np.uint64)
        b = np.ascontiguousarray(uids_b, dtype=np.uint64)
        with np.errstate(over="ignore"):
            mixed = _splitmix64(a * _GOLDEN) ^ _splitmix64((b + _U(1)) * _MIX1)
        return self.uniform(mixed, step, channel)

    # ---- convenience --------------------------------------------------- #
    def spawn_seed(self, uid: int, tag: int = 0) -> int:
        """A plain integer seed derived for one agent (for a nested RNG)."""
        return int(_key(self.root, np.array([uid]), tag, CH_SPARE)[0])
