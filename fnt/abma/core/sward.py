"""The grass layer — an environment the animals wear down and share.

Everything else in ABMA that an animal changes, it changes for itself: its own
energy, its own marks, its own memory of home. Grass is the first part of the
world that is genuinely *common*. An animal that walks a route flattens it, and
the flattened route is then cheaper and faster for **every** animal, including
ones that never contributed. A vole that stops to clip a path is paying, in
time it could have spent foraging, for a public good.

That is the whole point of modelling it. Runway networks are one of the most
conspicuous things voles actually do in the field, and they are not explicable
from an individual optimum — they are a shared structure that emerges because
movement and clipping happen to modify the substrate.

Three couplings
---------------
**Height slows movement.** Deep sward is hard going; a worn trail is faster than
open ground ever was. The speed multiplier is anchored at ``speed_ref_cm``
(factor 1) and runs from ``speed_max_factor`` on bare ground down to
``speed_min_factor`` at ``max_cm``.

**Height costs energy.** Pushing through grass is charged per metre travelled
and per centimetre of sward, on top of the flat-ground locomotion cost. So a
trail is not just faster, it is cheaper — which is what makes maintaining one
rational.

**Movement and chewing lower it.** Walking flattens grass a little
(``trample_cm_per_m``). Chewing flattens it several times faster
(``chew_rate_multiplier``) but the animal must stop to do it, for a bout whose
length scales with how tall the grass is. Clipping yields **no energy**: the
cost of trail maintenance is the foraging time it displaces, and nothing else.
That is a deliberate modelling choice — real voles do eat what they clip — made
so that trail-building and feeding can be told apart in a result.

Regrowth is what makes it a *maintenance* problem rather than a one-off
demolition. Without it the first week flattens the enclosure permanently and
there is no ongoing decision to study. With it, a trail decays unless it is
used, so route fidelity and trail structure reinforce each other — or don't.

Honesty
-------
Every rate here is a **free** parameter in the sense of
:data:`fnt.abma.core.project.SOURCES`. Sward heights (6-10 cm), and the fact
that voles clip runways, are field-realistic; how many centimetres a single
pass removes is not measured, it is chosen so that trails take days rather than
minutes or months to appear.
"""
from __future__ import annotations

import math

import numpy as np


class SwardField:
    """A grid of grass height (cm) that agents trample, clip, and regrow."""

    def __init__(self, width: float, height: float, params, rng=None):
        self.p = params
        self.cell = max(0.02, float(params.cell_size))
        self.nx = max(1, int(math.ceil(width / self.cell)))
        self.ny = max(1, int(math.ceil(height / self.cell)))
        rng = rng if rng is not None else np.random.default_rng(0)
        lo, hi = float(params.initial_min_cm), float(params.initial_max_cm)
        self.height = rng.uniform(min(lo, hi), max(lo, hi),
                                  (self.ny, self.nx)).astype(np.float32)
        if params.patchiness > 0:
            self._make_patchy(rng, float(params.patchiness))
        self.initial_mean = float(self.height.mean())
        #: cumulative cm removed, split by cause — the two are separate
        #: behaviours and a result should be able to attribute a trail
        self.trampled_cm = 0.0
        self.chewed_cm = 0.0

    def _make_patchy(self, rng, strength: float) -> None:
        """Smooth the initial field so the sward is patchy, not white noise.

        A box blur repeated a few times approximates a gaussian; the result is
        renormalised back to the requested height band so patchiness changes
        the *texture* of the sward, not how much of it there is.
        """
        lo, hi = float(self.p.initial_min_cm), float(self.p.initial_max_cm)
        k = max(1, int(round(strength * 6)))
        h = self.height.astype(np.float64)
        for _ in range(k):
            h = (h
                 + np.roll(h, 1, 0) + np.roll(h, -1, 0)
                 + np.roll(h, 1, 1) + np.roll(h, -1, 1)) / 5.0
        span = h.max() - h.min()
        if span > 1e-9:
            h = (h - h.min()) / span * (max(lo, hi) - min(lo, hi)) + min(lo, hi)
        self.height = h.astype(np.float32)

    # ------------------------------------------------------------------ #
    def cells_of(self, P: np.ndarray):
        col = np.clip((P[:, 0] / self.cell).astype(np.int32), 0, self.nx - 1)
        row = np.clip((P[:, 1] / self.cell).astype(np.int32), 0, self.ny - 1)
        return row, col

    def sample(self, P: np.ndarray) -> np.ndarray:
        """Grass height (cm) under each agent."""
        if len(P) == 0:
            return np.zeros(0)
        row, col = self.cells_of(P)
        return self.height[row, col].astype(np.float64)

    # ------------------------------------------------------------------ #
    def grow(self, dt: float, season_factor: float = 1.0) -> None:
        """Regrow toward ``max_cm``, slowing as the sward closes in on it.

        Logistic rather than linear so a bare trail recovers quickly at first
        and then tails off, and so growth cannot overshoot the ceiling however
        long the timestep is.
        """
        rate = float(self.p.regrowth_cm_per_day) * max(0.0, season_factor)
        if rate <= 0:
            return
        top = float(self.p.max_cm)
        gain = rate * (dt / 86400.0) * (1.0 - self.height / max(1e-6, top))
        np.clip(self.height + gain, 0.0, top, out=self.height)

    def trample(self, P: np.ndarray, metres: np.ndarray) -> None:
        """Wear the sward down under animals that moved this step."""
        per_m = float(self.p.trample_cm_per_m)
        if per_m <= 0 or len(P) == 0:
            return
        moved = np.asarray(metres, float)
        active = moved > 1e-9
        if not active.any():
            return
        self._reduce(P[active], per_m * moved[active], "trampled")

    def chew(self, P: np.ndarray, dt: float, chewing: np.ndarray) -> np.ndarray:
        """Clip the sward under animals that are chewing. Returns cm removed.

        Several times faster than walking over it — that ratio
        (``chew_rate_multiplier``) is what makes stopping worth the time.
        """
        removed = np.zeros(len(P))
        if not np.any(chewing):
            return removed
        idx = np.nonzero(chewing)[0]
        rate = (float(self.p.chew_cm_per_s)
                * float(self.p.chew_rate_multiplier))
        want = np.full(len(idx), rate * dt)
        removed[idx] = self._reduce(P[idx], want, "chewed")
        return removed

    def _reduce(self, P: np.ndarray, amount: np.ndarray, cause: str):
        """Lower the cells under ``P``, never below the clipping floor.

        Returns the cm actually removed per agent, which is less than asked
        for once the sward is already down to the floor — so a trail cannot be
        chewed into a negative-height hole, and the bookkeeping stays honest
        about how much work went nowhere.
        """
        row, col = self.cells_of(P)
        floor = float(self.p.chew_floor_cm)
        done = np.zeros(len(P))
        # a Python loop, but bounded by the cohort size (tens), and correct
        # when two animals stand on the same cell — np.add.at cannot clamp
        for k in range(len(P)):
            r, c = row[k], col[k]
            take = min(float(amount[k]), float(self.height[r, c]) - floor)
            if take <= 0:
                continue
            self.height[r, c] -= take
            done[k] = take
        total = float(done.sum())
        if cause == "chewed":
            self.chewed_cm += total
        else:
            self.trampled_cm += total
        return done

    # ------------------------------------------------------------------ #
    # How height turns into movement
    # ------------------------------------------------------------------ #
    def speed_factor(self, h) -> np.ndarray:
        """Speed multiplier for grass of height ``h`` (cm).

        Anchored so ``speed_ref_cm`` is neutral: below it an animal is on a
        worn path and gains speed up to ``speed_max_factor``; above it the
        sward drags, down to ``speed_min_factor`` at ``max_cm``.
        """
        h = np.asarray(h, float)
        ref = max(1e-6, float(self.p.speed_ref_cm))
        top = max(ref + 1e-6, float(self.p.max_cm))
        fast = float(self.p.speed_max_factor)
        slow = float(self.p.speed_min_factor)
        below = 1.0 + (fast - 1.0) * np.clip((ref - h) / ref, 0.0, 1.0)
        above = 1.0 - (1.0 - slow) * np.clip((h - ref) / (top - ref), 0.0, 1.0)
        return np.where(h <= ref, below, above)

    def push_kj(self, mass_g, metres, h) -> np.ndarray:
        """Extra kJ spent forcing a body through grass of height ``h``.

        Charged on top of the flat-ground locomotion cost, per metre and per
        centimetre of sward, so a worn trail is cheaper as well as quicker —
        which is what makes maintaining one pay for itself.
        """
        return (float(self.p.push_kj_per_kg_m_per_cm)
                * (np.asarray(mass_g, float) / 1000.0)
                * np.asarray(metres, float) * np.asarray(h, float))

    def chew_bout_s(self, h) -> np.ndarray:
        """How long a clipping bout lasts — longer in taller grass."""
        return (float(self.p.chew_seconds_per_cm)
                * np.clip(np.asarray(h, float), 0.0, None))

    # ------------------------------------------------------------------ #
    # Readouts
    # ------------------------------------------------------------------ #
    def mean_height(self) -> float:
        return float(self.height.mean())

    #: A cell counts as trail once it is worn to this share of the sward's
    #: own starting height. Relative, because "below 8 cm" would report half
    #: the enclosure as trail on day zero simply from the initial 6-10 cm
    #: spread — measuring the starting distribution, not the animals' work.
    TRAIL_FRACTION_OF_INITIAL = 0.6

    def trail_fraction(self) -> float:
        """Share of the arena the animals have actually worn into runway."""
        cut = self.TRAIL_FRACTION_OF_INITIAL * max(1e-6, self.initial_mean)
        return float((self.height < cut).mean())

    def image(self, max_side: int = 160):
        """(rgba uint8, extent) — the sward, dark where it is worn to a trail.

        Trails are the visible output of the whole mechanism, so this exists to
        make them watchable while a run happens rather than inferable from a
        CSV afterwards.
        """
        top = max(1e-6, float(self.p.max_cm))
        h = np.clip(self.height / top, 0.0, 1.0)
        ny, nx = h.shape
        step = max(1, int(math.ceil(max(ny, nx) / max(8, max_side))))
        if step > 1:
            ty, tx = (ny // step) * step, (nx // step) * step
            h = h[:ty, :tx].reshape(ty // step, step, tx // step,
                                    step).mean(axis=(1, 3))
            ny, nx = ty, tx
        rgba = np.zeros(h.shape + (4,), np.float32)
        # tall sward = saturated green, worn trail = pale bare earth
        rgba[..., 0] = 0.42 - 0.28 * h
        rgba[..., 1] = 0.38 + 0.20 * h
        rgba[..., 2] = 0.24 - 0.14 * h
        rgba[..., 3] = 0.55
        out = (np.clip(rgba, 0.0, 1.0) * 255).astype(np.uint8)
        return out, (0.0, nx * self.cell, 0.0, ny * self.cell)
