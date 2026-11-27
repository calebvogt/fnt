"""Rest, warmth and attachment: the needs that make a vole seek company.

Food and water explain where an animal goes when it is hungry or thirsty. They
do not explain the thing a group of prairie voles spends most of its time
doing, which is resting, in contact, with particular others. This module adds
the three needs behind that (:class:`fnt.abma.core.config.SocialParams`), and
nothing about what to do with them:

* **tired** rises while an animal is up and falls only while it rests;
* **cold** is high for an animal resting alone in the open, lower in a shelter,
  lower again against another body, and absent while it is moving;
* **lonely** (separation distress) rises in an animal that is attached to
  another and cannot smell it nearby.

and one memory:

* **attachment** ``A[i, j]``: how attached animal i is to the *odour* of
  animal j. It grows with contact and with mating, only as far as i can smell
  and tell apart j, and an attachment already formed holds back a new one.

Smell matters here in exactly two places, and both are sensory. An attachment
is learned from an odour, so an animal that cannot take that odour in forms
none (olfactory bulb removal prevents partner preference formation in female
prairie voles: Williams et al. 1992, Physiol Behav 52:635). And an attachment
is *read* from an odour: the partner signal, the relief of separation distress
and the aversion to an unattached animal at close quarters all pass through
what the animal smells right now. Warmth passes through none of it. So taking
away the nose does not make an animal less social; it removes what made its
sociality selective. Whether that is enough to turn a cohort of pairs into one
undifferentiated huddle is the question the module exists to let a run ask.

There are two noses. The main olfactory epithelium (``smell_ability``) takes
in volatile odour at a distance; the vomeronasal organ (``vno_ability``)
takes in involatile cues on contact. Which of them an attachment needs is a
setting, because it is not settled: removing the vomeronasal organ alone also
prevents mating-induced partner preference in female prairie voles, with main
olfaction intact (Curtis, Liu & Wang 2001, Brain Res 901:167). ``SocialParams.
bond_needs_moe``, ``bond_needs_vno`` and ``read_by_vno`` state the hypothesis.

The engine calls :meth:`SocialState.step` once per step and
:meth:`SocialState.on_mating` when a pair mates; :func:`fnt.abma.core.senses.
sense` calls :meth:`SocialState.odours` and reads the three bars.
"""
from __future__ import annotations

import numpy as np


class SocialState:
    """Per-animal social needs and the attachment matrix of one simulation."""

    def __init__(self, sim):
        n = sim.n
        self.p = sim.cfg.social
        self.tired = np.zeros(n)
        self.cold = np.zeros(n)
        self.lonely = np.zeros(n)
        #: A[i, j]: how attached animal i is to the odour of animal j (0-1)
        self.A = np.zeros((n, n))
        #: bookkeeping for a run's summary: seconds resting, seconds touching
        #: another animal, seconds touching the animal it is most attached to
        self.rest_s = np.zeros(n)
        self.touch_s = np.zeros(n)
        self.partner_touch_s = np.zeros(n)
        #: a mating as the body reports it to the brain: 1 when it happens,
        #: fading over ``mating_trace_s``
        self.mated = np.zeros(n)
        #: the pull toward the other sex (0-100; stays 0 unless libido_rise_h > 0)
        self.libido = np.zeros(n)
        self._sig = None
        self._shelter = self._shelters(sim)

    # ---- roster --------------------------------------------------------- #
    def resize(self, n: int) -> None:
        have = len(self.tired)
        if n <= have:
            return
        pad = n - have
        for name in ("tired", "cold", "lonely", "rest_s", "touch_s", "partner_touch_s", "mated", "libido"):
            setattr(self, name, np.concatenate([getattr(self, name), np.zeros(pad)]))
        A = np.zeros((n, n))
        A[:have, :have] = self.A
        self.A = A

    @staticmethod
    def _shelters(sim):
        """Walled boxes ``(cx, cy, hw, hd)`` and huts ``(cx, cy, r)``."""
        a = sim.cfg.arena
        boxes = np.array([(z.x, z.y, z.w / 2.0, z.d / 2.0) for z in getattr(a, "resource_zones", [])],
                         float).reshape(-1, 4)
        huts = np.array([(h.x, h.y, 0.5 * max(h.w, 0.05)) for h in getattr(a, "huts", [])],
                        float).reshape(-1, 3)
        return boxes, huts

    def in_shelter(self, P) -> np.ndarray:
        boxes, huts = self._shelter
        out = np.zeros(len(P), bool)
        if len(boxes):
            cx, cy, hw, hd = boxes.T
            out |= ((np.abs(P[:, 0][:, None] - cx[None, :]) <= hw[None, :])
                    & (np.abs(P[:, 1][:, None] - cy[None, :]) <= hd[None, :])).any(axis=1)
        if len(huts):
            d = np.linalg.norm(P[:, None, :] - huts[None, :, :2], axis=2)
            out |= (d <= huts[None, :, 2]).any(axis=1)
        return out

    def signatures(self, sim) -> np.ndarray:
        """Each animal's own odour signature, ``(n, 24)``: three of 24
        components, drawn once from the run's seed and the animal's stable
        uid. An assumption (outbred animals smell different), not a known
        carrier: a vole has one MUP gene, so not the mouse's MUP barcode."""
        n = sim.n
        if self._sig is None or len(self._sig) != n:
            out = np.zeros((n, 24))
            for i in range(n):
                rng = np.random.default_rng([int(sim._root_seed) & 0x7FFFFFFF, int(sim.uid[i]) & 0x7FFFFFFF, 0x5349])
                out[i, rng.choice(24, 3, replace=False)] = 1.0
            self._sig = out
        return self._sig * sim.identity[:n, None]

    # ---- what the nose makes of the others ------------------------------ #
    def bond(self) -> np.ndarray:
        """How attached each animal is to anyone at all (its strongest tie)."""
        return self.A.max(axis=1) if self.A.size else np.zeros(len(self.tired))

    def partner(self) -> np.ndarray:
        """The animal each one is most attached to, or -1."""
        if not self.A.size:
            return np.full(len(self.tired), -1)
        j = self.A.argmax(axis=1)
        return np.where(self.A[np.arange(len(j)), j] > 0.05, j, -1)

    def odours(self, sim, dist, unit):
        """Body odour of the others, sorted by attachment, before the reader's
        own acuity is applied (the caller scales by ``smell``, as it does for
        every olfactory channel).

        Returns ``(partner_level, partner_vec, stranger_level, stranger_vec)``:
        levels 0-1 and world-frame direction vectors ``(n, 2)``. *Partner* is
        the odour of animals this one is attached to, weighted by how much;
        *stranger* is the odour of the rest. ``identity_signal`` scales how
        much of an animal's odour says who it is: an animal with none of it is
        smelled, but neither as a partner nor as a stranger.
        """
        n = sim.n
        self.resize(n)
        lam = max(1e-6, float(self.p.partner_reach_m))
        alive = sim.alive.astype(float)
        w = np.exp(-np.where(np.isfinite(dist), dist, 1e9) / lam) * alive[None, :] * sim.identity[None, :]
        out = []
        for wt in (self.A[:n, :n], 1.0 - self.A[:n, :n]):
            wc = w * wt
            np.fill_diagonal(wc, 0.0)
            lvl = np.clip(wc.sum(axis=1), 0.0, 1.0)
            g = np.einsum("ij,ijk->ik", wc, unit)
            gm = np.linalg.norm(g, axis=1) + 1e-12
            out += [lvl, g / gm[:, None] * (lvl / (lvl + 0.05))[:, None]]
        return tuple(out)

    def _contact(self, sim, dist) -> np.ndarray:
        """Who is close enough to whom for the vomeronasal organ to work."""
        alive = sim.alive
        return (dist < sim.contact_r) & alive[None, :] & alive[:, None]

    def contact_odours(self, sim, dist):
        """What the vomeronasal organ makes of the animals in contact:
        ``(partner, stranger)``, each 0-1. Silent unless an attachment can be
        read that way (``read_by_vno``), and for an animal without the organ."""
        n = sim.n
        if not self.p.read_by_vno:
            return np.zeros(n), np.zeros(n)
        self.resize(n)
        w = self._contact(sim, dist) * sim.identity[None, :] * sim.vno[:, None]
        A = self.A[:n, :n]
        return np.clip((w * A).sum(axis=1), 0.0, 1.0), np.clip((w * (1.0 - A)).sum(axis=1), 0.0, 1.0)

    def _learns(self, sim) -> np.ndarray:
        """``L[i, j]``: how well animal i can take in what makes j itself,
        through whichever nose an attachment needs."""
        p, n = self.p, sim.n
        L = (sim.recognition_matrix()[:n, :n] if p.bond_needs_moe
             else np.broadcast_to(sim.identity[None, :], (n, n)))
        return L * sim.vno[:, None] if p.bond_needs_vno else L

    def _reads(self, sim, near, contact) -> np.ndarray:
        """``R[i, j]``: how strongly animal i is taking in j's identity right
        now: by smell, fading with distance (``near``), and by the vomeronasal
        organ in contact if an attachment can be read that way."""
        R = sim.smell[:, None] * near
        if self.p.read_by_vno:
            R = np.maximum(R, sim.vno[:, None] * contact)
        return R * sim.identity[None, :]

    # ---- one step ------------------------------------------------------- #
    def step(self, sim, dt: float, dist, speed_frac) -> None:
        """Advance the needs and the attachments by one engine step.

        ``dist`` is the pairwise distance matrix of this step (inf on the
        diagonal); ``speed_frac`` each animal's speed as a fraction of its easy
        pace (0 = still)."""
        p, n = self.p, sim.n
        self.resize(n)
        alive = sim.alive
        dth = dt / 3600.0
        sf = np.clip(np.asarray(speed_frac, float), 0.0, 1.0)
        resting = (sf < p.rest_speed_frac) & alive

        # ---- tired ----------------------------------------------------- #
        up = p.tired_rise_h + p.tired_move_h * sf
        self.tired = np.clip(self.tired + dth * np.where(resting, -p.tired_fall_h, up), 0.0, 100.0)

        # ---- cold ------------------------------------------------------ #
        d = np.where(alive[None, :] & alive[:, None], dist, np.inf)
        touching = d < p.huddle_r_m
        k = touching.sum(axis=1)
        target = p.cold_alone * (1.0 - sf)                       # moving warms
        target = target * np.where(self.in_shelter(sim.P), 1.0 - p.shelter_warmth, 1.0)
        target = target * (1.0 - p.huddle_warmth) ** k
        self.cold = np.clip(self.cold + (target - self.cold) * min(1.0, dt / max(1e-6, p.cold_tau_s)), 0.0, 100.0)

        # ---- attachment ------------------------------------------------ #
        recog = self._learns(sim)                                # how well i takes in what makes j itself
        A = self.A
        contact = d < sim.contact_r
        same = sim.sex_m[:, None] == sim.sex_m[None, :]
        grow = p.bond_rate_h * dth * contact * recog * np.where(same, p.peer_bond, 1.0)
        A += grow * (1.0 - A) * self._room(A)
        A -= p.bond_decay_h * dth * A * ~contact
        np.fill_diagonal(A, 0.0)
        np.clip(A, 0.0, 1.0, out=A)

        # ---- lonely: attached, and the partner is not smelled nearby --- #
        bond = A.max(axis=1)
        near = np.clip(1.0 - d / max(1e-6, p.partner_reach_m), 0.0, 1.0)
        with_partner = np.clip((A * self._reads(sim, near, contact)).sum(axis=1) / np.maximum(bond, 1e-6), 0.0, 1.0)
        in_brain = None
        if p.bond_in_brain:                  # the brain's own memory decides, if it has one
            read = getattr(sim.policy, "partner_memory", None)
            in_brain = read(n) if read is not None else None
        if in_brain is not None:
            bond = in_brain[0]
            with_partner = np.clip(in_brain[1] / max(1e-6, p.match_full), 0.0, 1.0)
        self.lonely = np.clip(self.lonely + dth * (p.lonely_rise_h * bond * (1.0 - with_partner)
                                                   - p.lonely_fall_h * with_partner), 0.0, 100.0)

        # ---- an unattached animal at close quarters, to an attached one - #
        closeness = np.clip(1.0 - d / max(1e-6, p.stranger_r_m), 0.0, 1.0)
        stranger = np.clip(((1.0 - A) * self._reads(sim, closeness, contact)).sum(axis=1), 0.0, 1.0)
        if in_brain is not None:
            # no table: someone is smelled close by, and what is smelled does not match what is stored
            someone = np.clip((closeness * sim.identity[None, :]).max(axis=1) * sim.smell, 0.0, 1.0)
            stranger = someone * (1.0 - with_partner)
        sim.stress = np.clip(sim.stress + dth * p.stranger_stress_h * bond * stranger, 0.0, 100.0)

        # ---- the pull toward the other sex ------------------------------- #
        if p.libido_rise_h > 0.0:
            other = sim.sex_m[:, None] != sim.sex_m[None, :]
            reach = max(1e-6, float(sim.cfg.policy.perception_r))
            air = np.clip((np.exp(-d / reach) * other * sim.identity[None, :]).sum(axis=1), 0.0, 1.0) * sim.smell
            skin = ((d < sim.contact_r) & other).any(axis=1) * sim.vno
            taken_in = np.maximum(air, skin)
            self.libido = np.clip(self.libido + dth * (p.libido_rise_h * (1.0 - taken_in)
                                                       - p.libido_fall_h * taken_in), 0.0, 100.0)

        for arr in (self.tired, self.cold, self.lonely, self.mated, self.libido):
            arr[~alive] = 0.0
        self.mated *= np.exp(-dt / max(1.0, p.mating_trace_s))

        # ---- bookkeeping ----------------------------------------------- #
        self.rest_s += dt * resting
        self.touch_s += dt * (k > 0)
        partner = self.partner()
        has = partner >= 0
        self.partner_touch_s += dt * (has & touching[np.arange(n), np.where(has, partner, 0)])

    def _room(self, A) -> np.ndarray:
        """How much room animal i has left for an attachment to j: all of it
        if it is attached to no one else, less the more it already is."""
        if A.shape[0] < 2:
            return np.ones_like(A)
        order = np.sort(A, axis=1)
        top, second = order[:, -1], order[:, -2]
        other = np.where(A >= top[:, None], second[:, None], top[:, None])   # strongest tie to anyone but j
        return np.clip(1.0 - self.p.bond_exclusive * other, 0.0, 1.0)

    def on_mating(self, sim, i: int, j: int) -> None:
        """A mating: each animal's attachment to the other steps up, as far as
        it can smell the other and has room for it."""
        self.resize(sim.n)
        self.mated[[i, j]] = 1.0
        self.libido[[i, j]] = np.maximum(0.0, self.libido[[i, j]] - self.p.libido_mating)
        recog = self._learns(sim)
        room = self._room(self.A)
        for a, b in ((i, j), (j, i)):
            self.A[a, b] = min(1.0, self.A[a, b] + self.p.mate_bond * recog[a, b]
                               * (1.0 - self.A[a, b]) * room[a, b])

    # ---- reading it back ------------------------------------------------ #
    def discomfort(self) -> np.ndarray:
        p = self.p
        return (p.w_tired * (self.tired / 100.0) ** 2 + p.w_cold * (self.cold / 100.0) ** 2
                + p.w_lonely * (self.lonely / 100.0) ** 2 + p.w_libido * (self.libido / 100.0) ** 2)
