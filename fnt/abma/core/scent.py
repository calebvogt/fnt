"""Scent marking — the substrate territoriality is built out of.

Rodents deposit urine marks as they move. A mark is a *thing left in a place*:
it says who was last here, it carries (in wild-type animals) an individual
signature, and it fades. Everything social that ABMA models on top of this is
emergent — no agent is told where its territory is, or how big a home range to
keep. It marks, it smells other animals' marks, and structure appears.

Why this is a field and not a per-agent attribute
-------------------------------------------------
A prescribed ``home_range_r`` answers the question the experiment is supposed to
ask. Marks don't: an animal that marks heavily and avoids foreign marks *ends
up* with a defended area, and its size falls out of density, marking effort,
olfactory acuity and decay rate. That makes "how does social structure organise
when marks fade in 6 h vs 6 days?" a runnable experiment rather than a
parameter you have to assume.

Representation
--------------
A regular grid over the arena. Per cell:
  ``owner``    index of the animal whose mark currently dominates (-1 = clean)
  ``strength`` 0..1 mark salience, decaying exponentially
  ``ident``    the signature strength *at the moment of deposit* — a MUP-KO
               animal leaves a real mark with ``ident`` 0, so others can smell
               that someone passed but not who.

Only the dominant mark per cell is kept ("who was last here"), which is what an
animal reads off a spot and keeps the field O(cells) rather than O(cells x
animals).
"""
from __future__ import annotations

import math

import numpy as np


class ScentField:
    """Decaying grid of identity-carrying scent marks over the arena."""

    #: marks below this are treated as gone (keeps the field sparse & cheap)
    EPS = 0.01

    def __init__(self, width: float, height: float, params):
        self.p = params
        self.cell = max(0.01, float(params.cell_size))
        self.nx = max(1, int(math.ceil(width / self.cell)))
        self.ny = max(1, int(math.ceil(height / self.cell)))
        self.owner = np.full((self.ny, self.nx), -1, np.int32)
        self.strength = np.zeros((self.ny, self.nx), np.float32)
        self.ident = np.zeros((self.ny, self.nx), np.float32)
        self._build_stencil(float(params.perception_r))

    # ------------------------------------------------------------------ #
    def _build_stencil(self, radius: float) -> None:
        """Offsets (in cells) within ``radius``, with their unit directions."""
        k = max(1, int(math.ceil(radius / self.cell)))
        dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
        dist = np.hypot(dx * self.cell, dy * self.cell)
        keep = (dist <= radius) & (dist > 1e-9)      # exclude the centre cell
        self._sdx = dx[keep].astype(np.int32)
        self._sdy = dy[keep].astype(np.int32)
        d = dist[keep]
        # unit vector from the animal toward that cell, in metres
        self._sux = (self._sdx * self.cell / d).astype(np.float32)
        self._suy = (self._sdy * self.cell / d).astype(np.float32)
        # nearer marks matter more (simple 1/(1+d) falloff)
        self._sw = (1.0 / (1.0 + d)).astype(np.float32)

    def cells_of(self, P: np.ndarray):
        """Map positions (n, 2) in metres to (row, col) indices, clipped."""
        col = np.clip((P[:, 0] / self.cell).astype(np.int32), 0, self.nx - 1)
        row = np.clip((P[:, 1] / self.cell).astype(np.int32), 0, self.ny - 1)
        return row, col

    # ------------------------------------------------------------------ #
    def decay(self, dt: float) -> None:
        """Fade every mark by the configured half-life."""
        hl_s = max(1e-6, float(self.p.half_life_h) * 3600.0)
        self.strength *= float(0.5 ** (dt / hl_s))
        gone = self.strength < self.EPS
        if gone.any():
            self.strength[gone] = 0.0
            self.owner[gone] = -1
            self.ident[gone] = 0.0

    def deposit(self, idxs, P: np.ndarray, amount, ident) -> None:
        """Lay marks for the agents in ``idxs`` at their positions.

        ``P`` is the FULL (n_agents, 2) position array and ``idxs`` indexes
        into it — the same integer is what gets stored as the cell's owner, so
        agent identity and row order are one and the same.

        A mark reinforces the cell if the depositor already owns it, otherwise
        it takes the cell over only if it lands stronger than what is there —
        so over-marking a rival is possible but costs more than topping up your
        own patch.
        """
        if len(idxs) == 0:
            return
        row, col = self.cells_of(P[idxs])
        amount = np.broadcast_to(np.asarray(amount, np.float32), (len(idxs),))
        ident = np.broadcast_to(np.asarray(ident, np.float32), (len(idxs),))
        for k, i in enumerate(idxs):
            r, c = row[k], col[k]
            cur = self.strength[r, c]
            if self.owner[r, c] == i:
                self.strength[r, c] = min(1.0, cur + amount[k])
                self.ident[r, c] = ident[k]
            elif amount[k] > cur:                 # over-mark a rival's spot
                self.owner[r, c] = i
                self.strength[r, c] = amount[k]
                self.ident[r, c] = ident[k]

    # ------------------------------------------------------------------ #
    def sample(self, P: np.ndarray, idx: np.ndarray):
        """Read the local scent landscape for every agent.

        Returns ``(own_vec, foreign_vec, own_level, foreign_level)``:
          ``own_vec``     pull toward the animal's own marks (site fidelity)
          ``foreign_vec`` push away from other animals' marks (territoriality)
          ``own_level``   0..1 how strongly it is standing in its own patch
          ``foreign_level`` 0..1 how much foreign scent is present here

        Foreign marks are weighted by their signature ``ident``: an anonymous
        mark (MUP-KO) still registers as "somebody passed" but carries far less
        territorial weight than a recognisable one.
        """
        n = len(P)
        if n == 0:
            z2 = np.zeros((0, 2))
            return z2, z2.copy(), np.zeros(0), np.zeros(0)
        row, col = self.cells_of(P)
        # (n, k) neighbourhood indices, clipped at the arena edge
        rr = np.clip(row[:, None] + self._sdy[None, :], 0, self.ny - 1)
        cc = np.clip(col[:, None] + self._sdx[None, :], 0, self.nx - 1)
        s = self.strength[rr, cc]                       # (n, k)
        o = self.owner[rr, cc]
        idn = self.ident[rr, cc]

        mine = o == idx[:, None]
        other = (o >= 0) & ~mine
        w = s * self._sw[None, :]

        # Normalised by the stencil's total weight, so these come back as
        # O(1) steering vectors. Without it the pull grows with how heavily an
        # area is marked and a heavy marker gets trapped in its own patch —
        # unable to out-vote its own scent even when starving.
        norm = max(1e-9, float(self._sw.sum()))

        # own marks: attraction toward the patch you have been maintaining
        wo = np.where(mine, w, 0.0)
        own_vec = np.stack([(wo * self._sux[None, :]).sum(1),
                            (wo * self._suy[None, :]).sum(1)], axis=1) / norm
        # foreign marks: repulsion, discounted when the signature is unreadable
        anon = float(self.p.anonymous_weight)
        wf = np.where(other, w * (anon + (1.0 - anon) * idn), 0.0)
        foreign_vec = -np.stack([(wf * self._sux[None, :]).sum(1),
                                 (wf * self._suy[None, :]).sum(1)],
                                axis=1) / norm

        own_level = np.clip(wo.sum(1) / norm, 0, 1)
        foreign_level = np.clip(np.where(other, w, 0.0).sum(1) / norm, 0, 1)
        return own_vec, foreign_vec, own_level, foreign_level

    # ------------------------------------------------------------------ #
    def occupancy(self):
        """(owner, strength) copies — for drawing the territory map."""
        return self.owner.copy(), self.strength.copy()

    def marked_fraction(self) -> float:
        """Share of the arena currently carrying a detectable mark."""
        return float((self.owner >= 0).mean())

    def coverage_by_owner(self, n_agents: int) -> np.ndarray:
        """Cells currently dominated by each agent — emergent territory size."""
        out = np.zeros(n_agents, int)
        live = self.owner[self.owner >= 0]
        if live.size:
            cnt = np.bincount(live, minlength=n_agents)
            out[:len(cnt)] = cnt[:n_agents]
        return out

    def area_by_owner(self, n_agents: int) -> np.ndarray:
        """Emergent territory area per agent (m²) — never prescribed."""
        return self.coverage_by_owner(n_agents) * (self.cell ** 2)
