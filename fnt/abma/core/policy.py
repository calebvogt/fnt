"""Agent decision policies — the swappable "brain" behind each step.

A :class:`Policy` turns the population's current state into a desired movement
direction per agent. The engine (``Simulation``) owns everything else: physics,
physiology, combat, recording. This seam keeps the *rule logic* editable and
leaves room for a future ``RLPolicy`` (a learned policy) without touching the
engine — rule-based agents and learning agents can share the same interface.

``decide(sim, elapsed_s)`` returns ``(desired, perception)``:
  * ``desired`` — (N, 2) unnormalised desired-heading vectors.
  * ``perception`` — a dict of reusable intermediates the engine also needs
    (``dist_home``, ``within``, ``rec_j``, ``need_food``, ``need_water``), so
    they are computed once.
"""
from __future__ import annotations

import numpy as np


class Policy:
    def decide(self, sim, elapsed_s: float):
        raise NotImplementedError


class RuleBasedPolicy(Policy):
    """The default hand-written ruleset. All weights are transparent dials."""

    k_home = 1.0          # home-range spring gain
    k_resource = 1.6      # resource-seeking gain
    k_social = 1.0        # social force gain
    k_territory = 2.0     # scent-marked territory avoidance gain
    k_random = 0.5        # exploratory noise gain
    perception_r = 0.6    # neighbour perception radius (m)
    heading_jitter = 0.5  # radians of heading drift per step

    def decide(self, sim, elapsed_s: float):
        P = sim.P
        n = sim.n

        # --- home-range spring ---
        hv = sim.home - P
        dist_home = np.linalg.norm(hv, axis=1) + 1e-9
        home_dir = hv / dist_home[:, None]
        w_home = self.k_home * np.clip(
            (dist_home - sim.home_r) / sim.home_r, 0, 1.5)
        desired = w_home[:, None] * home_dir

        # --- resource seeking ---
        need_food = np.clip(sim.hunger - 0.5, 0, 1)
        need_water = np.clip(sim.thirst - 0.5, 0, 1)
        if sim.food.shape[0]:
            desired += self.k_resource * sim._seek(P, sim.food, need_food)
        if sim.water.shape[0]:
            desired += self.k_resource * sim._seek(P, sim.water, need_water)

        # --- pairwise social / territorial forces (olfaction-gated) ---
        diff = P[None, :, :] - P[:, None, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        unit = diff / (dist[:, :, None] + 1e-9)
        alive_f = sim.alive.astype(float)
        within = (dist < self.perception_r) * alive_f[:, None] * alive_f[None, :]

        recog = np.outer(sim.smell, sim.identity)
        s = sim.sex_m
        mm = np.outer(s, s)
        ff = np.outer(1 - s, 1 - s)
        opp = 1.0 - mm - ff
        rec_j = sim._receptivity(elapsed_s)

        w_opp = (0.3 + rec_j[None, :]) * sim.social[:, None] * opp
        w_ff = sim.social[:, None] * (0.4 + 0.6 * recog) * ff
        w_mm = -sim.aggr[:, None] * (0.3 + 0.7 * recog) * mm
        W = (w_opp + w_ff + w_mm) * within
        social_vec = np.einsum("ij,ijk->ik", W, unit) * self.k_social
        smag = np.linalg.norm(social_vec, axis=1)
        cap = np.clip(smag, 0, 2.0)
        social_vec = np.where(smag[:, None] > 1e-9,
                              social_vec / (smag[:, None] + 1e-9) * cap[:, None],
                              social_vec)
        desired += social_vec

        # --- scent-marked territory avoidance ---
        to_home = P[:, None, :] - sim.home[None, :, :]
        dth = np.linalg.norm(to_home, axis=2)
        np.fill_diagonal(dth, np.inf)
        inside = np.clip((sim.home_r[None, :] - dth) / sim.home_r[None, :], 0, 1)
        same_sex = mm + ff
        terr_w = (self.k_territory * sim.smell[:, None] * sim.identity[None, :]
                  * (0.3 + sim.aggr[None, :]) * inside * same_sex
                  * alive_f[:, None] * alive_f[None, :])
        terr_dir = to_home / (dth[:, :, None] + 1e-9)
        desired += np.einsum("ij,ijk->ik", terr_w, terr_dir)

        # --- correlated random walk (per-agent turn rate + wander gain) ---
        sim.H += sim.rng.normal(0.0, 1.0, n) * np.clip(sim.turn_rate, 1e-3, None)
        rand_dir = np.stack([np.cos(sim.H), np.sin(sim.H)], axis=1)
        desired += sim.wander[:, None] * rand_dir

        return desired, {
            "dist": dist, "dist_home": dist_home, "within": within,
            "rec_j": rec_j, "need_food": need_food, "need_water": need_water,
        }
