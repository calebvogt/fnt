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
    def decide(self, sim, elapsed_s: float, dt: float):
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

    # Trait tuning (turn_rate defaults etc.) was calibrated at this timestep;
    # jitter scales with sqrt(dt/REF) so path tortuosity is dt-invariant while
    # existing configs behave identically at the reference dt.
    REF_DT = 2.0

    def decide(self, sim, elapsed_s: float, dt: float):
        P = sim.P
        n = sim.n
        # config-driven weights (fall back to the class defaults)
        pp = getattr(sim.cfg, "policy", None)
        if pp is not None:
            self.k_home = pp.k_home
            self.k_resource = pp.k_resource
            self.k_social = pp.k_social
            self.k_territory = pp.k_territory
            self.k_random = pp.k_random
            self.perception_r = pp.perception_r
            thr = pp.forage_threshold
        else:
            thr = 0.5

        # --- home-range spring ---
        hv = sim.home - P
        dist_home = np.linalg.norm(hv, axis=1) + 1e-9
        home_dir = hv / dist_home[:, None]
        w_home = self.k_home * np.clip(
            (dist_home - sim.home_r) / sim.home_r, 0, 1.5)
        desired = w_home[:, None] * home_dir

        # --- resource seeking ---
        span = max(1e-6, 1.0 - thr)
        need_food = np.clip((sim.hunger - thr) / span, 0, 1)
        need_water = np.clip((sim.thirst - thr) / span, 0, 1)
        # a hungry animal suspends home fidelity to make a foraging trip
        rel = getattr(pp, "forage_releases_home", 0.0) if pp is not None else 0.0
        desired *= (1.0 - rel * np.maximum(need_food, need_water))[:, None]
        # steer for the access point (a walled zone's doorway), not the centre
        food_t = getattr(sim, "food_seek", sim.food)
        water_t = getattr(sim, "water_seek", sim.water)
        if food_t.shape[0]:
            desired += self.k_resource * sim._seek(P, food_t, need_food)
        if water_t.shape[0]:
            desired += self.k_resource * sim._seek(P, water_t, need_water)

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
        # heading drift is diffusive: sd grows with sqrt(dt), so tortuosity per
        # simulated second doesn't change when the integration step does
        jit = np.clip(sim.turn_rate, 1e-3, None) * np.sqrt(dt / self.REF_DT)
        sim.H += sim.rng.normal(0.0, 1.0, n) * jit
        rand_dir = np.stack([np.cos(sim.H), np.sin(sim.H)], axis=1)
        desired += sim.wander[:, None] * rand_dir

        return desired, {
            "dist": dist, "dist_home": dist_home, "within": within,
            "rec_j": rec_j, "need_food": need_food, "need_water": need_water,
        }
