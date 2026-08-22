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
            thr = 50.0

        # Scent marking, when enabled, replaces BOTH geometric shortcuts below
        # (the home-range spring and the home_r territory proxy). Site fidelity
        # then means "stay where my own marks are" and territoriality means
        # "avoid where someone else's are" — so home-range size and territory
        # boundaries are outcomes of the run, not parameters of it.
        scented = getattr(sim, "scent", None) is not None
        own_vec = foreign_vec = None
        own_lvl = foreign_lvl = None
        if scented:
            idx = np.arange(n)
            own_vec, foreign_vec, own_lvl, foreign_lvl = sim.scent.sample(P, idx)

        # --- home-range spring (geometric; legacy path only) ---
        hv = sim.home - P
        dist_home = np.linalg.norm(hv, axis=1) + 1e-9
        if scented:
            home_dir = hv / dist_home[:, None]
            # Site fidelity has two channels. (1) Scent: come back to where my
            # own marks are — olfaction-dependent, so it fails under anosmia.
            k_sh = getattr(pp, "k_scent_home", 1.2) if pp is not None else 1.2
            desired = (k_sh * sim.smell)[:, None] * own_vec
            # (2) Spatial memory: an animal remembers where it has been living
            # even with no working nose. `home` tracks REALISED occupancy, so
            # this anchors without prescribing a home-range size — the settling
            # distance falls out of memory gain vs. this animal's wander drive.
            # Without it, anosmia would make animals disperse across the whole
            # arena rather than lose territorial structure while staying put,
            # which is the actual methimazole phenotype.
            k_mem = getattr(pp, "k_memory", 0.35) if pp is not None else 0.35
            w_mem = np.clip(k_mem * dist_home, 0.0, 3.0)
            desired += w_mem[:, None] * home_dir
        else:
            home_dir = hv / dist_home[:, None]
            w_home = self.k_home * np.clip(
                (dist_home - sim.home_r) / sim.home_r, 0, 1.5)
            desired = w_home[:, None] * home_dir

        # --- resource seeking (hunger/thirst are 0-100 bars) ---
        span = max(1e-6, 100.0 - thr)
        need_food = np.clip((sim.hunger - thr) / span, 0, 1)
        need_water = np.clip((sim.thirst - thr) / span, 0, 1)
        # a hungry animal suspends home fidelity to make a foraging trip
        rel = getattr(pp, "forage_releases_home", 0.0) if pp is not None else 0.0
        desired *= (1.0 - rel * np.maximum(need_food, need_water))[:, None]
        # steer for the access point (a walled zone's doorway), not the centre
        food_t = getattr(sim, "food_seek", sim.food)
        water_t = getattr(sim, "water_seek", sim.water)
        # Go after the MORE urgent need rather than summing both pulls: when
        # food and water sit in opposite directions the two vectors cancel and
        # the animal starves standing between them.
        go_food = need_food >= need_water
        res_vec = np.zeros((n, 2))
        if food_t.shape[0]:
            res_vec += self.k_resource * sim._seek(
                P, food_t, np.where(go_food, need_food, 0.0))
        if water_t.shape[0]:
            res_vec += self.k_resource * sim._seek(
                P, water_t, np.where(go_food, 0.0, need_water))

        # A walled resource zone has one small doorway. An animal inside it
        # that wants something outside — water, almost always — would otherwise
        # steer straight into a wall and stay there until it dehydrated, belly
        # full. Inside a zone, head for the gap first.
        rects = getattr(sim, "food_rects", np.zeros((0, 4)))
        doors = getattr(sim, "zone_doors", np.zeros((0, 2)))
        if len(rects) and len(doors):
            cx, cy, hw, hd = rects.T
            inz = ((np.abs(P[:, 0][:, None] - cx[None, :]) <= hw[None, :])
                   & (np.abs(P[:, 1][:, None] - cy[None, :]) <= hd[None, :]))
            leaving = inz.any(axis=1) & (~go_food) & (need_water > 0)
            if leaving.any():
                j = np.argmax(inz, axis=1)
                to_door = doors[j] - P
                dd = np.linalg.norm(to_door, axis=1)[:, None] + 1e-9
                exit_vec = self.k_resource * (to_door / dd) * need_water[:, None]
                res_vec = np.where(leaving[:, None], exit_vec, res_vec)
        desired += res_vec

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

        # --- territory avoidance ---
        if scented:
            # Real marks: keep away from ground another animal has marked. The
            # response needs BOTH a nose that works (sim.smell) and a mark that
            # carries a readable signature (folded into foreign_vec via the
            # mark's stored identity) — so anosmia and MUP-knockout degrade
            # spacing by different routes, which is the point of the model.
            k_sa = getattr(pp, "k_scent_avoid", 2.2) if pp is not None else 2.2
            # ...but a hungry animal trespasses. Without this, an animal that
            # does not hold the patch containing the food starves rather than
            # cross a boundary, and territoriality silently becomes lethal.
            avoid = (k_sa * sim.smell) * (1.0 - rel * np.maximum(need_food,
                                                                 need_water))
            desired += avoid[:, None] * foreign_vec
        else:
            to_home = P[:, None, :] - sim.home[None, :, :]
            dth = np.linalg.norm(to_home, axis=2)
            np.fill_diagonal(dth, np.inf)
            inside = np.clip((sim.home_r[None, :] - dth) / sim.home_r[None, :],
                             0, 1)
            same_sex = mm + ff
            terr_w = (self.k_territory * sim.smell[:, None]
                      * sim.identity[None, :] * (0.3 + sim.aggr[None, :])
                      * inside * same_sex
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
            # handed back so the engine can drive counter-marking without
            # sampling the scent field a second time
            "scent_own": own_lvl, "scent_foreign": foreign_lvl,
        }
