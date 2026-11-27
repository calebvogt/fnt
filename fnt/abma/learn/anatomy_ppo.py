"""Raising a brain wired from anatomy.

The network of :mod:`fnt.abma.core.anatomy` starts as random draws inside the
wiring of its spec. This trains those weights the way a learned brain is
trained (:mod:`fnt.abma.learn.ppo`): lives in the nursery's randomised worlds
on the real engine, scored only by the change in the body's own discomfort
(hunger, thirst, stress, injury, fatigue) plus a mating bonus. Nothing rewards
approaching food, marking, biting or fleeing.

What training may and may not change:

* A weight exists only where the spec wires two cells, and keeps the sign the
  spec gives that projection (an excitatory projection cannot become
  inhibitory; a cell of a mixed projection keeps the sign it drew). Within an
  area the local wiring is free in sign. Where a sense enters is fixed.
* Receptor levels, the areas, and the dynamics (time constant, each area's own
  inhibition) are not trained. They are the anatomy.

While it is being trained the inside of the network is quiet and the muscles
try things: Gaussian noise is added to the drive of the muscle units only, and
that is the "action" the learner scores. At experiment time the usual small
noise in every cell is back, which is a difference between how it was raised
and how it is run.

The critic that estimates how well things are going is a training scaffold
beside the brain, not part of the anatomy; it rides along in the weights file
so training can be resumed, and the simulation never reads it.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import anatomy
from ..core.senses import SENSORS
from . import nursery
from .nursery import Curriculum

WEIGHTS_FORMAT = 1


@dataclass
class AnatomyHyper:
    """Learning settings. The defaults suit a four-core machine."""
    worlds: int = 24             # nursery worlds per iteration
    workers: int = 4
    gamma: float = 0.998         # a tick is 0.25 s, so the horizon is long in ticks
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatches: int = 8
    target_kl: float = 0.03
    chunk: int = 32              # steps of backpropagation through time
    entropy: float = 0.002
    value_coef: float = 0.5
    max_grad: float = 0.5
    log_std_init: float = -1.0   # how hard the muscles try things at first
    #: the most noise training may put on a muscle unit's drive (log of its
    #: sd). Left free it grew until the noise did the moving and the network
    #: commanded nothing (2026-11-26).
    log_std_max: float = -0.5
    #: iterations at the start in which only the critic learns
    critic_warmup: int = 0
    wiring_seed: int = 0         # which random draw training starts from
    #: threads the weight update may use (0 = PyTorch's default, every core);
    #: a few is right on a machine that is busy with other work
    torch_threads: int = 0
    seed: int = 0


def _inv_softplus(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 1e-4)
    return np.where(w > 20.0, w, np.log(np.expm1(np.minimum(w, 20.0))))


class TorchAnatomy(nn.Module):
    """The trainable twin of :class:`fnt.abma.core.anatomy.AnatomicalBrain`."""

    def __init__(self, brain: anatomy.AnatomicalBrain, log_std_init: float = -1.0):
        super().__init__()
        N, I = brain.n, len(SENSORS)
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)     # noqa: E731
        self.n, self.alpha, self.k_inh = N, float(brain.alpha), float(brain.k_inh)
        self.register_buffer("S", f(brain.S))                         # allowed sign, 0 = not wired
        self.register_buffer("wired", f(brain.S != 0))
        self.register_buffer("local", f(brain.local_mask))
        self.register_buffer("in_mask", f(brain.in_mask))
        self.register_buffer("pool", f(brain.pool))
        self.register_buffer("out_idx", torch.as_tensor(brain.out_units, dtype=torch.long))
        # neuromodulators: the level is a body state (a sensor column) or the
        # mean activity of the area that releases it
        dens, src, body = [], [], []
        for mid, source, d in brain.mods:
            row = np.zeros(N)
            col = -1
            if mid in brain._body_col:
                col = brain._body_col[mid]
            elif source:
                s = brain.sl[source]
                row[s] = 1.0 / (s.stop - s.start)
            else:
                continue
            dens.append(d)
            src.append(row)
            body.append(col)
        self.register_buffer("dens", f(np.array(dens).reshape(len(dens), N)))
        self.register_buffer("src", f(np.array(src).reshape(len(src), N)))
        # body-state modulators: which sensor column feeds which modulator row
        pick = np.zeros((len(SENSORS), len(body)))
        for m, col in enumerate(body):
            if col >= 0:
                pick[col, m] = 1.0
        self.register_buffer("body_pick", f(pick))
        #: plastic projections, as the brain has them (see AnatomicalBrain.learn)
        self.tick_s = float(brain.tick_s)
        self.plastic = [dict(p, pre_mask_t=f(p["pre_mask"])) for p in brain.plastic]
        #: the brain's levels (``dynamics.sweep``): units, pooling, which of a
        #: level's units are muscles and where those sit in the action
        self.sweep, self.balance_steps = bool(brain.sweep), int(brain.BALANCE_STEPS)
        self.levels = []
        where_out = {int(u_): j for j, u_ in enumerate(brain.out_units)}
        for lv in brain.levels:
            pos = [j for j, u_ in enumerate(lv["idx"]) if int(u_) in where_out]
            self.levels.append({"idx": torch.as_tensor(lv["idx"], dtype=torch.long), "pool": f(lv["pool"]),
                                "outpos": torch.as_tensor(pos, dtype=torch.long),
                                "actcols": torch.as_tensor([where_out[int(lv["idx"][j])] for j in pos],
                                                           dtype=torch.long),
                                "np_idx": lv["idx"]})
        if self.levels:
            cat = np.concatenate([lv["idx"] for lv in brain.levels])
            self.register_buffer("inv", torch.as_tensor(np.argsort(cat), dtype=torch.long))
        self.theta = nn.Parameter(f(_inv_softplus(np.abs(brain.W0)) * (brain.S != 0)))
        self.w_loc = nn.Parameter(f(brain.W0 * brain.local_mask))
        self.w_in = nn.Parameter(f(brain.Win))
        self.bias = nn.Parameter(f(brain.bias))
        self.log_std = nn.Parameter(torch.full((len(brain.out_units),), float(log_std_init)))
        self.critic = nn.Sequential(nn.Linear(N + I, 96), nn.Tanh(), nn.Linear(96, 1))

    def weights(self):
        """``(W, Win)`` as the simulation uses them, with the spec's wiring and
        signs enforced by construction."""
        W = self.S * F.softplus(self.theta) * self.wired + self.w_loc * self.local
        return W, self.w_in * self.in_mask

    def initial_fast(self, n: int) -> list:
        return [torch.zeros(n, p["post"].stop - p["post"].start, p["pre"].stop - p["pre"].start)
                for p in self.plastic]

    @staticmethod
    def _above(rate, threshold):
        return ((rate - threshold) / max(1e-6, 1.0 - threshold)).clamp(0.0, 1.0)

    def _sweep_tick(self, r, o, drive_t, gain, act_t, alive, fast, Wl):
        """One tick of ``dynamics.sweep``: -> (every cell's drive before any
        muscle noise, the new rates, what each plastic source cell sent)."""
        cur, xs, srcs = r, [], [None] * len(self.plastic)
        for lv, Wt in zip(self.levels, Wl):
            idx = lv["idx"]
            e = cur @ Wt + drive_t[:, idx]
            for k, (p, Fp) in enumerate(zip(self.plastic, fast)):
                if p["post"].start in lv["np_idx"]:
                    src = self._above(cur[:, p["pre"]], p["pre_threshold"]) * p["pre_mask_t"]
                    srcs[k] = src
                    loc = np.searchsorted(lv["np_idx"], np.arange(p["post"].start, p["post"].stop))
                    extra = torch.zeros_like(e)
                    extra[:, loc] = torch.einsum("bij,bj->bi", Fp, src)
                    e = e + extra
            g = gain[:, idx]
            has_out = act_t is not None and len(lv["outpos"]) > 0
            a_out = torch.tanh(F.relu(act_t[:, lv["actcols"]])) if has_out else None
            with torch.no_grad():                 # the balance point is found, not differentiated
                lo, hi = torch.zeros_like(e), torch.ones_like(e)
                for _ in range(self.balance_steps):
                    mid = 0.5 * (lo + hi)
                    rs = torch.tanh(F.relu(g * (e - self.k_inh * mid)))
                    if has_out:                   # the muscles did what they were recorded doing
                        rs[:, lv["outpos"]] = a_out
                    up = rs @ lv["pool"].T > mid
                    lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
                m = 0.5 * (lo + hi)
            # the same m, now with the derivative it really has with respect to what drives the area
            x0 = g * (e - self.k_inh * m)
            rs0 = torch.tanh(F.relu(x0))
            if has_out:
                rs0 = rs0.index_copy(1, lv["outpos"], a_out)
            s = rs0 @ lv["pool"].T
            with torch.no_grad():
                slope = (1.0 - rs0 ** 2) * (x0 > 0) * g
                if has_out:
                    slope[:, lv["outpos"]] = 0.0
                denom = 1.0 + self.k_inh * (slope @ lv["pool"].T)
            m = m + (s - s.detach()) / denom
            x = g * (e - self.k_inh * m)
            xs.append(x)
            rs = torch.tanh(F.relu(x))
            if has_out:
                rs = rs.index_copy(1, lv["outpos"], a_out)
            val = r[:, idx] + self.alpha * (rs - r[:, idx])
            if alive is not None:
                val = val * alive[:, idx]
            cur = cur.index_copy(1, idx, val)
        return torch.cat(xs, dim=1)[:, self.inv], cur, srcs

    def forward(self, obs, r0, act, alive=None, fast=None, return_fast=False, trace=None):
        """``obs`` (T, B, sensors), ``r0`` (B, units), ``act`` (T, B, muscle
        units: the drive each muscle unit actually had) -> the mean drive the
        network gave those units (T, B, .), value (T, B), final state.
        ``alive`` (B, units) of 1 and 0 keeps an animal's dead cells at zero.
        ``fast`` is each animal's experience-written weights at the start
        (zeros if not given); ``return_fast=True`` also returns them as they
        are at the end."""
        if self.plastic and fast is None:
            fast = self.initial_fast(obs.shape[1])
        fast = [f_.detach() for f_ in (fast or [])]
        W, Win = self.weights()
        Wt, srct = (W - self.k_inh * self.pool).T, self.src.T
        T = obs.shape[0]
        body = obs.clamp(0, 1) @ self.body_pick                      # (T, B, M); zero where an area is the source
        drive = obs @ Win.T + self.bias                              # the senses' part, every tick at once
        r, mus, vals = r0, [], []
        Wl = [W[lv["idx"]].T for lv in self.levels]
        for t in range(T):
            o = obs[t]
            gain = 1.0 + (r @ srct + body[t]) @ self.dens
            if self.sweep:
                x, new, srcs = self._sweep_tick(r, o, drive[t], gain, None if act is None else act[t],
                                                alive, fast, Wl)
                mus.append(x[:, self.out_idx])
                vals.append(self.critic(torch.cat([r, o], dim=1)).squeeze(-1))
                if trace is not None:
                    trace.append((new.detach(), x.detach() if act is None
                                  else x.detach().index_copy(1, self.out_idx, act[t])))
                self._learn(fast, srcs, r, new, o)
                r = new
                continue
            xin = r @ Wt + drive[t]
            srcs = []
            for p, Fp in zip(self.plastic, fast):
                src = self._above(r[:, p["pre"]], p["pre_threshold"]) * p["pre_mask_t"]
                srcs.append(src)
                extra = torch.zeros_like(xin)
                extra[:, p["post"]] = torch.einsum("bij,bj->bi", Fp, src)
                xin = xin + extra
            x = gain * xin
            mus.append(x[:, self.out_idx])
            vals.append(self.critic(torch.cat([r, o], dim=1)).squeeze(-1))
            if act is not None:
                x = x.index_copy(1, self.out_idx, act[t])
            new = r + self.alpha * (torch.tanh(F.relu(x)) - r)
            if alive is not None:
                new = new * alive
            if trace is not None:
                trace.append((new.detach(), x.detach()))
            self._learn(fast, srcs, r, new, o)
            r = new
        if return_fast:
            return torch.stack(mus), torch.stack(vals), r, fast
        return torch.stack(mus), torch.stack(vals), r

    def _learn(self, fast, srcs, r, new, o) -> None:
        """One tick of experience written into ``fast`` (a list, changed in
        place): the twin of ``AnatomicalBrain.learn``."""
        if not self.plastic:
            return
        with torch.no_grad():                # experience, not something training reaches into
            for k, (p, src) in enumerate(zip(self.plastic, srcs)):
                g = torch.ones(r.shape[0])
                for kind, ref in p["gates"]:
                    level = o[:, ref].clamp(0, 1) if kind == "body" else r[:, ref].mean(dim=1)
                    g = g * self._above(level, p["gate_threshold"])
                Fp = fast[k]
                Fp = Fp + self.tick_s * (p["rate"] * g[:, None, None] * new[:, p["post"], None]
                                         * src.detach()[:, None, :] - p["decay"] * Fp)
                Fp = Fp.clamp(min=0.0)
                total = Fp.sum(dim=2, keepdim=True)
                fast[k] = Fp * (p["cap"] / total.clamp(min=1e-9)).clamp(max=1.0)

    def export(self) -> dict:
        W, Win = self.weights()
        out = {"W": W, "Win": Win, "bias": self.bias, "log_std": self.log_std,
               "c0w": self.critic[0].weight, "c0b": self.critic[0].bias,
               "c2w": self.critic[2].weight, "c2b": self.critic[2].bias}
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in out.items()}

    def load_critic(self, w: dict) -> None:
        if "c0w" in w and tuple(w["c0w"].shape) == tuple(self.critic[0].weight.shape):
            with torch.no_grad():
                self.critic[0].weight.copy_(torch.as_tensor(w["c0w"]))
                self.critic[0].bias.copy_(torch.as_tensor(w["c0b"]))
                self.critic[2].weight.copy_(torch.as_tensor(w["c2w"]))
                self.critic[2].bias.copy_(torch.as_tensor(w["c2b"]))
        if "log_std" in w and len(w["log_std"]) == len(self.log_std):
            with torch.no_grad():
                self.log_std.copy_(torch.as_tensor(w["log_std"]))


def _worker(args):
    weights, meta, seeds, cur_d = args
    return nursery.rollout(weights, meta, seeds, cur_d)


class AnatomyTrainer:
    """Owns the learner, the worker pool and the log."""

    def __init__(self, spec_name: str, hp: AnatomyHyper, cur: Curriculum, out_dir: str,
                 name: str = "anatomy", resume: str | None = None):
        self.hp, self.cur, self.out_dir, self.name, self.spec_name = hp, cur, out_dir, name, spec_name
        os.makedirs(out_dir, exist_ok=True)
        torch.manual_seed(hp.seed)
        # Gradients carried back through many ticks of saturated units shrink
        # to subnormal floats, which the processor handles about ten times
        # slower than ordinary ones. Treating them as zero changes the
        # weights by less than 1e-7 and makes the update about nine times
        # faster (measured 2026-11-25: 46 s -> 5 s).
        torch.set_flush_denormal(True)
        self.spec = anatomy.load_spec(spec_name)
        self.agent_steps, start = 0, None
        if resume:
            start, meta = anatomy.load_weights(resume)
            if meta.get("anatomy") != spec_name:
                raise ValueError(f"{resume} was raised for {meta.get('anatomy')!r}, not {spec_name!r}")
            self.agent_steps = int(meta.get("agent_steps", 0))
            hp.wiring_seed = int(meta.get("wiring_seed", hp.wiring_seed))
        brain = anatomy.AnatomicalBrain(self.spec, seed=hp.wiring_seed, tick_s=cur.tick_s, weights=start)
        if hp.torch_threads > 0:
            torch.set_num_threads(int(hp.torch_threads))
        self.net = TorchAnatomy(brain, hp.log_std_init)
        if start is not None:
            self.net.load_critic(start)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=hp.lr, eps=1e-5)
        self.iteration = 0
        self._seed_base = hp.seed * 1_000_003 + self.agent_steps % 1_000_003
        self.pool = mp.get_context("spawn").Pool(hp.workers) if hp.workers > 1 else None
        self.log_path = os.path.join(out_dir, f"{name}.training_log.jsonl")
        self.t0 = time.time()

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def meta(self) -> dict:
        return {"format": WEIGHTS_FORMAT, "kind": "anatomy", "anatomy": self.spec_name, "name": self.name,
                "wiring_seed": int(self.hp.wiring_seed), "sensors": list(SENSORS), "tick_s": self.cur.tick_s,
                "agent_steps": int(self.agent_steps), "trained": True, "species": self.cur.species,
                "curriculum": self.cur.to_dict(), "hyper": asdict(self.hp)}

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(self.out_dir, f"{self.name}.npz")
        return anatomy.save_weights(path, self.net.export(), self.meta())

    # ---- one iteration -------------------------------------------------- #
    def collect(self) -> dict:
        hp = self.hp
        w, meta = self.net.export(), self.meta()
        base = self._seed_base + self.iteration * 100_003
        seeds = [base + k for k in range(hp.worlds)]
        cur_d = self.cur.to_dict()
        if self.pool is None:
            parts = [nursery.rollout(w, meta, seeds, cur_d)]
        else:
            split = [list(s) for s in np.array_split(seeds, hp.workers) if len(s)]
            parts = self.pool.map(_worker, [(w, meta, s, cur_d) for s in split])
        return {"obs": np.concatenate([p["obs"] for p in parts], axis=1),
                "u": np.concatenate([p["u"] for p in parts], axis=1),
                "rew": np.concatenate([p["rew"] for p in parts], axis=1),
                "stats": np.concatenate([p["stats"] for p in parts], axis=0),
                "alive": np.concatenate([p["alive"] for p in parts], axis=0)}

    def update(self, data: dict) -> dict:
        hp, net = self.hp, self.net
        for key in ("obs", "u", "rew"):
            if not np.isfinite(data[key]).all():
                raise FloatingPointError(f"the nursery returned a non-finite value in '{key}'")
        obs, u, rew = (torch.as_tensor(data[k]) for k in ("obs", "u", "rew"))
        alive = torch.as_tensor(data["alive"])          # (A, units): the cells each animal had
        T, A = rew.shape
        L = hp.chunk
        n_chunks = T // L
        T = n_chunks * L
        obs, u, rew = obs[:T], u[:T], rew[:T]

        with torch.no_grad():                 # score the lives with the brain that lived them
            r = torch.zeros(A, net.n)
            r_start = torch.zeros(n_chunks, A, net.n)
            fast = net.initial_fast(A)                  # every nursery life starts with nothing written
            fast_start = [torch.zeros(n_chunks, *f_.shape) for f_ in fast]
            mu = torch.zeros(T, A, u.shape[2])
            val = torch.zeros(T, A)
            for c in range(n_chunks):
                r_start[c] = r
                for k, f_ in enumerate(fast):
                    fast_start[k][c] = f_
                sl = slice(c * L, (c + 1) * L)
                mu[sl], val[sl], r, fast = net(obs[sl], r, u[sl], alive, fast=fast, return_fast=True)
            logp_old = torch.distributions.Normal(mu, net.log_std.exp()).log_prob(u).sum(-1)
            nxt = torch.cat([val[1:], val[-1:]], dim=0)       # time ran out; nothing ended
            delta = rew + hp.gamma * nxt - val
            adv = torch.zeros_like(rew)
            run = torch.zeros(A)
            for t in range(T - 1, -1, -1):
                run = delta[t] + hp.gamma * hp.lam * run
                adv[t] = run
            ret = adv + val

        def chunked(x):                       # (T, A, ...) -> (L, chunks*A, ...)
            x = x.reshape(n_chunks, L, A, *x.shape[2:])
            return x.transpose(0, 1).reshape(L, n_chunks * A, *x.shape[3:])

        obs_c, u_c, adv_c, ret_c = chunked(obs), chunked(u), chunked(adv), chunked(ret)
        logp_c, val_c = chunked(logp_old), chunked(val)
        r_c = r_start.reshape(n_chunks * A, net.n)
        fast_c = [f_.reshape(n_chunks * A, *f_.shape[2:]) for f_ in fast_start]
        alive_c = alive.repeat(n_chunks, 1)             # sequence c * A + a is animal a
        n_seq = n_chunks * A
        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n_upd, stop = 0, False
        for _ in range(hp.epochs):
            if stop:
                break
            for idx in torch.randperm(n_seq).chunk(hp.minibatches):
                mu_n, v_n, _ = net(obs_c[:, idx], r_c[idx], u_c[:, idx], alive_c[idx],
                                   fast=[f_[idx] for f_ in fast_c])
                dist = torch.distributions.Normal(mu_n, net.log_std.exp())
                logp = dist.log_prob(u_c[:, idx]).sum(-1)
                ratio = (logp - logp_c[:, idx]).clamp(-20.0, 20.0).exp()
                a = adv_c[:, idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - hp.clip, 1 + hp.clip) * a).mean()
                v_old = val_c[:, idx]
                v_clip = v_old + (v_n - v_old).clamp(-hp.clip, hp.clip)
                tgt = ret_c[:, idx]
                v_loss = 0.5 * torch.max((v_n - tgt) ** 2, (v_clip - tgt) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + hp.value_coef * v_loss - hp.entropy * ent
                if self.iteration < hp.critic_warmup:        # the brain is not touched yet
                    loss = hp.value_coef * v_loss + 0.0 * (pg + ent)
                self.opt.zero_grad()
                loss.backward()
                if self.iteration < hp.critic_warmup:
                    for p_ in (net.theta, net.w_loc, net.w_in, net.bias, net.log_std):
                        p_.grad = None
                gnorm = nn.utils.clip_grad_norm_(net.parameters(), hp.max_grad)
                if not (torch.isfinite(loss) and torch.isfinite(gnorm)):
                    self.opt.zero_grad()              # leave the weights as they are
                    stats["skipped"] = stats.get("skipped", 0.0) + 1.0
                    continue
                self.opt.step()
                with torch.no_grad():
                    net.log_std.clamp_(-2.5, hp.log_std_max)
                    kl = (logp_c[:, idx] - logp).mean().item()
                    stats["pg"] += pg.item()
                    stats["v"] += v_loss.item()
                    stats["ent"] += ent.item()
                    stats["kl"] += kl
                    stats["clipfrac"] += ((ratio - 1).abs() > hp.clip).float().mean().item()
                n_upd += 1
                if kl > hp.target_kl and self.iteration >= hp.critic_warmup:
                    stop = True
                    break
        skipped = stats.pop("skipped", 0.0)
        out = {k: v / max(1, n_upd) for k, v in stats.items()}
        out["skipped"] = skipped
        out["updates"] = n_upd
        out["explained_var"] = float(1 - (ret - val).var() / (ret.var() + 1e-8))
        out["return"] = float(rew.sum(0).mean())
        return out

    def step(self) -> dict:
        t0 = time.time()
        data = self.collect()
        t1 = time.time()
        info = self.update(data)
        t2 = time.time()
        T, A = data["rew"].shape
        self.agent_steps += T * A
        self.iteration += 1
        info.update(nursery.summarise(data["stats"]))
        info.update(iteration=self.iteration, agent_steps=self.agent_steps, animals=A,
                    sec_collect=round(t1 - t0, 2), sec_update=round(t2 - t1, 2),
                    elapsed_min=round((time.time() - self.t0) / 60, 2))
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(info) + "\n")
        return info
