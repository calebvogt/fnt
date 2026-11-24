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
    wiring_seed: int = 0         # which random draw training starts from
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

    def forward(self, obs, r0, act):
        """``obs`` (T, B, sensors), ``r0`` (B, units), ``act`` (T, B, muscle
        units: the drive each muscle unit actually had) -> the mean drive the
        network gave those units (T, B, .), value (T, B), final state."""
        W, Win = self.weights()
        Wt, srct = (W - self.k_inh * self.pool).T, self.src.T
        T = obs.shape[0]
        body = obs.clamp(0, 1) @ self.body_pick                      # (T, B, M); zero where an area is the source
        drive = obs @ Win.T + self.bias                              # the senses' part, every tick at once
        r, mus, vals = r0, [], []
        for t in range(T):
            o = obs[t]
            gain = 1.0 + (r @ srct + body[t]) @ self.dens
            x = gain * (r @ Wt + drive[t])
            mus.append(x[:, self.out_idx])
            vals.append(self.critic(torch.cat([r, o], dim=1)).squeeze(-1))
            x = x.index_copy(1, self.out_idx, act[t])
            r = r + self.alpha * (torch.tanh(F.relu(x)) - r)
        return torch.stack(mus), torch.stack(vals), r

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
        self.spec = anatomy.load_spec(spec_name)
        self.agent_steps, start = 0, None
        if resume:
            start, meta = anatomy.load_weights(resume)
            if meta.get("anatomy") != spec_name:
                raise ValueError(f"{resume} was raised for {meta.get('anatomy')!r}, not {spec_name!r}")
            self.agent_steps = int(meta.get("agent_steps", 0))
            hp.wiring_seed = int(meta.get("wiring_seed", hp.wiring_seed))
        brain = anatomy.AnatomicalBrain(self.spec, seed=hp.wiring_seed, tick_s=cur.tick_s, weights=start)
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
                "stats": np.concatenate([p["stats"] for p in parts], axis=0)}

    def update(self, data: dict) -> dict:
        hp, net = self.hp, self.net
        obs, u, rew = (torch.as_tensor(data[k]) for k in ("obs", "u", "rew"))
        T, A = rew.shape
        L = hp.chunk
        n_chunks = T // L
        T = n_chunks * L
        obs, u, rew = obs[:T], u[:T], rew[:T]

        with torch.no_grad():                 # score the lives with the brain that lived them
            r = torch.zeros(A, net.n)
            r_start = torch.zeros(n_chunks, A, net.n)
            mu = torch.zeros(T, A, u.shape[2])
            val = torch.zeros(T, A)
            for c in range(n_chunks):
                r_start[c] = r
                sl = slice(c * L, (c + 1) * L)
                mu[sl], val[sl], r = net(obs[sl], r, u[sl])
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
        n_seq = n_chunks * A
        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n_upd, stop = 0, False
        for _ in range(hp.epochs):
            if stop:
                break
            for idx in torch.randperm(n_seq).chunk(hp.minibatches):
                mu_n, v_n, _ = net(obs_c[:, idx], r_c[idx], u_c[:, idx])
                dist = torch.distributions.Normal(mu_n, net.log_std.exp())
                logp = dist.log_prob(u_c[:, idx]).sum(-1)
                ratio = (logp - logp_c[:, idx]).exp()
                a = adv_c[:, idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a, ratio.clamp(1 - hp.clip, 1 + hp.clip) * a).mean()
                v_old = val_c[:, idx]
                v_clip = v_old + (v_n - v_old).clamp(-hp.clip, hp.clip)
                tgt = ret_c[:, idx]
                v_loss = 0.5 * torch.max((v_n - tgt) ** 2, (v_clip - tgt) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + hp.value_coef * v_loss - hp.entropy * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), hp.max_grad)
                self.opt.step()
                with torch.no_grad():
                    net.log_std.clamp_(-2.5, 0.5)
                    kl = (logp_c[:, idx] - logp).mean().item()
                    stats["pg"] += pg.item()
                    stats["v"] += v_loss.item()
                    stats["ent"] += ent.item()
                    stats["kl"] += kl
                    stats["clipfrac"] += ((ratio - 1).abs() > hp.clip).float().mean().item()
                n_upd += 1
                if kl > hp.target_kl:
                    stop = True
                    break
        out = {k: v / max(1, n_upd) for k, v in stats.items()}
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
