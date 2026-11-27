"""Proximal policy optimisation for an ABMA brain.

The learner and the animal are the same network written twice: once in numpy
(:class:`fnt.abma.core.brain.Brain`, which is what runs inside a simulation and
inside the rollout workers) and once here in PyTorch, which is the only place
gradients are needed. :meth:`TorchBrain.export` and :meth:`TorchBrain.load_brain`
move the weights between them, name for name.

The algorithm is ordinary recurrent PPO (Schulman et al. 2017) with generalised
advantage estimation. Each iteration:

1. worker processes run one episode in each of a batch of freshly generated
   nursery worlds (:mod:`.nursery`), every animal driven by the current brain;
2. the trajectories come back, a value function scores them, and the policy is
   nudged toward the actions that turned out better than expected — in chunks
   short enough to backpropagate through time;
3. the new weights go back to the workers.

One brain is shared by every animal in every world, so each animal's rivals are
copies of itself: social behaviour, where it appears, is learned in self-play.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from ..core.brain import Brain, BRAIN_FORMAT
from ..core.senses import SENSORS, MODULATORS, SenseParams
from . import nursery
from .nursery import Curriculum


@dataclass
class Hyper:
    """Learning settings. Defaults suit a laptop and a small recurrent net."""
    n_sensory: int = 64
    n_rnn: int = 64
    worlds: int = 96             # nursery worlds per iteration
    workers: int = 8             # rollout processes
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 6e-4
    # Collecting experience on the real engine is the expensive part, so each
    # batch is worked harder than a fast simulator would justify: more passes
    # and smaller minibatches, held in check by the KL stop below.
    epochs: int = 6
    minibatches: int = 12
    target_kl: float = 0.03      # stop an iteration's updates past this
    chunk: int = 32              # steps of backpropagation through time
    entropy: float = 0.005
    value_coef: float = 0.5
    max_grad: float = 0.5
    max_turn: float = 1.5        # radians per tick
    #: spread of the initial neuromodulator weights (see TorchBrain)
    gain_init: float = 1.5
    #: threads the weight update may use (0 = PyTorch's default, every core).
    #: On a machine that is busy with other work, every core is too many: the
    #: update's threads wait on each other and an update that takes seconds
    #: takes minutes.
    torch_threads: int = 0
    seed: int = 0


class TorchBrain(nn.Module):
    """The trainable twin of :class:`fnt.abma.core.brain.Brain`."""

    def __init__(self, n_sensory: int = 64, n_rnn: int = 64,
                 gain_init: float = 1.5, n_out: int = 2):
        super().__init__()
        own = {name: i for i, name in enumerate(SENSORS)}
        self.register_buffer(
            "mod_idx", torch.tensor([own[m] for m in MODULATORS]))
        self.W1 = nn.Linear(len(SENSORS), n_sensory)
        self.Wg = nn.Linear(len(MODULATORS), n_sensory)
        self.gru = nn.GRU(n_sensory, n_rnn)
        # turn, speed and, for a brain raised where animals can hurt each
        # other, a bite
        self.Wo = nn.Linear(n_rnn, n_out)
        self.value = nn.Linear(n_rnn, 1)
        self.log_std = nn.Parameter(torch.full((n_out,), -0.5))
        nn.init.orthogonal_(self.W1.weight, math.sqrt(2))
        nn.init.orthogonal_(self.Wo.weight, 0.01)
        nn.init.orthogonal_(self.value.weight, 1.0)
        for lin in (self.W1, self.Wg, self.Wo, self.value):
            nn.init.zeros_(lin.bias)
        if n_out > 2:
            with torch.no_grad():           # biting starts rare, not half the time
                self.Wo.bias[2:] = -1.0
        # Every sensory unit is born with its own random sensitivity to each
        # neuromodulator — some are turned up by hunger, some down, some
        # ignore it — the way a receptor is expressed in some cells and not
        # others. Which sensitivity ends up *used* for what is learned.
        # Starting them all at zero instead (every gain exactly 1) left the
        # body's needs with almost no say: after 4M animal-ticks gains still
        # varied by a few percent, and a thirsty animal chose food as often
        # as a hungry one did.
        nn.init.normal_(self.Wg.weight, 0.0, float(gain_init))
        with torch.no_grad():               # gain 1 for a middling animal
            self.Wg.bias.copy_(-0.5 * self.Wg.weight.sum(dim=1))

    def forward(self, obs, h0):
        """``obs`` (T, B, n_sensors), ``h0`` (1, B, n_rnn) -> mean motor
        command (T, B, n_out), value (T, B), final state."""
        gain = 2.0 * torch.sigmoid(self.Wg(obs[..., self.mod_idx]))
        s = torch.tanh(self.W1(obs)) * gain
        out, hT = self.gru(s, h0)
        return self.Wo(out), self.value(out).squeeze(-1), hT

    # ---- to and from the numpy brain ------------------------------------ #
    def export(self, meta: dict) -> Brain:
        g = self.gru
        w = {
            "W1": self.W1.weight, "b1": self.W1.bias,
            "Wg": self.Wg.weight, "bg": self.Wg.bias,
            "W_ih": g.weight_ih_l0, "W_hh": g.weight_hh_l0,
            "b_ih": g.bias_ih_l0, "b_hh": g.bias_hh_l0,
            "Wo": self.Wo.weight, "bo": self.Wo.bias,
            "log_std": self.log_std,
            # the critic rides along so training can resume from a brain file;
            # the simulation never reads it
            "Wv": self.value.weight, "bv": self.value.bias,
        }
        return Brain({k: v.detach().cpu().numpy() for k, v in w.items()}, meta)

    def load_brain(self, brain: Brain, wake: tuple = ()) -> None:
        """Take up an existing brain's weights, matched by name.

        The body may have gained senses, neuromodulators or a motor output
        since that brain was raised. Its own inputs go to the columns that now
        carry them; a sense it never had starts unconnected (zero weights), so
        it behaves as it did until it learns otherwise; an output it never had
        keeps this network's fresh, quiet initialisation.
        """
        w = {k: torch.as_tensor(v) for k, v in brain.w.items()}
        s_now = {name: i for i, name in enumerate(SENSORS)}
        m_now = {name: i for i, name in enumerate(MODULATORS)}
        s_cols = [s_now[s] for s in brain.sensors]
        gone = [m for m in brain.modulators if m not in m_now]
        if gone:
            raise ValueError(f"this brain uses neuromodulators the body no longer has: {gone}")
        m_cols = [m_now[m] for m in brain.modulators]
        # ``wake``: neuromodulators this training is about; of those, the ones the brain never had
        m_new = [i for i, name in enumerate(MODULATORS) if i not in m_cols and name in wake]
        k = min(w["Wo"].shape[0], self.Wo.out_features)
        with torch.no_grad():
            self.W1.weight.zero_()
            self.W1.weight[:, s_cols] = w["W1"]
            self.W1.bias.copy_(w["b1"])
            fresh = self.Wg.weight.clone()
            self.Wg.weight.zero_()
            self.Wg.weight[:, m_cols] = w["Wg"]
            self.Wg.bias.copy_(w["bg"])
            if m_new:
                # A neuromodulator the body has gained since: every sensory
                # unit gets its own random sensitivity to it, as a new brain's
                # units do (half strength, and centred so that a middling
                # level leaves the gains where they were). Left at zero, a new
                # need has almost no say in what the animal does for millions
                # of ticks; see the note on ``gain_init`` above.
                self.Wg.weight[:, m_new] = 0.5 * fresh[:, m_new]
                self.Wg.bias -= 0.3 * self.Wg.weight[:, m_new].sum(dim=1)
            self.gru.weight_ih_l0.copy_(w["W_ih"])
            self.gru.weight_hh_l0.copy_(w["W_hh"])
            self.gru.bias_ih_l0.copy_(w["b_ih"])
            self.gru.bias_hh_l0.copy_(w["b_hh"])
            self.Wo.weight[:k] = w["Wo"][:k]
            self.Wo.bias[:k] = w["bo"][:k]
            self.log_std[:k] = w["log_std"][:k]
            if "Wv" in w:
                self.value.weight.copy_(w["Wv"]); self.value.bias.copy_(w["bv"])


def _brain_meta(hp: Hyper, cur: Curriculum, name: str, extra: dict) -> dict:
    return {
        "format": BRAIN_FORMAT, "name": name, "sensors": list(SENSORS),
        "modulators": list(MODULATORS), "tick_s": cur.tick_s,
        "max_turn": hp.max_turn, "senses": SenseParams().to_dict(),
        "trained": True, "species": cur.species,
        "curriculum": cur.to_dict(), "hyper": asdict(hp), **extra,
    }


def _worker(args):
    weights, meta, seeds, cur_d = args
    return nursery.rollout(weights, meta, seeds, cur_d)


class Trainer:
    """Owns the learner, the worker pool and the log."""

    def __init__(self, hp: Hyper, cur: Curriculum, out_dir: str,
                 name: str = "brain", resume: Brain | None = None):
        self.hp, self.cur, self.out_dir, self.name = hp, cur, out_dir, name
        os.makedirs(out_dir, exist_ok=True)
        torch.manual_seed(hp.seed)
        # gradients through many ticks shrink to subnormal floats, which are
        # about ten times slower to compute with; treat them as zero (as the
        # anatomy trainer does)
        torch.set_flush_denormal(True)
        if hp.torch_threads > 0:
            torch.set_num_threads(int(hp.torch_threads))
        self.net = TorchBrain(hp.n_sensory, hp.n_rnn, hp.gain_init,
                              n_out=3 if cur.contest else 2)
        self.agent_steps = 0
        if resume is not None:
            self.net.load_brain(resume, wake=("tired", "cold", "lonely") if cur.life else ())
            self.agent_steps = int(resume.meta.get("agent_steps", 0))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=hp.lr, eps=1e-5)
        self.iteration = 0
        self._seed_base = hp.seed * 1_000_003 + self.agent_steps % 1_000_003
        self.pool = (mp.get_context("spawn").Pool(hp.workers)
                     if hp.workers > 1 else None)
        self.log_path = os.path.join(out_dir, "training_log.jsonl")
        self.t0 = time.time()

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    # ---- brains --------------------------------------------------------- #
    def brain(self) -> Brain:
        return self.net.export(_brain_meta(
            self.hp, self.cur, self.name,
            {"agent_steps": self.agent_steps, "iterations": self.iteration,
             "trained_on": time.strftime("%Y-%m-%d")}))

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(self.out_dir, f"{self.name}.npz")
        return self.brain().save(path)

    # ---- one iteration -------------------------------------------------- #
    def collect(self) -> dict:
        hp = self.hp
        brain = self.brain()
        base = self._seed_base + self.iteration * 100_003
        seeds = [base + k for k in range(hp.worlds)]
        cur_d = self.cur.to_dict()
        if self.pool is None:
            parts = [nursery.rollout(brain.w, brain.meta, seeds, cur_d)]
        else:
            split = [list(s) for s in np.array_split(seeds, hp.workers)
                     if len(s)]
            parts = self.pool.map(
                _worker, [(brain.w, brain.meta, s, cur_d) for s in split])
        return {"obs": np.concatenate([p["obs"] for p in parts], axis=1),
                "u": np.concatenate([p["u"] for p in parts], axis=1),
                "rew": np.concatenate([p["rew"] for p in parts], axis=1),
                "stats": np.concatenate([p["stats"] for p in parts], axis=0)}

    def update(self, data: dict) -> dict:
        hp, net = self.hp, self.net
        for key in ("obs", "u", "rew"):
            if not np.isfinite(data[key]).all():
                raise FloatingPointError(
                    f"the nursery returned a non-finite value in '{key}': a "
                    "sense, a need or the reward is broken, and training on "
                    "it would silently destroy the brain")
        obs = torch.as_tensor(data["obs"])            # (T, A, I)
        u = torch.as_tensor(data["u"])                # (T, A, 2)
        rew = torch.as_tensor(data["rew"])            # (T, A)
        T, A = rew.shape
        L = hp.chunk
        n_chunks = T // L
        T = n_chunks * L
        obs, u, rew = obs[:T], u[:T], rew[:T]

        # ---- score the trajectories with the brain that produced them ---- #
        with torch.no_grad():
            h = torch.zeros(1, A, hp.n_rnn)
            h_start = torch.zeros(n_chunks, A, hp.n_rnn)
            mu = torch.zeros(T, A, net.Wo.out_features)
            val = torch.zeros(T, A)
            for c in range(n_chunks):
                h_start[c] = h[0]
                sl = slice(c * L, (c + 1) * L)
                mu[sl], val[sl], h = net(obs[sl], h)
            std = net.log_std.exp()
            logp_old = torch.distributions.Normal(mu, std).log_prob(u).sum(-1)
            # an episode ends because time ran out, not because anything
            # happened, so the last value stands in for the one after it
            nxt = torch.cat([val[1:], val[-1:]], dim=0)
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

        obs_c, u_c = chunked(obs), chunked(u)
        adv_c, ret_c = chunked(adv), chunked(ret)
        logp_c, val_c = chunked(logp_old), chunked(val)
        h_c = h_start.reshape(1, n_chunks * A, hp.n_rnn)
        n_seq = n_chunks * A

        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n_upd = 0
        stop = False
        for _ in range(hp.epochs):
            if stop:
                break
            perm = torch.randperm(n_seq)
            for idx in perm.chunk(hp.minibatches):
                mu_n, v_n, _ = net(obs_c[:, idx], h_c[:, idx])
                dist = torch.distributions.Normal(mu_n, net.log_std.exp())
                logp = dist.log_prob(u_c[:, idx]).sum(-1)
                ratio = (logp - logp_c[:, idx]).exp()
                a = adv_c[:, idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = -torch.min(ratio * a,
                                ratio.clamp(1 - hp.clip, 1 + hp.clip) * a).mean()
                v_old = val_c[:, idx]
                v_clip = v_old + (v_n - v_old).clamp(-hp.clip, hp.clip)
                r = ret_c[:, idx]
                v_loss = 0.5 * torch.max((v_n - r) ** 2,
                                         (v_clip - r) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + hp.value_coef * v_loss - hp.entropy * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), hp.max_grad)
                self.opt.step()
                with torch.no_grad():
                    net.log_std.clamp_(-2.5, 0.5)
                    stats["pg"] += pg.item()
                    stats["v"] += v_loss.item()
                    stats["ent"] += ent.item()
                    kl = (logp_c[:, idx] - logp).mean().item()
                    stats["kl"] += kl
                    stats["clipfrac"] += ((ratio - 1).abs() > hp.clip
                                          ).float().mean().item()
                n_upd += 1
                if kl > hp.target_kl:
                    stop = True
                    break
        out = {k: v / max(1, n_upd) for k, v in stats.items()}
        out["updates"] = n_upd
        var = ret.var()
        out["explained_var"] = float(1 - (ret - val).var() / (var + 1e-8))
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
        info.update(iteration=self.iteration, agent_steps=self.agent_steps,
                    animals=A, sec_collect=round(t1 - t0, 2),
                    sec_update=round(t2 - t1, 2),
                    log_std=[round(float(x), 3)
                             for x in self.net.log_std.detach()],
                    elapsed_min=round((time.time() - self.t0) / 60, 2))
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(info) + "\n")
        return info
