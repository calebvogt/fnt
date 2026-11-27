"""Teach an anatomy brain by example.

    python -m fnt.abma.learn.imitate --name vole_anat --teacher vole_doorways --start prairie_vole_settled --own 0.5

Reward alone has not so far taught a brain wired from anatomy to forage in the
time we can run (2026-11: flat after 30M ticks). This does: a generic learned
brain (the teacher, itself raised by reward alone) lives nursery lives; the
anatomy brain is shown what the teacher sensed, tick by tick, and its weights
are moved, inside the spec's wiring and signs exactly as in reinforcement
learning, until its own muscles command what the teacher commanded.
Supervised, so the gradient is exact.

``--own X`` lets the student live a share of the lives itself, with the
teacher saying what it would have done there. Without it the student only
ever sees the places a competent animal goes, and is lost the first time its
own mistake takes it elsewhere (measured: hungry-at-food 0.10 -> 0.39 in five
iterations once half the lives were its own).

Every few iterations the student is put in fresh nursery worlds on its own
and scored: "ON ITS OWN" is the number that matters. Senses the teacher does
not have are zero in what the student is shown on the teacher's lives.

A brain started this way has to be described as such. Reward-only training
(``python -m fnt.abma.learn --anatomy SPEC --resume NAME --critic-warmup 8``)
continues from it and has been seen to hold and improve it.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from ..core import anatomy
from ..core.brain import Brain, find_brain
from ..core.senses import SENSORS
from . import nursery
from .anatomy_ppo import TorchAnatomy
from .nursery import Curriculum

REST = {"BITE": 0.05, "DET": 0.10, "EUS": 0.50}     # muscles the teacher does not have: keep them where they were
TURN_BASE = 0.30


def _teach(args):
    w, meta, seeds, cur_d = args
    return nursery.rollout(w, meta, seeds, cur_d)


def main():
    ap = argparse.ArgumentParser(prog="python -m fnt.abma.learn.imitate",
                                 description="Teach an anatomy brain by example.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--spec", default="prairie_vole")
    ap.add_argument("--teacher", default="vole_doorways")
    ap.add_argument("--start", default="prairie_vole_settled",
                    help="anatomy weights to start from (see fnt.abma.learn.settle)")
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--boxes", type=float, default=0.0)
    ap.add_argument("--anosmic", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--life", action="store_true", help="worlds in which animals tire, get cold and attach")
    ap.add_argument("--bond-in-brain", action="store_true",
                    help="with --life: lives begin with the dealt pairs held together (see nursery)")
    ap.add_argument("--attached", type=float, default=0.5)
    ap.add_argument("--own", type=float, default=0.0,
                    help="share of each iteration's worlds that the STUDENT lives in (the teacher then says what it "
                         "would have done there): a student shown only the teacher's lives never sees the places "
                         "its own mistakes take it")
    a = ap.parse_args()
    torch.set_num_threads(a.torch_threads)
    torch.set_flush_denormal(True)

    spec = anatomy.load_spec(a.spec)
    cur = Curriculum(species=spec.get("species", "House mouse"), boxes=a.boxes, anosmic=a.anosmic,
                     life=a.life, bond_in_brain=a.bond_in_brain, attached=a.attached)
    cur.tick_s = float(spec.get("dynamics", {}).get("tick_s", 0.25))
    teacher = Brain.load(find_brain(a.teacher))
    t_w = teacher.weights if hasattr(teacher, "weights") else teacher.w
    take = np.asarray(teacher._take)
    start, meta0 = anatomy.load_weights(a.start)
    brain = anatomy.AnatomicalBrain(spec, seed=int(meta0.get("wiring_seed", 0)), tick_s=cur.tick_s, weights=start)
    assert brain.alpha == 1.0, "the muscle targets below assume a unit follows its drive within a tick"
    net = TorchAnatomy(brain)
    params = [net.theta, net.w_loc, net.w_in, net.bias]
    opt = torch.optim.Adam(params, lr=a.lr, eps=1e-5)
    out_names = [r["id"] for r in spec["regions"] if r["class"] == "output" for _ in range(r["n"])]
    col = {n: [j for j, m in enumerate(out_names) if m == n] for n in set(out_names)}
    rest_t = torch.tensor([REST.get(n, 0.0) for n in out_names])
    rest_w = torch.tensor([0.1 if n in REST else 0.0 for n in out_names])
    out_dir = anatomy.user_dir()
    log_path = os.path.join(out_dir, a.name + ".imitation_log.jsonl")
    s_meta = {"format": 1, "kind": "anatomy", "anatomy": a.spec, "name": a.name,
              "wiring_seed": int(meta0.get("wiring_seed", 0)), "sensors": list(SENSORS), "tick_s": cur.tick_s,
              "agent_steps": 0, "trained": True, "species": cur.species,
              "imitated": {"teacher": a.teacher, "start": a.start}}
    pool = mp.get_context("spawn").Pool(a.workers)
    t0, it = time.time(), 0
    print(f"teaching '{a.name}' ({a.spec}, from {a.start}) by the example of '{a.teacher}': {a.worlds} worlds x "
          f"{cur.steps} ticks of {cur.tick_s:g} s per iteration", flush=True)
    try:
        while (time.time() - t0) / 60.0 < a.minutes:
            it += 1
            for g_ in opt.param_groups:               # the step shrinks to a fifth over the run
                g_["lr"] = a.lr * (1.0 - 0.8 * min(1.0, (time.time() - t0) / 60.0 / a.minutes))
            seeds = [10_000_019 * it + k for k in range(a.worlds)]
            n_own = int(round(a.own * a.worlds)) if it > 1 else 0
            jobs = []
            if n_own:                                 # the student's own lives (a little noise, to see more places)
                arrays = net.export()
                arrays["log_std"] = np.full_like(arrays["log_std"], -1.5)
                jobs += [(arrays, dict(s_meta), list(s), cur.to_dict())
                         for s in np.array_split(seeds[:n_own], max(1, a.workers // 2)) if len(s)]
            n_first = len(jobs)
            jobs += [(t_w, teacher.meta, list(s), cur.to_dict())
                     for s in np.array_split(seeds[n_own:], a.workers) if len(s)]
            parts = pool.map(_teach, jobs)
            fulls = []
            for k, p in enumerate(parts):
                if k < n_first:
                    fulls.append(p["obs"])            # an anatomy brain's rollout records every sense
                else:
                    f_ = np.zeros((*p["obs"].shape[:2], len(SENSORS)), np.float32)
                    f_[:, :, take] = p["obs"]
                    fulls.append(f_)
            full = np.concatenate(fulls, axis=1)
            T, A = full.shape[:2]
            # what the teacher commands, tick by tick, for exactly these senses (no exploration noise)
            h = np.zeros((A, teacher.n_rnn), np.float32)
            t_u = np.zeros((T, A, 2), np.float32)
            for t in range(T):
                motor, h, _ = teacher.forward(full[t], h)
                t_u[t] = motor[:, :2]
            obs = torch.as_tensor(full)
            u = torch.as_tensor(t_u)
            turn = (teacher.max_turn / brain.max_turn) * torch.tanh(u[..., 0])     # what the teacher's body did,
            speed = torch.sigmoid(u[..., 1])                                       # in the student's units
            target = rest_t.repeat(T, A, 1)
            target[..., col["TURN"][0]] = TURN_BASE - 0.5 * turn
            target[..., col["TURN"][1]] = TURN_BASE + 0.5 * turn
            for j in col["SPEED"]:
                target[..., j] = speed
            weight = rest_w.repeat(T, A, 1)
            weight[..., col["TURN"]] = 1.0
            weight[..., col["SPEED"]] = 1.0

            L = a.chunk
            n_chunks = T // L
            with torch.no_grad():                     # where every chunk starts, with the student as it is now
                r = torch.zeros(A, net.n)
                r_start = torch.zeros(n_chunks, A, net.n)
                fast = net.initial_fast(A)           # what each animal's own experience has written so far
                fast_start = [torch.zeros(n_chunks, *f_.shape) for f_ in fast]
                for c in range(n_chunks):
                    r_start[c] = r
                    for k, f_ in enumerate(fast):
                        fast_start[k][c] = f_
                    _, _, r, fast = net(obs[c * L:(c + 1) * L], r, None, fast=fast, return_fast=True)

            def chunked(x):
                x = x[:n_chunks * L].reshape(n_chunks, L, A, *x.shape[2:])
                return x.transpose(0, 1).reshape(L, n_chunks * A, *x.shape[3:])

            obs_c, tgt_c, w_c = chunked(obs), chunked(target), chunked(weight)
            r_c = r_start.reshape(n_chunks * A, net.n)
            fast_c = [f_.reshape(n_chunks * A, *f_.shape[2:]) for f_ in fast_start]
            tot, n_upd = 0.0, 0
            for _ in range(a.epochs):
                for idx in torch.randperm(n_chunks * A).chunk(a.minibatches):
                    mu, _, _ = net(obs_c[:, idx], r_c[idx], None, fast=[f_[idx] for f_ in fast_c])
                    rate = torch.tanh(torch.relu(mu))
                    tg, ww = tgt_c[:, idx], w_c[:, idx]
                    # a muscle cell below threshold passes no gradient: pull it up when it should be firing
                    loss = (ww * ((rate - tg) ** 2 + 0.1 * torch.relu(-mu) * (tg > 0.02))).sum(-1).mean()
                    opt.zero_grad()
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                    if torch.isfinite(loss) and torch.isfinite(gn):
                        opt.step()
                        tot += float(loss.detach())
                        n_upd += 1
            with torch.no_grad():                     # how well it now does what the teacher did
                r = torch.zeros(A, net.n)
                fast = net.initial_fast(A)
                rates = []
                for c in range(n_chunks):
                    mu, _, r, fast = net(obs[c * L:(c + 1) * L], r, None, fast=fast, return_fast=True)
                    rates.append(torch.tanh(torch.relu(mu)))
                rate = torch.cat(rates)
                s_turn = rate[..., col["TURN"][1]] - rate[..., col["TURN"][0]]
                s_speed = rate[..., col["SPEED"]].mean(-1)
                tt, ts = turn[:n_chunks * L], speed[:n_chunks * L]
                r2 = lambda y, p: float(1 - ((y - p) ** 2).mean() / y.var())     # noqa: E731
                info = {"iteration": it, "loss": tot / max(1, n_upd), "r2_turn": r2(tt, s_turn),
                        "r2_speed": r2(ts, s_speed), "elapsed_min": round((time.time() - t0) / 60, 2)}
            line = (f"it {it:3d}  loss {info['loss']:.4f}  turn R2 {info['r2_turn']:+.3f}  speed R2 "
                    f"{info['r2_speed']:+.3f}")
            if it % a.eval_every == 0 or it == 1:     # the student on its own, in fresh worlds
                arrays = net.export()
                arrays["log_std"] = np.full_like(arrays["log_std"], -2.0)
                anatomy.save_weights(os.path.join(out_dir, a.name + ".npz"), arrays, dict(s_meta, agent_steps=it * T * A))
                ev = [list(s) for s in np.array_split([777_000 + k for k in range(24)], a.workers)]
                st = np.concatenate([p["stats"] for p in pool.map(
                    _teach, [(arrays, dict(s_meta), s, cur.to_dict()) for s in ev])], axis=0)
                sm = nursery.summarise(st)
                info.update({k: float(sm[k]) for k in (
                    "food_time_hungry", "food_time_sated", "water_time_thirsty", "right_choice", "speed",
                    "rest_tired", "rest_fresh", "touch", "touch_anosmic", "partner_share_intact",
                    "partner_share_anosmic", "lonely_attached", "cold") if k in sm})
                line += (f"   ON ITS OWN: hungry at food {info['food_time_hungry']:.3f}  thirsty at water "
                         f"{info['water_time_thirsty']:.3f}  right choice {info['right_choice']:.3f}  speed "
                         f"{info['speed']:.3f}")
                if a.life:
                    line += (f"  rest tired/fresh {info['rest_tired']:.2f}/{info['rest_fresh']:.2f}  touch "
                             f"{info['touch']:.2f}  with partner nose/none {info['partner_share_intact']:.2f}/"
                             f"{info['partner_share_anosmic']:.2f}  lonely {info['lonely_attached']:.0f}")
            print(line + f"   [{info['elapsed_min']:.0f} min]", flush=True)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(info) + "\n")
    finally:
        pool.close()
    print("wrote", os.path.join(out_dir, a.name + ".npz"))


if __name__ == "__main__":
    main()
