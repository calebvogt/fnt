"""Give every cell of an anatomy brain a working resting level before it is raised.

    python -m fnt.abma.learn.settle --anatomy prairie_vole --name prairie_vole_settled

The untrained network (random weights inside the spec's wiring) leaves some
areas switched off for good and others pinned on. In the first prairie vole
draft the lateral hypothalamus was below threshold in 100% of cell-ticks, the
VTA in 83%, the PVN in 70%: no dopamine, no oxytocin, and no learning signal
through any of them, because a cell below threshold passes no gradient.

Here each cell's resting drive (its bias, a trained quantity anyway) is moved
until its mean rate over nursery lives sits at a target: the stand-in for the
homeostatic regulation of intrinsic excitability. Weights are not touched.

Targets, as a fraction of maximum rate: sense cells 0.15, every other area
0.25; muscles: the two TURN units equal (no built-in circling), SPEED 0.35 (it
walks), BITE 0.05, bladder wall 0.10, sphincter 0.50.

The result is written beside trained anatomy weights; raise or teach from it
with ``--resume NAME`` (:mod:`fnt.abma.learn`) or ``--start NAME``
(:mod:`fnt.abma.learn.imitate`).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..core import anatomy
from ..core.senses import SENSORS
from . import nursery
from .anatomy_ppo import AnatomyHyper, TorchAnatomy
from .nursery import Curriculum

#: resting rates of the muscles (fraction of maximum)
MUSCLE = {"TURN": 0.30, "SPEED": 0.35, "BITE": 0.05, "DET": 0.10, "EUS": 0.50}
SENSE, OTHER = 0.15, 0.25


def rates(net: TorchAnatomy, data: dict):
    """Mean rate of every cell over the lives in ``data`` and how often its
    drive was below threshold, by replaying them through the trainer's twin
    (whatever the spec's dynamics are)."""
    obs, u = torch.as_tensor(data["obs"]), torch.as_tensor(data["u"])
    alive = torch.as_tensor(data["alive"])
    T, A = obs.shape[:2]
    trace = []
    with torch.no_grad():
        net(obs, torch.zeros(A, net.n), u, alive, trace=trace)
    keep = trace[T // 8:]                        # let it settle from the all-zero start
    m = torch.stack([r_.mean(0) for r_, _ in keep]).mean(0)
    below = torch.stack([(x_ <= 0).float().mean(0) for _, x_ in keep]).mean(0)
    return m.numpy(), below.numpy()


def settle(spec_name: str, name: str = "", iterations: int = 24, worlds: int = 6,
           seed: int = 0, quiet: bool = False) -> str:
    """Settle ``spec_name`` and write the weights; returns their path."""
    spec = anatomy.load_spec(spec_name)
    name = name or f"{spec_name}_settled"
    cur = Curriculum(species=spec.get("species", "House mouse"))
    cur.tick_s = float(spec.get("dynamics", {}).get("tick_s", 0.25))
    brain = anatomy.AnatomicalBrain(spec, seed=seed, tick_s=cur.tick_s)
    net = TorchAnatomy(brain, AnatomyHyper().log_std_init)
    target = np.zeros(brain.n)
    for r_ in spec["regions"]:
        target[brain.sl[r_["id"]]] = MUSCLE.get(r_["id"], SENSE if r_["class"] == "sense" else OTHER)
    meta = {"kind": "anatomy", "anatomy": spec_name, "wiring_seed": seed}
    for it in range(iterations):
        data = nursery.rollout(net.export(), meta, [5000 + 10 * it + k for k in range(worlds)], cur.to_dict())
        m, below = rates(net, data)
        err = target - m
        if not quiet:
            areas = {i: float(np.abs(err[brain.sl[i]]).mean()) for i in brain.ids}
            worst = sorted(areas, key=areas.get)[-4:]
            print(f"it {it:2d}  mean |rate - target| {np.abs(err).mean():.3f}  cells below threshold >90% of the "
                  f"time: {int((below > 0.9).sum())}/{brain.n}  worst areas: "
                  + ", ".join(f"{i} {areas[i]:.2f}" for i in worst), flush=True)
        # big steps first, then smaller ones: areas in a loop can flip together
        scale = 1.0 if it < iterations * 0.4 else 0.5 if it < iterations * 0.7 else 0.25
        with torch.no_grad():
            net.bias += torch.as_tensor(np.clip(1.2 * err, -0.4, 0.4) * scale, dtype=torch.float32)
    arrays = net.export()
    if not quiet:
        m, below = rates(net, nursery.rollout(arrays, meta, list(range(7000, 7008)), cur.to_dict()))
        print("\nafter settling: area  mean rate  share of cell-ticks below threshold")
        for i in brain.ids:
            s = brain.sl[i]
            print(f"  {i:6s} {m[s].mean():.2f}  {below[s].mean():.2f}")
    path = os.path.join(anatomy.user_dir(), name + ".npz")
    anatomy.save_weights(path, arrays, {
        "format": 1, "kind": "anatomy", "anatomy": spec_name, "name": name, "wiring_seed": seed,
        "sensors": list(SENSORS), "tick_s": cur.tick_s, "agent_steps": 0, "trained": False,
        "species": cur.species,
        "settled": {"targets": {"sense": SENSE, "other": OTHER, **MUSCLE}, "iterations": iterations}})
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m fnt.abma.learn.settle", description=__doc__.split("\n\n")[0])
    p.add_argument("--anatomy", required=True, metavar="SPEC")
    p.add_argument("--name", default="", help="name of the weights to write (default SPEC_settled)")
    p.add_argument("--iterations", type=int, default=24)
    p.add_argument("--worlds", type=int, default=6, help="nursery worlds per iteration")
    p.add_argument("--wiring-seed", type=int, default=0)
    a = p.parse_args(argv)
    print("wrote", settle(a.anatomy, a.name, a.iterations, a.worlds, a.wiring_seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
