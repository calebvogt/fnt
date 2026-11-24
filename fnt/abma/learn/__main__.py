"""Train a brain from the command line.

    python -m fnt.abma.learn --name my_vole --minutes 20
    python -m fnt.abma.learn --name my_vole --resume my_vole --minutes 10

The brain is written to ``~/ABMA_brains/<name>.npz`` (or ``--out``) every few
iterations, so a run can be stopped at any point and what it had learned so far
is already on disk. Ask for it in an experiment with ``--brain <name>``.
"""
from __future__ import annotations

import argparse
import sys
import time

from ..core.brain import Brain, find_brain, user_dir
from .nursery import Curriculum
from .ppo import Hyper, Trainer


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m fnt.abma.learn",
        description="Raise an ABMA brain by reinforcement in nursery worlds.")
    p.add_argument("--name", default="brain", help="Name of the brain to write.")
    p.add_argument("--out", default=user_dir(),
                   help="Folder for the brain and its training log.")
    p.add_argument("--resume", help="Start from an existing brain (name or path).")
    p.add_argument("--anatomy", metavar="SPEC",
                   help="Train a brain wired from this anatomy spec instead "
                        "(e.g. c57_male): the wiring and the sign of every "
                        "projection stay as the spec gives them, and only "
                        "the weights inside them learn. Ticks are 0.25 s. "
                        "Use it afterwards with: python -m fnt.abma --anatomy "
                        "SPEC --anatomy-weights NAME")
    p.add_argument("--wiring-seed", type=int, default=0,
                   help="With --anatomy: the random draw training starts from.")
    p.add_argument("--minutes", type=float, default=20.0,
                   help="Wall-clock training budget.")
    p.add_argument("--iterations", type=int, default=0,
                   help="Stop after this many iterations (0 = use --minutes).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--worlds", type=int, default=96,
                   help="Nursery worlds per iteration.")
    p.add_argument("--steps", type=int, default=1024,
                   help="Brain ticks per episode.")
    p.add_argument("--species", default="Prairie vole")
    p.add_argument("--animals", default="2,4,8",
                   help="Group sizes to raise it in, comma-separated.")
    p.add_argument("--arena", default="1.6,4.0",
                   help="Enclosure side range in metres, min,max.")
    p.add_argument("--solitary", action="store_true",
                   help="No contests or matings: foraging only.")
    p.add_argument("--one-resource", type=float, default=0.5,
                   help="Fraction of worlds with only food or only water.")
    p.add_argument("--boxes", type=float, default=0.5,
                   help="Fraction of worlds whose food is in walled boxes "
                        "with doorways.")
    p.add_argument("--anosmic", type=float, default=0.25,
                   help="Fraction of animals raised with no sense of smell.")
    p.add_argument("--born-at-resource", type=float, default=0.2,
                   help="Fraction of animals put down at food or water. A "
                        "nose-less animal that starts at water while hungry "
                        "has to learn to leave it and search.")
    p.add_argument("--box-water", type=float, default=0.5,
                   help="Chance that a box holds water as well as food.")
    p.add_argument("--towers", type=float, default=0.5,
                   help="Chance that open water is a solid tower rather than "
                        "a patch to walk over.")
    p.add_argument("--born-at-door", type=float, default=0.15,
                   help="Fraction of animals put down just outside a "
                        "doorway of a box.")
    p.add_argument("--contest", action="store_true",
                   help="Raise it in worlds where animals can bite, sprint, "
                        "tire, hide under huts and die of their injuries. The "
                        "brain gets a third output, the bite. Resuming a brain "
                        "raised without it is fine: it starts out not biting.")
    p.add_argument("--huts", type=float, default=0.6,
                   help="With --contest: fraction of worlds that have huts "
                        "(cover an animal cannot be bitten under).")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--entropy", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=5)
    a = p.parse_args(argv)

    cur = Curriculum(
        species=a.species, steps=a.steps, social=not a.solitary,
        animals=tuple(int(x) for x in a.animals.split(",")),
        arena_m=tuple(float(x) for x in a.arena.split(",")),
        one_resource=a.one_resource, boxes=a.boxes, anosmic=a.anosmic,
        born_at_door=a.born_at_door, born_at_resource=a.born_at_resource,
        box_water=a.box_water, towers=a.towers,
        contest=a.contest, huts=a.huts)
    if a.anatomy:
        return _raise_anatomy(a, cur)
    hp = Hyper(workers=a.workers, worlds=a.worlds, lr=a.lr,
               entropy=a.entropy, seed=a.seed)
    resume = Brain.load(find_brain(a.resume)) if a.resume else None
    tr = Trainer(hp, cur, a.out, name=a.name, resume=resume)
    deadline = time.time() + a.minutes * 60.0
    print(f"raising '{a.name}' in {a.out}  ({hp.worlds} worlds x {cur.steps} "
          f"ticks per iteration, {hp.workers} workers)", flush=True)
    try:
        while True:
            info = tr.step()
            print(f"it {info['iteration']:4d}  {info['agent_steps'] / 1e6:7.2f}M "
                  f"ret {info['return']:+7.3f}  "
                  f"food|hungry {info['food_time_hungry']:.2f} "
                  f"food|sated {info['food_time_sated']:.2f}  "
                  f"water|thirsty {info['water_time_thirsty']:.2f} "
                  f"water|sated {info['water_time_sated']:.2f}  "
                  f"right {info['right_choice']:.2f}  "
                  f"box nose/none {info['box_found_intact']:.2f}/"
                  f"{info['box_found_anosmic']:.2f}  "
                  f"dH {info['d_hunger']:+5.1f} dT {info['d_thirst']:+5.1f}  "
                  f"spd {info['speed']:.2f} "
                  + (f"bites m/f {info['bites_male']:.1f}/{info['bites_female']:.1f} "
                     f"bitten {info['bitten']:.2f} died {info['died']:.3f} "
                     f"health {info['health']:.0f}  " if a.contest else "")
                  + f"ev {info['explained_var']:+.2f} "
                  f"kl {info['kl']:.3f} "
                  f"[{info['sec_collect']:.0f}s+{info['sec_update']:.0f}s]",
                  flush=True)
            if info["iteration"] % a.save_every == 0:
                tr.save()
            if a.iterations and info["iteration"] >= a.iterations:
                break
            if not a.iterations and time.time() >= deadline:
                break
    except KeyboardInterrupt:
        print("stopped.")
    finally:
        path = tr.save()
        tr.close()
    print(f"wrote {path}")
    return 0




def _raise_anatomy(a, cur) -> int:
    """Train a brain wired from an anatomy spec (see ``learn/anatomy_ppo``)."""
    import time

    from ..core import anatomy
    from .anatomy_ppo import AnatomyHyper, AnatomyTrainer

    cur.tick_s = 0.25                          # the network's own tick
    if a.species == "Prairie vole":
        cur.species = "House mouse"
    out = a.out if a.out != user_dir() else anatomy.user_dir()
    hp = AnatomyHyper(workers=a.workers, worlds=a.worlds, lr=min(a.lr, 3e-4), entropy=min(a.entropy, 0.002),
                      seed=a.seed, wiring_seed=a.wiring_seed)
    tr = AnatomyTrainer(a.anatomy, hp, cur, out, name=a.name, resume=a.resume)
    print(f"raising '{a.name}' on anatomy '{a.anatomy}' in {out}  ({hp.worlds} worlds x {cur.steps} ticks of "
          f"{cur.tick_s:g} s, {hp.workers} workers)", flush=True)
    t0, path = time.time(), None
    try:
        while True:
            info = tr.step()
            print(f"it {info['iteration']:4d}  {info['agent_steps'] / 1e6:7.2f}M ticks  "
                  f"return {info['return']:+7.3f}  hungry at food {info.get('food_time_hungry', float('nan')):.3f}  "
                  f"thirsty at water {info.get('water_time_thirsty', float('nan')):.3f}  kl {info['kl']:.4f}  "
                  f"collect {info['sec_collect']:.0f}s update {info['sec_update']:.0f}s", flush=True)
            if info["iteration"] % max(1, a.save_every) == 0:
                path = tr.save()
            if a.iterations and info["iteration"] >= a.iterations:
                break
            if not a.iterations and (time.time() - t0) / 60.0 >= a.minutes:
                break
    except KeyboardInterrupt:
        print("stopped.")
    finally:
        path = tr.save()
        tr.close()
    print(f"wrote {path}")
    print(f"use it: python -m fnt.abma --anatomy {a.anatomy} --anatomy-weights {a.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
