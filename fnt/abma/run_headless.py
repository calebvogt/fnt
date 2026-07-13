"""Headless / batch runner for ABMA — run experiments without the GUI.

Examples
--------
Write a starter config you can edit::

    python -m fnt.abma --write-default my_experiment.json

Run an experiment from a config, into a project folder, and analyse it::

    python -m fnt.abma my_experiment.json --out ~/ABMA_projects --analyze

Override a few fields for a quick test or a batch sweep::

    python -m fnt.abma my_experiment.json --out ~/runs --trials 5 --days 3 --parallel
"""
from __future__ import annotations

import argparse
import os
import sys

from .core.config import ExperimentConfig, default_vole_experiment
from .core.runner import run_experiment


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m fnt.abma",
        description="Run an ABMA in-silico experiment from the command line.")
    p.add_argument("config", nargs="?",
                   help="Path to an experiment config JSON (from the GUI or "
                        "--write-default).")
    p.add_argument("--out", default=os.path.join(os.path.expanduser("~"),
                                                  "ABMA_projects"),
                   help="Parent folder; a subfolder named after the experiment "
                        "is created inside it.")
    p.add_argument("--write-default", metavar="PATH",
                   help="Write the default vole config to PATH and exit.")
    p.add_argument("--trials", type=int, help="Override number of trials.")
    p.add_argument("--days", type=float, help="Override duration in days.")
    p.add_argument("--seed", type=int, help="Override base random seed.")
    p.add_argument("--parallel", action="store_true",
                   help="Run trials in parallel (no live view).")
    p.add_argument("--analyze", action="store_true",
                   help="Run built-in socio-spatial analysis afterwards.")
    args = p.parse_args(argv)

    if args.write_default:
        default_vole_experiment().to_json(args.write_default)
        print(f"Wrote default config to {args.write_default}")
        return 0

    if not args.config:
        p.error("a config JSON is required (or use --write-default)")

    cfg = ExperimentConfig.from_json(args.config)
    if args.trials is not None:
        cfg.n_trials = args.trials
    if args.days is not None:
        cfg.days = args.days
    if args.seed is not None:
        cfg.seed = args.seed
    if args.parallel:
        cfg.parallel = True

    project_dir = os.path.join(args.out, cfg.name)
    run_experiment(cfg, project_dir, log_cb=print, analyze=args.analyze)
    print(f"\nProject written to {project_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
