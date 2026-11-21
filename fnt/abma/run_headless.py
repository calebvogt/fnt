"""Headless / batch runner for ABMA — run experiments without the GUI.

Examples
--------
Write a starter config you can edit::

    python -m fnt.abma --write-default my_experiment.json

Run an experiment from a config, into a project folder, and analyse it::

    python -m fnt.abma my_experiment.json --out ~/ABMA_projects --analyze

Override a few fields for a quick test or a batch sweep::

    python -m fnt.abma my_experiment.json --out ~/runs --trials 5 --days 3 --parallel

Give the animals a learned brain, and ask what one of them would have done if
it had been less hungry six hours in::

    python -m fnt.abma --list-brains
    python -m fnt.abma --preset vole --brain vole_forager --days 2 --out ~/runs
    python -m fnt.abma --preset vole --twin hunger-30@0 --out ~/runs

Run the saline-vs-methimazole dose-response — the design ABMA exists for —
without touching the GUI::

    python -m fnt.abma --anosmia-study --out ~/studies --doses 0,0.5,0.75,1 \
        --replicates 4 --days 3

Arms are paired by seed, so replicate *r* starts identically in every arm.
Results land in ``results/metrics_long.csv`` (tidy, for R) and
``results/comparison.csv``. The primary readout is ``mean_dist_MM``: emergent
male-male spacing, which nothing in the config prescribes.
"""
from __future__ import annotations

import argparse
import os
import sys

from .core.config import ExperimentConfig, default_vole_experiment
from .core.runner import run_experiment


def study_base(args) -> ExperimentConfig | None:
    """The cohort and enclosure an ``--anosmia-study`` compares its arms in.

    The study used to accept only a config file. It now takes the same
    one-line description an ordinary run does (``--preset``, ``--males``,
    ``--females``, ``--season``) and ``--brain``, so "3 males and 3 females in
    this enclosure for 5 days, two replicates intact and two on methimazole"
    is one command. ``None`` means nothing was described: the study's own
    default (the vole cage) applies.
    """
    from .core.compose import design

    described = args.preset or args.males or args.females
    if not (args.config or described or args.brain):
        return None
    cfg = design(
        preset=args.preset,
        base=(ExperimentConfig.from_json(args.config) if args.config
              else None),
        males=args.males, females=args.females, species=args.species,
        days=args.days, seed=args.seed, name=args.name, season=args.season,
        year=args.year) if (args.config or described) else None
    if cfg is None:
        from .core.presets import vole_anosmia
        cfg = vole_anosmia()
    if args.brain:
        from .core.brain import find_brain
        find_brain(args.brain)                 # raises with the list of names
        cfg.brain.enabled = True
        cfg.brain.name = args.brain
    return cfg


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
    p.add_argument("--anosmia-study", action="store_true",
                   help="Run the methimazole dose-response study instead of a "
                        "single experiment. Uses CONFIG as the base if given, "
                        "otherwise the prairie-vole preset.")
    p.add_argument("--doses", default="0,0.5,0.75,1.0",
                   help="Comma-separated methimazole doses (0 = saline).")
    p.add_argument("--replicates", type=int, default=4,
                   help="Replicates per condition (paired seeds across arms).")
    p.add_argument("--scalar-nose", action="store_true",
                   help="Use the scalar recognition gate instead of the "
                        "receptor/signature olfactory model.")

    d = p.add_argument_group(
        "describe a run",
        "Build a config from a description instead of a JSON file. Combine "
        "with --watch to open it on screen, or run it headless as usual.")
    d.add_argument("--preset", help="Arena/paradigm to start from; matched "
                                    "loosely, e.g. 'voleterra'.")
    d.add_argument("--list-presets", action="store_true",
                   help="Print the available presets and exit.")
    d.add_argument("--species", default="Prairie vole",
                   help="Species for --males/--females (default: Prairie vole).")
    d.add_argument("--males", type=int, default=0,
                   help="Replace the cohort with this many males.")
    d.add_argument("--females", type=int, default=0,
                   help="Replace the cohort with this many females.")
    d.add_argument("--season", choices=["winter", "spring", "summer", "fall",
                                        "autumn"],
                   help="Release date, which at a real site sets day length, "
                        "solar angle and grass growth together.")
    d.add_argument("--year", type=int, default=2026,
                   help="Year for --season (default 2026).")
    d.add_argument("--name", help="Name the experiment (sets the run folder).")
    d.add_argument("--save-config", metavar="PATH",
                   help="Write the composed config to PATH as well.")
    d.add_argument("--watch", action="store_true",
                   help="Open the ABMA window and run it on screen.")
    d.add_argument("--no-autostart", action="store_true",
                   help="With --watch, load the design but do not start it.")
    d.add_argument("--dry-run", action="store_true",
                   help="Print what would run, then exit without running.")
    m = p.add_argument_group(
        "movie",
        "Film the start of a run instead of running the experiment: the "
        "enclosure from above, each animal with a trail, hunger and thirst "
        "underneath. Written to <out>/<name>/movie.gif.")
    m.add_argument("--movie", nargs="?", const="gif", choices=["gif", "mp4"],
                   help="Write an animation (gif by default; mp4 needs "
                        "ffmpeg).")
    m.add_argument("--movie-hours", type=float, default=0.5,
                   help="How much of the run to film, from release "
                        "(default 0.5 h).")
    m.add_argument("--movie-deprived", type=float, metavar="LEVEL",
                   help="Release every animal this hungry and thirsty "
                        "(0-100), so foraging starts at once.")
    m.add_argument("--movie-vs-mmz", action="store_true",
                   help="Side by side: the cohort as described, and the same "
                        "animals after methimazole.")
    b = p.add_argument_group(
        "learned brains",
        "Drive the animals with a trained neural network instead of the "
        "rule-based policy, and ask counterfactual questions of them.")
    b.add_argument("--brain", metavar="NAME",
                   help="Give every animal this learned brain (a name from "
                        "--list-brains, or a path to a .npz).")
    b.add_argument("--list-brains", action="store_true",
                   help="Print the brains that can be asked for, and exit.")
    b.add_argument("--twin", action="append", metavar="TWEAK",
                   help="Live one life twice: run to --fork-hours, copy the "
                        "world, apply TWEAK to the copy, and compare. E.g. "
                        "'hunger-30@0' (animal 0 thirty points less hungry), "
                        "'smell_ability=0@males', 'rnn:12=1@0~10' (clamp unit "
                        "12 on for 10 s; needs --brain). Repeatable.")
    b.add_argument("--assay", metavar="WHICH",
                   help="Bench-test the animals instead of running an "
                        "experiment: 'approach' (find the food: hungry, sated, "
                        "anosmic), 'choice' (food or water, hungry vs "
                        "thirsty), 'search' (area-restricted search after "
                        "food is lost), or 'all'. Needs no config; add "
                        "--brain to test a learned brain, omit it to test the "
                        "rule-based animal. 'vitals' runs your config for "
                        "--days and reports whether the cohort feeds and "
                        "waters itself.")
    b.add_argument("--assay-trials", type=int, default=40,
                   help="Animals per condition in a bench assay (default 40).")
    b.add_argument("--fork-hours", type=float, default=6.0,
                   help="When the two lives part (default 6 h into the run).")
    b.add_argument("--horizon-hours", type=float, default=6.0,
                   help="How long both lives are followed (default 6 h).")
    args = p.parse_args(argv)

    if args.list_brains:
        from .core.brain import Brain, list_brains
        found = list_brains()
        if not found:
            print("No brains yet. Train one with: python -m fnt.abma.learn")
        for name, path in found:
            b = Brain.load(path)
            print(f"{name}\n    {b.describe()}\n    {path}")
            cu = b.command_units(3)
            print("    units one synapse from the muscles — turn: "
                  + ", ".join(f"{u} ({'left' if w > 0 else 'right'})"
                              for u, w in cu["turn"])
                  + "; speed: "
                  + ", ".join(f"{u} ({'faster' if w > 0 else 'slower'})"
                              for u, w in cu["speed"]))
        return 0

    if args.assay and args.assay != "vitals":
        from .core import assays
        try:
            make = assays.policy_factory(args.brain)
            _, report = assays.bench(make, args.assay,
                                     trials=args.assay_trials,
                                     species=args.species,
                                     seed=args.seed or 0)
        except (ValueError, FileNotFoundError) as e:
            p.error(str(e))
        print(f"{args.species}, "
              + (f"learned brain '{args.brain}'" if args.brain
                 else "rule-based") + "\n")
        print(report)
        return 0

    if args.list_presets:
        from .core.presets import all_presets
        for pr in all_presets():
            print(f"{pr.name}\n    {pr.description}")
        return 0

    if args.write_default:
        default_vole_experiment().to_json(args.write_default)
        print(f"Wrote default config to {args.write_default}")
        return 0

    if args.anosmia_study:
        from .core.study import anosmia_study, run_study

        try:
            base = study_base(args)
        except (ValueError, FileNotFoundError) as e:
            p.error(str(e))
        try:
            doses = tuple(float(d) for d in args.doses.split(",") if d.strip())
        except ValueError:
            p.error(f"--doses must be numbers, got {args.doses!r}")
        if len(doses) < 2:
            p.error("--doses needs at least two levels to compare")
        study = anosmia_study(base=base, doses=doses,
                              replicates=args.replicates, days=args.days,
                              mechanistic_nose=not args.scalar_nose)
        if args.name:
            study.name = args.name
        if args.seed is not None:
            study.base_seed = args.seed
        study_dir = os.path.join(args.out, study.name)
        if base is not None:
            from .core.compose import summary as _summary
            base.n_trials = args.replicates     # what the study will run
            print(_summary(base))
        print(f"{len(doses)} arms (methimazole dose "
              f"{', '.join(f'{d:g}' for d in doses)}) x {args.replicates} "
              f"replicate(s), {study.base.days:g} days each"
              + (f", learned brain '{args.brain}'" if args.brain else ""))
        if args.dry_run:
            return 0
        run_study(study, study_dir, log_cb=print)
        print(f"\nStudy written to {study_dir}")
        print("  results/metrics_long.csv   tidy long table, for R")
        print("  results/comparison.csv     per-metric arm vs reference")
        return 0

    described = args.preset or args.males or args.females
    if not args.config and not described:
        p.error("a config JSON is required (or --preset/--males/--females, "
                "or --write-default)")

    from .core.compose import design, summary

    try:
        cfg = design(
            preset=args.preset,
            base=(ExperimentConfig.from_json(args.config) if args.config
                  else None),
            males=args.males, females=args.females, species=args.species,
            days=args.days, trials=args.trials, seed=args.seed,
            name=args.name, season=args.season, year=args.year)
    except ValueError as e:
        p.error(str(e))
    if args.parallel:
        cfg.parallel = True
    if args.brain:
        from .core.brain import find_brain
        try:
            find_brain(args.brain)
        except FileNotFoundError as e:
            p.error(str(e))
        cfg.brain.enabled = True
        cfg.brain.name = args.brain
        if not args.name:
            cfg.name = f"{cfg.name}_brain"
    if args.save_config:
        # the target folder often does not exist yet — asking someone to mkdir
        # before saving a config they just described is pure friction
        parent = os.path.dirname(os.path.abspath(args.save_config))
        os.makedirs(parent, exist_ok=True)
        cfg.to_json(args.save_config)
        print(f"Config written to {args.save_config}")

    print(summary(cfg))
    if args.dry_run:
        return 0

    if args.movie:
        from .core import movie

        out_dir = os.path.join(args.out, cfg.name)
        os.makedirs(out_dir, exist_ok=True)
        seconds = max(float(cfg.dt), args.movie_hours * 3600.0)
        # about 300 frames whatever the length, so a longer film plays faster
        # rather than growing
        sample = max(float(cfg.dt), seconds / 300.0)
        prep = (movie.arrive_deprived(args.movie_deprived)
                if args.movie_deprived is not None else None)
        nervous = (f"learned brain '{cfg.brain.name}'" if cfg.brain.enabled
                   else "rule-based")
        clips = [movie.film(cfg, seconds, sample_s=sample, prepare=prep,
                            title="Intact" if args.movie_vs_mmz else cfg.name)]
        if args.movie_vs_mmz:
            clips.append(movie.film(movie.anosmic(cfg), seconds,
                                    sample_s=sample, prepare=prep,
                                    title="Anosmic (methimazole)"))
        n = clips[0].xy.shape[1]
        path = movie.render(
            clips, os.path.join(out_dir, "movie." + args.movie),
            caption=f"{n} animals, {nervous}")
        print("wrote", path)
        return 0

    if args.assay == "vitals":
        from .core.assays import format_vitals, vitals
        print()
        print(format_vitals(vitals(cfg, days=float(cfg.days))))
        return 0

    if args.twin:
        from .core.twin import run_twin
        from .core.twin_plot import save_png
        import json

        out_dir = os.path.join(args.out, cfg.name)
        os.makedirs(out_dir, exist_ok=True)
        try:
            res = run_twin(cfg, args.twin,
                           fork_at_s=args.fork_hours * 3600.0,
                           horizon_s=args.horizon_hours * 3600.0)
        except ValueError as e:
            p.error(str(e))
        print(json.dumps(res.summary(), indent=2))
        print("\nwrote", res.save(os.path.join(out_dir, "twin.npz")))
        print("wrote", save_png(res, os.path.join(out_dir, "twin.png"), cfg))
        return 0

    if args.watch:
        from .gui.launch import watch
        return watch(cfg, out_dir=args.out,
                     autostart=not args.no_autostart)

    project_dir = os.path.join(args.out, cfg.name)
    run_experiment(cfg, project_dir, log_cb=print, analyze=args.analyze)
    print(f"\nProject written to {project_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
