"""Experiment orchestration: build a project folder and run N trials.

A project folder is self-describing and analysis-ready::

    <project>/
      config.json              full ExperimentConfig (re-runnable)
      README.txt               provenance / how it was generated
      data/
        uwb_S001_processed.csv   canonical trajectory (FNT schema)
        events_S001.csv
        agents_S001.csv
        ... (one set per trial)
"""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .config import ExperimentConfig
from .simulation import Simulation, run_trial


def grid_offsets(n: int, w: float, h: float, gap: float | None = None):
    """Lay out ``n`` replicate chambers in a grid; return (dx, dy) offsets (m)."""
    if gap is None:
        gap = 0.25 * max(w, h)
    cols = max(1, math.ceil(math.sqrt(n)))
    return [((i % cols) * (w + gap), (i // cols) * (h + gap)) for i in range(n)]


def make_project(config: ExperimentConfig, project_dir: str) -> str:
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    config.to_json(os.path.join(project_dir, "config.json"))
    with open(os.path.join(project_dir, "README.txt"), "w") as fh:
        fh.write(
            "ABMA simulated experiment\n"
            f"name: {config.name}\n"
            f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"trials: {config.n_trials}  days: {config.days}  "
            f"agents/trial: {config.total_agents()}\n\n"
            "data/uwb_<trial>_processed.csv matches FNT UWB preprocessing output\n"
            "so downstream FNT/R proximity, network, and GBI analyses run as-is.\n"
        )
    return data_dir


def run_experiment(config: ExperimentConfig, project_dir: str,
                   progress_cb=None, frame_cb=None, log_cb=None,
                   frame_interval_s: float = 300.0,
                   analyze: bool = False, meta_cb=None) -> list[dict]:
    """Run all trials sequentially, with optional live frame streaming.

    progress_cb(overall_fraction), frame_cb(frame_dict) for trial 0 only,
    log_cb(str) for human-readable status. If ``analyze`` is set, run the
    built-in socio-spatial analysis over the finished project.
    """
    data_dir = make_project(config, project_dir)
    results = []
    n = config.n_trials

    def _log(msg):
        if log_cb:
            log_cb(msg)

    if config.parallel and n > 1:
        _log(f"Running {n} trials in parallel ({config.n_workers} workers)...")
        cfg_d = config.to_dict()
        args = [(cfg_d, i, data_dir) for i in range(n)]
        with ProcessPoolExecutor(max_workers=config.n_workers) as ex:
            for done, res in enumerate(ex.map(run_trial, args), 1):
                results.append(res)
                _log(f"  finished {res['trial_id']}")
                if progress_cb:
                    progress_cb(done / n)
    else:
        # replicates run in lockstep so they can be watched side-by-side
        _log(f"Running {n} replicate chamber(s) in lockstep "
             f"({config.total_agents()} agents each, {config.days} days)...")
        results = _run_live(config, data_dir, progress_cb, frame_cb, log_cb,
                            frame_interval_s, meta_cb)

    _log(f"Done. {n} trial(s) written to {data_dir}")

    if analyze:
        from .analysis import analyze_experiment
        _log("Analysing socio-spatial metrics...")
        try:
            analyze_experiment(project_dir, log_cb=_log)
        except Exception as e:  # analysis is best-effort; never fail the run
            _log(f"  analysis skipped: {e}")
    return results


def _run_live(config, data_dir, progress_cb, frame_cb, log_cb,
              frame_interval_s, meta_cb):
    """Step every replicate together; stream a combined multi-chamber frame."""
    from .recorder import (TrajectoryRecorder, EventRecorder, ConditionRecorder,
                           write_agents_table)
    n = config.n_trials
    sims = [Simulation(config, trial_index=i) for i in range(n)]
    offsets = grid_offsets(n, config.arena.width, config.arena.height)

    recs = []
    for sim in sims:
        base = {k: os.path.join(data_dir, f"{k}_{sim.trial_id}.csv")
                for k in ("events", "condition", "agents")}
        traj = os.path.join(data_dir, f"uwb_{sim.trial_id}_processed.csv")
        rec = TrajectoryRecorder(traj, sim.trial_id, sim.start_dt, sim.agents)
        ev = EventRecorder(base["events"], sim.trial_id, sim.start_dt)
        cr = ConditionRecorder(base["condition"], sim.trial_id, sim.start_dt,
                               sim.agents)
        write_agents_table(base["agents"], sim.agents)
        recs.append({"traj": rec, "evt": ev, "cond": cr, "paths": {
            "trajectory": traj, "events": base["events"],
            "condition": base["condition"], "agents": base["agents"],
            "trial_id": sim.trial_id}})

    if meta_cb is not None:  # combined static meta, keyed by GLOBAL index
        combined, gi = [], 0
        for si, sim in enumerate(sims):
            for a in sim.agent_static():
                a = dict(a)
                a["index"] = gi           # global order == frame array order
                a["replicate"] = si + 1
                a["group"] = f"{a['group']} · rep{si + 1}"
                combined.append(a)
                gi += 1
        meta_cb(combined)

    total_s = config.days * 86400.0
    dt = config.dt
    n_steps = int(total_s / dt)
    rec_every = max(1, int(round(config.record_interval / dt)))
    cond_every = max(1, int(round(max(300.0, config.record_interval) / dt)))
    frame_every = max(1, int(round(frame_interval_s / dt)))
    report_every = max(1, n_steps // 100)

    elapsed = 0.0
    try:
        for k in range(n_steps):
            for si, sim in enumerate(sims):
                sim.step(elapsed, dt, events=recs[si]["evt"])
            elapsed += dt
            if k % rec_every == 0:
                for sim, r in zip(sims, recs):
                    r["traj"].record(elapsed, sim.P[:, 0], sim.P[:, 1])
            if k % cond_every == 0:
                for sim, r in zip(sims, recs):
                    r["cond"].record(elapsed, sim.health, sim.energy, sim.hunger,
                                     sim.thirst, sim.stress, sim.mass,
                                     sim.smell < 0.5)
            if frame_cb is not None and k % frame_every == 0:
                frame_cb(_combined_frame(sims, offsets, elapsed))
            if progress_cb is not None and k % report_every == 0:
                progress_cb(k / n_steps)
    finally:
        for r in recs:
            r["traj"].close()
            r["evt"].close()
            r["cond"].close()
    if progress_cb is not None:
        progress_cb(1.0)
    return [r["paths"] for r in recs]


_FRAME_KEYS = ("heading", "sex_m", "alive", "color", "size", "shape",
               "health", "energy", "hunger", "thirst", "stress", "mass",
               "anosmic", "estrus", "activity", "fights_won", "fights_lost",
               "matings", "dist_today")


def _combined_frame(sims, offsets, elapsed):
    """Concatenate all replicates' agents into one frame, offset into the grid."""
    frames = [s._frame(elapsed) for s in sims]
    out = {
        "trial": "grid", "elapsed": elapsed,
        "day": int(elapsed // 86400) + 1,
        "hour": sims[0]._hour(elapsed), "is_day": sims[0]._is_day(elapsed),
        "x": np.concatenate([f["x"] + offsets[i][0]
                             for i, f in enumerate(frames)]),
        "y": np.concatenate([f["y"] + offsets[i][1]
                             for i, f in enumerate(frames)]),
    }
    for key in _FRAME_KEYS:
        out[key] = np.concatenate([f[key] for f in frames])
    return out
