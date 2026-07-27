"""ABMA projects — the unit a user creates, saves, reopens and shares.

A project is one self-contained folder holding the world, the population, the
protocol, the parameter provenance, and every run ever executed from it::

    <project>/
      project.json              manifest + working ExperimentConfig + provenance
      runs/
        run_001_20260726-1432/
          config.json           exact config used for this run (with its seed)
          README.txt
          data/*.csv            trajectories, events, conditions, agents

Runs are *append-only*: each execution allocates a new folder, so a project
accumulates a reproducible history rather than overwriting itself. Re-running
``config.json`` from any run folder reproduces that run exactly.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .config import ExperimentConfig

SCHEMA_VERSION = 1
MANIFEST = "project.json"

# How a parameter's value was arrived at. Recording this is what keeps a
# game-like tuning UI honest: a slider is fine, an undocumented number is not.
SOURCES = ("measured", "literature", "estimated", "free", "default")


def default_root() -> str:
    """Where projects live unless the user picks somewhere else."""
    return os.path.join(os.path.expanduser("~"), "ABMA_Projects")


def _slug(name: str) -> str:
    s = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    return re.sub(r"\s+", "_", s) or "project"


@dataclass
class Project:
    """An open project: its folder, metadata, working config and provenance."""

    path: str
    name: str = "project"
    description: str = ""
    created: str = ""
    modified: str = ""
    config: ExperimentConfig | None = None
    provenance: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ---- lifecycle ---------------------------------------------------- #
    @staticmethod
    def create(name: str, config: ExperimentConfig, root: str | None = None,
               description: str = "") -> "Project":
        root = root or default_root()
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, _slug(name))
        n = 2
        while os.path.exists(path):            # never clobber an existing project
            path = os.path.join(root, f"{_slug(name)}_{n}")
            n += 1
        os.makedirs(os.path.join(path, "runs"), exist_ok=True)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        p = Project(path=path, name=name, description=description,
                    created=now, modified=now, config=config)
        p.save()
        return p

    @staticmethod
    def load(path: str) -> "Project":
        if os.path.isfile(path):               # tolerate being given the manifest
            path = os.path.dirname(path)
        with open(os.path.join(path, MANIFEST)) as fh:
            d = json.load(fh)
        cfg = d.get("config")
        return Project(
            path=path,
            name=d.get("name", os.path.basename(path)),
            description=d.get("description", ""),
            created=d.get("created", ""),
            modified=d.get("modified", ""),
            config=ExperimentConfig.from_dict(cfg) if cfg else None,
            provenance=d.get("provenance", {}) or {},
        )

    def save(self) -> None:
        self.modified = time.strftime("%Y-%m-%dT%H:%M:%S")
        os.makedirs(os.path.join(self.path, "runs"), exist_ok=True)
        payload = {
            "schema": SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "created": self.created,
            "modified": self.modified,
            "config": self.config.to_dict() if self.config else None,
            "provenance": self.provenance,
        }
        tmp = os.path.join(self.path, MANIFEST + ".tmp")
        with open(tmp, "w") as fh:             # atomic: never leave a half file
            json.dump(payload, fh, indent=2)
        os.replace(tmp, os.path.join(self.path, MANIFEST))

    # ---- runs ---------------------------------------------------------- #
    @property
    def runs_dir(self) -> str:
        return os.path.join(self.path, "runs")

    def new_run_dir(self, label: str = "") -> str:
        """Allocate the next run folder. Runs are never overwritten."""
        os.makedirs(self.runs_dir, exist_ok=True)
        used = [d for d in os.listdir(self.runs_dir)
                if d.startswith("run_") and
                os.path.isdir(os.path.join(self.runs_dir, d))]
        nxt = 1
        for d in used:
            m = re.match(r"run_(\d+)", d)
            if m:
                nxt = max(nxt, int(m.group(1)) + 1)
        stamp = time.strftime("%Y%m%d-%H%M")
        suffix = f"_{_slug(label)}" if label else ""
        path = os.path.join(self.runs_dir, f"run_{nxt:03d}_{stamp}{suffix}")
        os.makedirs(path, exist_ok=True)
        return path

    def runs(self) -> list[dict]:
        """Summarise every run in the project, newest first."""
        out = []
        if not os.path.isdir(self.runs_dir):
            return out
        for d in sorted(os.listdir(self.runs_dir), reverse=True):
            rp = os.path.join(self.runs_dir, d)
            if not os.path.isdir(rp):
                continue
            info = {"id": d, "path": rp, "days": None, "trials": None,
                    "agents": None, "seed": None, "when": ""}
            try:
                info["when"] = time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(rp)))
            except OSError:
                pass
            cfg_p = os.path.join(rp, "config.json")
            if os.path.isfile(cfg_p):
                try:
                    c = ExperimentConfig.from_json(cfg_p)
                    info.update(days=c.days, trials=c.n_trials,
                                agents=c.total_agents(), seed=c.seed)
                except Exception:
                    pass
            data = os.path.join(rp, "data")
            info["n_files"] = len(os.listdir(data)) if os.path.isdir(data) else 0
            out.append(info)
        return out

    # ---- provenance ----------------------------------------------------- #
    def set_provenance(self, param: str, unit: str = "", source: str = "estimated",
                       note: str = "") -> None:
        """Record how a parameter's value was arrived at.

        ``param`` is a dotted path into the config, e.g. ``arena.width``.
        """
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        self.provenance[param] = {"unit": unit, "source": source, "note": note}

    def provenance_summary(self) -> dict[str, int]:
        counts = {s: 0 for s in SOURCES}
        for rec in self.provenance.values():
            counts[rec.get("source", "default")] = \
                counts.get(rec.get("source", "default"), 0) + 1
        return counts


def list_projects(root: str | None = None) -> list[dict]:
    """Rows for the Load screen: name, path, when, and a one-line summary."""
    root = root or default_root()
    if not os.path.isdir(root):
        return []
    rows = []
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d)
        man = os.path.join(path, MANIFEST)
        if not os.path.isfile(man):
            continue
        try:
            with open(man) as fh:
                m = json.load(fh)
        except Exception:
            continue
        cfg = m.get("config") or {}
        arena = cfg.get("arena", {})
        n_agents = sum(g.get("count", 0) for g in cfg.get("groups", []))
        runs_dir = os.path.join(path, "runs")
        n_runs = len([x for x in os.listdir(runs_dir)
                      if os.path.isdir(os.path.join(runs_dir, x))]) \
            if os.path.isdir(runs_dir) else 0
        rows.append({
            "name": m.get("name", d),
            "path": path,
            "modified": m.get("modified", ""),
            "description": m.get("description", ""),
            "n_agents": n_agents,
            "n_runs": n_runs,
            "summary": (f"{arena.get('width', 0):g}×{arena.get('height', 0):g} m · "
                        f"{n_agents} agents · {n_runs} run(s)"),
        })
    rows.sort(key=lambda r: r["modified"], reverse=True)
    return rows
