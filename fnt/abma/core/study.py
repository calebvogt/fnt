"""Studies — run several conditions side by side and compare them.

A single run tells you what happened once. A *study* is the unit that answers
the questions ABMA exists for: "what changes if I lesion this?", "what changes
if I take the water away?". It is::

    base config  ×  conditions (overrides)  ×  replicates

Each condition is one execution of the base config with a few values patched,
run for ``replicates`` trials. Results are collapsed into a tidy long table
(condition, replicate, metric, value) plus a comparison against a reference
condition, so the output drops straight into R/ggplot.

Seeds
-----
``seed_policy="paired"`` (default) gives every condition the *same* seed
sequence, so replicate *r* starts from identical initial conditions in every
arm. Differences are then attributable to the manipulation rather than to
different starting positions — the simulation equivalent of a paired design.
``"independent"`` gives each condition a disjoint seed block instead.

Layout on disk::

    <study>/
      study.json
      conditions/01_control/{config.json,data/}
      conditions/02_anosmic/{config.json,data/}
      results/metrics_long.csv
      results/comparison.csv
"""
from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import ExperimentConfig

SEED_POLICIES = ("paired", "independent")
_SEED_BLOCK = 10_000          # spacing between conditions when independent


# --------------------------------------------------------------------------- #
# Override paths
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+|\*)\]")


def parse_path(path: str) -> list:
    """Split an override path into keys.

    ``'groups[0].traits.smell_ability'`` -> ``['groups', 0, 'traits', ...]``.
    ``[*]`` means *every* element, so ``groups[*].traits.smell_ability``
    lesions all groups at once — the usual case for an ablation.
    """
    out = []
    for name, idx in _TOKEN.findall(path):
        out.append(name if not idx else ("*" if idx == "*" else int(idx)))
    if not out:
        raise ValueError(f"empty override path: {path!r}")
    return out


def _targets(data, keys):
    """Resolve all containers matching ``keys[:-1]``, expanding any ``*``."""
    cur = [data]
    for k in keys[:-1]:
        nxt = []
        for c in cur:
            if k == "*":
                nxt.extend(c)
            else:
                nxt.append(c[k])
        cur = nxt
    return cur


def set_path(data: dict, path: str, value: Any) -> None:
    """Set a dotted/indexed path inside a config *dict*, in place."""
    keys = parse_path(path)
    last = keys[-1]
    try:
        targets = _targets(data, keys)
        if not targets:
            raise KeyError("no matching element")
        for t in targets:
            if last == "*":
                for i in range(len(t)):
                    t[i] = value
            else:
                t[last] = value
    except (KeyError, IndexError, TypeError) as e:
        raise KeyError(f"override path {path!r} failed: {e}") from e


def get_path(data: dict, path: str) -> Any:
    keys = parse_path(path)
    targets = _targets(data, keys)
    last = keys[-1]
    vals = [t[last] for t in targets] if last != "*" else [v for t in targets
                                                          for v in t]
    return vals[0] if len(vals) == 1 else vals


# --------------------------------------------------------------------------- #
# Study definition
# --------------------------------------------------------------------------- #
@dataclass
class Condition:
    """One arm: the base config with ``overrides`` applied."""
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "overrides": self.overrides,
                "description": self.description}


@dataclass
class Study:
    name: str
    base: ExperimentConfig
    conditions: list[Condition] = field(default_factory=list)
    replicates: int = 4
    seed_policy: str = "paired"
    base_seed: int = 0
    reference: str = ""          # condition to compare against ("" = first)
    description: str = ""

    # ---- construction -------------------------------------------------- #
    def config_for(self, index: int) -> ExperimentConfig:
        """Materialise the config for condition ``index`` (all replicates)."""
        if self.seed_policy not in SEED_POLICIES:
            raise ValueError(f"seed_policy must be one of {SEED_POLICIES}")
        cond = self.conditions[index]
        d = copy.deepcopy(self.base.to_dict())
        for path, value in (cond.overrides or {}).items():
            set_path(d, path, value)
        cfg = ExperimentConfig.from_dict(d)
        cfg.n_trials = self.replicates
        # trial i inside a run draws seed = cfg.seed + i (see Simulation), so
        # a shared base seed makes replicate i identical across conditions.
        cfg.seed = self.base_seed + (
            0 if self.seed_policy == "paired" else index * _SEED_BLOCK)
        cfg.name = f"{_slug(self.name)}_{_slug(cond.name)}"
        return cfg

    def ref_index(self) -> int:
        names = [c.name for c in self.conditions]
        if self.reference and self.reference in names:
            return names.index(self.reference)
        return 0

    # ---- serialization -------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "name": self.name, "description": self.description,
            "replicates": self.replicates, "seed_policy": self.seed_policy,
            "base_seed": self.base_seed, "reference": self.reference,
            "conditions": [c.to_dict() for c in self.conditions],
            "base": self.base.to_dict(),
        }

    def to_json(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Study":
        return Study(
            name=d.get("name", "study"),
            base=ExperimentConfig.from_dict(d["base"]),
            conditions=[Condition(**c) for c in d.get("conditions", [])],
            replicates=d.get("replicates", 4),
            seed_policy=d.get("seed_policy", "paired"),
            base_seed=d.get("base_seed", 0),
            reference=d.get("reference", ""),
            description=d.get("description", ""),
        )

    @staticmethod
    def from_json(path: str) -> "Study":
        with open(path) as fh:
            return Study.from_dict(json.load(fh))


def _slug(s: str) -> str:
    s = "".join(c if c.isalnum() or c in " _-" else "_" for c in str(s)).strip()
    return re.sub(r"\s+", "_", s) or "x"


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def run_study(study: Study, study_dir: str, progress_cb: Callable | None = None,
              log_cb: Callable | None = None, analyze: bool = True,
              cancel_cb: Callable | None = None) -> dict:
    """Run every condition, then collapse results into tidy tables.

    Returns ``{"dir", "conditions", "metrics_csv", "comparison_csv"}``.
    """
    from .runner import run_experiment

    os.makedirs(study_dir, exist_ok=True)
    study.to_json(os.path.join(study_dir, "study.json"))
    cdir = os.path.join(study_dir, "conditions")
    os.makedirs(cdir, exist_ok=True)

    def _log(m):
        if log_cb:
            log_cb(m)

    n = len(study.conditions)
    out_dirs = []
    for i, cond in enumerate(study.conditions):
        if cancel_cb and cancel_cb():
            _log("Study cancelled.")
            break
        cfg = study.config_for(i)
        cd = os.path.join(cdir, f"{i + 1:02d}_{_slug(cond.name)}")
        os.makedirs(cd, exist_ok=True)
        _log(f"[{i + 1}/{n}] condition '{cond.name}' "
             f"({study.replicates} replicate(s), seed {cfg.seed})")
        run_experiment(cfg, cd, log_cb=log_cb, analyze=False,
                       cancel_cb=cancel_cb)
        out_dirs.append((cond.name, cd))
        if progress_cb:
            progress_cb((i + 1) / max(1, n))

    res = {"dir": study_dir, "conditions": out_dirs,
           "metrics_csv": None, "comparison_csv": None}
    if analyze and out_dirs:
        _log("Collecting metrics...")
        try:
            res.update(collect_results(study, study_dir))
        except Exception as e:                    # analysis is best-effort
            _log(f"  metric collection skipped: {e}")
    return res


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
_ID_COLS = ("Trial", "n_agents")


def collect_results(study: Study, study_dir: str) -> dict:
    """Analyse every trial and write metrics_long.csv + comparison.csv."""
    import glob
    import pandas as pd
    from .analysis import analyze_trial

    rows = []
    cdir = os.path.join(study_dir, "conditions")
    for i, cond in enumerate(study.conditions):
        cd = os.path.join(cdir, f"{i + 1:02d}_{_slug(cond.name)}", "data")
        if not os.path.isdir(cd):
            continue
        zones = study.base.arena.zones or None
        for traj in sorted(glob.glob(os.path.join(cd, "uwb_*_processed.csv"))):
            try:
                summ = analyze_trial(traj, cd, zones=zones)
            except Exception:
                continue
            trial = str(summ.get("Trial", ""))
            m = re.search(r"(\d+)$", trial)
            rep = int(m.group(1)) if m else len(rows) + 1
            for k, v in summ.items():
                if k in _ID_COLS or not isinstance(v, (int, float)):
                    continue
                rows.append({"condition": cond.name, "replicate": rep,
                             "metric": k, "value": float(v)})
    rdir = os.path.join(study_dir, "results")
    os.makedirs(rdir, exist_ok=True)
    long_csv = os.path.join(rdir, "metrics_long.csv")
    long_df = pd.DataFrame(rows, columns=["condition", "replicate",
                                          "metric", "value"])
    long_df.to_csv(long_csv, index=False)

    comp_csv = os.path.join(rdir, "comparison.csv")
    compare(long_df, study, comp_csv)
    return {"metrics_csv": long_csv, "comparison_csv": comp_csv}


def compare(long_df, study: Study, out_csv: str | None = None):
    """Per metric × condition summary, differenced against the reference arm.

    Descriptive only: means, spread, difference and a standardised effect
    size. With a handful of replicates these are effect estimates, not
    hypothesis tests — no p-values are reported and none should be inferred.
    """
    import numpy as np
    import pandas as pd

    if long_df is None or long_df.empty:
        empty = pd.DataFrame(columns=["metric", "condition", "n", "mean", "sd"])
        if out_csv:
            empty.to_csv(out_csv, index=False)
        return empty

    ref = study.conditions[study.ref_index()].name if study.conditions else None
    paired = study.seed_policy == "paired"
    out = []
    for metric, mdf in long_df.groupby("metric"):
        by = {c: g.set_index("replicate")["value"]
              for c, g in mdf.groupby("condition")}
        rvals = by.get(ref)
        for cond, vals in by.items():
            row = {"metric": metric, "condition": cond, "n": int(vals.size),
                   "mean": float(vals.mean()),
                   "sd": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                   "is_reference": cond == ref}
            if rvals is not None and cond != ref:
                row["diff_vs_ref"] = float(vals.mean() - rvals.mean())
                # pooled-SD standardised difference (Cohen's d)
                if vals.size > 1 and rvals.size > 1:
                    s1, s2 = vals.std(ddof=1), rvals.std(ddof=1)
                    n1, n2 = vals.size, rvals.size
                    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2)
                                 / max(1, n1 + n2 - 2))
                    row["cohens_d"] = (float((vals.mean() - rvals.mean()) / sp)
                                       if sp > 1e-12 else 0.0)
                # paired design: difference within matched replicates
                if paired:
                    common = vals.index.intersection(rvals.index)
                    if len(common):
                        d = vals.loc[common] - rvals.loc[common]
                        row["paired_diff_mean"] = float(d.mean())
                        row["paired_diff_sd"] = (float(d.std(ddof=1))
                                                 if len(d) > 1 else 0.0)
            out.append(row)
    cols = ["metric", "condition", "is_reference", "n", "mean", "sd",
            "diff_vs_ref", "cohens_d", "paired_diff_mean", "paired_diff_sd"]
    df = pd.DataFrame(out)
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    df = df.sort_values(["metric", "condition"]).reset_index(drop=True)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


# --------------------------------------------------------------------------- #
# Convenience builders for the two designs ABMA is built for
# --------------------------------------------------------------------------- #
def lesion_study(name: str, base: ExperimentConfig, trait_path: str,
                 levels: dict[str, Any], replicates: int = 4,
                 reference: str = "") -> Study:
    """Vary one trait/parameter across named levels (e.g. an ablation series).

    ``levels`` maps condition name -> value, e.g.
    ``{"intact": 1.0, "partial": 0.5, "anosmic": 0.0}``.
    """
    conds = [Condition(name=k, overrides={trait_path: v},
                       description=f"{trait_path} = {v}")
             for k, v in levels.items()]
    return Study(name=name, base=base, conditions=conds, replicates=replicates,
                 reference=reference or next(iter(levels), ""))


def environment_study(name: str, base: ExperimentConfig,
                      variants: dict[str, dict[str, Any]],
                      replicates: int = 4, reference: str = "") -> Study:
    """Compare worlds: each variant is a name -> {path: value} override set."""
    conds = [Condition(name=k, overrides=v) for k, v in variants.items()]
    return Study(name=name, base=base, conditions=conds, replicates=replicates,
                 reference=reference or next(iter(variants), ""))
