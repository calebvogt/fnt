"""Built-in socio-spatial analysis of ABMA output.

Closes the loop: after a trial is simulated, derive the same summary metrics you
compute on real UWB data — pairwise proximity, daily social edgelists, individual
space use — without leaving ABMA. Because the trajectory schema matches FNT's UWB
output, these mirror the FNT/R proximity + network analyses.

Outputs (per trial), written next to the trajectory CSV:
  edgelist_daily_<trial>.csv   per-day, per-pair contact time / bouts / distance
  space_use_<trial>.csv        per-agent centroid, home-range area, path length
And an experiment-level ``analysis_summary.csv`` aggregating across trials.
"""
from __future__ import annotations

import glob
import itertools
import os

import numpy as np
import pandas as pd


def _sample_dt(time_sec: np.ndarray) -> float:
    ts = np.unique(time_sec)
    return float(np.median(np.diff(ts))) if ts.size > 1 else 1.0


def analyze_trial(traj_csv: str, out_dir: str | None = None,
                  proximity_threshold: float = 0.15, zones=None) -> dict:
    """Analyse one trajectory CSV; write edgelist + space-use, return a summary."""
    df = pd.read_csv(traj_csv)
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(traj_csv))
    trial = str(df["Trial"].iloc[0])
    ids = sorted(df["sexid"].unique())
    sex = df.groupby("sexid")["sex"].first().to_dict()
    sdt = _sample_dt(df["time_sec"].values)

    px = df.pivot_table(index="time_sec", columns="sexid", values="smoothed_x")
    py = df.pivot_table(index="time_sec", columns="sexid", values="smoothed_y")
    day_of = df.groupby("time_sec")["Day"].first()

    # ---- daily pairwise edgelist ---------------------------------------- #
    edges = []
    pair_mean_dist = {"F-F": [], "M-M": [], "F-M": []}
    dyads_in_contact = 0
    for a, b in itertools.combinations(ids, 2):
        if a not in px or b not in px:
            continue
        d = np.sqrt((px[a] - px[b]) ** 2 + (py[a] - py[b]) ** 2)
        ptype = "-".join(sorted([sex[a], sex[b]]))
        ptype = "F-M" if ptype == "F-M" else ptype
        pair_mean_dist.setdefault(ptype, []).append(float(d.mean()))
        contact_any = False
        for day, sub_idx in day_of.groupby(day_of).groups.items():
            dd = (d.loc[sub_idx] < proximity_threshold).astype(int).values
            if dd.size == 0:
                continue
            total_time = float(dd.sum() * sdt)
            if total_time <= 0:
                continue
            contact_any = True
            n_bouts = int((dd[0] == 1) + ((dd[1:] == 1) & (dd[:-1] == 0)).sum())
            mean_d = float(d.loc[sub_idx][d.loc[sub_idx] < proximity_threshold].mean())
            edges.append([trial, int(day), a, b, total_time, n_bouts,
                          round(mean_d, 4), sex[a], sex[b], ptype])
        dyads_in_contact += int(contact_any)

    edge_df = pd.DataFrame(edges, columns=[
        "Trial", "Day", "animal1", "animal2", "total_time_s", "n_bouts",
        "mean_distance", "sex1", "sex2", "pair_type"])
    edge_path = os.path.join(out_dir, f"edgelist_daily_{trial}.csv")
    edge_df.to_csv(edge_path, index=False)

    # ---- per-agent space use -------------------------------------------- #
    rows = []
    for aid in ids:
        g = df[df["sexid"] == aid]
        sx, sy = g["smoothed_x"].std(), g["smoothed_y"].std()
        # 95% bivariate-normal ellipse area (2.45 sigma semi-axes)
        area95 = float(np.pi * (2.45 * sx) * (2.45 * sy))
        path_len = float(np.sqrt(np.diff(g["smoothed_x"]) ** 2
                                 + np.diff(g["smoothed_y"]) ** 2).sum())
        rows.append([trial, aid, sex[aid], round(g["smoothed_x"].mean(), 4),
                     round(g["smoothed_y"].mean(), 4), round(area95, 4),
                     round(path_len, 2)])
    space_df = pd.DataFrame(rows, columns=[
        "Trial", "sexid", "sex", "centroid_x", "centroid_y",
        "home_range_area_m2", "path_length_m"])
    space_path = os.path.join(out_dir, f"space_use_{trial}.csv")
    space_df.to_csv(space_path, index=False)

    n_dyads = len(list(itertools.combinations(ids, 2)))
    summary = {
        "Trial": trial,
        "n_agents": len(ids),
        "mean_dist_FF": _nanmean(pair_mean_dist.get("F-F")),
        "mean_dist_MM": _nanmean(pair_mean_dist.get("M-M")),
        "mean_dist_FM": _nanmean(pair_mean_dist.get("F-M")),
        "mean_home_range_m2": round(float(space_df["home_range_area_m2"].mean()), 4),
        "network_density": round(dyads_in_contact / n_dyads, 4) if n_dyads else 0.0,
    }
    # ---- zone occupancy (e.g. OFT centre time) --------------------------- #
    if zones:
        zrows = []
        for aid in ids:
            g = df[df["sexid"] == aid]
            gx, gy = g["smoothed_x"].values, g["smoothed_y"].values
            ntot = len(gx)
            for z in zones:
                inside = ((gx >= z.x) & (gx <= z.x + z.w)
                          & (gy >= z.y) & (gy <= z.y + z.h))
                frac = float(inside.mean()) if ntot else 0.0
                zrows.append([trial, aid, sex[aid], z.name, z.role,
                              round(frac, 4), round(frac * ntot * sdt, 1)])
        zdf = pd.DataFrame(zrows, columns=[
            "Trial", "sexid", "sex", "zone", "role", "frac_time", "seconds"])
        zdf.to_csv(os.path.join(out_dir, f"zone_occupancy_{trial}.csv"),
                   index=False)
        centers = zdf[zdf["role"] == "center"]
        if not centers.empty:
            summary["center_time_pct"] = round(
                float(centers["frac_time"].mean()) * 100, 1)

    # ---- dominance from combat events + event counts --------------------- #
    evt = os.path.join(out_dir, f"events_{trial}.csv")
    if os.path.exists(evt):
        ev = pd.read_csv(evt)
        for kind in ("mating", "fight", "death"):
            summary[f"n_{kind}"] = int((ev["event"] == kind).sum())
        _write_dominance(ev, ids, sex, trial, out_dir)

    # ---- condition summary (final mass, mean health/stress) -------------- #
    cond = os.path.join(out_dir, f"condition_{trial}.csv")
    if os.path.exists(cond):
        cd = pd.read_csv(cond)
        last = cd.sort_values("time_sec").groupby("sexid").tail(1)
        summary["mean_final_mass_g"] = round(float(last["mass"].mean()), 2)
        summary["mean_health"] = round(float(cd["health"].mean()), 1)
        summary["mean_stress"] = round(float(cd["stress"].mean()), 1)
    return summary


def _write_dominance(ev, ids, sex, trial, out_dir):
    """Dyadic win/loss tallies and (unweighted) David's score per agent."""
    fights = ev[ev["event"] == "fight"]
    if fights.empty:
        return
    wins = {a: {b: 0 for b in ids} for a in ids}
    for _, r in fights.iterrows():
        if r["actor"] in wins and r["target"] in wins[r["actor"]]:
            wins[r["actor"]][r["target"]] += 1  # actor = winner
    rows = []
    for a in ids:
        won = sum(wins[a][b] for b in ids)
        lost = sum(wins[b][a] for b in ids)
        ds = 0.0
        for b in ids:
            tot = wins[a][b] + wins[b][a]
            if tot:
                ds += wins[a][b] / tot
        rows.append([trial, a, sex[a], won, lost,
                     round(won / (won + lost), 3) if won + lost else None,
                     round(ds, 3)])
    df = pd.DataFrame(rows, columns=[
        "Trial", "sexid", "sex", "wins", "losses", "win_rate", "davids_score"])
    df = df.sort_values("davids_score", ascending=False)
    df.to_csv(os.path.join(out_dir, f"dominance_{trial}.csv"), index=False)


def analyze_experiment(project_dir: str, proximity_threshold: float = 0.15,
                       log_cb=None) -> str:
    """Analyse every trial in ``<project_dir>/data`` and write a summary CSV."""
    data_dir = os.path.join(project_dir, "data")
    if not os.path.isdir(data_dir):
        data_dir = project_dir
    # zones (if any) come from the project's config.json
    zones = []
    for cand in (os.path.join(project_dir, "config.json"),
                 os.path.join(os.path.dirname(data_dir), "config.json")):
        if os.path.exists(cand):
            try:
                from .config import ExperimentConfig
                zones = ExperimentConfig.from_json(cand).arena.zones
            except Exception:
                zones = []
            break
    trajs = sorted(glob.glob(os.path.join(data_dir, "uwb_*_processed.csv")))
    summaries = []
    for t in trajs:
        if log_cb:
            log_cb(f"  analysing {os.path.basename(t)}")
        summaries.append(analyze_trial(t, data_dir, proximity_threshold,
                                       zones=zones))
    if not summaries:
        raise FileNotFoundError(f"No uwb_*_processed.csv found in {data_dir}")
    out = os.path.join(data_dir, "analysis_summary.csv")
    pd.DataFrame(summaries).to_csv(out, index=False)
    if log_cb:
        log_cb(f"  wrote {os.path.basename(out)}")
    return out


def _nanmean(vals):
    if not vals:
        return None
    return round(float(np.nanmean(vals)), 4)
