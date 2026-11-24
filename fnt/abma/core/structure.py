"""Hierarchy or territories? Readouts of social structure from zone visits.

The question these answer: do the animals of one sex sort into a single rank
order that holds everywhere (one pole), or into owners of different places,
each of which wins at home (several poles)?

Everything here is computed from one table of **zone visits** ("bouts"), one
row per visit::

    animal   who
    zone     where
    start    when it began, seconds (any origin)
    stop     when it ended

That is what an RFID antenna at a resource zone gives for real animals, and
what :func:`bouts_from_positions` derives from a simulated trajectory. Because
both go through the same functions, a simulated cohort and a real one are
measured by exactly the same rule, including that rule's blind spots.

Nothing here scores a fight directly. A **displacement** is read from the
visits: one animal is alone at a zone, a second joins it, and then one of the
two is alone again. The one that stays is the winner, the one that was there
first is the resident. That is the "R, RI, I" sequence used for the field mice
in Vogt et al. 2024 (BMC Biology 22:35).

Readouts (see :func:`structure_summary`):

*Who holds what.* ``owner_share`` (how exclusively each zone is used),
``poles`` (the effective number of animals that own zones: 1 = one animal owns
every zone, Z = each of Z zones has a different owner), ``landless`` (animals
owning nothing), ``owner_stays`` (a zone has the same owner the next day),
``access_gini`` (how unequal time at the zones is), ``shared``
(how much of the occupied time has two or more animals present).

*Who beats whom, and does it depend on where.* ``landau_h`` and ``ttri`` (how
linear and transitive the displacement matrix is), ``dci`` (how one-sided
dyads are), ``resident_win`` (the animal that was there first stays),
``owner_win`` (yesterday's owner of the zone stays) and ``rank_transfer`` (an
animal's record at the *other* zones predicts who stays at this one). In a
hierarchy rank transfers and ownership adds nothing; among territory holders
ownership predicts the winner and rank does not transfer.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

BOUT_COLS = ["animal", "zone", "start", "stop"]


# --------------------------------------------------------------------------- #
# Who holds what
# --------------------------------------------------------------------------- #
def _clean(bouts: pd.DataFrame, min_dur_s: float = 1.0) -> pd.DataFrame:
    b = bouts[BOUT_COLS + [c for c in ("day",) if c in bouts.columns]].copy()
    b["start"] = b["start"].astype(float)
    # a single RFID read is a visit of no measured length; give it the floor
    b["stop"] = np.maximum(b["stop"].astype(float), b["start"] + float(min_dur_s))
    return b


def zone_time(bouts: pd.DataFrame, animals=None, zones=None) -> pd.DataFrame:
    """Seconds each animal spent at each zone (animals x zones; zeros kept)."""
    b = _clean(bouts)
    t = (b.assign(dur=b["stop"] - b["start"])
         .pivot_table(index="animal", columns="zone", values="dur", aggfunc="sum", fill_value=0.0))
    if animals is not None:
        t = t.reindex(index=list(animals), fill_value=0.0)
    if zones is not None:
        t = t.reindex(columns=list(zones), fill_value=0.0)
    return t


def zone_control(bouts: pd.DataFrame, animals=None, zones=None) -> pd.DataFrame:
    """Per zone: its owner (the animal with the most time), that animal's share,
    the seconds of use and how many animals used it. Unused zones have no owner."""
    t = zone_time(bouts, animals, zones)
    tot = t.sum(axis=0)
    out = pd.DataFrame({
        "owner": [t[z].idxmax() if tot[z] > 0 else None for z in t.columns],
        "owner_share": [t[z].max() / tot[z] if tot[z] > 0 else np.nan for z in t.columns],
        "seconds": tot.values,
        "users": [(t[z] > 0).sum() for z in t.columns],
    }, index=t.columns)
    out.index.name = "zone"
    return out


def gini(x) -> float:
    """Gini coefficient of non-negative values (0 = equal, towards 1 = one has all)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _states(bouts: pd.DataFrame):
    """Per zone, the sequence of (time, set of animals present). Ends are applied
    before starts at the same instant, so a hand-over is not a meeting."""
    b = _clean(bouts)
    for zone, g in b.groupby("zone", sort=True):
        ev = sorted([(s, 1, a) for a, s in zip(g["animal"], g["start"])]
                    + [(e, 0, a) for a, e in zip(g["animal"], g["stop"])],
                    key=lambda x: (x[0], x[1]))
        count: dict = {}
        seq = []
        for t, kind, a in ev:
            before = frozenset(k for k, v in count.items() if v > 0)
            count[a] = count.get(a, 0) + (1 if kind else -1)
            after = frozenset(k for k, v in count.items() if v > 0)
            if after != before:
                if seq and seq[-1][0] == t:
                    seq[-1] = (t, after)
                else:
                    seq.append((t, after))
        yield zone, seq


def copresence(bouts: pd.DataFrame) -> pd.DataFrame:
    """Per zone: seconds with at least one animal present, and with two or more."""
    rows = []
    for zone, seq in _states(bouts):
        occ = two = 0.0
        for (t0, s), (t1, _) in zip(seq[:-1], seq[1:]):
            if len(s) >= 1:
                occ += t1 - t0
            if len(s) >= 2:
                two += t1 - t0
        rows.append((zone, occ, two))
    return pd.DataFrame(rows, columns=["zone", "occupied_s", "shared_s"]).set_index("zone")


# --------------------------------------------------------------------------- #
# Who beats whom
# --------------------------------------------------------------------------- #
def displacements(bouts: pd.DataFrame, min_together_s: float = 0.0,
                  min_after_s: float = 0.0) -> pd.DataFrame:
    """Displacements read from visits: alone, joined by a second, one left alone.

    One row per event: ``time`` (when the loser left), ``zone``, ``winner`` (the
    one that stayed), ``loser``, ``resident`` (the one that was there first),
    ``together_s`` (how long both were present) and ``after_s`` (how long the
    winner then stayed alone). Meetings of three or more are not scored.
    """
    rows = []
    for zone, seq in _states(bouts):
        for k in range(1, len(seq) - 1):
            (_, a), (t1, pair), (t2, c) = seq[k - 1], seq[k], seq[k + 1]
            if len(a) == 1 and len(pair) == 2 and len(c) == 1 and a < pair and c < pair:
                winner = next(iter(c))
                loser = next(iter(pair - c))
                t3 = seq[k + 2][0] if k + 2 < len(seq) else t2
                if t2 - t1 >= min_together_s and t3 - t2 >= min_after_s:
                    rows.append((t2, zone, winner, loser, next(iter(a)), t2 - t1, t3 - t2))
    return pd.DataFrame(rows, columns=["time", "zone", "winner", "loser", "resident",
                                       "together_s", "after_s"])


def win_matrix(disp: pd.DataFrame, animals) -> pd.DataFrame:
    """Counts of row-beat-column over ``animals``."""
    animals = list(animals)
    W = pd.DataFrame(0.0, index=animals, columns=animals)
    for w, l in zip(disp["winner"], disp["loser"]):
        if w in W.index and l in W.index:
            W.loc[w, l] += 1
    return W


def landau_h(W) -> float:
    """Landau's h (0 = no order, 1 = a perfectly linear order). Dyads go to the
    majority winner; tied and unobserved dyads count half each way."""
    W = np.asarray(W, float)
    n = len(W)
    if n < 3:
        return np.nan
    D = np.where(W > W.T, 1.0, np.where(W < W.T, 0.0, 0.5))
    np.fill_diagonal(D, 0.0)
    v = D.sum(axis=1)
    return float(12.0 / (n ** 3 - n) * ((v - (n - 1) / 2.0) ** 2).sum())


def dci(W) -> float:
    """Directional consistency: 0 = dyads split evenly, 1 = every dyad one-way."""
    W = np.asarray(W, float)
    tot = W.sum()
    return float(np.abs(W - W.T).sum() / 2.0 / tot) if tot > 0 else np.nan


def triangle_transitivity(W) -> float:
    """Shizuka & McDonald's t_tri: among triads in which every dyad has a
    winner, 1 = all transitive, 0 = as many cycles as chance gives."""
    W = np.asarray(W, float)
    n = len(W)
    D = W > W.T
    trans = cyc = 0
    for i, j, k in itertools.combinations(range(n), 3):
        if all(D[a, b] or D[b, a] for a, b in ((i, j), (j, k), (i, k))):
            wins = sorted((int(D[i, j]) + int(D[i, k]), int(D[j, i]) + int(D[j, k]),
                           int(D[k, i]) + int(D[k, j])))
            if wins == [1, 1, 1]:
                cyc += 1
            else:
                trans += 1
    if trans + cyc == 0:
        return np.nan
    return float(4.0 * (trans / (trans + cyc) - 0.75))


def davids_score(W: pd.DataFrame) -> pd.Series:
    """Normalised David's score per animal (de Vries et al. 2006), from dyadic
    win proportions corrected for the number of interactions."""
    A = np.asarray(W, float)
    n = len(A)
    tot = A + A.T
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(tot > 0, (A + 0.5) / (tot + 1.0), 0.0)     # Dij
    np.fill_diagonal(P, 0.0)
    Q = np.where(tot > 0, 1.0 - P, 0.0)
    np.fill_diagonal(Q, 0.0)
    w, l = P.sum(axis=1), Q.sum(axis=1)
    ds = w + P.dot(w) - l - Q.dot(l)
    return pd.Series((ds + n * (n - 1) / 2.0) / n, index=W.index)


def _owner_win(disp: pd.DataFrame, owner_of) -> tuple[float, int]:
    """Share of displacements won by the zone's owner, where exactly one of the
    two animals is the owner. ``owner_of(row)`` gives the owner to use."""
    win = n = 0
    for r in disp.itertuples(index=False):
        o = owner_of(r)
        if o is None or (o == r.winner) == (o == r.loser):
            continue
        n += 1
        win += int(o == r.winner)
    return (win / n if n else np.nan), n


def _rank_transfer(disp: pd.DataFrame) -> tuple[float, int]:
    """Does a record made elsewhere predict who stays here? For each
    displacement, each animal's win rate at the OTHER zones; the share won by
    the animal with the better one (events where they tie are left out)."""
    if disp.empty:
        return np.nan, 0
    w = disp.groupby(["winner", "zone"]).size().unstack(fill_value=0)
    l = disp.groupby(["loser", "zone"]).size().unstack(fill_value=0)
    zones = sorted(set(w.columns) | set(l.columns))
    ids = sorted(set(w.index) | set(l.index))
    w = w.reindex(index=ids, columns=zones, fill_value=0)
    l = l.reindex(index=ids, columns=zones, fill_value=0)
    hit = n = 0
    for r in disp.itertuples(index=False):
        other = [z for z in zones if z != r.zone]
        rate = {}
        for a in (r.winner, r.loser):
            ww, ll = w.loc[a, other].sum(), l.loc[a, other].sum()
            rate[a] = ww / (ww + ll) if ww + ll > 0 else np.nan
        if np.isnan(rate[r.winner]) or np.isnan(rate[r.loser]) or rate[r.winner] == rate[r.loser]:
            continue
        n += 1
        hit += int(rate[r.winner] > rate[r.loser])
    return (hit / n if n else np.nan), n


# --------------------------------------------------------------------------- #
# The summary
# --------------------------------------------------------------------------- #
def structure_summary(bouts: pd.DataFrame, animals=None, zones=None,
                      min_together_s: float = 0.0, min_after_s: float = 0.0) -> dict:
    """Every readout for one group of animals (pass one sex of one trial).

    ``animals`` is the full roster, so an animal that never visits a zone still
    counts as landless; ``zones`` is the full list of zones on offer. With a
    ``day`` column, ownership is decided per day and ``owner_win`` uses the
    previous day's owner, so a win cannot make its own winner the owner.
    """
    animals = sorted(bouts["animal"].unique()) if animals is None else list(animals)
    zones = sorted(bouts["zone"].unique()) if zones is None else list(zones)
    n = len(animals)
    days = sorted(bouts["day"].unique()) if "day" in bouts.columns else [None]

    # --- who holds what, per day, then averaged over days -----------------
    per_day, owners_by_day = [], {}
    for d in days:
        b = bouts if d is None else bouts[bouts["day"] == d]
        t = zone_time(b, animals, zones)
        zc = zone_control(b, animals, zones)
        used = zc[zc["seconds"] > 0]
        owners_by_day[d] = zc["owner"].to_dict()
        if used.empty:
            continue
        owned = used["owner"].value_counts()
        cp = copresence(b)
        per_day.append(dict(
            owner_share=float(t.max(axis=0).sum() / t.values.sum()),
            zones_used=len(used),
            owners=len(owned),
            poles=float(1.0 / ((owned / len(used)) ** 2).sum()),
            top_owner_zones=float(owned.max() / len(used)),
            landless=float(1.0 - len(owned) / n),
            access_gini=gini(t.sum(axis=1).values),
            shared=float(cp["shared_s"].sum() / cp["occupied_s"].sum()) if cp["occupied_s"].sum() > 0 else np.nan,
            zone_hours_per_animal=float(t.values.sum() / 3600.0 / n),
        ))
    out = {"n_animals": n, "n_zones": len(zones), "n_days": len(per_day)}
    if per_day:
        out.update(pd.DataFrame(per_day).mean().to_dict())
        out["poles_max"] = float(min(n, out["zones_used"]))
    # does a zone keep its owner from one day to the next?
    kept = [owners_by_day[d0][z] == owners_by_day[d1][z]
            for d0, d1 in zip(days[:-1], days[1:]) for z in zones
            if owners_by_day[d0].get(z) is not None and owners_by_day[d1].get(z) is not None]
    out["owner_stays"] = float(np.mean(kept)) if kept else np.nan

    # --- who beats whom ----------------------------------------------------
    disp = displacements(bouts, min_together_s, min_after_s)
    disp = disp[disp["winner"].isin(animals) & disp["loser"].isin(animals)]
    W = win_matrix(disp, animals)
    out["displacements"] = len(disp)
    out["dyads_seen"] = int(((W.values + W.values.T) > 0).sum() // 2)
    out["dyads_possible"] = n * (n - 1) // 2
    out["landau_h"] = landau_h(W) if len(disp) else np.nan
    out["ttri"] = triangle_transitivity(W)
    out["dci"] = dci(W)
    out["top_win_share"] = float(W.sum(axis=1).max() / len(disp)) if len(disp) else np.nan
    out["resident_win"] = float((disp["winner"] == disp["resident"]).mean()) if len(disp) else np.nan
    if "day" in bouts.columns and len(disp):
        # the day of each displacement, from the visit table's own clock
        edges = bouts.groupby("day")["start"].min().sort_index()
        dd = edges.index.values[np.clip(np.searchsorted(edges.values, disp["time"].values, side="right") - 1,
                                        0, len(edges) - 1)]
        disp = disp.assign(day=dd)
        prev = {d: days[i - 1] for i, d in enumerate(days) if i > 0}
        out["owner_win"], out["owner_win_n"] = _owner_win(
            disp, lambda r: owners_by_day.get(prev.get(r.day), {}).get(r.zone))
    else:
        own = owners_by_day[days[0]]
        out["owner_win"], out["owner_win_n"] = _owner_win(disp, lambda r: own.get(r.zone))
    out["rank_transfer"], out["rank_transfer_n"] = _rank_transfer(disp)
    return out


# --------------------------------------------------------------------------- #
# Simulated runs -> the same visit table
# --------------------------------------------------------------------------- #
def bouts_from_positions(traj: pd.DataFrame, zones, x: str = "smoothed_x", y: str = "smoothed_y",
                         animal: str = "sexid", time: str = "time_sec",
                         max_gap_s: float | None = None) -> pd.DataFrame:
    """Zone visits from a trajectory in FNT's UWB schema.

    ``zones`` are rectangles with ``x``, ``y`` (centre), ``w``, ``d`` and
    ``label`` (:class:`~fnt.abma.core.config.ResourceZone`). A visit runs from
    the first sample inside a zone to the last, and ends when the animal is
    sampled outside it or a gap longer than ``max_gap_s`` (default: 1.5 sample
    intervals) passes. The table carries ``sex`` and ``day`` when the trajectory does.
    """
    ts = np.sort(traj[time].unique())
    step = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
    gap = 1.5 * step if max_gap_s is None else float(max_gap_s)
    t0 = float(ts[0]) if len(ts) else 0.0
    rows = []
    for a, g in traj.sort_values(time).groupby(animal, sort=False):
        gx, gy, gt = g[x].values, g[y].values, g[time].values.astype(float)
        sex = g["sex"].iloc[0] if "sex" in g else None
        for zi, z in enumerate(zones):
            inside = (np.abs(gx - z.x) <= z.w / 2.0) & (np.abs(gy - z.y) <= z.d / 2.0)
            idx = np.nonzero(inside)[0]
            if not len(idx):
                continue
            # a visit breaks where the samples inside stop being consecutive in time
            brk = np.nonzero((np.diff(idx) > 1) | (np.diff(gt[idx]) > gap))[0]
            first = np.concatenate([[0], brk + 1])
            last = np.concatenate([brk, [len(idx) - 1]])
            label = getattr(z, "label", "") or f"Z{zi + 1}"
            for f, l in zip(first, last):
                s, e = gt[idx[f]], gt[idx[l]] + step
                rows.append((a, sex, label, s - t0, e - t0, int((s - t0) // 86400) + 1))
    return pd.DataFrame(rows, columns=["animal", "sex", "zone", "start", "stop", "day"])


def bouts_from_run(project_dir: str, trial: str | None = None) -> pd.DataFrame:
    """The visit table of a finished ABMA run (its resource zones).

    One trial, or every trial in the folder when ``trial`` is None. The table
    carries ``trial``, ``sex`` and ``day``; an animal that never entered a zone
    still appears once, with an empty zone, so it stays on the roster.
    """
    import glob

    from .config import ExperimentConfig

    project_dir = os.path.expanduser(project_dir)
    cfg = ExperimentConfig.from_json(os.path.join(project_dir, "config.json"))
    files = sorted(glob.glob(os.path.join(project_dir, "data", f"uwb_{trial or '*'}_processed.csv")))
    if not files:
        raise FileNotFoundError(f"no uwb_*_processed.csv under {project_dir}")
    out = []
    for f in files:
        traj = pd.read_csv(f)
        b = bouts_from_positions(traj, cfg.arena.resource_zones)
        roster = traj.groupby("sexid")["sex"].first()
        absent = roster[~roster.index.isin(b["animal"])]
        b = pd.concat([b, pd.DataFrame({"animal": absent.index, "sex": absent.values})], ignore_index=True)
        b.insert(0, "trial", str(traj["Trial"].iloc[0]))
        out.append(b)
    return pd.concat(out, ignore_index=True)


def structure_table(visits: pd.DataFrame, by=("trial", "sex"), days=None, zones=None,
                    min_together_s: float = 0.0, min_after_s: float = 0.0) -> pd.DataFrame:
    """:func:`structure_summary` for every group of a visit table, one row each.

    ``by`` names the grouping columns (those absent from the table are
    dropped); give one sex per group. The roster of a group is every animal
    listed for it, including rows with an empty zone. The zones on offer are
    ``zones``, or else every zone visited by anyone who shares the group's
    columns other than ``sex``. ``days`` = (first, last) keeps those days.
    """
    by = [c for c in by if c in visits.columns]
    v = visits.copy()
    rows = []
    place = [c for c in by if c != "sex"]
    seen = (v.dropna(subset=["zone"]).groupby(place)["zone"].unique() if place
            else None)
    for key, g in (v.groupby(by) if by else [((), v)]):
        key = key if isinstance(key, tuple) else (key,)
        roster = sorted(g["animal"].unique())
        if zones is not None:
            zl = list(zones)
        elif seen is not None:
            k = tuple(key[by.index(c)] for c in place)
            zl = sorted(seen.loc[k if len(k) > 1 else k[0]])
        else:
            zl = sorted(v["zone"].dropna().unique())
        b = g.dropna(subset=["zone", "start", "stop"])
        if days is not None and "day" in b.columns:
            b = b[(b["day"] >= days[0]) & (b["day"] <= days[1])]
        if b.empty:
            continue
        rows.append({**dict(zip(by, key)), **structure_summary(b, roster, zl, min_together_s, min_after_s)})
    return pd.DataFrame(rows)
