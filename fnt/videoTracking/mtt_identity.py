"""Who is who across several cameras: names for Mask Tracker tracks from a position-tracking system (UWB),
carried along tracks and handed between cameras at doorways.

Built for the VoleCosm mesocosm (2026-11-18): four resource-zone cameras, each looking into one box, and an overhead
camera over the open floor, every vole wearing a UWB tag. UWB alone is too coarse to name voles moment by moment
indoors (slow tags, smoothing, radio bounce), but it is trustworthy when a vole is alone. So names are **anchored**
where the match is unambiguous and **carried** along the video tracks from there:

1. **Tracklets**: each camera's tracks, cut where they jump faster than an animal moves (a tracker swap) or are lost
   for long. Overhead positions go to floor metres through a floor map (``mtt_floor``).
2. **Anchors** (a frame where a tracklet's animal is plain): a zone camera that sees one vole while UWB has exactly
   one animal in that box and every other animal clear of it; an overhead vole whose nearest UWB animal is close,
   the next nearest far, and no other vole near it. A tracklet whose anchors name two animals holds a swap and is
   cut between them.
3. **Links** (probably the same animal): *handoffs* - a zone tracklet ends in one of its camera's exits (an
   ``exits.json``) and an overhead tracklet starts by that box within a few seconds, or the other way round;
   *bridges* - in one camera a tracklet ends and another starts close by soon after (mask flicker, a tracker cut).
4. **Names**: one integer programme labels every tracklet at once: anchors decide, the plain UWB distance breaks
   ties, and each link whose ends get different names costs its weight, so a name travels forward and backward
   along a chain. No animal is on two tracklets at once (except the two views of a vole in a doorway), nor in two
   places further apart than it could run in the time between. A tracklet nothing anchors or links is left
   unnamed rather than guessed. A chain whose evidence barely favours its name over the next is marked unsure.

    python -m fnt.videoTracking.mtt_identity <identity.json>

``identity.json``::

    {"start": "2026-02-07T12:20:00-07:00", "fps": 30, "frames": 63000, "uwb_offset_s": -121,
     "uwb": "<csv>", "uwb_columns": {"animal": "SexID", "time": "sec", "x": "x", "y": "y"},
     "zones": "<csv: zone_name, x, y (box corners, metres)>",
     "cameras": {"cam1": {"tracks": "<..._MaskTracker folder>", "zone": "Z1", "exits": "<exits.json>"}, ...,
                 "cam5": {"tracks": "<..._MaskTracker folder>", "floor_map": "<floor_map.json>"}},
     "out": "<folder>", "params": {...}}

``start`` is the clock time of frame 0 (every camera's tracks share frame numbers: clock-synced clips, e.g.
``fnt-grid --clips``); UWB second = clock second + ``uwb_offset_s``. Writes ``identities.csv`` (one row per
tracklet: camera, object_id, first and last frame, animal, unsure, margin, anchors) and ``links.csv``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Params:
    min_conf: float = 0.35           # detections below this are left out
    min_frames: int = 8              # shorter tracklets are left out (detections)
    jump_px: Dict[str, float] = field(default_factory=lambda: {"zone": 60, "overhead": 25})   # px per frame
    lost_frames: int = 20            # a track lost this long is cut
    anchor_near: float = 0.30        # m: an overhead anchor's UWB match is this close ...
    anchor_clear: float = 0.60       # ... and every other animal (and vole) this far
    anchor_every: int = 3            # anchors are looked for on every Nth frame
    min_run: int = 5                 # anchored samples for a run to count as a different animal (a swap)
    exit_px: float = 60              # a zone tracklet ends "in a doorway" within this of an exit
    link_s: float = 2.5              # handoff window (s)
    link_m: float = 0.5              # the overhead end of a handoff lies this close to the box (m)
    bridge_frames: int = 60
    bridge_px: Dict[str, List[float]] = field(default_factory=lambda: {"zone": [40, 15], "overhead": [15, 6]})
    w_handoff: float = 400.0         # link weights, in anchor-frame units
    w_bridge: float = 300.0
    anchor_w: float = 2.0            # per anchor frame
    weak: float = 0.2                # weight of the plain UWB distance (m per frame)
    unknown_m: float = 1.0           # an unnamed tracklet costs as much as the worst UWB match
    vmax: float = 1.0                # m/s between two places
    pair_s: float = 5.0              # tracklets further apart in time are not compared
    sure: float = 40.0               # a chain is unsure below this margin
    time_limit_s: float = 900.0


def box_dist(x, y, box):
    x0, x1, y0, y1 = box
    return np.hypot(np.maximum(np.maximum(x0 - x, 0), x - x1), np.maximum(np.maximum(y0 - y, 0), y - y1))


def _tracks_csv(path: str) -> str:
    if path.endswith(".csv"):
        return path
    return os.path.join(path, "trajectories.csv")


class Identity:
    def __init__(self, cfg: Dict):
        from .mtt_floor import FloorMap
        self.cfg = cfg
        self.p = Params(**cfg.get("params", {}))
        self.fps = float(cfg.get("fps", 30))
        self.n = int(cfg["frames"])
        start = datetime.fromisoformat(cfg["start"]).timestamp()
        z = pd.read_csv(cfg["zones"])
        self.boxes = {k: (g.x.min(), g.x.max(), g.y.min(), g.y.max()) for k, g in z.groupby("zone_name")}
        self.box_c = {k: np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2]) for k, b in self.boxes.items()}
        cols = {"animal": "SexID", "time": "sec", "x": "x", "y": "y", **cfg.get("uwb_columns", {})}
        u = pd.read_csv(cfg["uwb"])
        t = start + float(cfg.get("uwb_offset_s", 0)) + np.arange(self.n) / self.fps
        u = u[u[cols["time"]].between(t[0] - 30, t[-1] + 30)]
        self.animals = sorted(u[cols["animal"]].unique())
        self.uxy = {a: np.stack([np.interp(t, g[cols["time"]], g[cols["x"]]), np.interp(t, g[cols["time"]], g[cols["y"]])], 1)
                    for a, g in u.sort_values(cols["time"]).groupby(cols["animal"])}
        self.cams = cfg["cameras"]
        self.floor = {c: FloorMap.load(v["floor_map"]) for c, v in self.cams.items() if v.get("floor_map")}
        self.exits = {}
        for c, v in self.cams.items():
            if v.get("exits"):
                with open(v["exits"], encoding="utf-8") as fh:
                    d = json.load(fh)
                self.exits[c] = [np.array(p, np.float32) for e in d.values() for p in e.get("exits", [])]

    # ---------------------------------------------------------------- tracklets
    def kind(self, cam):
        return "overhead" if cam in self.floor else "zone"

    def _make(self, cam, obj, fr, xy):
        t = dict(id=-1, cam=cam, obj=int(obj), f0=int(fr[0]), f1=int(fr[-1]), frames=np.asarray(fr, int),
                 xy=np.asarray(xy, float))
        t["floor"] = self.floor[cam].to_floor(t["xy"]) if cam in self.floor else None
        return t

    def tracklets(self) -> List[Dict]:
        p, out = self.p, []
        for cam, v in self.cams.items():
            tj = pd.read_csv(_tracks_csv(v["tracks"]))
            tj = tj[(tj.confidence >= p.min_conf) & (tj.frame < self.n)]
            jump = p.jump_px[self.kind(cam)]
            for obj, g in tj.sort_values("frame").groupby("object_id"):
                fr, xy = g.frame.to_numpy(), g[["x", "y"]].to_numpy()
                step = np.hypot(*np.diff(xy, axis=0).T) / np.maximum(np.diff(fr), 1)
                cut = np.flatnonzero((step > jump) | (np.diff(fr) > p.lost_frames)) + 1
                for piece in np.split(np.arange(len(fr)), cut):
                    if len(piece) >= p.min_frames:
                        out.append(self._make(cam, obj, fr[piece], xy[piece]))
        return out

    def uwb_cost(self, t) -> np.ndarray:
        """Mean UWB distance (m, capped at 1) of a tracklet to each animal."""
        f = t["frames"][::3]
        out = []
        for a in self.animals:
            uv = self.uxy[a][f]
            if t["floor"] is not None:
                d = np.linalg.norm(t["floor"][::3] - uv, axis=1)
            else:
                d = box_dist(uv[:, 0], uv[:, 1], self.boxes[self.cams[t["cam"]]["zone"]])
            out.append(float(np.mean(np.minimum(d, 1.0))))
        return np.array(out)

    # ---------------------------------------------------------------- anchors
    def anchors(self, tl) -> List[tuple]:
        p = self.p
        active: Dict[str, Dict[int, list]] = {c: {} for c in self.cams}
        for t in tl:
            for k, f in enumerate(t["frames"]):
                if f % p.anchor_every == 0:
                    active[t["cam"]].setdefault(int(f), []).append((t["id"], k))
        A = np.array([self.uxy[a] for a in self.animals])                # animals x frames x 2
        ev = []
        for cam, by_frame in active.items():
            for f, here in by_frame.items():
                if cam not in self.floor:
                    if len(here) != 1:
                        continue
                    bd = box_dist(A[:, f, 0], A[:, f, 1], self.boxes[self.cams[cam]["zone"]])
                    inside = np.flatnonzero(bd <= 0.05)
                    if len(inside) == 1 and np.all(np.delete(bd, inside[0]) >= p.anchor_clear):
                        ev.append((here[0][0], f, int(inside[0])))
                else:
                    pos = np.array([tl[i]["floor"][k] for i, k in here])
                    for n_, (i, k) in enumerate(here):
                        others = np.delete(pos, n_, axis=0)
                        if len(others) and np.min(np.linalg.norm(others - pos[n_], axis=1)) < p.anchor_clear:
                            continue
                        d = np.linalg.norm(A[:, f] - pos[n_], axis=1)
                        o = np.argsort(d)
                        if d[o[0]] <= p.anchor_near and (len(o) < 2 or d[o[1]] >= p.anchor_clear):
                            ev.append((i, f, int(o[0])))
        return ev

    def split_swaps(self, tl, ev):
        p, out, n_cuts = self.p, [], 0
        by_t: Dict[int, list] = {}
        for i, f, a in ev:
            by_t.setdefault(i, []).append((f, a))
        for t in tl:
            runs = []
            for f, a in sorted(by_t.get(t["id"], [])):
                if runs and runs[-1][0] == a:
                    runs[-1][2] = f; runs[-1][3] += 1
                else:
                    runs.append([a, f, f, 1])
            keep = []
            for r in (r for r in runs if r[3] >= p.min_run):
                if keep and keep[-1][0] == r[0]:
                    keep[-1][2] = r[2]
                else:
                    keep.append(r)
            cuts = [(keep[i][2] + keep[i + 1][1]) // 2 for i in range(len(keep) - 1)]
            if not cuts:
                out.append(t)
                continue
            for piece in np.split(np.arange(len(t["frames"])), np.searchsorted(t["frames"], cuts)):
                if len(piece) >= p.min_frames:
                    out.append(self._make(t["cam"], t["obj"], t["frames"][piece], t["xy"][piece]))
            n_cuts += len(cuts)
        return out, n_cuts

    # ---------------------------------------------------------------- links
    def _near_exit(self, cam, x, y):
        import cv2
        return any(cv2.pointPolygonTest(q, (float(x), float(y)), True) > -self.p.exit_px for q in self.exits.get(cam, []))

    def links(self, tl) -> List[Dict]:
        p, out = self.p, []

        def add(i, j, kind, w):
            out.append(dict(a=i, b=j, kind=kind, w=float(w), cam_a=tl[i]["cam"], cam_b=tl[j]["cam"],
                            frame_a=tl[i]["f1"], frame_b=tl[j]["f0"]))

        over = [t for t in tl if t["floor"] is not None]
        for z in (t for t in tl if t["floor"] is None and t["cam"] in self.exits):
            box = self.boxes[self.cams[z["cam"]]["zone"]]
            if z["f1"] < self.n - 5 and self._near_exit(z["cam"], *z["xy"][-1]):            # zone -> floor
                for o in over:
                    dt = o["f0"] - z["f1"]
                    if -self.fps <= dt <= self.fps * p.link_s and o["f0"] > 2 and box_dist(*o["floor"][0], box) <= p.link_m:
                        add(z["id"], o["id"], f"handoff {z['cam']}->{o['cam']}", p.w_handoff * np.exp(-abs(dt) / self.fps))
            if z["f0"] > 2 and self._near_exit(z["cam"], *z["xy"][0]):                       # floor -> zone
                for o in over:
                    dt = z["f0"] - o["f1"]
                    if -self.fps <= dt <= self.fps * p.link_s and o["f1"] < self.n - 5 and box_dist(*o["floor"][-1], box) <= p.link_m:
                        add(o["id"], z["id"], f"handoff {o['cam']}->{z['cam']}", p.w_handoff * np.exp(-abs(dt) / self.fps))
        for cam in self.cams:
            base, speed = p.bridge_px[self.kind(cam)]
            mine = sorted((t for t in tl if t["cam"] == cam), key=lambda t: t["f0"])
            starts = np.array([t["f0"] for t in mine])
            for a in mine:
                lo_, hi_ = np.searchsorted(starts, a["f1"] + 1), np.searchsorted(starts, a["f1"] + p.bridge_frames, "right")
                for b in mine[lo_:hi_]:
                    gap = b["f0"] - a["f1"]
                    d = np.hypot(*(b["xy"][0] - a["xy"][-1]))
                    reach = base + speed * gap
                    if d <= reach:
                        add(a["id"], b["id"], f"bridge {cam}", p.w_bridge * np.exp(-gap / 30) * (1 - 0.5 * d / reach))
        return out

    # ---------------------------------------------------------------- names
    def _place(self, t, end):
        if t["floor"] is not None:
            return t["floor"][-1 if end else 0]
        return self.box_c[self.cams[t["cam"]]["zone"]]

    def _too_far(self, a, b):
        if a["f1"] > b["f0"]:
            a, b = b, a
        gap = max(b["f0"] - a["f1"], 0) / self.fps
        slack = 0.35 * ((a["floor"] is None) + (b["floor"] is None))           # a box is ~0.5 m across
        return np.linalg.norm(self._place(a, True) - self._place(b, False)) > self.p.vmax * gap + slack + 0.15

    def _doorway_pair(self, a, b, both):
        if (a["floor"] is None) == (b["floor"] is None):
            return False
        z, o = (a, b) if b["floor"] is not None else (b, a)
        if "zone" not in self.cams[z["cam"]]:
            return False
        fo = o["floor"][np.isin(o["frames"], list(both))]
        box = self.boxes[self.cams[z["cam"]]["zone"]]
        return len(fo) > 0 and np.median(box_dist(fo[:, 0], fo[:, 1], box)) <= self.p.link_m

    def name(self, tl, anchor, links):
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
        p = self.p
        nT, nA = len(tl), len(self.animals) + 1
        Cm = np.array([list(t["cost"]) + [p.unknown_m] for t in tl])
        nfr = np.array([len(t["frames"]) for t in tl], float)
        nL = len(links)
        nx, ny = nT * nA, nL * nA
        ev = p.weak * Cm * nfr[:, None]
        ev[:, :-1] -= p.anchor_w * anchor
        c = np.concatenate([ev.ravel(), np.repeat([lk["w"] / 2 for lk in links], nA)])
        R_, C_, V_, lo, hi = [], [], [], [], []

        def row(entries, l, h):
            r = len(lo)
            for k, v in entries:
                R_.append(r); C_.append(k); V_.append(v)
            lo.append(l); hi.append(h)

        for t in range(nT):
            row([(t * nA + a, 1) for a in range(nA)], 1, 1)
        for li, lk in enumerate(links):
            for a in range(nA):
                row([(nx + li * nA + a, 1), (lk["a"] * nA + a, -1), (lk["b"] * nA + a, 1)], 0, np.inf)
                row([(nx + li * nA + a, 1), (lk["a"] * nA + a, 1), (lk["b"] * nA + a, -1)], 0, np.inf)
        handoffs = {frozenset((lk["a"], lk["b"])) for lk in links if lk["kind"].startswith("handoff")}
        fsets = [set(t["frames"].tolist()) for t in tl]
        order = sorted(range(nT), key=lambda i: tl[i]["f0"])
        for k, i in enumerate(order):
            for j in order[k + 1:]:
                if tl[j]["f0"] > tl[i]["f1"] + p.pair_s * self.fps:
                    break
                if tl[j]["f0"] <= tl[i]["f1"]:
                    both = fsets[i] & fsets[j]
                    clash = (frozenset((i, j)) not in handoffs and not self._doorway_pair(tl[i], tl[j], both)) \
                        if len(both) > 3 else self._too_far(tl[i], tl[j])
                else:
                    clash = self._too_far(tl[i], tl[j])
                if clash:
                    for a in range(nA - 1):
                        row([(i * nA + a, 1), (j * nA + a, 1)], 0, 1)
        A_ = coo_matrix((V_, (R_, C_)), shape=(len(lo), nx + ny)).tocsr()
        res = milp(c, constraints=LinearConstraint(A_, lo, hi), integrality=np.r_[np.ones(nx), np.zeros(ny)],
                   bounds=Bounds(0, 1), options={"time_limit": p.time_limit_s})
        if res.x is None:
            raise RuntimeError(f"naming failed: {res.message}")
        lab = res.x[:nx].reshape(nT, nA).argmax(1)
        parent = list(range(nT))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for lk in links:
            if lab[lk["a"]] == lab[lk["b"]]:
                parent[find(lk["a"])] = find(lk["b"])
        margin = np.full(nT, np.inf)
        chains: Dict[int, list] = {}
        for t in range(nT):
            chains.setdefault(find(t), []).append(t)
        for members in chains.values():
            a = lab[members[0]]
            if a < nA - 1:
                e = ev[members].sum(0)[:-1]
                margin[members] = float(np.delete(e, a).min() - e[a]) if nA > 2 else np.inf
        return lab, margin, Cm

    # ---------------------------------------------------------------- all
    def run(self, log=print) -> pd.DataFrame:
        tl = self.tracklets()
        for i, t in enumerate(tl):
            t["id"] = i
        tl, n_cuts = self.split_swaps(tl, self.anchors(tl))
        anchor = np.zeros((len(tl), len(self.animals)))
        for i, t in enumerate(tl):
            t["id"] = i
            t["cost"] = self.uwb_cost(t)
        for i, f, a in self.anchors(tl):
            anchor[i, a] += self.p.anchor_every
        log(f"{len(tl)} tracklets ({n_cuts} cut where their anchors changed animal); anchors on "
            f"{int((anchor.sum(1) > 0).sum())}")
        links = self.links(tl)
        lab, margin, Cm = self.name(tl, anchor, links)
        rows = []
        for t in tl:
            a = int(lab[t["id"]])
            name = self.animals[a] if a < len(self.animals) else ""
            rows.append(dict(camera=t["cam"], object_id=t["obj"], frame_first=t["f0"], frame_last=t["f1"],
                             detections=len(t["frames"]), animal=name,
                             unsure=bool(name and margin[t["id"]] < self.p.sure),
                             margin=round(float(margin[t["id"]]), 1) if name else "",
                             anchors=" ".join(f"{self.animals[j]}:{int(v)}" for j, v in enumerate(anchor[t["id"]]) if v),
                             uwb_m=round(float(Cm[t["id"], a]), 3) if name else ""))
        self.links_ = pd.DataFrame(links)
        if len(self.links_):
            self.links_["kept"] = [lab[lk["a"]] == lab[lk["b"]] for lk in links]
        out = pd.DataFrame(rows).sort_values(["camera", "frame_first"])
        named = out[out.animal != ""].detections.sum() / max(out.detections.sum(), 1)
        log(f"{len(links)} candidate links, {int(self.links_.kept.sum()) if len(links) else 0} kept; named "
            f"{named:.0%} of tracked detections")
        return out


def lookup(identities: pd.DataFrame):
    """(camera, object_id, frame) -> (animal or "", unsure): which tracklet a detection belongs to."""
    by = {}
    for r in identities.itertuples():
        by.setdefault((r.camera, int(r.object_id)), []).append((int(r.frame_first), int(r.frame_last),
                                                               r.animal if isinstance(r.animal, str) else "",
                                                               bool(r.unsure)))

    def get(cam, obj, frame):
        for f0, f1, a, u in by.get((cam, int(obj)), ()):
            if f0 <= frame <= f1:
                return a, u
        return "", False
    return get


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="identity.json (see the module's description)")
    a = ap.parse_args(argv)
    with open(a.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    out = cfg.get("out") or os.path.dirname(os.path.abspath(a.config))
    os.makedirs(out, exist_ok=True)
    idn = Identity(cfg)
    res = idn.run()
    res.to_csv(os.path.join(out, "identities.csv"), index=False)
    idn.links_.to_csv(os.path.join(out, "links.csv"), index=False)
    print("wrote", os.path.join(out, "identities.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
