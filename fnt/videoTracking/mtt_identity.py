"""Who is who across several cameras: names for Mask Tracker tracks from a position-tracking system (UWB),
carried along tracks and handed between cameras at doorways.

Built for the VoleCosm mesocosm (2026-11-18): four resource-zone cameras, each looking into one box, and an overhead
camera over the open floor, every vole wearing a UWB tag. UWB alone is too coarse to name voles moment by moment
indoors (slow tags, smoothing, radio bounce), but it is trustworthy when a vole is alone. So names are **anchored**
where the match is unambiguous and **carried** along the video tracks from there:

1. **Tracklets**: each camera's tracks, cut where they jump faster than an animal moves (a tracker swap) or are lost
   for long. Overhead positions go to floor metres through a floor map (``mtt_floor``). On the overhead camera a
   second mask on an animal already detected in that frame (masks overlapping, typically the whole body plus the
   head alone) is left out first, keeping the larger mask (a camera's ``"dedupe"``: true/false overrides).
2. **Anchors** (a frame where a tracklet's animal is plain): a zone camera that sees one vole while UWB has exactly
   one animal in that box and every other animal clear of it; an overhead vole whose nearest UWB animal is close,
   the next nearest far, and no other vole near it. A tracklet whose anchors name two animals holds a swap and is
   cut between them.
3. **Links** (probably the same animal): *handoffs* - a zone tracklet ends in one of its camera's exits (an
   ``exits.json``) and an overhead tracklet starts by that box within a few seconds, or the other way round;
   *bridges* - in one camera a tracklet ends and another starts close by soon after (mask flicker, a tracker cut);
   *doorway links* - one vole seen at once by a zone camera (touching an exit) and by the overhead camera (at that
   exit's door on the floor): a strong link. Door positions come from the data (the overhead position of the one
   vole by the box while one zone detection touches the exit) and are kept in ``doors.json`` in the ``out``
   folder (edit it by hand, or ``--derive-doors`` to place them again).
4. **Names**: one integer programme labels every tracklet at once: anchors decide, the plain UWB distance breaks
   ties, and each link whose ends get different names costs its weight (an unnamed end costs nothing), so a name
   travels forward and backward along a chain. No animal is on two tracklets at once (except the two views of a vole in a doorway), nor in two
   places further apart than it could run in the time between. A tracklet nothing anchors or links is left
   unnamed rather than guessed. A chain whose evidence barely favours its name over the next is marked unsure.
5. **Corrections** (``corrections.csv`` in the ``out`` folder, written by the Fusion View window): rows of
   camera, object_id, frame_first, frame_last, animal (``none``: not an animal). Tracklets are cut at each
   correction's ends and the stretch inside is held to the name given; every other rule still applies, so one
   correction renames whatever its chain carries. A correction the rules can't meet is reported, not forced.

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
tracklet: camera, object_id, first and last frame, animal, unsure, margin, anchors) and ``links.csv``, plus
``doors.json``, ``duplicates.csv`` (detections left out) and ``door_links.csv`` when there are any.
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

FAR = 1e3    # m: where an animal not in the arena is placed (never near anything)


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
    present_pad_s: float = 2.0       # s: an animal counts as in the arena this long before its first UWB fix
    unknown_m: float = 1.0           # an unnamed tracklet costs as much as the worst UWB match
    vmax: float = 1.0                # m/s between two places
    pair_s: float = 5.0              # tracklets further apart in time are not compared
    sure: float = 40.0               # a chain is unsure below this margin
    time_limit_s: float = 900.0
    w_fixed: float = 1e6             # cost of breaking a correction (only when the hard rules leave no choice)
    dedupe_iou: float = 0.3          # two masks in one frame overlapping this much (IoU) ...
    dedupe_contain: float = 0.6      # ... or one this much inside the other (of the smaller's area) are one animal
    door_touch_px: float = 15        # a zone detection's box this close to an exit polygon touches the exit
    door_box_m: float = 0.3          # deriving a door: the one overhead vole within this of the box (m)
    door_min_n: int = 5              # frames needed to place a door
    door_m: float = 0.15             # an overhead detection this close to a door (m) is the vole in that doorway
    door_min_frames: int = 2         # frames a zone/overhead pair must share a doorway to be linked
    w_door: float = 2000.0           # weight of a doorway link (much stronger than a handoff)
    doorway_q: float = 0.9           # a zone and an overhead tracklet seen at once are one vole in a doorway when this
                                     # share of the overhead one's shared frames is within link_m of the box (was
                                     # the median: a tracklet half by the box and half out on the floor passed)
    conflict_rounds: int = 3        # re-solves after cutting tracklets left unnamed by a hard conflict (0: off)
    conflict_margin_s: float = 0.5   # s added to the travel time (at vmax) kept clear on each side of a conflict


def box_dist(x, y, box):
    x0, x1, y0, y1 = box
    return np.hypot(np.maximum(np.maximum(x0 - x, 0), x - x1), np.maximum(np.maximum(y0 - y, 0), y - y1))


def _tracks_csv(path: str) -> str:
    if path.endswith(".csv"):
        return path
    return os.path.join(path, "trajectories.csv")


def _mask_store(path: str) -> Optional[str]:
    """The ``*_tracks.h5`` (masks) beside a camera's tracks, or None."""
    d = os.path.dirname(path) if path.endswith(".csv") else path
    found = sorted(glob.glob(os.path.join(glob.escape(d), "*_tracks.h5")))
    return found[0] if found else None


def find_duplicates(tj: pd.DataFrame, store: str, iou: float = 0.3, contain: float = 0.6) -> pd.DataFrame:
    """Detections that are a second mask on an animal already detected in the same frame.

    The overhead model sometimes puts two masks on one vole (typically the whole body and the head/cap alone). Two
    masks of one frame are one animal when their IoU exceeds ``iou`` or the intersection covers ``contain`` of the
    smaller; the larger (containing) mask is kept, since the higher-confidence one is often the head alone. Mask
    overlap only: two real animals side by side don't overlap. Returns one row per dropped detection (frame, drop,
    keep, iou, contained)."""
    from .track_store import TrackMaskReader
    rows = []
    multi = tj[tj.duplicated("frame", keep=False)]
    if multi.empty:
        return pd.DataFrame(columns=["frame", "drop", "keep", "iou", "contained"])
    has_box = all(c in tj for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
    rd = TrackMaskReader(store)
    try:
        for f, g in multi.groupby("frame"):
            objs = g.object_id.astype(int).tolist()
            if has_box:                                   # pairs whose boxes meet; nothing else can overlap
                b = g[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
                pairs = [(i, j) for i in range(len(b)) for j in range(i + 1, len(b))
                         if b[i, 0] <= b[j, 2] and b[j, 0] <= b[i, 2] and b[i, 1] <= b[j, 3] and b[j, 1] <= b[i, 3]]
                if not pairs:
                    continue
            else:
                pairs = [(i, j) for i in range(len(objs)) for j in range(i + 1, len(objs))]
            row_of = {int(rd.object_id[r]): r for r in rd.rows_for_frame(int(f))}
            masks: Dict[int, Optional[np.ndarray]] = {}

            def mask(o):
                if o not in masks:
                    masks[o] = rd.mask_for_row(row_of[o]) if o in row_of else None
                return masks[o]
            gone = set()
            for i, j in pairs:
                a, b_ = objs[i], objs[j]
                if a in gone or b_ in gone:
                    continue
                ma, mb = mask(a), mask(b_)
                if ma is None or mb is None:
                    continue
                inter = int(np.logical_and(ma, mb).sum())
                if not inter:
                    continue
                sa, sb = int(ma.sum()), int(mb.sum())
                v_iou, v_cont = inter / (sa + sb - inter), inter / max(min(sa, sb), 1)
                if v_iou > iou or v_cont > contain:
                    keep, drop = (a, b_) if sa >= sb else (b_, a)
                    gone.add(drop)
                    rows.append(dict(frame=int(f), drop=drop, keep=keep, iou=round(v_iou, 3),
                                     contained=round(v_cont, 3)))
    finally:
        rd.close()
    return pd.DataFrame(rows, columns=["frame", "drop", "keep", "iou", "contained"])


def _touches(poly: np.ndarray, tj: pd.DataFrame, px: float) -> np.ndarray:
    """Which detections' boxes touch (or lie within ``px`` of) an exit polygon; the centre point without boxes."""
    import cv2
    x0, y0 = poly.min(0) - px
    x1, y1 = poly.max(0) + px
    if all(c in tj for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
        b = tj[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float)
    else:
        xy = tj[["x", "y"]].to_numpy(float)
        b = np.c_[xy, xy]
    out = np.zeros(len(b), bool)
    near = np.flatnonzero((b[:, 0] <= x1) & (b[:, 2] >= x0) & (b[:, 1] <= y1) & (b[:, 3] >= y0))
    for k in near:
        bx1, by1, bx2, by2 = b[k]
        if cv2.pointPolygonTest(poly, ((bx1 + bx2) / 2, (by1 + by2) / 2), True) > -px:
            out[k] = True
        elif bx2 > bx1 and by2 > by1:
            box = np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], np.float32)
            out[k] = cv2.intersectConvexConvex(poly, box)[0] > 1
    return out


def doors_path(cfg: Dict, cfg_path: Optional[str] = None) -> str:
    out = cfg.get("out") or (os.path.dirname(os.path.abspath(cfg_path)) if cfg_path else "")
    return cfg.get("doors") or (os.path.join(out, "doors.json") if out else "")


NOT_ANIMAL = "none"


def corrections_path(cfg: Dict, cfg_path: Optional[str] = None) -> str:
    out = cfg.get("out") or (os.path.dirname(os.path.abspath(cfg_path)) if cfg_path else "")
    return cfg.get("corrections") or (os.path.join(out, "corrections.csv") if out else "")


def read_corrections(path: str) -> pd.DataFrame:
    cols = ["camera", "object_id", "frame_first", "frame_last", "animal"]
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path, keep_default_na=False)[cols]


class Identity:
    def __init__(self, cfg: Dict, corrections: Optional[pd.DataFrame] = None):
        from .mtt_floor import FloorMap
        self.cfg = cfg
        self.corrections = (corrections if corrections is not None
                            else read_corrections(corrections_path(cfg))).reset_index(drop=True)
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
        self.uxy, self.present = {}, {}
        for a, g in u.sort_values(cols["time"]).groupby(cols["animal"]):
            ts = g[cols["time"]].to_numpy(float)
            xy = np.stack([np.interp(t, ts, g[cols["x"]]), np.interp(t, ts, g[cols["y"]])], 1)
            # before an animal's first fix (not yet released) and after its last it is not in the arena: placed far
            # away, so it is never an anchor nor near a vole, and a tracklet mostly outside that span can't take it
            on = (t >= ts[0] - self.p.present_pad_s) & (t <= ts[-1] + self.p.present_pad_s)
            xy[~on] = FAR
            self.uxy[a], self.present[a] = xy, on
        self.cams = cfg["cameras"]
        self.floor = {c: FloorMap.load(v["floor_map"]) for c, v in self.cams.items() if v.get("floor_map")}
        self.exits = {}
        for c, v in self.cams.items():
            if v.get("exits"):
                with open(v["exits"], encoding="utf-8") as fh:
                    d = json.load(fh)
                self.exits[c] = [np.array(p, np.float32) for e in d.values() for p in e.get("exits", [])]
        self.doors: Dict[tuple, Dict] = {}               # (zone camera, exit index) -> {"xy", "n", "spread_m"}
        self.duplicates_ = pd.DataFrame()
        self.door_links_ = pd.DataFrame(columns=["zone", "over", "frames", "f0", "f1"])
        self._tj: Dict[str, pd.DataFrame] = {}
        self._touch: Dict[tuple, np.ndarray] = {}

    # ---------------------------------------------------------------- tracklets
    def kind(self, cam):
        return "overhead" if cam in self.floor else "zone"

    def dedupe_on(self, cam) -> bool:
        """Duplicate masks are merged on overhead cameras unless a camera's ``"dedupe"`` says otherwise."""
        return bool(self.cams[cam].get("dedupe", self.kind(cam) == "overhead"))

    def detections(self, cam) -> pd.DataFrame:
        """A camera's detections as named: confident enough, in range, a second mask on one animal left out."""
        if cam in self._tj:
            return self._tj[cam]
        p, v = self.p, self.cams[cam]
        tj = pd.read_csv(_tracks_csv(v["tracks"]))
        tj = tj[(tj.confidence >= p.min_conf) & (tj.frame < self.n)].reset_index(drop=True)
        store = _mask_store(v["tracks"]) if self.dedupe_on(cam) else None
        if store:
            dup = find_duplicates(tj, store, p.dedupe_iou, p.dedupe_contain)
            if len(dup):
                key = pd.MultiIndex.from_frame(tj[["frame", "object_id"]].astype(int))
                drop = key.isin(pd.MultiIndex.from_frame(dup[["frame", "drop"]].astype(int)))
                # the kept mask stands for both: it takes the higher confidence of the pair
                best = dup.merge(tj[["frame", "object_id", "confidence"]], left_on=["frame", "drop"],
                                 right_on=["frame", "object_id"]).groupby(["frame", "keep"]).confidence.max()
                ki = pd.MultiIndex.from_arrays([tj.frame.astype(int), tj.object_id.astype(int)])
                up = best.reindex(ki).to_numpy()
                tj["confidence"] = np.fmax(tj.confidence.to_numpy(float), up)
                tj = tj[~drop].reset_index(drop=True)
                self.duplicates_ = pd.concat([self.duplicates_, dup.assign(camera=cam)], ignore_index=True)
        self._tj[cam] = tj
        return tj

    def touching(self, cam, k) -> np.ndarray:
        """Which of a zone camera's detections touch its exit ``k``."""
        if (cam, k) not in self._touch:
            self._touch[(cam, k)] = _touches(self.exits[cam][k], self.detections(cam), self.p.door_touch_px)
        return self._touch[(cam, k)]

    def derive_doors(self) -> Dict[tuple, Dict]:
        """The floor position of each zone-camera exit, from the data: the median overhead position of the one vole
        by that box while exactly one zone detection touches the exit. An exit seen fewer than ``door_min_n`` times
        gets no door (and no doorway links)."""
        p, doors = self.p, {}
        if not self.floor:
            return doors
        ocam = next(iter(self.floor))
        o = self.detections(ocam)
        fl = self.floor[ocam].to_floor(o[["x", "y"]].to_numpy(float))
        by_f = {int(f): idx for f, idx in o.groupby("frame").indices.items()}
        for cam, polys in self.exits.items():
            if cam in self.floor or "zone" not in self.cams[cam]:
                continue
            box = self.boxes[self.cams[cam]["zone"]]
            z = self.detections(cam)
            for k in range(len(polys)):
                hit = z[self.touching(cam, k)]
                n_by = hit.groupby("frame").size()
                pts = []
                for f in n_by.index[n_by == 1]:
                    idx = by_f.get(int(f))
                    if idx is None:
                        continue
                    d = box_dist(fl[idx, 0], fl[idx, 1], box)
                    if (d <= p.door_box_m).sum() == 1:
                        pts.append(fl[idx][d <= p.door_box_m][0])
                if len(pts) >= p.door_min_n:
                    pts = np.array(pts)
                    med = np.median(pts, 0)
                    doors[(cam, k)] = dict(xy=med, n=len(pts),
                                           spread_m=float(np.median(np.linalg.norm(pts - med, axis=1))))
        return doors

    @staticmethod
    def save_doors(doors: Dict[tuple, Dict], path: str):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({f"{c}:{k}": dict(x=round(float(v["xy"][0]), 4), y=round(float(v["xy"][1]), 4), n=int(v["n"]),
                                        spread_m=round(float(v["spread_m"]), 4)) for (c, k), v in doors.items()},
                      fh, indent=1)

    @staticmethod
    def load_doors(path: str) -> Dict[tuple, Dict]:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return {(k.rsplit(":", 1)[0], int(k.rsplit(":", 1)[1])): dict(xy=np.array([v["x"], v["y"]], float),
                                                                       n=v.get("n", 0), spread_m=v.get("spread_m", 0))
                for k, v in d.items()}

    def use_doors(self, path: Optional[str] = None, rederive: bool = False, log=print):
        """Door positions from ``path`` when it exists (so a hand-edited file is kept), else derived and saved
        there."""
        path = path if path is not None else doors_path(self.cfg)
        if path and os.path.exists(path) and not rederive:
            self.doors = self.load_doors(path)
            log(f"doors: {len(self.doors)} from {path}")
            return
        self.doors = self.derive_doors()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.save_doors(self.doors, path)
        log(f"doors: {len(self.doors)} derived from the data" + (f" -> {path}" if path else ""))

    def _make(self, cam, obj, fr, xy):
        t = dict(id=-1, cam=cam, obj=int(obj), f0=int(fr[0]), f1=int(fr[-1]), frames=np.asarray(fr, int),
                 xy=np.asarray(xy, float))
        t["floor"] = self.floor[cam].to_floor(t["xy"]) if cam in self.floor else None
        return t

    def tracklets(self) -> List[Dict]:
        p, out = self.p, []
        for cam in self.cams:
            tj = self.detections(cam)
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

    def apply_corrections(self, tl):
        """Cut tracklets at each correction's ends; return them and {tracklet index: animal index or None}."""
        cuts: Dict[tuple, set] = {}
        for r in self.corrections.itertuples():
            cuts.setdefault((r.camera, int(r.object_id)), set()).update((int(r.frame_first), int(r.frame_last) + 1))
        out = []
        for t in tl:
            c = sorted(f for f in cuts.get((t["cam"], t["obj"]), ()) if t["f0"] < f <= t["f1"])
            if not c:
                out.append(t)
                continue
            for piece in np.split(np.arange(len(t["frames"])), np.searchsorted(t["frames"], c)):
                if len(piece):
                    out.append(self._make(t["cam"], t["obj"], t["frames"][piece], t["xy"][piece]))
        fixed = {}
        for i, t in enumerate(out):
            for r in self.corrections.itertuples():
                if (r.camera, int(r.object_id)) == (t["cam"], t["obj"]) and int(r.frame_first) <= t["f0"]                         and t["f1"] <= int(r.frame_last):
                    a = str(r.animal)
                    if a != NOT_ANIMAL and a not in self.animals:
                        raise ValueError(f"correction names {a}, not one of {', '.join(self.animals)}")
                    fixed[i] = None if a == NOT_ANIMAL else self.animals.index(a)
        return out, fixed

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
        out += self.door_links(tl, out)
        return out

    def door_links(self, tl, links) -> List[Dict]:
        """Doorway links: a frame where exactly one zone detection touches an exit and exactly one overhead detection
        lies within ``door_m`` of that exit's door shows one vole in two views, so their tracklets get a strong link
        (``w_door``). A pair that is already a handoff has that link raised to ``w_door`` instead. Not a handoff:
        the two views may overlap in time only while the overhead one stays by the box (the doorway rule)."""
        p = self.p
        self.door_links_ = pd.DataFrame(columns=["zone", "over", "frames", "f0", "f1"])
        if not self.doors:
            return []
        owner = {}                                       # (camera, object, frame) -> tracklet
        over: Dict[int, list] = {}                       # frame -> [(tracklet, floor xy)] overhead
        for t in tl:
            for k, f in enumerate(t["frames"]):
                owner[(t["cam"], t["obj"], int(f))] = t["id"]
                if t["floor"] is not None:
                    over.setdefault(int(f), []).append((t["id"], t["floor"][k]))
        pairs: Dict[tuple, list] = {}
        for (cam, k), door in self.doors.items():
            if cam not in self.exits or k >= len(self.exits[cam]):
                continue
            z = self.detections(cam)
            hit = z[self.touching(cam, k)]
            one = hit[hit.groupby("frame").frame.transform("size") == 1]
            for f, obj in zip(one.frame.astype(int), one.object_id.astype(int)):
                zi = owner.get((cam, obj, f))
                near = [i for i, q in over.get(f, ()) if np.hypot(*(q - door["xy"])) <= p.door_m]
                if zi is not None and len(near) == 1:
                    pairs.setdefault((zi, near[0]), []).append(f)
        out, rows = [], []
        by_pair = {frozenset((lk["a"], lk["b"])): lk for lk in links}
        for (zi, oi), frs in pairs.items():
            if len(frs) < p.door_min_frames:
                continue
            rows.append(dict(zone=zi, over=oi, frames=len(frs), f0=min(frs), f1=max(frs)))
            lk = by_pair.get(frozenset((zi, oi)))
            if lk is not None:
                lk["w"] = max(lk["w"], p.w_door)
                if "+door" not in lk["kind"]:
                    lk["kind"] += "+door"
                continue
            a, b = (zi, oi) if tl[zi]["f0"] <= tl[oi]["f0"] else (oi, zi)
            lk = dict(a=a, b=b, kind=f"door {tl[zi]['cam']}", w=float(p.w_door), cam_a=tl[a]["cam"],
                      cam_b=tl[b]["cam"], frame_a=tl[a]["f1"], frame_b=tl[b]["f0"])
            by_pair[frozenset((zi, oi))] = lk
            out.append(lk)
        self.door_links_ = pd.DataFrame(rows, columns=["zone", "over", "frames", "f0", "f1"])
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
        return len(fo) > 0 and np.quantile(box_dist(fo[:, 0], fo[:, 1], box), self.p.doorway_q) <= self.p.link_m

    def clashes(self, tl, links) -> List[tuple]:
        """Pairs of tracklets that can never be the same animal (the hard rules): seen at the same time in two
        places (a handoff or a doorway pair excepted), or too far apart for the time between them."""
        p = self.p
        handoffs = {frozenset((lk["a"], lk["b"])) for lk in links if lk["kind"].startswith("handoff")}
        fsets = [set(t["frames"].tolist()) for t in tl]
        order = sorted(range(len(tl)), key=lambda i: tl[i]["f0"])
        out = []
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
                    out.append((i, j))
        return out

    def conflict_cuts(self, tl, lab, anchor, Cm, links) -> Dict[int, List[int]]:
        """Where to cut tracklets the hard rules left unnamed. A tracklet whose evidence (anchors, UWB) names an
        animal that a clashing tracklet holds lost all of its frames to that clash, often a long one to a short
        overlap. Cut it around the other tracklet's span, widened on each side by the time an animal needs (at
        ``vmax``, plus ``conflict_margin_s``) to get between the two places, so only that stretch is in doubt and
        the rest can take the name. Returns {tracklet: frames to cut before}."""
        p = self.p
        U = len(self.animals)
        nfr = np.array([len(t["frames"]) for t in tl], float)
        ev = p.weak * Cm * nfr[:, None]
        ev[:, :-1] -= p.anchor_w * anchor
        want = {}
        for i in np.flatnonzero(np.asarray(lab) == U):
            a = int(np.argmin(ev[i, :-1]))
            if ev[i, a] < ev[i, U] and nfr[i] >= 2 * p.min_frames:
                want[int(i)] = a
        def pos(t, f):                                   # where tracklet t is at (or nearest to) frame f
            if t["floor"] is None:
                return self.box_c[self.cams[t["cam"]]["zone"]]
            return t["floor"][min(np.searchsorted(t["frames"], f), len(t["frames"]) - 1)]

        def pad(t, o, f, end):                           # frames t needs to get to (from) o's start (end)
            slack = 0.35 * ((t["floor"] is None) + (o["floor"] is None)) + 0.15
            d = np.linalg.norm(pos(t, f) - self._place(o, end))
            return int(np.ceil((max(d - slack, 0) / p.vmax + p.conflict_margin_s) * self.fps))

        cuts: Dict[int, set] = {}
        for i, j in self.clashes(tl, links):
            for u, v in ((i, j), (j, i)):
                if u not in want or int(lab[v]) != want[u]:
                    continue
                t, o = tl[u], tl[v]
                w0 = o["f0"] - pad(t, o, o["f0"], False)
                w1 = o["f1"] + pad(t, o, o["f1"], True)
                c = {f for f in (w0, w1 + 1) if t["f0"] < f <= t["f1"]}
                if c:
                    cuts.setdefault(u, set()).update(c)
        return {i: sorted(c) for i, c in cuts.items()}

    def cut(self, tl, cuts: Dict[int, List[int]]) -> List[Dict]:
        """The tracklets with each one in ``cuts`` cut before those frames (every piece kept)."""
        out = []
        for i, t in enumerate(tl):
            if i not in cuts:
                out.append(t)
                continue
            for piece in np.split(np.arange(len(t["frames"])), np.searchsorted(t["frames"], cuts[i])):
                if len(piece):
                    out.append(self._make(t["cam"], t["obj"], t["frames"][piece], t["xy"][piece]))
        return out

    def name(self, tl, anchor, links, fixed=None):
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
        p = self.p
        nT, nA = len(tl), len(self.animals) + 1
        Cm = np.array([list(t["cost"]) + [p.unknown_m] for t in tl])
        nfr = np.array([len(t["frames"]) for t in tl], float)
        nL = len(links)
        nx, ny = nT * nA, nL
        ev = p.weak * Cm * nfr[:, None]
        ev[:, :-1] -= p.anchor_w * anchor
        c = np.concatenate([ev.ravel(), [lk["w"] for lk in links]])
        ub = np.ones(nx + ny)
        for i, t in enumerate(tl):                       # an animal not yet in the arena (or gone) can't be named
            for a, an in enumerate(self.animals):
                if self.present[an][t["frames"]].mean() < 0.5:
                    ub[i * nA + a] = 0
        for i, a in (fixed or {}).items():               # a correction: every other choice costs w_fixed
            k = nA - 1 if a is None else a
            c[i * nA:(i + 1) * nA] += p.w_fixed
            c[i * nA + k] -= p.w_fixed
            ub[i * nA + k] = 1                           # the person reviewing knows better than the UWB span
        R_, C_, V_, lo, hi = [], [], [], [], []

        def row(entries, l, h):
            r = len(lo)
            for k, v in entries:
                R_.append(r); C_.append(k); V_.append(v)
            lo.append(l); hi.append(h)

        for t in range(nT):
            row([(t * nA + a, 1) for a in range(nA)], 1, 1)
        # a link costs its weight when its two ends carry different names; a named end beside an unnamed one is
        # free, so an unnamed (often spurious) tracklet doesn't pull its anchored neighbours unnamed
        U = nA - 1
        for li, lk in enumerate(links):
            i, j = lk["a"], lk["b"]
            for a in range(U):
                row([(nx + li, 1), (i * nA + a, -1), (j * nA + a, 1), (j * nA + U, 1)], 0, np.inf)
                row([(nx + li, 1), (j * nA + a, -1), (i * nA + a, 1), (i * nA + U, 1)], 0, np.inf)
        for i, j in self.clashes(tl, links):
            for a in range(nA - 1):
                row([(i * nA + a, 1), (j * nA + a, 1)], 0, 1)
        A_ = coo_matrix((V_, (R_, C_)), shape=(len(lo), nx + ny)).tocsr()
        res = milp(c, constraints=LinearConstraint(A_, lo, hi), integrality=np.r_[np.ones(nx), np.zeros(ny)],
                   bounds=Bounds(0, ub), options={"time_limit": p.time_limit_s})
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
    def run(self, log=print, doors: Optional[str] = None, rederive_doors: bool = False) -> pd.DataFrame:
        """Name every tracklet. ``doors``: the door-position file (default ``doors_path``: ``doors.json`` in the
        ``out`` folder), read when it exists, else derived from the data and written there."""
        tl = self.tracklets()
        if len(self.duplicates_):
            log(f"{len(self.duplicates_)} duplicate mask(s) left out (a second mask on an animal already detected): "
                + ", ".join(f"{c} {n}" for c, n in self.duplicates_.camera.value_counts().items()))
        if self.exits and self.floor:
            self.use_doors(doors, rederive_doors, log)
        for i, t in enumerate(tl):
            t["id"] = i
        tl, n_cuts = self.split_swaps(tl, self.anchors(tl))

        def prepare(tl):
            tl, fixed = self.apply_corrections(tl)
            anchor = np.zeros((len(tl), len(self.animals)))
            for i, t in enumerate(tl):
                t["id"] = i
                if "cost" not in t:
                    t["cost"] = self.uwb_cost(t)
            for i, f, a in self.anchors(tl):
                anchor[i, a] += self.p.anchor_every
            return tl, fixed, anchor

        tl, fixed, anchor = prepare(tl)
        log(f"{len(tl)} tracklets ({n_cuts} cut where their anchors changed animal); anchors on "
            f"{int((anchor.sum(1) > 0).sum())}")
        links = self.links(tl)
        lab, margin, Cm = self.name(tl, anchor, links, fixed)
        # A tracklet a hard rule left unnamed loses all its frames to the clash, often a long one to a short
        # overlap: cut it where the clash starts and ends and solve again, so only that stretch stays in doubt.
        self.conflict_cuts_ = 0
        for _ in range(self.p.conflict_rounds):
            cuts = self.conflict_cuts(tl, lab, anchor, Cm, links)
            if not cuts:
                break
            self.conflict_cuts_ += sum(len(c) for c in cuts.values())
            tl, fixed, anchor = prepare(self.cut(tl, cuts))
            links = self.links(tl)
            lab, margin, Cm = self.name(tl, anchor, links, fixed)
        if self.conflict_cuts_:
            log(f"{self.conflict_cuts_} cut(s) where a hard conflict had left a tracklet unnamed; "
                f"{len(tl)} tracklets")
        self.broken_ = []                                # corrections the hard rules could not meet
        for i, a in fixed.items():
            if int(lab[i]) != (len(self.animals) if a is None else a):
                self.broken_.append((tl[i]["cam"], tl[i]["obj"], tl[i]["f0"], tl[i]["f1"]))
        if self.broken_:
            log(f"{len(self.broken_)} correction stretch(es) could not be met (e.g. the same animal twice at once): "
                + "; ".join(f"{c} #{o} {f0}-{f1}" for c, o, f0, f1 in self.broken_[:5]))
        rows = []
        for t in tl:
            a = int(lab[t["id"]])
            name = self.animals[a] if a < len(self.animals) else ""
            is_fixed = t["id"] in fixed
            rows.append(dict(camera=t["cam"], object_id=t["obj"], frame_first=t["f0"], frame_last=t["f1"],
                             detections=len(t["frames"]), animal=name, fixed=is_fixed,
                             unsure=bool(name and not is_fixed and margin[t["id"]] < self.p.sure),
                             margin=round(float(margin[t["id"]]), 1) if name else "",
                             anchors=" ".join(f"{self.animals[j]}:{int(v)}" for j, v in enumerate(anchor[t["id"]]) if v),
                             uwb_m=round(float(Cm[t["id"], a]), 3) if name else ""))
        self.links_ = pd.DataFrame(links, columns=["a", "b", "kind", "w", "cam_a", "cam_b", "frame_a", "frame_b"])
        self.links_["kept"] = [lab[lk["a"]] == lab[lk["b"]] for lk in links]
        out = pd.DataFrame(rows).sort_values(["camera", "frame_first"])
        named = out[out.animal != ""].detections.sum() / max(out.detections.sum(), 1)
        n_door = int(self.links_.kind.str.contains("door").sum()) if len(links) else 0
        log(f"{len(links)} candidate links ({n_door} doorway), {int(self.links_.kept.sum()) if len(links) else 0} "
            f"kept; named {named:.0%} of tracked detections")
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
    ap.add_argument("--derive-doors", action="store_true",
                    help="place the doors from the data again, replacing doors.json")
    a = ap.parse_args(argv)
    with open(a.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    out = cfg.get("out") or os.path.dirname(os.path.abspath(a.config))
    os.makedirs(out, exist_ok=True)
    idn = Identity(cfg, read_corrections(corrections_path(cfg, a.config)))
    res = idn.run(doors=doors_path(cfg, a.config), rederive_doors=a.derive_doors)
    res.to_csv(os.path.join(out, "identities.csv"), index=False)
    idn.links_.to_csv(os.path.join(out, "links.csv"), index=False)
    if len(idn.duplicates_):
        idn.duplicates_.to_csv(os.path.join(out, "duplicates.csv"), index=False)
    if len(getattr(idn, "door_links_", ())):
        idn.door_links_.to_csv(os.path.join(out, "door_links.csv"), index=False)
    print("wrote", os.path.join(out, "identities.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
