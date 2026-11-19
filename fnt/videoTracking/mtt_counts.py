"""How many animals are in view, through huddles (Mask Tracker).

A mask model tells animals apart when they are apart. In a huddle it cannot:
animals lie on and under each other, and how many masks the model draws there
says little about how many animals there are. Counting does not need it to.
Animals do not appear or vanish in the middle of the floor: they come in and go
out through the camera's exits (drawn in the Exits window, ``mask_tracker_cli
exits``; saved in ``exits.json`` beside the videos), and an animal lost
anywhere else is hiding, not gone. So the count in a huddle is the count of the
animals that went into it minus those that came out, whenever they did that,
even after the huddle was there first (the animals leaving it show how many it
held).

Steps, on a tracked video's saved masks (``<video>_tracks.h5``; Save masks on)
or, without them, on ``trajectories.csv``'s boxes:

1. groups: on each analysed frame, masks that touch are one group (a lone
   animal is a group of one).
2. segments: a group followed from frame to frame while nothing joins or
   leaves it. A segment ends where groups meet or split, or where it is lost.
   Short losses (missed frames) are bridged: a segment ending and another
   starting close by soon after may be the same animals.
3. counts: one integer count per segment, solved for the whole video at once
   (a small integer program, scipy's HiGHS): animals are conserved where
   segments meet and split; they arrive and leave cheaply at the exits and at
   the video's start and end; a segment starting or ending anywhere else is
   animals coming out of or going into hiding (still there, out of view: the
   hidden count, which can never go below zero); a lone animal-sized mask is
   at least one animal, a group at least as many as its area holds; never more
   than the trial's animals. Without exits drawn, every start and end is an
   arrival or a departure at a moderate cost, and nothing hides.

Writes, next to the trajectories (``<video>_MaskTracker/``):

    counts.csv      per analysed frame: animals in view, hiding, in the
                    camera's space (both), groups, masks
    segments.csv    per segment: frames, place, masks, area, solved count
    count_events.csv  animals arriving and leaving at the exits, and going into
                    and coming out of hiding (where, how many)

Run: ``python -m fnt.videoTracking.mtt_counts <videos or their _MaskTracker
folders> [--animals N]``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class CountSettings:
    min_score: float = 0.4        # masks below this are left out
    scale: int = 4                # groups are found on a grid of 1/scale of the frame
    touch_px: float = 8.0         # masks this close (full-frame px) are one group
    gap_s: float = 2.0            # a segment lost and another found within this time ...
    gap_px: float = 200.0         # ... and distance may be the same animals
    single_area: float = 0.0      # one animal's mask area (px); 0: from lone confident masks
    animals: int = 0              # animals in the trial (0: unknown, no cap)
    # costs (per animal) of explaining a segment's start or end
    cost_exit: float = 0.3        # arriving or leaving at an exit
    cost_hide: float = 2.0        # going into or coming out of hiding (anywhere else)
    cost_hidden_at_start: float = 1.0   # each animal already hiding when the video starts
    cost_unknown: float = 1.0     # no exits drawn: every start and end costs this
    near_px: float = 60.0         # a segment's end this close to an exit is at it
    # how much a segment's look counts: per second of it (at most max_weight_s) for what it
    # shows, once per segment for animals beyond it (underneath others: conservation decides)
    weight_single: float = 3.0    # a lone animal-sized mask: at least one animal (not a false mask)
    weight_floor: float = 3.0     # a group: at least what its area holds
    weight_single_over: float = 1.0   # per animal beyond one under a lone mask (one on another)
    weight_over: float = 0.5      # per animal beyond a group's masks and area (under others)
    max_weight_s: float = 20.0
    # a mask at least this share of one animal, not lying over a bigger mask, is a whole
    # animal: animals side by side touch (one group, little more area than one curled-up
    # animal) but each keeps its own mask; the pieces of one animal split apart are smaller
    whole_frac: float = 0.6


# ------------------------------------------------------------------ inputs
@dataclass
class Detections:
    """A tracked video's detections, by analysed frame."""
    fps: float
    width: int
    height: int
    frames: np.ndarray                      # analysed frames with detections, ascending
    rows: Dict[int, List[int]]              # frame -> row indices
    score: np.ndarray
    area: np.ndarray
    bbox: np.ndarray                        # [N, 4] x1 y1 x2 y2
    centroid: np.ndarray                    # [N, 2]
    mask_cells: Optional[callable] = None   # row -> (ys, xs) on the grid, or None (boxes only)
    step: int = 1
    first_frame: int = 0
    last_frame: int = 0
    source: str = ""


def _rle_cells(counts: np.ndarray, h: int, w: int, f: int) -> Tuple[np.ndarray, np.ndarray]:
    """Grid cells (1/f scale) covered by an uncompressed column-major RLE."""
    c = np.asarray(counts, np.int64)
    ends = np.cumsum(c)
    starts = ends - c
    on_s, on_len = starts[1::2], c[1::2]                # runs alternate 0s, 1s
    keep = on_len > 0
    on_s, on_len = on_s[keep], on_len[keep]
    if not on_s.size:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    x = on_s // h
    y0 = on_s % h
    y1 = np.minimum(y0 + on_len - 1, h - 1)             # a run wrapping a column is cut there
    gy0, gy1 = y0 // f, y1 // f
    n = gy1 - gy0 + 1
    gx = np.repeat(x // f, n)
    gy = np.repeat(gy0, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    return gy, gx


def load_detections(out_dir: str) -> Detections:
    """A ``<video>_MaskTracker`` folder's detections: its mask store if there
    is one, else trajectories.csv (boxes only)."""
    h5 = [f for f in os.listdir(out_dir) if f.endswith("_tracks.h5")]
    settings = {}
    rs = os.path.join(out_dir, "run_settings.json")
    if os.path.exists(rs):
        with open(rs, encoding="utf-8") as fh:
            settings = json.load(fh)
    step = int((settings.get("settings") or {}).get("frame_step", 1) or 1)
    if h5:
        from .track_store import open_h5
        path = os.path.join(out_dir, h5[0])
        f = open_h5(path, "r")
        a = f.attrs
        fps, w, h = float(a.get("fps", 30.0)), int(a["width"]), int(a["height"])
        frame = f["frame"][:]
        rle_start, rle_len = f["rle_start"][:], f["rle_len"][:]
        counts_ds = f["rle_counts"]
        all_counts = counts_ds[:] if counts_ds.size < 400_000_000 else None

        def cells(row: int, scale: int):
            n = int(rle_len[row])
            if n == 0:
                return None
            s = int(rle_start[row])
            c = all_counts[s:s + n] if all_counts is not None else counts_ds[s:s + n]
            return _rle_cells(c, h, w, scale)
        det = Detections(fps=fps, width=w, height=h, frames=np.array([]), rows={},
                         score=f["score"][:].astype(float), area=f["area"][:].astype(float),
                         bbox=f["bbox"][:].astype(float), centroid=f["centroid"][:].astype(float),
                         mask_cells=cells, source=path)
        n_frames = int(a.get("frame_count", 0))
    else:
        tj = pd.read_csv(os.path.join(out_dir, "trajectories.csv"))
        frame = tj["frame"].to_numpy()
        fps = float(settings.get("fps", 30.0))
        w, h = int(settings.get("width", 1920)), int(settings.get("height", 1080))
        det = Detections(fps=fps, width=w, height=h, frames=np.array([]), rows={},
                         score=tj["confidence"].to_numpy(float), area=tj["mask_area"].to_numpy(float),
                         bbox=tj[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float),
                         centroid=tj[["x", "y"]].to_numpy(float), source=os.path.join(out_dir, "trajectories.csv"))
        n_frames = int(settings.get("frames", frame.max() + 1 if frame.size else 0))
    order = np.argsort(frame, kind="stable")
    fr_sorted = frame[order]
    bounds = np.flatnonzero(np.diff(fr_sorted)) + 1
    for chunk in np.split(order, bounds):
        if chunk.size:
            det.rows[int(frame[chunk[0]])] = chunk.tolist()
    det.frames = np.array(sorted(det.rows), dtype=np.int64)
    det.step = step
    det.first_frame = 0
    det.last_frame = max(n_frames - 1, int(det.frames[-1]) if det.frames.size else 0)
    return det


# ------------------------------------------------------------------- exits
@dataclass
class Exits:
    """A camera's exits (mtt_exits) as the distance (px) to the nearest one."""
    dist: Optional[np.ndarray] = None

    @classmethod
    def for_video(cls, video: str, w: int, h: int) -> "Exits":
        """The exits drawn for ``video``'s camera in its folder's exits.json."""
        from .frame_activity import camera_key
        from .mtt_exits import load_exits
        try:
            entry = load_exits(os.path.dirname(os.path.abspath(video))).get(camera_key(video)[1])
        except (OSError, ValueError):
            entry = None
        if not entry or not entry.get("exits"):
            return cls()
        return cls.from_polygons(entry["exits"], tuple(entry.get("frame_size") or (w, h)), w, h)

    @classmethod
    def from_polygons(cls, polys, frame_size, w: int, h: int) -> "Exits":
        sx, sy = w / frame_size[0], h / frame_size[1]
        m = np.full((h, w), 255, np.uint8)
        for p in polys:
            pts = np.round(np.asarray(p, float) * [sx, sy]).astype(np.int32)
            cv2.fillPoly(m, [pts], 0)
        return cls(dist=cv2.distanceTransform(m, cv2.DIST_L2, 3))

    @property
    def drawn(self) -> bool:
        return self.dist is not None

    def kind_at(self, x: float, y: float, near: float) -> str:
        """'exit' at (or within ``near`` px of) an exit, else 'hiding'."""
        yi = int(min(max(y, 0), self.dist.shape[0] - 1))
        xi = int(min(max(x, 0), self.dist.shape[1] - 1))
        return "exit" if self.dist[yi, xi] <= near else "hiding"


# ------------------------------------------------------------- 1-2 segments
@dataclass
class Segment:
    id: int
    f0: int
    f1: int
    n_masks: List[int] = field(default_factory=list)
    area: List[float] = field(default_factory=list)
    own: List[Tuple[float, ...]] = field(default_factory=list)   # each mask's area not under a bigger one
    xy0: Tuple[float, float] = (0.0, 0.0)
    xy1: Tuple[float, float] = (0.0, 0.0)
    preds: List[int] = field(default_factory=list)
    succs: List[int] = field(default_factory=list)
    bridged_from: List[int] = field(default_factory=list)   # gap links (missed frames)


def _frame_groups(det: Detections, rows: List[int], cfg: CountSettings):
    """Groups on one frame: (label grid, [(rows, area px, (cx, cy), own areas px)])."""
    f = cfg.scale
    gh, gw = (det.height + f - 1) // f, (det.width + f - 1) // f
    occ = np.zeros((gh, gw), np.uint8)
    cells = []
    for r in rows:
        c = det.mask_cells(r, f) if det.mask_cells is not None else None
        if c is None:                                   # box only
            x1, y1, x2, y2 = (det.bbox[r] / f).astype(int)
            yy, xx = np.mgrid[max(y1, 0):min(y2 + 1, gh), max(x1, 0):min(x2 + 1, gw)]
            c = (yy.ravel(), xx.ravel())
        cells.append(c)
        occ[c] = 1
    k = max(1, int(round(cfg.touch_px / f)))
    grown = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    n, lab = cv2.connectedComponents(grown, connectivity=8)
    lab = lab * occ                                     # groups on the animals' own cells
    # each mask's own area: its cells not already under a bigger mask (biggest first),
    # so a second mask drawn over the same animal adds little
    taken = np.zeros_like(occ)
    own = {}
    for i in sorted(range(len(rows)), key=lambda i: -cells[i][0].size):
        c = cells[i]
        own[i] = float((taken[c] == 0).sum() * f * f)
        taken[c] = 1
    groups: Dict[int, List[int]] = {}
    for i, (r, c) in enumerate(zip(rows, cells)):
        if c[0].size == 0:
            continue
        g = int(np.bincount(lab[c][lab[c] > 0]).argmax()) if (lab[c] > 0).any() else 0
        if g:
            groups.setdefault(g, []).append(i)
    out = []
    for g, ii in sorted(groups.items()):
        ys, xs = np.nonzero(lab == g)
        out.append((g, [rows[i] for i in ii], float(ys.size * f * f),
                    (float(xs.mean() * f), float(ys.mean() * f)), tuple(own[i] for i in ii)))
    return lab, out


def build_segments(det: Detections, cfg: CountSettings, log=print) -> List[Segment]:
    """Groups followed from frame to frame (steps 1-2)."""
    segs: List[Segment] = []
    prev_lab, prev_ids, prev_frame = None, {}, None
    t0 = time.time()
    gap_frames = det.step * 1.5
    for k, fr in enumerate(det.frames):
        rows = [r for r in det.rows[int(fr)] if det.score[r] >= cfg.min_score]
        if not rows:
            continue
        lab, groups = _frame_groups(det, rows, cfg)
        cur_ids = {}
        links: Dict[int, List[int]] = {}           # current group -> previous segments
        back: Dict[int, List[int]] = {}            # previous segment -> current groups
        if prev_lab is not None and fr - prev_frame <= gap_frames:
            both = (prev_lab > 0) & (lab > 0)
            if both.any():
                pairs = np.unique(prev_lab[both].astype(np.int64) * 65536 + lab[both])
                for p in pairs:
                    a, b = int(p // 65536), int(p % 65536)
                    if a in prev_ids:
                        links.setdefault(b, []).append(prev_ids[a])
                        back.setdefault(prev_ids[a], []).append(b)
            # no overlap at all (a quick move): the nearest lone pair within reach
            lone_prev = [s for s in prev_ids.values() if s not in back]
            lone_cur = [g for g, *_ in groups if g not in links]
            for g in lone_cur:
                cx, cy = next(c for gg, _, _, c, _ in groups if gg == g)
                best = None
                for s in lone_prev:
                    sx, sy = segs[s].xy1
                    d = np.hypot(cx - sx, cy - sy)
                    if d < cfg.gap_px / 2 and (best is None or d < best[0]):
                        best = (d, s)
                if best is not None:
                    links[g] = [best[1]]
                    back[best[1]] = [g]
                    lone_prev.remove(best[1])
        for g, rs, area, c, own in groups:
            prev = links.get(g, [])
            if len(prev) == 1 and len(back.get(prev[0], [])) == 1:      # carries on
                s = segs[prev[0]]
            else:                                                     # new segment
                s = Segment(id=len(segs), f0=int(fr), f1=int(fr), xy0=c)
                for p in prev:
                    s.preds.append(p)
                    segs[p].succs.append(s.id)
                segs.append(s)
            s.f1 = int(fr)
            s.xy1 = c
            s.n_masks.append(len(rs))
            s.area.append(area)
            s.own.append(own)
            cur_ids[g] = s.id
        prev_lab, prev_ids, prev_frame = lab, cur_ids, int(fr)
        if log and k and k % 20000 == 0:
            log(f"[counts] {k}/{len(det.frames)} frames, {len(segs)} segments ({time.time() - t0:.0f} s)")
    # bridge short losses: an end with nothing after it, a start with nothing
    # before it, close in time and place (a few missed frames)
    gap = cfg.gap_s * det.fps
    ends = sorted((s for s in segs if not s.succs), key=lambda s: s.f1)
    starts = sorted((s for s in segs if not s.preds), key=lambda s: s.f0)
    taken = set()
    for e in ends:
        best = None
        for s in starts:
            if s.f0 <= e.f1:
                continue
            if s.f0 - e.f1 > gap:
                break
            if s.id in taken:
                continue
            d = np.hypot(s.xy0[0] - e.xy1[0], s.xy0[1] - e.xy1[1])
            if d <= cfg.gap_px and (best is None or d < best[0]):
                best = (d, s)
        if best is not None:
            s = best[1]
            taken.add(s.id)
            e.succs.append(s.id)
            s.preds.append(e.id)
            s.bridged_from.append(e.id)
    if log:
        log(f"[counts] {len(det.frames)} frames -> {len(segs)} segments, {len(taken)} gaps bridged "
            f"({time.time() - t0:.0f} s)")
    return segs


# ---------------------------------------------------------------- 3 counts
def single_area(det: Detections, segs: List[Segment]) -> float:
    """One animal's mask area: the median of confident lone masks."""
    a = [ar for s in segs for n, ar in zip(s.n_masks, s.area) if n == 1]
    return float(np.median(a)) if a else 1.0


def solve_counts(det: Detections, segs: List[Segment], exits: Exits, cfg: CountSettings,
                 log=print) -> Dict:
    """One integer count per segment (step 3). Returns the solution and costs."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    one = cfg.single_area or single_area(det, segs)
    fps = det.fps
    n_s = len(segs)
    # variables: n (segments) | x (links) | b (births) | d (deaths) | u, v (look deviations)
    links = [(p, s.id) for s in segs for p in s.preds]
    n_l = len(links)
    iN, iX, iB, iD = 0, n_s, n_s + n_l, 2 * n_s + n_l
    iU, iV = 3 * n_s + n_l, 4 * n_s + n_l
    # hiding: animals that went out of sight away from an exit, at each moment
    # some went in or came out (P[0]: hiding when the video starts)
    hide_t = sorted({s.f1 + 0.5 for s in segs} | {float(s.f0) for s in segs})
    iP = 5 * n_s + n_l
    n_var = iP + len(hide_t) + 1
    c = np.zeros(n_var)
    lo, hi = np.zeros(n_var), np.full(n_var, np.inf)
    cap = cfg.animals if cfg.animals > 0 else 50
    hi[iN:iN + n_s] = cap

    start_kind, end_kind, look = [], [], []
    for s in segs:
        dur = min((s.f1 - s.f0) / fps + det.step / fps, cfg.max_weight_s)
        nm = float(np.median(s.n_masks))
        ratio = float(np.median(s.area)) / one
        lone = nm <= 1 and 0.5 <= ratio <= 1.5
        # whole animals seen apart: masks big enough to be one animal each
        whole = int(np.median([sum(a >= cfg.whole_frac * one for a in o) for o in s.own]))
        # a group holds at least the whole animals it shows and what its area shows
        # (animals overlap, so a little less than whole multiples), maybe more (some
        # underneath)
        lower = 1 if lone else max(1, whole, int(np.floor(ratio + 0.3)))
        upper = 1 if lone else max(lower, int(np.ceil(nm)), int(np.ceil(ratio)))
        look.append((lone, lower, upper, dur, nm, ratio, whole))

        def kind(f, xy, video_edge):
            if video_edge:
                return "video_edge"
            if not exits.drawn:
                return "unknown"
            return exits.kind_at(xy[0], xy[1], cfg.near_px)
        # the video's own first and last frames (not the first and last detections)
        start_kind.append(kind(s.f0, s.xy0, s.f0 <= det.first_frame + 2 * det.step))
        end_kind.append(kind(s.f1, s.xy1, s.f1 >= det.last_frame - 2 * det.step))

    cost_of = {"video_edge": 0.05, "exit": cfg.cost_exit, "hiding": cfg.cost_hide,
               "unknown": cfg.cost_unknown}
    for i, s in enumerate(segs):
        c[iB + i] = cost_of[start_kind[i]]
        c[iD + i] = cost_of[end_kind[i]]
        lone, lower, upper, dur, nm, ratio, whole = look[i]
        if lone:
            c[iU + i] = cfg.weight_single * dur      # n below 1 (a false detection)
            c[iV + i] = cfg.weight_single_over       # n above 1 (one on top of another)
        else:
            c[iU + i] = cfg.weight_floor * dur       # n below what the area shows
            c[iV + i] = cfg.weight_over              # n above masks and area (underneath)

    c[iP] = cfg.cost_hidden_at_start
    A = lil_matrix((4 * n_s + len(hide_t), n_var))
    lb, ub = [], []
    in_links: Dict[int, List[int]] = {}
    out_links: Dict[int, List[int]] = {}
    for j, (p, q) in enumerate(links):
        out_links.setdefault(p, []).append(j)
        in_links.setdefault(q, []).append(j)
    r = 0
    for i in range(n_s):
        lone, lower, upper, *_ = look[i]
        # n_i = sum(in links) + b_i
        A[r, iN + i] = 1
        for j in in_links.get(i, []):
            A[r, iX + j] = -1
        A[r, iB + i] = -1
        lb.append(0); ub.append(0); r += 1
        # n_i = sum(out links) + d_i
        A[r, iN + i] = 1
        for j in out_links.get(i, []):
            A[r, iX + j] = -1
        A[r, iD + i] = -1
        lb.append(0); ub.append(0); r += 1
        # u_i >= lower - n_i ; v_i >= n_i - upper
        A[r, iU + i] = 1
        A[r, iN + i] = 1
        lb.append(lower); ub.append(np.inf); r += 1
        A[r, iV + i] = 1
        A[r, iN + i] = -1
        lb.append(-upper); ub.append(np.inf); r += 1
    # P[k] = P[k-1] + those going into hiding at hide_t[k] - those coming out
    k_of = {t: k + 1 for k, t in enumerate(hide_t)}
    went = {}
    for i, s in enumerate(segs):
        if end_kind[i] == "hiding":
            went.setdefault(k_of[s.f1 + 0.5], []).append((iD + i, -1))
        if start_kind[i] == "hiding":
            went.setdefault(k_of[float(s.f0)], []).append((iB + i, 1))
    for k in range(1, len(hide_t) + 1):
        A[r, iP + k] = 1
        A[r, iP + k - 1] = -1
        for var, sign in went.get(k, []):
            A[r, var] = sign
        lb.append(0); ub.append(0); r += 1
    cons = [LinearConstraint(A.tocsr(), np.array(lb, float), np.array(ub, float))]

    # never more than the trial's animals at once (checked where segments start)
    if cfg.animals > 0:
        by_start = sorted(range(n_s), key=lambda i: segs[i].f0)
        starts = sorted({segs[i].f0 for i in by_start})
        C = lil_matrix((len(starts), n_var))
        active: List[int] = []
        j = 0
        for k, t in enumerate(starts):
            while j < n_s and segs[by_start[j]].f0 <= t:
                active.append(by_start[j])
                j += 1
            active = [i for i in active if segs[i].f1 >= t]
            for i in active:
                C[k, iN + i] = 1
            C[k, iP + k_of[float(t)]] = 1                     # and those hiding
        cons.append(LinearConstraint(C.tocsr(), -np.inf, cfg.animals))

    t0 = time.time()
    res = milp(c, constraints=cons, integrality=np.ones(n_var), bounds=Bounds(lo, hi),
               options={"time_limit": 600, "mip_rel_gap": 0.01})
    if log:
        log(f"[counts] solve: {res.message} ({n_s} segments, {n_l} links, {time.time() - t0:.0f} s)")
    if res.x is None:
        raise RuntimeError(f"count solve failed: {res.message}")
    x = np.round(res.x).astype(int)
    return dict(n=x[iN:iN + n_s], x=x[iX:iX + n_l], b=x[iB:iB + n_s], d=x[iD:iD + n_s],
                hiding=x[iP:], hide_t=hide_t,
                links=links, start_kind=start_kind, end_kind=end_kind, look=look, single_area=one,
                objective=float(res.fun))


# ----------------------------------------------------------------- outputs
def write_outputs(out_dir: str, det: Detections, segs: List[Segment], sol: Dict,
                  cfg: CountSettings) -> Dict[str, str]:
    fps = det.fps
    n = sol["n"]
    seg_rows = []
    for i, s in enumerate(segs):
        lone, lower, upper, dur, nm, ratio, whole = sol["look"][i]
        seg_rows.append(dict(segment=s.id, frame_start=s.f0, frame_end=s.f1,
                             time_start_s=round(s.f0 / fps, 2), time_end_s=round(s.f1 / fps, 2),
                             x_start=round(s.xy0[0]), y_start=round(s.xy0[1]),
                             x_end=round(s.xy1[0]), y_end=round(s.xy1[1]),
                             masks_median=nm, masks_max=int(max(s.n_masks)), area_animals=round(ratio, 2),
                             whole_masks=whole, at_least=lower,
                             animals=int(n[i]), arrived=int(sol["b"][i]), left=int(sol["d"][i]),
                             start_at=sol["start_kind"][i], end_at=sol["end_kind"][i],
                             from_segments=" ".join(map(str, s.preds)),
                             to_segments=" ".join(map(str, s.succs))))
    sdf = pd.DataFrame(seg_rows)
    # per analysed frame
    nf = det.last_frame + 1
    vis = np.zeros(nf, np.int32)
    grp = np.zeros(nf, np.int32)
    msk = np.zeros(nf, np.int32)
    for i, s in enumerate(segs):
        vis[s.f0:s.f1 + 1] += n[i]
        grp[s.f0:s.f1 + 1] += 1
    for fr, rows in det.rows.items():
        msk[fr] = sum(det.score[r] >= cfg.min_score for r in rows)
    # hiding: the solved count after each moment some went into or came out of hiding
    hid = np.full(nf, int(sol["hiding"][0]), np.int32)
    for k, t in enumerate(sol["hide_t"]):
        hid[int(np.ceil(t)):] = int(sol["hiding"][k + 1])
    frames = np.arange(0, nf, det.step)
    cdf = pd.DataFrame(dict(frame=frames, time_s=np.round(frames / fps, 3), in_view=vis[frames],
                            hiding=hid[frames], in_space=vis[frames] + hid[frames],
                            groups=grp[frames], masks=msk[frames]))
    ev = []
    for i, s in enumerate(segs):
        for what, k, fr, xy, kind in (("arrive", sol["b"][i], s.f0, s.xy0, sol["start_kind"][i]),
                                      ("leave", sol["d"][i], s.f1, s.xy1, sol["end_kind"][i])):
            if k and kind != "video_edge":
                if kind == "hiding":
                    what = "out of hiding" if what == "arrive" else "into hiding"
                ev.append(dict(frame=fr, time_s=round(fr / fps, 2), event=what, animals=int(k), at=kind,
                               x=round(xy[0]), y=round(xy[1]), segment=s.id))
    edf = pd.DataFrame(ev, columns=["frame", "time_s", "event", "animals", "at", "x", "y",
                                    "segment"]).sort_values("frame")
    paths = {k: os.path.join(out_dir, f) for k, f in
             (("counts", "counts.csv"), ("segments", "segments.csv"), ("events", "count_events.csv"))}
    cdf.to_csv(paths["counts"], index=False)
    sdf.to_csv(paths["segments"], index=False)
    edf.to_csv(paths["events"], index=False)
    with open(os.path.join(out_dir, "counts_settings.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(asdict(cfg), single_area_used=sol["single_area"], source=det.source,
                       segments=len(segs), objective=sol["objective"]), fh, indent=2)
    return paths


def count_video(target: str, cfg: Optional[CountSettings] = None, log=print,
                exits: Optional[Exits] = None) -> Dict[str, str]:
    """Count the animals of one tracked video (its path or its _MaskTracker
    folder). The exits are its camera's in exits.json beside it (or ``exits``)."""
    cfg = cfg or CountSettings()
    out_dir = target if target.rstrip("\\/").endswith("_MaskTracker") else \
        os.path.splitext(target)[0] + "_MaskTracker"
    video = out_dir.rstrip("\\/")[:-len("_MaskTracker")] + os.path.splitext(target)[1] \
        if not target.endswith("_MaskTracker") else ""
    det = load_detections(out_dir)
    if not video:
        rs = os.path.join(out_dir, "run_settings.json")
        if os.path.exists(rs):
            with open(rs, encoding="utf-8") as fh:
                video = json.load(fh).get("video", "")
    if exits is None:
        exits = Exits.for_video(video, det.width, det.height) if video else Exits()
    if log:
        log(f"[counts] {os.path.basename(out_dir)}: {len(det.frames)} analysed frames with detections, "
            f"step {det.step}, {'masks' if det.mask_cells else 'boxes'}, "
            f"exits {'drawn' if exits.drawn else 'not drawn (every start and end costs the same)'}")
    segs = build_segments(det, cfg, log=log)
    sol = solve_counts(det, segs, exits, cfg, log=log)
    return write_outputs(out_dir, det, segs, sol, cfg)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m fnt.videoTracking.mtt_counts", description=__doc__.split("\n\n")[0])
    p.add_argument("targets", nargs="+", help="Tracked videos or their _MaskTracker folders.")
    p.add_argument("--animals", type=int, default=0, help="Animals in the trial (a cap; 0: none).")
    p.add_argument("--min-score", type=float, default=CountSettings.min_score)
    a = p.parse_args(argv)
    cfg = CountSettings(animals=a.animals, min_score=a.min_score)
    for t in a.targets:
        count_video(t, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
