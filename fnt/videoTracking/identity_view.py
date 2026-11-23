"""The picture of a multi-camera tracking run that the Fusion View window shows and ``tracked_grid`` renders:
the clock-synced camera clips in a grid, each tracked animal's mask in its colour with its midline keypoints and its
name (from ``mtt_identity``), and optionally the UWB (or other position-tracking) data over it - where UWB puts each
animal on the overhead camera, who UWB puts in each box, the tails of both, and a floor map with the video and UWB
positions side by side - so the alignment the names rest on can be judged by eye.

A ``Scene`` is built from the ``identity.json`` that ``mtt_identity`` ran on; ``compose`` puts one frame of every
camera into the grid and ``draw`` draws the chosen ``Overlays`` on it. Frame n of every clip is the same moment
(``fnt-grid --clips``).
"""
from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

KP_ORDER = ["nose", "neck", "mid", "hip", "tailbase", "tailtip"]
PALETTE = [(60, 76, 231), (34, 156, 255), (182, 89, 155), (219, 152, 52), (113, 204, 46), (15, 196, 241),
           (160, 160, 255), (200, 120, 60), (90, 220, 220), (240, 240, 120)]          # BGR
GREY = (170, 170, 170)


@dataclass
class Overlays:
    masks: bool = True
    keypoints: bool = True
    names: bool = True
    legend: bool = True
    track_tails: bool = False      # each video track's recent path
    uwb: bool = False              # UWB positions on the overhead camera, who UWB puts in each box
    uwb_tails: bool = False        # each animal's recent UWB path
    floor: bool = False            # a floor map (metres) with the video and UWB positions, in a free cell
    tail_s: float = 3.0


def video_of(track_dir: str) -> str:
    """The clip a ``<video>_MaskTracker`` folder was made from (beside it)."""
    stem = os.path.basename(track_dir.rstrip("\\/"))[:-len("_MaskTracker")]
    cands = [p for p in glob.glob(os.path.join(os.path.dirname(track_dir.rstrip("\\/")), stem + ".*"))
             if not p.endswith((".csv", ".json"))]
    if not cands:
        raise FileNotFoundError(f"no video beside {track_dir}")
    return cands[0]


def load_config(cfg_path: str):
    """identity.json -> (cfg, out folder, {camera: {dir, clip, store, midline, trajectories}})."""
    with open(cfg_path, encoding="utf-8-sig") as fh:
        cfg = json.load(fh)
    out_dir = cfg.get("out") or os.path.dirname(os.path.abspath(cfg_path))
    cams = {}
    for c, v in cfg["cameras"].items():
        d = v["tracks"] if not v["tracks"].endswith(".csv") else os.path.dirname(v["tracks"])
        store = glob.glob(os.path.join(d, "*_tracks.h5"))
        mid = glob.glob(os.path.join(d, "*_midline.csv"))
        cams[c] = dict(dir=d, clip=video_of(d), store=store[0] if store else None, midline=mid[0] if mid else None,
                       trajectories=v["tracks"] if v["tracks"].endswith(".csv") else os.path.join(d, "trajectories.csv"))
    return cfg, out_dir, cams


def frame_step_of(cfg: Dict, cams: Dict) -> int:
    """The frame step the tracks were made at: identity.json's ``frame_step``, else Mask Tracker's run settings."""
    if cfg.get("frame_step"):
        return int(cfg["frame_step"])
    for v in cams.values():
        p = os.path.join(v["dir"], "run_settings.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return int(json.load(fh).get("settings", {}).get("frame_step", 1) or 1)
    return 1


class _Names:
    """(camera, object_id, frame) -> (animal, unsure, fixed), from identities.csv rows."""

    def __init__(self, ids: pd.DataFrame):
        self.by: Dict[tuple, tuple] = {}
        fixed = ids["fixed"].astype(bool) if "fixed" in ids else pd.Series(False, index=ids.index)
        for (cam, obj), g in ids.assign(fixed=fixed).sort_values("frame_first").groupby(["camera", "object_id"]):
            self.by[(cam, int(obj))] = (g.frame_first.to_numpy(int), g.frame_last.to_numpy(int),
                                        [a if isinstance(a, str) else "" for a in g.animal],
                                        g.unsure.astype(bool).to_numpy(), g.fixed.to_numpy(bool))

    def get(self, cam, obj, frame):
        r = self.by.get((cam, int(obj)))
        if r is None:
            return "", False, False
        i = np.searchsorted(r[0], frame, "right") - 1
        if i >= 0 and frame <= r[1][i]:
            return r[2][i], bool(r[3][i]), bool(r[4][i])
        return "", False, False

    def row(self, cam, obj, frame):
        """(first, last) frame of the tracklet holding this detection, or None."""
        r = self.by.get((cam, int(obj)))
        if r is None:
            return None
        i = np.searchsorted(r[0], frame, "right") - 1
        return (int(r[0][i]), int(r[1][i])) if i >= 0 and frame <= r[1][i] else None


class Scene:
    def __init__(self, cfg_path: str, size: str = "1080", hold: Optional[int] = None,
                 identities: Optional[pd.DataFrame] = None, with_uwb: bool = True):
        from fnt.videoProcessing.camera_grid_cli import MESOCOSM_LAYOUT, parse_layout
        from .track_store import TrackMaskReader, long_path
        self.cfg_path = cfg_path
        self.cfg, self.out_dir, self.cams = load_config(cfg_path)
        cfg = self.cfg
        self.fps = float(cfg.get("fps", 30))
        self.n_frames = int(cfg["frames"])
        self.start = datetime.fromisoformat(cfg["start"])
        self.title = cfg.get("title", "")
        self.hold = int(hold or frame_step_of(cfg, self.cams))
        self.layout = parse_layout(cfg.get("layout", MESOCOSM_LAYOUT), cfg.get("centre_scale", 1.2))
        self.set_size(size)
        self.readers = {c: TrackMaskReader(v["store"]) for c, v in self.cams.items() if v["store"]}
        self._blocks: "OrderedDict[tuple, np.ndarray]" = OrderedDict()   # (cam, k) -> a block of RLE runs
        self._dets: "OrderedDict[tuple, list]" = OrderedDict()            # (cam, frame) -> detections
        self._rle_ds: Dict[str, object] = {}
        self.size_of = {c: ((r.width, r.height) if r.width else (1920, 1080)) for c, r in self.readers.items()}
        # keypoints, as arrays sorted by frame
        self.kp = {}
        for c, v in self.cams.items():
            if v["midline"]:
                m = pd.read_csv(long_path(v["midline"])).sort_values("frame")
                pts = np.stack([m[[f"{q}_x", f"{q}_y"]].to_numpy(float) for q in KP_ORDER if f"{q}_x" in m], 1)
                self.kp[c] = (m.frame.to_numpy(int), m.object_id.to_numpy(int), pts)
        # the identity model gives the UWB per frame, the floor map and the boxes
        from .mtt_identity import Identity
        self.idn = Identity(cfg) if with_uwb and cfg.get("uwb") else None
        self.floor = self.idn.floor if self.idn else {}
        # trajectories (for tails and the floor map), per camera and object
        self.tracks: Dict[str, Dict[int, tuple]] = {}
        for c, v in self.cams.items():
            if not os.path.exists(long_path(v["trajectories"])):
                continue
            tj = pd.read_csv(long_path(v["trajectories"]), usecols=["frame", "object_id", "x", "y", "confidence"])
            tj = tj[tj.confidence >= 0.35].sort_values("frame")
            per = {}
            for obj, g in tj.groupby("object_id"):
                xy = g[["x", "y"]].to_numpy(float)
                fl = self.floor[c].to_floor(xy) if c in self.floor else None
                per[int(obj)] = (g.frame.to_numpy(int), xy, fl)
            self.tracks[c] = per
        if identities is None:
            p = os.path.join(self.out_dir, "identities.csv")
            identities = pd.read_csv(p, keep_default_na=False) if os.path.exists(p) else pd.DataFrame(
                columns=["camera", "object_id", "frame_first", "frame_last", "animal", "unsure"])
        self.set_identities(identities)

    def set_size(self, size: str):
        """The canvas: "4k" (3840x2160) or "1080"; cells, legend and floor map laid out on it."""
        from fnt.videoProcessing.camera_grid_encode import EncodeSettings
        layout = self.layout
        self.size = "4k" if size == "4k" else "1080"
        self.W, self.H = (3840, 2160) if self.size == "4k" else (1920, 1080)
        self.s = self.W / 3840
        geo = EncodeSettings(width=self.W, height=self.H)
        self.cells = {cam: geo.cell_rect(layout, r, c) for (r, c), cam in layout.assignments.items() if cam in self.cams}
        free = [(r, c) for r in range(layout.rows) for c in range(layout.cols) if (r, c) not in layout.assignments]
        leg = free[len(free) // 2 - 1] if free else (0, 0)
        self.legend_cell = geo.cell_rect(layout, *leg)
        rest = [f for f in free if f != leg]
        same_row = [f for f in rest if f[0] == leg[0]]
        self.floor_cell = geo.cell_rect(layout, *(same_row or rest)[0]) if rest else None

    # ---------------------------------------------------------------- names
    def set_identities(self, ids: pd.DataFrame):
        self.ids = ids
        self.names_at = _Names(ids)
        found = sorted(a for a in ids.animal.unique() if isinstance(a, str) and a) if len(ids) else []
        self.animals = sorted(set(self.idn.animals) | set(found)) if self.idn else found
        self.names = {a: self.cfg.get("names", {}).get(a, a) for a in self.animals}
        self.colours = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(self.animals)}

    def colour(self, animal):
        return self.colours.get(animal, GREY)

    def clock(self, n) -> datetime:
        return self.start + timedelta(seconds=n / self.fps)

    def analysed(self, n) -> int:
        return n - n % self.hold

    # ---------------------------------------------------------------- geometry
    def to_canvas(self, cam, xy):
        cx, cy, cw, ch = self.cells[cam]
        w, h = self.size_of.get(cam, (1920, 1080))
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.column_stack([cx + xy[:, 0] * cw / w, cy + xy[:, 1] * ch / h])

    def cell_at(self, x, y) -> Optional[str]:
        for cam, (cx, cy, cw, ch) in self.cells.items():
            if cx <= x < cx + cw and cy <= y < cy + ch:
                return cam
        return None

    def uwb_at(self, n) -> Dict[str, np.ndarray]:
        """{animal: (x, y) m} for the animals UWB has in the arena at frame n."""
        if not self.idn or not 0 <= n < self.idn.n:
            return {}
        return {a: xy[n] for a, xy in self.idn.uxy.items() if self.idn.present[a][n]}

    def hit(self, n, x, y) -> Optional[Dict]:
        """The detection under canvas point (x, y) on frame n (its mask, else the nearest centre within reach)."""
        import cv2
        cam = self.cell_at(x, y)
        if cam is None or cam not in self.readers:
            return None
        cx, cy, cw, ch = self.cells[cam]
        w, h = self.size_of[cam]
        px, py = (x - cx) * w / cw, (y - cy) * h / ch
        nd = self.analysed(n)
        best, best_d = None, 60.0
        for det in self.detections(cam, nd):
            m, (ox, oy) = det["mask"], det["origin"]
            if m is not None and 0 <= int(py) - oy < m.shape[0] and 0 <= int(px) - ox < m.shape[1]                     and m[int(py) - oy, int(px) - ox]:
                best = det
                break
            d = float(np.hypot(det["centroid"][0] - px, det["centroid"][1] - py))
            if d < best_d:
                best, best_d = det, d
        if best is None:
            return None
        obj = best["object_id"]
        animal, unsure, fixed = self.names_at.get(cam, obj, nd)
        return dict(cam=cam, obj=obj, frame=nd, animal=animal, unsure=unsure, fixed=fixed,
                    tracklet=self.names_at.row(cam, obj, nd) or self.run_around(cam, obj, nd))

    def run_around(self, cam, obj, frame, gap=20) -> Optional[Tuple[int, int]]:
        """The stretch of an object's frames around ``frame`` without a gap longer than ``gap``."""
        t = self.tracks.get(cam, {}).get(int(obj))
        if t is None:
            return None
        fr = t[0]
        i = int(np.clip(np.searchsorted(fr, frame), 0, len(fr) - 1))
        breaks = np.flatnonzero(np.diff(fr) > gap)
        lo = breaks[breaks < i]
        hi = breaks[breaks >= i]
        return int(fr[lo[-1] + 1] if len(lo) else fr[0]), int(fr[hi[0]] if len(hi) else fr[-1])

    def uwb_distances(self, cam, obj, f0, f1) -> Dict[str, float]:
        """Mean UWB distance (m) of a stretch of a track to every animal (box distance on a zone camera)."""
        t = self.tracks.get(cam, {}).get(int(obj))
        if t is None or not self.idn:
            return {}
        sel = (t[0] >= f0) & (t[0] <= f1)
        if not sel.any():
            return {}
        tr = self.idn._make(cam, obj, t[0][sel], t[1][sel])
        return dict(zip(self.idn.animals, (float(v) for v in self.idn.uwb_cost(tr))))

    # ---------------------------------------------------------------- detections
    BLOCKS = 256            # blocks of RLE runs kept (one compressed chunk each, 256 kB)
    DETS = 256              # frames of decoded detections kept

    def _runs(self, cam, a, b) -> np.ndarray:
        """rle_counts[a:b] of a camera's store, read a whole compressed chunk at a time and kept: the store is
        gzip-chunked, so reading each detection's few runs on its own unpacks the same chunk over and over."""
        ds = self._rle_ds.get(cam)
        if ds is None:
            ds = self._rle_ds[cam] = self.readers[cam]._f["rle_counts"]
        B = int(ds.chunks[0]) if ds.chunks else 1 << 16
        parts = []
        for k in range(a // B, (b - 1) // B + 1):
            blk = self._blocks.get((cam, k))
            if blk is None:
                blk = ds[k * B:(k + 1) * B]
                self._blocks[(cam, k)] = blk
                if len(self._blocks) > self.BLOCKS:
                    self._blocks.popitem(last=False)
            else:
                self._blocks.move_to_end((cam, k))
            parts.append(blk[max(a - k * B, 0):min(b - k * B, B)])
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def detections(self, cam, frame) -> List[Dict]:
        """A frame's detections, each with its mask cut to its box (``mask``, at ``origin``): decoding only what
        is drawn keeps playback quick, and the last frames' are kept for redrawing."""
        key = (cam, int(frame))
        got = self._dets.get(key)
        if got is not None:
            self._dets.move_to_end(key)
            return got
        got = self._detections(cam, frame)
        self._dets[key] = got
        if len(self._dets) > self.DETS:
            self._dets.popitem(last=False)
        return got

    def _detections(self, cam, frame) -> List[Dict]:
        r = self.readers.get(cam)
        if r is None:
            return []
        out = []
        for row in r.rows_for_frame(frame):
            bb = r.bbox[row]
            x0, y0 = max(int(bb[0]) - 2, 0), max(int(bb[1]) - 2, 0)
            x1, y1 = min(int(bb[2]) + 3, r.width), min(int(bb[3]) + 3, r.height)
            mask = None
            n = int(r._rle_len[row])
            if n and x1 > x0 and y1 > y0:
                counts = self._runs(cam, int(r._rle_start[row]), int(r._rle_start[row]) + n).astype(np.int64)
                pos = np.concatenate([[0], np.cumsum(np.maximum(counts, 0))])
                lo, hi = x0 * r.height, x1 * r.height         # column-major RLE: only the box's columns
                st, en = pos[1:-1:2], pos[2::2]               # the runs of mask pixels
                keep = (en > lo) & (st < hi)
                buf = np.zeros(hi - lo, bool)
                for a, b in zip(np.maximum(st[keep], lo) - lo, np.minimum(en[keep], hi) - lo):
                    buf[a:b] = True
                mask = buf.reshape((x1 - x0, r.height)).T[y0:y1]
            out.append(dict(object_id=int(r.object_id[row]), score=float(r.score[row]), bbox=bb.tolist(),
                            centroid=r.centroid[row].tolist(), mask=mask, origin=(x0, y0)))
        return out

    # ---------------------------------------------------------------- drawing
    def compose(self, frames: Dict[str, np.ndarray]) -> np.ndarray:
        """The grid of one frame per camera (frames already at their cell's size are pasted as they are)."""
        import cv2
        img = np.zeros((self.H, self.W, 3), np.uint8)
        for c, fr in frames.items():
            if c in self.cells and fr is not None:
                x, y, w, h = self.cells[c]
                if fr.shape[0] != h or fr.shape[1] != w:
                    fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
                img[y:y + h, x:x + w] = fr
        return img

    def _label(self, img, text, x, y, colour, scale, boxed=False):
        import cv2
        s, th = self.s, max(1, round(2 * self.s))
        scale *= s
        (tw, tht), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        pad = max(3, round(8 * s))                     # names on a black box: readable on any background
        x = int(np.clip(x - tw / 2, pad, self.W - tw - pad)); y = int(max(tht + pad, y))
        cv2.rectangle(img, (x - pad, y - tht - pad), (x + tw + pad, y + bl + pad), (0, 0, 0), -1)
        if boxed:                                      # a corrected name: white frame
            cv2.rectangle(img, (x - pad, y - tht - pad), (x + tw + pad, y + bl + pad), (255, 255, 255), th)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, th, cv2.LINE_AA)

    def draw(self, img: np.ndarray, n: int, ov: Overlays = Overlays()) -> np.ndarray:
        import cv2
        s, th = self.s, max(1, round(2 * self.s))
        nd = self.analysed(n)
        tail = int(round(ov.tail_s * self.fps))
        where = {a: [] for a in self.animals}
        video_pos: Dict[str, tuple] = {}               # animal -> ("floor", xy) | ("box", zone)
        texts = []
        for cam, (cx, cy, cw, ch) in self.cells.items():
            sub = img[cy:cy + ch, cx:cx + cw]          # drawing clipped to the cell
            w, h = self.size_of.get(cam, (1920, 1080))
            sx, sy = cw / w, ch / h
            if ov.track_tails:
                for obj, (fr, xy, _) in self.tracks.get(cam, {}).items():
                    i0, i1 = np.searchsorted(fr, nd - tail), np.searchsorted(fr, nd, "right")
                    if i1 - i0 < 2:
                        continue
                    pts = np.round(xy[i0:i1] * [sx, sy]).astype(np.int32)
                    for k in range(1, len(pts)):
                        if fr[i0 + k] - fr[i0 + k - 1] > 3 * self.hold:
                            continue
                        a = self.names_at.get(cam, obj, int(fr[i0 + k]))[0]
                        cv2.line(sub, tuple(pts[k - 1]), tuple(pts[k]), self.colour(a), th, cv2.LINE_AA)
            for det in self.detections(cam, nd):
                name, unsure, fixed = self.names_at.get(cam, det["object_id"], nd)
                if not name and det["score"] < 0.35:
                    continue
                col = self.colour(name)
                if ov.masks and det["mask"] is not None:
                    cs, _ = cv2.findContours(det["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE, offset=det["origin"])
                    cs = [np.round(c.astype(np.float32) * [sx, sy]).astype(np.int32) for c in cs]
                    if cs:
                        bx, by, bw, bh = cv2.boundingRect(np.concatenate(cs))
                        x0, y0, x1, y1 = max(bx, 0), max(by, 0), min(bx + bw + 1, cw), min(by + bh + 1, ch)
                        if x1 > x0 and y1 > y0:
                            roi = sub[y0:y1, x0:x1]
                            lay = roi.copy()
                            cv2.fillPoly(lay, [c - [x0, y0] for c in cs], col)
                            sub[y0:y1, x0:x1] = cv2.addWeighted(lay, 0.35, roi, 0.65, 0)
                        cv2.polylines(sub, cs, True, col, th, cv2.LINE_AA)
                bx1, by1, bx2, _ = det["bbox"]
                if ov.names:
                    texts.append(((self.names[name] + ("?" if unsure else "")) if name else "?",
                                  cx + (bx1 + bx2) / 2 * sx, cy + by1 * sy - 10 * s, col,
                                  0.9 if cw > 1300 * s else 0.75, fixed))
                if name:
                    where[name].append(cam)
                    if cam in self.floor:
                        video_pos[name] = ("floor", self.floor[cam].to_floor(np.array([det["centroid"]], float))[0])
                    elif self.idn and "zone" in self.idn.cams.get(cam, {}):
                        video_pos.setdefault(name, ("box", self.idn.cams[cam]["zone"]))
            if ov.keypoints and cam in self.kp:
                fr, ob, pts_all = self.kp[cam]
                i0, i1 = np.searchsorted(fr, nd), np.searchsorted(fr, nd, "right")
                for k in range(i0, i1):
                    name = self.names_at.get(cam, int(ob[k]), nd)[0]
                    p = pts_all[k]
                    p = p[~np.isnan(p).any(1)]
                    p = np.round(p * [sx, sy]).astype(np.int32)
                    if len(p) > 1:
                        cv2.polylines(sub, [p], False, (255, 255, 255), th, cv2.LINE_AA)
                    for i, q in enumerate(p):
                        cv2.circle(sub, tuple(q), max(2, round((6 if i == 0 else 4) * s)),
                                   (255, 255, 255) if i else self.colour(name), -1, cv2.LINE_AA)
            if (ov.uwb or ov.uwb_tails) and self.idn:
                self._draw_uwb(sub, cam, n, tail, ov, sx, sy)
        for t in texts:
            self._label(img, *t[:5], boxed=t[5])
        if ov.floor and self.floor_cell and self.idn:
            self._draw_floor(img, n, nd, tail, ov, video_pos)
        if ov.legend:
            self._draw_legend(img, n, where, video_pos, ov)
        return img

    def _draw_uwb(self, sub, cam, n, tail, ov, sx, sy):
        import cv2
        s, th = self.s, max(1, round(2 * self.s))
        here = self.uwb_at(n)
        if cam in self.floor:                          # overhead: where UWB puts each animal
            fm = self.floor[cam]
            for a in here:
                if ov.uwb_tails:
                    f = np.arange(max(n - tail, 0), n + 1, 3)
                    f = f[self.idn.present[a][f]]
                    if len(f) > 1:
                        q = np.round(fm.to_pixels(self.idn.uxy[a][f]) * [sx, sy]).astype(np.int32)
                        for p in q:
                            cv2.circle(sub, tuple(p), max(2, round(4 * s)), self.colour(a), -1, cv2.LINE_AA)
                if ov.uwb:
                    p = np.round(fm.to_pixels(here[a][None]) [0] * [sx, sy]).astype(np.int32)
                    self._uwb_mark(sub, p, a, max(5, round(16 * s)), 0.75)
        elif ov.uwb and "zone" in self.idn.cams.get(cam, {}):   # zone camera: who UWB puts in this box
            from .mtt_identity import box_dist
            box = self.idn.boxes[self.idn.cams[cam]["zone"]]
            inside = [a for a, xy in here.items() if box_dist(xy[0], xy[1], box) <= 0.05]
            near = [a for a, xy in here.items() if 0.05 < box_dist(xy[0], xy[1], box) <= 0.3]
            y = int(70 * s)
            for txt, group, faded in (("UWB in box:", inside, False), ("UWB near:", near, True)):
                if not group:
                    continue
                cv2.rectangle(sub, (int(8 * s), y - int(30 * s)), (int(560 * s), y + int(12 * s)), (0, 0, 0), -1)
                cv2.putText(sub, txt, (int(14 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * s, (200, 200, 200), th,
                            cv2.LINE_AA)
                x = int(190 * s)
                for a in group:
                    col = self.colour(a) if not faded else tuple(int(v * 0.6) for v in self.colour(a))
                    cv2.putText(sub, self.names[a], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * s, col, th, cv2.LINE_AA)
                    x += int(120 * s)
                y += int(46 * s)

    def _uwb_mark(self, img, p, a, r, scale):
        """Where UWB puts an animal: a filled dot in its colour with its name beside it."""
        import cv2
        s, th = self.s, max(1, round(2 * self.s))
        p = (int(p[0]), int(p[1]))
        cv2.circle(img, p, r + max(1, round(3 * s)), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, p, r, self.colour(a), -1, cv2.LINE_AA)
        text = self.names.get(a, a)
        (tw, tht), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale * s, th)
        x, y = p[0] + r + int(6 * s), p[1] + tht // 2
        pad = max(2, round(5 * s))
        cv2.rectangle(img, (x - pad, y - tht - pad), (x + tw + pad, y + bl + pad), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale * s, self.colour(a), th, cv2.LINE_AA)

    def _floor_geometry(self):
        fx, fy, fw, fh = self.floor_cell
        xs, ys = [], []
        for b in self.idn.boxes.values():
            xs += [b[0], b[1]]; ys += [b[2], b[3]]
        for fm in self.floor.values():
            if fm.arena:
                xs += fm.arena[:2]; ys += fm.arena[2:]
        x0, x1, y0, y1 = min(xs) - 0.1, max(xs) + 0.1, min(ys) - 0.1, max(ys) + 0.1
        top = int(60 * self.s)
        k = min((fw - 20 * self.s) / (x1 - x0), (fh - top - 10 * self.s) / (y1 - y0))
        ox = fx + (fw - k * (x1 - x0)) / 2
        oy = fy + top
        return lambda xy: np.column_stack([ox + (np.asarray(xy, float).reshape(-1, 2)[:, 0] - x0) * k,
                                           oy + (y1 - np.asarray(xy, float).reshape(-1, 2)[:, 1]) * k])

    def _draw_floor(self, img, n, nd, tail, ov, video_pos):
        import cv2
        s, th = self.s, max(1, round(2 * self.s))
        fx, fy, fw, fh = self.floor_cell
        P = self._floor_geometry()
        ip = lambda xy: tuple(np.round(P(xy)[0]).astype(int))
        cv2.putText(img, "floor (m): square = video, dot + name = UWB", (int(fx + 20 * s), int(fy + 40 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (200, 200, 200), th, cv2.LINE_AA)
        for fm in self.floor.values():
            if fm.arena:
                x0, x1, y0, y1 = fm.arena
                cv2.rectangle(img, ip((x0, y1)), ip((x1, y0)), (110, 110, 110), th, cv2.LINE_AA)
        for z, b in self.idn.boxes.items():
            cv2.rectangle(img, ip((b[0], b[3])), ip((b[1], b[2])), (90, 90, 90), th, cv2.LINE_AA)
            cv2.putText(img, z, ip((b[0], b[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (140, 140, 140), th, cv2.LINE_AA)
        here = self.uwb_at(n)
        for a, xy in here.items():
            if ov.uwb_tails:
                f = np.arange(max(n - tail, 0), n + 1, 3)
                f = f[self.idn.present[a][f]]
                for p in np.round(P(self.idn.uxy[a][f])).astype(int):
                    cv2.circle(img, tuple(p), max(1, round(3 * s)), self.colour(a), -1, cv2.LINE_AA)
            self._uwb_mark(img, ip(xy), a, max(4, round(12 * s)), 0.6)
        for cam, per in self.tracks.items():
            if cam not in self.floor:
                continue
            for obj, (fr, _, fl) in per.items():
                i0, i1 = np.searchsorted(fr, nd - (tail if ov.track_tails else 0)), np.searchsorted(fr, nd, "right")
                if i1 <= i0 or fr[i1 - 1] != nd:
                    continue
                a = self.names_at.get(cam, obj, nd)[0]
                q = np.round(P(fl[i0:i1])).astype(np.int32)
                if len(q) > 1:
                    cv2.polylines(img, [q], False, self.colour(a), th, cv2.LINE_AA)
                self._video_mark(img, q[-1], a)
        in_box: Dict[str, List[str]] = {}
        for a, (kind, v) in video_pos.items():
            if kind == "box":
                in_box.setdefault(v, []).append(a)
        for z, group in in_box.items():
            b = self.idn.boxes[z]
            for i, a in enumerate(group):
                x = b[0] + (b[1] - b[0]) * (i + 1) / (len(group) + 1)
                self._video_mark(img, ip((x, (b[2] + b[3]) / 2)), a)

    def _video_mark(self, img, p, a):
        import cv2
        k = max(3, round(10 * self.s))
        cv2.rectangle(img, (int(p[0]) - k, int(p[1]) - k), (int(p[0]) + k, int(p[1]) + k), self.colour(a), -1)

    def _draw_legend(self, img, n, where, video_pos, ov):
        import cv2
        from .mtt_identity import box_dist
        s, th = self.s, max(1, round(2 * self.s))
        lx, ly, _, _ = self.legend_cell
        cv2.putText(img, f"{self.title} {self.clock(n):%Y-%m-%d %H:%M:%S}".strip(), (int(lx + 30 * s), int(ly + 60 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1 * s, (230, 230, 230), th, cv2.LINE_AA)
        note = "names: anchored when alone (UWB), carried along tracks; ? = unsure"
        if ov.uwb or ov.floor:
            note = "mask/square = video, dot + name = UWB; distance = video to UWB"
        cv2.putText(img, note, (int(lx + 30 * s), int(ly + 105 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                    (160, 160, 160), th, cv2.LINE_AA)
        here = self.uwb_at(n) if (ov.uwb or ov.floor) else {}
        for i, a in enumerate(self.animals):
            y = int(ly + (170 + i * 70) * s)
            cv2.circle(img, (int(lx + 50 * s), int(y - 10 * s)), max(4, round(16 * s)), self.colours[a], -1, cv2.LINE_AA)
            seen = ", ".join(sorted(set(where[a]))) or "not seen"
            if a in here and a in video_pos:
                kind, v = video_pos[a]
                d = (np.linalg.norm(v - here[a]) if kind == "floor"
                     else box_dist(here[a][0], here[a][1], self.idn.boxes[v]))
                seen += f"  {float(d):.2f} m"
            cv2.putText(img, f"{self.names[a]}  {seen}", (int(lx + 85 * s), y), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s,
                        (230, 230, 230) if where[a] else (120, 120, 120), th, cv2.LINE_AA)
