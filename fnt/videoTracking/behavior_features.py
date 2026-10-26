"""Behavioural classification inputs for a stretch of video, from instance masks alone.

For one window of a video (a minute or so at the full frame rate) this runs the mask
model on every frame, links the detections into tracks, and measures each animal with
:mod:`mask_pose`: spine, nose and tail base, body length, speed, and motion in the head,
middle and rear thirds; and each pair of animals: nose-nose and nose-to-tail-base
distances, centroid distance (all in body lengths) and contact. Simple rules on those
measures then propose candidate bouts -- nose-nose and anogenital sniffing, chasing,
sniffing or grooming in place, immobility and huddling -- for a person to confirm or
correct in :mod:`behavior_review`. Confirmed bouts are the classifier's first labels.

Outputs, per window, in ``out_dir`` with the stem ``<video stem>_<first frame>``:

    _tracks.h5     Mask Tracker track store (every mask with its track id)
    _tracks.csv    one row per animal per frame: frame, t, track, cx, cy, area, speed,
                   body_len, nose_x/y, tb_x/y, heading, bend, tail, head_ev,
                   me_head/me_mid/me_rear
    _pairs.csv     one row per ordered pair per frame: focal, other, nose_nose,
                   nose_tailbase, centroid_dist, contact
    _bouts.csv     candidate bouts: kind, focal, other, frame0, frame1, dur_s, video,
                   tracks_csv, tracks_h5 (ready for behavior_review)

Distances are in body lengths and speeds in body lengths per second, so the thresholds
carry between cameras and frame rates.
"""
from __future__ import annotations

import csv
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import mask_pose as mp

# Tracking.
MAX_GAP_S = 1.0          # a track survives this long without a detection
MIN_TRACK_S = 1.0        # shorter tracks are dropped as flicker
STITCH_GAP_S = 3.0       # pieces this far apart in time may be one animal (see stitch_tracks)
STITCH_DIST_BL = 1.0     # ... if the next starts within this many body lengths of the last
# Candidate rules. BL = body length, nose to tail base (the scratchpad prototype measured
# with the tail in, about 1.8 BL, and these are its thresholds converted).
NOSE_NEAR_BL = 0.5
NOSE_MIN_S = 0.5         # facial investigation
ANOGENITAL_MIN_S = 1.0   # anogenital and flank: a brush past the rear is not an investigation
HEAD_CONF = 1.0          # |head evidence| below this: which end is the head is a guess
JOIN_GAP_S = 1.0         # facial and anogenital bouts this close are one investigation
ACTIVE_FRAC = 0.3        # a sniff needs the actor's head busy on this share of the bout
# Follow: one animal moving after another, heading at it. Chase: the same, fast (the
# follower over CHASE_FAST_BLS on most of the bout) -- an aggressive pursuit with a flight.
CHASE_FOCAL_BLS, CHASE_OTHER_BLS, CHASE_DIST_BL, CHASE_ANGLE, CHASE_MIN_S = 2.7, 1.8, 4.5, 35, 0.5
CHASE_FAST_BLS = 6.0
STILL_BLS, HEAD_RATIO, HEAD_MIN_S = 0.27, 2.0, 1.0
IMMOBILE_BLS, IMMOBILE_MIN_S = 0.18, 3.0

Detector = Callable[[np.ndarray], List[Tuple[np.ndarray, float]]]


# ---------------------------------------------------------------- detection
def yolo_detector(model_dir: str, conf: float = 0.25, imgsz: Optional[int] = None,
                  device: str = "") -> Detector:
    """Frame -> [(full-frame bool mask, score)], filtered as batch inference filters
    (specks dropped, one mask per animal, the model's ROI applied)."""
    from ultralytics import YOLO
    from .mask_tracker_batch import drop_fragments, one_per_animal, resolve_weights
    from .mtt_roi import model_roi

    model = YOLO(resolve_weights(model_dir))
    roi = model_roi(model_dir)
    if imgsz is None:
        try:
            import json
            with open(os.path.join(model_dir, "training_config.json")) as fh:
                imgsz = int(json.load(fh).get("imgsz", 640))
        except Exception:
            imgsz = 640

    def detect(frame):
        h, w = frame.shape[:2]
        img = frame if roi is None else roi.apply(frame)[0]
        r = model.predict(img, imgsz=imgsz, conf=conf, retina_masks=True,
                          device=device or None, verbose=False)[0]
        if r.masks is None or not len(r.masks):
            return []
        raw = r.masks.data.cpu().numpy() > 0.5
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
        ih, iw = img.shape[:2]
        if raw.shape[1:] != (ih, iw):
            raw = np.stack([cv2.resize(m.astype(np.uint8), (iw, ih),
                                       interpolation=cv2.INTER_NEAREST) > 0 for m in raw])
        if roi is not None:
            raw, kept = roi.to_frame(raw, h, w)
            raw, scores = raw[kept], scores[kept]
        masks = [drop_fragments(m) for m in raw]
        keep = one_per_animal(masks, scores)
        return [(masks[k], float(scores[k])) for k in keep if masks[k].any()]

    return detect


def _as_det(mask: np.ndarray, score: float) -> Optional[Dict]:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return {"bbox": (x1, y1, x2, y2), "crop": mask[y1:y2, x1:x2].copy(), "score": float(score),
            "centroid": np.array([xs.mean(), ys.mean()])}


# ---------------------------------------------------------------- tracking
def _box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def mask_iou(a: Dict, b: Dict) -> float:
    if _box_iou(a["bbox"], b["bbox"]) == 0:
        return 0.0
    x1, y1 = min(a["bbox"][0], b["bbox"][0]), min(a["bbox"][1], b["bbox"][1])
    x2, y2 = max(a["bbox"][2], b["bbox"][2]), max(a["bbox"][3], b["bbox"][3])
    ma = np.zeros((y2 - y1, x2 - x1), bool)
    mb = ma.copy()
    ax1, ay1, ax2, ay2 = a["bbox"]
    bx1, by1, bx2, by2 = b["bbox"]
    ma[ay1 - y1:ay2 - y1, ax1 - x1:ax2 - x1] = a["crop"]
    mb[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] = b["crop"]
    u = np.count_nonzero(ma | mb)
    return np.count_nonzero(ma & mb) / u if u else 0.0


def link_tracks(dets: Sequence[Sequence[Dict]], fps: float, max_gap_s: float = MAX_GAP_S,
                min_track_s: float = MIN_TRACK_S) -> Dict[int, Dict[int, Dict]]:
    """Detections per frame -> {track id: {frame index: detection}}.

    Each frame's detections are matched to the live tracks by mask overlap (Hungarian),
    falling back to centroid distance under 0.6 of a rough body length when masks don't
    overlap (a fast animal between frames). Unmatched detections start new tracks; a track
    survives ``max_gap_s`` without a detection; tracks shorter than ``min_track_s`` are
    dropped. Identity through merges and occlusions is not attempted.
    """
    from scipy.optimize import linear_sum_assignment

    tracks: Dict[int, Dict[int, Dict]] = {}
    last: Dict[int, Tuple[int, Dict]] = {}
    next_id = 1
    max_gap = max(1, int(round(max_gap_s * fps)))
    for t, frame_dets in enumerate(dets):
        alive = [k for k, (lt, _) in last.items() if t - lt <= max_gap]
        cost = np.ones((len(alive), len(frame_dets)))
        for i, k in enumerate(alive):
            prev = last[k][1]
            for j, d in enumerate(frame_dets):
                iou = mask_iou(prev, d)
                if iou > 0.05:
                    cost[i, j] = 1 - iou
                else:
                    dist = np.linalg.norm(prev["centroid"] - d["centroid"])
                    body = 2 * np.sqrt(prev["crop"].sum())
                    if dist < 0.6 * body:
                        cost[i, j] = 0.95 + 0.04 * dist / body
        used = set()
        if len(alive) and len(frame_dets):
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 0.999:
                    tracks[alive[i]][t] = frame_dets[j]
                    last[alive[i]] = (t, frame_dets[j])
                    used.add(j)
        for j, d in enumerate(frame_dets):
            if j not in used:
                tracks[next_id] = {t: d}
                last[next_id] = (t, d)
                next_id += 1
    tracks = stitch_tracks(tracks, fps)
    min_len = max(1, int(round(min_track_s * fps)))
    return {k: v for k, v in tracks.items() if len(v) >= min_len}


def stitch_tracks(tracks: Dict[int, Dict[int, Dict]], fps: float, max_gap_s: float = STITCH_GAP_S,
                  max_dist_bl: float = STITCH_DIST_BL, max_overlap: int = 3) -> Dict[int, Dict[int, Dict]]:
    """Join track pieces of one animal that lost its detection briefly.

    A piece that starts up to ``max_gap_s`` after another ends (or overlaps its last
    ``max_overlap`` frames: a brief double detection), within ``max_dist_bl`` body lengths of
    where that one ended, continues it -- but only when each is the other's best match, so
    two animals close together are never joined across. Body length here is estimated from
    the mask area (1.5 x its square root), before any keypoints exist. Where pieces
    overlap, the earlier piece's detections are kept."""
    def ends(tr):
        t0, t1 = min(tr), max(tr)
        area = float(np.median([d["crop"].sum() for d in tr.values()]))
        return t0, t1, tr[t0]["centroid"], tr[t1]["centroid"], 1.5 * np.sqrt(area)

    info = {k: ends(v) for k, v in tracks.items()}
    max_gap = int(round(max_gap_s * fps))
    cands = []
    for a, (a0, a1, _, a_end, a_bl) in info.items():
        for b, (b0, b1, b_start, _, _) in info.items():
            if a == b or b1 <= a1:
                continue
            gap = b0 - a1
            if -max_overlap <= gap <= max_gap:
                d = float(np.linalg.norm(b_start - a_end)) / max(a_bl, 1.0)
                if d < max_dist_bl:
                    cands.append((d + max(gap, 0) / fps * 0.1, a, b))
    best_next: Dict[int, Tuple[float, int]] = {}
    best_prev: Dict[int, Tuple[float, int]] = {}
    for c, a, b in cands:
        if a not in best_next or c < best_next[a][0]:
            best_next[a] = (c, b)
        if b not in best_prev or c < best_prev[b][0]:
            best_prev[b] = (c, a)
    links = {a: b for a, (_, b) in best_next.items() if best_prev.get(b, (None, None))[1] == a}
    heads = [k for k in tracks if k not in set(links.values())]
    out: Dict[int, Dict[int, Dict]] = {}
    for h in sorted(heads):
        merged = dict(tracks[h])
        k = h
        while k in links:
            k = links[k]
            for t, d in tracks[k].items():
                merged.setdefault(t, d)
        out[h] = merged
    return out


# ---------------------------------------------------------------- features
MERGED_X = 1.6           # a mask this many times one animal's area is several animals


def mark_merged(rows: List[Dict]):
    """Flag masks that are several animals merged into one, in place.

    One animal's area is the window's median over masks with a tail seen (a visible tail
    means the mask is a single animal), else over all masks. A track's own median would
    miss a pile that stays merged all window -- the case that matters for huddles."""
    single = [float(r["area"]) for r in rows if int(float(r.get("tail") or 0))]
    ref = float(np.median(single if len(single) >= 20 else [float(r["area"]) for r in rows])) if rows else 0.0
    for r in rows:
        r["merged"] = int(ref > 0 and float(r["area"]) > MERGED_X * ref)


def track_rows(tracks: Dict[int, Dict[int, Dict]], frames: Sequence[int], fps: float) -> List[Dict]:
    """Per animal per frame: geometry, oriented keypoints and speed (no pixels yet)."""
    rows = []
    for k, tr in tracks.items():
        ts = sorted(tr)
        geoms = [mp.mask_geometry(tr[t]["crop"], tr[t]["bbox"][:2]) for t in ts]
        cents = [tr[t]["centroid"] for t in ts]
        oriented, conf = mp.orient_track(geoms, cents, fps)
        kps = [mp.keypoints(g) if g is not None else None for g in oriented]
        # One body length per track: the median nose-to-tail-base length (tail excluded).
        lens = [kp["body_length"] for kp in kps if kp is not None and kp["body_length"] > 0]
        body = float(np.median(lens)) if lens else 1.0
        areas = [tr[t]["crop"].sum() for t in ts]
        for i, t in enumerate(ts):
            d = tr[t]
            i0, i1 = max(0, i - 2), min(len(ts) - 1, i + 2)
            dt_s = (ts[i1] - ts[i0]) / fps
            speed = (np.linalg.norm(cents[i1] - cents[i0]) / dt_s / body) if dt_s else 0.0
            r = {"frame": int(frames[t]), "t": t, "track": k, "area": int(areas[i]),
                 "cx": float(cents[i][0]),
                 "cy": float(cents[i][1]), "speed": float(speed), "body_ref": body,
                 "vx": float(cents[i1][0] - cents[i0][0]) / dt_s if dt_s else 0.0,
                 "vy": float(cents[i1][1] - cents[i0][1]) / dt_s if dt_s else 0.0}
            if oriented[i] is not None:
                kp = kps[i]
                r.update(body_len=kp["body_length"], nose_x=float(kp["nose"][0]),
                         nose_y=float(kp["nose"][1]), tb_x=float(kp["tail_base"][0]),
                         tb_y=float(kp["tail_base"][1]), heading=kp["heading"], bend=kp["bend"],
                         tail=int(bool(oriented[i]["tail"])), head_ev=conf[i],
                         _spine=kp["body_spine"], _kp=kp)
            rows.append(r)
    mark_merged(rows)
    return rows


def add_pixel_features(rows: List[Dict], tracks, video: str, frames: Sequence[int],
                       crop_size: int = 0):
    """Motion energy by body third from frame differences (and, with ``crop_size``,
    body-aligned crops). Reads the window's frames once, in order. Returns (crops, keys)."""
    by_t: Dict[int, List[Dict]] = {}
    for r in rows:
        by_t.setdefault(r["t"], []).append(r)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    prev = None
    crops, keys = [], []
    for t in range(len(frames)):
        ok, img = cap.read()
        if not ok:
            break
        grey = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        diff = cv2.absdiff(grey, prev) if prev is not None else np.zeros_like(grey)
        prev = grey
        for r in by_t.get(t, []):
            if "_spine" not in r:
                r.update(me_head=np.nan, me_mid=np.nan, me_rear=np.nan)
                continue
            d = tracks[r["track"]][t]
            r["me_head"], r["me_mid"], r["me_rear"] = mp.motion_by_thirds(
                diff, d["crop"], d["bbox"][:2], r["_spine"])
            if crop_size:
                crops.append(mp.body_aligned_crop(grey, d["crop"], d["bbox"][:2], r["_kp"], crop_size))
                keys.append((r["track"], r["frame"]))
    cap.release()
    return crops, keys


def pair_rows(rows: List[Dict], tracks) -> List[Dict]:
    """Every ordered pair of animals on every frame where both have keypoints."""
    from scipy import ndimage

    by_t: Dict[int, List[Dict]] = {}
    for r in rows:
        if "_kp" in r:
            by_t.setdefault(r["t"], []).append(r)
    out = []
    for t, rs in by_t.items():
        for a in rs:
            da = tracks[a["track"]][t]
            grown = {"bbox": da["bbox"], "crop": ndimage.binary_dilation(da["crop"], iterations=4)}
            for b in rs:
                if a is b:
                    continue
                d = mp.pair_distances(a["_kp"], b["_kp"], a["body_ref"])
                db = tracks[b["track"]][t]
                contact = mask_iou(grown, db) > 0
                out.append({"frame": a["frame"], "t": t, "focal": a["track"], "other": b["track"],
                            "centroid_dist": d["centroid"], "nose_nose": d["nose_nose"],
                            "nose_tailbase": d["nose_tailbase"], "contact": int(contact)})
    return out


# ---------------------------------------------------------------- candidate bouts
def _f(r: Dict, k: str) -> float:
    try:
        v = float(r.get(k, np.nan))
    except (TypeError, ValueError):
        return np.nan
    return v


def candidate_bouts(rows: List[Dict], pairs: List[Dict], fps: float, n_frames: int) -> List[Dict]:
    """Rule-based candidate bouts (first guesses, to be tuned against review):

    nose_nose   noses within NOSE_NEAR_BL for NOSE_MIN_S (each pair once): facial
                investigation, mutual
    anogenital  focal nose within NOSE_NEAR_BL of the other's tail base, not nose-nose, for
                ANOGENITAL_MIN_S
    flank       focal nose within NOSE_NEAR_BL of the other's mid-body, neither of the above,
                for ANOGENITAL_MIN_S
    follow      focal faster than CHASE_FOCAL_BLS heading at the other (< CHASE_ANGLE deg),
                the other faster than CHASE_OTHER_BLS moving away, within CHASE_DIST_BL;
                chase when the follower is over CHASE_FAST_BLS on most of it
    look_around body still (< STILL_BLS) with head motion over HEAD_RATIO x rear motion
                (looking around, vigilance; or grooming in place)
    immobile    slower than IMMOBILE_BLS for IMMOBILE_MIN_S; huddle when in contact with
                another animal, or when the mask is a merged blob of several

    The nose rules count a frame only when the animals involved have a confident head
    call (|head evidence| >= HEAD_CONF: a tail seen, or a strong taper or direction of
    travel), and keep a bout only when the actor's head is busy (head motion above the
    window's median) on at least ACTIVE_FRAC of it. Two resting mice side by side otherwise
    read as endless short sniffs, and a guessed head end as a sniff of the wrong end.

    ``rows`` and ``pairs`` are the dicts written to the tracks and pairs CSVs (numbers may
    be strings), so bouts can be rebuilt from saved windows (:func:`bouts_from_csv`).
    """
    T = n_frames
    key = {(int(r["t"]), int(r["track"])): r for r in rows}
    bouts = []

    def add(kind, focal, other, a, b):
        bouts.append({"kind": kind, "focal": focal, "other": other, "t0": a, "t1": b,
                      "dur_s": round((b - a + 1) / fps, 2)})

    heads = np.array([_f(r, "me_head") for r in rows])
    heads = heads[~np.isnan(heads)]
    busy_floor = float(np.median(heads)) if len(heads) else 0.0
    me = np.array([np.nanmean([_f(r, "me_head"), _f(r, "me_mid"), _f(r, "me_rear")]) for r in rows])
    me = me[~np.isnan(me)]
    head_floor = float(np.percentile(me, 60)) if len(me) else 0.0

    def confident(r):
        return r is not None and abs(_f(r, "head_ev")) >= HEAD_CONF

    def busy(track, a, b):
        flags = [_f(key[(t, track)], "me_head") > busy_floor for t in range(a, b + 1) if (t, track) in key]
        return bool(flags) and np.mean(flags) >= ACTIVE_FRAC

    by_pair: Dict[Tuple[int, int], Dict[int, Dict]] = {}
    for p in pairs:
        by_pair.setdefault((int(p["focal"]), int(p["other"])), {})[int(p["t"])] = p
    min_nose = max(1, int(round(NOSE_MIN_S * fps)))
    for (f, o), ps in by_pair.items():
        nn = np.zeros(T, bool)
        ag = np.zeros(T, bool)
        fl = np.zeros(T, bool)
        ch = np.zeros(T, bool)
        fast = np.zeros(T, bool)
        for t, p in ps.items():
            F, O = key.get((t, f)), key.get((t, o))
            close_nn = _f(p, "nose_nose") < NOSE_NEAR_BL
            close_ag = _f(p, "nose_tailbase") < NOSE_NEAR_BL
            both = confident(F) and confident(O)
            nn[t] = close_nn and both
            ag[t] = close_ag and not close_nn and both
            if both and not close_nn and not close_ag:
                mid = np.array([(_f(O, "nose_x") + _f(O, "tb_x")) / 2, (_f(O, "nose_y") + _f(O, "tb_y")) / 2])
                bl = _f(F, "body_ref") if not np.isnan(_f(F, "body_ref")) else _f(F, "body_len")
                d = np.linalg.norm(np.array([_f(F, "nose_x"), _f(F, "nose_y")]) - mid) / max(bl, 1e-6)
                fl[t] = d < NOSE_NEAR_BL
            if (F and O and confident(F) and _f(p, "centroid_dist") < CHASE_DIST_BL
                    and _f(F, "speed") > CHASE_FOCAL_BLS and _f(O, "speed") > CHASE_OTHER_BLS):
                to_o = np.array([_f(O, "cx") - _f(F, "cx"), _f(O, "cy") - _f(F, "cy")])
                h = np.radians(_f(F, "heading"))
                hd = np.array([np.cos(h), np.sin(h)])
                ang = np.degrees(np.arccos(np.clip(hd @ to_o / (np.linalg.norm(to_o) + 1e-9), -1, 1)))
                ch[t] = ang < CHASE_ANGLE and np.array([_f(O, "vx"), _f(O, "vy")]) @ to_o > 0
                fast[t] = ch[t] and _f(F, "speed") > CHASE_FAST_BLS
        min_ag = max(1, int(round(ANOGENITAL_MIN_S * fps)))
        if f < o:
            for a, b in mp.runs(nn, min_nose):
                if busy(f, a, b) or busy(o, a, b):
                    add("nose_nose", f, o, a, b)
        for a, b in mp.runs(ag, min_ag):
            if busy(f, a, b):
                add("anogenital", f, o, a, b)
        for a, b in mp.runs(fl, min_ag):
            if busy(f, a, b):
                add("flank", f, o, a, b)
        for a, b in mp.runs(ch, max(1, int(round(CHASE_MIN_S * fps)))):
            add("chase" if fast[a:b + 1].mean() > 0.5 else "follow", f, o, a, b)
    contact_t = {(int(p["t"]), int(p["focal"])) for p in pairs if int(p["contact"])}
    by_track: Dict[int, Dict[int, Dict]] = {}
    for r in rows:
        by_track.setdefault(int(r["track"]), {})[int(r["t"])] = r
    for k, rs in by_track.items():
        still = np.zeros(T, bool)
        head = np.zeros(T, bool)
        imm = np.zeros(T, bool)
        merged = np.zeros(T, bool)
        for t, r in rs.items():
            still[t] = _f(r, "speed") < STILL_BLS
            imm[t] = _f(r, "speed") < IMMOBILE_BLS
            merged[t] = bool(int(_f(r, "merged") or 0))
            h, rr = _f(r, "me_head"), _f(r, "me_rear")
            head[t] = (not np.isnan(h)) and h > HEAD_RATIO * (rr if not np.isnan(rr) else 0) and h > head_floor
        for a, b in mp.runs(still & head, max(1, int(round(HEAD_MIN_S * fps)))):
            add("look_around", k, "", a, b)
        for a, b in mp.runs(imm, max(1, int(round(IMMOBILE_MIN_S * fps))), gap=5):
            together = np.mean([((t, k) in contact_t) or merged[t] for t in range(a, b + 1)]) > 0.5
            add("huddle" if together else "immobile", k, "", a, b)
    return join_investigations(bouts, fps)


def join_investigations(bouts: List[Dict], fps: float, max_gap_s: float = JOIN_GAP_S) -> List[Dict]:
    """One bout per investigation sequence of a pair, pre-split where it changes.

    An actor's anogenital and flank bouts and the nose-nose bouts of the same two animals,
    less than ``max_gap_s`` apart, become one ``investigation`` bout whose ``segments`` list
    each part's first frame index and kind (a gap goes to the part after it), when the
    sequence holds more than one kind. The reviewer then confirms or moves the split
    instead of finding it. Each bout joins at most one sequence."""
    gap = int(round(max_gap_s * fps))
    used = set()
    joined = []
    ano = sorted((b for b in bouts if b["kind"] in ("anogenital", "flank")), key=lambda b: -b["dur_s"])
    for seed in ano:
        if id(seed) in used:
            continue
        f, o = seed["focal"], seed["other"]
        pool = [b for b in bouts if id(b) not in used and (
            (b["kind"] in ("anogenital", "flank") and b["focal"] == f and b["other"] == o) or
            (b["kind"] == "nose_nose" and {b["focal"], b["other"]} == {f, o}))]
        pool.sort(key=lambda b: b["t0"])
        i = pool.index(seed)
        lo = hi = i
        while lo > 0 and pool[lo]["t0"] - pool[lo - 1]["t1"] <= gap:
            lo -= 1
        while hi < len(pool) - 1 and pool[hi + 1]["t0"] - pool[hi]["t1"] <= gap:
            hi += 1
        chain = pool[lo:hi + 1]
        if len({b["kind"] for b in chain}) < 2:
            continue
        segs = []
        for b in chain:
            if not segs or segs[-1][1] != b["kind"]:
                segs.append((b["t0"], b["kind"]))
        segs[0] = (chain[0]["t0"], segs[0][1])
        t0, t1 = chain[0]["t0"], max(b["t1"] for b in chain)
        joined.append({"kind": "investigation", "focal": f, "other": o, "t0": t0, "t1": t1,
                       "dur_s": round((t1 - t0 + 1) / fps, 2), "segments_t": segs})
        used.update(id(b) for b in chain)
    return [b for b in bouts if id(b) not in used] + joined


def bouts_from_csv(tracks_csv: str, pairs_csv: str, fps: float) -> List[Dict]:
    """Candidate bouts rebuilt from a saved window's tracks and pairs CSVs (no detection),
    so changed rules can be applied to windows already run. Velocities missing from older
    files are recomputed from the centroids."""
    with open(tracks_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(pairs_csv, newline="", encoding="utf-8") as fh:
        pairs = list(csv.DictReader(fh))
    if rows and "vx" not in rows[0]:
        by: Dict[str, List[Dict]] = {}
        for r in rows:
            by.setdefault(r["track"], []).append(r)
        for rs in by.values():
            rs.sort(key=lambda r: int(r["t"]))
            for i, r in enumerate(rs):
                a, b = rs[max(0, i - 2)], rs[min(len(rs) - 1, i + 2)]
                dt = (int(b["t"]) - int(a["t"])) / fps
                r["vx"] = (float(b["cx"]) - float(a["cx"])) / dt if dt else 0.0
                r["vy"] = (float(b["cy"]) - float(a["cy"])) / dt if dt else 0.0
    mark_merged(rows)
    n = max(int(r["t"]) for r in rows) + 1 if rows else 0
    frame_of = {int(r["t"]): int(r["frame"]) for r in rows}
    bouts = candidate_bouts(rows, pairs, fps, n)
    for b in bouts:
        b["frame0"], b["frame1"] = frame_of[b["t0"]], frame_of[b["t1"]]
        b["segments"] = _segments_text(b, lambda t: frame_of[t])
    return bouts


def rebuild_bouts(bouts_csv: str, fps: Optional[float] = None) -> int:
    """Rewrite a window's bouts CSV with the current rules, from its tracks and pairs CSVs.
    Returns the number of bouts."""
    stem = bouts_csv[:-len("_bouts.csv")]
    with open(bouts_csv, newline="", encoding="utf-8") as fh:
        old = list(csv.DictReader(fh))
    video = old[0]["video"] if old else ""
    if fps is None:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        cap.release()
    bouts = bouts_from_csv(stem + "_tracks.csv", stem + "_pairs.csv", fps)
    for b in bouts:
        b.update(video=video, tracks_csv=stem + "_tracks.csv", tracks_h5=stem + "_tracks.h5")
    _write_csv(bouts_csv, BOUT_COLS, bouts)
    return len(bouts)


# ---------------------------------------------------------------- writing
TRACK_COLS = ["frame", "t", "track", "area", "merged", "cx", "cy", "vx", "vy", "speed", "body_ref",
              "body_len", "nose_x", "nose_y", "tb_x", "tb_y", "heading", "bend", "tail", "head_ev",
              "me_head", "me_mid", "me_rear"]
PAIR_COLS = ["frame", "t", "focal", "other", "centroid_dist", "nose_nose", "nose_tailbase", "contact"]
BOUT_COLS = ["kind", "focal", "other", "frame0", "frame1", "dur_s", "video", "tracks_csv", "tracks_h5",
             "segments"]


def _segments_text(b: Dict, frame_of) -> str:
    """``frame:kind;...`` for a pre-split bout (frame indices -> video frames)."""
    return ";".join(f"{frame_of(t)}:{k}" for t, k in b.get("segments_t", []))


def _write_csv(path, cols, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(v, 3) if isinstance(v, float) else v) for c, v in r.items() if c in cols})
    os.replace(tmp, path)


def write_track_store(path: str, video: str, fps: float, shape, frames, tracks, model: str = ""):
    from .track_store import TrackMaskWriter

    h, w = shape
    writer = TrackMaskWriter(path, video_path=video, fps=fps, frame_count=len(frames),
                             width=w, height=h, model_dir=model)
    by_t: Dict[int, Dict[int, Dict]] = {}
    for k, tr in tracks.items():
        for t, d in tr.items():
            by_t.setdefault(t, {})[k] = d
    for t in range(len(frames)):
        matched = {}
        for k, d in by_t.get(t, {}).items():
            m = np.zeros((h, w), bool)
            x1, y1, x2, y2 = d["bbox"]
            m[y1:y2, x1:x2] = d["crop"]
            matched[k] = {"mask": m, "bbox": list(d["bbox"]), "centroid": tuple(d["centroid"]),
                          "score": d["score"], "area": int(d["crop"].sum())}
        writer.add_frame(int(frames[t]), matched)
    writer.close()


def run_window(video: str, start_frame: int, n_frames: int, out_dir: str,
               detector: Detector, model_name: str = "", crop_size: int = 0,
               progress: Optional[Callable[[int, int], None]] = None) -> Dict[str, str]:
    """Detect, track, measure and propose bouts for frames ``start_frame`` onward.
    Returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (start_frame + n_frames)
    start_frame = max(0, min(int(start_frame), total - 1))
    end = min(total, start_frame + int(n_frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames, dets, shape = [], [], None
    for i, fi in enumerate(range(start_frame, end)):
        ok, img = cap.read()
        if not ok:
            break
        shape = img.shape[:2]
        frames.append(fi)
        dets.append([d for d in (_as_det(m, s) for m, s in detector(img)) if d is not None])
        if progress:
            progress(i + 1, end - start_frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {video} at {start_frame}")
    tracks = link_tracks(dets, fps)
    rows = track_rows(tracks, frames, fps)
    crops, keys = add_pixel_features(rows, tracks, video, frames, crop_size)
    pairs = pair_rows(rows, tracks)
    bouts = candidate_bouts(rows, pairs, fps, len(frames))
    stem = f"{os.path.splitext(os.path.basename(video))[0]}_{frames[0]}"
    paths = {k: os.path.join(out_dir, stem + suffix) for k, suffix in
             (("tracks_h5", "_tracks.h5"), ("tracks_csv", "_tracks.csv"),
              ("pairs_csv", "_pairs.csv"), ("bouts_csv", "_bouts.csv"))}
    write_track_store(paths["tracks_h5"], video, fps, shape, frames, tracks, model_name)
    _write_csv(paths["tracks_csv"], TRACK_COLS, sorted(rows, key=lambda r: (r["track"], r["t"])))
    _write_csv(paths["pairs_csv"], PAIR_COLS, pairs)
    for b in bouts:
        b.update(frame0=frames[b["t0"]], frame1=frames[b["t1"]], video=video,
                 tracks_csv=paths["tracks_csv"], tracks_h5=paths["tracks_h5"],
                 segments=_segments_text(b, lambda t: frames[t]))
    _write_csv(paths["bouts_csv"], BOUT_COLS, bouts)
    if crop_size and crops:
        paths["crops"] = os.path.join(out_dir, stem + "_crops.npz")
        np.savez_compressed(paths["crops"], crops=np.array(crops), keys=np.array(keys))
    return paths


def build_session(bouts_csvs: Sequence[str], out_csv: str, per_kind: int = 0, seed: int = 0) -> int:
    """Pool windows' candidate bouts into one review session, numbered, kinds interleaved.

    With ``per_kind`` > 0 each kind keeps at most that many bouts, taken in turn from the
    windows (each window's longest first), so a few busy minutes don't fill the session.
    Returns the number of bouts."""
    import random

    by_kind: Dict[str, Dict[str, List[Dict]]] = {}
    for p in bouts_csvs:
        with open(p, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                by_kind.setdefault(r["kind"], {}).setdefault(p, []).append(r)
    rng = random.Random(seed)
    chosen: Dict[str, List[Dict]] = {}
    for kind, wins in by_kind.items():
        for v in wins.values():
            v.sort(key=lambda r: -float(r["dur_s"]))
        order = sorted(wins)
        rng.shuffle(order)
        out, depth = [], 0
        while any(depth < len(wins[w]) for w in order):
            for w in order:
                if depth < len(wins[w]) and (not per_kind or len(out) < per_kind):
                    out.append(wins[w][depth])
            depth += 1
            if per_kind and len(out) >= per_kind:
                break
        chosen[kind] = out
    ordered, rank = [], 0
    while any(rank < len(v) for v in chosen.values()):
        for kind in sorted(chosen):
            if rank < len(chosen[kind]):
                ordered.append(chosen[kind][rank])
        rank += 1
    for i, r in enumerate(ordered):
        r["bout_id"] = str(i)
    _write_csv(out_csv, ["bout_id"] + BOUT_COLS, ordered)
    return len(ordered)
