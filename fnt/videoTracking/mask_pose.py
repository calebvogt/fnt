"""Mask-first pose: keypoints and pixel features from instance masks, with no keypoint labels.

The mask says where an animal is; its shape and the pixels under it say what it is doing.
Keypoint trackers need every point dragged into place on every training frame. Here the
points come from the outline the mask model already draws:

* **Spine.** The longest path through the mask's skeleton, extended out to the outline,
  with the body's width (twice the distance to the edge) measured along it.
* **Tail.** A thin run (under ``TAIL_WIDTH_FRAC`` of the widest point) reaching one end
  of the spine. Labelled masks usually include the tail, and it settles which end is
  which: the tail base is where the run starts, and the nose is the far end.
* **Head or rear without a tail.** The head third tapers more than the rump third, and
  animals mostly move head first. :func:`orient_track` weighs these cues with the tail
  frame by frame and keeps the head on the same end over a track unless the evidence turns
  strongly (a Viterbi path over the two orientations).
* **Pixels under the mask.** Frame-difference motion energy in the head, middle and rear
  thirds (a still body with a busy head is sniffing or grooming), and a body-aligned crop
  (nose up, background zeroed) for image embeddings.

Distances are in body lengths, so thresholds carry between cameras. Nothing here is
rodent-specific except the head cues: for other animals the spine and widths still hold,
and the orientation is simply weakly determined.
"""
from __future__ import annotations

import collections
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

TAIL_WIDTH_FRAC = 0.35     # thinner than this fraction of the widest point is tail
TAIL_MIN_FRAC = 0.15       # a tail run must cover this much of the spine
HOLE_FILL_FRAC = 0.03      # holes smaller than this fraction of the mask are filled
SPINE_POINTS = 41
FLIP_COST = 6.0            # evidence needed to move the head to the other end mid-track


def _longest_path(skel: np.ndarray) -> np.ndarray:
    """Pixel path (y, x) between the two skeleton points farthest apart along it."""
    pts = np.argwhere(skel)
    if len(pts) < 2:
        return pts
    index = {tuple(p): i for i, p in enumerate(pts)}
    nbrs: List[List[int]] = [[] for _ in pts]
    for i, (y, x) in enumerate(pts):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                j = index.get((y + dy, x + dx))
                if j is not None and j != i:
                    nbrs[i].append(j)

    def bfs(start):
        dist = {start: 0}
        parent = {start: -1}
        queue = collections.deque([start])
        while queue:
            u = queue.popleft()
            for v in nbrs[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
        return max(dist, key=dist.get), parent

    a, _ = bfs(0)
    b, parent = bfs(a)
    path = []
    while b != -1:
        path.append(pts[b])
        b = parent[b]
    return np.array(path)


def fill_small_holes(mask: np.ndarray, frac: float = HOLE_FILL_FRAC) -> np.ndarray:
    """Fill holes smaller than ``frac`` of the mask. A tail curled back onto the body
    encloses a large hole; filling that would turn the loop into body."""
    m = mask.astype(bool)
    limit = max(4, int(frac * m.sum()))
    inv = (~m).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    out = m.copy()
    h, w = m.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_edge = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_edge and area < limit:
            out[lbl == i] = True
    return out


def mask_geometry(crop: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> Optional[Dict]:
    """Spine, widths and tail of one mask.

    Args:
        crop: boolean mask, usually cut to the detection's box.
        offset: (x, y) of the crop's top-left corner in the frame.

    Returns ``{"spine": (N, 2) x, y from end A to end B, "width": (N,), "length": px,
    "tail": None or (end, run)}`` where ``end`` is 0 for A and 1 for B and ``run`` the
    number of spine points the tail covers; or None for a mask too small to measure.
    """
    from skimage.morphology import skeletonize

    pad = 2
    m = np.pad(fill_small_holes(crop), pad)
    dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    path = _longest_path(skeletonize(m))
    if len(path) < 5:
        return None
    p = path[:, ::-1].astype(float)                        # x, y
    k = max(3, len(p) // 15)
    p = np.array([p[max(0, i - k):i + k + 1].mean(0) for i in range(len(p))])
    # The skeleton stops short of the outline; carry each end out to the edge.
    ends = []
    for end, inner in ((p[0], p[min(5, len(p) - 1)]), (p[-1], p[max(-6, -len(p))])):
        u = end - inner
        n = np.linalg.norm(u)
        u = u / n if n else u
        q = end.copy()
        steps = []
        for _ in range(200):
            q = q + u
            yi, xi = int(round(q[1])), int(round(q[0]))
            if not (0 <= yi < m.shape[0] and 0 <= xi < m.shape[1]) or not m[yi, xi]:
                break
            steps.append(q.copy())
        ends.append(np.array(steps).reshape(-1, 2))
    p = np.vstack([ends[0][::-1], p, ends[1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    if s[-1] < 10:
        return None
    si = np.linspace(0, s[-1], SPINE_POINTS)
    spine = np.stack([np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1])], 1)
    yy = np.clip(np.round(spine[:, 1]).astype(int), 0, m.shape[0] - 1)
    xx = np.clip(np.round(spine[:, 0]).astype(int), 0, m.shape[1] - 1)
    width = 2 * dt[yy, xx]
    thin = width < TAIL_WIDTH_FRAC * width.max()
    tail = None
    for end in (0, 1):
        seq = thin if end == 0 else thin[::-1]
        run = 0
        while run < len(seq) and seq[run]:
            run += 1
        if run > TAIL_MIN_FRAC * SPINE_POINTS and (tail is None or run > tail[1]):
            tail = (end, run)
    spine = spine - pad + np.asarray(offset, float)
    return {"spine": spine, "width": width, "length": float(s[-1]), "tail": tail}


def flip_geometry(g: Dict) -> Dict:
    """The same geometry with ends A and B exchanged."""
    out = dict(g)
    out["spine"] = g["spine"][::-1].copy()
    out["width"] = g["width"][::-1].copy()
    if g["tail"]:
        out["tail"] = (1 - g["tail"][0], g["tail"][1])
    return out


def taper_evidence(g: Dict) -> float:
    """Positive when end A looks like the head: its third of the body (tail excluded) is
    narrower than end B's. In body-width units, so roughly -1 to 1."""
    w = g["width"]
    lo, hi = 0, len(w)
    if g["tail"]:
        end, run = g["tail"]
        if end == 0:
            lo = run
        else:
            hi = len(w) - run
    body = w[lo:hi]
    if len(body) < 6:
        return 0.0
    t = max(2, len(body) // 3)
    return float((body[-t:].mean() - body[:t].mean()) / (body.max() + 1e-6))


def orient_track(geoms: Sequence[Optional[Dict]], centroids: Sequence[np.ndarray],
                 fps: float, flip_cost: float = FLIP_COST) -> Tuple[List[Optional[Dict]], List[float]]:
    """Put the head at end A on every frame of one track.

    ``geoms`` and ``centroids`` are per frame (None where there is no geometry). Endpoint
    labels are first made consistent along the track (A stays the end nearest the last A),
    then each frame's evidence that A is the head -- tail (+-3), taper (x3), moving head
    first (up to +-2 once faster than half a body length per second) -- is combined by a
    two-state Viterbi path that pays ``flip_cost`` to switch ends.

    Returns the oriented geometries (end A = nose) and the per-frame evidence for the
    chosen orientation (large positive = confident).
    """
    geoms = [dict(g) if g is not None else None for g in geoms]
    prev = None
    for i, g in enumerate(geoms):
        if g is None:
            continue
        a, b = g["spine"][0], g["spine"][-1]
        if prev is not None and (np.linalg.norm(a - prev[1]) + np.linalg.norm(b - prev[0])
                                 < np.linalg.norm(a - prev[0]) + np.linalg.norm(b - prev[1])):
            geoms[i] = g = flip_geometry(g)
        prev = (g["spine"][0], g["spine"][-1])
    lengths = [g["length"] for g in geoms if g is not None]
    body = float(np.median(lengths)) if lengths else 1.0
    n = len(geoms)
    ev = np.zeros(n)
    for i, g in enumerate(geoms):
        if g is None:
            continue
        e = 3.0 * taper_evidence(g)
        if g["tail"]:
            e += 3.0 if g["tail"][0] == 1 else -3.0
        i0, i1 = max(0, i - 2), min(n - 1, i + 2)
        if i1 > i0 and centroids[i0] is not None and centroids[i1] is not None:
            v = (np.asarray(centroids[i1]) - np.asarray(centroids[i0])) * fps / (i1 - i0)
            speed = np.linalg.norm(v) / body
            axis = g["spine"][0] - g["spine"][-1]
            if speed > 0.5 and np.linalg.norm(axis):
                cos = v @ axis / (np.linalg.norm(v) * np.linalg.norm(axis))
                e += 2.0 * cos * min(1.0, speed / 2.0)
        ev[i] = e
    score = np.zeros(2)
    back = []
    for e in ev:
        stay, flip = score, score[::-1] - flip_cost
        back.append((flip > stay).astype(int))
        score = np.maximum(stay, flip) + np.array([e / 2, -e / 2])
    state = int(np.argmax(score))
    states = []
    for choice in reversed(back):
        states.append(state)
        if choice[state]:
            state = 1 - state
    states.reverse()
    out, conf = [], []
    for g, s, e in zip(geoms, states, ev):
        if g is None:
            out.append(None)
            conf.append(0.0)
        else:
            out.append(g if s == 0 else flip_geometry(g))
            conf.append(float(e if s == 0 else -e))
    return out, conf


def keypoints(g: Dict) -> Dict:
    """Nose, tail base, body midpoint, body length (tail excluded) and bend from an oriented
    geometry (end A = nose)."""
    sp = g["spine"]
    n = len(sp)
    tb_idx = n - 1 - g["tail"][1] if (g["tail"] and g["tail"][0] == 1) else n - 1
    nose, tb, mid = sp[0], sp[tb_idx], sp[tb_idx // 2]
    body = float(np.linalg.norm(np.diff(sp[:tb_idx + 1], axis=0), axis=1).sum())
    v1, v2 = nose - mid, mid - tb
    bend = float(np.degrees(np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)))
    heading = float(np.degrees(np.arctan2(nose[1] - tb[1], nose[0] - tb[0])))
    return {"nose": nose, "tail_base": tb, "mid": mid, "body_length": body, "bend": bend,
            "heading": heading, "body_spine": sp[:tb_idx + 1]}


def motion_by_thirds(diff: np.ndarray, crop: np.ndarray, offset: Tuple[int, int],
                     body_spine: np.ndarray) -> Tuple[float, float, float]:
    """Mean frame difference under the mask in the head, middle and rear thirds, each pixel
    assigned to the third of its nearest body-spine point."""
    ys, xs = np.nonzero(crop)
    if not len(ys) or len(body_spine) < 3:
        return (np.nan, np.nan, np.nan)
    pts = np.stack([xs + offset[0], ys + offset[1]], 1).astype(float)
    step = max(1, len(body_spine) // 12)
    anchors = body_spine[::step]
    nearest = np.argmin(((pts[:, None, :] - anchors[None]) ** 2).sum(-1), axis=1) * step
    part = np.minimum(2, (nearest * 3) // len(body_spine))
    vals = diff[ys + offset[1], xs + offset[0]].astype(float)
    return tuple(float(vals[part == i].mean()) if np.any(part == i) else np.nan for i in range(3))


def body_aligned_crop(grey: np.ndarray, crop: np.ndarray, offset: Tuple[int, int],
                      kp: Dict, size: int = 96) -> np.ndarray:
    """The animal's pixels rotated nose-up, centred on the body, two body lengths across,
    background zeroed."""
    nose, tb = kp["nose"], kp["tail_base"]
    c = (nose + tb) / 2
    angle = np.degrees(np.arctan2(nose[1] - tb[1], nose[0] - tb[0])) + 90
    side = max(16.0, 2 * kp["body_length"])
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), angle, size / side)
    M[:, 2] += size / 2 - c
    full = np.zeros(grey.shape[:2], np.uint8)
    h, w = crop.shape
    full[offset[1]:offset[1] + h, offset[0]:offset[0] + w] = crop
    g = cv2.warpAffine(grey, M, (size, size))
    m = cv2.warpAffine(full, M, (size, size), flags=cv2.INTER_NEAREST)
    return np.where(m > 0, g, 0).astype(np.uint8)


def pair_distances(a: Dict, b: Dict, body_length: float) -> Dict[str, float]:
    """Nose-nose, nose-to-tail-base (a's nose to b's tail base) and centroid distance, in
    body lengths of ``a``."""
    bl = max(body_length, 1e-6)
    return {"nose_nose": float(np.linalg.norm(a["nose"] - b["nose"]) / bl),
            "nose_tailbase": float(np.linalg.norm(a["nose"] - b["tail_base"]) / bl),
            "centroid": float(np.linalg.norm(a["mid"] - b["mid"]) / bl)}


def runs(flags: Sequence[bool], min_len: int, gap: int = 3) -> List[Tuple[int, int]]:
    """(first, last) index of each run of True, bridging gaps of up to ``gap`` False, kept
    when at least ``min_len`` long. The bout finder behind every candidate rule."""
    idx = np.flatnonzero(np.asarray(flags, bool))
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap + 1:
            out.append((int(start), int(prev)))
            start = i
        prev = i
    out.append((int(start), int(prev)))
    return [(a, b) for a, b in out if b - a + 1 >= min_len]
