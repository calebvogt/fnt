"""Midline keypoints from Mask Tracker masks: a body centreline and keypoints along it, for every frame of a track.

For each frame's mask (from ``<video>_MaskTracker/<video>_tracks.h5``, written when a video is tracked with
``--save-masks``): keep the largest piece, fill holes, smooth the outline; thin it to its skeleton and keep the longest
path through it (the spine); extend both ends along their direction to the mask edge; resample. Which end is the head
is decided per frame from a head cue, then made consistent over time (a two-state Viterbi pass that penalises the
path ends jumping between frames):

* ``bright`` (default): a bright marker on the head, such as a tracking tag or headcap: the largest patch inside the
  mask brighter than the body's median by ``--bright-delta``; the path end nearest it is the head;
* ``motion``: the end leading the direction of travel (any animal; uninformative while it sits still, and weak while
  it moves: on VoleCosm open-field videos it agreed with the head marker on ~70% of moving frames);
* ``none``: continuity only (the first frame's orientation is arbitrary).

The thin run of points at the back end (under a third of the body width) is the tail; the body midline ends at its
base. Keypoints sit at 0, 25, 50, 75 and 100% of the body midline (nose, neck, mid, hip, tailbase), plus the tail tip
when the tail is in the mask. Per frame: body length, head and body direction, head-body angle, bend, total turning,
and the marker's area (a quality signal). Written to ``<video>_MaskTracker/<video>_midline.csv``.

    python -m fnt.videoTracking.mtt_midline <videos, folders or .txt lists> [--head bright|motion|none] [--workers N]

Small animals are analysed enlarged: when a video's median mask is under the open-field size the thresholds were set
on (~7000 px, a vole filling ~120 px of an open-field frame), each crop is scaled up to that size first (``--scale
auto``; a mesocosm overhead vole of ~560 px is enlarged ~3.5x) and everything is written back in frame pixels.

Pixel units throughout (divide by the video's px/cm for cm). By default one animal (a multi-animal store uses its
object with the most frames); ``--objects all`` does every tracked animal, one row per animal per analysed frame."""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np

N_MID = 41                     # resampled body-midline points
KP_FRAC = {"nose": 0.0, "neck": 0.25, "mid": 0.5, "hip": 0.75, "tailbase": 1.0}
PAD = 25                       # crop margin around the box (px)
MIN_AREA = 500                 # smaller masks are skipped (px)
MARKER_MIN_AREA = 120          # a bright head marker counts from this size (px)
REF_AREA = 7000                # the mask size (px) the thresholds above were set on (VoleCosm open field)
MAX_SCALE = 6.0


# ----------------------------------------------------------------------------------------------- one frame
def _smooth_mask(m):
    from scipy import ndimage as ndi
    lab, n = ndi.label(m)
    if n == 0:
        return None
    if n > 1:
        m = lab == (1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))))
    m = ndi.binary_fill_holes(m)
    m = ndi.gaussian_filter(m.astype(np.float32), 2.0) > 0.5
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (1 + int(np.argmax(ndi.sum(m, lab, range(1, n + 1)))))
    return m


_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _longest_path(m):
    """The longest path through the mask's skeleton (x, y), or None."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from skimage.morphology import skeletonize
    sk = skeletonize(m)
    ys, xs = np.nonzero(sk)
    n = ys.size
    if n < 5:
        return None
    idx = -np.ones(sk.shape, np.int64)
    idx[ys, xs] = np.arange(n)
    H, W = sk.shape
    rows, cols, wts = [], [], []
    for dy, dx in _OFFS:
        yy, xx = ys + dy, xs + dx
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        j = np.full(n, -1)
        j[ok] = idx[yy[ok], xx[ok]]
        k = j >= 0
        rows.append(np.nonzero(k)[0])
        cols.append(j[k])
        wts.append(np.full(k.sum(), np.hypot(dy, dx)))
    G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    ncomp, labels = connected_components(G, directed=False)
    if ncomp > 1:
        keep = np.nonzero(labels == np.argmax(np.bincount(labels)))[0]
        G = G[keep][:, keep]
        ys, xs = ys[keep], xs[keep]
    d0 = dijkstra(G, indices=0, directed=False)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1)))
    da, pred = dijkstra(G, indices=a, directed=False, return_predecessors=True)
    b = int(np.argmax(np.where(np.isfinite(da), da, -1)))
    path = [b]
    while path[-1] != a:
        path.append(int(pred[path[-1]]))
    path = path[::-1]
    return np.stack([xs[path], ys[path]], 1).astype(np.float64)


def _extend_end(pts, m, back=10, max_len=80):
    """March from the last point along the end's direction until leaving the mask."""
    p, q = pts[-1], pts[max(0, len(pts) - 1 - back)]
    d = p - q
    nd = np.hypot(*d)
    if nd < 1e-6:
        return pts
    d = d / nd
    H, W = m.shape
    ext = []
    for s in range(1, max_len):
        r = p + d * s
        xi, yi = int(round(r[0])), int(round(r[1]))
        if not (0 <= xi < W and 0 <= yi < H) or not m[yi, xi]:
            break
        ext.append(r)
    return np.vstack([pts, ext]) if ext else pts


def _resample(pts, n):
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(pts[:1], n, 0), 0.0
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1), float(s[-1])


def _marker(gray, m, delta):
    """The bright head marker inside the mask: centre, area, contrast (or area 0)."""
    from scipy import ndimage as ndi
    inner = ndi.binary_erosion(m, iterations=4)
    v = gray[inner]
    if v.size < 50:
        return None
    body = float(np.median(v))
    hat = inner & (gray > body + delta)
    lab, n = ndi.label(hat)
    if n == 0:
        return dict(x=np.nan, y=np.nan, area=0, contrast=0.0)
    sizes = ndi.sum(hat, lab, range(1, n + 1))
    k = 1 + int(np.argmax(sizes))
    yy, xx = np.nonzero(lab == k)
    return dict(x=float(xx.mean()), y=float(yy.mean()), area=int(sizes.max()), contrast=float(gray[yy, xx].mean() - body))


def analyse_frame(mask, gray=None, delta=40.0):
    """One frame's cropped mask (bool) and grey image of the same crop: the oriented-later midline, or None."""
    from scipy import ndimage as ndi
    m = _smooth_mask(mask)
    if m is None or m.sum() < MIN_AREA:
        return None
    path = _longest_path(m)
    if path is None or len(path) < 5:
        return None
    path = _extend_end(path, m)
    path = _extend_end(path[::-1], m)[::-1]
    full, _ = _resample(path, 120)
    full = np.stack([ndi.gaussian_filter1d(full[:, 0], 2, mode="nearest"),
                     ndi.gaussian_filter1d(full[:, 1], 2, mode="nearest")], 1)
    full, _ = _resample(full, 120)
    dt = ndi.distance_transform_edt(m)
    H, W = m.shape
    xi = np.clip(np.round(full[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(full[:, 1]).astype(int), 0, H - 1)
    mk = _marker(gray, m, delta) if gray is not None else None
    ev = 0.0
    if mk is not None and mk["area"] >= MARKER_MIN_AREA:
        j = int(np.argmin(np.hypot(full[:, 0] - mk["x"], full[:, 1] - mk["y"])))
        ev = 1.0 - 2.0 * j / (len(full) - 1)          # +1: the marker is at the path's start
    return dict(full=full, width=2.0 * dt[yi, xi], marker=mk, ev_marker=ev, area=int(m.sum()))


# ----------------------------------------------------------------------------------------------- orientation
def _motion_evidence(results, step=5):
    """Per frame, +1 when the path's start leads the centroid's motion, -1 when its end does (0 while still)."""
    T = len(results)
    c = np.full((T, 2), np.nan)
    for t, r in enumerate(results):
        if r is not None:
            c[t] = (r["full"] / r.get("scale", 1.0) + r["off"]).mean(0)
    ev = np.zeros(T)
    for t, r in enumerate(results):
        if r is None:
            continue
        a, b = max(t - step, 0), min(t + step, T - 1)
        if np.isnan(c[a]).any() or np.isnan(c[b]).any():
            continue
        v = c[b] - c[a]
        sp = np.hypot(*v)
        if sp < 0.5 * step:                             # under ~0.5 px per frame: no evidence
            continue
        f = r["full"]
        axis = f[0] - f[-1]
        na = np.hypot(*axis)
        if na < 1e-6:
            continue
        ev[t] = float(np.clip(np.dot(v, axis) / (sp * na), -1, 1))
    return ev


def orient(results, evidence, a=4.0, scale=25.0):
    """Per frame 0 (the path's start is the head) or 1: head evidence plus continuity of the path ends."""
    T = len(results)
    cost = np.zeros((T, 2))
    back = np.zeros((T, 2), int)
    prev = None
    for t, r in enumerate(results):
        if r is None:
            cost[t] = cost[t - 1].min() if t else 0
            back[t] = np.argmin(cost[t - 1]) if t else 0
            prev = None
            continue
        u = np.array([-a * evidence[t], a * evidence[t]])
        f = r["full"] + r["off"]
        ends = [(f[12], f[-13]), (f[-13], f[12])]
        if prev is None or t == 0:
            cost[t] = (cost[t - 1].min() if t else 0.0) + u
            back[t] = np.argmin(cost[t - 1]) if t else 0
        else:
            for o in (0, 1):
                c = [cost[t - 1, po] + (np.hypot(*(ends[o][0] - prev[po][0])) + np.hypot(*(ends[o][1] - prev[po][1]))) / scale
                     for po in (0, 1)]
                back[t, o] = int(np.argmin(c))
                cost[t, o] = min(c) + u[o]
        prev = ends
    o = np.zeros(T, int)
    o[-1] = int(np.argmin(cost[-1]))
    for t in range(T - 1, 0, -1):
        o[t - 1] = back[t, o[t]]
    return o


def _ang(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def keypoints(r, o) -> Dict:
    """Orient one frame's midline, split off the tail, place the keypoints and measure the posture (frame px)."""
    full = r["full"][::-1] if o else r["full"]
    width = r["width"][::-1] if o else r["width"]
    off = r["off"]
    s = float(r.get("scale", 1.0))                       # analysed enlarged by s: lengths /s, areas /s^2 on output
    wb = float(np.percentile(width, 90))
    thin = width < 0.33 * wb
    k = len(full) - 1
    while k > len(full) // 2 and thin[k]:
        k -= 1
    tail = full[k:]
    tail_len = float(np.hypot(*np.diff(tail, axis=0).T).sum()) if len(tail) > 1 else 0.0
    has_tail = tail_len >= 15.0
    body = full[: k + 1] if has_tail else full
    mid, body_len = _resample(body, N_MID)
    mid = mid / s + off
    body_len, tail_len = body_len / s, tail_len / s
    kp = {name: mid[int(round(fr * (N_MID - 1)))] for name, fr in KP_FRAC.items()}
    th = np.unwrap(np.arctan2(*np.diff(mid, axis=0).T[::-1]))
    head_dir, body_dir = _ang(kp["nose"] - kp["neck"]), _ang(kp["neck"] - kp["tailbase"])
    out = dict(body_len=body_len, tail_len=tail_len if has_tail else 0.0, has_tail=has_tail,
               head_dir=head_dir, body_dir=body_dir, head_body_angle=_wrap(head_dir - body_dir),
               bend=_wrap(_ang(kp["nose"] - kp["mid"]) - _ang(kp["mid"] - kp["tailbase"])),
               total_turn=float(np.degrees(th[-1] - th[0])), abs_curv=float(np.degrees(np.abs(np.diff(th)).sum())),
               max_width=wb / s, mask_area=int(round(r["area"] / s ** 2)))
    mk = r.get("marker")
    out["marker_area"] = int(round(mk["area"] / s ** 2)) if mk else 0
    for name, p in kp.items():
        out[f"{name}_x"], out[f"{name}_y"] = float(p[0]), float(p[1])
    tip = full[-1] / s + off if has_tail else (np.nan, np.nan)
    out["tailtip_x"], out["tailtip_y"] = float(tip[0]), float(tip[1])
    return out


# ----------------------------------------------------------------------------------------------- one video
def analysis_scale(areas) -> float:
    """How much to enlarge a video's crops so its median mask is the open-field size the thresholds were set on."""
    a = np.asarray(areas, float)
    a = a[a > 0]
    if not len(a):
        return 1.0
    return float(np.clip(np.sqrt(REF_AREA / np.median(a)), 1.0, MAX_SCALE).round(2))


def store_path_for(video: str) -> str:
    from .mtt_runs import output_dir_for
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(output_dir_for(video), f"{stem}_tracks.h5")


def _store_plan(video: str, objects: str, scale: Optional[float]):
    """The objects to analyse, which store rows hold them on each frame, the scale and the frame count."""
    import cv2
    from .track_store import TrackMaskReader
    st = TrackMaskReader(store_path_for(video))
    if not st.has_masks:
        raise ValueError(f"{video}: its track has no masks (track it with --save-masks)")
    if scale is None:
        scale = analysis_scale(getattr(st, "area", ()))
    ids, counts = np.unique(st.object_id, return_counts=True)
    keep = set(int(i) for i in ids) if objects == "all" else {int(ids[np.argmax(counts)])}
    rows_at: Dict[int, List] = {}                              # frame -> [(object, store row)]
    seen = set()
    for i, (fr, oid) in enumerate(zip(st.frame, st.object_id)):
        if int(oid) in keep and (int(fr), int(oid)) not in seen:
            seen.add((int(fr), int(oid)))
            rows_at.setdefault(int(fr), []).append((int(oid), i))
    cap = cv2.VideoCapture(video)
    last = max(rows_at) if rows_at else -1
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (last + 1)
    cap.release()
    end = min(n, last + 1) if objects == "all" else n
    return st, keep, rows_at, float(scale), end


def _analyse_range(args):
    """Analyse frames [f_lo, f_hi) of one video (one worker's share): {object: [(frame, result)]}."""
    video, head, delta, objects, scale, f_lo, f_hi = args
    import cv2
    st, keep, rows_at, scale, _ = _store_plan(video, objects, scale)
    cap = cv2.VideoCapture(video)
    if f_lo:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)
    per: Dict[int, List] = {o: [] for o in keep}
    for fi in range(f_lo, f_hi):
        want = rows_at.get(fi)
        if objects == "all" and not want:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        for oid in keep:
            r = None
            row = next((i for o, i in (want or []) if o == oid), None)
            if row is not None:
                m_full = st.mask_for_row(row)
                if m_full is not None and m_full.any():
                    x1, y1, x2, y2 = st.bbox[row]
                    x0, y0 = max(int(x1) - PAD, 0), max(int(y1) - PAD, 0)
                    xe, ye = min(int(x2) + PAD, m_full.shape[1]), min(int(y2) + PAD, m_full.shape[0])
                    m_crop = m_full[y0:ye, x0:xe].astype(np.uint8)
                    g_crop = (cv2.cvtColor(frame[y0:ye, x0:xe], cv2.COLOR_BGR2GRAY)
                              if head == "bright" else None)
                    if scale > 1.0:
                        size = (int(round(m_crop.shape[1] * scale)), int(round(m_crop.shape[0] * scale)))
                        m_crop = cv2.resize(m_crop, size, interpolation=cv2.INTER_NEAREST)
                        if g_crop is not None:
                            g_crop = cv2.resize(g_crop, size, interpolation=cv2.INTER_CUBIC)
                    r = analyse_frame(m_crop.astype(bool), None if g_crop is None else g_crop.astype(float), delta)
                    if r is not None:
                        r["off"] = np.array([x0, y0], float)
                        r["scale"] = scale
            if objects == "all" and row is None:
                continue
            per[oid].append((fi, r))
    cap.release()
    return per


def midline_for_video(video: str, head: str = "bright", delta: float = 40.0, out_csv: Optional[str] = None,
                      objects: str = "largest", scale: Optional[float] = None, frame_workers: int = 1) -> str:
    """Midline keypoints for every frame of a tracked video (needs its mask store); returns the CSV path.

    ``objects="largest"`` (open field: one animal) keeps the object with the most frames and writes a row for every
    video frame. ``"all"`` (several animals, e.g. a mesocosm) does every object, each oriented along its own frames,
    and writes one row per object per analysed frame with its ``object_id`` (frames it was not detected on are left
    out; with a frame step only the analysed frames have masks). ``scale``: enlarge each crop this much before the
    analysis; None picks it from the median mask size (1 at open-field size and above). ``frame_workers``: split the
    video's frames across this many processes (each frame's midline is independent; the head/tail orientation
    over time is then done on the joined results, so the output is the same as one process)."""
    import pandas as pd
    _, keep, _, scale, end = _store_plan(video, objects, scale)
    edges = np.linspace(0, end, max(1, frame_workers) + 1).round().astype(int)
    jobs = [(video, head, delta, objects, scale, int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    if len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            parts = list(ex.map(_analyse_range, jobs))
    else:
        parts = [_analyse_range(j) for j in jobs]
    per: Dict[int, List] = {o: [] for o in keep}
    for part in parts:                                         # in frame order: the ranges are consecutive
        for o, items in part.items():
            per[o].extend(items)
    rows = []
    for oid, items in sorted(per.items()):
        res = [r for _, r in items]
        if head == "bright":
            ev = np.array([r["ev_marker"] if r else 0.0 for r in res])
        elif head == "motion":
            ev = _motion_evidence(res)
        else:
            ev = np.zeros(len(res))
        o = orient(res, ev) if res else []
        for k, (fi, r) in enumerate(items):
            base = dict(frame=fi, object_id=oid) if objects == "all" else dict(frame=fi)
            if r is None:
                rows.append(dict(base, ok=False))
            else:
                d = keypoints(r, o[k])
                d.update(base, ok=True, head_evidence=float(ev[k]))
                rows.append(d)
    df = pd.DataFrame(rows)
    if objects == "all" and len(df):
        df = df.sort_values(["frame", "object_id"])
    out_csv = out_csv or os.path.join(os.path.dirname(store_path_for(video)),
                                      os.path.splitext(os.path.basename(video))[0] + "_midline.csv")
    from .track_store import long_path
    df.to_csv(long_path(out_csv), index=False)          # NVR camera names run store paths past 260 characters
    return out_csv


def _one(args):
    video, head, delta, objects, scale, frame_workers = args
    t0 = time.time()
    try:
        return video, midline_for_video(video, head, delta, objects=objects, scale=scale,
                                        frame_workers=frame_workers), None, time.time() - t0
    except Exception as e:                                # one bad video doesn't stop the rest
        return video, None, str(e), time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+", help="tracked videos (with masks saved), folders or .txt lists")
    ap.add_argument("--head", choices=("bright", "motion", "none"), default="bright",
                    help="the head cue: a bright head marker (default), the direction of travel, or none")
    ap.add_argument("--bright-delta", type=float, default=40.0,
                    help="a head marker is this much brighter (grey levels) than the body's median")
    ap.add_argument("--objects", choices=("largest", "all"), default="largest",
                    help="largest: the one animal with the most frames (open field, default); all: every tracked "
                         "animal, one row per animal per analysed frame with its object_id (mesocosm)")
    ap.add_argument("--scale", default="auto",
                    help="enlarge each crop this much before the analysis (small animals, e.g. a mesocosm overhead "
                         "view); auto (default): from the median mask size, 1 at open-field size")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--frame-workers", type=int, default=1,
                    help="split each video's frames across this many processes (a long or crowded video; the "
                         "result is the same as one process)")
    ap.add_argument("--skip-existing", action="store_true", help="leave out videos that already have a midline CSV")
    a = ap.parse_args(argv)
    scale = None if str(a.scale).lower() == "auto" else float(a.scale)
    from .mtt_runs import find_track_videos
    from .track_store import store_exists
    vids = [v for v in find_track_videos(a.videos) if store_exists(store_path_for(v))]
    if a.skip_existing:
        vids = [v for v in vids if not store_exists(store_path_for(v).replace("_tracks.h5", "_midline.csv"))]
    if not vids:
        print("no tracked videos with a mask store found", file=sys.stderr)
        return 1
    print(f"midlines for {len(vids)} video(s), {a.workers} at once"
          + (f", each split {a.frame_workers} ways" if a.frame_workers > 1 else "") + f", head cue: {a.head}", flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, (v, a.head, a.bright_delta, a.objects, scale, a.frame_workers)) for v in vids]
        for i, f in enumerate(as_completed(futs), 1):
            v, out, err, s = f.result()
            print(f"[{i}/{len(vids)}] {'ok ' if out else 'FAILED'} {os.path.basename(v)} ({s:.0f} s){'' if out else ': ' + err}",
                  flush=True)
            failed += out is None
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
