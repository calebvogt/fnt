"""A camera grid with the tracked animals drawn on: masks in each animal's colour, midline keypoints and its name
over its head (from ``mtt_identity``), a legend of who is in which camera.

The grid is built from the clock-synced single-camera clips the tracks were made on (``fnt-grid --clips``; frame
n of every clip is the same moment), laid out as ``fnt-grid`` lays it out. The clip is split into parts drawn in
parallel and joined without re-encoding; a window shows each part's latest frame and the progress.

    python -m fnt.videoProcessing.tracked_grid <identity.json> [--size 4k|1080] [--workers 8] [--frames A:B]
        [--hold N] [--out FILE] [--no-window]

``identity.json`` is the one ``mtt_identity`` ran on (it names each camera's Mask Tracker folder; the clip is the
video beside it), with ``identities.csv`` in its ``out`` folder. ``--hold N``: tracks made on every Nth frame keep
each analysed frame's masks on screen until the next one. Optional keys in identity.json: ``names`` ({animal:
label}), ``title``, ``layout`` (default the mesocosm one), ``centre_scale`` (default 1.2).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from multiprocessing import Array, Process

import numpy as np

KP_ORDER = ["nose", "neck", "mid", "hip", "tailbase", "tailtip"]
PALETTE = [(60, 76, 231), (34, 156, 255), (182, 89, 155), (219, 152, 52), (113, 204, 46), (15, 196, 241),
           (160, 160, 255), (200, 120, 60), (90, 220, 220), (240, 240, 120)]          # BGR
GREY = (170, 170, 170)


def _video_of(track_dir: str) -> str:
    stem = os.path.basename(track_dir.rstrip("\\/"))[:-len("_MaskTracker")]
    cands = [p for p in glob.glob(os.path.join(os.path.dirname(track_dir.rstrip("\\/")), stem + ".*"))
             if not p.endswith((".csv", ".json"))]
    if not cands:
        raise FileNotFoundError(f"no video beside {track_dir}")
    return cands[0]


def _plan(cfg_path: str):
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    out_dir = cfg.get("out") or os.path.dirname(os.path.abspath(cfg_path))
    cams = {}
    for c, v in cfg["cameras"].items():
        d = v["tracks"] if not v["tracks"].endswith(".csv") else os.path.dirname(v["tracks"])
        store = glob.glob(os.path.join(d, "*_tracks.h5"))
        mid = glob.glob(os.path.join(d, "*_midline.csv"))
        cams[c] = dict(clip=_video_of(d), store=store[0] if store else None, midline=mid[0] if mid else None)
    return cfg, out_dir, cams


def worker(k, f0, f1, size, out_path, prev_dir, progress, cfg_path, hold):
    import cv2
    import pandas as pd
    from fnt.videoProcessing.camera_grid_cli import MESOCOSM_LAYOUT, parse_layout
    from fnt.videoProcessing.camera_grid_encode import EncodeSettings
    from fnt.videoTracking.mtt_identity import lookup
    from fnt.videoTracking.track_store import TrackMaskReader, long_path
    cfg, out_dir, cams = _plan(cfg_path)
    ids = pd.read_csv(os.path.join(out_dir, "identities.csv"), keep_default_na=False)
    who = lookup(ids)
    animals = sorted(a for a in ids.animal.unique() if a)
    names = {a: cfg.get("names", {}).get(a, a) for a in animals}
    colours = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(animals)}
    W, Hh = (3840, 2160) if size == "4k" else (1920, 1080)
    s = W / 3840
    layout = parse_layout(cfg.get("layout", MESOCOSM_LAYOUT), cfg.get("centre_scale", 1.2))
    geo = EncodeSettings(width=W, height=Hh)
    cells = {cam: geo.cell_rect(layout, r, c) for (r, c), cam in layout.assignments.items() if cam in cams}
    free = [(r, c) for r in range(layout.rows) for c in range(layout.cols) if (r, c) not in layout.assignments]
    lx, ly, _, _ = geo.cell_rect(layout, *(free[len(free) // 2 - 1] if free else (0, 0)))
    readers = {c: TrackMaskReader(v["store"]) for c, v in cams.items() if v["store"]}
    kp = {}
    for c, v in cams.items():
        if v["midline"]:
            m = pd.read_csv(long_path(v["midline"]))
            m = m[(m.frame >= f0 - hold) & (m.frame < f1)]
            kp[c] = {n: g for n, g in m.groupby("frame")}
    caps = {c: cv2.VideoCapture(v["clip"]) for c, v in cams.items() if c in cells}
    for cp in caps.values():
        cp.set(cv2.CAP_PROP_POS_FRAMES, f0)
    start = datetime.fromisoformat(cfg["start"])
    title = cfg.get("title", "")
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{Hh}",
                           "-r", str(cfg.get("fps", 30)), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2", out_path], stdin=subprocess.PIPE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    th = max(1, round(2 * s))

    def label(img, text, x, y, colour, scale):
        scale *= s
        (tw, tht), bl = cv2.getTextSize(text, font, scale, th)
        pad = max(3, round(8 * s))                     # names on a black box: readable on any background
        x = int(np.clip(x - tw / 2, pad, W - tw - pad)); y = int(max(tht + pad, y))
        cv2.rectangle(img, (x - pad, y - tht - pad), (x + tw + pad, y + bl + pad), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), font, scale, colour, th, cv2.LINE_AA)

    for n in range(f0, f1):
        img = np.zeros((Hh, W, 3), np.uint8)
        ok = True
        for c, cp in caps.items():
            good, fr = cp.read()
            if not good:
                ok = False
                break
            x, y, w, h = cells[c]
            img[y:y + h, x:x + w] = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
        if not ok:
            break
        nd = n - n % hold
        where = {a: [] for a in animals}
        texts = []
        for cam, (cx, cy, cw, ch) in cells.items():
            sx, sy = cw / 1920, ch / 1080
            for det in (readers[cam].detections_for_frame(nd) if cam in readers else []):
                name, unsure = who(cam, det["object_id"], nd)
                if not name and det["score"] < 0.35:
                    continue
                col = colours.get(name, GREY)
                if det["mask"] is not None:
                    cs, _ = cv2.findContours(det["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cs = [np.round(c.astype(np.float32) * [sx, sy] + [cx, cy]).astype(np.int32) for c in cs]
                    if cs:
                        bx, by, bw, bh = cv2.boundingRect(np.concatenate(cs))
                        x0, y0, x1, y1 = max(bx, 0), max(by, 0), min(bx + bw + 1, W), min(by + bh + 1, Hh)
                        roi = img[y0:y1, x0:x1]
                        lay = roi.copy()
                        cv2.fillPoly(lay, [c - [x0, y0] for c in cs], col)
                        img[y0:y1, x0:x1] = cv2.addWeighted(lay, 0.35, roi, 0.65, 0)
                        cv2.polylines(img, cs, True, col, th, cv2.LINE_AA)
                bx1, by1, bx2, _ = det["bbox"]
                texts.append(((names[name] + ("?" if unsure else "")) if name else "?",
                              cx + (bx1 + bx2) / 2 * sx, cy + by1 * sy - 10 * s, col, 0.9 if cw > 1300 * s else 0.75))
                if name:
                    where[name].append(cam)
            for _, r in (kp.get(cam, {}).get(nd, pd.DataFrame())).iterrows():
                name, _ = who(cam, int(r.object_id), nd)
                pts = [(cx + r[f"{q}_x"] * sx, cy + r[f"{q}_y"] * sy) for q in KP_ORDER
                       if f"{q}_x" in r and pd.notna(r[f"{q}_x"])]
                pts = np.round(np.array(pts)).astype(np.int32)
                if len(pts) > 1:
                    cv2.polylines(img, [pts], False, (255, 255, 255), th, cv2.LINE_AA)
                for i, p in enumerate(pts):
                    cv2.circle(img, tuple(p), max(2, round((6 if i == 0 else 4) * s)),
                               (255, 255, 255) if i else colours.get(name, GREY), -1, cv2.LINE_AA)
        for t in texts:
            label(img, *t)
        clock = start + timedelta(seconds=n / float(cfg.get("fps", 30)))
        cv2.putText(img, f"{title} {clock:%Y-%m-%d %H:%M:%S}".strip(), (int(lx + 30 * s), int(ly + 60 * s)), font,
                    1.1 * s, (230, 230, 230), th, cv2.LINE_AA)
        cv2.putText(img, "names: anchored when alone (UWB), carried along tracks; ? = unsure",
                    (int(lx + 30 * s), int(ly + 105 * s)), font, 0.8 * s, (160, 160, 160), th, cv2.LINE_AA)
        for i, a in enumerate(animals):
            y = int(ly + (170 + i * 70) * s)
            cv2.circle(img, (int(lx + 50 * s), int(y - 10 * s)), max(4, round(16 * s)), colours[a], -1, cv2.LINE_AA)
            seen = ", ".join(sorted(set(where[a]))) or "not seen"
            cv2.putText(img, f"{names[a]}  {seen}", (int(lx + 85 * s), y), font, 1.0 * s,
                        (230, 230, 230) if where[a] else (120, 120, 120), th, cv2.LINE_AA)
        ff.stdin.write(img.tobytes())
        progress[k] = n - f0 + 1
        if (n - f0) % 15 == 0:
            tmp = os.path.join(prev_dir, f"w{k}.tmp.jpg")
            cv2.imwrite(tmp, cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
            try:                   # the window may be reading the last preview (Windows locks it): skip it
                os.replace(tmp, os.path.join(prev_dir, f"w{k}.jpg"))
            except OSError:
                pass
    ff.stdin.close()
    ff.wait()


def render(cfg_path, size="4k", workers=8, frames=None, hold=1, out=None, window=True) -> str:
    import cv2
    cfg, out_dir, _ = _plan(cfg_path)
    f0, f1 = frames or (0, int(cfg["frames"]))
    out = out or os.path.join(out_dir, "tracked_grid" + ("" if size == "4k" else "_1080")
                              + (f"_f{f0}-{f1}" if frames else "") + ".mp4")
    work = tempfile.mkdtemp(prefix="fnt_tracked_")
    edges = np.linspace(f0, f1, workers + 1).round().astype(int)
    progress = Array("i", workers)
    parts = [os.path.join(work, f"part{k:02d}.mp4") for k in range(workers)]
    procs = [Process(target=worker, args=(k, edges[k], edges[k + 1], size, parts[k], work, progress, cfg_path, hold))
             for k in range(workers)]
    t0 = time.time()
    for p in procs:
        p.start()
    total = f1 - f0
    win = f"FNT - drawing tracked grid ({workers} at once)"
    if window:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cols = min(4, workers)
    rows = int(np.ceil(workers / cols))
    while any(p.is_alive() for p in procs):
        if window:
            tiles = []
            for k in range(workers):
                pth = os.path.join(work, f"w{k}.jpg")
                im = cv2.imread(pth) if os.path.exists(pth) else None
                im = np.zeros((360, 640, 3), np.uint8) if im is None else im
                n_k = edges[k + 1] - edges[k]
                cv2.rectangle(im, (0, 344), (int(640 * progress[k] / max(n_k, 1)), 360), (80, 200, 80), -1)
                cv2.putText(im, f"part {k + 1}: {progress[k]}/{n_k}", (8, 336), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (230, 230, 230), 1, cv2.LINE_AA)
                tiles.append(im)
            tiles += [np.zeros((360, 640, 3), np.uint8)] * (rows * cols - len(tiles))
            grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
            done, el = sum(progress), time.time() - t0
            rate = done / el if el else 0
            bar = np.full((100, grid.shape[1], 3), 30, np.uint8)
            cv2.rectangle(bar, (0, 70), (int(grid.shape[1] * done / total), 100), (80, 200, 80), -1)
            cv2.putText(bar, f"{done}/{total} frames, {rate:.1f} frames/s, "
                             f"{((total - done) / rate / 60) if rate else 0:.1f} min left", (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (230, 230, 230), 2, cv2.LINE_AA)
            cv2.imshow(win, np.vstack([grid, bar]))
            cv2.waitKey(500)
        else:
            time.sleep(1)
    if window:
        cv2.destroyAllWindows()
    bad = [k + 1 for k, p in enumerate(procs) if p.exitcode or progress[k] != edges[k + 1] - edges[k]]
    if bad:                                    # never join a video with a part missing frames
        raise RuntimeError(f"tracked grid: part(s) {bad} stopped early; nothing written (parts in {work})")
    lst = os.path.join(work, "parts.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="the identity.json mtt_identity ran on")
    ap.add_argument("--size", choices=["4k", "1080"], default="4k")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--frames", help="A:B, a stretch only")
    ap.add_argument("--hold", type=int, default=1, help="the frame step the tracks were made at")
    ap.add_argument("--out")
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    fr = tuple(int(v) for v in a.frames.split(":")) if a.frames else None
    out = render(a.config, a.size, a.workers, fr, a.hold, a.out, not a.no_window)
    print(f"wrote {out} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
