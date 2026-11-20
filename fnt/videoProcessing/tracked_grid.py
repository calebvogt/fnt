"""A camera grid with the tracked animals drawn on: masks in each animal's colour, midline keypoints and its name
over its head (from ``mtt_identity``), a legend of who is in which camera.

The grid is built from the clock-synced single-camera clips the tracks were made on (``fnt-grid --clips``; frame
n of every clip is the same moment), laid out as ``fnt-grid`` lays it out. The clip is split into parts drawn in
parallel and joined without re-encoding; a window shows each part's latest frame and the progress.

    python -m fnt.videoProcessing.tracked_grid <identity.json> [--size 4k|1080] [--workers 8] [--frames A:B]
        [--hold N] [--uwb] [--tails S] [--floor] [--out FILE] [--no-window]

``identity.json`` is the one ``mtt_identity`` ran on (it names each camera's Mask Tracker folder; the clip is the
video beside it), with ``identities.csv`` in its ``out`` folder. ``--hold N``: tracks made on every Nth frame keep
each analysed frame's masks on screen until the next one. Optional keys in identity.json: ``names`` ({animal:
label}), ``title``, ``layout`` (default the mesocosm one), ``centre_scale`` (default 1.2).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Array, Process

import numpy as np

from fnt.videoTracking.identity_view import KP_ORDER, PALETTE, GREY, Overlays, load_config  # noqa: F401

_plan = load_config


def worker(k, f0, f1, size, out_path, prev_dir, progress, cfg_path, hold, overlays=None):
    import cv2
    from fnt.videoTracking.identity_view import Scene
    scene = Scene(cfg_path, size, hold or None)
    ov = Overlays(**(overlays or {}))
    W, Hh = scene.W, scene.H
    caps = {c: cv2.VideoCapture(v["clip"]) for c, v in scene.cams.items() if c in scene.cells}
    for cp in caps.values():
        cp.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{Hh}",
                           "-r", str(scene.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2", out_path], stdin=subprocess.PIPE)
    for n in range(f0, f1):
        frames = {}
        for c, cp in caps.items():
            good, fr = cp.read()
            if not good:
                break
            frames[c] = fr
        if len(frames) < len(caps):
            break
        img = scene.draw(scene.compose(frames), n, ov)
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


def render(cfg_path, size="4k", workers=8, frames=None, hold=None, out=None, window=True, overlays=None) -> str:
    import cv2
    cfg, out_dir, _ = _plan(cfg_path)
    f0, f1 = frames or (0, int(cfg["frames"]))
    out = out or os.path.join(out_dir, "tracked_grid" + ("" if size == "4k" else "_1080")
                              + (f"_f{f0}-{f1}" if frames else "") + ".mp4")
    work = tempfile.mkdtemp(prefix="fnt_tracked_")
    edges = np.linspace(f0, f1, workers + 1).round().astype(int)
    progress = Array("i", workers)
    parts = [os.path.join(work, f"part{k:02d}.mp4") for k in range(workers)]
    procs = [Process(target=worker, args=(k, edges[k], edges[k + 1], size, parts[k], work, progress, cfg_path, hold,
                                          overlays))
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
    ap.add_argument("--hold", type=int, default=None,
                    help="the frame step the tracks were made at (default: identity.json frame_step or the run settings)")
    ap.add_argument("--uwb", action="store_true", help="draw the UWB positions and who UWB puts in each box")
    ap.add_argument("--tails", type=float, default=0, metavar="S", help="draw video and UWB tails S seconds long")
    ap.add_argument("--floor", action="store_true", help="draw a floor map with video and UWB positions")
    ap.add_argument("--out")
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    fr = tuple(int(v) for v in a.frames.split(":")) if a.frames else None
    ov = dict(uwb=a.uwb, floor=a.floor, track_tails=a.tails > 0, uwb_tails=a.tails > 0 and (a.uwb or a.floor),
              tail_s=a.tails or 3.0)
    out = render(a.config, a.size, a.workers, fr, a.hold, a.out, not a.no_window, ov)
    print(f"wrote {out} in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
