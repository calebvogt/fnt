"""The behaviour video with the photometry scrolling underneath it.

Every output frame is one input frame, so the overlay never drifts from the video: the panel
under frame k shows the traces around frame k's own time (k / fps). Each row is a window of
``window_s`` seconds centred on that time, with a cursor in the middle. Rows are drawn with
OpenCV and the frames piped to ffmpeg, so an hour of 1080p video takes minutes, not hours.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

BG = (30, 30, 30)
GRID = (70, 70, 70)
TEXT = (210, 210, 210)
CURSOR = (60, 60, 230)          # BGR red
GREEN, VIOLET, BLUE, AMBER = (122, 175, 27), (214, 79, 138), (214, 120, 42), (25, 178, 250)


@dataclass
class Row:
    """One trace row: samples (time, values), a label and a BGR colour.
    ``second`` is an optional second trace drawn on the same axes."""
    label: str
    time: np.ndarray
    values: np.ndarray
    colour: tuple
    second: Optional[np.ndarray] = None
    second_colour: tuple = VIOLET
    second_label: str = ""
    zero_line: bool = False


def _limits(*arrays):
    vals = np.concatenate([a[np.isfinite(a)] for a in arrays if a is not None])
    if not len(vals):
        return -1.0, 1.0
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def _polyline(img, xs, ys, colour, thickness=2):
    import cv2
    ok = np.isfinite(ys)
    if not ok.any():
        return
    # split at NaN runs (dropouts) so they show as gaps
    breaks = np.flatnonzero(np.diff(ok.astype(int)) != 0) + 1
    for seg in np.split(np.arange(len(ys)), breaks):
        if len(seg) > 1 and ok[seg[0]]:
            pts = np.column_stack([xs[seg], ys[seg]]).astype(np.int32)
            cv2.polylines(img, [pts], False, colour, thickness, lineType=cv2.LINE_AA)


def draw_panel(rows: List[Row], t_now: float, width: int, row_h: int, window_s: float,
               limits) -> np.ndarray:
    """The trace panel for one moment."""
    import cv2
    h = row_h * len(rows)
    img = np.full((h, width, 3), BG, np.uint8)
    left, right = 150, width - 20
    x_mid = (left + right) // 2
    t0, t1 = t_now - window_s / 2, t_now + window_s / 2
    for k, (row, (lo, hi)) in enumerate(zip(rows, limits)):
        top = k * row_h + 10
        bottom = (k + 1) * row_h - 24
        cv2.line(img, (left, bottom), (right, bottom), GRID, 1)
        for tick in np.arange(np.ceil(t0), t1, 5.0 if window_s <= 40 else 10.0):
            x = int(left + (tick - t0) / window_s * (right - left))
            cv2.line(img, (x, top), (x, bottom), GRID, 1)
            if k == len(rows) - 1:
                cv2.putText(img, f"{tick:.0f}", (x - 12, bottom + 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, TEXT, 1, cv2.LINE_AA)
        i0, i1 = np.searchsorted(row.time, [t0, t1])
        tt = row.time[i0:i1]
        xs = left + (tt - t0) / window_s * (right - left)

        def y_of(v):
            return bottom - (v - lo) / (hi - lo) * (bottom - top)

        if row.zero_line and lo < 0 < hi:
            y0 = int(y_of(0))
            cv2.line(img, (left, y0), (right, y0), GRID, 1)
        if row.second is not None:
            _polyline(img, xs, y_of(row.second[i0:i1]), row.second_colour)
        _polyline(img, xs, y_of(row.values[i0:i1]), row.colour)
        cv2.putText(img, row.label, (10, top + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, row.colour, 2,
                    cv2.LINE_AA)
        if row.second_label:
            cv2.putText(img, row.second_label, (10, top + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        row.second_colour, 2, cv2.LINE_AA)
        cv2.putText(img, f"{hi:.3g}", (left - 60, top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT,
                    1, cv2.LINE_AA)
        cv2.putText(img, f"{lo:.3g}", (left - 60, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT,
                    1, cv2.LINE_AA)
    cv2.line(img, (x_mid, 0), (x_mid, h), CURSOR, 2)
    cv2.putText(img, f"t = {t_now:7.2f} s", (right - 150, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                TEXT, 1, cv2.LINE_AA)
    return img


def render_combined_video(video_path: str, out_path: str, rows: List[Row], *, window_s: float = 20.0,
                          row_h: int = 170, width: Optional[int] = None, grayscale: bool = True,
                          crf: int = 23, progress: Optional[Callable[[int, int], None]] = None,
                          should_stop: Optional[Callable[[], bool]] = None) -> bool:
    """Write the video with ``rows`` under it to ``out_path`` (H.264 mp4).

    Returns False if stopped; the partial file is then removed.
    """
    import os
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw, vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = width or vw
    vid_h = int(round(vh * width / vw)) // 2 * 2
    panel_h = (row_h * len(rows)) // 2 * 2
    limits = [_limits(r.values, r.second) for r in rows]
    partial = out_path + ".partial"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{width}x{vid_h + panel_h}", "-r", f"{fps}", "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-f", "mp4", partial]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        from fnt.usv.usv_detector.power import keep_full_speed
        keep_full_speed(proc.pid)
    except Exception:
        pass
    ok = True
    try:
        for k in range(n):
            if should_stop and should_stop():
                ok = False
                break
            got, frame = cap.read()
            if not got:
                break
            if grayscale:
                frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            if frame.shape[1] != width or frame.shape[0] != vid_h:
                frame = cv2.resize(frame, (width, vid_h), interpolation=cv2.INTER_AREA)
            panel = draw_panel(rows, k / fps, width, row_h, window_s, limits)[:panel_h]
            proc.stdin.write(np.vstack([frame, panel]).tobytes())
            if progress and k % 300 == 0:
                progress(k, n)
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.wait()
    if ok and proc.returncode == 0:
        os.replace(partial, out_path)
        return True
    if os.path.exists(partial):
        os.remove(partial)
    if ok:
        raise RuntimeError(f"ffmpeg failed writing {out_path} (exit {proc.returncode})")
    return False
