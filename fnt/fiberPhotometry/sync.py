"""Putting photometry on the behaviour video's frames.

Each video frame gets the photometry samples recorded while it was on screen (from its own time
to the next frame's), averaged. Frame times come from the video file itself: frame k of a
constant-rate video is shown at k / fps.

Doric note: the .doric file logs fewer video timestamps than the camera writes frames. In pilot
P001 the mp4s hold 18-20% more frames than the .doric has timestamps for, yet both span the same
time (e.g. 102,773 frames at 30 fps = 3,425.8 s against timestamps ending at 3,425.7 s). So
pairing the n-th timestamp with the n-th frame drifts by minutes over an hour; timing frames by
the video's own rate does not. ``check_video_clock`` reports the comparison.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def video_frame_times(video_path: str) -> Tuple[np.ndarray, float]:
    """(frame times in s, fps) of a constant-rate video, counting its frames."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or fps <= 0 or n <= 0:
        raise OSError(f"no frame rate or frame count in {video_path}")
    return np.arange(n) / fps, float(fps)


def recording_frame_times(data, video_path: str) -> Tuple[np.ndarray, float]:
    """(frame times in s on the photometry clock, fps) for ``data``'s video.

    Timed by the video's own rate (``video_frame_times``) unless the recording logged one
    hardware time per frame (``data.video_frame_clock``, e.g. a camera TTL). Then frame k takes
    the k-th logged time: surplus logged times at the end are dropped, as the Warden lab MATLAB
    did, and frames past the last logged time continue at the median interval.
    """
    ft, fps = video_frame_times(video_path)
    logged = getattr(data, "video_timestamps", None)
    if not getattr(data, "video_frame_clock", False) or logged is None or len(logged) < 2:
        return ft, fps
    logged = np.asarray(logged, float)
    n = len(ft)
    if n <= len(logged):
        return logged[:n], fps
    step = float(np.median(np.diff(logged)))
    return np.concatenate([logged, logged[-1] + step * np.arange(1, n - len(logged) + 1)]), fps


def check_video_clock(frame_times: np.ndarray, logged_times: Optional[np.ndarray]) -> Dict:
    """Compare the video's own frame times with timestamps the acquisition system logged."""
    out = {"video_frames": int(len(frame_times)),
           "video_span_s": float(frame_times[-1]) if len(frame_times) else 0.0}
    if logged_times is not None and len(logged_times):
        out.update(logged_frames=int(len(logged_times)),
                   logged_span_s=float(logged_times[-1] - logged_times[0]),
                   frame_count_ratio=float(len(frame_times) / len(logged_times)))
    return out


def align_to_frames(time: np.ndarray, values: np.ndarray, frame_times: np.ndarray,
                    offset_s: float = 0.0) -> np.ndarray:
    """Mean of ``values`` over each frame's interval; NaN samples are ignored.

    A frame with no finite sample inside its interval takes the nearest finite sample, unless
    that sample is more than one frame interval away (then NaN). ``offset_s`` is added to the
    frame times (video start relative to photometry time zero).
    """
    t = np.asarray(time, float)
    v = np.asarray(values, float)
    ft = np.asarray(frame_times, float) + offset_s
    if len(ft) == 0:
        return np.array([])
    step = np.median(np.diff(ft)) if len(ft) > 1 else 1 / 30
    edges = np.append(ft, ft[-1] + step)
    idx = np.searchsorted(t, edges)
    ok = np.isfinite(v)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(ok)])
    cnt = ccnt[idx[1:]] - ccnt[idx[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (csum[idx[1:]] - csum[idx[:-1]]) / cnt
    empty = cnt == 0
    if empty.any() and ok.any():
        tf, vf = t[ok], v[ok]
        j = np.clip(np.searchsorted(tf, ft[empty]), 1, len(tf) - 1)
        left, right = tf[j - 1], tf[j]
        near = np.where(np.abs(ft[empty] - left) <= np.abs(right - ft[empty]), j - 1, j)
        dist = np.abs(tf[near] - ft[empty])
        out[empty] = np.where(dist <= step, vf[near], np.nan)
    return out
