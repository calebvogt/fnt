"""Where animals can leave a fixed camera's view: its exits.

An enclosed space (a nest box, a resource zone) is entered and left only through a few
places, such as the mouth of an entrance tube. A tracker that knows them can keep counting
a mouse it has lost sight of anywhere else: it is hiding, not gone.

Exits are drawn per camera framing, so they live with the videos: ``exits.json`` in each
video folder, keyed by camera (the video name up to its first "_", as in frame_activity):

    {"CAM13": {"exits": [[[x, y], ...], ...], "frame_size": [w, h],
               "source": "checked", "score": 0.78}}

``source`` is "checked" once a person has drawn or confirmed them, otherwise how they were
placed (e.g. "auto", with the match ``score``).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .frame_activity import camera_key

EXITS_FILE = "exits.json"
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

Polygon = List[Tuple[float, float]]


def exits_path(folder: str) -> str:
    return os.path.join(folder, EXITS_FILE)


def load_exits(folder: str) -> Dict[str, dict]:
    """The folder's exits by camera ({} when there is no file)."""
    p = exits_path(folder)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_camera_exits(folder: str, camera: str, exits: Sequence[Polygon],
                      frame_size: Tuple[int, int], source: str = "checked", **extra) -> None:
    """Set one camera's exits in the folder's file, keeping the other cameras'."""
    data = load_exits(folder)
    entry = {"exits": [[[round(float(x), 1), round(float(y), 1)] for x, y in p]
                       for p in exits if len(p) >= 3],
             "frame_size": [int(v) for v in frame_size], "source": source}
    entry.update(extra)
    data[camera] = entry
    tmp = exits_path(folder) + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, exits_path(folder))


def cameras_in(items: Sequence[str]) -> Dict[Tuple[str, str], List[str]]:
    """{(folder, camera): its videos in name order} from video files and folders
    (searched recursively)."""
    videos = []
    for it in items:
        if os.path.isdir(it):
            for root, _, files in os.walk(it):
                videos += [os.path.join(root, f) for f in files if f.lower().endswith(VIDEO_EXTS)]
        elif it.lower().endswith(VIDEO_EXTS):
            videos.append(it)
    out: Dict[Tuple[str, str], List[str]] = {}
    for v in sorted(videos, key=lambda p: (os.path.dirname(p), os.path.basename(p))):
        out.setdefault(camera_key(v), []).append(v)
    return out


def median_frame(videos: Sequence[str], n: int = 15) -> Optional[np.ndarray]:
    """The median of the first frames of ``n`` clips spread over ``videos`` (BGR): animals
    and passing changes drop out, the fixtures stay."""
    if not videos:
        return None
    pick = [videos[int(i)] for i in np.unique(np.linspace(0, len(videos) - 1, n).round())]
    frames = []
    for v in pick:
        cap = cv2.VideoCapture(v)
        ok, fr = cap.read()
        cap.release()
        if ok and (not frames or fr.shape == frames[0].shape):
            frames.append(fr)
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
