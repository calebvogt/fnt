"""What a whole-video tracking run leaves behind, beyond its outputs.

Each ``<video>_MaskTracker/`` folder gets ``run_settings.json`` (the model, every
Inference-tab setting, the FNT version, when, how fast, how many frames had no
detection) and ``inference.log`` (the run's log for that video). With those,
a folder says how it was made, and ``--settings-from`` repeats it exactly.
"""
from __future__ import annotations

import json
import os
import platform
from datetime import datetime
from typing import Dict, Iterable, List, Optional

SETTINGS_FILE = "run_settings.json"
LOG_FILE = "inference.log"
OUTPUT_SUFFIX = "_MaskTracker"


def output_dir_for(video: str) -> str:
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(os.path.dirname(video), f"{stem}{OUTPUT_SUFFIX}")


def is_tracking_file(name: str) -> bool:
    """A file (or folder) in a <video>_MaskTracker folder that tracking wrote; other tools' files kept there, such
    as the ROI Tool's drawing and outputs (``*_roi*``), are not."""
    return "_roi" not in name


def has_tracking_output(video: str) -> bool:
    """The video's output folder holds a track (its trajectories, mask store or run record)."""
    out = output_dir_for(video)
    if not os.path.isdir(out):
        return False
    return any(n == SETTINGS_FILE or n.endswith("trajectories.csv") or n.endswith("_tracks.h5")
               or n == "trajectories_detailed.csv" for n in os.listdir(out))


def move_tracking_aside(video: str) -> Optional[str]:
    """Before re-tracking: move what tracking wrote in the video's output folder to
    ``_TO_DELETE_tracking/<video>_MaskTracker/`` beside the video (nothing is deleted); the folder and the other
    tools' files in it (the ROI Tool's drawing) stay. Returns where the files went, or None when there was nothing."""
    return move_tracking_aside_dir(output_dir_for(video))


def move_tracking_aside_dir(out: str) -> Optional[str]:
    """``move_tracking_aside`` for an output folder: what tracking wrote in ``out`` goes to
    ``_TO_DELETE_tracking/<folder>/`` beside it (where the video is). Returns where, or None."""
    import shutil
    from datetime import datetime
    out = os.path.normpath(out)
    if not os.path.isdir(out):
        return None
    files = [n for n in os.listdir(out) if is_tracking_file(n)]
    if not files:
        return None
    dest = os.path.join(os.path.dirname(out), "_TO_DELETE_tracking", os.path.basename(out))
    if os.path.exists(dest):
        dest += datetime.now().strftime("_%Y%m%d_%H%M%S")
    os.makedirs(dest)
    for n in files:
        shutil.move(os.path.join(out, n), os.path.join(dest, n))
    return dest


def fnt_version() -> str:
    try:
        from importlib.metadata import version
        return version("fnt")
    except Exception:
        return "unknown"


class LogTee:
    """Stands in for sys.stdout while one video runs: lines go on to the
    stream that was there (if any) and are kept, time-stamped, for the log."""

    def __init__(self, stream):
        self.stream = stream
        self.lines: List[str] = []

    def write(self, s):
        if self.stream is not None:
            self.stream.write(s)
        if s and s.strip():
            stamp = datetime.now().strftime("%H:%M:%S")
            self.lines += [f"{stamp} {ln}" for ln in s.rstrip().splitlines() if ln.strip()]
        return len(s) if s else 0

    def flush(self):
        if self.stream is not None and hasattr(self.stream, "flush"):
            self.stream.flush()


def write_run_record(output_dir: str, record: Dict, log_lines: Iterable[str] = ()) -> None:
    """run_settings.json and inference.log in a video's output folder."""
    rec = dict(record)
    rec.setdefault("fnt_version", fnt_version())
    rec.setdefault("host", platform.node())
    with open(os.path.join(output_dir, SETTINGS_FILE), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, default=str)
    lines = list(log_lines)
    if lines:
        with open(os.path.join(output_dir, LOG_FILE), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def read_run_settings(path: str) -> Dict:
    """The record of a run, from its output folder or its run_settings.json."""
    if os.path.isdir(path):
        path = os.path.join(path, SETTINGS_FILE)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_track_videos(items: Iterable[str]) -> List[str]:
    """Videos to track from files, folders (recursively) and .txt lists,
    leaving out what tracking writes (anything inside a ``_MaskTracker`` folder)."""
    from .mask_tracker_batch import find_videos
    out = []
    for v in find_videos(list(items)):
        parts = os.path.normpath(v).split(os.sep)
        if any(p.endswith(OUTPUT_SUFFIX) for p in parts[:-1]):
            continue
        out.append(v)
    return out
