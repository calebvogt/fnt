"""What a whole-video tracking run leaves behind, beyond its outputs.

Each ``<video>_MaskTracker/`` folder gets ``run_settings.json`` (the model, every
Inference-tab setting, the FNT version, when, how fast, how many frames had no
detection) and ``inference.log`` (the run's log for that video). With those,
a folder says how it was made, and ``--settings-from`` repeats it exactly.

``MaskTracker_summary.csv`` in a video folder is one row per tracked video,
rebuilt from those folders (an older run without ``run_settings.json`` is read
from its trajectories and mask store as far as they go).
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from typing import Dict, Iterable, List, Optional

SETTINGS_FILE = "run_settings.json"
LOG_FILE = "inference.log"
SUMMARY_FILE = "MaskTracker_summary.csv"
OUTPUT_SUFFIX = "_MaskTracker"


def output_dir_for(video: str) -> str:
    stem = os.path.splitext(os.path.basename(video))[0]
    return os.path.join(os.path.dirname(video), f"{stem}{OUTPUT_SUFFIX}")


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


# ---------------------------------------------------------------------------
# Folder summary
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "video", "status", "frames", "frames_processed", "frames_without_detection",
    "pct_without_detection", "tracks", "fps_processed", "elapsed_min", "finished",
    "model", "confidence", "max_objects", "masks", "resolution", "workers",
    "fnt_version", "output_dir",
]


def _row_from_record(video: str, out_dir: str, rec: Dict) -> Dict:
    s = rec.get("settings", {})
    frames = rec.get("frames")
    miss = rec.get("frames_without_detection")
    done = rec.get("frames_processed")
    return {
        "video": os.path.basename(video),
        "status": "done" if rec.get("complete") else "incomplete",
        "frames": frames,
        "frames_processed": done,
        "frames_without_detection": miss,
        "pct_without_detection": round(100 * miss / done, 2) if miss is not None and done else None,
        "tracks": rec.get("tracks"),
        "fps_processed": rec.get("fps_processed"),
        "elapsed_min": round(rec["elapsed_s"] / 60, 1) if rec.get("elapsed_s") else None,
        "finished": rec.get("finished"),
        "model": os.path.basename(os.path.normpath(rec.get("model_dir", ""))),
        "confidence": s.get("confidence_threshold"),
        "max_objects": s.get("max_detections"),
        "masks": s.get("use_masks"),
        "resolution": s.get("inference_size") or "trained",
        "workers": (rec.get("parallel") or {}).get("workers", 1),
        "fnt_version": rec.get("fnt_version"),
        "output_dir": out_dir,
    }


def _row_without_record(video: str, out_dir: str) -> Dict:
    """An output folder from before run_settings.json: what its files tell."""
    row = {c: None for c in SUMMARY_COLUMNS}
    row.update(video=os.path.basename(video), output_dir=out_dir, status="no record")
    csv = os.path.join(out_dir, "trajectories.csv")
    if not os.path.exists(csv):
        row["status"] = "incomplete"
        return row
    try:
        import pandas as pd
        df = pd.read_csv(csv, usecols=["frame", "object_id"])
        row["tracks"] = int(df["object_id"].nunique())
        seen = int(df["frame"].nunique())
    except Exception:
        seen = None
    h5 = [f for f in os.listdir(out_dir) if f.endswith("_tracks.h5")]
    if h5:
        try:
            import h5py
            with h5py.File(os.path.join(out_dir, h5[0]), "r") as f:
                row["frames"] = int(f.attrs.get("frame_count", 0)) or None
                row["model"] = os.path.basename(os.path.normpath(str(f.attrs.get("model_dir", ""))))
                row["finished"] = str(f.attrs.get("created", ""))
                row["masks"] = bool(f.attrs.get("has_masks", False))
        except Exception:
            pass
    if row["frames"] and seen is not None:
        row["frames_processed"] = row["frames"]
        row["frames_without_detection"] = row["frames"] - seen
        row["pct_without_detection"] = round(100 * (row["frames"] - seen) / row["frames"], 2)
    return row


def folder_summary(folder: str) -> List[Dict]:
    """One row per video in ``folder`` that has an output folder."""
    rows = []
    for name in sorted(os.listdir(folder)):
        out_dir = os.path.join(folder, name)
        if (name.startswith("_") or not name.endswith(OUTPUT_SUFFIX)
                or not os.path.isdir(out_dir)):        # "_TO_DELETE..." bins are not videos
            continue
        stem = name[: -len(OUTPUT_SUFFIX)]
        video = next((os.path.join(folder, f) for f in os.listdir(folder)
                      if os.path.splitext(f)[0] == stem and os.path.isfile(os.path.join(folder, f))),
                     os.path.join(folder, stem))
        if os.path.exists(os.path.join(out_dir, SETTINGS_FILE)):
            try:
                rows.append(_row_from_record(video, out_dir, read_run_settings(out_dir)))
                continue
            except Exception:
                pass
        rows.append(_row_without_record(video, out_dir))
    return rows


def write_folder_summary(folder: str) -> Optional[str]:
    """MaskTracker_summary.csv in ``folder``; None when nothing there is tracked."""
    import csv
    rows = folder_summary(folder)
    if not rows:
        return None
    path = os.path.join(folder, SUMMARY_FILE)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


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


def print_summary(rows: List[Dict], stream=None) -> None:
    stream = stream or sys.stdout
    for r in rows:
        miss = r.get("pct_without_detection")
        stream.write(f"{r['video']}: {r['status']}, {r.get('frames_processed') or '?'} frames, "
                     f"{'?' if miss is None else miss}% without detection, "
                     f"{r.get('tracks') if r.get('tracks') is not None else '?'} track(s), "
                     f"{r.get('fps_processed') or '?'} fps, model {r.get('model') or '?'}\n")
