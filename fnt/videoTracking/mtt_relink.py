"""Link a finished Mask Tracker run again, without re-detecting.

``python -m fnt.videoTracking.mask_tracker_cli relink <output folders...>``
reads each ``<video>_MaskTracker`` folder's saved detections (the
``*_tracks.h5`` mask store; ``trajectories.csv`` alone for box-only runs),
joins them into tracks again with the current linker (or the settings given),
and writes the folder's outputs anew: ``trajectories.csv``, the mask store
with the new ids, ``run_settings.json`` and ``inference.log``. What the run
wrote before moves to ``_TO_DELETE_tracking/<folder>/`` beside it (nothing is
deleted), as ``track --redo`` does. The tracked video shows the old ids, so it
moves aside too; ``--tracked-video`` draws a new one from the source video.

Detection settings (frame step, object limit, model) stay the run's; linker
settings not given take the current defaults, not the old run's, so an old
"never end tracks" run is linked the new way unless asked otherwise.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

#: Settings that belong to the linker (relink takes the defaults for these unless given).
LINKER_KEYS = ("linker", "max_disappeared_frames", "gate_px", "iou_weight", "two_pass_conf",
               "motion", "takeover", "matching_algorithm")


def decode_crop(counts: np.ndarray, height: int) -> Optional[Tuple[int, int, np.ndarray]]:
    """One detection's COCO RLE (column-major runs, zeros first) decoded into
    (x0, y0, mask cropped to its own box) without building the full frame."""
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size < 2:
        return None
    ends = np.cumsum(counts)
    starts = ends - counts
    s, n = starts[1::2], counts[1::2]
    keep = n > 0
    s, n = s[keep], n[keep]
    total = int(n.sum())
    if total == 0:
        return None
    # every pixel index of the runs of ones, without a Python loop
    offs = np.repeat(s - np.concatenate(([0], np.cumsum(n)[:-1])), n) + np.arange(total)
    rows, cols = offs % height, offs // height
    y0, x0 = int(rows.min()), int(cols.min())
    crop = np.zeros((int(rows.max()) - y0 + 1, int(cols.max()) - x0 + 1), dtype=bool)
    crop[rows - y0, cols - x0] = True
    return x0, y0, crop


def _store_in(folder: str) -> Optional[str]:
    hits = sorted(n for n in os.listdir(folder) if n.endswith("_tracks.h5"))
    return os.path.join(folder, hits[0]) if hits else None


def _categories(rec: Dict, folder_csv: Optional[str]) -> Dict:
    """The model's classes ({"1": "vole"}), from its training_config.json, else read
    back from the old trajectories' object names."""
    model_dir = rec.get("model_dir") or ""
    cfg = os.path.join(model_dir, "training_config.json")
    if model_dir and os.path.isfile(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                cats = json.load(f).get("categories") or {}
            if cats:
                return cats
        except Exception:
            pass
    if folder_csv and os.path.isfile(folder_csv):
        import pandas as pd
        df = pd.read_csv(folder_csv, usecols=lambda c: c in ("label", "object_name"))
        if "object_name" in df:
            return {str(int(l)): str(n).rsplit("_", 1)[0]
                    for l, n in df.dropna().groupby("label")["object_name"].first().items()}
    return {}


def relink_settings(rec: Dict, overrides: Optional[Dict] = None):
    """MaskInferenceConfig for relinking a run: its detection settings, the
    linker's defaults, then ``overrides``."""
    from dataclasses import fields
    from .mask_tracker_inference import MaskInferenceConfig
    names = {f.name for f in fields(MaskInferenceConfig)}
    base = {k: v for k, v in (rec.get("settings") or {}).items()
            if k in names and k not in LINKER_KEYS}
    base.update({k: v for k, v in (overrides or {}).items() if k in names})
    return MaskInferenceConfig(**base)


def _detections(folder: str, store: Optional[str]):
    """Every saved detection: a dict of arrays (frame, centroid, bbox, score,
    label, area, shape fields) plus a per-row crop getter. From the mask store,
    else trajectories.csv (boxes only)."""
    from .mask_tracker_inference import SHAPE_FIELDS
    if store:
        from .track_store import open_h5
        with open_h5(store, "r") as f:
            a = dict(f.attrs)
            d = {k: f[k][:] for k in ("frame", "label", "score", "area", "bbox", "centroid",
                                      "rle_start", "rle_len", "object_id")}
            d["shape"] = {n: f[n][:] for n in SHAPE_FIELDS if n in f}
            d["counts"] = f["rle_counts"][:] if bool(a.get("has_masks", True)) else None
        d["attrs"] = a
        return d
    import pandas as pd
    tj = pd.read_csv(os.path.join(folder, "trajectories.csv")).sort_values(["frame", "object_id"])
    d = {"frame": tj["frame"].to_numpy(int), "label": tj["label"].to_numpy(int),
         "score": tj["confidence"].to_numpy(float), "area": tj["mask_area"].to_numpy(int),
         "bbox": tj[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].to_numpy(float),
         "centroid": tj[["x", "y"]].to_numpy(float), "object_id": tj["object_id"].to_numpy(int),
         "shape": {n: tj[n].to_numpy(float) for n in SHAPE_FIELDS if n in tj},
         "counts": None, "attrs": {}, "table": tj}
    return d


def relink_folder(folder: str, overrides: Optional[Dict] = None, tracked_video: bool = False,
                  log: Callable[[str], None] = print,
                  progress: Optional[Callable[[int, int], None]] = None) -> Dict:
    """Link one ``<video>_MaskTracker`` folder again (see the module doc).
    Returns {"folder", "aside", "tracks", "singletons", "detections", "linker"}."""
    from . import mtt_runs
    from .mask_tracker_inference import MultiObjectTracker, SHAPE_FIELDS
    folder = os.path.normpath(folder)
    t0 = time.time()
    lines: List[str] = []

    def say(msg):
        lines.append(f"{datetime.now():%H:%M:%S} {msg}")
        log(msg)

    rec: Dict = {}
    if os.path.isfile(os.path.join(folder, mtt_runs.SETTINGS_FILE)):
        rec = mtt_runs.read_run_settings(folder)
    store = _store_in(folder)
    old_csv = os.path.join(folder, "trajectories.csv")
    if not store and not os.path.isfile(old_csv):
        raise FileNotFoundError(f"{folder}: no mask store or trajectories.csv to link")
    config = relink_settings(rec, overrides)
    cats = _categories(rec, old_csv)
    d = _detections(folder, store)
    a = d["attrs"]
    H = int(a.get("height", 0) or rec.get("height", 0) or 0)
    W = int(a.get("width", 0) or rec.get("width", 0) or 0)
    fps = float(a.get("fps", 0) or rec.get("fps", 0) or 30.0)
    n_frames = int(a.get("frame_count", 0) or rec.get("frames", 0) or (int(d["frame"].max()) + 1
                                                                        if len(d["frame"]) else 0))
    step = max(1, int(config.frame_step or 1))
    n = len(d["frame"])
    say(f"[relink] {os.path.basename(folder)}: {n} detections, {n_frames} frames, every {step}, "
        f"linker {config.linker}")

    tracker = MultiObjectTracker(config)
    want_crop = tracker.linker == "bytetrack" and d["counts"] is not None and H > 0
    new_ids = np.zeros(n, dtype=np.int64)
    frames = d["frame"].astype(np.int64)
    order = np.argsort(frames, kind="stable")
    bounds = np.searchsorted(frames[order], np.arange(0, n_frames + step, step))
    frame_hw = (H, W) if H and W else None
    total_steps = (n_frames + step - 1) // step
    for si, f in enumerate(range(0, n_frames, step)):
        rows = order[bounds[si]:bounds[si + 1]] if si + 1 < len(bounds) else order[bounds[si]:]
        rows = [int(r) for r in rows if frames[r] == f]
        dets = []
        for r in rows:
            crop = None
            if want_crop and int(d["rle_len"][r]) > 0:
                s0 = int(d["rle_start"][r])
                crop = decode_crop(d["counts"][s0:s0 + int(d["rle_len"][r])], H)
            dets.append({
                "row": r, "centroid": (float(d["centroid"][r][0]), float(d["centroid"][r][1])),
                "bbox": [float(v) for v in d["bbox"][r]], "mask": None,
                "score": float(d["score"][r]), "label": int(d["label"][r]), "area": int(d["area"][r]),
                "shape": {k: (None if not np.isfinite(v[r]) else float(v[r]))
                          for k, v in d["shape"].items()},
                "crop": crop,
            })
        matched = tracker.update_dets(dets, f, fps, frame_hw)
        for oid, det in matched.items():
            new_ids[det["row"]] = oid
        if progress and (si % 500 == 0 or si + 1 == total_steps):
            progress(si + 1, total_steps)
    # detections on frames off the analysed grid (should not happen) keep no id
    resolved = tracker.resolved_settings()
    kept = new_ids > 0
    n_tracks = len(np.unique(new_ids[kept & (new_ids < 1_000_001)]))
    say(f"[relink] {n_tracks} tracks, {tracker.n_singletons} low-confidence detections on their own, "
        f"{int((~kept).sum())} left out; {resolved}")

    aside = mtt_runs.move_tracking_aside_dir(folder)
    say(f"[relink] old outputs -> {aside}")
    import pandas as pd
    from .mask_tracker_inference import add_names_and_velocities
    if store:
        new_store = os.path.join(folder, os.path.basename(store))
        shutil.copy2(os.path.join(aside, os.path.basename(store)), new_store)
        from .track_store import TrackMaskReader, open_h5, trajectories_dataframe
        with open_h5(new_store, "a") as f:
            f["object_id"][:] = new_ids.astype(np.int32)
            f.attrs["relinked"] = datetime.now().isoformat(timespec="seconds")
            f.attrs["linker"] = json.dumps(resolved)
            for k in ("proofread", "proofread_at"):
                if k in f.attrs:
                    del f.attrs[k]
        with TrackMaskReader(new_store) as rd:
            df = trajectories_dataframe(rd, categories=cats or None)
    else:
        tj = d["table"].copy()
        tj["object_id"] = new_ids
        tj = tj[tj["object_id"] > 0].drop(columns=[c for c in ("object_name", "vx", "vy", "speed")
                                                   if c in tj])
        tj = tj.sort_values(["object_id", "frame"]).reset_index(drop=True)
        add_names_and_velocities(tj, cats or None)
        order_cols = ["frame", "time_s", "object_id", "object_name", "label", "x", "y", "bbox_x1",
                      "bbox_y1", "bbox_x2", "bbox_y2", "mask_area", *SHAPE_FIELDS, "confidence",
                      "vx", "vy", "speed"]
        df = tj[[c for c in order_cols if c in tj.columns]]
    df.to_csv(os.path.join(folder, "trajectories.csv"), index=False)
    say(f"[relink] trajectories.csv: {len(df)} rows, {df['object_id'].nunique() if len(df) else 0} ids")

    video_out = None
    video = rec.get("video") or str(a.get("video_path", ""))
    if tracked_video:
        if store and video and os.path.isfile(video):
            video_out = _draw_video(video, os.path.join(folder, os.path.basename(store)), cats, step,
                                    folder, say)
        else:
            say("[relink] no tracked video: the source video or mask store is missing")

    from dataclasses import asdict
    new_rec = dict(rec)
    new_rec.update(settings=asdict(config), linker_resolved=resolved, tracks=n_tracks,
                   create_tracked_video=bool(video_out), complete=True,
                   relinked={"at": datetime.now().isoformat(timespec="seconds"), "aside": aside,
                             "seconds": round(time.time() - t0, 1),
                             "previous_settings": rec.get("settings")})
    old_log = os.path.join(aside, mtt_runs.LOG_FILE) if aside else ""
    if old_log and os.path.isfile(old_log):
        with open(old_log, encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh] + ["", "--- relinked ---"] + lines
    mtt_runs.write_run_record(folder, new_rec, lines)
    return {"folder": folder, "aside": aside, "tracks": n_tracks, "singletons": tracker.n_singletons,
            "detections": n, "linker": resolved, "tracked_video": video_out}


def _draw_video(video: str, store: str, cats: Dict, step: int, folder: str, say) -> Optional[str]:
    """The tracked video again, from the source video and the relinked mask store."""
    import cv2
    from pathlib import Path
    from .mask_tracker_inference import _draw_annotations
    from .track_store import TrackMaskReader
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        say(f"[relink] cannot open {video}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = os.path.join(folder, f"{Path(video).stem}_annotated.mp4")
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (w, h))
    with TrackMaskReader(store) as rd:
        i = 0
        while True:
            if step > 1 and i % step:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            matched = {}
            for det in rd.detections_for_frame(i):
                if det["object_id"] > 0:
                    det["centroid"] = tuple(det["centroid"])
                    matched[det["object_id"]] = det
            writer.write(_draw_annotations(frame, matched, cats))
            i += 1
    cap.release()
    writer.release()
    say(f"[relink] tracked video: {out}")
    return out


def run_relink(folders: List[str], overrides: Optional[Dict] = None, tracked_video: bool = False,
               window: bool = True) -> int:
    """Relink folders one after another; with ``window``, in a progress window
    that closes itself when every folder succeeded (it stays up on a failure)."""
    if not window:
        bad = 0
        for fo in folders:
            try:
                relink_folder(fo, overrides, tracked_video)
            except Exception as e:
                bad += 1
                print(f"[relink] FAILED {fo}: {e}", flush=True)
        return 1 if bad else 0
    return _run_window(folders, overrides, tracked_video)


def _run_window(folders, overrides, tracked_video) -> int:
    import sys
    from PyQt5.QtCore import QThread, QTimer, pyqtSignal
    from PyQt5.QtWidgets import (QApplication, QLabel, QProgressBar, QPlainTextEdit, QVBoxLayout,
                                 QWidget)

    class Worker(QThread):
        line = pyqtSignal(str)
        step = pyqtSignal(int, int, int)            # folder index, done, total
        done = pyqtSignal(int)                      # failures

        def run(self):
            bad = 0
            for k, fo in enumerate(folders):
                try:
                    relink_folder(fo, overrides, tracked_video, log=self.line.emit,
                                  progress=lambda a, b, k=k: self.step.emit(k, a, b))
                except Exception as e:                       # shown, and the window stays up
                    bad += 1
                    self.line.emit(f"[relink] FAILED {fo}: {e}")
            self.done.emit(bad)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = QWidget()
    win.setWindowTitle(f"Mask Tracker relink — {len(folders)} folder(s)")
    lay = QVBoxLayout(win)
    head = QLabel(f"0 of {len(folders)} folders")
    bar = QProgressBar()
    bar.setRange(0, 1000)
    text = QPlainTextEdit()
    text.setReadOnly(True)
    for wdg in (head, bar, text):
        lay.addWidget(wdg)
    win.resize(900, 420)
    worker = Worker()
    worker.line.connect(text.appendPlainText)

    def on_step(k, a, b):
        head.setText(f"{k} of {len(folders)} folders done · {os.path.basename(folders[k])}")
        bar.setValue(int(1000 * (k + a / max(1, b)) / len(folders)))

    result = {"bad": 0}

    def on_done(bad):
        result["bad"] = bad
        bar.setValue(1000)
        if bad:
            head.setText(f"{len(folders) - bad} of {len(folders)} relinked · {bad} FAILED (see below)")
        else:
            head.setText(f"all {len(folders)} relinked · closing")
            QTimer.singleShot(3000, win.close)

    worker.step.connect(on_step)
    worker.done.connect(on_done)
    win.show()
    worker.start()
    app.exec_()
    worker.wait()
    return 1 if result["bad"] else 0
