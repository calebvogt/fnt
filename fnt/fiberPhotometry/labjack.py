"""Reading LabJack LJStreamUD .dat recordings (the Warden lab rig, Caleb Vogt's 2019-2020 PhD data).

LJStreamUD writes tab-separated text: a header ``Time v0 v1 ... v13`` and one row per scan (250 Hz
on that rig). The lab's MATLAB (``ReferenceMaterials/PhotometryBehaviorCombinedVideo.m``) reads
``v0`` as the 473 nm signal, ``v1`` as the 405 nm isosbestic control and ``v2`` as the camera's
frame TTL. That TTL is the frame clock: each frame is marked where ``v2`` drops below 1 V. The
recording gives one hardware time per camera frame, so the video is timed by those marks rather
than by its nominal frame rate. That matters on this rig because the camera started a few seconds
after the photometry and ran at about 30.07 fps against the file's 30.

Returns the same ``DoricFileData`` container as ``doric.py``, so everything downstream is shared.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .doric import DoricChannelInfo, DoricFileData, DoricVideoInfo

SIGNAL = "v0"          # 473 nm
CONTROL = "v1"         # 405 nm
CAMERA_TTL = "v2"
TTL_THRESHOLD_V = 1.0
WAVELENGTHS = {SIGNAL: 473, CONTROL: 405}
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".wmv", ".mkv")


def is_labjack_dat(path) -> bool:
    """True when ``path`` is a .dat file whose first line is LJStreamUD's ``Time v0 v1 ...`` header."""
    p = Path(path)
    if p.suffix.lower() != ".dat" or not p.is_file():
        return False
    with open(p, "r", errors="replace") as f:
        cols = f.readline().strip().split("\t")
    return len(cols) >= 3 and cols[0] == "Time" and cols[1] == "v0"


def ttl_frame_samples(ttl: np.ndarray, threshold: float = TTL_THRESHOLD_V) -> np.ndarray:
    """Sample index (0-based) of each camera frame: the last sample before ``ttl`` drops below
    ``threshold``. That is the sample MATLAB's ``find(diff(ttl < 1) == 1)`` marks."""
    low = np.asarray(ttl) < threshold
    return np.flatnonzero(np.diff(low.astype(np.int8)) == 1)


def frame_count(video) -> int:
    import cv2
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return n


class LabJackFileReader:
    """Reader for LJStreamUD .dat recordings, with the same interface as ``DoricFileReader``."""

    def __init__(self, filepath, video: Optional[str] = None):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        if not is_labjack_dat(self.filepath):
            raise ValueError(f"not a LabJack LJStreamUD .dat file: {self.filepath.name}")
        self.video = video
        self._table: Optional[pd.DataFrame] = None

    def _read(self) -> pd.DataFrame:
        if self._table is None:
            self._table = pd.read_csv(self.filepath, sep="\t", dtype=np.float64)
        return self._table

    def scan_file(self) -> DoricFileData:
        d = self._read()
        t = d["Time"].to_numpy()
        fs = float(1 / np.median(np.diff(t))) if len(t) > 1 else 0.0
        chans = []
        for c in d.columns[1:]:
            wl = WAVELENGTHS.get(c)
            name = f"{c} (camera TTL)" if c == CAMERA_TTL else c
            chans.append(DoricChannelInfo(path=c, time_path="Time", wavelength=wl, name=name,
                                          sampling_rate=fs, n_samples=len(t)))
        data = DoricFileData(filepath=self.filepath, all_channels=chans)
        data.signal_channel = next((c for c in chans if c.path == SIGNAL), None)
        data.isosbestic_channel = next((c for c in chans if c.path == CONTROL), None)
        if CAMERA_TTL in d.columns:
            marks = ttl_frame_samples(d[CAMERA_TTL].to_numpy())
            video = self.video or self._pair_video(len(marks))
            if video and len(marks):
                data.video_info = DoricVideoInfo(time_path=CAMERA_TTL, relative_path=Path(video).name,
                                                 n_frames=len(marks), absolute_path=str(video))
                span = t[marks[-1]] - t[marks[0]]
                data.video_info.fps = (len(marks) - 1) / span if span > 0 else 0.0
        return data

    def load_data(self, data: DoricFileData) -> DoricFileData:
        d = self._read()
        t = d["Time"].to_numpy()
        if data.signal_channel:
            data.signal_data, data.signal_time = d[data.signal_channel.path].to_numpy(), t
        if data.isosbestic_channel:
            data.isosbestic_data, data.isosbestic_time = d[data.isosbestic_channel.path].to_numpy(), t
        if data.video_info and CAMERA_TTL in d.columns:
            data.video_timestamps = t[ttl_frame_samples(d[CAMERA_TTL].to_numpy())]
            data.video_frame_clock = True
        return data

    def _pair_video(self, n_marks: int) -> Optional[str]:
        """The video beside the .dat whose frame count is closest to its TTL count (the only one,
        if there is one). Subfolders such as ``processing/`` are not searched."""
        vids: List[Path] = sorted(p for p in self.filepath.parent.iterdir()
                                  if p.suffix.lower() in VIDEO_SUFFIXES and p.is_file())
        if len(vids) <= 1:
            return str(vids[0]) if vids else None
        return str(min(vids, key=lambda v: abs(frame_count(v) - n_marks)))
