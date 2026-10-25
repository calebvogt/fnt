"""Video QC: the video filmed alongside a stretch of a recording.

When a spectrogram alone cannot say what made a sound (a running wheel, a
chewing mouse, a call), the camera that watched the microphone can. A MAD
project names that video in ``<project_dir>/video_qc.json``. It is a file of
its own rather than a field of ``mad_project_info.json`` so that an older FNT,
which rewrites the project file with only the fields it knows, cannot drop it.

The file says how to get from a recording to its video::

    {
      "trial_regex":   "/(T\\d{3})_",          # trial id, from the recording's path
      "channel_regex": "_(ch\\d)",             # channel id, from its file name
      "audio_start": [                         # its wall-clock start; first rule that works
        {"regex": "T(\\d{4}-\\d{2}-\\d{2}_\\d{2}-\\d{2}-\\d{2})_",
         "format": "%Y-%m-%d_%H-%M-%S"},
        {"avisoft_log": true}                  # Avisoft's .LOG beside sequence-numbered files
      ],
      "cameras": {"ch1": "CAM16", "ch2": "CAM15", "ch3": "CAM13", "ch4": "CAM14"},
      "sources": [                             # where each camera's files are
        {"root": "Z:/.../video_ResourceZones", "cameras": ["CAM13", "CAM14", "CAM15", "CAM16"]},
        {"root": "Z:/.../video_Overhead", "cameras": ["CAM12"]}
      ],
      "trial_folder": "{trial}_*",             # a trial's folder under a root
      "video_file": "{camera}_*.mp4",          # a camera's files in it
      "video_start": {"regex": "_(\\d{4}-\\d{2}-\\d{2}_\\d{2}_\\d{2}_\\d{2}_\\d{3})\\.mp4$",
                      "format": "%Y-%m-%d_%H_%M_%S_%f"},
      "offsets_s": {"T006": 0.0}               # per trial: video clock minus audio clock
    }

A moment ``t`` seconds into a recording is the wall-clock time ``start + t``,
plus the trial's offset on the video clock; the frame shown is the one of the
camera's file that covers that time. Both clocks come from file names (or the
Avisoft log), so the match is as good as the two clocks were: check the time a
camera burns into its frames against the one shown with the clip.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

VIDEO_QC_FILENAME = 'video_qc.json'


@dataclass
class VideoQCConfig:
    """A project's video source; see the module docstring."""
    trial_regex: str = r"/(T\d{3})_"
    channel_regex: str = r"_(ch\d)"
    audio_start: List[Dict] = field(default_factory=lambda: [{"avisoft_log": True}])
    cameras: Dict[str, str] = field(default_factory=dict)
    sources: List[Dict] = field(default_factory=list)
    trial_folder: str = "{trial}_*"
    video_file: str = "{camera}_*.mp4"
    video_start: Dict = field(default_factory=dict)
    offsets_s: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict) -> 'VideoQCConfig':
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, project_dir: Optional[str]) -> Optional['VideoQCConfig']:
        """The project's video source, or None if it names none."""
        if not project_dir:
            return None
        path = os.path.join(project_dir, VIDEO_QC_FILENAME)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def all_cameras(self) -> List[str]:
        """Every camera the sources name, in their order."""
        out: List[str] = []
        for s in self.sources:
            for c in s.get('cameras', []):
                if c not in out:
                    out.append(c)
        return out

    def root_of(self, camera: str) -> Optional[str]:
        for s in self.sources:
            if camera in s.get('cameras', []):
                return s.get('root')
        return None


def _slash(path: str) -> str:
    return str(path).replace('\\', '/')


def recording_ids(wav: str, cfg: VideoQCConfig) -> Tuple[Optional[str], Optional[str]]:
    """(trial, channel) of a recording, from its path and file name."""
    m = re.search(cfg.trial_regex, _slash(wav))
    trial = m.group(1) if m else None
    m = re.search(cfg.channel_regex, os.path.basename(wav))
    if m is None:       # channel only in the folder name ("ch1_z4/")
        m = re.search(cfg.channel_regex, _slash(os.path.dirname(wav)) + '_')
    channel = m.group(1) if m else None
    return trial, channel


def cameras_for(wav: str, cfg: VideoQCConfig) -> List[str]:
    """The cameras a recording can be checked against: its channel's camera
    first, then every other camera of the sources."""
    _, channel = recording_ids(wav, cfg)
    own = cfg.cameras.get(channel) if channel else None
    rest = [c for c in cfg.all_cameras() if c != own]
    return ([own] if own else []) + rest


def _avisoft_start(wav: str) -> Optional[datetime]:
    """Start of a sequence-numbered Avisoft file (``..._T0000042.wav``) from the
    channel folder's .LOG: date, time and file name, tab-separated."""
    m = re.search(r"_(T\d+\.wav)$", os.path.basename(wav), re.IGNORECASE)
    if not m:
        return None
    seq = m.group(1).lower()
    for log in sorted(glob.glob(os.path.join(os.path.dirname(wav), "*.LOG"))):
        try:
            with open(log, errors='ignore') as f:
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) >= 3 and parts[2].lower() == seq:
                        return datetime.strptime(f"{parts[0]} {parts[1]}", "%m/%d/%y %H:%M:%S")
        except (OSError, ValueError):
            continue
    return None


def recording_start(wav: str, cfg: VideoQCConfig) -> Optional[datetime]:
    """Wall-clock start of a recording: the first ``audio_start`` rule that
    gives one."""
    name = os.path.basename(wav)
    for rule in cfg.audio_start:
        if rule.get('avisoft_log'):
            t = _avisoft_start(wav)
            if t is not None:
                return t
        elif rule.get('regex'):
            m = re.search(rule['regex'], name)
            if m:
                try:
                    return datetime.strptime(m.group(1), rule['format'])
                except ValueError:
                    continue
    return None


def _file_start(path: str, cfg: VideoQCConfig) -> Optional[datetime]:
    m = re.search(cfg.video_start.get('regex', r"$^"), os.path.basename(path))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), cfg.video_start['format'])
    except (ValueError, KeyError):
        return None


@lru_cache(maxsize=64)
def _camera_files(folder: str, pattern: str) -> Tuple[str, ...]:
    """A camera's files in one trial folder (listing a network folder of ~900
    files takes a moment, so each is listed once per session)."""
    return tuple(sorted(glob.glob(os.path.join(folder, pattern))))


@lru_cache(maxsize=4096)
def _video_info(path: str) -> Tuple[float, int]:
    """(fps, frame count) of a video file."""
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return float(fps), n


def video_files(trial: str, camera: str, cfg: VideoQCConfig) -> List[Tuple[datetime, str]]:
    """(start, path) of every file of one camera in one trial, by start."""
    root = cfg.root_of(camera)
    if not root or not trial:
        return []
    out = []
    for folder in sorted(glob.glob(os.path.join(root, cfg.trial_folder.format(trial=trial)))):
        if not os.path.isdir(folder):
            continue
        for p in _camera_files(folder, cfg.video_file.format(camera=camera)):
            t = _file_start(p, cfg)
            if t is not None:
                out.append((t, p))
    out.sort()
    return out


@dataclass
class VideoSpan:
    """Frames ``f0``..``f1`` (inclusive) of one file; frame ``f0`` is at
    ``t0_s`` seconds into the recording."""
    path: str
    camera: str
    file_start: datetime
    fps: float
    f0: int
    f1: int
    t0_s: float

    @property
    def n_frames(self) -> int:
        return self.f1 - self.f0 + 1


@dataclass
class VideoClip:
    """Frames covering a stretch of a recording, and the time of each."""
    camera: str
    times_s: np.ndarray                      # seconds into the recording
    frames: List[np.ndarray]                 # greyscale, uint8
    spans: List[VideoSpan]
    rec_start: datetime
    offset_s: float = 0.0

    def index_at(self, t_s: float) -> int:
        """The frame showing ``t_s`` seconds into the recording (the last one
        that started at or before it)."""
        if not len(self.times_s):
            return -1
        i = int(np.searchsorted(self.times_s, t_s, side='right')) - 1
        return min(max(i, 0), len(self.times_s) - 1)

    def frame_info(self, i: int) -> Tuple[str, int, datetime]:
        """(file, frame number in it, its time on the video clock) of frame i."""
        t = float(self.times_s[i])
        for s in self.spans:
            k = int(round((t - s.t0_s) * s.fps)) + s.f0
            if s.f0 <= k <= s.f1:
                return (os.path.basename(s.path), k,
                        s.file_start + timedelta(seconds=k / s.fps))
        s = self.spans[-1]
        return os.path.basename(s.path), s.f1, s.file_start + timedelta(seconds=s.f1 / s.fps)


class VideoQCError(RuntimeError):
    """Why no video could be found (shown to the user as is)."""


def locate(wav: str, t0_s: float, t1_s: float, camera: str,
           cfg: VideoQCConfig) -> Tuple[List[VideoSpan], datetime, float]:
    """The frames of ``camera`` that cover ``t0_s``..``t1_s`` of a recording:
    (spans, the recording's wall-clock start, the trial's clock offset).
    A stretch that crosses from one file into the next gives two spans."""
    trial, _ = recording_ids(wav, cfg)
    if trial is None:
        raise VideoQCError(f"No trial id in the recording's path ({cfg.trial_regex!r}).")
    start = recording_start(wav, cfg)
    if start is None:
        raise VideoQCError(f"No start time for {os.path.basename(wav)}: not in its "
                           "name, and no Avisoft log entry for it.")
    offset = float(cfg.offsets_s.get(trial, 0.0))
    files = video_files(trial, camera, cfg)
    if not files:
        raise VideoQCError(f"No {camera} video for {trial} under "
                           f"{cfg.root_of(camera) or '(no source for this camera)'}.")
    w0 = start + timedelta(seconds=t0_s + offset)
    w1 = start + timedelta(seconds=t1_s + offset)
    spans: List[VideoSpan] = []
    for i, (fs, path) in enumerate(files):
        nxt = files[i + 1][0] if i + 1 < len(files) else None
        if nxt is not None and nxt <= w0:
            continue                       # ends before the stretch starts
        if fs > w1:
            break
        fps, n = _video_info(path)
        if fps <= 0 or n <= 0:
            continue
        fe = fs + timedelta(seconds=n / fps)
        if fe <= w0:
            continue                       # a gap in the recording
        a = max(0, int(math.floor((w0 - fs).total_seconds() * fps)))
        b = min(n - 1, int(math.ceil((w1 - fs).total_seconds() * fps)))
        if b < a:
            continue
        t_a = (fs + timedelta(seconds=a / fps) - start).total_seconds() - offset
        spans.append(VideoSpan(path, camera, fs, fps, a, b, t_a))
    if not spans:
        raise VideoQCError(f"{camera} has no video at {w0:%Y-%m-%d %H:%M:%S} "
                           f"({trial}): the camera was not recording then.")
    return spans, start, offset


def read_clip(wav: str, t0_s: float, t1_s: float, camera: str, cfg: VideoQCConfig,
              width: int = 640, max_frames: int = 900,
              progress: Optional[Callable[[int, int], None]] = None,
              cancelled: Optional[Callable[[], bool]] = None) -> VideoClip:
    """Decode the frames covering ``t0_s``..``t1_s`` of a recording, scaled to
    ``width`` px and greyscale. A stretch longer than ``max_frames`` frames
    keeps every k-th frame, so a zoomed-out view stays quick and small."""
    import cv2
    spans, start, offset = locate(wav, t0_s, t1_s, camera, cfg)
    total = sum(s.n_frames for s in spans)
    step = max(1, int(math.ceil(total / max(1, max_frames))))
    times: List[float] = []
    frames: List[np.ndarray] = []
    done = 0
    for s in spans:
        cap = cv2.VideoCapture(s.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s.f0)
            for k in range(s.f0, s.f1 + 1):
                if cancelled is not None and cancelled():
                    raise VideoQCError("Cancelled.")
                if (k - s.f0) % step:
                    ok = cap.grab()          # skip without decoding to an image
                else:
                    ok, fr = cap.read()
                    if ok:
                        if fr.ndim == 3:
                            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                        h = int(round(fr.shape[0] * width / fr.shape[1]))
                        frames.append(cv2.resize(fr, (width, h), interpolation=cv2.INTER_AREA))
                        times.append(s.t0_s + (k - s.f0) / s.fps)
                if not ok:
                    break
                done += 1
                if progress is not None and done % 15 == 0:
                    progress(done, total)
        finally:
            cap.release()
    if not frames:
        raise VideoQCError(f"Could not decode {camera}'s frames ({os.path.basename(spans[0].path)}).")
    return VideoClip(camera, np.asarray(times), frames, spans, start, offset)
