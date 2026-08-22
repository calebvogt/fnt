"""Recording layout and action logging for MuseStudio.

Each recording gets a timestamped parent folder:

    YYYY-MM-DD_HHMMSS_FNT_MuseStudio_recording/
        recording_config.json     # settings + metadata snapshot
        session_logs.txt          # every GUI action, timestamped to the ms
        Data/
            Muse/                 # raw device streams (EEG, optics, accel/gyro, battery)
            Video/                # webcam.mp4, webcam_timestamps.csv
            Audio/                # audio_events.csv (binaural stimulus log)
            Analysis/             # synchrony.csv (derived PLV metrics)
            Events/               # events.csv — protocol phases on the LSL clock

``session_logs.txt`` is the human-readable narrative (wall clock).
``Data/Events/events.csv`` is its machine-readable counterpart on the **LSL
clock**, which is what lets review segment the signals by protocol phase.

``SessionLogger`` buffers actions from the moment the window opens, so when a
recording starts the log already contains the lead-up (scan, connect, …).
"""

import json
import os
from collections import deque
from datetime import datetime

FOLDER_SUFFIX = "FNT_MuseStudio_recording"


class RecordingSession:
    """Creates and exposes the per-recording directory tree."""

    def __init__(self, base_dir, subject=""):
        """``subject`` is folded into the folder name when given, so a
        recording identifies its subject on the filesystem alone — no need to
        open a config file to know whose night or session this was."""
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.subject = subject or ""
        parts = [stamp] + ([self.subject] if self.subject else []) + [FOLDER_SUFFIX]
        self.name = "_".join(parts)
        self.root = os.path.join(base_dir, self.name)
        self.data_dir = os.path.join(self.root, "Data")
        self.muse_dir = os.path.join(self.data_dir, "Muse")
        self.video_dir = os.path.join(self.data_dir, "Video")
        self.audio_dir = os.path.join(self.data_dir, "Audio")
        self.analysis_dir = os.path.join(self.data_dir, "Analysis")
        self.events_dir = os.path.join(self.data_dir, "Events")
        for d in (self.root, self.data_dir, self.muse_dir, self.video_dir,
                  self.audio_dir, self.analysis_dir, self.events_dir):
            os.makedirs(d, exist_ok=True)
        self.config_path = os.path.join(self.root, "recording_config.json")
        self.log_path = os.path.join(self.root, "session_logs.txt")

    def write_config(self, config):
        """Write (or overwrite) recording_config.json."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)


class SessionLogger:
    """Timestamped GUI-action log with a pre-recording in-memory buffer."""

    # Bounded so an all-night recording can't grow this without limit. Once the
    # file is open every line goes straight to disk anyway; the buffer only
    # exists to hold the pre-recording lead-up.
    MAX_BUFFERED = 5000

    def __init__(self):
        self._events = deque(maxlen=self.MAX_BUFFERED)
        self._file = None

    def log(self, text):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # ms
        line = f"{stamp}  {text}"
        self._events.append(line)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()

    def start_file(self, path):
        """Open the log file and flush everything buffered so far."""
        self._file = open(path, "w", encoding="utf-8")
        if self._events:
            self._file.write("\n".join(self._events) + "\n")
            self._file.flush()

    def stop_file(self):
        if self._file is not None:
            self._file.close()
            self._file = None
