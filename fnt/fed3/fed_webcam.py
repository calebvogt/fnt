"""Webcam / USB camera capture for the FED3 tab.

Adapted from :mod:`fnt.musestudio.webcam`. The difference is the time base:
MuseStudio stamps frames with the LSL clock because that is what its EEG inlets
use. FED3 has no LSL, so frames are stamped with
:func:`fnt.fed3.fed_session.host_now` — the same host wall clock used for
behavioural events and interaction logs. A pellet at ``host_time`` and a frame
at ``host_time`` are directly comparable with no alignment step.

The .mp4 is written at a nominal fixed FPS; ``<camera>_frames.csv`` holds the
true per-frame timing, which is what should be used for analysis. Webcam frame
intervals are not reliably uniform, so trusting the container's frame rate would
introduce drift over a long session.
"""

import csv
import os
import threading
import time

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from .fed_session import host_now

FRAME_FIELDS = ["frame_index", "host_time", "host_monotonic"]


def list_cameras(max_index=6):
    """Camera indices that open, scanning contiguously from 0.

    Opening a device briefly activates it and can trigger the OS camera
    permission prompt, so call this on demand rather than at import.
    """
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        opened = cap.isOpened()
        cap.release()
        if opened:
            found.append(i)
        else:
            break        # indices are contiguous; stop at the first gap
    return found


class WebcamRecorder(QThread):
    """Captures frames from one camera for live preview and recording.

    Recording is armed independently of capture: the preview can run while
    idle, and :meth:`start_recording` begins writing without restarting the
    device (which would cost several seconds of warm-up).
    """

    frame_ready = pyqtSignal(QImage)
    opened = pyqtSignal(int, int, float)     # width, height, fps
    error = pyqtSignal(str)
    recording_started = pyqtSignal(str)      # video path

    def __init__(self, camera_index=0, label=None, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.label = label or f"camera{camera_index}"
        self._running = False
        self._lock = threading.Lock()
        self._writer = None
        self._ts_file = None
        self._ts_writer = None
        self._frame_idx = 0
        self._fps = 30.0
        self._size = (0, 0)
        self._pending = None        # (video_path, frames_path) armed before size known
        self._dropped = 0

    # --- recording control (GUI thread) ----------------------------------

    def start_recording(self, video_path, frames_path):
        """Arm recording. Opens the writer now if the frame size is known,
        otherwise on the first frame, so a session started while the camera is
        still warming up still captures from its first available frame."""
        with self._lock:
            if self._size != (0, 0):
                self._open_writer_locked(video_path, frames_path)
            else:
                self._pending = (video_path, frames_path)
        return True

    def stop_recording(self):
        """Close the writer and return the number of frames recorded."""
        with self._lock:
            frames = self._frame_idx
            self._pending = None
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            if self._ts_file is not None:
                self._ts_file.close()
                self._ts_file = None
                self._ts_writer = None
        return frames

    def is_recording(self):
        with self._lock:
            return self._writer is not None

    def frame_count(self):
        with self._lock:
            return self._frame_idx

    def stop(self):
        self._running = False

    # --- capture thread ---------------------------------------------------

    def _open_writer_locked(self, video_path, frames_path):
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(video_path, fourcc, self._fps, self._size)
        self._ts_file = open(frames_path, "w", newline="", encoding="utf-8")
        self._ts_writer = csv.writer(self._ts_file)
        self._ts_writer.writerow(FRAME_FIELDS)
        self._frame_idx = 0
        self._pending = None
        self.recording_started.emit(video_path)

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"Could not open camera {self.camera_index}.")
            return
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            self._fps = fps if fps and fps > 1 else 30.0
            self._size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.opened.emit(self._size[0], self._size[1], self._fps)

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                # Stamp before any work so the timestamp reflects capture, not
                # encode time.
                host_ts = host_now()
                mono = time.monotonic()

                with self._lock:
                    if self._writer is None and self._pending is not None:
                        self._open_writer_locked(*self._pending)
                    if self._writer is not None:
                        # Some webcams change resolution as they warm up.
                        if (frame.shape[1], frame.shape[0]) != self._size:
                            frame = cv2.resize(frame, self._size)
                        self._writer.write(frame)
                        self._ts_writer.writerow(
                            [self._frame_idx, f"{host_ts:.6f}", f"{mono:.6f}"])
                        self._frame_idx += 1
                        if self._frame_idx % 300 == 0:
                            self._ts_file.flush()   # bound loss if we crash
                self.frame_ready.emit(_to_qimage(frame))
        except Exception as exc:  # noqa: BLE001 - surfaced to the GUI
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.stop_recording()
            cap.release()


def _to_qimage(frame_bgr):
    """Copy an OpenCV BGR frame into a standalone QImage safe to cross threads."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
