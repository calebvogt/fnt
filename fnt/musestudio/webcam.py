"""Webcam capture for MuseStudio: live preview plus recording synchronized
with the Muse data.

Synchronization: each frame is stamped with LSL ``local_clock()`` — the same
time base the Muse LSL inlets use (OpenMuse's streamer runs on the same host, so
the clock is shared across processes). Frame timestamps are written to
``webcam_timestamps.csv`` (frame_index, lsl_timestamp) alongside the video, so
frames can be aligned to the EEG/OPTICS CSVs after the fact. The .mp4 itself is
written at a nominal fixed FPS; the CSV holds the true per-frame timing.
"""

import csv
import os
import threading
import time

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


def list_cameras(max_index=6):
    """Probe camera indices and return those that open (contiguous from 0).

    Opening a device briefly activates it (and may trigger the OS camera
    permission prompt on first use), so this is best called on demand.
    """
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        opened = cap.isOpened()
        cap.release()
        if opened:
            found.append(i)
        else:
            break   # indices are contiguous; stop at the first gap
    return found


_local_clock = None


def _lsl_clock():
    """Return the LSL local clock (shared with Muse timestamps)."""
    global _local_clock
    if _local_clock is None:
        from mne_lsl.lsl import local_clock
        _local_clock = local_clock
    return _local_clock()


class WebcamThread(QThread):
    """Captures frames for live preview and (optionally) records them."""

    frame_ready = pyqtSignal(QImage)
    opened = pyqtSignal(int, int, float)   # width, height, fps
    error = pyqtSignal(str)

    def __init__(self, camera_index=0, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False
        self._lock = threading.Lock()
        self._writer = None
        self._ts_file = None
        self._ts_writer = None
        self._frame_idx = 0
        self._fps = 30.0
        self._size = (0, 0)
        self._pending_dir = None      # record was requested before size was known

    # --- recording control (called from GUI thread) ---
    def start_recording(self, session_dir):
        """Request recording into ``session_dir``. Opens the writer immediately
        if the frame size is known, otherwise defers to the first frame (so a
        session started before the camera warms up still records). Returns True
        if recording is armed."""
        with self._lock:
            if self._size != (0, 0):
                self._open_writer_locked(session_dir)
            else:
                self._pending_dir = session_dir
        return True

    def _open_writer_locked(self, session_dir):
        video_path = os.path.join(session_dir, "webcam.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(video_path, fourcc, self._fps, self._size)
        self._ts_file = open(
            os.path.join(session_dir, "webcam_timestamps.csv"), "w", newline=""
        )
        self._ts_writer = csv.writer(self._ts_file)
        self._ts_writer.writerow(["frame_index", "lsl_timestamp"])
        self._frame_idx = 0
        self._pending_dir = None

    def stop_recording(self):
        with self._lock:
            frames = self._frame_idx
            self._pending_dir = None
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

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"Could not open camera {self.camera_index}.")
            return
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            self._fps = fps if fps and fps > 1 else 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._size = (w, h)
            self.opened.emit(w, h, self._fps)

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                ts = _lsl_clock()
                with self._lock:
                    # Open a deferred writer now that the frame size is known.
                    if self._writer is None and self._pending_dir is not None:
                        self._open_writer_locked(self._pending_dir)
                    if self._writer is not None:
                        # Guard against a late size change (some webcams warm up).
                        if (frame.shape[1], frame.shape[0]) != self._size:
                            frame = cv2.resize(frame, self._size)
                        self._writer.write(frame)
                        self._ts_writer.writerow([self._frame_idx, f"{ts:.6f}"])
                        self._frame_idx += 1
                self.frame_ready.emit(_to_qimage(frame))
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.stop_recording()
            cap.release()


def _to_qimage(frame_bgr):
    """Convert an OpenCV BGR frame to a standalone QImage (copied, thread-safe)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return img.copy()
