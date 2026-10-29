"""Qt side of tracking several videos at once (``mask_tracker_parallel``).

``ParallelInferenceWorker`` relays the worker processes' messages as signals;
it has the control surface of the one-at-a-time ``InferenceWorker``
(``request_pause/resume/stop``, ``_pause_flag``, ``video_finished``,
``all_done``, ``log``), so the Inference tab drives either. ``InferenceGrid``
is the tile view that takes the place of the preview pane while it runs: one
small tile per running video with its masks, progress and speed.
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QFrame, QGridLayout, QLabel, QProgressBar, QSizePolicy,
                             QVBoxLayout, QWidget)

from .mask_tracker_parallel import ParallelInference


class ParallelInferenceWorker(QThread):
    video_started = pyqtSignal(int, int, str)          # slot, video index, path
    video_progress = pyqtSignal(int, int, int, float)  # slot, frame, total, fps
    video_frame = pyqtSignal(int, object, int)         # slot, preview RGB, frame
    video_finished = pyqtSignal(object)                # result dict (as InferenceWorker)
    video_failed = pyqtSignal(int, str)                # slot, message
    worker_exited = pyqtSignal(int)                    # slot: no more videos for it
    all_done = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, video_paths: List[str], model_dir: str, config_dict: Dict,
                 n_workers: int, opts: Optional[Dict] = None):
        super().__init__()
        self.runner = ParallelInference(video_paths, model_dir, config_dict, n_workers, opts)
        self.n_workers = self.runner.n_workers
        self.fps: Dict[int, float] = {}
        self._stop_flag = False

    @property
    def _pause_flag(self) -> bool:
        return self.runner.paused

    def request_pause(self):
        self.runner.pause()

    def request_resume(self):
        self.runner.resume()

    def request_stop(self):
        self._stop_flag = True
        self.runner.stop()

    def total_fps(self) -> float:
        return sum(self.fps.values())

    def run(self):
        try:
            self.runner.start()
        except Exception as e:
            self.error.emit(f"Could not start the worker processes: {e}")
            return
        self.log.emit(f"[MTT Parallel] {len(self.runner.video_paths)} videos, "
                      f"{self.n_workers} at once")
        latest_frame: Dict[int, tuple] = {}
        t_last_frames = 0.0
        while not self.runner.done():
            for m in self.runner.messages(timeout=0.1):
                kind, wid = m[0], m[1]
                if kind == "log":
                    self.log.emit(f"[v{wid + 1}] {m[2]}")
                elif kind == "started":
                    self.fps[wid] = 0.0
                    self.video_started.emit(wid, m[2], m[3])
                elif kind == "progress":
                    self.fps[wid] = m[5]
                    self.video_progress.emit(wid, m[3], m[4], m[5])
                elif kind == "frame":
                    latest_frame[wid] = (m[3], m[4])   # only the newest is drawn
                elif kind == "finished":
                    self.fps.pop(wid, None)
                    m[3]["worker"] = wid
                    self.video_finished.emit(m[3])
                elif kind == "stopped":
                    self.fps.pop(wid, None)
                elif kind == "error":
                    self.fps.pop(wid, None)
                    self.video_failed.emit(wid, m[3])
                    self.log.emit(f"[v{wid + 1}] ERROR {m[3]}")
                elif kind == "exit":
                    self.fps.pop(wid, None)
                    self.worker_exited.emit(wid)
            now = time.time()
            if latest_frame and now - t_last_frames >= 0.1:
                for wid, (rgb, f) in latest_frame.items():
                    self.video_frame.emit(wid, rgb, f)
                latest_frame.clear()
                t_last_frames = now
        self.all_done.emit()

    def shutdown(self, timeout: float = 10.0):
        """Stop the workers (they remove their partial outputs) and wait."""
        self.request_stop()
        if self.isRunning():
            self.wait(int(timeout * 1000))
        self.runner.terminate(timeout=1.0)


class _Tile(QFrame):
    """One running video: name, the latest frame with its masks, progress."""

    def __init__(self, slot: int, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QFrame { background-color: #141414; border: 1px solid #333; "
                           "border-radius: 3px; }")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 3, 4, 3)
        lay.setSpacing(2)
        self.lbl_name = QLabel(f"Worker {slot + 1}: waiting")
        self.lbl_name.setStyleSheet("color: #cccccc; font-size: 10px; border: none;")
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.lbl_img.setMinimumSize(80, 45)
        self.lbl_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_img.setStyleSheet("background-color: #000000; border: none;")
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(5)
        self.bar.setStyleSheet("QProgressBar { background-color: #2a2a2a; border: none; }"
                               "QProgressBar::chunk { background-color: #2979ff; }")
        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("color: #999999; font-size: 10px; border: none;")
        lay.addWidget(self.lbl_name)
        lay.addWidget(self.lbl_img, 1)
        lay.addWidget(self.bar)
        lay.addWidget(self.lbl_stats)
        self._image: Optional[QImage] = None

    def set_frame(self, rgb: np.ndarray):
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        self._image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._show()

    def _show(self):
        if self._image is None:
            return
        size = self.lbl_img.size()
        if size.width() < 2 or size.height() < 2:
            return
        self.lbl_img.setPixmap(QPixmap.fromImage(self._image).scaled(
            size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._show()


class InferenceGrid(QWidget):
    """Tiles for the videos being tracked at once, plus a summary line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("color: #cccccc; font-size: 11px; font-weight: bold;")
        lay.addWidget(self.lbl_summary)
        self._grid_host = QWidget()
        self._grid = QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(4)
        lay.addWidget(self._grid_host, 1)
        self.tiles: List[_Tile] = []
        self._names: Dict[int, str] = {}

    def set_slots(self, n: int):
        for t in self.tiles:
            t.setParent(None)
            t.deleteLater()
        self.tiles = []
        self._names = {}
        cols = max(1, math.ceil(math.sqrt(n)))
        for i in range(n):
            t = _Tile(i, self._grid_host)
            self._grid.addWidget(t, i // cols, i % cols)
            self.tiles.append(t)
        rows = max(1, math.ceil(n / cols))
        for r in range(rows):
            self._grid.setRowStretch(r, 1)
        for c in range(cols):
            self._grid.setColumnStretch(c, 1)

    def video_started(self, slot: int, idx: int, path: str):
        if 0 <= slot < len(self.tiles):
            name = os.path.basename(path)
            self._names[slot] = name
            t = self.tiles[slot]
            t.lbl_name.setText(f"{idx + 1}. {name}")
            t.lbl_name.setStyleSheet("color: #cccccc; font-size: 10px; border: none;")
            t.bar.setValue(0)
            t.lbl_stats.setText("loading the model…")

    def video_progress(self, slot: int, frame: int, total: int, fps: float):
        if 0 <= slot < len(self.tiles):
            t = self.tiles[slot]
            t.bar.setMaximum(max(1, total))
            t.bar.setValue(frame)
            left = (total - frame) / fps if fps > 0 else 0
            t.lbl_stats.setText(f"frame {frame}/{total} · {fps:.1f} fps · "
                                f"{int(left // 60)}:{int(left % 60):02d} left")

    def video_frame(self, slot: int, rgb, frame: int):
        if 0 <= slot < len(self.tiles) and rgb is not None:
            self.tiles[slot].set_frame(rgb)

    def video_failed(self, slot: int, msg: str):
        if 0 <= slot < len(self.tiles):
            t = self.tiles[slot]
            t.lbl_name.setStyleSheet("color: #ef5350; font-size: 10px; border: none;")
            t.lbl_stats.setText(msg[:120])

    def video_done(self, slot: int, text: str):
        if 0 <= slot < len(self.tiles):
            self.tiles[slot].bar.setValue(self.tiles[slot].bar.maximum())
            self.tiles[slot].lbl_stats.setText(text)

    def worker_exited(self, slot: int):
        if 0 <= slot < len(self.tiles):
            t = self.tiles[slot]
            if not t.lbl_stats.text().startswith(("done", "finished")):
                t.lbl_stats.setText(t.lbl_stats.text() + " · worker stopped")
            else:
                t.lbl_stats.setText(t.lbl_stats.text() + " · no more videos")
