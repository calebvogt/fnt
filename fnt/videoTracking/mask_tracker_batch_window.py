"""The window of a batch video inference run (``mask_tracker_cli infer-videos``): which video,
how far, the last frame with its masks, and Pause / Stop, so a long run can be watched
at a glance rather than trusted from a log."""
from __future__ import annotations

import os
import time
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel, QProgressBar,
                             QPushButton, QTextEdit, QVBoxLayout)

from .mask_tracker_batch import run_batch

COLORS = [(255, 170, 40), (60, 200, 255), (200, 110, 255), (90, 230, 120), (255, 120, 160),
          (80, 140, 255), (255, 220, 70), (160, 255, 220)]


class BatchWorker(QThread):
    video_started = pyqtSignal(int, int, str)
    frame_done = pyqtSignal(int, int, object)      # sample, samples, preview RGB
    log = pyqtSignal(str)
    finished_run = pyqtSignal(object)

    def __init__(self, model, videos, step, conf, imgsz, device):
        super().__init__()
        self.args = (model, videos, step, conf, imgsz, device)
        self.stop_flag = False
        self.pause_flag = False
        self._last_preview = 0.0

    def run(self):
        model, videos, step, conf, imgsz, device = self.args

        def on_frame(si, n, frame, masks):
            now = time.time()
            if now - self._last_preview < 0.5 and si != n - 1:   # repaint at most twice a second
                self.frame_done.emit(si, n, None)
                return
            self._last_preview = now
            img = frame.copy()
            for k, m in enumerate(masks):
                c = COLORS[k % len(COLORS)]
                over = img.copy()
                over[m] = c
                img = cv2.addWeighted(over, 0.4, img, 0.6, 0)
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, c, 2)
            h, w = img.shape[:2]
            img = cv2.resize(img, (640, int(640 * h / w)), interpolation=cv2.INTER_AREA)
            self.frame_done.emit(si, n, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        try:
            res = run_batch(model, videos, step, conf, imgsz, device,
                            on_video=lambda k, n, v: self.video_started.emit(k, n, v),
                            on_frame=on_frame, should_stop=lambda: self.stop_flag,
                            is_paused=lambda: self.pause_flag, log=self.log.emit)
        except Exception as e:
            self.log.emit(f"[batch] stopped: {e}")
            res = {"error": str(e)}
        self.finished_run.emit(res)


class BatchWindow(QDialog):
    def __init__(self, worker: BatchWorker, title: str):
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.setStyleSheet("QDialog { background: #2b2b2b; color: #cccccc; }"
                           "QLabel { color: #cccccc; } QTextEdit { background: #1a1a1a; color: #b0b0b0;"
                           " font-family: monospace; font-size: 10px; }")
        self.worker = worker
        self.t0 = time.time()
        v = QVBoxLayout(self)
        self.lbl_video = QLabel("Loading the model…")
        v.addWidget(self.lbl_video)
        self.bar_videos = QProgressBar()
        self.bar_frames = QProgressBar()
        v.addWidget(self.bar_videos)
        v.addWidget(self.bar_frames)
        self.preview = QLabel()
        self.preview.setMinimumSize(640, 360)
        self.preview.setAlignment(Qt.AlignCenter)
        v.addWidget(self.preview, 1)
        row = QHBoxLayout()
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.btn_pause.clicked.connect(self._pause)
        self.btn_stop.clicked.connect(self._stop)
        row.addStretch()
        row.addWidget(self.btn_pause)
        row.addWidget(self.btn_stop)
        v.addLayout(row)
        self.logbox = QTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumHeight(140)
        v.addWidget(self.logbox)
        worker.video_started.connect(self._video)
        worker.frame_done.connect(self._frame)
        worker.log.connect(self._log)
        worker.finished_run.connect(self._finished)
        self.resize(700, 760)

    def _video(self, k, n, path):
        self.bar_videos.setMaximum(n)
        self.bar_videos.setValue(k)
        el = time.time() - self.t0
        eta = f", about {el / k * (n - k) / 3600:.1f} h left" if k else ""
        self.lbl_video.setText(f"Video {k + 1} of {n}: {os.path.basename(path)}{eta}")

    def _frame(self, si, n, rgb):
        self.bar_frames.setMaximum(n)
        self.bar_frames.setValue(si + 1)
        if rgb is not None:
            h, w = rgb.shape[:2]
            img = QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format_RGB888).copy()
            self.preview.setPixmap(QPixmap.fromImage(img))

    def _log(self, msg):
        self.logbox.append(msg)
        print(msg, flush=True)

    def _pause(self):
        self.worker.pause_flag = not self.worker.pause_flag
        self.btn_pause.setText("Resume" if self.worker.pause_flag else "Pause")

    def _stop(self):
        self.worker.stop_flag = True
        self.btn_stop.setEnabled(False)
        self.lbl_video.setText("Stopping after this frame… (the run resumes where it stopped)")

    def _finished(self, res):
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.bar_videos.setValue(self.bar_videos.maximum())
        if "error" in res:
            self.lbl_video.setText(f"Stopped: {res['error']}")
        else:
            self.lbl_video.setText(f"Done: {res['done']} of {res['videos']} video(s) in "
                                   f"{res['seconds'] / 3600:.1f} h with {res['model']}. Close this window.")

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop_flag = True
            self.worker.wait(30000)
        event.accept()


def run_with_window(model, videos: List[str], step, conf, imgsz: Optional[int], device) -> int:
    app = QApplication.instance() or QApplication([])
    worker = BatchWorker(model, videos, step, conf, imgsz, device)
    win = BatchWindow(worker, f"Mask Tracker — Batch Inference ({len(videos)} videos)")
    win.show()
    worker.start()
    return app.exec_()
