"""The Fiber Photometry tool's preview: the behaviour video with the traces aligned under it.

Scrub with the slider (or arrow keys), play at 0.25-8x, and click the session overview to jump.
The traces show a window centred on the current frame's time (frame k at k / fps of the video,
the same timing the outputs use); dropouts show as gaps. Playback keeps to the wall clock: when the
PC cannot decode every frame at the chosen speed it skips frames rather than slowing down. For a
wireless headstage two more rows show its movement (accelerometer) and its link strength.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QElapsedTimer, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout,
    QWidget,
)

GREEN, VIOLET, BLUE, AMBER, RED = "#1baf7a", "#a070e0", "#4a90e2", "#f0a020", "#e05050"
GREY, CYAN = "#b0b0b0", "#40c8d0"


class PreviewPane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.fps = 30.0
        self.n_frames = 0
        self.frame = 0
        self.r = None
        self._last_read = -1
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.title = QLabel("Select a recording to preview it")
        self.title.setStyleSheet("font-weight: bold;")
        lay.addWidget(self.title)

        self.video = QLabel("no video")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(320, 180)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setStyleSheet("background: #111; color: #777;")
        lay.addWidget(self.video, 5)

        pg.setConfigOptions(antialias=False, background="#1e1e1e", foreground="#cccccc")
        self.plots = pg.GraphicsLayoutWidget()
        self.p_sig = self.plots.addPlot(row=0, col=0)
        self.p_dff = self.plots.addPlot(row=1, col=0)
        self.p_z = self.plots.addPlot(row=2, col=0)
        self.p_mov = self.plots.addPlot(row=3, col=0)
        self.p_link = self.plots.addPlot(row=4, col=0)
        self.p_all = self.plots.addPlot(row=5, col=0)
        for p, label in ((self.p_sig, "signal (V)"), (self.p_dff, "dF/F (%)"), (self.p_z, "z-score"),
                         (self.p_mov, "movement"), (self.p_link, "link (dBm)"),
                         (self.p_all, "session dF/F")):
            p.setLabel("left", label)
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setMouseEnabled(x=False, y=False)
            p.hideButtons()
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
        for p in (self.p_dff, self.p_z, self.p_mov, self.p_link):
            p.setXLink(self.p_sig)
        for p in (self.p_sig, self.p_dff, self.p_z, self.p_mov, self.p_link):
            p.getViewBox().setAutoVisible(y=True)
            p.enableAutoRange(axis="y")
        self.p_link.setLabel("bottom", "time (s)")
        for p in (self.p_mov, self.p_link):
            p.setMaximumHeight(90)
        self.p_all.setMaximumHeight(110)
        self.p_sig.addLegend(offset=(5, 2), labelTextSize="8pt")
        self.c_sig = self.p_sig.plot(pen=pg.mkPen(GREEN, width=1), name="470 nm signal")
        self.c_fit = self.p_sig.plot(pen=pg.mkPen(VIOLET, width=1),
                                     name="fitted control (415 nm scaled onto the signal)")
        self.c_dff = self.p_dff.plot(pen=pg.mkPen(BLUE, width=1))
        self.c_z = self.p_z.plot(pen=pg.mkPen(AMBER, width=1))
        self.c_mov = self.p_mov.plot(pen=pg.mkPen(GREY, width=1))
        self.c_link = self.p_link.plot(pen=pg.mkPen(CYAN, width=1))
        self.c_all = self.p_all.plot(pen=pg.mkPen(BLUE, width=1))
        self.cursors = [pg.InfiniteLine(angle=90, pen=pg.mkPen(RED, width=1.5))
                        for _ in range(6)]
        for p, c in zip((self.p_sig, self.p_dff, self.p_z, self.p_mov, self.p_link, self.p_all),
                        self.cursors):
            p.addItem(c)
        self.p_all.scene().sigMouseClicked.connect(self._overview_click)
        lay.addWidget(self.plots, 6)

        row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedWidth(70)
        self.btn_play.clicked.connect(self.toggle_play)
        row.addWidget(self.btn_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.seek)
        row.addWidget(self.slider, 1)
        self.lbl_time = QLabel("")
        self.lbl_time.setMinimumWidth(230)
        row.addWidget(self.lbl_time)
        row.addWidget(QLabel("Speed"))
        self.speed = QComboBox()
        for s in ("0.25x", "0.5x", "1x", "2x", "4x", "8x"):
            self.speed.addItem(s, float(s[:-1]))
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self._restart_timer)
        row.addWidget(self.speed)
        row.addWidget(QLabel("Window"))
        self.window = QDoubleSpinBox()
        self.window.setRange(2, 600)
        self.window.setValue(20)
        self.window.setSuffix(" s")
        self.window.valueChanged.connect(lambda _v: self._show_traces())
        row.addWidget(self.window)
        lay.addLayout(row)
        self._set_enabled(False)

    def _set_enabled(self, on):
        for w in (self.btn_play, self.slider, self.speed, self.window):
            w.setEnabled(on)

    # ------------------------------------------------------------------ data
    def clear(self, message="Select a recording to preview it"):
        self.stop()
        if self.cap is not None:
            self.cap.release()
        self.cap, self.r = None, None
        self.title.setText(message)
        self.video.setText("no video")
        self.video.setPixmap(QPixmap())
        for c in (self.c_sig, self.c_fit, self.c_dff, self.c_z, self.c_mov, self.c_link, self.c_all):
            c.setData([], [])
        self._set_enabled(False)

    def set_recording(self, name: str, result, video_path: Optional[str], label: str = "",
                      movement=None, link=None):
        """Show ``result`` (a PhotometryResult) with its video. ``movement`` and ``link`` are
        optional (time, values) pairs: headstage acceleration and wireless signal strength."""
        import cv2
        self.clear(f"{name}   {label}")
        self.r = result
        t = result.time
        for plot, curve, extra in ((self.p_mov, self.c_mov, movement),
                                   (self.p_link, self.c_link, link)):
            plot.setVisible(extra is not None)
            if extra is not None:
                curve.setData(*extra, connect="finite")
        sig = np.where(result.dropout, np.nan, result.signal)
        fit = np.where(result.dropout, np.nan, result.fitted_control)
        self.c_sig.setData(t, sig, connect="finite")
        self.c_fit.setData(t, fit, connect="finite")
        self.c_dff.setData(t, result.dff, connect="finite")
        self.c_z.setData(t, result.z, connect="finite")
        # session overview: 0.5-s means, light enough to redraw
        step = max(1, int(round(0.5 / np.median(np.diff(t)))))
        n = len(t) // step * step
        with np.errstate(invalid="ignore"):
            self.c_all.setData(t[:n].reshape(-1, step).mean(1),
                               np.nanmean(result.dff[:n].reshape(-1, step), 1), connect="finite")
        self.p_all.setXRange(t[0], t[-1], padding=0)
        if video_path:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                self.cap = cap
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.cap is None:
            self.video.setText("no video for this recording")
            self.fps = 30.0
            self.n_frames = int((t[-1] - t[0]) * self.fps)
        self._last_read = -1
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self.n_frames - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._set_enabled(True)
        self.seek(0)

    # ------------------------------------------------------------------ playback
    def toggle_play(self):
        if self.timer.isActive():
            self.stop()
        else:
            self.btn_play.setText("Pause")
            self._restart_timer()

    def stop(self):
        self.timer.stop()
        self.btn_play.setText("Play")

    def _restart_timer(self):
        """(Re)start playback from the current frame at the chosen speed, on the wall clock."""
        if self.btn_play.text() != "Pause":
            return
        self._clock = QElapsedTimer()
        self._clock.start()
        self._play_from = self.frame
        self.timer.start(30)                                  # redraw ~33 times a second

    def _tick(self):
        elapsed = self._clock.elapsed() / 1000
        target = self._play_from + int(elapsed * self.fps * self.speed.currentData())
        if target >= self.n_frames:
            self.slider.setValue(self.n_frames - 1)
            self.stop()
            return
        if target > self.frame:
            self.slider.setValue(target)

    def seek(self, k: int):
        if self.r is None:
            return
        self.frame = int(k)
        self._show_frame()
        self._show_traces()

    def _show_frame(self):
        if self.cap is None:
            return
        import cv2
        k = self.frame
        ahead = k - self._last_read
        if 1 < ahead <= 10:                        # playing faster than 1x: skip by grabbing
            for _ in range(ahead - 1):
                self.cap.grab()
        elif ahead != 1:                           # a jump: seek
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        got, img = self.cap.read()
        self._last_read = k
        if not got:
            return
        h, w = img.shape[:2]
        qimg = QImage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).data, w, h, 3 * w, QImage.Format_RGB888)
        self.video.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.FastTransformation))

    def _show_traces(self):
        if self.r is None:
            return
        t_now = self.frame / self.fps
        half = self.window.value() / 2
        self.p_sig.setXRange(t_now - half, t_now + half, padding=0)
        for c in self.cursors:
            c.setPos(t_now)
        i = min(np.searchsorted(self.r.time, t_now), len(self.r.time) - 1)
        val = self.r.dff[i]
        self.lbl_time.setText(f"t {t_now:8.2f} s   frame {self.frame + 1:,}/{self.n_frames:,}   "
                              f"dF/F {'gap' if np.isnan(val) else f'{val:+.2f}%'}")

    def _overview_click(self, ev):
        if self.r is None:
            return
        vb = self.p_all.getViewBox()
        if not self.p_all.sceneBoundingRect().contains(ev.scenePos()):
            return
        t = vb.mapSceneToView(ev.scenePos()).x()
        self.slider.setValue(int(np.clip(t * self.fps, 0, self.n_frames - 1)))

    def keyPressEvent(self, ev):
        step = {Qt.Key_Right: 1, Qt.Key_Left: -1, Qt.Key_PageDown: int(self.fps * 5),
                Qt.Key_PageUp: -int(self.fps * 5)}.get(ev.key())
        if step and self.r is not None:
            self.slider.setValue(int(np.clip(self.frame + step, 0, self.n_frames - 1)))
        elif ev.key() == Qt.Key_Space:
            self.toggle_play()
        else:
            super().keyPressEvent(ev)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.r is not None and self.cap is not None:
            self._last_read = -1                   # redraw at the new size
            self._show_frame()

    def closeEvent(self, ev):
        self.clear()
        super().closeEvent(ev)
