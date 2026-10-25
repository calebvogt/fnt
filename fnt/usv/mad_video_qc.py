"""MAD's Video QC window: the camera that watched a microphone, over the
stretch of the recording on screen.

Opened from MAD's "Video QC" button. The project names its video in
``video_qc.json`` (see :mod:`fnt.usv.usv_detector.mad_video`); the window
shows the channel's own camera by default and any other camera of the project
on request. It follows MAD's playhead: while the audio plays, the frame keeps
pace with it, and dragging the slider or playing the video here moves the
playhead line on the spectrogram, so a sound and the frame it goes with are
always on screen together.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PyQt5.QtCore import QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider, QVBoxLayout)

from fnt.usv.usv_detector.mad_video import (VideoClip, VideoQCConfig, VideoQCError,
                                            cameras_for, read_clip, recording_ids)

#: Playback speeds offered for the video's own player.
SPEEDS = (0.1, 0.25, 0.5, 1.0)


class _ClipWorker(QThread):
    """Decodes one clip off the GUI thread."""
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)

    def __init__(self, wav, t0, t1, camera, cfg, parent=None):
        super().__init__(parent)
        self._args = (wav, t0, t1, camera, cfg)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        wav, t0, t1, camera, cfg = self._args
        try:
            clip = read_clip(wav, t0, t1, camera, cfg,
                             progress=lambda a, b: self.progress.emit(a, b),
                             cancelled=lambda: self._cancel)
        except VideoQCError as e:
            self.failed.emit(str(e))
            return
        except Exception as e:                      # a bad file must not take MAD down
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.done.emit(clip)


class MADVideoQCWindow(QDialog):
    """One instance per MAD session, hidden rather than destroyed on close."""

    #: Seconds into the recording, when the slider or this window's own player
    #: moves (not when MAD's playhead drives it).
    position_changed = pyqtSignal(float)
    #: Emitted on show/hide so MAD can clear the playhead line it drew for us.
    visibility_changed = pyqtSignal(bool)

    def __init__(self, main=None):
        # Un-parented, like MAD's other pop-outs, so it can sit on another
        # screen while the spectrogram is in use.
        super().__init__(None)
        self._main = main
        self._clip: Optional[VideoClip] = None
        self._worker: Optional[_ClipWorker] = None
        self._request = None                 # (wav, t0, t1, cfg) of the clip shown or loading
        self._index = -1
        self._driven = False                 # MAD's audio is playing and drives the frame
        self._mark = None                    # (t0, t1) of the selected call, if any

        self.setWindowTitle("Video QC")
        self.setModal(False)
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(820, 560)

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Camera:"))
        self.combo_camera = QComboBox()
        self.combo_camera.setToolTip(
            "The microphone's own camera comes first. Any other camera of the "
            "project can be pulled for the same stretch of time.")
        self.combo_camera.activated.connect(lambda _i: self._reload())
        top.addWidget(self.combo_camera)
        self.btn_pull = QPushButton("Pull Current View")
        self.btn_pull.setToolTip(
            "Load the video for the stretch now shown in MAD's spectrogram "
            "(plus 1 s either side).")
        self.btn_pull.clicked.connect(self._pull_from_main)
        top.addWidget(self.btn_pull)
        top.addStretch(1)
        top.addWidget(QLabel("Speed:"))
        self.combo_speed = QComboBox()
        for s in SPEEDS:
            self.combo_speed.addItem(f"{s:g}x", s)
        self.combo_speed.setCurrentIndex(SPEEDS.index(0.25))
        self.combo_speed.setToolTip("Speed of this window's own player (Play here, "
                                    "silent). MAD's Play sets its own pace.")
        self.combo_speed.currentIndexChanged.connect(self._set_timer_interval)
        top.addWidget(self.combo_speed)
        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedWidth(70)
        self.btn_play.clicked.connect(self._toggle_play)
        top.addWidget(self.btn_play)
        v.addLayout(top)

        self.lbl_frame = QLabel("Press Video QC in MAD to load the video for the view.")
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        self.lbl_frame.setMinimumSize(320, 180)
        self.lbl_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_frame.setStyleSheet("background:#111; color:#aab4bd;")
        v.addWidget(self.lbl_frame, 1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        v.addWidget(self.slider)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#aab4bd; font-size:10px;")
        v.addWidget(self.lbl_status)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self._set_timer_interval()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def request(self, wav: str, t0_s: float, t1_s: float, cfg: VideoQCConfig,
                camera: Optional[str] = None) -> None:
        """Load the video for ``t0_s``..``t1_s`` of ``wav``; ``camera`` None =
        the channel's own camera."""
        cams = cameras_for(wav, cfg)
        trial, channel = recording_ids(wav, cfg)
        keep = self.combo_camera.currentData() if self._request and \
            self._request[0] == wav else None
        self.combo_camera.blockSignals(True)
        self.combo_camera.clear()
        for i, c in enumerate(cams):
            own = (i == 0 and channel and cfg.cameras.get(channel) == c)
            self.combo_camera.addItem(f"{c} ({channel}'s camera)" if own else c, c)
        want = camera or keep or (cams[0] if cams else None)
        if want in cams:
            self.combo_camera.setCurrentIndex(cams.index(want))
        self.combo_camera.blockSignals(False)
        self._request = (wav, float(t0_s), float(t1_s), cfg)
        if not cams:
            self._show_message(f"The project's video_qc.json names no camera for "
                               f"{os.path.basename(wav)} ({trial}, {channel}).")
            return
        self._reload()

    def _pull_from_main(self):
        m = self._main
        if m is not None and hasattr(m, '_show_video_qc'):
            m._show_video_qc()

    def _reload(self):
        if self._request is None:
            return
        wav, t0, t1, cfg = self._request
        camera = self.combo_camera.currentData()
        if not camera:
            return
        self._stop_own_play()
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        self._clip = None
        self._index = -1
        self.slider.setEnabled(False)
        self.setWindowTitle(f"Video QC — {camera} — {os.path.basename(wav)}")
        self._show_message(f"Loading {camera}, {t0:.2f}–{t1:.2f} s of "
                           f"{os.path.basename(wav)}…")
        w = _ClipWorker(wav, t0, t1, camera, cfg, parent=self)
        w.done.connect(lambda clip, w=w: self._on_loaded(w, clip))
        w.failed.connect(lambda msg, w=w: self._on_failed(w, msg))
        w.progress.connect(lambda a, b, w=w: self._on_progress(w, a, b))
        self._worker = w
        w.start()

    def _on_progress(self, w, a, b):
        if w is self._worker:
            self.lbl_status.setText(f"Decoding frame {a} of {b}…")

    def _on_failed(self, w, msg):
        if w is not self._worker:
            return
        self._worker = None
        self._show_message(msg)
        self.lbl_status.setText("")

    def _on_loaded(self, w, clip: VideoClip):
        if w is not self._worker:
            return                               # superseded by a newer request
        self._worker = None
        self._clip = clip
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(clip.frames) - 1)
        self.slider.blockSignals(False)
        self.slider.setEnabled(True)
        start = self._request[1] if self._request else 0.0
        pos = getattr(getattr(self._main, 'spectrogram', None), 'playback_position', None)
        self.show_time(pos if pos is not None else start + 1.0)

    def _show_message(self, text: str):
        self.lbl_frame.setPixmap(QPixmap())
        self.lbl_frame.setText(text)

    # ------------------------------------------------------------------
    # Showing frames
    # ------------------------------------------------------------------
    def set_call_mark(self, t0_s: Optional[float], t1_s: Optional[float]):
        """Frames inside ``t0_s``..``t1_s`` (the selected call) get a red frame."""
        self._mark = None if t0_s is None or t1_s is None else (float(t0_s), float(t1_s))
        self._render()

    def show_time(self, t_s: float):
        """Show the frame at ``t_s`` seconds into the recording."""
        if self._clip is None:
            return
        i = self._clip.index_at(t_s)
        self._show_index(i)

    def _show_index(self, i: int):
        if self._clip is None or not (0 <= i < len(self._clip.frames)):
            return
        self._index = i
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self._render()

    def _render(self):
        clip = self._clip
        if clip is None or self._index < 0:
            return
        fr = np.ascontiguousarray(clip.frames[self._index])
        h, w = fr.shape[:2]
        img = QImage(fr.data, w, h, w, QImage.Format_Grayscale8).copy()
        pm = QPixmap.fromImage(img).scaled(
            self.lbl_frame.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        t = float(clip.times_s[self._index])
        name, k, when = clip.frame_info(self._index)
        p = QPainter(pm)
        try:
            p.setRenderHint(QPainter.Antialiasing)
            if self._mark and self._mark[0] - 0.034 <= t <= self._mark[1] + 0.034:
                p.setPen(QPen(QColor(230, 40, 40), 6))
                p.drawRect(QRectF(3, 3, pm.width() - 6, pm.height() - 6))
            f = QFont()
            f.setPointSize(9)
            f.setBold(True)
            p.setFont(f)
            text = (f"{clip.camera}  ·  t = {t:.2f} s  ·  {when:%Y-%m-%d %H:%M:%S}."
                    f"{when.microsecond // 1000:03d}  ·  {name} frame {k}")
            p.setPen(QColor(0, 0, 0))
            p.drawText(QRectF(9, 7, pm.width() - 14, 20), Qt.AlignLeft, text)
            p.setPen(QColor(255, 230, 90))
            p.drawText(QRectF(8, 6, pm.width() - 14, 20), Qt.AlignLeft, text)
        finally:
            p.end()
        self.lbl_frame.setText("")
        self.lbl_frame.setPixmap(pm)
        off = f"; clock offset {clip.offset_s:+.2f} s" if clip.offset_s else ""
        step = (len(clip.frames) > 1 and
                float(np.median(np.diff(clip.times_s))) or 0.0)
        self.lbl_status.setText(
            f"Frame {self._index + 1} of {len(clip.frames)}"
            f"{f' (every {step:.2f} s)' if step > 0.1 else ''}  ·  video clock from "
            f"the file names{off}. Compare with the time the camera burns into "
            f"the frame.")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()

    # ------------------------------------------------------------------
    # Slider and this window's own (silent) player
    # ------------------------------------------------------------------
    def _on_slider(self, i: int):
        if self._clip is None:
            return
        self._show_index(i)
        if not self._driven:
            self.position_changed.emit(float(self._clip.times_s[i]))

    def _set_timer_interval(self, *_):
        speed = float(self.combo_speed.currentData() or 0.25)
        fps = 15.0
        if self._clip is not None and self._clip.spans:
            fps = self._clip.spans[0].fps
        self._timer.setInterval(max(10, int(round(1000.0 / (fps * speed)))))

    def _toggle_play(self):
        if self._timer.isActive():
            self._stop_own_play()
            return
        if self._clip is None or self._driven:
            return
        if self._index >= len(self._clip.frames) - 1:
            self._show_index(0)
        self._set_timer_interval()
        self._timer.start()
        self.btn_play.setText("Pause")

    def _stop_own_play(self):
        self._timer.stop()
        self.btn_play.setText("Play")

    def _step(self):
        if self._clip is None or self._index >= len(self._clip.frames) - 1:
            self._stop_own_play()
            return
        self._show_index(self._index + 1)
        self.position_changed.emit(float(self._clip.times_s[self._index]))

    # ------------------------------------------------------------------
    # MAD's playhead (its audio Play)
    # ------------------------------------------------------------------
    def playback_started(self, start_s: float, end_s: float):
        self._stop_own_play()
        self._driven = True
        self.btn_play.setEnabled(False)
        self.show_time(start_s)

    def playback_tick(self, pos_s: float):
        self.show_time(pos_s)

    def playback_stopped(self):
        self._driven = False
        self.btn_play.setEnabled(True)

    # ------------------------------------------------------------------
    def showEvent(self, e):
        super().showEvent(e)
        self.visibility_changed.emit(True)

    def hideEvent(self, e):
        super().hideEvent(e)
        self._stop_own_play()
        self.visibility_changed.emit(False)

    def closeEvent(self, e):
        # Hidden, not destroyed: reopening keeps the clip and the camera choice.
        if self._worker is not None:
            self._worker.cancel()
        e.ignore()
        self.hide()
