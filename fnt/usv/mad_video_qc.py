"""MAD's Fusion QC window: what was happening at a microphone over the stretch
of the recording on screen — the camera that watched it, the mice that camera's
Mask Tracker model finds in each frame, and the mice the RFID reader saw in
that zone.

Opened from MAD's "Fusion QC" button. The project names its sources in
``video_qc.json`` (see :mod:`fnt.usv.usv_detector.mad_video`); the window shows
the channel's own camera by default and any other camera of the project on
request. It follows MAD's playhead: while the audio plays, the frame keeps pace
with it, and dragging the slider or playing the video here moves the playhead
line on the spectrogram, so a sound, the frame and the RFID reads it goes with
are on screen together.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget)

from fnt.usv.usv_detector.mad_video import (VideoClip, VideoQCConfig, VideoQCError,
                                            ZoneRFID, cameras_for, masks_at, read_clip,
                                            recording_ids, segment_clip, zone_rfid)

#: Playback speeds offered for the video's own player.
SPEEDS = (0.1, 0.25, 0.5, 1.0)
#: Outline colours, one per mouse found in a frame (not identities).
MASK_COLORS = [(255, 200, 0), (0, 200, 255), (255, 90, 200), (120, 255, 120),
               (255, 140, 60), (170, 140, 255), (255, 255, 255), (60, 220, 200)]


class _ClipWorker(QThread):
    """Decodes one clip off the GUI thread, then reads its RFID and segments
    its mice (each sent as soon as it is ready)."""
    done = pyqtSignal(object)
    rfid_done = pyqtSignal(object)
    masks_done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, wav, t0, t1, camera, cfg, want_masks, parent=None):
        super().__init__(parent)
        self._args = (wav, t0, t1, camera, cfg)
        self._want_masks = want_masks
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        wav, t0, t1, camera, cfg = self._args
        model = (cfg.mask_models or {}).get(camera) if self._want_masks else None
        try:
            clip = read_clip(wav, t0, t1, camera, cfg, width=1024 if model else 640,
                             progress=lambda a, b: self.progress.emit(f"Decoding frame {a} of {b}…"),
                             cancelled=lambda: self._cancel)
        except VideoQCError as e:
            self.failed.emit(str(e))
            return
        except Exception as e:                      # a bad file must not take MAD down
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.done.emit(clip)
        try:
            self.rfid_done.emit(zone_rfid(wav, t0, t1, cfg))
        except Exception as e:
            self.progress.emit(f"RFID could not be read: {type(e).__name__}: {e}")
        if model and not self._cancel:
            try:
                every = max(1, int(cfg.mask_every or 1))
                masks = segment_clip(
                    clip, model, every=every, device=cfg.mask_device or "cpu",
                    progress=lambda a, b: self.progress.emit(f"Finding mice: frame {a} of {b}…"),
                    cancelled=lambda: self._cancel)
                if not self._cancel:
                    self.masks_done.emit((masks, every))
            except Exception as e:
                self.progress.emit(f"Masks could not be made: {type(e).__name__}: {e}")


class _RFIDStrip(QWidget):
    """One row per mouse the reader saw at the zone: its visit bouts (grey)
    and each read (a tick), over the clip's time; a line at the frame shown."""

    ROW_H = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rfid: Optional[ZoneRFID] = None
        self._t0, self._t1 = 0.0, 1.0
        self._now: Optional[float] = None
        self._rows: List[str] = []
        self.setMinimumHeight(24)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, rfid: Optional[ZoneRFID], t0: float, t1: float):
        self._rfid, self._t0, self._t1 = rfid, t0, max(t1, t0 + 1e-6)
        codes = []
        if rfid is not None:
            for df in (rfid.bouts, rfid.reads):
                if df is not None and len(df):
                    codes += list(df["code"])
        self._rows = sorted(set(codes))
        self.setFixedHeight(max(24, 6 + self.ROW_H * max(1, len(self._rows))))
        self.update()

    def set_now(self, t: Optional[float]):
        self._now = t
        self.update()

    def _x(self, t, x0, w):
        return x0 + (t - self._t0) / (self._t1 - self._t0) * w

    def paintEvent(self, e):
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), QColor(24, 26, 28))
            x0, w = 70.0, max(10.0, self.width() - 78.0)
            f = QFont()
            f.setPointSize(8)
            p.setFont(f)
            if not self._rows:
                p.setPen(QColor(150, 160, 170))
                msg = ("No RFID data named in video_qc.json." if self._rfid is None
                       else f"No reads at zone {self._rfid.zone}'s antenna in this stretch.")
                p.drawText(QRectF(8, 4, self.width() - 16, 16), Qt.AlignLeft, msg)
                return
            sex = {}
            if self._rfid.bouts is not None and len(self._rfid.bouts):
                sex = dict(zip(self._rfid.bouts["code"], self._rfid.bouts["sex"]))
            for r, code in enumerate(self._rows):
                y = 3 + r * self.ROW_H
                p.setPen(QColor(200, 210, 220))
                s = sex.get(code)
                p.drawText(QRectF(4, y, x0 - 8, self.ROW_H), Qt.AlignVCenter | Qt.AlignLeft,
                           f"{code}{f' ({s})' if isinstance(s, str) else ''}")
                b = self._rfid.bouts
                if b is not None and len(b):
                    for t0, t1 in b.loc[b["code"] == code, ["t0_s", "t1_s"]].itertuples(index=False):
                        a, z = self._x(max(t0, self._t0), x0, w), self._x(min(t1, self._t1), x0, w)
                        p.fillRect(QRectF(a, y + 4, max(1.0, z - a), self.ROW_H - 8), QColor(80, 86, 92))
                rd = self._rfid.reads
                if rd is not None and len(rd):
                    p.setPen(QPen(QColor(120, 220, 255), 2))
                    for t in rd.loc[rd["code"] == code, "t_s"]:
                        x = self._x(t, x0, w)
                        p.drawLine(QPointF(x, y + 2), QPointF(x, y + self.ROW_H - 2))
            if self._now is not None:
                x = self._x(self._now, x0, w)
                p.setPen(QPen(QColor(255, 80, 80), 1))
                p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        finally:
            p.end()


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
        self._masks: Dict[int, List] = {}
        self._mask_every = 1
        self._rfid: Optional[ZoneRFID] = None

        self.setWindowTitle("Fusion QC")
        self.setModal(False)
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(860, 680)

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
            "Load the stretch now shown in MAD's spectrogram (plus 1 s either side).")
        self.btn_pull.clicked.connect(self._pull_from_main)
        top.addWidget(self.btn_pull)
        self.chk_masks = QCheckBox("Mice")
        self.chk_masks.setChecked(True)
        self.chk_masks.setToolTip(
            "Outline the mice the camera's Mask Tracker model finds (every few "
            "frames, on the CPU while MAD holds the GPU). Colours tell mice apart "
            "within a frame only; they are not identities.")
        self.chk_masks.toggled.connect(lambda _on: self._render())
        top.addWidget(self.chk_masks)
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

        self.lbl_frame = QLabel("Press Fusion QC in MAD to load the view.")
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        self.lbl_frame.setMinimumSize(320, 180)
        self.lbl_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_frame.setStyleSheet("background:#111; color:#aab4bd;")
        v.addWidget(self.lbl_frame, 1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider)
        v.addWidget(self.slider)

        self.lbl_rfid = QLabel("")
        self.lbl_rfid.setWordWrap(True)
        self.lbl_rfid.setStyleSheet("color:#d0d8e0; font-size:11px;")
        v.addWidget(self.lbl_rfid)
        self.rfid_strip = _RFIDStrip()
        self.rfid_strip.setToolTip(
            "RFID at the microphone's zone: one row per mouse; grey = its visit "
            "bouts, blue ticks = its reads at the zone's antenna, red line = the "
            "frame shown. The reader cycles its antennas, so a mouse can sit in "
            "the zone between reads.")
        v.addWidget(self.rfid_strip)

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
        """Load ``t0_s``..``t1_s`` of ``wav``; ``camera`` None = the channel's
        own camera (or the one chosen before, for the same recording)."""
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
        self._masks, self._rfid = {}, None
        self._index = -1
        self.slider.setEnabled(False)
        self.lbl_rfid.setText("")
        self.rfid_strip.set_data(None, t0, t1)
        self.setWindowTitle(f"Fusion QC — {camera} — {os.path.basename(wav)}")
        self._show_message(f"Loading {camera}, {t0:.2f}–{t1:.2f} s of "
                           f"{os.path.basename(wav)}…")
        w = _ClipWorker(wav, t0, t1, camera, cfg, self.chk_masks.isChecked(), parent=self)
        w.done.connect(lambda clip, w=w: self._on_loaded(w, clip))
        w.rfid_done.connect(lambda r, w=w: self._on_rfid(w, r))
        w.masks_done.connect(lambda m, w=w: self._on_masks(w, m))
        w.failed.connect(lambda msg, w=w: self._on_failed(w, msg))
        w.progress.connect(lambda msg, w=w: self._on_progress(w, msg))
        w.finished.connect(lambda w=w: self._on_worker_finished(w))
        self._worker = w
        w.start()

    def _on_progress(self, w, msg):
        if w is self._worker:
            self.lbl_status.setText(msg)

    def _on_worker_finished(self, w):
        if w is self._worker:
            self._worker = None
            self._render()

    def _on_failed(self, w, msg):
        if w is not self._worker:
            return
        self._show_message(msg)
        self.lbl_status.setText("")

    def _on_loaded(self, w, clip: VideoClip):
        if w is not self._worker:
            return                               # superseded by a newer request
        self._clip = clip
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(clip.frames) - 1)
        self.slider.blockSignals(False)
        self.slider.setEnabled(True)
        start = self._request[1] if self._request else 0.0
        pos = getattr(getattr(self._main, 'spectrogram', None), 'playback_position', None)
        self.show_time(pos if pos is not None else start + 1.0)

    def _on_rfid(self, w, rfid: Optional[ZoneRFID]):
        if w is not self._worker or self._request is None:
            return
        self._rfid = rfid
        _, t0, t1, _ = self._request
        self.rfid_strip.set_data(rfid, t0, t1)
        if self._clip is not None and self._index >= 0:
            self.rfid_strip.set_now(float(self._clip.times_s[self._index]))
        self.lbl_rfid.setText(self._rfid_summary(rfid))

    def _rfid_summary(self, rfid: Optional[ZoneRFID]) -> str:
        if rfid is None:
            return "RFID: none named in video_qc.json."
        ants = ", ".join(str(a) for a in rfid.antennas) or "?"
        parts = []
        b, rd = rfid.bouts, rfid.reads
        codes = sorted(set(list(b["code"]) if b is not None and len(b) else []) |
                       set(list(rd["code"]) if rd is not None and len(rd) else []))
        sex = dict(zip(b["code"], b["sex"])) if b is not None and len(b) else {}
        for c in codes:
            n = int((rd["code"] == c).sum()) if rd is not None and len(rd) else 0
            s = sex.get(c)
            parts.append(f"{c}{f' ({s})' if isinstance(s, str) else ''}: {n} read{'s' if n != 1 else ''}")
        who = "; ".join(parts) if parts else "nobody"
        sync = ("" if rfid.offset_s else
                "  ·  RFID clock not yet synced to the audio (its clock was set by a "
                "Bluetooth push; offset to be estimated)")
        return f"RFID, zone {rfid.zone} (antenna {ants}), this stretch: {who}{sync}"

    def _on_masks(self, w, payload):
        if w is not self._worker:
            return
        self._masks, self._mask_every = payload
        self._render()

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
        self._show_index(self._clip.index_at(t_s))

    def _show_index(self, i: int):
        if self._clip is None or not (0 <= i < len(self._clip.frames)):
            return
        self._index = i
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.rfid_strip.set_now(float(self._clip.times_s[i]))
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
        dets = masks_at(self._masks, self._index, self._mask_every) if self._masks else None
        p = QPainter(pm)
        try:
            p.setRenderHint(QPainter.Antialiasing)
            if dets and self.chk_masks.isChecked():
                sc = pm.width() / float(w)
                for j, (cnt, score) in enumerate(dets):
                    poly = QPolygonF([QPointF(x * sc, y * sc) for x, y in cnt])
                    p.setPen(QPen(QColor(*MASK_COLORS[j % len(MASK_COLORS)]), 2))
                    p.drawPolygon(poly)
            if self._mark and self._mark[0] - 0.034 <= t <= self._mark[1] + 0.034:
                p.setPen(QPen(QColor(230, 40, 40), 6))
                p.drawRect(QRectF(3, 3, pm.width() - 6, pm.height() - 6))
            f = QFont()
            f.setPointSize(9)
            f.setBold(True)
            p.setFont(f)
            mice = "" if dets is None else f"  ·  mice seen: {len(dets)}"
            text = (f"{clip.camera}  ·  t = {t:.2f} s  ·  {when:%Y-%m-%d %H:%M:%S}."
                    f"{when.microsecond // 1000:03d}  ·  {name} frame {k}{mice}")
            p.setPen(QColor(0, 0, 0))
            p.drawText(QRectF(9, 7, pm.width() - 14, 20), Qt.AlignLeft, text)
            p.setPen(QColor(255, 230, 90))
            p.drawText(QRectF(8, 6, pm.width() - 14, 20), Qt.AlignLeft, text)
        finally:
            p.end()
        self.lbl_frame.setText("")
        self.lbl_frame.setPixmap(pm)
        if self._worker is None:
            off = f"; clock offset {clip.offset_s:+.2f} s" if clip.offset_s else ""
            step = (len(clip.frames) > 1 and float(np.median(np.diff(clip.times_s))) or 0.0)
            masks = ""
            if self._request and (self._request[3].mask_models or {}).get(clip.camera):
                masks = f"  ·  mice outlined every {self._mask_every} frame(s)" if self._masks else ""
            self.lbl_status.setText(
                f"Frame {self._index + 1} of {len(clip.frames)}"
                f"{f' (every {step:.2f} s)' if step > 0.1 else ''}{masks}  ·  video clock "
                f"from the file names{off}; compare with the time the camera burns into "
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
