"""Fusion View: scroll through a multi-camera tracking run with the names over the animals, see the UWB data
beside the video, and correct the names where they are wrong.

Opens the ``identity.json`` that ``mtt_identity`` ran on and shows the camera grid (``identity_view``), drawn live,
nothing rendered first:

* **Play and scroll**: slider, play/pause at 0.25-8x, step by an analysed frame, a second or ten; a timeline with a
  row per animal (which camera names it, faint where unsure; click to go there).
* **Overlays** (toggles): masks, keypoints, names, video tails, UWB positions (rings on the overhead camera; who UWB
  puts in each box on the zone cameras), UWB tails, and a floor map with the video and UWB positions side by side.
  The legend gives each animal's distance between its video and UWB positions.
* **Go to** the moments worth checking: unsure names, a track whose name changes (a possible swap), handoffs between
  cameras, long unnamed tracks, your corrections.
* **Correct**: click an animal: its track, frames, name, anchors and its UWB distance to every animal; name the
  stretch (or from here on, or up to here) as an animal or as not an animal. Corrections are saved to
  ``corrections.csv`` beside ``identities.csv``; **Re-solve names** runs ``mtt_identity`` again with them held (a
  correction carries along whatever its chain links to) and redraws.

    python -m fnt.videoTracking.fusion_view <identity.json>

Keys: Space play/pause; Left/Right step; Shift+Left/Right 1 s; Ctrl+Left/Right 10 s; N / P next / previous moment
of the chosen kind.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from PyQt5.QtCore import QPoint, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton, QSlider, QSplitter, QVBoxLayout,
    QWidget,
)

from .identity_view import Overlays, Scene
from .mtt_identity import NOT_ANIMAL, corrections_path, read_corrections

CAM_COLOURS = [(90, 160, 255), (255, 170, 60), (120, 210, 120), (230, 110, 200), (200, 200, 90), (110, 220, 230)]
JUMPS = ["unsure name", "name change on a track", "handoff between cameras", "long unnamed track", "correction"]


def _bgra(bgr: np.ndarray):
    """(QImage, buffer): the picture as Qt's native 32-bit format, on the buffer (keep it while the QImage is
    used); converting once here is far cheaper than an RGB888 copy and lets Qt paint it without converting."""
    import cv2
    buf = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    h, w = buf.shape[:2]
    return QImage(buf.data, w, h, 4 * w, QImage.Format_RGB32), buf


class _Kept:
    """Pictures kept by key up to a size in MB, the least recently used dropped first (thread-safe)."""

    def __init__(self, mb: float):
        import threading
        from collections import OrderedDict
        self.max, self.bytes = mb * 2 ** 20, 0
        self.items: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            img = self.items.get(key)
            if img is not None:
                self.items.move_to_end(key)
            return img

    def put(self, key, img):
        with self.lock:
            if key in self.items:
                return
            self.items[key] = img
            self.bytes += img.nbytes
            while self.bytes > self.max and len(self.items) > 1:
                _, old = self.items.popitem(last=False)
                self.bytes -= old.nbytes

    def clear(self):
        with self.lock:
            self.items.clear()
            self.bytes = 0


class Player:
    """Frame n of every camera's clip, each scaled to its cell.

    One reader thread per camera, so the cameras decode side by side. Decoded frames are kept whole (``WHOLE_MB``)
    and scaled to their cells (``CELL_MB``): redrawing a frame (an overlay toggled) decodes nothing and zooming (a
    new cell size) is only a resize. Reads on in order going forward; a short step back decodes the analysed
    frames before it in the same pass (the decoder has to start from an earlier key frame anyway), so the next
    steps back are instant; ``prefetch`` decodes the next frame while this one is drawn."""
    WHOLE_MB = 1200
    CELL_MB = 600
    AHEAD = 45              # frames: read on instead of seeking
    FILL = 24               # analysed frames decoded before a step back
    BACK = 301              # a step back of up to this many frames fills; a longer jump only seeks

    def __init__(self, clips: Dict[str, str], hold: int = 1):
        import cv2
        self.cv2 = cv2
        self.caps = {c: cv2.VideoCapture(p) for c, p in clips.items()}
        self.pos = {c: 0 for c in clips}
        self.hold = max(1, int(hold))
        self.pools = {c: ThreadPoolExecutor(max_workers=1) for c in clips}     # one decoder per camera
        self.whole = _Kept(self.WHOLE_MB)                                        # (cam, n) -> frame
        self.cells = _Kept(self.CELL_MB)                                         # (cam, n, w, h) -> cell
        self.pending: Dict[tuple, object] = {}

    def _fit(self, fr, size):
        w, h = size
        if fr.shape[1] == w and fr.shape[0] == h:
            return fr
        return self.cv2.resize(fr, (w, h), interpolation=self.cv2.INTER_AREA)

    # decoding (on the camera's own thread)
    def _decode(self, cam, n, size):
        key = (cam, n) + tuple(size)
        img = self.cells.get(key)
        if img is not None:
            return img
        fr = self.whole.get((cam, n))
        if fr is None:
            cap, pos, fill = self.caps[cam], self.pos[cam], ()
            if not (pos <= n <= pos + self.AHEAD):
                start = n
                if 0 < pos - n <= self.BACK:                    # a step back: the frames before it too
                    start = max(0, n - self.FILL * self.hold)
                    fill = set(range(n - self.hold, start - 1, -self.hold))
                cap.set(self.cv2.CAP_PROP_POS_FRAMES, start)
                self.pos[cam] = start
            while self.pos[cam] < n:
                p = self.pos[cam]
                if p in fill:
                    ok, f = cap.read()
                    if ok:
                        self.whole.put((cam, p), f)
                else:
                    cap.grab()
                self.pos[cam] = p + 1
            ok, fr = cap.read()
            self.pos[cam] = n + 1 if ok else -10 ** 9
            if not ok:
                return None
            self.whole.put((cam, n), fr)
        img = self._fit(fr, size)
        self.cells.put(key, img)
        return img

    def _submit(self, cam, n, size):
        key = (cam, n) + tuple(size)
        fut = self.pending.get(key)
        if fut is None:
            fut = self.pools[cam].submit(self._decode, cam, n, size)
            self.pending[key] = fut
            fut.add_done_callback(lambda _f, k=key: self.pending.pop(k, None))
        return fut

    def read(self, n, sizes: Dict[str, tuple]) -> Dict[str, np.ndarray]:
        """{camera: frame n scaled to sizes[camera] (w, h)} for the cameras in sizes."""
        out, futs = {}, {}
        for cam, size in sizes.items():
            if cam not in self.caps:
                continue
            img = self.cells.get((cam, n) + tuple(size))
            if img is not None:
                out[cam] = img
            else:
                futs[cam] = self._submit(cam, n, size)
        for cam, fut in futs.items():
            out[cam] = fut.result()
        return out

    def prefetch(self, n, sizes: Dict[str, tuple]):
        """Start decoding frame n (playing: the next frame while this one is drawn)."""
        for cam, size in sizes.items():
            if cam in self.caps and self.cells.get((cam, n) + tuple(size)) is None:
                self._submit(cam, n, size)

    def close(self):
        for p in self.pools.values():
            p.shutdown(wait=True, cancel_futures=True)
        for c in self.caps.values():
            c.release()
        self.whole.clear()
        self.cells.clear()


class GridView(QWidget):
    """The grid, zoomable: pinch (trackpad or touch screen), Ctrl+scroll or the mouse wheel zoom about the
    pointer; drag (or two-finger scroll on a trackpad) to move about; a click without dragging picks an animal."""
    clicked = pyqtSignal(float, float, QPoint)            # canvas x, y; global position
    zoomed = pyqtSignal(float)
    MAX_ZOOM = 12.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img: Optional[QImage] = None
        self.zoom = 1.0
        self.centre = (0.5, 0.5)                           # the middle of the view, as a share of the canvas
        self._press = None
        self._dragged = False
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background: black;")
        self.setFocusPolicy(Qt.StrongFocus)
        self.grabGesture(Qt.PinchGesture)

    def set_image(self, img: QImage):
        self.img = img
        self.update()

    def set_frame(self, bgr: np.ndarray):
        """Show a BGR picture (kept as Qt's native format; the buffer lives as long as the image shown)."""
        self.img, self._buf = _bgra(bgr)
        self.update()

    # geometry: the fitted rectangle on screen, and the part of the canvas shown in it
    def _target(self) -> QRectF:
        if self.img is None:
            return QRectF()
        k = min(self.width() / self.img.width(), self.height() / self.img.height())
        w, h = self.img.width() * k, self.img.height() * k
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def _source(self) -> QRectF:
        W, H = self.img.width(), self.img.height()
        w, h = W / self.zoom, H / self.zoom
        x = float(np.clip(self.centre[0] * W - w / 2, 0, W - w))
        y = float(np.clip(self.centre[1] * H - h / 2, 0, H - h))
        return QRectF(x, y, w, h)

    def _to_canvas(self, pos):
        r, s = self._target(), self._source()
        return (s.x() + (pos.x() - r.x()) * s.width() / r.width(),
                s.y() + (pos.y() - r.y()) * s.height() / r.height())

    def _set_view(self, zoom, centre):
        self.zoom = float(np.clip(zoom, 1.0, self.MAX_ZOOM))
        hw, hh = 0.5 / self.zoom, 0.5 / self.zoom
        self.centre = (float(np.clip(centre[0], hw, 1 - hw)), float(np.clip(centre[1], hh, 1 - hh)))
        self.zoomed.emit(self.zoom)
        self.update()

    def zoom_by(self, factor, pos=None):
        """Zoom by factor, keeping the canvas point under pos (default: the middle of the view) where it is."""
        if self.img is None:
            return
        r = self._target()
        pos = pos if pos is not None and r.contains(pos) else r.center()
        qx, qy = self._to_canvas(pos)
        z = float(np.clip(self.zoom * factor, 1.0, self.MAX_ZOOM))
        W, H = self.img.width(), self.img.height()
        w, h = W / z, H / z
        x0 = qx - (pos.x() - r.x()) * w / r.width()
        y0 = qy - (pos.y() - r.y()) * h / r.height()
        self._set_view(z, ((x0 + w / 2) / W, (y0 + h / 2) / H))

    def fit(self):
        if self.img is not None:
            self._set_view(1.0, (0.5, 0.5))

    def pan_pixels(self, dx, dy):
        """Move the picture by dx, dy screen pixels."""
        if self.img is None or self.zoom <= 1.0:
            return
        r = self._target()
        self._set_view(self.zoom, (self.centre[0] - dx / (r.width() * self.zoom),
                                   self.centre[1] - dy / (r.height() * self.zoom)))

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self.img is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform, not self._dragged)     # smooth once the drag stops
            p.drawImage(self._target(), self.img, self._source())

    # input
    def event(self, e):
        from PyQt5.QtCore import QEvent
        if e.type() == QEvent.NativeGesture and e.gestureType() == Qt.ZoomNativeGesture:    # macOS trackpad pinch
            self.zoom_by(1.0 + e.value(), e.pos())
            return True
        if e.type() == QEvent.Gesture:                                                       # touch screen pinch
            g = e.gesture(Qt.PinchGesture)
            if g is not None:
                self.zoom_by(g.scaleFactor(), self.mapFromGlobal(g.centerPoint().toPoint()))
                return True
        return super().event(e)

    def wheelEvent(self, e):
        pix = e.pixelDelta()
        if not (e.modifiers() & Qt.ControlModifier) and not pix.isNull() and e.phase() != Qt.NoScrollPhase:
            self.pan_pixels(pix.x(), pix.y())               # two-finger scroll on a trackpad: move about
            return
        steps = e.angleDelta().y() / 120.0
        if steps:
            self.zoom_by(1.25 ** steps, e.pos())

    def mousePressEvent(self, e):
        if e.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._press, self._dragged = e.pos(), False

    def mouseMoveEvent(self, e):
        if self._press is None:
            return
        d = e.pos() - self._press
        if self._dragged or abs(d.x()) + abs(d.y()) > 4:
            self._dragged = True
            self.setCursor(Qt.ClosedHandCursor)
            self.pan_pixels(d.x(), d.y())
            self._press = e.pos()

    def mouseReleaseEvent(self, e):
        press, dragged = self._press, self._dragged
        self._press, self._dragged = None, False
        self.unsetCursor()
        if dragged:
            self.update()
        if press is None or dragged or e.button() != Qt.LeftButton:
            return
        r = self._target()
        if self.img is None or not r.contains(e.pos()):
            return
        x, y = self._to_canvas(e.pos())
        self.clicked.emit(x, y, e.globalPos())

    def mouseDoubleClickEvent(self, e):
        self.mousePressEvent(e)


class Timeline(QWidget):
    """A row per animal: where it is named (coloured by camera, faint where unsure); click or drag to go there."""
    seek = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: List[str] = []
        self.spans: List[tuple] = []          # (row, f0, f1, cam index, unsure)
        self.marks: List[int] = []
        self.n = 1
        self.playhead = 0
        self.setMinimumHeight(40)

    def set_data(self, scene: Scene, marks: List[int]):
        self.rows = list(scene.animals)
        cams = list(scene.cells)
        self.n = max(scene.n_frames, 1)
        self.spans = []
        for r in scene.ids.itertuples():
            if isinstance(r.animal, str) and r.animal in self.rows:
                self.spans.append((self.rows.index(r.animal), int(r.frame_first), int(r.frame_last),
                                   cams.index(r.camera) if r.camera in cams else 0, bool(r.unsure)))
        self.marks = marks
        self.cams = cams
        self.setFixedHeight(18 + 16 * len(self.rows))
        self.update()

    def set_playhead(self, n):
        self.playhead = n
        self.update()

    def _x(self, f):
        return 90 + (self.width() - 100) * f / self.n

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(25, 25, 25))
        p.setPen(QColor(200, 200, 200))
        for i, a in enumerate(self.rows):
            p.drawText(4, 28 + 16 * i, a)
        for row, f0, f1, ci, unsure in self.spans:
            r, g, b = CAM_COLOURS[ci % len(CAM_COLOURS)]
            col = QColor(r, g, b, 90 if unsure else 255)
            x0, x1 = self._x(f0), max(self._x(f1), self._x(f0) + 1)
            p.fillRect(QRectF(x0, 16 + 16 * row, x1 - x0, 13), col)
        p.setPen(QPen(QColor(230, 230, 230, 120), 1))
        for f in self.marks:
            x = self._x(f)
            p.drawLine(int(x), 2, int(x), 12)
        x = 4
        for ci, c in enumerate(getattr(self, "cams", [])):
            r, g, b = CAM_COLOURS[ci % len(CAM_COLOURS)]
            p.fillRect(QRectF(self.width() - 60 * (len(self.cams) - ci), 2, 12, 10), QColor(r, g, b))
            p.setPen(QColor(200, 200, 200))
            p.drawText(int(self.width() - 60 * (len(self.cams) - ci) + 15), 11, c)
        p.setPen(QPen(QColor(255, 60, 60), 2))
        xp = int(self._x(self.playhead))
        p.drawLine(xp, 0, xp, self.height())

    def mousePressEvent(self, e):
        self._go(e.pos().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._go(e.pos().x())

    def _go(self, x):
        f = int(np.clip((x - 90) / max(self.width() - 100, 1) * self.n, 0, self.n - 1))
        self.seek.emit(f)


class Solver(QThread):
    done = pyqtSignal(object, object, str)
    failed = pyqtSignal(str)

    def __init__(self, cfg, corrections, parent=None):
        super().__init__(parent)
        self.cfg, self.corrections = cfg, corrections

    def run(self):
        from .mtt_identity import Identity
        lines = []
        try:
            idn = Identity(self.cfg, self.corrections)
            res = idn.run(log=lines.append)
            self.done.emit(res, idn.links_, "\n".join(lines))
        except Exception as e:                                   # shown to the person reviewing
            self.failed.emit(f"{type(e).__name__}: {e}")


class ReviewWindow(QMainWindow):
    def __init__(self, cfg_path: str):
        super().__init__()
        self.cfg_path = cfg_path
        self.scene = Scene(cfg_path, "1080")
        self.player = Player({c: v["clip"] for c, v in self.scene.cams.items() if c in self.scene.cells},
                             hold=self.scene.hold)
        self.corr_path = corrections_path(self.scene.cfg, cfg_path)
        self.corrections = read_corrections(self.corr_path)
        lp = os.path.join(self.scene.out_dir, "links.csv")
        self.links = pd.read_csv(lp) if os.path.exists(lp) else pd.DataFrame()
        self.n = 0
        self._want: Optional[int] = None
        self._shown: set = set()
        self.playing = False
        self.dirty = False
        self.solver: Optional[Solver] = None
        self.setWindowTitle(f"Fusion View - {self.scene.title or os.path.basename(cfg_path)}")
        self._build()
        self._refresh_names()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.show_frame(0)

    # ---------------------------------------------------------------- layout
    def _build(self):
        self.view = GridView()
        self.view.clicked.connect(self._clicked)
        self.timeline = Timeline()
        self.timeline.seek.connect(self.request_frame)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.scene.n_frames - 1)
        self.slider.sliderReleased.connect(lambda: self.show_frame(self.slider.value()))
        self.slider.valueChanged.connect(self._slider_moved)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.speed = QComboBox()
        self.speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x", "8x"])
        self.speed.setCurrentText("1x")
        self.speed.currentTextChanged.connect(lambda _: self._restart_clock())
        self.time_lbl = QLabel()
        bar = QHBoxLayout()
        for txt, d in (("-10 s", -300), ("-1 s", -30), ("<", -1), (">", 1), ("+1 s", 30), ("+10 s", 300)):
            b = QPushButton(txt)
            b.setFixedWidth(52)
            b.clicked.connect(lambda _, d=d: self.step(d))
            bar.addWidget(b)
            if txt == "<":
                bar.addWidget(self.play_btn)
        bar.addWidget(self.speed)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.time_lbl)
        for txt, tip, fn in (("-", "Zoom out (-)", lambda: self.view.zoom_by(0.8)),
                             ("Fit", "Whole grid (0)", self.view.fit),
                             ("+", "Zoom in (+; or pinch, Ctrl+scroll, mouse wheel)", lambda: self.view.zoom_by(1.25))):
            b = QPushButton(txt)
            b.setFixedWidth(40)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        self.zoom_lbl = QLabel("1.0x")
        self.zoom_lbl.setFixedWidth(46)
        bar.addWidget(self.zoom_lbl)
        self.view.zoomed.connect(self._zoomed)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.view, 1)
        lv.addWidget(self.timeline)
        lv.addLayout(bar)

        side = QWidget()
        side.setMaximumWidth(420)
        sv = QVBoxLayout(side)
        ov = QGroupBox("Show")
        ovl = QVBoxLayout(ov)
        self.checks = {}
        for key, txt, on in (("masks", "Masks", True), ("keypoints", "Keypoints", True), ("names", "Names", True),
                             ("track_tails", "Video tails", False), ("uwb", "UWB positions", True),
                             ("uwb_tails", "UWB tails", False), ("floor", "Floor map (video + UWB)", True),
                             ("legend", "Legend", True)):
            cb = QCheckBox(txt)
            cb.setChecked(on)
            cb.toggled.connect(lambda _: self.show_frame(self.n))
            ovl.addWidget(cb)
            self.checks[key] = cb
        row = QHBoxLayout()
        row.addWidget(QLabel("Tail length (s)"))
        self.tail = QDoubleSpinBox()
        self.tail.setRange(0.5, 60)
        self.tail.setValue(3)
        self.tail.valueChanged.connect(lambda _: self.show_frame(self.n))
        row.addWidget(self.tail)
        ovl.addLayout(row)
        self.analysed_only = QCheckBox(f"Play analysed frames only (every {self.scene.hold})")
        self.analysed_only.setChecked(self.scene.hold > 1)
        self.analysed_only.setVisible(self.scene.hold > 1)
        ovl.addWidget(self.analysed_only)
        sv.addWidget(ov)

        jg = QGroupBox("Go to")
        jl = QVBoxLayout(jg)
        self.jump_kind = QComboBox()
        self.jump_kind.addItems(JUMPS)
        self.jump_kind.currentTextChanged.connect(lambda _: self._update_jump_label())
        jl.addWidget(self.jump_kind)
        jr = QHBoxLayout()
        prev_b, next_b = QPushButton("< Previous (P)"), QPushButton("Next (N) >")
        prev_b.clicked.connect(lambda: self.jump(-1))
        next_b.clicked.connect(lambda: self.jump(1))
        jr.addWidget(prev_b)
        jr.addWidget(next_b)
        jl.addLayout(jr)
        self.jump_lbl = QLabel()
        jl.addWidget(self.jump_lbl)
        sv.addWidget(jg)

        ig = QGroupBox("Selected track (click an animal)")
        il = QVBoxLayout(ig)
        self.info = QLabel("-")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        il.addWidget(self.info)
        sv.addWidget(ig)

        cg = QGroupBox("Corrections")
        cl = QVBoxLayout(cg)
        self.corr_list = QListWidget()
        self.corr_list.itemDoubleClicked.connect(lambda it: self.show_frame(int(it.data(Qt.UserRole))))
        cl.addWidget(self.corr_list)
        cr = QHBoxLayout()
        rm = QPushButton("Remove")
        rm.clicked.connect(self._remove_correction)
        self.solve_btn = QPushButton("Re-solve names")
        self.solve_btn.clicked.connect(self.resolve)
        cr.addWidget(rm)
        cr.addWidget(self.solve_btn)
        cl.addLayout(cr)
        self.status = QLabel()
        self.status.setWordWrap(True)
        cl.addWidget(self.status)
        sv.addWidget(cg, 1)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setSizes([1500, 360])
        self.setCentralWidget(split)
        m = self.menuBar().addMenu("File")
        a = QAction("Open identity.json...", self)
        a.triggered.connect(self._open)
        m.addAction(a)
        self._fill_corrections()

    # ---------------------------------------------------------------- frames
    def overlays(self) -> Overlays:
        return Overlays(**{k: cb.isChecked() for k, cb in self.checks.items()}, tail_s=self.tail.value())

    def _cell_sizes(self) -> Dict[str, tuple]:
        """{camera: cell (w, h)} for the cameras in view: zoomed in, the others are not decoded."""
        v, sc = self.view, self.scene
        cells = sc.cells
        if v.zoom > 1.0:
            hw = hh = 0.5 / v.zoom
            x0, x1 = (v.centre[0] - hw) * sc.W, (v.centre[0] + hw) * sc.W
            y0, y1 = (v.centre[1] - hh) * sc.H, (v.centre[1] + hh) * sc.H
            cells = {c: r for c, r in cells.items() if r[0] < x1 and r[0] + r[2] > x0 and r[1] < y1 and r[1] + r[3] > y0}
        return {c: (w, h) for c, (_, _, w, h) in cells.items()}

    def request_frame(self, n):
        """Show frame n once the events waiting now are handled: dragging along the timeline draws the latest
        position only, not every position passed."""
        first = self._want is None
        self._want = int(n)
        if first:
            QTimer.singleShot(0, self._show_wanted)

    def _show_wanted(self):
        n, self._want = self._want, None
        if n is not None:
            self.show_frame(n)

    def show_frame(self, n):
        n = int(np.clip(n, 0, self.scene.n_frames - 1))
        self.n = n
        sizes = self._cell_sizes()
        self._shown = set(sizes)
        img = self.scene.draw(self.scene.compose(self.player.read(n, sizes)), n, self.overlays())
        self.view.set_frame(img)
        self.timeline.set_playhead(n)
        self.slider.blockSignals(True)
        self.slider.setValue(n)
        self.slider.blockSignals(False)
        self.time_lbl.setText(f"{self.scene.clock(n):%H:%M:%S}  frame {n}")

    def _zoomed(self, z):
        """Draw at 4K once zoomed in, so the cameras and labels stay sharp (1080 whole, for speed)."""
        self.zoom_lbl.setText(f"{z:.1f}x")
        size = "4k" if z > 1.4 else "1080"
        if size != self.scene.size:
            self.scene.set_size(size)
            self.show_frame(self.n)
        elif not set(self._cell_sizes()) <= self._shown:          # moved onto a camera not drawn yet
            self.show_frame(self.n)

    def _slider_moved(self, v):
        if self.slider.isSliderDown():                           # dragging: the time only; the frame on release
            self.time_lbl.setText(f"{self.scene.clock(v):%H:%M:%S}  frame {v}")
        else:
            self.show_frame(v)

    def step(self, d):
        if abs(d) == 1 and self.scene.hold > 1:
            d *= self.scene.hold
        self.show_frame(self.n + d)

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing:
            self._restart_clock()
            self.timer.start(10)
        else:
            self.timer.stop()

    def _restart_clock(self):
        self.t0, self.f0 = time.time(), self.n

    def _tick(self):
        rate = float(self.speed.currentText().rstrip("x")) * self.scene.fps
        target = self.f0 + int((time.time() - self.t0) * rate)
        if self.analysed_only.isChecked():
            target -= target % self.scene.hold
        if target >= self.scene.n_frames:
            self.toggle_play()
            return
        if target > self.n:
            self.show_frame(target)
            # the nearest frame that can come next, decoded while this one is on screen (never one beyond it: a
            # later frame than the one asked for would mean going back)
            nxt = target + (self.scene.hold if self.analysed_only.isChecked() else 1)
            if nxt < self.scene.n_frames:
                self.player.prefetch(nxt, self._cell_sizes())

    # ---------------------------------------------------------------- names and moments
    def _refresh_names(self):
        ids = self.scene.ids
        self.targets = {k: [] for k in JUMPS}
        if len(ids):
            ok = ids[ids.animal.astype(str) != ""]
            self.targets["unsure name"] = sorted(ok[ok.unsure.astype(bool) & (ok.detections >= 10)].frame_first)
            for _, g in ids.sort_values("frame_first").groupby(["camera", "object_id"]):
                a = g.animal.astype(str).to_numpy()
                for k in range(1, len(g)):
                    if a[k] != a[k - 1] and a[k] and a[k - 1]:
                        self.targets["name change on a track"].append(int(g.frame_first.iloc[k]))
            self.targets["name change on a track"].sort()
            un = ids[(ids.animal.astype(str) == "") & (ids.detections >= 30)]
            self.targets["long unnamed track"] = sorted(un.frame_first)
        if len(self.links) and "kept" in self.links:
            h = self.links[self.links.kind.str.startswith("handoff") & self.links.kept.astype(bool)]
            self.targets["handoff between cameras"] = sorted(h.frame_b)
        self.targets["correction"] = sorted(self.corrections.frame_first.astype(int)) if len(self.corrections) else []
        self.timeline.set_data(self.scene, self.targets["name change on a track"])
        self._update_jump_label()

    def _update_jump_label(self):
        t = self.targets.get(self.jump_kind.currentText(), [])
        before = sum(1 for f in t if f <= self.n)
        self.jump_lbl.setText(f"{before} of {len(t)} before here")

    def jump(self, d):
        t = self.targets.get(self.jump_kind.currentText(), [])
        if not t:
            return
        if d > 0:
            nxt = [f for f in t if f > self.n]
            f = nxt[0] if nxt else t[0]
        else:
            prv = [f for f in t if f < self.n]
            f = prv[-1] if prv else t[-1]
        self.show_frame(int(f))
        self._update_jump_label()

    # ---------------------------------------------------------------- selecting and correcting
    def _clicked(self, x, y, gpos):
        hit = self.scene.hit(self.n, x, y)
        if not hit:
            return
        cam, obj, nd = hit["cam"], hit["obj"], hit["frame"]
        f0, f1 = hit["tracklet"] or (nd, nd)
        d = self.scene.uwb_distances(cam, obj, f0, f1)
        row = self.scene.ids[(self.scene.ids.camera == cam) & (self.scene.ids.object_id == obj)
                             & (self.scene.ids.frame_first == f0)]
        anchors = row.anchors.iloc[0] if len(row) and "anchors" in row else ""
        margin = row.margin.iloc[0] if len(row) and "margin" in row else ""
        name = hit["animal"] or "unnamed"
        self.info.setText(
            f"<b>{cam} track {obj}</b>, frames {f0}-{f1} ({(f1 - f0) / self.scene.fps:.1f} s)<br>"
            f"name: <b>{name}</b>{' (unsure)' if hit['unsure'] else ''}{' (corrected)' if hit['fixed'] else ''}"
            f"{f', margin {margin}' if margin != '' else ''}<br>anchors: {anchors or 'none'}<br>"
            "mean UWB distance over the stretch:<br>" +
            "<br>".join(f"&nbsp;&nbsp;{a}: {v:.2f} m" for a, v in sorted(d.items(), key=lambda kv: kv[1])))
        menu = QMenu(self)
        head = menu.addAction(f"{cam} track {obj}: {name}, frames {f0}-{f1}")
        head.setEnabled(False)
        menu.addSeparator()
        for title, a, b in (("Name this stretch as", f0, f1), ("From here on", nd, f1), ("Up to here", f0, nd)):
            sub = menu.addMenu(title)
            for an in self.scene.animals:
                act = sub.addAction(self.scene.names.get(an, an))
                act.triggered.connect(lambda _, an=an, a=a, b=b: self.add_correction(cam, obj, a, b, an))
            sub.addSeparator()
            act = sub.addAction("Not an animal")
            act.triggered.connect(lambda _, a=a, b=b: self.add_correction(cam, obj, a, b, NOT_ANIMAL))
        menu.addSeparator()
        clr = menu.addAction("Remove corrections on this track")
        clr.triggered.connect(lambda: self.clear_corrections(cam, obj))
        menu.exec_(gpos)

    def add_correction(self, cam, obj, f0, f1, animal):
        c = self.corrections
        # a new correction replaces older ones on the same frames of the same track
        keep = ~((c.camera == cam) & (c.object_id.astype(int) == int(obj)) & (c.frame_first.astype(int) >= f0)
                 & (c.frame_last.astype(int) <= f1))
        row = pd.DataFrame([dict(camera=cam, object_id=int(obj), frame_first=int(f0), frame_last=int(f1),
                                 animal=animal)])
        self.corrections = pd.concat([c[keep], row], ignore_index=True)
        self._save_corrections()

    def clear_corrections(self, cam, obj):
        c = self.corrections
        self.corrections = c[~((c.camera == cam) & (c.object_id.astype(int) == int(obj)))].reset_index(drop=True)
        self._save_corrections()

    def _remove_correction(self):
        r = self.corr_list.currentRow()
        if r >= 0:
            self.corrections = self.corrections.drop(self.corrections.index[r]).reset_index(drop=True)
            self._save_corrections()

    def _save_corrections(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.corr_path)), exist_ok=True)
        self.corrections.to_csv(self.corr_path, index=False)
        self.dirty = True
        self._fill_corrections()
        self.status.setText(f"{len(self.corrections)} correction(s) saved; Re-solve names to apply them.")

    def _fill_corrections(self):
        self.corr_list.clear()
        for r in self.corrections.itertuples():
            it = QListWidgetItem(f"{r.camera} #{r.object_id}  {self.scene.clock(int(r.frame_first)):%H:%M:%S}"
                                 f"-{self.scene.clock(int(r.frame_last)):%H:%M:%S}  ->  {r.animal}")
            it.setData(Qt.UserRole, int(r.frame_first))
            self.corr_list.addItem(it)

    def resolve(self):
        if self.solver and self.solver.isRunning():
            return
        self.solve_btn.setEnabled(False)
        self.status.setText("Re-solving names... (about half a minute for a 35-min test)")
        self.solver = Solver(self.scene.cfg, self.corrections, self)
        self.solver.done.connect(self._solved)
        self.solver.failed.connect(self._solve_failed)
        self.solver.start()

    def _solved(self, ids, links, log):
        ids.to_csv(os.path.join(self.scene.out_dir, "identities.csv"), index=False)
        links.to_csv(os.path.join(self.scene.out_dir, "links.csv"), index=False)
        self.scene.set_identities(ids)
        self.links = links
        self.dirty = False
        self._refresh_names()
        self.show_frame(self.n)
        self.solve_btn.setEnabled(True)
        self.status.setText(log)

    def _solve_failed(self, msg):
        self.solve_btn.setEnabled(True)
        self.status.setText("Re-solve failed: " + msg)

    # ---------------------------------------------------------------- misc
    def keyPressEvent(self, e):
        k, mod = e.key(), e.modifiers()
        big = 300 if mod & Qt.ControlModifier else 30 if mod & Qt.ShiftModifier else 1
        if k == Qt.Key_Space:
            self.toggle_play()
        elif k == Qt.Key_Right:
            self.step(big)
        elif k == Qt.Key_Left:
            self.step(-big)
        elif k == Qt.Key_N:
            self.jump(1)
        elif k == Qt.Key_P:
            self.jump(-1)
        elif k in (Qt.Key_Plus, Qt.Key_Equal):
            self.view.zoom_by(1.25)
        elif k in (Qt.Key_Minus, Qt.Key_Underscore):
            self.view.zoom_by(0.8)
        elif k == Qt.Key_0:
            self.view.fit()
        else:
            super().keyPressEvent(e)

    def _open(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open identity.json", os.path.dirname(self.cfg_path), "*.json")
        if p:
            w = ReviewWindow(p)
            w.show()
            self._next = w

    def closeEvent(self, e):
        if self.dirty and QMessageBox.question(
                self, "Fusion View", "Corrections are saved but not applied (Re-solve names). Close anyway?"
        ) != QMessageBox.Yes:
            e.ignore()
            return
        self.timer.stop()
        self.player.close()
        super().closeEvent(e)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", nargs="?", help="the identity.json mtt_identity ran on")
    a = ap.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    path = a.config or QFileDialog.getOpenFileName(None, "Open identity.json", "", "*.json")[0]
    if not path:
        return 0
    w = ReviewWindow(path)
    w.resize(1800, 1000)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
