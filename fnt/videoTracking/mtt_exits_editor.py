"""Drawing each camera's exits (mtt_exits) on its median frame.

``ExitsWindow`` lists every folder x camera; unchecked ones first, then by the given focus
order. Click the corners round an exit (the mouth of the entrance tube), and close it as in
the ROI editor (first corner again, double-click, right-click or Enter). Corners can be
dragged. Space confirms the camera's exits as they are and moves on; any edit is saved as
checked when you leave the camera.
"""
from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget, QPushButton,
                             QVBoxLayout, QWidget)

from .mtt_exits import load_exits, median_frame, save_camera_exits
from .mtt_roi_editor import SNAP_PX, _ROICanvas

EXIT_COLOR = QColor("#f2d024")


class _ExitCanvas(_ROICanvas):
    """The ROI canvas with one kind of outline, the exits, drawn over the frame unshaded."""

    def __init__(self, parent=None):
        super().__init__(np.zeros((1080, 1920, 3), np.uint8), None, parent)
        self.loading = "Loading..."

    def set_frame(self, frame_rgb: Optional[np.ndarray], exits: Sequence) -> None:
        self.cancel_outline()
        self._drag = None
        self.highlight = None
        if frame_rgb is not None:
            self._rgb = np.ascontiguousarray(frame_rgb[:, :, :3])
            self.img_h, self.img_w = self._rgb.shape[:2]
            self._pixmap = self._to_pixmap(self._rgb)
            self.loading = ""
        self.areas = {"include": [[tuple(q) for q in p] for p in exits], "exclude": []}
        self.update()

    def mousePressEvent(self, ev):
        if not self.loading:
            super().mousePressEvent(ev)

    def exits(self) -> List[List[Tuple[float, float]]]:
        return [list(p) for p in self.areas["include"]]

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        if self.loading:
            painter.setPen(QColor("#aaaaaa"))
            painter.drawText(self.rect(), Qt.AlignCenter, self.loading)
            painter.end()
            return
        ox, oy = self._origin()
        s = self._scale()
        painter.drawPixmap(QRectF(ox, oy, self.img_w * s, self.img_h * s), self._pixmap,
                           QRectF(0, 0, self.img_w, self.img_h))
        for i, p in enumerate(self.areas["include"]):
            poly = QPolygonF([QPointF(*self.to_widget(*q)) for q in p])
            painter.setPen(QPen(EXIT_COLOR, 3.0 if self.highlight == ("include", i) else 2.0))
            painter.setBrush(QBrush(QColor(242, 208, 36, 60)))
            painter.drawPolygon(poly)
            painter.setBrush(QColor(255, 255, 255, 200))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            for q in p:
                painter.drawEllipse(QPointF(*self.to_widget(*q)), 3, 3)
        if self._points:
            pts = list(self._points) + ([self._hover] if self._hover else [])
            painter.setPen(QPen(EXIT_COLOR, 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolyline(QPolygonF([QPointF(*self.to_widget(*q)) for q in pts]))
            painter.setPen(QPen(QColor("#ffffff"), 2.0))
            painter.drawEllipse(QPointF(*self.to_widget(*self._points[0])), SNAP_PX * 0.6,
                                SNAP_PX * 0.6)
        painter.end()


class ExitsWindow(QWidget):
    """Step through ``cameras`` ({(folder, camera): videos}) and draw their exits."""
    _loaded = pyqtSignal(object)

    def __init__(self, cameras: Dict[Tuple[str, str], List[str]],
                 focus: Sequence[str] = (), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mask Tracker — Exits")
        rank = {k: i for i, k in enumerate(focus)}
        self.cameras = cameras
        self.saved = {k: load_exits(k[0]).get(k[1], {}) for k in cameras}
        self.keys = sorted(cameras, key=lambda k: (self.saved[k].get("source") == "checked",
                                                   rank.get(self.label(k), len(rank)),
                                                   self.label(k)))
        self.frames: Dict[Tuple[str, str], np.ndarray] = {}
        self._pending = set()
        self.cur: Optional[Tuple[str, str]] = None
        self._dirty = False

        self.canvas = _ExitCanvas(self)
        self.canvas.changed.connect(self._edited)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._go)
        self.lbl = QLabel()
        self.lbl.setWordWrap(True)
        hint = QLabel("Click the corners round each exit (the open end of the entrance tube, "
                      "where mice come and go). Close it on the first corner, double-click, "
                      "right-click or Enter. Drag corners to adjust; Backspace drops the last "
                      "corner, Esc the outline. Delete removes the selected (or last) exit."
                      "\n\nSpace: these exits are right, next camera. "
                      "PgDn / PgUp: next / previous. Edits are saved when you move on.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        b_ok = QPushButton("Right, next (Space)")
        b_ok.clicked.connect(self.confirm_next)
        b_del = QPushButton("Delete exit (Del)")
        b_del.clicked.connect(self._delete)
        b_prev = QPushButton("Previous")
        b_prev.clicked.connect(lambda: self.step(-1))
        b_next = QPushButton("Next")
        b_next.clicked.connect(lambda: self.step(1))
        self.exit_list = QListWidget()
        self.exit_list.setMaximumHeight(90)
        self.exit_list.currentRowChanged.connect(self._highlight)
        for b in (b_ok, b_del, b_prev, b_next):
            b.setAutoDefault(False)
            b.setFocusPolicy(Qt.NoFocus)

        side = QVBoxLayout()
        side.addWidget(QLabel("Cameras:"))
        side.addWidget(self.list, 1)
        side.addWidget(self.lbl)
        side.addWidget(QLabel("Exits:"))
        side.addWidget(self.exit_list)
        side.addWidget(b_del)
        side.addWidget(hint)
        row = QHBoxLayout()
        row.addWidget(b_prev)
        row.addWidget(b_next)
        side.addLayout(row)
        side.addWidget(b_ok)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(280)
        banner = QLabel(
            "Outline where mice enter and leave this camera's view (for a resource zone: the open "
            "end of the white entrance tube). The tracker then keeps counting a mouse that "
            "vanishes anywhere else: it is hiding, not gone.<br><b>Yellow = the current exit.</b> "
            "On the open end? Press <b>Space</b>. Wrong? Press <b>Del</b>, click the corners "
            "round the open end, press <b>Enter</b>, then <b>Space</b>.")
        banner.setWordWrap(True)
        banner.setStyleSheet("font-size: 13px; padding: 4px;")
        left = QVBoxLayout()
        left.addWidget(banner)
        left.addWidget(self.canvas, 1)
        lay = QHBoxLayout(self)
        lay.addLayout(left, 1)
        lay.addWidget(side_w)
        screen = QApplication.primaryScreen()
        if screen is not None:
            g = screen.availableGeometry()
            self.resize(int(g.width() * 0.85), int(g.height() * 0.85))

        self._loaded.connect(self._on_loaded)
        self._fill_list()
        if self.keys:
            self.list.setCurrentRow(0)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def label(key: Tuple[str, str]) -> str:
        return f"{os.path.basename(key[0])}/{key[1]}"

    def _status(self, key) -> str:
        s = self.saved[key]
        if s.get("source") == "checked":
            return "checked"
        if s.get("exits"):
            return s.get("source", "?") + (f" {s['score']:.2f}" if "score" in s else "")
        return "none"

    def _fill_list(self) -> None:
        row = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for k in self.keys:
            self.list.addItem(f"{self.label(k)}   [{self._status(k)}]")
        if row >= 0:
            self.list.setCurrentRow(row)
        self.list.blockSignals(False)

    def _load(self, key) -> None:
        if key in self.frames or key in self._pending:
            return
        self._pending.add(key)

        def work():
            fr = median_frame(self.cameras[key])
            self._loaded.emit((key, fr))
        threading.Thread(target=work, daemon=True).start()

    def _on_loaded(self, item) -> None:
        key, fr = item
        self._pending.discard(key)
        self.frames[key] = (cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if fr is not None
                            else np.zeros((1080, 1920, 3), np.uint8))
        if key == self.cur:
            self._show()

    def _show(self) -> None:
        key = self.cur
        s = self.saved[key]
        fr = self.frames.get(key)
        exits = s.get("exits", [])
        if fr is not None and s.get("frame_size"):
            w0, h0 = s["frame_size"]
            sx, sy = fr.shape[1] / w0, fr.shape[0] / h0
            exits = [[(x * sx, y * sy) for x, y in p] for p in exits]
        if fr is None:
            self.canvas.loading = f"Loading the median frame of {self.label(key)}..."
        self.canvas.set_frame(fr, exits)
        self._dirty = False
        self._refresh_info()

    def _refresh_info(self) -> None:
        key = self.cur
        self.lbl.setText(f"<b>{self.label(key)}</b><br>{len(self.cameras[key])} videos; "
                         f"exits: {self._status(key)}" + (" (edited)" if self._dirty else ""))
        self.exit_list.blockSignals(True)
        self.exit_list.clear()
        for i, p in enumerate(self.canvas.exits()):
            self.exit_list.addItem(f"Exit {i + 1} ({len(p)} corners)")
        self.exit_list.blockSignals(False)

    # ------------------------------------------------------------ actions
    def _save(self, confirmed: bool) -> None:
        key = self.cur
        if key is None or (not self._dirty and not confirmed) or self.canvas.loading:
            return
        self.canvas.close_outline()
        extra = {k: v for k, v in self.saved[key].items()
                 if k not in ("exits", "frame_size", "source")}
        save_camera_exits(key[0], key[1], self.canvas.exits(),
                          (self.canvas.img_w, self.canvas.img_h), "checked", **extra)
        self.saved[key] = load_exits(key[0]).get(key[1], {})
        self._dirty = False

    def _go(self, row: int) -> None:
        if not 0 <= row < len(self.keys):
            return
        self._save(confirmed=False)
        self.cur = self.keys[row]
        for k in self.keys[row:row + 3]:                 # this one and the next two
            self._load(k)
        self._show()
        self._fill_list()
        self.canvas.setFocus()

    def step(self, d: int) -> None:
        row = self.list.currentRow() + d
        if 0 <= row < len(self.keys):
            self.list.setCurrentRow(row)
        else:
            self._save(confirmed=False)
            self._fill_list()

    def confirm_next(self) -> None:
        self._save(confirmed=True)
        self._fill_list()
        self.step(1)

    def _edited(self) -> None:
        self._dirty = True
        self._refresh_info()

    def _highlight(self, row: int) -> None:
        self.canvas.highlight = ("include", row) if row >= 0 else None
        self.canvas.update()

    def _delete(self) -> None:
        row = self.exit_list.currentRow()
        if row < 0:
            row = len(self.canvas.areas["include"]) - 1          # none selected: the last one
        if 0 <= row < len(self.canvas.areas["include"]):
            del self.canvas.areas["include"][row]
            self.canvas.highlight = None
            self.canvas._edited()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Space and not self.canvas._points:
            self.confirm_next()
        elif ev.key() == Qt.Key_Delete:
            self._delete()
        elif ev.key() == Qt.Key_PageDown:
            self.step(1)
        elif ev.key() == Qt.Key_PageUp:
            self.step(-1)
        elif ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.canvas.close_outline()
        else:
            super().keyPressEvent(ev)

    def closeEvent(self, ev):
        self._save(confirmed=False)
        super().closeEvent(ev)
