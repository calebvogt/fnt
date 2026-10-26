"""Drawing a project's region of interest (mtt_roi) on a frame.

``ROIEditorDialog`` shows a frame: drag a box for the crop, click out the
corners of each area to exclude (right-click, double-click or Enter closes
it). ``paint_roi`` shades what the model will not see, here and over Mask
Tracker's preview.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (QButtonGroup, QDialog, QHBoxLayout, QLabel, QListWidget,
                             QPushButton, QRadioButton, QSizePolicy, QVBoxLayout, QWidget)

from .mtt_roi import ROI

SHADE = QColor(0, 0, 0, 130)
CROP_COLOR = QColor("#4d8fe0")
EXCLUDE_COLOR = QColor("#e05a4d")


def paint_roi(painter: QPainter, roi: ROI, img_w: int, img_h: int,
              to_widget: Callable[[float, float], Tuple[float, float]],
              highlight: int = -1) -> None:
    """Shade the parts of an ``img_w`` x ``img_h`` frame outside ``roi``
    (cut away or excluded) and outline its crop and excluded areas."""
    r = roi.for_size(img_w, img_h)

    def poly(points) -> QPolygonF:
        return QPolygonF([QPointF(*to_widget(x, y)) for x, y in points])

    def rect(ax, ay, bx, by) -> QRectF:
        (wx0, wy0), (wx1, wy1) = to_widget(ax, ay), to_widget(bx, by)
        return QRectF(QPointF(wx0, wy0), QPointF(wx1, wy1))

    x0, y0, x1, y1 = r.crop or (0, 0, img_w, img_h)
    frame = QPainterPath()
    frame.addRect(rect(0, 0, img_w, img_h))
    keep = QPainterPath()
    keep.addRect(rect(x0, y0, x1, y1))
    for p in r.exclude:
        hole = QPainterPath()
        hole.addPolygon(poly(p))
        hole.closeSubpath()
        keep = keep.subtracted(hole)
    painter.save()
    painter.setPen(Qt.NoPen)
    painter.setBrush(SHADE)
    painter.drawPath(frame.subtracted(keep))
    painter.setBrush(Qt.NoBrush)
    if r.crop:
        painter.setPen(QPen(CROP_COLOR, 1.5, Qt.DashLine))
        painter.drawRect(rect(x0, y0, x1, y1))
    for i, p in enumerate(r.exclude):
        painter.setPen(QPen(EXCLUDE_COLOR, 2.5 if i == highlight else 1.2,
                            Qt.SolidLine if i == highlight else Qt.DashLine))
        painter.drawPolygon(poly(p))
    painter.restore()


class _ROICanvas(QWidget):
    """The frame, fitted to the widget, with the ROI being drawn on it."""
    changed = pyqtSignal()

    def __init__(self, frame_rgb: np.ndarray, roi: Optional[ROI], parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        rgb = np.ascontiguousarray(frame_rgb[:, :, :3])
        self.img_h, self.img_w = rgb.shape[:2]
        self._pixmap = QPixmap.fromImage(
            QImage(rgb.data, self.img_w, self.img_h, 3 * self.img_w, QImage.Format_RGB888).copy())
        r = roi.for_size(self.img_w, self.img_h) if roi is not None else ROI()
        self.crop: Optional[Tuple[int, int, int, int]] = r.crop
        self.exclude: List[List[Tuple[float, float]]] = [list(p) for p in r.exclude]
        self.mode = "crop"
        self.highlight = -1
        self._drag_start: Optional[Tuple[float, float]] = None
        self._points: List[Tuple[float, float]] = []      # the excluded area being drawn
        self._hover: Optional[Tuple[float, float]] = None

    # ---------------------------------------------------------- geometry
    def _scale(self) -> float:
        return min(self.width() / self.img_w, self.height() / self.img_h)

    def _origin(self) -> Tuple[float, float]:
        s = self._scale()
        return (self.width() - self.img_w * s) / 2, (self.height() - self.img_h * s) / 2

    def to_widget(self, x: float, y: float) -> Tuple[float, float]:
        s = self._scale()
        ox, oy = self._origin()
        return ox + x * s, oy + y * s

    def to_image(self, wx: float, wy: float) -> Tuple[float, float]:
        s = self._scale()
        ox, oy = self._origin()
        return (min(max((wx - ox) / s, 0.0), float(self.img_w)),
                min(max((wy - oy) / s, 0.0), float(self.img_h)))

    def roi(self) -> Optional[ROI]:
        r = ROI(crop=self.crop, exclude=[list(p) for p in self.exclude],
                frame_size=(self.img_w, self.img_h))
        return None if r.is_empty() else r

    # ------------------------------------------------------------ editing
    def close_area(self) -> None:
        if len(self._points) >= 3:
            self.exclude.append(list(self._points))
            self.changed.emit()
        self._points = []
        self.update()

    def cancel_area(self) -> None:
        self._points = []
        self.update()

    def mousePressEvent(self, ev):
        p = self.to_image(ev.x(), ev.y())
        if self.mode == "crop" and ev.button() == Qt.LeftButton:
            self._drag_start = p
            self.crop = None
        elif self.mode == "exclude":
            if ev.button() == Qt.LeftButton:
                self._points.append(p)
            elif ev.button() == Qt.RightButton:
                self.close_area()
        self.update()

    def mouseMoveEvent(self, ev):
        self._hover = self.to_image(ev.x(), ev.y())
        if self.mode == "crop" and self._drag_start is not None:
            self._set_crop(self._drag_start, self._hover)
        self.update()

    def mouseReleaseEvent(self, ev):
        if self.mode == "crop" and self._drag_start is not None and ev.button() == Qt.LeftButton:
            self._set_crop(self._drag_start, self.to_image(ev.x(), ev.y()))
            self._drag_start = None
            if self.crop and (self.crop[2] - self.crop[0] < 8 or self.crop[3] - self.crop[1] < 8):
                self.crop = None                # a click, not a box
            self.changed.emit()
            self.update()

    def mouseDoubleClickEvent(self, ev):
        if self.mode == "exclude":
            # The double-click's first press already added this corner.
            self.close_area()

    def keyPressEvent(self, ev):
        if self.mode == "exclude" and ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.close_area()
        elif ev.key() == Qt.Key_Escape and self._points:
            self.cancel_area()
        elif ev.key() == Qt.Key_Backspace and self._points:
            self._points.pop()
            self.update()
        else:
            super().keyPressEvent(ev)

    def _set_crop(self, a, b) -> None:
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        self.crop = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))

    # ------------------------------------------------------------ drawing
    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        ox, oy = self._origin()
        s = self._scale()
        painter.drawPixmap(QRectF(ox, oy, self.img_w * s, self.img_h * s), self._pixmap,
                           QRectF(0, 0, self.img_w, self.img_h))
        r = self.roi()
        if r is not None:
            paint_roi(painter, r, self.img_w, self.img_h, self.to_widget, self.highlight)
        if self._points:
            pts = list(self._points) + ([self._hover] if self._hover else [])
            painter.setPen(QPen(EXCLUDE_COLOR, 1.5))
            painter.drawPolyline(QPolygonF([QPointF(*self.to_widget(*q)) for q in pts]))
            painter.setBrush(EXCLUDE_COLOR)
            for q in self._points:
                painter.drawEllipse(QPointF(*self.to_widget(*q)), 3, 3)
        painter.end()


class ROIEditorDialog(QDialog):
    """Draw the crop and excluded areas on ``frame_rgb``; ``roi()`` after
    Save is the result (None: whole frames)."""

    def __init__(self, frame_rgb: np.ndarray, roi: Optional[ROI] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Region of interest")
        self.resize(1280, 820)
        self.canvas = _ROICanvas(frame_rgb, roi, self)

        self.rb_crop = QRadioButton("Crop box")
        self.rb_crop.setToolTip("Drag a box around the area the animals can reach.\n"
                                "The model sees only this box: more pixels per mouse.")
        self.rb_exclude = QRadioButton("Exclude area")
        self.rb_exclude.setToolTip(
            "Click the corners of an area the model must ignore (e.g. a box lid it\n"
            "sees mice through). Right-click, double-click or Enter closes the area;\n"
            "Backspace removes the last corner, Esc drops the area.")
        self.rb_crop.setChecked(True)
        modes = QButtonGroup(self)
        for rb in (self.rb_crop, self.rb_exclude):
            modes.addButton(rb)
            rb.toggled.connect(self._mode_changed)

        self.list_exclude = QListWidget()
        self.list_exclude.setMaximumWidth(220)
        self.list_exclude.currentRowChanged.connect(self._select_area)
        btn_del = QPushButton("Delete area")
        btn_del.clicked.connect(self._delete_area)
        btn_clear_crop = QPushButton("Clear crop")
        btn_clear_crop.clicked.connect(self._clear_crop)
        btn_clear = QPushButton("Clear all")
        btn_clear.clicked.connect(self._clear_all)
        self.lbl_info = QLabel()
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        btn_save = QPushButton("Save")
        btn_save.setDefault(False)
        btn_save.setAutoDefault(False)
        btn_save.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setAutoDefault(False)
        btn_cancel.clicked.connect(self.reject)

        side = QVBoxLayout()
        side.addWidget(QLabel("Draw:"))
        side.addWidget(self.rb_crop)
        side.addWidget(self.rb_exclude)
        side.addSpacing(8)
        side.addWidget(QLabel("Excluded areas:"))
        side.addWidget(self.list_exclude, 1)
        side.addWidget(btn_del)
        side.addWidget(btn_clear_crop)
        side.addWidget(btn_clear)
        side.addWidget(self.lbl_info)
        row = QHBoxLayout()
        row.addWidget(btn_save)
        row.addWidget(btn_cancel)
        side.addLayout(row)
        lay = QHBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addLayout(side)

        self.canvas.changed.connect(self._refresh)
        self._refresh()
        self.canvas.setFocus()

    def roi(self) -> Optional[ROI]:
        return self.canvas.roi()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            return              # Enter closes an excluded area (the canvas), not the dialog
        super().keyPressEvent(ev)

    def _mode_changed(self) -> None:
        self.canvas.cancel_area()
        self.canvas.mode = "exclude" if self.rb_exclude.isChecked() else "crop"
        self.canvas.setCursor(Qt.CrossCursor)
        self.canvas.setFocus()

    def _refresh(self) -> None:
        cur = self.list_exclude.currentRow()
        self.list_exclude.blockSignals(True)
        self.list_exclude.clear()
        for i, p in enumerate(self.canvas.exclude):
            self.list_exclude.addItem(f"Area {i + 1} ({len(p)} corners)")
        self.list_exclude.blockSignals(False)
        if 0 <= cur < self.list_exclude.count():
            self.list_exclude.setCurrentRow(cur)
        r = self.roi()
        info = f"Frame {self.canvas.img_w}x{self.canvas.img_h}. "
        info += r.describe() if r is not None else "No ROI: the model sees whole frames."
        self.lbl_info.setText(info)
        self.canvas.update()

    def _select_area(self, row: int) -> None:
        self.canvas.highlight = row
        self.canvas.update()

    def _delete_area(self) -> None:
        row = self.list_exclude.currentRow()
        if 0 <= row < len(self.canvas.exclude):
            del self.canvas.exclude[row]
            self.canvas.highlight = -1
            self._refresh()

    def _clear_crop(self) -> None:
        self.canvas.crop = None
        self._refresh()

    def _clear_all(self) -> None:
        self.canvas.crop = None
        self.canvas.exclude = []
        self.canvas.cancel_area()
        self.canvas.highlight = -1
        self._refresh()
