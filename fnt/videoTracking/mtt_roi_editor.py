"""Drawing a project's region of interest (mtt_roi) on a frame.

``ROIEditorDialog`` shows a frame. Click out the corners of the area the
model sees (Include) and of any holes in it (Exclude); a click on the first
corner, a double-click, a right-click or Enter closes the outline. Corners
of finished outlines can be dragged. ``paint_roi`` shades what the model
will not see, here and over Mask Tracker's preview.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap,
                         QPolygonF)
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QDialog, QHBoxLayout,
                             QLabel, QListWidget, QPushButton, QRadioButton, QSizePolicy,
                             QVBoxLayout, QWidget)

from .mtt_roi import ROI

SHADE = QColor(0, 0, 0, 165)
INCLUDE_COLOR = QColor("#4d8fe0")
EXCLUDE_COLOR = QColor("#e05a4d")
SNAP_PX = 12          # a click this close to the first corner closes the outline


def paint_roi(painter: QPainter, roi: ROI, img_w: int, img_h: int,
              to_widget: Callable[[float, float], Tuple[float, float]],
              highlight: Optional[Tuple[str, int]] = None) -> None:
    """Shade the parts of an ``img_w`` x ``img_h`` frame the model will not
    see: dark outside the included area, dark and hatched red in excluded
    holes. Outline both."""
    r = roi.for_size(img_w, img_h)

    def poly(points) -> QPolygonF:
        return QPolygonF([QPointF(*to_widget(x, y)) for x, y in points])

    def path(polys) -> QPainterPath:
        out = QPainterPath()
        for p in polys:
            piece = QPainterPath()
            piece.addPolygon(poly(p))
            piece.closeSubpath()
            out = out.united(piece)
        return out

    (ax, ay), (bx, by) = to_widget(0, 0), to_widget(img_w, img_h)
    frame = QPainterPath()
    frame.addRect(QRectF(QPointF(ax, ay), QPointF(bx, by)))
    inside = path(r.include) if r.include else frame
    holes = path(r.exclude).intersected(inside)
    painter.save()
    painter.setPen(Qt.NoPen)
    painter.setBrush(SHADE)
    painter.drawPath(frame.subtracted(inside))
    painter.drawPath(holes)
    painter.setBrush(QBrush(QColor(224, 90, 77, 150), Qt.BDiagPattern))
    painter.drawPath(holes)
    painter.setBrush(Qt.NoBrush)
    for kind, polys, color in (("include", r.include, INCLUDE_COLOR),
                               ("exclude", r.exclude, EXCLUDE_COLOR)):
        for i, p in enumerate(polys):
            hot = highlight == (kind, i)
            painter.setPen(QPen(color, 3.0 if hot else 1.6, Qt.SolidLine))
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
        self.setCursor(Qt.CrossCursor)
        self._rgb = np.ascontiguousarray(frame_rgb[:, :, :3])
        self.img_h, self.img_w = self._rgb.shape[:2]
        self._pixmap = self._to_pixmap(self._rgb)
        self._model_pixmap: Optional[QPixmap] = None
        self.model_view = False
        r = roi.for_size(self.img_w, self.img_h) if roi is not None else ROI()
        self.areas = {"include": [list(p) for p in r.include],
                      "exclude": [list(p) for p in r.exclude]}
        self.mode = "include"
        self.highlight: Optional[Tuple[str, int]] = None
        self._points: List[Tuple[float, float]] = []      # the outline being drawn
        self._hover: Optional[Tuple[float, float]] = None
        self._drag: Optional[Tuple[str, int, int]] = None  # (kind, area, corner)

    def _to_pixmap(self, rgb: np.ndarray) -> QPixmap:
        rgb = np.ascontiguousarray(rgb)
        return QPixmap.fromImage(QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                                        QImage.Format_RGB888).copy())

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

    def _near(self, a, wx: float, wy: float) -> bool:
        ax, ay = self.to_widget(*a)
        return math.hypot(ax - wx, ay - wy) <= SNAP_PX

    def _corner_at(self, wx: float, wy: float) -> Optional[Tuple[str, int, int]]:
        for kind in ("exclude", "include"):
            for i, p in enumerate(self.areas[kind]):
                for k, q in enumerate(p):
                    if self._near(q, wx, wy):
                        return kind, i, k
        return None

    def roi(self) -> Optional[ROI]:
        r = ROI(include=[list(p) for p in self.areas["include"]],
                exclude=[list(p) for p in self.areas["exclude"]],
                frame_size=(self.img_w, self.img_h))
        return None if r.is_empty() else r

    # ------------------------------------------------------------ editing
    def set_mode(self, mode: str) -> None:
        self.cancel_outline()
        self.mode = mode

    def close_outline(self) -> None:
        if len(self._points) >= 3:
            self.areas[self.mode].append(list(self._points))
            self._edited()
        self._points = []
        self.update()

    def cancel_outline(self) -> None:
        self._points = []
        self.update()

    def _edited(self) -> None:
        self._model_pixmap = None
        self.changed.emit()

    def mousePressEvent(self, ev):
        self.setFocus()
        wx, wy = ev.x(), ev.y()
        if ev.button() == Qt.RightButton:
            self.close_outline()
            return
        if ev.button() != Qt.LeftButton:
            return
        if self._points:
            if len(self._points) >= 3 and self._near(self._points[0], wx, wy):
                self.close_outline()            # back on the first corner
            else:
                self._points.append(self.to_image(wx, wy))
        else:
            hit = self._corner_at(wx, wy)
            if hit is not None:
                self._drag = hit                # move a finished outline's corner
            else:
                self._points = [self.to_image(wx, wy)]
        self.update()

    def mouseMoveEvent(self, ev):
        self._hover = self.to_image(ev.x(), ev.y())
        if self._drag is not None:
            kind, i, k = self._drag
            self.areas[kind][i][k] = self._hover
        elif not self._points:
            self.setCursor(Qt.SizeAllCursor if self._corner_at(ev.x(), ev.y()) else Qt.CrossCursor)
        self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag is not None:
            self._drag = None
            self._edited()

    def mouseDoubleClickEvent(self, ev):
        # The double-click's first press already added this corner.
        if ev.button() == Qt.LeftButton:
            self.close_outline()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.close_outline()
        elif ev.key() == Qt.Key_Escape and self._points:
            self.cancel_outline()
        elif ev.key() == Qt.Key_Backspace and self._points:
            self._points.pop()
            self.update()
        else:
            super().keyPressEvent(ev)

    # ------------------------------------------------------------ drawing
    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        ox, oy = self._origin()
        s = self._scale()
        target = QRectF(ox, oy, self.img_w * s, self.img_h * s)
        r = self.roi()
        if self.model_view and r is not None:
            if self._model_pixmap is None:       # what the model gets: the box, grey outside
                cut, (x0, y0) = r.apply(self._rgb)
                full = np.full_like(self._rgb, 40)
                full[y0:y0 + cut.shape[0], x0:x0 + cut.shape[1]] = cut
                self._model_pixmap = self._to_pixmap(full)
            painter.drawPixmap(target, self._model_pixmap, QRectF(0, 0, self.img_w, self.img_h))
        else:
            painter.drawPixmap(target, self._pixmap, QRectF(0, 0, self.img_w, self.img_h))
            if r is not None:
                paint_roi(painter, r, self.img_w, self.img_h, self.to_widget, self.highlight)
        if self._points:
            color = INCLUDE_COLOR if self.mode == "include" else EXCLUDE_COLOR
            pts = list(self._points) + ([self._hover] if self._hover else [])
            painter.setPen(QPen(color, 2.0))
            painter.drawPolyline(QPolygonF([QPointF(*self.to_widget(*q)) for q in pts]))
            painter.setBrush(color)
            for q in self._points[1:]:
                painter.drawEllipse(QPointF(*self.to_widget(*q)), 3, 3)
            # The first corner, larger: click it to close the outline.
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#ffffff"), 2.0))
            painter.drawEllipse(QPointF(*self.to_widget(*self._points[0])), SNAP_PX * 0.6,
                                SNAP_PX * 0.6)
        elif not self.model_view:
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.setBrush(QColor(255, 255, 255, 160))
            for polys in self.areas.values():
                for p in polys:
                    for q in p:
                        painter.drawEllipse(QPointF(*self.to_widget(*q)), 2.5, 2.5)
        painter.end()


class ROIEditorDialog(QDialog):
    """Draw the included and excluded areas on ``frame_rgb``; ``roi()``
    after Save is the result (None: whole frames)."""

    def __init__(self, frame_rgb: np.ndarray, roi: Optional[ROI] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Region of interest")
        screen = QApplication.primaryScreen()
        if screen is not None:
            g = screen.availableGeometry()
            self.resize(int(g.width() * 0.9), int(g.height() * 0.9))
        else:
            self.resize(1280, 820)
        self.canvas = _ROICanvas(frame_rgb, roi, self)

        self.rb_include = QRadioButton("Include area")
        self.rb_include.setToolTip(
            "Outline the area the animals can be in (e.g. the arena floor). The model\n"
            "sees only this area, cut out of the frame: more pixels per mouse.\n"
            "Several outlines are joined. None: the whole frame.")
        self.rb_exclude = QRadioButton("Exclude area")
        self.rb_exclude.setToolTip(
            "Outline a hole in the included area the model must ignore\n"
            "(e.g. a box lid it sees mice through).")
        self.rb_include.setChecked(True)
        modes = QButtonGroup(self)
        for rb in (self.rb_include, self.rb_exclude):
            modes.addButton(rb)
            rb.toggled.connect(self._mode_changed)
        hint = QLabel("Click the corners of an outline. Click the first corner again "
                      "(or double-click, right-click, Enter) to close it. Backspace removes "
                      "the last corner, Esc drops the outline. Drag a finished outline's "
                      "corners to adjust it.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        self.chk_model_view = QCheckBox("Show what the model sees")
        self.chk_model_view.setToolTip("The frame as training and prediction prepare it: cut to\n"
                                       "the box around the included area, the rest grey.")
        self.chk_model_view.toggled.connect(self._model_view)

        self.list_areas = QListWidget()
        self.list_areas.setMaximumWidth(240)
        self.list_areas.currentRowChanged.connect(self._select_area)
        btn_del = QPushButton("Delete outline")
        btn_del.clicked.connect(self._delete_area)
        btn_clear = QPushButton("Clear all")
        btn_clear.clicked.connect(self._clear_all)
        self.lbl_info = QLabel()
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        btn_save = QPushButton("Save")
        btn_save.setAutoDefault(False)
        btn_save.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setAutoDefault(False)
        btn_cancel.clicked.connect(self.reject)
        for b in (btn_del, btn_clear):
            b.setAutoDefault(False)

        side = QVBoxLayout()
        side.addWidget(QLabel("Draw:"))
        side.addWidget(self.rb_include)
        side.addWidget(self.rb_exclude)
        side.addWidget(hint)
        side.addSpacing(6)
        side.addWidget(self.chk_model_view)
        side.addSpacing(6)
        side.addWidget(QLabel("Outlines:"))
        side.addWidget(self.list_areas, 1)
        side.addWidget(btn_del)
        side.addWidget(btn_clear)
        side.addWidget(self.lbl_info)
        row = QHBoxLayout()
        row.addWidget(btn_save)
        row.addWidget(btn_cancel)
        side.addLayout(row)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(250)
        lay = QHBoxLayout(self)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(side_w)

        self.canvas.changed.connect(self._refresh)
        self._refresh()
        self.canvas.setFocus()

    def roi(self) -> Optional[ROI]:
        return self.canvas.roi()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.canvas.close_outline()     # closes an outline, never the dialog
            return
        super().keyPressEvent(ev)

    def accept(self):
        self.canvas.close_outline()         # an outline left open is kept, not lost
        super().accept()

    def _rows(self) -> List[Tuple[str, int]]:
        return [(k, i) for k in ("include", "exclude") for i in range(len(self.canvas.areas[k]))]

    def _mode_changed(self) -> None:
        self.canvas.set_mode("exclude" if self.rb_exclude.isChecked() else "include")
        self.canvas.setFocus()

    def _model_view(self, on: bool) -> None:
        self.canvas.model_view = on
        self.canvas.update()

    def _refresh(self) -> None:
        cur = self.list_areas.currentRow()
        self.list_areas.blockSignals(True)
        self.list_areas.clear()
        for kind, i in self._rows():
            n = len(self.canvas.areas[kind][i])
            self.list_areas.addItem(f"{kind.capitalize()} {i + 1} ({n} corners)")
        self.list_areas.blockSignals(False)
        if 0 <= cur < self.list_areas.count():
            self.list_areas.setCurrentRow(cur)
        r = self.roi()
        info = f"Frame {self.canvas.img_w}x{self.canvas.img_h}. "
        info += ("The model sees " + r.describe() + "." if r is not None
                 else "No ROI: the model sees whole frames.")
        self.lbl_info.setText(info)
        self.canvas.update()

    def _select_area(self, row: int) -> None:
        rows = self._rows()
        self.canvas.highlight = rows[row] if 0 <= row < len(rows) else None
        self.canvas.update()

    def _delete_area(self) -> None:
        rows = self._rows()
        row = self.list_areas.currentRow()
        if 0 <= row < len(rows):
            kind, i = rows[row]
            del self.canvas.areas[kind][i]
            self.canvas.highlight = None
            self.canvas._edited()

    def _clear_all(self) -> None:
        self.canvas.areas = {"include": [], "exclude": []}
        self.canvas.cancel_outline()
        self.canvas.highlight = None
        self.canvas._edited()

