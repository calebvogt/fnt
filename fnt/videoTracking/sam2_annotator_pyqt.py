"""FNT Mask Tracker Tool (MTT).

Unified GUI for instance segmentation annotation and Mask R-CNN training.
Two tabs:
  - Annotator: load videos, extract frames, annotate with manual/AI masks
  - Training: augment dataset and train a torchvision Mask R-CNN

All code written from scratch for FieldNeuroToolbox.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QGroupBox, QSpinBox, QComboBox, QDialog, QAction, QMenuBar,
    QListWidget, QListWidgetItem, QSizePolicy, QScrollArea,
    QInputDialog, QStatusBar, QFrame, QMenu, QTabWidget,
    QDoubleSpinBox, QDialogButtonBox, QLineEdit, QCheckBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush,
    QPolygonF, QWheelEvent, QMouseEvent, QKeyEvent,
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

DARK_STYLESHEET = """
    QMainWindow, QWidget { background-color: #2b2b2b; color: #cccccc; }
    QMenuBar { background-color: #2b2b2b; color: #cccccc; }
    QMenuBar::item:selected { background-color: #3c3c3c; }
    QMenu { background-color: #2b2b2b; color: #cccccc; border: 1px solid #555555; }
    QMenu::item:selected { background-color: #2979ff; }
    QGroupBox { border: 1px solid #555555; border-radius: 4px;
                margin-top: 8px; padding-top: 14px; font-weight: bold; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
    QPushButton { background-color: #3c3c3c; border: 1px solid #555555;
                  border-radius: 3px; padding: 5px 10px; color: #cccccc; }
    QPushButton:hover { background-color: #4a4a4a; }
    QPushButton:pressed { background-color: #555555; }
    QPushButton:disabled { color: #666666; background-color: #333333; }
    QPushButton:checked { background-color: #2979ff; color: white; }
    QListWidget { background-color: #1e1e1e; border: 1px solid #444444;
                  color: #cccccc; }
    QListWidget::item:selected { background-color: #2979ff; color: white; }
    QComboBox { background-color: #3c3c3c; border: 1px solid #555555;
                border-radius: 3px; padding: 3px 6px; color: #cccccc; }
    QComboBox QAbstractItemView { background-color: #3c3c3c; color: #cccccc;
                                   selection-background-color: #2979ff; }
    QSpinBox, QDoubleSpinBox { background-color: #3c3c3c; border: 1px solid #555555;
                                border-radius: 3px; padding: 2px; color: #cccccc; }
    QProgressBar { border: 1px solid #555555; border-radius: 3px;
                   text-align: center; background-color: #1e1e1e; color: #cccccc; }
    QProgressBar::chunk { background-color: #2979ff; border-radius: 3px; }
    QScrollArea { border: none; }
    QTabWidget::pane { border: 1px solid #555555; }
    QTabBar::tab { background-color: #3c3c3c; color: #cccccc; padding: 6px 16px;
                   border: 1px solid #555555; border-bottom: none; border-radius: 3px 3px 0 0; }
    QTabBar::tab:selected { background-color: #2b2b2b; color: white; }
    QTabBar::tab:hover { background-color: #4a4a4a; }
    QScrollBar:vertical {
        background-color: #1e1e1e; width: 12px;
        border: 1px solid #3f3f3f; border-radius: 3px;
    }
    QScrollBar::handle:vertical {
        background-color: #0078d4; border-radius: 2px; min-height: 20px; margin: 1px;
    }
    QScrollBar::handle:vertical:hover { background-color: #106ebe; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background-color: #1e1e1e; }
    QScrollBar:horizontal {
        background-color: #1e1e1e; height: 14px;
        border: 1px solid #3f3f3f; border-radius: 3px;
    }
    QScrollBar::handle:horizontal {
        background-color: #0078d4; border-radius: 2px; min-width: 20px; margin: 1px;
    }
    QScrollBar::handle:horizontal:hover { background-color: #106ebe; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background-color: #1e1e1e; }
    QStatusBar { background-color: #252525; color: #999999; }
    QLabel { color: #cccccc; }
"""

CLASS_COLORS = [
    (0, 120, 255),   # blue
    (255, 80, 0),    # orange
    (0, 200, 80),    # green
    (200, 0, 200),   # purple
    (255, 200, 0),   # yellow
    (0, 200, 200),   # cyan
    (255, 100, 100), # red
    (100, 255, 200), # mint
]


def _class_color(index: int) -> Tuple[int, int, int]:
    return CLASS_COLORS[index % len(CLASS_COLORS)]


# ======================================================================
# Annotation data structures
# ======================================================================
class AnnotationObject:
    def __init__(self, points: List[Tuple[float, float]], category: str,
                 ann_id: int = -1, is_ai: bool = False):
        self.points = list(points)
        self.category = category
        self.ann_id = ann_id
        self.is_ai = is_ai


# ======================================================================
# Category selector dialog (replaces QInputDialog dropdown)
# ======================================================================
class CategorySelectDialog(QDialog):
    """List-based category selector with the last-used category pre-selected."""

    def __init__(self, categories: List[str], last_used: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Assign Category")
        self.setMinimumWidth(260)
        self.setStyleSheet(
            "QDialog { background-color: #2b2b2b; color: #cccccc; }"
            "QListWidget { background-color: #1e1e1e; border: 1px solid #444; color: #cccccc; }"
            "QListWidget::item { padding: 4px 8px; }"
            "QListWidget::item:selected { background-color: #2979ff; color: white; }"
            "QPushButton { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 5px 14px; color: #cccccc; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
        )
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select category for this annotation:"))

        self.cat_list = QListWidget()
        for cat in categories:
            self.cat_list.addItem(cat)
        self.cat_list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.cat_list)

        btn_row = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)
        self.btn_ok = QPushButton("OK")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_ok)
        layout.addLayout(btn_row)

        # Pre-select last used category
        sel_row = 0
        if last_used:
            for i in range(self.cat_list.count()):
                if self.cat_list.item(i).text() == last_used:
                    sel_row = i
                    break
        if self.cat_list.count() > 0:
            self.cat_list.setCurrentRow(sel_row)
            self.cat_list.setFocus()

    def selected_category(self) -> Optional[str]:
        item = self.cat_list.currentItem()
        return item.text() if item else None


# ======================================================================
# Edit Classes dialog
# ======================================================================
class EditClassesDialog(QDialog):
    def __init__(self, categories: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Classes")
        self.setMinimumWidth(300)
        self.setStyleSheet(
            "QDialog { background-color: #2b2b2b; color: #cccccc; }"
            "QListWidget { background-color: #1e1e1e; border: 1px solid #444; color: #cccccc; }"
            "QListWidget::item:selected { background-color: #2979ff; color: white; }"
            "QLineEdit { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 4px; color: #cccccc; }"
            "QPushButton { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 5px 10px; color: #cccccc; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
        )
        layout = QVBoxLayout(self)

        self.class_list = QListWidget()
        for cat in categories:
            self.class_list.addItem(cat)
        layout.addWidget(self.class_list)

        add_row = QHBoxLayout()
        self.txt_new = QLineEdit()
        self.txt_new.setPlaceholderText("New class name...")
        self.txt_new.returnPressed.connect(self._add_class)
        add_row.addWidget(self.txt_new, 1)
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self._add_class)
        add_row.addWidget(btn_add)
        layout.addLayout(add_row)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.clicked.connect(self._remove_class)
        layout.addWidget(btn_remove)

        btn_close = QPushButton("Done")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        self._removed: List[str] = []

    def _add_class(self):
        name = self.txt_new.text().strip()
        if not name:
            return
        for i in range(self.class_list.count()):
            if self.class_list.item(i).text() == name:
                return
        self.class_list.addItem(name)
        self.txt_new.clear()

    def _remove_class(self):
        row = self.class_list.currentRow()
        if row < 0:
            return
        name = self.class_list.item(row).text()
        self._removed.append(name)
        self.class_list.takeItem(row)

    def get_categories(self) -> List[str]:
        return [self.class_list.item(i).text() for i in range(self.class_list.count())]

    def get_removed(self) -> List[str]:
        return list(self._removed)


# ======================================================================
# Preview widget with zoom, pan, polygon drawing, point dragging
# ======================================================================
class AnnotationPreviewWidget(QWidget):
    annotation_accepted = pyqtSignal()
    ai_prediction_requested = pyqtSignal()
    zoom_changed = pyqtSignal(float)
    mode_changed = pyqtSignal(str)
    advance_frame_requested = pyqtSignal()
    annotation_edited = pyqtSignal(int)
    delete_annotation_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self._image_rgb: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._img_w = 0
        self._img_h = 0

        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._panning = False
        self._pan_start = None

        self.drawing_mode = "navigate"
        self._drawing_active = False
        self._current_points: List[Tuple[float, float]] = []
        self._drawing_accepted = False

        self._ai_positive_points: List[Tuple[int, int]] = []
        self._ai_negative_points: List[Tuple[int, int]] = []
        self._ai_mask: Optional[np.ndarray] = None
        self._ai_mask_contour_pts: List[Tuple[float, float]] = []
        self._ai_score: float = 0.0

        self._drag_obj_idx: Optional[int] = None
        self._drag_pt_idx: Optional[int] = None
        self._dragging_point = False

        self._editing_obj_idx: Optional[int] = None

        self.annotations: List[AnnotationObject] = []

        self._pending_annotation: Optional[List[Tuple[float, float]]] = None
        self._pending_is_ai = False

    def set_frame(self, image_rgb: np.ndarray):
        h, w = image_rgb.shape[:2]
        self._image_rgb = image_rgb
        self._img_w, self._img_h = w, h
        bytes_per_line = 3 * w
        qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._reset_view()
        self.update()

    def clear(self):
        self._pixmap = None
        self._image_rgb = None
        self._clear_drawing()
        self.annotations.clear()
        self.update()

    def _clear_drawing(self):
        self._drawing_active = False
        self._drawing_accepted = False
        self._current_points.clear()
        self._ai_positive_points.clear()
        self._ai_negative_points.clear()
        self._ai_mask = None
        self._ai_mask_contour_pts.clear()
        self._ai_score = 0.0
        self._editing_obj_idx = None
        self._pending_annotation = None
        self.update()

    def _reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

    def _base_scale(self) -> float:
        if self._pixmap is None:
            return 1.0
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        return min(ww / pw, wh / ph)

    def _effective_scale(self) -> float:
        return self._base_scale() * self._zoom

    def _img_to_widget(self, ix: float, iy: float) -> Tuple[float, float]:
        s = self._effective_scale()
        base = self._base_scale()
        dw, dh = self._img_w * base, self._img_h * base
        ox = (self.width() - dw) / 2 + self._pan_x
        oy = (self.height() - dh) / 2 + self._pan_y
        return ox + ix * s, oy + iy * s

    def _widget_to_img(self, wx: float, wy: float) -> Optional[Tuple[float, float]]:
        s = self._effective_scale()
        if s == 0:
            return None
        base = self._base_scale()
        dw, dh = self._img_w * base, self._img_h * base
        ox = (self.width() - dw) / 2 + self._pan_x
        oy = (self.height() - dh) / 2 + self._pan_y
        ix = (wx - ox) / s
        iy = (wy - oy) / s
        if 0 <= ix < self._img_w and 0 <= iy < self._img_h:
            return ix, iy
        return None

    def _point_radius(self) -> float:
        return max(2.0, min(6.0, 4.0 / self._zoom))

    def _find_point_at(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        threshold = max(8.0, 10.0 / self._zoom)
        for oi, ann in enumerate(self.annotations):
            for pi, (px, py) in enumerate(ann.points):
                sx, sy = self._img_to_widget(px, py)
                if math.hypot(wx - sx, wy - sy) < threshold:
                    return oi, pi
        return None

    def _find_annotation_at(self, wx: float, wy: float) -> Optional[int]:
        pt = QPointF(wx, wy)
        for oi in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[oi]
            if len(ann.points) < 3:
                continue
            poly = QPolygonF()
            for px, py in ann.points:
                sx, sy = self._img_to_widget(px, py)
                poly.append(QPointF(sx, sy))
            if poly.containsPoint(pt, Qt.OddEvenFill):
                return oi
        return None

    # -- Events --
    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap is None:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        old_zoom = self._zoom
        self._zoom = max(1.0, min(20.0, self._zoom * factor))
        if self._zoom == old_zoom:
            return
        mx, my = event.x(), event.y()
        self._pan_x = mx - (mx - self._pan_x) * (self._zoom / old_zoom)
        self._pan_y = my - (my - self._pan_y) * (self._zoom / old_zoom)
        self.zoom_changed.emit(self._zoom)
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self._pixmap is None or event.button() != Qt.LeftButton:
            return
        hit_idx = self._find_annotation_at(event.x(), event.y())
        if hit_idx is not None:
            self._editing_obj_idx = hit_idx
            self.mode_changed.emit("Editing Mask")
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self._pixmap is None:
            return

        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = (event.x(), event.y(), self._pan_x, self._pan_y)
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.RightButton:
            if self.drawing_mode == "ai" and self._drawing_active:
                coords = self._widget_to_img(event.x(), event.y())
                if coords:
                    self._ai_negative_points.append((int(coords[0]), int(coords[1])))
                    self.update()
                    self._request_ai_prediction()
            else:
                self._show_context_menu(event.globalPos(), event.x(), event.y())
            return

        if event.button() == Qt.LeftButton:
            if self.drawing_mode == "navigate":
                if self._editing_obj_idx is not None:
                    hit = self._find_point_at(event.x(), event.y())
                    if hit is not None and hit[0] == self._editing_obj_idx:
                        self._drag_obj_idx, self._drag_pt_idx = hit
                        self._dragging_point = True
                        return
                self._panning = True
                self._pan_start = (event.x(), event.y(), self._pan_x, self._pan_y)
                self.setCursor(Qt.ClosedHandCursor)
                return

            coords = self._widget_to_img(event.x(), event.y())
            if coords is None:
                return
            ix, iy = coords

            hit = self._find_point_at(event.x(), event.y())
            if hit is not None:
                self._drag_obj_idx, self._drag_pt_idx = hit
                self._dragging_point = True
                return

            if self.drawing_mode == "manual":
                if not self._drawing_active:
                    self._drawing_active = True
                    self._drawing_accepted = False
                    self._current_points.clear()
                self._current_points.append((ix, iy))
                self.update()

            elif self.drawing_mode == "ai":
                if not self._drawing_active:
                    self._drawing_active = True
                    self._ai_positive_points.clear()
                    self._ai_negative_points.clear()
                    self._ai_mask = None
                    self._ai_mask_contour_pts.clear()
                self._ai_positive_points.append((int(ix), int(iy)))
                self.update()
                self._request_ai_prediction()

    def _show_context_menu(self, global_pos, wx: float, wy: float):
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #2b2b2b; color: #cccccc; border: 1px solid #555; }"
            "QMenu::item:selected { background-color: #2979ff; }"
            "QMenu::item:disabled { color: #666666; }"
        )
        act_manual = menu.addAction("Add Manual Mask")
        act_ai = menu.addAction("Add AI-Assisted Mask")
        menu.addSeparator()
        hit_idx = self._find_annotation_at(wx, wy)
        act_edit = menu.addAction("Edit Mask")
        act_delete = menu.addAction("Delete Mask")
        if hit_idx is None:
            act_edit.setEnabled(False)
            act_delete.setEnabled(False)
        action = menu.exec_(global_pos)
        if action == act_manual:
            self.drawing_mode = "manual"
            self.mode_changed.emit("Manual Mask")
        elif action == act_ai:
            self.drawing_mode = "ai"
            self.mode_changed.emit("AI-Assisted Mask")
        elif action == act_edit and hit_idx is not None:
            self._editing_obj_idx = hit_idx
            self.mode_changed.emit("Editing Mask")
            self.update()
        elif action == act_delete and hit_idx is not None:
            self.delete_annotation_requested.emit(hit_idx)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning and self._pan_start is not None:
            dx = event.x() - self._pan_start[0]
            dy = event.y() - self._pan_start[1]
            self._pan_x = self._pan_start[2] + dx
            self._pan_y = self._pan_start[3] + dy
            self.update()
            return
        if self._dragging_point and self._drag_obj_idx is not None:
            coords = self._widget_to_img(event.x(), event.y())
            if coords is not None:
                ann = self.annotations[self._drag_obj_idx]
                ann.points[self._drag_pt_idx] = coords
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._panning:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            return
        if self._dragging_point:
            self._dragging_point = False
            self._drag_obj_idx = None
            self._drag_pt_idx = None

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self._editing_obj_idx is not None:
                idx = self._editing_obj_idx
                self._editing_obj_idx = None
                self.mode_changed.emit("Navigate")
                self.update()
                self.annotation_edited.emit(idx)
            else:
                self._finish_annotation()
        elif event.key() == Qt.Key_Escape:
            if self._editing_obj_idx is not None:
                self._editing_obj_idx = None
                self.mode_changed.emit("Navigate")
                self.update()
            else:
                self._clear_drawing()
                self.drawing_mode = "navigate"
                self.mode_changed.emit("Navigate")
        elif event.key() == Qt.Key_Space:
            self.advance_frame_requested.emit()
        elif event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._zoom = min(20.0, self._zoom * 1.2)
            self.zoom_changed.emit(self._zoom)
            self.update()
        elif event.key() == Qt.Key_Minus:
            self._zoom = max(1.0, self._zoom / 1.2)
            self.zoom_changed.emit(self._zoom)
            self.update()
        elif event.key() == Qt.Key_0:
            self._reset_view()
            self.zoom_changed.emit(self._zoom)
            self.update()
        else:
            super().keyPressEvent(event)

    def _finish_annotation(self):
        if self.drawing_mode == "manual" and self._drawing_active and len(self._current_points) >= 3:
            self._pending_annotation = list(self._current_points)
            self._pending_is_ai = False
            self._drawing_accepted = True
            self.update()
            self.annotation_accepted.emit()
            self.drawing_mode = "navigate"
            self.mode_changed.emit("Navigate")
        elif self.drawing_mode == "ai" and self._drawing_active and self._ai_mask_contour_pts:
            self._pending_annotation = list(self._ai_mask_contour_pts)
            self._pending_is_ai = True
            self._drawing_accepted = True
            self.update()
            self.annotation_accepted.emit()
            self.drawing_mode = "navigate"
            self.mode_changed.emit("Navigate")

    def accept_annotation(self, category: str, ann_id: int = -1):
        if self._pending_annotation is None:
            return
        ann = AnnotationObject(
            self._pending_annotation, category, ann_id=ann_id,
            is_ai=self._pending_is_ai,
        )
        self.annotations.append(ann)
        self._clear_drawing()

    def get_pending_points(self) -> Optional[List[Tuple[float, float]]]:
        return self._pending_annotation

    def get_pending_ai_mask(self) -> Optional[np.ndarray]:
        if self._pending_is_ai and self._ai_mask is not None:
            return self._ai_mask
        return None

    def set_ai_mask(self, mask: np.ndarray, score: float):
        self._ai_mask = mask
        self._ai_score = score
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            eps = 0.005 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, eps, True)
            self._ai_mask_contour_pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        else:
            self._ai_mask_contour_pts = []
        self.update()

    def _request_ai_prediction(self):
        self.ai_prediction_requested.emit()

    # -- Painting --
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._pixmap is None:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            painter.end()
            return

        s = self._effective_scale()
        base = self._base_scale()
        dw, dh = self._img_w * base, self._img_h * base
        ox = (self.width() - dw) / 2 + self._pan_x
        oy = (self.height() - dh) / 2 + self._pan_y

        painter.drawPixmap(
            int(ox), int(oy),
            int(self._img_w * s), int(self._img_h * s),
            self._pixmap,
        )

        pr = self._point_radius()
        cat_counters: Dict[str, int] = {}

        for ai, ann in enumerate(self.annotations):
            ci = ai % len(CLASS_COLORS)
            r, g, b = CLASS_COLORS[ci]
            color = QColor(r, g, b)
            is_editing = (ai == self._editing_obj_idx)

            if len(ann.points) >= 3:
                poly = QPolygonF()
                for px, py in ann.points:
                    wx, wy = self._img_to_widget(px, py)
                    poly.append(QPointF(wx, wy))
                painter.setBrush(QBrush(QColor(r, g, b, 50)))
                if is_editing:
                    painter.setPen(QPen(color, 2.0, Qt.DashLine))
                else:
                    painter.setPen(QPen(color, 1.5))
                painter.drawPolygon(poly)

            if is_editing:
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.setBrush(QBrush(color))
                for px, py in ann.points:
                    wx, wy = self._img_to_widget(px, py)
                    painter.drawEllipse(QPointF(wx, wy), pr + 1, pr + 1)

            if ann.category and len(ann.points) >= 3:
                cat_counters[ann.category] = cat_counters.get(ann.category, 0) + 1
                instance_num = cat_counters[ann.category]
                cx = sum(p[0] for p in ann.points) / len(ann.points)
                cy = sum(p[1] for p in ann.points) / len(ann.points)
                wcx, wcy = self._img_to_widget(cx, cy)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.drawEllipse(QPointF(wcx, wcy), 3, 3)
                label = f"{ann.category}_{instance_num}"
                font = painter.font()
                font.setPixelSize(max(10, min(14, int(12 * self._zoom ** 0.3))))
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QPen(QColor(0, 0, 0), 3))
                painter.drawText(QPointF(wcx + 5, wcy - 5), label)
                painter.setPen(QPen(QColor(255, 255, 255)))
                painter.drawText(QPointF(wcx + 5, wcy - 5), label)

        if self._drawing_active:
            if self.drawing_mode == "manual":
                self._paint_manual_drawing(painter, pr)
            elif self.drawing_mode == "ai":
                self._paint_ai_drawing(painter, pr)

        if self._drawing_accepted and self._pending_annotation:
            pen = QPen(QColor(0, 220, 0), 2)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 220, 0, 40)))
            poly = QPolygonF()
            for px, py in self._pending_annotation:
                wx, wy = self._img_to_widget(px, py)
                poly.append(QPointF(wx, wy))
            if len(self._pending_annotation) >= 3:
                painter.drawPolygon(poly)

        painter.end()

    def _paint_manual_drawing(self, painter: QPainter, pr: float):
        if not self._current_points:
            return
        painter.setPen(QPen(QColor(255, 220, 50), 2))
        for i in range(len(self._current_points) - 1):
            x1, y1 = self._img_to_widget(*self._current_points[i])
            x2, y2 = self._img_to_widget(*self._current_points[i + 1])
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
        painter.setBrush(QBrush(QColor(255, 220, 50)))
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        for px, py in self._current_points:
            wx, wy = self._img_to_widget(px, py)
            painter.drawEllipse(QPointF(wx, wy), pr, pr)

    def _paint_ai_drawing(self, painter: QPainter, pr: float):
        if self._ai_mask_contour_pts and len(self._ai_mask_contour_pts) >= 3:
            painter.setPen(QPen(QColor(255, 220, 50), 2))
            painter.setBrush(QBrush(QColor(255, 220, 50, 40)))
            poly = QPolygonF()
            for px, py in self._ai_mask_contour_pts:
                wx, wy = self._img_to_widget(px, py)
                poly.append(QPointF(wx, wy))
            painter.drawPolygon(poly)


# ======================================================================
# Worker threads
# ======================================================================
class SAM2LoadWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, segmenter):
        super().__init__()
        self.segmenter = segmenter

    def run(self):
        try:
            self.segmenter.load_model()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class SAM2PredictWorker(QThread):
    finished = pyqtSignal(np.ndarray, float)
    error = pyqtSignal(str)

    def __init__(self, segmenter, pos_points, neg_points):
        super().__init__()
        self.segmenter = segmenter
        self.pos_points = pos_points
        self.neg_points = neg_points

    def run(self):
        try:
            mask, score = self.segmenter.predict_mask(self.pos_points, self.neg_points)
            self.finished.emit(mask, score)
        except Exception as e:
            self.error.emit(str(e))


class AugmentWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, coco_json, images_dir, output_dir):
        super().__init__()
        self.coco_json = coco_json
        self.images_dir = images_dir
        self.output_dir = output_dir

    def run(self):
        try:
            from .mask_tracker_augmentation import augment_coco_dataset
            out_path = augment_coco_dataset(
                self.coco_json, self.images_dir, self.output_dir,
                progress_callback=lambda cur, total: self.progress.emit(cur, total),
            )
            self.finished.emit(out_path)
        except Exception as e:
            self.error.emit(str(e))


class TrainWorker(QThread):
    progress = pyqtSignal(int, int, object)  # iteration, total, metrics
    finished = pyqtSignal(object)  # summary dict
    error = pyqtSignal(str)

    def __init__(self, config_dict):
        super().__init__()
        self.config_dict = config_dict
        self._stop_flag = False
        self._pause_flag = False

    def request_stop(self):
        self._stop_flag = True
        self._pause_flag = False

    def request_pause(self):
        self._pause_flag = True

    def request_resume(self):
        self._pause_flag = False

    def run(self):
        # Suppress Metal stderr noise (harmless "command buffer" warnings)
        # at the file-descriptor level so it doesn't flood the terminal.
        import platform
        _saved_fd = None
        _null_fd = None
        if platform.system() == "Darwin":
            try:
                _saved_fd = os.dup(2)
                _null_fd = os.open(os.devnull, os.O_WRONLY)
                os.dup2(_null_fd, 2)
            except OSError:
                _saved_fd = None

        try:
            print("[MTT TrainWorker] Starting training thread...")
            print(f"[MTT TrainWorker] Config: {self.config_dict}")
            from .mask_tracker_training import MaskRCNNTrainingConfig, train_mask_rcnn
            config = MaskRCNNTrainingConfig(**self.config_dict)
            import time as _pause_time

            def _check_stop():
                while self._pause_flag and not self._stop_flag:
                    _pause_time.sleep(0.2)
                return self._stop_flag

            result = train_mask_rcnn(
                config,
                progress=lambda it, total, metrics: self.progress.emit(it, total, metrics),
                should_stop=_check_stop,
            )
            print(f"[MTT TrainWorker] Training finished: {result.get('iterations_completed')} iterations")
            self.finished.emit(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            if _saved_fd is not None:
                os.dup2(_saved_fd, 2)
                os.close(_saved_fd)
            if _null_fd is not None:
                os.close(_null_fd)


class InferenceWorker(QThread):
    progress = pyqtSignal(int, int)  # frame_idx, total_frames
    video_finished = pyqtSignal(object)  # result dict
    all_done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, video_paths, model_dir, config_dict):
        super().__init__()
        self.video_paths = list(video_paths)
        self.model_dir = model_dir
        self.config_dict = config_dict
        self._stop_flag = False
        self._pause_flag = False

    def request_stop(self):
        self._stop_flag = True
        self._pause_flag = False

    def request_pause(self):
        self._pause_flag = True

    def request_resume(self):
        self._pause_flag = False

    def run(self):
        try:
            from .mask_tracker_inference import MaskInferenceConfig, run_inference_on_video
            import time as _pause_time

            config = MaskInferenceConfig(**self.config_dict)

            def _check_stop():
                while self._pause_flag and not self._stop_flag:
                    _pause_time.sleep(0.2)
                return self._stop_flag

            for video_path in self.video_paths:
                if self._stop_flag:
                    break
                result = run_inference_on_video(
                    video_path,
                    self.model_dir,
                    config,
                    progress=lambda f, t: self.progress.emit(f, t),
                    should_stop=_check_stop,
                )
                result["video_path"] = video_path
                self.video_finished.emit(result)

            self.all_done.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class LiveLossPlot(QDialog):
    """Real-time training loss plot displayed during Mask R-CNN training."""

    def __init__(self, total_iterations: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Progress")
        self.setMinimumSize(600, 450)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self._iterations = []
        self._losses = []
        self._total = total_iterations

        try:
            import matplotlib
            matplotlib.use("Qt5Agg")
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(6, 3.5), dpi=100)
            self._fig.patch.set_facecolor("#1e1e1e")
            self._ax = self._fig.add_subplot(111)
            self._ax.set_facecolor("#1e1e1e")
            self._ax.set_xlabel("Iteration", color="#cccccc")
            self._ax.set_ylabel("Loss", color="#cccccc")
            self._ax.set_title("Training Loss", color="#cccccc")
            self._ax.tick_params(colors="#999999")
            for spine in self._ax.spines.values():
                spine.set_color("#555555")
            self._ax.set_xlim(0, 10)
            self._line, = self._ax.plot([], [], color="#2979ff", linewidth=1.5)
            self._fig.tight_layout()

            self._canvas = FigureCanvasQTAgg(self._fig)
            self._has_mpl = True
        except ImportError:
            self._canvas = QLabel("matplotlib not installed — loss values shown below")
            self._canvas.setAlignment(Qt.AlignCenter)
            self._canvas.setStyleSheet("color: #cccccc; font-size: 13px;")
            self._has_mpl = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._canvas)

        self._lbl_status = QLabel("Waiting for first iteration...")
        self._lbl_status.setStyleSheet("color: #cccccc; font-size: 12px;")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl_status)

    def add_point(self, iteration: int, loss: float):
        self._iterations.append(iteration)
        self._losses.append(loss)

        self._lbl_status.setText(
            f"Iteration {iteration} / {self._total}    Loss: {loss:.4f}"
        )

        if not self._has_mpl:
            return
        self._line.set_data(self._iterations, self._losses)
        self._ax.set_xlim(0, max(iteration * 1.1, 10))
        self._ax.set_ylim(0, max(self._losses) * 1.1 + 0.01)
        self._canvas.draw_idle()


class FrameExtractWorker(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, video_paths, frame_indices_per_video, output_dir):
        super().__init__()
        self.video_paths = video_paths
        self.frame_indices_per_video = frame_indices_per_video
        self.output_dir = output_dir

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            saved = []
            total = sum(len(v) for v in self.frame_indices_per_video.values())
            done = 0
            for vpath, frame_idxs in self.frame_indices_per_video.items():
                cap = cv2.VideoCapture(vpath)
                if not cap.isOpened():
                    continue
                stem = Path(vpath).stem
                for fidx in sorted(frame_idxs):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                    ret, frame = cap.read()
                    if ret:
                        fname = f"{stem}_frame_{fidx:06d}.png"
                        out_path = os.path.join(self.output_dir, fname)
                        cv2.imwrite(out_path, frame)
                        saved.append((vpath, fidx, out_path))
                    done += 1
                    self.progress.emit(done, total)
                cap.release()
            self.finished.emit(saved)
        except Exception as e:
            self.error.emit(str(e))


# ======================================================================
# Main window
# ======================================================================
class SAM2AnnotatorWindow(QMainWindow):
    BASE_TITLE = "FNT Mask Tracker Tool"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.BASE_TITLE)
        self.setMinimumSize(1100, 750)
        self.resize(1400, 900)
        self.setStyleSheet(DARK_STYLESHEET)

        self._project_dir: Optional[str] = None
        self._project_config: Dict = {}

        self.video_paths: List[str] = []
        self.current_video_idx: int = -1
        self._video_cap: Optional[cv2.VideoCapture] = None
        self._video_frame_count: int = 0
        self._video_fps: float = 30.0

        self._extracted_frames: List[Tuple[str, int, str]] = []
        self.current_frame_idx: int = -1
        self._output_dir: Optional[str] = None

        self._segmenter = None
        self._sam2_worker: Optional[SAM2LoadWorker] = None
        self._predict_worker: Optional[SAM2PredictWorker] = None
        self._image_set = False
        self._ai_enabled = False

        self._inference_worker: Optional[InferenceWorker] = None
        self._track_model_dirs: List[str] = []

        from .mask_tracker_annotator import COCOAnnotationManager
        self._coco = COCOAnnotationManager()
        self._categories: List[str] = []
        self._last_used_category: str = ""

        self._build_menu_bar()
        self._build_ui()
        self._update_nav_state()

    # ==================================================================
    # Menu bar
    # ==================================================================
    def _build_menu_bar(self):
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        file_menu = menu_bar.addMenu("File")

        act_new = QAction("New Project...", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self._new_project)
        file_menu.addAction(act_new)

        act_open = QAction("Open Project...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_project)
        file_menu.addAction(act_open)

        act_save = QAction("Save Project", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_export = QAction("Export COCO JSON...", self)
        act_export.triggered.connect(self._export_coco)
        file_menu.addAction(act_export)

        act_import = QAction("Import COCO JSON...", self)
        act_import.triggered.connect(self._import_coco)
        file_menu.addAction(act_import)

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left panel: tabs
        self.tab_widget = QTabWidget()
        fm = self.fontMetrics()
        min_w = max(340, fm.averageCharWidth() * 52 + 40)
        max_w = max(440, fm.averageCharWidth() * 70 + 40)
        self.tab_widget.setMinimumWidth(min_w)
        self.tab_widget.setMaximumWidth(max_w)

        self._build_annotator_tab()
        self._build_training_tab()
        self._build_tracking_tab()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self.preview = AnnotationPreviewWidget()
        self.preview.annotation_accepted.connect(self._on_annotation_accepted)
        self.preview.ai_prediction_requested.connect(self._request_ai_prediction)
        self.preview.zoom_changed.connect(
            lambda z: self.lbl_zoom_info.setText(f"{int(z * 100)}%")
        )
        self.preview.mode_changed.connect(self._on_mode_changed)
        self.preview.advance_frame_requested.connect(self._next_frame)
        self.preview.annotation_edited.connect(self._on_annotation_edited)
        self.preview.delete_annotation_requested.connect(self._on_delete_annotation_by_index)
        right_layout.addWidget(self.preview, 1)

        # Info row below preview
        info_row = QWidget()
        info_layout = QHBoxLayout(info_row)
        info_layout.setContentsMargins(5, 2, 5, 2)
        info_layout.setSpacing(8)

        self.lbl_mode = QLabel("Navigate")
        self.lbl_mode.setStyleSheet(
            "color: #2979ff; font-weight: bold; font-size: 11px; padding: 2px 8px;"
        )
        info_layout.addWidget(self.lbl_mode)

        self.lbl_frame_info = QLabel("Frame: — / —")
        self.lbl_frame_info.setStyleSheet("color: #999999;")
        info_layout.addWidget(self.lbl_frame_info)

        self.lbl_ann_stats = QLabel("")
        self.lbl_ann_stats.setStyleSheet("color: #999999; font-size: 10px;")
        info_layout.addWidget(self.lbl_ann_stats)

        info_layout.addStretch()

        self.btn_edit_classes = QPushButton("Edit Classes")
        self.btn_edit_classes.setStyleSheet(
            "QPushButton { background-color: #2979ff; color: white; font-weight: bold; "
            "padding: 3px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #448aff; }"
        )
        self.btn_edit_classes.clicked.connect(self._edit_classes)
        info_layout.addWidget(self.btn_edit_classes)

        self.lbl_zoom_info = QLabel("100%")
        self.lbl_zoom_info.setStyleSheet("color: #999999;")
        info_layout.addWidget(self.lbl_zoom_info)
        right_layout.addWidget(info_row)

        main_layout.addWidget(self.tab_widget)
        main_layout.addWidget(right_panel, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("File > New Project to begin, or load videos directly")

    def _build_annotator_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        self._create_videos_section(layout)
        self._create_extraction_section(layout)
        self._create_frames_section(layout)

        # Help text
        info = QLabel(
            "Right-click preview to add Manual or AI Mask.\n"
            "Manual: click to place vertices, Enter to accept.\n"
            "AI: left-click to include, right-click to exclude, Enter to accept.\n"
            "Right-click a mask to Edit or Delete it.\n"
            "Left-click + drag to pan. Scroll to zoom. Space = next frame."
        )
        info.setStyleSheet("color: #888888; font-size: 9px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        scroll.setWidget(widget)
        self.tab_widget.addTab(scroll, "Annotation")

    def _build_training_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # Training data summary
        summary_group = QGroupBox("Training Data")
        summary_vbox = QVBoxLayout()
        summary_vbox.setSpacing(2)
        self.lbl_train_images = QLabel("Annotated images: —")
        self.lbl_train_annotations = QLabel("Total annotations: —")
        self.lbl_train_classes = QLabel("Classes: —")
        for lbl in (self.lbl_train_images, self.lbl_train_annotations, self.lbl_train_classes):
            lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
            summary_vbox.addWidget(lbl)
        summary_group.setLayout(summary_vbox)
        layout.addWidget(summary_group)

        # Blue arrow buttons for spin boxes — generate tiny arrow PNGs
        # at runtime for cross-platform consistency.
        import tempfile
        arrow_dir = tempfile.mkdtemp(prefix="fnt_arrows_")
        for name, points in [("up", [(0,5),(4,0),(8,5)]), ("dn", [(0,0),(4,5),(8,0)])]:
            img = QImage(8, 6, QImage.Format_ARGB32)
            img.fill(QColor(0, 0, 0, 0))
            p = QPainter(img)
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(QBrush(QColor(255, 255, 255)))
            p.setPen(Qt.NoPen)
            poly = QPolygonF([QPointF(x, y) for x, y in points])
            p.drawPolygon(poly)
            p.end()
            img.save(os.path.join(arrow_dir, f"{name}.png"))
        _up_path = os.path.join(arrow_dir, "up.png").replace("\\", "/")
        _dn_path = os.path.join(arrow_dir, "dn.png").replace("\\", "/")
        self._spin_style = (
            "QSpinBox, QDoubleSpinBox {"
            "  background-color: #2b2b2b; color: #cccccc; border: 1px solid #555;"
            "}"
            "QSpinBox::up-button, QDoubleSpinBox::up-button,"
            "QSpinBox::down-button, QDoubleSpinBox::down-button {"
            "  background-color: #2979ff; border: none; width: 16px;"
            "}"
            "QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,"
            "QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {"
            "  background-color: #448aff;"
            "}"
            f"QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{"
            f"  image: url({_up_path}); width: 8px; height: 6px;"
            f"}}"
            f"QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{"
            f"  image: url({_dn_path}); width: 8px; height: 6px;"
            f"}}"
        )
        spin_style = self._spin_style

        # Training section
        train_group = QGroupBox("Train Mask R-CNN")
        train_vbox = QVBoxLayout()
        train_vbox.setSpacing(4)

        tip_device = (
            "Hardware accelerator for training.\n\n"
            "Auto (recommended): selects best available GPU automatically.\n"
            "  Priority: NVIDIA CUDA > Apple MPS > CPU.\n"
            "CUDA: NVIDIA GPU — fastest, requires NVIDIA hardware.\n"
            "MPS: Apple Silicon GPU via Metal Performance Shaders.\n"
            "  Uses unified memory shared with CPU. First iteration\n"
            "  is slow (~10-30s) due to Metal kernel compilation.\n"
            "CPU: always available but significantly slower."
        )
        row = QHBoxLayout()
        lbl = QLabel("Device:")
        lbl.setToolTip(tip_device)
        row.addWidget(lbl)
        self.combo_device = QComboBox()
        self.combo_device.addItems(["Auto", "CPU", "CUDA (NVIDIA)", "MPS (Apple Silicon)"])
        self.combo_device.setToolTip(tip_device)
        row.addWidget(self.combo_device)
        train_vbox.addLayout(row)

        tip_backbone = (
            "CNN backbone network that extracts visual features from images.\n"
            "All use FPN (Feature Pyramid Network) for multi-scale detection.\n\n"
            "MobileNetV3-Large FPN (~22M params): recommended default.\n"
            "  Uses depthwise separable convolutions — much faster than\n"
            "  ResNet with good accuracy. Best speed/accuracy tradeoff.\n"
            "MobileNetV3-Small FPN (~19M params): lightest and fastest.\n"
            "  Best for quick experiments or very limited GPU memory.\n"
            "  Slightly lower accuracy on complex scenes.\n"
            "ResNet-50 FPN (~44M params): heaviest, highest accuracy.\n"
            "  Uses standard convolutions stacked 50 layers deep.\n"
            "  Best for final production models when accuracy matters most."
        )
        row = QHBoxLayout()
        lbl = QLabel("Backbone:")
        lbl.setToolTip(tip_backbone)
        row.addWidget(lbl)
        self.combo_backbone = QComboBox()
        self.combo_backbone.addItems([
            "MobileNetV3-Large FPN",
            "MobileNetV3-Small FPN",
            "ResNet-50 FPN",
        ])
        self.combo_backbone.setToolTip(tip_backbone)
        row.addWidget(self.combo_backbone)
        train_vbox.addLayout(row)

        tip_freeze = (
            "Freeze pretrained backbone weights (default: on).\n\n"
            "ON (recommended for <100 annotated images): the backbone\n"
            "  was pretrained on ImageNet (1M+ images) and already knows\n"
            "  how to extract edges, textures, and shapes. Only the\n"
            "  detection/mask heads are trained to recognize your objects.\n"
            "  Much faster, uses less memory, and avoids overfitting.\n"
            "OFF: fine-tunes the entire network end-to-end. Use only\n"
            "  with large annotation sets (100+) where you want the\n"
            "  backbone to adapt to domain-specific features (e.g.,\n"
            "  infrared or unusual imaging modalities)."
        )
        self.chk_freeze_backbone = QCheckBox("Freeze backbone (train heads only)")
        self.chk_freeze_backbone.setChecked(True)
        self.chk_freeze_backbone.setToolTip(tip_freeze)
        train_vbox.addWidget(self.chk_freeze_backbone)

        tip_optimizer = (
            "Algorithm that updates model weights during training.\n\n"
            "AdamW (recommended): maintains adaptive learning rates for\n"
            "  each parameter independently. Converges faster, especially\n"
            "  on small datasets (<100 images). Less sensitive to the\n"
            "  learning rate setting. Uses slightly more GPU memory.\n"
            "SGD: classical stochastic gradient descent with momentum.\n"
            "  Requires more careful learning rate tuning and more\n"
            "  iterations, but can generalize better on large datasets.\n"
            "  Standard choice in academic benchmarks."
        )
        row = QHBoxLayout()
        lbl = QLabel("Optimizer:")
        lbl.setToolTip(tip_optimizer)
        row.addWidget(lbl)
        self.combo_optimizer = QComboBox()
        self.combo_optimizer.addItems(["AdamW", "SGD"])
        self.combo_optimizer.setToolTip(tip_optimizer)
        row.addWidget(self.combo_optimizer)
        train_vbox.addLayout(row)

        tip_iterations = (
            "Total number of weight updates to perform.\n\n"
            "Each iteration loads one batch of images, computes the\n"
            "loss, and updates the model weights. With AdamW and a\n"
            "frozen backbone, 200-500 iterations is often sufficient.\n"
            "With SGD or unfrozen backbone, 1000-5000 may be needed.\n"
            "Watch the loss plot — if loss is still dropping at the\n"
            "end, increase iterations. If it plateaus early, the\n"
            "early stopping feature will handle it automatically."
        )
        row = QHBoxLayout()
        lbl = QLabel("Iterations:")
        lbl.setToolTip(tip_iterations)
        row.addWidget(lbl)
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(100, 100000)
        self.spin_iterations.setValue(1000)
        self.spin_iterations.setSingleStep(100)
        self.spin_iterations.setToolTip(tip_iterations)
        self.spin_iterations.setStyleSheet(spin_style)
        row.addWidget(self.spin_iterations)
        train_vbox.addLayout(row)

        tip_lr = (
            "Controls how much weights change per iteration.\n\n"
            "Higher = faster convergence but risk of instability\n"
            "  (loss may spike or diverge).\n"
            "Lower = more stable but slower convergence.\n\n"
            "Recommended: 0.001-0.01 for AdamW, 0.005-0.02 for SGD.\n"
            "A warmup phase gradually ramps the LR from near-zero\n"
            "to this value over the first ~10% of iterations,\n"
            "preventing early instability."
        )
        row = QHBoxLayout()
        lbl = QLabel("Learning rate:")
        lbl.setToolTip(tip_lr)
        row.addWidget(lbl)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 0.1)
        self.spin_lr.setValue(0.005)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setToolTip(tip_lr)
        self.spin_lr.setStyleSheet(spin_style)
        row.addWidget(self.spin_lr)
        train_vbox.addLayout(row)

        tip_batch = (
            "Number of images processed per iteration (default: 2).\n\n"
            "Larger batches give smoother, more stable gradient\n"
            "estimates but use proportionally more GPU memory.\n"
            "Batch 2: good default for most setups.\n"
            "Batch 1: use if you hit out-of-memory errors.\n"
            "Batch 4+: use with small images or large GPU memory.\n"
            "Note: with very small datasets, larger batches can\n"
            "mean fewer unique images seen per epoch."
        )
        row = QHBoxLayout()
        lbl = QLabel("Batch size:")
        lbl.setToolTip(tip_batch)
        row.addWidget(lbl)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 16)
        self.spin_batch.setValue(2)
        self.spin_batch.setToolTip(tip_batch)
        self.spin_batch.setStyleSheet(spin_style)
        row.addWidget(self.spin_batch)
        train_vbox.addLayout(row)

        tip_val = (
            "Fraction of annotated images held out for validation.\n\n"
            "These images are NOT used for training. After each epoch,\n"
            "validation loss is checked to detect overfitting (model\n"
            "memorizes training images but fails on new ones).\n"
            "0.20 (20%): good default — e.g., 14 images → 11 train, 3 val.\n"
            "0.10: use with very small datasets (<10 images) to keep\n"
            "  more images for training.\n"
            "0.00: train on all images. Not recommended — no way to\n"
            "  detect overfitting."
        )
        row = QHBoxLayout()
        lbl = QLabel("Validation split:")
        lbl.setToolTip(tip_val)
        row.addWidget(lbl)
        self.spin_val_frac = QDoubleSpinBox()
        self.spin_val_frac.setRange(0.0, 0.5)
        self.spin_val_frac.setValue(0.2)
        self.spin_val_frac.setSingleStep(0.05)
        self.spin_val_frac.setDecimals(2)
        self.spin_val_frac.setToolTip(tip_val)
        self.spin_val_frac.setStyleSheet(spin_style)
        row.addWidget(self.spin_val_frac)
        train_vbox.addLayout(row)

        tip_min_size = (
            "Image shortest edge is resized to this many pixels.\n\n"
            "Auto-detected from your annotations to keep the smallest\n"
            "labeled object at ≥24px after resize. Directly controls\n"
            "training speed and GPU memory usage:\n"
            "  256px: fastest, lowest memory, coarsest detail.\n"
            "  384px: good balance for most animal tracking.\n"
            "  512px+: preserves fine detail for small objects.\n\n"
            "Images are also pre-resized on CPU before GPU transfer\n"
            "to avoid memory pressure. Reduce this value if training\n"
            "is slow or you hit out-of-memory errors."
        )
        row = QHBoxLayout()
        lbl = QLabel("Image min size:")
        lbl.setToolTip(tip_min_size)
        row.addWidget(lbl)
        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(256, 2048)
        self.spin_min_size.setValue(480)
        self.spin_min_size.setSingleStep(64)
        self.spin_min_size.setToolTip(tip_min_size)
        self.spin_min_size.setStyleSheet(spin_style)
        row.addWidget(self.spin_min_size)
        train_vbox.addLayout(row)

        self.chk_augment = QCheckBox("Data augmentation (~8x expansion)")
        self.chk_augment.setChecked(False)
        self.chk_augment.setToolTip(
            "Apply random transforms to training images (default: off).\n\n"
            "Generates augmented copies using random horizontal flips,\n"
            "brightness/contrast jitter, and spatial transforms.\n"
            "Expands the effective dataset ~8x, reducing overfitting\n"
            "when you have few annotated images (<20).\n\n"
            "OFF: faster training, use when iterating quickly.\n"
            "ON: slower to start (generates augmented images), but\n"
            "  produces more robust models that generalize better."
        )
        train_vbox.addWidget(self.chk_augment)

        tip_early_stop = (
            "Automatically stop training when loss stops improving.\n\n"
            "Monitors a 20-iteration moving average of training loss.\n"
            "If loss does not improve by >1% for 'patience' consecutive\n"
            "iterations, training stops and the best weights are kept.\n\n"
            "Patience 50-100: aggressive — stops quickly, saves time.\n"
            "Patience 200+: conservative — allows longer plateaus\n"
            "  before giving up, may find better minima.\n\n"
            "The best model weights (lowest loss) are always saved\n"
            "separately regardless of when training stops."
        )
        es_row = QHBoxLayout()
        self.chk_early_stop = QCheckBox("Early stopping, patience:")
        self.chk_early_stop.setChecked(True)
        self.chk_early_stop.setToolTip(tip_early_stop)
        es_row.addWidget(self.chk_early_stop)
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(20, 1000)
        self.spin_patience.setValue(100)
        self.spin_patience.setSingleStep(10)
        self.spin_patience.setToolTip(tip_early_stop)
        self.spin_patience.setStyleSheet(spin_style)
        es_row.addWidget(self.spin_patience)
        train_vbox.addLayout(es_row)

        self.btn_train = QPushButton("Train Model")
        self.btn_train.setStyleSheet(
            "QPushButton { background-color: #2979ff; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #448aff; }"
            "QPushButton:disabled { background-color: #333333; color: #666666; }"
        )
        self.btn_train.clicked.connect(self._start_training)
        train_vbox.addWidget(self.btn_train)

        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        train_vbox.addWidget(self.train_progress)

        self.lbl_train_status = QLabel("")
        self.lbl_train_status.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_train_status.setWordWrap(True)
        train_vbox.addWidget(self.lbl_train_status)

        btn_row = QHBoxLayout()
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setStyleSheet(
            "QPushButton { background-color: #f57c00; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #fb8c00; }"
        )
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_pause.setVisible(False)
        btn_row.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #e53935; }"
        )
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_stop.setVisible(False)
        btn_row.addWidget(self.btn_stop)
        train_vbox.addLayout(btn_row)

        train_group.setLayout(train_vbox)
        layout.addWidget(train_group)

        layout.addStretch()
        scroll.setWidget(widget)
        self.tab_widget.addTab(scroll, "Training")

    def _build_tracking_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # --- Model Selection ---
        model_group = QGroupBox("Trained Model")
        model_vbox = QVBoxLayout()
        model_vbox.setSpacing(4)

        model_row = QHBoxLayout()
        self.combo_track_model = QComboBox()
        self.combo_track_model.setToolTip(
            "Select a trained Mask R-CNN model from this project.\n"
            "Models are saved in the project's models/ directory after training."
        )
        self.combo_track_model.currentIndexChanged.connect(self._on_track_model_changed)
        model_row.addWidget(self.combo_track_model, 1)
        btn_refresh_models = QPushButton("Refresh")
        btn_refresh_models.setToolTip("Rescan the models/ directory for trained models.")
        btn_refresh_models.clicked.connect(self._refresh_model_list)
        model_row.addWidget(btn_refresh_models)
        model_vbox.addLayout(model_row)

        self.lbl_track_model_info = QLabel("No model selected")
        self.lbl_track_model_info.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_track_model_info.setWordWrap(True)
        model_vbox.addWidget(self.lbl_track_model_info)

        model_group.setLayout(model_vbox)
        layout.addWidget(model_group)

        # --- Video Queue ---
        video_group = QGroupBox("Video Queue")
        video_vbox = QVBoxLayout()
        video_vbox.setSpacing(4)

        vid_btn_row = QHBoxLayout()
        btn_add_vids = QPushButton("Add Videos")
        btn_add_vids.setToolTip("Add individual video files to the tracking queue.")
        btn_add_vids.clicked.connect(self._add_tracking_videos)
        vid_btn_row.addWidget(btn_add_vids)
        btn_add_vid_folder = QPushButton("Add Folder")
        btn_add_vid_folder.setToolTip(
            "Add all video files from a folder to the tracking queue."
        )
        btn_add_vid_folder.clicked.connect(self._add_tracking_folder)
        vid_btn_row.addWidget(btn_add_vid_folder)
        btn_clear_vids = QPushButton("Clear")
        btn_clear_vids.setToolTip("Remove all videos from the queue.")
        btn_clear_vids.clicked.connect(self._clear_tracking_videos)
        vid_btn_row.addWidget(btn_clear_vids)
        video_vbox.addLayout(vid_btn_row)

        self.list_track_videos = QListWidget()
        self.list_track_videos.setMaximumHeight(120)
        self.list_track_videos.setToolTip("Videos queued for tracking. All will be processed sequentially.")
        video_vbox.addWidget(self.list_track_videos)

        video_group.setLayout(video_vbox)
        layout.addWidget(video_group)

        # --- Inference Settings ---
        settings_group = QGroupBox("Inference Settings")
        settings_vbox = QVBoxLayout()
        settings_vbox.setSpacing(4)

        row0 = QHBoxLayout()
        row0.addWidget(QLabel("Max objects:"))
        self.spin_track_max_det = QSpinBox()
        self.spin_track_max_det.setRange(1, 100)
        self.spin_track_max_det.setValue(1)
        self.spin_track_max_det.setStyleSheet(self._spin_style)
        self.spin_track_max_det.setToolTip(
            "Maximum number of objects/masks to detect per frame (1–100).\n"
            "Set this to the number of animals or objects you expect\n"
            "in the video. Only the top-N highest-confidence detections\n"
            "are kept each frame. Default 1 for single-animal tracking."
        )
        row0.addWidget(self.spin_track_max_det)
        settings_vbox.addLayout(row0)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Confidence:"))
        self.spin_track_confidence = QDoubleSpinBox()
        self.spin_track_confidence.setRange(0.05, 0.99)
        self.spin_track_confidence.setValue(0.5)
        self.spin_track_confidence.setSingleStep(0.05)
        self.spin_track_confidence.setStyleSheet(self._spin_style)
        self.spin_track_confidence.setToolTip(
            "Minimum detection confidence score (0.05–0.99).\n"
            "Higher = fewer but more reliable detections.\n"
            "Lower = more detections but more false positives.\n"
            "Default 0.5 is a good starting point."
        )
        row1.addWidget(self.spin_track_confidence)
        settings_vbox.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Max disappear:"))
        self.combo_track_disappear = QComboBox()
        self.combo_track_disappear.addItems(["Never", "30", "60", "90", "150", "300"])
        self.combo_track_disappear.setToolTip(
            "Max frames an object can vanish before its track is dropped.\n"
            "Never: tracks persist for the entire video (recommended).\n"
            "30: ~1 second at 30fps. Use if objects leave and re-enter\n"
            "and you want separate track IDs for each appearance."
        )
        row2.addWidget(self.combo_track_disappear)
        row2.addWidget(QLabel("frames"))
        settings_vbox.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("IoU threshold:"))
        self.spin_track_iou = QDoubleSpinBox()
        self.spin_track_iou.setRange(0.05, 0.95)
        self.spin_track_iou.setValue(0.3)
        self.spin_track_iou.setSingleStep(0.05)
        self.spin_track_iou.setStyleSheet(self._spin_style)
        self.spin_track_iou.setToolTip(
            "Minimum mask IoU to match a detection to an existing track (0.05–0.95).\n"
            "Higher = stricter matching (fewer identity swaps, but more track breaks).\n"
            "Lower = more lenient (tracks persist longer, but may swap IDs).\n"
            "Default 0.3 works well for most scenarios."
        )
        row3.addWidget(self.spin_track_iou)
        settings_vbox.addLayout(row3)

        row_match = QHBoxLayout()
        row_match.addWidget(QLabel("Matching:"))
        self.combo_track_matching = QComboBox()
        self.combo_track_matching.addItems(["Hungarian (optimal)", "Greedy (fast)"])
        self.combo_track_matching.setToolTip(
            "Algorithm for matching detections to existing tracks.\n"
            "Hungarian: globally optimal assignment via linear sum assignment.\n"
            "  Best accuracy, slightly slower. Recommended default.\n"
            "Greedy: assigns nearest centroid pairs in order of distance.\n"
            "  Faster, but can make suboptimal assignments with many objects."
        )
        row_match.addWidget(self.combo_track_matching)
        settings_vbox.addLayout(row_match)

        row_res = QHBoxLayout()
        row_res.addWidget(QLabel("Resolution:"))
        self.combo_track_resolution = QComboBox()
        self.combo_track_resolution.addItems([
            "Trained (default)", "256px", "384px", "512px", "640px", "800px",
        ])
        self.combo_track_resolution.setToolTip(
            "Max image dimension sent to the model for inference.\n"
            "Trained: uses the resolution the model was trained at.\n"
            "Lower = faster inference but less accurate on small objects.\n"
            "For large objects relative to the frame, 384–512px is usually sufficient."
        )
        row_res.addWidget(self.combo_track_resolution)
        settings_vbox.addLayout(row_res)

        self.chk_track_masks = QCheckBox("Generate mask overlays (slower)")
        self.chk_track_masks.setChecked(False)
        self.chk_track_masks.setToolTip(
            "When enabled, the model also predicts pixel-level masks for each\n"
            "detection and draws them on the annotated video. ~2x slower.\n"
            "When disabled, uses bounding boxes only — faster and sufficient\n"
            "for tracking since centroids are computed from box centers."
        )
        settings_vbox.addWidget(self.chk_track_masks)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Device:"))
        self.combo_track_device = QComboBox()
        self.combo_track_device.addItems(["CPU", "Auto", "CUDA (GPU)", "MPS (Apple Silicon)"])
        self.combo_track_device.setToolTip(
            "Hardware device for inference.\n"
            "CPU: recommended for Apple Silicon Macs (~5x faster than MPS for detection).\n"
            "Auto: uses CUDA if available, otherwise CPU.\n"
            "CUDA: NVIDIA GPU (fastest when available).\n"
            "MPS: Apple Silicon GPU via Metal (slower than CPU for detection models)."
        )
        row4.addWidget(self.combo_track_device)
        settings_vbox.addLayout(row4)

        settings_group.setLayout(settings_vbox)
        layout.addWidget(settings_group)

        # --- Controls ---
        ctrl_group = QGroupBox("Run Tracking")
        ctrl_vbox = QVBoxLayout()
        ctrl_vbox.setSpacing(4)

        btn_row = QHBoxLayout()
        self.btn_start_tracking = QPushButton("Start Tracking")
        self.btn_start_tracking.setStyleSheet(
            "QPushButton { background-color: #2979ff; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #448aff; }"
            "QPushButton:disabled { background-color: #333333; color: #666666; }"
        )
        self.btn_start_tracking.setToolTip("Run the selected model on all queued videos.")
        self.btn_start_tracking.clicked.connect(self._start_tracking)
        self.btn_start_tracking.setEnabled(False)
        btn_row.addWidget(self.btn_start_tracking)

        self.btn_track_pause = QPushButton("Pause")
        self.btn_track_pause.setStyleSheet(
            "QPushButton { background-color: #f57c00; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #ff9800; }"
        )
        self.btn_track_pause.clicked.connect(self._toggle_tracking_pause)
        self.btn_track_pause.setVisible(False)
        btn_row.addWidget(self.btn_track_pause)

        self.btn_track_stop = QPushButton("Stop")
        self.btn_track_stop.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #e53935; }"
        )
        self.btn_track_stop.clicked.connect(self._stop_tracking)
        self.btn_track_stop.setVisible(False)
        btn_row.addWidget(self.btn_track_stop)
        ctrl_vbox.addLayout(btn_row)

        self.lbl_track_status = QLabel("Ready")
        self.lbl_track_status.setStyleSheet("color: #999999;")
        ctrl_vbox.addWidget(self.lbl_track_status)

        self.track_progress = QProgressBar()
        self.track_progress.setVisible(False)
        ctrl_vbox.addWidget(self.track_progress)

        ctrl_group.setLayout(ctrl_vbox)
        layout.addWidget(ctrl_group)

        # --- Results ---
        results_group = QGroupBox("Results")
        results_vbox = QVBoxLayout()
        self.list_track_results = QListWidget()
        self.list_track_results.setMaximumHeight(160)
        self.list_track_results.setToolTip(
            "Completed videos with track counts and CSV output paths.\n"
            "Double-click to open the CSV file location."
        )
        self.list_track_results.itemDoubleClicked.connect(self._open_tracking_result)
        results_vbox.addWidget(self.list_track_results)
        results_group.setLayout(results_vbox)
        layout.addWidget(results_group)

        layout.addStretch()
        scroll.setWidget(widget)
        self.tab_widget.addTab(scroll, "Tracking")

    # ------------------------------------------------------------------
    # Annotator tab sections
    # ------------------------------------------------------------------
    def _create_videos_section(self, layout):
        group = QGroupBox("1. Videos")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_add_folder = QPushButton("Add Folder(s)...")
        self.btn_add_folder.clicked.connect(self._add_video_folder)
        btn_row.addWidget(self.btn_add_folder)
        self.btn_add_files = QPushButton("Add File(s)...")
        self.btn_add_files.clicked.connect(self._add_video_files)
        btn_row.addWidget(self.btn_add_files)
        vbox.addLayout(btn_row)

        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(120)
        self.video_list.currentRowChanged.connect(self._on_video_selected)
        vbox.addWidget(self.video_list)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)
        self.btn_prev_video = QPushButton("< Prev")
        self.btn_prev_video.clicked.connect(self._prev_video)
        nav_row.addWidget(self.btn_prev_video)
        self.lbl_video_num = QLabel("0 / 0")
        self.lbl_video_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_video_num, 1)
        self.btn_next_video = QPushButton("Next >")
        self.btn_next_video.clicked.connect(self._next_video)
        nav_row.addWidget(self.btn_next_video)
        vbox.addLayout(nav_row)

        self.btn_remove_video = QPushButton("Remove Selected")
        self.btn_remove_video.clicked.connect(self._remove_selected_video)
        vbox.addWidget(self.btn_remove_video)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_extraction_section(self, layout):
        group = QGroupBox("2. Frame Extraction")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        row = QHBoxLayout()
        row.addWidget(QLabel("Method:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Uniform sample", "Random sample", "Every Nth frame"])
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        row.addWidget(self.combo_method, 1)
        vbox.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Target:"))
        self.combo_target = QComboBox()
        self.combo_target.addItems(["All videos", "Current video"])
        row.addWidget(self.combo_target, 1)
        vbox.addLayout(row)

        row = QHBoxLayout()
        self.lbl_count = QLabel("Frames per video:")
        row.addWidget(self.lbl_count)
        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 10000)
        self.spin_count.setValue(20)
        row.addWidget(self.spin_count)
        vbox.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Output:"))
        self.lbl_output_dir = QLabel("(auto)")
        self.lbl_output_dir.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_output_dir.setWordWrap(True)
        row.addWidget(self.lbl_output_dir, 1)
        self.btn_browse_output = QPushButton("...")
        self.btn_browse_output.setFixedWidth(30)
        self.btn_browse_output.clicked.connect(self._browse_output_dir)
        row.addWidget(self.btn_browse_output)
        vbox.addLayout(row)

        self.btn_generate = QPushButton("Generate Frames")
        self.btn_generate.setStyleSheet(
            "QPushButton { background-color: #2979ff; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #448aff; }"
            "QPushButton:disabled { background-color: #333333; color: #666666; }"
        )
        self.btn_generate.clicked.connect(self._generate_frames)
        vbox.addWidget(self.btn_generate)

        self.extract_progress = QProgressBar()
        self.extract_progress.setVisible(False)
        vbox.addWidget(self.extract_progress)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_frames_section(self, layout):
        group = QGroupBox("3. Extracted Frames")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        self.frame_list = QListWidget()
        self.frame_list.setMaximumHeight(140)
        self.frame_list.currentRowChanged.connect(self._on_frame_selected)
        vbox.addWidget(self.frame_list)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)
        self.btn_prev_frame = QPushButton("< Prev")
        self.btn_prev_frame.clicked.connect(self._prev_frame)
        nav_row.addWidget(self.btn_prev_frame)
        self.lbl_frame_num = QLabel("0 / 0")
        self.lbl_frame_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_frame_num, 1)
        self.btn_next_frame = QPushButton("Next >")
        self.btn_next_frame.clicked.connect(self._next_frame)
        nav_row.addWidget(self.btn_next_frame)
        vbox.addLayout(nav_row)

        self.btn_load_frames = QPushButton("Load Existing Frames...")
        self.btn_load_frames.clicked.connect(self._load_existing_frames)
        vbox.addWidget(self.btn_load_frames)

        group.setLayout(vbox)
        layout.addWidget(group)

    # ==================================================================
    # Project management
    # ==================================================================
    def _new_project(self):
        d = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not d:
            return
        self._project_dir = d
        os.makedirs(os.path.join(d, "training_frames"), exist_ok=True)
        os.makedirs(os.path.join(d, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(d, "models"), exist_ok=True)
        self._output_dir = os.path.join(d, "training_frames")
        self.lbl_output_dir.setText(self._output_dir)
        self.lbl_output_dir.setStyleSheet("color: #cccccc; font-size: 10px;")
        self._project_config = {
            "project_dir": d,
            "video_paths": [],
            "categories": [],
            "sam2_model_path": None,
        }
        self._coco.auto_save_path = os.path.join(d, "annotations", "annotations.json")
        self._save_project_config()
        self.setWindowTitle(f"{self.BASE_TITLE} — {os.path.basename(d)}")
        self.status_bar.showMessage(f"Created project: {d}")

    def _open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project Config", "",
            "JSON Files (project_config.json);;All Files (*)"
        )
        if not path:
            return
        self._load_project(path)

    def _load_project(self, config_path: str):
        with open(config_path) as f:
            cfg = json.load(f)
        self._project_dir = os.path.dirname(config_path)
        self._project_config = cfg

        saved_paths = cfg.get("video_paths", [])
        self.video_paths = [p for p in saved_paths if os.path.exists(p)]
        if len(self.video_paths) < len(saved_paths):
            n_missing = len(saved_paths) - len(self.video_paths)
            print(f"[MTT] Skipped {n_missing} video path(s) that no longer exist.")
        self._refresh_video_list(auto_select=bool(self.video_paths))

        for cat in cfg.get("categories", []):
            if cat not in self._categories:
                self._coco.add_category(cat)
                self._categories.append(cat)

        self._output_dir = os.path.join(self._project_dir, "training_frames")
        self.lbl_output_dir.setText(self._output_dir)
        self.lbl_output_dir.setStyleSheet("color: #cccccc; font-size: 10px;")

        if os.path.isdir(self._output_dir):
            self._scan_frames_dir(self._output_dir)
            if self._extracted_frames:
                self.frame_list.setCurrentRow(0)

        ann_path = os.path.join(self._project_dir, "annotations", "annotations.json")
        if os.path.exists(ann_path):
            self._coco.load(ann_path)
            self._categories = [c["name"] for c in self._coco.categories]
        self._coco.auto_save_path = ann_path

        self._refresh_frame_list()

        self.setWindowTitle(f"{self.BASE_TITLE} — {os.path.basename(self._project_dir)}")
        self.status_bar.showMessage(f"Opened project: {self._project_dir}")
        self._update_ann_stats()

    def _save_project(self):
        if self._project_dir is None:
            self._new_project()
            if self._project_dir is None:
                return
        self._project_config["video_paths"] = list(self.video_paths)
        self._project_config["categories"] = list(self._categories)
        self._save_project_config()
        ann_dir = os.path.join(self._project_dir, "annotations")
        os.makedirs(ann_dir, exist_ok=True)
        ann_path = os.path.join(ann_dir, "annotations.json")
        if self._coco.annotations:
            self._coco.export(ann_path)
        self.status_bar.showMessage(f"Project saved: {self._project_dir}")

    def _save_project_config(self):
        if self._project_dir is None:
            return
        path = os.path.join(self._project_dir, "project_config.json")
        with open(path, "w") as f:
            json.dump(self._project_config, f, indent=2)

    # ==================================================================
    # Videos logic
    # ==================================================================
    def _add_video_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if not folder:
            return
        added = 0
        for f in sorted(os.listdir(folder)):
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                full = os.path.join(folder, f)
                if full not in self.video_paths:
                    self.video_paths.append(full)
                    added += 1
        if added == 0:
            QMessageBox.information(self, "No Videos", "No video files found in selected folder.")
            return
        self._refresh_video_list()
        self.status_bar.showMessage(f"Added {added} video(s) from {os.path.basename(folder)}")

    def _add_video_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
        )
        if not files:
            return
        added = 0
        for f in files:
            if f not in self.video_paths:
                self.video_paths.append(f)
                added += 1
        self._refresh_video_list()
        self.status_bar.showMessage(f"Added {added} video(s)")

    def _refresh_video_list(self, auto_select: bool = True):
        self.video_list.blockSignals(True)
        self.video_list.clear()
        for vp in self.video_paths:
            item = QListWidgetItem(os.path.basename(vp))
            item.setToolTip(vp)
            self.video_list.addItem(item)
        self.video_list.blockSignals(False)
        if auto_select and self.video_paths and self.current_video_idx < 0:
            self.video_list.setCurrentRow(0)
        self._update_nav_state()

    def _remove_selected_video(self):
        row = self.video_list.currentRow()
        if row < 0:
            return
        self.video_paths.pop(row)
        self._close_video()
        self.current_video_idx = -1
        self._refresh_video_list()
        if self.video_paths:
            self.video_list.setCurrentRow(min(row, len(self.video_paths) - 1))

    def _on_video_selected(self, row: int):
        if row < 0 or row >= len(self.video_paths):
            return
        self.current_video_idx = row
        self._load_video(self.video_paths[row])
        self._update_nav_state()

    def _prev_video(self):
        if self.current_video_idx > 0:
            self.video_list.setCurrentRow(self.current_video_idx - 1)

    def _next_video(self):
        if self.current_video_idx < len(self.video_paths) - 1:
            self.video_list.setCurrentRow(self.current_video_idx + 1)

    def _load_video(self, path: str):
        self._close_video()
        self._video_cap = cv2.VideoCapture(path)
        if not self._video_cap.isOpened():
            QMessageBox.warning(self, "Error", f"Could not open video:\n{path}")
            return
        self._video_frame_count = int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_fps = self._video_cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._show_video_frame(0)
        self.status_bar.showMessage(
            f"Loaded: {os.path.basename(path)} "
            f"({self._video_frame_count} frames, {self._video_fps:.1f} fps)"
        )

    def _close_video(self):
        if self._video_cap is not None:
            self._video_cap.release()
            self._video_cap = None
        self._video_frame_count = 0

    def _show_video_frame(self, frame_idx: int):
        if self._video_cap is None:
            return
        self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._video_cap.read()
        if ret:
            self.preview.set_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # ==================================================================
    # Frame extraction
    # ==================================================================
    def _on_method_changed(self, idx):
        if idx == 2:
            self.lbl_count.setText("Stride (N):")
            self.spin_count.setValue(30)
        else:
            self.lbl_count.setText("Frames per video:")
            self.spin_count.setValue(20)

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._output_dir = d
            self.lbl_output_dir.setText(d)
            self.lbl_output_dir.setStyleSheet("color: #cccccc; font-size: 10px;")

    def _compute_frame_indices(self, total_frames: int) -> List[int]:
        method = self.combo_method.currentIndex()
        n = self.spin_count.value()
        if method == 0:
            if n >= total_frames:
                return list(range(total_frames))
            step = total_frames / n
            return [int(i * step) for i in range(n)]
        elif method == 1:
            n = min(n, total_frames)
            return sorted(random.sample(range(total_frames), n))
        elif method == 2:
            return list(range(0, total_frames, max(1, n)))
        return []

    def _generate_frames(self):
        if not self.video_paths:
            QMessageBox.warning(self, "No Videos", "Load videos first.")
            return
        target = self.combo_target.currentIndex()
        if target == 1:
            if self.current_video_idx < 0:
                QMessageBox.warning(self, "No Video", "Select a video first.")
                return
            vids = [self.video_paths[self.current_video_idx]]
        else:
            vids = list(self.video_paths)

        if self._project_dir is None:
            QMessageBox.information(
                self, "Save Project First",
                "Please save a project before generating frames.",
            )
            self._new_project()
            if self._project_dir is None:
                return

        if self._output_dir is None:
            self._output_dir = os.path.join(self._project_dir, "training_frames")
            self.lbl_output_dir.setText(self._output_dir)
            self.lbl_output_dir.setStyleSheet("color: #cccccc; font-size: 10px;")

        frame_map = {}
        for vp in vids:
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                continue
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            indices = self._compute_frame_indices(total)
            if indices:
                frame_map[vp] = indices

        if not frame_map:
            QMessageBox.warning(self, "No Frames", "Could not compute frames for any video.")
            return

        total = sum(len(v) for v in frame_map.values())
        self.extract_progress.setMaximum(total)
        self.extract_progress.setValue(0)
        self.extract_progress.setVisible(True)
        self.btn_generate.setEnabled(False)

        self._extract_worker = FrameExtractWorker(vids, frame_map, self._output_dir)
        self._extract_worker.progress.connect(lambda d, t: self.extract_progress.setValue(d))
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_worker.error.connect(self._on_extract_error)
        self._extract_worker.start()

    def _on_extract_finished(self, results: list):
        self.extract_progress.setVisible(False)
        self.btn_generate.setEnabled(True)
        self._extracted_frames = results
        self._refresh_frame_list()
        self.status_bar.showMessage(f"Extracted {len(results)} frames to {self._output_dir}")
        if self._extracted_frames:
            self.frame_list.setCurrentRow(0)

    def _on_extract_error(self, msg: str):
        self.extract_progress.setVisible(False)
        self.btn_generate.setEnabled(True)
        QMessageBox.critical(self, "Extraction Error", msg)

    # ==================================================================
    # Extracted frames
    # ==================================================================
    def _refresh_frame_list(self):
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for _, _, fp in self._extracted_frames:
            filename = os.path.basename(fp)
            n_ann = self._count_annotations_for_file(filename)
            if n_ann > 0:
                label = f"✔ {filename} ({n_ann})"
            else:
                label = f"   {filename}"
            item = QListWidgetItem(label)
            if n_ann > 0:
                item.setForeground(QColor("#4fc456"))
            self.frame_list.addItem(item)
        self.frame_list.blockSignals(False)
        self._update_nav_state()

    def _count_annotations_for_file(self, filename: str) -> int:
        if filename not in self._coco._image_id_map:
            return 0
        img_id = self._coco._image_id_map[filename]
        return len(self._coco.get_annotations_for_image(img_id))

    def _scan_frames_dir(self, folder: str):
        paths = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        )
        self._extracted_frames = [("", 0, p) for p in paths]
        self._refresh_frame_list()

    def _load_existing_frames(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Frames Folder")
        if not folder:
            return
        self._output_dir = folder
        self.lbl_output_dir.setText(folder)
        self.lbl_output_dir.setStyleSheet("color: #cccccc; font-size: 10px;")
        self._scan_frames_dir(folder)
        if self._extracted_frames:
            self.frame_list.setCurrentRow(0)
        self.status_bar.showMessage(f"Loaded {len(self._extracted_frames)} frames from {folder}")

    def _on_frame_selected(self, row: int):
        if row < 0 or row >= len(self._extracted_frames):
            return
        self.current_frame_idx = row
        _, _, img_path = self._extracted_frames[row]
        self._load_annotation_frame(img_path)
        self._update_nav_state()

    def _prev_frame(self):
        if self.current_frame_idx > 0:
            self.frame_list.setCurrentRow(self.current_frame_idx - 1)

    def _next_frame(self):
        if self.current_frame_idx < len(self._extracted_frames) - 1:
            self.frame_list.setCurrentRow(self.current_frame_idx + 1)

    def _load_annotation_frame(self, path: str):
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "Error", f"Could not read image:\n{path}")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.preview.clear()
        self.preview.set_frame(img_rgb)
        self._image_set = False
        if self._segmenter is not None and self._segmenter.predictor is not None:
            self._segmenter.set_image(img_rgb)
            self._image_set = True
        self._load_frame_annotations(path)
        self._update_info_labels()

    def _load_frame_annotations(self, path: str):
        filename = os.path.basename(path)
        self.preview.annotations.clear()

        if filename not in self._coco._image_id_map:
            self._update_ann_stats()
            return

        img_id = self._coco._image_id_map[filename]
        anns = self._coco.get_annotations_for_image(img_id)

        for ann in anns:
            cat_name = self._coco.get_category_name(ann["category_id"])
            points = []
            if ann["segmentation"]:
                seg = ann["segmentation"][0]
                for i in range(0, len(seg), 2):
                    points.append((float(seg[i]), float(seg[i + 1])))
            ann_obj = AnnotationObject(points, cat_name, ann_id=ann["id"])
            self.preview.annotations.append(ann_obj)

        self.preview.update()
        self._update_ann_stats()

    # ==================================================================
    # Mode changes
    # ==================================================================
    def _on_mode_changed(self, mode_name: str):
        self.lbl_mode.setText(mode_name)
        if self.preview.drawing_mode == "ai":
            self._ai_enabled = True
            self._ensure_sam2_loaded()
        else:
            self._ai_enabled = self.preview.drawing_mode == "ai"

    def _ensure_sam2_loaded(self):
        if self._segmenter is not None and self._segmenter.predictor is not None:
            return
        from .sam2_checkpoint_manager import SAM2_CHECKPOINTS, _FNT_REPO_ROOT, _find_existing_checkpoints

        search_dirs = [
            _FNT_REPO_ROOT / "sam_models_local",
            _FNT_REPO_ROOT / "SAM_models",
        ]
        custom_path = self._project_config.get("sam2_model_path")
        if custom_path and os.path.isdir(custom_path):
            search_dirs.insert(0, Path(custom_path))
        found = _find_existing_checkpoints(*search_dirs)

        if found:
            names = list(found.keys())
            descriptions = [f"{n} ({SAM2_CHECKPOINTS.get(n, {}).get('size_mb', '?')} MB)" for n in names]
            choice, ok = QInputDialog.getItem(
                self, "Select SAM2 Model", "Found existing SAM2 model(s). Select one:",
                descriptions, 0, False,
            )
            if ok:
                idx = descriptions.index(choice)
                self._load_sam2_model(str(found[names[idx]]), SAM2_CHECKPOINTS[names[idx]]["config"])
            else:
                self.preview.drawing_mode = "navigate"
                self.lbl_mode.setText("Navigate")
                self._ai_enabled = False
        else:
            reply = QMessageBox.question(
                self, "SAM2 Model Required",
                "No SAM2 model found. Would you like to download one?\n\n"
                "(Requires the sam2 Python package:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git)",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._show_sam2_download_dialog()
            else:
                self.preview.drawing_mode = "navigate"
                self.lbl_mode.setText("Navigate")
                self._ai_enabled = False

    def _show_sam2_download_dialog(self):
        try:
            from .sam2_checkpoint_manager import SAM2CheckpointDialog
            dialog = SAM2CheckpointDialog(parent=self)
            if dialog.exec_() == QDialog.Accepted:
                ckpt = dialog.get_checkpoint_path()
                config = dialog.get_config_name()
                if ckpt:
                    self._load_sam2_model(str(ckpt), config)
                    return
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
        self.preview.drawing_mode = "navigate"
        self.lbl_mode.setText("Navigate")
        self._ai_enabled = False

    def _load_sam2_model(self, checkpoint_path: str, config_name: str):
        try:
            from .mask_tracker_annotator import SAM2ImageSegmenter
        except ImportError as e:
            QMessageBox.critical(
                self, "Missing Package",
                f"SAM2 package not installed.\n\n"
                f"Install with:\n  pip install git+https://github.com/facebookresearch/sam2.git\n\n"
                f"Error: {e}"
            )
            self.preview.drawing_mode = "navigate"
            self.lbl_mode.setText("Navigate")
            self._ai_enabled = False
            return
        self._segmenter = SAM2ImageSegmenter(checkpoint_path, config_name)
        self.status_bar.showMessage(f"Loading SAM2 ({os.path.basename(checkpoint_path)})...")
        self._sam2_worker = SAM2LoadWorker(self._segmenter)
        self._sam2_worker.finished.connect(self._on_sam2_loaded)
        self._sam2_worker.error.connect(self._on_sam2_error)
        self._sam2_worker.start()
        self._project_config["sam2_model_path"] = os.path.dirname(checkpoint_path)

    def _on_sam2_loaded(self):
        self.status_bar.showMessage(f"SAM2 loaded ({self._segmenter.device})")
        if self.current_frame_idx >= 0 and not self._image_set:
            _, _, path = self._extracted_frames[self.current_frame_idx]
            img = cv2.imread(path)
            if img is not None:
                self._segmenter.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                self._image_set = True

    def _on_sam2_error(self, msg: str):
        self.status_bar.showMessage(f"SAM2 error: {msg}")
        QMessageBox.critical(self, "SAM2 Error", msg)
        self.preview.drawing_mode = "navigate"
        self.lbl_mode.setText("Navigate")
        self._ai_enabled = False

    # ==================================================================
    # AI prediction
    # ==================================================================
    def _request_ai_prediction(self):
        if self._segmenter is None or self._segmenter.predictor is None:
            return
        if not self._image_set:
            return
        pos = list(self.preview._ai_positive_points)
        neg = list(self.preview._ai_negative_points) or None
        if not pos:
            return
        self._predict_worker = SAM2PredictWorker(self._segmenter, pos, neg)
        self._predict_worker.finished.connect(self._on_prediction_done)
        self._predict_worker.error.connect(self._on_prediction_error)
        self._predict_worker.start()

    def _on_prediction_done(self, mask: np.ndarray, score: float):
        self.preview.set_ai_mask(mask, score)
        self.status_bar.showMessage(f"SAM2 prediction (score={score:.3f})")

    def _on_prediction_error(self, msg: str):
        self.status_bar.showMessage(f"Prediction error: {msg}")

    # ==================================================================
    # Class management
    # ==================================================================
    def _edit_classes(self):
        dialog = EditClassesDialog(list(self._categories), parent=self)
        if dialog.exec_() == QDialog.Accepted:
            for removed in dialog.get_removed():
                if removed in self._categories:
                    self._coco.remove_category(removed)
                    self._categories.remove(removed)
            new_cats = dialog.get_categories()
            for cat in new_cats:
                if cat not in self._categories:
                    self._categories.append(cat)
                    self._coco.add_category(cat)
            if self.current_frame_idx >= 0:
                _, _, path = self._extracted_frames[self.current_frame_idx]
                self._load_frame_annotations(path)
            self._update_ann_stats()

    # ==================================================================
    # Annotation logic
    # ==================================================================
    def _on_annotation_accepted(self):
        if self._categories:
            dialog = CategorySelectDialog(
                self._categories, self._last_used_category, parent=self
            )
            if dialog.exec_() != QDialog.Accepted:
                self.preview._clear_drawing()
                return
            cat = dialog.selected_category()
            if not cat:
                self.preview._clear_drawing()
                return
        else:
            cat, ok = QInputDialog.getText(
                self, "Assign Category", "Type category name (e.g. 'vole'):"
            )
            if not ok or not cat.strip():
                self.preview._clear_drawing()
                return
            cat = cat.strip()

        if cat not in self._categories:
            self._categories.append(cat)
            self._coco.add_category(cat)

        self._last_used_category = cat

        if self.current_frame_idx < 0:
            return

        _, _, path = self._extracted_frames[self.current_frame_idx]
        filename = os.path.basename(path)
        img = cv2.imread(path)
        h, w = img.shape[:2]
        img_id = self._coco.get_or_add_image(filename, w, h)
        cat_id = self._coco._category_name_map[cat]

        pending_mask = self.preview.get_pending_ai_mask()
        if pending_mask is not None:
            mask = pending_mask
        else:
            points = self.preview.get_pending_points()
            if not points or len(points) < 3:
                self.preview._clear_drawing()
                return
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
            mask = mask.astype(bool)

        ann_id = self._coco.add_annotation(img_id, cat_id, mask)
        if ann_id < 0:
            QMessageBox.warning(self, "Error", "Annotation too small.")
            self.preview._clear_drawing()
            return

        self.preview.accept_annotation(cat, ann_id=ann_id)
        self._update_ann_stats()
        self.status_bar.showMessage(f"Annotation saved: {cat} (id={ann_id})")

    def _on_annotation_edited(self, obj_idx: int):
        if obj_idx < 0 or obj_idx >= len(self.preview.annotations):
            return
        ann = self.preview.annotations[obj_idx]
        if ann.ann_id >= 0:
            self._coco.update_annotation_polygon(ann.ann_id, ann.points)
            self.status_bar.showMessage(f"Annotation {ann.ann_id} updated")

    def _on_delete_annotation_by_index(self, obj_idx: int):
        if obj_idx < 0 or obj_idx >= len(self.preview.annotations):
            return
        ann = self.preview.annotations[obj_idx]
        ann_id = ann.ann_id
        if ann_id >= 0:
            self._coco.remove_annotation(ann_id)
        self.preview.annotations.pop(obj_idx)
        self.preview.update()
        self._update_ann_stats()
        self.status_bar.showMessage(f"Annotation {ann_id} deleted")

    # ==================================================================
    # Export / Import (kept in File menu)
    # ==================================================================
    def _export_coco(self):
        stats = self._coco.get_stats()
        if stats["num_annotations"] == 0:
            QMessageBox.warning(self, "No Annotations", "Create some annotations first.")
            return
        default_path = ""
        if self._project_dir:
            default_path = os.path.join(self._project_dir, "annotations", "annotations.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export COCO JSON", default_path, "JSON Files (*.json)"
        )
        if not path:
            return
        self._coco.export(path)
        self.status_bar.showMessage(f"Exported {stats['num_annotations']} annotations to {path}")

    def _import_coco(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import COCO JSON", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        self._coco.load(path)
        self._categories = [c["name"] for c in self._coco.categories]
        if self.current_frame_idx >= 0:
            _, _, img_path = self._extracted_frames[self.current_frame_idx]
            self._load_frame_annotations(img_path)
        self._update_ann_stats()
        self.status_bar.showMessage(f"Imported annotations from {path}")

    # ==================================================================
    # Training
    # ==================================================================
    def _start_training(self):
        stats = self._coco.get_stats()
        if stats["num_annotations"] == 0:
            QMessageBox.warning(self, "No Annotations", "Annotate some frames first.")
            return
        if not self._project_dir:
            QMessageBox.warning(self, "No Project", "Save a project first.")
            return

        ann_path = self._coco.auto_save_path
        if not ann_path or not os.path.exists(ann_path):
            ann_path = os.path.join(self._project_dir, "annotations", "annotations.json")
            self._coco.export(ann_path)

        images_dir = self._output_dir or os.path.join(self._project_dir, "training_frames")

        if self.chk_augment.isChecked():
            self.btn_train.setEnabled(False)
            self.lbl_train_status.setText("Augmenting dataset...")
            self.status_bar.showMessage("Running data augmentation...")
            aug_dir = os.path.join(os.path.dirname(images_dir), "augmented")
            self._pending_train_config = {
                "images_dir": images_dir,
                "ann_path": ann_path,
                "aug_dir": aug_dir,
            }
            self._augment_worker = AugmentWorker(ann_path, images_dir, aug_dir)
            self._augment_worker.progress.connect(
                lambda c, t: self.lbl_train_status.setText(f"Augmenting... {c}/{t} images")
            )
            self._augment_worker.finished.connect(self._on_augment_then_train)
            self._augment_worker.error.connect(self._on_train_error)
            self._augment_worker.start()
        else:
            self._launch_training(ann_path, images_dir)

    def _on_augment_then_train(self, aug_output_path: str):
        aug_dir = self._pending_train_config["aug_dir"]
        aug_json = os.path.join(aug_dir, "annotations.json")
        if os.path.exists(aug_json):
            self._launch_training(aug_json, aug_dir)
        else:
            self._launch_training(
                self._pending_train_config["ann_path"],
                self._pending_train_config["images_dir"],
            )

    def _launch_training(self, coco_json: str, images_dir: str):
        combo_text = self.combo_device.currentText()
        if combo_text.startswith("Auto"):
            device_choice = "auto"
        elif combo_text == "CPU":
            device_choice = "cpu"
        elif combo_text.startswith("CUDA"):
            device_choice = "cuda"
        elif combo_text.startswith("MPS"):
            device_choice = "mps"
        else:
            device_choice = "auto"
        backbone_text = self.combo_backbone.currentText()
        if "Small" in backbone_text:
            backbone = "mobilenet_v3_small"
        elif "Large" in backbone_text:
            backbone = "mobilenet_v3_large"
        else:
            backbone = "resnet50"
        optimizer = "adamw" if self.combo_optimizer.currentText() == "AdamW" else "sgd"
        config_dict = {
            "coco_json_path": coco_json,
            "images_dir": images_dir,
            "output_dir": os.path.join(self._project_dir, "models"),
            "num_iterations": self.spin_iterations.value(),
            "learning_rate": self.spin_lr.value(),
            "batch_size": self.spin_batch.value(),
            "val_fraction": self.spin_val_frac.value(),
            "min_size": self.spin_min_size.value(),
            "device": device_choice,
            "backbone": backbone,
            "freeze_backbone": self.chk_freeze_backbone.isChecked(),
            "optimizer": optimizer,
            "early_stop_patience": self.spin_patience.value() if self.chk_early_stop.isChecked() else 0,
        }

        self.btn_train.setEnabled(False)
        total = self.spin_iterations.value()
        self.train_progress.setMaximum(total)
        self.train_progress.setValue(0)
        self.train_progress.setVisible(True)
        self.btn_pause.setVisible(True)
        self.btn_pause.setText("Pause")
        self.btn_stop.setVisible(True)
        self.lbl_train_status.setText("Starting training...")

        self._loss_plot = LiveLossPlot(total_iterations=total, parent=self)
        self._loss_plot.show()

        self._train_worker = TrainWorker(config_dict)
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.start()

    def _toggle_pause(self):
        if not hasattr(self, "_train_worker") or not self._train_worker.isRunning():
            return
        if self._train_worker._pause_flag:
            self._train_worker.request_resume()
            self.btn_pause.setText("Pause")
            self.btn_pause.setStyleSheet(
                "QPushButton { background-color: #f57c00; color: white; font-weight: bold; }"
                "QPushButton:hover { background-color: #fb8c00; }"
            )
            self.lbl_train_status.setText("Resumed training...")
        else:
            self._train_worker.request_pause()
            self.btn_pause.setText("Resume")
            self.btn_pause.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; font-weight: bold; }"
                "QPushButton:hover { background-color: #43a047; }"
            )
            self.lbl_train_status.setText("Training paused")

    def _stop_training(self):
        if hasattr(self, "_train_worker") and self._train_worker.isRunning():
            self._train_worker.request_stop()
            self.lbl_train_status.setText("Stopping training...")

    def _on_train_progress(self, iteration: int, total: int, metrics: dict):
        self.train_progress.setValue(iteration)
        loss = metrics.get("loss", 0.0)
        self.lbl_train_status.setText(f"Iteration {iteration}/{total}  —  loss: {loss:.4f}")
        if hasattr(self, "_loss_plot") and self._loss_plot is not None:
            self._loss_plot.add_point(iteration, loss)

    def _on_train_finished(self, summary: dict):
        self.train_progress.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_pause.setVisible(False)
        self.btn_stop.setVisible(False)
        if hasattr(self, "_loss_plot") and self._loss_plot is not None:
            self._loss_plot.close()
            self._loss_plot = None

        model_path = summary.get("model_path", "")
        run_dir = summary.get("run_dir", "")
        early = summary.get("early_stopped", False)
        iters = summary.get("iterations_completed", "?")
        best = summary.get("best_loss")
        best_str = f"{best:.4f}" if best is not None else "N/A"
        stopped_by_user = hasattr(self, "_train_worker") and self._train_worker._stop_flag

        if stopped_by_user and not early:
            status = f"Training stopped by user at iteration {iters}. Best loss: {best_str}"
            title = "Training Stopped"
            detail = (f"Training was stopped at iteration {iters}.\n"
                      f"Best loss: {best_str}\n\n"
                      f"Model saved to:\n{run_dir}")
        elif early:
            status = f"Training stopped early (plateau) at iteration {iters}. Best loss: {best_str}"
            title = "Training Complete (Early Stop)"
            detail = (f"Training stopped early — loss plateaued after {iters} iterations.\n"
                      f"Best loss: {best_str}\n\n"
                      f"Model saved to:\n{run_dir}")
        else:
            status = f"Training complete ({iters} iterations). Best loss: {best_str}"
            title = "Training Complete"
            detail = (f"Completed {iters} iterations.\n"
                      f"Best loss: {best_str}\n\n"
                      f"Model saved to:\n{run_dir}")

        self.lbl_train_status.setText(status)
        QMessageBox.information(self, title, detail)
        self.status_bar.showMessage(f"Training complete: {model_path}")

    def _on_train_error(self, msg: str):
        self.train_progress.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_pause.setVisible(False)
        self.btn_stop.setVisible(False)
        if hasattr(self, "_loss_plot") and self._loss_plot is not None:
            self._loss_plot.close()
            self._loss_plot = None
        self.lbl_train_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Training Error", msg)

    # ==================================================================
    # Tracking
    # ==================================================================
    def _refresh_model_list(self):
        self.combo_track_model.clear()
        self._track_model_dirs = []
        if not self._project_dir:
            self.lbl_track_model_info.setText("No project open")
            self._update_tracking_button_state()
            return

        models_root = os.path.join(self._project_dir, "models")
        if not os.path.isdir(models_root):
            self.lbl_track_model_info.setText("No models/ directory found — train a model first")
            self._update_tracking_button_state()
            return

        for entry in sorted(os.listdir(models_root)):
            run_dir = os.path.join(models_root, entry)
            if not os.path.isdir(run_dir):
                continue
            has_weights = (
                os.path.exists(os.path.join(run_dir, "weights_best.pt"))
                or os.path.exists(os.path.join(run_dir, "weights.pt"))
            )
            if not has_weights:
                continue

            label = entry
            config_path = os.path.join(run_dir, "training_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        cfg = json.load(f)
                    backbone = cfg.get("backbone", "?")
                    n_cls = cfg.get("num_classes", "?")
                    label = f"{entry}  [{backbone}, {n_cls} classes]"
                except Exception:
                    pass

            self.combo_track_model.addItem(label)
            self._track_model_dirs.append(run_dir)

        if not self._track_model_dirs:
            self.lbl_track_model_info.setText("No trained models found — train a model first")
        else:
            self._on_track_model_changed(0)
        self._update_tracking_button_state()

    def _on_track_model_changed(self, index: int):
        if index < 0 or index >= len(getattr(self, "_track_model_dirs", [])):
            self.lbl_track_model_info.setText("No model selected")
            return
        run_dir = self._track_model_dirs[index]
        config_path = os.path.join(run_dir, "training_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                cats = cfg.get("categories", {})
                cat_names = list(cats.values()) if isinstance(cats, dict) else cats
                max_sz = cfg.get("max_size", 800)
                self.lbl_track_model_info.setText(
                    f"Path: {run_dir}\n"
                    f"Backbone: {cfg.get('backbone', '?')}  |  "
                    f"Classes: {', '.join(str(c) for c in cat_names) if cat_names else '?'}  |  "
                    f"Min size: {cfg.get('min_size', '?')}"
                )
                self.combo_track_resolution.setItemText(0, f"Trained ({max_sz}px)")
            except Exception:
                self.lbl_track_model_info.setText(f"Path: {run_dir}")
                self.combo_track_resolution.setItemText(0, "Trained (default)")
        else:
            self.lbl_track_model_info.setText(f"Path: {run_dir}")
            self.combo_track_resolution.setItemText(0, "Trained (default)")
        self._update_tracking_button_state()

    def _add_tracking_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)",
        )
        for f in files:
            self.list_track_videos.addItem(os.path.basename(f))
            self.list_track_videos.item(
                self.list_track_videos.count() - 1
            ).setData(Qt.UserRole, f)
        self._update_tracking_button_state()

    def _add_tracking_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if not folder:
            return
        for fname in sorted(os.listdir(folder)):
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                full = os.path.join(folder, fname)
                self.list_track_videos.addItem(fname)
                self.list_track_videos.item(
                    self.list_track_videos.count() - 1
                ).setData(Qt.UserRole, full)
        self._update_tracking_button_state()

    def _clear_tracking_videos(self):
        self.list_track_videos.clear()
        self._update_tracking_button_state()

    def _update_tracking_button_state(self):
        has_model = bool(getattr(self, "_track_model_dirs", []))
        has_videos = self.list_track_videos.count() > 0
        self.btn_start_tracking.setEnabled(has_model and has_videos)

    def _start_tracking(self):
        model_idx = self.combo_track_model.currentIndex()
        if model_idx < 0 or model_idx >= len(self._track_model_dirs):
            QMessageBox.warning(self, "No Model", "Select a trained model first.")
            return
        if self.list_track_videos.count() == 0:
            QMessageBox.warning(self, "No Videos", "Add videos to the queue first.")
            return

        model_dir = self._track_model_dirs[model_idx]
        video_paths = []
        for i in range(self.list_track_videos.count()):
            video_paths.append(self.list_track_videos.item(i).data(Qt.UserRole))

        existing = []
        for vp in video_paths:
            out_dir = os.path.join(
                str(Path(vp).parent), f"{Path(vp).stem}_MaskTracker"
            )
            if os.path.isdir(out_dir):
                existing.append(os.path.basename(vp))

        if existing:
            names = "\n".join(f"  • {n}" for n in existing)
            reply = QMessageBox.question(
                self,
                "Existing Results Found",
                f"The following video(s) already have MaskTracker output:\n\n"
                f"{names}\n\n"
                f"Overwrite existing results?",
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return

        combo_text = self.combo_track_device.currentText()
        if combo_text.startswith("Auto"):
            device = "auto"
        elif combo_text == "CPU":
            device = "cpu"
        elif combo_text.startswith("CUDA"):
            device = "cuda"
        elif combo_text.startswith("MPS"):
            device = "mps"
        else:
            device = "auto"

        disappear_text = self.combo_track_disappear.currentText()
        max_disappear = 0 if disappear_text == "Never" else int(disappear_text)

        match_text = self.combo_track_matching.currentText()
        matching_algo = "greedy" if match_text.startswith("Greedy") else "hungarian"

        res_text = self.combo_track_resolution.currentText()
        inference_size = 0 if res_text.startswith("Trained") else int(res_text.replace("px", ""))

        config_dict = {
            "confidence_threshold": self.spin_track_confidence.value(),
            "max_detections": self.spin_track_max_det.value(),
            "max_disappeared_frames": max_disappear,
            "iou_match_threshold": self.spin_track_iou.value(),
            "matching_algorithm": matching_algo,
            "inference_size": inference_size,
            "use_masks": self.chk_track_masks.isChecked(),
            "device": device,
        }

        self._track_video_paths = video_paths
        self._track_videos_done = 0
        self._track_total_videos = len(video_paths)
        self.list_track_results.clear()

        self.btn_start_tracking.setEnabled(False)
        self.btn_track_pause.setVisible(True)
        self.btn_track_pause.setText("Pause")
        self.btn_track_stop.setVisible(True)
        self.track_progress.setVisible(True)
        self.track_progress.setValue(0)
        self.lbl_track_status.setText(
            f"Processing video 1 of {self._track_total_videos}: "
            f"{os.path.basename(video_paths[0])}"
        )

        self._inference_worker = InferenceWorker(video_paths, model_dir, config_dict)
        self._inference_worker.progress.connect(self._on_tracking_progress)
        self._inference_worker.video_finished.connect(self._on_video_finished)
        self._inference_worker.all_done.connect(self._on_all_tracking_done)
        self._inference_worker.error.connect(self._on_tracking_error)
        self._inference_worker.start()

    def _on_tracking_progress(self, frame_idx: int, total_frames: int):
        self.track_progress.setMaximum(total_frames)
        self.track_progress.setValue(frame_idx)
        vid_num = self._track_videos_done + 1
        vid_name = os.path.basename(self._track_video_paths[self._track_videos_done])
        self.lbl_track_status.setText(
            f"Processing video {vid_num} of {self._track_total_videos}: "
            f"{vid_name}  —  frame {frame_idx}/{total_frames}"
        )

    def _on_video_finished(self, result: dict):
        self._track_videos_done += 1
        video_name = os.path.basename(result.get("video_path", ""))
        n_tracks = result.get("num_tracks", 0)
        output_dir = result.get("output_dir", "")
        total_frames = result.get("total_frames", 0)

        item = QListWidgetItem(
            f"{video_name}: {n_tracks} tracks, {total_frames} frames -> {os.path.basename(output_dir)}/"
        )
        item.setData(Qt.UserRole, output_dir)
        item.setForeground(QColor("#4fc456"))
        self.list_track_results.addItem(item)

        if self._track_videos_done < self._track_total_videos:
            next_name = os.path.basename(
                self._track_video_paths[self._track_videos_done]
            )
            self.lbl_track_status.setText(
                f"Processing video {self._track_videos_done + 1} of "
                f"{self._track_total_videos}: {next_name}"
            )
            self.track_progress.setValue(0)

    def _on_all_tracking_done(self):
        self.track_progress.setVisible(False)
        self.btn_start_tracking.setEnabled(True)
        self.btn_track_pause.setVisible(False)
        self.btn_track_stop.setVisible(False)
        self.lbl_track_status.setText(
            f"Done — {self._track_videos_done} video(s) processed"
        )
        self.status_bar.showMessage(
            f"Tracking complete: {self._track_videos_done} video(s)"
        )

    def _toggle_tracking_pause(self):
        if not hasattr(self, "_inference_worker") or not self._inference_worker.isRunning():
            return
        if self._inference_worker._pause_flag:
            self._inference_worker.request_resume()
            self.btn_track_pause.setText("Pause")
            self.btn_track_pause.setStyleSheet(
                "QPushButton { background-color: #f57c00; color: white; font-weight: bold; "
                "padding: 8px; border-radius: 3px; }"
                "QPushButton:hover { background-color: #ff9800; }"
            )
            self.lbl_track_status.setText("Resumed tracking...")
        else:
            self._inference_worker.request_pause()
            self.btn_track_pause.setText("Resume")
            self.btn_track_pause.setStyleSheet(
                "QPushButton { background-color: #2e7d32; color: white; font-weight: bold; "
                "padding: 8px; border-radius: 3px; }"
                "QPushButton:hover { background-color: #43a047; }"
            )
            self.lbl_track_status.setText("Tracking paused")

    def _stop_tracking(self):
        if hasattr(self, "_inference_worker") and self._inference_worker.isRunning():
            self._inference_worker.request_stop()
            self.lbl_track_status.setText("Stopping tracking...")

    def _on_tracking_error(self, msg: str):
        self.track_progress.setVisible(False)
        self.btn_start_tracking.setEnabled(True)
        self.btn_track_pause.setVisible(False)
        self.btn_track_stop.setVisible(False)
        self.lbl_track_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Tracking Error", msg)

    def _open_tracking_result(self, item):
        output_dir = item.data(Qt.UserRole)
        if output_dir and os.path.isdir(output_dir):
            import subprocess, platform
            if platform.system() == "Darwin":
                subprocess.Popen(["open", output_dir])
            elif platform.system() == "Windows":
                subprocess.Popen(["explorer", output_dir])
            else:
                subprocess.Popen(["xdg-open", output_dir])

    def _on_tab_changed(self, index: int):
        tab_name = self.tab_widget.tabText(index)
        if tab_name == "Training":
            self._refresh_training_summary()
            self._refresh_device_label()
        elif tab_name == "Tracking":
            self._refresh_model_list()
            self.preview.clear()
            self.preview.update()
        elif tab_name == "Annotation":
            if self.current_frame_idx >= 0 and self.current_frame_idx < len(self._extracted_frames):
                _, _, path = self._extracted_frames[self.current_frame_idx]
                self._load_annotation_frame(path)

    def _refresh_device_label(self):
        if self.combo_device.currentIndex() == 0:
            try:
                from .mask_tracker_training import _resolve_device, get_device_description
                dev = _resolve_device("auto")
                desc = get_device_description(dev)
                self.combo_device.setItemText(0, f"Auto ({desc})")
            except Exception:
                self.combo_device.setItemText(0, "Auto")

    def _refresh_training_summary(self):
        stats = self._coco.get_stats()
        n_img = stats["images_with_annotations"]
        n_ann = stats["num_annotations"]
        cats = stats["categories"]
        self.lbl_train_images.setText(f"Annotated images: {n_img}")
        self.lbl_train_annotations.setText(f"Total annotations: {n_ann}")
        if cats:
            self.lbl_train_classes.setText(f"Classes ({len(cats)}): {', '.join(cats)}")
        else:
            self.lbl_train_classes.setText("Classes: —")

        suggested, details = self._coco.suggest_min_size()
        if suggested is not None:
            self.spin_min_size.setValue(suggested)
            self.spin_min_size.setToolTip(
                f"Auto-detected: {suggested}px\n"
                f"Smallest mask edge: {details['smallest_mask_dim']:.0f}px, "
                f"image shortest edge: {details['shortest_edge']}px\n"
                f"Keeps smallest object ≥24px after resize.\n"
                "Smaller = faster training, larger = finer detail."
            )
            print(f"[MTT] Auto min_size={suggested}px "
                  f"(smallest mask={details['smallest_mask_dim']:.0f}px, "
                  f"image edge={details['shortest_edge']}px)")

    # ==================================================================
    # Helpers
    # ==================================================================
    def _update_nav_state(self):
        n_vids = len(self.video_paths)
        self.btn_prev_video.setEnabled(self.current_video_idx > 0)
        self.btn_next_video.setEnabled(self.current_video_idx < n_vids - 1)
        self.btn_remove_video.setEnabled(self.current_video_idx >= 0)
        self.lbl_video_num.setText(
            f"{self.current_video_idx + 1} / {n_vids}" if n_vids else "0 / 0"
        )
        n_frames = len(self._extracted_frames)
        self.btn_prev_frame.setEnabled(self.current_frame_idx > 0)
        self.btn_next_frame.setEnabled(self.current_frame_idx < n_frames - 1)
        self.lbl_frame_num.setText(
            f"{self.current_frame_idx + 1} / {n_frames}" if n_frames else "0 / 0"
        )

    def _update_info_labels(self):
        n_frames = len(self._extracted_frames)
        self.lbl_frame_info.setText(
            f"Frame: {self.current_frame_idx + 1} of {n_frames}" if n_frames else "Frame: — / —"
        )
        self.lbl_zoom_info.setText(f"{int(self.preview._zoom * 100)}%")

    def _update_ann_stats(self):
        stats = self._coco.get_stats()
        n_this_frame = len(self.preview.annotations)
        if stats["num_annotations"] == 0:
            self.lbl_ann_stats.setText("")
        else:
            self.lbl_ann_stats.setText(
                f"{n_this_frame} on frame  |  "
                f"{stats['num_annotations']} total across {stats['images_with_annotations']} images"
            )
            self.lbl_ann_stats.setStyleSheet("color: #4caf50; font-size: 10px;")
        self._update_frame_list_item(self.current_frame_idx)

    def _update_frame_list_item(self, row: int):
        if row < 0 or row >= len(self._extracted_frames):
            return
        item = self.frame_list.item(row)
        if item is None:
            return
        _, _, fp = self._extracted_frames[row]
        filename = os.path.basename(fp)
        n_ann = self._count_annotations_for_file(filename)
        if n_ann > 0:
            item.setText(f"✔ {filename} ({n_ann})")
            item.setForeground(QColor("#4fc456"))
        else:
            item.setText(f"   {filename}")
            item.setForeground(QColor("#cccccc"))

    def closeEvent(self, event):
        self._close_video()
        stats = self._coco.get_stats()
        if stats["num_annotations"] > 0:
            reply = QMessageBox.question(
                self, "Unsaved Work",
                f"You have {stats['num_annotations']} annotations.\n"
                "Save project before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Yes:
                self._save_project()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        event.accept()
