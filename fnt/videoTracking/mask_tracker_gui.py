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
import shutil
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
    QTreeWidget, QTreeWidgetItem, QHeaderView,
    QInputDialog, QStatusBar, QFrame, QMenu, QTabWidget, QTabBar,
    QDoubleSpinBox, QDialogButtonBox, QLineEdit, QCheckBox, QSlider,
    QTextEdit, QSplitter,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QTimer
from PyQt5.QtGui import (
    QIcon, QImage, QPixmap, QFont, QColor, QPainter, QPen, QBrush,
    QPolygonF, QWheelEvent, QMouseEvent, QKeyEvent, QPainterPath,
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
    QTreeWidget { background-color: #1e1e1e; border: 1px solid #444444;
                  color: #cccccc; }
    QTreeWidget::item:selected { background-color: #2979ff; color: white; }
    QHeaderView::section { background-color: #333333; color: #cccccc;
                           border: 1px solid #444444; padding: 2px 6px; }
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


BEHAVIOR_PRESETS = {
    "Rodent Solo": [
        "locomotion", "idle", "rearing", "self-grooming",
    ],
    "Rodent Social": [
        "locomotion", "idle", "rearing", "self-grooming",
        "huddle", "attack", "flee",
    ],
}


# ======================================================================
# Annotation data structures
# ======================================================================
class AnnotationObject:
    def __init__(self, points: List[Tuple[float, float]], category: str,
                 ann_id: int = -1, is_ai: bool = False, inferred: bool = False):
        self.points = list(points)
        self.category = category
        self.ann_id = ann_id
        self.is_ai = is_ai
        self.inferred = inferred


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
# Edit Object Classes dialog
# ======================================================================
class EditClassesDialog(QDialog):
    def __init__(self, categories: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Object Classes")
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
    approve_annotation_requested = pyqtSignal(int)

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

        self.annotation_keys_enabled = True

        self._drag_obj_idx: Optional[int] = None
        self._drag_pt_idx: Optional[int] = None
        self._dragging_point = False

        self._editing_obj_idx: Optional[int] = None

        self.annotations: List[AnnotationObject] = []

        self._pending_annotation: Optional[List[Tuple[float, float]]] = None
        self._pending_is_ai = False

    def set_frame(self, image_rgb: np.ndarray, reset_view: bool = False):
        h, w = image_rgb.shape[:2]
        self._image_rgb = image_rgb
        self._img_w, self._img_h = w, h
        bytes_per_line = 3 * w
        qimg = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        if reset_view or self._zoom == 1.0:
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

    def _insert_contour_point(self, wx: float, wy: float):
        """Insert a new editable vertex on the contour edge closest to the click."""
        if self._editing_obj_idx is None:
            return
        ann = self.annotations[self._editing_obj_idx]
        pts = ann.points
        if len(pts) < 3:
            return

        best_dist = float("inf")
        best_seg = -1
        best_proj = None

        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            ax, ay = self._img_to_widget(*pts[i])
            bx, by = self._img_to_widget(*pts[j])
            dx, dy = bx - ax, by - ay
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-6:
                continue
            t = max(0.0, min(1.0, ((wx - ax) * dx + (wy - ay) * dy) / seg_len_sq))
            px, py = ax + t * dx, ay + t * dy
            d = math.hypot(wx - px, wy - py)
            if d < best_dist:
                best_dist = d
                best_seg = i
                best_proj = (t, pts[i], pts[j])

        max_dist = max(15.0, 20.0 / self._zoom)
        if best_proj is None or best_dist > max_dist:
            return

        t, (x0, y0), (x1, y1) = best_proj
        new_x = x0 + t * (x1 - x0)
        new_y = y0 + t * (y1 - y0)
        insert_idx = best_seg + 1
        ann.points.insert(insert_idx, (new_x, new_y))

        self._drag_obj_idx = self._editing_obj_idx
        self._drag_pt_idx = insert_idx
        self._dragging_point = True
        self.update()

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

    def mousePressEvent(self, event: QMouseEvent):
        if self._pixmap is None:
            return

        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = (event.x(), event.y(), self._pan_x, self._pan_y)
            self.setCursor(Qt.ClosedHandCursor)
            return

        if event.button() == Qt.RightButton:
            if self.annotation_keys_enabled:
                if self._editing_obj_idx is not None:
                    self._insert_contour_point(event.x(), event.y())
                    return
                if self.drawing_mode == "ai" and self._drawing_active:
                    coords = self._widget_to_img(event.x(), event.y())
                    if coords:
                        self._ai_negative_points.append((int(coords[0]), int(coords[1])))
                        self.update()
                        self._request_ai_prediction()
                    return
                self._show_context_menu(event.globalPos(), event.x(), event.y())
            elif hasattr(self, "_cls_mask_handler") and self._cls_mask_handler:
                coords = self._widget_to_img(event.x(), event.y())
                if coords:
                    handled = self._cls_mask_handler(
                        int(coords[0]), int(coords[1]), event.globalPos()
                    )
                    if handled:
                        return
            return

        if event.button() == Qt.LeftButton:
            if not self.annotation_keys_enabled:
                self._panning = True
                self._pan_start = (event.x(), event.y(), self._pan_x, self._pan_y)
                self.setCursor(Qt.ClosedHandCursor)
                return

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

            if self._editing_obj_idx is not None:
                return

            if self.drawing_mode == "manual":
                if not self._drawing_active:
                    self._drawing_active = True
                    self._drawing_accepted = False
                    self._current_points.clear()
                self._current_points.append((ix, iy))
                self.update()

            elif self.drawing_mode == "ai":
                # In AI mode, single-click pans (like navigate mode).
                # Double-click places points — see mouseDoubleClickEvent.
                self._panning = True
                self._pan_start = (event.x(), event.y(), self._pan_x, self._pan_y)
                self.setCursor(Qt.ClosedHandCursor)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Double-click places AI mask points when in SAM labeling mode."""
        if self._pixmap is None:
            return
        if event.button() != Qt.LeftButton:
            return
        if not self.annotation_keys_enabled:
            return
        if self.drawing_mode != "ai":
            return

        coords = self._widget_to_img(event.x(), event.y())
        if coords is None:
            return
        ix, iy = coords

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
        menu.addSeparator()
        hit_idx = self._find_annotation_at(wx, wy)
        act_approve = menu.addAction("✔ Approve Mask")
        act_edit = menu.addAction("Edit Mask")
        act_delete = menu.addAction("Delete Mask")
        if hit_idx is None:
            act_approve.setEnabled(False)
            act_edit.setEnabled(False)
            act_delete.setEnabled(False)
        elif hit_idx is not None:
            ann = self.annotations[hit_idx]
            if not ann.inferred:
                act_approve.setEnabled(False)
        # Approve all inferred on this frame
        act_approve_all = None
        has_any_inferred = any(a.inferred for a in self.annotations)
        if has_any_inferred:
            menu.addSeparator()
            act_approve_all = menu.addAction("✔ Approve All on Frame")

        act_clear_all = None
        if self.annotations:
            menu.addSeparator()
            act_clear_all = menu.addAction("Clear All Masks on Frame")

        action = menu.exec_(global_pos)
        if action == act_manual:
            self.drawing_mode = "manual"
            self.mode_changed.emit("Manual Mask")
        elif action == act_approve and hit_idx is not None:
            self.approve_annotation_requested.emit(hit_idx)
        elif action == act_approve_all and act_approve_all is not None:
            # Approve all inferred annotations on this frame
            for i in range(len(self.annotations) - 1, -1, -1):
                if self.annotations[i].inferred:
                    self.approve_annotation_requested.emit(i)
        elif action == act_edit and hit_idx is not None:
            if self.annotations[hit_idx].inferred:
                self.approve_annotation_requested.emit(hit_idx)
            self._editing_obj_idx = hit_idx
            self.mode_changed.emit("Editing Mask")
            self.update()
        elif action == act_delete and hit_idx is not None:
            self.delete_annotation_requested.emit(hit_idx)
        elif action == act_clear_all and act_clear_all is not None:
            for i in range(len(self.annotations) - 1, -1, -1):
                self.delete_annotation_requested.emit(i)

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
        if not self.annotation_keys_enabled:
            super().keyPressEvent(event)
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self._editing_obj_idx is not None:
                idx = self._editing_obj_idx
                self._editing_obj_idx = None
                self.mode_changed.emit("Navigate")
                self.update()
                self.annotation_edited.emit(idx)
            elif self._drawing_active:
                self._finish_annotation()
            else:
                self.advance_frame_requested.emit()
        elif event.key() == Qt.Key_Escape:
            if self._editing_obj_idx is not None:
                self._editing_obj_idx = None
                self.mode_changed.emit("Navigate")
                self.update()
            elif self.drawing_mode == "ai" and self._drawing_active:
                # Cancel current AI prediction but stay in AI mode
                self._clear_drawing()
                self.update()
            else:
                self._clear_drawing()
                self.drawing_mode = "navigate"
                self.mode_changed.emit("Navigate")
        elif event.key() == Qt.Key_Space:
            self.advance_frame_requested.emit()
        elif event.key() == Qt.Key_A and not self._drawing_active:
            # Approve all inferred annotations on current frame
            has_inferred = any(a.inferred for a in self.annotations)
            if has_inferred:
                for i in range(len(self.annotations) - 1, -1, -1):
                    if self.annotations[i].inferred:
                        self.approve_annotation_requested.emit(i)
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
            self._clear_drawing()
            self.mode_changed.emit("AI-Assisted Mask")

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
            if ann.inferred:
                r, g, b = (255, 200, 0)
            else:
                ci = ai % len(CLASS_COLORS)
                r, g, b = CLASS_COLORS[ci]
            color = QColor(r, g, b)
            is_editing = (ai == self._editing_obj_idx)

            if len(ann.points) >= 3:
                poly = QPolygonF()
                for px, py in ann.points:
                    wx, wy = self._img_to_widget(px, py)
                    poly.append(QPointF(wx, wy))
                painter.setBrush(Qt.NoBrush)
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
                if ann.inferred:
                    label += " [inferred]"
                font = painter.font()
                font.setPixelSize(max(10, min(14, int(12 * self._zoom ** 0.3))))
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QPen(QColor(0, 0, 0), 3))
                painter.drawText(QPointF(wcx + 5, wcy - 5), label)
                lbl_color = QColor(255, 200, 0) if ann.inferred else QColor(255, 255, 255)
                painter.setPen(QPen(lbl_color))
                painter.drawText(QPointF(wcx + 5, wcy - 5), label)

        if self._drawing_active:
            if self.drawing_mode == "manual":
                self._paint_manual_drawing(painter, pr)
            elif self.drawing_mode == "ai":
                self._paint_ai_drawing(painter, pr)

        if self._drawing_accepted and self._pending_annotation:
            pen = QPen(QColor(0, 220, 0), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
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
            painter.setBrush(Qt.NoBrush)
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

    def __init__(self, coco_json, images_dir, output_dir, config=None):
        super().__init__()
        self.coco_json = coco_json
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.config = config

    def run(self):
        try:
            from .mask_tracker_augmentation import augment_coco_dataset
            out_path = augment_coco_dataset(
                self.coco_json, self.images_dir, self.output_dir,
                config=self.config,
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
            import time as _pause_time

            arch = self.config_dict.pop("_architecture", "maskrcnn")

            def _check_stop():
                while self._pause_flag and not self._stop_flag:
                    _pause_time.sleep(0.2)
                return self._stop_flag

            if arch == "yolo":
                from .yolo_training import YOLOTrainingConfig, train_yolo_seg
                config = YOLOTrainingConfig(**self.config_dict)
                result = train_yolo_seg(
                    config,
                    progress=lambda it, total, metrics: self.progress.emit(it, total, metrics),
                    should_stop=_check_stop,
                )
            else:
                from .mask_tracker_training import MaskRCNNTrainingConfig, train_mask_rcnn
                config = MaskRCNNTrainingConfig(**self.config_dict)
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


class PostTrainInferenceWorker(QThread):
    """Run the trained model on unlabeled frames and emit inferred annotations."""
    progress = pyqtSignal(int, int)  # current_frame, total_frames
    frame_result = pyqtSignal(str, object)  # filename, list of {segmentation, bbox, area, category_id}
    finished = pyqtSignal(int)  # total inferred count
    error = pyqtSignal(str)

    def __init__(self, model_dir: str, frame_paths: list, max_det: int = 1,
                 confidence: float = 0.5, categories: dict = None):
        super().__init__()
        self.model_dir = model_dir
        self.frame_paths = frame_paths
        self.max_det = max_det
        self.confidence = confidence
        self.categories = categories or {}
        self._stop_flag = False

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        try:
            import json as _json

            config_path = os.path.join(self.model_dir, "training_config.json")
            arch = "maskrcnn"
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = _json.load(f)
                arch = cfg.get("architecture", "maskrcnn")

            if arch == "yolov11-seg":
                from .yolo_inference import YOLOInference
                engine = YOLOInference(self.model_dir, device="auto", use_masks=True)
            else:
                from .mask_tracker_inference import MaskRCNNInference
                engine = MaskRCNNInference(self.model_dir, device="auto")

            engine.load_model()

            total = len(self.frame_paths)
            inferred_count = 0

            for i, path in enumerate(self.frame_paths):
                if self._stop_flag:
                    break

                img = cv2.imread(path)
                if img is None:
                    self.progress.emit(i + 1, total)
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = engine.predict(
                    img_rgb,
                    confidence_threshold=self.confidence,
                    max_detections=self.max_det,
                )

                detections = []
                n_det = len(result.get("scores", []))
                for j in range(n_det):
                    mask = None
                    if result.get("masks") is not None and j < len(result["masks"]):
                        mask = result["masks"][j]

                    if mask is not None:
                        from .mask_tracker_annotator import mask_to_coco_polygons, mask_bbox
                        polygons = mask_to_coco_polygons(mask)
                        if not polygons:
                            continue
                        bbox = list(mask_bbox(mask))
                        area = int(mask.sum())
                    else:
                        # Use bounding box as a rectangle polygon
                        box = result["boxes"][j]
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        polygons = [[x1, y1, x2, y1, x2, y2, x1, y2]]
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        area = int((x2 - x1) * (y2 - y1))

                    label = int(result["labels"][j])
                    score = float(result["scores"][j])
                    detections.append({
                        "segmentation": polygons,
                        "bbox": bbox,
                        "area": area,
                        "category_id": label,
                        "score": score,
                    })

                if detections:
                    filename = os.path.basename(path)
                    self.frame_result.emit(filename, detections)
                    inferred_count += len(detections)

                self.progress.emit(i + 1, total)

            self.finished.emit(inferred_count)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class InferenceWorker(QThread):
    progress = pyqtSignal(int, int)  # frame_idx, total_frames
    frame_ready = pyqtSignal(object, int)  # annotated_rgb (np.ndarray), frame_idx
    video_finished = pyqtSignal(object)  # result dict
    all_done = pyqtSignal()
    error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, video_paths, model_dir, config_dict,
                 classifier_dir=None, nc_threshold=0.5,
                 cls_window=15, show_preview=True,
                 min_bout=5, uncertain_gap=0.10,
                 create_tracked_video=True):
        super().__init__()
        self.video_paths = list(video_paths)
        self.model_dir = model_dir
        self.config_dict = config_dict
        self.classifier_dir = classifier_dir
        self.nc_threshold = nc_threshold
        self.cls_window = cls_window
        self.show_preview = show_preview
        self.min_bout = min_bout
        self.uncertain_gap = uncertain_gap
        self.create_tracked_video = create_tracked_video
        self._stop_flag = False
        self._pause_flag = False

    def request_stop(self):
        self._stop_flag = True
        self._pause_flag = False

    def request_pause(self):
        self._pause_flag = True

    def request_resume(self):
        self._pause_flag = False

    def _load_classifier(self):
        """Load the behavior classifier model. Returns (model, class_names, device) or None."""
        if not self.classifier_dir:
            return None
        import torch
        from torchvision import models
        cfg_path = os.path.join(self.classifier_dir, "classifier_config.json")
        weights_path = os.path.join(self.classifier_dir, "best_classifier.pth")
        if not os.path.isfile(cfg_path) or not os.path.isfile(weights_path):
            print(f"[Classifier Inference] Missing config or weights in {self.classifier_dir}")
            return None

        with open(cfg_path) as f:
            cfg = json.load(f)
        class_names = cfg.get("class_names", [])
        n_classes = cfg.get("n_classes", len(class_names))
        backbone = cfg.get("backbone", "ResNet-18")

        from torch import nn
        if backbone == "ResNet-34":
            model = models.resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif backbone == "MobileNetV3":
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        else:
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, n_classes)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

        model.load_state_dict(torch.load(weights_path, map_location=device))
        model = model.to(device)
        model.eval()

        print(f"[Classifier Inference] Loaded {backbone} ({n_classes} classes: "
              f"{', '.join(class_names)}) on {device.upper()}")
        return model, class_names, device

    @staticmethod
    def _apply_min_bout_filter(rows, min_bout):
        """Replace behavior bouts shorter than min_bout frames with NC.

        Groups consecutive same-behavior predictions per object. Bouts
        shorter than min_bout are relabeled NC to suppress flickering.
        """
        from itertools import groupby

        per_obj = {}
        for i, r in enumerate(rows):
            per_obj.setdefault(r["object_id"], []).append(i)

        for obj_id, indices in per_obj.items():
            labels = [rows[i]["behavior"] for i in indices]
            for _, grp in groupby(enumerate(labels), key=lambda x: x[1]):
                grp_list = list(grp)
                if len(grp_list) < min_bout and grp_list[0][1] != "NC":
                    for local_pos, _ in grp_list:
                        rows[indices[local_pos]]["behavior"] = "NC"

    def run(self):
        import io, sys
        _real_stdout = sys.stdout

        class _LogCapture(io.TextIOBase):
            def __init__(self, worker, real):
                self._worker = worker
                self._real = real
            def write(self, s):
                if s and s.strip():
                    self._worker.log.emit(s.rstrip())
                return self._real.write(s)
            def flush(self):
                return self._real.flush()

        sys.stdout = _LogCapture(self, _real_stdout)
        try:
            from .mask_tracker_inference import MaskInferenceConfig, run_inference_on_video
            from .silhouette_extractor import generate_composite
            import time as _pause_time

            config = MaskInferenceConfig(**self.config_dict)

            cls_info = self._load_classifier()
            cls_model = cls_class_names = cls_device = None
            torch = None
            if cls_info:
                cls_model, cls_class_names, cls_device = cls_info
                import torch

            def _check_stop():
                while self._pause_flag and not self._stop_flag:
                    _pause_time.sleep(0.2)
                return self._stop_flag

            def _on_frame(frame_rgb, frame_idx):
                if self.show_preview:
                    self.frame_ready.emit(frame_rgb, frame_idx)

            for video_path in self.video_paths:
                if self._stop_flag:
                    break

                from collections import deque
                mask_buffers: dict = {}
                current_labels: dict = {}
                behavior_rows: list = []

                def _track_cb(matched, frame_idx, fps):
                    if cls_model is None:
                        return None

                    for obj_id, det in matched.items():
                        if obj_id not in mask_buffers:
                            mask_buffers[obj_id] = deque(maxlen=self.cls_window)
                        mask = det.get("mask")
                        if mask is not None:
                            bbox = det["bbox"]
                            h_f = mask.shape[0]
                            w_f = mask.shape[1]
                            x1 = max(0, int(round(float(bbox[0]))))
                            y1 = max(0, int(round(float(bbox[1]))))
                            x2 = min(w_f, int(round(float(bbox[2]))))
                            y2 = min(h_f, int(round(float(bbox[3]))))
                            if x2 > x1 and y2 > y1:
                                mask_buffers[obj_id].append({
                                    "crop": mask[y1:y2, x1:x2],
                                    "bbox": (x1, y1, x2, y2),
                                })
                            else:
                                mask_buffers[obj_id].append(None)
                        else:
                            mask_buffers[obj_id].append(None)

                    for obj_id in list(mask_buffers.keys()):
                        buf = mask_buffers[obj_id]
                        if len(buf) < self.cls_window:
                            continue

                        window = list(buf)
                        crops = [w["crop"] if w is not None else None for w in window]
                        boxes = [w["bbox"] if w is not None else None for w in window]
                        composite = generate_composite(crops, output_size=(128, 128), bboxes=boxes)
                        img = composite.astype(np.float32) / 255.0
                        inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(cls_device)
                        with torch.no_grad():
                            logits = cls_model(inp)
                            probs = torch.softmax(logits, dim=1)[0]
                            sorted_probs, sorted_idx = probs.sort(descending=True)
                            conf = sorted_probs[0].item()
                            pred_idx = sorted_idx[0].item()
                            gap = (sorted_probs[0] - sorted_probs[1]).item() if len(sorted_probs) > 1 else conf

                        if conf < self.nc_threshold or gap < self.uncertain_gap:
                            beh_name = "NC"
                        else:
                            beh_name = cls_class_names[pred_idx]
                        current_labels[obj_id] = (beh_name, conf)

                        behavior_rows.append({
                            "frame": frame_idx,
                            "object_id": obj_id,
                            "behavior": beh_name,
                            "confidence": round(conf, 4),
                            "gap": round(gap, 4),
                        })

                    return current_labels if current_labels else None

                result = run_inference_on_video(
                    video_path,
                    self.model_dir,
                    config,
                    progress=lambda f, t: self.progress.emit(f, t),
                    should_stop=_check_stop,
                    frame_callback=_on_frame,
                    track_callback=_track_cb if cls_model else None,
                    create_tracked_video=self.create_tracked_video,
                )
                result["video_path"] = video_path

                if self._stop_flag:
                    out_dir = result.get("output_dir", "")
                    if out_dir and os.path.isdir(out_dir):
                        import shutil
                        shutil.rmtree(out_dir, ignore_errors=True)
                        print(f"[MTT Inference] Removed incomplete output: {out_dir}")
                    break

                if behavior_rows:
                    per_obj_first = {}
                    for row in behavior_rows:
                        oid = row["object_id"]
                        if oid not in per_obj_first:
                            per_obj_first[oid] = row
                    backfill = []
                    for oid, first_row in per_obj_first.items():
                        first_frame = first_row["frame"]
                        for fi in range(first_frame):
                            backfill.append({
                                "frame": fi,
                                "object_id": oid,
                                "behavior": first_row["behavior"],
                                "confidence": first_row["confidence"],
                                "gap": first_row["gap"],
                            })
                    if backfill:
                        behavior_rows = backfill + behavior_rows

                    self._apply_min_bout_filter(behavior_rows, self.min_bout)

                    import csv
                    out_dir = result.get("output_dir", "")
                    beh_csv = os.path.join(out_dir, "behavior_predictions.csv")
                    fieldnames = ["frame", "object_id", "behavior", "confidence", "gap"]
                    behavior_rows.sort(key=lambda r: (r["object_id"], r["frame"]))
                    with open(beh_csv, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        w.writeheader()
                        w.writerows(behavior_rows)
                    result["behavior_csv"] = beh_csv
                    n_pred = len(behavior_rows)
                    n_nc = sum(1 for r in behavior_rows if r["behavior"] == "NC")
                    n_objs = len(set(r["object_id"] for r in behavior_rows))
                    print(f"[Classifier Inference] Saved {n_pred} predictions "
                          f"({n_nc} NC) for {n_objs} objects → {beh_csv}")

                self.video_finished.emit(result)

            self.all_done.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
        finally:
            sys.stdout = _real_stdout


class _ClipExtractWorker(QThread):
    """Extract masks for a short clip (N frames) at a specific position."""
    finished_ok = pyqtSignal(dict)  # {obj_id: [list of det dicts per frame]}
    error = pyqtSignal(str)

    def __init__(self, model_dir, video_path, start_frame, clip_length,
                 confidence, max_det):
        super().__init__()
        self.model_dir = model_dir
        self.video_path = video_path
        self.start_frame = start_frame
        self.clip_length = clip_length
        self.confidence = confidence
        self.max_det = max_det

    def run(self):
        try:
            from .silhouette_extractor import SilhouetteExtractor, _detect_architecture
            from .mask_tracker_inference import MaskInferenceConfig, MultiObjectTracker

            architecture = _detect_architecture(self.model_dir)
            if architecture == "yolov11-seg":
                from .yolo_inference import YOLOInference
                inference = YOLOInference(
                    self.model_dir, device="cpu",
                    inference_size=0, use_masks=True,
                )
            else:
                from .mask_tracker_inference import MaskRCNNInference
                inference = MaskRCNNInference(
                    self.model_dir, device="cpu",
                    inference_size=0, use_masks=True,
                )
            inference.load_model()

            config = MaskInferenceConfig(
                confidence_threshold=self.confidence,
                max_detections=self.max_det,
                use_masks=True,
            )
            tracker = MultiObjectTracker(config)

            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            # {obj_id: [det_or_None per clip frame]}
            object_frames: dict = {}

            for i in range(self.clip_length):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = inference.predict(
                    frame_rgb, self.confidence, self.max_det,
                )
                fi = self.start_frame + i
                matched = tracker.update(
                    detections, fi, fps, frame_hw=(frame_h, frame_w),
                )
                seen = set()
                for obj_id, det in matched.items():
                    if obj_id not in object_frames:
                        object_frames[obj_id] = [None] * i
                    object_frames[obj_id].append(det)
                    seen.add(obj_id)
                for obj_id in object_frames:
                    if obj_id not in seen:
                        object_frames[obj_id].append(None)

            cap.release()
            self.finished_ok.emit(object_frames)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class _BatchClipExtractWorker(QThread):
    """Extract multiple clips from videos, saving each to disk."""
    clip_ready = pyqtSignal(int, str)  # clip_idx, clip_dir
    all_done = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_dir, clip_positions, clip_length, confidence,
                 max_det, clips_base_dir, start_clip_num):
        super().__init__()
        self.model_dir = model_dir
        self.clip_positions = clip_positions  # [(video_path, start_frame), ...]
        self.clip_length = clip_length
        self.confidence = confidence
        self.max_det = max_det
        self.clips_base_dir = clips_base_dir
        self.start_clip_num = start_clip_num
        self._stop_flag = False

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        try:
            from .silhouette_extractor import _detect_architecture
            from .mask_tracker_inference import MaskInferenceConfig, MultiObjectTracker

            architecture = _detect_architecture(self.model_dir)
            if architecture == "yolov11-seg":
                from .yolo_inference import YOLOInference
                inference = YOLOInference(
                    self.model_dir, device="cpu",
                    inference_size=0, use_masks=True,
                )
            else:
                from .mask_tracker_inference import MaskRCNNInference
                inference = MaskRCNNInference(
                    self.model_dir, device="cpu",
                    inference_size=0, use_masks=True,
                )
            inference.load_model()

            config = MaskInferenceConfig(
                confidence_threshold=self.confidence,
                max_detections=self.max_det,
                use_masks=True,
            )

            current_cap = None
            current_path = None

            for clip_idx, (video_path, start_frame) in enumerate(self.clip_positions):
                if self._stop_flag:
                    break

                if video_path != current_path:
                    if current_cap is not None:
                        current_cap.release()
                    current_cap = cv2.VideoCapture(video_path)
                    current_path = video_path

                fps = current_cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_h = int(current_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_w = int(current_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                current_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                tracker = MultiObjectTracker(config)
                object_frames: dict = {}
                raw_frames = []

                for i in range(self.clip_length):
                    ret, frame = current_cap.read()
                    if not ret:
                        break
                    raw_frames.append(frame)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections = inference.predict(
                        frame_rgb, self.confidence, self.max_det,
                    )
                    fi = start_frame + i
                    matched = tracker.update(
                        detections, fi, fps, frame_hw=(frame_h, frame_w),
                    )
                    seen = set()
                    for obj_id, det in matched.items():
                        if obj_id not in object_frames:
                            object_frames[obj_id] = [None] * i
                        object_frames[obj_id].append(det)
                        seen.add(obj_id)
                    for obj_id in object_frames:
                        if obj_id not in seen:
                            object_frames[obj_id].append(None)

                # Save clip to disk
                clip_num = self.start_clip_num + clip_idx
                clip_id = f"clip_{clip_num:04d}"
                clip_dir = os.path.join(self.clips_base_dir, clip_id)
                os.makedirs(clip_dir, exist_ok=True)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    os.path.join(clip_dir, "frames.mp4"),
                    fourcc, fps, (frame_w, frame_h),
                )
                for f in raw_frames:
                    writer.write(f)
                writer.release()

                mask_arrays = {}
                for obj_id, fdata in object_frames.items():
                    per_frame = []
                    for det in fdata:
                        if det is not None and det.get("mask") is not None:
                            per_frame.append(det["mask"].astype(bool))
                        else:
                            per_frame.append(
                                np.zeros((frame_h, frame_w), dtype=bool)
                            )
                    mask_arrays[f"obj_{obj_id}"] = np.stack(per_frame, axis=0)
                np.savez_compressed(
                    os.path.join(clip_dir, "masks.npz"), **mask_arrays,
                )

                objects = {}
                for obj_id in object_frames:
                    objects[str(obj_id)] = {"behavior": None, "color": None}
                meta = {
                    "clip_id": clip_id,
                    "source_video": video_path,
                    "start_frame": start_frame,
                    "clip_length": len(raw_frames),
                    "fps": fps,
                    "frame_width": frame_w,
                    "frame_height": frame_h,
                    "num_objects": len(object_frames),
                    "status": "pending",
                    "objects": objects,
                }
                with open(os.path.join(clip_dir, "meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)

                self.clip_ready.emit(clip_idx, clip_dir)

            if current_cap is not None:
                current_cap.release()

            self.all_done.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class _ClassifierTrainWorker(QThread):
    """Train a behavior classifier on composite images."""
    epoch_done = pyqtSignal(int, float, float, float)  # epoch, train_loss, val_loss, val_acc
    log_message = pyqtSignal(str)
    finished = pyqtSignal(object)  # summary dict
    error = pyqtSignal(str)

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self._stop_flag = False

    def request_stop(self):
        self._stop_flag = True

    def run(self):
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
            self._train()
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

    def _train(self):
        import time as _time
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset, random_split
        from torchvision import transforms, models

        composite_paths = self.config["composite_paths"]
        labels = self.config["labels"]
        class_names = self.config["class_names"]
        epochs = self.config["epochs"]
        lr = self.config["lr"]
        batch_size = self.config["batch_size"]
        val_split = self.config["val_split"]
        augment = self.config["augment"]
        freeze_backbone = self.config.get("freeze_backbone", False)
        backbone = self.config["backbone"]
        output_dir = self.config["output_dir"]

        os.makedirs(output_dir, exist_ok=True)
        train_start = _time.time()

        self.log_message.emit(f"[Classifier] Output directory: {output_dir}")
        self.log_message.emit(
            f"[Classifier] Dataset: {len(composite_paths)} composites, "
            f"{len(class_names)} classes ({', '.join(class_names)})"
        )
        self.log_message.emit(
            f"[Classifier] Config: backbone={backbone}, epochs={epochs}, "
            f"lr={lr:.4f}, batch_size={batch_size}, val_split={val_split}, augment={augment}"
        )

        class CompositeDataset(Dataset):
            def __init__(self, paths, targets, transform=None):
                self.paths = paths
                self.targets = targets
                self.transform = transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                img = cv2.imread(self.paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (128, 128))
                img = img.astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)
                if self.transform:
                    img = self.transform(img)
                return img, self.targets[idx]

        aug_transform = None
        if augment:
            aug_rotation = self.config.get("aug_rotation", 15)
            aug_list = []
            if aug_rotation >= 180:
                aug_list.append(transforms.RandomHorizontalFlip())
                aug_list.append(transforms.RandomVerticalFlip())
                aug_list.append(transforms.RandomRotation(180))
            elif aug_rotation > 0:
                aug_list.append(transforms.RandomHorizontalFlip())
                aug_list.append(transforms.RandomRotation(aug_rotation))
            if aug_list:
                aug_transform = transforms.Compose(aug_list)

        n_total = len(composite_paths)
        n_val = max(1, int(n_total * val_split)) if val_split > 0 else 0

        if n_val > 0:
            import random as _rnd
            from collections import defaultdict as _ddict
            per_class: dict[int, list[int]] = _ddict(list)
            for i, lbl in enumerate(labels):
                per_class[lbl].append(i)

            train_idx, val_idx = [], []
            for cls_id, idxs in per_class.items():
                _rnd.shuffle(idxs)
                n_cls_val = max(1, int(len(idxs) * val_split))
                val_idx.extend(idxs[:n_cls_val])
                train_idx.extend(idxs[n_cls_val:])

            if not train_idx:
                train_idx = val_idx[:]

            train_paths = [composite_paths[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_paths = [composite_paths[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            train_ds = CompositeDataset(train_paths, train_labels, aug_transform)
            val_ds = CompositeDataset(val_paths, val_labels, None)
            n_train = len(train_idx)
            n_val = len(val_idx)
        else:
            train_ds = CompositeDataset(composite_paths, labels, aug_transform)
            val_ds = None
            n_train = n_total
            n_val = 0

        self.log_message.emit(
            f"[Classifier] Split: {n_train} train, {n_val} val (stratified per class)"
        )

        n_classes = len(class_names)

        from collections import Counter as _Counter
        train_counts = _Counter(train_ds.targets if hasattr(train_ds, "targets") else train_labels)
        for cls_idx, cls_name in enumerate(class_names):
            cnt = train_counts.get(cls_idx, 0)
            self.log_message.emit(f"[Classifier]   {cls_name}: {cnt} train")

        if n_train < 10 * n_classes:
            self.log_message.emit(
                f"[Classifier] ⚠ Small dataset: {n_train} train samples for {n_classes} classes. "
                f"Consider labeling more clips (aim for 10+ per class) for better results."
            )
        if n_val > 0 and n_val < n_classes:
            self.log_message.emit(
                f"[Classifier] ⚠ Val set ({n_val}) smaller than class count ({n_classes}). "
                f"Val metrics will be noisy."
            )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
        if backbone == "ResNet-34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif backbone == "MobileNetV3":
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        else:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, n_classes)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            if backbone == "MobileNetV3":
                for param in model.classifier[-1].parameters():
                    param.requires_grad = True
            else:
                for param in model.fc.parameters():
                    param.requires_grad = True

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        model = model.to(device)
        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log_message.emit(
            f"[Classifier] Model: {backbone} ({n_total:,} params, "
            f"{n_trainable:,} trainable) on {device.upper()}"
        )
        if freeze_backbone:
            self.log_message.emit(
                f"[Classifier] Backbone frozen — training only the classifier head"
            )
        self.log_message.emit(
            f"[Classifier] Optimizer: AdamW (lr={lr:.4f}, weight_decay=1e-4) + CosineAnnealingLR"
        )

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        early_stopping = self.config.get("early_stopping", True)
        patience = self.config.get("patience", 50) if early_stopping else epochs

        if early_stopping:
            self.log_message.emit(f"[Classifier] Starting training for {epochs} epochs (early stopping patience={patience})")
        else:
            self.log_message.emit(f"[Classifier] Starting training for {epochs} epochs (early stopping disabled)")
        best_epoch = 0
        all_train_losses = []
        all_val_losses = []
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            if self._stop_flag:
                self.log_message.emit("[Classifier] Training stopped by user.")
                break

            model.train()
            running_loss = 0.0
            n_batches = 0
            for imgs, targets in train_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
            scheduler.step()
            train_loss = running_loss / max(n_batches, 1)

            val_loss = 0.0
            val_acc = 0.0
            if val_loader is not None:
                model.eval()
                v_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for imgs, targets in val_loader:
                        imgs = imgs.to(device)
                        targets = targets.to(device)
                        outputs = model(imgs)
                        v_loss += criterion(outputs, targets).item()
                        preds = outputs.argmax(dim=1)
                        correct += (preds == targets).sum().item()
                        total += targets.size(0)
                val_loss = v_loss / max(len(val_loader), 1)
                val_acc = correct / max(total, 1)

                all_val_losses.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_val_acc = val_acc
                    patience_counter = 0
                    torch.save(model.state_dict(),
                               os.path.join(output_dir, "best_classifier.pth"))
                    self.log_message.emit(
                        f"[Classifier] Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
                        f"val_loss={val_loss:.4f}  val_acc={val_acc:.1%}  ★ best"
                    )
                else:
                    patience_counter += 1
                    self.log_message.emit(
                        f"[Classifier] Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
                        f"val_loss={val_loss:.4f}  val_acc={val_acc:.1%}  "
                        f"(patience {patience_counter}/{patience})"
                    )
            else:
                torch.save(model.state_dict(),
                           os.path.join(output_dir, "best_classifier.pth"))
                self.log_message.emit(
                    f"[Classifier] Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}"
                )

            all_train_losses.append(train_loss)
            self.epoch_done.emit(epoch, train_loss, val_loss, val_acc)

            if patience_counter >= patience:
                self.log_message.emit(
                    f"[Classifier] Early stopping at epoch {epoch} "
                    f"(val loss did not improve for {patience} epochs)"
                )
                break

        elapsed = _time.time() - train_start
        final_epoch = epoch if not self._stop_flag else epoch - 1

        torch.save(model.state_dict(),
                   os.path.join(output_dir, "last_classifier.pth"))
        config_out = {
            "backbone": backbone,
            "class_names": class_names,
            "n_classes": n_classes,
            "input_size": 128,
            "epochs_trained": final_epoch,
        }
        with open(os.path.join(output_dir, "classifier_config.json"), "w") as f:
            json.dump(config_out, f, indent=2)

        # Save per-epoch training log
        import csv as _csv
        log_path = os.path.join(output_dir, "training_log.csv")
        with open(log_path, "w", newline="") as lf:
            writer = _csv.writer(lf)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
            for ei in range(len(all_train_losses)):
                t_loss = all_train_losses[ei]
                v_loss = all_val_losses[ei] if ei < len(all_val_losses) else ""
                v_acc = ""
                writer.writerow([ei + 1, f"{t_loss:.6f}", v_loss if v_loss == "" else f"{v_loss:.6f}", v_acc])

        # Save training summary
        summary_out = {
            "backbone": backbone,
            "class_names": class_names,
            "n_classes": n_classes,
            "epochs_trained": final_epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "best_val_acc": best_val_acc,
            "final_train_loss": all_train_losses[-1] if all_train_losses else None,
            "final_val_loss": all_val_losses[-1] if all_val_losses else None,
            "training_time_sec": round(elapsed, 1),
            "device": str(device),
            "learning_rate": lr,
            "batch_size": batch_size,
            "n_train": n_train,
            "n_val": n_val,
            "early_stopping": early_stopping,
        }
        with open(os.path.join(output_dir, "training_summary.json"), "w") as sf:
            json.dump(summary_out, sf, indent=2)

        self.log_message.emit("─" * 60)
        self.log_message.emit(
            f"[Classifier] Training complete: {final_epoch} epochs in {elapsed:.1f}s "
            f"({elapsed / max(final_epoch, 1):.2f}s/epoch)"
        )
        if best_epoch > 0:
            self.log_message.emit(
                f"[Classifier] Best model: epoch {best_epoch}  "
                f"val_loss={best_val_loss:.4f}  val_acc={best_val_acc:.1%}"
            )

        if all_train_losses and all_val_losses:
            final_train = all_train_losses[-1]
            final_val = all_val_losses[-1]
            if final_val > 2 * final_train and final_train < 0.1:
                self.log_message.emit(
                    f"[Classifier] ⚠ Overfitting detected: train loss ({final_train:.4f}) "
                    f"is much lower than val loss ({final_val:.4f}). "
                    f"Try: more training data, enable augmentation, or use a smaller backbone."
                )
            if best_val_acc < 0.5 and n_val > 0:
                self.log_message.emit(
                    f"[Classifier] ⚠ Low val accuracy ({best_val_acc:.1%}). "
                    f"The model may not have enough data to learn meaningful patterns. "
                    f"Try labeling more clips (aim for 20+ per class)."
                )
        if len(all_train_losses) >= 3:
            first_third = all_train_losses[:len(all_train_losses) // 3]
            last_third = all_train_losses[-(len(all_train_losses) // 3):]
            if first_third and last_third:
                early_avg = sum(first_third) / len(first_third)
                late_avg = sum(last_third) / len(last_third)
                if early_avg > 0 and (early_avg - late_avg) / early_avg < 0.1:
                    self.log_message.emit(
                        f"[Classifier] ⚠ Train loss did not drop meaningfully "
                        f"(early avg: {early_avg:.4f}, late avg: {late_avg:.4f}). "
                        f"Consider: labeling more clips, increasing learning rate, "
                        f"or checking that composite images look correct."
                    )

        self.log_message.emit(f"[Classifier] Saved: best_classifier.pth, last_classifier.pth, classifier_config.json")
        self.log_message.emit(f"[Classifier] Output: {output_dir}")

        self.finished.emit({
            "output_dir": output_dir,
            "epochs_trained": final_epoch,
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "class_names": class_names,
        })


class _SilhouetteExtractionWorker(QThread):
    progress = pyqtSignal(int, int, str)  # frame_idx, total_frames, video_name
    video_done = pyqtSignal(str, int)  # video_name, num_tracks
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_dir, video_paths, output_dir, confidence, max_det):
        super().__init__()
        self.model_dir = model_dir
        self.video_paths = list(video_paths)
        self.output_dir = output_dir
        self.confidence = confidence
        self.max_det = max_det
        self._stop_flag = False

    def request_stop(self):
        self._stop_flag = True

    def run(self):
        try:
            from .silhouette_extractor import SilhouetteExtractor

            extractor = SilhouetteExtractor(self.model_dir)
            extractor.load_model()

            for vpath in self.video_paths:
                if self._stop_flag:
                    break
                vname = os.path.basename(vpath)
                result = extractor.extract_video(
                    vpath,
                    self.output_dir,
                    confidence_threshold=self.confidence,
                    max_detections=self.max_det,
                    progress=lambda f, t: self.progress.emit(f, t, vname),
                    should_stop=lambda: self._stop_flag,
                )
                self.video_done.emit(vname, result["num_tracks"])

            self.finished.emit()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class LiveLossPlot(QWidget):
    """Real-time training loss and validation mAP plot for segmentation.

    Matches the ClassifierLossPlot style: train loss, best loss, and
    mAP50 (mask) on a secondary axis — three clean lines.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._iterations = []
        self._losses = []
        self._best_losses = []
        self._map_iters = []
        self._map_vals = []
        self._total = 100

        try:
            import matplotlib
            matplotlib.use("Qt5Agg")
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(8, 4), dpi=100)
            self._fig.patch.set_facecolor("#1e1e1e")
            self._ax = self._fig.add_subplot(111)
            self._setup_axes()
            self._train_line, = self._ax.plot([], [], color="#2979ff", linewidth=1.8,
                                              label="Train loss")
            self._best_line, = self._ax.plot([], [], color="#ff9800", linewidth=1.8,
                                             linestyle="--", label="Best loss")
            self._ax2 = self._ax.twinx()
            self._ax2.yaxis.set_label_position("right")
            self._ax2.yaxis.tick_right()
            self._ax2.set_ylabel("mAP", color="#66bb6a")
            self._ax2.tick_params(axis="y", colors="#66bb6a")
            self._ax2.set_ylim(0, 1.05)
            self._ax2.spines["right"].set_color("#66bb6a")
            self._map_line, = self._ax2.plot([], [], color="#66bb6a", linewidth=1.4,
                                             marker="o", markersize=3,
                                             label="Val mAP50")
            all_handles = [self._train_line, self._best_line, self._map_line]
            all_labels = [h.get_label() for h in all_handles]
            self._ax.legend(all_handles, all_labels,
                            loc="upper right", fontsize=8,
                            facecolor="#2b2b2b", edgecolor="#555555",
                            labelcolor="#cccccc", framealpha=0.9)
            self._fig.subplots_adjust(right=0.88, bottom=0.15)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._has_mpl = True
        except ImportError:
            self._canvas = QLabel("matplotlib not installed — loss values shown below")
            self._canvas.setAlignment(Qt.AlignCenter)
            self._canvas.setStyleSheet("color: #cccccc; font-size: 13px;")
            self._has_mpl = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas, 1)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color: #cccccc; font-size: 11px;")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl_status)

    def _setup_axes(self):
        self._ax.set_facecolor("#1e1e1e")
        self._ax.set_xlabel("Epoch", color="#cccccc")
        self._ax.set_ylabel("Loss (log scale)", color="#cccccc")
        self._ax.set_title("Segmentation Training", color="#cccccc", fontsize=11)
        self._ax.set_yscale("log")
        self._ax.tick_params(colors="#999999")
        for spine in self._ax.spines.values():
            spine.set_color("#555555")
        self._ax.set_xlim(0, 10)

    def reset(self, total_iterations: int):
        self._iterations.clear()
        self._losses.clear()
        self._best_losses.clear()
        self._map_iters.clear()
        self._map_vals.clear()
        self._total = total_iterations
        self._lbl_status.setText("Waiting for first epoch...")
        if self._has_mpl:
            self._ax.cla()
            self._ax2.cla()
            self._ax2.yaxis.set_label_position("right")
            self._ax2.yaxis.tick_right()
            self._ax2.set_ylabel("mAP", color="#66bb6a")
            self._ax2.tick_params(axis="y", colors="#66bb6a")
            self._ax2.set_ylim(0, 1.05)
            self._ax2.spines["right"].set_color("#66bb6a")
            self._setup_axes()
            self._train_line, = self._ax.plot([], [], color="#2979ff", linewidth=1.8,
                                              label="Train loss")
            self._best_line, = self._ax.plot([], [], color="#ff9800", linewidth=1.8,
                                             linestyle="--", label="Best loss")
            self._map_line, = self._ax2.plot([], [], color="#66bb6a", linewidth=1.4,
                                             marker="o", markersize=3,
                                             label="Val mAP50")
            all_handles = [self._train_line, self._best_line, self._map_line]
            all_labels = [h.get_label() for h in all_handles]
            self._ax.legend(all_handles, all_labels,
                            loc="upper right", fontsize=8,
                            facecolor="#2b2b2b", edgecolor="#555555",
                            labelcolor="#cccccc", framealpha=0.9)
            self._canvas.draw_idle()

    def add_point(self, iteration: int, loss: float, metrics: dict = None):
        self._iterations.append(iteration)
        self._losses.append(loss)

        best = metrics.get("best_loss", loss) if metrics else loss
        self._best_losses.append(best)

        parts = [f"Epoch {iteration}/{self._total}    Loss: {loss:.4f}"]
        map50 = None
        if metrics:
            map50 = metrics.get("mAP50(M)") or metrics.get("mAP50(B)")
            if map50 is not None:
                parts.append(f"mAP50: {map50:.3f}")
        self._lbl_status.setText("    ".join(parts))

        if not self._has_mpl:
            return

        self._train_line.set_data(self._iterations, self._losses)
        self._best_line.set_data(self._iterations, self._best_losses)

        if map50 is not None:
            self._map_iters.append(iteration)
            self._map_vals.append(map50)
            self._map_line.set_data(self._map_iters, self._map_vals)

        self._ax.set_xlim(0, max(iteration * 1.1, 10))
        positive = [v for v in self._losses + self._best_losses if v > 0]
        if positive:
            y_min = min(positive) * 0.5
            y_max = max(positive) * 2.0
            self._ax.set_ylim(y_min, y_max)
        self._canvas.draw_idle()


class ClassifierLossPlot(QWidget):
    """Real-time training/validation loss and accuracy plot for the behavior classifier."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._epochs = []
        self._train_losses = []
        self._val_losses = []
        self._val_accs = []
        self._total = 50

        try:
            import matplotlib
            matplotlib.use("Qt5Agg")
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self._fig = Figure(figsize=(8, 4), dpi=100)
            self._fig.patch.set_facecolor("#1e1e1e")
            self._ax = self._fig.add_subplot(111)
            self._setup_axes()
            self._train_line, = self._ax.plot([], [], color="#2979ff", linewidth=1.8,
                                              label="Train loss")
            self._val_line, = self._ax.plot([], [], color="#ff9800", linewidth=1.8,
                                            linestyle="--", label="Val loss")
            self._ax2 = self._ax.twinx()
            self._ax2.yaxis.set_label_position("right")
            self._ax2.yaxis.tick_right()
            self._ax2.set_ylabel("Accuracy", color="#66bb6a")
            self._ax2.tick_params(axis="y", colors="#66bb6a")
            self._ax2.set_ylim(0, 1.05)
            self._ax2.spines["right"].set_color("#66bb6a")
            self._acc_line, = self._ax2.plot([], [], color="#66bb6a", linewidth=1.4,
                                             marker="o", markersize=3,
                                             label="Val accuracy")
            all_handles = [self._train_line, self._val_line, self._acc_line]
            all_labels = [h.get_label() for h in all_handles]
            self._ax.legend(all_handles, all_labels,
                            loc="upper right", fontsize=8,
                            facecolor="#2b2b2b", edgecolor="#555555",
                            labelcolor="#cccccc", framealpha=0.9)
            self._fig.subplots_adjust(right=0.88, bottom=0.15)
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._has_mpl = True
        except ImportError:
            self._canvas = QLabel("matplotlib not installed — loss values shown below")
            self._canvas.setAlignment(Qt.AlignCenter)
            self._canvas.setStyleSheet("color: #cccccc; font-size: 13px;")
            self._has_mpl = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas, 1)
        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("color: #cccccc; font-size: 11px;")
        self._lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl_status)

    def _setup_axes(self):
        self._ax.set_facecolor("#1e1e1e")
        self._ax.set_xlabel("Epoch", color="#cccccc")
        self._ax.set_ylabel("Loss", color="#cccccc")
        self._ax.set_title("Classifier Training", color="#cccccc", fontsize=11)
        self._ax.tick_params(colors="#999999")
        for spine in self._ax.spines.values():
            spine.set_color("#555555")
        self._ax.set_xlim(0, 10)

    def reset(self, total_epochs: int):
        self._epochs.clear()
        self._train_losses.clear()
        self._val_losses.clear()
        self._val_accs.clear()
        self._total = total_epochs
        self._lbl_status.setText("Waiting for first epoch...")
        if self._has_mpl:
            self._ax.cla()
            self._ax2.cla()
            self._ax2.yaxis.set_label_position("right")
            self._ax2.yaxis.tick_right()
            self._ax2.set_ylabel("Accuracy", color="#66bb6a")
            self._ax2.tick_params(axis="y", colors="#66bb6a")
            self._ax2.set_ylim(0, 1.05)
            self._ax2.spines["right"].set_color("#66bb6a")
            self._setup_axes()
            self._train_line, = self._ax.plot([], [], color="#2979ff", linewidth=1.8,
                                              label="Train loss")
            self._val_line, = self._ax.plot([], [], color="#ff9800", linewidth=1.8,
                                            linestyle="--", label="Val loss")
            self._acc_line, = self._ax2.plot([], [], color="#66bb6a", linewidth=1.4,
                                             marker="o", markersize=3,
                                             label="Val accuracy")
            all_handles = [self._train_line, self._val_line, self._acc_line]
            all_labels = [h.get_label() for h in all_handles]
            self._ax.legend(all_handles, all_labels,
                            loc="upper right", fontsize=8,
                            facecolor="#2b2b2b", edgecolor="#555555",
                            labelcolor="#cccccc", framealpha=0.9)
            self._canvas.draw_idle()

    def add_point(self, epoch: int, train_loss: float,
                  val_loss: float = None, val_acc: float = None):
        self._epochs.append(epoch)
        self._train_losses.append(train_loss)

        parts = [f"Epoch {epoch}/{self._total}    Train loss: {train_loss:.4f}"]
        if val_loss is not None:
            self._val_losses.append(val_loss)
            parts.append(f"Val loss: {val_loss:.4f}")
        if val_acc is not None:
            self._val_accs.append(val_acc)
            parts.append(f"Val acc: {val_acc:.1%}")
        self._lbl_status.setText("    ".join(parts))

        if not self._has_mpl:
            return

        self._train_line.set_data(self._epochs, self._train_losses)
        if self._val_losses:
            val_epochs = self._epochs[-len(self._val_losses):]
            self._val_line.set_data(val_epochs, self._val_losses)
        if self._val_accs:
            acc_epochs = self._epochs[-len(self._val_accs):]
            self._acc_line.set_data(acc_epochs, self._val_accs)

        self._ax.set_xlim(0, max(epoch * 1.1, 10))
        all_losses = self._train_losses + self._val_losses
        y_max = max(all_losses) * 1.1 + 0.01 if all_losses else 1.0
        self._ax.set_ylim(0, y_max)
        self._canvas.draw_idle()


class TrainingSamplePreview(QWidget):
    """Shows sample validation predictions during/after training.

    Displays a grid of annotated images so the user can visually assess
    model quality without leaving the Training tab.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._images: List[QPixmap] = []
        self._labels: List[str] = []
        self.setMinimumHeight(100)

    def clear(self):
        self._images.clear()
        self._labels.clear()
        self.update()

    def set_images(self, images_rgb: list, labels: list = None):
        """Set a list of (numpy RGB image) to display in a grid."""
        self._images.clear()
        self._labels = labels or []
        for img_rgb in images_rgb:
            h, w = img_rgb.shape[:2]
            bpl = 3 * w
            qimg = QImage(img_rgb.data, w, h, bpl, QImage.Format_RGB888)
            self._images.append(QPixmap.fromImage(qimg.copy()))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if not self._images:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(
                self.rect(), Qt.AlignCenter,
                "Sample predictions will appear here during training"
            )
            painter.end()
            return

        n = len(self._images)
        cols = min(n, max(1, self.width() // 200))
        rows = (n + cols - 1) // cols
        cell_w = self.width() // cols
        cell_h = self.height() // rows
        margin = 4

        for i, pm in enumerate(self._images):
            r, c = divmod(i, cols)
            x = c * cell_w + margin
            y = r * cell_h + margin
            avail_w = cell_w - 2 * margin
            avail_h = cell_h - 2 * margin - 16  # room for label
            scaled = pm.scaled(
                avail_w, avail_h, Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            dx = x + (avail_w - scaled.width()) // 2
            dy = y + (avail_h - scaled.height()) // 2
            painter.drawPixmap(dx, dy, scaled)

            if i < len(self._labels):
                painter.setPen(QColor(180, 180, 180))
                font = painter.font()
                font.setPixelSize(10)
                painter.setFont(font)
                painter.drawText(
                    x, y + avail_h + 2, avail_w, 14,
                    Qt.AlignCenter, self._labels[i],
                )

        painter.end()


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
            skipped = 0
            total = sum(len(v) for v in self.frame_indices_per_video.values())
            done = 0
            for vpath, frame_idxs in self.frame_indices_per_video.items():
                cap = cv2.VideoCapture(vpath)
                if not cap.isOpened():
                    continue
                stem = Path(vpath).stem
                for fidx in sorted(frame_idxs):
                    fname = f"{stem}_frame_{fidx:06d}.png"
                    out_path = os.path.join(self.output_dir, fname)
                    if os.path.exists(out_path):
                        # Don't overwrite existing frames
                        saved.append((vpath, fidx, out_path))
                        skipped += 1
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                        ret, frame = cap.read()
                        if ret:
                            cv2.imwrite(out_path, frame)
                            saved.append((vpath, fidx, out_path))
                    done += 1
                    self.progress.emit(done, total)
                cap.release()
            if skipped:
                print(f"[MTT] Skipped {skipped} existing frames")
            self.finished.emit(saved)
        except Exception as e:
            self.error.emit(str(e))


# ======================================================================
# Main window
# ======================================================================
class MaskTrackerWindow(QMainWindow):
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
        self._video_frame_idx: int = 0
        self._annot_in_video_mode: bool = False

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
        self._cls_model_dirs: List[str] = []

        from .mask_tracker_annotator import COCOAnnotationManager
        self._coco = COCOAnnotationManager()
        self._categories: List[str] = []
        self._last_used_category: str = ""

        self._build_menu_bar()
        self._build_ui()
        self._update_nav_state()

        # Tab activity animation — pixel-art running mouse on right side
        self._mouse_pixmaps = self._build_mouse_sprites()
        self._tab_anim_tick = 0
        self._mouse_active = {}  # track which tabs currently show the mouse
        self._tab_anim_timer = QTimer(self)
        self._tab_anim_timer.timeout.connect(self._update_tab_animations)
        self._tab_anim_timer.start(150)

    @staticmethod
    def _build_mouse_sprites() -> list:
        """Generate 4 frames of a running mouse as 16x16 QPixmaps."""
        frames = []
        sz = 16
        leg_poses = [
            (2, 2, -2, 1),
            (1, 3, -1, 3),
            (-1, 1, 2, 2),
            (0, 3, 0, 3),
        ]
        body_color = QColor(200, 200, 200)
        ear_color = QColor(255, 180, 180)
        leg_color = QColor(180, 180, 180)
        tail_color = QColor(170, 170, 170)
        eye_color = QColor(40, 40, 40)

        for fdx, fdy, bdx, bdy in leg_poses:
            pix = QPixmap(sz, sz)
            pix.fill(QColor(0, 0, 0, 0))
            p = QPainter(pix)
            p.setRenderHint(QPainter.Antialiasing)

            p.setPen(QPen(tail_color, 1.2))
            tail = QPainterPath()
            tail.moveTo(2, 8)
            tail.cubicTo(0, 5, 1, 2, 3, 3)
            p.drawPath(tail)

            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(body_color))
            p.drawEllipse(3, 5, 9, 6)
            p.drawEllipse(10, 4, 5, 5)

            p.setBrush(QBrush(ear_color))
            p.drawEllipse(11, 2, 3, 3)
            p.drawEllipse(13, 3, 3, 3)

            p.setBrush(QBrush(eye_color))
            p.drawEllipse(13, 5, 2, 2)

            p.setPen(QPen(leg_color, 1.3))
            p.drawLine(10, 10, int(10 + fdx), int(10 + fdy))
            p.drawLine(5, 10, int(5 + bdx), int(10 + bdy))

            p.end()
            frames.append(pix)
        return frames

    def _update_tab_animations(self):
        self._tab_anim_tick = (self._tab_anim_tick + 1) % len(self._mouse_pixmaps)
        pixmap = self._mouse_pixmaps[self._tab_anim_tick]

        active_map = {
            0: (  # Segmentation (mask model training)
                (hasattr(self, "_train_worker")
                 and self._train_worker is not None
                 and self._train_worker.isRunning())
                or (hasattr(self, "_post_infer_worker")
                    and self._post_infer_worker is not None
                    and self._post_infer_worker.isRunning())
            ),
            1: (  # Classification
                (hasattr(self, "_clip_extract_worker")
                 and self._clip_extract_worker is not None
                 and self._clip_extract_worker.isRunning())
                or (hasattr(self, "_batch_worker")
                    and self._batch_worker is not None
                    and self._batch_worker.isRunning())
                or (hasattr(self, "_cls_train_worker")
                    and self._cls_train_worker is not None
                    and self._cls_train_worker.isRunning())
            ),
            2: (  # Inference
                hasattr(self, "_inference_worker")
                and self._inference_worker is not None
                and self._inference_worker.isRunning()
            ),
        }

        tab_bar = self.tab_widget.tabBar()
        for tab_idx, active in active_map.items():
            was_active = self._mouse_active.get(tab_idx, False)
            if active:
                if not was_active:
                    lbl = QLabel()
                    lbl.setFixedSize(18, 16)
                    lbl.setPixmap(pixmap)
                    tab_bar.setTabButton(tab_idx, QTabBar.RightSide, lbl)
                    self._mouse_active[tab_idx] = lbl
                else:
                    self._mouse_active[tab_idx].setPixmap(pixmap)
            else:
                if was_active:
                    tab_bar.setTabButton(tab_idx, QTabBar.RightSide, None)
                    self._mouse_active[tab_idx] = False

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
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(5)
        splitter.setStyleSheet(
            "QSplitter::handle { background-color: #444444; }"
        )

        # Left panel: tabs
        self.tab_widget = QTabWidget()
        fm = self.fontMetrics()
        min_w = max(340, fm.averageCharWidth() * 52 + 40)
        self.tab_widget.setMinimumWidth(min_w)

        self._build_annotator_tab()
        self._build_classifier_tab()
        self._build_tracking_tab()
        self._inference_tab_idx = 2
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self._preview_title_row = QWidget()
        _pt_layout = QHBoxLayout(self._preview_title_row)
        _pt_layout.setContentsMargins(4, 2, 4, 2)
        _pt_layout.setSpacing(6)
        self._lbl_preview_title = QLabel("")
        self._lbl_preview_title.setStyleSheet(
            "color: #aaaaaa; font-size: 11px; font-weight: bold;"
        )
        _pt_layout.addWidget(self._lbl_preview_title)
        _pt_layout.addStretch()
        self._btn_toggle_training = QPushButton("Show Training Graph")
        self._btn_toggle_training.setStyleSheet(
            "QPushButton { background-color: #2979ff; color: #ffffff; border: none; "
            "border-radius: 3px; padding: 2px 8px; font-size: 10px; }"
            "QPushButton:hover { background-color: #448aff; }"
        )
        self._btn_toggle_training.setVisible(False)
        self._btn_toggle_training.clicked.connect(self._toggle_cls_training_viz)
        _pt_layout.addWidget(self._btn_toggle_training)
        self._preview_title_row.setVisible(False)
        right_layout.addWidget(self._preview_title_row)

        self.preview = AnnotationPreviewWidget()
        self.preview.annotation_accepted.connect(self._on_annotation_accepted)
        self.preview.ai_prediction_requested.connect(self._request_ai_prediction)
        self.preview.zoom_changed.connect(
            lambda z: self.lbl_zoom_info.setText(f"{int(z * 100)}%")
        )
        self.preview.mode_changed.connect(self._on_mode_changed)
        self.preview.advance_frame_requested.connect(self._on_advance_frame)
        self.preview.annotation_edited.connect(self._on_annotation_edited)
        self.preview.delete_annotation_requested.connect(self._on_delete_annotation_by_index)
        self.preview.approve_annotation_requested.connect(self._on_approve_annotation_by_index)
        right_layout.addWidget(self.preview, 1)

        # Info row below preview
        self._info_row = QWidget()
        info_row = self._info_row
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

        self.btn_sam_toggle = QPushButton("SAM Labeling")
        self.btn_sam_toggle.setCheckable(True)
        self.btn_sam_toggle.setChecked(False)
        self.btn_sam_toggle.setToolTip(
            "Toggle SAM2 AI-assisted labeling mode.\n\n"
            "When active, double-click on an object to place a\n"
            "positive point. SAM2 will predict a segmentation mask.\n"
            "Right-click to add negative (exclude) points.\n"
            "Enter to accept the mask, or toggle off to cancel.\n\n"
            "Single-click drag still pans the view while in SAM mode."
        )
        self._sam_toggle_off_style = (
            "QPushButton { background-color: #424242; color: #cccccc; font-weight: bold; "
            "padding: 3px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #616161; }"
        )
        self._sam_toggle_on_style = (
            "QPushButton { background-color: #2e7d32; color: white; font-weight: bold; "
            "padding: 3px 12px; border-radius: 3px; border: 2px solid #66bb6a; }"
            "QPushButton:hover { background-color: #388e3c; }"
        )
        self.btn_sam_toggle.setStyleSheet(self._sam_toggle_off_style)
        self.btn_sam_toggle.toggled.connect(self._on_sam_toggle)
        info_layout.addWidget(self.btn_sam_toggle)

        self.btn_edit_classes = QPushButton("Edit Object Classes")
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

        # Store annotation-specific widgets for show/hide on tab switch
        self._annotation_bar_widgets = [
            self.lbl_mode, self.lbl_frame_info, self.lbl_ann_stats,
            self.btn_sam_toggle, self.btn_edit_classes, self.lbl_zoom_info,
        ]
        # Classifier info label (hidden by default, shown on Classifier tab)
        self.lbl_classifier_info = QLabel("")
        self.lbl_classifier_info.setStyleSheet("color: #999999; font-size: 11px;")
        self.lbl_classifier_info.setVisible(False)
        info_layout.addWidget(self.lbl_classifier_info)

        # Edit Behavior Classes button (hidden by default, shown on Classifier tab)
        self.btn_edit_behaviors = QPushButton("Edit Behavior Classes")
        self.btn_edit_behaviors.setStyleSheet(
            "QPushButton { background-color: #7b1fa2; color: white; font-weight: bold; "
            "padding: 3px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #9c27b0; }"
        )
        self.btn_edit_behaviors.setToolTip(
            "Add, edit, or remove behavior categories for classification."
        )
        self.btn_edit_behaviors.clicked.connect(self._show_behavior_categories_popup)
        self.btn_edit_behaviors.setVisible(False)
        info_layout.addWidget(self.btn_edit_behaviors)

        # Tracking info label (hidden by default, shown on Tracking tab)
        self.lbl_tracking_info = QLabel("")
        self.lbl_tracking_info.setStyleSheet("color: #999999; font-size: 11px;")
        self.lbl_tracking_info.setVisible(False)
        info_layout.addWidget(self.lbl_tracking_info)

        # Classifier frame slider (hidden by default, shown on Classifier tab)
        self._cls_nav_bar = QWidget()
        cls_nav_layout = QHBoxLayout(self._cls_nav_bar)
        cls_nav_layout.setContentsMargins(5, 2, 5, 2)
        cls_nav_layout.setSpacing(5)
        self._cls_frame_slider = QSlider(Qt.Horizontal)
        self._cls_frame_slider.setMinimum(0)
        self._cls_frame_slider.setMaximum(0)
        self._cls_frame_slider.valueChanged.connect(self._on_cls_slider_changed)
        self._cls_frame_slider.sliderPressed.connect(self._on_cls_slider_pressed)
        self._cls_frame_slider.setStyleSheet(
            "QSlider::groove:horizontal { background: #3c3c3c; height: 6px; "
            "border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #2979ff; width: 12px; "
            "margin: -4px 0; border-radius: 6px; }"
            "QSlider::sub-page:horizontal { background: #2979ff; border-radius: 3px; }"
        )
        self.lbl_cls_frame_num = QLabel("0 / 0")
        self.lbl_cls_frame_num.setStyleSheet("color: #cccccc; font-size: 10px;")
        self.lbl_cls_frame_num.setMinimumWidth(80)
        cls_nav_layout.addWidget(self._cls_frame_slider, 1)
        cls_nav_layout.addWidget(self.lbl_cls_frame_num)

        playback_btn_style = (
            "QPushButton { background-color: #3c3c3c; color: #cccccc; "
            "border: 1px solid #555; border-radius: 3px; padding: 2px 8px; "
            "font-size: 12px; min-width: 24px; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
            "QPushButton:checked { background-color: #2979ff; color: white; }"
        )
        self._btn_cls_play_pause = QPushButton("▶")
        self._btn_cls_play_pause.setToolTip("Play / Pause (Space)")
        self._btn_cls_play_pause.setStyleSheet(playback_btn_style)
        self._btn_cls_play_pause.clicked.connect(self._cls_toggle_playback)
        cls_nav_layout.addWidget(self._btn_cls_play_pause)

        self._combo_cls_speed = QComboBox()
        self._combo_cls_speed.setToolTip("Playback speed")
        self._combo_cls_speed.addItems(
            ["0.25x", "0.5x", "1x", "1.25x", "1.5x", "2x", "3x", "4x", "8x"]
        )
        self._combo_cls_speed.setCurrentIndex(2)  # default 1x
        self._combo_cls_speed.setStyleSheet(
            "QComboBox { background-color: #3c3c3c; color: #cccccc; "
            "border: 1px solid #555; border-radius: 3px; padding: 2px 6px; "
            "font-size: 11px; min-width: 44px; }"
            "QComboBox QAbstractItemView { background-color: #3c3c3c; "
            "color: #cccccc; selection-background-color: #2979ff; }"
        )
        self._combo_cls_speed.currentIndexChanged.connect(self._on_cls_speed_changed)
        self._cls_speed_multiplier = 1.0
        cls_nav_layout.addWidget(self._combo_cls_speed)

        self._cls_nav_bar.setVisible(False)
        right_layout.addWidget(self._cls_nav_bar)

        # Training visualization panel (hidden by default, shown on Training tab)
        self._training_viz_panel = QWidget()
        training_viz_layout = QVBoxLayout(self._training_viz_panel)
        training_viz_layout.setContentsMargins(4, 4, 4, 4)
        training_viz_layout.setSpacing(4)

        self._loss_plot = LiveLossPlot(parent=self._training_viz_panel)
        training_viz_layout.addWidget(self._loss_plot, 3)

        self._seg_log_text = QTextEdit()
        self._seg_log_text.setReadOnly(True)
        self._seg_log_text.setStyleSheet(
            "QTextEdit { background-color: #1a1a1a; color: #b0b0b0; "
            "font-family: monospace; font-size: 10px; border: 1px solid #333; }"
        )
        self._seg_log_text.setLineWrapMode(QTextEdit.NoWrap)
        training_viz_layout.addWidget(self._seg_log_text, 1)

        btn_copy_seg_log = QPushButton("Copy Training Logs to Clipboard")
        btn_copy_seg_log.setStyleSheet(
            "QPushButton { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 4px 10px; color: #cccccc; font-size: 10px; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
        )
        btn_copy_seg_log.clicked.connect(self._copy_seg_train_log)
        training_viz_layout.addWidget(btn_copy_seg_log)

        self._training_viz_panel.setVisible(False)
        right_layout.addWidget(self._training_viz_panel, 1)

        # Classifier training visualization panel (hidden by default)
        self._cls_training_viz_panel = QWidget()
        cls_train_viz_layout = QVBoxLayout(self._cls_training_viz_panel)
        cls_train_viz_layout.setContentsMargins(4, 4, 4, 4)
        cls_train_viz_layout.setSpacing(4)
        self._cls_loss_plot = ClassifierLossPlot(parent=self._cls_training_viz_panel)
        cls_train_viz_layout.addWidget(self._cls_loss_plot, 3)

        # Training log text box
        self._cls_log_text = QTextEdit()
        self._cls_log_text.setReadOnly(True)
        self._cls_log_text.setStyleSheet(
            "QTextEdit { background-color: #1a1a1a; color: #b0b0b0; "
            "font-family: monospace; font-size: 10px; border: 1px solid #333; }"
        )
        self._cls_log_text.setLineWrapMode(QTextEdit.NoWrap)
        cls_train_viz_layout.addWidget(self._cls_log_text, 1)

        btn_copy_log = QPushButton("Copy Training Logs to Clipboard")
        btn_copy_log.setStyleSheet(
            "QPushButton { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 4px 10px; color: #cccccc; font-size: 10px; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
        )
        btn_copy_log.clicked.connect(self._copy_cls_train_log)
        cls_train_viz_layout.addWidget(btn_copy_log)

        self._cls_training_viz_panel.setVisible(False)
        right_layout.addWidget(self._cls_training_viz_panel, 1)

        # Inference log panel (hidden by default, shown during inference)
        self._infer_log_panel = QWidget()
        infer_log_layout = QVBoxLayout(self._infer_log_panel)
        infer_log_layout.setContentsMargins(4, 4, 4, 4)
        infer_log_layout.setSpacing(4)

        self._infer_log_text = QTextEdit()
        self._infer_log_text.setReadOnly(True)
        self._infer_log_text.setStyleSheet(
            "QTextEdit { background-color: #1a1a1a; color: #b0b0b0; "
            "font-family: monospace; font-size: 10px; border: 1px solid #333; }"
        )
        self._infer_log_text.setLineWrapMode(QTextEdit.NoWrap)
        infer_log_layout.addWidget(self._infer_log_text, 1)

        btn_copy_infer_log = QPushButton("Copy Inference Logs to Clipboard")
        btn_copy_infer_log.setStyleSheet(
            "QPushButton { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 4px 10px; color: #cccccc; font-size: 10px; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
        )
        btn_copy_infer_log.clicked.connect(self._copy_infer_log)
        infer_log_layout.addWidget(btn_copy_infer_log)

        self._infer_log_panel.setVisible(False)
        right_layout.addWidget(self._infer_log_panel, 0)

        splitter.addWidget(self.tab_widget)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        max_left = max(600, fm.averageCharWidth() * 95 + 40)
        splitter.setSizes([min_w, 800])
        right_panel.setMinimumWidth(300)
        self.tab_widget.setMaximumWidth(max_left)
        main_layout.addWidget(splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("File > New Project to begin, or load videos directly")

    def _build_annotator_tab(self):
        # Blue arrow buttons for spin boxes — generate tiny arrow PNGs
        # at runtime for cross-platform consistency.  Defined early so all
        # sections (extraction, training, classifier, inference) can use it.
        if not hasattr(self, "_spin_style"):
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
            "Right-click preview to add Manual Mask or Edit/Delete masks.\n"
            "Manual: click to place vertices, Enter to accept.\n"
            "SAM: toggle SAM Labeling button, double-click to place points,\n"
            "  right-click to exclude, Enter to accept.\n"
            "Left-click + drag to pan. Scroll to zoom. Space/Enter = next frame.\n"
            "E = extract current video frame to Training Frames Queue.\n"
            "Arrow keys scrub video (Shift ±10, Ctrl ±100)."
        )
        info.setStyleSheet("color: #888888; font-size: 9px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # --- Training section (appended to annotator tab) ---
        self._build_training_section(layout)

        layout.addStretch()
        scroll.setWidget(widget)
        self.tab_widget.addTab(scroll, "Mask")

    def _build_training_section(self, layout):
        """Build the model training UI section (appended to the Annotate tab)."""
        spin_style = self._spin_style

        # Training section
        train_group = QGroupBox("Train Model")
        train_vbox = QVBoxLayout()
        train_vbox.setSpacing(4)

        # Training data summary (inline at top)
        self.lbl_train_annotations = QLabel("Total annotations: —")
        self.lbl_train_classes = QLabel("Classes: —")
        for lbl in (self.lbl_train_annotations, self.lbl_train_classes):
            lbl.setStyleSheet("color: #cccccc; font-size: 11px;")
            train_vbox.addWidget(lbl)

        # Hardware info (inline)
        self.lbl_hw_info = QLabel("Detecting hardware...")
        self.lbl_hw_info.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_hw_info.setWordWrap(True)
        train_vbox.addWidget(self.lbl_hw_info)
        self._populate_hw_info()

        # Separator before controls
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3f3f3f;")
        train_vbox.addWidget(sep)

        tip_arch = (
            "Detection and segmentation architecture.\n\n"
            "YOLOv11-nano-seg: smallest and fastest (2.8M params,\n"
            "  ~6 MB). Coarser masks but very fast. Best when speed\n"
            "  matters more than mask precision.\n\n"
            "YOLOv11-small-seg (recommended): good balance of speed\n"
            "  and accuracy (9.4M params, ~12 MB). High-quality\n"
            "  instance masks with tight contour boundaries.\n\n"
            "YOLOv11-medium-seg: more capacity (22M params, ~44 MB).\n"
            "  Better accuracy, moderate training time. Good for\n"
            "  larger datasets or complex scenes.\n\n"
            "YOLOv11-large-seg: high accuracy (27M params, ~54 MB).\n"
            "  Best results with large datasets. Slower training.\n\n"
            "Mask R-CNN (MobileNetV3): two-stage detector. Accurate\n"
            "  with small datasets but slower (~3-10 fps on CPU).\n"
            "  Use as fallback if YOLO doesn't train well."
        )
        row = QHBoxLayout()
        lbl = QLabel("Architecture:")
        lbl.setToolTip(tip_arch)
        row.addWidget(lbl)
        self.combo_architecture = QComboBox()
        self.combo_architecture.addItems([
            "YOLOv11-nano-seg",
            "YOLOv11-small-seg",
            "YOLOv11-medium-seg",
            "YOLOv11-large-seg",
            "Mask R-CNN",
        ])
        self.combo_architecture.setToolTip(tip_arch)
        self.combo_architecture.currentIndexChanged.connect(self._on_architecture_changed)
        row.addWidget(self.combo_architecture)
        train_vbox.addLayout(row)

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
        self._maskrcnn_widgets = []
        row = QHBoxLayout()
        self._lbl_backbone = QLabel("Backbone:")
        self._lbl_backbone.setToolTip(tip_backbone)
        row.addWidget(self._lbl_backbone)
        self.combo_backbone = QComboBox()
        self.combo_backbone.addItems([
            "MobileNetV3-Large FPN",
            "MobileNetV3-Small FPN",
            "ResNet-50 FPN",
        ])
        self.combo_backbone.setToolTip(tip_backbone)
        row.addWidget(self.combo_backbone)
        self._maskrcnn_widgets.extend([self._lbl_backbone, self.combo_backbone])
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
            "Maximum training epochs (YOLO) or iterations (Mask R-CNN).\n\n"
            "YOLO: each epoch processes all images once. 100-1000 epochs\n"
            "  is typical. At ~1-2s/epoch, 1000 epochs takes ~20-30 min.\n"
            "Mask R-CNN: each iteration processes one batch. 200-2000\n"
            "  iterations is typical.\n\n"
            "Early stopping will stop training automatically when loss\n"
            "plateaus (see patience setting below), so it is safe to\n"
            "set this higher than needed. The best model weights are\n"
            "always saved regardless of when training stops."
        )
        row = QHBoxLayout()
        lbl = QLabel("Training Iterations:")
        lbl.setToolTip(tip_iterations)
        row.addWidget(lbl)
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(10, 100000)
        self.spin_iterations.setValue(1000)
        self.spin_iterations.setSingleStep(100)
        self.spin_iterations.setToolTip(tip_iterations)
        self.spin_iterations.setStyleSheet(spin_style)
        row.addWidget(self.spin_iterations)
        train_vbox.addLayout(row)

        tip_patience = (
            "Early stopping patience (number of epochs/iterations).\n\n"
            "Training stops automatically if the loss does not improve\n"
            "for this many consecutive epochs. The best model weights\n"
            "(lowest loss) are always saved regardless.\n\n"
            "Lower patience (10-30): stops sooner, faster iteration.\n"
            "Higher patience (50-200): allows the model more time to\n"
            "  recover from temporary loss plateaus.\n\n"
            "Default: 50. Set higher for larger datasets or if you\n"
            "see training stopped too early."
        )
        row = QHBoxLayout()
        lbl = QLabel("Early Stop Patience:")
        lbl.setToolTip(tip_patience)
        row.addWidget(lbl)
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(5, 1000)
        self.spin_patience.setValue(50)
        self.spin_patience.setSingleStep(10)
        self.spin_patience.setToolTip(tip_patience)
        self.spin_patience.setStyleSheet(spin_style)
        row.addWidget(self.spin_patience)
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
        self._lbl_min_size = QLabel("Image min size:")
        self._lbl_min_size.setToolTip(tip_min_size)
        row.addWidget(self._lbl_min_size)
        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(256, 2048)
        self.spin_min_size.setValue(480)
        self.spin_min_size.setSingleStep(64)
        self.spin_min_size.setToolTip(tip_min_size)
        self.spin_min_size.setStyleSheet(spin_style)
        row.addWidget(self.spin_min_size)
        self._maskrcnn_widgets.extend([self._lbl_min_size, self.spin_min_size])
        train_vbox.addLayout(row)

        self.chk_augment = QCheckBox("Data augmentation")
        self.chk_augment.setChecked(False)
        self.chk_augment.setToolTip(
            "Apply random transforms to training images each epoch.\n"
            "Expands the effective dataset, reducing overfitting\n"
            "when you have few annotated images."
        )
        train_vbox.addWidget(self.chk_augment)

        self._aug_options_frame = QFrame()
        aug_vbox = QVBoxLayout(self._aug_options_frame)
        aug_vbox.setContentsMargins(16, 4, 4, 4)
        aug_vbox.setSpacing(4)

        aug_lbl_style = "color: #cccccc; font-size: 11px;"
        aug_spin_style = spin_style

        def _aug_row(label_text, tooltip=""):
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl = QLabel(label_text)
            lbl.setStyleSheet(aug_lbl_style)
            if tooltip:
                lbl.setToolTip(tooltip)
            row.addWidget(lbl)
            return row, lbl

        # --- Rotation ---
        rot_row, _ = _aug_row("Rotation:",
            "Random continuous rotation applied each batch.\n"
            "±180° for top-down views (animal faces any direction).\n"
            "±15° for side views (animal is roughly horizontal).\n"
            "Select 'None' to disable rotation augmentation.")
        self.combo_aug_rotation = QComboBox()
        self.combo_aug_rotation.addItems(["None", "±15°", "±180°"])
        self.combo_aug_rotation.setCurrentIndex(0)
        self.combo_aug_rotation.setMinimumWidth(80)
        self.combo_aug_rotation.setStyleSheet(spin_style)
        self.combo_aug_rotation.setToolTip(
            "±180° — top-down recordings\n"
            "±15° — side-view recordings\n"
            "None — no rotation augmentation"
        )
        rot_row.addWidget(self.combo_aug_rotation)
        rot_row.addStretch()
        aug_vbox.addLayout(rot_row)

        # --- Scale ---
        self.chk_aug_scale = QCheckBox("Scale")
        self.chk_aug_scale.setChecked(False)
        self.chk_aug_scale.setStyleSheet(aug_lbl_style)
        self.chk_aug_scale.setToolTip(
            "Random zoom in/out around the image center.\n"
            "Helps the model handle animals at varying distances."
        )
        scale_row = QHBoxLayout()
        scale_row.setSpacing(4)
        scale_row.addWidget(self.chk_aug_scale)
        lbl_smin = QLabel("min:")
        lbl_smin.setStyleSheet(aug_lbl_style)
        self.spin_aug_scale_min = QDoubleSpinBox()
        self.spin_aug_scale_min.setRange(0.50, 0.99)
        self.spin_aug_scale_min.setValue(0.80)
        self.spin_aug_scale_min.setSingleStep(0.05)
        self.spin_aug_scale_min.setDecimals(2)
        self.spin_aug_scale_min.setStyleSheet(aug_spin_style)
        lbl_smax = QLabel("max:")
        lbl_smax.setStyleSheet(aug_lbl_style)
        self.spin_aug_scale_max = QDoubleSpinBox()
        self.spin_aug_scale_max.setRange(1.01, 2.00)
        self.spin_aug_scale_max.setValue(1.20)
        self.spin_aug_scale_max.setSingleStep(0.05)
        self.spin_aug_scale_max.setDecimals(2)
        self.spin_aug_scale_max.setStyleSheet(aug_spin_style)
        scale_row.addWidget(lbl_smin)
        scale_row.addWidget(self.spin_aug_scale_min)
        scale_row.addWidget(lbl_smax)
        scale_row.addWidget(self.spin_aug_scale_max)
        scale_row.addStretch()
        self._scale_min_max = [lbl_smin, self.spin_aug_scale_min,
                               lbl_smax, self.spin_aug_scale_max]
        for w in self._scale_min_max:
            w.setVisible(False)
        self.chk_aug_scale.toggled.connect(
            lambda on: [w.setVisible(on) for w in self._scale_min_max]
        )
        aug_vbox.addLayout(scale_row)

        # --- Brightness ---
        self.chk_aug_brightness = QCheckBox("Brightness")
        self.chk_aug_brightness.setChecked(False)
        self.chk_aug_brightness.setStyleSheet(aug_lbl_style)
        bright_row = QHBoxLayout()
        bright_row.setSpacing(4)
        bright_row.addWidget(self.chk_aug_brightness)
        lbl_bmin = QLabel("min:")
        lbl_bmin.setStyleSheet(aug_lbl_style)
        self.spin_aug_bright_min = QSpinBox()
        self.spin_aug_bright_min.setRange(-100, -1)
        self.spin_aug_bright_min.setValue(-10)
        self.spin_aug_bright_min.setSuffix("%")
        self.spin_aug_bright_min.setStyleSheet(aug_spin_style)
        lbl_bmax = QLabel("max:")
        lbl_bmax.setStyleSheet(aug_lbl_style)
        self.spin_aug_bright_max = QSpinBox()
        self.spin_aug_bright_max.setRange(1, 100)
        self.spin_aug_bright_max.setValue(10)
        self.spin_aug_bright_max.setSuffix("%")
        self.spin_aug_bright_max.setStyleSheet(aug_spin_style)
        bright_row.addWidget(lbl_bmin)
        bright_row.addWidget(self.spin_aug_bright_min)
        bright_row.addWidget(lbl_bmax)
        bright_row.addWidget(self.spin_aug_bright_max)
        bright_row.addStretch()
        self._bright_min_max = [lbl_bmin, self.spin_aug_bright_min,
                                lbl_bmax, self.spin_aug_bright_max]
        for w in self._bright_min_max:
            w.setVisible(False)
        self.chk_aug_brightness.toggled.connect(
            lambda on: [w.setVisible(on) for w in self._bright_min_max]
        )
        aug_vbox.addLayout(bright_row)

        # --- Contrast ---
        self.chk_aug_contrast = QCheckBox("Contrast")
        self.chk_aug_contrast.setChecked(False)
        self.chk_aug_contrast.setStyleSheet(aug_lbl_style)
        self.chk_aug_contrast.setToolTip(
            "Randomly adjust image contrast.\n"
            "Helps with varying lighting conditions between sessions."
        )
        contrast_row = QHBoxLayout()
        contrast_row.setSpacing(4)
        contrast_row.addWidget(self.chk_aug_contrast)
        lbl_cmin = QLabel("min:")
        lbl_cmin.setStyleSheet(aug_lbl_style)
        self.spin_aug_contrast_min = QSpinBox()
        self.spin_aug_contrast_min.setRange(-100, -1)
        self.spin_aug_contrast_min.setValue(-10)
        self.spin_aug_contrast_min.setSuffix("%")
        self.spin_aug_contrast_min.setStyleSheet(aug_spin_style)
        lbl_cmax = QLabel("max:")
        lbl_cmax.setStyleSheet(aug_lbl_style)
        self.spin_aug_contrast_max = QSpinBox()
        self.spin_aug_contrast_max.setRange(1, 100)
        self.spin_aug_contrast_max.setValue(10)
        self.spin_aug_contrast_max.setSuffix("%")
        self.spin_aug_contrast_max.setStyleSheet(aug_spin_style)
        contrast_row.addWidget(lbl_cmin)
        contrast_row.addWidget(self.spin_aug_contrast_min)
        contrast_row.addWidget(lbl_cmax)
        contrast_row.addWidget(self.spin_aug_contrast_max)
        contrast_row.addStretch()
        self._contrast_min_max = [lbl_cmin, self.spin_aug_contrast_min,
                                  lbl_cmax, self.spin_aug_contrast_max]
        for w in self._contrast_min_max:
            w.setVisible(False)
        self.chk_aug_contrast.toggled.connect(
            lambda on: [w.setVisible(on) for w in self._contrast_min_max]
        )
        aug_vbox.addLayout(contrast_row)

        self._aug_expansion_label = QLabel("")
        self._aug_expansion_label.setStyleSheet("color: #999999; font-size: 10px;")
        aug_vbox.addWidget(self._aug_expansion_label)

        # Connect all aug widgets to update summary label
        self._aug_update_widgets = [
            self.combo_aug_rotation, self.chk_aug_scale,
            self.chk_aug_brightness, self.chk_aug_contrast,
        ]
        self.combo_aug_rotation.currentIndexChanged.connect(
            lambda _: self._update_aug_expansion_label()
        )
        for chk in [self.chk_aug_scale, self.chk_aug_brightness, self.chk_aug_contrast]:
            chk.toggled.connect(self._update_aug_expansion_label)

        self._aug_options_frame.setVisible(False)
        train_vbox.addWidget(self._aug_options_frame)

        def _toggle_aug_opts(checked):
            self._aug_options_frame.setVisible(checked)
            if checked:
                self._update_aug_expansion_label()
        self.chk_augment.toggled.connect(_toggle_aug_opts)

        # --- Post-training inference (active learning) ---
        self.chk_post_infer = QCheckBox("Run inference on unlabeled frames after training")
        self.chk_post_infer.setChecked(False)
        self.chk_post_infer.setToolTip(
            "After training completes, automatically run the new model\n"
            "on all extracted frames that don't have manual annotations.\n\n"
            "Inferred masks appear in yellow on the Annotator tab.\n"
            "Right-click → 'Approve Mask' to accept them, or delete\n"
            "and re-annotate. Approved masks are included in the next\n"
            "training round, rapidly expanding your dataset.\n\n"
            "This is an active learning loop:\n"
            "  Annotate → Train → Auto-infer → Review → Retrain"
        )
        train_vbox.addWidget(self.chk_post_infer)

        post_infer_row = QHBoxLayout()
        self.lbl_post_infer_max = QLabel("  Max objects per frame:")
        self.lbl_post_infer_max.setToolTip(
            "Maximum number of objects to detect per frame during\n"
            "post-training inference. Set to the expected number of\n"
            "animals/objects in each frame."
        )
        post_infer_row.addWidget(self.lbl_post_infer_max)
        self.spin_post_infer_max = QSpinBox()
        self.spin_post_infer_max.setRange(0, 100)
        self.spin_post_infer_max.setValue(0)
        self.spin_post_infer_max.setSpecialValueText("Unlimited")
        self.spin_post_infer_max.setStyleSheet(spin_style)
        self.spin_post_infer_max.setToolTip(
            "Maximum number of objects to detect per frame during\n"
            "post-training inference. 'Unlimited' detects all objects\n"
            "above the confidence threshold."
        )
        post_infer_row.addWidget(self.spin_post_infer_max)
        train_vbox.addLayout(post_infer_row)

        post_conf_row = QHBoxLayout()
        self.lbl_post_infer_conf = QLabel("  Confidence threshold:")
        self.lbl_post_infer_conf.setToolTip(
            "Minimum confidence score for a detection to be kept.\n"
            "Lower values detect more objects but may include false\n"
            "positives. Higher values are more selective.\n\n"
            "Default 0.30 is appropriate for early training rounds\n"
            "with limited data. Increase to 0.5+ once the model\n"
            "has been trained on more annotated images."
        )
        post_conf_row.addWidget(self.lbl_post_infer_conf)
        self.spin_post_infer_conf = QDoubleSpinBox()
        self.spin_post_infer_conf.setRange(0.05, 0.95)
        self.spin_post_infer_conf.setValue(0.30)
        self.spin_post_infer_conf.setSingleStep(0.05)
        self.spin_post_infer_conf.setDecimals(2)
        self.spin_post_infer_conf.setStyleSheet(spin_style)
        self.spin_post_infer_conf.setToolTip(
            "Minimum confidence score (0.05–0.95).\n"
            "Default 0.30 works well for early active learning rounds."
        )
        post_conf_row.addWidget(self.spin_post_infer_conf)
        train_vbox.addLayout(post_conf_row)

        # Show/hide post-infer options based on checkbox
        def _toggle_post_infer_opts(checked):
            self.lbl_post_infer_max.setVisible(checked)
            self.spin_post_infer_max.setVisible(checked)
            self.lbl_post_infer_conf.setVisible(checked)
            self.spin_post_infer_conf.setVisible(checked)
        self.chk_post_infer.toggled.connect(_toggle_post_infer_opts)
        self.lbl_post_infer_max.setVisible(False)
        self.spin_post_infer_max.setVisible(False)
        self.lbl_post_infer_conf.setVisible(False)
        self.spin_post_infer_conf.setVisible(False)

        self._on_architecture_changed(0)

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

    # ==================================================================
    # Classifier tab
    # ==================================================================
    def _build_classifier_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        spin_style = getattr(self, "_spin_style", "")

        # Classifier state
        self._cls_video_cap = None
        self._cls_frame_idx = 0
        self._cls_total_frames = 0
        self._cls_fps = 30.0
        self._cls_video_path = None
        self._cls_clip_masks = {}    # obj_id -> list of {mask, bbox, centroid, label}
        self._cls_clip_start = 0
        self._cls_mask_categories = {}  # {int_label: str_name} from model's training_config
        self._cls_model_loaded = None
        self._cls_annotations = []   # list of annotation dicts
        self._batch_clips = []       # batch extraction clip queue
        self._cls_in_queue_mode = False
        self._cls_pending_clip_mode = False
        self._cls_playback_offset = 0
        self._cls_playback_clip_idx = -1
        self._cls_speed_multiplier = 1.0
        self._cls_playback_was_playing = False
        self._pending_frames = []
        self._queue_frames = []

        # --- Load Videos ---
        vid_group = QGroupBox("Load Videos")
        vid_vbox = QVBoxLayout()
        vid_vbox.setSpacing(4)

        vid_btn_row = QHBoxLayout()
        btn_add_cls_vids = QPushButton("Add Videos")
        btn_add_cls_vids.setToolTip("Add videos for behavioral annotation.")
        btn_add_cls_vids.clicked.connect(self._add_classifier_videos)
        vid_btn_row.addWidget(btn_add_cls_vids)
        btn_clear_cls_vids = QPushButton("Clear")
        btn_clear_cls_vids.setToolTip("Remove all annotation videos.")
        btn_clear_cls_vids.clicked.connect(self._clear_classifier_videos)
        vid_btn_row.addWidget(btn_clear_cls_vids)
        vid_vbox.addLayout(vid_btn_row)

        self.list_cls_videos = QListWidget()
        self.list_cls_videos.setMaximumHeight(80)
        self.list_cls_videos.currentItemChanged.connect(self._on_cls_video_selected)
        vid_vbox.addWidget(self.list_cls_videos)

        vid_group.setLayout(vid_vbox)
        layout.addWidget(vid_group)

        # --- Mask Model ---
        model_group = QGroupBox("Mask Model")
        model_vbox = QVBoxLayout()
        model_vbox.setSpacing(4)

        model_row = QHBoxLayout()
        self.combo_cls_model = QComboBox()
        self.combo_cls_model.setToolTip(
            "Select a trained mask model for silhouette extraction.\n"
            "Defaults to the most recently trained model."
        )
        model_row.addWidget(self.combo_cls_model, 1)
        btn_refresh_cls = QPushButton("Refresh")
        btn_refresh_cls.setToolTip("Rescan the models/ directory.")
        btn_refresh_cls.clicked.connect(self._refresh_classifier_models)
        model_row.addWidget(btn_refresh_cls)
        model_vbox.addLayout(model_row)

        self.lbl_cls_model_info = QLabel("No model selected")
        self.lbl_cls_model_info.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_cls_model_info.setWordWrap(True)
        model_vbox.addWidget(self.lbl_cls_model_info)

        model_group.setLayout(model_vbox)
        layout.addWidget(model_group)

        # Behavior categories list (managed via popup, not shown in left panel)
        self.list_cls_categories = QListWidget()
        self.list_cls_categories.setVisible(False)

        # --- Clip Extraction ---
        clip_group = QGroupBox("Clip Extraction")
        clip_vbox = QVBoxLayout()
        clip_vbox.setSpacing(4)

        row_clip = QHBoxLayout()
        row_clip.addWidget(QLabel("Clip length:"))
        self.spin_cls_clip_length = QSpinBox()
        self.spin_cls_clip_length.setRange(5, 60)
        self.spin_cls_clip_length.setValue(15)
        self.spin_cls_clip_length.setStyleSheet(spin_style)
        self.spin_cls_clip_length.setToolTip(
            "Number of frames per clip (5-60).\n"
            "Default 15 frames = ~0.5s at 30fps.\n"
            "Shorter = fast behaviors (grooming).\n"
            "Longer = slow behaviors (exploration).\n\n"
            "Applies to both batch extraction (Extract Clips)\n"
            "and manual single-clip extraction (press E)."
        )
        row_clip.addWidget(self.spin_cls_clip_length)
        row_clip.addWidget(QLabel("frames"))
        clip_vbox.addLayout(row_clip)

        manual_note = QLabel(
            "Tip: press E to manually extract a single clip at the current frame."
        )
        manual_note.setStyleSheet("color: #888888; font-size: 9px;")
        manual_note.setWordWrap(True)
        clip_vbox.addWidget(manual_note)

        row_conf = QHBoxLayout()
        row_conf.addWidget(QLabel("Confidence:"))
        self.spin_cls_confidence = QDoubleSpinBox()
        self.spin_cls_confidence.setRange(0.05, 0.99)
        self.spin_cls_confidence.setValue(0.5)
        self.spin_cls_confidence.setSingleStep(0.05)
        self.spin_cls_confidence.setStyleSheet(spin_style)
        self.spin_cls_confidence.setToolTip(
            "Minimum detection confidence for mask extraction."
        )
        row_conf.addWidget(self.spin_cls_confidence)
        clip_vbox.addLayout(row_conf)

        row_max = QHBoxLayout()
        row_max.addWidget(QLabel("Max objects:"))
        self.spin_cls_max_det = QSpinBox()
        self.spin_cls_max_det.setRange(0, 100)
        self.spin_cls_max_det.setValue(0)
        self.spin_cls_max_det.setSpecialValueText("Unlimited")
        self.spin_cls_max_det.setStyleSheet(spin_style)
        self.spin_cls_max_det.setToolTip(
            "Maximum objects to detect per frame.\n"
            "'Unlimited' detects all objects above the confidence threshold."
        )
        row_max.addWidget(self.spin_cls_max_det)
        clip_vbox.addLayout(row_max)

        row_method = QHBoxLayout()
        row_method.addWidget(QLabel("Method:"))
        self.combo_batch_method = QComboBox()
        self.combo_batch_method.addItems([
            "Uniform sample", "Random sample", "Every Nth frame",
        ])
        self.combo_batch_method.setToolTip(
            "How to select clip start positions across the video.\n"
            "Uniform: evenly spaced across the video.\n"
            "Random: randomly sampled positions.\n"
            "Every Nth: one clip every N frames."
        )
        row_method.addWidget(self.combo_batch_method, 1)
        clip_vbox.addLayout(row_method)

        row_nclips = QHBoxLayout()
        self.lbl_batch_count = QLabel("Clips per video:")
        row_nclips.addWidget(self.lbl_batch_count)
        self.spin_batch_count = QSpinBox()
        self.spin_batch_count.setRange(1, 500)
        self.spin_batch_count.setValue(20)
        self.spin_batch_count.setStyleSheet(spin_style)
        self.spin_batch_count.setToolTip(
            "Number of clips to extract per video.\n"
            "For 'Every Nth frame', this is the stride in frames."
        )
        row_nclips.addWidget(self.spin_batch_count)
        clip_vbox.addLayout(row_nclips)

        self.combo_batch_method.currentIndexChanged.connect(
            lambda idx: self.lbl_batch_count.setText(
                "Stride (frames):" if idx == 2 else "Clips per video:"
            )
        )

        self.btn_batch_extract = QPushButton("Extract Clips")
        self.btn_batch_extract.setStyleSheet(
            "QPushButton { background-color: #2979ff; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #448aff; }"
            "QPushButton:disabled { background-color: #333333; color: #666666; }"
        )
        self.btn_batch_extract.setToolTip(
            "Extract clips from all loaded videos in the background.\n"
            "You can start labeling as clips finish extracting."
        )
        self.btn_batch_extract.clicked.connect(self._start_batch_extraction)
        clip_vbox.addWidget(self.btn_batch_extract)

        self.lbl_cls_extract_status = QLabel(
            "Scrub to a behavior, press E to extract clip"
        )
        self.lbl_cls_extract_status.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_cls_extract_status.setWordWrap(True)
        clip_vbox.addWidget(self.lbl_cls_extract_status)

        clip_group.setLayout(clip_vbox)
        layout.addWidget(clip_group)

        # --- Clip Queue ---
        queue_group = QGroupBox("Clip Queue")
        queue_vbox = QVBoxLayout()
        queue_vbox.setSpacing(4)

        self.list_clip_queue = QTreeWidget()
        self.list_clip_queue.setMaximumHeight(160)
        self.list_clip_queue.setColumnCount(3)
        self.list_clip_queue.setHeaderLabels(["Clip", "Frames", "Action"])
        cq_header = self.list_clip_queue.header()
        cq_header.setStretchLastSection(False)
        cq_header.setSectionResizeMode(0, QHeaderView.Stretch)
        cq_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        cq_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        cq_header.setSortIndicatorShown(True)
        cq_header.setSectionsClickable(True)
        self.list_clip_queue.setRootIsDecorated(False)
        self.list_clip_queue.setSortingEnabled(True)
        self.list_clip_queue.sortByColumn(0, Qt.AscendingOrder)
        self.list_clip_queue.setToolTip(
            "Extracted clips. Yellow = pending, Green = labeled.\n"
            "Click a clip or press Up/Down to navigate.\n"
            "Selected clip auto-plays in the preview."
        )
        self.list_clip_queue.currentItemChanged.connect(self._on_clip_queue_selected)
        queue_vbox.addWidget(self.list_clip_queue)

        self.lbl_cls_annotation_stats = QLabel("No clips yet")
        self.lbl_cls_annotation_stats.setStyleSheet("color: #cccccc; font-size: 10px;")
        self.lbl_cls_annotation_stats.setWordWrap(True)
        queue_vbox.addWidget(self.lbl_cls_annotation_stats)

        del_btn_row = QHBoxLayout()
        self.btn_clear_all_labels = QPushButton("Clear Labels")
        self.btn_clear_all_labels.setStyleSheet(
            "QPushButton { background-color: #f9a825; color: #1a1a1a; font-weight: bold; "
            "padding: 4px 8px; border-radius: 3px; font-size: 10px; }"
            "QPushButton:hover { background-color: #fdd835; }"
        )
        self.btn_clear_all_labels.setToolTip("Reset all clips to pending (remove behavior assignments).")
        self.btn_clear_all_labels.clicked.connect(self._clear_all_clip_labels)
        del_btn_row.addWidget(self.btn_clear_all_labels)

        self.btn_delete_selected_clip = QPushButton("Delete Selected")
        self.btn_delete_selected_clip.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; font-weight: bold; "
            "padding: 4px 8px; border-radius: 3px; font-size: 10px; }"
            "QPushButton:hover { background-color: #e53935; }"
        )
        self.btn_delete_selected_clip.setToolTip("Delete the selected clip from disk.")
        self.btn_delete_selected_clip.clicked.connect(self._delete_selected_clips)
        del_btn_row.addWidget(self.btn_delete_selected_clip)

        self.btn_delete_all_clips = QPushButton("Delete All")
        self.btn_delete_all_clips.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; font-weight: bold; "
            "padding: 4px 8px; border-radius: 3px; font-size: 10px; }"
            "QPushButton:hover { background-color: #e53935; }"
        )
        self.btn_delete_all_clips.setToolTip("Delete all clips from disk.")
        self.btn_delete_all_clips.clicked.connect(self._delete_all_clips)
        del_btn_row.addWidget(self.btn_delete_all_clips)
        queue_vbox.addLayout(del_btn_row)

        self.btn_reextract_clips = QPushButton("Re-extract Clips")
        self.btn_reextract_clips.setStyleSheet(
            "QPushButton { background-color: #1565c0; color: white; font-weight: bold; "
            "padding: 5px 10px; border-radius: 3px; font-size: 10px; }"
            "QPushButton:hover { background-color: #1976d2; }"
        )
        self.btn_reextract_clips.setToolTip(
            "Re-run the currently selected mask model on all existing clips.\n"
            "Use after training a new mask model to update silhouette masks\n"
            "without losing your behavioral annotations."
        )
        self.btn_reextract_clips.clicked.connect(self._reextract_clips)
        queue_vbox.addWidget(self.btn_reextract_clips)

        # Enable multi-selection in clip queue
        self.list_clip_queue.setSelectionMode(QTreeWidget.ExtendedSelection)

        queue_group.setLayout(queue_vbox)
        layout.addWidget(queue_group)

        # --- Train Classifier ---
        cls_train_group = QGroupBox("Train Classifier")
        cls_train_vbox = QVBoxLayout()
        cls_train_vbox.setSpacing(4)
        spin_style = self._spin_style

        cls_train_info = QLabel(
            "Train a CNN on time-colored silhouette contour composites\n"
            "from your annotated clips."
        )
        cls_train_info.setStyleSheet("color: #888888; font-size: 9px;")
        cls_train_info.setWordWrap(True)
        cls_train_vbox.addWidget(cls_train_info)

        # Training data summary
        self.lbl_cls_train_data = QLabel("Training data: —")
        self.lbl_cls_train_data.setStyleSheet("color: #cccccc; font-size: 10px;")
        self.lbl_cls_train_data.setWordWrap(True)
        cls_train_vbox.addWidget(self.lbl_cls_train_data)

        # System info
        self.lbl_cls_system_info = QLabel("System: detecting...")
        self.lbl_cls_system_info.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_cls_system_info.setWordWrap(True)
        cls_train_vbox.addWidget(self.lbl_cls_system_info)
        self._populate_cls_system_info()

        self.lbl_cls_train_reqs = QLabel("")
        self.lbl_cls_train_reqs.setStyleSheet(
            "color: #ff9800; font-size: 9px; padding: 2px 0px;"
        )
        self.lbl_cls_train_reqs.setWordWrap(True)
        cls_train_vbox.addWidget(self.lbl_cls_train_reqs)

        # -- Backbone --
        tip_backbone = (
            "CNN architecture used as the image classifier.\n\n"
            "ResNet-18: lightweight, fast to train, good for small datasets\n"
            "  (20-200 composites). Recommended starting point.\n"
            "ResNet-34: deeper network with more capacity. Better for\n"
            "  larger datasets (200+) or when behaviors are visually\n"
            "  similar and harder to distinguish.\n"
            "MobileNetV3: very small and fast. Good for quick experiments\n"
            "  or when you plan to run inference on CPU. Slightly less\n"
            "  accurate than ResNet-18 on small datasets."
        )
        row = QHBoxLayout()
        lbl = QLabel("Backbone:")
        lbl.setToolTip(tip_backbone)
        row.addWidget(lbl)
        self.combo_cls_backbone = QComboBox()
        self.combo_cls_backbone.addItems(["ResNet-18", "ResNet-34", "MobileNetV3"])
        self.combo_cls_backbone.setToolTip(tip_backbone)
        row.addWidget(self.combo_cls_backbone)
        cls_train_vbox.addLayout(row)

        self.chk_cls_freeze = QCheckBox("Freeze backbone (recommended for <100 clips)")
        self.chk_cls_freeze.setChecked(True)
        self.chk_cls_freeze.setToolTip(
            "Freeze the pretrained CNN backbone and only train\n"
            "the final classification layer.\n\n"
            "ON (default): uses ImageNet features as-is and only\n"
            "  learns behavior → class mapping. Much less prone to\n"
            "  overfitting with small datasets (<100 composites).\n"
            "OFF: fine-tunes the entire network. Better when you\n"
            "  have 100+ composites per class, but risks overfitting\n"
            "  on small datasets."
        )
        cls_train_vbox.addWidget(self.chk_cls_freeze)

        # -- Training Iterations --
        tip_iters = (
            "Number of full passes (epochs) through the training data.\n\n"
            "Each iteration processes every composite image once. More\n"
            "iterations give the model more chances to learn, but too\n"
            "many can cause overfitting.\n\n"
            "If early stop patience is set, training will stop\n"
            "automatically when validation loss plateaus.\n\n"
            "50: good default for most datasets.\n"
            "100-200: use with larger or more complex datasets."
        )
        row = QHBoxLayout()
        lbl = QLabel("Training Iterations:")
        lbl.setToolTip(tip_iters)
        row.addWidget(lbl)
        self.spin_cls_epochs = QSpinBox()
        self.spin_cls_epochs.setRange(100, 5000)
        self.spin_cls_epochs.setValue(500)
        self.spin_cls_epochs.setSingleStep(100)
        self.spin_cls_epochs.setToolTip(tip_iters)
        self.spin_cls_epochs.setStyleSheet(spin_style)
        row.addWidget(self.spin_cls_epochs)
        cls_train_vbox.addLayout(row)

        # -- Early Stop Patience --
        tip_patience = (
            "Stop training if validation loss does not improve\n"
            "for this many consecutive epochs.\n\n"
            "50: good default — gives the model enough time to\n"
            "  recover from temporary plateaus.\n"
            "Lower values stop sooner but risk cutting off learning.\n"
            "Set equal to Training Iterations to disable early stopping."
        )
        row = QHBoxLayout()
        lbl = QLabel("Early Stop Patience:")
        lbl.setToolTip(tip_patience)
        row.addWidget(lbl)
        self.spin_cls_patience = QSpinBox()
        self.spin_cls_patience.setRange(5, 500)
        self.spin_cls_patience.setValue(50)
        self.spin_cls_patience.setSingleStep(10)
        self.spin_cls_patience.setToolTip(tip_patience)
        self.spin_cls_patience.setStyleSheet(spin_style)
        row.addWidget(self.spin_cls_patience)
        cls_train_vbox.addLayout(row)

        # -- Learning Rate --
        tip_cls_lr = (
            "Controls how much model weights change per training step.\n\n"
            "Higher = faster learning but risk of instability (loss\n"
            "  spikes or model fails to converge).\n"
            "Lower = more stable but slower training.\n\n"
            "0.001: recommended default for AdamW optimizer.\n"
            "0.0001: use if training loss is unstable or spiky.\n"
            "0.01: use if training is very slow to converge.\n\n"
            "A cosine annealing schedule gradually reduces the LR\n"
            "toward zero over the course of training."
        )
        row = QHBoxLayout()
        lbl = QLabel("Learning rate:")
        lbl.setToolTip(tip_cls_lr)
        row.addWidget(lbl)
        self.spin_cls_lr = QDoubleSpinBox()
        self.spin_cls_lr.setRange(0.0001, 0.01)
        self.spin_cls_lr.setValue(0.001)
        self.spin_cls_lr.setSingleStep(0.0001)
        self.spin_cls_lr.setDecimals(4)
        self.spin_cls_lr.setToolTip(tip_cls_lr)
        self.spin_cls_lr.setStyleSheet(spin_style)
        row.addWidget(self.spin_cls_lr)
        cls_train_vbox.addLayout(row)

        # -- Batch Size --
        tip_cls_batch = (
            "Number of composite images processed per training step.\n\n"
            "Larger batches give more stable gradient estimates and\n"
            "faster training, but use more GPU memory.\n\n"
            "16: good default for 128x128 composites (very small images).\n"
            "8: use if you have very few training examples (<20).\n"
            "32: use with larger datasets (100+) for faster training."
        )
        row = QHBoxLayout()
        lbl = QLabel("Batch size:")
        lbl.setToolTip(tip_cls_batch)
        row.addWidget(lbl)
        self.spin_cls_batch = QSpinBox()
        self.spin_cls_batch.setRange(4, 64)
        self.spin_cls_batch.setValue(16)
        self.spin_cls_batch.setToolTip(tip_cls_batch)
        self.spin_cls_batch.setStyleSheet(spin_style)
        row.addWidget(self.spin_cls_batch)
        cls_train_vbox.addLayout(row)

        # -- Validation Split --
        tip_cls_val = (
            "Fraction of composites held out for validation.\n\n"
            "These images are NOT used for training. After each epoch,\n"
            "validation loss is computed to detect overfitting and\n"
            "trigger early stopping.\n\n"
            "0.20 (20%%): good default. E.g., 50 clips → 40 train, 10 val.\n"
            "0.10: use with very small datasets (<20 clips).\n"
            "0.00: train on all clips. Not recommended — early stopping\n"
            "  won't work and overfitting can't be detected."
        )
        row = QHBoxLayout()
        lbl = QLabel("Validation split:")
        lbl.setToolTip(tip_cls_val)
        row.addWidget(lbl)
        self.spin_cls_val = QDoubleSpinBox()
        self.spin_cls_val.setRange(0.0, 0.5)
        self.spin_cls_val.setValue(0.20)
        self.spin_cls_val.setSingleStep(0.05)
        self.spin_cls_val.setDecimals(2)
        self.spin_cls_val.setToolTip(tip_cls_val)
        self.spin_cls_val.setStyleSheet(spin_style)
        row.addWidget(self.spin_cls_val)
        cls_train_vbox.addLayout(row)

        # -- Data Augmentation --
        aug_group = QGroupBox("Data augmentation")
        aug_vbox = QVBoxLayout()
        aug_vbox.setSpacing(4)

        rot_tip = (
            "Random continuous rotation applied each batch.\n"
            "±180° for top-down views (animal faces any direction).\n"
            "±15° for side views (animal is roughly horizontal).\n"
            "Select 'None' to disable rotation augmentation.\n\n"
            "Silhouette composites have black backgrounds,\n"
            "so rotation is the most useful augmentation."
        )
        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rotation:"))
        self.combo_cls_aug_rotation = QComboBox()
        self.combo_cls_aug_rotation.addItems(["None", "±15°", "±180°"])
        self.combo_cls_aug_rotation.setCurrentIndex(0)
        self.combo_cls_aug_rotation.setMinimumWidth(80)
        self.combo_cls_aug_rotation.setToolTip(rot_tip)
        self.combo_cls_aug_rotation.setStyleSheet(spin_style)
        rot_row.addWidget(self.combo_cls_aug_rotation)
        aug_vbox.addLayout(rot_row)

        aug_group.setLayout(aug_vbox)
        cls_train_vbox.addWidget(aug_group)

        self.btn_train_classifier = QPushButton("Train Action Classifier")
        self.btn_train_classifier.setStyleSheet(
            "QPushButton { background-color: #7b1fa2; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #9c27b0; }"
            "QPushButton:disabled { background-color: #333333; color: #666666; }"
        )
        self.btn_train_classifier.setToolTip(
            "Train a behavior classifier on the annotated clips.\n"
            "Requires at least 2 behavior categories with labeled clips."
        )
        self.btn_train_classifier.clicked.connect(self._start_classifier_training)
        self.btn_train_classifier.setEnabled(False)
        cls_train_vbox.addWidget(self.btn_train_classifier)

        self.cls_train_progress = QProgressBar()
        self.cls_train_progress.setVisible(False)
        cls_train_vbox.addWidget(self.cls_train_progress)

        cls_train_btn_row = QHBoxLayout()
        cls_train_btn_row.setSpacing(6)
        self.btn_cls_stop = QPushButton("Stop Training")
        self.btn_cls_stop.setStyleSheet(
            "QPushButton { background-color: #c62828; border: none; border-radius: 3px; "
            "padding: 4px 12px; color: #ffffff; font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background-color: #e53935; }"
        )
        self.btn_cls_stop.clicked.connect(self._stop_cls_training)
        self.btn_cls_stop.setVisible(False)
        cls_train_btn_row.addWidget(self.btn_cls_stop)
        cls_train_btn_row.addStretch()
        cls_train_vbox.addLayout(cls_train_btn_row)

        self.lbl_cls_train_status = QLabel("")
        self.lbl_cls_train_status.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_cls_train_status.setWordWrap(True)
        cls_train_vbox.addWidget(self.lbl_cls_train_status)

        cls_train_group.setLayout(cls_train_vbox)
        layout.addWidget(cls_train_group)

        layout.addStretch()
        scroll.setWidget(widget)
        self.tab_widget.addTab(scroll, "Actions")

    def _build_tracking_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # --- Video Queue (top) ---
        video_group = QGroupBox("Video Queue")
        video_vbox = QVBoxLayout()
        video_vbox.setSpacing(4)

        vid_btn_row = QHBoxLayout()
        btn_add_vids = QPushButton("Add Videos")
        btn_add_vids.setToolTip("Add individual video files to the inference queue.")
        btn_add_vids.clicked.connect(self._add_tracking_videos)
        vid_btn_row.addWidget(btn_add_vids)
        btn_add_vid_folder = QPushButton("Add Folder")
        btn_add_vid_folder.setToolTip(
            "Add all video files from a folder to the inference queue."
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
        self.list_track_videos.setToolTip("Videos queued for inference. All will be processed sequentially.")
        self.list_track_videos.currentItemChanged.connect(self._on_tracking_video_selected)
        video_vbox.addWidget(self.list_track_videos)

        video_group.setLayout(video_vbox)
        layout.addWidget(video_group)

        # --- Tracking Settings (model + inference params) ---
        settings_group = QGroupBox("Tracking Settings")
        settings_vbox = QVBoxLayout()
        settings_vbox.setSpacing(4)

        model_row = QHBoxLayout()
        lbl_model = QLabel("Tracking model:")
        model_row.addWidget(lbl_model)
        self.combo_track_model = QComboBox()
        self.combo_track_model.setToolTip(
            "Select a trained detection/segmentation model from this project.\n"
            "Models are saved in the project's models/ directory after training."
        )
        self.combo_track_model.currentIndexChanged.connect(self._on_track_model_changed)
        model_row.addWidget(self.combo_track_model, 1)
        btn_refresh_models = QPushButton("Refresh")
        btn_refresh_models.setToolTip("Rescan the models/ directory for trained models.")
        btn_refresh_models.clicked.connect(self._refresh_model_list)
        model_row.addWidget(btn_refresh_models)
        settings_vbox.addLayout(model_row)

        self.lbl_track_model_info = QLabel("No model selected")
        self.lbl_track_model_info.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_track_model_info.setWordWrap(True)
        settings_vbox.addWidget(self.lbl_track_model_info)

        self.lbl_track_system_info = QLabel("")
        self.lbl_track_system_info.setStyleSheet("color: #888888; font-size: 10px;")
        self.lbl_track_system_info.setWordWrap(True)
        settings_vbox.addWidget(self.lbl_track_system_info)

        row0 = QHBoxLayout()
        row0.addWidget(QLabel("Max objects:"))
        self.spin_track_max_det = QSpinBox()
        self.spin_track_max_det.setRange(0, 100)
        self.spin_track_max_det.setValue(0)
        self.spin_track_max_det.setSpecialValueText("Unlimited")
        self.spin_track_max_det.setStyleSheet(self._spin_style)
        self.spin_track_max_det.setToolTip(
            "Maximum number of objects/masks to detect per frame.\n"
            "'Unlimited' detects all objects above the confidence\n"
            "threshold. Set a number to keep only the top-N highest-\n"
            "confidence detections each frame."
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

        # --- Behavior Classification ---
        cls_group = QGroupBox("Behavior Classification")
        cls_vbox = QVBoxLayout()
        cls_vbox.setSpacing(4)

        self.chk_behavior_cls = QCheckBox("Enable behavior classification")
        self.chk_behavior_cls.setChecked(False)
        self.chk_behavior_cls.setToolTip(
            "Run a trained behavior classifier on tracked objects.\n\n"
            "When enabled, each tracked object accumulates mask data\n"
            "over a sliding window, generates a silhouette composite,\n"
            "and classifies the behavior using the selected model.\n\n"
            "Requires a trained classifier model from the Classifier tab."
        )
        self.chk_behavior_cls.toggled.connect(self._on_behavior_cls_toggled)
        cls_vbox.addWidget(self.chk_behavior_cls)

        self._behavior_cls_container = QWidget()
        cls_opts = QVBoxLayout()
        cls_opts.setContentsMargins(0, 0, 0, 0)
        cls_opts.setSpacing(4)

        row_bcm = QHBoxLayout()
        lbl_bcm = QLabel("Classifier model:")
        row_bcm.addWidget(lbl_bcm)
        self.combo_behavior_model = QComboBox()
        self.combo_behavior_model.setToolTip(
            "Select a trained behavior classifier model.\n"
            "Models are saved in action_classifier/model/ after training."
        )
        row_bcm.addWidget(self.combo_behavior_model, 1)
        btn_refresh_cls = QPushButton("Refresh")
        btn_refresh_cls.setToolTip("Rescan for trained classifier models.")
        btn_refresh_cls.clicked.connect(self._refresh_classifier_model_list)
        row_bcm.addWidget(btn_refresh_cls)
        cls_opts.addLayout(row_bcm)

        self.lbl_behavior_model_info = QLabel("No classifier model found")
        self.lbl_behavior_model_info.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_behavior_model_info.setWordWrap(True)
        cls_opts.addWidget(self.lbl_behavior_model_info)

        row_nc = QHBoxLayout()
        lbl_nc = QLabel("NC threshold:")
        lbl_nc.setToolTip(
            "Confidence threshold for classification.\n"
            "Predictions below this confidence are labeled NC (not classified)."
        )
        row_nc.addWidget(lbl_nc)
        self.spin_nc_threshold = QDoubleSpinBox()
        self.spin_nc_threshold.setRange(0.10, 0.95)
        self.spin_nc_threshold.setValue(0.50)
        self.spin_nc_threshold.setSingleStep(0.05)
        self.spin_nc_threshold.setDecimals(2)
        self.spin_nc_threshold.setStyleSheet(self._spin_style)
        self.spin_nc_threshold.setToolTip(
            "Minimum softmax confidence to assign a behavior label.\n\n"
            "0.50 (default): predictions must be >50% confident.\n"
            "Lower: more predictions, but more noise.\n"
            "Higher: fewer predictions, but more reliable."
        )
        row_nc.addWidget(self.spin_nc_threshold)
        cls_opts.addLayout(row_nc)

        row_win = QHBoxLayout()
        lbl_win = QLabel("Window size:")
        lbl_win.setToolTip("Number of frames used to generate each behavior composite.")
        row_win.addWidget(lbl_win)
        self.spin_cls_window = QSpinBox()
        self.spin_cls_window.setRange(5, 60)
        self.spin_cls_window.setValue(15)
        self.spin_cls_window.setSingleStep(5)
        self.spin_cls_window.setStyleSheet(self._spin_style)
        self.spin_cls_window.setToolTip(
            "Number of consecutive frames used to build a silhouette\n"
            "composite for behavior classification.\n\n"
            "15 (default): ~0.5s at 30fps. Good for quick behaviors.\n"
            "30: ~1s. Better for slower behaviors like grooming.\n"
            "Should roughly match the clip length used during training."
        )
        row_win.addWidget(self.spin_cls_window)
        row_win.addWidget(QLabel("frames"))
        cls_opts.addLayout(row_win)

        row_bout = QHBoxLayout()
        lbl_bout = QLabel("Min bout length:")
        lbl_bout.setToolTip(
            "Minimum consecutive frames for a behavior label to be kept.\n"
            "Shorter bouts are relabeled NC to suppress flickering."
        )
        row_bout.addWidget(lbl_bout)
        self.spin_min_bout = QSpinBox()
        self.spin_min_bout.setRange(1, 30)
        self.spin_min_bout.setValue(5)
        self.spin_min_bout.setStyleSheet(self._spin_style)
        self.spin_min_bout.setToolTip(
            "Behavior bouts shorter than this many frames\n"
            "are replaced with NC (not classified).\n\n"
            "5 (default): suppresses brief flickering between classes.\n"
            "1: no filtering — keep every per-frame prediction.\n"
            "10-15: aggressive smoothing for noisy predictions."
        )
        row_bout.addWidget(self.spin_min_bout)
        row_bout.addWidget(QLabel("frames"))
        cls_opts.addLayout(row_bout)

        row_gap = QHBoxLayout()
        lbl_gap = QLabel("Uncertainty gap:")
        lbl_gap.setToolTip(
            "Minimum gap between top-2 class probabilities.\n"
            "If the gap is smaller, the prediction is labeled NC."
        )
        row_gap.addWidget(lbl_gap)
        self.spin_uncertain_gap = QDoubleSpinBox()
        self.spin_uncertain_gap.setRange(0.0, 0.50)
        self.spin_uncertain_gap.setValue(0.10)
        self.spin_uncertain_gap.setSingleStep(0.05)
        self.spin_uncertain_gap.setDecimals(2)
        self.spin_uncertain_gap.setStyleSheet(self._spin_style)
        self.spin_uncertain_gap.setToolTip(
            "If the difference between the top prediction's confidence\n"
            "and the second-best is below this threshold, the frame\n"
            "is labeled NC (ambiguous prediction).\n\n"
            "0.10 (default): top class must be ≥10%% more confident\n"
            "  than the runner-up.\n"
            "0.00: disable this check (rely on NC threshold only).\n"
            "0.20: stricter — requires a clear winner."
        )
        row_gap.addWidget(self.spin_uncertain_gap)
        cls_opts.addLayout(row_gap)

        self._behavior_cls_container.setLayout(cls_opts)
        cls_vbox.addWidget(self._behavior_cls_container)

        cls_group.setLayout(cls_vbox)
        layout.addWidget(cls_group)

        self._behavior_cls_container.setVisible(False)

        # --- Run Inference ---
        ctrl_group = QGroupBox("Run Inference")
        ctrl_vbox = QVBoxLayout()
        ctrl_vbox.setSpacing(4)

        self.chk_create_tracked_video = QCheckBox("Create tracked video")
        self.chk_create_tracked_video.setChecked(True)
        self.chk_create_tracked_video.setToolTip(
            "Save an annotated video with tracking overlays.\n\n"
            "ON: write an annotated .mp4 with bounding boxes / masks.\n"
            "OFF: skip video output for faster processing (CSV only)."
        )
        ctrl_vbox.addWidget(self.chk_create_tracked_video)

        self.chk_inference_preview = QCheckBox("Show inference preview")
        self.chk_inference_preview.setChecked(True)
        self.chk_inference_preview.setToolTip(
            "Display annotated frames in the preview window during inference.\n\n"
            "ON: see live tracking + behavior overlays as videos are processed.\n"
            "OFF: skip rendering to the preview window for faster processing."
        )
        ctrl_vbox.addWidget(self.chk_inference_preview)

        btn_row = QHBoxLayout()
        self.btn_start_tracking = QPushButton("Run Inference")
        self.btn_start_tracking.setStyleSheet(
            "QPushButton { background-color: #2979ff; color: white; font-weight: bold; "
            "padding: 8px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #448aff; }"
            "QPushButton:disabled { background-color: #333333; color: #666666; }"
        )
        self.btn_start_tracking.setToolTip("Run inference on all queued videos.")
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

        self.track_queue_progress = QProgressBar()
        self.track_queue_progress.setStyleSheet(
            "QProgressBar { border: 1px solid #555; border-radius: 3px; "
            "background-color: #1a1a1a; height: 14px; text-align: center; "
            "color: #ffffff; font-size: 10px; }"
            "QProgressBar::chunk { background-color: #43a047; }"
        )
        self.track_queue_progress.setFormat("Queue: %v / %m videos")
        self.track_queue_progress.setVisible(False)
        ctrl_vbox.addWidget(self.track_queue_progress)

        ctrl_group.setLayout(ctrl_vbox)
        layout.addWidget(ctrl_group)

        # --- Results ---
        results_group = QGroupBox("Results")
        results_vbox = QVBoxLayout()
        self.list_track_results = QListWidget()
        self.list_track_results.setMaximumHeight(160)
        self.list_track_results.setToolTip(
            "Completed videos with track counts and CSV output paths.\n"
            "Double-click to open the output folder."
        )
        self.list_track_results.itemDoubleClicked.connect(self._open_tracking_result)
        results_vbox.addWidget(self.list_track_results)
        results_group.setLayout(results_vbox)
        layout.addWidget(results_group)

        layout.addStretch()
        scroll.setWidget(widget)
        self.tab_widget.addTab(scroll, "Inference")
        self._apply_system_defaults()

    def _apply_system_defaults(self):
        """Auto-detect hardware and set optimal tracking defaults."""
        try:
            from .mask_tracker_inference import detect_system_profile
            profile = detect_system_profile()
        except Exception:
            return

        dev = profile["recommended_device"]
        device_map = {"cpu": "CPU", "cuda": "CUDA (GPU)", "mps": "MPS (Apple Silicon)"}
        target_text = device_map.get(dev, "CPU")
        for i in range(self.combo_track_device.count()):
            if self.combo_track_device.itemText(i).startswith(target_text.split()[0]):
                self.combo_track_device.setCurrentIndex(i)
                break

        res = profile["recommended_resolution"]
        if res > 0:
            target = f"{res}px"
            for i in range(self.combo_track_resolution.count()):
                if self.combo_track_resolution.itemText(i).startswith(str(res)):
                    self.combo_track_resolution.setCurrentIndex(i)
                    break

        self.chk_track_masks.setChecked(profile["recommended_use_masks"])

        chip = profile.get("chip", "Unknown")
        ram = profile.get("ram_gb")
        ram_str = f", {ram}GB RAM" if ram else ""
        masks_str = "masks" if profile["recommended_use_masks"] else "boxes"
        res_str = f"{res}px" if res > 0 else "trained"
        self.lbl_track_system_info.setText(
            f"Detected: {chip}{ram_str} — defaults: {dev}, {res_str}, {masks_str}"
        )

    # ------------------------------------------------------------------
    # Annotator tab sections
    # ------------------------------------------------------------------
    def _create_videos_section(self, layout):
        group = QGroupBox("Add Videos")
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
        group = QGroupBox("Frame Extraction")
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
        self.spin_count.setStyleSheet(self._spin_style)
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
        group = QGroupBox("Training Frames Queue")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        self._frame_confidence: dict = {}
        self._shuffle_mode = False

        self.frame_list = QTreeWidget()
        self.frame_list.setMaximumHeight(320)
        self.frame_list.setColumnCount(2)
        self.frame_list.setHeaderLabels(["Frame", "Conf"])
        header = self.frame_list.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSortIndicatorShown(True)
        header.setSectionsClickable(True)
        self.frame_list.setRootIsDecorated(False)
        self.frame_list.setSortingEnabled(True)
        self.frame_list.sortByColumn(0, Qt.AscendingOrder)
        self.frame_list.currentItemChanged.connect(self._on_frame_item_changed)
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

        self.btn_shuffle = QPushButton("Shuffle")
        self.btn_shuffle.setCheckable(True)
        self.btn_shuffle.setChecked(False)
        self.btn_shuffle.setStyleSheet(
            "QPushButton { background-color: #3c3c3c; border: 1px solid #555555; "
            "border-radius: 3px; padding: 3px 8px; color: #cccccc; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
            "QPushButton:checked { background-color: #2979ff; color: white; }"
        )
        self.btn_shuffle.setToolTip(
            "When enabled, Space/Enter advances to a\n"
            "random unlabeled/inferred frame instead of\n"
            "the next one in list order."
        )
        self.btn_shuffle.toggled.connect(self._on_shuffle_toggled)
        nav_row.addWidget(self.btn_shuffle)
        vbox.addLayout(nav_row)

        delete_row = QHBoxLayout()
        delete_row.setSpacing(4)
        self.btn_delete_unlabeled = QPushButton("Delete Unlabeled Frames")
        self.btn_delete_unlabeled.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; font-weight: bold; "
            "padding: 3px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #e53935; }"
        )
        self.btn_delete_unlabeled.setToolTip(
            "Remove all frames that have no annotations\n"
            "(no manual or inferred masks) from the project."
        )
        self.btn_delete_unlabeled.clicked.connect(self._delete_unlabeled_frames)
        delete_row.addWidget(self.btn_delete_unlabeled)

        self.btn_delete_frame = QPushButton("Delete Frame")
        self.btn_delete_frame.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; font-weight: bold; "
            "padding: 3px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #e53935; }"
        )
        self.btn_delete_frame.setToolTip(
            "Delete the current frame from the project.\n"
            "The image file and its annotations will be\n"
            "permanently removed."
        )
        self.btn_delete_frame.clicked.connect(self._delete_current_frame)
        delete_row.addWidget(self.btn_delete_frame)
        vbox.addLayout(delete_row)

        self.btn_clear_inferences = QPushButton("Clear Inferences")
        self.btn_clear_inferences.setStyleSheet(
            "QPushButton { background-color: #e65100; color: white; font-weight: bold; "
            "padding: 3px 12px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #f57c00; }"
        )
        self.btn_clear_inferences.setToolTip(
            "Remove all inferred (AI-generated) annotations.\n"
            "Manually approved annotations are kept."
        )
        self.btn_clear_inferences.clicked.connect(self._clear_all_inferences)
        vbox.addWidget(self.btn_clear_inferences)

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
        self._cls_data_loaded = False
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

    def _migrate_classifier_dir(self):
        old = os.path.join(self._project_dir, "behavior_classifier")
        new = os.path.join(self._project_dir, "action_classifier")
        if not os.path.isdir(old):
            return
        import shutil
        if os.path.isdir(new):
            for item in os.listdir(old):
                src = os.path.join(old, item)
                dst = os.path.join(new, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                elif os.path.isdir(src):
                    for sub in os.listdir(src):
                        s2 = os.path.join(src, sub)
                        d2 = os.path.join(dst, sub)
                        if not os.path.exists(d2):
                            shutil.move(s2, d2)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.rename(old, new)
        print(f"[MTT] Migrated behavior_classifier -> action_classifier")

    def _load_project(self, config_path: str):
        with open(config_path) as f:
            cfg = json.load(f)
        self._project_dir = os.path.dirname(config_path)
        self._migrate_classifier_dir()
        self._cls_data_loaded = False
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
                self._select_frame_row(0)

        ann_path = os.path.join(self._project_dir, "annotations", "annotations.json")
        if os.path.exists(ann_path):
            self._coco.load(ann_path)
            self._categories = [c["name"] for c in self._coco.categories]
        self._coco.auto_save_path = ann_path

        self._rebuild_frame_confidence()
        self._refresh_frame_list()

        self.setWindowTitle(f"{self.BASE_TITLE} — {os.path.basename(self._project_dir)}")
        self.status_bar.showMessage(f"Opened project: {self._project_dir}")
        self._update_ann_stats()

        # Refresh whichever tab is currently visible
        current_tab_idx = self.tab_widget.currentIndex()
        if current_tab_idx == 0:  # Annotate (includes training section)
            self._refresh_training_summary()
            self._refresh_device_label()
        elif current_tab_idx == 1:  # Classify
            self._refresh_classifier_models()
        elif current_tab_idx == 2:  # Infer
            self._refresh_model_list()
            self._refresh_classifier_model_list()

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
        self._annot_in_video_mode = True

        # Deselect extracted frames — video and frame preview are mutually exclusive
        self.frame_list.blockSignals(True)
        self.frame_list.clearSelection()
        self._select_frame_row(-1)
        self.frame_list.blockSignals(False)

        self.preview.annotations.clear()
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
        self._video_frame_idx = 0

    def _show_video_frame(self, frame_idx: int):
        if self._video_cap is None:
            return
        frame_idx = max(0, min(frame_idx, self._video_frame_count - 1))
        self._video_frame_idx = frame_idx
        self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._video_cap.read()
        if ret:
            if self.preview.annotations:
                self.preview.annotations.clear()
            self.preview.set_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.lbl_frame_info.setText(
                f"Frame: {frame_idx + 1} / {self._video_frame_count}"
            )

    def _extract_frame_from_video(self):
        if self._video_cap is None or self.current_video_idx < 0:
            return
        if self._project_dir is None:
            QMessageBox.information(
                self, "Save Project First",
                "Please save a project before extracting frames.",
            )
            self._new_project()
            if self._project_dir is None:
                return
        if self._output_dir is None:
            self._output_dir = os.path.join(self._project_dir, "training_frames")
            self.lbl_output_dir.setText(self._output_dir)
            self.lbl_output_dir.setStyleSheet("color: #cccccc; font-size: 10px;")
        os.makedirs(self._output_dir, exist_ok=True)

        vpath = self.video_paths[self.current_video_idx]
        fidx = self._video_frame_idx
        stem = Path(vpath).stem
        fname = f"{stem}_frame_{fidx:06d}.png"
        out_path = os.path.join(self._output_dir, fname)

        if os.path.exists(out_path):
            self.status_bar.showMessage(f"Frame already extracted: {fname}", 3000)
        else:
            self._video_cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = self._video_cap.read()
            if not ret:
                self.status_bar.showMessage("Failed to read frame from video", 3000)
                return
            cv2.imwrite(out_path, frame)

        existing_paths = {fp for _, _, fp in self._extracted_frames}
        if out_path not in existing_paths:
            self._extracted_frames.append((vpath, fidx, out_path))
            self._extracted_frames.sort(key=lambda x: os.path.basename(x[2]))
            self._refresh_frame_list()

        # Select the extracted frame, switching to frame mode
        for i, (_, _, fp) in enumerate(self._extracted_frames):
            if fp == out_path:
                self._select_frame_row(i)
                break

        self.status_bar.showMessage(f"Extracted frame: {fname}", 3000)

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

        # Merge new frames with existing ones (append, don't replace)
        existing_paths = {fp for _, _, fp in self._extracted_frames}
        new_count = 0
        for item in results:
            if item[2] not in existing_paths:
                self._extracted_frames.append(item)
                new_count += 1

        # Sort by filename for consistent ordering
        self._extracted_frames.sort(key=lambda x: os.path.basename(x[2]))
        self._refresh_frame_list()
        total = len(self._extracted_frames)
        self.status_bar.showMessage(
            f"Added {new_count} new frames ({total} total) in {self._output_dir}"
        )
        if self._extracted_frames and self.current_frame_idx < 0:
            self._select_frame_row(0)

        QMessageBox.information(
            self, "Frames Ready",
            f"{new_count} frames extracted ({total} total).\n\n"
            f"Right-click the preview to start labeling objects.\n"
            f"AI-assisted labeling is recommended — left-click to "
            f"include regions, right-click to exclude, Enter to accept.",
        )

    def _on_extract_error(self, msg: str):
        self.extract_progress.setVisible(False)
        self.btn_generate.setEnabled(True)
        QMessageBox.critical(self, "Extraction Error", msg)

    # ==================================================================
    # Extracted frames
    # ==================================================================
    class _NumericTreeItem(QTreeWidgetItem):
        """QTreeWidgetItem that sorts the confidence column numerically."""
        def __lt__(self, other):
            col = self.treeWidget().sortColumn() if self.treeWidget() else 0
            if col == 1:
                a = self.data(1, Qt.UserRole)
                b = other.data(1, Qt.UserRole)
                if a is None:
                    a = -1.0
                if b is None:
                    b = -1.0
                return a < b
            return super().__lt__(other)

    def _refresh_frame_list(self):
        self.frame_list.blockSignals(True)
        self.frame_list.setSortingEnabled(False)
        self.frame_list.clear()
        for idx, (_, _, fp) in enumerate(self._extracted_frames):
            filename = os.path.basename(fp)
            n_total, n_inferred = self._count_annotations_for_file(filename)
            n_approved = n_total - n_inferred
            if n_total > 0 and n_inferred == n_total:
                label = f"● {filename} ({n_total})"
                color = QColor("#e6c830")
            elif n_total > 0 and n_inferred > 0:
                label = f"◐ {filename} ({n_approved}+{n_inferred})"
                color = QColor("#e6c830")
            elif n_total > 0:
                label = f"✔ {filename} ({n_total})"
                color = QColor("#4fc456")
            else:
                label = f"   {filename}"
                color = QColor("#cccccc")

            conf = self._frame_confidence.get(filename)
            conf_text = f"{conf:.2f}" if conf is not None else ""

            item = self._NumericTreeItem([label, conf_text])
            item.setData(0, Qt.UserRole, idx)
            item.setData(1, Qt.UserRole, conf if conf is not None else -1.0)
            item.setForeground(0, color)
            if conf is not None:
                if conf < 0.3:
                    item.setForeground(1, QColor("#e53935"))
                elif conf < 0.6:
                    item.setForeground(1, QColor("#e6c830"))
                else:
                    item.setForeground(1, QColor("#4fc456"))
            self.frame_list.addTopLevelItem(item)
        self.frame_list.setSortingEnabled(True)
        self.frame_list.blockSignals(False)
        self._update_nav_state()

    def _recalc_current_frame_confidence(self):
        """Recalculate confidence for the current frame after approval/deletion."""
        if self.current_frame_idx < 0:
            return
        _, _, fp = self._extracted_frames[self.current_frame_idx]
        filename = os.path.basename(fp)
        if filename not in self._coco._image_id_map:
            self._frame_confidence.pop(filename, None)
            self._update_frame_list_item(self.current_frame_idx)
            return
        img_id = self._coco._image_id_map[filename]
        max_score = 0.0
        has_inferred = False
        for ann in self._coco.get_annotations_for_image(img_id):
            if ann.get("inferred", False):
                s = ann.get("score", 0.0)
                if s > max_score:
                    max_score = s
                has_inferred = True
        if has_inferred:
            self._frame_confidence[filename] = max_score
        else:
            self._frame_confidence.pop(filename, None)
        self._update_frame_list_item(self.current_frame_idx)

    def _rebuild_frame_confidence(self):
        """Rebuild _frame_confidence dict from stored annotation scores."""
        self._frame_confidence.clear()
        img_id_to_name = {img["id"]: img["file_name"] for img in self._coco.images}
        for ann in self._coco.annotations:
            if not ann.get("inferred", False):
                continue
            score = ann.get("score")
            if score is None:
                continue
            fname = img_id_to_name.get(ann["image_id"], "")
            if fname:
                cur = self._frame_confidence.get(fname, 0.0)
                if score > cur:
                    self._frame_confidence[fname] = score

    def _count_annotations_for_file(self, filename: str) -> Tuple[int, int]:
        """Return (total_annotations, inferred_count) for a file."""
        if filename not in self._coco._image_id_map:
            return (0, 0)
        img_id = self._coco._image_id_map[filename]
        anns = self._coco.get_annotations_for_image(img_id)
        total = len(anns)
        n_inferred = sum(1 for a in anns if a.get("inferred", False))
        return (total, n_inferred)

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
            self._select_frame_row(0)
        self.status_bar.showMessage(f"Loaded {len(self._extracted_frames)} frames from {folder}")

    def _on_frame_item_changed(self, current, _previous):
        """Handle QTreeWidget currentItemChanged signal."""
        if current is None:
            return
        row = current.data(0, Qt.UserRole)
        if row is None or row < 0 or row >= len(self._extracted_frames):
            return
        self._dismiss_training_viz()
        self.current_frame_idx = row
        self._annot_in_video_mode = False

        self.video_list.blockSignals(True)
        self.video_list.clearSelection()
        self.video_list.setCurrentRow(-1)
        self.video_list.blockSignals(False)

        _, _, img_path = self._extracted_frames[row]
        self._load_annotation_frame(img_path)
        self._update_nav_state()

    def _select_frame_row(self, row: int):
        """Select the QTreeWidget item whose stored frame index == row."""
        if row < 0:
            self.frame_list.setCurrentItem(None)
            return
        for i in range(self.frame_list.topLevelItemCount()):
            item = self.frame_list.topLevelItem(i)
            if item.data(0, Qt.UserRole) == row:
                self.frame_list.setCurrentItem(item)
                return

    def _on_shuffle_toggled(self, checked: bool):
        self._shuffle_mode = checked

    def _prev_frame(self):
        if self.current_frame_idx > 0:
            self._select_frame_row(self.current_frame_idx - 1)

    def _on_advance_frame(self):
        if self._annot_in_video_mode and self._video_cap is not None:
            new_idx = self._video_frame_idx + 1
            if new_idx < self._video_frame_count:
                self._show_video_frame(new_idx)
        else:
            self._next_frame()

    def _next_frame(self):
        import random
        n = len(self._extracted_frames)
        if n == 0:
            return

        # Collect all frames that need review: unlabeled or inferred-only
        candidates = []
        for i in range(n):
            if i == self.current_frame_idx:
                continue
            _, _, fp = self._extracted_frames[i]
            filename = os.path.basename(fp)
            total, n_inferred = self._count_annotations_for_file(filename)
            if total == 0 or total == n_inferred:
                candidates.append(i)

        if not candidates:
            if self.current_frame_idx < n - 1:
                self._select_frame_row(self.current_frame_idx + 1)
            return

        if self._shuffle_mode:
            self._select_frame_row(random.choice(candidates))
        else:
            forward = [i for i in candidates if i > self.current_frame_idx]
            if forward:
                self._select_frame_row(forward[0])
            else:
                self._select_frame_row(candidates[0])

    def _delete_current_frame(self):
        if self.current_frame_idx < 0 or self.current_frame_idx >= len(self._extracted_frames):
            return
        _, _, fp = self._extracted_frames[self.current_frame_idx]
        filename = os.path.basename(fp)

        # Check if frame has annotations
        n_total, _ = self._count_annotations_for_file(filename)
        msg = f"Delete training frame '{filename}' from project folder?"
        if n_total > 0:
            msg += f"\n\nThis frame has {n_total} annotation(s) that will also be removed."
        msg += "\n\nThis cannot be undone."

        reply = QMessageBox.warning(
            self, "Delete Training Frame", msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Remove annotations and image record for this frame
        if filename in self._coco._image_id_map:
            img_id = self._coco._image_id_map[filename]
            anns = self._coco.get_annotations_for_image(img_id)
            for ann in anns:
                self._coco.remove_annotation(ann["id"])
            self._coco.images = [
                img for img in self._coco.images if img["id"] != img_id
            ]
            del self._coco._image_id_map[filename]
            self._coco._auto_save()

        # Delete file from disk
        try:
            if os.path.exists(fp):
                os.remove(fp)
        except OSError as e:
            QMessageBox.warning(self, "Error", f"Could not delete file:\n{e}")

        # Remove from extracted frames list
        self._extracted_frames.pop(self.current_frame_idx)

        # Adjust current index
        if len(self._extracted_frames) == 0:
            self.current_frame_idx = -1
            self.preview.clear()
        elif self.current_frame_idx >= len(self._extracted_frames):
            self.current_frame_idx = len(self._extracted_frames) - 1

        self._refresh_frame_list()
        if self.current_frame_idx >= 0:
            self._select_frame_row(self.current_frame_idx)
        self._update_ann_stats()
        self.status_bar.showMessage(f"Deleted frame: {filename}")

    def _delete_unlabeled_frames(self):
        """Delete all frames that have no annotations (manual or inferred)."""
        to_delete = []
        for idx, (_, _, fp) in enumerate(self._extracted_frames):
            filename = os.path.basename(fp)
            total, _ = self._count_annotations_for_file(filename)
            if total == 0:
                to_delete.append((idx, fp, filename))

        if not to_delete:
            QMessageBox.information(
                self, "No Unlabeled Frames",
                "All frames in the queue have at least one annotation.",
            )
            return

        reply = QMessageBox.warning(
            self, "Delete Unlabeled Frames",
            f"Delete {len(to_delete)} unlabeled frame(s) from the project folder?\n\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        for _, fp, filename in to_delete:
            if filename in self._coco._image_id_map:
                img_id = self._coco._image_id_map[filename]
                self._coco.images = [
                    img for img in self._coco.images if img["id"] != img_id
                ]
                del self._coco._image_id_map[filename]
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except OSError:
                pass

        deleted_indices = {idx for idx, _, _ in to_delete}
        self._extracted_frames = [
            f for i, f in enumerate(self._extracted_frames)
            if i not in deleted_indices
        ]
        self._coco._auto_save()

        if len(self._extracted_frames) == 0:
            self.current_frame_idx = -1
            self.preview.clear()
        elif self.current_frame_idx >= len(self._extracted_frames):
            self.current_frame_idx = len(self._extracted_frames) - 1

        self._refresh_frame_list()
        if self.current_frame_idx >= 0:
            self._select_frame_row(self.current_frame_idx)
        self._update_ann_stats()
        self.status_bar.showMessage(
            f"Deleted {len(to_delete)} unlabeled frame(s)."
        )

    def _clear_all_inferences(self):
        """Remove all inferred annotations across all frames."""
        inferred_anns = [a for a in self._coco.annotations if a.get("inferred", False)]
        if not inferred_anns:
            QMessageBox.information(
                self, "No Inferences",
                "There are no inferred annotations to remove.",
            )
            return

        inferred_image_ids = set(a["image_id"] for a in inferred_anns)
        n_frames = len(inferred_image_ids)

        reply = QMessageBox.warning(
            self, "Clear Inferences",
            f"Remove {len(inferred_anns)} inferred annotation(s) "
            f"from {n_frames} frame(s)?\n\n"
            "Manually approved annotations will not be affected.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        ids_to_remove = [a["id"] for a in inferred_anns]
        for ann_id in ids_to_remove:
            self._coco.remove_annotation(ann_id)

        self._frame_confidence.clear()

        self._refresh_frame_list()
        if self.current_frame_idx >= 0:
            _, _, path = self._extracted_frames[self.current_frame_idx]
            self._load_frame_annotations(path)
        self._update_ann_stats()
        self.status_bar.showMessage(
            f"Cleared {len(ids_to_remove)} inferred annotation(s) from {n_frames} frame(s)."
        )

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
            is_inferred = ann.get("inferred", False)
            ann_obj = AnnotationObject(
                points, cat_name, ann_id=ann["id"], inferred=is_inferred,
            )
            self.preview.annotations.append(ann_obj)

        self.preview.update()
        self._update_ann_stats()

    # ==================================================================
    # Mode changes
    # ==================================================================
    def _on_mode_changed(self, mode_name: str):
        if mode_name == "AI-Assisted Mask":
            self.lbl_mode.setText("AI Mask — Enter=accept, Esc=reject")
        else:
            self.lbl_mode.setText(mode_name)
        if self.preview.drawing_mode == "ai":
            self._ai_enabled = True
            self._ensure_sam2_loaded()
            # Sync toggle button without retriggering
            self.btn_sam_toggle.blockSignals(True)
            self.btn_sam_toggle.setChecked(True)
            self.btn_sam_toggle.setStyleSheet(self._sam_toggle_on_style)
            self.btn_sam_toggle.blockSignals(False)
        else:
            self._ai_enabled = self.preview.drawing_mode == "ai"
            self.btn_sam_toggle.blockSignals(True)
            self.btn_sam_toggle.setChecked(False)
            self.btn_sam_toggle.setStyleSheet(self._sam_toggle_off_style)
            self.btn_sam_toggle.blockSignals(False)

    def _on_sam_toggle(self, checked: bool):
        """Toggle SAM2 AI labeling mode on/off via the button."""
        if checked:
            self.preview.drawing_mode = "ai"
            self.preview.mode_changed.emit("AI-Assisted Mask")
        else:
            # Cancel any in-progress AI drawing
            self.preview._clear_drawing()
            self.preview.drawing_mode = "navigate"
            self.lbl_mode.setText("Navigate")
            self._ai_enabled = False
            self.btn_sam_toggle.setStyleSheet(self._sam_toggle_off_style)

    def _ensure_sam2_loaded(self):
        if self._segmenter is not None and self._segmenter.predictor is not None:
            return
        from .sam2_checkpoint_manager import SAM2_CHECKPOINTS, LOCAL_MODELS_DIR, _LEGACY_DIRS, _find_existing_checkpoints

        search_dirs = [LOCAL_MODELS_DIR] + list(_LEGACY_DIRS)
        custom_path = self._project_config.get("sam2_model_path")
        if custom_path and os.path.isdir(custom_path):
            search_dirs.insert(0, Path(custom_path))
        found = _find_existing_checkpoints(*search_dirs)

        if found:
            names = list(found.keys())
            descriptions = [f"{n} ({SAM2_CHECKPOINTS.get(n, {}).get('size_mb', '?')} MB)" for n in names]
            descriptions.append("Download additional model...")
            choice, ok = QInputDialog.getItem(
                self, "Select SAM2 Model", "Found existing SAM2 model(s). Select one:",
                descriptions, 0, False,
            )
            if ok:
                if choice == "Download additional model...":
                    self._show_sam2_download_dialog()
                else:
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
        self.status_bar.showMessage(
            f"SAM2 prediction (score={score:.3f}) — Enter to accept, Escape to reject"
        )

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

    def _on_approve_annotation_by_index(self, obj_idx: int):
        if obj_idx < 0 or obj_idx >= len(self.preview.annotations):
            return
        ann = self.preview.annotations[obj_idx]
        if ann.ann_id >= 0 and ann.inferred:
            self._coco.approve_annotation(ann.ann_id)
            ann.inferred = False
            self.preview.update()
            self._recalc_current_frame_confidence()
            self._update_ann_stats()
            self.status_bar.showMessage(f"Annotation {ann.ann_id} approved")

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
    def _populate_hw_info(self):
        try:
            from .mask_tracker_inference import detect_system_profile
            p = detect_system_profile()
            lines = []
            lines.append(f"Chip: {p.get('chip', 'Unknown')}")
            lines.append(f"CPU cores: {p.get('cpu_cores', '?')}  |  RAM: {p.get('ram_gb', '?')} GB")
            gpu_parts = []
            if p.get("has_cuda"):
                gpu_parts.append(f"CUDA ({p.get('cuda_name', 'Unknown')})")
            if p.get("has_mps"):
                gpu_parts.append("MPS (Apple Metal)")
            if not gpu_parts:
                gpu_parts.append("None (CPU only)")
            lines.append(f"GPU: {', '.join(gpu_parts)}")
            lines.append(f"Recommended: {p['recommended_device'].upper()}, "
                         f"{p['recommended_resolution']}px, "
                         f"{'masks' if p['recommended_use_masks'] else 'boxes'}")
            self.lbl_hw_info.setText("\n".join(lines))
        except Exception as e:
            self.lbl_hw_info.setText(f"Hardware detection failed: {e}")

    def _populate_cls_system_info(self):
        try:
            from .mask_tracker_inference import detect_system_profile
            p = detect_system_profile()
            parts = [p.get("chip", "Unknown")]
            if p.get("has_cuda"):
                parts.append(f"CUDA ({p.get('cuda_name', '?')})")
            elif p.get("has_mps"):
                parts.append("MPS (Apple Metal)")
            else:
                parts.append("CPU only")
            parts.append(f"{p.get('ram_gb', '?')} GB RAM")
            self.lbl_cls_system_info.setText(f"System: {' · '.join(parts)}")
        except Exception:
            self.lbl_cls_system_info.setText("System: detection failed")

    def _on_architecture_changed(self, index: int):
        arch_text = self.combo_architecture.currentText()
        is_maskrcnn = arch_text.startswith("Mask R-CNN")
        for w in self._maskrcnn_widgets:
            w.setVisible(is_maskrcnn)
        if not is_maskrcnn:
            is_small = "small" in arch_text.lower()
            self.spin_batch.setValue(2 if is_small else 8)
            self.spin_lr.setValue(0.01)
        else:
            self.spin_batch.setValue(2)
            self.spin_lr.setValue(0.005)
            self.combo_device.setToolTip(
                "Hardware device for training.\n"
                "Auto: uses CUDA if available, otherwise best available.\n"
                "CPU: slower but always works.\n"
                "CUDA: NVIDIA GPU — fastest.\n"
                "MPS: Apple Silicon — fast for Mask R-CNN."
            )
        if self.chk_augment.isChecked():
            self._update_aug_expansion_label()

    def _update_aug_expansion_label(self):
        parts = []
        rot_idx = self.combo_aug_rotation.currentIndex()
        if rot_idx == 1:
            parts.append("rot ±15°")
        elif rot_idx == 2:
            parts.append("rot ±180°")
        if self.chk_aug_scale.isChecked():
            lo = self.spin_aug_scale_min.value()
            hi = self.spin_aug_scale_max.value()
            parts.append(f"scale {lo:.2f}–{hi:.2f}×")
        if self.chk_aug_brightness.isChecked():
            parts.append("brightness")
        if self.chk_aug_contrast.isChecked():
            parts.append("contrast")
        if parts:
            self._aug_expansion_label.setText(
                f"  Online: {', '.join(parts)} (applied per batch)"
            )
        else:
            self._aug_expansion_label.setText("  No augmentations selected")

    def _build_aug_config(self):
        from .mask_tracker_augmentation import AugmentationConfig
        cfg = AugmentationConfig()
        rot_idx = self.combo_aug_rotation.currentIndex()
        if rot_idx == 0:
            cfg.horizontal_flip = False
            cfg.vertical_flip = False
            cfg.rotate_90 = False
            cfg.rotate_180 = False
            cfg.rotate_270 = False
            cfg.continuous_rotation = False
        elif rot_idx == 1:
            cfg.horizontal_flip = True
            cfg.vertical_flip = False
            cfg.rotate_90 = False
            cfg.rotate_180 = False
            cfg.rotate_270 = False
            cfg.continuous_rotation = True
            cfg.rotation_max_angle = 15.0
        else:
            cfg.horizontal_flip = True
            cfg.vertical_flip = True
            cfg.rotate_90 = True
            cfg.rotate_180 = True
            cfg.rotate_270 = True
            cfg.continuous_rotation = True
            cfg.rotation_max_angle = 180.0
        cfg.random_scale = self.chk_aug_scale.isChecked()
        cfg.scale_range = (
            self.spin_aug_scale_min.value(),
            self.spin_aug_scale_max.value(),
        )
        cfg.brightness = self.chk_aug_brightness.isChecked()
        cfg.contrast = self.chk_aug_contrast.isChecked()
        cfg.gaussian_noise = False
        cfg.gaussian_blur = False
        return cfg

    def _start_training(self):
        stats = self._coco.get_stats()
        if stats["num_annotations"] == 0:
            QMessageBox.warning(self, "No Annotations", "Annotate some frames first.")
            return
        if not self._project_dir:
            QMessageBox.warning(self, "No Project", "Save a project first.")
            return

        # Export training annotations, excluding inferred (unapproved) masks
        ann_path = os.path.join(self._project_dir, "annotations", "annotations_train.json")
        self._coco.export(ann_path, exclude_inferred=True)

        # Check that we have actual training data after excluding inferred
        n_approved = sum(
            1 for a in self._coco.annotations if not a.get("inferred", False)
        )
        if n_approved == 0:
            QMessageBox.warning(
                self, "No Approved Annotations",
                "All annotations are inferred (pending review).\n"
                "Approve some annotations first, then retrain.",
            )
            return

        images_dir = self._output_dir or os.path.join(self._project_dir, "training_frames")

        n_train_images = len(set(
            a["image_id"] for a in self._coco.annotations
            if not a.get("inferred", False)
        ))

        arch_text = self.combo_architecture.currentText()
        is_yolo = arch_text.startswith("YOLO")
        use_aug = self.chk_augment.isChecked()

        if use_aug and not is_yolo:
            # Mask R-CNN: offline augmentation (write augmented images to disk)
            self.btn_train.setEnabled(False)
            self.lbl_train_status.setText("Augmenting dataset...")
            self.status_bar.showMessage("Running data augmentation...")
            aug_dir = os.path.join(os.path.dirname(images_dir), "augmented")
            self._pending_train_config = {
                "images_dir": images_dir,
                "ann_path": ann_path,
                "aug_dir": aug_dir,
                "n_train_images": n_train_images,
            }
            aug_config = self._build_aug_config()
            self._augment_worker = AugmentWorker(ann_path, images_dir, aug_dir, aug_config)
            self._augment_worker.progress.connect(
                lambda c, t: self.lbl_train_status.setText(f"Augmenting... {c}/{t} images")
            )
            self._augment_worker.finished.connect(self._on_augment_then_train)
            self._augment_worker.error.connect(self._on_train_error)
            self._augment_worker.start()
        else:
            # YOLO: augmentation is applied online (no disk writes)
            self._launch_training(ann_path, images_dir, n_train_images)

    def _on_augment_then_train(self, aug_output_path: str):
        aug_dir = self._pending_train_config["aug_dir"]
        aug_json = os.path.join(aug_dir, "annotations.json")
        n_img = self._pending_train_config.get("n_train_images", 0)
        if os.path.exists(aug_json):
            self._launch_training(aug_json, aug_dir, n_img)
        else:
            self._launch_training(
                self._pending_train_config["ann_path"],
                self._pending_train_config["images_dir"],
                n_img,
            )

    def _launch_training(self, coco_json: str, images_dir: str,
                          n_train_images: int = 0):
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

        arch_text = self.combo_architecture.currentText()
        is_yolo = arch_text.startswith("YOLO")

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if is_yolo:
            arch_lower = arch_text.lower()
            if "large" in arch_lower:
                model_variant = "yolo11l-seg"
            elif "medium" in arch_lower:
                model_variant = "yolo11m-seg"
            elif "small" in arch_lower:
                model_variant = "yolo11s-seg"
            else:
                model_variant = "yolo11n-seg"
            optimizer = self.combo_optimizer.currentText()
            run_name = f"{timestamp}_yolo_n={n_train_images}"

            # Map augmentation config to Ultralytics online params
            aug_fliplr = 0.0
            aug_flipud = 0.0
            aug_degrees = 0.0
            aug_scale = 0.0
            aug_hsv_v = 0.0
            aug_hsv_s = 0.0
            if self.chk_augment.isChecked():
                rot_idx = self.combo_aug_rotation.currentIndex()
                if rot_idx == 1:
                    aug_fliplr = 0.5
                    aug_degrees = 15.0
                elif rot_idx == 2:
                    aug_fliplr = 0.5
                    aug_flipud = 0.5
                    aug_degrees = 180.0
                if self.chk_aug_scale.isChecked():
                    s_lo = self.spin_aug_scale_min.value()
                    s_hi = self.spin_aug_scale_max.value()
                    aug_scale = max(1.0 - s_lo, s_hi - 1.0)
                if self.chk_aug_brightness.isChecked():
                    b_pct = max(abs(self.spin_aug_bright_min.value()),
                                abs(self.spin_aug_bright_max.value()))
                    aug_hsv_v = b_pct / 100.0
                if self.chk_aug_contrast.isChecked():
                    c_pct = max(abs(self.spin_aug_contrast_min.value()),
                                abs(self.spin_aug_contrast_max.value()))
                    aug_hsv_s = c_pct / 100.0

            config_dict = {
                "coco_json_path": coco_json,
                "images_dir": images_dir,
                "output_dir": os.path.join(self._project_dir, "models"),
                "run_name": run_name,
                "num_iterations": self.spin_iterations.value(),
                "learning_rate": self.spin_lr.value(),
                "batch_size": self.spin_batch.value(),
                "val_fraction": self.spin_val_frac.value(),
                "device": device_choice,
                "model_variant": model_variant,
                "imgsz": 640,
                "early_stop_patience": self.spin_patience.value(),
                "freeze_backbone": self.chk_freeze_backbone.isChecked(),
                "optimizer": optimizer,
                "aug_fliplr": aug_fliplr,
                "aug_flipud": aug_flipud,
                "aug_degrees": aug_degrees,
                "aug_scale": aug_scale,
                "aug_hsv_v": aug_hsv_v,
                "aug_hsv_s": aug_hsv_s,
                "_architecture": "yolo",
            }
        else:
            backbone_text = self.combo_backbone.currentText()
            if "Small" in backbone_text:
                backbone = "mobilenet_v3_small"
            elif "Large" in backbone_text:
                backbone = "mobilenet_v3_large"
            else:
                backbone = "resnet50"
            optimizer = "adamw" if self.combo_optimizer.currentText() == "AdamW" else "sgd"
            run_name = f"{timestamp}_maskrcnn_n={n_train_images}"

            config_dict = {
                "coco_json_path": coco_json,
                "images_dir": images_dir,
                "output_dir": os.path.join(self._project_dir, "models"),
                "run_name": run_name,
                "num_iterations": self.spin_iterations.value(),
                "learning_rate": self.spin_lr.value(),
                "batch_size": self.spin_batch.value(),
                "val_fraction": self.spin_val_frac.value(),
                "min_size": self.spin_min_size.value(),
                "device": device_choice,
                "backbone": backbone,
                "freeze_backbone": self.chk_freeze_backbone.isChecked(),
                "optimizer": optimizer,
                "early_stop_patience": self.spin_patience.value(),
                "_architecture": "maskrcnn",
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

        # Reset embedded loss plot and log for new run
        self._loss_plot.reset(total)
        self._seg_log_text.clear()

        arch = config_dict.get("_architecture", "?")
        variant = config_dict.get("model_variant", config_dict.get("backbone", "?"))

        import platform as _plat
        self._seg_log_text.append(
            f"[System] Python {_plat.python_version()}  |  "
            f"OS: {_plat.system()} {_plat.release()}"
        )
        try:
            import torch as _torch
            self._seg_log_text.append(f"[System] PyTorch {_torch.__version__}")
            if _torch.cuda.is_available():
                gpu = _torch.cuda.get_device_name(0)
                mem = _torch.cuda.get_device_properties(0).total_memory / 1024**3
                self._seg_log_text.append(
                    f"[System] GPU: {gpu} ({mem:.1f} GB)  |  "
                    f"CUDA: {_torch.version.cuda or 'N/A'}"
                )
            elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                self._seg_log_text.append("[System] Apple MPS available")
            else:
                self._seg_log_text.append("[System] No GPU detected — training on CPU")
        except ImportError:
            pass
        try:
            import ultralytics as _ul
            self._seg_log_text.append(f"[System] Ultralytics {_ul.__version__}")
        except Exception:
            pass

        # Resolve actual training device for display
        requested_device = config_dict.get("device", "auto")
        actual_device = requested_device
        try:
            import torch as _torch
            if requested_device == "auto":
                if _torch.cuda.is_available():
                    actual_device = "cuda"
                elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                    actual_device = "mps"
                else:
                    actual_device = "cpu"
        except ImportError:
            pass

        self._seg_log_text.append("")
        self._seg_log_text.append(f"[Segmentation] Architecture: {arch} ({variant})")
        self._seg_log_text.append(
            f"[Segmentation] epochs={total}, lr={config_dict.get('learning_rate', 0):.4f}, "
            f"batch={config_dict.get('batch_size', 0)}, "
            f"device={actual_device}, "
            f"val_split={config_dict.get('val_fraction', 0)}"
        )
        self._seg_log_text.append(
            f"[Segmentation] freeze_backbone={config_dict.get('freeze_backbone', False)}, "
            f"optimizer={config_dict.get('optimizer', '?')}, "
            f"patience={config_dict.get('early_stop_patience', '?')}"
        )
        self._seg_log_text.append(
            f"[Segmentation] Training images: {n_train_images} (approved)"
        )

        # Log augmentation settings
        if arch == "yolo":
            aug_parts = []
            if config_dict.get("aug_fliplr", 0) > 0:
                aug_parts.append(f"fliplr={config_dict['aug_fliplr']}")
            if config_dict.get("aug_flipud", 0) > 0:
                aug_parts.append(f"flipud={config_dict['aug_flipud']}")
            if config_dict.get("aug_degrees", 0) > 0:
                aug_parts.append(f"degrees=±{config_dict['aug_degrees']:.0f}°")
            if config_dict.get("aug_scale", 0) > 0:
                aug_parts.append(f"scale=±{config_dict['aug_scale']:.2f}")
            if config_dict.get("aug_hsv_v", 0) > 0:
                aug_parts.append(f"hsv_v={config_dict['aug_hsv_v']:.1f}")
            if config_dict.get("aug_hsv_s", 0) > 0:
                aug_parts.append(f"hsv_s={config_dict['aug_hsv_s']:.1f}")
            if aug_parts:
                self._seg_log_text.append(
                    f"[Segmentation] Online augmentation: {', '.join(aug_parts)}"
                )
            else:
                self._seg_log_text.append("[Segmentation] Augmentation: disabled")

        self._seg_log_text.append("")

        self._train_worker = TrainWorker(config_dict)
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.start()

        self._training_viz_panel.setVisible(True)
        self.preview.setVisible(False)
        self._info_row.setVisible(False)

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
        self._loss_plot.add_point(iteration, loss, metrics)

        lr = metrics.get("lr", 0)
        best = metrics.get("best_loss", 0)
        parts = [f"Epoch {iteration}/{total}",
                 f"loss={loss:.4f}",
                 f"best={best:.4f}",
                 f"lr={lr:.6f}"]
        for key in ("box_loss", "seg_loss", "cls_loss", "dfl_loss"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")
        for key in ("mAP50(M)", "mAP50(B)", "mAP50-95(M)", "mAP50-95(B)"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")
        if "epoch_time" in metrics:
            parts.append(f"{metrics['epoch_time']:.1f}s/epoch")
        self._seg_log_text.append("  ".join(parts))

    def _dismiss_training_viz(self):
        if self._training_viz_panel.isVisible():
            self._training_viz_panel.setVisible(False)
            self.preview.setVisible(True)
            self._info_row.setVisible(True)

    def _on_train_finished(self, summary: dict):
        self.btn_pause.setVisible(False)
        self.btn_stop.setVisible(False)

        model_path = summary.get("model_path", "")
        run_dir = summary.get("run_dir", "")
        early = summary.get("early_stopped", False)
        iters = summary.get("iterations_completed", "?")
        best = summary.get("best_loss")
        best_str = f"{best:.4f}" if best is not None else "N/A"
        stopped_by_user = hasattr(self, "_train_worker") and self._train_worker._stop_flag

        if stopped_by_user and not early:
            status = f"Training stopped by user at iteration {iters}. Best loss: {best_str}"
        elif early:
            status = f"Training stopped early (plateau) at iteration {iters}. Best loss: {best_str}"
        else:
            status = f"Training complete ({iters} iterations). Best loss: {best_str}"

        self._seg_log_text.append("")
        self._seg_log_text.append(f"[Segmentation] {status}")
        self._seg_log_text.append(f"[Segmentation] Model saved to: {run_dir}")

        # Store summary for the combined dialog after inference
        self._last_train_summary = {
            "status": status,
            "run_dir": run_dir,
            "model_path": model_path,
            "early": early,
            "stopped_by_user": stopped_by_user,
            "iters": iters,
            "best_str": best_str,
        }

        # Trigger post-training inference if enabled, otherwise show dialog now
        if self.chk_post_infer.isChecked() and run_dir:
            self.lbl_train_status.setText("Training done — running inference on unlabeled frames...")
            self._seg_log_text.append("")
            self._run_post_train_inference(run_dir)
        else:
            self.train_progress.setVisible(False)
            self.btn_train.setEnabled(True)
            self.lbl_train_status.setText(status)
            self.status_bar.showMessage(f"Training complete: {model_path}")
            self._show_train_complete_dialog()

    def _run_post_train_inference(self, model_dir: str):
        """Run the trained model on unlabeled extracted frames."""
        if not self._extracted_frames:
            self.lbl_train_status.setText("No extracted frames for post-training inference.")
            return

        # Collect frames without manual (non-inferred) annotations
        unlabeled_paths = []
        for _, _, fp in self._extracted_frames:
            filename = os.path.basename(fp)
            if filename in self._coco._image_id_map:
                img_id = self._coco._image_id_map[filename]
                n_manual = self._coco.count_annotations_for_image(
                    img_id, exclude_inferred=True,
                )
                if n_manual > 0:
                    continue  # has manual annotations — skip
            unlabeled_paths.append(fp)

        if not unlabeled_paths:
            self.lbl_train_status.setText(
                "All frames already have annotations — no inference needed."
            )
            return

        # Remove any existing inferred annotations from these frames
        # (in case of a previous inference run)
        for fp in unlabeled_paths:
            filename = os.path.basename(fp)
            if filename in self._coco._image_id_map:
                img_id = self._coco._image_id_map[filename]
                existing = self._coco.get_annotations_for_image(img_id)
                for ann in existing:
                    if ann.get("inferred", False):
                        self._coco.remove_annotation(ann["id"])

        max_det = self.spin_post_infer_max.value() or 100

        self.btn_train.setEnabled(False)
        self.train_progress.setMaximum(len(unlabeled_paths))
        self.train_progress.setValue(0)
        self.train_progress.setVisible(True)
        confidence = self.spin_post_infer_conf.value()
        self.lbl_train_status.setText(
            f"Running inference on {len(unlabeled_paths)} unlabeled frames..."
        )
        self._seg_log_text.append(
            f"[Inference] {len(unlabeled_paths)} unlabeled frames, "
            f"confidence>={confidence:.2f}, max_det={max_det}"
        )

        self._post_infer_n_unlabeled = len(unlabeled_paths)
        self._post_infer_worker = PostTrainInferenceWorker(
            model_dir=model_dir,
            frame_paths=unlabeled_paths,
            max_det=max_det,
            confidence=confidence,
        )
        self._post_infer_worker.progress.connect(self._on_post_infer_progress)
        self._post_infer_worker.frame_result.connect(self._on_post_infer_frame)
        self._post_infer_worker.finished.connect(self._on_post_infer_finished)
        self._post_infer_worker.error.connect(self._on_post_infer_error)
        self._post_infer_worker.start()

    def _on_post_infer_progress(self, current: int, total: int):
        self.train_progress.setValue(current)
        self.lbl_train_status.setText(
            f"Auto-inferring frame {current}/{total}..."
        )

    def _on_post_infer_frame(self, filename: str, detections: list):
        """Add inferred annotations for a single frame."""
        frame_path = None
        for _, _, fp in self._extracted_frames:
            if os.path.basename(fp) == filename:
                frame_path = fp
                break
        if frame_path is None:
            return

        img = cv2.imread(frame_path)
        if img is None:
            return
        h, w = img.shape[:2]

        img_id = self._coco.get_or_add_image(filename, w, h)

        max_score = 0.0
        for det in detections:
            label = det["category_id"]
            cat_id = None
            coco_cats_by_idx = {}
            for ci, cat in enumerate(self._coco.categories):
                coco_cats_by_idx[ci + 1] = cat["id"]
            cat_id = coco_cats_by_idx.get(label)
            if cat_id is None and self._coco.categories:
                cat_id = self._coco.categories[0]["id"]
            if cat_id is None:
                continue

            score = det.get("score", 0.0)
            if score > max_score:
                max_score = score

            self._coco.add_annotation_from_polygon(
                image_id=img_id,
                category_id=cat_id,
                segmentation=det["segmentation"],
                bbox=det["bbox"],
                area=det["area"],
                inferred=True,
                score=score,
            )

        self._frame_confidence[filename] = max_score

    def _on_post_infer_finished(self, count: int):
        self.train_progress.setVisible(False)
        self.btn_train.setEnabled(True)

        # Count how many of the unlabeled frames got inferred annotations
        n_inferred_frames = 0
        for _, _, fp in self._extracted_frames:
            fn = os.path.basename(fp)
            if fn in self._coco._image_id_map:
                img_id = self._coco._image_id_map[fn]
                for ann in self._coco.get_annotations_for_image(img_id):
                    if ann.get("inferred", False):
                        n_inferred_frames += 1
                        break

        available = getattr(self, "_post_infer_n_unlabeled", len(self._extracted_frames))
        infer_status = (
            f"Post-training inference: {count} masks inferred "
            f"across {n_inferred_frames}/{available} frames"
        )
        self.lbl_train_status.setText(f"{infer_status}. Review inferred masks.")

        # Log inference results
        self._seg_log_text.append(f"[Inference] {infer_status}")
        if count == 0:
            self._seg_log_text.append(
                "[Inference] No detections — model may need more training data "
                "or a lower confidence threshold."
            )

        self._refresh_frame_list()
        if self.current_frame_idx >= 0:
            _, _, path = self._extracted_frames[self.current_frame_idx]
            self._load_frame_annotations(path)

        model_path = getattr(self, "_last_train_summary", {}).get("model_path", "")
        self.status_bar.showMessage(f"Training + inference complete: {model_path}")
        self._show_train_complete_dialog(infer_status)

    def _on_post_infer_error(self, msg: str):
        self.train_progress.setVisible(False)
        self.btn_train.setEnabled(True)
        self.lbl_train_status.setText(f"Post-inference error: {msg}")
        self._seg_log_text.append(f"[Inference] ERROR: {msg}")
        self._show_train_complete_dialog(f"Post-inference error: {msg}")

    def _show_train_complete_dialog(self, infer_info: str = None):
        s = getattr(self, "_last_train_summary", {})
        run_dir = s.get("run_dir", "")
        iters = s.get("iters", "?")
        best_str = s.get("best_str", "N/A")

        if s.get("stopped_by_user"):
            title = "Training Stopped"
        elif s.get("early"):
            title = "Training Complete (Early Stop)"
        else:
            title = "Training Complete"

        lines = [s.get("status", "Training finished.")]
        if infer_info:
            lines.append("")
            lines.append(infer_info)
        lines.append("")
        lines.append(f"Model saved to:\n{run_dir}")

        QMessageBox.information(self, title, "\n".join(lines))

    def _on_train_error(self, msg: str):
        self.train_progress.setVisible(False)
        self.btn_train.setEnabled(True)
        self.btn_pause.setVisible(False)
        self.btn_stop.setVisible(False)
        self._dismiss_training_viz()
        self.lbl_train_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Training Error", msg)

    def _show_training_samples(self, model_dir: str):
        """Run the trained model on a few random frames and show predictions."""
        if not self._extracted_frames:
            return

        # Pick up to 4 random frames
        sample_paths = []
        indices = list(range(len(self._extracted_frames)))
        random.shuffle(indices)
        for i in indices[:4]:
            _, _, fp = self._extracted_frames[i]
            sample_paths.append(fp)

        if not sample_paths:
            return

        try:
            import json as _json
            config_path = os.path.join(model_dir, "training_config.json")
            arch = "maskrcnn"
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = _json.load(f)
                arch = cfg.get("architecture", "maskrcnn")

            if arch == "yolov11-seg":
                from .yolo_inference import YOLOInference
                engine = YOLOInference(model_dir, device="auto", use_masks=True)
            else:
                from .mask_tracker_inference import MaskRCNNInference
                engine = MaskRCNNInference(model_dir, device="auto")
            engine.load_model()

            annotated_images = []
            labels = []
            for fp in sample_paths:
                img = cv2.imread(fp)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = engine.predict(img_rgb, confidence_threshold=0.3)

                # Draw detections on image
                overlay = img_rgb.copy()
                n = len(result.get("scores", []))
                for j in range(n):
                    score = float(result["scores"][j])
                    box = result["boxes"][j]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    color = (41, 121, 255)  # blue

                    # Draw mask contour if available
                    if result.get("masks") is not None and j < len(result["masks"]):
                        mask_u8 = result["masks"][j].astype(np.uint8) * 255
                        contours, _ = cv2.findContours(
                            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(overlay, contours, -1, color, 2)
                    else:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                    # Score label
                    cv2.putText(
                        overlay, f"{score:.2f}", (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    )

                annotated_images.append(overlay)
                labels.append(
                    f"{os.path.basename(fp)} — {n} det"
                )

            if annotated_images:
                self._training_sample_preview.set_images(annotated_images, labels)

        except Exception as e:
            print(f"[MTT] Sample preview error: {e}")

    # ==================================================================
    # Classifier
    # ==================================================================
    def _refresh_classifier_models(self):
        self.combo_cls_model.clear()
        self._cls_model_dirs = []
        if not self._project_dir:
            self.lbl_cls_model_info.setText("No project open")
            return
        models_dir = os.path.join(self._project_dir, "models")
        if not os.path.isdir(models_dir):
            self.lbl_cls_model_info.setText("No models found")
            return
        for entry in sorted(os.listdir(models_dir)):
            entry_path = os.path.join(models_dir, entry)
            config_path = os.path.join(entry_path, "training_config.json")
            if os.path.isdir(entry_path) and os.path.exists(config_path):
                self._cls_model_dirs.append(entry_path)
                label = entry
                try:
                    with open(config_path) as f:
                        cfg = json.load(f)
                    arch = cfg.get("architecture", "maskrcnn")
                    cats = cfg.get("categories", {})
                    cat_str = ", ".join(cats.values()) if cats else "unknown"
                    label = f"{entry}  [{arch}: {cat_str}]"
                except Exception:
                    pass
                self.combo_cls_model.addItem(label)
        if not self._cls_model_dirs:
            self.lbl_cls_model_info.setText("No trained models found — train one first")
        else:
            last_idx = len(self._cls_model_dirs) - 1
            self.combo_cls_model.setCurrentIndex(last_idx)
            self.lbl_cls_model_info.setText(
                f"{len(self._cls_model_dirs)} model(s) — using latest"
            )
        self._update_classifier_state()

    def _update_classifier_state(self):
        has_model = len(getattr(self, "_cls_model_dirs", [])) > 0
        has_video = getattr(self, "_cls_video_cap", None) is not None
        self.btn_batch_extract.setEnabled(has_model and has_video)
        self.btn_train_classifier.setEnabled(False)

    def _show_behavior_categories_popup(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Behavior Categories")
        dlg.setMinimumWidth(320)
        dlg.setStyleSheet(
            "QDialog { background-color: #2b2b2b; color: #cccccc; }"
            "QListWidget { background-color: #1e1e1e; border: 1px solid #444; color: #cccccc; }"
            "QListWidget::item:selected { background-color: #7b1fa2; color: white; }"
            "QLineEdit { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 4px; color: #cccccc; }"
            "QPushButton { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 5px 10px; color: #cccccc; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
        )
        layout = QVBoxLayout(dlg)

        info = QLabel(
            "Define behaviors to classify. Right-click a mask\n"
            "in the preview to assign a behavior."
        )
        info.setStyleSheet("color: #888888; font-size: 9px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Preset loadouts
        preset_row = QHBoxLayout()
        lbl_preset = QLabel("Presets:")
        lbl_preset.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        preset_row.addWidget(lbl_preset)
        combo_preset = QComboBox()
        combo_preset.addItems(list(BEHAVIOR_PRESETS.keys()))
        combo_preset.setStyleSheet(
            "QComboBox { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 3px; color: #cccccc; }"
        )
        combo_preset.setToolTip(
            "Load a preset set of behavior categories.\n"
            "This will replace any existing categories."
        )
        preset_row.addWidget(combo_preset, 1)
        btn_load_preset = QPushButton("Load")
        btn_load_preset.setToolTip("Replace current categories with the selected preset.")
        preset_row.addWidget(btn_load_preset)
        layout.addLayout(preset_row)

        cat_list = QListWidget()
        for i in range(self.list_cls_categories.count()):
            src = self.list_cls_categories.item(i)
            item = QListWidgetItem(src.text())
            item.setForeground(src.foreground())
            item.setData(Qt.UserRole, src.data(Qt.UserRole))
            cat_list.addItem(item)
        layout.addWidget(cat_list)

        add_row = QHBoxLayout()
        txt_new = QLineEdit()
        txt_new.setPlaceholderText("New behavior name...")
        add_row.addWidget(txt_new, 1)
        btn_add = QPushButton("Add")
        add_row.addWidget(btn_add)
        layout.addLayout(add_row)

        colors = ["#e53935", "#43a047", "#1e88e5", "#fb8c00",
                  "#8e24aa", "#00acc1", "#ffb300", "#6d4c41"]

        def _add():
            name = txt_new.text().strip()
            if not name:
                return
            for j in range(cat_list.count()):
                if cat_list.item(j).data(Qt.UserRole)["name"] == name:
                    return
            color = colors[cat_list.count() % len(colors)]
            item = QListWidgetItem(f"● {name}")
            item.setForeground(QColor(color))
            item.setData(Qt.UserRole, {"name": name, "color": color})
            cat_list.addItem(item)
            txt_new.clear()

        btn_add.clicked.connect(_add)
        txt_new.returnPressed.connect(_add)

        def _load_preset():
            preset_name = combo_preset.currentText()
            behaviors = BEHAVIOR_PRESETS.get(preset_name, [])
            if not behaviors:
                return
            if cat_list.count() > 0:
                reply = QMessageBox.question(
                    dlg, "Replace Categories?",
                    f"Replace current categories with '{preset_name}' preset?\n\n"
                    f"Behaviors: {', '.join(behaviors)}",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
                )
                if reply != QMessageBox.Yes:
                    return
            cat_list.clear()
            for i, name in enumerate(behaviors):
                color = colors[i % len(colors)]
                item = QListWidgetItem(f"● {name}")
                item.setForeground(QColor(color))
                item.setData(Qt.UserRole, {"name": name, "color": color})
                cat_list.addItem(item)

        btn_load_preset.clicked.connect(_load_preset)

        btn_row = QHBoxLayout()
        btn_edit = QPushButton("Edit")
        btn_remove = QPushButton("Remove")
        btn_row.addWidget(btn_edit)
        btn_row.addWidget(btn_remove)
        layout.addLayout(btn_row)

        def _edit():
            item = cat_list.currentItem()
            if not item:
                return
            data = item.data(Qt.UserRole)
            name, ok = QInputDialog.getText(
                dlg, "Edit Category", "Behavior name:", text=data["name"]
            )
            if ok and name.strip():
                data["name"] = name.strip()
                item.setText(f"● {name.strip()}")
                item.setData(Qt.UserRole, data)

        def _remove():
            row = cat_list.currentRow()
            if row >= 0:
                cat_list.takeItem(row)

        btn_edit.clicked.connect(_edit)
        btn_remove.clicked.connect(_remove)

        btn_done = QPushButton("Done")
        btn_done.setStyleSheet(
            "QPushButton { background-color: #7b1fa2; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #9c27b0; }"
        )
        btn_done.clicked.connect(dlg.accept)
        layout.addWidget(btn_done)

        if dlg.exec_() == QDialog.Accepted:
            new_names = set()
            for i in range(cat_list.count()):
                d = cat_list.item(i).data(Qt.UserRole)
                if d:
                    new_names.add(d["name"])

            if not self._check_and_reset_clips_for_removed_behaviors(new_names):
                return

            self.list_cls_categories.clear()
            for i in range(cat_list.count()):
                src = cat_list.item(i)
                item = QListWidgetItem(src.text())
                item.setForeground(src.foreground())
                item.setData(Qt.UserRole, src.data(Qt.UserRole))
                self.list_cls_categories.addItem(item)

    def _add_behavior_category(self):
        name, ok = QInputDialog.getText(
            self, "Add Behavior Category",
            "Behavior name (e.g. grooming, rearing, locomotion):"
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        colors = ["#e53935", "#43a047", "#1e88e5", "#fb8c00",
                  "#8e24aa", "#00acc1", "#ffb300", "#6d4c41"]
        color = colors[self.list_cls_categories.count() % len(colors)]
        item = QListWidgetItem(f"● {name}")
        item.setForeground(QColor(color))
        item.setData(Qt.UserRole, {"name": name, "color": color})
        self.list_cls_categories.addItem(item)

    def _edit_behavior_category(self):
        item = self.list_cls_categories.currentItem()
        if not item:
            return
        data = item.data(Qt.UserRole)
        name, ok = QInputDialog.getText(
            self, "Edit Category", "Behavior name:", text=data["name"]
        )
        if not ok or not name.strip():
            return
        data["name"] = name.strip()
        item.setText(f"● {name.strip()}")
        item.setData(Qt.UserRole, data)

    def _remove_behavior_category(self):
        row = self.list_cls_categories.currentRow()
        if row >= 0:
            self.list_cls_categories.takeItem(row)

    def _add_classifier_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos for Annotation", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm)"
        )
        for f in files:
            if not any(
                self.list_cls_videos.item(i).data(Qt.UserRole) == f
                for i in range(self.list_cls_videos.count())
            ):
                item = QListWidgetItem(os.path.basename(f))
                item.setData(Qt.UserRole, f)
                self.list_cls_videos.addItem(item)
        if files:
            self.list_cls_videos.setCurrentRow(self.list_cls_videos.count() - 1)

    def _clear_classifier_videos(self):
        self.list_cls_videos.clear()
        self._release_cls_video()
        self.preview.clear()
        self.preview.update()
        self._update_classifier_state()

    def _release_cls_video(self):
        if self._cls_video_cap is not None:
            self._cls_video_cap.release()
            self._cls_video_cap = None
        self._cls_video_path = None
        self._cls_total_frames = 0
        self._cls_frame_idx = 0
        self._cls_clip_masks = {}

    def _dismiss_cls_training_viz(self):
        if self._cls_training_viz_panel.isVisible():
            self._cls_training_viz_panel.setVisible(False)
            self.preview.setVisible(True)
            self._info_row.setVisible(True)
            self._cls_nav_bar.setVisible(True)
            self._btn_toggle_training.setText("Show Training Graph")
            self._btn_toggle_training.setVisible(True)

    def _toggle_cls_training_viz(self):
        showing_graph = self._cls_training_viz_panel.isVisible()
        if showing_graph:
            self._cls_training_viz_panel.setVisible(False)
            self.preview.setVisible(True)
            self._info_row.setVisible(True)
            self._cls_nav_bar.setVisible(True)
            self._btn_toggle_training.setText("Show Training Graph")
            self._lbl_preview_title.setText("Clip Preview — Right-click mask to label")
        else:
            self.preview.setVisible(False)
            self._info_row.setVisible(False)
            self._cls_nav_bar.setVisible(False)
            self._cls_training_viz_panel.setVisible(True)
            self._btn_toggle_training.setText("Show Clip Preview")
            cls_active = (
                hasattr(self, "_cls_train_worker")
                and self._cls_train_worker is not None
                and self._cls_train_worker.isRunning()
            )
            self._lbl_preview_title.setText(
                "Training Preview" if cls_active else "Training Complete"
            )

    def _on_cls_video_selected(self, current, previous):
        if current is None:
            return
        video_path = current.data(Qt.UserRole)
        if not video_path or not os.path.exists(video_path):
            return

        self._dismiss_cls_training_viz()

        # Deselect clip queue — video and queue are mutually exclusive
        self._exit_queue_mode()
        self.list_clip_queue.blockSignals(True)
        self.list_clip_queue.clearSelection()
        self.list_clip_queue.setCurrentItem(None)
        self.list_clip_queue.blockSignals(False)

        if video_path == self._cls_video_path:
            return

        self._release_cls_video()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.lbl_classifier_info.setText(f"Cannot open: {os.path.basename(video_path)}")
            return

        self._cls_video_cap = cap
        self._cls_video_path = video_path
        self._cls_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._cls_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._cls_frame_idx = 0

        self._cls_frame_slider.blockSignals(True)
        self._cls_frame_slider.setMaximum(max(self._cls_total_frames - 1, 0))
        self._cls_frame_slider.setValue(0)
        self._cls_frame_slider.blockSignals(False)

        self._show_cls_frame(0)
        self._update_classifier_state()
        self.setFocus()

    def _show_cls_frame(self, frame_idx: int):
        if self._cls_video_cap is None:
            return
        if self.preview.annotations:
            self.preview.annotations.clear()
        self._lbl_preview_title.setText("Video Preview")
        frame_idx = max(0, min(frame_idx, self._cls_total_frames - 1))
        self._cls_video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cls_video_cap.read()
        if not ret:
            return

        self._cls_frame_idx = frame_idx
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Overlay extracted clip masks if we have them
        if self._cls_clip_masks and self._cls_clip_start <= frame_idx:
            clip_offset = frame_idx - self._cls_clip_start
            frame_rgb = self._overlay_clip_masks(frame_rgb, clip_offset)

        self.preview.set_frame(frame_rgb)

        w = int(self._cls_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cls_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        t = frame_idx / self._cls_fps if self._cls_fps > 0 else 0
        vname = os.path.basename(self._cls_video_path or "")

        self.lbl_classifier_info.setText(
            f"{vname}  —  {w}x{h}  frame {frame_idx}/{self._cls_total_frames}  "
            f"({t:.1f}s)  [Left/Right: navigate, E: extract clip]"
        )
        self.lbl_cls_frame_num.setText(
            f"{frame_idx} / {self._cls_total_frames}"
        )

        self._cls_frame_slider.blockSignals(True)
        self._cls_frame_slider.setValue(frame_idx)
        self._cls_frame_slider.blockSignals(False)

    def _overlay_clip_masks(self, frame_rgb: np.ndarray, clip_offset: int) -> np.ndarray:
        overlay = frame_rgb.copy()
        colors = [
            (233, 57, 53), (67, 160, 71), (30, 136, 229),
            (251, 140, 0), (142, 36, 170), (0, 172, 193),
        ]
        for i, (obj_id, frames_data) in enumerate(self._cls_clip_masks.items()):
            if clip_offset >= len(frames_data):
                continue
            det = frames_data[clip_offset]
            if det is None:
                continue
            mask = det.get("mask")
            if mask is None:
                continue

            color = colors[i % len(colors)]
            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)

            bbox = det["bbox"]
            x1, y1 = int(bbox[0]), int(bbox[1])

            # Build label: "ClassName: obj_id" (e.g. "Vole: 1")
            cat_label = det.get("label")
            cats = getattr(self, "_cls_mask_categories", {})
            if cat_label is not None and cats:
                class_name = cats.get(cat_label, f"class{cat_label}")
            elif cats:
                # Disk-loaded masks lack a label field; use first category
                class_name = next(iter(cats.values()))
            else:
                class_name = None
            if class_name:
                label_text = f"{class_name.capitalize()}: {obj_id}"
            else:
                label_text = f"ID {obj_id}"
            clip_idx = getattr(self, "_cls_playback_clip_idx", -1)
            if 0 <= clip_idx < len(self._batch_clips):
                obj_info = self._batch_clips[clip_idx].get("objects", {}).get(str(obj_id), {})
                beh = obj_info.get("behavior")
                if beh:
                    label_text += f": {beh}"

            text_org = (x1, max(y1 - 5, 12))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            tx, ty = text_org
            cv2.rectangle(
                overlay,
                (tx - 1, ty - th - 2),
                (tx + tw + 1, ty + baseline + 1),
                (0, 0, 0), cv2.FILLED,
            )
            cv2.putText(
                overlay, label_text, text_org,
                font, font_scale, color, thickness, cv2.LINE_AA,
            )
        return overlay

    def _on_cls_slider_pressed(self):
        self._stop_clip_playback()

    def _on_cls_slider_changed(self, value: int):
        self._stop_clip_playback()
        if self._cls_in_queue_mode:
            self._cls_playback_offset = value
            self._show_queue_frame(value)
        elif self._cls_pending_clip_mode:
            self._cls_playback_offset = value
            self._show_pending_frame(value)
        else:
            self._show_cls_frame(value)

    def _cls_navigate(self, delta: int):
        if self._cls_in_queue_mode:
            frames = getattr(self, "_queue_frames", [])
            if frames:
                new_off = max(0, min(
                    self._cls_playback_offset + delta, len(frames) - 1
                ))
                self._cls_playback_offset = new_off
                self._show_queue_frame(new_off)
        elif self._cls_pending_clip_mode:
            frames = getattr(self, "_pending_frames", [])
            if frames:
                new_off = max(0, min(self._cls_playback_offset + delta, len(frames) - 1))
                self._cls_playback_offset = new_off
                self._show_pending_frame(new_off)
        else:
            new_idx = self._cls_frame_idx + delta
            if 0 <= new_idx < self._cls_total_frames:
                self._show_cls_frame(new_idx)

    def keyPressEvent(self, event):
        if self.tab_widget.currentIndex() == 0:  # Annotation tab
            if self._annot_in_video_mode and self._video_cap is not None:
                if event.key() == Qt.Key_Left:
                    mod = event.modifiers()
                    if mod & Qt.ShiftModifier:
                        delta = -10
                    elif mod & Qt.ControlModifier:
                        delta = -100
                    else:
                        delta = -1
                    self._show_video_frame(self._video_frame_idx + delta)
                    return
                elif event.key() == Qt.Key_Right:
                    mod = event.modifiers()
                    if mod & Qt.ShiftModifier:
                        delta = 10
                    elif mod & Qt.ControlModifier:
                        delta = 100
                    else:
                        delta = 1
                    self._show_video_frame(self._video_frame_idx + delta)
                    return
                elif event.key() == Qt.Key_E:
                    self._extract_frame_from_video()
                    return
        elif self.tab_widget.currentIndex() == 1:  # Classify tab
            if event.key() == Qt.Key_Escape:
                if self._cls_pending_clip_mode:
                    self._exit_pending_clip_mode()
                elif self._cls_in_queue_mode:
                    self._exit_queue_mode()
                return
            elif event.key() == Qt.Key_Left:
                self._stop_clip_playback()
                mod = event.modifiers()
                if mod & Qt.ShiftModifier:
                    self._cls_navigate(-10)
                elif mod & Qt.ControlModifier:
                    self._cls_navigate(-100)
                else:
                    self._cls_navigate(-1)
                return
            elif event.key() == Qt.Key_Right:
                self._stop_clip_playback()
                mod = event.modifiers()
                if mod & Qt.ShiftModifier:
                    self._cls_navigate(10)
                elif mod & Qt.ControlModifier:
                    self._cls_navigate(100)
                else:
                    self._cls_navigate(1)
                return
            elif event.key() in (Qt.Key_Up, Qt.Key_Down):
                if self._cls_in_queue_mode or self._cls_pending_clip_mode:
                    was_playing = hasattr(self, "_clip_play_timer") and self._clip_play_timer.isActive()
                    self._cls_playback_was_playing = was_playing
                    self._stop_clip_playback()
                    cur_item = self.list_clip_queue.currentItem()
                    cur = self.list_clip_queue.indexOfTopLevelItem(cur_item) if cur_item else -1
                    total = self.list_clip_queue.topLevelItemCount()
                    if total > 0:
                        if event.key() == Qt.Key_Up:
                            nxt = max(0, cur - 1)
                        else:
                            nxt = min(total - 1, cur + 1)
                        if nxt != cur:
                            self.list_clip_queue.setCurrentItem(self.list_clip_queue.topLevelItem(nxt))
                    return
            elif event.key() == Qt.Key_E:
                if self._cls_pending_clip_mode:
                    return
                self._exit_queue_mode()
                self._extract_clip_at_current_frame()
                return
            elif event.key() == Qt.Key_D:
                if self._cls_in_queue_mode:
                    self._delete_current_clip()
                    return
            elif event.key() == Qt.Key_S:
                if self._cls_in_queue_mode:
                    self._advance_to_next_unlabeled_clip()
                    return
            elif event.key() == Qt.Key_Space:
                if hasattr(self, "_clip_play_timer") and self._clip_play_timer.isActive():
                    self._stop_clip_playback()
                    self._cls_playback_was_playing = False
                elif (self._cls_in_queue_mode
                      or self._cls_pending_clip_mode
                      or self._cls_video_cap is not None):
                    self._cls_play_resume()
                    self._cls_playback_was_playing = True
                return
        super().keyPressEvent(event)

    class _ClipQueueTreeItem(QTreeWidgetItem):
        """Tree item that sorts Frames numerically and Clip ignoring the status prefix."""
        def __lt__(self, other):
            col = self.treeWidget().sortColumn() if self.treeWidget() else 0
            if col == 1:
                try:
                    return float(self.text(1)) < float(other.text(1))
                except ValueError:
                    return self.text(1) < other.text(1)
            if col == 0:
                return self.text(0).lstrip("✔● ") < other.text(0).lstrip("✔● ")
            return super().__lt__(other)

    def _clip_queue_item_by_idx(self, idx: int):
        """Find the clip queue tree item whose stored clip index matches idx.

        Items move when the user sorts columns, so visual row order cannot be
        used to map back to self._batch_clips indices.
        """
        for i in range(self.list_clip_queue.topLevelItemCount()):
            item = self.list_clip_queue.topLevelItem(i)
            if item.data(0, Qt.UserRole) == idx:
                return item
        return None

    def _make_clip_queue_item(self, clip: dict, idx: int) -> 'QTreeWidgetItem':
        """Build a QTreeWidgetItem for the clip queue table."""
        vname = os.path.basename(clip.get("source_video", ""))
        frames = str(clip.get("clip_length", "?"))
        status = clip.get("status", "pending")
        objects = clip.get("objects", {})

        behaviors = list(dict.fromkeys(
            v["behavior"] for v in objects.values()
            if v.get("behavior")
        ))
        if status == "labeled" and behaviors:
            if len(behaviors) > 1:
                action_text = f"multi ({', '.join(behaviors)})"
            else:
                action_text = behaviors[0]
            color = QColor("#4fc456")
            prefix = "✔ "
        else:
            action_text = ""
            color = QColor("#e6c830")
            prefix = "● "

        item = self._ClipQueueTreeItem([prefix + vname, frames, action_text])
        for col in range(3):
            item.setForeground(col, color)
        item.setData(0, Qt.UserRole, idx)
        return item

    def _update_clip_queue_item(self, item: 'QTreeWidgetItem', clip: dict):
        """Update an existing clip queue tree item in-place."""
        vname = os.path.basename(clip.get("source_video", ""))
        frames = str(clip.get("clip_length", "?"))
        status = clip.get("status", "pending")
        objects = clip.get("objects", {})

        behaviors = list(dict.fromkeys(
            v["behavior"] for v in objects.values()
            if v.get("behavior")
        ))
        if status == "labeled" and behaviors:
            if len(behaviors) > 1:
                action_text = f"multi ({', '.join(behaviors)})"
            else:
                action_text = behaviors[0]
            color = QColor("#4fc456")
            prefix = "✔ "
        else:
            action_text = ""
            color = QColor("#e6c830")
            prefix = "● "

        item.setText(0, prefix + vname)
        item.setText(1, frames)
        item.setText(2, action_text)
        for col in range(3):
            item.setForeground(col, color)

    def _load_mask_categories(self, model_dir: str):
        """Load category map from a mask model's training_config.json."""
        self._cls_mask_categories = {}
        try:
            config_path = os.path.join(model_dir, "training_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                cats = cfg.get("categories", {})
                self._cls_mask_categories = {int(k): v for k, v in cats.items()}
        except Exception:
            pass

    def _extract_clip_at_current_frame(self):
        if self._cls_video_cap is None:
            QMessageBox.warning(self, "No Video", "Load a video first.")
            return
        model_idx = self.combo_cls_model.currentIndex()
        if model_idx < 0 or model_idx >= len(getattr(self, "_cls_model_dirs", [])):
            QMessageBox.warning(self, "No Model", "Select a mask model first.")
            return

        model_dir = self._cls_model_dirs[model_idx]
        clip_length = self.spin_cls_clip_length.value()
        confidence = self.spin_cls_confidence.value()
        max_det = self.spin_cls_max_det.value() or 100
        start_frame = self._cls_frame_idx

        self._cls_clip_masks = {}
        self._cls_clip_start = start_frame
        self._load_mask_categories(model_dir)
        self.lbl_cls_extract_status.setText("Extracting clip masks...")
        self.btn_batch_extract.setEnabled(False)

        self._clip_extract_worker = _ClipExtractWorker(
            model_dir, self._cls_video_path,
            start_frame, clip_length, confidence, max_det,
        )
        self._clip_extract_worker.finished_ok.connect(self._on_clip_extract_done)
        self._clip_extract_worker.error.connect(self._on_clip_extract_error)
        self._clip_extract_worker.start()

    def _on_clip_extract_done(self, clip_data: dict):
        self._cls_clip_masks = clip_data
        self.btn_batch_extract.setEnabled(True)
        n_obj = len(clip_data)
        clip_len = self.spin_cls_clip_length.value()

        if n_obj == 0:
            self.lbl_cls_extract_status.setText("No objects detected.")
            self._show_cls_frame(self._cls_clip_start)
            QMessageBox.warning(
                self, "No Detections",
                "No objects detected in this clip.\n\n"
                "Possible causes:\n"
                "• The selected model may be undertrained (stopped early)\n"
                "• Confidence threshold may be too high\n"
                "• No recognizable objects at this position\n\n"
                "Try a different frame, lower the confidence, or select "
                "a different model.",
            )
            return

        has_any_mask = any(
            det is not None and det.get("mask") is not None
            for frames in clip_data.values()
            for det in frames
        )
        if not has_any_mask:
            self.lbl_cls_extract_status.setText("No masks in detections.")
            self._show_cls_frame(self._cls_clip_start)
            QMessageBox.warning(
                self, "No Masks",
                "Objects were detected but none had masks.\n\n"
                "The model may not support instance segmentation, "
                "or it may be undertrained. Try a different model.",
            )
            return

        self._enter_pending_clip_mode(n_obj, clip_len)

    def _on_clip_extract_error(self, msg: str):
        self.btn_batch_extract.setEnabled(True)
        self.lbl_cls_extract_status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Extraction Error", msg)

    def _enter_pending_clip_mode(self, n_obj: int, clip_len: int):
        self._cls_pending_clip_mode = True
        self._cls_playback_offset = 0

        # Pre-read clip frames into memory to avoid repeated VideoCapture seeks
        self._pending_frames = []
        if self._cls_video_cap is not None:
            self._cls_video_cap.set(cv2.CAP_PROP_POS_FRAMES, self._cls_clip_start)
            for _ in range(clip_len):
                ret, frame = self._cls_video_cap.read()
                if not ret:
                    break
                self._pending_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        self._cls_frame_slider.blockSignals(True)
        self._cls_frame_slider.setMinimum(0)
        self._cls_frame_slider.setMaximum(max(len(self._pending_frames) - 1, 0))
        self._cls_frame_slider.setValue(0)
        self._cls_frame_slider.blockSignals(False)

        self.lbl_cls_extract_status.setText(
            f"Pending clip — {n_obj} object(s), {len(self._pending_frames)} frames. "
            f"Right-click mask to label, Escape to cancel."
        )
        self.preview._cls_mask_handler = self._on_cls_mask_right_click

        self._show_pending_frame(0)
        self._cls_play_resume()

    def _show_pending_frame(self, offset: int):
        frames = getattr(self, "_pending_frames", [])
        if not frames:
            return
        self._lbl_preview_title.setText("Clip Preview — Right-click mask to label · Esc to exit")
        offset = max(0, min(offset, len(frames) - 1))
        frame_rgb = frames[offset].copy()

        if self._cls_clip_masks:
            frame_rgb = self._overlay_clip_masks(frame_rgb, offset)

        self.preview.set_frame(frame_rgb)
        self._cls_frame_idx = self._cls_clip_start + offset

        clip_len = len(frames)
        self.lbl_classifier_info.setText(
            f"PENDING CLIP  —  frame {offset + 1}/{clip_len}  "
            f"[Right-click mask to label, Esc to cancel]"
        )
        self.lbl_cls_frame_num.setText(f"{offset} / {clip_len}")

        self._cls_frame_slider.blockSignals(True)
        self._cls_frame_slider.setValue(offset)
        self._cls_frame_slider.blockSignals(False)

    def _exit_pending_clip_mode(self):
        self._stop_clip_playback()
        self._cls_pending_clip_mode = False
        self._cls_clip_masks = {}
        self._pending_frames = []
        if self._cls_video_cap is not None:
            self._cls_frame_slider.blockSignals(True)
            self._cls_frame_slider.setMinimum(0)
            self._cls_frame_slider.setMaximum(
                max(self._cls_total_frames - 1, 0)
            )
            self._cls_frame_slider.setValue(self._cls_frame_idx)
            self._cls_frame_slider.blockSignals(False)
            self._show_cls_frame(self._cls_frame_idx)
        self.lbl_cls_extract_status.setText("")

    def _on_cls_mask_right_click(self, img_x: int, img_y: int, global_pos):
        if not self._cls_clip_masks:
            return False

        if self._cls_in_queue_mode or self._cls_pending_clip_mode:
            clip_offset = getattr(self, "_cls_playback_offset", 0)
        else:
            clip_offset = self._cls_frame_idx - self._cls_clip_start
        if clip_offset < 0:
            return False

        hit_obj_id = None
        for obj_id, frames_data in self._cls_clip_masks.items():
            if clip_offset >= len(frames_data):
                continue
            det = frames_data[clip_offset]
            if det is None:
                continue
            mask = det.get("mask")
            if mask is None:
                continue
            if 0 <= img_y < mask.shape[0] and 0 <= img_x < mask.shape[1]:
                if mask[img_y, img_x]:
                    hit_obj_id = obj_id
                    break

        if hit_obj_id is None:
            return False

        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #2b2b2b; color: #cccccc; border: 1px solid #555; }"
            "QMenu::item:selected { background-color: #2979ff; }"
        )
        current_behavior = None
        clip_idx = getattr(self, "_cls_playback_clip_idx", -1)
        if 0 <= clip_idx < len(self._batch_clips):
            obj_info = self._batch_clips[clip_idx].get("objects", {}).get(str(hit_obj_id), {})
            current_behavior = obj_info.get("behavior")

        cats = getattr(self, "_cls_mask_categories", {})
        if cats:
            _cls_name = next(iter(cats.values()), "Object")
            _obj_label = f"{_cls_name.capitalize()} {hit_obj_id}"
        else:
            _obj_label = f"Object {hit_obj_id}"
        menu.addAction(f"{_obj_label} — Assign behavior:").setEnabled(False)
        menu.addSeparator()
        for i in range(self.list_cls_categories.count()):
            cat_item = self.list_cls_categories.item(i)
            data = cat_item.data(Qt.UserRole)
            label = f"● {data['name']}"
            if current_behavior and data["name"] == current_behavior:
                label += "  [selected]"
            act = menu.addAction(label)
            act.setData({"obj_id": hit_obj_id, "behavior": data["name"], "color": data["color"]})

        menu.addSeparator()
        add_new_act = menu.addAction("+ Add new behavior...")

        # Preset loadouts submenu
        preset_menu = menu.addMenu("Load preset...")
        preset_menu.setStyleSheet(
            "QMenu { background-color: #2b2b2b; color: #cccccc; border: 1px solid #555; }"
            "QMenu::item:selected { background-color: #2979ff; }"
        )
        preset_actions = {}
        for preset_name, behaviors in BEHAVIOR_PRESETS.items():
            act = preset_menu.addAction(f"{preset_name}  ({', '.join(behaviors)})")
            preset_actions[id(act)] = (preset_name, behaviors)

        chosen = menu.exec_(global_pos)
        if not chosen:
            return True

        if id(chosen) in preset_actions:
            _, behaviors = preset_actions[id(chosen)]
            new_names = set(behaviors)
            if not self._check_and_reset_clips_for_removed_behaviors(new_names):
                return True
            colors = ["#e53935", "#43a047", "#1e88e5", "#fb8c00",
                      "#8e24aa", "#00acc1", "#ffb300", "#6d4c41"]
            self.list_cls_categories.clear()
            for i, name in enumerate(behaviors):
                color = colors[i % len(colors)]
                item = QListWidgetItem(f"● {name}")
                item.setForeground(QColor(color))
                item.setData(Qt.UserRole, {"name": name, "color": color})
                self.list_cls_categories.addItem(item)
            self.status_bar.showMessage(
                f"Loaded preset: {', '.join(behaviors)}", 3000
            )
            return True

        if chosen is add_new_act:
            name, ok = QInputDialog.getText(
                self, "New Behavior", "Behavior name:"
            )
            if not ok or not name.strip():
                return True
            name = name.strip()
            colors = ["#e53935", "#43a047", "#1e88e5", "#fb8c00",
                      "#8e24aa", "#00acc1", "#ffb300", "#6d4c41"]
            color = colors[self.list_cls_categories.count() % len(colors)]
            item = QListWidgetItem(f"● {name}")
            item.setForeground(QColor(color))
            item.setData(Qt.UserRole, {"name": name, "color": color})
            self.list_cls_categories.addItem(item)
            label_data = {"obj_id": hit_obj_id, "behavior": name, "color": color}
        elif chosen.data():
            label_data = chosen.data()
        else:
            return True

        if self._cls_in_queue_mode:
            self._label_batch_clip(label_data)
        else:
            self._export_manual_clip(label_data)
        return True

    # ==================================================================
    # Clip storage helpers (disk-backed clips)
    # ==================================================================
    def _clips_base_dir(self) -> str:
        return os.path.join(self._project_dir, "action_classifier", "clips")

    def _next_clip_id(self) -> str:
        base = self._clips_base_dir()
        if not os.path.isdir(base):
            return "clip_0001"
        existing = [
            d for d in os.listdir(base)
            if d.startswith("clip_") and os.path.isdir(os.path.join(base, d))
        ]
        if not existing:
            return "clip_0001"
        nums = []
        for d in existing:
            try:
                nums.append(int(d.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return f"clip_{max(nums) + 1:04d}" if nums else "clip_0001"

    def _save_clip_to_disk(self, video_path, start_frame, clip_length,
                           fps, masks_dict, status="pending"):
        clip_id = self._next_clip_id()
        clip_dir = os.path.join(self._clips_base_dir(), clip_id)
        os.makedirs(clip_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            os.path.join(clip_dir, "frames.mp4"),
            fourcc, fps, (frame_w, frame_h),
        )
        actual_frames = 0
        for _ in range(clip_length):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            actual_frames += 1
        writer.release()
        cap.release()

        mask_arrays = {}
        for obj_id, frames_data in masks_dict.items():
            per_frame = []
            for det in frames_data:
                if det is not None and det.get("mask") is not None:
                    per_frame.append(det["mask"].astype(bool))
                else:
                    per_frame.append(np.zeros((frame_h, frame_w), dtype=bool))
            mask_arrays[f"obj_{obj_id}"] = np.stack(per_frame, axis=0)
        np.savez_compressed(os.path.join(clip_dir, "masks.npz"), **mask_arrays)

        objects = {}
        for obj_id in masks_dict:
            objects[str(obj_id)] = {"behavior": None, "color": None}

        meta = {
            "clip_id": clip_id,
            "source_video": video_path,
            "start_frame": start_frame,
            "clip_length": actual_frames,
            "fps": fps,
            "frame_width": frame_w,
            "frame_height": frame_h,
            "num_objects": len(masks_dict),
            "status": status,
            "objects": objects,
        }
        with open(os.path.join(clip_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        return clip_dir

    def _load_clip_frames(self, clip_dir):
        mp4_path = os.path.join(clip_dir, "frames.mp4")
        if not os.path.exists(mp4_path):
            return []
        cap = cv2.VideoCapture(mp4_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames

    def _load_clip_masks(self, clip_dir):
        npz_path = os.path.join(clip_dir, "masks.npz")
        if not os.path.exists(npz_path):
            return {}
        data = np.load(npz_path)
        masks_dict = {}
        for key in data.files:
            obj_id = int(key.replace("obj_", ""))
            arr = data[key]  # (N, H, W) bool
            frames_data = []
            for i in range(arr.shape[0]):
                mask = arr[i]
                ys, xs = np.where(mask)
                if len(ys) == 0:
                    frames_data.append(None)
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                frames_data.append({
                    "mask": mask,
                    "bbox": (x1, y1, x2, y2),
                    "centroid": (cx, cy),
                })
            masks_dict[obj_id] = frames_data
        return masks_dict

    def _save_clip_label(self, clip_dir, obj_id, behavior, color):
        meta_path = os.path.join(clip_dir, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["objects"][str(obj_id)] = {"behavior": behavior, "color": color}
        any_labeled = any(
            v["behavior"] is not None for v in meta["objects"].values()
        )
        meta["status"] = "labeled" if any_labeled else "pending"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        masks_dict = self._load_clip_masks(clip_dir)
        if obj_id in masks_dict:
            from .silhouette_extractor import generate_composite

            ux1, uy1, ux2, uy2 = float("inf"), float("inf"), 0, 0
            for det in masks_dict[obj_id]:
                if det is None:
                    continue
                bx1, by1, bx2, by2 = (
                    int(det["bbox"][0]), int(det["bbox"][1]),
                    int(det["bbox"][2]), int(det["bbox"][3]),
                )
                ux1 = min(ux1, bx1)
                uy1 = min(uy1, by1)
                ux2 = max(ux2, bx2)
                uy2 = max(uy2, by2)

            if ux2 <= ux1 or uy2 <= uy1:
                return

            pad = int(max(ux2 - ux1, uy2 - uy1) * 0.05)
            ux1 = max(0, int(ux1) - pad)
            uy1 = max(0, int(uy1) - pad)
            ux2 = int(ux2) + pad
            uy2 = int(uy2) + pad

            mask_crops = []
            for det in masks_dict[obj_id]:
                if det is None:
                    mask_crops.append(None)
                    continue
                mask = det["mask"]
                uy2_c = min(uy2, mask.shape[0])
                ux2_c = min(ux2, mask.shape[1])
                crop = mask[uy1:uy2_c, ux1:ux2_c]
                mask_crops.append(crop.astype(bool))
            composite = generate_composite(mask_crops, output_size=(128, 128))
            comp_path = os.path.join(clip_dir, f"composite_obj{obj_id}.png")
            cv2.imwrite(comp_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

    def _delete_clip_from_disk(self, clip_dir):
        if os.path.isdir(clip_dir):
            shutil.rmtree(clip_dir)

    def _scan_clips_from_disk(self):
        base = self._clips_base_dir()
        if not os.path.isdir(base):
            return []
        results = []
        for d in sorted(os.listdir(base)):
            clip_dir = os.path.join(base, d)
            meta_path = os.path.join(clip_dir, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                results.append((clip_dir, meta))
            except Exception:
                continue
        return results

    def _export_manual_clip(self, label_data: dict):
        if not self._project_dir:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return
        if not self._cls_clip_masks:
            return

        reply = QMessageBox.question(
            self, "Export Clip",
            f"Export clip as '{label_data['behavior']}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        clip_dir = self._save_clip_to_disk(
            self._cls_video_path,
            self._cls_clip_start,
            self.spin_cls_clip_length.value(),
            self._cls_fps,
            self._cls_clip_masks,
            status="labeled",
        )

        self._save_clip_label(
            clip_dir, label_data["obj_id"],
            label_data["behavior"], label_data["color"],
        )

        meta_path = os.path.join(clip_dir, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        idx = len(self._batch_clips)
        self._batch_clips.append({
            "clip_dir": clip_dir,
            "clip_id": meta["clip_id"],
            "source_video": self._cls_video_path,
            "start_frame": self._cls_clip_start,
            "clip_length": meta["clip_length"],
            "status": "labeled",
            "objects": meta.get("objects", {}),
        })

        item = self._make_clip_queue_item(self._batch_clips[idx], idx)
        self.list_clip_queue.blockSignals(True)
        self.list_clip_queue.addTopLevelItem(item)
        self.list_clip_queue.clearSelection()
        self.list_clip_queue.setCurrentItem(None)
        self.list_clip_queue.blockSignals(False)

        self._stop_clip_playback()
        self._cls_pending_clip_mode = False
        self._cls_clip_masks = {}
        self._pending_frames = []

        if self._cls_video_cap is not None:
            self._cls_frame_slider.blockSignals(True)
            self._cls_frame_slider.setMinimum(0)
            self._cls_frame_slider.setMaximum(
                max(self._cls_total_frames - 1, 0)
            )
            self._cls_frame_slider.setValue(self._cls_frame_idx)
            self._cls_frame_slider.blockSignals(False)

        self._show_cls_frame(self._cls_frame_idx)
        self.setFocus()
        self._refresh_cls_annotation_stats()
        self.lbl_cls_extract_status.setText(
            f"Clip exported as '{label_data['behavior']}'. "
            f"Continue scrubbing or press E for another clip."
        )

    def _load_classifier_data_from_disk(self):
        if not self._project_dir:
            return

        saved_clips = self._scan_clips_from_disk()
        if not saved_clips:
            return

        self._batch_clips = []
        self.list_clip_queue.setSortingEnabled(False)
        self.list_clip_queue.clear()

        colors_default = ["#e53935", "#43a047", "#1e88e5", "#fb8c00",
                          "#8e24aa", "#00acc1", "#ffb300", "#6d4c41"]
        seen_behaviors = {}

        for idx, (clip_dir, meta) in enumerate(saved_clips):
            clip_id = meta.get("clip_id", "?")
            source = meta.get("source_video", "")
            start = meta.get("start_frame", 0)
            status = meta.get("status", "pending")
            objects = meta.get("objects", {})

            clip_data = {
                "clip_dir": clip_dir,
                "clip_id": clip_id,
                "source_video": source,
                "start_frame": start,
                "clip_length": meta.get("clip_length", 15),
                "status": status,
                "objects": objects,
            }
            self._batch_clips.append(clip_data)

            item = self._make_clip_queue_item(clip_data, idx)
            self.list_clip_queue.addTopLevelItem(item)

            for v in objects.values():
                beh = v.get("behavior")
                clr = v.get("color")
                if beh and beh not in seen_behaviors:
                    seen_behaviors[beh] = clr or colors_default[
                        len(seen_behaviors) % len(colors_default)
                    ]

        self.list_clip_queue.setSortingEnabled(True)

        if seen_behaviors and self.list_cls_categories.count() == 0:
            for beh, clr in seen_behaviors.items():
                item = QListWidgetItem(f"● {beh}")
                item.setForeground(QColor(clr))
                item.setData(Qt.UserRole, {"name": beh, "color": clr})
                self.list_cls_categories.addItem(item)

        self._cls_data_loaded = True

    def _refresh_cls_annotation_stats(self):
        if not self._batch_clips:
            self.lbl_cls_annotation_stats.setText("No clips yet")
            self.lbl_cls_train_data.setText("Training data: —")
            self.lbl_cls_train_reqs.setText(
                "⚠ Requires at least 2 behavior categories with labeled clips to train."
            )
            self.btn_train_classifier.setEnabled(False)
            return
        from collections import Counter
        counts = Counter()
        n_composites = 0
        for clip in self._batch_clips:
            for obj_id_str, v in clip.get("objects", {}).items():
                beh = v.get("behavior")
                if beh:
                    counts[beh] += 1
                    comp = os.path.join(
                        clip.get("clip_dir", ""),
                        f"composite_obj{obj_id_str}.png",
                    )
                    if clip.get("clip_dir") and os.path.isfile(comp):
                        n_composites += 1
        if not counts:
            total_clips = len(self._batch_clips)
            labeled = sum(1 for c in self._batch_clips if c["status"] == "labeled")
            self.lbl_cls_annotation_stats.setText(
                f"{total_clips} clips ({labeled} labeled)"
            )
            self.lbl_cls_train_data.setText("Training data: no labeled clips yet")
            self.lbl_cls_train_reqs.setText(
                "⚠ Requires at least 2 behavior categories with labeled clips to train."
            )
            self.btn_train_classifier.setEnabled(False)
            return
        labeled_clips = sum(1 for c in self._batch_clips if c["status"] == "labeled")
        total_clips = len(self._batch_clips)
        parts = [f"{name}: {n}" for name, n in counts.most_common()]
        self.lbl_cls_annotation_stats.setText(
            f"{labeled_clips}/{total_clips} clips labeled — " + ", ".join(parts)
        )

        n_categories = len(counts)
        total_labels = sum(counts.values())
        data_lines = [
            f"Composites: {n_composites}  |  Classes: {n_categories}",
            "  ".join(f"{name}: {n}" for name, n in counts.most_common()),
        ]
        self.lbl_cls_train_data.setText("\n".join(data_lines))

        can_train = n_categories >= 2
        self.btn_train_classifier.setEnabled(can_train)
        if can_train:
            self.lbl_cls_train_reqs.setText("")
        else:
            self.lbl_cls_train_reqs.setText(
                f"⚠ Need at least 2 behavior categories (have {n_categories}). "
                f"Label clips with different behaviors to enable training."
            )

    def _get_current_category_names(self) -> set:
        names = set()
        for i in range(self.list_cls_categories.count()):
            data = self.list_cls_categories.item(i).data(Qt.UserRole)
            if data:
                names.add(data["name"])
        return names

    def _check_and_reset_clips_for_removed_behaviors(
        self, new_names: set, parent=None,
    ) -> bool:
        """Check if switching to *new_names* would orphan any labeled clips.

        If so, show a confirmation dialog with a breakdown.  On confirm (or if
        nothing is affected), reset the orphaned clips and return True.
        On cancel, return False.
        """
        if parent is None:
            parent = self
        affected: dict[str, int] = {}
        for clip in self._batch_clips:
            if clip["status"] != "labeled":
                continue
            for v in clip.get("objects", {}).values():
                beh = v.get("behavior")
                if beh and beh not in new_names:
                    affected[beh] = affected.get(beh, 0) + 1

        if not affected:
            return True

        total = sum(affected.values())
        breakdown = "\n".join(f"  • {name}: {n} clip(s)" for name, n in sorted(affected.items()))
        reply = QMessageBox.warning(
            parent, "Reset Labeled Clips?",
            f"This will reset {total} labeled clip(s) back to pending:\n\n"
            f"{breakdown}\n\n"
            f"Affected clips will need to be re-labeled.\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return False

        removed_names = set(affected.keys())
        for idx, clip in enumerate(self._batch_clips):
            if clip["status"] != "labeled":
                continue
            needs_reset = False
            for v in clip.get("objects", {}).values():
                beh = v.get("behavior")
                if beh and beh in removed_names:
                    needs_reset = True
                    break
            if not needs_reset:
                continue

            for v in clip.get("objects", {}).values():
                if v.get("behavior") and v["behavior"] in removed_names:
                    v["behavior"] = None
                    v["color"] = None

            any_labeled = any(
                v.get("behavior") is not None for v in clip.get("objects", {}).values()
            )
            clip["status"] = "labeled" if any_labeled else "pending"

            if clip.get("clip_dir"):
                meta_path = os.path.join(clip["clip_dir"], "meta.json")
                if os.path.isfile(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    for k, v in meta.get("objects", {}).items():
                        if v.get("behavior") and v["behavior"] in removed_names:
                            v["behavior"] = None
                            v["color"] = None
                    any_labeled_disk = any(
                        v.get("behavior") is not None
                        for v in meta.get("objects", {}).values()
                    )
                    meta["status"] = "labeled" if any_labeled_disk else "pending"
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)

            item = self._clip_queue_item_by_idx(idx)
            if item:
                self._update_clip_queue_item(item, clip)

        self._refresh_cls_annotation_stats()
        return True

    # --- Batch extraction ---
    def _start_batch_extraction(self):
        model_idx = self.combo_cls_model.currentIndex()
        if model_idx < 0 or model_idx >= len(getattr(self, "_cls_model_dirs", [])):
            QMessageBox.warning(self, "No Model", "Select a mask model first.")
            return
        if self.list_cls_videos.count() == 0:
            QMessageBox.warning(self, "No Videos", "Add annotation videos first.")
            return
        if not self._project_dir:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return

        model_dir = self._cls_model_dirs[model_idx]
        self._load_mask_categories(model_dir)
        clip_length = self.spin_cls_clip_length.value()
        confidence = self.spin_cls_confidence.value()
        max_det = self.spin_cls_max_det.value() or 100
        method = self.combo_batch_method.currentIndex()
        n_clips = self.spin_batch_count.value()

        clip_positions = []
        video_fps = {}
        for i in range(self.list_cls_videos.count()):
            vpath = self.list_cls_videos.item(i).data(Qt.UserRole)
            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            video_fps[vpath] = fps_val
            if total <= clip_length:
                continue

            max_start = total - clip_length
            if method == 0:  # Uniform
                starts = [int(j * max_start / max(n_clips - 1, 1))
                          for j in range(n_clips)]
            elif method == 1:  # Random
                starts = sorted(random.sample(
                    range(max_start), min(n_clips, max_start)
                ))
            else:  # Every Nth frame
                stride = n_clips
                starts = list(range(0, max_start, stride))

            for s in starts:
                clip_positions.append((vpath, s))

        if not clip_positions:
            QMessageBox.warning(self, "No Clips", "Videos too short for clip extraction.")
            return

        clips_base = self._clips_base_dir()
        os.makedirs(clips_base, exist_ok=True)

        # Determine starting clip number
        existing = self._scan_clips_from_disk()
        if existing:
            last_num = max(
                int(m["clip_id"].split("_")[1])
                for _, m in existing
            )
            start_num = last_num + 1
        else:
            start_num = 1

        # Add placeholder items to queue
        queue_start_idx = len(self._batch_clips)
        self.list_clip_queue.setSortingEnabled(False)
        for idx, (vpath, start) in enumerate(clip_positions):
            clip_data = {
                "clip_dir": None,
                "clip_id": None,
                "source_video": vpath,
                "start_frame": start,
                "clip_length": clip_length,
                "status": "extracting",
                "objects": {},
            }
            self._batch_clips.append(clip_data)
            item = self._make_clip_queue_item(clip_data, queue_start_idx + idx)
            self.list_clip_queue.addTopLevelItem(item)
        self.list_clip_queue.setSortingEnabled(True)

        self.btn_batch_extract.setEnabled(False)
        self.btn_reextract_clips.setEnabled(False)
        self.lbl_cls_annotation_stats.setText(
            f"Extracting {len(clip_positions)} clips..."
        )

        self._batch_queue_offset = queue_start_idx
        self._batch_worker = _BatchClipExtractWorker(
            model_dir, clip_positions, clip_length, confidence, max_det,
            clips_base, start_num,
        )
        self._batch_worker.clip_ready.connect(self._on_batch_clip_ready)
        self._batch_worker.all_done.connect(self._on_batch_done)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.start()

    def _on_batch_clip_ready(self, clip_idx, clip_dir):
        real_idx = getattr(self, "_batch_queue_offset", 0) + clip_idx
        if real_idx >= len(self._batch_clips):
            return
        meta_path = os.path.join(clip_dir, "meta.json")
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            return

        self._batch_clips[real_idx].update({
            "clip_dir": clip_dir,
            "clip_id": meta["clip_id"],
            "status": "pending",
            "objects": meta.get("objects", {}),
            "clip_length": meta.get("clip_length", 15),
        })

        item = self._clip_queue_item_by_idx(real_idx)
        if item:
            self._update_clip_queue_item(item, self._batch_clips[real_idx])

        done = sum(1 for c in self._batch_clips if c["clip_dir"] is not None)
        total = len(self._batch_clips)
        self.lbl_cls_annotation_stats.setText(
            f"Extracted {done}/{total} clips"
        )

    def _on_batch_done(self):
        self.btn_batch_extract.setEnabled(True)
        self.btn_reextract_clips.setEnabled(True)
        self._refresh_cls_annotation_stats()

    def _on_batch_error(self, msg):
        self.btn_batch_extract.setEnabled(True)
        self.btn_reextract_clips.setEnabled(True)
        QMessageBox.critical(self, "Batch Extraction Error", msg)

    def _on_clip_queue_selected(self, current, previous):
        if current is None:
            self._stop_clip_playback()
            if self._cls_in_queue_mode:
                self._cls_in_queue_mode = False
                self._cls_clip_masks = {}
                self._queue_frames = []
                self._lbl_preview_title.setText("")
                self.preview.set_frame(np.zeros((100, 100, 3), dtype=np.uint8))
            return
        idx = current.data(0, Qt.UserRole)
        if idx is None or idx >= len(getattr(self, "_batch_clips", [])):
            return

        self._dismiss_cls_training_viz()

        # Deselect video list — video and queue are mutually exclusive
        self.list_cls_videos.blockSignals(True)
        self.list_cls_videos.clearSelection()
        self.list_cls_videos.setCurrentItem(None)
        self.list_cls_videos.blockSignals(False)

        clip = self._batch_clips[idx]

        if clip["clip_dir"] is None:
            self.lbl_cls_extract_status.setText("Clip still extracting...")
            return

        self._cls_playback_clip_idx = idx
        self._cls_playback_offset = 0
        self._cls_in_queue_mode = True
        self._cls_pending_clip_mode = False

        # Ensure mask categories are loaded from the selected model
        if not getattr(self, "_cls_mask_categories", {}):
            model_idx = self.combo_cls_model.currentIndex()
            if 0 <= model_idx < len(getattr(self, "_cls_model_dirs", [])):
                self._load_mask_categories(self._cls_model_dirs[model_idx])

        # Load frames and masks from disk
        self._queue_frames = self._load_clip_frames(clip["clip_dir"])
        self._cls_clip_masks = self._load_clip_masks(clip["clip_dir"])
        self._cls_clip_start = 0  # queue frames are 0-indexed

        clip_length = len(self._queue_frames)
        if clip_length == 0:
            self.lbl_cls_extract_status.setText("Clip has no frames.")
            return

        # Set slider range for clip playback
        self._cls_frame_slider.blockSignals(True)
        self._cls_frame_slider.setMinimum(0)
        self._cls_frame_slider.setMaximum(clip_length - 1)
        self._cls_frame_slider.setValue(0)
        self._cls_frame_slider.blockSignals(False)

        # Determine fps from meta
        meta_path = os.path.join(clip["clip_dir"], "meta.json")
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            self._cls_fps = meta.get("fps", 30.0)
        except Exception:
            self._cls_fps = 30.0

        self._show_queue_frame(0)
        if self._cls_playback_was_playing:
            self._cls_play_resume()
        n_obj = len(self._cls_clip_masks)
        status = clip.get("status", "pending")
        if status == "labeled":
            behaviors = [
                v["behavior"] for v in clip.get("objects", {}).values()
                if v.get("behavior")
            ]
            self.lbl_cls_extract_status.setText(
                f"Labeled: {', '.join(behaviors)}. Right-click to re-label."
            )
        else:
            self.lbl_cls_extract_status.setText(
                f"Playing clip — {n_obj} object(s). Right-click mask to label."
            )

    def _show_queue_frame(self, offset):
        if not hasattr(self, "_queue_frames") or not self._queue_frames:
            return
        self._lbl_preview_title.setText("Clip Preview — Right-click mask to label")
        offset = max(0, min(offset, len(self._queue_frames) - 1))
        frame_rgb = self._queue_frames[offset].copy()

        if self._cls_clip_masks:
            frame_rgb = self._overlay_clip_masks(frame_rgb, offset)

        self.preview.set_frame(frame_rgb)
        self._cls_frame_idx = offset

        clip_idx = getattr(self, "_cls_playback_clip_idx", -1)
        clip = self._batch_clips[clip_idx] if 0 <= clip_idx < len(self._batch_clips) else {}
        clip_id = clip.get("clip_id", "?")
        total = len(self._queue_frames)
        self.lbl_classifier_info.setText(
            f"{clip_id}  —  frame {offset + 1}/{total}"
        )
        self.lbl_cls_frame_num.setText(f"{offset} / {total}")

        self._cls_frame_slider.blockSignals(True)
        self._cls_frame_slider.setValue(offset)
        self._cls_frame_slider.blockSignals(False)

    def _advance_clip_playback(self):
        if self._cls_in_queue_mode:
            frames = getattr(self, "_queue_frames", [])
            if not frames:
                return
            self._cls_playback_offset = (self._cls_playback_offset + 1) % len(frames)
            self._show_queue_frame(self._cls_playback_offset)
        elif self._cls_pending_clip_mode:
            frames = getattr(self, "_pending_frames", [])
            if not frames:
                return
            self._cls_playback_offset = (self._cls_playback_offset + 1) % len(frames)
            self._show_pending_frame(self._cls_playback_offset)
        elif self._cls_video_cap is not None:
            new_idx = self._cls_frame_idx + 1
            if new_idx >= self._cls_total_frames:
                self._stop_clip_playback()
                return
            self._show_cls_frame(new_idx)

    def _stop_clip_playback(self):
        if hasattr(self, "_clip_play_timer") and self._clip_play_timer.isActive():
            self._clip_play_timer.stop()
        if hasattr(self, "_btn_cls_play_pause"):
            self._btn_cls_play_pause.setText("▶")

    def _cls_play_resume(self):
        if not hasattr(self, "_clip_play_timer"):
            self._clip_play_timer = QTimer(self)
            self._clip_play_timer.timeout.connect(self._advance_clip_playback)
        if self._clip_play_timer.isActive():
            return
        fps = self._cls_fps if self._cls_fps > 0 else 30.0
        interval = max(1, int(1000 / (fps * self._cls_speed_multiplier)))
        self._clip_play_timer.start(interval)
        self._btn_cls_play_pause.setText("⏸")

    def _cls_toggle_playback(self):
        if hasattr(self, "_clip_play_timer") and self._clip_play_timer.isActive():
            self._stop_clip_playback()
            self._cls_playback_was_playing = False
        elif (self._cls_in_queue_mode
              or self._cls_pending_clip_mode
              or self._cls_video_cap is not None):
            self._cls_play_resume()
            self._cls_playback_was_playing = True
        self.setFocus()

    def _on_cls_speed_changed(self, idx):
        speed_map = [0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0]
        self._cls_speed_multiplier = speed_map[idx] if idx < len(speed_map) else 1.0
        if hasattr(self, "_clip_play_timer") and self._clip_play_timer.isActive():
            fps = self._cls_fps if self._cls_fps > 0 else 30.0
            interval = max(1, int(1000 / (fps * self._cls_speed_multiplier)))
            self._clip_play_timer.setInterval(interval)

    def _exit_queue_mode(self):
        self._stop_clip_playback()
        self._cls_in_queue_mode = False
        self._cls_pending_clip_mode = False
        self._cls_clip_masks = {}
        self._queue_frames = []
        # Restore slider to full video range
        if self._cls_video_cap is not None:
            self._cls_frame_slider.blockSignals(True)
            self._cls_frame_slider.setMinimum(0)
            self._cls_frame_slider.setMaximum(
                max(self._cls_total_frames - 1, 0)
            )
            self._cls_frame_slider.setValue(self._cls_frame_idx)
            self._cls_frame_slider.blockSignals(False)
            self._show_cls_frame(self._cls_frame_idx)
        self.lbl_cls_extract_status.setText("")

    def _advance_to_next_unlabeled_clip(self):
        was_playing = hasattr(self, "_clip_play_timer") and self._clip_play_timer.isActive()
        self._cls_playback_was_playing = was_playing
        self._stop_clip_playback()
        if not self._batch_clips:
            return
        cur_item = self.list_clip_queue.currentItem()
        current = self.list_clip_queue.indexOfTopLevelItem(cur_item) if cur_item else -1
        n = len(self._batch_clips)
        for offset in range(1, n + 1):
            idx = (current + offset) % n
            clip = self._batch_clips[idx]
            if clip["status"] == "pending" and clip["clip_dir"] is not None:
                target = self._clip_queue_item_by_idx(idx)
                if target is not None:
                    self.list_clip_queue.setCurrentItem(target)
                return
        self.lbl_cls_extract_status.setText("All clips labeled!")
        n_labeled = sum(1 for c in self._batch_clips if c["status"] == "labeled")
        QMessageBox.information(
            self, "All Clips Labeled",
            f"All {n_labeled} clip(s) have been labeled.\n\n"
            f"You can now train the classifier, or add more videos\n"
            f"and extract additional clips.",
        )

    def _delete_current_clip(self):
        idx = getattr(self, "_cls_playback_clip_idx", -1)
        if idx < 0 or idx >= len(self._batch_clips):
            return

        reply = QMessageBox.question(
            self, "Delete Clip",
            "Delete this clip from disk?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        was_playing = hasattr(self, "_clip_play_timer") and self._clip_play_timer.isActive()
        self._cls_playback_was_playing = was_playing
        self._stop_clip_playback()

        clip = self._batch_clips[idx]
        if clip.get("clip_dir"):
            self._delete_clip_from_disk(clip["clip_dir"])
        self._batch_clips.pop(idx)

        self._cls_in_queue_mode = False
        self._cls_clip_masks = {}
        self._queue_frames = []

        self.list_clip_queue.setSortingEnabled(False)
        self.list_clip_queue.clear()
        for i, c in enumerate(self._batch_clips):
            item = self._make_clip_queue_item(c, i)
            self.list_clip_queue.addTopLevelItem(item)
        self.list_clip_queue.setSortingEnabled(True)

        self._refresh_cls_annotation_stats()

        if self._batch_clips:
            self._advance_to_next_unlabeled_clip()
        else:
            self._lbl_preview_title.setText("")
            self.preview.set_frame(np.zeros((100, 100, 3), dtype=np.uint8))

    def _label_batch_clip(self, label_data: dict):
        idx = getattr(self, "_cls_playback_clip_idx", -1)
        if idx < 0 or idx >= len(self._batch_clips):
            return

        clip = self._batch_clips[idx]
        if not clip["clip_dir"]:
            return

        self._save_clip_label(
            clip["clip_dir"], label_data["obj_id"],
            label_data["behavior"], label_data.get("color", "#cccccc"),
        )

        clip["status"] = "labeled"
        clip["objects"][str(label_data["obj_id"])] = {
            "behavior": label_data["behavior"],
            "color": label_data.get("color", "#cccccc"),
        }

        item = self._clip_queue_item_by_idx(idx)
        if item:
            self._update_clip_queue_item(item, clip)

        self._refresh_cls_annotation_stats()
        self._advance_to_next_unlabeled_clip()

    def _delete_selected_clips(self):
        selected = self.list_clip_queue.selectedItems()
        if not selected:
            return
        reply = QMessageBox.question(
            self, "Delete Clips",
            f"Delete {len(selected)} clip(s) from disk?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._stop_clip_playback()
        self._cls_in_queue_mode = False
        self._cls_pending_clip_mode = False

        indices = sorted(
            [item.data(0, Qt.UserRole) for item in selected if item.data(0, Qt.UserRole) is not None],
            reverse=True,
        )
        next_row = min(indices) if indices else 0
        for idx in indices:
            if 0 <= idx < len(self._batch_clips):
                clip = self._batch_clips[idx]
                if clip.get("clip_dir"):
                    self._delete_clip_from_disk(clip["clip_dir"])
                self._batch_clips.pop(idx)

        # Rebuild queue list
        self.list_clip_queue.setSortingEnabled(False)
        self.list_clip_queue.clear()
        for idx, clip in enumerate(self._batch_clips):
            item = self._make_clip_queue_item(clip, idx)
            self.list_clip_queue.addTopLevelItem(item)
        self.list_clip_queue.setSortingEnabled(True)

        if self.list_clip_queue.topLevelItemCount() > 0:
            select_idx = min(next_row, self.list_clip_queue.topLevelItemCount() - 1)
            target = self._clip_queue_item_by_idx(select_idx)
            if target is not None:
                self.list_clip_queue.setCurrentItem(target)
        else:
            self.preview.setPixmap(QPixmap())

        self._refresh_cls_annotation_stats()

    def _delete_all_clips(self):
        if not self._batch_clips:
            return
        reply = QMessageBox.question(
            self, "Delete All Clips",
            f"Delete all {len(self._batch_clips)} clips from disk?\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._stop_clip_playback()
        self._cls_in_queue_mode = False
        self._cls_pending_clip_mode = False

        for clip in self._batch_clips:
            if clip.get("clip_dir"):
                self._delete_clip_from_disk(clip["clip_dir"])

        self._batch_clips = []
        self.list_clip_queue.clear()
        self._refresh_cls_annotation_stats()

    def _clear_all_clip_labels(self):
        labeled = [c for c in self._batch_clips if c["status"] == "labeled"]
        if not labeled:
            return
        n_annotations = sum(
            1 for c in labeled
            for v in c.get("objects", {}).values()
            if v.get("behavior")
        )
        reply = QMessageBox.warning(
            self, "Clear All Labels",
            f"Are you sure you want to clear {n_annotations} annotation(s) "
            f"from {len(labeled)} clip(s)?\n\n"
            f"All behavior assignments will be removed.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        for idx, clip in enumerate(self._batch_clips):
            if clip["status"] != "labeled":
                continue
            for v in clip.get("objects", {}).values():
                v["behavior"] = None
                v["color"] = None
            clip["status"] = "pending"

            if clip.get("clip_dir"):
                meta_path = os.path.join(clip["clip_dir"], "meta.json")
                if os.path.isfile(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    for v in meta.get("objects", {}).values():
                        v["behavior"] = None
                        v["color"] = None
                    meta["status"] = "pending"
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)

                import glob
                for comp in glob.glob(os.path.join(clip["clip_dir"], "composite_obj*.png")):
                    os.remove(comp)

            item = self._clip_queue_item_by_idx(idx)
            if item:
                self._update_clip_queue_item(item, clip)

        self._refresh_cls_annotation_stats()
        self.status_bar.showMessage(
            f"Cleared labels from {len(labeled)} clip(s).", 3000
        )

    def _reextract_clips(self):
        """Re-run the selected mask model on all existing clip positions."""
        model_idx = self.combo_cls_model.currentIndex()
        if model_idx < 0 or model_idx >= len(getattr(self, "_cls_model_dirs", [])):
            QMessageBox.warning(self, "No Model", "Select a mask model first.")
            return
        existing = [c for c in self._batch_clips if c.get("clip_dir")]
        if not existing:
            QMessageBox.warning(
                self, "No Clips",
                "No clips to re-extract. Extract clips first.",
            )
            return

        # Count labeled clips for the prompt
        n_labeled = sum(1 for c in existing if c.get("status") == "labeled")
        n_total = len(existing)

        # Prompt: keep labels or reset?
        msg = QMessageBox(self)
        msg.setWindowTitle("Re-extract Clips")
        msg.setIcon(QMessageBox.Question)
        if n_labeled > 0:
            msg.setText(
                f"Re-run the mask model on {n_total} clip(s) "
                f"({n_labeled} labeled).\n\n"
                "This will regenerate all silhouette masks using the "
                "currently selected model.\n\n"
                "How should existing behavior labels be handled?"
            )
            btn_keep = msg.addButton("Keep Labels", QMessageBox.AcceptRole)
            btn_reset = msg.addButton("Reset to Pending", QMessageBox.DestructiveRole)
        else:
            msg.setText(
                f"Re-run the mask model on {n_total} clip(s)?\n\n"
                "This will regenerate all silhouette masks using the "
                "currently selected model."
            )
            btn_keep = None
            btn_reset = msg.addButton("Re-extract", QMessageBox.AcceptRole)
        btn_cancel = msg.addButton(QMessageBox.Cancel)
        msg.exec_()

        if msg.clickedButton() == btn_cancel:
            return
        keep_labels = btn_keep is not None and msg.clickedButton() == btn_keep

        # Collect positions and old labels from existing clips
        positions = []
        old_labels = []
        for clip in existing:
            source = clip.get("source_video")
            start = clip.get("start_frame")
            if source and start is not None:
                positions.append((source, start))
                if keep_labels and clip.get("status") == "labeled":
                    old_labels.append(clip.get("objects", {}))
                else:
                    old_labels.append({})

        if not positions:
            return

        # Delete old clip directories
        import shutil
        for clip in existing:
            clip_dir = clip.get("clip_dir")
            if clip_dir and os.path.isdir(clip_dir):
                shutil.rmtree(clip_dir)

        # Clear queue
        self._batch_clips.clear()
        self.list_clip_queue.clear()

        # Store re-extraction state for applying labels after
        self._reextract_old_labels = old_labels
        self._reextract_keep_labels = keep_labels

        # Build placeholder items
        model_dir = self._cls_model_dirs[model_idx]
        clips_base = self._clips_base_dir()
        os.makedirs(clips_base, exist_ok=True)
        clip_length = self.spin_cls_clip_length.value()
        confidence = self.spin_cls_confidence.value()
        max_det = self.spin_cls_max_det.value() or 100

        self.list_clip_queue.setSortingEnabled(False)
        for idx, (vpath, start) in enumerate(positions):
            clip_data = {
                "clip_dir": None,
                "clip_id": None,
                "source_video": vpath,
                "start_frame": start,
                "clip_length": clip_length,
                "status": "extracting",
                "objects": {},
            }
            self._batch_clips.append(clip_data)
            item = self._make_clip_queue_item(clip_data, idx)
            self.list_clip_queue.addTopLevelItem(item)
        self.list_clip_queue.setSortingEnabled(True)

        self.btn_batch_extract.setEnabled(False)
        self.btn_reextract_clips.setEnabled(False)
        self.lbl_cls_annotation_stats.setText(
            f"Re-extracting {len(positions)} clips..."
        )

        self._batch_queue_offset = 0
        self._batch_worker = _BatchClipExtractWorker(
            model_dir, positions, clip_length, confidence, max_det,
            clips_base, 1,  # restart numbering from 1
        )
        self._batch_worker.clip_ready.connect(self._on_batch_clip_ready)
        self._batch_worker.all_done.connect(self._on_reextract_done)
        self._batch_worker.error.connect(self._on_batch_error)
        self._batch_worker.start()

    def _on_reextract_done(self):
        """Called when re-extraction finishes — re-apply labels if requested."""
        self.btn_batch_extract.setEnabled(True)
        self.btn_reextract_clips.setEnabled(True)

        old_labels = getattr(self, "_reextract_old_labels", [])
        keep = getattr(self, "_reextract_keep_labels", False)

        if keep and old_labels:
            applied = 0
            for idx, clip in enumerate(self._batch_clips):
                if idx >= len(old_labels):
                    break
                saved_objs = old_labels[idx]
                if not saved_objs:
                    continue
                clip_dir = clip.get("clip_dir")
                if not clip_dir:
                    continue

                new_objects = clip.get("objects", {})
                # Map old labels to new objects by position
                old_items = [
                    (k, v) for k, v in saved_objs.items()
                    if v.get("behavior")
                ]
                new_keys = list(new_objects.keys())
                for i, (old_key, old_val) in enumerate(old_items):
                    if i < len(new_keys):
                        target_key = new_keys[i]
                        behavior = old_val["behavior"]
                        color = old_val.get("color")
                        self._save_clip_label(
                            clip_dir, target_key, behavior, color,
                        )
                        clip["objects"][target_key] = {
                            "behavior": behavior, "color": color,
                        }
                        clip["status"] = "labeled"
                        applied += 1

                # Update queue item display
                item = self._clip_queue_item_by_idx(idx)
                if item:
                    self._update_clip_queue_item(item, clip)

            self.status_bar.showMessage(
                f"Re-extraction complete. Restored {applied} label(s).", 5000
            )
        else:
            self.status_bar.showMessage(
                f"Re-extraction complete. {len(self._batch_clips)} clips ready for labeling.",
                5000,
            )

        self._reextract_old_labels = []
        self._reextract_keep_labels = False
        self._refresh_cls_annotation_stats()

    def _start_classifier_training(self):
        composite_paths = []
        labels = []
        class_name_set = set()

        for clip in self._batch_clips:
            if clip["status"] != "labeled" or clip["clip_dir"] is None:
                continue
            for obj_id_str, obj_info in clip.get("objects", {}).items():
                beh = obj_info.get("behavior")
                if not beh:
                    continue
                comp_path = os.path.join(
                    clip["clip_dir"], f"composite_obj{obj_id_str}.png"
                )
                if os.path.isfile(comp_path):
                    composite_paths.append(comp_path)
                    class_name_set.add(beh)
                    labels.append(beh)

        class_names = sorted(class_name_set)
        if len(class_names) < 2:
            QMessageBox.warning(
                self, "Insufficient Data",
                "Need at least 2 behavior categories with composite images.\n"
                "Label more clips and ensure composites were generated."
            )
            return

        label_indices = [class_names.index(l) for l in labels]

        from datetime import datetime
        backbone_tag = self.combo_cls_backbone.currentText().replace("-", "").replace(" ", "")
        n_clips = len(composite_paths)
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{backbone_tag}_n={n_clips}"
        output_dir = os.path.join(
            self._project_dir, "action_classifier", "model", run_name
        )

        rot_idx = self.combo_cls_aug_rotation.currentIndex()
        aug_rotation = {0: 0, 1: 15, 2: 180}.get(rot_idx, 0)

        patience_val = self.spin_cls_patience.value()
        early_stop = patience_val < self.spin_cls_epochs.value()

        config = {
            "composite_paths": composite_paths,
            "labels": label_indices,
            "class_names": class_names,
            "epochs": self.spin_cls_epochs.value(),
            "lr": self.spin_cls_lr.value(),
            "batch_size": self.spin_cls_batch.value(),
            "val_split": self.spin_cls_val.value(),
            "augment": aug_rotation > 0,
            "aug_rotation": aug_rotation,
            "freeze_backbone": self.chk_cls_freeze.isChecked(),
            "early_stopping": early_stop,
            "patience": patience_val,
            "backbone": self.combo_cls_backbone.currentText(),
            "output_dir": output_dir,
        }

        self.btn_train_classifier.setEnabled(False)
        self.cls_train_progress.setMaximum(config["epochs"])
        self.cls_train_progress.setValue(0)
        self.cls_train_progress.setVisible(True)
        self.btn_cls_stop.setVisible(True)
        self.btn_cls_stop.setEnabled(True)
        self.lbl_cls_train_status.setText("Starting classifier training...")

        # Show loss plot in preview area
        self.preview.setVisible(False)
        self._info_row.setVisible(False)
        self._cls_nav_bar.setVisible(False)
        self._lbl_preview_title.setText("Training Preview")
        self._preview_title_row.setVisible(True)
        self._btn_toggle_training.setVisible(False)
        self._cls_training_viz_panel.setVisible(True)
        self._cls_loss_plot.reset(config["epochs"])
        self._cls_log_text.clear()

        self._cls_train_worker = _ClassifierTrainWorker(config)
        self._cls_train_worker.epoch_done.connect(self._on_cls_train_epoch)
        self._cls_train_worker.log_message.connect(self._on_cls_train_log)
        self._cls_train_worker.finished.connect(self._on_cls_train_finished)
        self._cls_train_worker.error.connect(self._on_cls_train_error)
        self._cls_train_worker.start()

    def _on_cls_train_log(self, msg):
        print(msg)
        self._cls_log_text.append(msg)
        self._cls_log_text.verticalScrollBar().setValue(
            self._cls_log_text.verticalScrollBar().maximum()
        )
        self.lbl_cls_train_status.setText(msg.replace("[Classifier] ", ""))

    def _copy_seg_train_log(self):
        from PyQt5.QtWidgets import QApplication
        text = self._seg_log_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_bar.showMessage("Training logs copied to clipboard.", 3000)

    def _copy_cls_train_log(self):
        from PyQt5.QtWidgets import QApplication
        text = self._cls_log_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_bar.showMessage("Training logs copied to clipboard.", 3000)

    def _copy_infer_log(self):
        from PyQt5.QtWidgets import QApplication
        text = self._infer_log_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status_bar.showMessage("Inference logs copied to clipboard.", 3000)

    def _on_infer_log(self, line: str):
        self._infer_log_text.append(line)
        sb = self._infer_log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _stop_cls_training(self):
        if hasattr(self, "_cls_train_worker") and self._cls_train_worker is not None and self._cls_train_worker.isRunning():
            self._cls_train_worker.request_stop()
            self.btn_cls_stop.setEnabled(False)
            self.lbl_cls_train_status.setText("Stopping training after current epoch...")

    def _on_cls_train_epoch(self, epoch, train_loss, val_loss, val_acc):
        self.cls_train_progress.setValue(epoch)
        self._cls_loss_plot.add_point(epoch, train_loss, val_loss, val_acc)
        parts = [f"Epoch {epoch}/{self.spin_cls_epochs.value()}",
                 f"train loss: {train_loss:.4f}"]
        if val_loss > 0:
            parts.append(f"val loss: {val_loss:.4f}")
        if val_acc > 0:
            parts.append(f"val acc: {val_acc:.1%}")
        self.lbl_cls_train_status.setText("  —  ".join(parts))

    def _on_cls_train_finished(self, summary):
        self.cls_train_progress.setVisible(False)
        self.btn_cls_stop.setVisible(False)
        self.btn_train_classifier.setEnabled(True)

        n_epochs = summary.get("epochs_trained", 0)
        best_loss = summary.get("best_val_loss")
        classes = summary.get("class_names", [])
        msg = f"Training complete — {n_epochs} epochs, {len(classes)} classes"
        if best_loss is not None:
            msg += f", best val loss: {best_loss:.4f}"
        self.lbl_cls_train_status.setText(msg)

        output_dir = summary.get("output_dir", "")
        self.status_bar.showMessage(
            f"Classifier saved to {output_dir}", 10000
        )

        # Restore preview area (keep loss plot visible until user navigates away)
        self._lbl_preview_title.setText("Training Complete")
        self._cls_train_worker = None

    def _on_cls_train_error(self, msg):
        self.cls_train_progress.setVisible(False)
        self.btn_cls_stop.setVisible(False)
        self.btn_train_classifier.setEnabled(True)
        self.lbl_cls_train_status.setText(f"Training error: {msg}")

        # Restore preview
        self._cls_training_viz_panel.setVisible(False)
        self._btn_toggle_training.setVisible(False)
        self.preview.setVisible(True)
        self._cls_nav_bar.setVisible(True)

        QMessageBox.critical(self, "Classifier Training Error", msg)

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
                    arch = cfg.get("architecture", "maskrcnn")
                    cats = cfg.get("categories", {})
                    cat_names = list(cats.values()) if isinstance(cats, dict) else cats
                    cat_str = ", ".join(str(c) for c in cat_names) if cat_names else "?"
                    if arch == "yolov11-seg":
                        variant = cfg.get("model_variant", "yolo11n-seg")
                        label = f"{entry}  [YOLO {variant}: {cat_str}]"
                    else:
                        backbone = cfg.get("backbone", "?")
                        label = f"{entry}  [{backbone}: {cat_str}]"
                except Exception:
                    pass

            self.combo_track_model.addItem(label)
            self._track_model_dirs.append(run_dir)

        if not self._track_model_dirs:
            self.lbl_track_model_info.setText("No trained models found — train a model first")
        else:
            last_idx = len(self._track_model_dirs) - 1
            self.combo_track_model.setCurrentIndex(last_idx)
            self._on_track_model_changed(last_idx)
        self._fit_combo_popup(self.combo_track_model)
        self._update_tracking_button_state()

    @staticmethod
    def _fit_combo_popup(combo: QComboBox):
        fm = combo.fontMetrics()
        max_w = 0
        for i in range(combo.count()):
            w = fm.horizontalAdvance(combo.itemText(i))
            if w > max_w:
                max_w = w
        combo.view().setMinimumWidth(max_w + 30)

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
                arch = cfg.get("architecture", "maskrcnn")
                if arch == "yolov11-seg":
                    variant = cfg.get("model_variant", "yolo11n-seg")
                    imgsz = cfg.get("imgsz", 640)
                    self.lbl_track_model_info.setText(
                        f"Path: {run_dir}\n"
                        f"Architecture: YOLO ({variant})  |  "
                        f"Classes: {', '.join(str(c) for c in cat_names) if cat_names else '?'}  |  "
                        f"Image size: {imgsz}px"
                    )
                    self.combo_track_resolution.setItemText(0, f"Trained ({imgsz}px)")
                    # YOLO should default to its trained resolution for best accuracy
                    self.combo_track_resolution.setCurrentIndex(0)
                    # YOLO produces masks at negligible cost — always enable
                    self.chk_track_masks.setChecked(True)
                    self.chk_track_masks.setEnabled(False)
                    self.chk_track_masks.setToolTip(
                        "Masks are always enabled for YOLO models (no speed penalty)."
                    )
                else:
                    max_sz = cfg.get("max_size", 800)
                    self.lbl_track_model_info.setText(
                        f"Path: {run_dir}\n"
                        f"Architecture: Mask R-CNN ({cfg.get('backbone', '?')})  |  "
                        f"Classes: {', '.join(str(c) for c in cat_names) if cat_names else '?'}  |  "
                        f"Min size: {cfg.get('min_size', '?')}"
                    )
                    self.combo_track_resolution.setItemText(0, f"Trained ({max_sz}px)")
                    # Mask R-CNN: user can choose (masks add ~48% cost)
                    self.chk_track_masks.setEnabled(True)
                    self.chk_track_masks.setToolTip(
                        "When enabled, the model also predicts pixel-level masks for each\n"
                        "detection and draws them on the annotated video. ~2x slower.\n"
                        "When disabled, uses bounding boxes only — faster and sufficient\n"
                        "for tracking since centroids are computed from box centers."
                    )
            except Exception:
                self.lbl_track_model_info.setText(f"Path: {run_dir}")
                self.combo_track_resolution.setItemText(0, "Trained (default)")
        else:
            self.lbl_track_model_info.setText(f"Path: {run_dir}")
            self.combo_track_resolution.setItemText(0, "Trained (default)")
        self._update_tracking_button_state()

    # -- Behavior Classification helpers --

    def _on_behavior_cls_toggled(self, enabled: bool):
        self._behavior_cls_container.setVisible(enabled)

    def _refresh_classifier_model_list(self):
        self.combo_behavior_model.clear()
        self._cls_model_dirs = []
        if not self._project_dir:
            self.lbl_behavior_model_info.setText("No project open")
            return

        cls_root = os.path.join(self._project_dir, "action_classifier", "model")
        if not os.path.isdir(cls_root):
            self.lbl_behavior_model_info.setText(
                "No classifier models found — train one in the Classifier tab"
            )
            return

        for entry in sorted(os.listdir(cls_root)):
            run_dir = os.path.join(cls_root, entry)
            cfg_path = os.path.join(run_dir, "classifier_config.json")
            if not os.path.isfile(cfg_path):
                continue
            best_path = os.path.join(run_dir, "best_classifier.pth")
            if not os.path.isfile(best_path):
                continue

            label = entry
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                classes = cfg.get("class_names", [])
                backbone = cfg.get("backbone", "?")
                n_cls = cfg.get("n_classes", len(classes))
                cls_str = ", ".join(classes) if classes else "?"
                label = f"{entry}  [{backbone}: {n_cls} classes — {cls_str}]"
            except Exception:
                pass

            self.combo_behavior_model.addItem(label)
            self._cls_model_dirs.append(run_dir)

        if not self._cls_model_dirs:
            self.lbl_behavior_model_info.setText(
                "No classifier models found — train one in the Classifier tab"
            )
            self.chk_behavior_cls.setChecked(False)
            self.chk_behavior_cls.setEnabled(False)
        else:
            self.chk_behavior_cls.setEnabled(True)
            last_idx = len(self._cls_model_dirs) - 1
            self.combo_behavior_model.setCurrentIndex(last_idx)
            run_dir = self._cls_model_dirs[last_idx]
            try:
                with open(os.path.join(run_dir, "classifier_config.json")) as f:
                    cfg = json.load(f)
                classes = cfg.get("class_names", [])
                self.lbl_behavior_model_info.setText(
                    f"Classes: {', '.join(classes)}  |  "
                    f"Backbone: {cfg.get('backbone', '?')}"
                )
            except Exception:
                self.lbl_behavior_model_info.setText(f"Path: {run_dir}")
            self._on_behavior_cls_toggled(self.chk_behavior_cls.isChecked())
        self._fit_combo_popup(self.combo_behavior_model)

    def _add_tracking_videos(self):
        had_none = self.list_track_videos.count() == 0
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
        if had_none and self.list_track_videos.count() > 0:
            self.list_track_videos.setCurrentRow(0)

    def _add_tracking_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if not folder:
            return
        had_none = self.list_track_videos.count() == 0
        for fname in sorted(os.listdir(folder)):
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                full = os.path.join(folder, fname)
                self.list_track_videos.addItem(fname)
                self.list_track_videos.item(
                    self.list_track_videos.count() - 1
                ).setData(Qt.UserRole, full)
        self._update_tracking_button_state()
        if had_none and self.list_track_videos.count() > 0:
            self.list_track_videos.setCurrentRow(0)

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
                f"Overwriting will clear all contents in the MaskTracker "
                f"output folder(s), including any ROI analysis saved there.\n\n"
                f"Overwrite existing results?",
                QMessageBox.Yes | QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return

            for vp in video_paths:
                out_dir = Path(vp).parent / f"{Path(vp).stem}_MaskTracker"
                if out_dir.is_dir():
                    shutil.rmtree(out_dir)

            roi_existing = []
            for vp in video_paths:
                stem = Path(vp).stem
                parent = Path(vp).parent
                for suffix in (f"{stem}_roiAnalysis", f"{stem}_roiAnalysis_SLEAP"):
                    roi_dir = parent / suffix
                    if roi_dir.is_dir() and any(roi_dir.iterdir()):
                        roi_existing.append(os.path.basename(vp))
                        break
            if roi_existing:
                roi_names = "\n".join(f"  • {n}" for n in roi_existing)
                roi_reply = QMessageBox.question(
                    self,
                    "ROI Analysis Found",
                    f"The following video(s) also have ROI analysis in a "
                    f"separate folder based on the previous tracking data:\n\n"
                    f"{roi_names}\n\n"
                    f"Clear the previous ROI analysis?\n"
                    f"(It will be invalid after re-tracking.)",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if roi_reply == QMessageBox.Yes:
                    for vp in video_paths:
                        stem = Path(vp).stem
                        parent = Path(vp).parent
                        for suffix in (f"{stem}_roiAnalysis", f"{stem}_roiAnalysis_SLEAP"):
                            roi_dir = parent / suffix
                            if roi_dir.is_dir():
                                shutil.rmtree(roi_dir)

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
            "max_detections": self.spin_track_max_det.value() or 100,
            "max_disappeared_frames": max_disappear,
            "iou_match_threshold": self.spin_track_iou.value(),
            "matching_algorithm": matching_algo,
            "inference_size": inference_size,
            "use_masks": self.chk_track_masks.isChecked(),
            "device": device,
        }

        if self.chk_behavior_cls.isChecked():
            config_dict["use_masks"] = True

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
        self.track_queue_progress.setMaximum(self._track_total_videos)
        self.track_queue_progress.setValue(0)
        self.track_queue_progress.setVisible(self._track_total_videos > 1)
        self.lbl_track_status.setText(
            f"Processing video 1 of {self._track_total_videos}: "
            f"{os.path.basename(video_paths[0])}"
        )

        self.preview.setEnabled(False)

        classifier_dir = None
        if self.chk_behavior_cls.isChecked():
            cls_idx = self.combo_behavior_model.currentIndex()
            if cls_idx >= 0 and cls_idx < len(self._cls_model_dirs):
                classifier_dir = self._cls_model_dirs[cls_idx]
            else:
                QMessageBox.warning(
                    self, "No Classifier Model",
                    "Behavior classification is enabled but no classifier model\n"
                    "is selected. Train a model in the Classifier tab first,\n"
                    "or uncheck 'Enable behavior classification'."
                )
                return

        self._inference_worker = InferenceWorker(
            video_paths, model_dir, config_dict,
            classifier_dir=classifier_dir,
            nc_threshold=self.spin_nc_threshold.value(),
            cls_window=self.spin_cls_window.value(),
            show_preview=self.chk_inference_preview.isChecked(),
            min_bout=self.spin_min_bout.value(),
            uncertain_gap=self.spin_uncertain_gap.value(),
            create_tracked_video=self.chk_create_tracked_video.isChecked(),
        )
        self._inference_worker.progress.connect(self._on_tracking_progress)
        self._inference_worker.frame_ready.connect(self._on_tracking_frame)
        self._inference_worker.video_finished.connect(self._on_video_finished)
        self._inference_worker.all_done.connect(self._on_all_tracking_done)
        self._inference_worker.error.connect(self._on_tracking_error)
        self._inference_worker.log.connect(self._on_infer_log)

        self._infer_log_text.clear()
        self._infer_log_panel.setVisible(True)
        self._inference_worker.start()

    def _on_tracking_frame(self, frame_rgb, frame_idx: int):
        """Display an annotated frame in the preview during live inference."""
        if frame_rgb is not None and self.tab_widget.currentIndex() == self._inference_tab_idx:
            self.preview.set_frame(frame_rgb)

    def _on_tracking_video_selected(self, current, previous):
        """Load first frame of selected video into preview."""
        if current is None:
            return
        video_path = current.data(Qt.UserRole)
        if not video_path or not os.path.exists(video_path):
            return
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.preview.set_frame(frame_rgb)
            duration = total / fps if fps > 0 else 0
            self.lbl_tracking_info.setText(
                f"{os.path.basename(video_path)}  —  "
                f"{w}x{h}, {total} frames, {fps:.0f}fps, {duration:.1f}s"
            )

    def _on_tracking_progress(self, frame_idx: int, total_frames: int):
        self.track_progress.setMaximum(total_frames)
        self.track_progress.setValue(frame_idx)
        vid_num = self._track_videos_done + 1
        vid_name = os.path.basename(self._track_video_paths[self._track_videos_done])
        status_text = (
            f"Processing video {vid_num} of {self._track_total_videos}: {vid_name}\n"
            f"Frame {frame_idx}/{total_frames}"
        )
        self.lbl_track_status.setText(status_text)
        self.lbl_tracking_info.setText(
            f"Frame {frame_idx}/{total_frames}  —  {vid_name}"
        )

    def _on_video_finished(self, result: dict):
        self._track_videos_done += 1
        self.track_queue_progress.setValue(self._track_videos_done)
        video_name = os.path.basename(result.get("video_path", ""))
        n_tracks = result.get("num_tracks", 0)
        output_dir = result.get("output_dir", "")
        total_frames = result.get("total_frames", 0)

        summary = f"{video_name}: {n_tracks} tracks, {total_frames} frames"
        if n_tracks == 0:
            summary += " ⚠ No detections — try lowering confidence threshold"
        if result.get("behavior_csv"):
            summary += " + behavior"
        summary += f" → {os.path.basename(output_dir)}/"

        item = QListWidgetItem(summary)
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
        self.track_queue_progress.setVisible(False)
        self.btn_start_tracking.setEnabled(True)
        self.btn_track_pause.setVisible(False)
        self.btn_track_stop.setVisible(False)
        self.preview.setEnabled(True)
        self.lbl_track_status.setText(
            f"Done — {self._track_videos_done} video(s) processed"
        )
        self.lbl_tracking_info.setText(
            f"Inference complete — {self._track_videos_done} video(s) processed"
        )
        self.status_bar.showMessage(
            f"Inference complete: {self._track_videos_done} video(s)"
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
        self.track_queue_progress.setVisible(False)
        self.btn_start_tracking.setEnabled(True)
        self.btn_track_pause.setVisible(False)
        self.btn_track_stop.setVisible(False)
        self.preview.setEnabled(True)
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
        # Tab indices: 0=Annotate, 1=Classify, 2=Infer
        is_annotation = index == 0
        is_classifier = index == 1
        is_tracking = index == 2

        self.preview.annotation_keys_enabled = is_annotation
        if not is_annotation:
            self.preview._clear_drawing()
            self.preview.drawing_mode = "navigate"
            self.preview._editing_obj_idx = None

        # Annotation bar: only on Annotate tab
        for w in self._annotation_bar_widgets:
            w.setVisible(is_annotation)
        # Classifier info + Edit Behaviors button: only on Classify tab
        self.lbl_classifier_info.setVisible(is_classifier)
        self.btn_edit_behaviors.setVisible(is_classifier)
        # Tracking info label: only on Infer tab
        self.lbl_tracking_info.setVisible(is_tracking)
        # Classifier: stop playback and clear mask handler when leaving tab
        if not is_classifier:
            self.preview._cls_mask_handler = None
            self._stop_clip_playback()

        # Check if classifier training is in progress
        cls_training_active = (
            hasattr(self, "_cls_train_worker")
            and self._cls_train_worker is not None
            and self._cls_train_worker.isRunning()
        )
        cls_show_loss = is_classifier and cls_training_active
        cls_graph_visible = is_classifier and self._cls_training_viz_panel.isVisible()

        # Frame slider + preview title: only on Classify tab
        self._cls_nav_bar.setVisible(
            is_classifier and not cls_training_active and not cls_graph_visible
        )
        self._preview_title_row.setVisible(is_classifier)
        # Check if mask model training is in progress
        mask_training_active = (
            hasattr(self, "_train_worker")
            and self._train_worker is not None
            and self._train_worker.isRunning()
        )
        mask_show_viz = is_annotation and mask_training_active

        # Classifier training loss panel
        self._cls_training_viz_panel.setVisible(cls_show_loss or cls_graph_visible)
        # Mask training visualization panel
        self._training_viz_panel.setVisible(mask_show_viz)
        # Preview + info row: hidden during any training viz
        show_preview = not cls_show_loss and not cls_graph_visible and not mask_show_viz
        self.preview.setVisible(show_preview)
        self._info_row.setVisible(show_preview)

        if is_annotation:
            self._refresh_training_summary()
            self._refresh_device_label()
            if self.current_frame_idx >= 0 and self.current_frame_idx < len(self._extracted_frames):
                _, _, path = self._extracted_frames[self.current_frame_idx]
                self._load_annotation_frame(path)
        elif is_classifier:
            self.preview.annotations.clear()
            self.preview._cls_mask_handler = self._on_cls_mask_right_click
            self._refresh_classifier_models()
            if not getattr(self, "_cls_data_loaded", False):
                self._load_classifier_data_from_disk()
            self._refresh_cls_annotation_stats()
            current = self.list_cls_videos.currentItem()
            if current:
                self._on_cls_video_selected(current, None)
            else:
                self.preview.clear()
                self.preview.update()
                self.lbl_classifier_info.setText(
                    "Add videos to begin annotating behaviors"
                )
            self.setFocus()
        elif is_tracking:
            if self._project_dir is None:
                reply = QMessageBox.question(
                    self, "No Project Loaded",
                    "No project is loaded. Tracking requires a project with "
                    "trained models.\n\nWould you like to open an existing project?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self._open_project()
            self._refresh_model_list()
            self._refresh_classifier_model_list()
            # Show first frame of selected video, or clear
            current = self.list_track_videos.currentItem()
            if current:
                self._on_tracking_video_selected(current, None)
            else:
                self.preview.clear()
                self.preview.update()
                self.lbl_tracking_info.setText("Add videos to begin inference")

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
        n_ann = stats["num_annotations"]
        n_inferred = sum(
            1 for a in self._coco.annotations if a.get("inferred", False)
        )
        n_approved = n_ann - n_inferred
        cats = stats["categories"]
        if n_inferred > 0:
            self.lbl_train_annotations.setText(
                f"Total annotations: {n_ann}  ({n_approved} approved, {n_inferred} inferred)"
            )
        else:
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
        self._refresh_training_summary()
        stats = self._coco.get_stats()
        n_this_frame = len(self.preview.annotations)
        n_inferred_frame = sum(1 for a in self.preview.annotations if a.inferred)
        if stats["num_annotations"] == 0:
            self.lbl_ann_stats.setText("")
        else:
            n_total_inferred = sum(
                1 for a in self._coco.annotations if a.get("inferred", False)
            )
            frame_str = f"{n_this_frame} on frame"
            if n_inferred_frame > 0:
                frame_str += f" ({n_inferred_frame} inferred)"
            total_str = f"{stats['num_annotations']} total across {stats['images_with_annotations']} images"
            if n_total_inferred > 0:
                total_str += f" ({n_total_inferred} inferred)"
            self.lbl_ann_stats.setText(f"{frame_str}  |  {total_str}")
            if n_inferred_frame > 0:
                self.lbl_ann_stats.setStyleSheet("color: #e6c830; font-size: 10px;")
            else:
                self.lbl_ann_stats.setStyleSheet("color: #4caf50; font-size: 10px;")
        self._update_frame_list_item(self.current_frame_idx)

    def _update_frame_list_item(self, row: int):
        if row < 0 or row >= len(self._extracted_frames):
            return
        # Find the tree item with this frame index
        item = None
        for i in range(self.frame_list.topLevelItemCount()):
            candidate = self.frame_list.topLevelItem(i)
            if candidate.data(0, Qt.UserRole) == row:
                item = candidate
                break
        if item is None:
            return
        _, _, fp = self._extracted_frames[row]
        filename = os.path.basename(fp)
        n_total, n_inferred = self._count_annotations_for_file(filename)
        n_approved = n_total - n_inferred
        if n_total > 0 and n_inferred == n_total:
            item.setText(0, f"● {filename} ({n_total})")
            item.setForeground(0, QColor("#e6c830"))
        elif n_total > 0 and n_inferred > 0:
            item.setText(0, f"◐ {filename} ({n_approved}+{n_inferred})")
            item.setForeground(0, QColor("#e6c830"))
        elif n_total > 0:
            item.setText(0, f"✔ {filename} ({n_total})")
            item.setForeground(0, QColor("#4fc456"))
        else:
            item.setText(0, f"   {filename}")
            item.setForeground(0, QColor("#cccccc"))

        conf = self._frame_confidence.get(filename)
        item.setText(1, f"{conf:.2f}" if conf is not None else "")
        item.setData(1, Qt.UserRole, conf if conf is not None else -1.0)
        if conf is not None:
            if conf < 0.3:
                item.setForeground(1, QColor("#e53935"))
            elif conf < 0.6:
                item.setForeground(1, QColor("#e6c830"))
            else:
                item.setForeground(1, QColor("#4fc456"))

    def closeEvent(self, event):
        self._close_video()
        # Annotations are auto-saved on every action, so no prompt needed.
        event.accept()
