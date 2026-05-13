"""Mask Tracker Pipeline — integrated SAM2 annotation, Mask R-CNN training, and inference GUI.

Single-window tool with three tabs (Annotate / Train / Track) on the left and
a shared image/video preview panel on the right.

Inspired by LabGym (Bing Ye Lab, U. Michigan) and EZannot (Yujia Hu).
All code written from scratch for FieldNeuroToolbox.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QListWidget, QListWidgetItem, QTabWidget, QSplitter, QFrame,
    QTextEdit, QSizePolicy, QScrollArea, QInputDialog, QSlider,
    QDialog,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

DARK_STYLE = """
    QMainWindow { background-color: #2b2b2b; color: #cccccc; }
    QWidget { background-color: #2b2b2b; color: #cccccc; }
    QPushButton {
        background-color: #0078d4; color: white; border: none;
        padding: 8px 16px; border-radius: 4px; font-weight: bold; min-height: 20px;
    }
    QPushButton:hover { background-color: #106ebe; }
    QPushButton:pressed { background-color: #005a9e; }
    QPushButton:disabled { background-color: #3f3f3f; color: #888888; }
    QGroupBox {
        font-weight: bold; border: 1px solid #3f3f3f; border-radius: 4px;
        margin-top: 10px; padding-top: 8px; color: #cccccc;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
    QLabel { color: #cccccc; background-color: transparent; }
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        padding: 5px; border: 1px solid #3f3f3f; border-radius: 3px;
        background-color: #1e1e1e; color: #cccccc;
    }
    QTextEdit {
        background-color: #1e1e1e; border: 1px solid #3f3f3f; color: #cccccc;
    }
    QProgressBar {
        border: 1px solid #3f3f3f; border-radius: 3px; text-align: center;
        background-color: #1e1e1e; color: #cccccc;
    }
    QProgressBar::chunk { background-color: #0078d4; }
    QTabWidget::pane { border: 1px solid #3f3f3f; background-color: #2b2b2b; }
    QTabBar::tab {
        background-color: #1e1e1e; color: #cccccc; padding: 8px 16px;
        margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px;
    }
    QTabBar::tab:selected { background-color: #2b2b2b; border-bottom: 2px solid #0078d4; }
    QTabBar::tab:hover:!selected { background-color: #3f3f3f; }
    QListWidget {
        background-color: #1e1e1e; border: 1px solid #3f3f3f; color: #cccccc;
    }
    QListWidget::item:selected { background-color: #0078d4; }
    QFrame { background-color: #2b2b2b; border-color: #3f3f3f; }
    QSlider::groove:horizontal {
        border: 1px solid #3f3f3f; height: 6px; background: #1e1e1e; border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #0078d4; width: 14px; margin: -4px 0; border-radius: 7px;
    }
    QScrollBar:vertical {
        background-color: #2b2b2b; width: 12px; border-radius: 6px;
    }
    QScrollBar::handle:vertical {
        background-color: #0078d4; border-radius: 4px; min-height: 20px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""

STOP_BUTTON_STYLE = """
    QPushButton {
        background-color: #c42b1c; color: white; border: none;
        padding: 8px 16px; border-radius: 4px; font-weight: bold; min-height: 20px;
    }
    QPushButton:hover { background-color: #a52314; }
    QPushButton:pressed { background-color: #8b1d10; }
    QPushButton:disabled { background-color: #3f3f3f; color: #888888; }
"""


# ──────────────────────────────────────────────────────────────────────
# Preview widget
# ──────────────────────────────────────────────────────────────────────
class PreviewWidget(QLabel):
    """Image preview with click interaction for annotation."""

    left_click = pyqtSignal(int, int)
    right_click = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(500, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3f3f3f;")
        self._original = None
        self._overlay_mask = None
        self._overlay_color = (0, 120, 212, 100)
        self._points_pos: List[Tuple[int, int]] = []
        self._points_neg: List[Tuple[int, int]] = []
        self._annotations: List[Tuple[np.ndarray, int, str]] = []
        self._click_enabled = False

    def set_image(self, image_rgb: np.ndarray):
        self._original = image_rgb.copy()
        self._overlay_mask = None
        self._points_pos.clear()
        self._points_neg.clear()
        self._annotations.clear()
        self._refresh()

    def set_mask_overlay(self, mask: Optional[np.ndarray]):
        self._overlay_mask = mask
        self._refresh()

    def set_existing_annotations(self, annotations: List[Tuple[np.ndarray, int, str]]):
        self._annotations = annotations
        self._refresh()

    def clear_points(self):
        self._points_pos.clear()
        self._points_neg.clear()
        self._overlay_mask = None
        self._refresh()

    def set_click_enabled(self, enabled: bool):
        self._click_enabled = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def _refresh(self):
        if self._original is None:
            return
        display = self._original.copy()

        for ann_mask, ann_cat, ann_name in self._annotations:
            color = self._category_color(ann_cat)
            overlay = display.copy()
            overlay[ann_mask] = color[:3]
            display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)
            contours, _ = cv2.findContours(ann_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, color[:3], 2)

        if self._overlay_mask is not None:
            overlay = display.copy()
            overlay[self._overlay_mask] = [0, 120, 212]
            display = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)
            contours, _ = cv2.findContours(
                self._overlay_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(display, contours, -1, (0, 200, 255), 2)

        for px, py in self._points_pos:
            cv2.circle(display, (px, py), 6, (0, 255, 0), -1)
            cv2.circle(display, (px, py), 7, (255, 255, 255), 1)
        for px, py in self._points_neg:
            cv2.circle(display, (px, py), 6, (255, 0, 0), -1)
            cv2.circle(display, (px, py), 7, (255, 255, 255), 1)

        h, w, _ = display.shape
        qimg = QImage(display.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

    def _widget_to_image(self, wx: int, wy: int) -> Optional[Tuple[int, int]]:
        if self._original is None or self.pixmap() is None:
            return None
        pm = self.pixmap()
        ox = (self.width() - pm.width()) // 2
        oy = (self.height() - pm.height()) // 2
        px, py = wx - ox, wy - oy
        if px < 0 or py < 0 or px >= pm.width() or py >= pm.height():
            return None
        ih, iw = self._original.shape[:2]
        ix = int(px * iw / pm.width())
        iy = int(py * ih / pm.height())
        return (max(0, min(ix, iw - 1)), max(0, min(iy, ih - 1)))

    def _category_color(self, cat_id: int) -> Tuple[int, int, int]:
        colors = [
            (0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (255, 0, 128), (0, 128, 255),
        ]
        return colors[(cat_id - 1) % len(colors)]

    def mousePressEvent(self, event):
        if not self._click_enabled:
            return
        coords = self._widget_to_image(event.x(), event.y())
        if coords is None:
            return
        if event.button() == Qt.LeftButton:
            self._points_pos.append(coords)
            self.left_click.emit(*coords)
        elif event.button() == Qt.RightButton:
            self._points_neg.append(coords)
            self.right_click.emit(*coords)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


# ──────────────────────────────────────────────────────────────────────
# Worker threads
# ──────────────────────────────────────────────────────────────────────
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
    mask_ready = pyqtSignal(object, float)
    error = pyqtSignal(str)

    def __init__(self, segmenter, pos_points, neg_points):
        super().__init__()
        self.segmenter = segmenter
        self.pos_points = pos_points
        self.neg_points = neg_points

    def run(self):
        try:
            mask, score = self.segmenter.predict_mask(self.pos_points, self.neg_points)
            self.mask_ready.emit(mask, score)
        except Exception as e:
            self.error.emit(str(e))


class TrainingWorker(QThread):
    progress_signal = pyqtSignal(int, int, dict)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from .mask_tracker_training import train_mask_rcnn
            summary = train_mask_rcnn(
                self.config,
                progress=lambda it, total, m: self.progress_signal.emit(it, total, m),
                should_stop=lambda: self._stop,
            )
            self.finished_signal.emit(summary)
        except Exception as e:
            self.error_signal.emit(str(e))


class InferenceWorker(QThread):
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, model_dir, config):
        super().__init__()
        self.video_path = video_path
        self.model_dir = model_dir
        self.config = config
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from .mask_tracker_inference import run_inference_on_video
            result = run_inference_on_video(
                self.video_path,
                self.model_dir,
                self.config,
                progress=lambda f, t: self.progress_signal.emit(f, t),
                should_stop=lambda: self._stop,
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))


# ──────────────────────────────────────────────────────────────────────
# Main GUI
# ──────────────────────────────────────────────────────────────────────
class MaskTrackerPipelineGUI(QMainWindow):
    """Single-window mask tracker pipeline with Annotate / Train / Track tabs."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask Tracker Pipeline - FieldNeuroToolbox")
        self.setGeometry(80, 80, 1300, 850)
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(DARK_STYLE)

        # State
        self._segmenter = None
        self._sam2_worker = None
        self._predict_worker = None
        self._training_worker = None
        self._inference_worker = None

        self._image_dir = ""
        self._image_files: List[str] = []
        self._current_image_idx = -1
        self._current_mask: Optional[np.ndarray] = None

        self._coco_manager = None
        self._exported_json_path = ""
        self._trained_model_dir = ""

        self._video_paths: List[str] = []

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # Header
        header = QFrame()
        header.setFrameStyle(QFrame.Box | QFrame.Raised)
        header.setStyleSheet("background-color: #1e1e1e; padding: 10px; border: 1px solid #3f3f3f;")
        header_layout = QVBoxLayout()
        header.setLayout(header_layout)
        title = QLabel("Mask Tracker Pipeline")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        header_layout.addWidget(title)
        subtitle = QLabel("SAM2-assisted annotation, Mask R-CNN training, and video tracking")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet("color: #999999; font-style: italic; background-color: transparent;")
        header_layout.addWidget(subtitle)
        main_layout.addWidget(header)

        # Splitter: left tabs | right preview
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)

        # Left: tabs
        self.tabs = QTabWidget()
        self._build_annotate_tab()
        self._build_train_tab()
        self._build_track_tab()
        splitter.addWidget(self.tabs)

        # Right: preview
        self.preview = PreviewWidget()
        self.preview.left_click.connect(self._on_preview_left_click)
        self.preview.right_click.connect(self._on_preview_right_click)
        splitter.addWidget(self.preview)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([400, 900])
        main_layout.addWidget(splitter)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #999999; padding: 4px;")
        main_layout.addWidget(self.status_label)

    # ── Annotate tab ──────────────────────────────────────────────────
    def _build_annotate_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        content.setLayout(layout)
        scroll.setWidget(content)
        self.tabs.addTab(scroll, "Annotate")

        # SAM2 model group
        model_group = QGroupBox("SAM2 Model")
        model_layout = QVBoxLayout()
        self.btn_select_model = QPushButton("Select SAM2 Model")
        self.btn_select_model.clicked.connect(self._select_sam2_model)
        model_layout.addWidget(self.btn_select_model)
        self.lbl_model_status = QLabel("No model loaded")
        self.lbl_model_status.setStyleSheet("color: #999999;")
        model_layout.addWidget(self.lbl_model_status)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Images group
        images_group = QGroupBox("Training Images")
        images_layout = QVBoxLayout()
        self.btn_load_images = QPushButton("Load Images Folder")
        self.btn_load_images.clicked.connect(self._load_images_folder)
        images_layout.addWidget(self.btn_load_images)
        self.btn_load_existing = QPushButton("Load Existing Annotations")
        self.btn_load_existing.clicked.connect(self._load_existing_annotations)
        images_layout.addWidget(self.btn_load_existing)
        self.lbl_images_count = QLabel("No images loaded")
        self.lbl_images_count.setStyleSheet("color: #999999;")
        images_layout.addWidget(self.lbl_images_count)

        nav_row = QHBoxLayout()
        self.btn_prev = QPushButton("<")
        self.btn_prev.setMaximumWidth(40)
        self.btn_prev.clicked.connect(self._prev_image)
        self.btn_prev.setEnabled(False)
        nav_row.addWidget(self.btn_prev)
        self.lbl_image_idx = QLabel("0 / 0")
        self.lbl_image_idx.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_image_idx)
        self.btn_next = QPushButton(">")
        self.btn_next.setMaximumWidth(40)
        self.btn_next.clicked.connect(self._next_image)
        self.btn_next.setEnabled(False)
        nav_row.addWidget(self.btn_next)
        images_layout.addLayout(nav_row)
        images_group.setLayout(images_layout)
        layout.addWidget(images_group)

        # Classes group
        classes_group = QGroupBox("Class Labels")
        classes_layout = QVBoxLayout()
        self.list_classes = QListWidget()
        self.list_classes.setMaximumHeight(100)
        classes_layout.addWidget(self.list_classes)
        cls_btn_row = QHBoxLayout()
        self.btn_add_class = QPushButton("Add Class")
        self.btn_add_class.clicked.connect(self._add_class)
        cls_btn_row.addWidget(self.btn_add_class)
        self.btn_remove_class = QPushButton("Remove")
        self.btn_remove_class.clicked.connect(self._remove_class)
        cls_btn_row.addWidget(self.btn_remove_class)
        classes_layout.addLayout(cls_btn_row)
        classes_group.setLayout(classes_layout)
        layout.addWidget(classes_group)

        # Annotation tools
        tools_group = QGroupBox("Annotation Tools")
        tools_layout = QVBoxLayout()
        info = QLabel("Left-click: positive point | Right-click: negative point")
        info.setStyleSheet("color: #999999; font-style: italic;")
        info.setWordWrap(True)
        tools_layout.addWidget(info)

        self.btn_add_object = QPushButton("Start Annotating Object")
        self.btn_add_object.clicked.connect(self._start_annotating)
        self.btn_add_object.setEnabled(False)
        tools_layout.addWidget(self.btn_add_object)

        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("Class:"))
        self.combo_class = QComboBox()
        cls_row.addWidget(self.combo_class)
        tools_layout.addLayout(cls_row)

        btn_row = QHBoxLayout()
        self.btn_accept = QPushButton("Accept Mask")
        self.btn_accept.clicked.connect(self._accept_annotation)
        self.btn_accept.setEnabled(False)
        btn_row.addWidget(self.btn_accept)
        self.btn_clear_pts = QPushButton("Clear Points")
        self.btn_clear_pts.clicked.connect(self._clear_annotation_points)
        self.btn_clear_pts.setEnabled(False)
        btn_row.addWidget(self.btn_clear_pts)
        tools_layout.addLayout(btn_row)

        self.lbl_mask_score = QLabel("")
        self.lbl_mask_score.setStyleSheet("color: #999999;")
        tools_layout.addWidget(self.lbl_mask_score)
        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)

        # Current image annotations
        ann_group = QGroupBox("Objects in Current Image")
        ann_layout = QVBoxLayout()
        self.list_annotations = QListWidget()
        self.list_annotations.setMaximumHeight(120)
        ann_layout.addWidget(self.list_annotations)
        self.btn_delete_ann = QPushButton("Delete Selected")
        self.btn_delete_ann.clicked.connect(self._delete_annotation)
        ann_layout.addWidget(self.btn_delete_ann)
        ann_group.setLayout(ann_layout)
        layout.addWidget(ann_group)

        # Export
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        self.btn_export = QPushButton("Export COCO JSON")
        self.btn_export.clicked.connect(self._export_annotations)
        self.btn_export.setEnabled(False)
        export_layout.addWidget(self.btn_export)
        self.lbl_export_status = QLabel("")
        self.lbl_export_status.setStyleSheet("color: #999999;")
        export_layout.addWidget(self.lbl_export_status)
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        layout.addStretch()

    # ── Train tab ─────────────────────────────────────────────────────
    def _build_train_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        content.setLayout(layout)
        scroll.setWidget(content)
        self.tabs.addTab(scroll, "Train")

        # Dataset group
        ds_group = QGroupBox("Dataset")
        ds_layout = QVBoxLayout()
        self.btn_browse_coco = QPushButton("Browse COCO JSON")
        self.btn_browse_coco.clicked.connect(self._browse_coco_json)
        ds_layout.addWidget(self.btn_browse_coco)
        self.lbl_dataset_info = QLabel("No dataset loaded")
        self.lbl_dataset_info.setStyleSheet("color: #999999;")
        self.lbl_dataset_info.setWordWrap(True)
        ds_layout.addWidget(self.lbl_dataset_info)
        ds_group.setLayout(ds_layout)
        layout.addWidget(ds_group)

        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Iterations:"))
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(50, 50000)
        self.spin_iterations.setValue(1000)
        row.addWidget(self.spin_iterations)
        params_layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Learning rate:"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 0.1)
        self.spin_lr.setValue(0.005)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setSingleStep(0.001)
        row2.addWidget(self.spin_lr)
        params_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Batch size:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 16)
        self.spin_batch.setValue(2)
        row3.addWidget(self.spin_batch)
        params_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Device:"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(["auto", "cuda", "mps", "cpu"])
        row4.addWidget(self.combo_device)
        params_layout.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("Output directory:"))
        self.txt_train_output = QLineEdit()
        self.txt_train_output.setPlaceholderText("Select output directory...")
        self.txt_train_output.setReadOnly(True)
        row5.addWidget(self.txt_train_output)
        self.btn_browse_output = QPushButton("...")
        self.btn_browse_output.setMaximumWidth(40)
        self.btn_browse_output.clicked.connect(self._browse_train_output)
        row5.addWidget(self.btn_browse_output)
        params_layout.addLayout(row5)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Training controls
        ctrl_row = QHBoxLayout()
        self.btn_start_train = QPushButton("Start Training")
        self.btn_start_train.clicked.connect(self._start_training)
        self.btn_start_train.setEnabled(False)
        ctrl_row.addWidget(self.btn_start_train)
        self.btn_stop_train = QPushButton("Stop")
        self.btn_stop_train.setStyleSheet(STOP_BUTTON_STYLE)
        self.btn_stop_train.clicked.connect(self._stop_training)
        self.btn_stop_train.setEnabled(False)
        ctrl_row.addWidget(self.btn_stop_train)
        layout.addLayout(ctrl_row)

        # Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        self.lbl_train_progress = QLabel("Ready")
        progress_layout.addWidget(self.lbl_train_progress)
        self.progress_train = QProgressBar()
        self.progress_train.setVisible(False)
        progress_layout.addWidget(self.progress_train)
        self.txt_train_log = QTextEdit()
        self.txt_train_log.setReadOnly(True)
        self.txt_train_log.setFont(QFont("Courier New", 9))
        self.txt_train_log.setMaximumHeight(250)
        progress_layout.addWidget(self.txt_train_log)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        layout.addStretch()

    # ── Track tab ─────────────────────────────────────────────────────
    def _build_track_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        content.setLayout(layout)
        scroll.setWidget(content)
        self.tabs.addTab(scroll, "Track")

        # Model group
        model_group = QGroupBox("Trained Model")
        model_layout = QVBoxLayout()
        self.btn_browse_model = QPushButton("Browse Model Directory")
        self.btn_browse_model.clicked.connect(self._browse_model_dir)
        model_layout.addWidget(self.btn_browse_model)
        self.lbl_model_info = QLabel("No model loaded")
        self.lbl_model_info.setStyleSheet("color: #999999;")
        self.lbl_model_info.setWordWrap(True)
        model_layout.addWidget(self.lbl_model_info)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Video group
        video_group = QGroupBox("Videos")
        video_layout = QVBoxLayout()
        vid_btn_row = QHBoxLayout()
        self.btn_add_videos = QPushButton("Add Videos")
        self.btn_add_videos.clicked.connect(self._add_track_videos)
        vid_btn_row.addWidget(self.btn_add_videos)
        self.btn_add_video_folder = QPushButton("Add Folder")
        self.btn_add_video_folder.clicked.connect(self._add_track_folder)
        vid_btn_row.addWidget(self.btn_add_video_folder)
        self.btn_clear_videos = QPushButton("Clear")
        self.btn_clear_videos.clicked.connect(self._clear_track_videos)
        vid_btn_row.addWidget(self.btn_clear_videos)
        video_layout.addLayout(vid_btn_row)
        self.list_videos = QListWidget()
        self.list_videos.setMaximumHeight(120)
        video_layout.addWidget(self.list_videos)
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # Inference settings
        settings_group = QGroupBox("Inference Settings")
        settings_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Confidence:"))
        self.spin_confidence = QDoubleSpinBox()
        self.spin_confidence.setRange(0.05, 0.99)
        self.spin_confidence.setValue(0.5)
        self.spin_confidence.setSingleStep(0.05)
        row.addWidget(self.spin_confidence)
        settings_layout.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Max disappear:"))
        self.spin_disappear = QSpinBox()
        self.spin_disappear.setRange(1, 300)
        self.spin_disappear.setValue(30)
        row2.addWidget(self.spin_disappear)
        row2.addWidget(QLabel("frames"))
        settings_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("IoU match threshold:"))
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.05, 0.95)
        self.spin_iou.setValue(0.3)
        self.spin_iou.setSingleStep(0.05)
        row3.addWidget(self.spin_iou)
        settings_layout.addLayout(row3)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Track controls
        ctrl_row = QHBoxLayout()
        self.btn_start_track = QPushButton("Start Tracking")
        self.btn_start_track.clicked.connect(self._start_tracking)
        self.btn_start_track.setEnabled(False)
        ctrl_row.addWidget(self.btn_start_track)
        self.btn_stop_track = QPushButton("Stop")
        self.btn_stop_track.setStyleSheet(STOP_BUTTON_STYLE)
        self.btn_stop_track.clicked.connect(self._stop_tracking)
        self.btn_stop_track.setEnabled(False)
        ctrl_row.addWidget(self.btn_stop_track)
        layout.addLayout(ctrl_row)

        # Progress
        progress_group = QGroupBox("Tracking Progress")
        progress_layout = QVBoxLayout()
        self.lbl_track_progress = QLabel("Ready")
        progress_layout.addWidget(self.lbl_track_progress)
        self.progress_track = QProgressBar()
        self.progress_track.setVisible(False)
        progress_layout.addWidget(self.progress_track)
        self.txt_track_log = QTextEdit()
        self.txt_track_log.setReadOnly(True)
        self.txt_track_log.setFont(QFont("Courier New", 9))
        self.txt_track_log.setMaximumHeight(250)
        progress_layout.addWidget(self.txt_track_log)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        layout.addStretch()

    # ══════════════════════════════════════════════════════════════════
    # ANNOTATE TAB LOGIC
    # ══════════════════════════════════════════════════════════════════
    def _select_sam2_model(self):
        try:
            from .sam2_checkpoint_manager import SAM2CheckpointDialog
            dialog = SAM2CheckpointDialog(parent=self)
            if dialog.exec_() == QDialog.Accepted:
                checkpoint_path = dialog.get_checkpoint_path()
                config_name = dialog.get_config_name()
                if checkpoint_path is None:
                    return

                from .mask_tracker_annotator import SAM2ImageSegmenter
                self._segmenter = SAM2ImageSegmenter(str(checkpoint_path), config_name)
                self.lbl_model_status.setText(f"Loading {checkpoint_path.name}...")
                self.lbl_model_status.setStyleSheet("color: #cccccc;")
                self.btn_select_model.setEnabled(False)

                self._sam2_worker = SAM2LoadWorker(self._segmenter)
                self._sam2_worker.finished.connect(self._on_sam2_loaded)
                self._sam2_worker.error.connect(self._on_sam2_error)
                self._sam2_worker.start()
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"SAM2 not available: {e}")

    def _on_sam2_loaded(self):
        self.lbl_model_status.setText(f"Model loaded on {self._segmenter.device}")
        self.lbl_model_status.setStyleSheet("color: #00cc00;")
        self.btn_select_model.setEnabled(True)
        self._update_annotate_ready()

    def _on_sam2_error(self, msg):
        self.lbl_model_status.setText(f"Error: {msg}")
        self.lbl_model_status.setStyleSheet("color: #ff4444;")
        self.btn_select_model.setEnabled(True)
        self._segmenter = None

    def _load_images_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Training Images Folder")
        if not folder:
            return
        self._image_dir = folder
        self._image_files = sorted([
            f for f in os.listdir(folder)
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS
        ])
        if not self._image_files:
            QMessageBox.information(self, "No Images", "No image files found in folder.")
            return

        from .mask_tracker_annotator import COCOAnnotationManager
        self._coco_manager = COCOAnnotationManager()
        self.lbl_images_count.setText(f"{len(self._image_files)} images in {os.path.basename(folder)}")
        self._current_image_idx = 0
        self._show_current_image()
        self._update_annotate_ready()

    def _load_existing_annotations(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load COCO JSON", "", "JSON Files (*.json)")
        if not path:
            return
        from .mask_tracker_annotator import COCOAnnotationManager
        self._coco_manager = COCOAnnotationManager()
        self._coco_manager.load(path)

        images_dir = QFileDialog.getExistingDirectory(self, "Select Images Directory (for loaded annotations)")
        if not images_dir:
            return
        self._image_dir = images_dir
        self._image_files = sorted([img["file_name"] for img in self._coco_manager.images])

        for cat in self._coco_manager.categories:
            self.list_classes.addItem(cat["name"])
            self.combo_class.addItem(cat["name"])

        self.lbl_images_count.setText(f"{len(self._image_files)} images (loaded from JSON)")
        self._exported_json_path = path
        self._current_image_idx = 0
        self._show_current_image()
        self._update_annotate_ready()

    def _show_current_image(self):
        if self._current_image_idx < 0 or self._current_image_idx >= len(self._image_files):
            return
        filename = self._image_files[self._current_image_idx]
        img_path = os.path.join(self._image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.preview.set_image(img_rgb)

        if self._segmenter and self._segmenter.predictor:
            self._segmenter.set_image(img_rgb)

        self.lbl_image_idx.setText(f"{self._current_image_idx + 1} / {len(self._image_files)}")
        self.btn_prev.setEnabled(self._current_image_idx > 0)
        self.btn_next.setEnabled(self._current_image_idx < len(self._image_files) - 1)

        self._refresh_annotation_list()
        self._show_existing_annotations()

    def _show_existing_annotations(self):
        if self._coco_manager is None:
            return
        filename = self._image_files[self._current_image_idx]
        if filename not in self._coco_manager._image_id_map:
            self.preview.set_existing_annotations([])
            return

        img_id = self._coco_manager._image_id_map[filename]
        anns = self._coco_manager.get_annotations_for_image(img_id)
        ann_data = []
        for ann in anns:
            h, w = self.preview._original.shape[:2] if self.preview._original is not None else (0, 0)
            if h == 0:
                continue
            from .mask_tracker_annotator import _polygon_to_mask as _p2m
            # Reuse the training module's polygon converter
            from .mask_tracker_training import _polygon_to_mask
            mask = _polygon_to_mask(ann["segmentation"], h, w).astype(bool)
            cat_name = self._coco_manager.get_category_name(ann["category_id"])
            ann_data.append((mask, ann["category_id"], cat_name))
        self.preview.set_existing_annotations(ann_data)

    def _refresh_annotation_list(self):
        self.list_annotations.clear()
        if self._coco_manager is None or self._current_image_idx < 0:
            return
        filename = self._image_files[self._current_image_idx]
        if filename not in self._coco_manager._image_id_map:
            return
        img_id = self._coco_manager._image_id_map[filename]
        for ann in self._coco_manager.get_annotations_for_image(img_id):
            cat_name = self._coco_manager.get_category_name(ann["category_id"])
            item = QListWidgetItem(f"[{ann['id']}] {cat_name} (area={ann['area']})")
            item.setData(Qt.UserRole, ann["id"])
            self.list_annotations.addItem(item)

    def _prev_image(self):
        if self._current_image_idx > 0:
            self._current_image_idx -= 1
            self._show_current_image()
            self.preview.set_click_enabled(False)
            self.btn_accept.setEnabled(False)
            self.btn_clear_pts.setEnabled(False)
            self._current_mask = None

    def _next_image(self):
        if self._current_image_idx < len(self._image_files) - 1:
            self._current_image_idx += 1
            self._show_current_image()
            self.preview.set_click_enabled(False)
            self.btn_accept.setEnabled(False)
            self.btn_clear_pts.setEnabled(False)
            self._current_mask = None

    def _add_class(self):
        name, ok = QInputDialog.getText(self, "Add Class", "Class name:")
        if ok and name.strip():
            name = name.strip()
            if self._coco_manager:
                self._coco_manager.add_category(name)
            self.list_classes.addItem(name)
            self.combo_class.addItem(name)

    def _remove_class(self):
        item = self.list_classes.currentItem()
        if item is None:
            return
        name = item.text()
        if self._coco_manager:
            self._coco_manager.remove_category(name)
        self.list_classes.takeItem(self.list_classes.row(item))
        idx = self.combo_class.findText(name)
        if idx >= 0:
            self.combo_class.removeItem(idx)

    def _update_annotate_ready(self):
        ready = (
            self._segmenter is not None
            and self._segmenter.predictor is not None
            and len(self._image_files) > 0
            and self.combo_class.count() > 0
        )
        self.btn_add_object.setEnabled(ready)
        self.btn_export.setEnabled(self._coco_manager is not None and len(self._coco_manager.annotations) > 0)

    def _start_annotating(self):
        if self.combo_class.count() == 0:
            QMessageBox.warning(self, "No Classes", "Add at least one class label first.")
            return
        self.preview.set_click_enabled(True)
        self.preview.clear_points()
        self._current_mask = None
        self.lbl_mask_score.setText("Click on object to segment...")
        self.btn_clear_pts.setEnabled(True)
        self.status_label.setText("Annotation mode: left-click = foreground, right-click = background")

    def _on_preview_left_click(self, x, y):
        self._run_sam2_prediction()

    def _on_preview_right_click(self, x, y):
        self._run_sam2_prediction()

    def _run_sam2_prediction(self):
        if self._segmenter is None or self._segmenter.predictor is None:
            return
        if not self.preview._points_pos:
            return

        pos = list(self.preview._points_pos)
        neg = list(self.preview._points_neg) if self.preview._points_neg else None

        self._predict_worker = SAM2PredictWorker(self._segmenter, pos, neg)
        self._predict_worker.mask_ready.connect(self._on_mask_ready)
        self._predict_worker.error.connect(lambda e: self.status_label.setText(f"Prediction error: {e}"))
        self._predict_worker.start()

    def _on_mask_ready(self, mask, score):
        self._current_mask = mask
        self.preview.set_mask_overlay(mask)
        self.lbl_mask_score.setText(f"Score: {score:.3f}")
        self.btn_accept.setEnabled(True)

    def _accept_annotation(self):
        if self._current_mask is None or self._coco_manager is None:
            return
        filename = self._image_files[self._current_image_idx]
        h, w = self._current_mask.shape
        img_id = self._coco_manager.get_or_add_image(filename, w, h)

        class_name = self.combo_class.currentText()
        if not class_name:
            QMessageBox.warning(self, "No Class", "Select a class label.")
            return
        cat_id = self._coco_manager.add_category(class_name)
        ann_id = self._coco_manager.add_annotation(img_id, cat_id, self._current_mask)
        if ann_id < 0:
            QMessageBox.warning(self, "Error", "Could not convert mask to polygon.")
            return

        self.preview.clear_points()
        self.preview.set_click_enabled(False)
        self._current_mask = None
        self.btn_accept.setEnabled(False)
        self.lbl_mask_score.setText(f"Annotation #{ann_id} saved")
        self._refresh_annotation_list()
        self._show_existing_annotations()
        self._update_annotate_ready()
        self.status_label.setText(f"Annotation saved. Total: {len(self._coco_manager.annotations)}")

    def _clear_annotation_points(self):
        self.preview.clear_points()
        self._current_mask = None
        self.btn_accept.setEnabled(False)
        self.lbl_mask_score.setText("")

    def _delete_annotation(self):
        item = self.list_annotations.currentItem()
        if item is None:
            return
        ann_id = item.data(Qt.UserRole)
        if self._coco_manager:
            self._coco_manager.remove_annotation(ann_id)
        self._refresh_annotation_list()
        self._show_existing_annotations()
        self._update_annotate_ready()

    def _export_annotations(self):
        if self._coco_manager is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save COCO JSON", "annotations.json", "JSON Files (*.json)")
        if not path:
            return
        self._coco_manager.export(path)
        self._exported_json_path = path
        stats = self._coco_manager.get_stats()
        self.lbl_export_status.setText(
            f"Exported: {stats['num_annotations']} annotations, "
            f"{stats['images_with_annotations']} images, "
            f"{stats['num_categories']} classes"
        )
        self.status_label.setText(f"Annotations exported to {path}")

    # ══════════════════════════════════════════════════════════════════
    # TRAIN TAB LOGIC
    # ══════════════════════════════════════════════════════════════════
    def _browse_coco_json(self):
        if self._exported_json_path and os.path.exists(self._exported_json_path):
            start_dir = os.path.dirname(self._exported_json_path)
        else:
            start_dir = ""
        path, _ = QFileDialog.getOpenFileName(self, "Select COCO JSON", start_dir, "JSON Files (*.json)")
        if not path:
            return
        self._exported_json_path = path
        try:
            import json
            with open(path) as f:
                data = json.load(f)
            n_img = len(data.get("images", []))
            n_ann = len(data.get("annotations", []))
            n_cat = len(data.get("categories", []))
            cats = [c["name"] for c in data.get("categories", [])]
            self.lbl_dataset_info.setText(
                f"{n_img} images, {n_ann} annotations, {n_cat} classes: {', '.join(cats)}"
            )
            self._update_train_ready()
        except Exception as e:
            self.lbl_dataset_info.setText(f"Error reading JSON: {e}")

    def _browse_train_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.txt_train_output.setText(d)
            self._update_train_ready()

    def _update_train_ready(self):
        ready = bool(self._exported_json_path) and bool(self.txt_train_output.text())
        self.btn_start_train.setEnabled(ready)

    def _start_training(self):
        from .mask_tracker_training import MaskRCNNTrainingConfig

        images_dir = self._image_dir
        if not images_dir:
            images_dir = QFileDialog.getExistingDirectory(self, "Select Images Directory (referenced by COCO JSON)")
            if not images_dir:
                return

        cfg = MaskRCNNTrainingConfig(
            coco_json_path=self._exported_json_path,
            images_dir=images_dir,
            output_dir=self.txt_train_output.text(),
            num_iterations=self.spin_iterations.value(),
            learning_rate=self.spin_lr.value(),
            batch_size=self.spin_batch.value(),
            device=self.combo_device.currentText(),
        )

        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)
        self.progress_train.setVisible(True)
        self.progress_train.setValue(0)
        self.txt_train_log.clear()

        self._training_worker = TrainingWorker(cfg)
        self._training_worker.progress_signal.connect(self._on_train_progress)
        self._training_worker.finished_signal.connect(self._on_train_finished)
        self._training_worker.error_signal.connect(self._on_train_error)
        self._training_worker.start()

    def _stop_training(self):
        if self._training_worker:
            self._training_worker.stop()
            self.btn_stop_train.setEnabled(False)

    def _on_train_progress(self, iteration, total, metrics):
        self.progress_train.setMaximum(total)
        self.progress_train.setValue(iteration)
        self.lbl_train_progress.setText(f"Iteration {iteration}/{total}")
        loss = metrics.get("loss", 0)
        lr = metrics.get("lr", 0)
        if iteration % 10 == 0 or iteration == total:
            self.txt_train_log.append(f"[{iteration}/{total}] loss={loss:.4f}  lr={lr:.6f}")
            sb = self.txt_train_log.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _on_train_finished(self, summary):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.progress_train.setVisible(False)

        self._trained_model_dir = summary.get("run_dir", "")
        self.lbl_train_progress.setText(
            f"Training complete! {summary.get('iterations_completed', 0)} iterations, "
            f"best loss: {summary.get('best_loss', 0):.4f}"
        )
        self.txt_train_log.append(f"\nModel saved to: {summary.get('model_path', '')}")
        self.lbl_model_info.setText(f"Model: {self._trained_model_dir}")
        self.status_label.setText("Training complete")
        QMessageBox.information(self, "Training Complete", f"Model saved to:\n{summary.get('run_dir', '')}")

    def _on_train_error(self, msg):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.progress_train.setVisible(False)
        self.lbl_train_progress.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Training Error", msg)

    # ══════════════════════════════════════════════════════════════════
    # TRACK TAB LOGIC
    # ══════════════════════════════════════════════════════════════════
    def _browse_model_dir(self):
        start_dir = self._trained_model_dir or ""
        d = QFileDialog.getExistingDirectory(self, "Select Model Directory", start_dir)
        if not d:
            return
        self._trained_model_dir = d
        weights = os.path.join(d, "weights_best.pt")
        if not os.path.exists(weights):
            weights = os.path.join(d, "weights.pt")
        if os.path.exists(weights):
            config_path = os.path.join(d, "training_config.json")
            info = f"Model: {os.path.basename(d)}"
            if os.path.exists(config_path):
                import json
                with open(config_path) as f:
                    cfg = json.load(f)
                n_cls = cfg.get("num_classes", "?")
                cats = cfg.get("categories", {})
                info += f"\nClasses: {n_cls}, Categories: {list(cats.values()) if isinstance(cats, dict) else cats}"
            self.lbl_model_info.setText(info)
            self.lbl_model_info.setStyleSheet("color: #00cc00;")
        else:
            self.lbl_model_info.setText("No weights found in directory")
            self.lbl_model_info.setStyleSheet("color: #ff4444;")
        self._update_track_ready()

    def _add_track_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm);;All Files (*.*)"
        )
        for f in files:
            if f not in self._video_paths:
                self._video_paths.append(f)
                self.list_videos.addItem(os.path.basename(f))
        self._update_track_ready()

    def _add_track_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if not folder:
            return
        VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        for f in sorted(os.listdir(folder)):
            full = os.path.join(folder, f)
            if os.path.isfile(full) and Path(f).suffix.lower() in VIDEO_EXTS:
                if full not in self._video_paths:
                    self._video_paths.append(full)
                    self.list_videos.addItem(f)
        self._update_track_ready()

    def _clear_track_videos(self):
        self._video_paths.clear()
        self.list_videos.clear()
        self._update_track_ready()

    def _update_track_ready(self):
        has_model = bool(self._trained_model_dir) and (
            os.path.exists(os.path.join(self._trained_model_dir, "weights_best.pt"))
            or os.path.exists(os.path.join(self._trained_model_dir, "weights.pt"))
        )
        self.btn_start_track.setEnabled(has_model and len(self._video_paths) > 0)

    def _start_tracking(self):
        from .mask_tracker_inference import MaskInferenceConfig

        config = MaskInferenceConfig(
            model_path=self._trained_model_dir,
            confidence_threshold=self.spin_confidence.value(),
            max_disappeared_frames=self.spin_disappear.value(),
            iou_match_threshold=self.spin_iou.value(),
            device=self.combo_device.currentText() if hasattr(self, "combo_device") else "auto",
        )

        self.btn_start_track.setEnabled(False)
        self.btn_stop_track.setEnabled(True)
        self.progress_track.setVisible(True)
        self.progress_track.setValue(0)
        self.txt_track_log.clear()

        self._pending_videos = list(self._video_paths)
        self._track_config = config
        self._track_results = []
        self._process_next_video()

    def _process_next_video(self):
        if not self._pending_videos:
            self._on_all_tracking_done()
            return
        video_path = self._pending_videos.pop(0)
        self.txt_track_log.append(f"\nProcessing: {os.path.basename(video_path)}")
        self.lbl_track_progress.setText(
            f"Video {len(self._video_paths) - len(self._pending_videos)}/{len(self._video_paths)}"
        )

        self._inference_worker = InferenceWorker(video_path, self._trained_model_dir, self._track_config)
        self._inference_worker.progress_signal.connect(self._on_track_progress)
        self._inference_worker.finished_signal.connect(self._on_video_tracked)
        self._inference_worker.error_signal.connect(self._on_track_error)
        self._inference_worker.start()

    def _stop_tracking(self):
        if self._inference_worker:
            self._inference_worker.stop()
        self._pending_videos.clear()
        self.btn_stop_track.setEnabled(False)

    def _on_track_progress(self, frame, total):
        self.progress_track.setMaximum(total)
        self.progress_track.setValue(frame)

    def _on_video_tracked(self, result):
        n_tracks = result.get("num_tracks", 0)
        csv_path = result.get("csv_path", "")
        self.txt_track_log.append(f"  {n_tracks} tracks, saved to {os.path.basename(csv_path)}")
        self._track_results.append(result)
        self._process_next_video()

    def _on_track_error(self, msg):
        self.txt_track_log.append(f"  Error: {msg}")
        self._process_next_video()

    def _on_all_tracking_done(self):
        self.btn_start_track.setEnabled(True)
        self.btn_stop_track.setEnabled(False)
        self.progress_track.setVisible(False)
        total_tracks = sum(r.get("num_tracks", 0) for r in self._track_results)
        self.lbl_track_progress.setText(
            f"Done! {len(self._track_results)} videos processed, {total_tracks} total tracks"
        )
        self.txt_track_log.append(f"\nAll videos processed. Total tracks: {total_tracks}")
        self.status_label.setText("Tracking complete")


def main():
    app = QApplication(sys.argv)
    window = MaskTrackerPipelineGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
