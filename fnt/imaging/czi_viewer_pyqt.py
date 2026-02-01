"""
CZI Viewer - PyQt5 GUI for viewing and processing Zeiss CZI microscopy images.

Follows the FNT pattern: left panel (controls) + right panel (image preview).
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QPoint
from PyQt5.QtGui import QFont, QImage, QPixmap, QPainter, QColor, QPen, QWheelEvent
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox, QScrollArea,
    QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QListWidgetItem,
    QCheckBox, QSlider, QStatusBar, QMessageBox, QSizePolicy, QLineEdit
)

from .czi_reader import CZIFileReader, CZIImageData, HAS_AICSPYLIBCZI, HAS_AICSIMAGEIO
from .image_processor import CZIImageProcessor, ChannelDisplaySettings


class CZILoadWorker(QThread):
    """Worker thread for loading CZI files."""
    progress = pyqtSignal(str)
    loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            self.progress.emit(f"Loading {Path(self.filepath).name}...")
            reader = CZIFileReader(self.filepath)
            data = reader.load_all_channels()
            self.loaded.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class ImagePreviewWidget(QWidget):
    """
    Custom widget for displaying microscopy images with zoom/pan.
    """

    annotation_clicked = pyqtSignal(int, int)  # x, y in image coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # Image state
        self.display_image: Optional[QImage] = None
        self.image_width = 0
        self.image_height = 0

        # View state
        self.zoom_level: float = 1.0
        self.pan_offset = QPointF(0, 0)

        # Interaction state
        self.is_panning: bool = False
        self.last_mouse_pos: Optional[QPoint] = None
        self.annotation_mode: bool = False

        # Styling
        self.setStyleSheet("background-color: #1e1e1e;")

    def set_image(self, image: np.ndarray):
        """
        Set image data (RGB numpy array) for display.

        Args:
            image: RGB uint8 array (H, W, 3)
        """
        if image is None:
            self.display_image = None
            self.update()
            return

        h, w = image.shape[:2]
        self.image_width = w
        self.image_height = h

        # Ensure contiguous and correct format
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB image
            bytes_per_line = 3 * w
            self.display_image = QImage(
                image.data, w, h, bytes_per_line, QImage.Format_RGB888
            ).copy()  # Copy to own the data
        else:
            # Grayscale - convert to RGB
            gray = image if len(image.shape) == 2 else image[:, :, 0]
            rgb = np.stack([gray, gray, gray], axis=2)
            bytes_per_line = 3 * w
            self.display_image = QImage(
                rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
            ).copy()

        self.update()

    def fit_to_window(self):
        """Reset zoom to fit image in widget."""
        if self.display_image is None:
            return

        widget_w = self.width()
        widget_h = self.height()

        scale_x = widget_w / self.image_width
        scale_y = widget_h / self.image_height

        self.zoom_level = min(scale_x, scale_y) * 0.95  # 5% margin
        self.pan_offset = QPointF(0, 0)
        self.update()

    def zoom_to_100(self):
        """Set zoom to 100% (1:1 pixel mapping)."""
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        self.update()

    def set_zoom(self, zoom: float):
        """Set zoom level."""
        self.zoom_level = max(0.01, min(50.0, zoom))
        self.update()

    def get_zoom_percent(self) -> int:
        """Get current zoom as percentage."""
        return int(self.zoom_level * 100)

    def paintEvent(self, event):
        """Paint the image with current zoom/pan state."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Fill background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self.display_image is None:
            # Draw placeholder text
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return

        # Calculate scaled size
        scaled_w = int(self.image_width * self.zoom_level)
        scaled_h = int(self.image_height * self.zoom_level)

        # Center in widget plus pan offset
        x = (self.width() - scaled_w) / 2 + self.pan_offset.x()
        y = (self.height() - scaled_h) / 2 + self.pan_offset.y()

        # Draw scaled image
        target_rect = QRectF(x, y, scaled_w, scaled_h)
        source_rect = QRectF(0, 0, self.image_width, self.image_height)

        painter.drawImage(target_rect, self.display_image, source_rect)

        # Draw annotation mode indicator
        if self.annotation_mode:
            painter.setPen(QPen(QColor(255, 200, 0), 2))
            painter.drawRect(2, 2, self.width() - 4, self.height() - 4)
            painter.setPen(QColor(255, 200, 0))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(10, 20, "Click to place annotation")

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if self.display_image is None:
            return

        # Get mouse position
        mouse_pos = event.pos()

        # Calculate zoom factor
        delta = event.angleDelta().y()
        if delta > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15

        old_zoom = self.zoom_level
        new_zoom = max(0.01, min(50.0, old_zoom * factor))

        # Adjust pan to zoom toward mouse position
        widget_center = QPointF(self.width() / 2, self.height() / 2)
        mouse_offset = QPointF(mouse_pos) - widget_center - self.pan_offset

        # Scale the offset
        scale_change = new_zoom / old_zoom
        new_offset = mouse_offset * scale_change

        self.pan_offset = self.pan_offset + mouse_offset - new_offset
        self.zoom_level = new_zoom

        self.update()

    def mousePressEvent(self, event):
        """Handle mouse press for panning or annotation placement."""
        if event.button() == Qt.LeftButton:
            if self.annotation_mode and self.display_image is not None:
                # Calculate image coordinates
                img_coords = self._widget_to_image_coords(event.pos())
                if img_coords:
                    self.annotation_clicked.emit(int(img_coords[0]), int(img_coords[1]))
            else:
                self.is_panning = True
                self.last_mouse_pos = event.pos()
                self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """Handle mouse drag for panning."""
        if self.is_panning and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += QPointF(delta)
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)

    def _widget_to_image_coords(self, widget_pos: QPoint) -> Optional[tuple]:
        """Convert widget coordinates to image coordinates."""
        if self.display_image is None:
            return None

        scaled_w = self.image_width * self.zoom_level
        scaled_h = self.image_height * self.zoom_level

        x = (self.width() - scaled_w) / 2 + self.pan_offset.x()
        y = (self.height() - scaled_h) / 2 + self.pan_offset.y()

        img_x = (widget_pos.x() - x) / self.zoom_level
        img_y = (widget_pos.y() - y) / self.zoom_level

        if 0 <= img_x < self.image_width and 0 <= img_y < self.image_height:
            return (img_x, img_y)
        return None


class ChannelControlWidget(QWidget):
    """Widget for controlling a single channel's display settings."""

    settings_changed = pyqtSignal()

    def __init__(self, channel_idx: int, channel_name: str,
                 suggested_color: str = "gray", parent=None):
        super().__init__(parent)
        self.channel_idx = channel_idx

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        # Enabled checkbox
        self.enabled_cb = QCheckBox()
        self.enabled_cb.setChecked(True)
        self.enabled_cb.stateChanged.connect(self._on_change)
        layout.addWidget(self.enabled_cb)

        # Channel name label
        name_label = QLabel(channel_name)
        name_label.setMinimumWidth(80)
        layout.addWidget(name_label)

        # Color selector
        self.color_combo = QComboBox()
        self.color_combo.addItems(['green', 'magenta', 'cyan', 'red', 'blue', 'yellow', 'gray'])
        self.color_combo.setCurrentText(suggested_color)
        self.color_combo.currentTextChanged.connect(self._on_change)
        layout.addWidget(self.color_combo)

        layout.addStretch()

    def _on_change(self):
        self.settings_changed.emit()

    def get_settings(self) -> ChannelDisplaySettings:
        """Get current settings for this channel."""
        return ChannelDisplaySettings(
            enabled=self.enabled_cb.isChecked(),
            color=self.color_combo.currentText()
        )

    def set_enabled(self, enabled: bool):
        self.enabled_cb.setChecked(enabled)

    def set_color(self, color: str):
        self.color_combo.setCurrentText(color)


class CZIViewerWindow(QMainWindow):
    """
    Main window for CZI microscopy image viewer.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FNT CZI Viewer")
        self.setMinimumSize(1000, 700)
        self.resize(1400, 900)

        # State
        self.czi_files: List[str] = []
        self.current_file_idx: int = 0
        self.image_data: Optional[CZIImageData] = None
        self.processor = CZIImageProcessor()
        self.channel_controls: Dict[int, ChannelControlWidget] = {}
        self.load_worker: Optional[CZILoadWorker] = None

        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        """Setup main UI layout."""
        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left scrollable panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(340)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        # Build sections
        self._create_input_section(left_layout)
        self._create_channel_section(left_layout)
        self._create_adjustments_section(left_layout)
        self._create_annotation_section(left_layout)
        self._create_export_section(left_layout)
        self._create_view_section(left_layout)

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)

        # Right preview panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview = ImagePreviewWidget()
        self.preview.annotation_clicked.connect(self._on_annotation_click)
        right_layout.addWidget(self.preview, 1)

        # Add to main
        main_layout.addWidget(left_scroll)
        main_layout.addWidget(right_panel, 1)

        self.setCentralWidget(central)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load a CZI file to begin")

    def _create_input_section(self, layout):
        """Create file input section."""
        group = QGroupBox("Input")
        group_layout = QVBoxLayout()

        # Buttons row
        btn_layout = QHBoxLayout()

        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.clicked.connect(self._add_folder)
        btn_layout.addWidget(self.btn_add_folder)

        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.clicked.connect(self._add_files)
        btn_layout.addWidget(self.btn_add_files)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear_files)
        self.btn_clear.setStyleSheet("background-color: #5c5c5c;")
        btn_layout.addWidget(self.btn_clear)

        group_layout.addLayout(btn_layout)

        # File list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(100)
        self.file_list.itemClicked.connect(self._on_file_selected)
        group_layout.addWidget(self.file_list)

        # Navigation row
        nav_layout = QHBoxLayout()

        self.btn_prev = QPushButton("<")
        self.btn_prev.setFixedWidth(40)
        self.btn_prev.clicked.connect(self._prev_file)
        nav_layout.addWidget(self.btn_prev)

        self.file_label = QLabel("No files loaded")
        self.file_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.file_label, 1)

        self.btn_next = QPushButton(">")
        self.btn_next.setFixedWidth(40)
        self.btn_next.clicked.connect(self._next_file)
        nav_layout.addWidget(self.btn_next)

        group_layout.addLayout(nav_layout)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_channel_section(self, layout):
        """Create channel display section."""
        group = QGroupBox("Channels")
        self.channel_layout = QVBoxLayout()

        self.channel_placeholder = QLabel("Load a file to see channels")
        self.channel_placeholder.setStyleSheet("color: #888888; font-style: italic;")
        self.channel_layout.addWidget(self.channel_placeholder)

        group.setLayout(self.channel_layout)
        layout.addWidget(group)

    def _create_adjustments_section(self, layout):
        """Create image adjustments section."""
        group = QGroupBox("Display Adjustments")
        group_layout = QVBoxLayout()

        # Brightness slider
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 300)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(self._on_adjustment_changed)
        brightness_layout.addWidget(self.brightness_slider)
        self.brightness_label = QLabel("1.0")
        self.brightness_label.setFixedWidth(35)
        brightness_layout.addWidget(self.brightness_label)
        group_layout.addLayout(brightness_layout)

        # Contrast slider
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self._on_adjustment_changed)
        contrast_layout.addWidget(self.contrast_slider)
        self.contrast_label = QLabel("1.0")
        self.contrast_label.setFixedWidth(35)
        contrast_layout.addWidget(self.contrast_label)
        group_layout.addLayout(contrast_layout)

        # Gamma slider
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 300)
        self.gamma_slider.setValue(100)
        self.gamma_slider.valueChanged.connect(self._on_adjustment_changed)
        gamma_layout.addWidget(self.gamma_slider)
        self.gamma_label = QLabel("1.0")
        self.gamma_label.setFixedWidth(35)
        gamma_layout.addWidget(self.gamma_label)
        group_layout.addLayout(gamma_layout)

        # Min/Max controls
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Min:"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(0, 1)
        self.min_spin.setSingleStep(0.01)
        self.min_spin.setValue(0)
        self.min_spin.setDecimals(2)
        self.min_spin.valueChanged.connect(self._on_adjustment_changed)
        range_layout.addWidget(self.min_spin)

        range_layout.addWidget(QLabel("Max:"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(0, 1)
        self.max_spin.setSingleStep(0.01)
        self.max_spin.setValue(1)
        self.max_spin.setDecimals(2)
        self.max_spin.valueChanged.connect(self._on_adjustment_changed)
        range_layout.addWidget(self.max_spin)
        group_layout.addLayout(range_layout)

        # Buttons row
        btn_layout = QHBoxLayout()
        self.btn_auto_levels = QPushButton("Auto Levels")
        self.btn_auto_levels.clicked.connect(self._auto_levels)
        btn_layout.addWidget(self.btn_auto_levels)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_adjustments)
        self.btn_reset.setStyleSheet("background-color: #5c5c5c;")
        btn_layout.addWidget(self.btn_reset)
        group_layout.addLayout(btn_layout)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_annotation_section(self, layout):
        """Create text annotation section."""
        group = QGroupBox("Annotations")
        group_layout = QVBoxLayout()

        # Text input
        text_layout = QHBoxLayout()
        text_layout.addWidget(QLabel("Text:"))
        self.annotation_text = QLineEdit()
        self.annotation_text.setPlaceholderText("Enter annotation text")
        text_layout.addWidget(self.annotation_text)
        group_layout.addLayout(text_layout)

        # Size and color
        options_layout = QHBoxLayout()
        options_layout.addWidget(QLabel("Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 72)
        self.font_size_spin.setValue(24)
        options_layout.addWidget(self.font_size_spin)

        options_layout.addWidget(QLabel("Color:"))
        self.annotation_color = QComboBox()
        self.annotation_color.addItems(['white', 'black', 'yellow', 'red', 'green', 'cyan', 'magenta'])
        options_layout.addWidget(self.annotation_color)
        options_layout.addStretch()
        group_layout.addLayout(options_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_add_annotation = QPushButton("Add Annotation")
        self.btn_add_annotation.clicked.connect(self._start_annotation_mode)
        btn_layout.addWidget(self.btn_add_annotation)

        self.btn_clear_annotations = QPushButton("Clear All")
        self.btn_clear_annotations.clicked.connect(self._clear_annotations)
        self.btn_clear_annotations.setStyleSheet("background-color: #5c5c5c;")
        btn_layout.addWidget(self.btn_clear_annotations)
        group_layout.addLayout(btn_layout)

        # Annotation list
        self.annotation_list = QListWidget()
        self.annotation_list.setMaximumHeight(60)
        group_layout.addWidget(self.annotation_list)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_export_section(self, layout):
        """Create export section."""
        group = QGroupBox("Export")
        group_layout = QVBoxLayout()

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.export_format = QComboBox()
        self.export_format.addItems(['PNG', 'TIFF'])
        format_layout.addWidget(self.export_format)
        format_layout.addStretch()
        group_layout.addLayout(format_layout)

        btn_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export Current")
        self.btn_export.clicked.connect(self._export_current)
        btn_layout.addWidget(self.btn_export)

        self.btn_export_all = QPushButton("Export All")
        self.btn_export_all.clicked.connect(self._export_all)
        btn_layout.addWidget(self.btn_export_all)
        group_layout.addLayout(btn_layout)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_view_section(self, layout):
        """Create view controls section."""
        group = QGroupBox("View Controls")
        group_layout = QVBoxLayout()

        zoom_layout = QHBoxLayout()
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(30)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        zoom_layout.addWidget(self.btn_zoom_out)

        zoom_layout.addWidget(QLabel("Zoom:"))
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        zoom_layout.addWidget(self.zoom_label)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(30)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        zoom_layout.addWidget(self.btn_zoom_in)

        zoom_layout.addStretch()
        group_layout.addLayout(zoom_layout)

        btn_layout = QHBoxLayout()
        self.btn_fit = QPushButton("Fit to Window")
        self.btn_fit.clicked.connect(self._fit_to_window)
        btn_layout.addWidget(self.btn_fit)

        self.btn_100 = QPushButton("Zoom 100%")
        self.btn_100.clicked.connect(self._zoom_100)
        btn_layout.addWidget(self.btn_100)
        group_layout.addLayout(btn_layout)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _apply_styles(self):
        """Apply FNT dark theme stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 8px;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                color: #cccccc;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3f3f3f;
                height: 6px;
                background: #1e1e1e;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: none;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QScrollArea {
                border: none;
            }
            QStatusBar {
                background-color: #1e1e1e;
                color: #cccccc;
                border-top: 1px solid #3f3f3f;
            }
            QCheckBox {
                color: #cccccc;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #3f3f3f;
                background-color: #1e1e1e;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #0078d4;
                background-color: #0078d4;
            }
        """)

    # =========================================================================
    # File handling
    # =========================================================================

    def _add_folder(self):
        """Browse for folder and add all CZI files."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            files = list(Path(folder).glob("*.czi"))
            files += list(Path(folder).glob("*.CZI"))
            self.czi_files.extend([str(f) for f in files])
            self._update_file_list()
            if self.czi_files and self.image_data is None:
                self.current_file_idx = 0
                self._load_current_file()

    def _add_files(self):
        """Browse for individual CZI files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select CZI Files", "", "CZI Files (*.czi *.CZI);;All Files (*.*)"
        )
        if files:
            self.czi_files.extend(files)
            self._update_file_list()
            if self.image_data is None:
                self.current_file_idx = len(self.czi_files) - len(files)
                self._load_current_file()

    def _clear_files(self):
        """Clear all loaded files."""
        self.czi_files.clear()
        self.current_file_idx = 0
        self.image_data = None
        self._update_file_list()
        self._clear_channel_controls()
        self.preview.set_image(None)
        self.status_bar.showMessage("Cleared all files")

    def _update_file_list(self):
        """Update the file list widget."""
        self.file_list.clear()
        for filepath in self.czi_files:
            self.file_list.addItem(Path(filepath).name)

        if self.czi_files:
            self.file_list.setCurrentRow(self.current_file_idx)
            self.file_label.setText(f"File {self.current_file_idx + 1}/{len(self.czi_files)}")
        else:
            self.file_label.setText("No files loaded")

    def _on_file_selected(self, item):
        """Handle file selection from list."""
        idx = self.file_list.row(item)
        if idx != self.current_file_idx:
            self.current_file_idx = idx
            self._load_current_file()

    def _prev_file(self):
        """Go to previous file."""
        if self.czi_files and self.current_file_idx > 0:
            self.current_file_idx -= 1
            self._load_current_file()
            self._update_file_list()

    def _next_file(self):
        """Go to next file."""
        if self.czi_files and self.current_file_idx < len(self.czi_files) - 1:
            self.current_file_idx += 1
            self._load_current_file()
            self._update_file_list()

    def _load_current_file(self):
        """Load the current CZI file."""
        if not self.czi_files:
            return

        if not HAS_AICSPYLIBCZI and not HAS_AICSIMAGEIO:
            QMessageBox.warning(
                self, "Missing Dependencies",
                "CZI file support requires additional packages.\n\n"
                "Install with:\n"
                "pip install aicspylibczi fsspec Pillow"
            )
            return

        filepath = self.czi_files[self.current_file_idx]

        # Use worker thread
        self.load_worker = CZILoadWorker(filepath)
        self.load_worker.progress.connect(self._on_load_progress)
        self.load_worker.loaded.connect(self._on_load_complete)
        self.load_worker.error.connect(self._on_load_error)
        self.load_worker.start()

    def _on_load_progress(self, message: str):
        self.status_bar.showMessage(message)

    def _on_load_complete(self, data: CZIImageData):
        self.image_data = data
        self._setup_channel_controls()
        self._update_display()
        self.preview.fit_to_window()
        self._update_zoom_label()

        n_channels = len(data.channel_data)
        self.status_bar.showMessage(
            f"Loaded: {Path(data.metadata.filepath).name} "
            f"({data.metadata.width}x{data.metadata.height}, {n_channels} channels)"
        )

    def _on_load_error(self, error: str):
        QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{error}")
        self.status_bar.showMessage(f"Error: {error}")

    # =========================================================================
    # Channel controls
    # =========================================================================

    def _clear_channel_controls(self):
        """Clear all channel control widgets."""
        for ctrl in self.channel_controls.values():
            ctrl.setParent(None)
            ctrl.deleteLater()
        self.channel_controls.clear()

        # Show placeholder
        self.channel_placeholder.show()

    def _setup_channel_controls(self):
        """Setup channel control widgets based on loaded image."""
        self._clear_channel_controls()

        if self.image_data is None:
            return

        self.channel_placeholder.hide()

        for channel_info in self.image_data.metadata.channels:
            ctrl = ChannelControlWidget(
                channel_info.index,
                channel_info.name,
                channel_info.suggested_color
            )
            ctrl.settings_changed.connect(self._on_channel_changed)
            self.channel_layout.addWidget(ctrl)
            self.channel_controls[channel_info.index] = ctrl

    def _on_channel_changed(self):
        """Handle channel settings change."""
        self._update_display()

    # =========================================================================
    # Display
    # =========================================================================

    def _get_current_settings(self) -> Dict[int, ChannelDisplaySettings]:
        """Get current display settings from all channel controls."""
        settings = {}

        # Get per-channel settings
        for idx, ctrl in self.channel_controls.items():
            channel_settings = ctrl.get_settings()

            # Apply global adjustments
            channel_settings.brightness = self.brightness_slider.value() / 100.0
            channel_settings.contrast = self.contrast_slider.value() / 100.0
            channel_settings.gamma = self.gamma_slider.value() / 100.0
            channel_settings.min_display = self.min_spin.value()
            channel_settings.max_display = self.max_spin.value()

            settings[idx] = channel_settings

        return settings

    def _update_display(self):
        """Update the image preview with current settings."""
        if self.image_data is None:
            return

        settings = self._get_current_settings()

        # Merge channels
        merged = self.processor.merge_channels(
            self.image_data.channel_data,
            settings
        )

        # Render annotations
        if self.processor.annotations:
            merged = self.processor.render_with_annotations(merged)

        # Convert to uint8 for display
        display_image = self.processor.to_uint8(merged)

        self.preview.set_image(display_image)

    def _on_adjustment_changed(self):
        """Handle global adjustment slider change."""
        # Update labels
        self.brightness_label.setText(f"{self.brightness_slider.value() / 100:.1f}")
        self.contrast_label.setText(f"{self.contrast_slider.value() / 100:.1f}")
        self.gamma_label.setText(f"{self.gamma_slider.value() / 100:.1f}")

        self._update_display()

    def _auto_levels(self):
        """Apply auto levels based on image histogram."""
        if self.image_data is None:
            return

        # Use first channel for auto levels
        first_channel = next(iter(self.image_data.channel_data.values()))
        min_val, max_val = self.processor.auto_levels(first_channel)

        self.min_spin.setValue(min_val)
        self.max_spin.setValue(max_val)
        self._update_display()

    def _reset_adjustments(self):
        """Reset all adjustments to defaults."""
        self.brightness_slider.setValue(100)
        self.contrast_slider.setValue(100)
        self.gamma_slider.setValue(100)
        self.min_spin.setValue(0)
        self.max_spin.setValue(1)
        self._update_display()

    # =========================================================================
    # Annotations
    # =========================================================================

    def _start_annotation_mode(self):
        """Enter annotation placement mode."""
        if self.image_data is None:
            QMessageBox.warning(self, "No Image", "Load an image first")
            return

        text = self.annotation_text.text().strip()
        if not text:
            QMessageBox.warning(self, "No Text", "Enter annotation text first")
            return

        self.preview.annotation_mode = True
        self.preview.update()
        self.status_bar.showMessage("Click on the image to place annotation")

    def _on_annotation_click(self, x: int, y: int):
        """Handle annotation placement click."""
        text = self.annotation_text.text().strip()
        if text:
            self.processor.add_annotation(
                text, x, y,
                self.font_size_spin.value(),
                self.annotation_color.currentText()
            )
            self.annotation_list.addItem(f'"{text}" at ({x}, {y})')
            self.annotation_text.clear()
            self._update_display()

        self.preview.annotation_mode = False
        self.preview.update()
        self.status_bar.showMessage("Annotation added")

    def _clear_annotations(self):
        """Clear all annotations."""
        self.processor.clear_annotations()
        self.annotation_list.clear()
        self._update_display()

    # =========================================================================
    # Export
    # =========================================================================

    def _export_current(self):
        """Export current image."""
        if self.image_data is None:
            QMessageBox.warning(self, "No Image", "Load an image first")
            return

        ext = self.export_format.currentText().lower()
        default_name = Path(self.image_data.metadata.filepath).stem + f"_processed.{ext}"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Image", default_name,
            f"{ext.upper()} Files (*.{ext});;All Files (*.*)"
        )

        if filepath:
            try:
                settings = self._get_current_settings()
                merged = self.processor.merge_channels(
                    self.image_data.channel_data,
                    settings
                )
                self.processor.export_image(merged, filepath, include_annotations=True)
                self.status_bar.showMessage(f"Exported: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _export_all(self):
        """Export all loaded images."""
        if not self.czi_files:
            QMessageBox.warning(self, "No Files", "No files loaded")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder:
            return

        ext = self.export_format.currentText().lower()
        exported = 0

        for filepath in self.czi_files:
            try:
                reader = CZIFileReader(filepath)
                data = reader.load_all_channels()

                settings = self._get_current_settings()
                merged = self.processor.merge_channels(data.channel_data, settings)

                output_name = Path(filepath).stem + f"_processed.{ext}"
                output_path = os.path.join(folder, output_name)
                self.processor.export_image(merged, output_path, include_annotations=False)
                exported += 1

            except Exception as e:
                self.status_bar.showMessage(f"Error exporting {filepath}: {e}")

        QMessageBox.information(
            self, "Export Complete",
            f"Exported {exported} of {len(self.czi_files)} files to:\n{folder}"
        )

    # =========================================================================
    # View controls
    # =========================================================================

    def _zoom_in(self):
        """Zoom in by 25%."""
        current = self.preview.zoom_level
        self.preview.set_zoom(current * 1.25)
        self._update_zoom_label()

    def _zoom_out(self):
        """Zoom out by 25%."""
        current = self.preview.zoom_level
        self.preview.set_zoom(current / 1.25)
        self._update_zoom_label()

    def _fit_to_window(self):
        """Fit image to window."""
        self.preview.fit_to_window()
        self._update_zoom_label()

    def _zoom_100(self):
        """Zoom to 100%."""
        self.preview.zoom_to_100()
        self._update_zoom_label()

    def _update_zoom_label(self):
        """Update zoom percentage label."""
        self.zoom_label.setText(f"{self.preview.get_zoom_percent()}%")


def main():
    """Run the CZI Viewer standalone."""
    app = QApplication(sys.argv)
    window = CZIViewerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
