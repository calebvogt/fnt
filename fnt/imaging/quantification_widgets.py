"""
Quantification widgets for the CZI viewer.

Provides QuantificationPanel (QWidget) for the left sidebar and
worker threads for background analysis. Supports multi-channel
analysis with colocalization and ROI-based density measurement.
"""

import os
import csv
import json
from typing import Optional, List, Dict

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QImage, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QComboBox, QSlider, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QHeaderView, QAbstractItemView,
    QMessageBox, QStackedWidget, QInputDialog, QScrollArea,
    QSizePolicy,
)

from .quantification import (
    ImageQuantifier, QuantificationConfig, QuantificationResult,
    MultiChannelConfig, MultiChannelResult, ColocalizationResult,
    ROIDefinition, ROIDensityResult, ROIChannelMetrics,
)

# Import CheckmarkCheckBox from the viewer module
from .czi_viewer_pyqt import CheckmarkCheckBox


# Color name -> RGB tuple for overlay rendering
OVERLAY_COLORS = {
    'green': (0, 255, 0),
    'magenta': (255, 0, 255),
    'cyan': (0, 255, 255),
    'red': (255, 0, 0),
    'blue': (0, 100, 255),
    'yellow': (255, 255, 0),
    'gray': (180, 180, 180),
    'white': (255, 255, 255),
}


class QuantificationWorker(QThread):
    """Run single-channel quantification analysis in a background thread."""
    finished = pyqtSignal(object)  # QuantificationResult
    error = pyqtSignal(str)

    def __init__(self, quantifier: ImageQuantifier, image: np.ndarray,
                 config: QuantificationConfig,
                 pixel_size_um: Optional[float] = None):
        super().__init__()
        self.quantifier = quantifier
        self.image = image
        self.config = config
        self.pixel_size_um = pixel_size_um

    def run(self):
        try:
            result = self.quantifier.detect_and_measure(
                self.image, self.config, self.pixel_size_um
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MultiChannelWorker(QThread):
    """Run multi-channel quantification in a background thread."""
    finished = pyqtSignal(object)  # MultiChannelResult
    error = pyqtSignal(str)
    progress = pyqtSignal(str, int, int)  # message, current_step, total_steps

    def __init__(self, quantifier: ImageQuantifier,
                 images: Dict[int, np.ndarray],
                 config: MultiChannelConfig,
                 pixel_size_um: Optional[float] = None,
                 channel_names: Optional[Dict[int, str]] = None):
        super().__init__()
        self.quantifier = quantifier
        self.images = images
        self.config = config
        self.pixel_size_um = pixel_size_um
        self.channel_names = channel_names or {}

    def run(self):
        try:
            from itertools import combinations

            ch_configs = self.config.channel_configs
            n_channels = len(ch_configs)
            n_pairs = len(list(combinations(ch_configs.keys(), 2)))
            n_rois = len(self.config.roi_definitions)
            total_steps = n_channels + (1 if n_pairs > 0 else 0) + (1 if n_rois > 0 else 0)
            step = 0

            # Per-channel detection
            channel_results = {}
            for ch_idx, ch_config in ch_configs.items():
                if ch_idx not in self.images:
                    continue
                name = self.channel_names.get(ch_idx, f"Channel {ch_idx}")
                self.progress.emit(f"Detecting cells in {name}...", step, total_steps)
                result = self.quantifier.detect_and_measure(
                    self.images[ch_idx], ch_config, self.pixel_size_um
                )
                result.channel_name = name
                channel_results[ch_idx] = result
                step += 1

            # Colocalization
            colocalizations = []
            ch_indices = sorted(channel_results.keys())
            if len(ch_indices) >= 2:
                self.progress.emit("Computing colocalization...", step, total_steps)
                for ch_a, ch_b in combinations(ch_indices, 2):
                    coloc = self.quantifier.compute_colocalization(
                        channel_results[ch_a], channel_results[ch_b],
                        self.images[ch_a], self.images[ch_b],
                        self.pixel_size_um,
                        self.channel_names.get(ch_a, f"Channel {ch_a}"),
                        self.channel_names.get(ch_b, f"Channel {ch_b}"),
                    )
                    colocalizations.append(coloc)
                step += 1

            # ROI densities
            roi_densities = []
            if self.config.roi_definitions:
                self.progress.emit("Computing ROI densities...", step, total_steps)
                roi_densities = self.quantifier.compute_roi_densities(
                    channel_results, self.config.roi_definitions,
                    self.pixel_size_um, self.channel_names,
                )
                step += 1

            self.progress.emit("Complete", total_steps, total_steps)

            result = MultiChannelResult(
                channel_results=channel_results,
                colocalizations=colocalizations,
                roi_densities=roi_densities,
                pixel_size_um=self.pixel_size_um,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ChannelQuantRow(QWidget):
    """Per-channel configuration row: enable checkbox, name, threshold method."""

    settings_changed = pyqtSignal()

    def __init__(self, channel_idx: int, channel_name: str,
                 channel_color: str = "gray", parent=None):
        super().__init__(parent)
        self.channel_idx = channel_idx
        self.channel_name = channel_name
        self.channel_color = channel_color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        # Row 1: enable checkbox + channel name + method combo
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        self.enable_cb = CheckmarkCheckBox("")
        self.enable_cb.setChecked(True)
        self.enable_cb.stateChanged.connect(lambda: self.settings_changed.emit())
        row1.addWidget(self.enable_cb)

        color_rgb = OVERLAY_COLORS.get(channel_color, (180, 180, 180))
        name_label = QLabel(channel_name)
        name_label.setStyleSheet(
            f"font-size: 10px; font-weight: bold; "
            f"color: rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]});"
        )
        row1.addWidget(name_label, 1)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Auto", "Triangle", "Li", "Manual"])
        self.method_combo.setFixedWidth(72)
        self.method_combo.setToolTip(
            "Threshold method:\n"
            "• Auto (Otsu) — Best for bimodal intensity distributions\n"
            "• Triangle — Good for dim signals on dark backgrounds\n"
            "• Li — Iterative minimum cross-entropy, good for uneven illumination\n"
            "• Manual — Set threshold value manually with the slider"
        )
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        row1.addWidget(self.method_combo)

        layout.addLayout(row1)

        # Row 2: manual threshold slider (hidden unless Manual)
        self.manual_widget = QWidget()
        mt_layout = QHBoxLayout(self.manual_widget)
        mt_layout.setContentsMargins(20, 0, 0, 0)
        mt_layout.setSpacing(4)

        mt_label = QLabel("Thresh:")
        mt_label.setStyleSheet("font-size: 9px;")
        mt_layout.addWidget(mt_label)

        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(0, 1000)
        self.thresh_slider.setValue(500)
        self.thresh_slider.valueChanged.connect(self._on_slider_changed)
        mt_layout.addWidget(self.thresh_slider, 1)

        self.thresh_value_label = QLabel("0.500")
        self.thresh_value_label.setFixedWidth(36)
        self.thresh_value_label.setStyleSheet("font-size: 9px;")
        mt_layout.addWidget(self.thresh_value_label)

        layout.addWidget(self.manual_widget)
        self.manual_widget.hide()

    def is_enabled(self) -> bool:
        return self.enable_cb.isChecked()

    def get_method(self) -> str:
        text = self.method_combo.currentText().lower()
        return "otsu" if text == "auto" else text

    def get_manual_threshold(self) -> float:
        return self.thresh_slider.value() / 1000.0

    def _on_method_changed(self, text: str):
        self.manual_widget.setVisible(text.lower() == "manual")
        self.settings_changed.emit()

    def reset(self):
        """Reset to defaults."""
        self.enable_cb.setChecked(True)
        self.method_combo.setCurrentIndex(0)
        self.thresh_slider.setValue(500)

    def _on_slider_changed(self, value: int):
        self.thresh_value_label.setText(f"{value / 1000.0:.3f}")
        self.settings_changed.emit()


class QuantificationPanel(QWidget):
    """
    Multi-channel quantification controls panel for the CZI viewer sidebar.

    Provides per-channel threshold configuration, multi-channel analysis
    with colocalization, ROI-based density measurement, and CSV export.
    Results are shown in togglable views: Summary, Per-Channel, Coloc,
    ROI Density.
    """

    # Signals to parent
    request_analysis = pyqtSignal()
    request_threshold_preview = pyqtSignal()
    overlay_updated = pyqtSignal(dict)
    overlay_cleared = pyqtSignal()
    request_roi_mode = pyqtSignal()
    request_overlay_export = pyqtSignal()      # Export overlay image
    request_batch_analysis = pyqtSignal()       # Batch analyze all files
    zoom_to_particle = pyqtSignal(float, float) # Pan/zoom to centroid (x, y)
    quant_mode_entered = pyqtSignal()  # Switch preview to quant mode
    quant_mode_exited = pyqtSignal()   # Switch preview back to viz mode

    MAX_TABLE_ROWS = 1000

    def __init__(self, title="Quantification", parent=None):
        super().__init__(parent)
        self._group_title = title
        self.quantifier = ImageQuantifier()
        self.current_result: Optional[MultiChannelResult] = None
        self.worker: Optional[MultiChannelWorker] = None
        self._roi_definitions: List[ROIDefinition] = []
        self._channel_rows: Dict[int, ChannelQuantRow] = {}
        self._channel_colors: Dict[int, str] = {}
        self._current_images: Optional[Dict[int, np.ndarray]] = None
        self._current_pixel_size: Optional[float] = None
        self._current_channel_names: Dict[int, str] = {}
        self._roi_counter = 0

        # Debounce timer for live threshold preview
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._do_threshold_preview)

        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        group = QGroupBox(self._group_title)
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # --- Per-channel config area (no scroll — expands naturally) ---
        self._channel_container = QWidget()
        self._channel_layout = QVBoxLayout(self._channel_container)
        self._channel_layout.setContentsMargins(0, 0, 0, 0)
        self._channel_layout.setSpacing(2)
        group_layout.addWidget(self._channel_container)

        # --- Background subtraction (independent from visualization) ---
        bg_row = QHBoxLayout()
        bg_label = QLabel("BG Sub:")
        bg_label.setStyleSheet("font-size: 10px;")
        bg_label.setToolTip(
            "Background subtraction applied to raw data before detection.\n"
            "Independent from the visualization settings above."
        )
        bg_row.addWidget(bg_label)
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(["None", "Rolling Ball", "Gaussian"])
        self.bg_method_combo.setToolTip(
            "Background subtraction method:\n"
            "• None — Use raw normalized data\n"
            "• Rolling Ball — Morphological opening (good for uneven illumination)\n"
            "• Gaussian — Gaussian blur subtraction (fast, smooth correction)"
        )
        self.bg_method_combo.currentTextChanged.connect(self._on_bg_method_changed)
        bg_row.addWidget(self.bg_method_combo, 1)
        group_layout.addLayout(bg_row)

        # BG radius row (hidden by default)
        self._bg_radius_widget = QWidget()
        br_layout = QHBoxLayout(self._bg_radius_widget)
        br_layout.setContentsMargins(0, 0, 0, 0)
        br_label = QLabel("Radius:")
        br_label.setStyleSheet("font-size: 10px;")
        br_layout.addWidget(br_label)
        self.bg_radius_spin = QDoubleSpinBox()
        self.bg_radius_spin.setRange(1, 200)
        self.bg_radius_spin.setValue(50.0)
        self.bg_radius_spin.setDecimals(0)
        self.bg_radius_spin.setSuffix(" px")
        br_layout.addWidget(self.bg_radius_spin, 1)
        group_layout.addWidget(self._bg_radius_widget)
        self._bg_radius_widget.hide()

        # --- Shared controls ---
        # Min/Max area
        min_row = QHBoxLayout()
        min_label = QLabel("Min area:")
        min_label.setStyleSheet("font-size: 10px;")
        min_row.addWidget(min_label)
        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0, 1000000)
        self.min_area_spin.setValue(10.0)
        self.min_area_spin.setDecimals(1)
        self.min_area_spin.setSuffix(" \u00b5m\u00b2")
        min_row.addWidget(self.min_area_spin, 1)
        group_layout.addLayout(min_row)

        max_row = QHBoxLayout()
        max_label = QLabel("Max area:")
        max_label.setStyleSheet("font-size: 10px;")
        max_row.addWidget(max_label)
        self.max_area_spin = QDoubleSpinBox()
        self.max_area_spin.setRange(0, 10000000)
        self.max_area_spin.setValue(10000.0)
        self.max_area_spin.setDecimals(1)
        self.max_area_spin.setSuffix(" \u00b5m\u00b2")
        max_row.addWidget(self.max_area_spin, 1)
        group_layout.addLayout(max_row)

        # Watershed checkbox (on by default)
        self.cb_watershed = CheckmarkCheckBox("Watershed separation")
        self.cb_watershed.setStyleSheet("font-size: 10px;")
        self.cb_watershed.setChecked(True)
        self.cb_watershed.setToolTip(
            "Separate touching cells using watershed segmentation.\n"
            "Recommended for densely packed cells."
        )
        group_layout.addWidget(self.cb_watershed)

        # Show mask toggle + contour mode
        mask_row = QHBoxLayout()
        mask_row.setSpacing(8)
        self.cb_show_mask = CheckmarkCheckBox("Show mask")
        self.cb_show_mask.setStyleSheet("font-size: 10px;")
        self.cb_show_mask.stateChanged.connect(self._on_mask_toggle)
        mask_row.addWidget(self.cb_show_mask)

        self.cb_contour_mode = CheckmarkCheckBox("Contour only")
        self.cb_contour_mode.setStyleSheet("font-size: 10px;")
        self.cb_contour_mode.setChecked(True)
        self.cb_contour_mode.setToolTip(
            "Show mask boundaries as outlines instead of filled overlay.\n"
            "Outlines keep the underlying signal visible."
        )
        self.cb_contour_mode.stateChanged.connect(self._on_mask_toggle)
        mask_row.addWidget(self.cb_contour_mode)
        mask_row.addStretch()
        group_layout.addLayout(mask_row)

        # --- ROI management ---
        roi_row = QHBoxLayout()
        self.btn_add_roi = QPushButton("Add ROI")
        self.btn_add_roi.clicked.connect(self._on_add_roi)
        roi_row.addWidget(self.btn_add_roi)
        self.btn_clear_rois = QPushButton("Clear ROIs")
        self.btn_clear_rois.setStyleSheet("background-color: #5c5c5c;")
        self.btn_clear_rois.clicked.connect(self._on_clear_rois)
        roi_row.addWidget(self.btn_clear_rois)
        group_layout.addLayout(roi_row)

        # ROI list container
        self._roi_list_widget = QWidget()
        self._roi_list_layout = QVBoxLayout(self._roi_list_widget)
        self._roi_list_layout.setContentsMargins(0, 0, 0, 0)
        self._roi_list_layout.setSpacing(2)
        self._roi_list_widget.hide()

        roi_scroll = QScrollArea()
        roi_scroll.setWidget(self._roi_list_widget)
        roi_scroll.setWidgetResizable(True)
        roi_scroll.setMaximumHeight(80)
        roi_scroll.setStyleSheet(
            "QScrollArea { border: none; background-color: transparent; }"
        )
        self._roi_scroll = roi_scroll
        self._roi_scroll.hide()
        group_layout.addWidget(self._roi_scroll)

        # --- Action buttons ---
        action_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self._run_analysis)
        action_row.addWidget(self.btn_run)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setStyleSheet("background-color: #5c5c5c;")
        self.btn_clear.clicked.connect(self.clear_results)
        action_row.addWidget(self.btn_clear)
        group_layout.addLayout(action_row)

        # --- Status label for progress feedback ---
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(
            "font-size: 9px; color: #999999; font-style: italic;"
        )
        self._status_label.hide()
        group_layout.addWidget(self._status_label)

        # --- Separator ---
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #3f3f3f;")
        group_layout.addWidget(sep)

        # --- Toggle buttons for result views ---
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(2)
        self._toggle_buttons: Dict[str, QPushButton] = {}
        toggle_style_active = (
            "background-color: #0078d4; color: white; "
            "padding: 3px 6px; font-size: 9px; min-height: 14px; "
            "border-radius: 3px; font-weight: bold;"
        )
        toggle_style_inactive = (
            "background-color: #3f3f3f; color: #aaaaaa; "
            "padding: 3px 6px; font-size: 9px; min-height: 14px; "
            "border-radius: 3px;"
        )
        for key, label in [("summary", "Summary"), ("per_channel", "Per-Ch"),
                           ("coloc", "Coloc"), ("roi_density", "ROI")]:
            btn = QPushButton(label)
            btn.setStyleSheet(toggle_style_inactive)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, k=key: self._switch_view(k))
            toggle_row.addWidget(btn)
            self._toggle_buttons[key] = btn
        group_layout.addLayout(toggle_row)

        # --- Stacked widget for result views ---
        self._results_stack = QStackedWidget()

        # View 0: Summary
        self._summary_view = QWidget()
        sv_layout = QVBoxLayout(self._summary_view)
        sv_layout.setContentsMargins(0, 4, 0, 0)
        self.summary_label = QLabel("No results")
        self.summary_label.setStyleSheet(
            "font-size: 10px; color: #aaaaaa; padding: 2px 0px;"
        )
        self.summary_label.setWordWrap(True)
        sv_layout.addWidget(self.summary_label)
        sv_layout.addStretch()
        self._results_stack.addWidget(self._summary_view)

        # View 1: Per-Channel
        self._per_ch_view = QWidget()
        pc_layout = QVBoxLayout(self._per_ch_view)
        pc_layout.setContentsMargins(0, 4, 0, 0)
        pc_layout.setSpacing(4)

        # Channel selector buttons
        self._per_ch_selector_layout = QHBoxLayout()
        self._per_ch_selector_layout.setSpacing(2)
        pc_layout.addLayout(self._per_ch_selector_layout)

        self._per_ch_summary = QLabel("")
        self._per_ch_summary.setStyleSheet("font-size: 10px; color: #aaaaaa;")
        self._per_ch_summary.setWordWrap(True)
        pc_layout.addWidget(self._per_ch_summary)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["ID", "Area", "Mean Int.", "Circ.", "X, Y"]
        )
        self.results_table.setMaximumHeight(160)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.verticalHeader().setDefaultSectionSize(20)
        self.results_table.setStyleSheet("""
            QTableWidget {
                font-size: 9px;
                background-color: #1e1e1e;
                gridline-color: #3f3f3f;
            }
            QHeaderView::section {
                background-color: #2b2b2b;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                font-size: 9px;
                padding: 2px;
            }
        """)
        self.results_table.itemSelectionChanged.connect(self._on_row_selected)
        pc_layout.addWidget(self.results_table)
        self._results_stack.addWidget(self._per_ch_view)

        # View 2: Colocalization
        self._coloc_view = QWidget()
        cl_layout = QVBoxLayout(self._coloc_view)
        cl_layout.setContentsMargins(0, 4, 0, 0)
        self._coloc_label = QLabel("No colocalization data")
        self._coloc_label.setStyleSheet("font-size: 10px; color: #aaaaaa;")
        self._coloc_label.setWordWrap(True)
        cl_layout.addWidget(self._coloc_label)
        cl_layout.addStretch()
        self._results_stack.addWidget(self._coloc_view)

        # View 3: ROI Density
        self._roi_density_view = QWidget()
        rd_layout = QVBoxLayout(self._roi_density_view)
        rd_layout.setContentsMargins(0, 4, 0, 0)

        self.roi_density_table = QTableWidget()
        self.roi_density_table.setMaximumHeight(140)
        self.roi_density_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.roi_density_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.roi_density_table.horizontalHeader().setStretchLastSection(True)
        self.roi_density_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.roi_density_table.verticalHeader().setVisible(False)
        self.roi_density_table.verticalHeader().setDefaultSectionSize(20)
        self.roi_density_table.setStyleSheet("""
            QTableWidget {
                font-size: 9px;
                background-color: #1e1e1e;
                gridline-color: #3f3f3f;
            }
            QHeaderView::section {
                background-color: #2b2b2b;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                font-size: 9px;
                padding: 2px;
            }
        """)
        rd_layout.addWidget(self.roi_density_table)

        self._roi_density_label = QLabel("No ROI density data")
        self._roi_density_label.setStyleSheet("font-size: 10px; color: #aaaaaa;")
        rd_layout.addWidget(self._roi_density_label)
        rd_layout.addStretch()
        self._results_stack.addWidget(self._roi_density_view)

        group_layout.addWidget(self._results_stack)

        # --- Export row ---
        export_row = QHBoxLayout()
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_export_csv.setEnabled(False)
        export_row.addWidget(self.btn_export_csv)

        self.btn_export_overlay = QPushButton("Export Overlay")
        self.btn_export_overlay.clicked.connect(
            lambda: self.request_overlay_export.emit()
        )
        self.btn_export_overlay.setEnabled(False)
        export_row.addWidget(self.btn_export_overlay)
        group_layout.addLayout(export_row)

        # --- Batch button ---
        self.btn_batch = QPushButton("Batch Analyze All")
        self.btn_batch.setToolTip(
            "Run analysis on all loaded CZI files using current settings.\n"
            "Exports a combined CSV with results for every file."
        )
        self.btn_batch.clicked.connect(
            lambda: self.request_batch_analysis.emit()
        )
        group_layout.addWidget(self.btn_batch)

        # --- Settings persistence ---
        settings_row = QHBoxLayout()
        self.btn_save_settings = QPushButton("Save Settings")
        self.btn_save_settings.setStyleSheet("background-color: #5c5c5c;")
        self.btn_save_settings.setToolTip(
            "Save current analysis settings to a JSON file\n"
            "for reproducibility and sharing."
        )
        self.btn_save_settings.clicked.connect(self._save_settings)
        settings_row.addWidget(self.btn_save_settings)

        self.btn_load_settings = QPushButton("Load Settings")
        self.btn_load_settings.setStyleSheet("background-color: #5c5c5c;")
        self.btn_load_settings.setToolTip(
            "Load analysis settings from a previously saved JSON file."
        )
        self.btn_load_settings.clicked.connect(self._load_settings)
        settings_row.addWidget(self.btn_load_settings)
        group_layout.addLayout(settings_row)

        group.setLayout(group_layout)
        main_layout.addWidget(group)

        # Start on summary view
        self._switch_view("summary")

        # Track which per-channel tab is active
        self._active_per_ch_idx: Optional[int] = None
        self._per_ch_buttons: Dict[int, QPushButton] = {}

    # =========================================================================
    # Public methods (called by CZIViewerWindow)
    # =========================================================================

    def set_channels(self, channel_infos: list):
        """Populate per-channel config rows from CZI metadata."""
        # Clear existing rows
        for row in self._channel_rows.values():
            row.setParent(None)
            row.deleteLater()
        self._channel_rows.clear()
        self._channel_colors.clear()

        # Remove stretch from layout
        while self._channel_layout.count() > 0:
            item = self._channel_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        for info in channel_infos:
            row = ChannelQuantRow(
                info.index, info.name,
                getattr(info, 'suggested_color', 'gray'),
            )
            row.settings_changed.connect(self._on_channel_settings_changed)
            self._channel_layout.addWidget(row)
            self._channel_rows[info.index] = row
            self._channel_colors[info.index] = getattr(info, 'suggested_color', 'gray')

        self._channel_layout.addStretch()

    def get_enabled_channels(self) -> List[int]:
        """Return list of channel indices that are enabled for analysis."""
        return [idx for idx, row in self._channel_rows.items() if row.is_enabled()]

    def get_selected_channel_index(self) -> Optional[int]:
        """Return first enabled channel index (backward compat)."""
        enabled = self.get_enabled_channels()
        return enabled[0] if enabled else None

    def set_roi(self, x: int, y: int, w: int, h: int):
        """Called when user draws an ROI. Prompt for label."""
        self._roi_counter += 1
        default_label = f"ROI {self._roi_counter}"
        label, ok = QInputDialog.getText(
            self, "ROI Label", "Enter a label for this ROI:",
            text=default_label,
        )
        if not ok or not label.strip():
            label = default_label

        roi = ROIDefinition(
            label=label.strip(), x=x, y=y, w=w, h=h,
        )
        self._roi_definitions.append(roi)
        self._update_roi_list()
        self._emit_roi_overlay()

    def add_roi(self, label: str, x: int, y: int, w: int, h: int):
        """Add an ROI directly (without dialog)."""
        roi = ROIDefinition(label=label, x=x, y=y, w=w, h=h)
        self._roi_definitions.append(roi)
        self._update_roi_list()
        self._emit_roi_overlay()

    def clear_results(self):
        """Clear all results, overlay, and ROIs."""
        self.current_result = None
        self.results_table.setRowCount(0)
        self.summary_label.setText("No results")
        self._coloc_label.setText("No colocalization data")
        self._status_label.hide()
        self._per_ch_summary.setText("")
        self._roi_density_label.setText("No ROI density data")
        self.roi_density_table.setRowCount(0)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_overlay.setEnabled(False)
        self._clear_per_ch_selector()
        self.overlay_cleared.emit()
        self.quant_mode_exited.emit()

    def run_multi_with_data(self, images: Dict[int, np.ndarray],
                            pixel_size_um: Optional[float],
                            channel_names: Dict[int, str]):
        """Run multi-channel quantification with provided image data."""
        self._current_images = images
        self._current_pixel_size = pixel_size_um
        self._current_channel_names = channel_names

        config = self._build_multi_config()

        self.btn_run.setEnabled(False)
        self.btn_run.setText("Analyzing...")
        self._status_label.setText("Starting analysis...")
        self._status_label.show()

        self.worker = MultiChannelWorker(
            self.quantifier, images, config, pixel_size_um, channel_names,
        )
        self.worker.finished.connect(self._on_multi_analysis_complete)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.start()

    def preview_threshold_with_data(self, images: Dict[int, np.ndarray],
                                    pixel_size_um: Optional[float],
                                    channel_names: Dict[int, str]):
        """Compute binary masks for live preview (no particle analysis)."""
        self._current_images = images
        self._current_pixel_size = pixel_size_um
        self._current_channel_names = channel_names

        if not self.cb_show_mask.isChecked():
            self.overlay_cleared.emit()
            return

        channel_overlays = {}
        for ch_idx, image in images.items():
            row = self._channel_rows.get(ch_idx)
            if row is None or not row.is_enabled():
                continue

            method = row.get_method()
            if method == "manual":
                thresh_val = row.get_manual_threshold()
            else:
                thresh_val = self.quantifier.compute_threshold(image, method)

            binary_mask = self.quantifier.threshold_image(image, thresh_val)
            color = OVERLAY_COLORS.get(
                self._channel_colors.get(ch_idx, 'cyan'), (0, 255, 255)
            )

            channel_overlays[ch_idx] = {
                'binary_mask': binary_mask,
                'color': color,
                'centroids': [],
                'labels': [],
            }

        if channel_overlays:
            overlay = {
                'channel_overlays': channel_overlays,
                'overlap_mask': self._compute_overlap_mask(channel_overlays),
                'roi_definitions': self._roi_definitions,
                'show_mask': True,
                'contour_mode': self.cb_contour_mode.isChecked(),
                'highlight_channel': None,
                'highlight_idx': -1,
            }
            self.overlay_updated.emit(overlay)
        else:
            self.overlay_cleared.emit()

    # =========================================================================
    # Internal methods
    # =========================================================================

    def _build_multi_config(self) -> MultiChannelConfig:
        """Build MultiChannelConfig from per-channel UI rows."""
        bg_method = self.get_bg_method()
        channel_configs = {}
        for ch_idx, row in self._channel_rows.items():
            if not row.is_enabled():
                continue
            channel_configs[ch_idx] = QuantificationConfig(
                channel_index=ch_idx,
                threshold_method=row.get_method(),
                manual_threshold=row.get_manual_threshold(),
                min_area_um2=self.min_area_spin.value(),
                max_area_um2=self.max_area_spin.value(),
                use_watershed=self.cb_watershed.isChecked(),
                apply_bg_subtraction=(bg_method != "none"),
                roi=None,  # Full image always; ROIs used for density only
            )
        return MultiChannelConfig(
            channel_configs=channel_configs,
            roi_definitions=list(self._roi_definitions),
        )

    def get_bg_method(self) -> str:
        """Return BG subtraction method as lowercase string."""
        method_map = {
            "None": "none",
            "Rolling Ball": "rolling_ball",
            "Gaussian": "gaussian",
        }
        return method_map.get(self.bg_method_combo.currentText(), "none")

    def get_bg_radius(self) -> float:
        """Return BG subtraction radius."""
        return self.bg_radius_spin.value()

    def _run_analysis(self):
        """Request parent to provide data and run analysis."""
        self.quant_mode_entered.emit()
        self.request_analysis.emit()

    def _on_multi_analysis_complete(self, result: MultiChannelResult):
        """Handle completed multi-channel analysis."""
        self.current_result = result
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Analysis")
        self.btn_export_csv.setEnabled(True)
        self.btn_export_overlay.setEnabled(True)
        self._status_label.hide()

        self._populate_summary(result)
        self._populate_per_channel(result)
        self._populate_coloc(result)
        self._populate_roi_density(result)
        self._emit_multi_overlay(result)

        # Switch to summary view
        self._switch_view("summary")

    def _on_analysis_error(self, error_msg: str):
        """Handle analysis error."""
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run Analysis")
        self._status_label.setText("Analysis failed")
        QMessageBox.warning(
            self, "Analysis Error",
            f"Quantification failed:\n{error_msg}"
        )

    def _on_worker_progress(self, message: str, current: int, total: int):
        """Update status label with worker progress."""
        self._status_label.setText(message)
        self._status_label.show()

    # --- Result population ---

    def _populate_summary(self, result: MultiChannelResult):
        """Populate the summary view with multi-channel overview."""
        lines = []
        for ch_idx in sorted(result.channel_results.keys()):
            ch_res = result.channel_results[ch_idx]
            color = self._channel_colors.get(ch_idx, 'gray')
            color_rgb = OVERLAY_COLORS.get(color, (180, 180, 180))
            name = ch_res.channel_name or f"Ch{ch_idx}"

            if ch_res.mean_area_um2 is not None:
                area_str = f"{ch_res.mean_area_um2:.1f} \u00b5m\u00b2"
            else:
                area_str = f"{ch_res.mean_area_px:.0f} px"

            lines.append(
                f"<span style='color: rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]}); "
                f"font-weight: bold;'>{name}</span>: "
                f"{ch_res.particle_count} cells | "
                f"Area: {ch_res.area_fraction * 100:.1f}% | "
                f"Mean Int: {ch_res.mean_intensity:.3f}"
            )

        # Colocalization summary
        for coloc in result.colocalizations:
            lines.append(
                f"<span style='color: #cccccc;'>Coloc</span>: "
                f"{coloc.channel_a_name}\u2194{coloc.channel_b_name} = "
                f"{coloc.a_in_b_percent:.1f}% / {coloc.b_in_a_percent:.1f}% | "
                f"Dice: {coloc.dice_coefficient:.2f}"
            )

        self.summary_label.setText("<br>".join(lines) if lines else "No results")

    def _populate_per_channel(self, result: MultiChannelResult):
        """Set up per-channel selector buttons and populate first channel."""
        self._clear_per_ch_selector()

        ch_indices = sorted(result.channel_results.keys())
        for ch_idx in ch_indices:
            ch_res = result.channel_results[ch_idx]
            name = ch_res.channel_name or f"Ch{ch_idx}"
            color = self._channel_colors.get(ch_idx, 'gray')
            color_rgb = OVERLAY_COLORS.get(color, (180, 180, 180))

            btn = QPushButton(name)
            btn.setStyleSheet(
                f"padding: 2px 8px; font-size: 9px; min-height: 14px; "
                f"border-radius: 3px; background-color: #3f3f3f; "
                f"color: rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]});"
            )
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(
                lambda checked, idx=ch_idx: self._show_per_channel(idx)
            )
            self._per_ch_selector_layout.addWidget(btn)
            self._per_ch_buttons[ch_idx] = btn

        self._per_ch_selector_layout.addStretch()

        # Show first channel by default
        if ch_indices:
            self._show_per_channel(ch_indices[0])

    def _show_per_channel(self, ch_idx: int):
        """Display particle table for a specific channel."""
        self._active_per_ch_idx = ch_idx
        if self.current_result is None:
            return

        ch_res = self.current_result.channel_results.get(ch_idx)
        if ch_res is None:
            return

        # Update button styles
        for idx, btn in self._per_ch_buttons.items():
            color = self._channel_colors.get(idx, 'gray')
            color_rgb = OVERLAY_COLORS.get(color, (180, 180, 180))
            if idx == ch_idx:
                btn.setStyleSheet(
                    f"padding: 2px 8px; font-size: 9px; min-height: 14px; "
                    f"border-radius: 3px; background-color: #0078d4; "
                    f"color: white; font-weight: bold;"
                )
            else:
                btn.setStyleSheet(
                    f"padding: 2px 8px; font-size: 9px; min-height: 14px; "
                    f"border-radius: 3px; background-color: #3f3f3f; "
                    f"color: rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]});"
                )

        # Summary line
        if ch_res.mean_area_um2 is not None:
            area_str = f"{ch_res.mean_area_um2:.1f} \u00b5m\u00b2"
            total_str = f"{ch_res.total_area_um2:.1f} \u00b5m\u00b2"
        else:
            area_str = f"{ch_res.mean_area_px:.0f} px"
            total_str = f"{ch_res.total_area_px:.0f} px"

        self._per_ch_summary.setText(
            f"Count: {ch_res.particle_count}  |  "
            f"Area Fraction: {ch_res.area_fraction * 100:.1f}%\n"
            f"Mean Area: {area_str}  |  "
            f"Mean Int: {ch_res.mean_intensity:.4f}\n"
            f"Total Area: {total_str}  |  "
            f"Integ. Density: {ch_res.integrated_density:.2f}"
        )

        # Populate table
        self._populate_table(ch_res)

    def _populate_table(self, result: QuantificationResult):
        """Fill the results table with particle measurements."""
        particles = result.particles
        display_count = min(len(particles), self.MAX_TABLE_ROWS)

        self.results_table.setRowCount(display_count)
        for i, p in enumerate(particles[:display_count]):
            id_item = QTableWidgetItem(str(p.label))
            id_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 0, id_item)

            if p.area_um2 is not None:
                area_text = f"{p.area_um2:.1f}"
            else:
                area_text = f"{p.area_px:.0f}"
            area_item = QTableWidgetItem(area_text)
            area_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.results_table.setItem(i, 1, area_item)

            int_item = QTableWidgetItem(f"{p.mean_intensity:.4f}")
            int_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.results_table.setItem(i, 2, int_item)

            circ_item = QTableWidgetItem(f"{p.circularity:.2f}")
            circ_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.results_table.setItem(i, 3, circ_item)

            pos_item = QTableWidgetItem(f"{p.centroid_x:.0f}, {p.centroid_y:.0f}")
            pos_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 4, pos_item)

        if len(particles) > self.MAX_TABLE_ROWS:
            self._per_ch_summary.setText(
                self._per_ch_summary.text() +
                f"\n(showing {self.MAX_TABLE_ROWS} of {len(particles)} in table)"
            )

    def _populate_coloc(self, result: MultiChannelResult):
        """Populate the colocalization view."""
        if not result.colocalizations:
            self._coloc_label.setText(
                "No colocalization data\n(requires 2+ channels)"
            )
            return

        lines = []
        for coloc in result.colocalizations:
            name_a = coloc.channel_a_name
            name_b = coloc.channel_b_name

            # Get total particle counts for context
            total_a = len(result.channel_results[coloc.channel_a_idx].particles) if coloc.channel_a_idx in result.channel_results else 0
            total_b = len(result.channel_results[coloc.channel_b_idx].particles) if coloc.channel_b_idx in result.channel_results else 0

            lines.append(
                f"<b>{name_a} \u2194 {name_b}</b>"
            )
            lines.append(
                f"  {name_a}\u2192{name_b}: "
                f"{coloc.a_in_b_count}/{total_a} "
                f"({coloc.a_in_b_percent:.1f}%)"
            )
            lines.append(
                f"  {name_b}\u2192{name_a}: "
                f"{coloc.b_in_a_count}/{total_b} "
                f"({coloc.b_in_a_percent:.1f}%)"
            )
            lines.append(
                f"  Dice: {coloc.dice_coefficient:.3f}"
            )
            if coloc.overlap_area_um2 is not None:
                lines.append(f"  Overlap: {coloc.overlap_area_um2:.1f} \u00b5m\u00b2")
            else:
                lines.append(f"  Overlap: {coloc.overlap_area_px:.0f} px")

            lines.append(
                f"  Cross-intensity:"
            )
            lines.append(
                f"    {name_a} in {name_b} cells: {coloc.mean_a_intensity_in_b:.4f}"
            )
            lines.append(
                f"    {name_b} in {name_a} cells: {coloc.mean_b_intensity_in_a:.4f}"
            )
            lines.append("")

        self._coloc_label.setText("\n".join(lines))

    def _populate_roi_density(self, result: MultiChannelResult):
        """Populate the ROI density view table."""
        if not result.roi_densities:
            self._roi_density_label.setText("No ROI density data\n(add ROIs before analysis)")
            self.roi_density_table.setRowCount(0)
            self.roi_density_table.setColumnCount(0)
            return

        self._roi_density_label.setText("")

        # Build dynamic columns based on channels present
        ch_indices = sorted(result.channel_results.keys())
        ch_names = {
            idx: result.channel_results[idx].channel_name or f"Ch{idx}"
            for idx in ch_indices
        }

        # Columns: ROI | Ch1 Count | Ch1 Density | Ch2 Count | Ch2 Density | ...
        headers = ["ROI"]
        for idx in ch_indices:
            headers.append(f"{ch_names[idx]} Count")
            headers.append(f"{ch_names[idx]} /mm\u00b2")

        self.roi_density_table.setColumnCount(len(headers))
        self.roi_density_table.setHorizontalHeaderLabels(headers)
        self.roi_density_table.setRowCount(len(result.roi_densities))

        for row_i, rd in enumerate(result.roi_densities):
            roi_item = QTableWidgetItem(rd.roi.label)
            roi_item.setTextAlignment(Qt.AlignCenter)
            self.roi_density_table.setItem(row_i, 0, roi_item)

            col = 1
            for ch_idx in ch_indices:
                metrics = rd.channel_results.get(ch_idx)
                if metrics:
                    count_item = QTableWidgetItem(str(metrics.particle_count))
                    count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.roi_density_table.setItem(row_i, col, count_item)

                    density_text = (
                        f"{metrics.density_per_mm2:.1f}"
                        if metrics.density_per_mm2 is not None else "N/A"
                    )
                    density_item = QTableWidgetItem(density_text)
                    density_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.roi_density_table.setItem(row_i, col + 1, density_item)
                col += 2

    def _clear_per_ch_selector(self):
        """Remove per-channel selector buttons."""
        while self._per_ch_selector_layout.count() > 0:
            item = self._per_ch_selector_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._per_ch_buttons.clear()
        self._active_per_ch_idx = None

    # --- Overlay ---

    def _emit_multi_overlay(self, result: MultiChannelResult):
        """Build multi-channel overlay dict and emit."""
        channel_overlays = {}
        for ch_idx, ch_res in result.channel_results.items():
            if ch_res.binary_mask is None:
                continue
            color = OVERLAY_COLORS.get(
                self._channel_colors.get(ch_idx, 'cyan'), (0, 255, 255)
            )
            channel_overlays[ch_idx] = {
                'binary_mask': ch_res.binary_mask,
                'color': color,
                'centroids': [
                    (p.centroid_x, p.centroid_y) for p in ch_res.particles
                ],
                'labels': [p.label for p in ch_res.particles],
            }

        overlay = {
            'channel_overlays': channel_overlays,
            'overlap_mask': self._compute_overlap_mask(channel_overlays),
            'roi_definitions': self._roi_definitions,
            'show_mask': self.cb_show_mask.isChecked(),
            'contour_mode': self.cb_contour_mode.isChecked(),
            'highlight_channel': self._active_per_ch_idx,
            'highlight_idx': -1,
        }
        self.overlay_updated.emit(overlay)

    def _compute_overlap_mask(self, channel_overlays: dict) -> Optional[np.ndarray]:
        """Compute overlap mask between all channel binary masks."""
        masks = [
            co['binary_mask'] for co in channel_overlays.values()
            if co.get('binary_mask') is not None
        ]
        if len(masks) < 2:
            return None

        # Start with first two masks and AND them
        overlap = masks[0] & masks[1]
        for m in masks[2:]:
            overlap = overlap | (masks[0] & m) | (masks[1] & m)

        return overlap if overlap.any() else None

    def _emit_roi_overlay(self):
        """Emit overlay update with just ROI rectangles (no analysis data)."""
        if self.current_result is not None:
            self._emit_multi_overlay(self.current_result)
        elif self._roi_definitions:
            overlay = {
                'channel_overlays': {},
                'overlap_mask': None,
                'roi_definitions': self._roi_definitions,
                'show_mask': False,
                'contour_mode': self.cb_contour_mode.isChecked(),
                'highlight_channel': None,
                'highlight_idx': -1,
            }
            self.overlay_updated.emit(overlay)

    # --- View toggle ---

    def _switch_view(self, view_key: str):
        """Switch between result views."""
        view_map = {
            "summary": 0,
            "per_channel": 1,
            "coloc": 2,
            "roi_density": 3,
        }
        idx = view_map.get(view_key, 0)
        self._results_stack.setCurrentIndex(idx)

        # Update button styles
        active_style = (
            "background-color: #0078d4; color: white; "
            "padding: 3px 6px; font-size: 9px; min-height: 14px; "
            "border-radius: 3px; font-weight: bold;"
        )
        inactive_style = (
            "background-color: #3f3f3f; color: #aaaaaa; "
            "padding: 3px 6px; font-size: 9px; min-height: 14px; "
            "border-radius: 3px;"
        )
        for key, btn in self._toggle_buttons.items():
            btn.setStyleSheet(active_style if key == view_key else inactive_style)

    # --- UI event handlers ---

    def _on_channel_settings_changed(self):
        """Handle per-channel settings change (trigger preview)."""
        if self.cb_show_mask.isChecked() and self._current_images is not None:
            self._preview_timer.start()

    def _on_bg_method_changed(self, method_text: str):
        """Show/hide BG radius based on method selection."""
        is_radius = method_text in ("Rolling Ball", "Gaussian")
        self._bg_radius_widget.setVisible(is_radius)

    def _on_mask_toggle(self, state):
        """Handle show/hide binary mask toggle."""
        if self.current_result is not None:
            self._emit_multi_overlay(self.current_result)
        elif state and self._current_images is not None:
            self._preview_timer.start()
        else:
            self.overlay_cleared.emit()

    def _on_row_selected(self):
        """Highlight the selected particle on the overlay."""
        if self.current_result is None or self._active_per_ch_idx is None:
            return

        rows = self.results_table.selectionModel().selectedRows()
        if not rows:
            return

        row_idx = rows[0].row()
        ch_res = self.current_result.channel_results.get(self._active_per_ch_idx)
        if ch_res is None or row_idx >= len(ch_res.particles):
            return

        # Re-emit overlay with highlight
        channel_overlays = {}
        for ch_idx, ch_r in self.current_result.channel_results.items():
            if ch_r.binary_mask is None:
                continue
            color = OVERLAY_COLORS.get(
                self._channel_colors.get(ch_idx, 'cyan'), (0, 255, 255)
            )
            channel_overlays[ch_idx] = {
                'binary_mask': ch_r.binary_mask,
                'color': color,
                'centroids': [
                    (p.centroid_x, p.centroid_y) for p in ch_r.particles
                ],
                'labels': [p.label for p in ch_r.particles],
            }

        overlay = {
            'channel_overlays': channel_overlays,
            'overlap_mask': self._compute_overlap_mask(channel_overlays),
            'roi_definitions': self._roi_definitions,
            'show_mask': self.cb_show_mask.isChecked(),
            'contour_mode': self.cb_contour_mode.isChecked(),
            'highlight_channel': self._active_per_ch_idx,
            'highlight_idx': row_idx,
        }
        self.overlay_updated.emit(overlay)

        # Zoom to the selected particle
        particle = ch_res.particles[row_idx]
        self.zoom_to_particle.emit(particle.centroid_x, particle.centroid_y)

    def _do_threshold_preview(self):
        """Execute debounced threshold preview."""
        self.request_threshold_preview.emit()

    def _on_add_roi(self):
        """Enter ROI drawing mode."""
        self.request_roi_mode.emit()

    def _on_clear_rois(self):
        """Clear all ROI definitions."""
        self._roi_definitions.clear()
        self._roi_counter = 0
        self._update_roi_list()
        if self.current_result is not None:
            self._emit_multi_overlay(self.current_result)
        else:
            self.overlay_cleared.emit()

    def _remove_roi(self, index: int):
        """Remove a specific ROI by index."""
        if 0 <= index < len(self._roi_definitions):
            self._roi_definitions.pop(index)
            self._update_roi_list()
            self._emit_roi_overlay()

    def _update_roi_list(self):
        """Update the ROI list display."""
        # Clear existing
        while self._roi_list_layout.count() > 0:
            item = self._roi_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._roi_definitions:
            self._roi_list_widget.hide()
            self._roi_scroll.hide()
            return

        for i, roi in enumerate(self._roi_definitions):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(4)

            label = QLabel(f"\u2022 {roi.label}: ({roi.x}, {roi.y}) {roi.w}\u00d7{roi.h}")
            label.setStyleSheet("font-size: 9px; color: #aaaaaa;")
            row_layout.addWidget(label, 1)

            remove_btn = QPushButton("\u2715")
            remove_btn.setFixedSize(16, 16)
            remove_btn.setStyleSheet(
                "background-color: #5c5c5c; color: #cccccc; "
                "font-size: 10px; padding: 0; border-radius: 2px; min-height: 14px;"
            )
            remove_btn.clicked.connect(
                lambda checked, idx=i: self._remove_roi(idx)
            )
            row_layout.addWidget(remove_btn)

            self._roi_list_layout.addWidget(row_widget)

        self._roi_list_widget.show()
        self._roi_scroll.show()

    # --- CSV Export ---

    def _export_csv(self):
        """Export multi-channel quantification results to CSV."""
        if self.current_result is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Quantification Results", "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if not filepath:
            return

        result = self.current_result
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)

                writer.writerow(["# Multi-Channel Quantification Results"])
                writer.writerow(["# Pixel Size (um)",
                                 result.pixel_size_um or "N/A"])
                writer.writerow([])

                # Per-channel sections
                for ch_idx in sorted(result.channel_results.keys()):
                    ch_res = result.channel_results[ch_idx]
                    ch_name = ch_res.channel_name or f"Channel {ch_idx}"

                    writer.writerow([f"# {ch_name} - Summary"])
                    writer.writerow(["# Threshold Method", ch_res.threshold_method])
                    writer.writerow(["# Threshold Value",
                                     f"{ch_res.threshold_value:.6f}"])
                    writer.writerow(["# Particle Count", ch_res.particle_count])
                    writer.writerow(["# Total Area (px)",
                                     f"{ch_res.total_area_px:.2f}"])
                    if ch_res.total_area_um2 is not None:
                        writer.writerow(["# Total Area (um2)",
                                         f"{ch_res.total_area_um2:.2f}"])
                    writer.writerow(["# Mean Area (px)",
                                     f"{ch_res.mean_area_px:.2f}"])
                    if ch_res.mean_area_um2 is not None:
                        writer.writerow(["# Mean Area (um2)",
                                         f"{ch_res.mean_area_um2:.2f}"])
                    writer.writerow(["# Mean Intensity",
                                     f"{ch_res.mean_intensity:.6f}"])
                    writer.writerow(["# Area Fraction",
                                     f"{ch_res.area_fraction:.6f}"])
                    writer.writerow(["# Integrated Density",
                                     f"{ch_res.integrated_density:.4f}"])
                    writer.writerow(["# Watershed", ch_res.watershed_used])
                    writer.writerow([])

                    # Particle data
                    has_um = (result.pixel_size_um is not None
                              and result.pixel_size_um > 0)
                    header = ["Label", "Area (px)"]
                    if has_um:
                        header.append("Area (um2)")
                    header.extend([
                        "Centroid X (px)", "Centroid Y (px)",
                        "Mean Intensity", "Integrated Intensity",
                        "Perimeter (px)", "Circularity",
                        "BBox Min Row", "BBox Min Col",
                        "BBox Max Row", "BBox Max Col",
                    ])
                    writer.writerow(header)

                    for p in ch_res.particles:
                        row = [p.label, f"{p.area_px:.2f}"]
                        if has_um:
                            row.append(
                                f"{p.area_um2:.2f}"
                                if p.area_um2 is not None else ""
                            )
                        row.extend([
                            f"{p.centroid_x:.2f}", f"{p.centroid_y:.2f}",
                            f"{p.mean_intensity:.6f}",
                            f"{p.integrated_intensity:.4f}",
                            f"{p.perimeter_px:.2f}", f"{p.circularity:.4f}",
                            p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3],
                        ])
                        writer.writerow(row)
                    writer.writerow([])

                # Colocalization section
                if result.colocalizations:
                    writer.writerow(["# Colocalization"])
                    writer.writerow([
                        "Channel A", "Channel B",
                        "A-in-B Count", "A-in-B %",
                        "B-in-A Count", "B-in-A %",
                        "Dice", "Overlap Area (px)",
                        "Overlap Area (um2)",
                        "Mean A Intensity in B",
                        "Mean B Intensity in A",
                    ])
                    for coloc in result.colocalizations:
                        writer.writerow([
                            coloc.channel_a_name, coloc.channel_b_name,
                            coloc.a_in_b_count,
                            f"{coloc.a_in_b_percent:.2f}",
                            coloc.b_in_a_count,
                            f"{coloc.b_in_a_percent:.2f}",
                            f"{coloc.dice_coefficient:.4f}",
                            f"{coloc.overlap_area_px:.2f}",
                            f"{coloc.overlap_area_um2:.2f}"
                            if coloc.overlap_area_um2 is not None else "N/A",
                            f"{coloc.mean_a_intensity_in_b:.6f}",
                            f"{coloc.mean_b_intensity_in_a:.6f}",
                        ])
                    writer.writerow([])

                # ROI Density section
                if result.roi_densities:
                    writer.writerow(["# ROI Density"])
                    writer.writerow([
                        "ROI", "Channel", "Count",
                        "Density (cells/mm2)", "Area Fraction",
                        "Mean Intensity",
                    ])
                    for rd in result.roi_densities:
                        for ch_idx in sorted(rd.channel_results.keys()):
                            m = rd.channel_results[ch_idx]
                            writer.writerow([
                                rd.roi.label, m.channel_name,
                                m.particle_count,
                                f"{m.density_per_mm2:.2f}"
                                if m.density_per_mm2 is not None else "N/A",
                                f"{m.area_fraction:.6f}",
                                f"{m.mean_intensity:.6f}",
                            ])
                    writer.writerow([])

            # Auto-save config alongside CSV
            config_path = os.path.splitext(filepath)[0] + "_config.json"
            try:
                settings = self._collect_settings()
                with open(config_path, 'w') as cf:
                    json.dump(settings, cf, indent=2)
            except Exception:
                pass  # Don't fail the export over config save

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export CSV:\n{str(e)}"
            )

    # --- Settings persistence ---

    def _save_settings(self):
        """Save current analysis settings to JSON."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Settings", "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not filepath:
            return

        try:
            settings = self._collect_settings()
            with open(filepath, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            QMessageBox.critical(
                self, "Save Error",
                f"Failed to save settings:\n{str(e)}"
            )

    def _load_settings(self):
        """Load analysis settings from JSON."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Analysis Settings", "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                settings = json.load(f)
            self._apply_settings(settings)
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load settings:\n{str(e)}"
            )

    def _collect_settings(self) -> dict:
        """Collect all current UI settings into a serializable dict."""
        channel_settings = {}
        for ch_idx, row in self._channel_rows.items():
            channel_settings[str(ch_idx)] = {
                'enabled': row.is_enabled(),
                'method': row.method_combo.currentText(),
                'manual_threshold': row.get_manual_threshold(),
            }

        return {
            'version': 1,
            'bg_method': self.bg_method_combo.currentText(),
            'bg_radius': self.bg_radius_spin.value(),
            'min_area_um2': self.min_area_spin.value(),
            'max_area_um2': self.max_area_spin.value(),
            'use_watershed': self.cb_watershed.isChecked(),
            'show_mask': self.cb_show_mask.isChecked(),
            'contour_mode': self.cb_contour_mode.isChecked(),
            'channels': channel_settings,
        }

    def _apply_settings(self, settings: dict):
        """Apply loaded settings to the UI widgets."""
        if not isinstance(settings, dict):
            return

        # Global settings
        if 'bg_method' in settings:
            idx = self.bg_method_combo.findText(settings['bg_method'])
            if idx >= 0:
                self.bg_method_combo.setCurrentIndex(idx)

        if 'bg_radius' in settings:
            self.bg_radius_spin.setValue(float(settings['bg_radius']))

        if 'min_area_um2' in settings:
            self.min_area_spin.setValue(float(settings['min_area_um2']))

        if 'max_area_um2' in settings:
            self.max_area_spin.setValue(float(settings['max_area_um2']))

        if 'use_watershed' in settings:
            self.cb_watershed.setChecked(bool(settings['use_watershed']))

        if 'show_mask' in settings:
            self.cb_show_mask.setChecked(bool(settings['show_mask']))

        if 'contour_mode' in settings:
            self.cb_contour_mode.setChecked(bool(settings['contour_mode']))

        # Per-channel settings (match by index)
        ch_settings = settings.get('channels', {})
        for ch_key, ch_data in ch_settings.items():
            ch_idx = int(ch_key)
            row = self._channel_rows.get(ch_idx)
            if row is None:
                continue

            if 'enabled' in ch_data:
                row.enable_cb.setChecked(bool(ch_data['enabled']))

            if 'method' in ch_data:
                idx = row.method_combo.findText(ch_data['method'])
                if idx >= 0:
                    row.method_combo.setCurrentIndex(idx)

            if 'manual_threshold' in ch_data:
                row.thresh_slider.setValue(
                    int(float(ch_data['manual_threshold']) * 1000)
                )
