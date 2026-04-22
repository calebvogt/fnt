"""
Deep Audio Detector - YOLO-based ML detection and labeling tool.

A comprehensive PyQt5 application for:
1. Creating and managing YOLO detection projects
2. Loading and browsing audio files
3. Manual labeling and ground-truthing
4. Training YOLO models for automated USV detection
5. Running ML inference on audio files

Follows a SLEAP/DAS paradigm where the project IS the model.

Author: FNT Project
"""

import math
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QTimer, QSettings, QUrl
from PyQt5.QtGui import QFont, QColor, QPainter, QImage, QPen, QBrush, QPolygonF, QKeySequence, QDesktopServices
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QGroupBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QListWidgetItem,
    QScrollArea, QSplitter, QStatusBar, QMessageBox, QScrollBar,
    QSizePolicy, QFrame, QCheckBox, QShortcut, QSlider,
    QDialog, QDialogButtonBox, QAction, QMenu, QInputDialog,
    QRadioButton, QButtonGroup, QFormLayout, QGridLayout,
)
from scipy import signal

from fnt.usv.audio_widgets import SpectrogramWidget, WaveformOverviewWidget
from fnt.usv.usv_detector.labels_io import (
    DAD_SUFFIX, load_labels, save_labels,
    find_existing_sibling_csv, merge_with_inference, scan_folders_for_wavs,
)

# Optional imports
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


# =============================================================================
# Worker Threads
# =============================================================================

class YOLOTrainingWorker(QThread):
    progress = pyqtSignal(str)       # status message
    epoch = pyqtSignal(int, dict)    # (epoch_index, metrics dict) — for live plot
    complete = pyqtSignal(str)       # model_path
    error = pyqtSignal(str)          # error message

    def __init__(self, dataset_yaml, output_dir, model_name,
                 pretrained_weights=None, model_variant='yolo11s.pt', imgsz=1280):
        super().__init__()
        self.dataset_yaml = dataset_yaml
        self.output_dir = output_dir
        self.model_name = model_name
        self.pretrained_weights = pretrained_weights
        self.model_variant = model_variant
        self.imgsz = imgsz

    def run(self):
        try:
            from fnt.usv.usv_detector.yolo_detector import train_yolo_model
            self.progress.emit(
                f"Training {self.model_name} ({self.model_variant}, imgsz={self.imgsz})..."
            )

            def _emit_epoch(epoch_idx, metrics):
                # Runs on ultralytics' training thread; Qt signals are
                # queued automatically across threads.
                self.epoch.emit(int(epoch_idx), dict(metrics or {}))

            model_path = train_yolo_model(
                self.dataset_yaml, self.output_dir, self.model_name,
                pretrained_weights=self.pretrained_weights,
                model_variant=self.model_variant,
                imgsz=self.imgsz,
                epoch_callback=_emit_epoch,
            )
            self.complete.emit(model_path)
        except Exception as e:
            self.error.emit(str(e))


class YOLOInferenceWorker(QThread):
    progress = pyqtSignal(str, int, int)  # filename, current, total
    file_complete = pyqtSignal(str, str, list, int)  # filename, filepath, detections, count
    all_complete = pyqtSignal(dict)  # results
    error = pyqtSignal(str, str)  # filename, error

    def __init__(self, files, model_path, config, confidence_threshold=None):
        super().__init__()
        self.files = files
        self.model_path = model_path
        self.config = config
        self.confidence_threshold = confidence_threshold

    def run(self):
        from fnt.usv.usv_detector.yolo_detector import run_yolo_inference
        results = {}
        for i, filepath in enumerate(self.files):
            filename = os.path.basename(filepath)
            try:
                self.progress.emit(filename, i + 1, len(self.files))
                detections = run_yolo_inference(
                    self.model_path, filepath, self.config,
                    confidence_threshold=self.confidence_threshold,
                )
                results[filepath] = detections
                self.file_complete.emit(filename, filepath, detections, len(detections))
            except Exception as e:
                self.error.emit(filename, str(e))
        self.all_complete.emit(results)


# =============================================================================
# Predict-menu dialogs (SLEAP-style)
# =============================================================================

class RunTrainingDialog(QDialog):
    """Options for ``Predict → Run Training…``.

    Collects model variant, image size, and (optionally) a chained inference
    pass. Returns via accessor methods; the main window applies the values
    and kicks off training.
    """

    def __init__(self, parent, *, n_positive, n_negative, n_files,
                 model_variant='yolo11s.pt', imgsz=1280,
                 auto_predict=False, auto_predict_scope='all'):
        super().__init__(parent)
        self.setWindowTitle("Run Training")
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)

        summary = QLabel(
            f"Training data: <b>{n_positive}</b> positive, "
            f"<b>{n_negative}</b> negative across <b>{n_files}</b> file(s).<br>"
            "Tiles are split 80/20 train/val per source file. Training stops "
            "automatically when validation loss plateaus."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        form = QFormLayout()
        form.setContentsMargins(0, 8, 0, 8)

        self.combo_model_variant = QComboBox()
        self.combo_model_variant.addItems(
            ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolov8n.pt']
        )
        idx = self.combo_model_variant.findText(model_variant)
        if idx >= 0:
            self.combo_model_variant.setCurrentIndex(idx)
        self.combo_model_variant.setToolTip(
            "YOLO weights to fine-tune from.\n"
            "• yolo11n — fastest, smallest, weakest recall on faint calls\n"
            "• yolo11s — recommended balance for small-object detection\n"
            "• yolo11m — highest accuracy, ~2× training time\n"
            "• yolov8n — legacy; only choose to reproduce older models\n"
            "Ultralytics auto-downloads the checkpoint on first use."
        )
        form.addRow("Model variant:", self.combo_model_variant)

        self.combo_imgsz = QComboBox()
        self.combo_imgsz.addItems(['640', '960', '1280', '1536'])
        self.combo_imgsz.setCurrentText(str(imgsz))
        self.combo_imgsz.setToolTip(
            "Training image size (px). Higher = better recall on small calls,\n"
            "slower training and more VRAM. 1280 is the standard for\n"
            "small-object detection; pick 1536 if tiny/low-SNR USVs are the\n"
            "bottleneck and you have the VRAM for it."
        )
        form.addRow("Image size:", self.combo_imgsz)

        layout.addLayout(form)

        # Optional chained inference group.
        chain_group = QGroupBox("After training")
        chain_layout = QVBoxLayout()
        self.chk_run_inference = QCheckBox("Also run inference with the new model")
        self.chk_run_inference.setChecked(bool(auto_predict))
        self.chk_run_inference.setToolTip(
            "When training finishes successfully, immediately run inference\n"
            "using the newly trained weights. Accepted calls are preserved;\n"
            "pending/ml rows are handled per your Run-Inference overwrite policy."
        )
        chain_layout.addWidget(self.chk_run_inference)

        scope_row = QHBoxLayout()
        scope_row.setContentsMargins(20, 0, 0, 0)
        self.radio_scope_current = QRadioButton("Current file only")
        self.radio_scope_current.setToolTip(
            "Run inference only on the currently displayed file."
        )
        self.radio_scope_all = QRadioButton("All project files")
        self.radio_scope_all.setToolTip(
            "Run inference on every .wav across all source folders of this project."
        )
        if auto_predict_scope == 'current':
            self.radio_scope_current.setChecked(True)
        else:
            self.radio_scope_all.setChecked(True)
        self._scope_group = QButtonGroup(self)
        self._scope_group.addButton(self.radio_scope_current)
        self._scope_group.addButton(self.radio_scope_all)
        scope_row.addWidget(self.radio_scope_current)
        scope_row.addWidget(self.radio_scope_all)
        scope_row.addStretch()
        chain_layout.addLayout(scope_row)

        self.chk_run_inference.toggled.connect(self.radio_scope_current.setEnabled)
        self.chk_run_inference.toggled.connect(self.radio_scope_all.setEnabled)
        self.radio_scope_current.setEnabled(self.chk_run_inference.isChecked())
        self.radio_scope_all.setEnabled(self.chk_run_inference.isChecked())

        chain_group.setLayout(chain_layout)
        layout.addWidget(chain_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Train")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def model_variant(self) -> str:
        return self.combo_model_variant.currentText()

    def imgsz(self) -> int:
        return int(self.combo_imgsz.currentText())

    def auto_predict(self) -> bool:
        return self.chk_run_inference.isChecked()

    def auto_predict_scope(self) -> str:
        return 'current' if self.radio_scope_current.isChecked() else 'all'


class RunInferenceDialog(QDialog):
    """Options for ``Predict → Run Inference…``.

    Returns confidence threshold, SAHI toggle, scope (current/all), and
    overwrite policy for pending/ml predictions already on disk.
    """

    def __init__(self, parent, *, confidence=0.15, use_sahi=True,
                 scope='all', overwrite_policy='replace_pending',
                 has_current_file=True, any_pending_exists=False):
        super().__init__(parent)
        self.setWindowTitle("Run Inference")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Run the current YOLO model on project audio and merge the results "
            "into each file's sibling CSV. Accepted, rejected, and user-drawn "
            "labels are always preserved."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Confidence row
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self.slider_confidence = QSlider(Qt.Horizontal)
        self.slider_confidence.setMinimum(5)   # 0.05
        self.slider_confidence.setMaximum(60)  # 0.60
        self.slider_confidence.setValue(int(round(max(0.05, min(0.60, confidence)) * 100)))
        self.slider_confidence.setToolTip(
            "Detection confidence threshold.\n"
            "• Lower → higher recall: more candidate boxes you can reject\n"
            "  with R, including more false positives.\n"
            "• Higher → cleaner output, may miss quiet or short calls.\n"
            "0.15 is the project default for high-recall review."
        )
        self.lbl_confidence = QLabel(f"{self.slider_confidence.value()/100:.2f}")
        self.lbl_confidence.setMinimumWidth(36)
        self.slider_confidence.valueChanged.connect(
            lambda v: self.lbl_confidence.setText(f"{v/100:.2f}")
        )
        conf_row.addWidget(self.slider_confidence, 1)
        conf_row.addWidget(self.lbl_confidence)
        layout.addLayout(conf_row)

        # SAHI
        self.chk_use_sahi = QCheckBox("Use SAHI sliced inference")
        self.chk_use_sahi.setChecked(bool(use_sahi))
        self.chk_use_sahi.setToolTip(
            "SAHI (Slicing Aided Hyper Inference) runs the detector on overlapping\n"
            "tile-sized slices of each full spectrogram. Typically 2–3× slower\n"
            "than plain tiling but markedly better recall on small or sparse calls.\n"
            "Recommended on unless you're sanity-checking a run."
        )
        layout.addWidget(self.chk_use_sahi)

        # Scope
        scope_group = QGroupBox("Run on")
        scope_layout = QHBoxLayout()
        self.radio_scope_current = QRadioButton("Current file")
        self.radio_scope_current.setToolTip(
            "Run inference only on the currently displayed file."
        )
        self.radio_scope_all = QRadioButton("All project files")
        self.radio_scope_all.setToolTip(
            "Run inference on every .wav across all source folders of this project."
        )
        if scope == 'current' and has_current_file:
            self.radio_scope_current.setChecked(True)
        else:
            self.radio_scope_all.setChecked(True)
        if not has_current_file:
            self.radio_scope_current.setEnabled(False)
        self._scope_group = QButtonGroup(self)
        self._scope_group.addButton(self.radio_scope_current)
        self._scope_group.addButton(self.radio_scope_all)
        scope_layout.addWidget(self.radio_scope_current)
        scope_layout.addWidget(self.radio_scope_all)
        scope_layout.addStretch()
        scope_group.setLayout(scope_layout)
        layout.addWidget(scope_group)

        # Overwrite policy
        policy_group = QGroupBox("Existing predictions")
        policy_layout = QVBoxLayout()
        self.radio_replace = QRadioButton(
            "Replace pending predictions (recommended)"
        )
        self.radio_replace.setToolTip(
            "Before running, delete previous ML-generated pending rows from each\n"
            "target file. Fresh predictions fully replace stale ones. Accepted\n"
            "and rejected calls are untouched."
        )
        self.radio_merge = QRadioButton(
            "Merge with existing pending predictions"
        )
        self.radio_merge.setToolTip(
            "Keep previous pending predictions; add new non-overlapping ones.\n"
            "Use this if you're running the same model with a different\n"
            "confidence or slicing setup and want to union the results."
        )
        if overwrite_policy == 'merge_keep_pending':
            self.radio_merge.setChecked(True)
        else:
            self.radio_replace.setChecked(True)
        self._policy_group = QButtonGroup(self)
        self._policy_group.addButton(self.radio_replace)
        self._policy_group.addButton(self.radio_merge)
        policy_layout.addWidget(self.radio_replace)
        policy_layout.addWidget(self.radio_merge)

        note = QLabel(
            "<i>Note:</i> accepted calls are always preserved. New inference boxes "
            "that overlap an accepted call are automatically dropped, so you "
            "never end up with a duplicate pending box shadowing a curated one."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        policy_layout.addWidget(note)
        policy_group.setLayout(policy_layout)
        layout.addWidget(policy_group)

        if any_pending_exists:
            hint = QLabel(
                "One or more target files already contain pending predictions."
            )
            hint.setStyleSheet("color: #f0a500; font-size: 10px;")
            hint.setWordWrap(True)
            layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Run")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def confidence(self) -> float:
        return self.slider_confidence.value() / 100.0

    def use_sahi(self) -> bool:
        return self.chk_use_sahi.isChecked()

    def scope(self) -> str:
        return 'current' if self.radio_scope_current.isChecked() else 'all'

    def overwrite_policy(self) -> str:
        return 'merge_keep_pending' if self.radio_merge.isChecked() else 'replace_pending'


# =============================================================================
# Main Deep Audio Detector Window
# =============================================================================

class DeepAudioDetectorWindow(QMainWindow):
    """Main window for Deep Audio Detector."""

    BASE_TITLE = "FNT Deep Audio Detector"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.BASE_TITLE)
        self.setMinimumSize(1000, 700)
        self.resize(1400, 900)

        # Live training plot window (lazy; created on first train).
        self._training_plot = None

        self.audio_files = []  # List of audio file paths
        self.current_file_idx = 0
        self.audio_data = None
        self.sample_rate = None
        self.detections_df = None  # Current file's detections DataFrame
        self.current_detection_idx = 0
        self.all_detections = {}  # filepath -> DataFrame
        self.detection_sources = {}  # filepath -> source string

        # YOLO project state
        self._yolo_project_config = None
        self._yolo_model_path = None

        # Training/inference options — plain state; edited via the dialogs
        # under the Predict menu. Default values are used until the user
        # opens a project (whose config may override them) or edits them.
        self._model_variant = 'yolo11s.pt'
        self._imgsz = 1280
        self._confidence = 0.15
        self._use_sahi = True
        # Train→predict chain settings.
        self._auto_predict_after_train = False
        self._auto_predict_scope = 'all'  # 'current' | 'all'
        # Inference overwrite policy for existing predictions.
        # 'replace_pending' removes pending/ml rows before merging fresh ones;
        # 'merge_keep_pending' leaves existing pending rows alone.
        self._inference_overwrite_policy = 'replace_pending'
        # Label-count cache so menu-action enable state can stay in sync
        # without the old lbl_train_data widget.
        self._n_positive_labels = 0
        self._n_negative_labels = 0

        # Playback state
        self.is_playing = False
        self.playback_speed = 1.0
        self.use_heterodyne = False
        self._playback_start_time = None
        self._playback_start_s = None
        self._playback_end_s = None
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(30)
        self._playback_timer.timeout.connect(self._update_playback_position)

        # Undo stack
        self.undo_stack = deque(maxlen=50)

        # Filter state
        self.filter_status = 'all'

        # Workers
        self._yolo_train_worker = None
        self._yolo_infer_worker = None

        self._setup_ui()
        self._apply_styles()
        self._setup_shortcuts()
        self._setup_pan_timers()
        self._restore_settings()

    # =========================================================================
    # UI Setup
    # =========================================================================

    def _setup_ui(self):
        """Setup the main UI."""
        self._setup_menu_bar()

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left panel (scrollable).
        #
        # Width is font-metric-based so Windows display scaling (125%/150%)
        # gets proportionally more room than macOS at 100%. Horizontal scroll
        # is AsNeeded rather than AlwaysOff as a safety net if a row overflows.
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        _fm = self.fontMetrics()
        _min_w = max(380, _fm.averageCharWidth() * 60 + 40)
        _max_w = max(480, _fm.averageCharWidth() * 80 + 40)
        left_scroll.setMinimumWidth(_min_w)
        left_scroll.setMaximumWidth(_max_w)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)

        # Build sections — training & inference live in the Predict menu
        # dialogs, not in the left panel (SLEAP-style layout).
        self._create_project_section(left_layout)
        self._create_training_data_section(left_layout)
        self._create_detection_section(left_layout)
        self._create_labeling_section(left_layout)

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)

        # Right panel (spectrogram)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        # Spectrogram
        self.spectrogram = SpectrogramWidget()
        self.spectrogram.detection_selected.connect(self.on_detection_selected)
        self.spectrogram.box_adjusted.connect(self.on_box_adjusted)
        self.spectrogram.drag_complete.connect(self.on_drag_complete)
        self.spectrogram.zoom_requested.connect(self.on_wheel_zoom)
        right_layout.addWidget(self.spectrogram, 1)

        # Waveform overview strip
        self.waveform_overview = WaveformOverviewWidget()
        self.waveform_overview.view_changed.connect(self.on_overview_clicked)
        right_layout.addWidget(self.waveform_overview)

        # Scrollbar row
        scroll_bar_row = QWidget()
        scroll_layout = QHBoxLayout(scroll_bar_row)
        scroll_layout.setContentsMargins(5, 2, 5, 2)
        scroll_layout.setSpacing(2)

        self.btn_pan_left = QPushButton("<")
        self.btn_pan_left.setObjectName("small_btn")
        self.btn_pan_left.setFixedWidth(24)
        self.btn_pan_left.setToolTip("Pan the spectrogram view left in time")
        self.btn_pan_left.clicked.connect(self.pan_left)
        scroll_layout.addWidget(self.btn_pan_left)

        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.setMaximum(1000)
        self.time_scrollbar.setToolTip("Scroll through the recording timeline")
        self.time_scrollbar.valueChanged.connect(self.on_scrollbar_changed)
        scroll_layout.addWidget(self.time_scrollbar, 1)

        self.btn_pan_right = QPushButton(">")
        self.btn_pan_right.setObjectName("small_btn")
        self.btn_pan_right.setFixedWidth(24)
        self.btn_pan_right.setToolTip("Pan the spectrogram view right in time")
        self.btn_pan_right.clicked.connect(self.pan_right)
        scroll_layout.addWidget(self.btn_pan_right)

        right_layout.addWidget(scroll_bar_row)

        # Controls row
        controls_bar = QWidget()
        controls_layout = QHBoxLayout(controls_bar)
        controls_layout.setContentsMargins(5, 2, 5, 2)
        controls_layout.setSpacing(4)

        # Window / Zoom
        controls_layout.addWidget(QLabel("Window:"))

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setObjectName("small_btn")
        self.btn_zoom_out.setFixedWidth(24)
        self.btn_zoom_out.setToolTip("Zoom out (show more time)")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        controls_layout.addWidget(self.btn_zoom_out)

        self.spin_view_window = QDoubleSpinBox()
        self.spin_view_window.setRange(0.1, 600.0)
        self.spin_view_window.setValue(2.0)
        self.spin_view_window.setSuffix(" s")
        self.spin_view_window.setFixedWidth(80)
        self.spin_view_window.setToolTip("Time window duration (seconds).\nControls how much of the recording is visible.\nSmaller = zoomed in, larger = zoomed out.\nAlso adjustable with mouse wheel on spectrogram.")
        self.spin_view_window.valueChanged.connect(self.on_view_window_changed)
        controls_layout.addWidget(self.spin_view_window)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setObjectName("small_btn")
        self.btn_zoom_in.setFixedWidth(24)
        self.btn_zoom_in.setToolTip("Zoom in (show less time)")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        controls_layout.addWidget(self.btn_zoom_in)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep1)

        # Frequency display range
        controls_layout.addWidget(QLabel("Freq:"))
        self.spin_display_min_freq = QSpinBox()
        self.spin_display_min_freq.setRange(0, 200000)
        self.spin_display_min_freq.setSingleStep(5000)
        self.spin_display_min_freq.setValue(0)
        self.spin_display_min_freq.setSuffix(" Hz")
        self.spin_display_min_freq.setFixedWidth(90)
        self.spin_display_min_freq.setToolTip("Minimum frequency displayed on the spectrogram (Hz).\nAdjust to zoom into the frequency range of interest.")
        self.spin_display_min_freq.valueChanged.connect(self.on_display_freq_changed)
        controls_layout.addWidget(self.spin_display_min_freq)

        controls_layout.addWidget(QLabel("-"))
        self.spin_display_max_freq = QSpinBox()
        self.spin_display_max_freq.setRange(1000, 250000)
        self.spin_display_max_freq.setSingleStep(5000)
        self.spin_display_max_freq.setValue(125000)
        self.spin_display_max_freq.setSuffix(" Hz")
        self.spin_display_max_freq.setFixedWidth(90)
        self.spin_display_max_freq.setToolTip("Maximum frequency displayed on the spectrogram (Hz).")
        self.spin_display_max_freq.valueChanged.connect(self.on_display_freq_changed)
        controls_layout.addWidget(self.spin_display_max_freq)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep2)

        # Colormap selector
        controls_layout.addWidget(QLabel("Color Map:"))
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(['viridis', 'magma', 'inferno', 'grayscale'])
        self.combo_colormap.setFixedWidth(85)
        self.combo_colormap.setToolTip("Spectrogram color scheme.")
        self.combo_colormap.currentTextChanged.connect(self.on_colormap_changed)
        controls_layout.addWidget(self.combo_colormap)

        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep3)

        # Playback controls
        self.btn_play = QPushButton("Play")
        self.btn_play.setToolTip("Play the current detection audio (Space key).\nAudio is slowed down according to the speed setting\nso ultrasonic calls become audible.")
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_play.setEnabled(False)
        controls_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: #5c5c5c;")
        self.btn_stop.setToolTip("Stop audio playback immediately.")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        controls_layout.addWidget(self.btn_stop)

        controls_layout.addWidget(QLabel("Speed:"))
        self._speed_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.0]
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(0, len(self._speed_values) - 1)
        self.slider_speed.setValue(len(self._speed_values) - 1)
        self.slider_speed.setFixedWidth(100)
        self.slider_speed.setToolTip("Playback speed multiplier.\nLower values slow audio down, shifting\nultrasonic frequencies into the audible range.")
        self.slider_speed.valueChanged.connect(self.on_speed_changed)
        controls_layout.addWidget(self.slider_speed)
        self.lbl_speed = QLabel("1.0x")
        self.lbl_speed.setFixedWidth(35)
        controls_layout.addWidget(self.lbl_speed)

        if not HAS_SOUNDDEVICE:
            lbl_warn = QLabel("No audio")
            lbl_warn.setStyleSheet("color: #d13438; font-size: 9px;")
            controls_layout.addWidget(lbl_warn)

        controls_layout.addStretch()

        right_layout.addWidget(controls_bar)

        # Add to main layout
        main_layout.addWidget(left_scroll)
        main_layout.addWidget(right_panel, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Welcome to Deep Audio Detector - Create or open a project to begin")

    def _make_label(self, text, tooltip=None, min_width=0):
        """Helper to create a label with optional tooltip and minimum width."""
        lbl = QLabel(text)
        if tooltip:
            lbl.setToolTip(tooltip)
        if min_width > 0:
            lbl.setMinimumWidth(min_width)
        return lbl

    def _create_project_section(self, layout):
        """Create project status section.

        Project lifecycle (New / Open / Close / Add Folder) lives on the
        File menu; this section is informational only.
        """
        group = QGroupBox("1. Project")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        self.lbl_project_name = QLabel("No project — use File → New Project…")
        self.lbl_project_name.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_project_name.setWordWrap(True)
        group_layout.addWidget(self.lbl_project_name)

        self.lbl_source_folders = QLabel("No source folders")
        self.lbl_source_folders.setStyleSheet("color: #777777; font-size: 9px;")
        self.lbl_source_folders.setWordWrap(True)
        group_layout.addWidget(self.lbl_source_folders)

        self.lbl_model_info = QLabel("No trained model")
        self.lbl_model_info.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_model_info.setWordWrap(True)
        group_layout.addWidget(self.lbl_model_info)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _setup_menu_bar(self):
        """Construct DAD's menu bar inside the DAD window itself.

        DAD is one of many tools inside FieldNeuroToolbox; using the macOS
        application-global menu bar (``self.menuBar()`` with native rendering)
        would plaster File / Labels / Predict across every FNT window while
        DAD is loaded. We force the menu bar to render as a child widget of
        the DAD main window so it scopes cleanly to this tool.
        """
        menubar = self.menuBar()
        # On macOS this is the critical flag — it pulls the menu bar out of
        # the system menu and renders it inline at the top of the DAD window.
        menubar.setNativeMenuBar(False)
        file_menu = menubar.addMenu("&File")

        act_new = QAction("&New Project…", self)
        act_new.setShortcut(QKeySequence.New)
        act_new.triggered.connect(self._menu_new_project)
        file_menu.addAction(act_new)

        act_open = QAction("&Open Project…", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self._menu_open_project)
        file_menu.addAction(act_open)

        self.menu_recent = file_menu.addMenu("Open &Recent")
        self._rebuild_recent_projects_menu()

        file_menu.addSeparator()

        self.act_add_folder = QAction("&Add Folder…", self)
        self.act_add_folder.setShortcut("Ctrl+Shift+O")
        self.act_add_folder.triggered.connect(self._menu_add_folder)
        self.act_add_folder.setEnabled(False)
        file_menu.addAction(self.act_add_folder)

        self.act_add_files = QAction("Add &Files…", self)
        self.act_add_files.triggered.connect(self._add_audio_files)
        self.act_add_files.setEnabled(False)
        file_menu.addAction(self.act_add_files)

        file_menu.addSeparator()

        self.act_close_project = QAction("&Close Project", self)
        self.act_close_project.triggered.connect(self._menu_close_project)
        self.act_close_project.setEnabled(False)
        file_menu.addAction(self.act_close_project)

        labels_menu = menubar.addMenu("&Labels")
        self.act_import_cad = QAction("Import from &CAD CSV…", self)
        self.act_import_cad.setToolTip(
            "Import a Classic Audio Detector sidecar CSV for the current file "
            "as reviewable accepted labels."
        )
        self.act_import_cad.triggered.connect(self._menu_import_cad)
        self.act_import_cad.setEnabled(False)
        labels_menu.addAction(self.act_import_cad)

        # SLEAP-style Predict menu — training and inference both pop out into
        # their own dialogs so the main window stays focused on labeling.
        predict_menu = menubar.addMenu("&Predict")

        self.act_run_training = QAction("Run &Training…", self)
        self.act_run_training.setShortcut("Ctrl+T")
        self.act_run_training.setToolTip(
            "Export labeled tiles and fine-tune a YOLO model on this project.\n"
            "Opens a dialog where you pick the model variant, image size, and\n"
            "optionally chain an inference pass on the resulting weights."
        )
        self.act_run_training.triggered.connect(self._menu_run_training)
        self.act_run_training.setEnabled(False)
        predict_menu.addAction(self.act_run_training)

        self.act_run_inference = QAction("Run &Inference…", self)
        self.act_run_inference.setShortcut("Ctrl+I")
        self.act_run_inference.setToolTip(
            "Run the current model on one or all project files. Opens a dialog\n"
            "for confidence, SAHI, scope, and how to handle existing predictions.\n"
            "Accepted calls are always preserved."
        )
        self.act_run_inference.triggered.connect(self._menu_run_inference)
        self.act_run_inference.setEnabled(False)
        predict_menu.addAction(self.act_run_inference)

    def _create_training_data_section(self, layout):
        """Create training data / file management section."""
        group = QGroupBox("2. Training Data")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.setToolTip("Select WAV files to add")
        self.btn_add_files.clicked.connect(self._add_audio_files)
        group_layout.addWidget(self.btn_add_files)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        group_layout.addWidget(self.file_list)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)

        self.btn_prev_file = QPushButton("< Prev")
        self.btn_prev_file.setObjectName("small_btn")
        self.btn_prev_file.setToolTip("Load the previous file in the list")
        self.btn_prev_file.clicked.connect(self._prev_file)
        self.btn_prev_file.setEnabled(False)
        nav_row.addWidget(self.btn_prev_file)

        self.lbl_file_num = QLabel("File 0/0")
        self.lbl_file_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_file_num, 1)

        self.btn_next_file = QPushButton("Next >")
        self.btn_next_file.setObjectName("small_btn")
        self.btn_next_file.setToolTip("Load the next file in the list")
        self.btn_next_file.clicked.connect(self._next_file)
        self.btn_next_file.setEnabled(False)
        nav_row.addWidget(self.btn_next_file)

        group_layout.addLayout(nav_row)

        self.lbl_data_summary = QLabel("No files loaded")
        self.lbl_data_summary.setStyleSheet("color: #999999; font-size: 10px;")
        group_layout.addWidget(self.lbl_data_summary)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_detection_section(self, layout):
        """Create current detection section."""
        group = QGroupBox("3. Current Detection")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.setSpacing(2)
        filter_row.addWidget(QLabel("Show:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItem("All", "all")
        self.combo_filter.addItem("Pending", "pending")
        self.combo_filter.addItem("Accepted", "accepted")
        self.combo_filter.addItem("Rejected", "rejected")
        self.combo_filter.addItem("Negative", "negative")
        self.combo_filter.setToolTip("Filter which detections to navigate through")
        self.combo_filter.currentIndexChanged.connect(self.on_filter_changed)
        filter_row.addWidget(self.combo_filter, 1)

        filter_row.addWidget(QLabel("Go:"))
        self.spin_jump = QSpinBox()
        self.spin_jump.setRange(1, 1)
        self.spin_jump.setFixedWidth(60)
        self.spin_jump.setToolTip("Jump directly to a detection by number.\nType a number and press Enter.")
        self.spin_jump.editingFinished.connect(self.on_jump_to_detection)
        filter_row.addWidget(self.spin_jump)
        group_layout.addLayout(filter_row)

        # Navigation
        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)

        self.btn_prev_det = QPushButton("<")
        self.btn_prev_det.setObjectName("small_btn")
        self.btn_prev_det.setToolTip("Go to previous detection (Left arrow key)")
        self.btn_prev_det.clicked.connect(self.prev_detection)
        self.btn_prev_det.setEnabled(False)
        nav_row.addWidget(self.btn_prev_det)

        self.lbl_det_num = QLabel("Det 0/0")
        self.lbl_det_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_det_num, 1)

        self.btn_next_det = QPushButton(">")
        self.btn_next_det.setObjectName("small_btn")
        self.btn_next_det.setToolTip("Go to next detection (Right arrow key)")
        self.btn_next_det.clicked.connect(self.next_detection)
        self.btn_next_det.setEnabled(False)
        nav_row.addWidget(self.btn_next_det)

        group_layout.addLayout(nav_row)

        # Time
        time_row = QHBoxLayout()
        time_row.setSpacing(2)
        time_row.addWidget(QLabel("Start:"))
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setDecimals(4)
        self.spin_start.setRange(0, 9999)
        self.spin_start.setSuffix(" s")
        self.spin_start.setToolTip("Start time of the selected detection (seconds).")
        self.spin_start.valueChanged.connect(self.on_time_changed)
        time_row.addWidget(self.spin_start, 1)

        time_row.addWidget(QLabel("Stop:"))
        self.spin_stop = QDoubleSpinBox()
        self.spin_stop.setDecimals(4)
        self.spin_stop.setRange(0, 9999)
        self.spin_stop.setSuffix(" s")
        self.spin_stop.setToolTip("End time of the selected detection (seconds).")
        self.spin_stop.valueChanged.connect(self.on_time_changed)
        time_row.addWidget(self.spin_stop, 1)
        group_layout.addLayout(time_row)

        # Frequency
        freq_row = QHBoxLayout()
        freq_row.setSpacing(2)
        freq_row.addWidget(QLabel("Min F:"))
        self.spin_det_min_freq = QSpinBox()
        self.spin_det_min_freq.setRange(0, 200000)
        self.spin_det_min_freq.setSingleStep(1000)
        self.spin_det_min_freq.setSuffix(" Hz")
        self.spin_det_min_freq.setToolTip("Minimum frequency of the selected detection (Hz).")
        self.spin_det_min_freq.valueChanged.connect(self.on_freq_changed)
        freq_row.addWidget(self.spin_det_min_freq, 1)

        freq_row.addWidget(QLabel("Max:"))
        self.spin_det_max_freq = QSpinBox()
        self.spin_det_max_freq.setRange(0, 200000)
        self.spin_det_max_freq.setSingleStep(1000)
        self.spin_det_max_freq.setSuffix(" Hz")
        self.spin_det_max_freq.setToolTip("Maximum frequency of the selected detection (Hz).")
        self.spin_det_max_freq.valueChanged.connect(self.on_freq_changed)
        freq_row.addWidget(self.spin_det_max_freq, 1)
        group_layout.addLayout(freq_row)

        # Info
        self.lbl_det_info = QLabel("Peak: -- Hz | Dur: -- ms")
        self.lbl_det_info.setStyleSheet("color: #999999;")
        group_layout.addWidget(self.lbl_det_info)

        # Status
        self.lbl_det_status = QLabel("Status: --")
        self.lbl_det_status.setStyleSheet("font-weight: bold;")
        group_layout.addWidget(self.lbl_det_status)

        # Progress
        self.lbl_progress = QLabel("0/0 reviewed")
        self.lbl_progress.setStyleSheet("color: #999999;")
        group_layout.addWidget(self.lbl_progress)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        group_layout.addWidget(self.progress_bar)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_labeling_section(self, layout):
        """Create labeling/action section."""
        group = QGroupBox("4. Labeling")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Accept/Reject
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(2)

        self.btn_accept = QPushButton("Accept")
        self.btn_accept.setObjectName("accept_btn")
        self.btn_accept.setToolTip("Mark current detection as a valid USV call (A key).")
        self.btn_accept.clicked.connect(self.accept_detection)
        self.btn_accept.setEnabled(False)
        btn_row1.addWidget(self.btn_accept)

        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setObjectName("reject_btn")
        self.btn_reject.setToolTip("Mark current detection as a false positive (R key).")
        self.btn_reject.clicked.connect(self.reject_detection)
        self.btn_reject.setEnabled(False)
        btn_row1.addWidget(self.btn_reject)

        self.btn_skip = QPushButton("Skip")
        self.btn_skip.setStyleSheet("background-color: #5c5c5c;")
        self.btn_skip.clicked.connect(self.skip_detection)
        self.btn_skip.setEnabled(False)
        self.btn_skip.setToolTip("Skip to the next detection without changing\nits current status (S key).")
        btn_row1.addWidget(self.btn_skip)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setStyleSheet("background-color: #5c5c5c;")
        self.btn_undo.clicked.connect(self.undo_action)
        self.btn_undo.setEnabled(False)
        self.btn_undo.setToolTip("Undo last labeling action (Ctrl+Z).")
        btn_row1.addWidget(self.btn_undo)

        group_layout.addLayout(btn_row1)

        # Batch labeling
        batch_row = QHBoxLayout()
        batch_row.setSpacing(2)

        self.btn_accept_all_pending = QPushButton("Accept All Pending")
        self.btn_accept_all_pending.setObjectName("accept_btn")
        self.btn_accept_all_pending.setToolTip("Accept all unreviewed detections at once.")
        self.btn_accept_all_pending.clicked.connect(self.accept_all_pending)
        self.btn_accept_all_pending.setEnabled(False)
        batch_row.addWidget(self.btn_accept_all_pending)

        self.btn_reject_all_pending = QPushButton("Reject All Pending")
        self.btn_reject_all_pending.setObjectName("reject_btn")
        self.btn_reject_all_pending.setToolTip("Reject all unreviewed detections at once.")
        self.btn_reject_all_pending.clicked.connect(self.reject_all_pending)
        self.btn_reject_all_pending.setEnabled(False)
        batch_row.addWidget(self.btn_reject_all_pending)

        group_layout.addLayout(batch_row)

        # Add/Delete
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(2)

        self.btn_add_usv = QPushButton("+ Add Label")
        self.btn_add_usv.setStyleSheet("background-color: #6b4c9a;")
        self.btn_add_usv.setToolTip("Manually draw a new USV detection box on the spectrogram.")
        self.btn_add_usv.clicked.connect(self.add_new_usv)
        self.btn_add_usv.setEnabled(False)
        btn_row2.addWidget(self.btn_add_usv)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet("background-color: #5c5c5c;")
        self.btn_delete.setToolTip("Permanently delete the currently selected detection (D key).")
        self.btn_delete.clicked.connect(self.delete_current)
        self.btn_delete.setEnabled(False)
        btn_row2.addWidget(self.btn_delete)

        group_layout.addLayout(btn_row2)

        # Delete pending / Delete all labels
        delete_row = QHBoxLayout()
        delete_row.setSpacing(2)

        self.btn_delete_pending = QPushButton("Delete Pending")
        self.btn_delete_pending.setStyleSheet("background-color: #5c5c5c;")
        self.btn_delete_pending.setToolTip("Permanently delete all unreviewed detections.")
        self.btn_delete_pending.clicked.connect(self.delete_all_pending)
        self.btn_delete_pending.setEnabled(False)
        delete_row.addWidget(self.btn_delete_pending)

        self.btn_delete_all_labels = QPushButton("Delete All Labels")
        self.btn_delete_all_labels.setStyleSheet("background-color: #8b0000;")
        self.btn_delete_all_labels.setToolTip("Delete ALL detections for the current file.")
        self.btn_delete_all_labels.clicked.connect(self.delete_all_labels)
        self.btn_delete_all_labels.setEnabled(False)
        delete_row.addWidget(self.btn_delete_all_labels)

        group_layout.addLayout(delete_row)

        # Statistics label
        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("color: #888888; font-size: 9px;")
        self.lbl_stats.setWordWrap(True)
        group_layout.addWidget(self.lbl_stats)

        # Instructions
        lbl_hint = QLabel("Keys: A=Accept R=Reject X=Negative D=Delete Space=Play Ctrl+Z=Undo")
        lbl_hint.setStyleSheet("color: #666666; font-size: 9px; font-style: italic;")
        lbl_hint.setWordWrap(True)
        group_layout.addWidget(lbl_hint)

        group.setLayout(group_layout)
        layout.addWidget(group)

    # Training and inference UI live in separate pop-out dialogs invoked
    # from the Predict menu — see RunTrainingDialog / RunInferenceDialog
    # below, and ``_menu_run_training`` / ``_menu_run_inference``.

    # =========================================================================
    # Styles
    # =========================================================================

    def _create_arrow_images(self):
        """Create small arrow PNG images for spinbox/combobox buttons."""
        import tempfile
        self._arrow_dir = tempfile.mkdtemp(prefix='dad_arrows_')

        up_img = QImage(9, 6, QImage.Format_ARGB32)
        up_img.fill(QColor(0, 0, 0, 0))
        painter = QPainter(up_img)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#cccccc'))
        painter.drawPolygon(QPolygonF([
            QPointF(4.5, 0.5), QPointF(8.5, 5.5), QPointF(0.5, 5.5)
        ]))
        painter.end()
        self._up_arrow_path = os.path.join(self._arrow_dir, 'up.png')
        up_img.save(self._up_arrow_path)

        down_img = QImage(9, 6, QImage.Format_ARGB32)
        down_img.fill(QColor(0, 0, 0, 0))
        painter = QPainter(down_img)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#cccccc'))
        painter.drawPolygon(QPolygonF([
            QPointF(4.5, 5.5), QPointF(0.5, 0.5), QPointF(8.5, 0.5)
        ]))
        painter.end()
        self._down_arrow_path = os.path.join(self._arrow_dir, 'down.png')
        down_img.save(self._down_arrow_path)

    def _apply_styles(self):
        """Apply dark theme styles."""
        self._create_arrow_images()
        up = self._up_arrow_path.replace('\\', '/')
        down = self._down_arrow_path.replace('\\', '/')
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
                font-family: Arial;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
                min-height: 18px;
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
            QPushButton#small_btn {
                padding: 4px 8px;
                min-height: 16px;
            }
            QPushButton#accept_btn {
                background-color: #107c10;
            }
            QPushButton#accept_btn:hover {
                background-color: #0e6b0e;
            }
            QPushButton#reject_btn {
                background-color: #d13438;
            }
            QPushButton#reject_btn:hover {
                background-color: #b52e31;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 6px;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 4px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                border-left: 1px solid #555555;
                border-bottom: 1px solid #555555;
                border-top-right-radius: 3px;
                background-color: #404040;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #505050;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
                background-color: #606060;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: url(UP_ARROW_PATH);
                width: 9px;
                height: 6px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                border-left: 1px solid #555555;
                border-top: 1px solid #555555;
                border-bottom-right-radius: 3px;
                background-color: #404040;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #505050;
            }
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #606060;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: url(DOWN_ARROW_PATH);
                width: 9px;
                height: 6px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 18px;
                border-left: 1px solid #555555;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
                background-color: #404040;
            }
            QComboBox::drop-down:hover {
                background-color: #505050;
            }
            QComboBox::down-arrow {
                image: url(DOWN_ARROW_PATH);
                width: 9px;
                height: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
                border: 1px solid #3f3f3f;
            }
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QListWidget::item {
                padding: 2px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QProgressBar {
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                text-align: center;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 12px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background-color: #0078d4;
                border-radius: 2px;
                min-height: 20px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #106ebe;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background-color: #1e1e1e;
            }
            QScrollBar:horizontal {
                background-color: #1e1e1e;
                height: 14px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QScrollBar::handle:horizontal {
                background-color: #0078d4;
                border-radius: 2px;
                min-width: 20px;
                margin: 1px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #106ebe;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background-color: #1e1e1e;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
            /* In-window menu bar — blue strip so it reads as part of DAD
               and is visually distinct from the black content area. */
            QMenuBar {
                background-color: #0078d4;
                color: white;
                padding: 2px 6px;
                font-weight: bold;
                spacing: 4px;
            }
            QMenuBar::item {
                background: transparent;
                color: white;
                padding: 4px 10px;
                border-radius: 3px;
            }
            QMenuBar::item:selected {
                background-color: #106ebe;
            }
            QMenuBar::item:pressed {
                background-color: #005a9e;
            }
            QMenu {
                background-color: #2b2b2b;
                color: #cccccc;
                border: 1px solid #0078d4;
                padding: 4px;
            }
            QMenu::item {
                padding: 4px 18px;
                border-radius: 2px;
            }
            QMenu::item:selected {
                background-color: #0078d4;
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background: #3f3f3f;
                margin: 4px 6px;
            }
        """.replace('UP_ARROW_PATH', up).replace('DOWN_ARROW_PATH', down))

    # =========================================================================
    # Shortcuts
    # =========================================================================

    def _setup_shortcuts(self):
        """Set up global keyboard shortcuts using QShortcut."""
        sc_next = QShortcut(QKeySequence(Qt.Key_N), self)
        sc_next.setContext(Qt.ApplicationShortcut)
        sc_next.activated.connect(self._shortcut_next_detection)

        sc_prev = QShortcut(QKeySequence(Qt.Key_B), self)
        sc_prev.setContext(Qt.ApplicationShortcut)
        sc_prev.activated.connect(self._shortcut_prev_detection)

        sc_pan_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        sc_pan_right.setContext(Qt.ApplicationShortcut)
        sc_pan_right.activated.connect(self._shortcut_pan_right)

        sc_pan_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        sc_pan_left.setContext(Qt.ApplicationShortcut)
        sc_pan_left.activated.connect(self._shortcut_pan_left)

        sc_zoom_in = QShortcut(QKeySequence(Qt.Key_Up), self)
        sc_zoom_in.setContext(Qt.ApplicationShortcut)
        sc_zoom_in.activated.connect(self._shortcut_zoom_in)

        sc_zoom_out = QShortcut(QKeySequence(Qt.Key_Down), self)
        sc_zoom_out.setContext(Qt.ApplicationShortcut)
        sc_zoom_out.activated.connect(self._shortcut_zoom_out)

        sc_add_usv = QShortcut(QKeySequence(Qt.Key_P), self)
        sc_add_usv.setContext(Qt.ApplicationShortcut)
        sc_add_usv.activated.connect(self._shortcut_add_usv)

        sc_skip = QShortcut(QKeySequence(Qt.Key_S), self)
        sc_skip.setContext(Qt.ApplicationShortcut)
        sc_skip.activated.connect(self._shortcut_skip)

    def _shortcut_next_detection(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_next_det.isEnabled():
            self._flash_button(self.btn_next_det)
            self.next_detection()

    def _shortcut_prev_detection(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_prev_det.isEnabled():
            self._flash_button(self.btn_prev_det)
            self.prev_detection()

    def _shortcut_pan_right(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.pan_right()

    def _shortcut_pan_left(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.pan_left()

    def _shortcut_zoom_out(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.zoom_out()

    def _shortcut_zoom_in(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.zoom_in()

    def _shortcut_add_usv(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_add_usv.isEnabled():
            self._flash_button(self.btn_add_usv)
            self.add_new_usv()

    def _shortcut_skip(self):
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_skip.isEnabled():
            self._flash_button(self.btn_skip)
            self.skip_detection()

    # =========================================================================
    # Pan Timers
    # =========================================================================

    def _setup_pan_timers(self):
        """Set up press-and-hold repeat timers for pan buttons."""
        self._pan_left_timer = QTimer(self)
        self._pan_left_timer.setInterval(100)
        self._pan_left_timer.timeout.connect(self.pan_left)

        self._pan_right_timer = QTimer(self)
        self._pan_right_timer.setInterval(100)
        self._pan_right_timer.timeout.connect(self.pan_right)

        self.btn_pan_left.pressed.connect(self._on_pan_left_pressed)
        self.btn_pan_left.released.connect(self._pan_left_timer.stop)
        self.btn_pan_right.pressed.connect(self._on_pan_right_pressed)
        self.btn_pan_right.released.connect(self._pan_right_timer.stop)

        self.btn_pan_left.clicked.disconnect(self.pan_left)
        self.btn_pan_right.clicked.disconnect(self.pan_right)

    def _on_pan_left_pressed(self):
        self.pan_left()
        self._pan_left_timer.start()

    def _on_pan_right_pressed(self):
        self.pan_right()
        self._pan_right_timer.start()

    # =========================================================================
    # File Management
    # =========================================================================

    def _add_audio_files(self):
        """Add individual WAV files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "WAV Files (*.wav *.WAV);;All Files (*.*)"
        )
        if not files:
            return

        duplicates = [f for f in files if f in self.audio_files]
        to_add = [f for f in files if f not in self.audio_files]

        if duplicates and not to_add:
            QMessageBox.information(
                self, "Already Imported",
                f"All {len(duplicates)} file(s) are already imported.")
            return

        if duplicates:
            self.status_bar.showMessage(
                f"Added {len(to_add)} new file(s), skipped {len(duplicates)} already imported")
        else:
            self.status_bar.showMessage(f"Added {len(to_add)} file(s)")

        had_files = len(self.audio_files) > 0
        for f in to_add:
            self.audio_files.append(f)
            self._try_load_detections(f)

        if had_files:
            self._refresh_file_list_items_full()
        else:
            self._update_file_list()

        self._update_data_summary()
        self._update_project_state()

    def _update_file_list(self):
        """Update file list display."""
        self.file_list.blockSignals(True)
        self.file_list.clear()

        for filepath in self.audio_files:
            filename = os.path.basename(filepath)
            parts = [filename]
            if filepath in self.all_detections:
                n_det = len(self.all_detections[filepath])
                parts.append(f"({n_det})")
            item = QListWidgetItem(" ".join(parts))
            item.setData(Qt.UserRole, filepath)
            self.file_list.addItem(item)

        n = len(self.audio_files)
        self.lbl_file_num.setText(f"File {self.current_file_idx + 1 if n > 0 else 0}/{n}")
        self.btn_prev_file.setEnabled(self.current_file_idx > 0)
        self.btn_next_file.setEnabled(self.current_file_idx < n - 1)

        if n > 0 and self.current_file_idx < n:
            self.file_list.setCurrentRow(self.current_file_idx)
        self.file_list.blockSignals(False)

        if n > 0 and self.current_file_idx < n:
            self._load_current_file()

        self._update_ui_state()

    def _refresh_file_list_items_full(self):
        """Rebuild file list widget without reloading the current file."""
        self.file_list.blockSignals(True)
        self.file_list.clear()

        for filepath in self.audio_files:
            filename = os.path.basename(filepath)
            parts = [filename]
            if filepath in self.all_detections:
                n_det = len(self.all_detections[filepath])
                parts.append(f"({n_det})")
            item = QListWidgetItem(" ".join(parts))
            item.setData(Qt.UserRole, filepath)
            self.file_list.addItem(item)

        n = len(self.audio_files)
        self.lbl_file_num.setText(f"File {self.current_file_idx + 1 if n > 0 else 0}/{n}")
        self.btn_prev_file.setEnabled(self.current_file_idx > 0)
        self.btn_next_file.setEnabled(self.current_file_idx < n - 1)

        if n > 0 and self.current_file_idx < n:
            self.file_list.setCurrentRow(self.current_file_idx)
        self.file_list.blockSignals(False)

        self._update_file_navigation()

    def _refresh_file_list_items(self):
        """Refresh file list display text without changing selection."""
        current_row = self.file_list.currentRow()
        self.file_list.blockSignals(True)

        for i, filepath in enumerate(self.audio_files):
            item = self.file_list.item(i)
            if item is None:
                continue
            filename = os.path.basename(filepath)
            parts = [filename]
            if filepath in self.all_detections:
                n_det = len(self.all_detections[filepath])
                parts.append(f"({n_det})")
            item.setText(" ".join(parts))

        self.file_list.blockSignals(False)
        if current_row >= 0:
            self.file_list.setCurrentRow(current_row)

    def _on_file_selected(self, row):
        """Handle file selection from list."""
        if 0 <= row < len(self.audio_files):
            self._store_current_detections()
            self.current_file_idx = row
            self._load_current_file()
            self._update_file_navigation()

    def _prev_file(self):
        """Go to previous file."""
        if self.current_file_idx > 0:
            self.file_list.setCurrentRow(self.current_file_idx - 1)

    def _next_file(self):
        """Go to next file."""
        if self.current_file_idx < len(self.audio_files) - 1:
            self.file_list.setCurrentRow(self.current_file_idx + 1)

    def _update_file_navigation(self):
        """Update file navigation buttons."""
        n = len(self.audio_files)
        self.lbl_file_num.setText(f"File {self.current_file_idx + 1}/{n}")
        self.btn_prev_file.setEnabled(self.current_file_idx > 0)
        self.btn_next_file.setEnabled(self.current_file_idx < n - 1)

    def _update_data_summary(self):
        """Update the data summary label."""
        n = len(self.audio_files)
        if n == 0:
            self.lbl_data_summary.setText("No files loaded")
        else:
            n_with_det = sum(1 for f in self.audio_files if f in self.all_detections)
            self.lbl_data_summary.setText(f"{n} files loaded, {n_with_det} with detections")

    def _load_current_file(self):
        """Load the current audio file."""
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            return

        filepath = self.audio_files[self.current_file_idx]
        self.status_bar.showMessage(f"Loading {os.path.basename(filepath)}...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            if HAS_SOUNDFILE:
                try:
                    self.audio_data, self.sample_rate = sf.read(filepath, dtype='float32')
                except Exception:
                    self.audio_data, self.sample_rate = self._load_with_ffmpeg(filepath)
            else:
                self.audio_data, self.sample_rate = self._load_with_ffmpeg(filepath)

            if self.audio_data.ndim > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)

            has_previous = self.spectrogram.total_duration > 0
            self.spectrogram.set_audio_data(self.audio_data, self.sample_rate,
                                            preserve_view=has_previous)
            self.waveform_overview.set_audio_data(self.audio_data, self.sample_rate)

            self.spin_display_min_freq.blockSignals(True)
            self.spin_display_max_freq.blockSignals(True)
            self.spin_display_min_freq.setValue(self.spectrogram.min_freq)
            self.spin_display_max_freq.setValue(self.spectrogram.max_freq)
            self.spin_display_min_freq.blockSignals(False)
            self.spin_display_max_freq.blockSignals(False)

            if has_previous:
                view_start, view_end = self.spectrogram.get_view_range()
                self.spin_view_window.blockSignals(True)
                self.spin_view_window.setValue(view_end - view_start)
                self.spin_view_window.blockSignals(False)

            if filepath in self.all_detections:
                self.detections_df = self.all_detections[filepath].copy()
                self._ensure_freq_bounds(self.detections_df)
                self.current_detection_idx = 0
            else:
                self._try_load_detections(filepath)

            self.undo_stack.clear()
            self.btn_undo.setEnabled(False)

            self._update_display()
            self.status_bar.showMessage(f"Loaded: {os.path.basename(filepath)}")

        except Exception as e:
            self.status_bar.showMessage(f"Error loading file: {e}")
            self.audio_data = None
        finally:
            QApplication.restoreOverrideCursor()

        self._update_ui_state()

    def _load_with_ffmpeg(self, filepath):
        """Load audio using ffmpeg."""
        import subprocess
        import tempfile

        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1', filepath
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        sr = int(result.stdout.strip())

        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
            temp_path = tmp.name

        try:
            convert_cmd = [
                'ffmpeg', '-i', filepath, '-f', 'f32le', '-acodec', 'pcm_f32le',
                '-ac', '1', '-y', temp_path
            ]
            subprocess.run(convert_cmd, capture_output=True, check=True)
            audio = np.fromfile(temp_path, dtype=np.float32)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return audio, sr

    def _try_load_detections(self, filepath):
        """Load existing sibling-CSV labels for a file.

        Reads the canonical ``<stem>_FNT_DAD_detections.csv`` first, then legacy
        suffixes (``_dad``, ``_yoloDetection``, ``_usv_yolo``). CAD-family
        suffixes (``_FNT_CAD_detections``, ``_cad``, ``_usv_dsp``,
        ``_usv_detections``) are deliberately NOT auto-imported — DAD is a
        standalone tool; users pull CAD labels in via ``Labels → Import…``.
        """
        df = load_labels(filepath)
        is_current = (
            self.audio_files
            and self.current_file_idx < len(self.audio_files)
            and filepath == self.audio_files[self.current_file_idx]
        )

        if df is None or len(df) == 0:
            if is_current:
                self.detections_df = None
                self.current_detection_idx = 0
            return

        if 'status' not in df.columns:
            df['status'] = 'pending'
        self._ensure_freq_bounds(df)
        self.all_detections[filepath] = df
        found = find_existing_sibling_csv(filepath)
        if found is not None:
            # 'dad', 'yoloDetection', etc. — used only for log/UI purposes now.
            self.detection_sources[filepath] = found.stem.rsplit("_", 1)[-1]
        if is_current:
            self.detections_df = df.copy()
            self.current_detection_idx = 0

    @staticmethod
    def _ensure_freq_bounds(df):
        """Ensure min_freq_hz and max_freq_hz columns exist."""
        if 'min_freq_hz' not in df.columns or df['min_freq_hz'].isna().all():
            if 'mean_freq_hz' in df.columns and 'freq_bandwidth_hz' in df.columns:
                df['min_freq_hz'] = df['mean_freq_hz'] - df['freq_bandwidth_hz'] / 2
                df['min_freq_hz'] = df['min_freq_hz'].clip(lower=0)
            else:
                df['min_freq_hz'] = 25000
        if 'max_freq_hz' not in df.columns or df['max_freq_hz'].isna().all():
            if 'mean_freq_hz' in df.columns and 'freq_bandwidth_hz' in df.columns:
                df['max_freq_hz'] = df['mean_freq_hz'] + df['freq_bandwidth_hz'] / 2
            else:
                df['max_freq_hz'] = 65000
        df['min_freq_hz'] = df['min_freq_hz'].fillna(25000)
        df['max_freq_hz'] = df['max_freq_hz'].fillna(65000)

    # =========================================================================
    # Detection Navigation and Display
    # =========================================================================

    def _update_display(self):
        """Update all display elements."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.lbl_det_num.setText("Det 0/0")
            self.lbl_det_info.setText("Peak: -- Hz | Dur: -- ms")
            self.lbl_det_status.setText("Status: --")
            self.spectrogram.set_detections([], -1)
            self.waveform_overview.set_detections([])
            self._update_progress()
            return

        n_det = len(self.detections_df)
        self.current_detection_idx = min(self.current_detection_idx, n_det - 1)
        self.current_detection_idx = max(0, self.current_detection_idx)

        det = self.detections_df.iloc[self.current_detection_idx]

        self.lbl_det_num.setText(f"Det {self.current_detection_idx + 1}/{n_det}")
        self.btn_prev_det.setEnabled(self.current_detection_idx > 0)
        self.btn_next_det.setEnabled(self.current_detection_idx < n_det - 1)

        self.spin_start.blockSignals(True)
        self.spin_stop.blockSignals(True)
        self.spin_start.setValue(det['start_seconds'])
        self.spin_stop.setValue(det['stop_seconds'])
        self.spin_start.blockSignals(False)
        self.spin_stop.blockSignals(False)

        self.spin_det_min_freq.blockSignals(True)
        self.spin_det_max_freq.blockSignals(True)
        min_f = det.get('min_freq_hz', 20000)
        max_f = det.get('max_freq_hz', 80000)
        min_f = 20000 if pd.isna(min_f) else int(min_f)
        max_f = 80000 if pd.isna(max_f) else int(max_f)
        self.spin_det_min_freq.setValue(min_f)
        self.spin_det_max_freq.setValue(max_f)
        self.spin_det_min_freq.blockSignals(False)
        self.spin_det_max_freq.blockSignals(False)

        peak = det.get('peak_freq_hz', 0)
        dur = det.get('duration_ms', 0)
        peak = 0 if pd.isna(peak) else peak
        dur = 0 if pd.isna(dur) else dur
        self.lbl_det_info.setText(f"Peak: {peak:.0f} Hz | Dur: {dur:.1f} ms")

        status = det.get('status', 'pending')
        self.lbl_det_status.setText(f"Status: {status.capitalize()}")
        if status == 'accepted':
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #107c10;")
        elif status == 'rejected':
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #d13438;")
        elif status == 'negative':
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #8b4513;")
        else:
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #999999;")

        self.spin_jump.blockSignals(True)
        self.spin_jump.setRange(1, max(1, n_det))
        self.spin_jump.setValue(self.current_detection_idx + 1)
        self.spin_jump.blockSignals(False)

        det_times = list(zip(
            self.detections_df['start_seconds'].tolist(),
            self.detections_df['stop_seconds'].tolist()
        ))
        self.waveform_overview.set_detections(det_times)

        self._update_spectrogram_view()
        self._update_progress()
        self._update_button_counts()
        self._update_statistics()

    def _update_spectrogram_view(self):
        """Update spectrogram to show current detection."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        det = self.detections_df.iloc[self.current_detection_idx]
        det_center = (det['start_seconds'] + det['stop_seconds']) / 2

        window = self.spin_view_window.value()
        view_start = det_center - window / 2
        view_end = det_center + window / 2

        total_dur = self.spectrogram.get_total_duration()
        if total_dur > 0:
            view_start = max(0, view_start)
            view_end = min(total_dur, view_end)

        self.spectrogram.set_view_range(view_start, view_end)
        self._update_scrollbar()
        self.waveform_overview.set_view_range(view_start, view_end)

        self._update_detection_boxes()

    def _update_scrollbar(self):
        """Update scrollbar position."""
        total_dur = self.spectrogram.get_total_duration()
        if total_dur <= 0:
            return

        view_start, view_end = self.spectrogram.get_view_range()
        view_center = (view_start + view_end) / 2

        pos = int(view_center / total_dur * 1000)
        self.time_scrollbar.blockSignals(True)
        self.time_scrollbar.setValue(pos)
        self.time_scrollbar.blockSignals(False)

    def _update_progress(self):
        """Update progress display."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.lbl_progress.setText("0/0 reviewed")
            self.progress_bar.setValue(0)
            return

        total = len(self.detections_df)
        reviewed = len(self.detections_df[self.detections_df['status'] != 'pending'])
        self.lbl_progress.setText(f"{reviewed}/{total} reviewed")
        self.progress_bar.setValue(int(reviewed / total * 100) if total > 0 else 0)

    def _update_detection_boxes(self):
        """Update detection boxes on spectrogram."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.spectrogram.set_detections([], -1)
            return

        detections = []
        for _, row in self.detections_df.iterrows():
            min_freq = row['min_freq_hz'] if 'min_freq_hz' in row and not pd.isna(row['min_freq_hz']) else 20000
            max_freq = row['max_freq_hz'] if 'max_freq_hz' in row and not pd.isna(row['max_freq_hz']) else 80000
            status = row['status'] if 'status' in row and not pd.isna(row['status']) else 'pending'

            det = {
                'start_seconds': row['start_seconds'],
                'stop_seconds': row['stop_seconds'],
                'min_freq_hz': min_freq,
                'max_freq_hz': max_freq,
                'status': status,
            }
            # Pass through peak_freq_N contour columns
            for col in row.index:
                if col.startswith('peak_freq_') and col != 'peak_freq_hz':
                    val = row[col]
                    if not pd.isna(val):
                        det[col] = val
            detections.append(det)

        self.spectrogram.set_detections(detections, self.current_detection_idx)

    def _update_detection_info_only(self):
        """Update detection info without changing view."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        det = self.detections_df.iloc[self.current_detection_idx]

        self.spin_start.blockSignals(True)
        self.spin_stop.blockSignals(True)
        self.spin_det_min_freq.blockSignals(True)
        self.spin_det_max_freq.blockSignals(True)

        self.spin_start.setValue(det['start_seconds'])
        self.spin_stop.setValue(det['stop_seconds'])

        min_f = det.get('min_freq_hz', 20000)
        max_f = det.get('max_freq_hz', 80000)
        min_f = 20000 if pd.isna(min_f) else int(min_f)
        max_f = 80000 if pd.isna(max_f) else int(max_f)
        self.spin_det_min_freq.setValue(min_f)
        self.spin_det_max_freq.setValue(max_f)

        self.spin_start.blockSignals(False)
        self.spin_stop.blockSignals(False)
        self.spin_det_min_freq.blockSignals(False)
        self.spin_det_max_freq.blockSignals(False)

        peak = det.get('peak_freq_hz', 0)
        dur = det.get('duration_ms', 0)
        peak = 0 if pd.isna(peak) else peak
        dur = 0 if pd.isna(dur) else dur
        self.lbl_det_info.setText(f"Peak: {peak:.0f} Hz | Dur: {dur:.1f} ms")

    def prev_detection(self):
        """Go to previous detection by time (respects filter)."""
        if self.detections_df is None:
            return
        filtered = self._get_filtered_indices()
        if not filtered:
            return
        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']
        earlier = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                   if self.detections_df.iloc[i]['start_seconds'] < current_time]
        if not earlier:
            earlier = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                       if i != self.current_detection_idx]
            if not earlier:
                return
            earlier.sort(key=lambda x: x[1])
            self.current_detection_idx = earlier[-1][0]
        else:
            earlier.sort(key=lambda x: x[1])
            self.current_detection_idx = earlier[-1][0]
        self._update_display()

    def next_detection(self):
        """Go to next detection by time (respects filter)."""
        if self.detections_df is None:
            return
        filtered = self._get_filtered_indices()
        if not filtered:
            return
        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']
        later = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                 if self.detections_df.iloc[i]['start_seconds'] > current_time]
        if not later:
            later = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                     if i != self.current_detection_idx]
            if not later:
                return
            later.sort(key=lambda x: x[1])
            self.current_detection_idx = later[0][0]
        else:
            later.sort(key=lambda x: x[1])
            self.current_detection_idx = later[0][0]
        self._update_display()

    def on_detection_selected(self, idx):
        """Handle detection selection from spectrogram."""
        if self.detections_df is not None and 0 <= idx < len(self.detections_df):
            self.current_detection_idx = idx
            self._update_display()

    def on_box_adjusted(self, idx, start_s, stop_s, min_freq, max_freq):
        """Handle box adjustment from spectrogram."""
        if self.detections_df is None or idx < 0 or idx >= len(self.detections_df):
            return

        self.detections_df.at[idx, 'start_seconds'] = start_s
        self.detections_df.at[idx, 'stop_seconds'] = stop_s
        self.detections_df.at[idx, 'min_freq_hz'] = min_freq
        self.detections_df.at[idx, 'max_freq_hz'] = max_freq
        self.detections_df.at[idx, 'duration_ms'] = (stop_s - start_s) * 1000

        if self.audio_files and self.current_file_idx < len(self.audio_files):
            filepath = self.audio_files[self.current_file_idx]
            self.all_detections[filepath] = self.detections_df.copy()

        self.spectrogram.update_detection(idx, start_s, stop_s, min_freq, max_freq)

        if idx == self.current_detection_idx:
            self._update_detection_info_only()

    def on_drag_complete(self):
        """Handle completion of box drag - save to CSV."""
        self._store_current_detections()
        self._update_button_counts()
        self._refresh_file_list_items()
        self._update_detection_boxes()

    # =========================================================================
    # View Controls
    # =========================================================================

    def on_scrollbar_changed(self, value):
        """Handle scrollbar change."""
        total_dur = self.spectrogram.get_total_duration()
        if total_dur <= 0:
            return

        center = value / 1000.0 * total_dur
        window = self.spin_view_window.value()

        view_start = max(0, center - window / 2)
        view_end = min(total_dur, center + window / 2)

        self.spectrogram.set_view_range(view_start, view_end)
        self._update_detection_boxes()
        self.waveform_overview.set_view_range(view_start, view_end)

    def pan_left(self):
        """Pan view left."""
        view_start, view_end = self.spectrogram.get_view_range()
        window = view_end - view_start
        shift = window * 0.25

        new_start = max(0, view_start - shift)
        new_end = new_start + window

        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()
        self._update_detection_boxes()
        self.waveform_overview.set_view_range(new_start, new_end)

    def pan_right(self):
        """Pan view right."""
        view_start, view_end = self.spectrogram.get_view_range()
        window = view_end - view_start
        shift = window * 0.25
        total = self.spectrogram.get_total_duration()

        new_end = min(total, view_end + shift)
        new_start = new_end - window

        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()
        self._update_detection_boxes()
        self.waveform_overview.set_view_range(new_start, new_end)

    def zoom_in(self):
        """Zoom in maintaining center."""
        current_start, current_end = self.spectrogram.get_view_range()
        center = (current_start + current_end) / 2
        new_window = max(0.1, (current_end - current_start) / 1.5)

        self.spin_view_window.blockSignals(True)
        self.spin_view_window.setValue(new_window)
        self.spin_view_window.blockSignals(False)

        total_dur = self.spectrogram.get_total_duration()
        new_start = max(0, center - new_window / 2)
        new_end = min(total_dur, center + new_window / 2)
        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()
        self._update_detection_boxes()
        self.waveform_overview.set_view_range(new_start, new_end)

    def zoom_out(self):
        """Zoom out maintaining center."""
        current_start, current_end = self.spectrogram.get_view_range()
        center = (current_start + current_end) / 2
        total_dur = self.spectrogram.get_total_duration()
        new_window = min(total_dur if total_dur > 0 else 600, (current_end - current_start) * 1.5)

        self.spin_view_window.blockSignals(True)
        self.spin_view_window.setValue(new_window)
        self.spin_view_window.blockSignals(False)

        new_start = max(0, center - new_window / 2)
        new_end = min(total_dur, center + new_window / 2)
        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()
        self._update_detection_boxes()
        self.waveform_overview.set_view_range(new_start, new_end)

    def on_view_window_changed(self):
        """Handle view window spinbox change."""
        self._update_spectrogram_view()

    def on_time_changed(self):
        """Handle time spinbox change."""
        if self.detections_df is None:
            return

        start = self.spin_start.value()
        stop = self.spin_stop.value()

        self.detections_df.at[self.current_detection_idx, 'start_seconds'] = start
        self.detections_df.at[self.current_detection_idx, 'stop_seconds'] = stop
        self.detections_df.at[self.current_detection_idx, 'duration_ms'] = (stop - start) * 1000

        self._update_detection_boxes()

    def on_freq_changed(self):
        """Handle frequency spinbox change."""
        if self.detections_df is None:
            return

        min_f = self.spin_det_min_freq.value()
        max_f = self.spin_det_max_freq.value()

        self.detections_df.at[self.current_detection_idx, 'min_freq_hz'] = min_f
        self.detections_df.at[self.current_detection_idx, 'max_freq_hz'] = max_f

        self._update_detection_boxes()

    def on_wheel_zoom(self, factor, center_time):
        """Handle mouse wheel zoom on spectrogram."""
        current_start, current_end = self.spectrogram.get_view_range()
        current_window = current_end - current_start
        new_window = current_window / factor
        total_dur = self.spectrogram.get_total_duration()
        new_window = max(0.05, min(total_dur if total_dur > 0 else 600, new_window))

        ratio = (center_time - current_start) / current_window if current_window > 0 else 0.5
        new_start = center_time - ratio * new_window
        new_end = new_start + new_window

        new_start = max(0, new_start)
        new_end = min(total_dur, new_end)

        self.spin_view_window.blockSignals(True)
        self.spin_view_window.setValue(new_end - new_start)
        self.spin_view_window.blockSignals(False)

        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()
        self._update_detection_boxes()
        self.waveform_overview.set_view_range(new_start, new_end)

    def on_overview_clicked(self, center_time):
        """Handle click on waveform overview to navigate."""
        window = self.spin_view_window.value()
        total_dur = self.spectrogram.get_total_duration()

        view_start = max(0, center_time - window / 2)
        view_end = min(total_dur, center_time + window / 2)

        self.spectrogram.set_view_range(view_start, view_end)
        self._update_scrollbar()
        self._update_detection_boxes()
        self.waveform_overview.set_view_range(view_start, view_end)

    def on_display_freq_changed(self):
        """Handle display frequency range change."""
        min_f = self.spin_display_min_freq.value()
        max_f = self.spin_display_max_freq.value()
        if max_f <= min_f:
            return
        self.spectrogram.min_freq = min_f
        self.spectrogram.max_freq = max_f
        self.spectrogram.cached_view_start = None
        self.spectrogram.cached_view_end = None
        self.spectrogram.spec_image = None
        if self.spectrogram.total_duration > 0:
            self.spectrogram._compute_view_spectrogram()
        self.spectrogram.update()

    def on_colormap_changed(self, name):
        """Handle colormap selection change."""
        self.spectrogram.set_colormap(name)

    def on_filter_changed(self):
        """Handle filter combo change."""
        self.filter_status = self.combo_filter.currentData()
        self._update_display()

    def on_jump_to_detection(self):
        """Jump to a specific detection number."""
        idx = self.spin_jump.value() - 1
        if self.detections_df is not None and 0 <= idx < len(self.detections_df):
            self.current_detection_idx = idx
            self._update_display()

    def _get_filtered_indices(self):
        """Get detection indices matching current filter."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return []
        if self.filter_status == 'all':
            return list(range(len(self.detections_df)))
        return list(self.detections_df.index[self.detections_df['status'] == self.filter_status])

    # =========================================================================
    # Labeling Actions
    # =========================================================================

    def accept_detection(self):
        """Mark current detection as accepted (USV)."""
        if self.detections_df is None:
            return
        old_status = self.detections_df.at[self.current_detection_idx, 'status']
        self.undo_stack.append(('label', self.current_detection_idx, old_status))
        self.btn_undo.setEnabled(True)
        self.detections_df.at[self.current_detection_idx, 'status'] = 'accepted'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self._update_project_state()
        self._auto_advance()

    def reject_detection(self):
        """Mark current detection as rejected."""
        if self.detections_df is None:
            return
        old_status = self.detections_df.at[self.current_detection_idx, 'status']
        self.undo_stack.append(('label', self.current_detection_idx, old_status))
        self.btn_undo.setEnabled(True)
        self.detections_df.at[self.current_detection_idx, 'status'] = 'rejected'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self._update_project_state()
        self._auto_advance()

    def mark_negative(self):
        """Mark current detection as negative (background) training data."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return
        if self.current_detection_idx >= len(self.detections_df):
            return
        old_status = self.detections_df.at[self.current_detection_idx, 'status']
        self.undo_stack.append(('label', self.current_detection_idx, old_status))
        self.btn_undo.setEnabled(True)
        self.detections_df.at[self.current_detection_idx, 'status'] = 'negative'
        nyquist = self.sample_rate / 2 if self.sample_rate else 125000
        self.detections_df.at[self.current_detection_idx, 'min_freq_hz'] = 0
        self.detections_df.at[self.current_detection_idx, 'max_freq_hz'] = nyquist
        self.detections_df.at[self.current_detection_idx, 'freq_bandwidth_hz'] = nyquist
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self._update_project_state()
        self._auto_advance()

    def skip_detection(self):
        """Skip to next detection by time without changing its status."""
        if self.detections_df is None:
            return
        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']
        all_dets = [(i, self.detections_df.iloc[i]['start_seconds'])
                    for i in range(len(self.detections_df))]
        all_dets.sort(key=lambda x: x[1])
        for idx, t in all_dets:
            if t > current_time:
                self.current_detection_idx = idx
                self._update_display()
                return
        n_pending = (self.detections_df['status'] == 'pending').sum()
        if n_pending == 0:
            self._update_display()
            self._prompt_next_file()
            return
        self.current_detection_idx = all_dets[0][0]
        self._update_display()

    def _update_button_counts(self):
        """Update Accept/Reject button labels with counts."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.btn_accept.setText("Accept")
            self.btn_reject.setText("Reject")
            return

        counts = self.detections_df['status'].value_counts()
        n_accepted = counts.get('accepted', 0)
        n_rejected = counts.get('rejected', 0)

        self.btn_accept.setText(f"Accept ({n_accepted})")
        self.btn_reject.setText(f"Reject ({n_rejected})")

    def add_new_usv(self):
        """Add a new USV detection at view center."""
        if self.audio_data is None:
            return

        view_start, view_end = self.spectrogram.get_view_range()
        view_center = (view_start + view_end) / 2
        view_duration = view_end - view_start

        new_duration = min(0.03, view_duration * 0.05)
        new_start = view_center - new_duration / 2
        new_stop = view_center + new_duration / 2

        freq_center = 40000
        freq_half = 4000
        new_row = {
            'start_seconds': new_start,
            'stop_seconds': new_stop,
            'duration_ms': new_duration * 1000,
            'min_freq_hz': freq_center - freq_half,
            'max_freq_hz': freq_center + freq_half,
            'peak_freq_hz': freq_center,
            'status': 'pending',
            'source': 'manual'
        }

        if self.detections_df is None:
            self.detections_df = pd.DataFrame([new_row])
        else:
            self.detections_df = pd.concat([self.detections_df, pd.DataFrame([new_row])],
                                           ignore_index=True)

        self.current_detection_idx = len(self.detections_df) - 1
        self._store_current_detections()
        self._update_display()
        self._update_ui_state()
        self.status_bar.showMessage("Added new USV detection")

    def delete_current(self):
        """Delete current detection."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        self.detections_df = self.detections_df.drop(self.current_detection_idx).reset_index(drop=True)

        if len(self.detections_df) == 0:
            self.detections_df = None
            self.current_detection_idx = 0
        elif self.current_detection_idx >= len(self.detections_df):
            self.current_detection_idx = len(self.detections_df) - 1

        self._store_current_detections()
        self._update_display()
        self._update_ui_state()
        self._update_project_state()
        self.status_bar.showMessage("Deleted detection")

    def delete_all_pending(self):
        """Delete all pending detections."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        pending_mask = self.detections_df['status'] == 'pending'
        n_pending = pending_mask.sum()

        if n_pending == 0:
            self.status_bar.showMessage("No pending detections to delete")
            return

        reply = QMessageBox.question(
            self, "Delete Pending",
            f"Delete {n_pending} pending detections?\n\nThis keeps only labeled detections.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.detections_df = self.detections_df[~pending_mask].reset_index(drop=True)

        if len(self.detections_df) == 0:
            self.detections_df = None
            self.current_detection_idx = 0
        else:
            self.current_detection_idx = min(self.current_detection_idx, len(self.detections_df) - 1)

        self._store_current_detections()
        self._update_display()
        self._update_ui_state()
        self._update_project_state()

        n_remaining = len(self.detections_df) if self.detections_df is not None else 0
        self.status_bar.showMessage(f"Deleted {n_pending} pending | {n_remaining} labeled remain")

    def delete_all_labels(self):
        """Delete ALL detections for the current file."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        n_total = len(self.detections_df)
        reply = QMessageBox.warning(
            self, "Warning",
            f"This action will delete all {n_total} pending and approved labels "
            "in this file.\n\nAre you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        if self.audio_files and self.current_file_idx < len(self.audio_files):
            filepath = self.audio_files[self.current_file_idx]
            # save_labels with empty df removes the canonical sibling CSV.
            try:
                save_labels(filepath, None)
            except Exception:
                pass
            # Also clean up the legacy _yoloDetection.csv so it doesn't resurface.
            base = Path(filepath).stem
            parent = Path(filepath).parent
            for legacy in ('_dad', '_yoloDetection', '_usv_yolo'):
                p = parent / f"{base}{legacy}.csv"
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            self.detection_sources.pop(filepath, None)
            self.all_detections.pop(filepath, None)

        self.detections_df = None
        self.current_detection_idx = 0
        self._store_current_detections()
        self._update_display()
        self._update_ui_state()
        self._update_project_state()
        self.status_bar.showMessage(f"Deleted all {n_total} detections")

    def _auto_advance(self):
        """Auto-advance to next pending detection by time."""
        if self.detections_df is None:
            return

        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']

        pending = []
        for i in range(len(self.detections_df)):
            if self.detections_df.iloc[i]['status'] == 'pending':
                pending.append((i, self.detections_df.iloc[i]['start_seconds']))

        if not pending:
            self._update_display()
            self._prompt_next_file()
            return

        pending.sort(key=lambda x: x[1])

        for idx, t in pending:
            if t > current_time:
                self.current_detection_idx = idx
                self._update_display()
                return

        self.current_detection_idx = pending[0][0]
        self._update_display()

    def _prompt_next_file(self):
        """Show dialog when all detections are curated."""
        box = QMessageBox(self)
        box.setWindowTitle("File Complete")
        box.setText("All detections in this file have been curated.\n\nMove to the next file?")
        btn_yes = box.addButton("Yes (Press Enter)", QMessageBox.AcceptRole)
        box.addButton("No", QMessageBox.RejectRole)
        box.setDefaultButton(btn_yes)
        box.exec_()
        if box.clickedButton() == btn_yes:
            self._advance_to_next_file_with_detections()

    def _advance_to_next_file_with_detections(self):
        """Auto-switch to the next file that has detections."""
        if not self.audio_files:
            return

        self._store_current_detections()

        for offset in range(1, len(self.audio_files)):
            next_idx = (self.current_file_idx + offset) % len(self.audio_files)
            filepath = self.audio_files[next_idx]

            has_dets = filepath in self.all_detections and len(self.all_detections[filepath]) > 0
            if not has_dets:
                if find_existing_sibling_csv(filepath) is not None:
                    has_dets = True

            if has_dets:
                self.file_list.setCurrentRow(next_idx)
                if self.detections_df is not None and len(self.detections_df) > 0:
                    pending = [(i, self.detections_df.iloc[i]['start_seconds'])
                               for i in range(len(self.detections_df))
                               if self.detections_df.iloc[i]['status'] == 'pending']
                    if pending:
                        pending.sort(key=lambda x: x[1])
                        self.current_detection_idx = pending[0][0]
                    else:
                        time_sorted = self.detections_df['start_seconds'].argsort()
                        self.current_detection_idx = time_sorted.iloc[0]
                    self._update_display()
                self.status_bar.showMessage(f"Advanced to {os.path.basename(filepath)}")
                return

        self.status_bar.showMessage("No more files with detections")

    def _store_current_detections(self):
        """Store current detections in all_detections dict and save to CSV."""
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            filepath = self.audio_files[self.current_file_idx]
            if self.detections_df is not None:
                self.all_detections[filepath] = self.detections_df.copy()
                self._save_detections_csv(filepath)
            elif filepath in self.all_detections:
                del self.all_detections[filepath]

    def _save_detections_csv(self, filepath):
        """Save current detections to the canonical sibling CSV
        (``<stem>_FNT_DAD_detections.csv``).

        If the in-memory df is empty/None, the canonical file is removed so the
        filesystem stays in sync. Legacy ``_dad.csv`` / ``_yoloDetection.csv``
        files are left untouched — upgrading is a one-way migration the next
        write performs.
        """
        try:
            written = save_labels(filepath, self.detections_df)
            if written is not None:
                self.detection_sources[filepath] = DAD_SUFFIX.lstrip('_')
        except Exception as e:
            self.status_bar.showMessage(f"Error saving CSV: {e}")

    # =========================================================================
    # Batch Labeling
    # =========================================================================

    def accept_all_pending(self):
        """Accept all pending detections."""
        if self.detections_df is None:
            return
        mask = self.detections_df['status'] == 'pending'
        n = mask.sum()
        if n == 0:
            self.status_bar.showMessage("No pending detections")
            return
        reply = QMessageBox.question(
            self, "Accept All Pending",
            f"Accept all {n} pending detections?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.undo_stack.append(('batch_label', list(self.detections_df[mask].index), 'pending'))
        self.btn_undo.setEnabled(True)
        self.detections_df.loc[mask, 'status'] = 'accepted'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self._update_project_state()
        self.status_bar.showMessage(f"Accepted {n} detections")

    def reject_all_pending(self):
        """Reject all pending detections."""
        if self.detections_df is None:
            return
        mask = self.detections_df['status'] == 'pending'
        n = mask.sum()
        if n == 0:
            self.status_bar.showMessage("No pending detections")
            return
        reply = QMessageBox.question(
            self, "Reject All Pending",
            f"Reject all {n} pending detections?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.undo_stack.append(('batch_label', list(self.detections_df[mask].index), 'pending'))
        self.btn_undo.setEnabled(True)
        self.detections_df.loc[mask, 'status'] = 'rejected'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self._update_project_state()
        self.status_bar.showMessage(f"Rejected {n} detections")

    # =========================================================================
    # Undo
    # =========================================================================

    def undo_action(self):
        """Undo the last labeling action."""
        if not self.undo_stack or self.detections_df is None:
            return

        action = self.undo_stack.pop()
        if action[0] == 'label':
            _, idx, old_status = action
            if 0 <= idx < len(self.detections_df):
                self.detections_df.at[idx, 'status'] = old_status
                self.current_detection_idx = idx
                self._store_current_detections()
                self._update_display()
                self._update_button_counts()
                self._update_project_state()
                self.status_bar.showMessage(f"Undid label change on detection {idx + 1}")
        elif action[0] == 'batch_label':
            _, indices, old_status = action
            for idx in indices:
                if 0 <= idx < len(self.detections_df):
                    self.detections_df.at[idx, 'status'] = old_status
            self._store_current_detections()
            self._update_display()
            self._update_button_counts()
            self._update_project_state()
            self.status_bar.showMessage(f"Undid batch label change on {len(indices)} detections")

        self.btn_undo.setEnabled(len(self.undo_stack) > 0)

    # =========================================================================
    # Playback
    # =========================================================================

    def on_speed_changed(self):
        """Handle playback speed slider change."""
        idx = self.slider_speed.value()
        speed = self._speed_values[idx]
        self.use_heterodyne = False
        self.playback_speed = speed
        self.lbl_speed.setText(f"{speed}x")

    def toggle_playback(self):
        """Toggle playback."""
        if self.is_playing:
            self.stop_playback()
        else:
            self.play_visible()

    def play_visible(self):
        """Play visible spectrogram region."""
        if not HAS_SOUNDDEVICE or self.audio_data is None:
            return

        start_s, stop_s = self.spectrogram.get_view_range()
        total_duration = len(self.audio_data) / self.sample_rate
        start_s = max(0, start_s)
        stop_s = min(total_duration, stop_s)

        start_sample = int(start_s * self.sample_rate)
        stop_sample = int(stop_s * self.sample_rate)
        segment = self.audio_data[start_sample:stop_sample].copy()

        try:
            if self.use_heterodyne:
                segment = self._heterodyne(segment)
                play_sr = 44100
            else:
                output_sr = 44100
                original_duration = len(segment) / self.sample_rate
                output_duration = original_duration / self.playback_speed
                n_output_samples = int(output_duration * output_sr)
                if n_output_samples < 100:
                    return
                segment = signal.resample(segment, n_output_samples).astype(np.float32)
                play_sr = output_sr

            sd.play(segment, play_sr)
            self.is_playing = True
            self.btn_play.setText("Playing...")

            import time as _time
            self._playback_latency = 0.15
            self._playback_start_time = _time.time()
            self._playback_start_s = start_s
            self._playback_end_s = stop_s
            self._playback_timer.start()

            self._set_playback_controls_enabled(False)

        except Exception as e:
            self.status_bar.showMessage(f"Playback error: {e}")

    def stop_playback(self):
        """Stop playback."""
        if HAS_SOUNDDEVICE:
            sd.stop()
        self.is_playing = False
        self.btn_play.setText("Play")
        self._playback_timer.stop()
        self.spectrogram.playback_position = None
        self.spectrogram.update()
        self._set_playback_controls_enabled(True)

    def _update_playback_position(self):
        """Timer callback to update the playback position line."""
        import time as _time
        if not self.is_playing or self._playback_start_time is None:
            self.stop_playback()
            return

        elapsed = _time.time() - self._playback_start_time - self._playback_latency
        elapsed = max(0.0, elapsed)
        audio_elapsed = elapsed * self.playback_speed
        current_pos = self._playback_start_s + audio_elapsed

        if current_pos >= self._playback_end_s:
            self.stop_playback()
            return

        self.spectrogram.playback_position = current_pos
        self.spectrogram.update()

    def _set_playback_controls_enabled(self, enabled):
        """Enable or disable controls during playback."""
        self.btn_pan_left.setEnabled(enabled)
        self.btn_pan_right.setEnabled(enabled)
        self.btn_zoom_in.setEnabled(enabled)
        self.btn_zoom_out.setEnabled(enabled)
        self.spin_view_window.setEnabled(enabled)
        self.time_scrollbar.setEnabled(enabled)
        self.btn_prev_det.setEnabled(enabled)
        self.btn_next_det.setEnabled(enabled)
        self.btn_prev_file.setEnabled(enabled)
        self.btn_next_file.setEnabled(enabled)
        self.slider_speed.setEnabled(enabled)
        self.file_list.setEnabled(enabled)

    def _heterodyne(self, segment, carrier_freq=40000):
        """Apply heterodyne transformation."""
        t = np.arange(len(segment)) / self.sample_rate
        mixed = segment * np.cos(2 * np.pi * carrier_freq * t)

        from scipy.signal import butter, filtfilt
        nyq = self.sample_rate / 2
        cutoff = min(10000, nyq * 0.9)
        b, a = butter(4, cutoff / nyq, btype='low')
        filtered = filtfilt(b, a, mixed)

        target_len = int(len(filtered) * 44100 / self.sample_rate)
        resampled = signal.resample(filtered, target_len)

        return resampled.astype(np.float32)

    # =========================================================================
    # Project Management
    # =========================================================================

    # ------------------------------------------------------------------
    # Project lifecycle (File menu handlers)
    # ------------------------------------------------------------------

    def _menu_new_project(self):
        """Create a new DAD project: prompt for name, then parent folder.

        No directory is auto-created on disk before the user picks a location —
        if the user cancels the folder picker, no state is left behind.
        """
        from fnt.usv.usv_detector.yolo_detector import create_project

        name, ok = QInputDialog.getText(
            self, "New Project", "Project name:", text="dad_v1"
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # Default to the user's home dir — do not auto-create a projects root.
        start_dir = os.path.expanduser("~")
        parent_dir = QFileDialog.getExistingDirectory(
            self, f"Choose where to save '{name}'", start_dir,
        )
        if not parent_dir:
            return
        project_dir = os.path.join(parent_dir, name)
        if os.path.exists(project_dir):
            QMessageBox.warning(
                self, "Project exists",
                f"A directory named '{name}' already exists in:\n{parent_dir}\n\n"
                "Choose a different name or open the existing project.")
            return

        try:
            config = create_project(project_dir)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project:\n{e}")
            return

        self._close_project(silent=True)
        self._activate_project(config)
        self._remember_recent_project(project_dir)
        self.status_bar.showMessage(f"Created project: {project_dir}")

    def _menu_open_project(self):
        """Open an existing project — user picks the project_info.json file."""
        from fnt.usv.usv_detector.yolo_detector import (
            YOLOProjectConfig, LEGACY_CONFIG_FILENAME,
        )
        start_dir = os.path.expanduser("~")

        config_path, _ = QFileDialog.getOpenFileName(
            self, "Open Project — select project_info.json",
            start_dir,
            "Project Info (project_info.json);;Legacy Config (project_config.json);;All Files (*)",
        )
        if not config_path:
            return

        basename = os.path.basename(config_path)
        if basename not in ("project_info.json", LEGACY_CONFIG_FILENAME):
            QMessageBox.warning(
                self, "Not a project",
                f"Expected project_info.json but got:\n{basename}")
            return

        try:
            config = YOLOProjectConfig.load(config_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{e}")
            return

        self._close_project(silent=True)
        self._activate_project(config)
        self._remember_recent_project(config.project_dir)

    def _menu_close_project(self):
        if self._yolo_project_config is None:
            return
        self._close_project(silent=False)

    def _menu_add_folder(self):
        """Add a folder of wavs to the current project (recursively scanned)."""
        if self._yolo_project_config is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Add folder of .wav files", os.path.expanduser("~"),
        )
        if not folder:
            return
        config = self._yolo_project_config
        if folder in config.source_folders:
            QMessageBox.information(
                self, "Already added",
                f"'{folder}' is already part of this project.")
            return
        config.source_folders.append(folder)
        try:
            config.save()
        except Exception:
            pass
        added = self._rescan_project_wavs()
        self.status_bar.showMessage(
            f"Added folder: {folder} (+{added} new wavs)")

    def _menu_import_cad(self):
        """Import a CAD sidecar CSV into the current file's label set."""
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            return
        filepath = self.audio_files[self.current_file_idx]
        base = Path(filepath).stem
        parent = Path(filepath).parent

        # Sensible default: look for any CAD-style sibling CSV.
        candidate = None
        for suffix in ("_FNT_CAD_detections", "_cad", "_usv_dsp", "_usv_detections", "_usv_yolo"):
            p = parent / f"{base}{suffix}.csv"
            if p.exists():
                candidate = str(p)
                break

        csv_path, _ = QFileDialog.getOpenFileName(
            self, f"Import CAD labels for {base}",
            candidate or str(parent),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not csv_path:
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            QMessageBox.critical(self, "Import failed", str(e))
            return

        required = {"start_seconds", "stop_seconds"}
        if not required.issubset(df.columns):
            QMessageBox.warning(
                self, "Invalid CSV",
                f"Expected columns {required} — got {set(df.columns)}.")
            return

        self._ensure_freq_bounds(df)
        df["status"] = "accepted"
        df["source"] = "cad_import"

        existing = self.detections_df if self.detections_df is not None else pd.DataFrame()
        combined = merge_with_inference(existing, df.to_dict("records"))
        self.detections_df = combined
        self.current_detection_idx = 0
        self._store_current_detections()
        self._update_display()
        self._update_ui_state()
        self.status_bar.showMessage(
            f"Imported {len(df)} labels from {os.path.basename(csv_path)}")

    # ------------------------------------------------------------------
    # Predict menu handlers
    # ------------------------------------------------------------------

    def _menu_run_training(self):
        """Open Run Training dialog and kick off training on accept."""
        if self._yolo_project_config is None:
            return
        try:
            from fnt.usv.usv_detector.yolo_detector import get_training_data_counts
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", str(e))
            return

        counts = get_training_data_counts(self.all_detections)
        if counts['n_positive'] < 1:
            QMessageBox.warning(
                self, "No Positive Labels",
                f"Need at least 1 accepted USV call to train.\n"
                f"Currently: {counts['n_positive']} positive, "
                f"{counts['n_negative']} negative.\n\n"
                "Use A to accept a call. X to mark a background region is\n"
                "optional but helps reduce false positives."
            )
            return

        dlg = RunTrainingDialog(
            self,
            n_positive=counts['n_positive'],
            n_negative=counts['n_negative'],
            n_files=counts['n_files_with_labels'],
            model_variant=self._model_variant,
            imgsz=self._imgsz,
            auto_predict=self._auto_predict_after_train,
            auto_predict_scope=self._auto_predict_scope,
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        self._model_variant = dlg.model_variant()
        self._imgsz = dlg.imgsz()
        self._auto_predict_after_train = dlg.auto_predict()
        self._auto_predict_scope = dlg.auto_predict_scope()

        self._train_model()

    def _menu_run_inference(self):
        """Open Run Inference dialog and kick off detection on accept."""
        if not self._yolo_model_path:
            QMessageBox.warning(
                self, "No Model",
                "No trained model is loaded for this project.\n\n"
                "Run 'Predict → Run Training…' first, or open a project that\n"
                "already contains a trained model.")
            return
        if not self.audio_files:
            QMessageBox.information(
                self, "No Files",
                "This project has no audio files yet. Use\n"
                "'File → Add Folder…' to load some wavs.")
            return

        # Flag whether any target file has pending rows, so the dialog can
        # surface a mild heads-up.
        any_pending = False
        for df in self.all_detections.values():
            if df is None or len(df) == 0:
                continue
            if (df['status'].astype(str) == 'pending').any():
                any_pending = True
                break

        has_current = bool(
            self.audio_files
            and 0 <= self.current_file_idx < len(self.audio_files)
        )

        dlg = RunInferenceDialog(
            self,
            confidence=self._confidence,
            use_sahi=self._use_sahi,
            scope='current' if has_current else 'all',
            overwrite_policy=self._inference_overwrite_policy,
            has_current_file=has_current,
            any_pending_exists=any_pending,
        )
        if dlg.exec_() != QDialog.Accepted:
            return

        self._confidence = dlg.confidence()
        self._use_sahi = dlg.use_sahi()
        self._inference_overwrite_policy = dlg.overwrite_policy()

        if dlg.scope() == 'current' and has_current:
            targets = [self.audio_files[self.current_file_idx]]
        else:
            targets = list(self.audio_files)

        self._detect_files(targets)

    # ------------------------------------------------------------------
    # Project state helpers
    # ------------------------------------------------------------------

    def _activate_project(self, config):
        """Switch the UI over to a freshly loaded project config."""
        self._yolo_project_config = config
        self._sync_ui_from_config(config)
        self.setWindowTitle(f"{self.BASE_TITLE} — {config.project_name}")
        self.lbl_project_name.setText(f"Project: {config.project_name}")
        self.lbl_project_name.setStyleSheet("color: #4CAF50; font-size: 10px;")

        self._yolo_model_path = None
        if config.models:
            last_model = config.models[-1]
            model_path = last_model.get('path', '')
            if os.path.exists(model_path):
                self._yolo_model_path = model_path
                model_name = last_model.get('name', 'unknown')
                n_pos = last_model.get('n_positive', '?')
                n_neg = last_model.get('n_negative', '?')
                self.lbl_model_info.setText(
                    f"Model: {model_name}\n"
                    f"Trained on: {n_pos} positive, {n_neg} negative"
                )
                self.lbl_model_info.setStyleSheet("color: #4CAF50; font-size: 10px;")
            else:
                self.lbl_model_info.setText("Model file not found (retrain needed)")
                self.lbl_model_info.setStyleSheet("color: #d13438; font-size: 10px;")
        else:
            self.lbl_model_info.setText("No trained model")
            self.lbl_model_info.setStyleSheet("color: #999999; font-size: 10px;")

        self.act_add_folder.setEnabled(True)
        self.act_add_files.setEnabled(True)
        self.act_close_project.setEnabled(True)

        # Load wavs and labels from all referenced source folders.
        self._rescan_project_wavs()
        self.status_bar.showMessage(f"Opened project: {config.project_dir}")
        self._update_source_folders_label()
        self._update_project_state()

    def _close_project(self, silent: bool = False):
        """Tear down the current project state."""
        if self._yolo_project_config is None:
            return
        if not silent:
            self.status_bar.showMessage(
                f"Closed project: {self._yolo_project_config.project_name}")
        # Flush any pending in-memory changes to disk before we forget them.
        self._store_current_detections()

        self._yolo_project_config = None
        self._yolo_model_path = None
        self.audio_files = []
        self.all_detections = {}
        self.detection_sources = {}
        self.detections_df = None
        self.current_detection_idx = 0
        self.audio_data = None
        self.sample_rate = None

        self.file_list.blockSignals(True)
        self.file_list.clear()
        self.file_list.blockSignals(False)

        self.setWindowTitle(self.BASE_TITLE)
        self.lbl_project_name.setText("No project — use File → New Project…")
        self.lbl_project_name.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_source_folders.setText("No source folders")
        self.lbl_model_info.setText("No trained model")
        self.lbl_model_info.setStyleSheet("color: #999999; font-size: 10px;")
        self.act_add_folder.setEnabled(False)
        self.act_add_files.setEnabled(False)
        self.act_close_project.setEnabled(False)
        self.act_import_cad.setEnabled(False)

        self.spectrogram.set_audio_data(None, None)
        self.waveform_overview.set_audio_data(None, None)
        self._update_data_summary()
        self._update_display()
        self._update_project_state()

    def _update_source_folders_label(self):
        """Update the 'source folders' status label with the current list."""
        cfg = self._yolo_project_config
        if cfg is None or not cfg.source_folders:
            self.lbl_source_folders.setText("No source folders (File → Add Folder…)")
            return
        lines = []
        for folder in cfg.source_folders:
            lines.append(os.path.basename(os.path.normpath(folder)) or folder)
        n = len(lines)
        self.lbl_source_folders.setText(
            f"{n} folder(s): " + ", ".join(lines[:3])
            + (" …" if n > 3 else "")
        )

    def _rescan_project_wavs(self) -> int:
        """Re-scan source_folders for wavs and (re)load sibling-CSV labels.

        Returns the number of newly added wavs since the last scan.
        """
        cfg = self._yolo_project_config
        if cfg is None:
            return 0

        wavs = [str(p) for p in scan_folders_for_wavs(cfg.source_folders)]
        existing = set(self.audio_files)
        new_wavs = [w for w in wavs if w not in existing]
        added = 0
        legacy_dad = []

        for w in wavs:
            if w not in self.audio_files:
                self.audio_files.append(w)
                added += 1
            if w not in self.all_detections:
                df = load_labels(w)
                if df is not None and len(df) > 0:
                    self.all_detections[w] = df
                    found = find_existing_sibling_csv(w)
                    if found:
                        self.detection_sources[w] = found.stem.rsplit("_", 1)[-1]
                        if found.name.endswith("_dad.csv"):
                            legacy_dad.append((w, found))

        if legacy_dad:
            self._offer_legacy_dad_rename(legacy_dad)

        if self.audio_files:
            self.current_file_idx = min(self.current_file_idx, len(self.audio_files) - 1)
            self._refresh_file_list_items_full()
            self._load_current_file()
        else:
            self.file_list.blockSignals(True)
            self.file_list.clear()
            self.file_list.blockSignals(False)

        self._update_source_folders_label()
        self._update_data_summary()
        self._update_project_state()
        return added

    def _offer_legacy_dad_rename(self, legacy_files):
        """Prompt the user to rename legacy ``_dad.csv`` files to
        ``_FNT_DAD_detections.csv``."""
        n = len(legacy_files)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Legacy CSV filenames detected")
        box.setText(
            f"{n} file{'s' if n != 1 else ''} in the project use the old "
            f"DAD naming convention (_dad.csv).\n\n"
            "Rename them to the new convention (_FNT_DAD_detections.csv)?"
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.Yes)
        if box.exec_() != QMessageBox.Yes:
            return

        renamed = 0
        skipped = 0
        errors = 0
        for filepath, old_path in legacy_files:
            new_path = old_path.with_name(
                old_path.name.replace('_dad.csv', '_FNT_DAD_detections.csv')
            )
            if new_path.exists():
                skipped += 1
                continue
            try:
                old_path.rename(new_path)
                self.detection_sources[filepath] = 'FNT_DAD_detections'
                renamed += 1
            except Exception as e:
                print(f"[DAD] Rename failed for {old_path}: {e}")
                errors += 1

        parts = [f"Renamed {renamed}"]
        if skipped:
            parts.append(f"{skipped} skipped (target exists)")
        if errors:
            parts.append(f"{errors} errors")
        if hasattr(self, 'statusBar'):
            try:
                self.statusBar().showMessage(" — ".join(parts), 5000)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Recent projects (QSettings list)
    # ------------------------------------------------------------------

    def _remember_recent_project(self, project_dir):
        settings = QSettings("FNT", "DeepAudioDetector")
        recents = settings.value("recent_projects", [], type=list) or []
        if project_dir in recents:
            recents.remove(project_dir)
        recents.insert(0, project_dir)
        recents = recents[:10]
        settings.setValue("recent_projects", recents)
        self._rebuild_recent_projects_menu()

    def _rebuild_recent_projects_menu(self):
        if not hasattr(self, "menu_recent"):
            return
        self.menu_recent.clear()
        settings = QSettings("FNT", "DeepAudioDetector")
        recents = settings.value("recent_projects", [], type=list) or []
        if not recents:
            act = QAction("(no recent projects)", self)
            act.setEnabled(False)
            self.menu_recent.addAction(act)
            return
        for path in recents:
            act = QAction(path, self)
            act.triggered.connect(lambda _checked=False, p=path: self._open_recent_project(p))
            self.menu_recent.addAction(act)

    def _open_recent_project(self, project_dir):
        from fnt.usv.usv_detector.yolo_detector import (
            YOLOProjectConfig, PROJECT_INFO_FILENAME, LEGACY_CONFIG_FILENAME,
        )
        candidate = os.path.join(project_dir, PROJECT_INFO_FILENAME)
        if not os.path.exists(candidate):
            candidate = os.path.join(project_dir, LEGACY_CONFIG_FILENAME)
        if not os.path.exists(candidate):
            QMessageBox.warning(
                self, "Missing",
                f"Project file not found at:\n{project_dir}\n\n"
                "It may have been moved or deleted.")
            return
        try:
            config = YOLOProjectConfig.load(candidate)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{e}")
            return
        self._close_project(silent=True)
        self._activate_project(config)
        self._remember_recent_project(config.project_dir)

    def _update_project_state(self):
        """Update label counts and Predict-menu enable state."""
        has_project = self._yolo_project_config is not None
        has_model = self._yolo_model_path is not None
        has_files = len(self.audio_files) > 0

        # Count labels across all files
        n_positive = 0
        n_negative = 0
        for filepath, df in self.all_detections.items():
            if df is None or len(df) == 0:
                continue
            n_positive += int((df['status'] == 'accepted').sum())
            n_negative += int(df['status'].isin(['negative', 'rejected']).sum())
        self._n_positive_labels = n_positive
        self._n_negative_labels = n_negative

        # Train: needs project + at least 1 positive.
        can_train = has_project and n_positive >= 1
        # Inference: needs model + files.
        can_infer = has_model and has_files
        if hasattr(self, 'act_run_training'):
            self.act_run_training.setEnabled(can_train)
        if hasattr(self, 'act_run_inference'):
            self.act_run_inference.setEnabled(can_infer)

    def _sync_ui_from_config(self, config):
        """Load training/inference defaults from the persisted project config."""
        self._model_variant = str(getattr(config, 'model_variant', 'yolo11s.pt'))
        self._imgsz = int(getattr(config, 'imgsz', 1280))
        conf = float(getattr(config, 'confidence_threshold', 0.15))
        self._confidence = max(0.05, min(0.60, conf))
        self._use_sahi = bool(getattr(config, 'use_sahi', True))

    # =========================================================================
    # Training
    # =========================================================================

    def _train_model(self):
        """Export training data and train a YOLO model.

        Reads options from ``self._model_variant`` / ``self._imgsz`` —
        RunTrainingDialog sets these before calling us. The legacy confirmation
        prompt is skipped because the dialog itself is the confirmation step.
        """
        if self._yolo_project_config is None:
            return

        try:
            from fnt.usv.usv_detector.yolo_detector import (
                export_training_data, write_yolo_dataset_yaml,
                get_training_data_counts
            )
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", str(e))
            return

        config = self._yolo_project_config
        counts = get_training_data_counts(self.all_detections)
        n_pos = counts['n_positive']
        n_neg = counts['n_negative']

        if n_pos < 1:
            QMessageBox.warning(
                self, "No Positive Labels",
                f"Need at least 1 accepted USV call to train.\n"
                f"Currently: {n_pos} positive, {n_neg} negative.\n\n"
                f"Use A to accept a call. X to mark a background region is\n"
                f"optional but helps reduce false positives."
            )
            return

        model_variant = self._model_variant
        imgsz = int(self._imgsz)

        # Persist model/imgsz choices into the project config so future
        # sessions and inference runs agree on tile size.
        config.model_variant = model_variant
        config.imgsz = imgsz
        config.tile_size = (imgsz, imgsz)
        try:
            config.save()
        except Exception:
            pass

        # Disable Predict menu actions so the user can't double-start.
        if hasattr(self, 'act_run_training'):
            self.act_run_training.setEnabled(False)
        if hasattr(self, 'act_run_inference'):
            self.act_run_inference.setEnabled(False)

        self.status_bar.showMessage("Exporting training data…")
        QApplication.processEvents()

        dataset_dir = os.path.join(config.project_dir, 'datasets', 'train')
        try:
            export_stats = export_training_data(
                self.audio_files, self.all_detections, dataset_dir, config,
                progress_callback=lambda msg, cur, tot: (
                    self.status_bar.showMessage(msg),
                    QApplication.processEvents(),
                )
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export training data:\n{e}")
            self._update_project_state()
            return

        self.status_bar.showMessage(
            f"Exported {export_stats['n_tiles']} tiles "
            f"({export_stats['n_positive']} positive, {export_stats['n_negative']} negative); "
            f"training…"
        )
        QApplication.processEvents()

        yaml_path = write_yolo_dataset_yaml(config.project_dir, dataset_dir)

        model_name = f"fntDADModel_n={n_pos}"
        model_dir = os.path.join(config.project_dir, 'models', model_name)

        prev_weights = None
        if config.models:
            prev_path = config.models[-1].get('path', '')
            if os.path.exists(prev_path):
                prev_weights = prev_path

        # Live training plot — separate top-level window, thread-safe signal.
        if self._training_plot is None:
            try:
                from fnt.usv.usv_detector.training_plot import TrainingPlotWindow
                self._training_plot = TrainingPlotWindow(self)
            except Exception:
                self._training_plot = None
        if self._training_plot is not None:
            self._training_plot.reset()
            self._training_plot.show()
            self._training_plot.raise_()

        self._yolo_train_worker = YOLOTrainingWorker(
            yaml_path, model_dir, model_name,
            pretrained_weights=prev_weights,
            model_variant=model_variant,
            imgsz=imgsz,
        )
        self._yolo_train_worker.progress.connect(self._on_train_progress)
        if self._training_plot is not None:
            self._yolo_train_worker.epoch.connect(self._training_plot.push_from_worker)
        self._yolo_train_worker.complete.connect(
            lambda path: self._on_train_complete(path, model_name, n_pos, n_neg)
        )
        self._yolo_train_worker.error.connect(self._on_train_error)
        self._yolo_train_worker.start()

    def _on_train_progress(self, message):
        """Forward training progress to the status bar."""
        self.status_bar.showMessage(message)

    def _on_train_complete(self, model_path, model_name, n_pos, n_neg):
        """Handle YOLO training completion."""
        self._yolo_model_path = model_path
        self.lbl_model_info.setText(
            f"Model: {model_name}\n"
            f"Trained on: {n_pos} positive, {n_neg} negative"
        )
        self.lbl_model_info.setStyleSheet("color: #4CAF50; font-size: 10px;")
        self.status_bar.showMessage(
            f"Training complete: {model_name} → {model_path}"
        )

        if self._yolo_project_config is not None:
            self._yolo_project_config.models.append({
                'name': model_name,
                'n_positive': n_pos,
                'n_negative': n_neg,
                'path': model_path,
                'date': datetime.now().isoformat(),
            })
            self._yolo_project_config.save()

        self._update_project_state()

        # Optional train → predict chain.
        if (self._auto_predict_after_train
                and self._yolo_project_config is not None
                and self.audio_files):
            if self._auto_predict_scope == 'current':
                targets = [self.audio_files[self.current_file_idx]] if self.audio_files else []
            else:
                targets = list(self.audio_files)
            if targets:
                self.status_bar.showMessage(
                    f"Auto-predict: running inference on {len(targets)} file(s)"
                )
                self._detect_files(targets)

    def _on_train_error(self, error_msg):
        """Handle YOLO training error."""
        self.status_bar.showMessage("Training failed")
        self._update_project_state()
        QMessageBox.critical(self, "Training Error", f"YOLO training failed:\n{error_msg}")

    # =========================================================================
    # Inference
    # =========================================================================

    def _detect_current(self):
        """Run YOLO detection on current file."""
        if not self._yolo_model_path or not self.audio_files:
            return
        filepath = self.audio_files[self.current_file_idx]
        self._detect_files([filepath])

    def _detect_all(self):
        """Run YOLO detection on all files."""
        if not self._yolo_model_path or not self.audio_files:
            return
        self._detect_files(list(self.audio_files))

    def _clear_pending_predictions(self, files):
        """Strip pending/ml rows from the given files' label sets in-memory.

        Accepted, rejected, and user-drawn rows are preserved. The canonical
        sibling CSV is rewritten in place so on-disk state matches memory.
        """
        for filepath in files:
            df = self.all_detections.get(filepath)
            if df is None or len(df) == 0:
                continue
            keep_mask = ~(
                (df['status'].astype(str) == 'pending')
                & (df.get('source', pd.Series(['ml'] * len(df))).astype(str) == 'ml')
            )
            kept = df[keep_mask].reset_index(drop=True)
            self.all_detections[filepath] = kept
            try:
                save_labels(filepath, kept)
            except Exception:
                pass

    def _detect_files(self, files):
        """Run YOLO inference on a list of files, honoring overwrite policy."""
        if not self._yolo_model_path or not self._yolo_project_config:
            return
        if not files:
            return

        # Optionally strip pending/ml rows before the merge so old predictions
        # don't accumulate. Accepted / rejected / user rows are always kept.
        if self._inference_overwrite_policy == 'replace_pending':
            self._clear_pending_predictions(files)
            # Refresh current-file view if it got touched.
            if self.audio_files and self.current_file_idx < len(self.audio_files):
                current = self.audio_files[self.current_file_idx]
                if current in files:
                    df = self.all_detections.get(current)
                    self.detections_df = df.copy() if df is not None and len(df) > 0 else None
                    self.current_detection_idx = 0
                    self._update_display()

        conf = float(self._confidence)
        use_sahi = bool(self._use_sahi)
        self._yolo_project_config.confidence_threshold = conf
        self._yolo_project_config.use_sahi = use_sahi

        if hasattr(self, 'act_run_training'):
            self.act_run_training.setEnabled(False)
        if hasattr(self, 'act_run_inference'):
            self.act_run_inference.setEnabled(False)

        mode = "SAHI" if use_sahi else "tiled"
        self.status_bar.showMessage(
            f"Running YOLO detection ({mode}, conf={conf:.2f}) on {len(files)} file(s)…"
        )

        self._yolo_infer_worker = YOLOInferenceWorker(
            files, self._yolo_model_path, self._yolo_project_config,
            confidence_threshold=conf,
        )
        self._yolo_infer_worker.progress.connect(self._on_infer_progress)
        self._yolo_infer_worker.file_complete.connect(self._on_infer_file_complete)
        self._yolo_infer_worker.all_complete.connect(self._on_infer_complete)
        self._yolo_infer_worker.error.connect(self._on_infer_error)
        self._yolo_infer_worker.start()

    def _on_infer_progress(self, filename, current, total):
        """Forward ML inference progress to the status bar."""
        pct = int(current / total * 100) if total else 0
        self.status_bar.showMessage(f"Detecting ({pct}%): {filename}")

    def _on_infer_file_complete(self, filename, filepath, detections, n_detections):
        """Handle ML file completion — merge with existing user labels and persist.

        New inference boxes arrive with ``status='pending'`` / ``source='ml'``.
        We merge them into any existing labels via ``merge_with_inference``,
        which preserves user-sourced rows verbatim and dedupes overlapping
        inference rows.
        """
        if isinstance(detections, pd.DataFrame):
            inference_rows = detections.to_dict('records')
        elif isinstance(detections, list):
            inference_rows = [dict(d) for d in detections]
        else:
            inference_rows = []

        # Normalize inference rows: status='pending', source='ml'.
        for row in inference_rows:
            row.setdefault('status', 'pending')
            row.setdefault('source', 'ml')

        existing = self.all_detections.get(filepath)
        if existing is None:
            existing = load_labels(filepath)  # pick up on-disk labels we haven't seen yet

        merged = merge_with_inference(existing, inference_rows)
        if len(merged) > 0:
            self._ensure_freq_bounds(merged)

        self.all_detections[filepath] = merged
        self.detection_sources[filepath] = DAD_SUFFIX.lstrip('_')

        # Persist via the canonical writer.
        try:
            save_labels(filepath, merged)
        except Exception as e:
            self.status_bar.showMessage(f"Error saving CSV for {filename}: {e}")

        # Update file list count.
        if filepath in self.audio_files:
            idx = self.audio_files.index(filepath)
            item = self.file_list.item(idx)
            if item:
                self.file_list.blockSignals(True)
                item.setText(f"{filename} ({len(merged)})")
                self.file_list.blockSignals(False)

        # Update current view if this is the displayed file.
        if (self.audio_files and self.current_file_idx < len(self.audio_files)
                and self.audio_files[self.current_file_idx] == filepath):
            self.detections_df = merged.copy() if len(merged) > 0 else None
            self.current_detection_idx = 0
            self._update_display()
            self._update_ui_state()

    def _on_infer_complete(self, results):
        """Handle ML inference batch completion."""
        # Prefer the post-merge totals we actually stored, so the status bar
        # matches what the user sees in the UI (user-drawn rows included).
        n_files = len(results)
        total_det = 0
        for filepath in results:
            df = self.all_detections.get(filepath)
            if isinstance(df, pd.DataFrame):
                total_det += len(df)
            elif isinstance(results[filepath], (list, pd.DataFrame)):
                total_det += len(results[filepath])

        self.status_bar.showMessage(
            f"Detected {total_det} USVs across {n_files} file(s)"
        )

        self._refresh_file_list_items()
        # Refresh the currently-displayed file's labels from what we merged.
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            current = self.audio_files[self.current_file_idx]
            merged = self.all_detections.get(current)
            if merged is not None and len(merged):
                self.detections_df = merged.copy()
                self._ensure_freq_bounds(self.detections_df)
            else:
                self.detections_df = None
            self.current_detection_idx = 0
            self._update_display()
        self._update_ui_state()
        self._update_project_state()

    def _on_infer_error(self, filename, error):
        """Handle ML inference error."""
        self.status_bar.showMessage(f"Inference error on {filename}: {error}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def _update_statistics(self):
        """Update detection statistics label."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.lbl_stats.setText("")
            return

        n = len(self.detections_df)
        durations = self.detections_df['stop_seconds'] - self.detections_df['start_seconds']
        dur_ms = durations * 1000
        total_dur = self.spectrogram.get_total_duration()
        cpm = n / (total_dur / 60) if total_dur > 0 else 0

        parts = [f"{n} calls | {cpm:.1f}/min"]
        parts.append(f"Dur: {dur_ms.mean():.1f}ms (SD {dur_ms.std():.1f})")

        if 'peak_freq_hz' in self.detections_df.columns:
            pf = self.detections_df['peak_freq_hz'].dropna()
            if len(pf) > 0:
                parts.append(f"Peak F: {pf.mean()/1000:.1f}kHz")

        self.lbl_stats.setText(" | ".join(parts))

    # =========================================================================
    # UI State Management
    # =========================================================================

    def _update_ui_state(self):
        """Update UI enabled states based on current state."""
        has_files = len(self.audio_files) > 0
        has_audio = self.audio_data is not None
        has_det = self.detections_df is not None and len(self.detections_df) > 0

        # Detection navigation
        self.btn_prev_det.setEnabled(bool(has_det and self.current_detection_idx > 0))
        self.btn_next_det.setEnabled(
            bool(has_det and self.current_detection_idx < len(self.detections_df) - 1) if has_det else False
        )

        # Labeling
        self.btn_accept.setEnabled(bool(has_det))
        self.btn_reject.setEnabled(bool(has_det))
        self.btn_skip.setEnabled(bool(has_det))
        self.btn_add_usv.setEnabled(bool(has_audio))
        self.btn_delete.setEnabled(bool(has_det))

        has_pending = bool(has_det and (self.detections_df['status'] == 'pending').any())
        self.btn_delete_pending.setEnabled(has_pending)
        self.btn_delete_all_labels.setEnabled(bool(has_det))
        self.btn_accept_all_pending.setEnabled(has_pending)
        self.btn_reject_all_pending.setEnabled(has_pending)

        # Playback
        self.btn_play.setEnabled(bool(has_audio and HAS_SOUNDDEVICE))
        self.btn_stop.setEnabled(bool(has_audio and HAS_SOUNDDEVICE))

        # Project state (train/detect)
        self._update_project_state()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            super().keyPressEvent(event)
            return

        key = event.key()
        modifiers = event.modifiers()

        if key == Qt.Key_Z and (modifiers & Qt.ControlModifier):
            self.undo_action()
            return

        if key == Qt.Key_A and self.btn_accept.isEnabled():
            self._flash_button(self.btn_accept)
            self.accept_detection()
        elif key == Qt.Key_R and self.btn_reject.isEnabled():
            self._flash_button(self.btn_reject)
            self.reject_detection()
        elif key == Qt.Key_X:
            self.mark_negative()
        elif key == Qt.Key_D and self.btn_delete.isEnabled():
            self._flash_button(self.btn_delete)
            self.delete_current()
        elif key == Qt.Key_Space and self.btn_play.isEnabled():
            self._flash_button(self.btn_play)
            self.toggle_playback()
        else:
            super().keyPressEvent(event)

    def _flash_button(self, button):
        """Briefly highlight a button to give visual feedback."""
        original_style = button.styleSheet()
        button.setStyleSheet("background-color: #ffffff; color: #000000;")
        QTimer.singleShot(120, lambda: button.setStyleSheet(original_style))

    # =========================================================================
    # Settings Persistence
    # =========================================================================

    def _restore_settings(self):
        """Restore window geometry and preferences from QSettings."""
        settings = QSettings("FNT", "DeepAudioDetector")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("windowState")
        if state:
            self.restoreState(state)
        view_window = settings.value("view_window", 2.0, type=float)
        self.spin_view_window.setValue(view_window)
        colormap = settings.value("colormap", "viridis", type=str)
        idx = self.combo_colormap.findText(colormap)
        if idx >= 0:
            self.combo_colormap.setCurrentIndex(idx)

    def closeEvent(self, event):
        """Save settings on close."""
        settings = QSettings("FNT", "DeepAudioDetector")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("view_window", self.spin_view_window.value())
        settings.setValue("colormap", self.combo_colormap.currentText())
        super().closeEvent(event)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    app = QApplication(sys.argv)
    window = DeepAudioDetectorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
