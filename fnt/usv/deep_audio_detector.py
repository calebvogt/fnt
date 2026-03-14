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
    QDialog, QDialogButtonBox
)
from scipy import signal

from fnt.usv.audio_widgets import SpectrogramWidget, WaveformOverviewWidget

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
    complete = pyqtSignal(str)       # model_path
    error = pyqtSignal(str)          # error message

    def __init__(self, dataset_yaml, output_dir, model_name, pretrained_weights=None):
        super().__init__()
        self.dataset_yaml = dataset_yaml
        self.output_dir = output_dir
        self.model_name = model_name
        self.pretrained_weights = pretrained_weights

    def run(self):
        try:
            from fnt.usv.usv_detector.yolo_detector import train_yolo_model
            self.progress.emit(f"Training {self.model_name} (auto early stopping)...")
            model_path = train_yolo_model(
                self.dataset_yaml, self.output_dir, self.model_name,
                pretrained_weights=self.pretrained_weights,
            )
            self.complete.emit(model_path)
        except Exception as e:
            self.error.emit(str(e))


class YOLOInferenceWorker(QThread):
    progress = pyqtSignal(str, int, int)  # filename, current, total
    file_complete = pyqtSignal(str, str, list, int)  # filename, filepath, detections, count
    all_complete = pyqtSignal(dict)  # results
    error = pyqtSignal(str, str)  # filename, error

    def __init__(self, files, model_path, config):
        super().__init__()
        self.files = files
        self.model_path = model_path
        self.config = config

    def run(self):
        from fnt.usv.usv_detector.yolo_detector import run_yolo_inference
        results = {}
        for i, filepath in enumerate(self.files):
            filename = os.path.basename(filepath)
            try:
                self.progress.emit(filename, i + 1, len(self.files))
                detections = run_yolo_inference(self.model_path, filepath, self.config)
                results[filepath] = detections
                self.file_complete.emit(filename, filepath, detections, len(detections))
            except Exception as e:
                self.error.emit(filename, str(e))
        self.all_complete.emit(results)


# =============================================================================
# Main Deep Audio Detector Window
# =============================================================================

class DeepAudioDetectorWindow(QMainWindow):
    """Main window for Deep Audio Detector."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FNT Deep Audio Detector")
        self.setMinimumSize(1000, 700)
        self.resize(1400, 900)

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
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left panel (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFixedWidth(380)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)

        # Build sections
        self._create_project_section(left_layout)
        self._create_training_data_section(left_layout)
        self._create_detection_section(left_layout)
        self._create_labeling_section(left_layout)
        self._create_train_section(left_layout)
        self._create_inference_section(left_layout)

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
        self._speed_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(0, len(self._speed_values) - 1)
        self.slider_speed.setValue(5)
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
        """Create project management section."""
        group = QGroupBox("1. Project")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_create_project = QPushButton("Create Project")
        self.btn_create_project.setObjectName("accept_btn")
        self.btn_create_project.setToolTip("Create a new YOLO detection project directory")
        self.btn_create_project.clicked.connect(self._create_project)
        btn_row.addWidget(self.btn_create_project)

        self.btn_open_project = QPushButton("Open Project")
        self.btn_open_project.setToolTip("Open an existing YOLO detection project")
        self.btn_open_project.clicked.connect(self._open_project)
        btn_row.addWidget(self.btn_open_project)

        group_layout.addLayout(btn_row)

        self.lbl_project_name = QLabel("No project")
        self.lbl_project_name.setStyleSheet("color: #999999; font-size: 10px;")
        group_layout.addWidget(self.lbl_project_name)

        self.lbl_model_info = QLabel("No trained model")
        self.lbl_model_info.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_model_info.setWordWrap(True)
        group_layout.addWidget(self.lbl_model_info)

        group.setLayout(group_layout)
        layout.addWidget(group)

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

    def _create_train_section(self, layout):
        """Create model training section."""
        group = QGroupBox("5. Train Model")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        self.lbl_train_data = QLabel("Labels: 0 positive, 0 negative")
        self.lbl_train_data.setStyleSheet("color: #999999; font-size: 10px;")
        group_layout.addWidget(self.lbl_train_data)

        self.btn_train = QPushButton("Export & Train")
        self.btn_train.setObjectName("accept_btn")
        self.btn_train.setToolTip("Export labeled data and train a YOLO model.")
        self.btn_train.clicked.connect(self._train_model)
        self.btn_train.setEnabled(False)
        group_layout.addWidget(self.btn_train)

        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        self.train_progress.setVisible(False)
        group_layout.addWidget(self.train_progress)

        self.lbl_train_status = QLabel("")
        self.lbl_train_status.setStyleSheet("color: #999999; font-size: 9px;")
        self.lbl_train_status.setWordWrap(True)
        group_layout.addWidget(self.lbl_train_status)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_inference_section(self, layout):
        """Create inference section."""
        group = QGroupBox("6. Inference")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        run_row = QHBoxLayout()
        run_row.setSpacing(2)

        self.btn_detect_current = QPushButton("Detect Current")
        self.btn_detect_current.setToolTip("Run YOLO detection on the current file.")
        self.btn_detect_current.clicked.connect(self._detect_current)
        self.btn_detect_current.setEnabled(False)
        run_row.addWidget(self.btn_detect_current)

        self.btn_detect_all = QPushButton("Detect All")
        self.btn_detect_all.setToolTip("Run YOLO detection on all loaded files.")
        self.btn_detect_all.clicked.connect(self._detect_all)
        self.btn_detect_all.setEnabled(False)
        run_row.addWidget(self.btn_detect_all)

        group_layout.addLayout(run_row)

        self.infer_progress = QProgressBar()
        self.infer_progress.setValue(0)
        self.infer_progress.setVisible(False)
        group_layout.addWidget(self.infer_progress)

        self.lbl_infer_status = QLabel("")
        self.lbl_infer_status.setStyleSheet("color: #999999; font-size: 9px;")
        self.lbl_infer_status.setWordWrap(True)
        group_layout.addWidget(self.lbl_infer_status)

        group.setLayout(group_layout)
        layout.addWidget(group)

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
        """Try to load existing detections for a file."""
        base = Path(filepath).stem
        parent = Path(filepath).parent

        for suffix in ['_yoloDetection', '_usv_yolo', '_usv_dsp', '_usv_detections']:
            csv_path = parent / f"{base}{suffix}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if 'status' not in df.columns:
                        df['status'] = 'pending'
                    self._ensure_freq_bounds(df)
                    self.all_detections[filepath] = df
                    self.detection_sources[filepath] = suffix.lstrip('_')
                    if filepath == self.audio_files[self.current_file_idx] if self.audio_files else False:
                        self.detections_df = df.copy()
                        self.current_detection_idx = 0
                    return
                except Exception:
                    pass

        if filepath == (self.audio_files[self.current_file_idx] if self.audio_files else None):
            self.detections_df = None
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
            base = Path(filepath).stem
            parent = Path(filepath).parent
            source = self.detection_sources.get(filepath, 'yoloDetection')
            csv_path = parent / f"{base}_{source}.csv"
            if csv_path.exists():
                csv_path.unlink()

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
                base = Path(filepath).stem
                parent = Path(filepath).parent
                for suffix in ['_yoloDetection', '_usv_yolo', '_usv_dsp', '_usv_detections']:
                    if (parent / f"{base}{suffix}.csv").exists():
                        has_dets = True
                        break

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
        """Save current detections to CSV file as {stem}_yoloDetection.csv."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        self.detections_df['call_number'] = range(1, len(self.detections_df) + 1)
        cols = self.detections_df.columns.tolist()
        if 'call_number' in cols:
            cols.remove('call_number')
            cols.insert(0, 'call_number')
            self.detections_df = self.detections_df[cols]

        base = Path(filepath).stem
        parent = Path(filepath).parent
        csv_path = parent / f"{base}_yoloDetection.csv"

        try:
            self.detections_df.to_csv(csv_path, index=False)
            self.detection_sources[filepath] = 'yoloDetection'
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

    def _create_project(self):
        """Create a new YOLO detection project."""
        project_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory for New Project"
        )
        if not project_dir:
            return

        try:
            from fnt.usv.usv_detector.yolo_detector import create_project
            self._yolo_project_config = create_project(project_dir)
            self._yolo_model_path = None
            self.lbl_project_name.setText(f"Project: {os.path.basename(project_dir)}")
            self.lbl_project_name.setStyleSheet("color: #4CAF50; font-size: 10px;")
            self.lbl_model_info.setText("No trained model")
            self.lbl_model_info.setStyleSheet("color: #999999; font-size: 10px;")
            self.status_bar.showMessage(f"Created project: {project_dir}")
            self._update_project_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project:\n{e}")

    def _open_project(self):
        """Open an existing YOLO detection project."""
        project_dir = QFileDialog.getExistingDirectory(
            self, "Select Existing Project Directory"
        )
        if not project_dir:
            return

        config_path = os.path.join(project_dir, 'project_config.json')
        if not os.path.exists(config_path):
            QMessageBox.warning(
                self, "Invalid Project",
                f"No project_config.json found in:\n{project_dir}\n\n"
                "Use 'Create Project' to create a new project."
            )
            return

        try:
            from fnt.usv.usv_detector.yolo_detector import YOLOProjectConfig
            self._yolo_project_config = YOLOProjectConfig.load(config_path)
            self.lbl_project_name.setText(f"Project: {os.path.basename(project_dir)}")
            self.lbl_project_name.setStyleSheet("color: #4CAF50; font-size: 10px;")

            # Check for existing models
            self._yolo_model_path = None
            if self._yolo_project_config.models:
                last_model = self._yolo_project_config.models[-1]
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

            self.status_bar.showMessage(f"Opened project: {project_dir}")
            self._update_project_state()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{e}")

    def _update_project_state(self):
        """Update label counts and enable/disable train & detect buttons."""
        has_project = self._yolo_project_config is not None
        has_model = self._yolo_model_path is not None
        has_files = len(self.audio_files) > 0

        # Count labels across all files
        n_positive = 0
        n_negative = 0
        for filepath, df in self.all_detections.items():
            if df is None or len(df) == 0:
                continue
            n_positive += (df['status'] == 'accepted').sum()
            n_negative += df['status'].isin(['negative', 'rejected']).sum()

        self.lbl_train_data.setText(f"Labels: {n_positive} positive, {n_negative} negative")

        # Train button: needs project + at least 1 positive + at least 1 negative
        can_train = has_project and n_positive >= 1 and n_negative >= 1
        self.btn_train.setEnabled(can_train)

        # Detect buttons: needs model + files
        self.btn_detect_current.setEnabled(has_model and has_files)
        self.btn_detect_all.setEnabled(has_model and has_files)

    # =========================================================================
    # Training
    # =========================================================================

    def _train_model(self):
        """Export training data and train YOLO model."""
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

        if n_pos < 1 or n_neg < 1:
            QMessageBox.warning(
                self, "Insufficient Labels",
                f"Need at least 1 positive and 1 negative label.\n"
                f"Currently: {n_pos} positive, {n_neg} negative.\n\n"
                f"Use A key to accept USV calls, X key to mark background."
            )
            return

        reply = QMessageBox.question(
            self, "Train YOLO Model",
            f"Training data: {n_pos} positive, {n_neg} negative "
            f"from {counts['n_files_with_labels']} files.\n\n"
            f"This will export spectrogram tiles and train a YOLOv8 model.\n"
            f"Training stops automatically when loss plateaus.\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.btn_train.setEnabled(False)
        self.train_progress.setVisible(True)
        self.train_progress.setValue(0)
        self.lbl_train_status.setText("Exporting training data...")
        QApplication.processEvents()

        dataset_dir = os.path.join(config.project_dir, 'datasets', 'train')
        try:
            export_stats = export_training_data(
                self.audio_files, self.all_detections, dataset_dir, config,
                progress_callback=lambda msg, cur, tot: (
                    self.lbl_train_status.setText(msg),
                    self.train_progress.setValue(int(cur / max(tot, 1) * 30)),
                    QApplication.processEvents(),
                )
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export training data:\n{e}")
            self.btn_train.setEnabled(True)
            self.train_progress.setVisible(False)
            return

        self.lbl_train_status.setText(
            f"Exported {export_stats['n_tiles']} tiles "
            f"({export_stats['n_positive']} positive, {export_stats['n_negative']} negative)"
        )
        self.train_progress.setValue(30)
        QApplication.processEvents()

        yaml_path = write_yolo_dataset_yaml(config.project_dir, dataset_dir)

        model_name = f"fntDADModel_n={n_pos}"
        model_dir = os.path.join(config.project_dir, 'models', model_name)

        prev_weights = None
        if config.models:
            prev_path = config.models[-1].get('path', '')
            if os.path.exists(prev_path):
                prev_weights = prev_path

        self.lbl_train_status.setText("Training YOLO model...")
        self.train_progress.setValue(35)
        QApplication.processEvents()

        self._yolo_train_worker = YOLOTrainingWorker(
            yaml_path, model_dir, model_name,
            pretrained_weights=prev_weights,
        )
        self._yolo_train_worker.progress.connect(self._on_train_progress)
        self._yolo_train_worker.complete.connect(
            lambda path: self._on_train_complete(path, model_name, n_pos, n_neg)
        )
        self._yolo_train_worker.error.connect(self._on_train_error)
        self._yolo_train_worker.start()

    def _on_train_progress(self, message):
        """Handle YOLO training progress."""
        self.lbl_train_status.setText(message)
        current = self.train_progress.value()
        if current < 95:
            self.train_progress.setValue(current + 1)

    def _on_train_complete(self, model_path, model_name, n_pos, n_neg):
        """Handle YOLO training completion."""
        self.train_progress.setValue(100)
        self.train_progress.setVisible(False)
        self.btn_train.setEnabled(True)

        self._yolo_model_path = model_path
        self.lbl_model_info.setText(
            f"Model: {model_name}\n"
            f"Trained on: {n_pos} positive, {n_neg} negative"
        )
        self.lbl_model_info.setStyleSheet("color: #4CAF50; font-size: 10px;")
        self.lbl_train_status.setText(f"Training complete! Model: {model_name}")
        self.status_bar.showMessage(f"YOLO model trained: {model_path}")

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

    def _on_train_error(self, error_msg):
        """Handle YOLO training error."""
        self.train_progress.setVisible(False)
        self.btn_train.setEnabled(True)
        self.lbl_train_status.setText("Training failed")
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

    def _detect_files(self, files):
        """Run YOLO inference on a list of files."""
        if not self._yolo_model_path or not self._yolo_project_config:
            return

        self.btn_detect_current.setEnabled(False)
        self.btn_detect_all.setEnabled(False)
        self.infer_progress.setVisible(True)
        self.infer_progress.setValue(0)
        self.lbl_infer_status.setText("Running YOLO detection...")

        self._yolo_infer_worker = YOLOInferenceWorker(
            files, self._yolo_model_path, self._yolo_project_config,
        )
        self._yolo_infer_worker.progress.connect(self._on_infer_progress)
        self._yolo_infer_worker.file_complete.connect(self._on_infer_file_complete)
        self._yolo_infer_worker.all_complete.connect(self._on_infer_complete)
        self._yolo_infer_worker.error.connect(self._on_infer_error)
        self._yolo_infer_worker.start()

    def _on_infer_progress(self, filename, current, total):
        """Handle ML inference progress."""
        self.infer_progress.setValue(int(current / total * 100))
        self.lbl_infer_status.setText(f"Detecting: {filename}")

    def _on_infer_file_complete(self, filename, filepath, detections, n_detections):
        """Handle ML file completion - store results and write CSV."""
        std_columns = ['call_number', 'start_seconds', 'stop_seconds', 'duration_ms',
                        'min_freq_hz', 'max_freq_hz', 'peak_freq_hz', 'freq_bandwidth_hz',
                        'max_power_db', 'mean_power_db', 'confidence', 'status', 'source']

        if isinstance(detections, pd.DataFrame):
            df = detections.copy()
        else:
            df = pd.DataFrame(detections)

        if len(df) > 0:
            if 'status' not in df.columns:
                df['status'] = 'pending'
            if 'source' not in df.columns:
                df['source'] = 'ml'
            if 'call_number' not in df.columns:
                df.insert(0, 'call_number', range(1, len(df) + 1))
        else:
            df = pd.DataFrame(columns=std_columns)

        self.all_detections[filepath] = df
        self.detection_sources[filepath] = 'yoloDetection'

        # Write CSV as {stem}_yoloDetection.csv
        base = Path(filepath).stem
        parent = Path(filepath).parent
        csv_path = parent / f"{base}_yoloDetection.csv"
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            self.status_bar.showMessage(f"Error saving CSV for {filename}: {e}")

        # Update file list
        if filepath in self.audio_files:
            idx = self.audio_files.index(filepath)
            item = self.file_list.item(idx)
            if item:
                self.file_list.blockSignals(True)
                item.setText(f"{filename} ({len(df)})")
                self.file_list.blockSignals(False)

        # Update current view if this is the displayed file
        if (self.audio_files and self.current_file_idx < len(self.audio_files)
                and self.audio_files[self.current_file_idx] == filepath):
            self.detections_df = df.copy()
            self._ensure_freq_bounds(self.detections_df)
            self.current_detection_idx = 0
            self._load_current_file()

    def _on_infer_complete(self, results):
        """Handle ML inference batch completion."""
        self.infer_progress.setValue(100)
        self.infer_progress.setVisible(False)
        self.btn_detect_current.setEnabled(True)
        self.btn_detect_all.setEnabled(True)

        total_det = sum(len(d) for d in results.values() if isinstance(d, (list, pd.DataFrame)))
        self.lbl_infer_status.setText(f"Complete: {total_det} detections in {len(results)} files")
        self.status_bar.showMessage(f"YOLO detection complete: {total_det} detections")

        self._refresh_file_list_items()
        self._load_current_file()
        self._update_project_state()

    def _on_infer_error(self, filename, error):
        """Handle ML inference error."""
        self.lbl_infer_status.setText(f"Error: {filename} - {error}")

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
