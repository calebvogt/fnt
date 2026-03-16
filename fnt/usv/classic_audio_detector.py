"""
Classic Audio Detector - DSP-based detection and labeling tool.

A comprehensive PyQt5 application for:
1. Loading and browsing audio files
2. Running DSP-based USV detection
3. Manual labeling and ground-truthing
4. Training Random Forest classifiers

Author: FNT Project
"""

import json
import math
import os
import sys
from collections import deque
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
    QDialog, QDialogButtonBox, QInputDialog, QLineEdit
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

class DSPDetectionWorker(QThread):
    """Worker thread for DSP detection."""
    progress = pyqtSignal(str, int, int)  # filename, current, total
    file_progress = pyqtSignal(float)  # fraction 0.0-1.0 within current file
    file_complete = pyqtSignal(str, str, list, int)  # filename, filepath, detections, n_detections
    all_complete = pyqtSignal(dict)  # results dict
    error = pyqtSignal(str, str)  # filename, error message

    def __init__(self, files: List[str], config: dict):
        super().__init__()
        self.files = files
        self.config = config
        self.results = {}
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        from fnt.usv.usv_detector.dsp_detector import DSPDetector
        from fnt.usv.usv_detector.config import USVDetectorConfig

        config = USVDetectorConfig(
            min_freq_hz=self.config.get('min_freq_hz', 20000),
            max_freq_hz=self.config.get('max_freq_hz', 65000),
            energy_threshold_db=self.config.get('energy_threshold_db', 10.0),
            min_duration_ms=self.config.get('min_duration_ms', 10.0),
            max_duration_ms=self.config.get('max_duration_ms', 1000.0),
            max_bandwidth_hz=self.config.get('max_bandwidth_hz', 20000.0),
            min_tonality=self.config.get('min_tonality', 0.3),
            min_call_freq_hz=self.config.get('min_call_freq_hz', 0.0),
            harmonic_filter=self.config.get('harmonic_filter', True),
            harmonic_label=self.config.get('harmonic_label', False),
            min_freq_gap_hz=self.config.get('min_freq_gap_hz', 5000.0),
            min_gap_ms=self.config.get('min_gap_ms', 5.0),
            noise_percentile=self.config.get('noise_percentile', 25.0),
            nperseg=self.config.get('nperseg', 512),
            noverlap=self.config.get('noverlap', 384),
            gpu_enabled=self.config.get('gpu_enabled', False),
            gpu_device=self.config.get('gpu_device', 'auto'),
            # Advanced noise rejection
            min_bandwidth_hz=self.config.get('min_bandwidth_hz', 0.0),
            min_snr_db=self.config.get('min_snr_db', 0.0),
            min_spectral_entropy=self.config.get('min_spectral_entropy', 0.0),
            max_spectral_entropy=self.config.get('max_spectral_entropy', 0.0),
            min_power_db=self.config.get('min_power_db', 0.0),
            max_power_db=self.config.get('max_power_db', 0.0),
            max_mean_sweep_rate=self.config.get('max_mean_sweep_rate', 0.0),
            max_contour_jitter=self.config.get('max_contour_jitter', 0.0),
            min_ici_ms=self.config.get('min_ici_ms', 0.0),
        )

        detector = DSPDetector(config)
        total = len(self.files)

        for i, filepath in enumerate(self.files):
            if self._stop_requested:
                break

            filename = os.path.basename(filepath)
            self.progress.emit(filename, i, total)
            self.file_progress.emit(0.0)

            try:
                def on_chunk_progress(fraction):
                    self.file_progress.emit(fraction)

                detections = detector.detect_file(filepath, progress_callback=on_chunk_progress)
                # If stop was requested during processing, discard this file's results
                if self._stop_requested:
                    break
                self.results[filepath] = detections
                self.file_complete.emit(filename, filepath, detections, len(detections))
            except Exception as e:
                if self._stop_requested:
                    break
                self.error.emit(filename, str(e))
                self.results[filepath] = []

        self.all_complete.emit(self.results)



# =============================================================================
# Main Classic Audio Detector Window
# =============================================================================

class ClassicAudioDetectorWindow(QMainWindow):
    """Main window for Classic Audio Detector."""

    # Species-specific DSP detection presets.
    # 'Manual' means user controls all parameters directly.
    # Add new species by adding a key with a dict of DSP config values.
    SPECIES_PROFILES = {
        'Manual': None,
        'Rodent USVs': {
            'min_freq_hz': 18000, 'max_freq_hz': 120000,
            'energy_threshold_db': 8.0,
            'min_duration_ms': 3.0, 'max_duration_ms': 3000.0,
            'max_bandwidth_hz': 25000, 'min_tonality': 0.05,
            'min_call_freq_hz': 15000, 'harmonic_filter': True,
            'min_freq_gap_hz': 5000,
            'min_gap_ms': 5.0, 'noise_percentile': 25.0,
            'nperseg': 512, 'noverlap': 384,
            # Advanced noise rejection
            'min_bandwidth_hz': 500, 'min_snr_db': 6.0,
            'min_spectral_entropy': 0.0, 'max_spectral_entropy': 0.0,
            'min_power_db': 0.0, 'max_power_db': 0.0,
            'max_mean_sweep_rate': 0.0, 'max_contour_jitter': 0.0,
            'min_ici_ms': 0.0,
        },
        'Prairie Vole USVs': {
            # Freq range: adults 20-45 kHz fundamental (Stewart 2015).
            # Cap at 55 kHz to capture full FM sweeps while excluding
            # most harmonic energy and broadband noise.
            'min_freq_hz': 20000, 'max_freq_hz': 55000,
            # Threshold: 10 dB — real calls are bright and well above noise.
            'energy_threshold_db': 10.0,
            # Duration: 8 ms min — noise fragments are 3-5 ms; real prairie
            # vole syllables are typically 10-40 ms (Stewart 2015).
            'min_duration_ms': 8.0, 'max_duration_ms': 500.0,
            # Bandwidth: 30 kHz max — real swept calls span 15-25 kHz;
            # 20 kHz was too tight and rejected legitimate FM sweeps.
            'max_bandwidth_hz': 30000,
            # Tonality: 0.20 — empirically tuned. Real calls measure
            # 0.14-0.62 tonality; 0.20 balances sensitivity vs noise.
            'min_tonality': 0.20,
            # Min call freq: 15 kHz — provides margin below the 20 kHz
            # fundamental floor without admitting low-frequency noise.
            'min_call_freq_hz': 15000,
            # Harmonic filter + labeling with 5 kHz gap.
            'harmonic_filter': True, 'harmonic_label': True,
            'min_freq_gap_hz': 5000,
            # Min gap: 4 ms — preserves rapid syllable sequences.
            'min_gap_ms': 4.0,
            'noise_percentile': 25.0,
            'nperseg': 512, 'noverlap': 384,
            # Advanced noise rejection — empirically tuned
            # Min bandwidth 1000 Hz: rejects pure-tone electrical noise.
            'min_bandwidth_hz': 1000,
            # Min SNR 8 dB: real calls are clearly above noise floor.
            'min_snr_db': 8.0,
            # Spectral entropy: disabled — not reliable at our FFT resolution.
            'min_spectral_entropy': 0.0, 'max_spectral_entropy': 0.0,
            # Power: disabled.
            'min_power_db': 0.0, 'max_power_db': 0.0,
            # Sweep/contour/ICI: disabled.
            'max_mean_sweep_rate': 0.0,
            'max_contour_jitter': 0.0,
            'min_ici_ms': 0.0,
        },
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FNT Classic Audio Detector")
        self.setMinimumSize(1000, 700)
        self.resize(1400, 900)  # Initial size, but resizable

        self.audio_files = []  # List of audio file paths
        self.current_file_idx = 0
        self.audio_data = None
        self.sample_rate = None
        self.detections_df = None  # Current file's detections
        self.current_detection_idx = 0
        self.all_detections = {}  # filepath -> DataFrame
        self.detection_sources = {}  # filepath -> 'dsp' | 'ml' | 'detections' (tracks CSV origin)
        self.dsp_queue = []  # Files queued for DSP detection
        self.dsp_queue = []  # Files queued for DSP detection

        # Playback state
        self.is_playing = False
        self.playback_speed = 1.0
        self.use_heterodyne = False
        self._playback_start_time = None  # Wall-clock time when playback started
        self._playback_start_s = None     # Audio time (seconds) where playback begins
        self._playback_end_s = None       # Audio time (seconds) where playback ends
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(30)  # ~33 fps update
        self._playback_timer.timeout.connect(self._update_playback_position)

        # Undo stack (stores (action, data) tuples)
        self.undo_stack = deque(maxlen=50)

        # Filter state
        self.filter_status = 'all'  # 'all', 'pending', 'accepted', 'rejected'

        # Workers
        self.dsp_worker = None

        self._setup_ui()
        self._apply_styles()
        self._setup_shortcuts()
        self._setup_pan_timers()
        self._restore_settings()

    def _setup_ui(self):
        """Setup the main UI."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left panel (scrollable - vertical only)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFixedWidth(380)  # Fixed width to prevent horizontal scroll

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)

        # Build sections
        self._create_input_section(left_layout)
        self._create_dsp_section(left_layout)
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

        # Scrollbar row (full width, matching spectrogram)
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

        # Controls row (window/zoom, freq, colormap, playback)
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
        self.spin_display_max_freq.setToolTip("Maximum frequency displayed on the spectrogram (Hz).\nPrairie vole USVs are typically 30-110 kHz.")
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
        self.combo_colormap.setToolTip("Spectrogram color scheme.\nViridis (default) and magma work well for USV visualization.\nGrayscale can be helpful for publications.")
        self.combo_colormap.currentTextChanged.connect(self.on_colormap_changed)
        controls_layout.addWidget(self.combo_colormap)

        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep3)

        # Playback controls (moved from left panel)
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
        # Speed slider: positions map to [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        self._speed_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(0, len(self._speed_values) - 1)
        self.slider_speed.setValue(5)  # Default 1.0x
        self.slider_speed.setFixedWidth(100)
        self.slider_speed.setToolTip("Playback speed multiplier.\nLower values slow audio down, shifting\nultrasonic frequencies into the audible range.\n0.1x is a good default for ~50 kHz USV calls.")
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
        self.status_bar.showMessage("Welcome to Classic Audio Detector - Load audio files to begin")

    def _create_input_section(self, layout):
        """Create input/file selection section."""
        group = QGroupBox("1. Input")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.setToolTip("Add all WAV files from a folder")
        self.btn_add_folder.clicked.connect(self.add_folder)
        btn_row.addWidget(self.btn_add_folder)

        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.setToolTip("Select individual WAV files to add")
        self.btn_add_files.clicked.connect(self.add_files)
        btn_row.addWidget(self.btn_add_files)

        self.btn_clear_files = QPushButton("Clear")
        self.btn_clear_files.setStyleSheet("background-color: #5c5c5c;")
        self.btn_clear_files.setToolTip("Remove all loaded files")
        self.btn_clear_files.clicked.connect(self.clear_files)
        btn_row.addWidget(self.btn_clear_files)

        group_layout.addLayout(btn_row)

        # File list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(100)
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        group_layout.addWidget(self.file_list)

        # File navigation
        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)

        self.btn_prev_file = QPushButton("< Prev")
        self.btn_prev_file.setObjectName("small_btn")
        self.btn_prev_file.setToolTip("Load the previous file in the list")
        self.btn_prev_file.clicked.connect(self.prev_file)
        self.btn_prev_file.setEnabled(False)
        nav_row.addWidget(self.btn_prev_file)

        self.lbl_file_num = QLabel("File 0/0")
        self.lbl_file_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_file_num, 1)

        self.btn_next_file = QPushButton("Next >")
        self.btn_next_file.setObjectName("small_btn")
        self.btn_next_file.setToolTip("Load the next file in the list")
        self.btn_next_file.clicked.connect(self.next_file)
        self.btn_next_file.setEnabled(False)
        nav_row.addWidget(self.btn_next_file)

        group_layout.addLayout(nav_row)

        # Open Folder button
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.setStyleSheet("background-color: #5c5c5c;")
        self.btn_open_folder.setToolTip("Open the folder containing the current file\nin the system file browser.")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setEnabled(False)
        group_layout.addWidget(self.btn_open_folder)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _make_label(self, text, tooltip=None, min_width=0):
        """Helper to create a label with optional tooltip and minimum width."""
        lbl = QLabel(text)
        if tooltip:
            lbl.setToolTip(tooltip)
        if min_width > 0:
            lbl.setMinimumWidth(min_width)
        return lbl

    def _create_dsp_section(self, layout):
        """Create DSP detection section."""
        group = QGroupBox("2. DSP Detection")
        group.setToolTip("Configure and run the DSP-based ultrasonic vocalization detector")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Species profile dropdown
        profile_row = QHBoxLayout()
        profile_row.setSpacing(4)
        profile_row.addWidget(self._make_label("Profile:", "Select a species profile to auto-fill\nDSP parameters, or Manual to set them yourself.", min_width=90))
        self.combo_species_profile = QComboBox()
        for name in self.SPECIES_PROFILES.keys():
            self.combo_species_profile.addItem(name)
        # Load custom profiles from disk
        self._custom_profiles = self._load_custom_profiles()
        for name in sorted(self._custom_profiles.keys()):
            self.combo_species_profile.addItem(name)
        self.combo_species_profile.setCurrentText("Manual")
        self.combo_species_profile.setToolTip("Species profile presets.\nSelecting a profile auto-fills all DSP parameters\nand locks them. Choose Manual to customize.")
        self.combo_species_profile.currentTextChanged.connect(self._on_species_profile_changed)
        profile_row.addWidget(self.combo_species_profile, 1)

        self.btn_save_profile = QPushButton("Save")
        self.btn_save_profile.setMaximumWidth(50)
        self.btn_save_profile.setToolTip("Save current settings as a custom profile")
        self.btn_save_profile.clicked.connect(self._save_custom_profile)
        profile_row.addWidget(self.btn_save_profile)

        self.btn_delete_profile = QPushButton("✕")
        self.btn_delete_profile.setMaximumWidth(26)
        self.btn_delete_profile.setToolTip("Delete the currently selected custom profile")
        self.btn_delete_profile.clicked.connect(self._delete_custom_profile)
        self.btn_delete_profile.setEnabled(False)
        profile_row.addWidget(self.btn_delete_profile)

        group_layout.addLayout(profile_row)

        # Frequency range
        freq_tip = ("Bandpass filter range for detection.\n"
                     "Only energy within this band is analyzed.\n"
                     "Prairie voles: typically 25-65 kHz.\n"
                     "Mice: typically 30-110 kHz.")
        freq_row = QHBoxLayout()
        freq_row.setSpacing(4)
        freq_row.addWidget(self._make_label("Freq Range:", freq_tip, min_width=90))

        self.spin_min_freq = QSpinBox()
        self.spin_min_freq.setRange(1000, 150000)
        self.spin_min_freq.setSingleStep(1000)
        self.spin_min_freq.setValue(20000)
        self.spin_min_freq.setSuffix(" Hz")
        self.spin_min_freq.setToolTip("Minimum frequency of the detection bandpass filter (Hz)")
        freq_row.addWidget(self.spin_min_freq, 1)

        freq_row.addWidget(QLabel("-"))

        self.spin_max_freq = QSpinBox()
        self.spin_max_freq.setRange(1000, 150000)
        self.spin_max_freq.setSingleStep(1000)
        self.spin_max_freq.setValue(65000)
        self.spin_max_freq.setSuffix(" Hz")
        self.spin_max_freq.setToolTip("Maximum frequency of the detection bandpass filter (Hz)")
        freq_row.addWidget(self.spin_max_freq, 1)

        group_layout.addLayout(freq_row)

        # Threshold
        thresh_tip = ("Energy threshold above the noise floor (dB).\n"
                      "Lower = more sensitive (more detections, more noise).\n"
                      "Higher = more conservative (fewer false positives).\n"
                      "Typical range: 6-15 dB.")
        thresh_row = QHBoxLayout()
        thresh_row.setSpacing(4)
        thresh_row.addWidget(self._make_label("Threshold:", thresh_tip, min_width=90))

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(1.0, 30.0)
        self.spin_threshold.setSingleStep(0.5)
        self.spin_threshold.setValue(10.0)
        self.spin_threshold.setSuffix(" dB")
        self.spin_threshold.setToolTip(thresh_tip)
        thresh_row.addWidget(self.spin_threshold)
        thresh_row.addStretch()

        group_layout.addLayout(thresh_row)

        # Duration
        dur_tip = ("Min/max call duration filter.\n"
                   "Detections outside this range are discarded.\n"
                   "Most USV calls are 5-200 ms.")
        dur_row = QHBoxLayout()
        dur_row.setSpacing(4)
        dur_row.addWidget(self._make_label("Duration:", dur_tip, min_width=90))

        self.spin_min_dur = QDoubleSpinBox()
        self.spin_min_dur.setRange(1.0, 100.0)
        self.spin_min_dur.setValue(10.0)
        self.spin_min_dur.setSuffix(" ms")
        self.spin_min_dur.setToolTip("Minimum call duration — shorter events are discarded (ms)")
        dur_row.addWidget(self.spin_min_dur, 1)

        dur_row.addWidget(QLabel("-"))

        self.spin_max_dur = QDoubleSpinBox()
        self.spin_max_dur.setRange(10.0, 5000.0)
        self.spin_max_dur.setValue(1000.0)
        self.spin_max_dur.setSuffix(" ms")
        self.spin_max_dur.setToolTip("Maximum call duration — longer events are discarded (ms)")
        dur_row.addWidget(self.spin_max_dur, 1)

        group_layout.addLayout(dur_row)

        # --- Noise Rejection Filters ---
        # Max bandwidth
        bw_tip = ("Maximum frequency bandwidth for a detection (Hz).\n"
                  "Detections spanning a wider frequency range are\n"
                  "discarded as broadband noise (cage bumps, movement).\n"
                  "Real USV calls are narrowband (typically <15 kHz).\n"
                  "Set to 0 to disable this filter.")
        bw_row = QHBoxLayout()
        bw_row.setSpacing(4)
        bw_row.addWidget(self._make_label("Max Bandwidth:", bw_tip, min_width=90))
        self.spin_max_bw = QSpinBox()
        self.spin_max_bw.setRange(0, 100000)
        self.spin_max_bw.setSingleStep(1000)
        self.spin_max_bw.setValue(20000)
        self.spin_max_bw.setSuffix(" Hz")
        self.spin_max_bw.setToolTip(bw_tip)
        bw_row.addWidget(self.spin_max_bw, 1)
        group_layout.addLayout(bw_row)

        # --- Advanced Options (collapsible) ---
        self.btn_advanced_toggle = QPushButton("▶ Advanced Options")
        self.btn_advanced_toggle.setFlat(True)
        self.btn_advanced_toggle.setStyleSheet("text-align: left; color: #aaaaaa; font-size: 11px; padding: 2px 0px;")
        self.btn_advanced_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_advanced_toggle.clicked.connect(self._toggle_advanced_options)
        group_layout.addWidget(self.btn_advanced_toggle)

        self.advanced_options_widget = QWidget()
        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(4)

        # Tonality (spectral purity)
        ton_tip = ("Minimum spectral purity score (0.0 - 1.0).\n"
                   "Measures how tonal vs. noisy a detection is.\n"
                   "1.0 = pure tone (energy at one frequency).\n"
                   "0.0 = broadband noise (energy spread everywhere).\n"
                   "Real USV calls are tonal (typically >0.3).\n"
                   "Higher = stricter, fewer false positives.\n"
                   "Set to 0 to disable this filter.")
        ton_row = QHBoxLayout()
        ton_row.setSpacing(4)
        ton_row.addWidget(self._make_label("Tonality:", ton_tip, min_width=90))
        self.spin_tonality = QDoubleSpinBox()
        self.spin_tonality.setRange(0.0, 1.0)
        self.spin_tonality.setSingleStep(0.05)
        self.spin_tonality.setDecimals(2)
        self.spin_tonality.setValue(0.30)
        self.spin_tonality.setToolTip(ton_tip)
        ton_row.addWidget(self.spin_tonality, 1)
        advanced_layout.addLayout(ton_row)

        # Min call frequency
        mcf_tip = ("Minimum actual frequency of a detection (Hz).\n"
                   "Detections whose lowest frequency falls below\n"
                   "this are discarded as likely broadband noise.\n"
                   "Broadband artifacts often extend down to the\n"
                   "detection floor, while real USVs start higher.\n"
                   "Set to 0 to disable this filter.")
        mcf_row = QHBoxLayout()
        mcf_row.setSpacing(4)
        mcf_row.addWidget(self._make_label("Min Call Freq:", mcf_tip, min_width=90))
        self.spin_min_call_freq = QSpinBox()
        self.spin_min_call_freq.setRange(0, 100000)
        self.spin_min_call_freq.setSingleStep(1000)
        self.spin_min_call_freq.setValue(0)
        self.spin_min_call_freq.setSuffix(" Hz")
        self.spin_min_call_freq.setToolTip(mcf_tip)
        mcf_row.addWidget(self.spin_min_call_freq, 1)
        advanced_layout.addLayout(mcf_row)

        # Harmonic filter checkbox
        harmonic_tip = ("Merge temporally overlapping detections that are\n"
                        "likely harmonics of the same call.\n"
                        "When enabled, if two detections overlap by >=80%\n"
                        "in time, only the lowest-frequency one (the\n"
                        "fundamental) is kept and the harmonic is discarded.\n"
                        "Recommended for species that produce strong harmonics.")
        self.chk_harmonic_filter = QCheckBox("Harmonic Filter")
        self.chk_harmonic_filter.setChecked(True)
        self.chk_harmonic_filter.setToolTip(harmonic_tip)
        advanced_layout.addWidget(self.chk_harmonic_filter)

        # Harmonic labeling checkbox
        harmonic_label_tip = ("Label harmonics instead of removing them.\n"
                              "When enabled, detections whose peak frequency\n"
                              "is ~2x or 3x another overlapping detection are\n"
                              "marked as 'harmonic' (yellow boxes) rather than\n"
                              "discarded. They are excluded from call counts\n"
                              "but preserved in the output CSV.\n\n"
                              "This runs after the Harmonic Filter above,\n"
                              "so both can be used together or independently.")
        self.chk_harmonic_label = QCheckBox("Label Harmonics")
        self.chk_harmonic_label.setChecked(False)
        self.chk_harmonic_label.setToolTip(harmonic_label_tip)
        advanced_layout.addWidget(self.chk_harmonic_label)

        # Frequency gap splitting
        freq_gap_tip = ("Minimum vertical frequency gap (Hz) to split\n"
                        "a single connected detection into two.\n"
                        "When a fundamental and harmonic are connected\n"
                        "by noise, this splits them so the harmonic\n"
                        "filter can remove the upper one.\n"
                        "Set to 0 to disable. 5000 Hz works well for\n"
                        "prairie vole USVs.")
        freq_gap_row = QHBoxLayout()
        freq_gap_row.setSpacing(4)
        freq_gap_row.addWidget(self._make_label("Freq Gap:", freq_gap_tip, min_width=90))
        self.spin_freq_gap = QSpinBox()
        self.spin_freq_gap.setRange(0, 30000)
        self.spin_freq_gap.setSingleStep(1000)
        self.spin_freq_gap.setValue(5000)
        self.spin_freq_gap.setSuffix(" Hz")
        self.spin_freq_gap.setToolTip(freq_gap_tip)
        freq_gap_row.addWidget(self.spin_freq_gap, 1)
        advanced_layout.addLayout(freq_gap_row)

        # Min gap
        gap_tip = ("Minimum silent gap between calls (ms).\n"
                   "Calls closer together than this are merged\n"
                   "into a single detection.")
        gap_row = QHBoxLayout()
        gap_row.setSpacing(4)
        gap_row.addWidget(self._make_label("Min Gap:", gap_tip, min_width=90))
        self.spin_min_gap = QDoubleSpinBox()
        self.spin_min_gap.setRange(0.0, 100.0)
        self.spin_min_gap.setValue(5.0)
        self.spin_min_gap.setSuffix(" ms")
        self.spin_min_gap.setToolTip(gap_tip)
        gap_row.addWidget(self.spin_min_gap, 1)
        advanced_layout.addLayout(gap_row)

        # --- Advanced Noise Rejection ---
        # Min bandwidth
        min_bw_tip = ("Minimum frequency bandwidth for a detection (Hz).\n"
                      "Very narrow detections (< 500 Hz) are often\n"
                      "electrical noise spikes or single-freq artifacts.\n"
                      "Set to 0 to disable this filter.")
        min_bw_row = QHBoxLayout()
        min_bw_row.setSpacing(4)
        min_bw_row.addWidget(self._make_label("Min Bandwidth:", min_bw_tip, min_width=90))
        self.spin_min_bw = QSpinBox()
        self.spin_min_bw.setRange(0, 50000)
        self.spin_min_bw.setSingleStep(100)
        self.spin_min_bw.setValue(0)
        self.spin_min_bw.setSuffix(" Hz")
        self.spin_min_bw.setToolTip(min_bw_tip)
        min_bw_row.addWidget(self.spin_min_bw, 1)
        advanced_layout.addLayout(min_bw_row)

        # Min SNR
        snr_tip = ("Minimum signal-to-noise ratio (dB).\n"
                   "SNR = peak power minus local noise floor.\n"
                   "Higher = keep only strong, clear calls.\n"
                   "Set to 0 to disable this filter.")
        snr_row = QHBoxLayout()
        snr_row.setSpacing(4)
        snr_row.addWidget(self._make_label("Min SNR:", snr_tip, min_width=90))
        self.spin_min_snr = QDoubleSpinBox()
        self.spin_min_snr.setRange(0.0, 50.0)
        self.spin_min_snr.setSingleStep(1.0)
        self.spin_min_snr.setValue(0.0)
        self.spin_min_snr.setSuffix(" dB")
        self.spin_min_snr.setToolTip(snr_tip)
        snr_row.addWidget(self.spin_min_snr, 1)
        advanced_layout.addLayout(snr_row)

        # Spectral entropy range
        ent_tip = ("Spectral entropy range (0.0 = pure tone, 1.0 = noise).\n"
                   "Min: reject too-pure artifacts (electrical tones).\n"
                   "Max: reject broadband noise events.\n"
                   "Real USV calls typically 0.1 - 0.6.\n"
                   "Set either to 0 to disable that bound.")
        ent_row = QHBoxLayout()
        ent_row.setSpacing(4)
        ent_row.addWidget(self._make_label("Entropy:", ent_tip, min_width=90))
        self.spin_min_entropy = QDoubleSpinBox()
        self.spin_min_entropy.setRange(0.0, 1.0)
        self.spin_min_entropy.setSingleStep(0.05)
        self.spin_min_entropy.setDecimals(2)
        self.spin_min_entropy.setValue(0.0)
        self.spin_min_entropy.setToolTip("Min spectral entropy (0=off)")
        ent_row.addWidget(self.spin_min_entropy, 1)
        ent_row.addWidget(QLabel("-"))
        self.spin_max_entropy = QDoubleSpinBox()
        self.spin_max_entropy.setRange(0.0, 1.0)
        self.spin_max_entropy.setSingleStep(0.05)
        self.spin_max_entropy.setDecimals(2)
        self.spin_max_entropy.setValue(0.0)
        self.spin_max_entropy.setToolTip("Max spectral entropy (0=off)")
        ent_row.addWidget(self.spin_max_entropy, 1)
        advanced_layout.addLayout(ent_row)

        # Power range
        pwr_tip = ("Power thresholds (dB).\n"
                   "Min: reject weak/quiet detections.\n"
                   "Max: reject clipping artifacts.\n"
                   "Set to 0 to disable.")
        pwr_row = QHBoxLayout()
        pwr_row.setSpacing(4)
        pwr_row.addWidget(self._make_label("Power:", pwr_tip, min_width=90))
        self.spin_min_power = QDoubleSpinBox()
        self.spin_min_power.setRange(-120.0, 0.0)
        self.spin_min_power.setSingleStep(1.0)
        self.spin_min_power.setValue(0.0)
        self.spin_min_power.setSuffix(" dB")
        self.spin_min_power.setToolTip("Min mean power (0=off)")
        pwr_row.addWidget(self.spin_min_power, 1)
        pwr_row.addWidget(QLabel("-"))
        self.spin_max_power = QDoubleSpinBox()
        self.spin_max_power.setRange(-120.0, 0.0)
        self.spin_max_power.setSingleStep(1.0)
        self.spin_max_power.setValue(0.0)
        self.spin_max_power.setSuffix(" dB")
        self.spin_max_power.setToolTip("Max peak power (0=off)")
        pwr_row.addWidget(self.spin_max_power, 1)
        advanced_layout.addLayout(pwr_row)

        # Max sweep rate
        sweep_tip = ("Maximum mean frequency sweep rate (kHz/ms).\n"
                     "Computed from peak frequency trajectory.\n"
                     "Very high rates may indicate noise artifacts.\n"
                     "Requires Freq Samples to be enabled.\n"
                     "Set to 0 to disable.")
        sweep_row = QHBoxLayout()
        sweep_row.setSpacing(4)
        sweep_row.addWidget(self._make_label("Max Sweep:", sweep_tip, min_width=90))
        self.spin_max_sweep = QDoubleSpinBox()
        self.spin_max_sweep.setRange(0.0, 100.0)
        self.spin_max_sweep.setSingleStep(0.5)
        self.spin_max_sweep.setDecimals(2)
        self.spin_max_sweep.setValue(0.0)
        self.spin_max_sweep.setSuffix(" kHz/ms")
        self.spin_max_sweep.setToolTip(sweep_tip)
        sweep_row.addWidget(self.spin_max_sweep, 1)
        advanced_layout.addLayout(sweep_row)

        # Max contour jitter
        jitter_tip = ("Maximum frequency contour jitter (kHz).\n"
                      "Measures how erratic the peak frequency trajectory is.\n"
                      "Real USV calls have smooth contours (low jitter).\n"
                      "Noise artifacts have erratic contours (high jitter).\n"
                      "Requires Freq Samples to be enabled.\n"
                      "Set to 0 to disable.")
        jitter_row = QHBoxLayout()
        jitter_row.setSpacing(4)
        jitter_row.addWidget(self._make_label("Max Jitter:", jitter_tip, min_width=90))
        self.spin_max_jitter = QDoubleSpinBox()
        self.spin_max_jitter.setRange(0.0, 50.0)
        self.spin_max_jitter.setSingleStep(0.5)
        self.spin_max_jitter.setDecimals(2)
        self.spin_max_jitter.setValue(0.0)
        self.spin_max_jitter.setSuffix(" kHz")
        self.spin_max_jitter.setToolTip(jitter_tip)
        jitter_row.addWidget(self.spin_max_jitter, 1)
        advanced_layout.addLayout(jitter_row)

        # Min ICI
        ici_tip = ("Minimum inter-call interval (ms).\n"
                   "Reject calls in suspiciously regular trains.\n"
                   "Some noise sources (e.g. 60Hz harmonics) produce\n"
                   "detections at regular short intervals.\n"
                   "Set to 0 to disable.")
        ici_row = QHBoxLayout()
        ici_row.setSpacing(4)
        ici_row.addWidget(self._make_label("Min ICI:", ici_tip, min_width=90))
        self.spin_min_ici = QDoubleSpinBox()
        self.spin_min_ici.setRange(0.0, 100.0)
        self.spin_min_ici.setSingleStep(1.0)
        self.spin_min_ici.setValue(0.0)
        self.spin_min_ici.setSuffix(" ms")
        self.spin_min_ici.setToolTip(ici_tip)
        ici_row.addWidget(self.spin_min_ici, 1)
        advanced_layout.addLayout(ici_row)

        # Noise percentile
        noise_tip = ("Percentile of spectrogram power used to\n"
                     "estimate the background noise floor.\n"
                     "Lower = assumes quieter background.\n"
                     "Typical: 20-30.")
        noise_row = QHBoxLayout()
        noise_row.setSpacing(4)
        noise_row.addWidget(self._make_label("Noise %tile:", noise_tip, min_width=90))
        self.spin_noise_pct = QDoubleSpinBox()
        self.spin_noise_pct.setRange(1.0, 50.0)
        self.spin_noise_pct.setValue(25.0)
        self.spin_noise_pct.setToolTip(noise_tip)
        noise_row.addWidget(self.spin_noise_pct, 1)
        advanced_layout.addLayout(noise_row)

        # FFT params
        fft_tip = ("FFT window size (samples). Larger = better\n"
                   "frequency resolution but worse time resolution.")
        overlap_tip = ("FFT overlap (samples). Higher overlap gives\n"
                       "smoother spectrograms but costs more compute.\n"
                       "Typical: 75% of FFT size.")
        fft_row = QHBoxLayout()
        fft_row.setSpacing(4)
        fft_row.addWidget(self._make_label("FFT:", fft_tip, min_width=34))
        self.spin_nperseg = QSpinBox()
        self.spin_nperseg.setRange(64, 2048)
        self.spin_nperseg.setSingleStep(64)
        self.spin_nperseg.setValue(512)
        self.spin_nperseg.setToolTip(fft_tip)
        fft_row.addWidget(self.spin_nperseg, 1)
        fft_row.addWidget(self._make_label("Overlap:", overlap_tip, min_width=50))
        self.spin_noverlap = QSpinBox()
        self.spin_noverlap.setRange(0, 1024)
        self.spin_noverlap.setSingleStep(64)
        self.spin_noverlap.setValue(384)
        self.spin_noverlap.setToolTip(overlap_tip)
        fft_row.addWidget(self.spin_noverlap, 1)
        advanced_layout.addLayout(fft_row)

        # Frequency samples (optional)
        freq_samp_tip = ("Sample peak frequency at N evenly-spaced\n"
                         "time points across each call. Enables\n"
                         "frequency contour visualization on accepted\n"
                         "detections (cyan line overlay).")
        freq_samp_row = QHBoxLayout()
        freq_samp_row.setSpacing(4)
        self.chk_freq_samples = QCheckBox("Freq Samples:")
        self.chk_freq_samples.setChecked(False)
        self.chk_freq_samples.setToolTip(freq_samp_tip)
        self.chk_freq_samples.toggled.connect(lambda checked: self.spin_freq_samples.setEnabled(checked))
        freq_samp_row.addWidget(self.chk_freq_samples)
        self.spin_freq_samples = QSpinBox()
        self.spin_freq_samples.setRange(3, 10)
        self.spin_freq_samples.setValue(5)
        self.spin_freq_samples.setEnabled(False)
        self.spin_freq_samples.setToolTip(freq_samp_tip)
        self.spin_freq_samples.setFixedWidth(60)
        freq_samp_row.addWidget(self.spin_freq_samples)
        freq_samp_row.addStretch()
        advanced_layout.addLayout(freq_samp_row)

        self.advanced_options_widget.setLayout(advanced_layout)
        self.advanced_options_widget.setVisible(False)
        group_layout.addWidget(self.advanced_options_widget)

        # GPU acceleration checkbox
        gpu_row = QHBoxLayout()
        gpu_row.setSpacing(4)
        self.chk_gpu_accel = QCheckBox("Enable GPU Acceleration")
        self.chk_gpu_accel.setChecked(False)
        self.chk_gpu_accel.setToolTip(
            "Use GPU for spectrogram computation (FFT).\n"
            "Supports NVIDIA CUDA and Apple Silicon MPS.\n"
            "Falls back to CPU if no compatible GPU found."
        )
        self.chk_gpu_accel.toggled.connect(self._on_gpu_toggle)
        gpu_row.addWidget(self.chk_gpu_accel)
        self.lbl_gpu_status = QLabel("")
        self.lbl_gpu_status.setStyleSheet("color: #999999; font-size: 9px;")
        gpu_row.addWidget(self.lbl_gpu_status)
        gpu_row.addStretch()
        group_layout.addLayout(gpu_row)
        self._selected_gpu_device = "auto"

        # Collect all DSP parameter widgets for profile enable/disable
        self._dsp_param_widgets = [
            self.spin_min_freq, self.spin_max_freq,
            self.spin_threshold,
            self.spin_min_dur, self.spin_max_dur,
            self.spin_max_bw, self.spin_tonality, self.spin_min_call_freq,
            self.chk_harmonic_filter, self.chk_harmonic_label,
            self.spin_freq_gap,
            self.spin_min_gap, self.spin_noise_pct,
            self.spin_nperseg, self.spin_noverlap,
            self.chk_freq_samples, self.spin_freq_samples,
            # Advanced noise rejection
            self.spin_min_bw, self.spin_min_snr,
            self.spin_min_entropy, self.spin_max_entropy,
            self.spin_min_power, self.spin_max_power,
            self.spin_max_sweep, self.spin_max_jitter, self.spin_min_ici,
        ]

        # Queue display
        self.lbl_queue = QLabel("Queue: 0 files")
        self.lbl_queue.setStyleSheet("color: #999999;")
        group_layout.addWidget(self.lbl_queue)

        # Queue buttons - row 1
        queue_row = QHBoxLayout()
        queue_row.setSpacing(2)

        self.btn_add_to_queue = QPushButton("Add Current")
        self.btn_add_to_queue.setToolTip("Add the currently selected file to the detection queue")
        self.btn_add_to_queue.clicked.connect(self.add_to_queue)
        self.btn_add_to_queue.setEnabled(False)
        queue_row.addWidget(self.btn_add_to_queue)

        self.btn_add_all_to_queue = QPushButton("Add All")
        self.btn_add_all_to_queue.setToolTip("Add all imported files to the detection queue")
        self.btn_add_all_to_queue.clicked.connect(self.add_all_to_queue)
        self.btn_add_all_to_queue.setEnabled(False)
        queue_row.addWidget(self.btn_add_all_to_queue)

        self.btn_clear_queue = QPushButton("Clear")
        self.btn_clear_queue.setStyleSheet("background-color: #5c5c5c;")
        self.btn_clear_queue.setToolTip("Remove all files from the detection queue")
        self.btn_clear_queue.clicked.connect(self.clear_queue)
        queue_row.addWidget(self.btn_clear_queue)

        group_layout.addLayout(queue_row)

        # Run button
        self.btn_run_dsp = QPushButton("Run DSP Detection")
        self.btn_run_dsp.setStyleSheet("background-color: #0078d4;")
        self.btn_run_dsp.setToolTip("Run DSP-based detection on all queued files")
        self.btn_run_dsp.clicked.connect(self.run_dsp_detection)
        self.btn_run_dsp.setEnabled(False)
        group_layout.addWidget(self.btn_run_dsp)

        # Process Preview Snapshot button
        self.btn_preview_snapshot = QPushButton("Process Preview Snapshot")
        self.btn_preview_snapshot.setStyleSheet("background-color: #2d7d46;")
        self.btn_preview_snapshot.setToolTip(
            "Run DSP detection on only the currently visible time range.\n"
            "Uses the current parameter settings. Results replace any\n"
            "existing detections for this file."
        )
        self.btn_preview_snapshot.clicked.connect(self.run_preview_snapshot)
        self.btn_preview_snapshot.setEnabled(False)
        group_layout.addWidget(self.btn_preview_snapshot)

        # Batch progress (tracks files completed)
        self.dsp_progress = QProgressBar()
        self.dsp_progress.setValue(0)
        self.dsp_progress.setVisible(False)
        self.dsp_progress.setFormat("Batch: %p%")
        group_layout.addWidget(self.dsp_progress)

        # Per-file progress (tracks chunks within current file)
        self.dsp_file_progress = QProgressBar()
        self.dsp_file_progress.setValue(0)
        self.dsp_file_progress.setVisible(False)
        self.dsp_file_progress.setFormat("File: %p%")
        self.dsp_file_progress.setMaximumHeight(12)
        group_layout.addWidget(self.dsp_file_progress)

        self.btn_stop_dsp = QPushButton("Stop Detection")
        self.btn_stop_dsp.setStyleSheet("background-color: #d13438;")
        self.btn_stop_dsp.setToolTip("Stop DSP detection after the current file finishes.\nResults for the file being processed will be discarded.")
        self.btn_stop_dsp.clicked.connect(self.stop_dsp_detection)
        self.btn_stop_dsp.setVisible(False)
        group_layout.addWidget(self.btn_stop_dsp)

        self.lbl_dsp_status = QLabel("")
        self.lbl_dsp_status.setStyleSheet("color: #999999; font-size: 9px;")
        group_layout.addWidget(self.lbl_dsp_status)

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
        self.combo_filter.setToolTip("Filter which detections to navigate through")
        self.combo_filter.currentIndexChanged.connect(self.on_filter_changed)
        filter_row.addWidget(self.combo_filter, 1)

        # Jump-to spinner
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
        self.spin_start.setToolTip("Start time of the selected detection (seconds).\nEdit to adjust the left boundary of the detection box.")
        self.spin_start.valueChanged.connect(self.on_time_changed)
        time_row.addWidget(self.spin_start, 1)

        time_row.addWidget(QLabel("Stop:"))
        self.spin_stop = QDoubleSpinBox()
        self.spin_stop.setDecimals(4)
        self.spin_stop.setRange(0, 9999)
        self.spin_stop.setSuffix(" s")
        self.spin_stop.setToolTip("End time of the selected detection (seconds).\nEdit to adjust the right boundary of the detection box.")
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
        self.spin_det_min_freq.setToolTip("Minimum frequency of the selected detection (Hz).\nEdit to adjust the bottom boundary of the detection box.")
        self.spin_det_min_freq.valueChanged.connect(self.on_freq_changed)
        freq_row.addWidget(self.spin_det_min_freq, 1)

        freq_row.addWidget(QLabel("Max:"))
        self.spin_det_max_freq = QSpinBox()
        self.spin_det_max_freq.setRange(0, 200000)
        self.spin_det_max_freq.setSingleStep(1000)
        self.spin_det_max_freq.setSuffix(" Hz")
        self.spin_det_max_freq.setToolTip("Maximum frequency of the selected detection (Hz).\nEdit to adjust the top boundary of the detection box.")
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

        # Progress (moved from section 6)
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
        self.btn_accept.setToolTip("Mark current detection as a valid USV call (A key).\nAccepted calls are saved and used for ML training.")
        self.btn_accept.clicked.connect(self.accept_detection)
        self.btn_accept.setEnabled(False)
        btn_row1.addWidget(self.btn_accept)

        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setObjectName("reject_btn")
        self.btn_reject.setToolTip("Mark current detection as a false positive (R key).\nRejected calls are excluded from analysis but kept for ML training.")
        self.btn_reject.clicked.connect(self.reject_detection)
        self.btn_reject.setEnabled(False)
        btn_row1.addWidget(self.btn_reject)

        self.btn_skip = QPushButton("Skip")
        self.btn_skip.setStyleSheet("background-color: #5c5c5c;")
        self.btn_skip.clicked.connect(self.skip_detection)
        self.btn_skip.setEnabled(False)
        self.btn_skip.setToolTip("Skip to the next detection without changing\nits current status (S key).\nUseful for reviewing without modifying labels.")
        btn_row1.addWidget(self.btn_skip)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setStyleSheet("background-color: #5c5c5c;")
        self.btn_undo.clicked.connect(self.undo_action)
        self.btn_undo.setEnabled(False)
        self.btn_undo.setToolTip("Undo last labeling action (Ctrl+Z).\nSupports undoing single and batch operations.")
        btn_row1.addWidget(self.btn_undo)

        group_layout.addLayout(btn_row1)

        # Batch labeling
        batch_row = QHBoxLayout()
        batch_row.setSpacing(2)

        self.btn_accept_all_pending = QPushButton("Accept All Pending")
        self.btn_accept_all_pending.setObjectName("accept_btn")
        self.btn_accept_all_pending.setToolTip("Accept all unreviewed detections at once.\nUseful after reviewing and only rejecting false positives.")
        self.btn_accept_all_pending.clicked.connect(self.accept_all_pending)
        self.btn_accept_all_pending.setEnabled(False)
        batch_row.addWidget(self.btn_accept_all_pending)

        self.btn_reject_all_pending = QPushButton("Reject All Pending")
        self.btn_reject_all_pending.setObjectName("reject_btn")
        self.btn_reject_all_pending.setToolTip("Reject all unreviewed detections at once.\nUseful to clear remaining detections after accepting valid calls.")
        self.btn_reject_all_pending.clicked.connect(self.reject_all_pending)
        self.btn_reject_all_pending.setEnabled(False)
        batch_row.addWidget(self.btn_reject_all_pending)

        group_layout.addLayout(batch_row)

        # Add/Delete
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(2)

        self.btn_add_usv = QPushButton("+ Add Label")
        self.btn_add_usv.setStyleSheet("background-color: #6b4c9a;")
        self.btn_add_usv.setToolTip("Manually draw a new USV detection box on the spectrogram.\nClick and drag on the spectrogram to define the region.")
        self.btn_add_usv.clicked.connect(self.add_new_usv)
        self.btn_add_usv.setEnabled(False)
        btn_row2.addWidget(self.btn_add_usv)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet("background-color: #5c5c5c;")
        self.btn_delete.setToolTip("Permanently delete the currently selected detection (D key).\nThis cannot be undone.")
        self.btn_delete.clicked.connect(self.delete_current)
        self.btn_delete.setEnabled(False)
        btn_row2.addWidget(self.btn_delete)

        group_layout.addLayout(btn_row2)

        # Delete pending / Delete all labels
        delete_row = QHBoxLayout()
        delete_row.setSpacing(2)

        self.btn_delete_pending = QPushButton("Delete Pending")
        self.btn_delete_pending.setStyleSheet("background-color: #5c5c5c;")
        self.btn_delete_pending.setToolTip("Permanently delete all unreviewed detections.\nUseful for clearing noise after accepting valid calls.")
        self.btn_delete_pending.clicked.connect(self.delete_all_pending)
        self.btn_delete_pending.setEnabled(False)
        delete_row.addWidget(self.btn_delete_pending)

        self.btn_delete_all_labels = QPushButton("Delete All Labels")
        self.btn_delete_all_labels.setStyleSheet("background-color: #8b0000;")
        self.btn_delete_all_labels.setToolTip("Delete ALL detections (pending + accepted + rejected)\nfor the current file. This cannot be undone.")
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
        lbl_hint = QLabel("Keys: A=Accept R=Reject D=Delete Space=Play Ctrl+Z=Undo")
        lbl_hint.setStyleSheet("color: #666666; font-size: 9px; font-style: italic;")
        lbl_hint.setWordWrap(True)
        group_layout.addWidget(lbl_hint)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _on_gpu_toggle(self, checked):
        """Handle GPU acceleration checkbox toggle."""
        if checked:
            self._show_gpu_detection_dialog()
        else:
            self.lbl_gpu_status.setText("")
            self._selected_gpu_device = "auto"

    def _show_gpu_detection_dialog(self):
        """Show dialog listing detected GPU devices."""
        try:
            from fnt.usv.usv_detector.gpu_utils import detect_available_devices
        except ImportError:
            QMessageBox.warning(self, "PyTorch Not Available",
                "PyTorch is required for GPU acceleration.\n\n"
                "Install with: pip install torch")
            self.chk_gpu_accel.blockSignals(True)
            self.chk_gpu_accel.setChecked(False)
            self.chk_gpu_accel.blockSignals(False)
            return

        devices = detect_available_devices()
        gpu_devices = [d for d in devices if d['type'] != 'cpu']

        if not gpu_devices:
            QMessageBox.warning(self, "No GPU Found",
                "No compatible GPU detected.\n\n"
                "Requires:\n"
                "  \u2022 NVIDIA GPU with CUDA support, or\n"
                "  \u2022 Apple Silicon with MPS support\n\n"
                "Processing will use CPU.")
            self.chk_gpu_accel.blockSignals(True)
            self.chk_gpu_accel.setChecked(False)
            self.chk_gpu_accel.blockSignals(False)
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("GPU Acceleration")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Detected compute devices:"))
        layout.addSpacing(4)

        for dev in devices:
            parts = []
            if dev['type'] != 'cpu':
                parts.append("\u2705")  # Green checkmark
            else:
                parts.append("   ")
            parts.append(f"{dev['name']}  ({dev['type'].upper()})")
            if dev.get('vram_mb'):
                parts.append(f" \u2014 {dev['vram_mb']:,} MB VRAM")
            lbl = QLabel("".join(parts))
            if dev['type'] != 'cpu':
                lbl.setStyleSheet("color: #4CAF50; font-weight: bold;")
            else:
                lbl.setStyleSheet("color: #999999;")
            layout.addWidget(lbl)

        layout.addSpacing(8)

        # Device selection
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Use device:"))
        combo = QComboBox()
        combo.addItem("Auto (best available)", "auto")
        for dev in gpu_devices:
            label = f"{dev['name']} ({dev['type'].upper()})"
            if dev.get('vram_mb'):
                label += f" \u2014 {dev['vram_mb']:,} MB"
            combo.addItem(label, dev['device'])
        sel_row.addWidget(combo, 1)
        layout.addLayout(sel_row)

        layout.addSpacing(8)

        # OK / Cancel
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            self._selected_gpu_device = combo.currentData()
            dev_name = combo.currentText()
            self.lbl_gpu_status.setText(f"\u26a1 {dev_name}")
            self.lbl_gpu_status.setStyleSheet("color: #4CAF50; font-size: 9px;")
        else:
            # User cancelled — uncheck
            self.chk_gpu_accel.blockSignals(True)
            self.chk_gpu_accel.setChecked(False)
            self.chk_gpu_accel.blockSignals(False)
            self.lbl_gpu_status.setText("")

    def _toggle_advanced_options(self):
        """Toggle visibility of advanced DSP options."""
        visible = not self.advanced_options_widget.isVisible()
        self.advanced_options_widget.setVisible(visible)
        arrow = "▼" if visible else "▶"
        self.btn_advanced_toggle.setText(f"{arrow} Advanced Options")

    def _on_species_profile_changed(self, profile_name):
        """Handle species profile dropdown change."""
        # Look up in built-in profiles first, then custom
        preset = self.SPECIES_PROFILES.get(profile_name)
        if preset is None:
            preset = self._custom_profiles.get(profile_name)

        # Enable delete button only for custom profiles
        is_custom = profile_name in self._custom_profiles
        self.btn_delete_profile.setEnabled(is_custom)

        if preset is None:
            # Manual mode — re-enable all DSP parameter widgets
            for w in self._dsp_param_widgets:
                w.setEnabled(True)
                w.setStyleSheet("")  # Clear locked styling
            # Respect freq_samples checkbox state
            self.spin_freq_samples.setEnabled(self.chk_freq_samples.isChecked())
        else:
            # Block signals on all spinboxes to avoid cascading updates
            spinboxes = [
                self.spin_min_freq, self.spin_max_freq, self.spin_threshold,
                self.spin_min_dur, self.spin_max_dur,
                self.spin_max_bw, self.spin_tonality, self.spin_min_call_freq,
                self.spin_freq_gap,
                self.spin_min_gap, self.spin_noise_pct,
                self.spin_nperseg, self.spin_noverlap,
                # Advanced noise rejection
                self.spin_min_bw, self.spin_min_snr,
                self.spin_min_entropy, self.spin_max_entropy,
                self.spin_min_power, self.spin_max_power,
                self.spin_max_sweep, self.spin_max_jitter, self.spin_min_ici,
            ]
            for s in spinboxes:
                s.blockSignals(True)

            self.spin_min_freq.setValue(preset['min_freq_hz'])
            self.spin_max_freq.setValue(preset['max_freq_hz'])
            self.spin_threshold.setValue(preset['energy_threshold_db'])
            self.spin_min_dur.setValue(preset['min_duration_ms'])
            self.spin_max_dur.setValue(preset['max_duration_ms'])
            self.spin_max_bw.setValue(preset['max_bandwidth_hz'])
            self.spin_tonality.setValue(preset['min_tonality'])
            self.spin_min_call_freq.setValue(preset['min_call_freq_hz'])
            self.spin_min_gap.setValue(preset['min_gap_ms'])
            self.spin_noise_pct.setValue(preset['noise_percentile'])
            self.spin_nperseg.setValue(preset['nperseg'])
            self.spin_noverlap.setValue(preset['noverlap'])
            self.chk_harmonic_filter.setChecked(preset.get('harmonic_filter', True))
            self.chk_harmonic_label.setChecked(preset.get('harmonic_label', False))
            self.spin_freq_gap.setValue(preset.get('min_freq_gap_hz', 5000))
            # Advanced noise rejection
            self.spin_min_bw.setValue(preset.get('min_bandwidth_hz', 0))
            self.spin_min_snr.setValue(preset.get('min_snr_db', 0.0))
            self.spin_min_entropy.setValue(preset.get('min_spectral_entropy', 0.0))
            self.spin_max_entropy.setValue(preset.get('max_spectral_entropy', 0.0))
            self.spin_min_power.setValue(preset.get('min_power_db', 0.0))
            self.spin_max_power.setValue(preset.get('max_power_db', 0.0))
            self.spin_max_sweep.setValue(preset.get('max_mean_sweep_rate', 0.0))
            self.spin_max_jitter.setValue(preset.get('max_contour_jitter', 0.0))
            self.spin_min_ici.setValue(preset.get('min_ici_ms', 0.0))

            for s in spinboxes:
                s.blockSignals(False)

            # Disable all DSP parameter widgets with strong visual indicator
            locked_style = "background-color: #1a1a1a; color: #555555;"
            for w in self._dsp_param_widgets:
                w.setEnabled(False)
                w.setStyleSheet(locked_style)


    # -------------------------------------------------------------------------
    # Custom profile management
    # -------------------------------------------------------------------------

    @staticmethod
    def _profiles_dir():
        """Return the directory for custom detection profiles."""
        d = os.path.join(os.path.expanduser("~"), ".fnt", "detection_profiles")
        os.makedirs(d, exist_ok=True)
        return d

    def _load_custom_profiles(self) -> dict:
        """Load all custom profiles from ~/.fnt/detection_profiles/."""
        profiles = {}
        profiles_dir = self._profiles_dir()
        if not os.path.isdir(profiles_dir):
            return profiles
        for fname in os.listdir(profiles_dir):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(profiles_dir, fname)) as f:
                        data = json.load(f)
                    name = data.pop('_profile_name', fname.replace('.json', ''))
                    profiles[name] = data
                except Exception:
                    continue
        return profiles

    def _save_custom_profile(self):
        """Save the current DSP settings as a custom profile."""
        # Count existing custom profiles for default name
        n = len(self._custom_profiles) + 1
        default_name = f"Custom Profile {n}"

        name, ok = QInputDialog.getText(
            self, "Save Detection Profile",
            "Profile name:",
            QLineEdit.Normal,
            default_name
        )
        if not ok or not name.strip():
            return

        name = name.strip()

        # Prevent overwriting built-in profiles
        if name in self.SPECIES_PROFILES:
            QMessageBox.warning(
                self, "Invalid Name",
                f"'{name}' is a built-in profile and cannot be overwritten.\n"
                "Please choose a different name."
            )
            return

        # Confirm overwrite if profile already exists
        if name in self._custom_profiles:
            reply = QMessageBox.question(
                self, "Overwrite Profile",
                f"Profile '{name}' already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Gather current settings
        profile_data = self._gather_dsp_config()
        profile_data['_profile_name'] = name

        # Save to disk
        safe_fname = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        filepath = os.path.join(self._profiles_dir(), f"{safe_fname}.json")
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)

        # Remove the _profile_name key from the in-memory dict
        profile_data_clean = {k: v for k, v in profile_data.items() if k != '_profile_name'}
        self._custom_profiles[name] = profile_data_clean

        # Add to dropdown if not already there
        if self.combo_species_profile.findText(name) == -1:
            self.combo_species_profile.addItem(name)

        # Select the newly saved profile
        self.combo_species_profile.setCurrentText(name)
        self.status_bar.showMessage(f"Profile '{name}' saved")

    def _delete_custom_profile(self):
        """Delete the currently selected custom profile."""
        name = self.combo_species_profile.currentText()
        if name not in self._custom_profiles:
            return

        reply = QMessageBox.question(
            self, "Delete Profile",
            f"Delete custom profile '{name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Remove from disk
        safe_fname = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        filepath = os.path.join(self._profiles_dir(), f"{safe_fname}.json")
        if os.path.exists(filepath):
            os.remove(filepath)

        # Remove from memory
        del self._custom_profiles[name]

        # Remove from dropdown and switch to Manual
        idx = self.combo_species_profile.findText(name)
        if idx >= 0:
            self.combo_species_profile.removeItem(idx)
        self.combo_species_profile.setCurrentText("Manual")
        self.status_bar.showMessage(f"Profile '{name}' deleted")

    def _setup_shortcuts(self):
        """Set up global keyboard shortcuts using QShortcut."""
        # Detection navigation: B = back, N = next
        sc_next = QShortcut(QKeySequence(Qt.Key_N), self)
        sc_next.setContext(Qt.ApplicationShortcut)
        sc_next.activated.connect(self._shortcut_next_detection)

        sc_prev = QShortcut(QKeySequence(Qt.Key_B), self)
        sc_prev.setContext(Qt.ApplicationShortcut)
        sc_prev.activated.connect(self._shortcut_prev_detection)

        # Spectrogram panning: Left/Right arrow keys
        sc_pan_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        sc_pan_right.setContext(Qt.ApplicationShortcut)
        sc_pan_right.activated.connect(self._shortcut_pan_right)

        sc_pan_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        sc_pan_left.setContext(Qt.ApplicationShortcut)
        sc_pan_left.activated.connect(self._shortcut_pan_left)

        # Window size: Up = zoom in (smaller window), Down = zoom out (larger window)
        sc_zoom_in = QShortcut(QKeySequence(Qt.Key_Up), self)
        sc_zoom_in.setContext(Qt.ApplicationShortcut)
        sc_zoom_in.activated.connect(self._shortcut_zoom_in)

        sc_zoom_out = QShortcut(QKeySequence(Qt.Key_Down), self)
        sc_zoom_out.setContext(Qt.ApplicationShortcut)
        sc_zoom_out.activated.connect(self._shortcut_zoom_out)

        # P = add new pending USV detection
        sc_add_usv = QShortcut(QKeySequence(Qt.Key_P), self)
        sc_add_usv.setContext(Qt.ApplicationShortcut)
        sc_add_usv.activated.connect(self._shortcut_add_usv)

        # S = skip detection (advance without changing status)
        sc_skip = QShortcut(QKeySequence(Qt.Key_S), self)
        sc_skip.setContext(Qt.ApplicationShortcut)
        sc_skip.activated.connect(self._shortcut_skip)

    def _shortcut_next_detection(self):
        """Handle N key shortcut — next detection."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_next_det.isEnabled():
            self._flash_button(self.btn_next_det)
            self.next_detection()

    def _shortcut_prev_detection(self):
        """Handle B key shortcut — previous detection."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_prev_det.isEnabled():
            self._flash_button(self.btn_prev_det)
            self.prev_detection()

    def _shortcut_pan_right(self):
        """Handle Right arrow — pan spectrogram right."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.pan_right()

    def _shortcut_pan_left(self):
        """Handle Left arrow — pan spectrogram left."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.pan_left()

    def _shortcut_zoom_out(self):
        """Handle Up arrow — increase window size (zoom out)."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.zoom_out()

    def _shortcut_zoom_in(self):
        """Handle Down arrow — decrease window size (zoom in)."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.zoom_in()

    def _shortcut_add_usv(self):
        """Handle P key — add new pending USV detection."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_add_usv.isEnabled():
            self._flash_button(self.btn_add_usv)
            self.add_new_usv()

    def _shortcut_skip(self):
        """Handle S key — skip to next detection without changing status."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_skip.isEnabled():
            self._flash_button(self.btn_skip)
            self.skip_detection()

    def _setup_pan_timers(self):
        """Set up press-and-hold repeat timers for pan buttons."""
        self._pan_left_timer = QTimer(self)
        self._pan_left_timer.setInterval(100)  # Repeat every 100ms while held
        self._pan_left_timer.timeout.connect(self.pan_left)

        self._pan_right_timer = QTimer(self)
        self._pan_right_timer.setInterval(100)
        self._pan_right_timer.timeout.connect(self.pan_right)

        # Connect press/release events
        self.btn_pan_left.pressed.connect(self._on_pan_left_pressed)
        self.btn_pan_left.released.connect(self._pan_left_timer.stop)
        self.btn_pan_right.pressed.connect(self._on_pan_right_pressed)
        self.btn_pan_right.released.connect(self._pan_right_timer.stop)

        # Disconnect the old clicked signals to avoid double-firing
        self.btn_pan_left.clicked.disconnect(self.pan_left)
        self.btn_pan_right.clicked.disconnect(self.pan_right)

    def _on_pan_left_pressed(self):
        """Handle pan left button press - immediate pan + start repeat timer."""
        self.pan_left()
        self._pan_left_timer.start()

    def _on_pan_right_pressed(self):
        """Handle pan right button press - immediate pan + start repeat timer."""
        self.pan_right()
        self._pan_right_timer.start()

    def _create_arrow_images(self):
        """Create small arrow PNG images for spinbox/combobox buttons."""
        import tempfile
        self._arrow_dir = tempfile.mkdtemp(prefix='usv_arrows_')

        # Up arrow (white triangle on transparent bg)
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

        # Down arrow
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
    # File Management
    # =========================================================================

    def add_folder(self):
        """Add all WAV files from a folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return

        # Use a dict to deduplicate (Windows glob is case-insensitive,
        # so *.wav and *.WAV can return the same files)
        seen = {}
        for f in list(Path(folder).glob("*.wav")) + list(Path(folder).glob("*.WAV")):
            key = str(f).lower()  # Normalize for case-insensitive FS
            if key not in seen:
                seen[key] = f
        wav_files = sorted(seen.values())
        if not wav_files:
            QMessageBox.warning(self, "No Files", "No WAV files found in folder.")
            return

        self._add_audio_files([str(f) for f in wav_files])

    def add_files(self):
        """Add individual WAV files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "WAV Files (*.wav *.WAV);;All Files (*.*)"
        )
        if not files:
            return

        self._add_audio_files(files)

    def _add_audio_files(self, new_files):
        """Add files to the import list, skipping duplicates with a warning."""
        duplicates = [f for f in new_files if f in self.audio_files]
        to_add = [f for f in new_files if f not in self.audio_files]

        if duplicates and not to_add:
            QMessageBox.information(
                self, "Already Imported",
                f"All {len(duplicates)} file{'s' if len(duplicates) != 1 else ''} "
                "are already imported. No new files added.")
            return

        if duplicates:
            self.status_bar.showMessage(
                f"Added {len(to_add)} new file{'s' if len(to_add) != 1 else ''}, "
                f"skipped {len(duplicates)} already imported")
        else:
            self.status_bar.showMessage(
                f"Added {len(to_add)} file{'s' if len(to_add) != 1 else ''}")

        had_files = len(self.audio_files) > 0
        for f in to_add:
            self.audio_files.append(f)

        # Scan new files for existing CSV detections
        self._scan_existing_csvs()

        if had_files:
            # Already have files loaded — just refresh the list display
            self._refresh_file_list_items_full()
        else:
            # First files being added — do full list build + load first file
            self._update_file_list()

    def _scan_existing_csvs(self):
        """Pre-scan for existing CSV detection files to show counts in file list."""
        for filepath in self.audio_files:
            if filepath in self.all_detections:
                continue
            base = Path(filepath).stem
            parent = Path(filepath).parent
            for suffix in ['_usv_dsp', '_usv_rf', '_usv_detections']:
                csv_path = parent / f"{base}{suffix}.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if 'status' not in df.columns:
                            df['status'] = 'pending'
                        self._ensure_freq_bounds(df)
                        self.all_detections[filepath] = df
                        self.detection_sources[filepath] = suffix.lstrip('_')
                        break
                    except Exception:
                        pass

    def clear_files(self):
        """Clear all files."""
        self.audio_files = []
        self.current_file_idx = 0
        self.audio_data = None
        self.detections_df = None
        self.all_detections = {}
        self.detection_sources = {}
        self.dsp_queue = []
        self.undo_stack.clear()
        self._update_file_list()
        self.spectrogram.set_audio_data(None, None)
        self.spectrogram.set_detections([], -1)
        self.waveform_overview.set_audio_data(None, None)
        self._update_ui_state()

    def _update_file_list(self):
        """Update file list display. Blocks signals to prevent
        _store_current_detections from overwriting freshly loaded data."""
        self.file_list.blockSignals(True)
        self.file_list.clear()

        for filepath in self.audio_files:
            filename = os.path.basename(filepath)
            # Build display text with queue indicator and detection count
            parts = []
            if filepath in self.dsp_queue:
                parts.append("\u2713")  # Checkmark for queued files
            parts.append(filename)
            if filepath in self.all_detections:
                n_det = len(self.all_detections[filepath])
                parts.append(f"({n_det})")
            display_text = " ".join(parts)
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, filepath)
            self.file_list.addItem(item)

        n = len(self.audio_files)
        self.lbl_file_num.setText(f"File {self.current_file_idx + 1 if n > 0 else 0}/{n}")
        self.btn_prev_file.setEnabled(self.current_file_idx > 0)
        self.btn_next_file.setEnabled(self.current_file_idx < n - 1)
        self.btn_add_to_queue.setEnabled(n > 0)
        self.btn_add_all_to_queue.setEnabled(n > 0)

        if n > 0 and self.current_file_idx < n:
            self.file_list.setCurrentRow(self.current_file_idx)
        self.file_list.blockSignals(False)

        if n > 0 and self.current_file_idx < n:
            self._load_current_file()

        self._update_ui_state()

    def _refresh_file_list_items_full(self):
        """Rebuild file list widget without reloading the current file.

        Used when adding more files to an already-populated list.
        """
        self.file_list.blockSignals(True)
        self.file_list.clear()

        for filepath in self.audio_files:
            filename = os.path.basename(filepath)
            parts = []
            if filepath in self.dsp_queue:
                parts.append("\u2713")
            parts.append(filename)
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
        self.btn_add_to_queue.setEnabled(n > 0)
        self.btn_add_all_to_queue.setEnabled(n > 0)

        if n > 0 and self.current_file_idx < n:
            self.file_list.setCurrentRow(self.current_file_idx)
        self.file_list.blockSignals(False)

        self._update_file_navigation()

    def on_file_selected(self, row):
        """Handle file selection from list."""
        if row >= 0 and row < len(self.audio_files):
            # Save current file's detections before switching
            self._store_current_detections()
            self.current_file_idx = row
            self._load_current_file()
            self._update_file_navigation()

    def prev_file(self):
        """Go to previous file."""
        if self.current_file_idx > 0:
            # Only change list selection — on_file_selected will handle
            # storing old detections and updating current_file_idx
            self.file_list.setCurrentRow(self.current_file_idx - 1)

    def next_file(self):
        """Go to next file."""
        if self.current_file_idx < len(self.audio_files) - 1:
            # Only change list selection — on_file_selected will handle
            # storing old detections and updating current_file_idx
            self.file_list.setCurrentRow(self.current_file_idx + 1)

    def _update_file_navigation(self):
        """Update file navigation buttons."""
        n = len(self.audio_files)
        self.lbl_file_num.setText(f"File {self.current_file_idx + 1}/{n}")
        self.btn_prev_file.setEnabled(self.current_file_idx > 0)
        self.btn_next_file.setEnabled(self.current_file_idx < n - 1)

    def _refresh_file_list_items(self):
        """Refresh file list display text without changing selection."""
        current_row = self.file_list.currentRow()
        self.file_list.blockSignals(True)

        for i, filepath in enumerate(self.audio_files):
            item = self.file_list.item(i)
            if item is None:
                continue
            filename = os.path.basename(filepath)
            # Build display text with queue indicator and detection count
            parts = []
            if filepath in self.dsp_queue:
                parts.append("\u2713")  # Checkmark for queued files
            parts.append(filename)
            if filepath in self.all_detections:
                n_det = len(self.all_detections[filepath])
                parts.append(f"({n_det})")
            item.setText(" ".join(parts))

        self.file_list.blockSignals(False)
        if current_row >= 0:
            self.file_list.setCurrentRow(current_row)

    def _load_current_file(self):
        """Load the current audio file with progress indication."""
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            return

        filepath = self.audio_files[self.current_file_idx]
        self.status_bar.showMessage(f"Loading {os.path.basename(filepath)}...")
        # Show a busy cursor during load
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            # Load audio
            if HAS_SOUNDFILE:
                try:
                    self.audio_data, self.sample_rate = sf.read(filepath, dtype='float32')
                except Exception:
                    self.audio_data, self.sample_rate = self._load_with_ffmpeg(filepath)
            else:
                self.audio_data, self.sample_rate = self._load_with_ffmpeg(filepath)

            if self.audio_data.ndim > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)

            # Preserve view settings if we already had a file loaded
            has_previous = self.spectrogram.total_duration > 0
            self.spectrogram.set_audio_data(self.audio_data, self.sample_rate,
                                            preserve_view=has_previous)
            self.waveform_overview.set_audio_data(self.audio_data, self.sample_rate)

            # Sync freq spinners with spectrogram's (possibly preserved) values
            self.spin_display_min_freq.blockSignals(True)
            self.spin_display_max_freq.blockSignals(True)
            self.spin_display_min_freq.setValue(self.spectrogram.min_freq)
            self.spin_display_max_freq.setValue(self.spectrogram.max_freq)
            self.spin_display_min_freq.blockSignals(False)
            self.spin_display_max_freq.blockSignals(False)

            # Sync the window spinner with the preserved view
            if has_previous:
                view_start, view_end = self.spectrogram.get_view_range()
                self.spin_view_window.blockSignals(True)
                self.spin_view_window.setValue(view_end - view_start)
                self.spin_view_window.blockSignals(False)

            # Load detections if available
            if filepath in self.all_detections:
                self.detections_df = self.all_detections[filepath].copy()
                # Ensure legacy data has freq bounds
                self._ensure_freq_bounds(self.detections_df)
                self.current_detection_idx = 0
            else:
                # Try to load from CSV
                self._try_load_detections(filepath)

            # Clear undo stack on file switch
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

        # Try different suffixes
        for suffix in ['_usv_dsp', '_usv_rf', '_usv_detections']:
            csv_path = parent / f"{base}{suffix}.csv"
            if csv_path.exists():
                try:
                    self.detections_df = pd.read_csv(csv_path)
                    if 'status' not in self.detections_df.columns:
                        self.detections_df['status'] = 'pending'
                    # Handle legacy CSVs missing min_freq_hz/max_freq_hz
                    self._ensure_freq_bounds(self.detections_df)
                    self.all_detections[filepath] = self.detections_df.copy()
                    # Track source suffix for correct auto-save naming
                    self.detection_sources[filepath] = suffix.lstrip('_')
                    self.current_detection_idx = 0
                    return
                except Exception:
                    pass

        self.detections_df = None
        self.current_detection_idx = 0
    @staticmethod
    def _ensure_freq_bounds(df):
        """Ensure min_freq_hz and max_freq_hz columns exist. Compute from
        mean_freq_hz and freq_bandwidth_hz if available, else use defaults."""
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
        # Fill any remaining NaNs
        df['min_freq_hz'] = df['min_freq_hz'].fillna(25000)
        df['max_freq_hz'] = df['max_freq_hz'].fillna(65000)

    # =========================================================================
    # DSP Detection
    # =========================================================================

    def _file_has_detections(self, filepath):
        """Check if a file has existing detections (in memory or on disk)."""
        if filepath in self.all_detections and len(self.all_detections[filepath]) > 0:
            return True
        # Check for CSV on disk
        base = Path(filepath).stem
        parent = Path(filepath).parent
        for suffix in ['_usv_dsp', '_usv_rf', '_usv_detections']:
            if (parent / f"{base}{suffix}.csv").exists():
                return True
        return False

    def add_to_queue(self):
        """Add current file to DSP queue."""
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            return

        filepath = self.audio_files[self.current_file_idx]
        if filepath not in self.dsp_queue:
            self.dsp_queue.append(filepath)
            self._update_queue_display()
            self.status_bar.showMessage(f"Added 1 file to queue")

    def add_all_to_queue(self):
        """Add all imported files to DSP queue."""
        if not self.audio_files:
            return

        files_to_add = [f for f in self.audio_files if f not in self.dsp_queue]
        if not files_to_add:
            self.status_bar.showMessage("All files already in queue")
            return

        for filepath in files_to_add:
            self.dsp_queue.append(filepath)

        self._update_queue_display()
        self.status_bar.showMessage(f"Added {len(files_to_add)} file{'s' if len(files_to_add) != 1 else ''} to queue")

    def clear_queue(self):
        """Clear DSP queue."""
        self.dsp_queue = []
        self._update_queue_display()

    def _update_queue_display(self):
        """Update queue display and file list checkmarks."""
        n = len(self.dsp_queue)
        self.lbl_queue.setText(f"Queue: {n} file{'s' if n != 1 else ''}")
        self.btn_run_dsp.setEnabled(n > 0)
        if n > 0:
            self.btn_run_dsp.setText(f"Run DSP Detection ({n} file{'s' if n != 1 else ''})")
        else:
            self.btn_run_dsp.setText("Run DSP Detection")
        # Update file list to show/hide checkmarks
        self._refresh_file_list_items()

    def _gather_dsp_config(self):
        """Gather current DSP parameter values from the UI into a config dict."""
        return {
            'min_freq_hz': self.spin_min_freq.value(),
            'max_freq_hz': self.spin_max_freq.value(),
            'energy_threshold_db': self.spin_threshold.value(),
            'min_duration_ms': self.spin_min_dur.value(),
            'max_duration_ms': self.spin_max_dur.value(),
            'max_bandwidth_hz': self.spin_max_bw.value(),
            'min_tonality': self.spin_tonality.value(),
            'min_call_freq_hz': self.spin_min_call_freq.value(),
            'harmonic_filter': self.chk_harmonic_filter.isChecked(),
            'harmonic_label': self.chk_harmonic_label.isChecked(),
            'min_freq_gap_hz': self.spin_freq_gap.value(),
            'min_gap_ms': self.spin_min_gap.value(),
            'noise_percentile': self.spin_noise_pct.value(),
            'nperseg': self.spin_nperseg.value(),
            'noverlap': self.spin_noverlap.value(),
            'freq_samples': self.spin_freq_samples.value() if self.chk_freq_samples.isChecked() else 0,
            'gpu_enabled': self.chk_gpu_accel.isChecked(),
            'gpu_device': getattr(self, '_selected_gpu_device', 'auto'),
            # Advanced noise rejection
            'min_bandwidth_hz': self.spin_min_bw.value(),
            'min_snr_db': self.spin_min_snr.value(),
            'min_spectral_entropy': self.spin_min_entropy.value(),
            'max_spectral_entropy': self.spin_max_entropy.value(),
            'min_power_db': self.spin_min_power.value(),
            'max_power_db': self.spin_max_power.value(),
            'max_mean_sweep_rate': self.spin_max_sweep.value(),
            'max_contour_jitter': self.spin_max_jitter.value(),
            'min_ici_ms': self.spin_min_ici.value(),
        }

    def run_preview_snapshot(self):
        """Run DSP detection on only the currently visible preview window."""
        if self.audio_data is None or self.sample_rate is None:
            self.status_bar.showMessage("No audio loaded")
            return

        filepath = self.audio_files[self.current_file_idx]
        view_start, view_end = self.spectrogram.get_view_range()

        # Extract the audio segment for the visible window
        start_sample = int(view_start * self.sample_rate)
        end_sample = int(view_end * self.sample_rate)
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_data), end_sample)
        audio_segment = self.audio_data[start_sample:end_sample]

        if len(audio_segment) == 0:
            self.status_bar.showMessage("Preview window is empty")
            return

        self.btn_preview_snapshot.setEnabled(False)
        self.btn_preview_snapshot.setText("Processing...")
        QApplication.processEvents()

        try:
            from fnt.usv.usv_detector.dsp_detector import DSPDetector
            from fnt.usv.usv_detector.config import USVDetectorConfig

            cfg = self._gather_dsp_config()
            detector_config = USVDetectorConfig(
                min_freq_hz=cfg.get('min_freq_hz', 20000),
                max_freq_hz=cfg.get('max_freq_hz', 65000),
                energy_threshold_db=cfg.get('energy_threshold_db', 10.0),
                min_duration_ms=cfg.get('min_duration_ms', 10.0),
                max_duration_ms=cfg.get('max_duration_ms', 1000.0),
                max_bandwidth_hz=cfg.get('max_bandwidth_hz', 20000.0),
                min_tonality=cfg.get('min_tonality', 0.3),
                min_call_freq_hz=cfg.get('min_call_freq_hz', 0.0),
                harmonic_filter=cfg.get('harmonic_filter', True),
                harmonic_label=cfg.get('harmonic_label', False),
                min_freq_gap_hz=cfg.get('min_freq_gap_hz', 5000.0),
                min_gap_ms=cfg.get('min_gap_ms', 5.0),
                noise_percentile=cfg.get('noise_percentile', 25.0),
                nperseg=cfg.get('nperseg', 512),
                noverlap=cfg.get('noverlap', 384),
                gpu_enabled=cfg.get('gpu_enabled', False),
                gpu_device=cfg.get('gpu_device', 'auto'),
                min_bandwidth_hz=cfg.get('min_bandwidth_hz', 0.0),
                min_snr_db=cfg.get('min_snr_db', 0.0),
                min_spectral_entropy=cfg.get('min_spectral_entropy', 0.0),
                max_spectral_entropy=cfg.get('max_spectral_entropy', 0.0),
                min_power_db=cfg.get('min_power_db', 0.0),
                max_power_db=cfg.get('max_power_db', 0.0),
                max_mean_sweep_rate=cfg.get('max_mean_sweep_rate', 0.0),
                max_contour_jitter=cfg.get('max_contour_jitter', 0.0),
                min_ici_ms=cfg.get('min_ici_ms', 0.0),
            )

            detector = DSPDetector(detector_config)

            seg_dur = len(audio_segment) / self.sample_rate
            print(f"[Preview Snapshot] view={view_start:.3f}-{view_end:.3f}s, "
                  f"samples={len(audio_segment)}, sr={self.sample_rate}, "
                  f"dur={seg_dur:.3f}s")
            print(f"[Preview Snapshot] Config: freq={detector_config.min_freq_hz}-"
                  f"{detector_config.max_freq_hz}Hz, thresh={detector_config.energy_threshold_db}dB, "
                  f"dur={detector_config.min_duration_ms}-{detector_config.max_duration_ms}ms, "
                  f"tonality={detector_config.min_tonality}, "
                  f"snr={detector_config.min_snr_db}dB, "
                  f"entropy={detector_config.min_spectral_entropy}-{detector_config.max_spectral_entropy}, "
                  f"min_bw={detector_config.min_bandwidth_hz}Hz")

            # Run detection with per-stage diagnostic logging
            from fnt.usv.usv_detector.spectrogram import bandpass_filter, compute_spectrogram_auto
            filtered = bandpass_filter(
                audio_segment, self.sample_rate,
                detector_config.min_freq_hz,
                detector_config.max_freq_hz
            )
            frequencies, times, Sxx_db = compute_spectrogram_auto(
                filtered, self.sample_rate,
                nperseg=detector_config.nperseg,
                noverlap=detector_config.noverlap,
                nfft=detector_config.nfft,
                window=detector_config.window_type,
                min_freq=detector_config.min_freq_hz,
                max_freq=detector_config.max_freq_hz,
                gpu_enabled=detector_config.gpu_enabled,
                gpu_device=detector_config.gpu_device,
            )
            print(f"[Preview Snapshot] Spectrogram: {Sxx_db.shape}, "
                  f"freq range {frequencies[0]:.0f}-{frequencies[-1]:.0f}Hz, "
                  f"time range {times[0]:.4f}-{times[-1]:.4f}s, "
                  f"dB range {Sxx_db.min():.1f} to {Sxx_db.max():.1f}")

            calls = detector._detect_from_spectrogram(frequencies, times, Sxx_db)
            print(f"[Preview Snapshot] After threshold+connected components: {len(calls)}")

            calls = detector._filter_by_duration(calls)
            print(f"[Preview Snapshot] After duration filter: {len(calls)}")

            calls = detector._merge_close_calls(calls)
            print(f"[Preview Snapshot] After merge close: {len(calls)}")

            calls = detector._filter_by_bandwidth(calls)
            print(f"[Preview Snapshot] After max bandwidth: {len(calls)}")

            calls = detector._filter_by_min_bandwidth(calls)
            print(f"[Preview Snapshot] After min bandwidth: {len(calls)}")

            calls = detector._filter_by_snr(calls)
            print(f"[Preview Snapshot] After SNR filter: {len(calls)}")

            calls = detector._compute_spectral_features(calls, frequencies, times, Sxx_db)

            # Diagnostic: show tonality and entropy distribution before filtering
            if calls:
                tonalities = [c.get('tonality', 0) for c in calls]
                entropies = [c.get('spectral_entropy', 0) for c in calls]
                print(f"[Preview Snapshot] Tonality stats: min={min(tonalities):.4f}, "
                      f"max={max(tonalities):.4f}, mean={sum(tonalities)/len(tonalities):.4f}, "
                      f"median={sorted(tonalities)[len(tonalities)//2]:.4f}")
                print(f"[Preview Snapshot] Entropy stats: min={min(entropies):.4f}, "
                      f"max={max(entropies):.4f}, mean={sum(entropies)/len(entropies):.4f}")
                # Show a few examples
                for i, c in enumerate(calls[:5]):
                    print(f"[Preview Snapshot]   call[{i}]: tonality={c.get('tonality',0):.4f}, "
                          f"entropy={c.get('spectral_entropy',0):.4f}, "
                          f"freq={c.get('min_freq_hz',0):.0f}-{c.get('max_freq_hz',0):.0f}Hz, "
                          f"dur={c.get('duration_ms',0):.1f}ms")

            calls = detector._filter_by_tonality(calls)
            print(f"[Preview Snapshot] After tonality filter (threshold={detector_config.min_tonality}): {len(calls)}")

            calls = detector._filter_by_spectral_entropy(calls)
            print(f"[Preview Snapshot] After spectral entropy ({detector_config.min_spectral_entropy}-{detector_config.max_spectral_entropy}): {len(calls)}")

            calls = detector._filter_by_min_freq(calls)
            print(f"[Preview Snapshot] After min freq filter: {len(calls)}")

            calls = detector._filter_by_power(calls)
            print(f"[Preview Snapshot] After power filter: {len(calls)}")

            calls = detector._filter_harmonics(calls)
            print(f"[Preview Snapshot] After harmonic filter: {len(calls)}")

            # Peak freq sampling
            n_samples = getattr(detector_config, 'freq_samples', 5)
            if n_samples and n_samples > 0:
                calls = detector._sample_peak_frequencies(calls, frequencies, times, Sxx_db)

            calls = detector._compute_contour_features(calls)
            calls = detector._filter_by_sweep_rate(calls)
            calls = detector._filter_by_contour_smoothness(calls)
            calls = detector._filter_by_ici(calls)
            print(f"[Preview Snapshot] Final detections: {len(calls)}")

            detections = calls

            # Offset all detection times by the view start
            for det in detections:
                det['start_seconds'] += view_start
                det['stop_seconds'] += view_start

            n_det = len(detections)
            import pandas as pd

            # Always clear existing detections within the preview window first
            if self.detections_df is not None and len(self.detections_df) > 0:
                mask_outside = (
                    (self.detections_df['stop_seconds'] <= view_start) |
                    (self.detections_df['start_seconds'] >= view_end)
                )
                self.detections_df = self.detections_df[mask_outside].copy()
            else:
                self.detections_df = None

            if n_det > 0:
                new_df = pd.DataFrame(detections)
                # Mark harmonics with special status, others as pending
                if 'is_harmonic' in new_df.columns:
                    new_df['status'] = new_df['is_harmonic'].apply(
                        lambda h: 'harmonic' if h else 'pending'
                    )
                else:
                    new_df['status'] = 'pending'
                self._ensure_freq_bounds(new_df)

                if self.detections_df is not None and len(self.detections_df) > 0:
                    self.detections_df = pd.concat([self.detections_df, new_df], ignore_index=True)
                    self.detections_df.sort_values('start_seconds', inplace=True)
                    self.detections_df.reset_index(drop=True, inplace=True)
                else:
                    self.detections_df = new_df

            # Handle empty result — set to None if nothing left
            if self.detections_df is not None and len(self.detections_df) == 0:
                self.detections_df = None

            # Update stored detections
            self.all_detections[filepath] = self.detections_df.copy() if self.detections_df is not None else pd.DataFrame()
            self.detection_sources[filepath] = 'usv_dsp'

            # Keep current_detection_idx valid without jumping the view
            if self.detections_df is not None and len(self.detections_df) > 0:
                self.current_detection_idx = min(self.current_detection_idx, len(self.detections_df) - 1)
                self.current_detection_idx = max(0, self.current_detection_idx)
            else:
                self.current_detection_idx = 0

            # Refresh display WITHOUT moving the spectrogram view
            self._update_display_no_scroll()
            if n_det > 0:
                n_harmonics = sum(1 for d in detections if d.get('is_harmonic', False))
                n_calls = n_det - n_harmonics
                harm_str = f" ({n_harmonics} harmonic)" if n_harmonics else ""
                self.status_bar.showMessage(
                    f"Preview snapshot: {n_calls} call{'s' if n_calls != 1 else ''}"
                    f"{harm_str} in {view_end - view_start:.1f}s window"
                )
            else:
                n_remaining = len(self.detections_df) if self.detections_df is not None else 0
                self.status_bar.showMessage(
                    f"Preview snapshot: no new detections in {view_end - view_start:.1f}s window "
                    f"(cleared window; {n_remaining} detections remain outside)"
                )

        except Exception as e:
            self.status_bar.showMessage(f"Preview snapshot error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_preview_snapshot.setEnabled(True)
            self.btn_preview_snapshot.setText("Process Preview Snapshot")
            self._update_ui_state()

    def run_dsp_detection(self):
        """Run DSP detection on queued files."""
        if not self.dsp_queue:
            return

        # Check which queued files already have detections
        files_with_dets = [f for f in self.dsp_queue if self._file_has_detections(f)]
        files_without_dets = [f for f in self.dsp_queue if f not in files_with_dets]
        n_with = len(files_with_dets)
        n_without = len(files_without_dets)
        n_total = len(self.dsp_queue)

        if n_with > 0:
            msg = (f"Queue: {n_total} file{'s' if n_total != 1 else ''}\n"
                   f"  - {n_with} file{'s' if n_with != 1 else ''} with existing detections\n"
                   f"  - {n_without} file{'s' if n_without != 1 else ''} without detections\n\n"
                   "How would you like to proceed?")
            box = QMessageBox(self)
            box.setWindowTitle("Existing Detections Found")
            box.setText(msg)
            btn_overwrite = box.addButton("Overwrite All", QMessageBox.AcceptRole)
            btn_skip = box.addButton("Skip Existing", QMessageBox.ActionRole)
            box.addButton("Cancel", QMessageBox.RejectRole)
            box.exec_()

            clicked = box.clickedButton()
            if clicked == btn_skip:
                if not files_without_dets:
                    self.status_bar.showMessage("All queued files already have detections — nothing to process")
                    return
                self.dsp_queue = files_without_dets
            elif clicked == btn_overwrite:
                pass  # Keep full queue
            else:
                return  # Cancel

        config = self._gather_dsp_config()

        # Start worker
        self.dsp_worker = DSPDetectionWorker(self.dsp_queue.copy(), config)
        self.dsp_worker.progress.connect(self._on_dsp_progress)
        self.dsp_worker.file_progress.connect(self._on_dsp_file_progress)
        self.dsp_worker.file_complete.connect(self._on_dsp_file_complete)
        self.dsp_worker.all_complete.connect(self._on_dsp_complete)
        self.dsp_worker.error.connect(self._on_dsp_error)

        self.dsp_progress.setVisible(True)
        self.dsp_progress.setValue(0)
        self.dsp_file_progress.setVisible(True)
        self.dsp_file_progress.setValue(0)
        self.btn_run_dsp.setEnabled(False)
        self.btn_run_dsp.setText("Running...")
        self.btn_stop_dsp.setVisible(True)

        self.dsp_worker.start()

    def stop_dsp_detection(self):
        """Stop DSP detection. The file currently being processed is discarded."""
        if hasattr(self, 'dsp_worker') and self.dsp_worker is not None:
            self.dsp_worker.stop()
            self.lbl_dsp_status.setText("Stopping after current file...")
            self.btn_stop_dsp.setEnabled(False)

    def _on_dsp_progress(self, filename, current, total):
        """Handle DSP batch progress update (file-level)."""
        self.dsp_progress.setValue(int(current / total * 100))
        self.lbl_dsp_status.setText(f"Processing ({current + 1}/{total}): {filename}")
        self.dsp_file_progress.setValue(0)

    def _on_dsp_file_progress(self, fraction):
        """Handle per-file progress update (chunk-level within a file)."""
        self.dsp_file_progress.setValue(int(fraction * 100))

    def _on_dsp_file_complete(self, filename, filepath, detections, n_detections):
        """Handle DSP file completion — write CSV immediately."""
        self.dsp_file_progress.setValue(100)
        self.lbl_dsp_status.setText(f"{filename}: {n_detections} detections")

        # Standard columns for detection CSVs
        std_columns = ['call_number', 'start_seconds', 'stop_seconds', 'duration_ms',
                        'min_freq_hz', 'max_freq_hz', 'peak_freq_hz', 'freq_bandwidth_hz',
                        'max_power_db', 'mean_power_db', 'status', 'source']

        # Build DataFrame from detection list
        if isinstance(detections, pd.DataFrame):
            df = detections.copy()
        else:
            df = pd.DataFrame(detections)

        if len(df) > 0:
            if 'status' not in df.columns:
                # Mark harmonics with special status if labeling is enabled
                if 'is_harmonic' in df.columns:
                    df['status'] = df['is_harmonic'].apply(
                        lambda h: 'harmonic' if h else 'pending'
                    )
                else:
                    df['status'] = 'pending'
            if 'source' not in df.columns:
                df['source'] = 'dsp'
            if 'call_number' not in df.columns:
                df.insert(0, 'call_number', range(1, len(df) + 1))
        else:
            df = pd.DataFrame(columns=std_columns)

        # Store in memory
        self.all_detections[filepath] = df
        self.detection_sources[filepath] = 'usv_dsp'

        # Write CSV immediately (crash-safe — results are on disk per file)
        base = Path(filepath).stem
        parent = Path(filepath).parent
        csv_path = parent / f"{base}_usv_dsp.csv"
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            self.status_bar.showMessage(f"Error saving CSV for {filename}: {e}")

        # Update file list item to show detection count
        if filepath in self.audio_files:
            idx = self.audio_files.index(filepath)
            item = self.file_list.item(idx)
            if item:
                parts = []
                if filepath in self.dsp_queue:
                    parts.append("\u2713")
                parts.append(filename)
                parts.append(f"({len(df)})")
                self.file_list.blockSignals(True)
                item.setText(" ".join(parts))
                self.file_list.blockSignals(False)

        # If this is the currently displayed file, update the view
        if (self.audio_files and self.current_file_idx < len(self.audio_files)
                and self.audio_files[self.current_file_idx] == filepath):
            self.detections_df = df.copy()
            self._ensure_freq_bounds(self.detections_df)
            self.current_detection_idx = 0
            self._load_current_file()

    def _on_dsp_complete(self, results):
        """Handle DSP batch completion. CSVs already written per-file."""
        was_stopped = hasattr(self, 'dsp_worker') and self.dsp_worker._stop_requested

        # Update UI
        self.dsp_progress.setValue(100)
        self.dsp_progress.setVisible(False)
        self.dsp_file_progress.setVisible(False)
        self.btn_stop_dsp.setVisible(False)
        self.btn_stop_dsp.setEnabled(True)
        self.btn_run_dsp.setEnabled(True)
        self.btn_run_dsp.setText("Run DSP Detection")

        total_det = sum(len(d) for d in results.values())
        if was_stopped:
            self.lbl_dsp_status.setText(f"Stopped. {len(results)} file{'s' if len(results) != 1 else ''} completed, {total_det} detections")
            self.status_bar.showMessage(f"DSP detection stopped. {len(results)} files completed.")
        else:
            self.lbl_dsp_status.setText(f"Complete: {total_det} total detections")
            self.status_bar.showMessage(f"DSP detection complete: {total_det} detections in {len(results)} files")

        # Clear queue and refresh display
        self.dsp_queue = []
        self._update_queue_display()

        self._refresh_file_list_items()
        self._load_current_file()

    def _on_dsp_error(self, filename, error):
        """Handle DSP error."""
        self.lbl_dsp_status.setText(f"Error: {filename} - {error}")

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

        # Navigation
        self.lbl_det_num.setText(f"Det {self.current_detection_idx + 1}/{n_det}")
        self.btn_prev_det.setEnabled(self.current_detection_idx > 0)
        self.btn_next_det.setEnabled(self.current_detection_idx < n_det - 1)

        # Time spinboxes
        self.spin_start.blockSignals(True)
        self.spin_stop.blockSignals(True)
        self.spin_start.setValue(det['start_seconds'])
        self.spin_stop.setValue(det['stop_seconds'])
        self.spin_start.blockSignals(False)
        self.spin_stop.blockSignals(False)

        # Frequency spinboxes (handle NaN values safely)
        self.spin_det_min_freq.blockSignals(True)
        self.spin_det_max_freq.blockSignals(True)
        min_f = det.get('min_freq_hz', 20000)
        max_f = det.get('max_freq_hz', 80000)
        # Handle NaN values
        min_f = 20000 if pd.isna(min_f) else int(min_f)
        max_f = 80000 if pd.isna(max_f) else int(max_f)
        self.spin_det_min_freq.setValue(min_f)
        self.spin_det_max_freq.setValue(max_f)
        self.spin_det_min_freq.blockSignals(False)
        self.spin_det_max_freq.blockSignals(False)

        # Info (handle NaN values)
        peak = det.get('peak_freq_hz', 0)
        dur = det.get('duration_ms', 0)
        peak = 0 if pd.isna(peak) else peak
        dur = 0 if pd.isna(dur) else dur
        self.lbl_det_info.setText(f"Peak: {peak:.0f} Hz | Dur: {dur:.1f} ms")

        # Status
        status = det.get('status', 'pending')
        self.lbl_det_status.setText(f"Status: {status.capitalize()}")
        if status == 'accepted':
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #107c10;")
        elif status == 'rejected':
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #d13438;")
        elif status == 'noise':
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #8b4513;")
        else:
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #999999;")

        # Update jump spinner
        self.spin_jump.blockSignals(True)
        self.spin_jump.setRange(1, max(1, n_det))
        self.spin_jump.setValue(self.current_detection_idx + 1)
        self.spin_jump.blockSignals(False)

        # Update waveform overview detection ticks
        det_times = list(zip(
            self.detections_df['start_seconds'].tolist(),
            self.detections_df['stop_seconds'].tolist()
        ))
        self.waveform_overview.set_detections(det_times)

        # Update spectrogram view
        self._update_spectrogram_view()
        self._update_progress()
        self._update_button_counts()
        self._update_button_counts()
        self._update_statistics()


    def _update_display_no_scroll(self):
        """Update display elements without moving the spectrogram view.

        Used by preview snapshot so the user's view stays in place.
        """
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
        elif status == 'noise':
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

        # Refresh detection boxes on the spectrogram WITHOUT moving the view
        self._update_detection_boxes()
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

        # Update detection boxes (reuses shared method to include contour data)
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

    def prev_detection(self):
        """Go to previous detection by time (respects filter)."""
        if self.detections_df is None:
            return
        filtered = self._get_filtered_indices()
        if not filtered:
            return
        # Get current detection's start time
        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']
        # Find all filtered detections that start before current, sorted by time
        earlier = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                   if self.detections_df.iloc[i]['start_seconds'] < current_time]
        if not earlier:
            # Wrap: also check detections at same time but different index
            earlier = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                       if i != self.current_detection_idx]
            if not earlier:
                return
            # Go to the last one temporally (wrap around)
            earlier.sort(key=lambda x: x[1])
            self.current_detection_idx = earlier[-1][0]
        else:
            # Go to the nearest earlier detection (latest start time before current)
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
        # Get current detection's start time
        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']
        # Find all filtered detections that start after current, sorted by time
        later = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                 if self.detections_df.iloc[i]['start_seconds'] > current_time]
        if not later:
            # Wrap: also check detections at same time but different index
            later = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                     if i != self.current_detection_idx]
            if not later:
                return
            # Go to the first one temporally (wrap around)
            later.sort(key=lambda x: x[1])
            self.current_detection_idx = later[0][0]
        else:
            # Go to the nearest later detection (earliest start time after current)
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

        # Update current file's stored detections (don't save yet - wait for drag complete)
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            filepath = self.audio_files[self.current_file_idx]
            self.all_detections[filepath] = self.detections_df.copy()

        # Update the spectrogram widget's detection in place (don't replace entire list during drag)
        self.spectrogram.update_detection(idx, start_s, stop_s, min_freq, max_freq)

        if idx == self.current_detection_idx:
            self._update_detection_info_only()

    def on_drag_complete(self):
        """Handle completion of box drag - save to CSV."""
        self._store_current_detections()
        self._update_button_counts()
        self._refresh_file_list_items()
        self._update_detection_boxes()

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

        # Handle NaN values safely
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
        self.lbl_det_info.setText(f"Peak: {peak:.0f} Hz | Dur: {dur:.1f} ms")

    def _update_detection_boxes(self):
        """Update detection boxes on spectrogram (includes peak_freq contour data)."""
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
            # Pass through peak_freq_N contour columns only if freq sampling is enabled
            if self.chk_freq_samples.isChecked():
                for col in row.index:
                    if col.startswith('peak_freq_') and col != 'peak_freq_hz':
                        val = row[col]
                        if not pd.isna(val):
                            det[col] = val
            detections.append(det)

        self.spectrogram.set_detections(detections, self.current_detection_idx)

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
        self._auto_advance()

    def skip_detection(self):
        """Skip to next detection by time without changing its status."""
        if self.detections_df is None:
            return
        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']
        # Find temporally next detection (any status)
        all_dets = [(i, self.detections_df.iloc[i]['start_seconds'])
                    for i in range(len(self.detections_df))]
        all_dets.sort(key=lambda x: x[1])
        for idx, t in all_dets:
            if t > current_time:
                self.current_detection_idx = idx
                self._update_display()
                return
        # At the last detection temporally — about to wrap
        # If fully curated, prompt to move to next file
        n_pending = (self.detections_df['status'] == 'pending').sum()
        if n_pending == 0:
            self._update_display()
            self._prompt_next_file()
            return
        # Wrap to earliest
        self.current_detection_idx = all_dets[0][0]
        self._update_display()

    def _update_button_counts(self):
        """Update Accept/Reject/Noise button labels with counts."""
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

        # Small square box centered at 40 kHz (typical USV range)
        freq_center = 40000
        freq_half = 4000  # ±4 kHz = 8 kHz span
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
            f"Delete {n_pending} pending detections?\n\n"
            "This keeps only labeled detections.",
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

        n_remaining = len(self.detections_df) if self.detections_df is not None else 0
        self.status_bar.showMessage(f"Deleted {n_pending} pending | {n_remaining} labeled remain")

    def delete_all_labels(self):
        """Delete ALL detections (pending + accepted + rejected) for the current file."""
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

        # Remove the CSV file on disk so labels don't reload
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            filepath = self.audio_files[self.current_file_idx]
            base = Path(filepath).stem
            parent = Path(filepath).parent
            source = self.detection_sources.get(filepath, 'usv_dsp')
            csv_path = parent / f"{base}_{source}.csv"
            if csv_path.exists():
                csv_path.unlink()

        self.detections_df = None
        self.current_detection_idx = 0
        self._store_current_detections()
        self._update_display()
        self._update_ui_state()
        self.status_bar.showMessage(f"Deleted all {n_total} detections")

    def _auto_advance(self):
        """Auto-advance to next pending detection by time, with wrap-around.

        After accept/reject, finds the temporally next pending detection.
        If none exist after the current position, wraps to the beginning.
        If ALL detections have been curated, prompts to move to next file.
        """
        if self.detections_df is None:
            return

        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']

        # Build list of all pending detections with their times
        pending = []
        for i in range(len(self.detections_df)):
            if self.detections_df.iloc[i]['status'] == 'pending':
                pending.append((i, self.detections_df.iloc[i]['start_seconds']))

        if not pending:
            # All detections have been curated — prompt for next file
            self._update_display()
            self._prompt_next_file()
            return

        # Sort by start time
        pending.sort(key=lambda x: x[1])

        # Find next pending after current time
        for idx, t in pending:
            if t > current_time:
                self.current_detection_idx = idx
                self._update_display()
                return

        # Wrap around — take the earliest pending detection
        self.current_detection_idx = pending[0][0]
        self._update_display()

    def _prompt_next_file(self):
        """Show dialog when all detections are curated, asking to move to next file."""
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

        # Save current file's detections first
        self._store_current_detections()

        # Search forward through files for one with detections
        for offset in range(1, len(self.audio_files)):
            next_idx = (self.current_file_idx + offset) % len(self.audio_files)
            filepath = self.audio_files[next_idx]

            has_dets = filepath in self.all_detections and len(self.all_detections[filepath]) > 0
            if not has_dets:
                # Check disk
                has_dets = self._file_has_detections(filepath)

            if has_dets:
                self.file_list.setCurrentRow(next_idx)
                # Jump to first pending detection by time, or first detection if none pending
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
                self.status_bar.showMessage(
                    f"Advanced to {os.path.basename(filepath)}")
                return

        self.status_bar.showMessage("No more files with detections")

    def _store_current_detections(self):
        """Store current detections in all_detections dict and save to CSV."""
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            filepath = self.audio_files[self.current_file_idx]
            if self.detections_df is not None:
                self.all_detections[filepath] = self.detections_df.copy()
                # Auto-save to CSV
                self._save_detections_csv(filepath)
            elif filepath in self.all_detections:
                del self.all_detections[filepath]

    def _save_detections_csv(self, filepath):
        """Save current detections to CSV file, preserving original naming."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        # Refresh call_number column before saving
        self.detections_df['call_number'] = range(1, len(self.detections_df) + 1)
        # Ensure call_number is the first column
        cols = self.detections_df.columns.tolist()
        if 'call_number' in cols:
            cols.remove('call_number')
            cols.insert(0, 'call_number')
            self.detections_df = self.detections_df[cols]

        base = Path(filepath).stem
        parent = Path(filepath).parent
        # Use the tracked source suffix, default to _usv_dsp
        source = self.detection_sources.get(filepath, 'usv_dsp')
        csv_path = parent / f"{base}_{source}.csv"

        try:
            self.detections_df.to_csv(csv_path, index=False)
        except Exception as e:
            self.status_bar.showMessage(f"Error saving CSV: {e}")

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
                # Resample to a standard output rate for smooth playback.
                # Playing at speed S means the output should have
                # (original_duration / S) seconds of audio at the output rate.
                output_sr = 44100
                original_duration = len(segment) / self.sample_rate
                output_duration = original_duration / self.playback_speed

                # Cap output duration to prevent enormous buffers
                max_play_duration = 120.0  # seconds
                if output_duration > max_play_duration:
                    self.status_bar.showMessage(
                        f"Playback too long ({output_duration:.0f}s at {self.playback_speed}x). "
                        f"Increase speed or reduce window size."
                    )
                    return

                n_output_samples = int(output_duration * output_sr)
                if n_output_samples < 100:
                    return
                segment = signal.resample(segment, n_output_samples).astype(np.float32)
                play_sr = output_sr

            # Normalize to prevent clipping
            peak = np.max(np.abs(segment))
            if peak > 0:
                segment = segment / peak * 0.9

            # Try playback, falling back to different sample rates if needed
            played = False
            for try_sr in [play_sr, 48000, 96000]:
                try:
                    if try_sr != play_sr:
                        # Resample to fallback rate
                        n_resamp = int(len(segment) * try_sr / play_sr)
                        play_segment = signal.resample(segment, n_resamp).astype(np.float32)
                    else:
                        play_segment = segment
                    sd.play(play_segment, try_sr)
                    played = True
                    break
                except sd.PortAudioError:
                    continue

            if not played:
                self.status_bar.showMessage("Playback error: no compatible audio output found")
                return

            self.is_playing = True
            self.btn_play.setText("Playing...")

            # Track playback position for the moving line.
            import time as _time
            self._playback_latency = 0.15
            self._playback_start_time = _time.time()
            self._playback_start_s = start_s
            self._playback_end_s = stop_s
            self._playback_timer.start()

            # Disable pan/zoom/navigation controls during playback
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
        # Re-enable controls
        self._set_playback_controls_enabled(True)

    def _update_playback_position(self):
        """Timer callback to update the playback position line."""
        import time as _time
        if not self.is_playing or self._playback_start_time is None:
            self.stop_playback()
            return

        # Subtract latency offset so the line matches when audio is heard
        elapsed = _time.time() - self._playback_start_time - self._playback_latency
        elapsed = max(0.0, elapsed)
        # Convert wall-clock elapsed to audio time using playback speed
        audio_elapsed = elapsed * self.playback_speed
        current_pos = self._playback_start_s + audio_elapsed

        if current_pos >= self._playback_end_s:
            # Playback finished
            self.stop_playback()
            return

        self.spectrogram.playback_position = current_pos
        self.spectrogram.update()

    def _set_playback_controls_enabled(self, enabled):
        """Enable or disable pan/zoom/navigation controls during playback."""
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

        # Low-pass filter
        from scipy.signal import butter, filtfilt
        nyq = self.sample_rate / 2
        cutoff = min(10000, nyq * 0.9)
        b, a = butter(4, cutoff / nyq, btype='low')
        filtered = filtfilt(b, a, mixed)

        # Resample to 44.1kHz
        target_len = int(len(filtered) * 44100 / self.sample_rate)
        resampled = signal.resample(filtered, target_len)

        return resampled.astype(np.float32)

    # =========================================================================
    # Save/Load
    # =========================================================================

    def open_output_folder(self):
        """Open folder containing current file (cross-platform)."""
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            folder = Path(self.audio_files[self.current_file_idx]).parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


    # =========================================================================
    # UI State Management
    # =========================================================================

    def _update_ui_state(self):
        """Update UI enabled states based on current state."""
        has_files = len(self.audio_files) > 0
        has_audio = self.audio_data is not None
        has_det = self.detections_df is not None and len(self.detections_df) > 0
        # Input
        self.btn_add_to_queue.setEnabled(has_files)
        self.btn_add_all_to_queue.setEnabled(has_files)

        # Detection navigation
        self.btn_prev_det.setEnabled(bool(has_det and self.current_detection_idx > 0))
        self.btn_next_det.setEnabled(bool(has_det and self.current_detection_idx < len(self.detections_df) - 1) if has_det else False)

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

        # Preview snapshot — available whenever audio is loaded
        self.btn_preview_snapshot.setEnabled(bool(has_audio))

        # Playback
        self.btn_play.setEnabled(bool(has_audio and HAS_SOUNDDEVICE))
        self.btn_stop.setEnabled(bool(has_audio and HAS_SOUNDDEVICE))

        # Open folder
        self.btn_open_folder.setEnabled(bool(has_files))


    def keyPressEvent(self, event):
        """Handle keyboard shortcuts. Skips when text-input widgets have focus."""
        # Don't intercept if a text-input widget has focus
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            super().keyPressEvent(event)
            return

        key = event.key()
        modifiers = event.modifiers()

        # Ctrl+Z = Undo
        if key == Qt.Key_Z and (modifiers & Qt.ControlModifier):
            self.undo_action()
            return

        if key == Qt.Key_A and self.btn_accept.isEnabled():
            self._flash_button(self.btn_accept)
            self.accept_detection()
        elif key == Qt.Key_R and self.btn_reject.isEnabled():
            self._flash_button(self.btn_reject)
            self.reject_detection()
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
                self.status_bar.showMessage(f"Undid label change on detection {idx + 1}")
        elif action[0] == 'batch_label':
            _, indices, old_status = action
            for idx in indices:
                if 0 <= idx < len(self.detections_df):
                    self.detections_df.at[idx, 'status'] = old_status
            self._store_current_detections()
            self._update_display()
            self._update_button_counts()
            self.status_bar.showMessage(f"Undid batch label change on {len(indices)} detections")

        self.btn_undo.setEnabled(len(self.undo_stack) > 0)

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
        # Push batch undo
        self.undo_stack.append(('batch_label', list(self.detections_df[mask].index), 'pending'))
        self.btn_undo.setEnabled(True)
        self.detections_df.loc[mask, 'status'] = 'accepted'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
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
        self.status_bar.showMessage(f"Rejected {n} detections")

    # =========================================================================
    # Filter / Jump
    # =========================================================================

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
    # Wheel Zoom / Overview Navigation / Display Frequency
    # =========================================================================

    def on_wheel_zoom(self, factor, center_time):
        """Handle mouse wheel zoom on spectrogram."""
        current_start, current_end = self.spectrogram.get_view_range()
        current_window = current_end - current_start
        new_window = current_window / factor
        total_dur = self.spectrogram.get_total_duration()
        new_window = max(0.05, min(total_dur if total_dur > 0 else 600, new_window))

        # Maintain zoom center at mouse position
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
        # Force spectrogram recompute
        self.spectrogram.cached_view_start = None
        self.spectrogram.cached_view_end = None
        self.spectrogram.spec_image = None
        if self.spectrogram.total_duration > 0:
            self.spectrogram._compute_view_spectrogram()
        self.spectrogram.update()

    def on_colormap_changed(self, name):
        """Handle colormap selection change."""
        self.spectrogram.set_colormap(name)

    # =========================================================================
    # Export
    # =========================================================================

    # =========================================================================
    # Detection Statistics
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
    # Settings Persistence
    # =========================================================================

    def _restore_settings(self):
        """Restore window geometry and preferences from QSettings."""
        settings = QSettings("FNT", "USVStudio")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("windowState")
        if state:
            self.restoreState(state)
        # Restore view preferences
        view_window = settings.value("view_window", 2.0, type=float)
        self.spin_view_window.setValue(view_window)
        colormap = settings.value("colormap", "viridis", type=str)
        idx = self.combo_colormap.findText(colormap)
        if idx >= 0:
            self.combo_colormap.setCurrentIndex(idx)

    def closeEvent(self, event):
        """Save settings on close."""
        settings = QSettings("FNT", "USVStudio")
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
    window = ClassicAudioDetectorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
