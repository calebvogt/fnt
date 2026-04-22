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

    def __init__(self, files: List[str], config: dict, noise_override: Optional[dict] = None):
        super().__init__()
        self.files = files
        self.config = config
        self.noise_override = noise_override  # dict with 'freqs' + 'noise_db', or None
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
            detect_harmonics=self.config.get('detect_harmonics', True),
            classify_call_types=self.config.get('classify_call_types', False),
            min_freq_gap_hz=self.config.get('min_freq_gap_hz', 5000.0),
            valley_split_ratio=self.config.get('valley_split_ratio', 0.30),
            min_gap_ms=self.config.get('min_gap_ms', 5.0),
            noise_percentile=self.config.get('noise_percentile', 25.0),
            noise_block_seconds=self.config.get('noise_block_seconds', 0.0),
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
        if self.noise_override:
            detector.noise_override = self.noise_override
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
# Pipeline Inspector Dialog (Phase 8)
# =============================================================================

class PipelineInspectorDialog(QDialog):
    """
    Matplotlib 2x3 grid of DSP pipeline stages for the current view window.

    Stages: raw spectrogram, bandpassed, noise floor, above-noise energy,
    binary mask (post-morphology), connected components.
    """

    def __init__(self, audio, sample_rate, view_range, dsp_config,
                 noise_override=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DSP Pipeline Inspector")
        self.resize(1200, 700)

        start_s, end_s = view_range
        start_sample = max(0, int(start_s * sample_rate))
        end_sample = min(len(audio), int(end_s * sample_rate))
        segment = audio[start_sample:end_sample]

        from fnt.usv.usv_detector.spectrogram import (
            bandpass_filter, compute_spectrogram_auto, estimate_noise_floor,
            estimate_noise_floor_blockwise,
        )
        from scipy.ndimage import label, binary_dilation, binary_erosion

        min_f = dsp_config.get('min_freq_hz', 20000)
        max_f = dsp_config.get('max_freq_hz', 65000)
        nperseg = dsp_config.get('nperseg', 512)
        noverlap = dsp_config.get('noverlap', 384)
        nfft = dsp_config.get('nfft', 1024)
        window = dsp_config.get('window_type', 'hann')
        thresh_db = dsp_config.get('energy_threshold_db', 10.0)
        percentile = dsp_config.get('noise_percentile', 25.0)
        block_s = dsp_config.get('noise_block_seconds', 0.0) or 0.0

        raw_f, raw_t, raw_Sxx = compute_spectrogram_auto(
            segment, sample_rate, nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, window=window, min_freq=min_f, max_freq=max_f,
        )
        filtered = bandpass_filter(segment, sample_rate, min_f, max_f)
        f, t, Sxx_db = compute_spectrogram_auto(
            filtered, sample_rate, nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, window=window, min_freq=min_f, max_freq=max_f,
        )

        if noise_override and noise_override.get('freqs') is not None:
            src_f = np.asarray(noise_override['freqs'], dtype=float)
            src_n = np.asarray(noise_override['noise_db'], dtype=float)
            noise_floor_1d = np.interp(f, src_f, src_n)
            noise_floor_display = noise_floor_1d[:, np.newaxis] * np.ones_like(Sxx_db)
            threshold = noise_floor_1d[:, np.newaxis] + thresh_db
            floor_title = "Noise Floor (user override)"
        elif block_s > 0:
            noise_floor_display = estimate_noise_floor_blockwise(Sxx_db, t, block_s, percentile)
            threshold = noise_floor_display + thresh_db
            floor_title = f"Noise Floor (block-wise, {block_s:.1f}s)"
        else:
            noise_floor_1d = estimate_noise_floor(Sxx_db, percentile)
            noise_floor_display = noise_floor_1d[:, np.newaxis] * np.ones_like(Sxx_db)
            threshold = noise_floor_1d[:, np.newaxis] + thresh_db
            floor_title = f"Noise Floor (p{percentile:.0f})"

        above_noise = Sxx_db - noise_floor_display
        mask = Sxx_db > threshold
        struct = np.ones((3, 3))
        mask_morph = binary_dilation(mask, structure=struct, iterations=1)
        mask_morph = binary_erosion(mask_morph, structure=struct, iterations=1)
        labeled, n_components = label(mask_morph)

        import matplotlib
        matplotlib.use('Qt5Agg', force=False)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

        layout = QVBoxLayout(self)
        fig = Figure(figsize=(14, 8), constrained_layout=True)
        canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        axes = fig.subplots(2, 3)
        extent = [t[0] + start_s, t[-1] + start_s, f[0] / 1000, f[-1] / 1000] if len(t) and len(f) else None
        raw_extent = ([raw_t[0] + start_s, raw_t[-1] + start_s,
                       raw_f[0] / 1000, raw_f[-1] / 1000]
                      if len(raw_t) and len(raw_f) else None)

        def _imshow(ax, Z, title, extent_, cmap='viridis', vmin=None, vmax=None):
            im = ax.imshow(Z, aspect='auto', origin='lower', extent=extent_,
                           cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('Freq (kHz)', fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        _imshow(axes[0, 0], raw_Sxx, "1. Raw Spectrogram (unfiltered)", raw_extent)
        _imshow(axes[0, 1], Sxx_db, f"2. Bandpassed ({min_f/1000:.0f}\u2013{max_f/1000:.0f} kHz)", extent)
        _imshow(axes[0, 2], noise_floor_display, f"3. {floor_title}", extent)
        _imshow(axes[1, 0], above_noise, "4. Sxx - Noise Floor (dB)", extent, cmap='magma')
        _imshow(axes[1, 1], mask_morph.astype(float),
                f"5. Mask post-morphology ({int(mask_morph.sum())} px)", extent, cmap='gray')
        _imshow(axes[1, 2], labeled, f"6. Connected Components ({n_components})",
                extent, cmap='tab20')

        fig.suptitle(
            f"Pipeline Inspector \u2014 view {start_s:.2f}\u2013{end_s:.2f}s"
            f" ({len(segment)/sample_rate:.2f}s, SR {sample_rate:,} Hz)",
            fontsize=11,
        )
        canvas.draw()


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
        'Prairie Vole USVs': {
            # Tuned to Ma et al. 2014 (Integr Zool 9:280-93), which documents
            # 14 adult prairie vole call types. Goals:
            #   • capture all 14 types (flat, up/down ramp, harmonic, 5 step
            #     variants, U, inverted-U, misc, complex, step composite)
            #   • one-shot detection — bias toward recall, reject noise via
            #     tonality+SNR+bandwidth, not duration
            #   • harmonics are *labeled* post-hoc (not removed); downstream
            #     classification handles the 14-way split

            # Fundamentals 20-80 kHz per paper; top raised to 80 kHz so we
            # also capture the harmonic component of 'harmonic' calls
            # (fundamental ~25, harmonic ~50) and expanded-range calls that
            # reach ~80 kHz during socio-sexual interaction.
            'min_freq_hz': 20000, 'max_freq_hz': 80000,

            # Threshold: 10 dB above per-bin noise floor.
            'energy_threshold_db': 10.0,

            # Duration: 5 ms floor — ramps and step fragments can be brief;
            # typical 'flat' calls are ~30 ms, complex calls can be long.
            'min_duration_ms': 5.0, 'max_duration_ms': 500.0,

            # Bandwidth: 40 kHz max — harmonic calls span fundamental+overtone
            # (~25-50 kHz = 25 kHz); complex/step-composite calls can span
            # more. 40 kHz leaves headroom without admitting broadband noise.
            'max_bandwidth_hz': 40000,

            # Tonality: 0.15 — step/harmonic calls have discontinuities that
            # lower the purity score; 0.15 keeps them while rejecting
            # broadband hiss and clicks.
            'min_tonality': 0.15,

            # Min call freq: 18 kHz — margin below the 20 kHz fundamental
            # without admitting the ~15 kHz cage-rustle / handling-noise band.
            'min_call_freq_hz': 18000,

            # Harmonic labeling (not removal) + 5 kHz gap matches the paper's
            # ≥5 kHz criterion for U/inverted-U and step jumps.
            'detect_harmonics': True,
            'min_freq_gap_hz': 5000, 'valley_split_ratio': 0.30,

            # Min gap: 20 ms — matches Ma et al.'s max-within-call
            # discontinuity. Prefers merging step/complex fragments into
            # a single detection over splitting. Occasional fusion of
            # distinct adjacent calls is acceptable; downstream cleanup
            # can split them.
            'min_gap_ms': 20.0,

            # Post-hoc call-type classification (Ma et al. 2014, 14 types).
            # Toggled in the GUI for the Prairie Vole profile only.
            'classify_call_types': True,

            'noise_percentile': 25.0,
            # 512-point FFT / 75% overlap matches Ma et al. (391 Hz res).
            'nperseg': 512, 'noverlap': 384,

            # Min bandwidth 1 kHz: rejects pure-tone electrical artifacts.
            'min_bandwidth_hz': 1000,
            # SNR 8 dB: real calls sit well above the per-bin floor.
            'min_snr_db': 8.0,

            # Advanced filters disabled — they add false negatives for the
            # rarer call types (step composite, complex) without meaningful
            # noise-rejection gain at this detection stage.
            'min_spectral_entropy': 0.0, 'max_spectral_entropy': 0.0,
            'min_power_db': 0.0, 'max_power_db': 0.0,
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

        # Undo / redo stacks (each entry is an (action_type, ...) tuple).
        # A fresh user labeling action always clears the redo stack.
        self.undo_stack = deque(maxlen=50)
        self.redo_stack = deque(maxlen=50)

        # Filter state
        self.filter_status = 'all'  # 'all', 'pending', 'accepted', 'rejected'

        # Per-bin noise floor override captured from a silence region.
        # When set (dict with 'freqs' and 'noise_db' 1D arrays), the detector
        # uses this in place of the percentile-based floor. Cleared by the user
        # via Clear Override or by _load_current_file when the file changes.
        self._noise_override = None

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

        # Left panel (scrollable).
        #
        # Width sizing is derived from the current font's average char width
        # so Windows at 125%/150% display scaling gets proportionally more
        # room than macOS at 100%. The previous hard-coded 340/500 px caps
        # clipped buttons like "Add Folder / Add Files / Clear" on Windows.
        # Horizontal scroll is AsNeeded rather than AlwaysOff so content is
        # always reachable even if the widest row overflows the cap.
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        _fm = self.fontMetrics()
        # "0" is close to the average advance width; buttons fit in ~55 chars,
        # mac fits ~60 comfortably, Windows at 125% needs closer to ~70.
        _min_cols = 55
        _max_cols = 80
        left_scroll.setMinimumWidth(max(340, _fm.averageCharWidth() * _min_cols + 40))
        left_scroll.setMaximumWidth(max(500, _fm.averageCharWidth() * _max_cols + 40))

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
        self.btn_play = QPushButton("Play (Space)")
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
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)  # Left panel: don't stretch
        splitter.setStretchFactor(1, 1)  # Right panel: takes remaining space
        splitter.setSizes([420, 800])    # Initial sizes
        main_layout.addWidget(splitter)

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
        profile_row.addWidget(self._make_label("Profile:",
            "Species-specific preset that auto-fills all preprocessing\n"
            "and detection parameters. Selecting a preset locks all\n"
            "parameters; choose 'Manual' to unlock and customize.\n"
            "Use 'Save' to store your current Manual settings as a\n"
            "reusable custom profile (saved to ~/.fnt/detection_profiles/).",
            min_width=80))
        self.combo_species_profile = QComboBox()
        for name in self.SPECIES_PROFILES.keys():
            self.combo_species_profile.addItem(name)
        # Load custom profiles from disk
        self._custom_profiles = self._load_custom_profiles()
        for name in sorted(self._custom_profiles.keys()):
            self.combo_species_profile.addItem(name)
        self.combo_species_profile.setCurrentText("Manual")
        self.combo_species_profile.setToolTip(
            "Species profile presets.\n"
            "Choosing a non-Manual profile auto-fills every DSP parameter\n"
            "and LOCKS all parameter widgets so they cannot be edited.\n"
            "Switch back to Manual to re-enable editing."
        )
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
        freq_tip = ("PREPROCESSING: Bandpass frequency range (Hz).\n"
                     "Only spectrogram rows within this range are analyzed;\n"
                     "energy outside is ignored entirely. This is applied\n"
                     "before thresholding and is visible in the Filter Overlay.\n\n"
                     "Set to match your species' vocalization range:\n"
                     "  Prairie voles: 20-55 kHz (fundamental)\n"
                     "  Mice: 30-110 kHz\n"
                     "  Rats (50 kHz): 35-80 kHz\n"
                     "  Rats (22 kHz alarm): 18-32 kHz\n\n"
                     "Each frequency row spans ~sample_rate/nfft Hz.\n"
                     "At 250 kHz / 512 nfft = ~488 Hz per row.")
        freq_row = QHBoxLayout()
        freq_row.setSpacing(4)
        freq_row.addWidget(self._make_label("Freq Range:", freq_tip, min_width=80))

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
        thresh_tip = ("Energy in dB above the per-frequency-bin noise floor required for a pixel to count as detection.\n"
                      "Typical: 8–12 dB. Lower = more sensitive / more false positives. Visible in Filter Overlay (F).")
        thresh_row = QHBoxLayout()
        thresh_row.setSpacing(4)
        thresh_row.addWidget(self._make_label("Threshold:", thresh_tip, min_width=80))

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(1.0, 30.0)
        self.spin_threshold.setSingleStep(0.5)
        self.spin_threshold.setValue(10.0)
        self.spin_threshold.setSuffix(" dB")
        self.spin_threshold.setToolTip(thresh_tip)
        thresh_row.addWidget(self.spin_threshold)
        thresh_row.addStretch()

        group_layout.addLayout(thresh_row)

        # Noise percentile (preprocessing — affects filter overlay)
        noise_tip = ("Percentile (per frequency bin, over the whole file) taken as the noise-floor baseline (%).\n"
                     "25 is a good default. Raise to 40–50 if the recording is call-dense and\n"
                     "low percentiles are themselves contaminated by calls. Visible in Filter Overlay (F).")
        noise_row = QHBoxLayout()
        noise_row.setSpacing(4)
        noise_row.addWidget(self._make_label("Noise %tile:", noise_tip, min_width=80))
        self.spin_noise_pct = QDoubleSpinBox()
        self.spin_noise_pct.setRange(1.0, 50.0)
        self.spin_noise_pct.setValue(25.0)
        self.spin_noise_pct.setToolTip(noise_tip)
        noise_row.addWidget(self.spin_noise_pct, 1)
        group_layout.addLayout(noise_row)

        # --- Detection Parameters (collapsible) ---
        self.btn_advanced_toggle = QPushButton("▶ Detection Parameters")
        self.btn_advanced_toggle.setFlat(True)
        self.btn_advanced_toggle.setStyleSheet("text-align: left; color: #aaaaaa; font-size: 11px; padding: 2px 0px;")
        self.btn_advanced_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_advanced_toggle.clicked.connect(self._toggle_advanced_options)
        group_layout.addWidget(self.btn_advanced_toggle)

        self.advanced_options_widget = QWidget()
        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(4)

        # === Widgets ordered to match detection pipeline execution order ===

        # 1. Frequency gap splitting (during connected component analysis)
        freq_gap_tip = ("DETECTION: Frequency gap for splitting merged blobs (Hz).\n"
                        "When a fundamental and its harmonic are joined into one\n"
                        "connected component (e.g. by a noise pixel bridge), this\n"
                        "parameter attempts to split them into separate detections.\n\n"
                        "Works with the Valley Split Ratio parameter (next row):\n"
                        "1. Binary gap: splits where there are literally no active\n"
                        "   pixels across a gap of this many Hz.\n"
                        "2. Energy valley: even when faint pixels bridge the gap,\n"
                        "   analyzes the per-row energy profile and splits at\n"
                        "   valleys whose depth exceeds Valley Split Ratio.\n\n"
                        "The gap is measured in frequency bins:\n"
                        "  gap_bins = freq_gap_hz / (sample_rate / nfft)\n"
                        "  e.g. 5000 Hz / 488 Hz = ~10 bins\n\n"
                        "Higher values = only split large gaps (less aggressive).\n"
                        "Lower values = split smaller gaps (more aggressive).\n"
                        "Set to 0 to disable.")
        freq_gap_row = QHBoxLayout()
        freq_gap_row.setSpacing(4)
        freq_gap_row.addWidget(self._make_label("Freq Gap:", freq_gap_tip, min_width=80))
        self.spin_freq_gap = QSpinBox()
        self.spin_freq_gap.setRange(0, 30000)
        self.spin_freq_gap.setSingleStep(1000)
        self.spin_freq_gap.setValue(5000)
        self.spin_freq_gap.setSuffix(" Hz")
        self.spin_freq_gap.setToolTip(freq_gap_tip)
        freq_gap_row.addWidget(self.spin_freq_gap, 1)
        advanced_layout.addLayout(freq_gap_row)

        # Valley split ratio (energy valley depth threshold)
        valley_tip = ("DETECTION: Energy valley depth ratio for splitting.\n"
                      "Used by Freq Gap's energy valley detection (Pass 2).\n\n"
                      "For each candidate split point, the algorithm computes:\n"
                      "  ratio = valley_energy / avg_peak_energy\n"
                      "in linear power (not dB). If ratio < this threshold,\n"
                      "the valley is deep enough to split.\n\n"
                      "Lower values = more conservative (only split deep valleys).\n"
                      "Higher values = more aggressive (split shallower dips).\n\n"
                      "  0.10 = valley must be 10 dB below peaks (very conservative)\n"
                      "  0.30 = valley must be ~5 dB below peaks (default)\n"
                      "  0.50 = valley must be ~3 dB below peaks (aggressive)\n\n"
                      "Only active when Freq Gap > 0.")
        valley_row = QHBoxLayout()
        valley_row.setSpacing(4)
        valley_row.addWidget(self._make_label("Valley Split:", valley_tip, min_width=80))
        self.spin_valley_ratio = QDoubleSpinBox()
        self.spin_valley_ratio.setRange(0.05, 0.95)
        self.spin_valley_ratio.setSingleStep(0.05)
        self.spin_valley_ratio.setDecimals(2)
        self.spin_valley_ratio.setValue(0.30)
        self.spin_valley_ratio.setToolTip(valley_tip)
        valley_row.addWidget(self.spin_valley_ratio, 1)
        advanced_layout.addLayout(valley_row)

        # 2. Duration filter
        dur_tip = ("DETECTION: Duration filter for connected components (ms).\n"
                   "After thresholding, contiguous pixel blobs (connected\n"
                   "components) are identified. Each blob's temporal extent\n"
                   "is measured in milliseconds. Blobs shorter than Min or\n"
                   "longer than Max are discarded.\n\n"
                   "Short noise speckle is typically 1-5 ms.\n"
                   "Prairie vole syllables: 8-100 ms.\n"
                   "Mouse syllables: 5-200 ms.\n"
                   "Rat 22 kHz alarm calls: 100-3000 ms.\n\n"
                   "Set Min higher to reject noise fragments.\n"
                   "Set Max lower to reject long broadband events.")
        dur_row = QHBoxLayout()
        dur_row.setSpacing(4)
        dur_row.addWidget(self._make_label("Duration:", dur_tip, min_width=80))

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

        advanced_layout.addLayout(dur_row)

        # 3. Merge close calls
        gap_tip = ("DETECTION: Minimum temporal gap between detections (ms).\n"
                   "After connected components are found, any two detections\n"
                   "separated by less than this gap are merged into one.\n"
                   "This prevents a single call with a brief amplitude dip\n"
                   "from being split into two detections.\n\n"
                   "Lower values (2-4 ms) = preserve fine temporal structure,\n"
                   "keep rapid syllable sequences as separate detections.\n"
                   "Higher values (10-30 ms) = merge syllables into phrases.\n\n"
                   "Within-sniff-cycle silences: typically < 20 ms.\n"
                   "Between-sniff silences: typically > 60 ms.\n"
                   "Set to 0 to disable merging.")
        gap_row = QHBoxLayout()
        gap_row.setSpacing(4)
        gap_row.addWidget(self._make_label("Min Gap:", gap_tip, min_width=80))
        self.spin_min_gap = QDoubleSpinBox()
        self.spin_min_gap.setRange(0.0, 100.0)
        self.spin_min_gap.setValue(5.0)
        self.spin_min_gap.setSuffix(" ms")
        self.spin_min_gap.setToolTip(gap_tip)
        gap_row.addWidget(self.spin_min_gap, 1)
        advanced_layout.addLayout(gap_row)

        # 4. Max bandwidth
        bw_tip = ("DETECTION: Maximum frequency bandwidth of a detection (Hz).\n"
                  "Measured as max_freq - min_freq of the connected component.\n"
                  "Blobs spanning wider than this are rejected as broadband\n"
                  "noise (cage bumps, movement artifacts, handling noise).\n\n"
                  "NOTE: If a fundamental and its harmonic are merged into\n"
                  "one blob, the bandwidth includes both. Use Freq Gap to\n"
                  "split them before this filter runs. Energy-based valley\n"
                  "detection will also split at dips between harmonics.\n\n"
                  "Typical USV bandwidth: 5-15 kHz (fundamental only).\n"
                  "FM sweeps can span 15-25 kHz.\n"
                  "Set to 0 to disable.")
        bw_row = QHBoxLayout()
        bw_row.setSpacing(4)
        bw_row.addWidget(self._make_label("Max Bandwidth:", bw_tip, min_width=80))
        self.spin_max_bw = QSpinBox()
        self.spin_max_bw.setRange(0, 100000)
        self.spin_max_bw.setSingleStep(1000)
        self.spin_max_bw.setValue(20000)
        self.spin_max_bw.setSuffix(" Hz")
        self.spin_max_bw.setToolTip(bw_tip)
        bw_row.addWidget(self.spin_max_bw, 1)
        advanced_layout.addLayout(bw_row)

        # 5. Min bandwidth
        min_bw_tip = ("DETECTION: Minimum frequency bandwidth (Hz).\n"
                      "Rejects detections whose frequency extent\n"
                      "(max_freq - min_freq) is narrower than this value.\n\n"
                      "Targets electrical noise artifacts: harmonics of\n"
                      "60 Hz power lines, oscillator leakage, and other\n"
                      "narrow-band interference appear as thin horizontal\n"
                      "lines with near-zero bandwidth.\n\n"
                      "Real USV calls, even constant-frequency rat 22 kHz\n"
                      "alarm calls, have >= 1 kHz bandwidth due to onset/\n"
                      "offset transients and slight FM.\n\n"
                      "Typical: 500-1000 Hz. Set to 0 to disable.")
        min_bw_row = QHBoxLayout()
        min_bw_row.setSpacing(4)
        min_bw_row.addWidget(self._make_label("Min Bandwidth:", min_bw_tip, min_width=80))
        self.spin_min_bw = QSpinBox()
        self.spin_min_bw.setRange(0, 50000)
        self.spin_min_bw.setSingleStep(100)
        self.spin_min_bw.setValue(0)
        self.spin_min_bw.setSuffix(" Hz")
        self.spin_min_bw.setToolTip(min_bw_tip)
        min_bw_row.addWidget(self.spin_min_bw, 1)
        advanced_layout.addLayout(min_bw_row)

        # 6. Min SNR
        snr_tip = ("DETECTION: Minimum signal-to-noise ratio (dB).\n"
                   "For each detection, SNR is computed as:\n"
                   "  snr_dB = max_power_dB - mean(noise_floor_dB)\n"
                   "where noise_floor is the per-frequency-row estimate\n"
                   "from the Noise %tile parameter, averaged across the\n"
                   "detection's frequency range.\n\n"
                   "This is different from the Threshold parameter:\n"
                   "  Threshold = per-pixel test during preprocessing.\n"
                   "  Min SNR = per-detection test on the blob's peak.\n\n"
                   "A detection can survive thresholding (enough pixels\n"
                   "above threshold) but still have low overall SNR.\n\n"
                   "Typical: 6-10 dB. Set to 0 to disable.")
        snr_row = QHBoxLayout()
        snr_row.setSpacing(4)
        snr_row.addWidget(self._make_label("Min SNR:", snr_tip, min_width=80))
        self.spin_min_snr = QDoubleSpinBox()
        self.spin_min_snr.setRange(0.0, 50.0)
        self.spin_min_snr.setSingleStep(1.0)
        self.spin_min_snr.setValue(0.0)
        self.spin_min_snr.setSuffix(" dB")
        self.spin_min_snr.setToolTip(snr_tip)
        snr_row.addWidget(self.spin_min_snr, 1)
        advanced_layout.addLayout(snr_row)

        # 7. Tonality (spectral purity)
        ton_tip = ("DETECTION: Minimum spectral purity (tonality) score.\n"
                   "Computed as: 1 - (geometric_mean / arithmetic_mean)\n"
                   "of the power spectrum within the detection's bounding box.\n\n"
                   "1.0 = pure tone (all energy at one frequency bin).\n"
                   "0.0 = white noise (energy spread uniformly).\n\n"
                   "IMPORTANT: Empirical tonality values depend on FFT\n"
                   "resolution and call duration. Short calls (3-10 ms) with\n"
                   "our default FFT (512 samples) typically measure 0.05-0.20\n"
                   "because the bounding box includes noise around the call.\n"
                   "Longer calls (20-50 ms) measure 0.15-0.60.\n\n"
                   "Literature values (e.g. DeepSqueak's 0.3 default) assume\n"
                   "different FFT parameters. Tune empirically using the\n"
                   "diagnostic output from Query Preview Snapshot.\n"
                   "Set to 0 to disable.")
        ton_row = QHBoxLayout()
        ton_row.setSpacing(4)
        ton_row.addWidget(self._make_label("Tonality:", ton_tip, min_width=80))
        self.spin_tonality = QDoubleSpinBox()
        self.spin_tonality.setRange(0.0, 1.0)
        self.spin_tonality.setSingleStep(0.05)
        self.spin_tonality.setDecimals(2)
        self.spin_tonality.setValue(0.30)
        self.spin_tonality.setToolTip(ton_tip)
        ton_row.addWidget(self.spin_tonality, 1)
        advanced_layout.addLayout(ton_row)

        # 8. Spectral entropy range
        ent_tip = ("DETECTION: Spectral entropy range (Shannon entropy).\n"
                   "Computed on the normalized power spectrum within the\n"
                   "detection's bounding box:\n"
                   "  H = -sum(p * log2(p)) / log2(N)\n"
                   "where p is the normalized power per frequency bin\n"
                   "and N is the number of bins. Normalized to [0, 1].\n\n"
                   "0.0 = all energy in one bin (pure tone / electrical).\n"
                   "1.0 = energy spread uniformly (white noise).\n\n"
                   "Min entropy: rejects pure-tone electrical artifacts\n"
                   "  (e.g. 60 Hz harmonics have entropy near 0).\n"
                   "Max entropy: rejects broadband noise events\n"
                   "  (cage bumps, handling noise have entropy > 0.85).\n\n"
                   "IMPORTANT: Like tonality, empirical values depend on\n"
                   "FFT resolution and call duration. With our default FFT,\n"
                   "real USVs typically measure 0.5-0.9 (not the 0.1-0.6\n"
                   "reported in literature with different parameters).\n"
                   "Set either bound to 0 to disable it.")
        ent_row = QHBoxLayout()
        ent_row.setSpacing(4)
        ent_row.addWidget(self._make_label("Entropy:", ent_tip, min_width=80))
        self.spin_min_entropy = QDoubleSpinBox()
        self.spin_min_entropy.setRange(0.0, 1.0)
        self.spin_min_entropy.setSingleStep(0.05)
        self.spin_min_entropy.setDecimals(2)
        self.spin_min_entropy.setValue(0.0)
        self.spin_min_entropy.setToolTip(
            "Minimum Shannon entropy of the power spectrum (0–1).\n"
            "0 = pure tone, 1 = white noise. Low min rejects pure-tone\n"
            "electrical artifacts (60 Hz harmonics). Typical USVs: 0.5–0.9\n"
            "with our default FFT (see full Entropy tooltip on the label). 0 = disabled."
        )
        ent_row.addWidget(self.spin_min_entropy, 1)
        ent_row.addWidget(QLabel("-"))
        self.spin_max_entropy = QDoubleSpinBox()
        self.spin_max_entropy.setRange(0.0, 1.0)
        self.spin_max_entropy.setSingleStep(0.05)
        self.spin_max_entropy.setDecimals(2)
        self.spin_max_entropy.setValue(0.0)
        self.spin_max_entropy.setToolTip(
            "Maximum Shannon entropy of the power spectrum (0–1).\n"
            "Low max rejects broadband noise (cage bumps, handling)\n"
            "which typically show entropy > 0.85. Typical USVs: 0.5–0.9.\n"
            "0 = disabled."
        )
        ent_row.addWidget(self.spin_max_entropy, 1)
        advanced_layout.addLayout(ent_row)

        # 9. Min call frequency
        mcf_tip = ("DETECTION: Minimum peak frequency of a detection (Hz).\n"
                   "Each detection's peak frequency (the frequency bin with\n"
                   "the highest power) is compared against this threshold.\n"
                   "Detections whose peak frequency falls below this value\n"
                   "are discarded.\n\n"
                   "Useful for rejecting broadband noise artifacts that\n"
                   "extend below the typical USV frequency range. Unlike\n"
                   "the Freq Range (preprocessing), this operates on the\n"
                   "detection's actual peak — not the bounding box edges.\n\n"
                   "Set below your species' lowest expected fundamental:\n"
                   "  Prairie voles: 15-20 kHz\n"
                   "  Mice: 25-30 kHz\n"
                   "Set to 0 to disable.")
        mcf_row = QHBoxLayout()
        mcf_row.setSpacing(4)
        mcf_row.addWidget(self._make_label("Min Call Freq:", mcf_tip, min_width=80))
        self.spin_min_call_freq = QSpinBox()
        self.spin_min_call_freq.setRange(0, 100000)
        self.spin_min_call_freq.setSingleStep(1000)
        self.spin_min_call_freq.setValue(0)
        self.spin_min_call_freq.setSuffix(" Hz")
        self.spin_min_call_freq.setToolTip(mcf_tip)
        mcf_row.addWidget(self.spin_min_call_freq, 1)
        advanced_layout.addLayout(mcf_row)

        # 10. Power range
        pwr_tip = ("DETECTION: Absolute power thresholds (dB).\n"
                   "Unlike SNR (relative to noise floor), these filter on\n"
                   "absolute power values from the spectrogram.\n\n"
                   "Min power: mean_power_dB across the detection must\n"
                   "  exceed this. Rejects faint/marginal detections.\n"
                   "Max power: max_power_dB must be below this. Rejects\n"
                   "  clipping artifacts and electrical saturation.\n\n"
                   "NOTE: Absolute power depends heavily on microphone\n"
                   "sensitivity, gain, and recording distance. These values\n"
                   "are NOT comparable across recording setups. Prefer\n"
                   "relative filters (SNR, Threshold) unless you have a\n"
                   "calibrated recording pipeline.\n\n"
                   "Set either to 0 to disable.")
        pwr_row = QHBoxLayout()
        pwr_row.setSpacing(4)
        pwr_row.addWidget(self._make_label("Power:", pwr_tip, min_width=80))
        self.spin_min_power = QDoubleSpinBox()
        self.spin_min_power.setRange(-120.0, 0.0)
        self.spin_min_power.setSingleStep(1.0)
        self.spin_min_power.setValue(0.0)
        self.spin_min_power.setSuffix(" dB")
        self.spin_min_power.setToolTip(
            "Minimum mean power (dB) within the detection box.\n"
            "Rejects faint/marginal detections. Recording-setup dependent;\n"
            "prefer Min SNR unless your pipeline is calibrated. 0 = disabled."
        )
        pwr_row.addWidget(self.spin_min_power, 1)
        pwr_row.addWidget(QLabel("-"))
        self.spin_max_power = QDoubleSpinBox()
        self.spin_max_power.setRange(-120.0, 0.0)
        self.spin_max_power.setSingleStep(1.0)
        self.spin_max_power.setValue(0.0)
        self.spin_max_power.setSuffix(" dB")
        self.spin_max_power.setToolTip(
            "Maximum peak power (dB) within the detection box.\n"
            "Rejects clipping / electrical saturation artifacts.\n"
            "Recording-setup dependent. 0 = disabled."
        )
        pwr_row.addWidget(self.spin_max_power, 1)
        advanced_layout.addLayout(pwr_row)

        # 11. Max sweep rate
        sweep_tip = ("DETECTION: Maximum mean frequency sweep rate (kHz/ms).\n"
                     "Requires Freq Samples to be enabled (checkbox below).\n\n"
                     "The detection's time span is divided into N equal\n"
                     "segments, and the peak frequency in each segment is\n"
                     "sampled. The mean sweep rate is:\n"
                     "  mean(|delta_freq| / delta_time) across all segments\n\n"
                     "Biological USV sweeps: typically 0.5-3.0 kHz/ms.\n"
                     "Mouse upsweep: ~60→80 kHz in 20 ms = 1.0 kHz/ms.\n"
                     "Rat flat calls: < 0.2 kHz/ms.\n"
                     "Noise transients: can exceed 5 kHz/ms.\n\n"
                     "Set to 0 to disable.")
        sweep_row = QHBoxLayout()
        sweep_row.setSpacing(4)
        sweep_row.addWidget(self._make_label("Max Sweep:", sweep_tip, min_width=80))
        self.spin_max_sweep = QDoubleSpinBox()
        self.spin_max_sweep.setRange(0.0, 100.0)
        self.spin_max_sweep.setSingleStep(0.5)
        self.spin_max_sweep.setDecimals(2)
        self.spin_max_sweep.setValue(0.0)
        self.spin_max_sweep.setSuffix(" kHz/ms")
        self.spin_max_sweep.setToolTip(sweep_tip)
        sweep_row.addWidget(self.spin_max_sweep, 1)
        advanced_layout.addLayout(sweep_row)

        # 12. Max contour jitter
        jitter_tip = ("DETECTION: Maximum frequency contour jitter (kHz).\n"
                      "Requires Freq Samples to be enabled (checkbox below).\n\n"
                      "Measures smoothness of the peak frequency trajectory\n"
                      "using mean absolute second derivative (acceleration):\n"
                      "  jitter = mean(|f[i+1] - 2*f[i] + f[i-1]|)\n"
                      "where f[i] are the sampled peak frequencies in kHz.\n\n"
                      "Low jitter = smooth frequency contour (real USV).\n"
                      "High jitter = erratic frequency jumps (noise).\n\n"
                      "Biological calls have smooth FM sweeps or flat\n"
                      "contours with jitter typically < 2-5 kHz.\n"
                      "Noise artifacts show random frequency jumps between\n"
                      "adjacent time samples.\n\n"
                      "Set to 0 to disable.")
        jitter_row = QHBoxLayout()
        jitter_row.setSpacing(4)
        jitter_row.addWidget(self._make_label("Max Jitter:", jitter_tip, min_width=80))
        self.spin_max_jitter = QDoubleSpinBox()
        self.spin_max_jitter.setRange(0.0, 50.0)
        self.spin_max_jitter.setSingleStep(0.5)
        self.spin_max_jitter.setDecimals(2)
        self.spin_max_jitter.setValue(0.0)
        self.spin_max_jitter.setSuffix(" kHz")
        self.spin_max_jitter.setToolTip(jitter_tip)
        jitter_row.addWidget(self.spin_max_jitter, 1)
        advanced_layout.addLayout(jitter_row)

        # 13. Min ICI
        ici_tip = ("DETECTION: Minimum inter-call interval (ms).\n"
                   "For each detection, the time gap to the previous\n"
                   "detection is measured. If the gap is shorter than\n"
                   "this threshold, the detection is discarded.\n\n"
                   "Targets periodic electrical noise that produces\n"
                   "detections at impossibly fast repetition rates\n"
                   "(e.g. 60 Hz harmonics = 16.7 ms intervals).\n\n"
                   "Natural USV inter-call intervals:\n"
                   "  Within sniff cycle: 7-20 ms\n"
                   "  Between sniff cycles: > 60 ms\n\n"
                   "Use cautiously — real calls within rapid bouts\n"
                   "can be very closely spaced (< 10 ms).\n"
                   "Set to 0 to disable.")
        ici_row = QHBoxLayout()
        ici_row.setSpacing(4)
        ici_row.addWidget(self._make_label("Min ICI:", ici_tip, min_width=80))
        self.spin_min_ici = QDoubleSpinBox()
        self.spin_min_ici.setRange(0.0, 100.0)
        self.spin_min_ici.setSingleStep(1.0)
        self.spin_min_ici.setValue(0.0)
        self.spin_min_ici.setSuffix(" ms")
        self.spin_min_ici.setToolTip(ici_tip)
        ici_row.addWidget(self.spin_min_ici, 1)
        advanced_layout.addLayout(ici_row)

        # 14. Detect Harmonics (post-hoc, last detection step)
        detect_harm_tip = ("DETECTION: Post-hoc harmonic detection.\n"
                           "After all filtering is complete, identifies detections\n"
                           "whose peak frequency is approximately 2x or 3x another\n"
                           "temporally overlapping detection (>= 50% overlap).\n\n"
                           "Harmonics are labeled (purple 'H' boxes) and excluded\n"
                           "from call counts, but preserved in the output CSV with\n"
                           "is_harmonic=True for downstream analysis.\n\n"
                           "Works best when Freq Gap splitting has already separated\n"
                           "fundamentals from harmonics into distinct connected\n"
                           "components. If they remain merged as one blob, there is\n"
                           "nothing to compare against.\n\n"
                           "Recommended ON for prairie voles (82% have harmonics)\n"
                           "and mice. Less critical for rat 22 kHz alarm calls.")
        self.chk_detect_harmonics = QCheckBox("Detect Harmonics")
        self.chk_detect_harmonics.setChecked(True)
        self.chk_detect_harmonics.setToolTip(detect_harm_tip)
        advanced_layout.addWidget(self.chk_detect_harmonics)

        # Post-hoc call-type classification (prairie vole only).
        # Shown/hidden by the profile dropdown handler; hidden by default.
        classify_tip = (
            "Classify each detection into one of 14 prairie vole call types\n"
            "(Ma et al. 2014): flat, upward_ramp, downward_ramp, harmonic,\n"
            "step_up, step_down, step_in_left/right/both, u_shape,\n"
            "inverted_u, miscellaneous, complex, step_composite.\n\n"
            "Rule-based heuristic using duration, bandwidth, and peak-freq\n"
            "contour samples. Adds a 'call_type' column to the CSV.\n"
            "Intended as a first-pass label — human review will be needed\n"
            "for edge cases, especially complex and step-in variants.")
        self.chk_classify_call_types = QCheckBox("Classify Call Types (Prairie Vole)")
        self.chk_classify_call_types.setChecked(True)
        self.chk_classify_call_types.setToolTip(classify_tip)
        self.chk_classify_call_types.setVisible(False)
        advanced_layout.addWidget(self.chk_classify_call_types)

        # --- Infrastructure settings ---

        # Adaptive (block-wise) noise floor
        adaptive_nf_tip = (
            "Recompute the per-freq noise floor in non-overlapping time blocks\n"
            "of N seconds, then linearly interpolate between blocks.\n"
            "Useful when background noise drifts over the recording\n"
            "(HVAC cycling, aging mic gain, etc.).\n\n"
            "  0 (unchecked) = single per-freq floor over the whole file (default).\n"
            "  30 s is a good starting value; longer = smoother, shorter = more adaptive."
        )
        adaptive_nf_row = QHBoxLayout()
        adaptive_nf_row.setSpacing(4)
        self.chk_adaptive_noise = QCheckBox("Adaptive noise floor:")
        self.chk_adaptive_noise.setChecked(False)
        self.chk_adaptive_noise.setToolTip(adaptive_nf_tip)
        self.chk_adaptive_noise.toggled.connect(
            lambda checked: self.spin_noise_block_s.setEnabled(checked)
        )
        adaptive_nf_row.addWidget(self.chk_adaptive_noise)
        self.spin_noise_block_s = QDoubleSpinBox()
        self.spin_noise_block_s.setRange(1.0, 600.0)
        self.spin_noise_block_s.setSingleStep(5.0)
        self.spin_noise_block_s.setValue(30.0)
        self.spin_noise_block_s.setSuffix(" s")
        self.spin_noise_block_s.setEnabled(False)
        self.spin_noise_block_s.setToolTip(adaptive_nf_tip)
        self.spin_noise_block_s.setFixedWidth(80)
        adaptive_nf_row.addWidget(self.spin_noise_block_s)
        adaptive_nf_row.addStretch()
        advanced_layout.addLayout(adaptive_nf_row)

        # FFT params
        fft_tip = ("FFT window size in samples (nperseg).\n"
                   "Controls the time-frequency resolution tradeoff:\n\n"
                   "  freq_resolution = sample_rate / nfft\n"
                   "  time_resolution = nfft / sample_rate\n\n"
                   "At 250 kHz sample rate:\n"
                   "  512 → 488 Hz/bin, 2.0 ms/frame (default)\n"
                   "  256 → 977 Hz/bin, 1.0 ms/frame\n"
                   "  1024 → 244 Hz/bin, 4.1 ms/frame\n\n"
                   "Larger = finer frequency detail but blurs short calls.\n"
                   "Smaller = sharper timing but coarser frequency bins.")
        overlap_tip = ("FFT overlap in samples (noverlap).\n"
                       "Number of samples shared between consecutive FFT\n"
                       "windows. Higher overlap = smoother time axis with\n"
                       "more columns in the spectrogram, at compute cost.\n\n"
                       "  hop_size = nperseg - noverlap\n"
                       "  time_step = hop_size / sample_rate\n\n"
                       "At 250 kHz with nperseg=512, noverlap=384:\n"
                       "  hop = 128 → 0.51 ms time step.\n\n"
                       "Typical: 75% of FFT size (384 for 512).")
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
        freq_samp_tip = ("Sample peak frequency at N evenly-spaced time points\n"
                         "across each detection. For each time segment, the\n"
                         "frequency bin with maximum power is recorded as\n"
                         "peak_freq_1 through peak_freq_N in the output CSV.\n\n"
                         "Enables:\n"
                         "  - Frequency contour visualization (cyan line overlay)\n"
                         "  - Sweep rate filtering (Max Sweep parameter)\n"
                         "  - Contour jitter filtering (Max Jitter parameter)\n"
                         "  - Call type classification (flat, up, down, chevron)\n\n"
                         "More samples = finer contour resolution but diminishing\n"
                         "returns above 5-7 for typical call durations.\n"
                         "Uncheck to skip contour analysis (faster detection).")
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

        # Filter Overlay button (after Detection Parameters section)
        overlay_tip = ("Overlay the thresholded detection mask on the spectrogram (shortcut: F).\n"
                       "Colored pixels = above threshold (signal the detector will analyze).\n"
                       "Black pixels = removed by preprocessing (band mask + noise floor + threshold).\n"
                       "Use this to tune Threshold and Noise %tile before running a batch.")
        self.btn_filter_overlay = QPushButton("Filter Overlay (F)")
        self.btn_filter_overlay.setCheckable(True)
        self.btn_filter_overlay.setToolTip(overlay_tip)
        self.btn_filter_overlay.toggled.connect(self._on_filter_overlay_toggled)
        group_layout.addWidget(self.btn_filter_overlay)

        self._selected_gpu_device = "auto"

        # Collect all DSP parameter widgets for profile enable/disable
        self._dsp_param_widgets = [
            self.spin_min_freq, self.spin_max_freq,
            self.spin_threshold,
            self.spin_min_dur, self.spin_max_dur,
            self.spin_max_bw, self.spin_tonality, self.spin_min_call_freq,
            self.chk_detect_harmonics,
            self.spin_freq_gap, self.spin_valley_ratio,
            self.spin_min_gap, self.spin_noise_pct,
            self.chk_adaptive_noise, self.spin_noise_block_s,
            self.spin_nperseg, self.spin_noverlap,
            self.chk_freq_samples, self.spin_freq_samples,
            # Advanced noise rejection
            self.spin_min_bw, self.spin_min_snr,
            self.spin_min_entropy, self.spin_max_entropy,
            self.spin_min_power, self.spin_max_power,
            self.spin_max_sweep, self.spin_max_jitter, self.spin_min_ici,
        ]

        # Connect filter-overlay-relevant spinners for live updates
        self.spin_min_freq.valueChanged.connect(self._push_filter_overlay_params)
        self.spin_max_freq.valueChanged.connect(self._push_filter_overlay_params)
        self.spin_threshold.valueChanged.connect(self._push_filter_overlay_params)
        self.spin_noise_pct.valueChanged.connect(self._push_filter_overlay_params)

        # Query Preview Snapshot button
        self.btn_preview_snapshot = QPushButton("Query Preview Snapshot (Q)")
        self.btn_preview_snapshot.setStyleSheet("background-color: #2d7d46;")
        self.btn_preview_snapshot.setToolTip(
            "Run the full DSP detection pipeline on only the currently\n"
            "visible spectrogram window (shortcut: Q).\n\n"
            "Uses all current parameter settings (both preprocessing\n"
            "and detection parameters).\n\n"
            "Existing detections within the visible time range are\n"
            "replaced; detections outside the window are preserved.\n\n"
            "Prints per-stage diagnostic counts to the terminal:\n"
            "  connected components → duration → merge → bandwidth\n"
            "  → tonality → entropy → power → harmonic → final\n\n"
            "Use this to rapidly iterate on parameter tuning without\n"
            "processing the entire file."
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

        # Silence-region noise calibration.
        calib_row = QHBoxLayout()
        calib_row.setSpacing(2)
        self.btn_calibrate_noise = QPushButton("Calibrate Noise from View")
        self.btn_calibrate_noise.setToolTip(
            "Capture a per-frequency-bin noise floor from the currently\n"
            "visible spectrogram window. Zoom to a quiet region first —\n"
            "the detector will use this floor in place of the percentile\n"
            "estimate until you Clear Override.\n\n"
            "Applies to both Preview Snapshot and Run Detection."
        )
        self.btn_calibrate_noise.clicked.connect(self._calibrate_noise_from_view)
        self.btn_calibrate_noise.setEnabled(False)
        calib_row.addWidget(self.btn_calibrate_noise, 1)
        self.btn_clear_noise_override = QPushButton("Clear")
        self.btn_clear_noise_override.setToolTip("Discard the captured noise floor and revert to the percentile estimate.")
        self.btn_clear_noise_override.clicked.connect(self._clear_noise_override)
        self.btn_clear_noise_override.setEnabled(False)
        calib_row.addWidget(self.btn_clear_noise_override)
        group_layout.addLayout(calib_row)

        self.lbl_noise_override = QLabel("Noise override: off")
        self.lbl_noise_override.setStyleSheet("color: #999999; font-size: 9px;")
        group_layout.addWidget(self.lbl_noise_override)

        # Pipeline Inspector button.
        self.btn_pipeline_inspector = QPushButton("Pipeline Inspector")
        self.btn_pipeline_inspector.setToolTip(
            "Open a matplotlib window showing each stage of the DSP pipeline\n"
            "for the currently visible spectrogram window:\n"
            "  1. Raw spectrogram (unfiltered)\n"
            "  2. Bandpassed spectrogram\n"
            "  3. Noise floor\n"
            "  4. Above-noise energy (Sxx - floor)\n"
            "  5. Binary mask after morphology\n"
            "  6. Connected components\n\n"
            "Useful for debugging why a call was or wasn't detected."
        )
        self.btn_pipeline_inspector.clicked.connect(self._open_pipeline_inspector)
        self.btn_pipeline_inspector.setEnabled(False)
        group_layout.addWidget(self.btn_pipeline_inspector)

        # Copy current DSP settings to the clipboard for sharing / troubleshooting.
        self.btn_copy_settings = QPushButton("Copy Settings to Clipboard")
        self.btn_copy_settings.setToolTip(
            "Copy all current DSP detection parameters to the clipboard as\n"
            "pretty-printed JSON. Paste into a chat to share your profile\n"
            "for troubleshooting or iteration."
        )
        self.btn_copy_settings.clicked.connect(self._copy_settings_to_clipboard)
        group_layout.addWidget(self.btn_copy_settings)

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

        # Queue display (after queue-button row)
        self.lbl_queue = QLabel("Queue: 0 files")
        self.lbl_queue.setStyleSheet("color: #999999;")
        group_layout.addWidget(self.lbl_queue)

        # GPU acceleration checkbox (directly above Run DSP Detection)
        gpu_row = QHBoxLayout()
        gpu_row.setSpacing(4)
        self.chk_gpu_accel = QCheckBox("Enable GPU Acceleration")
        self.chk_gpu_accel.setChecked(False)
        self.chk_gpu_accel.setToolTip(
            "Use GPU for spectrogram / STFT computation.\n"
            "Backends: CUDA (NVIDIA) or MPS (Apple Silicon) via PyTorch.\n"
            "Falls back silently to CPU if no compatible GPU is found.\n"
            "Benefit is significant on long files; negligible on short clips."
        )
        self.chk_gpu_accel.toggled.connect(self._on_gpu_toggle)
        gpu_row.addWidget(self.chk_gpu_accel)
        self.lbl_gpu_status = QLabel("")
        self.lbl_gpu_status.setStyleSheet("color: #999999; font-size: 9px;")
        gpu_row.addWidget(self.lbl_gpu_status)
        gpu_row.addStretch()
        group_layout.addLayout(gpu_row)

        # Run button
        self.btn_run_dsp = QPushButton("Run DSP Detection")
        self.btn_run_dsp.setStyleSheet("background-color: #0078d4;")
        self.btn_run_dsp.setToolTip(
            "Run DSP-based detection on all queued files.\n"
            "Writes <wav>_cad.csv next to each file.\n"
            "Safe to stop mid-batch — completed files stay;\n"
            "the currently processing file is discarded."
        )
        self.btn_run_dsp.clicked.connect(self.run_dsp_detection)
        self.btn_run_dsp.setEnabled(False)
        group_layout.addWidget(self.btn_run_dsp)

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

        self.btn_prev_det = QPushButton("< (B)")
        self.btn_prev_det.setObjectName("small_btn")
        self.btn_prev_det.setToolTip("Go to previous detection (B key)")
        self.btn_prev_det.clicked.connect(self.prev_detection)
        self.btn_prev_det.setEnabled(False)
        nav_row.addWidget(self.btn_prev_det)

        self.lbl_det_num = QLabel("Det 0/0")
        self.lbl_det_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_det_num, 1)

        self.btn_next_det = QPushButton("(N) >")
        self.btn_next_det.setObjectName("small_btn")
        self.btn_next_det.setToolTip("Go to next detection (N key)")
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
        _label_font = QFont()
        _label_font.setPointSize(10)

        self.btn_accept = QPushButton("Accept (A)")
        self.btn_accept.setFont(_label_font)
        self.btn_accept.setObjectName("accept_btn")
        self.btn_accept.setToolTip(
            "Mark current detection as a valid USV call (A key).\n"
            "Accepted calls are written to <wav>_cad.csv and can\n"
            "later be imported into DAD for model training."
        )
        self.btn_accept.clicked.connect(self.accept_detection)
        self.btn_accept.setEnabled(False)
        btn_row1.addWidget(self.btn_accept)

        self.btn_reject = QPushButton("Reject (R)")
        self.btn_reject.setFont(_label_font)
        self.btn_reject.setObjectName("reject_btn")
        self.btn_reject.setToolTip(
            "Mark current detection as a false positive (R key).\n"
            "Rejected calls are kept in the CSV as negative examples;\n"
            "they do not count toward your USV totals."
        )
        self.btn_reject.clicked.connect(self.reject_detection)
        self.btn_reject.setEnabled(False)
        btn_row1.addWidget(self.btn_reject)

        self.btn_harmonic = QPushButton("Harmonic (H)")
        self.btn_harmonic.setFont(_label_font)
        self.btn_harmonic.setStyleSheet("background-color: #6b3fa0;")
        self.btn_harmonic.clicked.connect(self.mark_harmonic)
        self.btn_harmonic.setEnabled(False)
        self.btn_harmonic.setToolTip("Mark current detection as a harmonic (H key).\n"
                                     "Sets status='accepted' and is_harmonic=True.\n"
                                     "Displayed as purple 'H' box. The is_harmonic\n"
                                     "column is independent of curation status:\n"
                                     "auto-detected harmonics remain 'pending' until\n"
                                     "manually reviewed.")
        btn_row1.addWidget(self.btn_harmonic)

        self.btn_skip = QPushButton("Skip (S)")
        self.btn_skip.setFont(_label_font)
        self.btn_skip.setStyleSheet("background-color: #5c5c5c;")
        self.btn_skip.clicked.connect(self.skip_detection)
        self.btn_skip.setEnabled(False)
        self.btn_skip.setToolTip("Skip to the next detection without changing\nits current status (S key).\nUseful for reviewing without modifying labels.")
        btn_row1.addWidget(self.btn_skip)

        import sys as _sys
        _undo_key = "⌘Z" if _sys.platform == "darwin" else "Ctrl+Z"
        self.btn_undo = QPushButton(f"Undo ({_undo_key})")
        self.btn_undo.setFont(_label_font)
        self.btn_undo.setStyleSheet("background-color: #5c5c5c;")
        self.btn_undo.clicked.connect(self.undo_action)
        self.btn_undo.setEnabled(False)
        self.btn_undo.setToolTip(
            "Undo last labeling action (Ctrl+Z).\n"
            "Redo with Ctrl+Shift+Z (or Ctrl+Y).\n"
            "Supports single and batch operations."
        )
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

        self.btn_add_usv = QPushButton("+ Add Label (P)")
        self.btn_add_usv.setStyleSheet("background-color: #6b4c9a;")
        self.btn_add_usv.setToolTip("Manually draw a new USV detection box on the spectrogram.\nClick and drag on the spectrogram to define the region.")
        self.btn_add_usv.clicked.connect(self.add_new_usv)
        self.btn_add_usv.setEnabled(False)
        btn_row2.addWidget(self.btn_add_usv)

        self.btn_delete = QPushButton("Delete (D)")
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
        self.btn_delete_all_labels.setToolTip(
            "Delete ALL detections (pending + accepted + rejected) for the current file,\n"
            "and remove its sibling CSV from disk. This CANNOT be undone by Undo;\n"
            "the only recovery is re-running DSP detection."
        )
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
        lbl_hint = QLabel(
            "Keys: A=Accept R=Reject H=Harmonic S=Skip D=Delete P=Add "
            "B/N=Prev/Next F=Filter Overlay Q=Preview Snapshot Space=Play "
            "Ctrl+Z=Undo Ctrl+Shift+Z=Redo"
        )
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
        self.btn_advanced_toggle.setText(f"{arrow} Detection Parameters")

    def _on_species_profile_changed(self, profile_name):
        """Handle species profile dropdown change."""
        # Look up in built-in profiles first, then custom
        preset = self.SPECIES_PROFILES.get(profile_name)
        if preset is None:
            preset = self._custom_profiles.get(profile_name)

        # Enable delete button only for custom profiles
        is_custom = profile_name in self._custom_profiles
        self.btn_delete_profile.setEnabled(is_custom)

        # Call-type classification is prairie-vole specific — show only there.
        is_prairie_vole = (profile_name == 'Prairie Vole USVs')
        self.chk_classify_call_types.setVisible(is_prairie_vole)
        if not is_prairie_vole:
            self.chk_classify_call_types.setChecked(False)

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
                self.spin_freq_gap, self.spin_valley_ratio,
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
            # Backward compat: old profiles may have harmonic_filter/harmonic_label
            detect_harm = preset.get('detect_harmonics',
                                     preset.get('harmonic_filter', True) or preset.get('harmonic_label', False))
            self.chk_detect_harmonics.setChecked(detect_harm)
            if is_prairie_vole:
                self.chk_classify_call_types.setChecked(
                    bool(preset.get('classify_call_types', False)))
            self.spin_freq_gap.setValue(preset.get('min_freq_gap_hz', 5000))
            self.spin_valley_ratio.setValue(preset.get('valley_split_ratio', 0.30))
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

            # Refresh filter overlay if active
            self._push_filter_overlay_params()

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

        # H = mark detection as harmonic
        sc_harmonic = QShortcut(QKeySequence(Qt.Key_H), self)
        sc_harmonic.setContext(Qt.ApplicationShortcut)
        sc_harmonic.activated.connect(self._shortcut_mark_harmonic)

        # V = toggle filter overlay
        sc_overlay = QShortcut(QKeySequence(Qt.Key_F), self)
        sc_overlay.setContext(Qt.ApplicationShortcut)
        sc_overlay.activated.connect(self._shortcut_toggle_filter_overlay)

        sc_snapshot = QShortcut(QKeySequence(Qt.Key_Q), self)
        sc_snapshot.setContext(Qt.ApplicationShortcut)
        sc_snapshot.activated.connect(self._shortcut_preview_snapshot)

    def _shortcut_preview_snapshot(self):
        """Handle Q key shortcut — run preview snapshot."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_preview_snapshot.isEnabled():
            self._flash_button(self.btn_preview_snapshot)
            self.run_preview_snapshot()

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

    def _shortcut_mark_harmonic(self):
        """Handle H key — mark detection as harmonic."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_harmonic.isEnabled():
            self._flash_button(self.btn_harmonic)
            self.mark_harmonic()

    def _shortcut_skip(self):
        """Handle S key — skip to next detection without changing status."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_skip.isEnabled():
            self._flash_button(self.btn_skip)
            self.skip_detection()

    def _shortcut_toggle_filter_overlay(self):
        """Handle V key — toggle filter overlay."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.btn_filter_overlay.toggle()

    def _on_filter_overlay_toggled(self, checked):
        """Handle filter overlay toggle."""
        if checked:
            self.btn_filter_overlay.setStyleSheet("background-color: #8b5cf6;")
            self._push_filter_overlay_params()
        else:
            self.btn_filter_overlay.setStyleSheet("")
            self.spectrogram.set_filter_overlay(False)

    def _push_filter_overlay_params(self):
        """Push current DSP filter params to the spectrogram overlay."""
        if not self.btn_filter_overlay.isChecked():
            return
        self.spectrogram.set_filter_overlay(True, {
            'min_freq_hz': self.spin_min_freq.value(),
            'max_freq_hz': self.spin_max_freq.value(),
            'noise_percentile': self.spin_noise_pct.value(),
            'energy_threshold_db': self.spin_threshold.value(),
        })

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
        # Track files whose CSVs use the legacy CAD suffix ``_usv_dsp`` so we
        # can offer a bulk rename to the new ``_cad`` convention. Only
        # ``_usv_dsp`` is treated as legacy CAD output; ``_usv_rf`` (Random
        # Forest) and ``_usv_detections`` (generic batch) are NOT auto-renamed.
        legacy_dsp = []

        for filepath in self.audio_files:
            if filepath in self.all_detections:
                continue
            base = Path(filepath).stem
            parent = Path(filepath).parent
            for suffix in ['_cad', '_usv_dsp', '_usv_rf', '_usv_detections']:
                csv_path = parent / f"{base}{suffix}.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if 'status' not in df.columns:
                            df['status'] = 'pending'
                        self._ensure_freq_bounds(df)
                        self.all_detections[filepath] = df
                        self.detection_sources[filepath] = suffix.lstrip('_')
                        if suffix == '_usv_dsp':
                            legacy_dsp.append((filepath, csv_path))
                        break
                    except Exception:
                        pass

        if legacy_dsp:
            self._offer_legacy_rename(legacy_dsp)

    def _offer_legacy_rename(self, legacy_files):
        """Prompt the user to rename legacy ``_usv_dsp.csv`` files to ``_cad.csv``."""
        n = len(legacy_files)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Legacy CSV filenames detected")
        box.setText(
            f"{n} file{'s' if n != 1 else ''} in the folder use the old "
            f"CAD naming convention (_usv_dsp.csv).\n\n"
            "Rename them to the new convention (_cad.csv)?"
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.Yes)
        if box.exec_() != QMessageBox.Yes:
            return

        renamed = 0
        skipped = 0
        errors = 0
        for filepath, old_path in legacy_files:
            new_path = old_path.with_name(old_path.name.replace('_usv_dsp.csv', '_cad.csv'))
            if new_path.exists():
                # Don't clobber an existing _cad.csv; leave the legacy in place.
                skipped += 1
                continue
            try:
                old_path.rename(new_path)
                self.detection_sources[filepath] = 'cad'
                renamed += 1
            except Exception as e:
                print(f"[CAD] Rename failed for {old_path}: {e}")
                errors += 1

        parts = [f"Renamed {renamed}"]
        if skipped:
            parts.append(f"{skipped} skipped (target exists)")
        if errors:
            parts.append(f"{errors} errors")
        self.status_bar.showMessage(" — ".join(parts))

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

            # Clear undo/redo history on file switch — indices refer to the prior file.
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.btn_undo.setEnabled(False)

            self._update_display()

            # Auto-adjust FFT settings for sample rate on first load
            # or when sample rate changes
            prev_sr = getattr(self, '_last_sample_rate', None)
            if prev_sr != self.sample_rate:
                self._auto_adjust_fft_for_sample_rate(self.sample_rate)
                self._last_sample_rate = self.sample_rate

            nyquist = self.sample_rate / 2
            self.status_bar.showMessage(
                f"Loaded: {os.path.basename(filepath)} "
                f"(SR: {self.sample_rate:,.0f} Hz, "
                f"Nyquist: {nyquist:,.0f} Hz)")

        except Exception as e:
            self.status_bar.showMessage(f"Error loading file: {e}")
            self.audio_data = None
        finally:
            QApplication.restoreOverrideCursor()

        self._update_ui_state()

    def _auto_adjust_fft_for_sample_rate(self, sample_rate):
        """Auto-adjust FFT parameters to maintain consistent frequency resolution.

        Targets ~500 Hz frequency resolution regardless of sample rate.
        Only adjusts if using "Manual" profile (doesn't override preset profiles).

        Sample rate → nperseg mapping (targeting ~500 Hz/bin):
            44100 Hz  → 128  (344 Hz/bin)
            96000 Hz  → 256  (375 Hz/bin)
            192000 Hz → 512  (375 Hz/bin)
            250000 Hz → 512  (488 Hz/bin)
            500000 Hz → 1024 (488 Hz/bin)

        Overlap is set to 75% of nperseg for good time resolution.
        """
        # Only auto-adjust in Manual mode
        profile = self.combo_species_profile.currentText()
        if profile != "Manual":
            return

        # Find nearest power-of-2 nperseg targeting ~500 Hz resolution
        target_resolution = 500.0  # Hz per bin
        ideal_nperseg = sample_rate / target_resolution
        # Round to nearest power of 2
        import math
        nperseg = int(2 ** round(math.log2(ideal_nperseg)))
        nperseg = max(64, min(2048, nperseg))  # Clamp to spinner range
        noverlap = int(nperseg * 0.75)

        actual_resolution = sample_rate / nperseg
        nyquist = sample_rate / 2

        self.spin_nperseg.blockSignals(True)
        self.spin_noverlap.blockSignals(True)
        self.spin_nperseg.setValue(nperseg)
        self.spin_noverlap.setValue(noverlap)
        self.spin_nperseg.blockSignals(False)
        self.spin_noverlap.blockSignals(False)

        print(f"[Auto FFT] SR={sample_rate:,.0f} Hz, Nyquist={nyquist:,.0f} Hz → "
              f"nperseg={nperseg}, overlap={noverlap}, "
              f"freq_resolution={actual_resolution:.0f} Hz/bin")

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
        for suffix in ['_cad', '_usv_dsp', '_usv_rf', '_usv_detections']:
            csv_path = parent / f"{base}{suffix}.csv"
            if csv_path.exists():
                try:
                    self.detections_df = pd.read_csv(csv_path)
                    if 'status' not in self.detections_df.columns:
                        self.detections_df['status'] = 'pending'
                    # Migrate legacy status='harmonic' to
                    # is_harmonic=True + status='pending'
                    if 'is_harmonic' not in self.detections_df.columns:
                        self.detections_df['is_harmonic'] = False
                    legacy_harm = self.detections_df['status'] == 'harmonic'
                    if legacy_harm.any():
                        self.detections_df.loc[legacy_harm, 'is_harmonic'] = True
                        self.detections_df.loc[legacy_harm, 'status'] = 'pending'
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
        for suffix in ['_cad', '_usv_dsp', '_usv_rf', '_usv_detections']:
            if (parent / f"{base}{suffix}.csv").exists():
                return True
        return False

    def _file_has_curated_detections(self, filepath):
        """Check if a file has any manually curated detections.

        A file is considered 'curated' if it has at least one detection
        with status 'accepted' or 'rejected'.  Auto-detected harmonics
        (is_harmonic=True with status='pending') do NOT count as curated
        since they haven't been manually reviewed.
        """
        df = self.all_detections.get(filepath)
        if df is not None and len(df) > 0 and 'status' in df.columns:
            curated = df[df['status'].isin(['accepted', 'rejected'])]
            return len(curated) > 0

        # Check on-disk CSV if not loaded in memory
        base = Path(filepath).stem
        parent = Path(filepath).parent
        for suffix in ['_cad', '_usv_dsp', '_usv_rf', '_usv_detections']:
            csv_path = parent / f"{base}{suffix}.csv"
            if csv_path.exists():
                try:
                    import pandas as pd
                    disk_df = pd.read_csv(csv_path)
                    if 'status' in disk_df.columns:
                        curated = disk_df[disk_df['status'].isin(['accepted', 'rejected'])]
                        return len(curated) > 0
                except Exception:
                    pass
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
            'detect_harmonics': self.chk_detect_harmonics.isChecked(),
            'classify_call_types': (
                self.chk_classify_call_types.isChecked()
                if hasattr(self, 'chk_classify_call_types') else False
            ),
            'min_freq_gap_hz': self.spin_freq_gap.value(),
            'valley_split_ratio': self.spin_valley_ratio.value(),
            'min_gap_ms': self.spin_min_gap.value(),
            'noise_percentile': self.spin_noise_pct.value(),
            'noise_block_seconds': (
                self.spin_noise_block_s.value() if self.chk_adaptive_noise.isChecked() else 0.0
            ),
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

    def _copy_settings_to_clipboard(self):
        """Copy current DSP detection settings to the clipboard as JSON."""
        cfg = self._gather_dsp_config()
        profile_name = None
        if hasattr(self, 'combo_profile'):
            profile_name = self.combo_profile.currentText()
        payload = {
            'profile': profile_name,
            'config': cfg,
            'noise_override_active': self._noise_override is not None,
        }
        try:
            text = json.dumps(payload, indent=2, sort_keys=True, default=float)
        except (TypeError, ValueError) as e:
            self.status_bar.showMessage(f"Copy failed: {e}")
            return
        QApplication.clipboard().setText(text)
        self.status_bar.showMessage(f"Copied {len(cfg)} DSP settings to clipboard")

    def _calibrate_noise_from_view(self):
        """Capture per-bin noise floor from the currently visible spectrogram window."""
        if self.audio_data is None or self.sample_rate is None:
            self.status_bar.showMessage("No audio loaded")
            return
        view_start, view_end = self.spectrogram.get_view_range()
        start_sample = max(0, int(view_start * self.sample_rate))
        end_sample = min(len(self.audio_data), int(view_end * self.sample_rate))
        if end_sample - start_sample < self.sample_rate * 0.05:
            self.status_bar.showMessage("Calibration window too short (need \u2265 50 ms)")
            return
        audio_segment = self.audio_data[start_sample:end_sample]
        try:
            from fnt.usv.usv_detector.spectrogram import bandpass_filter, compute_spectrogram_auto
        except ImportError as e:
            self.status_bar.showMessage(f"Calibration import failed: {e}")
            return
        cfg = self._gather_dsp_config()
        min_f = cfg.get('min_freq_hz', 20000)
        max_f = cfg.get('max_freq_hz', 65000)
        pct = float(cfg.get('noise_percentile', 25.0))
        filtered = bandpass_filter(audio_segment, self.sample_rate, min_f, max_f)
        frequencies, _times, Sxx_db = compute_spectrogram_auto(
            filtered, self.sample_rate,
            nperseg=cfg.get('nperseg', 512),
            noverlap=cfg.get('noverlap', 384),
            nfft=cfg.get('nfft', 1024),
            window=cfg.get('window_type', 'hann'),
            min_freq=min_f, max_freq=max_f,
            gpu_enabled=cfg.get('gpu_enabled', False),
            gpu_device=cfg.get('gpu_device', 'auto'),
        )
        if Sxx_db.size == 0:
            self.status_bar.showMessage("Calibration window produced empty spectrogram")
            return
        noise_db = np.percentile(Sxx_db, pct, axis=1)
        self._noise_override = {
            'freqs': np.asarray(frequencies, dtype=float),
            'noise_db': np.asarray(noise_db, dtype=float),
            'view_start_s': float(view_start),
            'view_end_s': float(view_end),
            'percentile': pct,
        }
        self._update_noise_override_label()
        self.btn_clear_noise_override.setEnabled(True)
        self.status_bar.showMessage(
            f"Noise floor captured: {len(frequencies)} bins "
            f"from {view_start:.2f}\u2013{view_end:.2f}s (p{pct:.0f})"
        )

    def _clear_noise_override(self):
        """Discard the captured noise override and revert to percentile estimate."""
        self._noise_override = None
        self.btn_clear_noise_override.setEnabled(False)
        self._update_noise_override_label()
        self.status_bar.showMessage("Noise override cleared")

    def _update_noise_override_label(self):
        """Refresh the override status label."""
        if not hasattr(self, 'lbl_noise_override'):
            return
        ov = self._noise_override
        if not ov:
            self.lbl_noise_override.setText("Noise override: off")
            self.lbl_noise_override.setStyleSheet("color: #999999; font-size: 9px;")
            return
        dur = ov.get('view_end_s', 0) - ov.get('view_start_s', 0)
        mean_db = float(np.mean(ov.get('noise_db', [0])))
        self.lbl_noise_override.setText(
            f"Noise override: {len(ov.get('freqs', []))} bins, "
            f"{dur:.2f}s window, mean {mean_db:.1f} dB"
        )
        self.lbl_noise_override.setStyleSheet("color: #8b5cf6; font-size: 9px;")

    def _open_pipeline_inspector(self):
        """Open the Pipeline Inspector dialog for the current view window."""
        if self.audio_data is None or self.sample_rate is None:
            self.status_bar.showMessage("No audio loaded")
            return
        try:
            dlg = PipelineInspectorDialog(
                audio=self.audio_data,
                sample_rate=self.sample_rate,
                view_range=self.spectrogram.get_view_range(),
                dsp_config=self._gather_dsp_config(),
                noise_override=self._noise_override,
                parent=self,
            )
        except Exception as e:
            self.status_bar.showMessage(f"Inspector failed: {e}")
            return
        dlg.show()
        # Keep a reference so the dialog isn't garbage-collected mid-display.
        self._pipeline_inspector = dlg

    def run_preview_snapshot(self):
        """Run DSP detection on only the currently visible preview window."""
        if self.audio_data is None or self.sample_rate is None:
            self.status_bar.showMessage("No audio loaded")
            return

        print("\n" + "=" * 60)
        print("  NEW PREVIEW SNAPSHOT")
        print("=" * 60)

        filepath = self.audio_files[self.current_file_idx]
        view_start, view_end = self.spectrogram.get_view_range()

        # Grow the analysis window so the noise floor matches what a full-file
        # run would compute. Without this, tight zooms dominated by calls yield
        # an inflated noise floor and miss detections the full run would find.
        file_dur = len(self.audio_data) / self.sample_rate
        cfg_preview = self._gather_dsp_config()
        chunk_dur = float(cfg_preview.get('chunk_duration_s', 10.0) or 10.0)
        min_analysis_dur = max(chunk_dur, view_end - view_start)
        view_mid = 0.5 * (view_start + view_end)
        analysis_start = max(0.0, view_mid - min_analysis_dur / 2.0)
        analysis_end = min(file_dur, analysis_start + min_analysis_dur)
        # If we hit the end of the file, back up the start so we keep full span
        if analysis_end - analysis_start < min_analysis_dur:
            analysis_start = max(0.0, analysis_end - min_analysis_dur)

        start_sample = max(0, int(analysis_start * self.sample_rate))
        end_sample = min(len(self.audio_data), int(analysis_end * self.sample_rate))
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
                detect_harmonics=cfg.get('detect_harmonics', True),
                classify_call_types=cfg.get('classify_call_types', False),
                min_freq_gap_hz=cfg.get('min_freq_gap_hz', 5000.0),
                valley_split_ratio=cfg.get('valley_split_ratio', 0.30),
                min_gap_ms=cfg.get('min_gap_ms', 5.0),
                noise_percentile=cfg.get('noise_percentile', 25.0),
                noise_block_seconds=cfg.get('noise_block_seconds', 0.0),
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
            if self._noise_override:
                detector.noise_override = self._noise_override

            seg_dur = len(audio_segment) / self.sample_rate
            print(f"\n{'='*60}", flush=True)
            print(f"  NEW PREVIEW SNAPSHOT", flush=True)
            print(f"{'='*60}", flush=True)
            print(f"[Preview Snapshot] view={view_start:.3f}-{view_end:.3f}s "
                  f"(analysis={analysis_start:.3f}-{analysis_end:.3f}s, "
                  f"grown by {(seg_dur - (view_end - view_start)):.2f}s "
                  f"to stabilize noise floor), "
                  f"samples={len(audio_segment)}, sr={self.sample_rate}, "
                  f"dur={seg_dur:.3f}s")
            print(f"[Preview Snapshot] Config: freq={detector_config.min_freq_hz}-"
                  f"{detector_config.max_freq_hz}Hz, thresh={detector_config.energy_threshold_db}dB, "
                  f"dur={detector_config.min_duration_ms}-{detector_config.max_duration_ms}ms, "
                  f"tonality={detector_config.min_tonality}, "
                  f"snr={detector_config.min_snr_db}dB, "
                  f"entropy={detector_config.min_spectral_entropy}-{detector_config.max_spectral_entropy}, "
                  f"min_bw={detector_config.min_bandwidth_hz}Hz, "
                  f"freq_gap={detector_config.min_freq_gap_hz}Hz, "
                  f"valley_split={detector_config.valley_split_ratio}")

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

            # Pipeline funnel — both to terminal (detailed) and to status bar (compact)
            funnel = []  # list of (label, count) tuples, drops-only
            def _stage(label, n_before, calls_after):
                """Record a stage only if it changed the count."""
                n_after = len(calls_after)
                if n_after != n_before:
                    funnel.append((label, n_after))

            n0 = 0
            calls = detector._detect_from_spectrogram(frequencies, times, Sxx_db)
            print(f"[Preview Snapshot] After threshold+connected components: {len(calls)}")
            funnel.append(("thresh", len(calls)))
            n0 = len(calls)

            calls = detector._filter_by_duration(calls)
            print(f"[Preview Snapshot] After duration filter: {len(calls)}")
            _stage("dur", n0, calls); n0 = len(calls)

            pre_merge = len(calls)
            calls = detector._merge_close_calls(calls)
            print(f"[Preview Snapshot] After merge close: {len(calls)} (merged {pre_merge - len(calls)})")
            _stage("merge", n0, calls); n0 = len(calls)

            calls = detector._filter_by_bandwidth(calls)
            print(f"[Preview Snapshot] After max bandwidth: {len(calls)}")
            _stage("maxBW", n0, calls); n0 = len(calls)

            calls = detector._filter_by_min_bandwidth(calls)
            print(f"[Preview Snapshot] After min bandwidth: {len(calls)}")
            _stage("minBW", n0, calls); n0 = len(calls)

            calls = detector._filter_by_snr(calls)
            print(f"[Preview Snapshot] After SNR filter: {len(calls)}")
            _stage("SNR", n0, calls); n0 = len(calls)

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
            _stage("tonal", n0, calls); n0 = len(calls)

            calls = detector._filter_by_spectral_entropy(calls)
            print(f"[Preview Snapshot] After spectral entropy ({detector_config.min_spectral_entropy}-{detector_config.max_spectral_entropy}): {len(calls)}")
            _stage("entropy", n0, calls); n0 = len(calls)

            calls = detector._filter_by_min_freq(calls)
            print(f"[Preview Snapshot] After min freq filter: {len(calls)}")
            _stage("minF", n0, calls); n0 = len(calls)

            calls = detector._filter_by_power(calls)
            print(f"[Preview Snapshot] After power filter: {len(calls)}")
            _stage("power", n0, calls); n0 = len(calls)

            if detector_config.detect_harmonics:
                calls = detector._label_harmonics(calls)
            n_harmonics_labeled = sum(1 for c in calls if c.get('is_harmonic', False))
            print(f"[Preview Snapshot] After harmonic labeling: {len(calls)} ({n_harmonics_labeled} labeled as harmonic)")

            # Peak freq sampling
            n_samples = getattr(detector_config, 'freq_samples', 5)
            if n_samples and n_samples > 0:
                calls = detector._sample_peak_frequencies(calls, frequencies, times, Sxx_db)

            calls = detector._compute_contour_features(calls)
            calls = detector._filter_by_sweep_rate(calls)
            _stage("sweep", n0, calls); n0 = len(calls)
            calls = detector._filter_by_contour_smoothness(calls)
            _stage("jitter", n0, calls); n0 = len(calls)
            calls = detector._filter_by_ici(calls)
            _stage("ICI", n0, calls); n0 = len(calls)

            # Post-hoc call-type classification (prairie vole 14-type scheme)
            if getattr(detector_config, 'classify_call_types', False):
                calls = detector._classify_call_types(calls)
                if calls:
                    type_counts = {}
                    for c in calls:
                        t = c.get('call_type', 'miscellaneous')
                        type_counts[t] = type_counts.get(t, 0) + 1
                    summary = ", ".join(f"{t}:{n}" for t, n in sorted(
                        type_counts.items(), key=lambda kv: -kv[1]))
                    print(f"[Preview Snapshot] Call types: {summary}")

            print(f"[Preview Snapshot] Final detections: {len(calls)}")
            if not funnel or funnel[-1][1] != len(calls):
                funnel.append(("final", len(calls)))

            # Offset all detection times by the analysis-window start,
            # not the view start — the pipeline ran on the grown window.
            for det in calls:
                det['start_seconds'] += analysis_start
                det['stop_seconds'] += analysis_start

            # Keep only detections overlapping the user's visible range.
            # This preserves the user's "preview = what I see" mental model
            # while still benefiting from the grown noise-floor estimate.
            n_total_analysis = len(calls)
            detections = [
                d for d in calls
                if d['stop_seconds'] > view_start and d['start_seconds'] < view_end
            ]
            n_det = len(detections)
            if n_total_analysis != n_det:
                print(f"[Preview Snapshot] In-view: {n_det} of {n_total_analysis} "
                      f"calls from grown analysis window")
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
                # All new detections start as pending; is_harmonic is a
                # separate boolean column (not a status value)
                new_df['status'] = 'pending'
                if 'is_harmonic' not in new_df.columns:
                    new_df['is_harmonic'] = False
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
            self.detection_sources[filepath] = 'cad'

            # Keep current_detection_idx valid without jumping the view
            if self.detections_df is not None and len(self.detections_df) > 0:
                self.current_detection_idx = min(self.current_detection_idx, len(self.detections_df) - 1)
                self.current_detection_idx = max(0, self.current_detection_idx)
            else:
                self.current_detection_idx = 0

            # Refresh display WITHOUT moving the spectrogram view
            self._update_display_no_scroll()
            # Compact drops-only pipeline funnel string, e.g. "thresh:45→dur:28→SNR:19→final:15"
            funnel_str = "→".join(f"{name}:{n}" for name, n in funnel) if funnel else "no survivors"
            if n_det > 0:
                n_harmonics = sum(1 for d in detections if d.get('is_harmonic', False))
                n_calls = n_det - n_harmonics
                harm_str = f" ({n_harmonics} harmonic)" if n_harmonics else ""
                self.status_bar.showMessage(
                    f"Preview snapshot: {n_calls} call{'s' if n_calls != 1 else ''}"
                    f"{harm_str} in {view_end - view_start:.1f}s window  |  {funnel_str}"
                )
            else:
                n_remaining = len(self.detections_df) if self.detections_df is not None else 0
                self.status_bar.showMessage(
                    f"Preview snapshot: no new detections in {view_end - view_start:.1f}s window "
                    f"({funnel_str}; {n_remaining} detections remain outside)"
                )

        except Exception as e:
            self.status_bar.showMessage(f"Preview snapshot error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_preview_snapshot.setEnabled(True)
            self.btn_preview_snapshot.setText("Query Preview Snapshot (Q)")
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
            # Categorize files with detections into curated vs uncurated
            files_curated = [f for f in files_with_dets
                             if self._file_has_curated_detections(f)]
            files_uncurated = [f for f in files_with_dets
                               if f not in files_curated]
            n_curated = len(files_curated)
            n_uncurated = len(files_uncurated)

            msg = (f"Queue: {n_total} file{'s' if n_total != 1 else ''}\n"
                   f"  • {n_without} without detections\n"
                   f"  • {n_uncurated} with only pending (uncurated) detections\n"
                   f"  • {n_curated} with manually curated detections "
                   f"(accepted/rejected)\n\n"
                   "How would you like to proceed?")
            box = QMessageBox(self)
            box.setWindowTitle("Existing Detections Found")
            box.setText(msg)
            btn_overwrite = box.addButton("Overwrite All", QMessageBox.AcceptRole)
            btn_skip_curated = box.addButton(
                "Protect Curated", QMessageBox.ActionRole)
            btn_skip = box.addButton("Skip All Existing", QMessageBox.ActionRole)
            box.addButton("Cancel", QMessageBox.RejectRole)
            box.exec_()

            clicked = box.clickedButton()
            if clicked == btn_skip:
                if not files_without_dets:
                    self.status_bar.showMessage(
                        "All queued files already have detections — nothing to process")
                    return
                self.dsp_queue = files_without_dets
            elif clicked == btn_skip_curated:
                # Keep files without detections + files with only pending detections
                safe_queue = files_without_dets + files_uncurated
                if not safe_queue:
                    self.status_bar.showMessage(
                        "All queued files have been manually curated — nothing to overwrite")
                    return
                skipped = n_curated
                self.dsp_queue = safe_queue
                if skipped > 0:
                    self.status_bar.showMessage(
                        f"Protecting {skipped} curated file{'s' if skipped != 1 else ''} — "
                        f"processing {len(safe_queue)} file{'s' if len(safe_queue) != 1 else ''}")
            elif clicked == btn_overwrite:
                pass  # Keep full queue
            else:
                return  # Cancel

        config = self._gather_dsp_config()

        # Start worker
        self.dsp_worker = DSPDetectionWorker(self.dsp_queue.copy(), config,
                                             noise_override=self._noise_override)
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
                        'max_power_db', 'mean_power_db', 'status', 'source',
                        'dsp_params_json']

        # Build DataFrame from detection list
        if isinstance(detections, pd.DataFrame):
            df = detections.copy()
        else:
            df = pd.DataFrame(detections)

        if len(df) > 0:
            if 'status' not in df.columns:
                # All detections start as pending; is_harmonic is a
                # separate boolean column (not a status value)
                df['status'] = 'pending'
            if 'is_harmonic' not in df.columns:
                df['is_harmonic'] = False
            if 'source' not in df.columns:
                df['source'] = 'dsp'
            if 'call_number' not in df.columns:
                df.insert(0, 'call_number', range(1, len(df) + 1))

            # Stamp the exact DSP config that produced these rows for reproducibility.
            if self.dsp_worker is not None:
                try:
                    df['dsp_params_json'] = json.dumps(
                        self.dsp_worker.config, sort_keys=True
                    )
                except (TypeError, ValueError):
                    pass
        else:
            df = pd.DataFrame(columns=std_columns)

        # Store in memory
        self.all_detections[filepath] = df
        self.detection_sources[filepath] = 'cad'

        # Write CSV immediately (crash-safe — results are on disk per file)
        base = Path(filepath).stem
        parent = Path(filepath).parent
        csv_path = parent / f"{base}_cad.csv"
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
        is_harmonic = det.get('is_harmonic', False)
        status_text = status.capitalize()
        if is_harmonic:
            status_text += " (Harmonic)"
        self.lbl_det_status.setText(f"Status: {status_text}")
        if is_harmonic:
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #b464ff;")
        elif status == 'accepted':
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
        is_harmonic = det.get('is_harmonic', False)
        status_text = status.capitalize()
        if is_harmonic:
            status_text += " (Harmonic)"
        self.lbl_det_status.setText(f"Status: {status_text}")
        if is_harmonic:
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #b464ff;")
        elif status == 'accepted':
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

    def _push_undo(self, entry):
        """Record a fresh user action on the undo stack and invalidate redo history."""
        self.undo_stack.append(entry)
        self.redo_stack.clear()
        self.btn_undo.setEnabled(True)

    def accept_detection(self):
        """Mark current detection as accepted (USV)."""
        if self.detections_df is None:
            return
        old_status = self.detections_df.at[self.current_detection_idx, 'status']
        self._push_undo(('label', self.current_detection_idx, old_status))
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
        self._push_undo(('label', self.current_detection_idx, old_status))
        self.detections_df.at[self.current_detection_idx, 'status'] = 'rejected'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self._auto_advance()

    def mark_harmonic(self):
        """Mark current detection as a harmonic.

        Sets status='accepted' (counts as curated) and is_harmonic=True.
        The is_harmonic flag is a separate boolean column independent of
        the curation status, so harmonics auto-detected by DSP stay
        'pending' until the user reviews them.
        """
        if self.detections_df is None:
            return
        old_status = self.detections_df.at[self.current_detection_idx, 'status']
        old_harmonic = self.detections_df.at[self.current_detection_idx, 'is_harmonic'] \
            if 'is_harmonic' in self.detections_df.columns else False
        self._push_undo(('harmonic', self.current_detection_idx, old_status, old_harmonic))
        self.detections_df.at[self.current_detection_idx, 'status'] = 'accepted'
        if 'is_harmonic' not in self.detections_df.columns:
            self.detections_df['is_harmonic'] = False
        self.detections_df.at[self.current_detection_idx, 'is_harmonic'] = True
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
            self.btn_accept.setText("Accept (A)")
            self.btn_reject.setText("Reject (R)")
            return

        counts = self.detections_df['status'].value_counts()
        n_accepted = counts.get('accepted', 0)
        n_rejected = counts.get('rejected', 0)

        self.btn_accept.setText(f"Accept {n_accepted} (A)")
        self.btn_reject.setText(f"Reject {n_rejected} (R)")

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
            source = self.detection_sources.get(filepath, 'cad')
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
            # Update the file list to reflect current detection counts
            self._refresh_file_list_items()

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
        # Use the tracked source suffix, default to _cad
        source = self.detection_sources.get(filepath, 'cad')
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
        self.btn_harmonic.setEnabled(bool(has_det))
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
        self.btn_calibrate_noise.setEnabled(bool(has_audio))
        self.btn_pipeline_inspector.setEnabled(bool(has_audio))

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

        # Ctrl+Shift+Z / Cmd+Shift+Z = Redo; Ctrl+Y = Redo (Windows convention)
        if key == Qt.Key_Z and (modifiers & Qt.ControlModifier) and (modifiers & Qt.ShiftModifier):
            self.redo_action()
            return
        if key == Qt.Key_Y and (modifiers & Qt.ControlModifier):
            self.redo_action()
            return
        # Ctrl+Z / Cmd+Z = Undo
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
        """Undo the last labeling action; push the inverse onto the redo stack."""
        if not self.undo_stack or self.detections_df is None:
            return

        action = self.undo_stack.pop()
        if action[0] == 'label':
            _, idx, old_status = action
            if 0 <= idx < len(self.detections_df):
                current_status = self.detections_df.at[idx, 'status']
                self.detections_df.at[idx, 'status'] = old_status
                self.redo_stack.append(('label', idx, current_status))
                self.current_detection_idx = idx
                self._store_current_detections()
                self._update_display()
                self._update_button_counts()
                self.status_bar.showMessage(f"Undid label change on detection {idx + 1}")
        elif action[0] == 'harmonic':
            _, idx, old_status, old_harmonic = action
            if 0 <= idx < len(self.detections_df):
                current_status = self.detections_df.at[idx, 'status']
                current_harmonic = (self.detections_df.at[idx, 'is_harmonic']
                                    if 'is_harmonic' in self.detections_df.columns else False)
                self.detections_df.at[idx, 'status'] = old_status
                if 'is_harmonic' in self.detections_df.columns:
                    self.detections_df.at[idx, 'is_harmonic'] = old_harmonic
                self.redo_stack.append(('harmonic', idx, current_status, current_harmonic))
                self.current_detection_idx = idx
                self._store_current_detections()
                self._update_display()
                self._update_button_counts()
                self.status_bar.showMessage(f"Undid harmonic label on detection {idx + 1}")
        elif action[0] == 'batch_label':
            _, indices, old_status = action
            # Capture the scalar status that was applied (uniform across the batch).
            applied_status = next(
                (self.detections_df.at[idx, 'status']
                 for idx in indices if 0 <= idx < len(self.detections_df)),
                None,
            )
            for idx in indices:
                if 0 <= idx < len(self.detections_df):
                    self.detections_df.at[idx, 'status'] = old_status
            if applied_status is not None:
                self.redo_stack.append(('batch_label', list(indices), applied_status))
            self._store_current_detections()
            self._update_display()
            self._update_button_counts()
            self.status_bar.showMessage(f"Undid batch label change on {len(indices)} detections")

        self.btn_undo.setEnabled(len(self.undo_stack) > 0)

    def redo_action(self):
        """Re-apply the most recently undone action."""
        if not self.redo_stack or self.detections_df is None:
            return

        action = self.redo_stack.pop()
        if action[0] == 'label':
            _, idx, new_status = action
            if 0 <= idx < len(self.detections_df):
                current_status = self.detections_df.at[idx, 'status']
                self.detections_df.at[idx, 'status'] = new_status
                self.undo_stack.append(('label', idx, current_status))
                self.current_detection_idx = idx
                self._store_current_detections()
                self._update_display()
                self._update_button_counts()
                self.status_bar.showMessage(f"Redid label change on detection {idx + 1}")
        elif action[0] == 'harmonic':
            _, idx, new_status, new_harmonic = action
            if 0 <= idx < len(self.detections_df):
                current_status = self.detections_df.at[idx, 'status']
                current_harmonic = (self.detections_df.at[idx, 'is_harmonic']
                                    if 'is_harmonic' in self.detections_df.columns else False)
                self.detections_df.at[idx, 'status'] = new_status
                if 'is_harmonic' not in self.detections_df.columns:
                    self.detections_df['is_harmonic'] = False
                self.detections_df.at[idx, 'is_harmonic'] = new_harmonic
                self.undo_stack.append(('harmonic', idx, current_status, current_harmonic))
                self.current_detection_idx = idx
                self._store_current_detections()
                self._update_display()
                self._update_button_counts()
                self.status_bar.showMessage(f"Redid harmonic label on detection {idx + 1}")
        elif action[0] == 'batch_label':
            _, indices, new_status = action
            pre_status = next(
                (self.detections_df.at[idx, 'status']
                 for idx in indices if 0 <= idx < len(self.detections_df)),
                None,
            )
            for idx in indices:
                if 0 <= idx < len(self.detections_df):
                    self.detections_df.at[idx, 'status'] = new_status
            if pre_status is not None:
                self.undo_stack.append(('batch_label', list(indices), pre_status))
            self._store_current_detections()
            self._update_display()
            self._update_button_counts()
            self.status_bar.showMessage(f"Redid batch label change on {len(indices)} detections")

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
        self._push_undo(('batch_label', list(self.detections_df[mask].index), 'pending'))
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
        self._push_undo(('batch_label', list(self.detections_df[mask].index), 'pending'))
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

    def _apply_dsp_config(self, cfg):
        """Push a DSP config dict (matching _gather_dsp_config output) into the UI widgets.

        Tolerates missing keys (leaves widget at its current value) and invalid values.
        Does NOT re-emit valueChanged signals for filter-overlay widgets until the end.
        """
        if not isinstance(cfg, dict):
            return

        def _set(widget, key, caster=None):
            if key not in cfg:
                return
            try:
                v = cfg[key]
                if caster is not None:
                    v = caster(v)
                widget.blockSignals(True)
                widget.setValue(v)
                widget.blockSignals(False)
            except (TypeError, ValueError):
                widget.blockSignals(False)

        _set(self.spin_min_freq, 'min_freq_hz', int)
        _set(self.spin_max_freq, 'max_freq_hz', int)
        _set(self.spin_threshold, 'energy_threshold_db', float)
        _set(self.spin_min_dur, 'min_duration_ms', float)
        _set(self.spin_max_dur, 'max_duration_ms', float)
        _set(self.spin_max_bw, 'max_bandwidth_hz', int)
        _set(self.spin_tonality, 'min_tonality', float)
        _set(self.spin_min_call_freq, 'min_call_freq_hz', int)
        _set(self.spin_freq_gap, 'min_freq_gap_hz', int)
        _set(self.spin_valley_ratio, 'valley_split_ratio', float)
        _set(self.spin_min_gap, 'min_gap_ms', float)
        _set(self.spin_noise_pct, 'noise_percentile', float)
        _set(self.spin_nperseg, 'nperseg', int)
        _set(self.spin_noverlap, 'noverlap', int)
        _set(self.spin_min_bw, 'min_bandwidth_hz', int)
        _set(self.spin_min_snr, 'min_snr_db', float)
        _set(self.spin_min_entropy, 'min_spectral_entropy', float)
        _set(self.spin_max_entropy, 'max_spectral_entropy', float)
        _set(self.spin_min_power, 'min_power_db', float)
        _set(self.spin_max_power, 'max_power_db', float)
        _set(self.spin_max_sweep, 'max_mean_sweep_rate', float)
        _set(self.spin_max_jitter, 'max_contour_jitter', float)
        _set(self.spin_min_ici, 'min_ici_ms', float)

        # Booleans / checkboxes
        if 'detect_harmonics' in cfg:
            try:
                self.chk_detect_harmonics.setChecked(bool(cfg['detect_harmonics']))
            except Exception:
                pass

        # freq_samples: stored as int; 0 means "disabled" in _gather_dsp_config
        if 'freq_samples' in cfg:
            try:
                n = int(cfg['freq_samples'])
                if n > 0:
                    self.chk_freq_samples.setChecked(True)
                    self.spin_freq_samples.setValue(n)
                else:
                    self.chk_freq_samples.setChecked(False)
            except (TypeError, ValueError):
                pass

        # noise_block_seconds: stored as float; 0.0 means "disabled".
        if 'noise_block_seconds' in cfg:
            try:
                s = float(cfg['noise_block_seconds'])
                if s > 0:
                    self.chk_adaptive_noise.setChecked(True)
                    self.spin_noise_block_s.setValue(s)
                else:
                    self.chk_adaptive_noise.setChecked(False)
            except (TypeError, ValueError):
                pass

        # Re-push filter-overlay-affecting values once so the overlay reflects them
        if hasattr(self, '_push_filter_overlay_params'):
            try:
                self._push_filter_overlay_params()
            except Exception:
                pass

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

        # Restore DSP parameters. Two cases:
        #   a) A non-Manual profile was active last session — re-select the
        #      profile; it pushes its own values and locks the widgets.
        #   b) Manual was active — push the saved DSP config dict into widgets.
        saved_profile = settings.value("dsp_profile", "Manual", type=str)
        if saved_profile and saved_profile != "Manual":
            idx = self.combo_species_profile.findText(saved_profile)
            if idx >= 0:
                # setCurrentText triggers _on_species_profile_changed which
                # applies the preset and locks the widgets.
                self.combo_species_profile.setCurrentText(saved_profile)
                return
            # If the saved profile name is no longer available (e.g. deleted
            # custom profile), fall through to restoring the raw DSP config.

        dsp_json = settings.value("dsp_config", "", type=str)
        if dsp_json:
            try:
                import json as _json
                cfg = _json.loads(dsp_json)
                self._apply_dsp_config(cfg)
            except Exception:
                pass

    def closeEvent(self, event):
        """Save settings on close."""
        settings = QSettings("FNT", "USVStudio")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("view_window", self.spin_view_window.value())
        settings.setValue("colormap", self.combo_colormap.currentText())

        # Persist DSP parameter state. Always save both the active profile name
        # and the raw config — if the user chose a non-Manual profile we'll
        # re-select it next launch; if they later delete that profile, the raw
        # config is still there as a fallback.
        try:
            import json as _json
            settings.setValue(
                "dsp_profile", self.combo_species_profile.currentText()
            )
            settings.setValue(
                "dsp_config", _json.dumps(self._gather_dsp_config())
            )
        except Exception:
            pass

        super().closeEvent(event)



# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    # HiDPI-safe scaling on Windows (125%/150% display scale). Must be set
    # before QApplication is constructed or the flag is ignored.
    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass
    app = QApplication(sys.argv)
    window = ClassicAudioDetectorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
