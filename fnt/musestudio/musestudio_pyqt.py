"""MuseStudio main window — connect to a Muse S Athena over BLE (via OpenMuse),
stream EEG/optics over LSL, live-plot the signals, and record them to CSV.

Brings together the pieces: live plots + numeric channel table, battery
readout, synchronized webcam capture, a binaural-beat generator with
closed-loop control from interhemispheric synchrony (PLV), a head-map + meter,
and a guided session runner (free record or a timed protocol).
"""

import csv
import math
import os
import time
from datetime import datetime

from PyQt5.QtCore import Qt, QSettings, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from fnt.musestudio import theme
from fnt.musestudio.analysis import BandPowerAnalyzer, HemodynamicsAnalyzer
from fnt.musestudio.binaural import (
    BinauralPanel, play_alert, play_complete, play_cue, play_resolved,
)
from fnt.musestudio.channel_table import LiveValuesPanel
from fnt.musestudio.dsp import curated_channels
from fnt.musestudio.live_plot import LiveSignalView
from fnt.musestudio.log_view import LogDialog
from fnt.musestudio.logbuffer import LOG
from fnt.musestudio.muse_stream import (
    LSLReaderThread, MuseRecorder, MuseStreamProcess, find_devices,
)
from fnt.musestudio.neuro_widgets import NeuroControls, NeuroView
from fnt.musestudio.protocol import PROTOCOLS, ProtocolRunner
from fnt.musestudio.questionnaire import QuestionnaireDialog
from fnt.musestudio.recording import RecordingSession, SessionLogger
from fnt.musestudio.review_view import ReviewPanel
from fnt.musestudio.session_summary import SessionSummaryDialog
from fnt.musestudio.speech import Speaker, list_voices
from fnt.musestudio.synchrony import SynchronyAnalyzer
from fnt.musestudio.viz import (
    BandHistoryPlot, BandPowerBars, LateralityView, SpectrogramView,
)
from fnt.musestudio.webcam import WebcamThread, list_cameras


def _fmt_time(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:d}:{seconds % 60:02d}"


# Sync-gate: advance once calibrated synchrony level holds above this for this long.
_SYNC_GATE_LEVEL = 0.6
_SYNC_GATE_HOLD_S = 10.0

# Stream watchdog: a stream is "stalled" after this long with no samples, and
# the failure is escalated if it stays down. Bluetooth hiccups of a few hundred
# ms are normal; multiple seconds means data is being lost.
_STALL_SECONDS = 2.5
_STALL_ESCALATE_S = 15.0


def _is_eeg(stream_name):
    return "EEG" in stream_name.upper()


def _is_battery(stream_name):
    return "BATTERY" in stream_name.upper()


def _is_optics(stream_name):
    """The Athena's fNIRS/PPG stream is published as Muse-OPTICS."""
    return "OPTIC" in stream_name.upper() or "NIRS" in stream_name.upper()


def _default_recording_dir():
    """Prefer the repo's gitignored LocalData/Muse when running from source."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))       # …/fnt
    local = os.path.join(repo, "LocalData", "Muse")
    if os.path.isdir(local):
        return local
    return os.path.join(os.path.expanduser("~"), "Documents")


def _short(stream_name):
    """"Muse-EEG (695A…)" -> "EEG" for status messages."""
    import re
    s = re.sub(r"\s*\(.*\)\s*$", "", str(stream_name))
    return re.sub(r"^Muse[-_]?", "", s) or s


_local_clock = None


def _lsl_now():
    """Current LSL clock value (shared with Muse/webcam timestamps)."""
    global _local_clock
    if _local_clock is None:
        from mne_lsl.lsl import local_clock
        _local_clock = local_clock
    return _local_clock()


class _ScanThread(QThread):
    """Runs ``OpenMuse find`` off the GUI thread."""
    result = pyqtSignal(list, str)
    failed = pyqtSignal(str)

    def run(self):
        try:
            devices, raw = find_devices()
            self.result.emit(devices, raw)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class SessionBanner(QFrame):
    """Prominent guided-session strip: phase name, instruction, countdown,
    progress, and a Continue button for wait-for-user phases."""

    continue_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "background-color: #17303f; border: 1px solid #2a6f97; border-radius: 6px;"
        )
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.title = QLabel("")
        self.title.setFont(QFont("Arial", 12, QFont.Bold))
        self.title.setStyleSheet("color: #7fd4ff; border: none;")
        top.addWidget(self.title)
        top.addStretch()
        self.countdown = QLabel("")
        self.countdown.setFont(QFont("Arial", 16, QFont.Bold))
        self.countdown.setStyleSheet("color: #ffffff; border: none;")
        top.addWidget(self.countdown)
        lay.addLayout(top)

        self.instruction = QLabel("")
        self.instruction.setWordWrap(True)
        self.instruction.setMinimumHeight(46)
        self.instruction.setStyleSheet("color: #dddddd; border: none;")
        lay.addWidget(self.instruction)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        bottom.addWidget(self.progress, stretch=1)
        self.continue_btn = QPushButton("Continue")
        self.continue_btn.clicked.connect(self.continue_clicked.emit)
        bottom.addWidget(self.continue_btn)
        lay.addLayout(bottom)
        self.hide()

    def show_free(self, recording=True):
        if recording:
            self.title.setText("Free recording")
            self.instruction.setText(
                "Recording all active streams (EEG, optics, webcam, audio events). "
                "Press Stop when you are finished."
            )
        else:
            self.title.setText("Free monitor")
            self.instruction.setText(
                "Live monitoring — nothing is being saved. Press Stop when finished."
            )
        self.progress.setRange(0, 0)  # busy indicator
        self.continue_btn.hide()
        self.show()

    def set_elapsed(self, seconds):
        self.countdown.setText(_fmt_time(math.floor(seconds)))

    def show_phase(self, phase, index, total, waiting):
        self.title.setText(f"Phase {index + 1}/{total} · {phase.name}")
        self.instruction.setText(phase.instruction)
        self.progress.setRange(0, total)
        self.progress.setValue(index + 1)
        self.countdown.setText("" if waiting else "")
        self.continue_btn.setVisible(waiting)
        self.show()

    def set_countdown(self, remaining):
        # Ceil so the display shows the seconds remaining and ticks evenly.
        self.countdown.setText(_fmt_time(math.ceil(remaining)))


class MuseStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Start capturing diagnostics (including native stdout/stderr from
        # liblsl and OpenCV) before anything else can fail.
        LOG.install()
        self._log_dialog = None
        self.stream_proc = None
        self.reader = None
        self.recorder = None
        self.scan_thread = None
        self.webcam = None
        self._audio_file = None
        self._audio_writer = None
        self._sync_file = None
        self._sync_writer = None
        self._event_file = None
        self._event_writer = None
        self._eeg_stream = None
        self._eeg_channels = []
        self._optics_channels = []
        self._device_address = None
        self._session_active = False
        self._recording_enabled = False
        self._free_start = 0.0
        self._current_phase = None
        self._sync_hold_start = None
        self.session = None            # active RecordingSession
        self.logger = SessionLogger()  # buffers GUI actions from window open

        # Default recording location persists across launches.
        self._settings = QSettings("FNT", "MuseStudio")
        self.output_dir = self._settings.value("recording_dir", _default_recording_dir())

        self.setWindowTitle("MuseStudio - FieldNeuroethologyToolbox")
        self.resize(1480, 900)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(theme.STYLESHEET)
        self._init_ui()

        # Guided-protocol runner and a free-record elapsed timer.
        self.runner = ProtocolRunner(self)
        self.runner.phase_started.connect(self._on_phase_started)
        self.runner.tick.connect(self._on_runner_tick)
        self.runner.finished.connect(self._on_runner_finished)
        self.runner.aborted.connect(self._on_runner_aborted)
        self._free_timer = QTimer(self)
        self._free_timer.setInterval(500)
        self._free_timer.timeout.connect(self._on_free_tick)

        # Watchdog for silently-dropped streams (Bluetooth/LSL dropouts would
        # otherwise leave gaps in a recording with nothing to show for it).
        self._last_sample = {}      # stream -> monotonic time of last chunk
        self._stalled = {}          # stream -> monotonic time the stall began
        self._escalated = {}        # stream -> already warned a second time
        self._dropouts = []         # (stream, start_wall, seconds) for the summary
        self._stall_timer = QTimer(self)
        self._stall_timer.setInterval(1000)
        self._stall_timer.timeout.connect(self._check_streams)
        self._stall_timer.start()

    # ------------------------------------------------------------------ UI
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

        # Experimental banner (full width, top).
        banner = QLabel(
            "⚠  Muse S Athena via OpenMuse (BLE). Decoding — especially fNIRS — is "
            "reverse-engineered and experimental; not affiliated with InteraXon."
        )
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "background-color: #4a3a10; color: #ffcc66; border: 1px solid #6a5520;"
            " border-radius: 4px; padding: 6px;"
        )
        root.addWidget(banner)

        main_split = QSplitter(Qt.Horizontal)
        self.left_scroll = self._build_left_column()
        # Thin always-visible rail to fold the control column away.
        self.left_toggle = QPushButton("◂")
        self.left_toggle.setCheckable(True)
        self.left_toggle.setChecked(True)
        self.left_toggle.setFixedWidth(18)
        self.left_toggle.setToolTip(
            "Collapse the controls column to give the data views the full window.")
        self.left_toggle.toggled.connect(self._on_left_toggled)
        left_wrap = QWidget()
        lw = QHBoxLayout(left_wrap)
        lw.setContentsMargins(0, 0, 0, 0)
        lw.setSpacing(0)
        lw.addWidget(self.left_scroll, stretch=1)
        lw.addWidget(self.left_toggle)

        main_split.addWidget(left_wrap)
        main_split.addWidget(self._build_right_view())
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([380, 1100])
        root.addWidget(main_split, stretch=1)

        self._wire_analyzer()
        self._init_speech()
        self._on_view_changed()   # apply initial Live view

    def _build_left_column(self):
        """Scrolling column of controls (like MAD / Mask Tracker)."""
        left = QWidget()
        col = QVBoxLayout(left)
        col.setContentsMargins(4, 4, 4, 4)
        col.setSpacing(8)

        # View selector at the very top.
        view_group = QGroupBox("View")
        vg = QHBoxLayout(view_group)
        self.live_radio = QRadioButton("Live View")
        self.live_radio.setChecked(True)
        self.live_radio.setToolTip("Stream data and play beats freely — nothing is saved.")
        self.rec_radio = QRadioButton("Recording View")
        self.rec_radio.setToolTip("Run a protocol or free recording and save it to disk.")
        self.live_radio.toggled.connect(self._on_view_changed)
        vg.addWidget(self.live_radio)
        vg.addWidget(self.rec_radio)
        vg.addStretch()
        col.addWidget(view_group)

        # Muse connection.
        muse_group = QGroupBox("Muse")
        mg = QVBoxLayout(muse_group)
        self.device_combo = QComboBox()
        self.device_combo.addItem("No devices — click Scan", None)
        self.device_combo.setToolTip("Muse devices found by the last scan. Pick one, then Connect.")
        mg.addWidget(self.device_combo)
        row = QHBoxLayout()
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self.on_scan)
        self.scan_btn.setToolTip("Search over Bluetooth for nearby Muse headbands.")
        row.addWidget(self.scan_btn)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect)
        self.connect_btn.setToolTip("Start streaming EEG/optics from the selected Muse.")
        row.addWidget(self.connect_btn)
        mg.addLayout(row)
        self.battery_label = QLabel("Battery: —")
        self.battery_label.setStyleSheet("color: #cccccc; font-weight: bold;")
        self.battery_label.setToolTip("Muse battery level (updates every few seconds).")
        mg.addWidget(self.battery_label)
        col.addWidget(muse_group)

        # Webcam.
        cam_group = QGroupBox("Webcam")
        cg = QVBoxLayout(cam_group)
        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip("Detected cameras. When on, video is recorded during a session.")
        self._populate_cameras()
        cg.addWidget(self.camera_combo)
        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.clicked.connect(self.on_toggle_camera)
        self.camera_btn.setToolTip("Turn the webcam preview on/off.")
        cg.addWidget(self.camera_btn)
        col.addWidget(cam_group)

        # Binaural generator (available in both views).
        self.binaural = BinauralPanel()
        self.binaural.tone_event.connect(self._on_audio_event)
        self.binaural.tone_event.connect(self._log_audio_event)
        col.addWidget(self.binaural)

        # Neurofeedback controls (band + calibrate).
        self.neuro_controls = NeuroControls()
        col.addWidget(self.neuro_controls)

        # Spoken guidance — the only way a guided protocol can guide you once
        # your eyes are closed.
        guide = QGroupBox("Spoken guidance")
        gg = QVBoxLayout(guide)
        self.speak_check = QCheckBox("Speak instructions")
        self.speak_check.setToolTip(
            "Read each protocol phase aloud when it begins, and announce "
            "problems and completion.\n\n"
            "Essential for eyes-closed sessions: the on-screen instructions and "
            "countdown are invisible once you settle. The binaural tone is "
            "automatically ducked while speaking so the voice stays clear."
        )
        self.speak_check.setChecked(self._settings.value("speech_enabled", True, type=bool))
        self.speak_check.toggled.connect(self._on_speech_toggled)
        gg.addWidget(self.speak_check)

        vrow = QHBoxLayout()
        vrow.addWidget(QLabel("Voice"))
        self.voice_combo = QComboBox()
        self.voice_combo.setToolTip(
            "Which system voice reads the instructions. Calmer, slower voices "
            "suit meditation protocols better.")
        vrow.addWidget(self.voice_combo, stretch=1)
        gg.addLayout(vrow)

        self.test_voice_btn = QPushButton("Test voice")
        self.test_voice_btn.setToolTip("Speak a sample line so you can set the volume before starting.")
        self.test_voice_btn.clicked.connect(self._on_test_voice)
        gg.addWidget(self.test_voice_btn)
        col.addWidget(guide)
        self._guidance_group = guide

        # Recording controls (shown only in Recording View).
        self.recording_group = QGroupBox("Recording")
        rg = QVBoxLayout(self.recording_group)
        rg.addWidget(QLabel("Recording protocol"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Free", "free")
        for key, proto in PROTOCOLS.items():
            self.mode_combo.addItem(proto.name, key)
        self.mode_combo.setToolTip(
            "Free: capture live streams until you press Stop.\n"
            "Protocols: a guided, timed trial with on-screen instructions."
        )
        rg.addWidget(self.mode_combo)
        brow = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.on_start_or_stop)
        self.start_btn.setToolTip("Run the selected session live, without saving any files.")
        brow.addWidget(self.start_btn)
        self.record_btn = QPushButton("Start + Record")
        self.record_btn.clicked.connect(lambda: self.on_start_session(record=True))
        self.record_btn.setToolTip("Run the selected session and save it to the recording folder.")
        brow.addWidget(self.record_btn)
        rg.addLayout(brow)
        rg.addWidget(QLabel("Default recording location:"))
        loc_row = QHBoxLayout()
        self.folder_label = QLabel(self.output_dir)
        self.folder_label.setStyleSheet("color: #999999;")
        self.folder_label.setToolTip(self.output_dir)
        self.folder_label.setWordWrap(True)
        loc_row.addWidget(self.folder_label, stretch=1)
        self.folder_btn = QPushButton("…")
        self.folder_btn.setFixedWidth(36)
        self.folder_btn.clicked.connect(self.on_choose_folder)
        self.folder_btn.setToolTip("Choose the default folder where each recording is created.")
        loc_row.addWidget(self.folder_btn)
        rg.addLayout(loc_row)
        self.review_btn = QPushButton("Review a recording…")
        self.review_btn.clicked.connect(self.on_review_recording)
        self.review_btn.setToolTip(
            "Open a finished recording and compare each protocol phase "
            "against baseline.")
        rg.addWidget(self.review_btn)
        col.insertWidget(1, self.recording_group)   # directly under the View group

        col.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(left)

        # Diagnostics is pinned *below* the scroll area rather than inside it,
        # so the troubleshooting buttons are reachable without scrolling —
        # which is exactly when you need them.
        diag = QGroupBox("Diagnostics")
        dg = QVBoxLayout(diag)
        dg.setSpacing(4)
        self.logs_btn = QPushButton("Session Logs")
        self.logs_btn.clicked.connect(self.on_show_logs)
        self.logs_btn.setToolTip(
            "Open the diagnostic log: errors, exceptions, Bluetooth/LSL stream\n"
            "messages and camera problems, including output from the underlying\n"
            "C libraries that would otherwise only appear in the terminal."
        )
        dg.addWidget(self.logs_btn)
        self.copy_logs_btn = QPushButton("Copy Logs to Clipboard")
        self.copy_logs_btn.clicked.connect(self.on_copy_logs)
        self.copy_logs_btn.setToolTip(
            "Copy the whole log plus platform and package versions to the\n"
            "clipboard — paste this when reporting a problem."
        )
        dg.addWidget(self.copy_logs_btn)

        container = QWidget()
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        cl.addWidget(scroll, stretch=1)
        cl.addWidget(diag)
        container.setMinimumWidth(330)
        container.setMaximumWidth(440)
        return container

    def _build_right_view(self):
        """Data view: session banner + tabbed views, with the raw signal first."""
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(4, 4, 4, 4)
        rlay.setSpacing(6)

        self.session_banner = SessionBanner()
        self.session_banner.continue_clicked.connect(self._on_continue)
        rlay.addWidget(self.session_banner)

        self.view_tabs = QTabWidget()
        self._live_tab = self._build_live_tab()
        self.view_tabs.addTab(self._live_tab, "Live signal")
        self.view_tabs.addTab(self._build_bands_tab(), "Bands")
        self.view_tabs.addTab(self._build_spectrogram_tab(), "Spectrogram")
        self.view_tabs.addTab(self._build_synchrony_tab(), "Synchrony")
        self.view_tabs.addTab(self._build_camera_tab(), "Camera")
        self.review = ReviewPanel()
        self.view_tabs.addTab(self.review, "Review")

        tips = [
            ("Live signal", "Raw waveforms as they arrive, at a fixed µV scale.\n"
                            "Use this to check electrode contact and watch the rhythm directly."),
            ("Bands", "How power is split across delta/theta/alpha/beta/gamma right now\n"
                      "and over the session, plus left/right hemisphere comparison."),
            ("Spectrogram", "Time–frequency map of one electrode. A steady bright band\n"
                            "near 10 Hz is an alpha rhythm; broad high-frequency haze is\n"
                            "usually muscle tension."),
            ("Synchrony", "Interhemispheric phase locking (PLV) between the left/right\n"
                          "electrode pairs — the measure the protocols try to move."),
            ("Camera", "Webcam preview. Recorded with frame timestamps during a session."),
            ("Review", "Open a finished recording and compare each protocol phase\n"
                       "against its baseline."),
        ]
        for i, (_, tip) in enumerate(tips):
            self.view_tabs.setTabToolTip(i, tip)
        self.view_tabs.currentChanged.connect(self._on_tab_changed)
        rlay.addWidget(self.view_tabs, stretch=1)

        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        rlay.addWidget(self.status_label)
        return right

    def _build_live_tab(self):
        """Primary view: raw signals at a real µV scale, plus the value table."""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(6, 8, 6, 6)
        lay.setSpacing(4)

        # Channel-set toggle: curated (the channels worth watching) vs every channel.
        top = QHBoxLayout()
        top.addWidget(QLabel("Channels"))
        self.channel_set_combo = QComboBox()
        self.channel_set_combo.addItem("Curated", "curated")
        self.channel_set_combo.addItem("All channels", "all")
        self.channel_set_combo.setToolTip(
            "Curated: the channels worth watching — the 4 scalp electrodes, the\n"
            "accelerometer, and the outer (cortex-reaching) optical detectors at\n"
            "the two hemodynamic wavelengths.\n\n"
            "All channels: everything the headband sends, including unconnected\n"
            "AUX inputs, gyroscope, inner (scalp) optodes and ambient-light\n"
            "references. More lanes means each one is drawn smaller.\n\n"
            "This only changes what is displayed — recording always captures\n"
            "every channel."
        )
        self.channel_set_combo.activated.connect(lambda _i: self._apply_channel_set())
        top.addWidget(self.channel_set_combo)
        top.addStretch()
        self.values_toggle = QPushButton("Values ▸")
        self.values_toggle.setCheckable(True)
        self.values_toggle.setChecked(True)
        self.values_toggle.setToolTip(
            "Fold the channel list away to give the signal plots the full width.")
        self.values_toggle.toggled.connect(self._on_values_toggled)
        top.addWidget(self.values_toggle)
        lay.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        self.plot = LiveSignalView()
        self.plot.setToolTip(
            "Live signals, newest on the right. EEG uses a fixed µV scale and a "
            "display-only 1–40 Hz filter; recording always saves raw data."
        )
        split.addWidget(self.plot)
        self.values_panel = LiveValuesPanel()
        self.values_panel.setToolTip(
            "Latest value of every channel, and which channels are drawn.\n"
            "Untick a channel to remove it from the plots; hover a channel name "
            "to see what it measures."
        )
        self.values_panel.setMaximumWidth(320)
        self.values_panel.visibility_changed.connect(self.plot.set_channel_visibility)
        split.addWidget(self.values_panel)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        split.setSizes([900, 280])
        lay.addWidget(split, stretch=1)
        return page

    def _on_values_toggled(self, shown):
        self.values_panel.setVisible(shown)
        self.values_toggle.setText("Values ▸" if shown else "Values ◂")

    def _apply_channel_set(self):
        """Apply the curated/all channel preset to plots and checkboxes."""
        mode = self.channel_set_combo.currentData()
        for stream, names in self.plot.streams().items():
            visible = (curated_channels(stream, names) if mode == "curated"
                       else list(names))
            self.values_panel.set_visible_channels(stream, visible)
            self.plot.set_channel_visibility(stream, visible)
        self._log(f"Channel set -> {mode}")

    def _build_bands_tab(self):
        """Band power now (bars) and over time (history), plus L/R laterality."""
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(6, 8, 6, 6)
        split = QSplitter(Qt.Vertical)

        top = QSplitter(Qt.Horizontal)
        self.band_bars = BandPowerBars()
        self.band_bars.setToolTip(
            "Share of total power in each band, averaged across electrodes.")
        top.addWidget(self.band_bars)
        self.band_history = BandHistoryPlot()
        self.band_history.setToolTip("How the band mix has evolved over the session.")
        top.addWidget(self.band_history)
        top.setSizes([420, 640])
        split.addWidget(top)

        self.laterality = LateralityView()
        self.laterality.setToolTip(
            "Left vs right comparison. EEG alpha asymmetry: alpha is inversely "
            "related to activation, so more alpha on a side means that side is "
            "relatively less engaged. fNIRS ΔOD is an uncalibrated proxy for "
            "blood-volume change."
        )
        split.addWidget(self.laterality)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 170])
        lay.addWidget(split)
        return page

    def _build_spectrogram_tab(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(6, 8, 6, 6)
        self.spectrogram = SpectrogramView()
        self.spectrogram.setToolTip(
            "Rolling time–frequency map of one electrode. A steady bright band "
            "around 10 Hz is an alpha rhythm."
        )
        self.spectrogram.channel_combo.activated.connect(
            lambda i: self.bands.set_spectrogram_channel(
                self.spectrogram.channel_combo.itemData(i) or 0)
        )
        lay.addWidget(self.spectrogram)
        return page

    def _build_synchrony_tab(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(6, 8, 6, 6)
        self.neuro_view = NeuroView()
        lay.addWidget(self.neuro_view)
        return page

    def _build_camera_tab(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(6, 8, 6, 6)
        self.camera_view = QLabel("Camera off")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(240, 200)
        self.camera_view.setStyleSheet(
            f"background-color: {theme.PLOT_BG}; color: {theme.TEXT_FAINT};"
            f" border: 1px solid {theme.BORDER}; border-radius: 6px;"
        )
        self.camera_view.setToolTip(
            "Live webcam preview. Recorded with frame timestamps during a session.")
        lay.addWidget(self.camera_view)
        return page

    def _wire_analyzer(self):
        # Frequency-domain analysis -> bands tab + spectrogram.
        self.bands = BandPowerAnalyzer(self)
        self.bands.updated.connect(self._on_band_metrics)
        self.bands.spectrogram_updated.connect(self.spectrogram.update_spectrogram)

        # Optics -> fNIRS laterality (uncalibrated ΔOD proxy).
        self.hemo = HemodynamicsAnalyzer(self)
        self.hemo.updated.connect(self.laterality.update_hemo)

        self.analyzer = SynchronyAnalyzer(self)
        self.analyzer.metrics_updated.connect(self._on_metrics)
        self.analyzer.status.connect(self.neuro_controls.set_status)
        self.analyzer.baseline_progress.connect(
            lambda f: self.neuro_controls.set_status(f"Calibrating baseline… {int(f * 100)}%")
        )
        self.analyzer.baseline_done.connect(
            lambda floor: self.neuro_controls.set_status(f"Baseline set (resting PLV {floor:.2f}).")
        )
        self.neuro_controls.band_changed.connect(self.analyzer.set_band)
        self.neuro_controls.band_changed.connect(lambda b: self._log(f"Synchrony band -> {b}"))
        self.neuro_controls.calibrate_requested.connect(lambda: self.analyzer.start_baseline(30.0))
        self.neuro_controls.calibrate_requested.connect(lambda: self._log("Calibrate baseline (30s)"))

    def _populate_cameras(self):
        """Fill the camera combo with actually-detected devices."""
        try:
            cams = list_cameras()
        except Exception:
            cams = [0]
        self.camera_combo.clear()
        if cams:
            for i in cams:
                self.camera_combo.addItem(f"Camera {i}", i)
        else:
            self.camera_combo.addItem("No camera detected", None)

    def _on_view_changed(self, *_):
        """Recording controls are only shown in Recording View."""
        recording = self.rec_radio.isChecked()
        self.recording_group.setVisible(recording)

    def _on_left_toggled(self, shown):
        self.left_scroll.setVisible(shown)
        self.left_toggle.setText("◂" if shown else "▸")

    def _on_tab_changed(self, _index):
        """Only the visible view redraws — keeps tab switching snappy while
        streaming (each hidden view would otherwise keep repainting)."""
        live = self.view_tabs.currentWidget() is self._live_tab
        self.plot.set_paused(not live)
        self.values_panel.set_paused(not live)

    # --------------------------------------------------------------- actions
    def on_scan(self):
        self.scan_btn.setEnabled(False)
        self._log("Scan for devices")
        self._set_status("Scanning for Muse devices…")
        self.scan_thread = _ScanThread()
        self.scan_thread.result.connect(self._on_scan_result)
        self.scan_thread.failed.connect(self._on_scan_failed)
        self.scan_thread.start()

    def _on_scan_result(self, devices, raw):
        self.scan_btn.setEnabled(True)
        self.device_combo.clear()
        if not devices:
            self.device_combo.addItem("No devices found", None)
            self._set_status("No Muse devices found. Is the headband on and nearby?")
            self._log("Scan found 0 devices")
            return
        for d in devices:
            self.device_combo.addItem(f"{d['name']}  ({d['address']})", d["address"])
        self._set_status(f"Found {len(devices)} device(s).")
        self._log(f"Scan found {len(devices)} device(s): "
                  + ", ".join(d["address"] for d in devices))

    def _on_scan_failed(self, msg):
        self.scan_btn.setEnabled(True)
        self._set_status("Scan failed.")
        QMessageBox.critical(self, "Scan failed", msg)

    def on_connect(self):
        if self.reader is not None:  # currently connected -> disconnect
            self.disconnect_stream()
            return

        address = self.device_combo.currentData()
        if not address:
            QMessageBox.warning(self, "No device", "Scan and select a Muse device first.")
            return
        self._device_address = address
        self._log(f"Connect to {address}")

        try:
            self.stream_proc = MuseStreamProcess(address)
            self.stream_proc.start()
        except FileNotFoundError:
            QMessageBox.critical(
                self, "OpenMuse not found",
                "The OpenMuse CLI was not found. Reinstall project dependencies:\n"
                "    pip install -e .",
            )
            return

        # Fresh views for the new session.
        self.plot.clear()
        self.values_panel.clear_values()
        self.battery_label.setText("Battery: —")

        self.reader = LSLReaderThread(address=address)
        self.reader.samples_ready.connect(self._on_samples)
        self.reader.connected.connect(self._on_connected)
        self.reader.disconnected.connect(self._on_disconnected)
        self.reader.error.connect(self._on_reader_error)
        self.reader.status.connect(self._set_status)
        self.reader.start()

        self.connect_btn.setText("Disconnect")
        self.device_combo.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self._set_status("Connecting… (starting OpenMuse stream)")

    def _on_connected(self, names):
        # The reader can be torn down (disconnect, or a stream error) while this
        # queued signal is still in flight — Qt would then deliver it against a
        # dead reader. Bail out instead of crashing.
        if self.reader is None:
            self._log("Ignored a late 'connected' signal — reader already stopped")
            return
        ch_names = self.reader.channel_names()
        # Battery is shown as a number, not plotted or listed as a channel.
        plot_names = {k: v for k, v in ch_names.items() if not _is_battery(k)}
        self.plot.set_channel_names(plot_names)
        self.values_panel.set_channel_names(plot_names)
        # Identify the EEG stream for synchrony analysis.
        self._eeg_stream = next((n for n in names if _is_eeg(n)), None)
        self._eeg_channels = ch_names.get(self._eeg_stream, []) if self._eeg_stream else []
        self.analyzer.reset()
        self.bands.reset()
        self.hemo.reset()
        self._last_sample.clear()
        self._stalled.clear()
        self._escalated.clear()

        # Real sample rates drive the display filters and every analyzer.
        rates = {n: self.reader.sample_rate(n) for n in names}
        self.plot.set_sample_rates(rates)
        if self._eeg_stream:
            fs = rates.get(self._eeg_stream)
            if fs:
                self.analyzer.set_sample_rate(fs)
            self.bands.configure(self._eeg_channels, fs)
            self.spectrogram.set_channels(self._eeg_channels)

        optics_stream = next((n for n in names if _is_optics(n)), None)
        self._optics_channels = ch_names.get(optics_stream, []) if optics_stream else []
        if optics_stream:
            self.hemo.configure(self._optics_channels, rates.get(optics_stream))

        # Panels are created lazily on first sample, so apply the curated
        # channel preset once data has actually started flowing.
        QTimer.singleShot(1200, self._apply_channel_set)

        self._set_status(f"Connected. Streaming: {', '.join(names)}")
        self._log(f"Connected — streams: {', '.join(names)}")

    def _on_samples(self, stream_name, timestamps, data):
        """Route incoming chunks: battery -> numeric label, everything else ->
        the scrolling plot and the live values table."""
        self._note_samples(stream_name)
        if _is_battery(stream_name):
            if len(data):
                pct = float(data[-1][0] if data.ndim == 2 else data[-1])
                if pct <= 1.0:      # some firmwares report a 0–1 fraction
                    pct *= 100.0
                self.battery_label.setText(f"Battery: {pct:.0f}%")
            return
        self.plot.add_samples(stream_name, timestamps, data)
        self.values_panel.add_samples(stream_name, timestamps, data)
        if stream_name == self._eeg_stream:
            self.analyzer.add_eeg(self._eeg_channels, data)
            self.bands.add_eeg(self._eeg_channels, data)
        elif _is_optics(stream_name):
            self.hemo.add_optics(self._optics_channels, data)

    # ------------------------------------------------------------- watchdog
    def _check_streams(self):
        """Detect streams that have stopped delivering, warn audibly, recover.

        A dropout mid-session is the worst failure mode here: with your eyes
        closed nothing on screen can tell you, and the recording quietly gains a
        hole. So a stall gets a distinct alert tone, a marker in events.csv, and
        an attempt to restart the streamer if its process died.
        """
        if self.reader is None or not self._last_sample:
            return
        now = time.monotonic()
        for stream, last in list(self._last_sample.items()):
            gap = now - last
            if gap >= _STALL_SECONDS and stream not in self._stalled:
                self._stalled[stream] = last
                LOG.add(f"STREAM STALLED: {stream} (no data for {gap:.1f}s)", "error")
                self._log(f"Stream stalled: {stream}")
                self._write_event("dropout", "stall_start", stream)
                self._set_status(f"⚠ {_short(stream)} stopped sending data…")
                play_alert()
                if self._session_active:
                    self._speak("Signal lost. Check the headband.")
                self._maybe_restart_streamer()
            elif gap >= _STALL_SECONDS and stream in self._stalled:
                if gap >= _STALL_ESCALATE_S and not self._escalated.get(stream):
                    self._escalated[stream] = True
                    LOG.add(f"STREAM STILL DOWN: {stream} after {gap:.0f}s", "error")
                    self._set_status(
                        f"⚠ {_short(stream)} still down after {gap:.0f}s — "
                        "check the headband is on and in range.")
                    play_alert()

    def _note_samples(self, stream_name):
        """Called on every chunk; clears a stall and reports the gap."""
        now = time.monotonic()
        if stream_name in self._stalled:
            gap = now - self._stalled.pop(stream_name)
            self._escalated.pop(stream_name, None)
            self._dropouts.append((stream_name, datetime.now(), gap))
            LOG.add(f"STREAM RESUMED: {stream_name} after {gap:.1f}s", "log")
            self._log(f"Stream resumed: {stream_name} (lost {gap:.1f}s)")
            self._write_event("dropout", "stall_end", f"{stream_name} {gap:.1f}s")
            self._set_status(f"{_short(stream_name)} resumed (lost {gap:.1f}s)")
            play_resolved()
        self._last_sample[stream_name] = now

    def _maybe_restart_streamer(self):
        """If the OpenMuse process died, that's why data stopped — restart it.

        (liblsl reconnects on its own while the streamer is alive, so we only
        intervene when the producer itself is gone.)
        """
        proc = self.stream_proc
        if proc is None or proc.is_alive():
            return
        LOG.add("OpenMuse streamer process died — restarting", "error")
        self._log("OpenMuse streamer died; restarting")
        self._write_event("dropout", "streamer_restart", "")
        try:
            proc.start()
        except Exception as exc:  # noqa: BLE001
            LOG.add(f"Streamer restart failed: {exc}", "error")

    def _on_band_metrics(self, m):
        """Band powers -> bars, history and the EEG half of the laterality view."""
        self.band_bars.update_metrics(m)
        self.laterality.update_bands(m)
        times, series = self.bands.history()
        self.band_history.update_history(times, series)

    def _on_metrics(self, m):
        """Synchrony update: refresh visuals, drive audio, log if recording."""
        self.neuro_view.update_metrics(m)
        # On good contact drive the tone from synchrony; on lost contact degrade
        # toward rough/quiet so a slipping headband is audible feedback.
        self.binaural.apply_synchrony(m.level if m.contact_ok else 0.0)
        self._check_sync_gate(m)

    def _check_sync_gate(self, m):
        """Advance a "synced" gate phase once synchrony is held above baseline."""
        phase = self._current_phase
        if phase is None or getattr(phase, "gate", "") != "synced":
            self._sync_hold_start = None
            return
        held = m.contact_ok and m.calibrated and m.level >= _SYNC_GATE_LEVEL
        now = time.monotonic()
        if held:
            if self._sync_hold_start is None:
                self._sync_hold_start = now
            elif now - self._sync_hold_start >= _SYNC_GATE_HOLD_S:
                self._log("Sync gate satisfied — advancing to heterodyne")
                self._sync_hold_start = None
                self.runner.skip_to_next()
        else:
            self._sync_hold_start = None
        if self._sync_writer is not None:
            self._sync_writer.writerow([
                f"{_lsl_now():.6f}", m.band,
                f"{m.plv.get('frontal', 0.0):.4f}",
                f"{m.plv.get('temporal', 0.0):.4f}",
                f"{m.plv_combined:.4f}", f"{m.level:.4f}", f"{m.drift_hz:.4f}",
                int(m.contact_ok), int(m.calibrated),
            ])
            self._sync_file.flush()

    def _on_reader_error(self, msg):
        self._set_status("Stream error.")
        LOG.add(f"STREAM ERROR: {msg}", "error")
        QMessageBox.critical(self, "Stream error", msg)
        self.disconnect_stream()

    def _on_disconnected(self):
        # Emitted when the reader loop ends. Sessions may still run headband-free.
        pass

    def disconnect_stream(self):
        self._log("Disconnect")
        if self._session_active:
            self._end_session(aborted=True)
        if self.recorder is not None:
            self._stop_recording()
        if self.reader is not None:
            self.reader.stop()
            self.reader.wait(3000)
            self.reader = None
        self._eeg_stream = None
        if self.stream_proc is not None:
            self.stream_proc.stop()
            self.stream_proc = None
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.battery_label.setText("Battery: —")
        self._set_status("Disconnected.")

    # --------------------------------------------------------------- session
    def on_start_or_stop(self):
        if self._session_active:
            self.on_stop_session()
        else:
            self.on_start_session(record=False)

    def on_start_session(self, record=False):
        """Run the selected mode. ``record`` controls whether files are saved.
        A session can run without the Muse connected (e.g. audio + webcam only)."""
        mode = self.mode_combo.currentData()
        self._recording_enabled = record
        self._log(f"Start session (mode={mode}, record={record})")
        self._begin_session_ui()
        if mode == "free":
            if record:
                self._start_recording()
                if self.recorder is None:      # recording failed to start
                    self._end_session(aborted=True)
                    return
            self._free_start = time.monotonic()
            self.session_banner.show_free(recording=record)
            self.session_banner.set_elapsed(0)
            self._free_timer.start()
        else:
            self.runner.start(PROTOCOLS[mode])

    def on_stop_session(self):
        # Aborts a running protocol or ends free monitoring/recording.
        if self.runner.is_running():
            self.runner.abort()   # -> _on_runner_aborted handles teardown
        else:
            self._end_session(aborted=True)

    def _begin_session_ui(self):
        self._session_active = True
        self.start_btn.setText("Stop")
        self.record_btn.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.connect_btn.setEnabled(False)

    def _end_session(self, aborted=False):
        self._free_timer.stop()
        self.binaural.protocol_audio_off()
        if self.recorder is not None:   # protocol's Done phase may have stopped it already
            self._stop_recording()
        self._session_active = False
        self._recording_enabled = False
        self._current_phase = None
        self._sync_hold_start = None
        self.start_btn.setText("Start")
        self.record_btn.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.connect_btn.setEnabled(True)
        self.session_banner.hide()
        if aborted:
            self._set_status("Session stopped.")

    def _on_free_tick(self):
        self.session_banner.set_elapsed(time.monotonic() - self._free_start)

    # --- protocol runner callbacks ---
    def _on_phase_started(self, phase, index, total):
        waiting = phase.duration is None
        self._current_phase = phase
        self._sync_hold_start = None
        self._log(f"Protocol phase {index + 1}/{total}: {phase.name}"
                  + (f" ({int(phase.duration)}s)" if phase.duration else ""))
        # LSL-stamped so review can segment the recording by phase.
        self._write_event("phase", phase.name,
                          f"{index + 1}/{total} dur={phase.duration or ''}")
        self.session_banner.show_phase(phase, index, total, waiting)
        # Cue beep marks the transition for eyes-closed users. Skip it when the
        # phase starts a tone (the tone itself is the cue) or one is already
        # playing, to avoid audio-device contention.
        if "audio_on" not in phase.actions and not self.binaural.player.is_playing():
            play_cue()
        # Speak the instruction — with eyes closed this is the only guidance.
        # A short delay lets the attention beep finish first.
        QTimer.singleShot(450, lambda p=phase: self._speak(p.spoken()))
        for action in phase.actions:
            if not self._run_phase_action(action, phase):
                break   # a failed action aborted the protocol

    def _has_eeg(self):
        return self.reader is not None and self._eeg_stream is not None

    def _run_phase_action(self, action, phase):
        """Perform a phase action; return False if it aborted the protocol."""
        if action == "start_recording":
            if self._recording_enabled and self.recorder is None:
                self._start_recording()
                if self.recorder is None:   # recording failed to start
                    self._set_status("Recording failed to start — protocol aborted.")
                    self.runner.abort()
                    return False
        elif action == "calibrate":
            # Only meaningful with a live EEG signal.
            if self._has_eeg():
                self.analyzer.start_baseline(float(phase.duration or 30.0))
                self._write_event("marker", "calibrate_start", phase.duration or 30.0)
            else:
                self._log("No EEG — skipping baseline calibration")
        elif action == "audio_on":
            p = phase.params
            # Closed-loop needs live EEG; without the headband, play clean tones.
            closed = p.get("closed_loop", False) and self._has_eeg()
            self.binaural.protocol_audio_on(
                p.get("base", 200), p.get("beat", 10), closed,
                mode=p.get("mode", "binaural"),
            )
        elif action == "audio_control":
            self.binaural.protocol_control_tone(phase.params.get("base", 200))
        elif action == "heterodyne_start":
            # Continue the AM tone open-loop and begin the offset ramp at Δ=0.
            self.binaural.loop_check.setChecked(False)
            self.binaural.set_heterodyne_offset(0.0)
        elif action == "audio_off":
            self.binaural.protocol_audio_off()
        elif action == "audio_fade_out":
            self.binaural.fade_out(5.0)
        elif action == "stop_recording":
            if self.recorder is not None:
                self._stop_recording()
        return True

    def _on_runner_tick(self, remaining, duration):
        self.session_banner.set_countdown(remaining)
        # Ramp a parameter across the phase (e.g. the heterodyne offset 0→to).
        phase = self._current_phase
        if phase is not None and phase.ramp and duration > 0:
            progress = max(0.0, min(1.0, 1.0 - remaining / duration))
            if phase.ramp.get("param") == "heterodyne_offset":
                self.binaural.set_heterodyne_offset(phase.ramp["to"] * progress)

    def _on_continue(self):
        self.runner.advance()

    def _on_runner_finished(self):
        self._log("Protocol complete")
        play_complete()      # distinct from a phase change, for eyes-closed use
        self._end_session(aborted=False)
        self._set_status("Protocol complete — recording saved.")

    def _on_runner_aborted(self):
        self._log("Protocol aborted")
        self._end_session(aborted=True)

    # --------------------------------------------------------------- recording
    def _start_recording(self):
        try:
            self.session = RecordingSession(self.output_dir)
            self.recorder = MuseRecorder(self.session.muse_dir)
        except Exception as exc:  # noqa: BLE001
            self.session = None
            QMessageBox.critical(self, "Recording failed", str(exc))
            return
        if self.reader is not None:      # no headband -> only video/audio recorded
            self.reader.start_recording(self.recorder)
        # Route each artifact to its Data subfolder by provenance.
        self._open_audio_log(self.session.audio_dir)       # stimulus log
        self._open_sync_log(self.session.analysis_dir)     # derived PLV
        self._open_event_log(self.session.events_dir)      # protocol timeline
        cam_note = ""
        if self.webcam is not None:
            self.webcam.start_recording(self.session.video_dir)  # opens on first frame
            cam_note = " + webcam"
        # Start the action log (flushes the buffered lead-up) and snapshot config.
        self.logger.start_file(self.session.log_path)
        self.session.write_config(self._build_config())
        self._log(f"Recording started -> {self.session.root}")
        self._set_status(f"Recording{cam_note} to {self.session.name}")

    def _stop_recording(self):
        self._close_event_log()
        self._close_audio_log()
        self._close_sync_log()
        frames = self.webcam.stop_recording() if self.webcam is not None else 0
        if self.reader is not None:
            self.reader.stop_recording()
        elif self.recorder is not None:
            self.recorder.stop()
        counts = self.recorder.counts() if self.recorder else {}
        root = self.session.root if self.session else None
        # Update the config with end-of-recording results, then finish the log.
        if self.session is not None:
            cfg = self._build_config()
            cfg["results"] = {"sample_counts": counts, "webcam_frames": frames}
            self.session.write_config(cfg)
        summary = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "no samples"
        if frames:
            summary += f", webcam: {frames} frames"
        self._log(f"Recording stopped ({summary})")
        self.logger.stop_file()
        self.recorder = None
        self.session = None
        self._set_status(f"Saved to {root} ({summary})")
        # Surface the result immediately — after an eyes-closed session you
        # shouldn't have to go hunting to find out whether it worked.
        self._show_summary(root, counts, frames)

    def _show_summary(self, root, counts, frames):
        if not root:
            return
        # Subjective ratings first — they're only reliable while the session is
        # still fresh, and drowsiness is needed to interpret the EEG at all.
        try:
            key = self.mode_combo.currentData()
            proto = PROTOCOLS.get(key)
            if proto is not None:
                labels = [p.name for p in proto.phases if p.duration]
                q = QuestionnaireDialog(root, proto.name, labels, self)
                q.exec_()
        except Exception as exc:  # noqa: BLE001
            LOG.add(f"Questionnaire failed: {exc}", "error")
        dropouts, self._dropouts = list(self._dropouts), []
        try:
            dialog = SessionSummaryDialog(root, counts, frames, dropouts, self)
            dialog.review_btn.clicked.connect(
                lambda: self._review_path(root, dialog))
            dialog.show()
        except Exception as exc:  # noqa: BLE001 - never let this hide a recording
            LOG.add(f"Session summary failed: {exc}", "error")

    def _review_path(self, path, dialog=None):
        self.view_tabs.setCurrentWidget(self.review)
        self.review.load(path)
        if dialog is not None:
            dialog.accept()

    def _build_config(self):
        """Snapshot of settings + metadata written to recording_config.json."""
        try:
            from importlib.metadata import version
            app_version = version("fnt")
        except Exception:
            app_version = "unknown"
        p = self.binaural._params()
        streams = self.reader.channel_names() if self.reader else {}
        return {
            "app": "FieldNeuroethologyToolbox / MuseStudio",
            "version": app_version,
            "created": datetime.now().isoformat(timespec="milliseconds"),
            "mode": self.mode_combo.currentData(),
            "protocol": (PROTOCOLS[self.mode_combo.currentData()].name
                         if self.mode_combo.currentData() in PROTOCOLS else None),
            "muse_address": self._device_address,
            "eeg_stream": self._eeg_stream,
            "streams": list(streams.keys()),
            "channels": streams,
            "sample_rates": {n: (self.reader.sample_rate(n) if self.reader else None)
                             for n in streams},
            "camera_enabled": self.webcam is not None,
            "synchrony_band": self.neuro_controls.band_combo.currentData(),
            "binaural": {
                "base_hz": p["base_hz"], "beat_hz": p["beat_hz"],
                "volume": p["volume"], "closed_loop": self.binaural.is_closed_loop(),
            },
            "paths": {
                "root": self.session.root, "data": self.session.data_dir,
                "muse": self.session.muse_dir, "video": self.session.video_dir,
                "audio": self.session.audio_dir, "analysis": self.session.analysis_dir,
                "events": self.session.events_dir,
            },
        }

    # --------------------------------------------------------------- speech
    def _init_speech(self):
        """Wire the speaker, populate voices and restore saved preferences."""
        self.speaker = Speaker(self)
        self.speaker.enabled = self.speak_check.isChecked()
        # Duck the tone whenever the voice is talking, restore when it stops.
        self.speaker.started.connect(lambda: self.binaural.set_ducked(True))
        self.speaker.finished.connect(lambda: self.binaural.set_ducked(False))

        voices = list_voices()
        self.voice_combo.clear()
        if voices:
            for label, ident in voices:
                self.voice_combo.addItem(label, ident)
            saved = self._settings.value("speech_voice", "")
            idx = self.voice_combo.findData(saved) if saved else -1
            # Samantha is the usual macOS default and reads calmly.
            if idx < 0:
                idx = max(0, self.voice_combo.findText("Samantha", Qt.MatchStartsWith))
            self.voice_combo.setCurrentIndex(idx)
            self.speaker.voice = self.voice_combo.currentData()
            self.voice_combo.activated.connect(self._on_voice_changed)
        else:
            self.voice_combo.addItem("No voices available", None)
            self.voice_combo.setEnabled(False)

        if not self.speaker.available():
            self.speak_check.setEnabled(False)
            self.test_voice_btn.setEnabled(False)
            self._guidance_group.setToolTip(
                "No text-to-speech backend found. On macOS this uses the built-in "
                "'say' command; elsewhere install pyttsx3 (pip install pyttsx3)."
            )
        LOG.add(f"Speech backend: {self.speaker.backend_name()}", "log")

    def _on_speech_toggled(self, on):
        self.speaker.enabled = on
        self._settings.setValue("speech_enabled", on)
        if not on:
            self.speaker.stop()
            self.binaural.set_ducked(False)
        self._log(f"Spoken guidance {'on' if on else 'off'}")

    def _on_voice_changed(self, _index):
        self.speaker.voice = self.voice_combo.currentData()
        self._settings.setValue("speech_voice", self.speaker.voice or "")
        self.speaker.say("This is the guidance voice.")

    def _on_test_voice(self):
        self.speaker.say(
            "This is the guidance voice. Close your eyes and breathe normally.")

    def _speak(self, text):
        """Speak if guidance is enabled; harmless no-op otherwise."""
        speaker = getattr(self, "speaker", None)
        if speaker is not None:
            speaker.say(text)

    # ------------------------------------------------------------ diagnostics
    def on_show_logs(self):
        if getattr(self, "_log_dialog", None) is None:
            self._log_dialog = LogDialog(self)
        self._log_dialog.show()
        self._log_dialog.raise_()
        self._log_dialog.activateWindow()

    def on_copy_logs(self):
        QApplication.clipboard().setText(LOG.report())
        n = LOG.count()
        self._set_status(f"Diagnostic report copied to clipboard ({n} log lines).")

    def on_review_recording(self):
        """Jump to the Review tab and open a recording folder."""
        self.view_tabs.setCurrentWidget(self.review)
        self.review.on_open()

    def on_choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose default recording location", self.output_dir
        )
        if folder:
            self.output_dir = folder
            self.folder_label.setText(folder)
            self.folder_label.setToolTip(folder)
            self._settings.setValue("recording_dir", folder)
            self._log(f"Default recording location set to {folder}")

    # ------------------------------------------------------------ audio log
    def _open_audio_log(self, session_dir):
        self._audio_file = open(
            os.path.join(session_dir, "audio_events.csv"), "w", newline=""
        )
        self._audio_writer = csv.writer(self._audio_file)
        self._audio_writer.writerow(
            ["lsl_timestamp", "event", "base_hz", "beat_hz",
             "left_hz", "right_hz", "volume"]
        )
        # Snapshot current tone state at record start.
        state = "playing" if self.binaural.is_playing() else "idle"
        self._on_audio_event({"event": f"record_start ({state})", **self.binaural._params()})

    def _close_audio_log(self):
        if self._audio_file is not None:
            self._on_audio_event({"event": "record_stop", **self.binaural._params()})
            self._audio_file.close()
            self._audio_file = None
            self._audio_writer = None

    # ----------------------------------------------------------- event log
    def _open_event_log(self, session_dir):
        """Machine-readable protocol timeline on the LSL clock (review reads this)."""
        self._event_file = open(
            os.path.join(session_dir, "events.csv"), "w", newline=""
        )
        self._event_writer = csv.writer(self._event_file)
        self._event_writer.writerow(["lsl_timestamp", "kind", "label", "detail"])
        self._write_event("session", "record_start", self.mode_combo.currentData() or "")
        # The phase that *starts* the recording already emitted its marker before
        # this log existed, so re-emit it here — otherwise the baseline phase
        # (the thing every other phase is compared against) would be missing.
        if self._current_phase is not None:
            self._write_event("phase", self._current_phase.name, "active at record start")

    def _close_event_log(self):
        if self._event_file is not None:
            self._write_event("session", "record_stop", "")
            self._event_file.close()
            self._event_file = None
            self._event_writer = None

    def _write_event(self, kind, label, detail=""):
        if self._event_writer is None:
            return
        self._event_writer.writerow([f"{_lsl_now():.6f}", kind, label, str(detail)])
        self._event_file.flush()

    def _open_sync_log(self, session_dir):
        self._sync_file = open(
            os.path.join(session_dir, "synchrony.csv"), "w", newline=""
        )
        self._sync_writer = csv.writer(self._sync_file)
        self._sync_writer.writerow(
            ["lsl_timestamp", "band", "plv_frontal", "plv_temporal",
             "plv_combined", "level", "drift_hz", "contact_ok", "calibrated"]
        )

    def _close_sync_log(self):
        if self._sync_file is not None:
            self._sync_file.close()
            self._sync_file = None
            self._sync_writer = None

    def _log_audio_event(self, payload):
        """Mirror binaural changes into the human-readable session log."""
        self._log(
            f"Audio {payload['event']}: L{payload['left_hz']:.0f}/"
            f"R{payload['right_hz']:.0f} Hz (beat {payload['beat_hz']}), "
            f"vol {payload['volume']}"
        )

    def _on_audio_event(self, payload):
        """Write a binaural-beat event row if a recording is active."""
        if self._audio_writer is None:
            return
        self._audio_writer.writerow([
            f"{_lsl_now():.6f}", payload["event"], payload["base_hz"],
            payload["beat_hz"], payload["left_hz"], payload["right_hz"],
            payload["volume"],
        ])
        self._audio_file.flush()

    # ------------------------------------------------------------------ camera
    def on_toggle_camera(self):
        if self.webcam is not None:
            self._stop_camera()
            return
        index = self.camera_combo.currentData()
        if index is None:
            QMessageBox.warning(self, "No camera", "No camera was detected.")
            return
        self.webcam = WebcamThread(camera_index=index)
        self.webcam.frame_ready.connect(self._on_frame)
        self.webcam.opened.connect(self._on_camera_opened)
        self.webcam.error.connect(self._on_camera_error)
        self.webcam.start()
        self.camera_btn.setText("Stop Camera")
        self.camera_combo.setEnabled(False)
        self._log(f"Camera {index} started")

    def _stop_camera(self):
        if self.webcam is None:
            return
        self.webcam.stop()
        if not self.webcam.wait(3000):   # camera read blocked -> force it down
            self.webcam.terminate()
            self.webcam.wait(1000)
        self.webcam = None
        self.camera_btn.setText("Start Camera")
        self.camera_combo.setEnabled(True)
        self.camera_view.setPixmap(QPixmap())   # clear last frame
        self.camera_view.setText("Camera off")
        self._log("Camera stopped")

    def _on_frame(self, qimage):
        pix = QPixmap.fromImage(qimage).scaled(
            self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.camera_view.setPixmap(pix)

    def _on_camera_opened(self, w, h, fps):
        self._set_status(f"Camera on: {w}x{h} @ {fps:.0f} fps")

    def _on_camera_error(self, msg):
        self._stop_camera()
        QMessageBox.critical(self, "Camera error", msg)

    def _set_status(self, msg):
        self.status_label.setText(msg)

    def _log(self, msg):
        """Record a timestamped GUI action to the session log (buffered).

        Also mirrored into the diagnostic buffer so the Session Logs window
        shows user actions interleaved with errors — that ordering is usually
        what explains a failure.
        """
        self.logger.log(msg)
        LOG.add(msg, "session")

    def closeEvent(self, event):
        self.disconnect_stream()
        self._stop_camera()
        speaker = getattr(self, "speaker", None)
        if speaker is not None:
            speaker.shutdown()
        self.binaural.close_audio()
        if self.scan_thread is not None and self.scan_thread.isRunning():
            self.scan_thread.wait(2000)
        super().closeEvent(event)
