"""MuseStudio main window — connect to a Muse S Athena over BLE (via OpenMuse),
stream EEG/optics over LSL, live-plot the signals, and record them to CSV.

Brings together the pieces: live plots + numeric channel table, battery
readout, synchronized webcam capture, a binaural-beat generator with
closed-loop control from interhemispheric synchrony (PLV), a head-map + meter,
and a guided session runner (free record or a timed protocol).
"""

import csv
import json
import logging
import math
import os
import time
from datetime import datetime, timezone

from PyQt5.QtCore import Qt, QSettings, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGroupBox,
    QInputDialog,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

from fnt.musestudio import theme
from fnt.musestudio.mindball_controller import MindballController, latest_ghost
from fnt.musestudio.mindball_view import MindballView
from fnt.musestudio.flight.controller import FlightController
from fnt.musestudio.flight.view import FlightView
from fnt.musestudio.audio_output import (
    advice as audio_advice, capability as audio_capability,
)
from fnt.musestudio.analysis import BandPowerAnalyzer, HemodynamicsAnalyzer
from fnt.musestudio.binaural import (
    BinauralPanel, play_alert, play_complete, play_cue,
    play_lateral_sequence, play_resolved,
)
from fnt.musestudio.channel_table import LiveValuesPanel
from fnt.musestudio.device_status import DeviceStatusBar
from fnt.musestudio.dsp import EEG_ELECTRODES, curated_channels
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
from fnt.musestudio.session_view import FullscreenSessionView
from fnt.musestudio.sleep import SleepGuard, fmt_duration, storage_check
from fnt.musestudio.speech import Speaker, list_voices
from fnt.musestudio.subjects import (
    HAIR_OVER_EARS, HANDEDNESS, SEX, SubjectRegistry, sanitize_id,
)
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
#
# The threshold must be per-stream: Muse-BATTERY legitimately delivers one
# sample every ~5 s, so a flat 2.5 s cutoff would fire a false "signal lost"
# alarm (with tones and speech) every battery sample, all session long.
_STALL_SECONDS = 2.5
_STALL_ESCALATE_S = 15.0
# How long a session may run with no EEG before it is stopped outright. Long
# enough to ride out a brief BLE hiccup (liblsl reconnects on its own), short
# enough that nobody sits through a five-minute protocol collecting nothing.
_EEG_LOSS_ABORT_S = 20.0
# How often to look again while nothing is connected. The operator usually
# switches the headband on after opening the app, so a single scan at start-up
# misses it entirely.
_RESCAN_EVERY_S = 20.0
# A stream can be open while the headband sends nothing. Idle (no session), that
# state is dropped after this long so the UI stops claiming a connection it does
# not have and the next scan can retry cleanly.
_IDLE_SILENT_DROP_S = 25.0
_STALL_UNKNOWN_RATE_S = 12.0     # streams whose rate we couldn't resolve


def _stall_threshold(rate):
    """Seconds of silence that counts as a stall, given a nominal rate (Hz)."""
    if not rate or rate <= 0:
        return _STALL_UNKNOWN_RATE_S
    # Three missed sample periods, but never tighter than the BLE-hiccup floor.
    return max(_STALL_SECONDS, 3.0 / float(rate))


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


# Reference-tone level for the pre-session audio check. Deliberately low:
# the subject raises their system volume to meet it, rather than the app
# picking a loudness for someone who may be wearing in-ear buds.
AUDIO_CHECK_VOLUME = 0.12

def _git_sha():
    """Short git SHA of the working tree, or "" outside a checkout."""
    try:
        import subprocess
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        out = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        sha = (out.stdout or "").strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        return sha + ("+dirty" if (dirty.stdout or "").strip() else "")
    except Exception:  # noqa: BLE001
        return ""


_local_clock = None


def _lsl_now():
    """Current LSL clock value (shared with Muse/webcam timestamps).

    Falls back to ``time.monotonic`` when mne_lsl is unavailable. That is not a
    cosmetic guard: this function is called while writing every event row, and
    when it raised, the exception propagated out of ``_open_event_log`` and
    ``_close_audio_log`` mid-recording. The protocol then ran happily to
    completion having written no events.csv at all -- the same silent-empty-
    recording failure this project has already been bitten by once, and worse
    here because events.csv is what every downstream analysis aligns against.

    The fallback clock does not share an origin with the Muse stream, so
    cross-stream alignment is meaningless in that mode; a recording made this
    way is for smoke-testing the plumbing, not for analysis. It is still far
    better than losing the timeline without saying so.
    """
    global _local_clock
    if _local_clock is None:
        try:
            from mne_lsl.lsl import local_clock
            _local_clock = local_clock
        except Exception:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "mne_lsl unavailable — event timestamps fall back to a local "
                "monotonic clock and will NOT align with Muse stream timestamps."
            )
            _local_clock = time.monotonic
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
    stop_clicked = pyqtSignal()

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
        # Stop lives here as well as in the left column. During a session the
        # banner is the one element guaranteed to be on screen and looked at,
        # and "how do I stop this?" must never be a hunt.
        self.stop_btn = QPushButton("Stop session")
        self.stop_btn.setToolTip("End the session now. A recording in progress "
                                 "is closed and saved.")
        self.stop_btn.setStyleSheet(
            f"background-color: #5b2020; border: 1px solid {theme.DANGER}; "
            f"color: #ffd9d9; font-weight: 600; padding: 4px 12px;")
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        bottom.addWidget(self.stop_btn)
        lay.addLayout(bottom)
        self.hide()

    def show_free(self, recording=True, compact=False):
        """``compact`` drops the explanatory line for modes that have their own
        display.

        In Mindball or Flight the banner sat above a full-screen game restating
        that live monitoring was happening — information the player already has,
        occupying the top of the very view they are trying to watch. The title,
        the clock and Stop are kept; the sentence is not.
        """
        if recording:
            self.title.setText("Recording")
            self.instruction.setText(
                "" if compact else
                "Recording all active streams (EEG, optics, webcam, audio events). "
                "Press Stop when you are finished."
            )
        else:
            self.title.setText("Live")
            self.instruction.setText(
                "" if compact else
                "Live monitoring — nothing is being saved. Press Stop when finished."
            )
        self.instruction.setVisible(not compact)
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
        LOG.install_crash_handlers()
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
        self._quality_file = None
        self._quality_writer = None
        self._event_file = None
        self._event_writer = None
        self._eeg_stream = None
        self._eeg_channels = []
        self._optics_channels = []
        self._device_address = None
        self._auto_connecting = False
        self._auto_started = False
        self._connected_at = None
        self._session_needs_eeg = False
        self._devices = []
        self._selected_address = None
        self._selected_name = ""
        self._session_active = False
        self._session_needs_eeg = False
        self._recording_enabled = False
        self._free_start = 0.0
        self._current_phase = None
        self._sync_hold_start = None
        self.session = None            # active RecordingSession
        self.logger = SessionLogger()  # buffers GUI actions from window open

        # Default recording location persists across launches.
        self._settings = QSettings("FNT", "MuseStudio")
        self.output_dir = self._settings.value("recording_dir", _default_recording_dir())

        self.setWindowTitle("MuseStudio - FieldNeuroToolbox")
        self.resize(1480, 900)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(theme.STYLESHEET)

        # Built BEFORE _init_ui(): _build_right_view() adds flight_view as a tab,
        # so the widget has to exist by then.
        self.flight = FlightController(self)
        self.flight_view = FlightView()
        self.flight.frame.connect(
            lambda st, tr: self.flight_view.set_frame(
                st, tr, self.flight.cfg.dead_zone, self.flight.cfg.full_scale_z))
        self.flight.status.connect(self._on_flight_status)
        self.flight.calibrating.connect(
            lambda f: self.flight_view.set_status(
                f"Calibrating baseline… {100*f:.0f}%" if f < 1.0 else ""))
        self.flight.phase_changed.connect(
            lambda ph: self._write_event("flight", ph, ""))

        self.mindball = MindballController(self)
        self.mindball_view = MindballView()
        self.mindball.frame.connect(
            lambda st, lab, hist: self.mindball_view.set_frame(st, lab, hist))
        self.mindball.status.connect(self._on_mindball_status)
        self.mindball.finished.connect(self._on_mindball_finished)
        self.flight.altitude_tone.connect(self._on_flight_tone)
        self.flight.cue.connect(self._on_flight_cue)
        self.flight.finished.connect(self._on_flight_finished)

        self._init_ui()

        # Guided-protocol runner and a free-record elapsed timer.
        self.runner = ProtocolRunner(self)
        self.runner.phase_started.connect(self._on_phase_started)
        self.runner.tick.connect(self._on_runner_tick)
        self.runner.finished.connect(self._on_runner_finished)
        self.runner.aborted.connect(self._on_runner_aborted)
        self.runner.phase_skipped.connect(self._on_phase_skipped)
        self.device_status.connect_clicked.connect(self.on_connect)

        # Keep looking while nothing is connected — the headband is usually
        # switched on after the app is already open.
        self._rescan_timer = QTimer(self)
        self._rescan_timer.timeout.connect(self._retry_scan)
        self._rescan_timer.start(int(_RESCAN_EVERY_S * 1000))

        self._free_timer = QTimer(self)
        self._free_timer.setInterval(500)
        self._free_timer.timeout.connect(self._on_free_tick)

        # Watchdog for silently-dropped streams (Bluetooth/LSL dropouts would
        # otherwise leave gaps in a recording with nothing to show for it).
        self._last_sample = {}      # stream -> monotonic time of last chunk
        self._stalled = {}          # stream -> monotonic time the stall began
        self._escalated = {}        # stream -> already warned a second time
        self._stall_limits = {}     # stream -> seconds of silence = stall
        self._battery_warned = {}   # one-shot low/critical battery warnings
        self._battery_last = None
        self._sleep_mode = False
        self._silence_alerts = False
        self._sleep_guard = SleepGuard()
        self.fullscreen_view = None
        self._dropouts = []         # (stream, start_wall, seconds) for the summary
        self._stall_timer = QTimer(self)
        self._stall_timer.setInterval(1000)
        self._stall_timer.timeout.connect(self._check_streams)
        self._stall_timer.start()

        # Live EEG throughput for the status bar — the fastest way to see that
        # a "connected" headband has actually stopped sending anything.
        self._rate_count = 0
        self._rate_timer = QTimer(self)
        self._rate_timer.setInterval(1000)
        self._rate_timer.timeout.connect(self._update_rate)
        self._rate_timer.start()

    # ------------------------------------------------------------------ UI
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)

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
        main_split.setSizes([500, 1040])
        root.addWidget(main_split, stretch=1)

        self._wire_analyzer()
        self._load_subjects()
        self._init_speech()

    def _build_left_column(self):
        """Scrolling column of controls (like MAD / Mask Tracker)."""
        left = QWidget()
        col = QVBoxLayout(left)
        col.setContentsMargins(4, 4, 4, 4)
        col.setSpacing(8)

        # View selector at the very top.

        # Muse connection.

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

        self.fullscreen_check = QCheckBox("Full-screen session")
        self.fullscreen_check.setToolTip(
            "During a protocol, take over the screen with a distraction-free "
            "phase display.\n\n"
            "Eyes-open blocks get a fixation cross to look at — a fixed gaze "
            "means fewer eye-movement artifacts in the control block.\n"
            "Eyes-closed blocks go near-black, because screen light reaching "
            "the retina through your eyelids suppresses the alpha rhythm being "
            "measured.\n\n"
            "Press Esc at any time to return to the normal window; the session "
            "keeps running."
        )
        self.fullscreen_check.setChecked(
            self._settings.value("fullscreen_session", True, type=bool))
        self.fullscreen_check.toggled.connect(
            lambda on: self._settings.setValue("fullscreen_session", on))
        gg.addWidget(self.fullscreen_check)
        col.addWidget(guide)
        self._guidance_group = guide

        # Who is being recorded. Sits directly under View so it is the first
        # thing set before a session, and hard to forget.
        self.subject_group = QGroupBox("Subject")
        sg = QVBoxLayout(self.subject_group)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("ID"))
        self.subject_combo = QComboBox()
        self.subject_combo.setEditable(True)
        self.subject_combo.setInsertPolicy(QComboBox.NoInsert)
        self.subject_combo.lineEdit().setPlaceholderText("e.g. S01")
        self.subject_combo.setToolTip(
            "Code identifying who is being recorded. It goes into the folder "
            "name, the config file and every report, so sessions can be grouped "
            "by person.\n\n"
            "Prefer a code (S01, CV01) over a real name — the code travels with "
            "any data or report you share.\n\n"
            "Previously used subjects appear in the list; their handedness is "
            "filled in automatically."
        )
        self.subject_combo.currentTextChanged.connect(self._on_subject_changed)
        srow.addWidget(self.subject_combo, stretch=1)
        sg.addLayout(srow)

        drow = QHBoxLayout()
        drow.addWidget(QLabel("Sex"))
        self.sex_combo = QComboBox()
        for v in SEX:
            self.sex_combo.addItem(v, v)
        self.sex_combo.setToolTip(
            "Recorded once per subject as a standard demographic covariate.")
        drow.addWidget(self.sex_combo, stretch=1)
        drow.addWidget(QLabel("Hair over ears"))
        self.hair_combo = QComboBox()
        for v in HAIR_OVER_EARS:
            self.hair_combo.addItem(v, v)
        self.hair_combo.setToolTip(
            "Whether hair sits between the ear sensors and the skin.\n\n"
            "This is here because it predicts data quality: in session 1 both "
            "TP9 and TP10 lost contact within two minutes because of long hair, "
            "while the forehead electrodes stayed perfect throughout. Recording "
            "it means a temporal-pair failure can be explained rather than "
            "mistaken for bad technique — and you know to clip the hair back "
            "before starting."
        )
        drow.addWidget(self.hair_combo, stretch=1)
        sg.addLayout(drow)

        hrow = QHBoxLayout()
        hrow.addWidget(QLabel("Handedness"))
        self.handed_combo = QComboBox()
        for h in HANDEDNESS:
            self.handed_combo.addItem(h, h)
        self.handed_combo.setToolTip(
            "Recorded once per subject.\n\n"
            "This matters here specifically: hemispheric lateralization differs "
            "systematically between right- and left-handers, so any left/right "
            "asymmetry or interhemispheric-synchrony result has to be read "
            "against it. Stored with the subject, not per session."
        )
        hrow.addWidget(self.handed_combo, stretch=1)
        sg.addLayout(hrow)

        self.session_label_edit = QLineEdit()
        self.session_label_edit.setPlaceholderText("session label (optional) — e.g. caffeine, pre-sleep")
        self.session_label_edit.setToolTip(
            "Free-text tag for *this* session's condition, saved into the "
            "recording config. Useful when comparing the same subject across "
            "conditions (caffeine, time of day, post-exercise…)."
        )
        sg.addWidget(self.session_label_edit)

        # Recording controls (shown only in Recording View).
        self.recording_group = QGroupBox("Recording")
        rg = QVBoxLayout(self.recording_group)
        # The session selector. This control used to be a flat list labelled
        # "Recording protocol" holding three unrelated kinds of thing —
        # open-ended modes, an interactive mode, and timed protocols — with the
        # only explanation buried in a tooltip. Nobody found Flight Calibration
        # in it, which is a fair verdict on the design rather than on the user.
        # Now: grouped under headings, and whatever is selected explains itself
        # in the label underneath, so the choice is legible without hovering.
        rg.addWidget(QLabel("Session"))
        self.mode_combo = QComboBox()
        self._mode_blurbs = {}

        def _add_heading(text):
            self.mode_combo.addItem(text)
            i = self.mode_combo.count() - 1
            item = self.mode_combo.model().item(i)
            item.setEnabled(False)
            f = item.font()
            f.setBold(True)
            item.setFont(f)

        def _add_mode(label, key, blurb):
            self.mode_combo.addItem(f"   {label}", key)
            self._mode_blurbs[key] = blurb

        _add_heading("Open-ended")
        _add_mode("Free monitor", "free",
                  "Capture whatever is streaming until you press Stop. "
                  "No guidance, no timing, no phases.")
        _add_mode("Sleep (overnight)", "sleep",
                  "Unattended all-night recording. Keeps the Mac awake, "
                  "silences every alert, pauses the live plots, and writes at "
                  "reduced precision — about 1 GB a night.")
        _add_mode("Mindball (vs your past self)", "mindball",
                  "Push a ball into your opponent's end by being the calmer "
                  "one — the science-museum game. You play a recording of a "
                  "previous match, or a practice partner if you have none yet.")
        _add_mode("Flight test (cued, 2:20)", "flight",
                  "Fly a craft on your own cortical alpha. Spoken cues alternate "
                  "CLIMB (eyes closed) and DESCEND (eyes open), so altitude can "
                  "be checked against the instruction. Stops on its own.")

        self.mode_combo.insertSeparator(self.mode_combo.count())
        _add_heading("Guided protocols")
        for key, proto in PROTOCOLS.items():
            # Count only phases that always run. Hardware-gated blocks (a
            # stereo-only check, say) are skipped on some setups, so including
            # them would advertise a length the session does not have.
            total = sum(ph.duration or 0 for ph in proto.phases if not ph.requires)
            mins = f"{int(total // 60)}:{int(total % 60):02d}"
            name = proto.name.split("(")[0].strip()
            self.mode_combo.addItem(f"   {name}  ·  {mins}", key)
            self._mode_blurbs[key] = proto.description

        self.mode_combo.setToolTip(
            "What this session will do. Open-ended runs until you stop it; "
            "a guided protocol steps through timed phases with spoken "
            "instructions.")
        rg.addWidget(self.mode_combo)

        # Says what the selected session actually does. The point is that no
        # choice here should require hovering, reading source, or guessing.
        self.mode_blurb = QLabel()
        self.mode_blurb.setWordWrap(True)
        self.mode_blurb.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # A word-wrapped QLabel defaults to a Preferred vertical policy, which in
        # a QVBoxLayout lets it soak up every spare pixel — this one grew to 480
        # px against a 38 px hint and pushed the whole Recording group out from
        # underneath the panel below it, so Recording rendered *behind* Webcam
        # and looked like it had disappeared. Minimum policy plus a ceiling keeps
        # it to its content, and the fixed ceiling also stops the left column
        # reflowing every time the selection changes.
        self.mode_blurb.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.mode_blurb.setMaximumHeight(74)
        self.mode_blurb.setStyleSheet(
            f"color: {theme.TEXT_DIM}; padding: 2px 4px 6px 4px;")
        rg.addWidget(self.mode_blurb)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        # Restore whatever was run last — most sessions are a repeat of the
        # previous one, so the common case should need no interaction at all.
        last = self._settings.value("last_session", "")
        restored = False
        for i in range(self.mode_combo.count()):
            if last and self.mode_combo.itemData(i) == last:
                self.mode_combo.setCurrentIndex(i)
                restored = True
                break
        if not restored:
            self.mode_combo.setCurrentIndex(1)   # index 0 is the "Open-ended" heading
        self._on_mode_changed()

        brow = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.on_start_or_stop)
        self.start_btn.setToolTip("Run the selected session live, without saving any files.")
        self.start_btn.setMinimumHeight(38)
        brow.addWidget(self.start_btn, stretch=1)
        self.record_btn = QPushButton("Start + Record")
        self.record_btn.setMinimumHeight(38)
        # The recording button is the primary action — an unrecorded session is
        # the exception, not the default.
        self.record_btn.setProperty("accent", "true")
        self.record_btn.clicked.connect(lambda: self.on_start_session(record=True))
        self.record_btn.setToolTip("Run the selected session and save it to the recording folder.")
        brow.addWidget(self.record_btn, stretch=2)
        rg.addLayout(brow)
        # Only the recording-*management* rows follow the Live/Recording view
        # toggle. The session selector and Start buttons never hide: they are
        # the primary action, and the app opens in Live View, so hiding them
        # meant a fresh window offered no visible way to start anything at all.
        self.recording_extras = QWidget()
        ex = QVBoxLayout(self.recording_extras)
        ex.setContentsMargins(0, 0, 0, 0)
        ex.addWidget(QLabel("Default recording location:"))
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
        ex.addLayout(loc_row)
        self.review_btn = QPushButton("Review a recording…")
        self.review_btn.clicked.connect(self.on_review_recording)
        self.review_btn.setToolTip(
            "Open a finished recording and compare each protocol phase "
            "against baseline.")
        ex.addWidget(self.review_btn)
        rg.addWidget(self.recording_extras)
        self.recording_extras.setVisible(True)
        # Final column order: who is being recorded, then what to run. The Muse
        # panel that used to sit above these is gone — connecting is automatic,
        # and its status plus a fallback Connect button now live in the
        # always-visible strip above the tabs. The Live/Recording View radios
        # are gone too: they toggled two rows and, by hiding the whole Recording
        # group in their default state, made the primary action invisible.
        col.insertWidget(0, self.recording_group)
        col.insertWidget(0, self.subject_group)

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
        # Wide enough for the two-button Recording row plus the scrollbar and
        # collapse rail — at 330 the "Start + Record" button was clipped.
        # 470/620, not 400/520: at 400 the Subject fields and the session
        # description were clipped on the right, which is what made the column
        # look broken rather than merely narrow.
        container.setMinimumWidth(470)
        container.setMaximumWidth(620)
        return container

    def _build_right_view(self):
        """Data view: session banner + tabbed views, with the raw signal first."""
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(4, 4, 4, 4)
        rlay.setSpacing(6)

        # Headband status is pinned above everything: it must be readable no
        # matter which tab is showing, and it is the fastest way to notice that
        # a "connected" device has stopped sending data.
        self.device_status = DeviceStatusBar()
        rlay.addWidget(self.device_status)

        self.session_banner = SessionBanner()
        self.session_banner.continue_clicked.connect(self._on_continue)
        self.session_banner.stop_clicked.connect(self.on_stop_session)
        rlay.addWidget(self.session_banner)

        self.view_tabs = QTabWidget()
        self._live_tab = self._build_live_tab()
        self.view_tabs.addTab(self._live_tab, "Live signal")
        self.view_tabs.addTab(self._build_bands_tab(), "Bands")
        self.view_tabs.addTab(self._build_spectrogram_tab(), "Spectrogram")
        self.view_tabs.addTab(self._build_synchrony_tab(), "Synchrony")
        self.view_tabs.addTab(self._build_camera_tab(), "Camera")
        self.view_tabs.addTab(self.flight_view, "Flight")
        self.view_tabs.addTab(self.mindball_view, "Mindball")
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
        self.values_panel.setMaximumWidth(380)
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

    def showEvent(self, event):
        """Begin looking for the headband as soon as the window appears.

        Deferred one event-loop turn so the window paints first — the operator
        should see the UI and the "Looking for your headband…" status together,
        not stare at nothing while a BLE scan blocks the first paint.
        """
        super().showEvent(event)
        if not self._auto_started:
            self._auto_started = True
            QTimer.singleShot(0, self.auto_start)

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
        # Never run two scans at once, and never leak the thread.
        #
        # This crashed the app. Each QThread needs a pipe for its event
        # dispatcher, and the periodic re-scan created a brand new _ScanThread
        # every 20 s while disconnected, dropping the previous one without
        # waiting for or deleting it. File descriptors ran out, and the next
        # thread's QEventDispatcherUNIXPrivate constructor called qFatal ->
        # SIGABRT. The macOS crash report pointed straight at it
        # (QMessageLogger::fatal inside QEventDispatcherUNIXPrivate).
        old = getattr(self, "scan_thread", None)
        if old is not None:
            try:
                if old.isRunning():
                    self._log("Scan already in progress — skipping")
                    return
            except RuntimeError:
                # The C++ object was already deleted by deleteLater; the Python
                # wrapper outlives it briefly. Treat that as "no scan running".
                self.scan_thread = None
        self.device_status.set_connect_enabled(False)
        self._log("Scan for devices")
        self._set_status("Scanning for Muse devices…")
        thread = _ScanThread(self)
        thread.result.connect(self._on_scan_result)
        thread.failed.connect(self._on_scan_failed)
        # Free the thread object once it finishes, so repeated scans over a long
        # idle period do not accumulate.
        thread.finished.connect(thread.deleteLater)
        # Drop our reference too, or the next scan inspects a deleted object.
        thread.finished.connect(lambda: setattr(self, "scan_thread", None))
        self.scan_thread = thread
        thread.start()

    def _on_scan_result(self, devices, raw):
        self.device_status.set_connect_enabled(True)
        self._devices = list(devices or [])
        if not devices:
            self.device_status.set_connected(False)
            self._set_status("No Muse devices found. Is the headband on and nearby?")
            self._log("Scan found 0 devices")
            self._auto_connecting = False
            return
        self._set_status(f"Found {len(devices)} device(s).")
        self._log(f"Scan found {len(devices)} device(s): "
                  + ", ".join(d["address"] for d in devices))

        if not self._auto_connecting:
            return
        self._auto_connecting = False
        # Pick the headband we used last; failing that, the only one in range.
        # Anything more ambiguous is left to the human — silently connecting to
        # whichever Muse happens to be nearest in a shared lab would be worse
        # than asking.
        remembered = self._settings.value("device_address", "")
        target = next((d for d in devices if d["address"] == remembered), None)
        if target is None and len(devices) == 1:
            target = devices[0]
        if target is None:
            self.device_status.set_connected(False)
            self._set_status(f"{len(devices)} headbands found — press Connect to "
                             "choose one.")
            return
        self._selected_address = target["address"]
        self._selected_name = target.get("name", "")
        self._log(f"Auto-connecting to {self._selected_address}")
        self.on_connect()

    def _retry_scan(self):
        """Keep looking for the headband while nothing is connected.

        The original scan fired once, when the window opened. That loses the
        natural order of operations: the operator opens the app, THEN picks up
        the headband and switches it on. By the time the Muse is advertising,
        the one scan has long finished and nothing ever looks again, so the app
        sits there saying "not connected" forever while the headband sits there
        advertising. Retrying closes that gap without any button to find.
        """
        if self.reader is not None or self._auto_connecting or self._session_active:
            return
        self._log("Re-scanning for the headband")
        self.auto_start()

    def auto_start(self):
        """Scan and connect without being asked. Called once when the tab opens.

        The window already knows which headband it used last, so making the
        operator press Scan, choose from a list, and press Connect is three
        clicks of pure ceremony. It stays cancellable — the Muse panel still has
        both buttons — and it never guesses between multiple unknown devices.
        """
        if self.reader is not None or self._auto_connecting:
            return
        self._auto_connecting = True
        self._set_status("Looking for your headband…")
        self.device_status.set_busy("Searching…")
        self.on_scan()

    def _on_scan_failed(self, msg):
        self.device_status.set_connect_enabled(True)
        self.device_status.set_connected(False)
        self._set_status("Scan failed.")
        QMessageBox.critical(self, "Scan failed", msg)

    def on_connect(self):
        if self.reader is not None:  # currently connected -> disconnect
            self.disconnect_stream()
            return

        address = self._selected_address
        if not address and len(self._devices) > 1:
            # Ambiguous: ask rather than guess. A one-shot dialog is the right
            # weight for something that only happens when auto-connect could not
            # decide; it does not warrant a permanent panel in the column.
            labels = [f"{d.get('name','Muse')}  ({d['address']})" for d in self._devices]
            choice, ok = QInputDialog.getItem(
                self, "Choose a headband",
                "More than one Muse is in range:", labels, 0, False)
            if not ok:
                return
            picked = self._devices[labels.index(choice)]
            address = self._selected_address = picked["address"]
            self._selected_name = picked.get("name", "")
        elif not address and len(self._devices) == 1:
            address = self._selected_address = self._devices[0]["address"]
            self._selected_name = self._devices[0].get("name", "")
        if not address:
            # Nothing known yet — scanning is the useful answer to "Connect".
            self.auto_start()
            return
        self._device_address = address
        # Remembered so the next session can connect without being asked. This
        # is the single biggest saving in the start-up flow: Scan, pick, Connect
        # is three clicks that the app already has enough information to skip.
        self._settings.setValue("device_address", address)
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

        self.reader = LSLReaderThread(
            address=address,
            streamer_dead=lambda: (self.stream_proc is not None
                                   and not self.stream_proc.is_alive()))
        self.reader.samples_ready.connect(self._on_samples)
        self.reader.connected.connect(self._on_connected)
        self.reader.disconnected.connect(self._on_disconnected)
        self.reader.error.connect(self._on_reader_error)
        self.reader.status.connect(self._set_status)
        self.reader.start()

        self.device_status.set_busy("Connecting…")
        self._set_status("Connecting… (starting OpenMuse stream)")

    def _on_connected(self, names):
        # The reader can be torn down (disconnect, or a stream error) while this
        # queued signal is still in flight — Qt would then deliver it against a
        # dead reader. Bail out instead of crashing.
        if self.reader is None:
            self._log("Ignored a late 'connected' signal — reader already stopped")
            return
        # When streams resolved. Used to notice a connection that resolves
        # outlets but never delivers a sample, which looks identical to a
        # working headband until you check the numbers.
        self._connected_at = time.monotonic()
        self._last_sample = {}
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
        self._battery_warned.clear()

        # Real sample rates drive the display filters and every analyzer.
        rates = {n: self.reader.sample_rate(n) for n in names}
        self.plot.set_sample_rates(rates)
        # Watchdog thresholds scale with each stream's nominal rate (battery
        # sends one sample per ~5 s and must not trip false alarms).
        self._stall_limits = {n: _stall_threshold(rates.get(n)) for n in names}
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

        self.device_status.set_connected(True, self._selected_name or "Connected")
        self._set_status(f"Connected. Streaming: {', '.join(names)}")
        self._log(f"Connected — streams: {', '.join(names)}")

    def _on_samples(self, stream_name, timestamps, data):
        """Route incoming chunks: battery -> numeric label, everything else ->
        the scrolling plot and the live values table."""
        self._note_samples(stream_name, len(timestamps) if timestamps is not None else 0)
        if _is_battery(stream_name):
            if len(data):
                pct = float(data[-1][0] if data.ndim == 2 else data[-1])
                if pct <= 1.0:      # some firmwares report a 0–1 fraction
                    pct *= 100.0
                self._update_battery(pct)
            return
        self.plot.add_samples(stream_name, timestamps, data)
        self.values_panel.add_samples(stream_name, timestamps, data)
        if stream_name == self._eeg_stream:
            # Feed the always-visible trace. One channel, decimated ~8x: this is
            # a liveness indicator, not a measurement, so it costs almost
            # nothing and must never compete with the real plots.
            try:
                col = 0
                for i, n in enumerate(self._eeg_channels or []):
                    if "AF8" in str(n).upper():
                        col = i
                        break
                self.device_status.push_signal(data[::8, col])
            except Exception:  # noqa: BLE001
                pass
            self.analyzer.add_eeg(self._eeg_channels, data)
            self.bands.add_eeg(self._eeg_channels, data)
            if self.flight.is_running():
                self.flight.add_eeg(self._eeg_channels, data)
            if self.mindball.is_running():
                self.mindball.add_eeg(self._eeg_channels, data)
        elif _is_optics(stream_name):
            self.hemo.add_optics(self._optics_channels, data)
        elif ("ACC" in str(stream_name).upper()
              or "GYRO" in str(stream_name).upper()):
            if self.flight.is_running():
                names = self.reader.channel_names().get(stream_name, []) \
                    if self.reader is not None else []
                self.flight.add_motion(names, data)

    def _update_rate(self):
        """Publish EEG samples/second to the status bar once per second."""
        if self.reader is None:
            self._rate_count = 0
            self.device_status.set_rate(None)
            return
        expected = None
        if self._eeg_stream:
            expected = self.reader.sample_rate(self._eeg_stream)
        self.device_status.set_rate(self._rate_count, expected)
        self._rate_count = 0

    def _update_battery(self, pct):
        """Battery readout with escalating warnings.

        A dying battery doesn't just end the session — first it browns out the
        BLE radio, which shows up as all four streams repeatedly 'breaking off
        and reconnecting'. Warning at 20% (and insisting at 10%) prevents a
        session that would be garbage anyway.
        """
        self._battery_last = pct
        self.device_status.set_battery(pct)
        if pct < 10:
            if not self._battery_warned.get("critical"):
                self._battery_warned["critical"] = True
                LOG.add(f"BATTERY CRITICAL: {pct:.0f}% — expect BLE dropouts; "
                        "charge before recording", "error")
                self._log(f"Battery critical ({pct:.0f}%)")
                self._set_status(
                    f"⚠ Battery {pct:.0f}% — streams will drop out. Charge the "
                    "headband before recording.")
                self._sound(play_alert, session_only=True)
                if self._session_active:
                    self._speak("Headband battery is critically low. The "
                                "recording may cut out.")
        elif pct < 20:
            if not self._battery_warned.get("low"):
                self._battery_warned["low"] = True
                LOG.add(f"Battery low: {pct:.0f}%", "log")
                self._set_status(f"Battery {pct:.0f}% — consider charging "
                                 "before a long session.")

    # ------------------------------------------------------------- watchdog
    def _check_streams(self):
        """Detect streams that have stopped delivering, warn audibly, recover.

        A dropout mid-session is the worst failure mode here: with your eyes
        closed nothing on screen can tell you, and the recording quietly gains a
        hole. So a stall gets a distinct alert tone, a marker in events.csv, and
        an attempt to restart the streamer if its process died.
        """
        if self.reader is None:
            return
        if not self._last_sample:
            # Never received a single sample since connecting. The old early
            # return here meant a connection that produced nothing was never
            # examined at all — no stall, no drop, no message — so the UI sat on
            # "connected" indefinitely. That is the state that wasted a session.
            self._check_never_delivered()
            return
        now = time.monotonic()
        for stream, last in list(self._last_sample.items()):
            gap = now - last
            limit = self._stall_limits.get(stream, _STALL_UNKNOWN_RATE_S)
            if gap >= limit and stream not in self._stalled:
                self._stalled[stream] = last
                LOG.add(f"STREAM STALLED: {stream} (no data for {gap:.1f}s)", "error")
                self._log(f"Stream stalled: {stream}")
                self._write_event("dropout", "stall_start", stream)
                self._set_status(f"⚠ {_short(stream)} stopped sending data…")
                self._sound(play_alert, session_only=True)
                if self._session_active:
                    self._speak("Signal lost. Check the headband.")
                self._maybe_restart_streamer()
            elif gap >= limit and stream in self._stalled:
                if gap >= _STALL_ESCALATE_S and not self._escalated.get(stream):
                    self._escalated[stream] = True
                    LOG.add(f"STREAM STILL DOWN: {stream} after {gap:.0f}s", "error")
                    self._set_status(
                        f"⚠ {_short(stream)} still down after {gap:.0f}s — "
                        "check the headband is on and in range.")
                    self._sound(play_alert, session_only=True)

        self._abort_if_no_eeg()
        self._drop_if_silent_idle()

    def _check_never_delivered(self):
        """Connected, but not one sample has ever arrived."""
        started = getattr(self, "_connected_at", None)
        if started is None:
            return
        gap = time.monotonic() - started
        if gap < _IDLE_SILENT_DROP_S:
            return
        detail = ""
        if self.stream_proc is not None and not self.stream_proc.is_alive():
            code = self.stream_proc.exit_code()
            detail = f" The streamer exited (code {code})."
        tail = (self.stream_proc.tail(6) if self.stream_proc is not None else "")
        if tail:
            LOG.add("OpenMuse streamer output:\n" + tail, "error")
        self._log(f"No data ever arrived after {gap:.0f}s.{detail}")
        self._set_status(
            f"Connected but no data after {gap:.0f}s.{detail} Dropping the "
            "connection — switch the headband off and on, and it will retry.")
        if self._session_active:
            self.on_stop_session()
        self.disconnect_stream()

    def _drop_if_silent_idle(self):
        """Disconnect a stream that is open but delivering nothing.

        "Connected" with no data and no battery is the single most confusing
        state this app can show — it looks like a working headband, and the
        Connect button reads "Disconnect", so the obvious recovery is not even
        offered. Dropping it turns an ambiguous state into an honest one and
        lets the periodic re-scan pick the device up again.
        """
        if self.reader is None or self._session_active:
            return
        last = max(self._last_sample.values()) if self._last_sample else None
        gap = time.monotonic() - last if last else _IDLE_SILENT_DROP_S + 1
        if gap < _IDLE_SILENT_DROP_S:
            return
        self._log(f"Dropping silent connection (no data for {gap:.0f}s)")
        self._set_status(
            "Headband stopped sending data — connection dropped. Make sure it is "
            "switched on, then it will be picked up automatically.")
        self.disconnect_stream()

    def _abort_if_no_eeg(self):
        """End a session that has lost the brain data it exists to collect.

        A protocol that keeps running without EEG wastes the subject's time and
        writes a file with a hole in it, or nothing at all. That has already
        happened here once: a 9-minute run completed against a browned-out
        headband that still reported "connected", and the recording was empty.

        Only sessions that STARTED with EEG are aborted. Running a protocol with
        no headband at all is a deliberate, supported case (audio-only rehearsal),
        and must not be broken by this.
        """
        if not self._session_active or not self._session_needs_eeg:
            return
        last = self._last_sample.get(self._eeg_stream)
        gap = time.monotonic() - last if last else _EEG_LOSS_ABORT_S + 1
        if gap < _EEG_LOSS_ABORT_S:
            return
        LOG.add(f"SESSION ABORTED: no EEG for {gap:.0f}s", "error")
        self._log(f"Session aborted — no EEG for {gap:.0f}s")
        self._write_event("session", "aborted_no_eeg", f"{gap:.1f}s")
        self._speak("Signal lost. Ending the session.")
        self._set_status(
            f"Session ended — no EEG for {gap:.0f}s. The headband stopped "
            "streaming, so there was nothing left to record.")
        self.on_stop_session()
        QMessageBox.warning(
            self, "Session ended — no EEG",
            f"No EEG arrived for {gap:.0f} seconds, so the session was stopped.\n\n"
            "The headband reports 'connected' whenever a stream is open, even "
            "when it has browned out or dropped out of range, so a session can "
            "otherwise run to completion and record nothing.\n\n"
            "Check the headband is switched on, charged, and seated, then start "
            "again.")

    def _note_samples(self, stream_name, n_samples=0):
        """Called on every chunk; clears a stall and reports the gap."""
        now = time.monotonic()
        if stream_name == self._eeg_stream and n_samples:
            self._rate_count += int(n_samples)
        if stream_name in self._stalled:
            gap = now - self._stalled.pop(stream_name)
            self._escalated.pop(stream_name, None)
            self._dropouts.append((stream_name, datetime.now(), gap))
            LOG.add(f"STREAM RESUMED: {stream_name} after {gap:.1f}s", "log")
            self._log(f"Stream resumed: {stream_name} (lost {gap:.1f}s)")
            self._write_event("dropout", "stall_end", f"{stream_name} {gap:.1f}s")
            self._set_status(f"{_short(stream_name)} resumed (lost {gap:.1f}s)")
            self._sound(play_resolved, session_only=True)
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
        self.device_status.set_contacts(m.contact_per_channel)
        self._write_quality(m)
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
        # Re-entrancy guard. This handler tears the reader down, and tearing it
        # down can emit another error, which re-entered here — the crash log
        # showed _on_reader_error twice in one stack, ending in
        # QThread::~QThread() calling qFatal ("destroyed while still running")
        # and SIGABRT. The second entry destroyed the thread the first was
        # still unwinding.
        if getattr(self, "_handling_reader_error", False):
            self._log(f"Suppressed re-entrant stream error: {msg}")
            return
        self._handling_reader_error = True
        try:
            self._handle_reader_error(msg)
        finally:
            self._handling_reader_error = False

    def _handle_reader_error(self, msg):
        # Append whatever the OpenMuse subprocess actually said. Its output was
        # captured and then discarded, so the operator saw only the app's guess
        # ("is the streamer running?") while the streamer's real error — the one
        # naming the actual problem — sat unread in a pipe.
        tail = ""
        if self.stream_proc is not None:
            tail = self.stream_proc.tail(10)
            code = self.stream_proc.exit_code()
            if code is not None:
                tail = f"(streamer exited, code {code})\n{tail}"
            if code == -6:
                # SIGABRT from CoreBluetooth. macOS grants Bluetooth access per
                # application, and a process launched from a terminal that has
                # not been granted it is killed the instant it touches the
                # radio — before printing anything. The symptom is a streamer
                # that dies silently, which is indistinguishable from a headband
                # problem unless you know to look at the exit code.
                msg = ("Bluetooth permission denied.\n\n"
                       "The OpenMuse streamer was killed by macOS the moment it "
                       "tried to use Bluetooth (SIGABRT).\n\n"
                       "This is not a headband problem. macOS grants Bluetooth "
                       "per application: launch MuseStudio from Terminal.app "
                       "(or whichever terminal you granted Bluetooth access), "
                       "not from an embedded/tool terminal.\n\n"
                       "System Settings → Privacy & Security → Bluetooth lists "
                       "the apps that have it.")
                LOG.add("BLUETOOTH PERMISSION DENIED (streamer got SIGABRT)", "error")
                self._set_status("Bluetooth permission denied — launch from Terminal.app.")
                QMessageBox.critical(self, "Bluetooth permission denied", msg)
                self.disconnect_stream()
                return
        self._set_status("Stream error.")
        LOG.add(f"STREAM ERROR: {msg}", "error")
        if tail:
            LOG.add("OpenMuse streamer said:\n" + tail, "error")
            msg = f"{msg}\n\n--- OpenMuse output ---\n{tail}"
        QMessageBox.critical(self, "Stream error", msg)
        self.disconnect_stream()

    def _on_disconnected(self):
        self.plot.clear()
        # Emitted when the reader loop ends. Sessions may still run headband-free.
        pass

    def disconnect_stream(self):
        self._log("Disconnect")
        # A scan starting during teardown was visible in the crash: a
        # _ScanThread sat in find_devices while the reader was erroring.
        self._auto_connecting = False
        if self._session_active:
            self._end_session(aborted=True)
        if self.recorder is not None:
            self._stop_recording()
        reader, self.reader = self.reader, None
        if reader is not None:
            # Drop our reference FIRST, then shut down: a queued signal
            # arriving mid-teardown must not find a half-dead reader.
            reader.stop()
            if not reader.wait(5000):
                # Qt calls qFatal from ~QThread if the thread is still running
                # when the object is destroyed, which aborts the whole process.
                # Keeping a reference parks the object instead: leaking one
                # thread is survivable, aborting the session is not.
                self._log("Reader thread did not stop in 5s — parking it")
                self._parked_threads = getattr(self, "_parked_threads", [])
                self._parked_threads.append(reader)
            else:
                reader.deleteLater()
        self._eeg_stream = None
        if self.stream_proc is not None:
            self.stream_proc.stop()
            self.stream_proc = None
        self.device_status.set_connected(False)
        self.device_status.set_connected(False)
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

        if mode == "mindball":
            if not self._begin_mindball(record):
                return
            self._begin_session_ui()
            self._free_start = time.monotonic()
            self.session_banner.show_free(recording=record, compact=True)
            self.session_banner.set_elapsed(0)
            self._free_timer.start()
            return

        if mode == "flight":
            if not self._begin_flight(record):
                return
            self._begin_session_ui()
            self._free_start = time.monotonic()
            self.session_banner.show_free(recording=record, compact=True)
            self.session_banner.set_elapsed(0)
            self._free_timer.start()
            return

        if mode == "sleep":
            if not self._begin_sleep(record):
                return
            self._begin_session_ui()
            self._free_start = time.monotonic()
            self.session_banner.show_free(recording=record)
            self.session_banner.set_elapsed(0)
            self._free_timer.start()
            return

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
            if self.fullscreen_check.isChecked():
                self._open_fullscreen()
            self.runner.start(PROTOCOLS[mode])

    # -------------------------------------------------------- mindball mode
    def _begin_mindball(self, record):
        if not self._has_eeg():
            QMessageBox.warning(
                self, "No EEG",
                "Mindball needs a live EEG stream — connect the Muse first.")
            return False
        if record:
            self._start_recording()
            if self.recorder is None:
                return False
        ghost = latest_ghost(self.output_dir)
        if ghost is None:
            self._log("No past Mindball match — using a practice partner")
        else:
            self._log(f"Mindball opponent: {ghost.label} "
                      f"({ghost.duration():.0f}s recorded)")
        fs = 256.0
        if self.reader is not None and self._eeg_stream:
            fs = self.reader.sample_rate(self._eeg_stream) or 256.0
        self.mindball.set_channels(self._eeg_channels)
        telemetry = self.session.analysis_dir if (record and self.session) else None
        self.mindball.start(opponent=ghost, telemetry_dir=telemetry, fs=fs)
        self.view_tabs.setCurrentWidget(self.mindball_view)
        return True

    def _end_mindball(self):
        if self.mindball.is_running():
            self.mindball.stop()
            self._log("Mindball ended")

    def _on_mindball_status(self, text):
        self.mindball_view.set_status(text)
        self._set_status(text)
        self._speak(text)
        # Deliberately NOT written to events.csv. Status is emitted at the
        # render rate, and it flooded the file: session 205111's events.csv is
        # 83 status rows out of 86. events.csv is what every downstream analysis
        # aligns against, so it must carry events, not UI chatter. The text is
        # still in the session log.
        self._log(f"mindball: {text}")

    def _on_mindball_finished(self, summary):
        self._log(f"Mindball result: {summary}")
        if self._event_writer is not None:
            self._write_event("mindball", "result", json.dumps(summary))
        if self._session_active:
            self.on_stop_session()

    # ---------------------------------------------------------- flight mode
    def _begin_flight(self, record):
        """Start a flight. Returns False to abort.

        Requires a live EEG stream, and says so rather than starting: a flight
        with no signal would sit on the ground while the pilot wondered what
        they were doing wrong, which is the least useful failure available.
        """
        if not self._has_eeg():
            QMessageBox.warning(
                self, "No EEG",
                "Flight Mode needs a live EEG stream — connect the Muse first.\n\n"
                "The craft is flown on cortical alpha; without the headband there "
                "is nothing to fly it with.")
            return False
        if record:
            self._start_recording()
            if self.recorder is None:
                return False
        self.flight.set_channels(self._eeg_channels)
        telemetry = self.session.analysis_dir if (record and self.session) else None
        from fnt.musestudio.flight.controller import FLIGHT_TEST
        self.flight.start(telemetry_dir=telemetry, schedule=FLIGHT_TEST)
        self.view_tabs.setCurrentWidget(self.flight_view)
        self._log(f"Flight started (record={record})")
        return True

    def _end_flight(self):
        if self.flight.is_running():
            self.flight.stop()
            self._log("Flight ended")
        # Unconditional: stopping by any route (Stop button, Escape, EEG loss,
        # schedule end) must leave silence behind.
        self._silence_flight_audio()

    def _on_flight_cue(self, label, speech):
        """Speak the next instruction. The pilot's eyes are shut half the time,
        so the cue has to be audible, not written."""
        self._write_event("flight_cue", label, speech)
        self.flight_view.set_status(speech)
        self._set_status(speech)
        self._speak(speech)

    def _on_flight_finished(self):
        # Order matters: silence, then end the session, then speak. The audio and
        # the live view must stop the instant the test does — announcing first
        # left the engine running and the craft on screen for the length of an
        # utterance, which reads as "it did not actually stop".
        self._silence_flight_audio()
        self.flight_view.set_status("Flight test complete.")
        if self._session_active:
            self.on_stop_session()
        self._speak("Flight test complete.")

    def _silence_flight_audio(self):
        try:
            self.binaural.player.set_engine(0.0)
        except Exception:  # noqa: BLE001
            pass
        self.binaural.protocol_audio_off()

    def _on_flight_tone(self, height01):
        """Sonify altitude. Pitch rises with height; silent on the ground.

        Two octaves from 220 Hz, mapped exponentially because pitch perception
        is logarithmic — a linear frequency map would make the bottom half of
        the climb almost inaudible as a change. Mono on purpose: this has to
        work on laptop speakers, where left and right reach both ears anyway.
        """
        if not self.flight.is_running():
            return
        from fnt.musestudio.flight.sim import FlightPhase
        phase = self.flight.craft.state.phase
        player = self.binaural.player
        # Silent on the ground. Altitude reaches 0 a little before the craft
        # latches to LANDED (there is a grace period), and a tone still sounding
        # at zero altitude reads as "something is happening" when nothing is.
        if (phase in (FlightPhase.GROUNDED, FlightPhase.LANDED)
                or self.flight.craft.state.altitude <= 0.5):
            if player.is_playing():
                player.set_engine(0.0)
                self.binaural.protocol_audio_off()
            return
        freq = 220.0 * (2.0 ** (2.0 * float(height01)))
        if not player.is_playing():
            self.binaural.set_voicing("soft", 0.40)
            player.set_mode("binaural")
            player.set_volume(self.binaural._volume)
            try:
                player.play()
            except Exception:  # noqa: BLE001
                return
            self.binaural.fade_in()
        player.set_frequencies(freq, freq)
        # Engine swells with effort, so working to climb sounds different from
        # coasting down even when the two are at the same height.
        thrust = max(0.0, float(self.flight.craft.state.thrust))
        # Pitch an octave under the altitude tone so the engine rises and falls
        # with the craft; level follows thrust so effort is audible separately.
        player.set_engine(0.20 + 0.45 * thrust, freq=freq * 0.5)

    def _on_flight_status(self, text):
        self.flight_view.set_status(text)
        self._set_status(text)
        self._log(f"flight: {text}")   # log, not events.csv — see _on_mindball_status

    # ----------------------------------------------------------- sleep mode
    def _begin_sleep(self, record):
        """Set up an unattended overnight recording. Returns False to abort."""
        # Disk first — running out at 4 a.m. loses the night.
        if record:
            ok, msg = storage_check(self.output_dir, hours=9.0)
            if not ok:
                QMessageBox.warning(self, "Not enough disk space",
                                    f"Overnight recording needs room:\n\n{msg}")
                return False
            self._log(f"Sleep storage check: {msg}")

        if self._battery_last is not None and self._battery_last < 60:
            answer = QMessageBox.question(
                self, "Headband battery",
                f"Battery is {self._battery_last:.0f}%.\n\n"
                "A full night needs a full charge — Muse's own 9-hour figure "
                "comes from a firmware low-power mode this app cannot reach, so "
                "streaming will drain faster than that.\n\nStart anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return False

        self._sleep_mode = True
        # Nothing may make a sound until morning.
        self._silence_alerts = True
        if getattr(self, "speaker", None) is not None:
            self._speech_was_enabled = self.speaker.enabled
            self.speaker.enabled = False
        self.binaural.protocol_audio_off()

        # Stop drawing: nine hours of 30 fps plotting is fan noise beside your
        # head, and nobody is watching.
        self.plot.set_paused(True)
        self.values_panel.set_paused(True)

        if self._sleep_guard.start():
            self._log("Wake-lock held (caffeinate) — system stays awake")
        else:
            LOG.add("Could not hold a wake-lock; the Mac may sleep and end "
                    "the recording", "error")
            QMessageBox.warning(
                self, "No wake-lock",
                "Could not prevent the Mac from sleeping. Set Energy Saver to "
                "never sleep, or the recording will stop.")

        if record:
            self._start_recording(precision=2)      # ~1 GB instead of 1.5
            if self.recorder is None:
                self._end_sleep()
                return False
        self._log("Sleep mode started")
        self._set_status("Sleep recording — press Stop in the morning.")
        return True

    def _end_sleep(self):
        """Undo everything sleep mode changed."""
        if not self._sleep_mode:
            return
        self._sleep_guard.stop()
        self._sleep_mode = False
        self._silence_alerts = False
        if getattr(self, "speaker", None) is not None:
            self.speaker.enabled = getattr(self, "_speech_was_enabled", True)
        live = self.view_tabs.currentWidget() is self._live_tab
        self.plot.set_paused(not live)
        self.values_panel.set_paused(not live)
        self._log("Sleep mode ended")

    def on_stop_session(self):
        # Aborts a running protocol or ends free monitoring/recording.
        if self.runner.is_running():
            self.runner.abort()   # -> _on_runner_aborted handles teardown
        else:
            self._end_session(aborted=True)

    def _begin_session_ui(self):
        self._session_active = True
        # Recorded at start: a session begun WITHOUT a headband is a supported
        # audio-only rehearsal and must never be auto-aborted. One begun WITH
        # EEG has lost the thing it exists for if the stream dies.
        self._session_needs_eeg = self._has_eeg()
        # Swap the emphasis: while a session runs, stopping is the only action
        # that matters, so it takes the accent and the full width and the other
        # button gets out of the way. Previously "Stop" was the small secondary
        # button beside a large accented "Start + Record", which read as
        # inactive and was easy to miss entirely.
        self.start_btn.setText("■  Stop session")
        self.start_btn.setProperty("accent", "true")
        self.start_btn.setStyleSheet(
            f"background-color: #5b2020; border: 1px solid {theme.DANGER}; "
            f"color: #ffd9d9; font-weight: 600;")
        self.record_btn.hide()
        self.mode_combo.setEnabled(False)
        self.device_status.set_connect_enabled(False)

    def _open_fullscreen(self):
        """Hand the screen over to the subject-facing phase display."""
        if self.fullscreen_view is None:
            self.fullscreen_view = FullscreenSessionView()
            self.fullscreen_view.exited.connect(self._on_fullscreen_escape)
            self.fullscreen_view.continue_pressed.connect(self._on_continue)
        self.fullscreen_view.showFullScreen()
        self.fullscreen_view.raise_()
        self.fullscreen_view.activateWindow()
        self.fullscreen_view.setFocus()
        self._log("Full-screen session view opened")

    def _on_fullscreen_escape(self):
        """Escape ends the session, rather than just leaving full screen.

        Previously it dropped back to the main window with the session still
        running, which reads as "the program is still going in the background"
        and is genuinely confusing. Escape is the universal get-me-out key, so
        it now means what it looks like it means. A recording in progress asks
        first, because an accidental keypress should not throw away a run.
        """
        if self._session_active and self.recorder is not None:
            reply = QMessageBox.question(
                self, "Stop the session?",
                "A recording is in progress.\n\nStop the session and save what "
                "has been captured so far?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                self._open_fullscreen()      # they meant to stay
                return
        self._close_fullscreen()
        if self._session_active:
            self.on_stop_session()

    def _close_fullscreen(self):
        if self.fullscreen_view is not None:
            self.fullscreen_view.close()
            self.fullscreen_view = None
            self._log("Full-screen session view closed")
            self.raise_()
            self.activateWindow()

    def _end_session(self, aborted=False):
        self._free_timer.stop()
        self._end_sleep()
        # Before _stop_recording, so the telemetry file is closed and flushed
        # while the session directory is still open.
        self._end_flight()
        self._end_mindball()
        self._close_fullscreen()
        self.binaural.protocol_audio_off()
        if self.recorder is not None:   # protocol's Done phase may have stopped it already
            self._stop_recording()
        self._session_active = False
        self._session_needs_eeg = False
        self._recording_enabled = False
        self._current_phase = None
        self._sync_hold_start = None
        self.start_btn.setText("Start")
        self.start_btn.setProperty("accent", "false")
        self.start_btn.setStyleSheet("")
        self.record_btn.show()
        self.record_btn.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.device_status.set_connect_enabled(True)
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
        if self.fullscreen_view is not None:
            self.fullscreen_view.show_phase(phase, index, total, waiting)
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
            self._sound(play_cue)
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
        elif action == "audio_check":
            # Runs in the Set-up phase, BEFORE recording and before the subject
            # closes their eyes. Two jobs: say out loud which device the Mac is
            # about to play through, and give the subject a deliberately QUIET
            # reference tone to set their system volume against.
            #
            # The tone is fixed at a low level on purpose. The safe direction to
            # get the level wrong is too soft -- the subject raises the system
            # volume until it is comfortable, and the app never chooses a
            # loudness on behalf of somebody wearing in-ear buds. Detection is a
            # prompt, never an interlock: macOS cannot tell us the output volume,
            # and no API knows what is actually in someone's ears.
            out, msg = audio_advice()
            self._audio_output = out
            # Hardware decides what this session can collect. Re-detected here
            # rather than at start-up, because the subject may only plug their
            # earbuds in once they are sitting down.
            self._audio_cap = audio_capability(out)
            self.runner.set_capabilities(self._audio_cap.tokens())
            for note in self._audio_cap.notes:
                self._log(f"Audio: {note}")
            self._log(f"Audio output: {out.name!r} transport={out.transport!r} "
                      f"kind={out.kind}")
            self._set_status(msg)
            if out.kind == "speakers":
                self._log("WARNING: default output is not headphones.")
            self._sound(lambda: play_cue(freq=660.0, dur=0.5, volume=AUDIO_CHECK_VOLUME))
        elif action == "lateral_cues":
            # Alternating L/R tones across the phase. Fire-and-forget: the cue
            # sequence is scheduled once and runs on the audio device's own
            # clock, so it costs the GUI thread nothing during the block.
            sides = ["L", "R"] * 10
            self._sound(lambda: play_lateral_sequence(sides))
            self._write_event("marker", "lateral_cues", "".join(sides))
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
                timbre=p.get("timbre"), depth=p.get("depth"),
            )
        elif action == "audio_control":
            p = phase.params
            self.binaural.protocol_control_tone(
                p.get("base", 200), timbre=p.get("timbre"), depth=p.get("depth"))
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

    def _on_mode_changed(self, _index=None):
        """Keep the blurb in step, and never rest on a heading row."""
        key = self.mode_combo.currentData()
        if key is not None:
            self._settings.setValue("last_session", key)
        if key is None:                     # a disabled heading or separator
            for i in range(self.mode_combo.currentIndex() + 1,
                           self.mode_combo.count()):
                if self.mode_combo.itemData(i) is not None:
                    self.mode_combo.setCurrentIndex(i)
                    return
            return
        self.mode_blurb.setText(self._mode_blurbs.get(key, ""))

    def _on_phase_skipped(self, phase, missing):
        """A phase the current hardware cannot support was skipped.

        Written to events.csv as well as the log, so a later analysis can tell
        an omitted block from a failed one — the two look identical if all you
        have is a gap in the timeline.
        """
        self._log(f"Phase skipped: {phase.name} (needs {missing})")
        self._write_event("phase_skipped", phase.name, missing)

    def _on_runner_tick(self, remaining, duration):
        self.session_banner.set_countdown(remaining)
        if self.fullscreen_view is not None:
            self.fullscreen_view.set_countdown(remaining)
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
        self._sound(play_complete)      # distinct from a phase change, for eyes-closed use
        self._end_session(aborted=False)
        self._set_status("Protocol complete — recording saved.")

    def _on_runner_aborted(self):
        self._log("Protocol aborted")
        self._end_session(aborted=True)

    # --------------------------------------------------------------- recording
    def _start_recording(self, precision=6):
        try:
            self.session = RecordingSession(self.output_dir, subject=self.subject_id())
            self.recorder = MuseRecorder(self.session.muse_dir, precision=precision)
        except Exception as exc:  # noqa: BLE001
            self.session = None
            QMessageBox.critical(self, "Recording failed", str(exc))
            return
        if self.reader is not None:      # no headband -> only video/audio recorded
            self.reader.start_recording(self.recorder)
        # Route each artifact to its Data subfolder by provenance.
        self._open_audio_log(self.session.audio_dir)       # stimulus log
        self._open_sync_log(self.session.analysis_dir)     # derived PLV
        self._open_quality_log(self.session.analysis_dir)  # per-electrode quality
        self._open_event_log(self.session.events_dir)      # protocol timeline
        cam_note = ""
        if self.webcam is not None:
            self.webcam.start_recording(self.session.video_dir)  # opens on first frame
            cam_note = " + webcam"
        # Start the action log (flushes the buffered lead-up) and snapshot config.
        self.device_status.set_recording(True)
        sid = self.subject_id()
        if sid:
            self.subjects.upsert(sid, handedness=self.handed_combo.currentData(),
                                 sex=self.sex_combo.currentData(),
                                 hair_over_ears=self.hair_combo.currentData())
            self.subjects.note_session(sid)
            self._settings.setValue('subject_id', sid)
            self._log(f'Subject: {sid}')
        self.logger.start_file(self.session.log_path)
        self.session.write_config(self._build_config())
        self._log(f"Recording started -> {self.session.root}")
        self._set_status(f"Recording{cam_note} to {self.session.name}")

    def _stop_recording(self):
        self._close_event_log()
        self._close_audio_log()
        self._close_sync_log()
        self._close_quality_log()
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
        self.device_status.set_recording(False)
        self.recorder = None
        self.session = None
        self._set_status(f"Saved to {root} ({summary})")
        # Surface the result immediately — after an eyes-closed session you
        # shouldn't have to go hunting to find out whether it worked. Deferred
        # one event-loop turn: this method can run inside a protocol phase's
        # action loop, and opening a modal questionnaire from there would block
        # the remaining actions and the banner update mid-phase.
        QTimer.singleShot(0, lambda: self._show_summary(root, counts, frames))

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
            "app": "FieldNeuroToolbox / MuseStudio",
            "subject": self._subject_block(),
            # Reproducibility. The installed-package version is stamped at
            # install time and read 0.1.7 in every recording while pyproject was
            # already at 0.1.10 — useless for telling which control law produced
            # a telemetry file. The git SHA is the only thing that actually
            # identifies the code.
            "git_sha": _git_sha(),
            # Anchor the LSL clock to wall time. Every data file is on the LSL
            # monotonic clock, which aligns the streams to each other perfectly
            # but to nothing outside this process — a stimulus PC, a second
            # recorder, or a camera cannot be aligned without this pair.
            "clock_anchor": {"lsl": round(_lsl_now(), 6),
                             "utc": datetime.now(timezone.utc).isoformat()},
            # Recorded so sessions are never silently pooled across different
            # audio hardware — a speakers session has strictly fewer feedback
            # dimensions than a headphones one and is not the same experiment.
            "audio": (self._audio_cap.as_dict()
                      if getattr(self, "_audio_cap", None) else None),
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

    # -------------------------------------------------------------- subject
    def _load_subjects(self):
        """Populate the subject list from the registry beside the recordings."""
        self.subjects = SubjectRegistry(self.output_dir)
        current = self.subject_combo.currentText()
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        self.subject_combo.addItems(self.subjects.ids())
        last = self._settings.value("subject_id", "")
        self.subject_combo.setCurrentText(current or last or "")
        self.subject_combo.blockSignals(False)
        self._on_subject_changed(self.subject_combo.currentText())

    def _on_subject_changed(self, text):
        """Fill in what we already know about this subject."""
        record = self.subjects.get(sanitize_id(text)) if getattr(
            self, "subjects", None) else {}
        for key, combo in (("handedness", self.handed_combo),
                           ("sex", self.sex_combo),
                           ("hair_over_ears", self.hair_combo)):
            value = record.get(key)
            if value:
                idx = combo.findData(value)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def subject_id(self):
        return sanitize_id(self.subject_combo.currentText())

    def _subject_block(self):
        """The subject metadata stored in recording_config.json."""
        sid = self.subject_id()
        record = self.subjects.get(sid) if sid else {}
        return {
            "id": sid,
            "handedness": self.handed_combo.currentData(),
            "sex": self.sex_combo.currentData(),
            "hair_over_ears": self.hair_combo.currentData(),
            "session_label": self.session_label_edit.text().strip(),
            "prior_sessions": int(record.get("sessions", 0)),
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

    def _sound(self, play_fn, session_only=False):
        """Play a cue unless the session is running silent.

        Overnight recordings suppress every tone: an alert at 3 a.m. is a worse
        outcome than the dropout it is reporting.

        ``session_only`` marks warning cues. Outside a running session a lost
        stream is almost always deliberate — you switched the headband off —
        and beeping at someone who is sitting there reviewing their data is
        just noise. Those events are still written to the log.
        """
        if self._silence_alerts:
            return
        if session_only and not self._session_active:
            return
        play_fn()

    def _speak(self, text):
        """Speak if guidance is enabled; harmless no-op otherwise."""
        if self._silence_alerts:      # overnight: nothing may wake the sleeper
            return
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
            self._load_subjects()      # registry lives beside the recordings
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

    def _write_quality(self, m):
        """Per-electrode signal quality, once per band update, on the LSL clock.

        This is the column that was missing from every recording ever made.
        Contact quality reached the status dots and nothing else: the only path
        to disk was synchrony.csv's contact_ok, and that writer sits behind a
        gate that fires for exactly one protocol phase in the whole app, so the
        file is header-only in all 13 sessions on disk.

        The consequence was that "was TP10 good during minute 12?" — the single
        most common question about a Muse recording, given that ear electrodes
        are the documented failure mode — could not be answered after the fact
        from any file. Now it can.
        """
        if self._quality_writer is None:
            return
        try:
            row = [f"{_lsl_now():.6f}"]
            for ch in EEG_ELECTRODES:
                ok = None
                for name, val in (m.contact_per_channel or {}).items():
                    if ch in str(name).upper():
                        ok = val
                        break
                rel = (m.relative or {}).get(ch) or {}
                row += ["" if ok is None else int(bool(ok)),
                        round(float(rel.get("alpha", 0.0)), 5)]
            self._quality_writer.writerow(row)
            self._quality_file.flush()
        except Exception:  # noqa: BLE001
            pass

    def _open_quality_log(self, session_dir):
        try:
            self._quality_file = open(
                os.path.join(session_dir, "signal_quality.csv"), "w", newline="")
            self._quality_writer = csv.writer(self._quality_file)
            head = ["lsl_timestamp"]
            for ch in EEG_ELECTRODES:
                head += [f"{ch}_ok", f"{ch}_alpha_rel"]
            self._quality_writer.writerow(head)
        except Exception:  # noqa: BLE001
            self._quality_file = None
            self._quality_writer = None

    def _close_quality_log(self):
        if getattr(self, "_quality_file", None) is not None:
            try:
                self._quality_file.close()
            except Exception:  # noqa: BLE001
                pass
        self._quality_file = None
        self._quality_writer = None

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
