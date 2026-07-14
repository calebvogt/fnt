"""MuseStudio main window — connect to a Muse S Athena over BLE (via OpenMuse),
stream EEG/optics over LSL, live-plot the signals, and record them to CSV.

Brings together the pieces: live plots + numeric channel table, battery
readout, synchronized webcam capture, a binaural-beat generator with
closed-loop control from interhemispheric synchrony (PLV), a head-map + meter,
and a guided session runner (free record or a timed protocol).
"""

import csv
import os
import time

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from fnt.musestudio.binaural import BinauralPanel, play_cue
from fnt.musestudio.channel_table import LiveValuesPanel
from fnt.musestudio.live_plot import MultiChannelScrollPlot
from fnt.musestudio.muse_stream import (
    LSLReaderThread, MuseRecorder, MuseStreamProcess, find_devices,
)
from fnt.musestudio.neuro_widgets import NeuroPanel
from fnt.musestudio.protocol import PROTOCOLS, ProtocolRunner
from fnt.musestudio.synchrony import SynchronyAnalyzer
from fnt.musestudio.webcam import WebcamThread


def _fmt_time(seconds):
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:d}:{seconds % 60:02d}"


def _is_eeg(stream_name):
    return "EEG" in stream_name.upper()


def _is_battery(stream_name):
    return "BATTERY" in stream_name.upper()


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

    def show_free(self):
        self.title.setText("Free recording")
        self.instruction.setText(
            "Recording all active streams (EEG, optics, webcam, audio events). "
            "Press Stop when you are finished."
        )
        self.progress.setRange(0, 0)  # busy indicator
        self.continue_btn.hide()
        self.show()

    def set_elapsed(self, seconds):
        self.countdown.setText(_fmt_time(seconds))

    def show_phase(self, phase, index, total, waiting):
        self.title.setText(f"Phase {index + 1}/{total} · {phase.name}")
        self.instruction.setText(phase.instruction)
        self.progress.setRange(0, total)
        self.progress.setValue(index + 1)
        self.countdown.setText("" if waiting else "")
        self.continue_btn.setVisible(waiting)
        self.show()

    def set_countdown(self, remaining):
        self.countdown.setText(_fmt_time(remaining))


class MuseStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stream_proc = None
        self.reader = None
        self.recorder = None
        self.scan_thread = None
        self.webcam = None
        self._audio_file = None
        self._audio_writer = None
        self._sync_file = None
        self._sync_writer = None
        self._eeg_stream = None
        self._eeg_channels = []
        self._session_active = False
        self._free_start = 0.0
        self.output_dir = os.path.join(os.path.expanduser("~"), "Documents")

        self.setWindowTitle("MuseStudio - FieldNeuroethologyToolbox")
        self.resize(1280, 800)
        self.setMinimumSize(960, 600)
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

    # ------------------------------------------------------------------ UI
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)

        # Experimental banner.
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

        # Control bar.
        controls = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(240)
        self.device_combo.addItem("No devices — click Scan", None)
        self.device_combo.setToolTip("Muse devices found by the last scan. Pick one, then Connect.")
        controls.addWidget(self.device_combo)

        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self.on_scan)
        self.scan_btn.setToolTip("Search over Bluetooth for nearby Muse headbands.")
        controls.addWidget(self.scan_btn)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect)
        self.connect_btn.setToolTip("Start streaming EEG/optics from the selected Muse.")
        controls.addWidget(self.connect_btn)

        # Session controls: pick a mode, then Start.
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Free record", "free")
        for key, proto in PROTOCOLS.items():
            self.mode_combo.addItem(proto.name, key)
        self.mode_combo.setToolTip(
            "Free record: capture everything until you press Stop.\n"
            "Protocols: a guided, timed trial with on-screen instructions."
        )
        controls.addWidget(self.mode_combo)

        self.start_btn = QPushButton("Start")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.on_start_or_stop)
        self.start_btn.setToolTip("Begin the selected session (records to the save folder).")
        controls.addWidget(self.start_btn)

        self.camera_combo = QComboBox()
        for i in range(4):
            self.camera_combo.addItem(f"Camera {i}", i)
        self.camera_combo.setToolTip("Which webcam to preview/record. Try another index if the feed is blank.")
        controls.addWidget(self.camera_combo)

        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.clicked.connect(self.on_toggle_camera)
        self.camera_btn.setToolTip("Turn the webcam preview on/off. When on, video is recorded during a session.")
        controls.addWidget(self.camera_btn)

        controls.addStretch()

        # Numeric battery readout (from the Muse-BATTERY stream; not plotted).
        self.battery_label = QLabel("Battery: —")
        self.battery_label.setStyleSheet(
            "color: #cccccc; font-weight: bold; padding: 0 10px;"
        )
        self.battery_label.setToolTip("Muse battery level (updates every few seconds).")
        controls.addWidget(self.battery_label)

        self.folder_btn = QPushButton("Save Folder…")
        self.folder_btn.clicked.connect(self.on_choose_folder)
        self.folder_btn.setToolTip("Where each session's timestamped folder of CSVs/video is written.")
        controls.addWidget(self.folder_btn)
        root.addLayout(controls)

        self.folder_label = QLabel(f"Save to: {self.output_dir}")
        self.folder_label.setStyleSheet("color: #999999;")
        root.addWidget(self.folder_label)

        # Guided-session instruction/countdown strip (hidden until a run starts).
        self.session_banner = SessionBanner()
        self.session_banner.continue_clicked.connect(self._on_continue)
        root.addWidget(self.session_banner)

        # Body: scrolling plot (left) beside a right column holding the webcam
        # preview (top) and the live numeric channel table (bottom).
        split = QSplitter(Qt.Horizontal)
        self.plot = MultiChannelScrollPlot()
        self.plot.setToolTip(
            "Live scrolling signals, newest on the right. One stacked panel per "
            "stream (EEG, optics); traces are auto-scaled per channel."
        )
        split.addWidget(self.plot)

        right = QSplitter(Qt.Vertical)
        self.camera_view = QLabel("Camera off")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumHeight(220)
        self.camera_view.setStyleSheet(
            "background-color: #111111; color: #777777; border: 1px solid #3f3f3f;"
        )
        self.camera_view.setToolTip("Live webcam preview. Recorded (with frame timestamps) during a session when the camera is on.")
        right.addWidget(self.camera_view)
        self.values_panel = LiveValuesPanel()
        self.values_panel.setToolTip("Latest value of every channel — the same columns written to the per-stream CSVs.")
        right.addWidget(self.values_panel)
        right.setStretchFactor(0, 1)
        right.setStretchFactor(1, 1)
        right.setSizes([300, 400])
        split.addWidget(right)

        # Neurofeedback column: head map + synchrony meter + controls.
        self.neuro = NeuroPanel()
        split.addWidget(self.neuro)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 1)
        split.setSizes([720, 300, 320])
        root.addWidget(split, stretch=1)

        # Synchrony analyzer (fed EEG on the GUI thread; QTimer drives it).
        self.analyzer = SynchronyAnalyzer(self)
        self.analyzer.metrics_updated.connect(self._on_metrics)
        self.analyzer.status.connect(self.neuro.set_status)
        self.analyzer.baseline_progress.connect(
            lambda f: self.neuro.set_status(f"Calibrating baseline… {int(f * 100)}%")
        )
        self.analyzer.baseline_done.connect(
            lambda floor: self.neuro.set_status(f"Baseline set (resting PLV {floor:.2f}).")
        )
        self.neuro.band_changed.connect(self.analyzer.set_band)
        self.neuro.calibrate_requested.connect(lambda: self.analyzer.start_baseline(30.0))

        # Binaural-beat generator (audio protocol, logged to the session folder).
        self.binaural = BinauralPanel()
        self.binaural.tone_event.connect(self._on_audio_event)
        root.addWidget(self.binaural)

        # Status line.
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("color: #cccccc;")
        root.addWidget(self.status_label)

    # --------------------------------------------------------------- actions
    def on_scan(self):
        self.scan_btn.setEnabled(False)
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
            return
        for d in devices:
            self.device_combo.addItem(f"{d['name']}  ({d['address']})", d["address"])
        self._set_status(f"Found {len(devices)} device(s).")

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
        ch_names = self.reader.channel_names()
        # Battery is shown as a number, not plotted or listed as a channel.
        plot_names = {k: v for k, v in ch_names.items() if not _is_battery(k)}
        self.plot.set_channel_names(plot_names)
        self.values_panel.set_channel_names(plot_names)
        # Identify the EEG stream for synchrony analysis.
        self._eeg_stream = next((n for n in names if _is_eeg(n)), None)
        self._eeg_channels = ch_names.get(self._eeg_stream, []) if self._eeg_stream else []
        self.analyzer.reset()
        if self._eeg_stream:
            fs = self.reader.sample_rate(self._eeg_stream)
            if fs:
                self.analyzer.set_sample_rate(fs)
        self.start_btn.setEnabled(True)
        self._set_status(f"Connected. Streaming: {', '.join(names)}")

    def _on_samples(self, stream_name, timestamps, data):
        """Route incoming chunks: battery -> numeric label, everything else ->
        the scrolling plot and the live values table."""
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

    def _on_metrics(self, m):
        """Synchrony update: refresh visuals, drive audio, log if recording."""
        self.neuro.update_metrics(m)
        # On good contact drive the tone from synchrony; on lost contact degrade
        # toward rough/quiet so a slipping headband is audible feedback.
        self.binaural.apply_synchrony(m.level if m.contact_ok else 0.0)
        if self._sync_writer is not None:
            self._sync_writer.writerow([
                f"{_lsl_now():.6f}", m.band,
                f"{m.plv.get('frontal', 0.0):.4f}",
                f"{m.plv.get('temporal', 0.0):.4f}",
                f"{m.plv_combined:.4f}", f"{m.level:.4f}",
                int(m.contact_ok), int(m.calibrated),
            ])
            self._sync_file.flush()

    def _on_reader_error(self, msg):
        self._set_status("Stream error.")
        QMessageBox.critical(self, "Stream error", msg)
        self.disconnect_stream()

    def _on_disconnected(self):
        # Emitted when the reader loop ends (clean stop or error).
        self.start_btn.setEnabled(False)

    def disconnect_stream(self):
        if self._session_active:
            self._end_session(aborted=True)
        if self.recorder is not None:
            self._stop_recording()
        if self.reader is not None:
            self.reader.stop()
            self.reader.wait(3000)
            self.reader = None
        if self.stream_proc is not None:
            self.stream_proc.stop()
            self.stream_proc = None
        self.connect_btn.setText("Connect")
        self.connect_btn.setEnabled(True)
        self.device_combo.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.start_btn.setEnabled(False)
        self.battery_label.setText("Battery: —")
        self._set_status("Disconnected.")

    # --------------------------------------------------------------- session
    def on_start_or_stop(self):
        if self._session_active:
            self.on_stop_session()
        else:
            self.on_start_session()

    def on_start_session(self):
        if self.reader is None:
            QMessageBox.warning(self, "Not connected", "Connect a Muse before starting.")
            return
        mode = self.mode_combo.currentData()
        self._begin_session_ui()
        if mode == "free":
            self._start_recording()
            if self.recorder is None:      # recording failed to start
                self._end_session(aborted=True)
                return
            self._free_start = time.monotonic()
            self.session_banner.show_free()
            self.session_banner.set_elapsed(0)
            self._free_timer.start()
        else:
            self.runner.start(PROTOCOLS[mode])

    def on_stop_session(self):
        # Aborts a running protocol or ends free recording.
        if self.runner.is_running():
            self.runner.abort()   # -> _on_runner_aborted handles teardown
        else:
            self._end_session(aborted=True)

    def _begin_session_ui(self):
        self._session_active = True
        self.start_btn.setText("Stop")
        self.mode_combo.setEnabled(False)
        self.connect_btn.setEnabled(False)

    def _end_session(self, aborted=False):
        self._free_timer.stop()
        self.binaural.protocol_audio_off()
        if self.recorder is not None:   # protocol's Done phase may have stopped it already
            self._stop_recording()
        self._session_active = False
        self.start_btn.setText("Start")
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
        self.session_banner.show_phase(phase, index, total, waiting)
        # Cue beep marks the transition for eyes-closed users. Skip it when the
        # phase starts a tone (the tone itself is the cue) or one is already
        # playing, to avoid audio-device contention.
        if "audio_on" not in phase.actions and not self.binaural.player.is_playing():
            play_cue()
        for action in phase.actions:
            if not self._run_phase_action(action, phase):
                break   # a failed action aborted the protocol

    def _run_phase_action(self, action, phase):
        """Perform a phase action; return False if it aborted the protocol."""
        if action == "start_recording":
            if self.recorder is None:
                self._start_recording()
                if self.recorder is None:   # recording failed to start
                    self._set_status("Recording failed to start — protocol aborted.")
                    self.runner.abort()
                    return False
        elif action == "calibrate":
            self.analyzer.start_baseline(float(phase.duration or 30.0))
        elif action == "audio_on":
            p = phase.params
            self.binaural.protocol_audio_on(
                p.get("base", 200), p.get("beat", 10), p.get("closed_loop", False)
            )
        elif action == "audio_off":
            self.binaural.protocol_audio_off()
        elif action == "stop_recording":
            if self.recorder is not None:
                self._stop_recording()
        return True

    def _on_runner_tick(self, remaining, _duration):
        self.session_banner.set_countdown(remaining)

    def _on_continue(self):
        self.runner.advance()

    def _on_runner_finished(self):
        self._end_session(aborted=False)
        self._set_status("Protocol complete — recording saved.")

    def _on_runner_aborted(self):
        self._end_session(aborted=True)

    # --------------------------------------------------------------- recording
    def _start_recording(self):
        if self.reader is None:
            return
        try:
            self.recorder = MuseRecorder(self.output_dir)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Recording failed", str(exc))
            return
        self.reader.start_recording(self.recorder)
        # Log the binaural-beat protocol and synchrony to the same session folder.
        self._open_audio_log(self.recorder.session_dir)
        self._open_sync_log(self.recorder.session_dir)
        # If the camera is live, record video into the same session folder,
        # timestamped on the shared LSL clock for sync with the Muse data.
        cam_note = ""
        if self.webcam is not None:
            self.webcam.start_recording(self.recorder.session_dir)  # opens on first frame
            cam_note = " + webcam"
        self._set_status(f"Recording{cam_note} to {self.recorder.session_dir}")

    def _stop_recording(self):
        self._close_audio_log()
        self._close_sync_log()
        frames = self.webcam.stop_recording() if self.webcam is not None else 0
        if self.reader is not None:
            session_dir = self.reader.stop_recording()
        else:
            session_dir = self.recorder.stop() if self.recorder else None
        counts = self.recorder.counts() if self.recorder else {}
        self.recorder = None
        summary = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "no samples"
        if frames:
            summary += f", webcam: {frames} frames"
        self._set_status(f"Saved to {session_dir} ({summary})")

    def on_choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose save folder", self.output_dir)
        if folder:
            self.output_dir = folder
            self.folder_label.setText(f"Save to: {folder}")

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

    def _open_sync_log(self, session_dir):
        self._sync_file = open(
            os.path.join(session_dir, "synchrony.csv"), "w", newline=""
        )
        self._sync_writer = csv.writer(self._sync_file)
        self._sync_writer.writerow(
            ["lsl_timestamp", "band", "plv_frontal", "plv_temporal",
             "plv_combined", "level", "contact_ok", "calibrated"]
        )

    def _close_sync_log(self):
        if self._sync_file is not None:
            self._sync_file.close()
            self._sync_file = None
            self._sync_writer = None

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
        self.webcam = WebcamThread(camera_index=index)
        self.webcam.frame_ready.connect(self._on_frame)
        self.webcam.opened.connect(self._on_camera_opened)
        self.webcam.error.connect(self._on_camera_error)
        self.webcam.start()
        self.camera_btn.setText("Stop Camera")
        self.camera_combo.setEnabled(False)

    def _stop_camera(self):
        if self.webcam is None:
            return
        self.webcam.stop()
        self.webcam.wait(3000)
        self.webcam = None
        self.camera_btn.setText("Start Camera")
        self.camera_combo.setEnabled(True)
        self.camera_view.setText("Camera off")

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

    def closeEvent(self, event):
        self.disconnect_stream()
        self._stop_camera()
        self.binaural.close_audio()
        if self.scan_thread is not None and self.scan_thread.isRunning():
            self.scan_thread.wait(2000)
        super().closeEvent(event)
