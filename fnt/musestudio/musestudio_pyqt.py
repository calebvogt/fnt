"""MuseStudio main window — connect to a Muse S Athena over BLE (via OpenMuse),
stream EEG/fNIRS over LSL, live-plot the signals, and record them to CSV.

V1 scope: connect -> stream -> record -> live plot. Webcam and audio biofeedback
are future iterations (see the project plan).
"""

import os

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from fnt.musestudio.live_plot import MultiChannelScrollPlot
from fnt.musestudio.muse_stream import (
    LSLReaderThread, MuseRecorder, MuseStreamProcess, find_devices,
)


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


class MuseStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stream_proc = None
        self.reader = None
        self.recorder = None
        self.scan_thread = None
        self.output_dir = os.path.join(os.path.expanduser("~"), "Documents")

        self.setWindowTitle("MuseStudio - FieldNeuroToolbox")
        self.resize(1100, 750)
        self._init_ui()

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
        self.device_combo.setMinimumWidth(280)
        self.device_combo.addItem("No devices — click Scan", None)
        controls.addWidget(self.device_combo)

        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self.on_scan)
        controls.addWidget(self.scan_btn)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect)
        controls.addWidget(self.connect_btn)

        self.record_btn = QPushButton("Record")
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self.on_record)
        controls.addWidget(self.record_btn)

        controls.addStretch()
        self.folder_btn = QPushButton("Save Folder…")
        self.folder_btn.clicked.connect(self.on_choose_folder)
        controls.addWidget(self.folder_btn)
        root.addLayout(controls)

        self.folder_label = QLabel(f"Save to: {self.output_dir}")
        self.folder_label.setStyleSheet("color: #999999;")
        root.addWidget(self.folder_label)

        # Live plot.
        self.plot = MultiChannelScrollPlot()
        root.addWidget(self.plot, stretch=1)

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
                'Install the muse extra:\n    pip install -e ".[muse]"',
            )
            return

        self.reader = LSLReaderThread()
        self.reader.samples_ready.connect(self.plot.add_samples)
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
        self.plot.set_channel_names(self.reader.channel_names())
        self.record_btn.setEnabled(True)
        self._set_status(f"Connected. Streaming: {', '.join(names)}")

    def _on_reader_error(self, msg):
        self._set_status("Stream error.")
        QMessageBox.critical(self, "Stream error", msg)
        self.disconnect_stream()

    def _on_disconnected(self):
        # Emitted when the reader loop ends (clean stop or error).
        self.record_btn.setEnabled(False)

    def disconnect_stream(self):
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
        self.record_btn.setEnabled(False)
        self._set_status("Disconnected.")

    def on_record(self):
        if self.recorder is None:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        if self.reader is None:
            return
        try:
            self.recorder = MuseRecorder(self.output_dir)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Recording failed", str(exc))
            return
        self.reader.start_recording(self.recorder)
        self.record_btn.setText("Stop")
        self._set_status(f"Recording to {self.recorder.session_dir}")

    def _stop_recording(self):
        if self.reader is not None:
            session_dir = self.reader.stop_recording()
        else:
            session_dir = self.recorder.stop() if self.recorder else None
        counts = self.recorder.counts() if self.recorder else {}
        self.recorder = None
        self.record_btn.setText("Record")
        summary = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "no samples"
        self._set_status(f"Saved to {session_dir} ({summary})")

    def on_choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose save folder", self.output_dir)
        if folder:
            self.output_dir = folder
            self.folder_label.setText(f"Save to: {folder}")

    def _set_status(self, msg):
        self.status_label.setText(msg)

    def closeEvent(self, event):
        self.disconnect_stream()
        if self.scan_thread is not None and self.scan_thread.isRunning():
            self.scan_thread.wait(2000)
        super().closeEvent(event)
