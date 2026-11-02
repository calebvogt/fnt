#!/usr/bin/env python3
"""Fiber Photometry tool (PyQt5): batch dF/F, z-score and video alignment, with a preview.

Left: recordings, pipeline and settings, run controls and log. Right: the selected recording's
video with its traces aligned underneath, to scrub or play (``preview_view.py``). Runs write a
dated run folder beside the recordings (``batch.py``). The pipelines are described in
``processing.py``; GuPPy's and PhAT's methods keep their licences in ``guppy_methods.py`` and
``phat_methods.py``.

Standalone:  python -m fnt.fiberPhotometry.cli gui [folder or recordings]
"""
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)

from fnt.theme import BLUE_BUTTON_STYLE, apply_dark_theme

from .batch import Settings, find_recordings, load_and_process, run_batch
from .labjack import VIDEO_SUFFIXES
from .processing import PIPELINE_LABELS, PIPELINES
from .preview_view import PreviewPane
from .sync import recording_frame_times

ZSCORE_METHODS = [("standard", "Whole session (mean, SD)"),
                  ("modified", "Robust (median, MAD)"),
                  ("baseline", "Baseline window (mean, SD)")]


class RunWorker(QThread):
    progress = pyqtSignal(str)
    file_done = pyqtSignal(int, str)
    all_done = pyqtSignal(str)

    def __init__(self, files: List[Path], settings: Settings, out_dir: Optional[Path]):
        super().__init__()
        self.files, self.settings, self.out_dir = files, settings, out_dir
        self._stop = False

    def request_stop(self):
        self._stop = True
        self.progress.emit("Stop requested: finishing the current step...")

    def run(self):
        try:
            out = run_batch(self.files, self.settings, self.out_dir, log=self.progress.emit,
                            should_stop=lambda: self._stop,
                            on_file=lambda k, row: self.file_done.emit(k, row.get("status", "")))
            self.all_done.emit(f"Finished. Outputs in {out}")
        except Exception as e:
            self.all_done.emit(f"Run failed: {e}")


class PreviewLoader(QThread):
    loaded = pyqtSignal(object, object)        # data, result
    failed = pyqtSignal(str)

    def __init__(self, recording: Path, settings: Settings):
        super().__init__()
        self.recording, self.settings = recording, settings

    def run(self):
        try:
            data, r = load_and_process(self.recording, self.settings)
            self.loaded.emit(data, r)
        except Exception as e:
            self.failed.emit(str(e))


class FiberPhotometryWindow(QWidget):
    def __init__(self, parent=None, paths=None):
        super().__init__(parent)
        self.setWindowTitle("FNT Fiber Photometry")
        self.resize(1500, 900)
        self.files: List[Path] = []
        self.worker: Optional[RunWorker] = None
        self.loader: Optional[PreviewLoader] = None
        apply_dark_theme()
        self._build()
        if paths:
            self.add_paths(paths)
            if self.files:
                self.file_list.setCurrentRow(0)

    # ------------------------------------------------------------------ layout
    def _build(self):
        outer = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)

        files = QGroupBox("Recordings")
        fl = QVBoxLayout(files)
        row = QHBoxLayout()
        for text, slot in (("Add folder", self.add_folder), ("Add files", self.add_files),
                           ("Remove", self.remove_selected), ("Clear", self.clear_all)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        fl.addLayout(row)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setMinimumHeight(110)
        self.file_list.currentRowChanged.connect(lambda _r: self.update_preview())
        fl.addWidget(self.file_list)
        ll.addWidget(files)

        pipe = QGroupBox("dF/F pipeline")
        pl = QVBoxLayout(pipe)
        self.pipeline = QComboBox()
        for key in PIPELINES:
            self.pipeline.addItem(PIPELINE_LABELS[key], key)
        self.pipeline.setToolTip(
            "GuPPy (default): moving average, robust (Tukey biweight) fit of the control onto the "
            "signal, dF/F = (signal - fit) / fit.\n"
            "Warden lab MATLAB: FIR low-pass, linear fit, dF/F = (signal - fit) / mean(signal).\n"
            "PhAT: each channel divided by its biexponential bleaching fit, reference matched by "
            "medians and interquartile ranges.")
        self.pipeline.currentIndexChanged.connect(lambda k: self.stack.setCurrentIndex(k))
        pl.addWidget(self.pipeline)
        self.stack = QStackedWidget()
        pages = {}
        for key in PIPELINES:
            w = QWidget()
            pages[key] = QFormLayout(w)
            self.stack.addWidget(w)
        f = pages["guppy"]
        self.smooth_guppy = self._spin(0.0, 5.0, 0.1, " s", 2)
        f.addRow("Moving average:", self.smooth_guppy)
        f = pages["matlab"]
        self.chk_fir = QCheckBox("FIR low-pass (filteredfitcaldff.m; off = fitcaldff.m)")
        self.chk_fir.setChecked(True)
        f.addRow(self.chk_fir)
        self.fir_cutoff = self._spin(0.5, 100.0, 20.0, " Hz", 1)
        f.addRow("Cutoff:", self.fir_cutoff)
        self.fir_order = QSpinBox(); self.fir_order.setRange(10, 1000); self.fir_order.setValue(100)
        f.addRow("Filter order:", self.fir_order)
        f = pages["phat"]
        self.smooth_phat = self._spin(0.0, 5.0, 0.1, " s", 2)
        f.addRow("Moving average:", self.smooth_phat)
        self.phat_bleaching = QComboBox()
        self.phat_bleaching.addItem("Each channel's own (PhAT)", "each channel")
        self.phat_bleaching.addItem("The control's, for both (keeps slow signal change)", "control")
        self.phat_bleaching.setToolTip(
            "PhAT divides each channel by its own bleaching fit, which also removes any slow change "
            "the signal has and the control does not. 'The control's' divides both channels by the "
            "control's bleaching curve, so a slow change in the signal alone is kept.")
        f.addRow("Bleaching removed:", self.phat_bleaching)
        self.biexp_r2 = self._spin(0.0, 1.0, 0.5, "", 2)
        self.biexp_r2.setToolTip("Bleaching fits with R^2 below this are rejected (median used)")
        f.addRow("Min fit R²:", self.biexp_r2)
        self.reference_fit = QComboBox(); self.reference_fit.addItems(["robust", "least squares"])
        f.addRow("Reference fit:", self.reference_fit)
        pl.addWidget(self.stack)

        common = QFormLayout()
        self.chk_dropouts = QCheckBox("Remove wireless dropouts (full and partial)")
        self.chk_dropouts.setChecked(True)
        self.chk_dropouts.setToolTip("Samples below 0.5 x a channel's median, or 0.85 x its 10-s running "
                                     "median, are bridged before filtering, left out of the fit and "
                                     "shown as gaps.")
        common.addRow(self.chk_dropouts)
        self.notch = QLineEdit()
        self.notch.setPlaceholderText("e.g. 3.32 9.94 (blank = none)")
        self.notch.setToolTip("Narrow lines to remove from both channels (zero-phase notch filters, "
                              "width = f / 30)")
        common.addRow("Notch (Hz):", self.notch)
        self.zscore = QComboBox()
        for key, label in ZSCORE_METHODS:
            self.zscore.addItem(label, key)
        self.zscore.currentIndexChanged.connect(
            lambda _k: self.z_window.setVisible(self.zscore.currentData() == "baseline"))
        common.addRow("z-score:", self.zscore)
        zrow = QHBoxLayout()
        self.z_start = self._spin(0, 1e6, 0, " s", 1)
        self.z_end = self._spin(0, 1e6, 60, " s", 1)
        zrow.addWidget(self.z_start); zrow.addWidget(QLabel("to")); zrow.addWidget(self.z_end)
        self.z_window = QWidget(); self.z_window.setLayout(zrow); self.z_window.setVisible(False)
        common.addRow("Baseline:", self.z_window)
        pl.addLayout(common)
        self.btn_preview = QPushButton("Update preview with these settings")
        self.btn_preview.clicked.connect(self.update_preview)
        pl.addWidget(self.btn_preview)
        ll.addWidget(pipe)

        out = QGroupBox("Outputs")
        of = QFormLayout(out)
        self.chk_video = QCheckBox("Video with the traces underneath")
        self.chk_video.setChecked(True)
        of.addRow(self.chk_video)
        self.video_window = self._spin(2, 600, 20, " s", 0)
        of.addRow("Trace window:", self.video_window)
        orow = QHBoxLayout()
        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText("new run folder: <recordings>/FNT_FiberPhotometry/<date>_<pipeline>")
        self.out_dir.setToolTip("Blank: a new dated run folder beside the recordings. A folder: write "
                                "into it and skip recordings already done there (resume).")
        b = QPushButton("...")
        b.setFixedWidth(30)
        b.clicked.connect(self._pick_out_dir)
        orow.addWidget(self.out_dir); orow.addWidget(b)
        of.addRow("Output:", orow)
        ll.addWidget(out)

        row = QHBoxLayout()
        self.btn_run = QPushButton("Process recordings")
        self.btn_run.setStyleSheet(BLUE_BUTTON_STYLE)
        self.btn_run.setMinimumHeight(34)
        self.btn_run.clicked.connect(self.start)
        self.btn_stop = QPushButton("Stop"); self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(lambda: self.worker and self.worker.request_stop())
        row.addWidget(self.btn_run, 1); row.addWidget(self.btn_stop)
        ll.addLayout(row)
        self.progress = QProgressBar(); self.progress.setVisible(False)
        ll.addWidget(self.progress)
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMinimumHeight(120)
        ll.addWidget(self.log_box, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(400)
        split.addWidget(scroll)
        self.preview = PreviewPane()
        split.addWidget(self.preview)
        split.setSizes([440, 1060])
        split.setStretchFactor(1, 1)
        self._update_buttons()

    @staticmethod
    def _spin(lo, hi, val, suffix, decimals):
        s = QDoubleSpinBox()
        s.setRange(lo, hi); s.setDecimals(decimals); s.setValue(val); s.setSuffix(suffix)
        return s

    def _pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder")
        if d:
            self.out_dir.setText(d)

    # ------------------------------------------------------------------ files
    def log(self, text):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def add_paths(self, paths):
        try:
            recs = find_recordings(paths)
        except FileNotFoundError as e:
            self.log(str(e))
            return 0
        n = 0
        for p in recs:
            if p in self.files:
                continue
            self.files.append(p)
            vids = p.parent if p.suffix.lower() == ".dat" else p.parent / f"{p.stem}_Videos"
            has_video = vids.is_dir() and any(v.suffix.lower() in VIDEO_SUFFIXES for v in vids.iterdir())
            # LabJack files share names across sessions (data_0_<date>.dat): show date/animal folders
            name = f"{p.parent.parent.name}/{p.parent.name}/{p.name}" if p.suffix.lower() == ".dat" else p.name
            item = QListWidgetItem(f"{name}   [{'video' if has_video else 'no video'}]")
            item.setData(Qt.UserRole, str(p))
            self.file_list.addItem(item)
            n += 1
        if n:
            self.log(f"Added {n} recording(s)")
        self._update_buttons()
        return n

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder with recordings")
        if folder:
            self.add_paths([folder])

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Recordings", "",
                                                "Recordings (*.doric *.dat);;Doric (*.doric);;LabJack LJStreamUD (*.dat)")
        if paths:
            self.add_paths(paths)

    def remove_selected(self):
        for item in self.file_list.selectedItems():
            self.files.remove(Path(item.data(Qt.UserRole)))
            self.file_list.takeItem(self.file_list.row(item))
        self._update_buttons()

    def clear_all(self):
        self.files.clear()
        self.file_list.clear()
        self.preview.clear()
        self._update_buttons()

    def _update_buttons(self):
        busy = self.worker is not None and self.worker.isRunning()
        self.btn_run.setEnabled(bool(self.files) and not busy)
        self.btn_preview.setEnabled(bool(self.files))
        self.btn_stop.setEnabled(busy)

    # ------------------------------------------------------------------ settings
    def settings(self) -> Settings:
        key = self.pipeline.currentData()
        z = self.zscore.currentData()
        try:
            notch = tuple(float(x) for x in self.notch.text().replace(",", " ").split())
        except ValueError:
            notch = ()
            self.log("Notch frequencies not understood; none applied.")
        return Settings(
            pipeline=key, remove_dropouts=self.chk_dropouts.isChecked(), notch_hz=notch,
            apply_filter=self.chk_fir.isChecked(), fir_cutoff=self.fir_cutoff.value(),
            fir_order=self.fir_order.value(),
            smooth_s=self.smooth_phat.value() if key == "phat" else self.smooth_guppy.value(),
            biexp_r2=self.biexp_r2.value(), reference_fit=self.reference_fit.currentText(),
            phat_bleaching=self.phat_bleaching.currentData(), zscore_method=z,
            zscore_baseline=(self.z_start.value(), self.z_end.value()) if z == "baseline" else None,
            create_video=self.chk_video.isChecked(), video_window_s=self.video_window.value())

    # ------------------------------------------------------------------ preview
    def update_preview(self):
        k = self.file_list.currentRow()
        if k < 0 or k >= len(self.files):
            return
        if self.loader is not None and self.loader.isRunning():
            self.loader.wait()
        rec = self.files[k]
        s = self.settings()
        self.preview.clear(f"Loading {rec.name} ...")
        self.loader = PreviewLoader(rec, s)
        self.loader.loaded.connect(lambda data, r: self._show_preview(rec, s, data, r))
        self.loader.failed.connect(lambda msg: self.preview.clear(f"{rec.name}: {msg}"))
        self.loader.start()

    def _show_preview(self, rec, s, data, r):
        video = data.video_info.absolute_path if data.video_info else None
        label = (f"{PIPELINE_LABELS[s.pipeline]}  |  dropouts {100 * r.info['dropout_fraction']:.1f}%  "
                 f"|  dF/F robust SD {1.4826 * np.nanmedian(np.abs(r.dff - np.nanmedian(r.dff))):.2f}%")
        movement = link = None
        if data.imu_time is not None and data.imu_accel_x is not None:
            a = np.sqrt(data.imu_accel_x ** 2 + data.imu_accel_y ** 2 + data.imu_accel_z ** 2)
            movement = (data.imu_time, np.abs(a - np.median(a)))
        if data.state_time is not None and data.signal_strength is not None:
            # -127 dBm is the headstage's "no link" value
            link = (data.state_time, np.where(data.signal_strength <= -120, np.nan,
                                              data.signal_strength))
        frame_times = None
        if video and data.video_frame_clock:
            try:
                frame_times = recording_frame_times(data, video)[0]
            except OSError:
                pass
        self.preview.set_recording(rec.name, r, video, label, movement=movement, link=link,
                                   frame_times=frame_times)
        self.log(f"Preview {rec.name}: {label}")

    # ------------------------------------------------------------------ run
    def start(self):
        s = self.settings()
        out = Path(self.out_dir.text()) if self.out_dir.text().strip() else None
        if out is None and len({f.parent for f in self.files}) > 1:
            QMessageBox.warning(self, "Several folders",
                                "These recordings come from several folders. Choose an output "
                                "folder, or process one folder at a time.")
            return
        self.progress.setRange(0, len(self.files)); self.progress.setValue(0)
        self.progress.setVisible(True)
        self.worker = RunWorker(list(self.files), s, out)
        self.worker.progress.connect(self.log)
        self.worker.file_done.connect(self._file_done)
        self.worker.all_done.connect(self._all_done)
        self.log(f"Processing {len(self.files)} recording(s): {PIPELINE_LABELS[s.pipeline]}")
        self.worker.start()
        self._update_buttons()

    def _file_done(self, k, status):
        self.progress.setValue(k + 1)
        item = self.file_list.item(k)
        if item is not None:
            item.setText(f"{self.files[k].name}   [{status}]")

    def _all_done(self, text):
        self.log(text)
        self.progress.setVisible(False)
        self._update_buttons()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(10000)
        self.preview.clear()
        super().closeEvent(event)


def main(paths=None):
    if paths is None:
        paths = sys.argv[1:]
    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(app)
    w = FiberPhotometryWindow(paths=paths)
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
