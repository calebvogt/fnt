"""File > Export...: MAD's folder-wide tables (mad_export), in their own window.

The tables are one row per recording, per detection and per labelled call,
written into one folder from the ``.mad`` files beside the recordings -- what
an analysis in R or pandas reads, as against File > Export Annotations to CSV,
which writes one CSV beside each recording. ``mad export`` writes the same
tables with the same options from the command line.

An export of a full run reads tens of thousands of files on a network share,
so it runs on a worker thread and reports as it goes; the window stays usable
and Stop leaves the previous tables whole. Opening the window also starts a
read of which model each recording was last analysed with (root attributes
only), so the Model list can say how many recordings each model has; Export
does not wait for it.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QVBoxLayout,
)

from fnt.usv.usv_detector import mad_export as mx

#: Workers still running after their window closed. A QThread whose Python
#: wrapper is collected while it runs takes the process down, and the window
#: that started one may be gone (closed mid-export) before it finishes.
_RUNNING: set = set()


def _hold(worker: QThread) -> None:
    _RUNNING.add(worker)
    worker.finished.connect(lambda w=worker: _RUNNING.discard(w))
    # Reaped by Qt once finished, not left parented to MAD for the session.
    worker.finished.connect(worker.deleteLater)


class _ScanWorker(QThread):
    """Find the input's .mad files and the model each was last analysed with."""
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(object)        # {mad path: {mtime, model}}
    error_signal = pyqtSignal(str)

    def __init__(self, folders, wavs, out_dir: str = "", parent=None):
        super().__init__(parent)
        self.folders, self.wavs, self.out_dir = list(folders), list(wavs), out_dir
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            stores = mx.find_stores(self.folders, self.wavs, lambda: self._stop)
            known = mx.previous_scan(self.out_dir, stores) if self.out_dir else {}
            scan = mx.scan_models(
                stores, known,
                progress=lambda _s, i, n, _name: self.progress_signal.emit(i, n),
                should_stop=lambda: self._stop)
            if not self._stop:
                self.finished_signal.emit(scan)
        except mx.ExportStopped:
            pass
        except Exception as e:                      # unreachable share, ACLs
            self.error_signal.emit(str(e))


class _ExportWorker(QThread):
    """Run :func:`mad_export.export` off the UI thread."""
    progress_signal = pyqtSignal(str, int, int, str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, opts: mx.ExportOptions, known: Optional[Dict] = None,
                 parent=None):
        super().__init__(parent)
        self.opts, self.known = opts, known
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            rep = mx.export(self.opts,
                            progress=lambda *a: self.progress_signal.emit(*a),
                            log=self.log_signal.emit,
                            should_stop=lambda: self._stop, known=self.known)
            self.finished_signal.emit(rep)
        except Exception as e:
            import traceback
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")


class MADExportDialog(QDialog):
    """Choose the recordings, the output folder and the tables, then export.

    ``wavs`` are the recordings offered as the first input (the open project's,
    or the Audio list's with no project), under ``wavs_label``. ``settings``
    (a QSettings) keeps the choices between sessions. ``main`` is MAD's window:
    workers are parented to it so closing MAD stops them, and its folder picker
    and log are used when it has them.
    """

    STAGES = {"scan": "Reading models", "detections": "Detections",
              "labels": "Labelled calls"}

    def __init__(self, main=None, wavs: Sequence[str] = (), wavs_label: str = "",
                 settings=None, start_dir: str = "", parented: bool = True):
        # Un-parented from MAD by default (see MADInspectorDialog): an owned
        # window stays above its owner, and this one can run for an hour.
        super().__init__(main if parented else None)
        self._main = main
        self._wavs = list(wavs)
        self._settings = settings
        self._start_dir = start_dir or os.path.expanduser("~")
        self._scan: Dict = {}
        self._scan_inputs = None            # the inputs self._scan is complete for
        self._scan_worker: Optional[_ScanWorker] = None
        self._export_worker: Optional[_ExportWorker] = None
        self.setWindowTitle("Export Tables")
        self.setModal(False)
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint
                            | Qt.WindowCloseButtonHint)
        self.resize(680, 700)
        v = QVBoxLayout(self)

        blurb = QLabel(
            "Writes folder-wide tables from the <b>.mad</b> files beside the "
            "recordings: one row per recording, one per detection and one per "
            "labelled call, for analysis in R or pandas. Columns are generic "
            "(path, folder, file name, the start time an Avisoft name gives); "
            "join trial, treatment and the like in your analysis.<br>"
            "<i>Exporting again into the same folder reads only the recordings "
            "whose .mad changed since.</i>")
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color: #bbbbbb; font-size: 10px;")
        v.addWidget(blurb)

        # --- recordings ---------------------------------------------------
        box_in = QGroupBox("Recordings")
        gi = QVBoxLayout(box_in)
        self.rb_list = QRadioButton(
            f"{wavs_label or 'Project recordings'} ({len(self._wavs):,})")
        self.rb_list.setEnabled(bool(self._wavs))
        self.rb_list.setToolTip(
            "Every recording registered in the open project (or, with no "
            "project, in the Audio list) that has a .mad file.")
        gi.addWidget(self.rb_list)
        self.rb_folders = QRadioButton("Folders, each walked to the bottom:")
        gi.addWidget(self.rb_folders)
        self.list_folders = QListWidget()
        self.list_folders.setMaximumHeight(90)
        gi.addWidget(self.list_folders)
        row = QHBoxLayout()
        self.btn_add = QPushButton("Add Folder(s)…")
        self.btn_add.clicked.connect(self._add_folders)
        row.addWidget(self.btn_add)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self._remove_folders)
        row.addWidget(self.btn_remove)
        row.addStretch(1)
        gi.addLayout(row)
        v.addWidget(box_in)

        # --- output -------------------------------------------------------
        box_out = QGroupBox("Output folder")
        go = QHBoxLayout(box_out)
        self.edit_out = QLineEdit()
        self.edit_out.setPlaceholderText("Folder for the tables")
        go.addWidget(self.edit_out, 1)
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_out)
        go.addWidget(b)
        v.addWidget(box_out)

        # --- what ---------------------------------------------------------
        box_what = QGroupBox("Export")
        gw = QVBoxLayout(box_what)
        self.chk_det = QCheckBox(
            f"Recordings and detections ({mx.RECORDINGS_CSV}, {mx.DETECTIONS_CSV})")
        self.chk_det.setToolTip(
            "One row per recording whose last inference was the model below, "
            "and one per detection of those recordings: score, review status, "
            "harmonic number, the contour measures of its mask and its "
            "brightness ridge. No audio is read.")
        gw.addWidget(self.chk_det)
        mrow = QHBoxLayout()
        mrow.addSpacing(22)
        mrow.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        self.combo_model.addItem("Most common model", "")
        self.combo_model.setToolTip(
            "Only recordings whose last inference was this model. The count "
            "beside each is how many of the recordings it analysed last.")
        mrow.addWidget(self.combo_model, 1)
        gw.addLayout(mrow)
        self.chk_labels = QCheckBox(f"Labelled calls ({mx.LABELS_CSV})")
        self.chk_labels.setToolTip(
            "One row per accepted call, from any model or painted by hand, "
            "with MAD's call metrics recomputed from the audio (as Export "
            "Annotations to CSV does), the contour measures and the ridge. "
            "Reads each labelled recording's audio, so it is the slow table.")
        gw.addWidget(self.chk_labels)
        self.chk_images = QCheckBox(
            f"Spectrogram images of labelled calls ({mx.IMAGES_DIR}/)")
        self.chk_images.setToolTip(
            "A PNG per labelled call: the call and 10 ms either side, every "
            "frequency bin (high at the top); grey = level as MAD stores it, "
            "alpha = the call's own mask.")
        gw.addWidget(self.chk_images)
        self.chk_harm = QCheckBox("Include harmonics")
        self.chk_harm.setToolTip(
            "Off: fundamentals only (harmonic_n blank or 1). On: every "
            "detection and labelled call, harmonics included.")
        gw.addWidget(self.chk_harm)
        self.chk_incr = QCheckBox("Skip recordings unchanged since the last export")
        self.chk_incr.setToolTip(
            "Copy the rows of every recording whose .mad has not changed from "
            "the tables already in the output folder, instead of reading it "
            "again. Turn off after updating FNT to recompute everything.")
        gw.addWidget(self.chk_incr)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Leave out .mad files written in the last"))
        self.spin_settle = QDoubleSpinBox()
        self.spin_settle.setRange(0.0, 1440.0)
        self.spin_settle.setDecimals(0)
        self.spin_settle.setSuffix(" min")
        self.spin_settle.setToolTip(
            "MAD may still be writing them (an inference run in progress); "
            "the next export includes them. 0 leaves nothing out.")
        srow.addWidget(self.spin_settle)
        srow.addStretch(1)
        gw.addLayout(srow)
        v.addWidget(box_what)

        # --- progress -----------------------------------------------------
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        v.addWidget(self.progress)
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #999999; font-size: 9px;")
        self.lbl_status.setWordWrap(True)
        v.addWidget(self.lbl_status)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("font-size: 10px;")
        v.addWidget(self.log_view, 1)

        row = QHBoxLayout()
        self.btn_export = QPushButton("Export")
        self.btn_export.setDefault(True)
        self.btn_export.clicked.connect(self._run)
        row.addWidget(self.btn_export)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip(
            "Stop after the current recording. The table being written is "
            "discarded; the previous one stays as it was.")
        self.btn_stop.clicked.connect(self._stop)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.close)
        row.addWidget(bb)
        v.addLayout(row)

        self._restore()
        self.chk_det.toggled.connect(self._sync)
        self.chk_labels.toggled.connect(self._on_labels_toggled)
        self.chk_images.toggled.connect(self._on_images_toggled)
        # Two exclusive buttons: one of them toggling is every change.
        self.rb_folders.toggled.connect(self._inputs_changed)
        self._sync()
        self._inputs_changed()

    # -- settings ---------------------------------------------------------
    def _value(self, key, default, typ):
        if self._settings is None:
            return default
        try:
            return self._settings.value(f"mad/export/{key}", default, type=typ)
        except Exception:
            return default

    def _restore(self):
        self.edit_out.setText(self._value("out_dir", "", str))
        for f in self._value("folders", [], list) or []:
            if f:
                self.list_folders.addItem(str(f))
        use_folders = self._value("use_folders", not self._wavs, bool)
        (self.rb_folders if use_folders or not self._wavs
         else self.rb_list).setChecked(True)
        self.chk_det.setChecked(self._value("detections", True, bool))
        self.chk_labels.setChecked(self._value("labels", False, bool))
        self.chk_images.setChecked(self._value("images", False, bool))
        self.chk_harm.setChecked(self._value("harmonics", False, bool))
        self.chk_incr.setChecked(self._value("incremental", True, bool))
        self.spin_settle.setValue(self._value("settle_min", mx.SETTLE_S / 60, float))

    def _save(self):
        if self._settings is None:
            return
        s = self._settings
        s.setValue("mad/export/out_dir", self.edit_out.text().strip())
        s.setValue("mad/export/folders", self._folders())
        s.setValue("mad/export/use_folders", self.rb_folders.isChecked())
        for key, chk in (("detections", self.chk_det), ("labels", self.chk_labels),
                         ("images", self.chk_images), ("harmonics", self.chk_harm),
                         ("incremental", self.chk_incr)):
            s.setValue(f"mad/export/{key}", chk.isChecked())
        s.setValue("mad/export/settle_min", float(self.spin_settle.value()))

    # -- inputs -----------------------------------------------------------
    def _folders(self) -> List[str]:
        return [self.list_folders.item(i).text()
                for i in range(self.list_folders.count())]

    def _inputs(self):
        """(folders, wavs) of the chosen input."""
        if self.rb_list.isChecked():
            return [], list(self._wavs)
        return self._folders(), []

    def _add_folders(self):
        picker = getattr(self._main, "_pick_folders", None)
        if callable(picker):
            folders = picker()             # MAD's multi-select folder dialog
        else:
            f = QFileDialog.getExistingDirectory(
                self, "Folder of recordings to export", self._start_dir)
            folders = [f] if f else []
        have = {os.path.normcase(f) for f in self._folders()}
        for f in folders:
            if os.path.normcase(f) not in have:
                self.list_folders.addItem(f)
        if folders:
            if self.rb_folders.isChecked():
                self._inputs_changed()
            else:
                self.rb_folders.setChecked(True)     # reads them via toggled

    def _remove_folders(self):
        for item in self.list_folders.selectedItems():
            self.list_folders.takeItem(self.list_folders.row(item))
        self._inputs_changed()

    def _browse_out(self):
        start = self.edit_out.text().strip() or self._start_dir
        d = QFileDialog.getExistingDirectory(self, "Folder for the tables", start)
        if d:
            self.edit_out.setText(d)
            self._inputs_changed()

    def _sync(self, *_):
        """The model only filters detections."""
        self.combo_model.setEnabled(self.chk_det.isChecked())

    def _on_images_toggled(self, on: bool):
        if on:                              # the images are of labelled calls
            self.chk_labels.setChecked(True)

    def _on_labels_toggled(self, on: bool):
        if not on:
            self.chk_images.setChecked(False)

    # -- model list -------------------------------------------------------
    def _inputs_changed(self, *_):
        """Read the models of the chosen input's .mad files, in the background,
        replacing any read already under way."""
        if self._export_worker is not None:
            return
        self._stop_scan()
        folders, wavs = self._inputs()
        key = (tuple(folders), tuple(wavs))
        if not folders and not wavs:
            self._fill_models({})
            return
        if key == self._scan_inputs:
            return
        self._scan, self._scan_inputs = {}, None
        self._fill_models(None)
        w = _ScanWorker(folders, wavs, self.edit_out.text().strip(), self._main)
        w.progress_signal.connect(self._on_scan_progress)
        w.finished_signal.connect(lambda scan, k=key: self._on_scanned(k, scan))
        w.error_signal.connect(self._on_scan_error)
        _hold(w)
        self._scan_worker = w
        w.start()

    def _stop_scan(self):
        """Abandon the model read under way, if any. request_stop only sets a
        Python flag, so it is safe on a worker Qt has already reaped."""
        w, self._scan_worker = self._scan_worker, None
        if w is None:
            return
        w.request_stop()
        for sig in ("finished_signal", "progress_signal", "error_signal"):
            try:
                getattr(w, sig).disconnect()
            except (TypeError, RuntimeError):
                pass

    def _on_scan_error(self, msg):
        self._scan_worker = None
        self._fill_models({})
        self.lbl_status.setText(f"Could not read the recordings: {msg}")

    def _on_scan_progress(self, i, n):
        if self._export_worker is None:
            self.lbl_status.setText(f"Reading which model each .mad was last "
                                    f"analysed with: {i + 1:,} of {n:,}")

    def _on_scanned(self, key, scan):
        self._scan_worker = None
        self._scan, self._scan_inputs = scan, key
        self._fill_models(scan)
        if self._export_worker is None:
            counts = mx.model_counts(scan)
            self.lbl_status.setText(
                f"{len(scan):,} .mad file(s); {sum(counts.values()):,} analysed "
                f"by {len(counts)} model(s)." if scan else
                "No .mad file found in the chosen recordings.")

    def _fill_models(self, scan):
        """The Model list: most common first, each with its recording count;
        ``None`` while the read is under way."""
        keep = self.combo_model.currentData() or ""
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        if scan is None:
            self.combo_model.addItem("Most common model (reading models…)", "")
        else:
            counts = mx.model_counts(scan)
            top = counts.most_common(1)
            self.combo_model.addItem(
                f"Most common: {top[0][0]} ({top[0][1]:,})" if top
                else "Most common model", "")
            for m, n in counts.most_common():
                self.combo_model.addItem(f"{m}  ({n:,} recording(s))", m)
        i = self.combo_model.findData(keep)
        self.combo_model.setCurrentIndex(max(0, i))
        self.combo_model.blockSignals(False)

    # -- run --------------------------------------------------------------
    def options(self) -> Optional[mx.ExportOptions]:
        """The export the dialog describes, or None (and a message) when it
        is not complete."""
        folders, wavs = self._inputs()
        out = self.edit_out.text().strip()
        if not folders and not wavs:
            self.lbl_status.setText("Choose the recordings: the project's, or "
                                    "folders.")
            return None
        if not out:
            self.lbl_status.setText("Choose a folder for the tables.")
            return None
        if not (self.chk_det.isChecked() or self.chk_labels.isChecked()
                or self.chk_images.isChecked()):
            self.lbl_status.setText("Tick at least one table to export.")
            return None
        return mx.ExportOptions(
            out_dir=out, folders=folders, wavs=wavs,
            detections=self.chk_det.isChecked(),
            labels=self.chk_labels.isChecked() or self.chk_images.isChecked(),
            images=self.chk_images.isChecked(),
            model=self.combo_model.currentData() or None,
            harmonics=self.chk_harm.isChecked(),
            incremental=self.chk_incr.isChecked(),
            settle_s=float(self.spin_settle.value()) * 60.0)

    def _run(self):
        opts = self.options()
        if opts is None:
            return
        self._save()
        folders, wavs = self._inputs()
        # A finished read of these inputs saves the export reading them again.
        known = (dict(self._scan)
                 if self._scan_inputs == (tuple(folders), tuple(wavs)) else None)
        self._stop_scan()
        self._set_running(True)
        self.log_view.clear()
        self.progress.setRange(0, 0)
        w = _ExportWorker(opts, known, self._main)
        w.progress_signal.connect(self._on_progress)
        w.log_signal.connect(self._on_log)
        w.finished_signal.connect(self._on_done)
        w.error_signal.connect(self._on_error)
        _hold(w)
        self._export_worker = w
        w.start()

    def _stop(self):
        w = self._export_worker
        if w is not None:
            w.request_stop()
            self.btn_stop.setEnabled(False)
            self.lbl_status.setText("Stopping after the current recording…")

    def _set_running(self, on: bool):
        for wdg in (self.btn_export, self.rb_list, self.rb_folders, self.btn_add,
                    self.btn_remove, self.edit_out, self.chk_det, self.chk_labels,
                    self.chk_images, self.chk_harm, self.chk_incr,
                    self.spin_settle, self.combo_model, self.list_folders):
            wdg.setEnabled(not on)
        if not on:
            self.rb_list.setEnabled(bool(self._wavs))
            self._sync()
        self.btn_stop.setEnabled(on)
        self.progress.setVisible(on)

    def _on_progress(self, stage, i, n, name):
        if self.progress.maximum() != n:
            self.progress.setRange(0, max(1, n))
        self.progress.setValue(min(i, n))
        self.lbl_status.setText(
            f"{self.STAGES.get(stage, stage)}: {min(i + 1, n):,} of {n:,}"
            + (f" — {name}" if name else ""))

    def _on_log(self, msg):
        self.log_view.appendPlainText(msg)

    def _on_done(self, rep):
        self._export_worker = None
        self._set_running(False)
        tables = rep.get("tables") or []
        parts = []
        if mx.DETECTIONS_CSV in tables:
            parts.append(f"{rep['recordings']:,} recording(s) with "
                         f"{rep['detection_rows']:,} detection(s)")
        if mx.LABELS_CSV in tables:
            parts.append(f"{rep['label_rows']:,} labelled call(s)")
        text = ("Exported " + " and ".join(parts) + f" to {rep['out_dir']}."
                if parts else "Nothing was exported.")
        if rep.get("stopped"):
            text = ("Stopped. " + (text + " " if parts else "")
                    + "The table being written when stopped is unchanged.")
        if rep.get("error"):
            text += f" ({rep['error']})"
        self.lbl_status.setText(text)
        main = self._main
        if main is not None:
            try:
                main.status_bar.showMessage(text)
                main._log(text)
            except Exception:
                pass

    def _on_error(self, msg):
        self._export_worker = None
        self._set_running(False)
        first = msg.split("\n", 1)[0]
        self.lbl_status.setText(f"Export failed: {first}")
        self.log_view.appendPlainText(msg)

    def closeEvent(self, event):
        """Closing stops a running export (the previous tables stay) and the
        model read, and keeps the choices for next time."""
        self._save()
        self._stop_scan()
        w = self._export_worker
        if w is not None:
            w.request_stop()
            main = self._main
            if main is not None:
                try:
                    main.status_bar.showMessage(
                        "Export stopped; the previous tables are unchanged.")
                except Exception:
                    pass
        super().closeEvent(event)
