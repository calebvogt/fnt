"""Twin experiment: fork one animal's life and watch the two versions part.

The dialog asks the three things a counterfactual needs — which animal, what
to change, and when — runs both lives in the background, and shows them on one
figure: the life as it was in black, the life with the change in red. See
:mod:`fnt.abma.core.twin` for what makes the comparison exact.
"""
from __future__ import annotations

import os
import traceback

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QPushButton, QProgressBar, QFileDialog,
    QMessageBox, QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from ..core.twin import Tweak, run_twin
from ..core.twin_plot import figure, save_png

#: (label, needs a brain, builder(amount, seconds, unit) -> Tweak fields,
#:  which of amount / unit / seconds the row uses, hint)
CHANGES = [
    ("A little less hungry", False,
     lambda a, s, u: dict(kind="state", name="hunger", op="add", value=-a),
     "amount", "Hunger drops by this many points (0–100 scale)."),
    ("Hungrier", False,
     lambda a, s, u: dict(kind="state", name="hunger", op="add", value=a),
     "amount", "Hunger rises by this many points."),
    ("Thirstier", False,
     lambda a, s, u: dict(kind="state", name="thirst", op="add", value=a),
     "amount", "Thirst rises by this many points."),
    ("More stressed", False,
     lambda a, s, u: dict(kind="state", name="stress", op="add", value=a),
     "amount", "Stress rises by this many points."),
    ("Loses its sense of smell, for good", False,
     lambda a, s, u: dict(kind="trait", name="smell_ability", op="set",
                          value=0.0),
     "", "Olfactory acuity set to zero from this moment on."),
    ("Cannot smell, for a while", True,
     lambda a, s, u: dict(kind="sense", name="olfaction", op="scale",
                          value=0.0, duration_s=s),
     "seconds", "Every olfactory input to its brain is silenced, then "
                "restored."),
    ("One brain unit driven on", True,
     lambda a, s, u: dict(kind="unit", name=f"rnn:{u}", op="set", value=1.0,
                          duration_s=s),
     "unit seconds", "Clamp a recurrent unit fully on, then release it. Find "
                     "unit numbers by hovering the Brain activity raster."),
    ("One brain unit silenced", True,
     lambda a, s, u: dict(kind="unit", name=f"rnn:{u}", op="scale", value=0.0,
                          duration_s=s),
     "unit seconds", "Hold a recurrent unit at zero, then release it."),
]


class _Worker(QThread):
    progress = pyqtSignal(float)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, cfg, tweak, fork_s, horizon_s):
        super().__init__()
        self.args = (cfg, tweak, fork_s, horizon_s)

    def run(self):
        cfg, tweak, fork_s, horizon_s = self.args
        try:
            self.done.emit(run_twin(
                cfg, [tweak], fork_at_s=fork_s, horizon_s=horizon_s,
                progress_cb=lambda f: self.progress.emit(float(f))))
        except Exception:
            self.failed.emit(traceback.format_exc())


class TwinDialog(QDialog):
    """Ask a what-if of one animal."""

    def __init__(self, cfg, animals: list, selected: int | None = None,
                 parent=None):
        """``animals`` is ``[(row, "F9001 · saline_F"), ...]``."""
        super().__init__(parent)
        self.cfg = cfg
        self.result = None
        self.worker = None
        self.setWindowTitle("Twin experiment — one life, lived twice")
        self.resize(1180, 760)
        lay = QVBoxLayout(self)

        intro = QLabel(
            "Run the experiment up to a moment, copy the whole world, change "
            "one thing about one animal in the copy, and follow both. Every "
            "random event is identical in the two until the change makes it "
            "otherwise, so any difference you see was caused by it.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#aab; font-size:11px;")
        lay.addWidget(intro)

        top = QHBoxLayout()
        form = QFormLayout()
        self.in_animal = QComboBox()
        for row, label in animals:
            self.in_animal.addItem(label, row)
        if selected is not None:
            k = self.in_animal.findData(int(selected))
            if k >= 0:
                self.in_animal.setCurrentIndex(k)
        has_brain = bool(getattr(cfg.brain, "enabled", False))
        self.in_change = QComboBox()
        for label, needs_brain, _, _, hint in CHANGES:
            self.in_change.addItem(label)
            if needs_brain and not has_brain:
                item = self.in_change.model().item(self.in_change.count() - 1)
                item.setEnabled(False)
                item.setToolTip("Needs a learned brain (Experiment ▸ "
                                "Nervous system).")
        self.in_change.currentIndexChanged.connect(self._on_change)
        self.in_amount = QDoubleSpinBox()
        self.in_amount.setRange(1, 100)
        self.in_amount.setValue(30)
        self.in_amount.setSuffix(" points")
        self.in_unit = QSpinBox()
        self.in_unit.setRange(0, 511)
        self.in_seconds = QDoubleSpinBox()
        self.in_seconds.setRange(2, 86400)
        self.in_seconds.setValue(10)
        self.in_seconds.setSuffix(" s")
        self.in_fork = QDoubleSpinBox()
        self.in_fork.setRange(0.0, max(0.1, cfg.days * 24.0))
        self.in_fork.setValue(min(6.0, cfg.days * 12.0))
        self.in_fork.setSuffix(" h into the run")
        self.in_fork.setSingleStep(0.5)
        self.in_horizon = QDoubleSpinBox()
        self.in_horizon.setRange(0.1, 240.0)
        self.in_horizon.setValue(6.0)
        self.in_horizon.setSuffix(" h")
        self.in_horizon.setSingleStep(0.5)
        form.addRow("Which animal", self.in_animal)
        form.addRow("What changes", self.in_change)
        form.addRow("By how much", self.in_amount)
        form.addRow("Which unit", self.in_unit)
        form.addRow("For how long", self.in_seconds)
        form.addRow("When", self.in_fork)
        form.addRow("Follow both for", self.in_horizon)
        self._form = form
        left = QWidget()
        left.setLayout(form)
        left.setMaximumWidth(380)
        top.addWidget(left)

        side = QVBoxLayout()
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#8a9099; font-size:11px;")
        side.addWidget(self.hint)
        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        self.verdict.setStyleSheet("font-size:13px;")
        side.addWidget(self.verdict, 1)
        row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Live it twice")
        self.btn_run.setObjectName("accept_btn")
        self.btn_run.clicked.connect(self._run)
        self.btn_save = QPushButton("Save figure + data…")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_save)
        row.addStretch()
        side.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setMaximumHeight(6)
        self.bar.setTextVisible(False)
        side.addWidget(self.bar)
        top.addLayout(side, 1)
        lay.addLayout(top)

        self.canvas_host = QVBoxLayout()
        lay.addLayout(self.canvas_host, 1)
        self.canvas = None
        self._on_change(0)

    # ------------------------------------------------------------------ #
    def _on_change(self, k: int) -> None:
        uses = CHANGES[k][3]
        self.hint.setText(CHANGES[k][4])
        for w, key in ((self.in_amount, "amount"), (self.in_unit, "unit"),
                       (self.in_seconds, "seconds")):
            on = key in uses
            w.setVisible(on)
            lab = self._form.labelForField(w)
            if lab is not None:
                lab.setVisible(on)

    def tweak(self) -> Tweak:
        k = self.in_change.currentIndex()
        fields = CHANGES[k][2](self.in_amount.value(), self.in_seconds.value(),
                               self.in_unit.value())
        return Tweak(target=str(self.in_animal.currentData()),
                     label="", **fields)

    def _run(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.btn_run.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.verdict.setText("Living both lives…")
        self.bar.setValue(0)
        self.worker = _Worker(self.cfg, self.tweak(),
                              self.in_fork.value() * 3600.0,
                              self.in_horizon.value() * 3600.0)
        self.worker.progress.connect(lambda f: self.bar.setValue(int(f * 100)))
        self.worker.done.connect(self._show)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _failed(self, tb: str) -> None:
        self.btn_run.setEnabled(True)
        self.verdict.setText("")
        QMessageBox.critical(self, "Twin experiment failed",
                             tb.strip().splitlines()[-1])

    def _show(self, res) -> None:
        self.result = res
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.bar.setValue(100)
        self.verdict.setText(res.sentence())
        if self.canvas is not None:
            self.canvas_host.removeWidget(self.canvas)
            self.canvas.setParent(None)
        self.canvas = FigureCanvasQTAgg(figure(res, self.cfg))
        self.canvas_host.addWidget(self.canvas)

    def _save(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save twin experiment", os.path.join(
                os.path.expanduser("~"), "ABMA_Projects", "twin.png"),
            "Figure (*.png)")
        if not path:
            return
        save_png(self.result, path, self.cfg)
        self.result.save(os.path.splitext(path)[0] + ".npz")

    def closeEvent(self, ev):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(100)
        super().closeEvent(ev)
