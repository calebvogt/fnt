"""
Live matplotlib window for YOLO training feedback.

Replaces stdout scroll during training with a compact 3-panel plot:
  1. Train/val box-loss vs epoch
  2. mAP50 + mAP50-95 vs epoch
  3. Precision + Recall vs epoch

The plot window lives in the Qt GUI thread. The training worker thread pushes
epoch metrics via ``push_from_worker`` (thread-safe; uses a Qt signal).

Usage (from the main window):

    self._plot = TrainingPlotWindow(self)
    self._plot.show()
    # In YOLOTrainingWorker, forward each epoch via a signal that calls:
    self._plot.push(epoch, metrics)
"""

from typing import Dict

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QDialog, QVBoxLayout

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    _HAS_MPL = True
except Exception:  # pragma: no cover - matplotlib is a required dep but be safe
    _HAS_MPL = False


# Metric keys we care about (ultralytics naming).
_LOSS_KEYS_TRAIN = ('train/box_loss', 'train/cls_loss', 'train/dfl_loss')
_LOSS_KEYS_VAL = ('val/box_loss',)
_MAP_KEYS = ('metrics/mAP50(B)', 'metrics/mAP50-95(B)')
_PR_KEYS = ('metrics/precision(B)', 'metrics/recall(B)')


def _first_present(metrics: Dict, keys) -> float:
    """Return the first metric value whose key is present, else NaN."""
    for k in keys:
        if k in metrics:
            try:
                return float(metrics[k])
            except (TypeError, ValueError):
                continue
        # Tolerate the (B) / non-(B) variants ultralytics sometimes emits.
        alt = k.replace('(B)', '')
        if alt in metrics:
            try:
                return float(metrics[alt])
            except (TypeError, ValueError):
                continue
    return float('nan')


class TrainingPlotWindow(QDialog):
    """Modeless dialog showing live training curves."""

    # Worker threads call ``push_from_worker`` which emits this signal. The
    # slot runs on the GUI thread so matplotlib stays happy.
    _metric_signal = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DAD Training")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(780, 520)

        self._epochs = []
        self._train_box_loss = []
        self._val_box_loss = []
        self._map50 = []
        self._map5095 = []
        self._precision = []
        self._recall = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        if _HAS_MPL:
            self._figure = Figure(figsize=(7.5, 5.0), tight_layout=True)
            self._canvas = FigureCanvas(self._figure)
            layout.addWidget(self._canvas)

            gs = self._figure.add_gridspec(3, 1, hspace=0.45)
            self._ax_loss = self._figure.add_subplot(gs[0, 0])
            self._ax_map = self._figure.add_subplot(gs[1, 0], sharex=self._ax_loss)
            self._ax_pr = self._figure.add_subplot(gs[2, 0], sharex=self._ax_loss)

            for ax in (self._ax_loss, self._ax_map, self._ax_pr):
                ax.grid(alpha=0.3)
            self._ax_loss.set_ylabel('Box loss')
            self._ax_map.set_ylabel('mAP')
            self._ax_pr.set_ylabel('P / R')
            self._ax_pr.set_xlabel('Epoch')

            (self._line_train,) = self._ax_loss.plot([], [], label='train', color='#1f77b4')
            (self._line_val,) = self._ax_loss.plot([], [], label='val', color='#ff7f0e')
            self._ax_loss.legend(loc='upper right', fontsize=8)

            (self._line_map50,) = self._ax_map.plot([], [], label='mAP50', color='#2ca02c')
            (self._line_map5095,) = self._ax_map.plot([], [], label='mAP50-95', color='#9467bd')
            self._ax_map.legend(loc='lower right', fontsize=8)
            self._ax_map.set_ylim(0, 1)

            (self._line_p,) = self._ax_pr.plot([], [], label='precision', color='#17becf')
            (self._line_r,) = self._ax_pr.plot([], [], label='recall', color='#d62728')
            self._ax_pr.legend(loc='lower right', fontsize=8)
            self._ax_pr.set_ylim(0, 1)
        else:
            from PyQt5.QtWidgets import QLabel
            layout.addWidget(QLabel("matplotlib not installed; install it to see live plots."))

        self._metric_signal.connect(self._on_metrics)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear accumulated history — call before a new training run."""
        self._epochs.clear()
        self._train_box_loss.clear()
        self._val_box_loss.clear()
        self._map50.clear()
        self._map5095.clear()
        self._precision.clear()
        self._recall.clear()
        if _HAS_MPL:
            for line in (
                self._line_train, self._line_val,
                self._line_map50, self._line_map5095,
                self._line_p, self._line_r,
            ):
                line.set_data([], [])
            self._canvas.draw_idle()

    def push_from_worker(self, epoch: int, metrics: Dict):
        """Thread-safe entry point for the training worker."""
        self._metric_signal.emit(int(epoch), dict(metrics or {}))

    # Alias kept for clarity in call sites that already live on the GUI thread.
    def push(self, epoch: int, metrics: Dict):
        self.push_from_worker(epoch, metrics)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_metrics(self, epoch: int, metrics: Dict):
        if not _HAS_MPL:
            return

        self._epochs.append(epoch)
        self._train_box_loss.append(_first_present(metrics, _LOSS_KEYS_TRAIN[:1]))
        self._val_box_loss.append(_first_present(metrics, _LOSS_KEYS_VAL))
        self._map50.append(_first_present(metrics, _MAP_KEYS[:1]))
        self._map5095.append(_first_present(metrics, _MAP_KEYS[1:]))
        self._precision.append(_first_present(metrics, _PR_KEYS[:1]))
        self._recall.append(_first_present(metrics, _PR_KEYS[1:]))

        self._line_train.set_data(self._epochs, self._train_box_loss)
        self._line_val.set_data(self._epochs, self._val_box_loss)
        self._line_map50.set_data(self._epochs, self._map50)
        self._line_map5095.set_data(self._epochs, self._map5095)
        self._line_p.set_data(self._epochs, self._precision)
        self._line_r.set_data(self._epochs, self._recall)

        for ax in (self._ax_loss, self._ax_map, self._ax_pr):
            ax.relim()
        # Autoscale loss axis (mAP/PR are clamped 0..1).
        self._ax_loss.autoscale_view()
        self._ax_map.set_xlim(0, max(1, max(self._epochs)))
        self._ax_pr.set_xlim(0, max(1, max(self._epochs)))

        self._canvas.draw_idle()
