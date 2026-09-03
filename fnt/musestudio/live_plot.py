"""Live scrolling signal view — the primary window onto the headband.

One stacked lane per visible channel, newest data on the right, streams ordered
EEG → motion → optics.

Design decisions that matter scientifically:

* **EEG lanes use a real, fixed µV scale.** The gain selector says how many µV
  span half a lane, so trace height means the same thing from second to second.
  (An auto-scaled trace silently rescales when you blink, which makes a 20 µV
  alpha rhythm and a 200 µV artifact look identical.)
* **EEG is display-filtered** (1–40 Hz + mains notch) with a causal streaming
  filter. Without it, DC drift and line noise dominate and you cannot see the
  rhythm you are trying to train. Recording is unaffected — raw data is written
  straight from the reader thread.
* **Non-EEG streams are not EEG-filtered.** fNIRS/optics hemodynamics live below
  1 Hz, so an EEG high-pass would erase exactly the signal of interest; those
  lanes are mean-removed and robustly auto-ranged instead.
"""

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from fnt.musestudio import theme
from fnt.musestudio.dsp import StreamingFilter

_MAX_RATE_HZ = 300          # ring-buffer sizing bound (> EEG 256 Hz)
_LANE = 1.0                 # vertical spacing between channel lanes
_LANE_PX = 26               # min pixels per lane so every label has room
GAIN_CHOICES = [10, 25, 50, 100, 200, 500]     # µV per half-lane
WINDOW_CHOICES = [2, 5, 10, 30]                # seconds


def _is_eeg(name):
    return "EEG" in str(name).upper()


def short_channel(name):
    """Lane label: drop the redundant stream prefix.

    ``OPTICS_LO_RED`` -> ``LO_RED``, ``EEG_TP9`` -> ``TP9``. Full names are
    long enough that pyqtgraph culls them for not fitting the axis.
    """
    s = str(name)
    for prefix in ("OPTICS_", "EEG_", "MUSE_"):
        if s.upper().startswith(prefix):
            return s[len(prefix):]
    return s


def stream_priority(name):
    """Display order: EEG first (the signal of interest), then motion, then optics."""
    upper = str(name).upper()
    if "EEG" in upper:
        return 0
    if "ACC" in upper or "GYRO" in upper:
        return 1
    if "OPTIC" in upper or "NIRS" in upper:
        return 2
    return 3


class _StreamPanel:
    """Ring buffers, curves and scaling for one LSL stream."""

    def __init__(self, glw, name, channel_names, fs):
        self.name = name
        self.channel_names = list(channel_names)
        self.is_eeg = _is_eeg(name)
        self.fs = float(fs)
        self.visible = list(self.channel_names)
        n = len(self.channel_names)
        maxlen = int(WINDOW_CHOICES[-1] * _MAX_RATE_HZ)

        self.t = deque(maxlen=maxlen)
        self.buffers = {ch: deque(maxlen=maxlen) for ch in self.channel_names}
        # EEG gets a causal display filter; optics must keep its sub-Hz content.
        self.filter = StreamingFilter(fs, n) if self.is_eeg else None

        self.plot = pg.PlotItem()
        units = "µV (fixed scale)" if self.is_eeg else "auto-scaled"
        theme.style_plot(self.plot, title=f"{name}   ·   {units}")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.showGrid(x=True, y=False, alpha=0.10)
        self.plot.getAxis("left").setWidth(72)

        self.curves = {}
        for i, ch in enumerate(self.channel_names):
            pen = pg.mkPen(theme.electrode_color(ch, i), width=1.2)
            self.curves[ch] = self.plot.plot(pen=pen)
        self._baselines = []
        self._apply_layout()

    # --- configuration ----------------------------------------------------
    def set_filtering(self, enabled):
        if self.is_eeg:
            self.filter = (StreamingFilter(self.fs, len(self.channel_names))
                           if enabled else None)

    def set_visible(self, names):
        """Show only ``names``; lanes compact so hidden channels leave no gap."""
        keep = [c for c in self.channel_names if c in set(names)]
        self.visible = keep or list(self.channel_names)
        self._apply_layout()

    def _apply_layout(self):
        """Re-stack lanes and rebuild axis labels for the visible channels."""
        for item in self._baselines:
            self.plot.removeItem(item)
        self._baselines = []

        n = len(self.visible)
        for ch, curve in self.curves.items():
            curve.setVisible(ch in self.visible)
        ticks = [((n - 1 - i) * _LANE, short_channel(self.visible[i]))
                 for i in range(n)]
        self.plot.getAxis("left").setTicks([ticks])
        self.plot.setYRange(-_LANE * 0.6, (n - 1) * _LANE + _LANE * 0.6, padding=0)
        for i in range(n):
            line = pg.InfiniteLine(pos=(n - 1 - i) * _LANE, angle=0,
                                   pen=pg.mkPen(theme.GRID, width=1))
            line.setZValue(-10)
            self.plot.addItem(line)
            self._baselines.append(line)
        # Guarantee vertical room for every label, or pyqtgraph culls them.
        self.plot.setMinimumHeight(max(90, n * _LANE_PX + 34))

    # --- data -------------------------------------------------------------
    def add(self, timestamps, data):
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        if self.filter is not None:
            data = self.filter.process(data)
        self.t.extend(np.asarray(timestamps, dtype=float).tolist())
        for i, ch in enumerate(self.channel_names):
            if i < data.shape[1]:
                self.buffers[ch].extend(data[:, i].tolist())

    def refresh(self, gain_uv, window_s):
        if len(self.t) < 2:
            return
        t = np.fromiter(self.t, dtype=float, count=len(self.t))
        t = t - t[-1]                       # newest at 0
        keep = t >= -float(window_s)
        t = t[keep]
        n = len(self.visible)
        for i, ch in enumerate(self.visible):
            buf = self.buffers[ch]
            y = np.fromiter(buf, dtype=float, count=len(buf))
            y = y[-len(keep):][keep] if len(y) >= len(keep) else y
            m = min(len(y), len(t))
            if m < 2:
                continue
            yy, tt = y[-m:], t[-m:]
            lane = (n - 1 - i) * _LANE
            if self.is_eeg:
                # Fixed scale: `gain_uv` µV spans half a lane. Clip so a big
                # artifact stays in its own lane instead of covering neighbours.
                scaled = np.clip(yy / float(gain_uv), -1.0, 1.0) * (_LANE * 0.5)
            else:
                scaled = _robust_norm(yy) * (_LANE * 0.45)
            self.curves[ch].setData(tt, lane + scaled)
        self.plot.setXRange(-float(window_s), 0, padding=0)


def _robust_norm(y):
    """Mean-remove and scale by a percentile range (outlier-tolerant)."""
    if len(y) == 0:
        return y
    y = y - np.mean(y)
    span = np.percentile(np.abs(y), 98)
    return y / span if span > 1e-12 else y


class LiveSignalView(QWidget):
    """Scrolling multi-stream view plus its own display controls."""

    def __init__(self, parent=None, refresh_hz=30):
        super().__init__(parent)
        theme.apply_pyqtgraph_defaults()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        bar = QHBoxLayout()
        bar.setContentsMargins(4, 0, 4, 0)
        bar.addWidget(QLabel("Scale"))
        self.gain_combo = QComboBox()
        for g in GAIN_CHOICES:
            self.gain_combo.addItem(f"±{g} µV", g)
        self.gain_combo.setCurrentIndex(GAIN_CHOICES.index(100))
        self.gain_combo.setToolTip(
            "Fixed EEG amplitude scale — how many µV span half a lane.\n\n"
            "Smaller values magnify the trace (good for spotting a weak alpha\n"
            "rhythm); larger values keep blinks and jaw clenches on-screen.\n"
            "Relaxed eyes-closed alpha is typically 20–50 µV.\n"
            "This changes the display only, never the recorded data."
        )
        bar.addWidget(self.gain_combo)

        bar.addSpacing(12)
        bar.addWidget(QLabel("Window"))
        self.window_combo = QComboBox()
        for w in WINDOW_CHOICES:
            self.window_combo.addItem(f"{w} s", w)
        self.window_combo.setCurrentIndex(WINDOW_CHOICES.index(5))
        self.window_combo.setToolTip(
            "How many seconds of history to show.\n\n"
            "Short windows (2–5 s) show individual waves; longer windows\n"
            "(10–30 s) make slow trends and drift easier to see, at the cost\n"
            "of detail. Longer windows redraw slightly more data per frame."
        )
        bar.addWidget(self.window_combo)

        bar.addSpacing(12)
        self.filter_check = QCheckBox("Filter 1–40 Hz + notch")
        self.filter_check.setChecked(True)
        self.filter_check.setToolTip(
            "Display-only band-pass and mains notch, applied to EEG lanes.\n\n"
            "On (recommended): removes DC drift and 50/60 Hz line noise so the\n"
            "brain rhythm is visible. Off: shows the unprocessed signal, useful\n"
            "for judging electrode contact and spotting hardware problems.\n"
            "Recording always saves the unfiltered raw signal either way."
        )
        self.filter_check.toggled.connect(self._on_filter_toggled)
        bar.addWidget(self.filter_check)

        bar.addStretch()
        self.hint = QLabel("Left sensors blue · right sensors amber")
        self.hint.setStyleSheet(f"color: {theme.TEXT_FAINT};")
        bar.addWidget(self.hint)
        root.addLayout(bar)

        # An empty plot area is indistinguishable from a broken one. Until the
        # first samples arrive this says what to do next, in the place the user
        # is already looking, rather than leaving a black rectangle.
        self.placeholder = QLabel(
            "No signal yet.\n\n"
            "Put the headband on and switch it on — MuseStudio looks for it and "
            "connects on its own.\nTraces appear here the moment data arrives.\n\n"
            "If it does not find the headband, use Scan then Connect in the Muse "
            "panel on the left.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet(
            f"color: {theme.TEXT_FAINT}; background: {theme.PLOT_BG}; "
            "font-size: 13px; line-height: 150%;")

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(theme.PLOT_BG)
        self.glw.hide()
        root.addWidget(self.placeholder, stretch=1)
        root.addWidget(self.glw, stretch=1)

        self._panels = {}
        self._channel_names = {}
        self._rates = {}
        self._paused = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_all)
        self._timer.start(int(1000 / refresh_hz))

    # --- configuration -----------------------------------------------------
    def set_channel_names(self, mapping):
        self._channel_names.update(mapping)

    def set_sample_rates(self, mapping):
        """Provide {stream_name: sample_rate} so filters are designed right."""
        self._rates.update({k: v for k, v in mapping.items() if v})

    def set_paused(self, paused):
        """Stop redrawing while this view is hidden (data still buffers)."""
        self._paused = bool(paused)

    def streams(self):
        return {name: panel.channel_names for name, panel in self._panels.items()}

    def set_channel_visibility(self, stream_name, visible_names):
        panel = self._panels.get(stream_name)
        if panel is not None:
            panel.set_visible(visible_names)

    # --- data --------------------------------------------------------------
    def add_samples(self, stream_name, timestamps, data):
        panel = self._panels.get(stream_name)
        if panel is None:
            names = self._channel_names.get(stream_name)
            if not names:
                n = data.shape[1] if getattr(data, "ndim", 1) == 2 else 1
                names = [f"ch{i}" for i in range(n)]
            panel = _StreamPanel(self.glw, stream_name, names,
                                 fs=self._rates.get(stream_name, 256.0))
            panel.set_filtering(self.filter_check.isChecked())
            self._panels[stream_name] = panel
            if self.placeholder.isVisible():
                self.placeholder.hide()
                self.glw.show()
            self._relayout()
        panel.add(timestamps, data)

    def _relayout(self):
        """Re-add every panel so streams appear in priority order."""
        self.glw.clear()
        for row, name in enumerate(sorted(self._panels, key=stream_priority)):
            self.glw.addItem(self._panels[name].plot, row=row, col=0)

    def _on_filter_toggled(self, on):
        for panel in self._panels.values():
            panel.set_filtering(on)

    def _refresh_all(self):
        if self._paused or not self.isVisible():
            return
        gain = self.gain_combo.currentData()
        window = self.window_combo.currentData()
        for panel in self._panels.values():
            panel.refresh(gain, window)

    def clear(self):
        """Drop every panel and return to the placeholder.

        Called on connect (fresh views for a new session) and on disconnect, so
        an idle window explains itself rather than showing an empty black plot
        that looks identical to a crashed one.
        """
        self.glw.clear()
        self._panels.clear()
        self.glw.hide()
        self.placeholder.show()


# Backwards-compatible alias (the window previously used this name).
MultiChannelScrollPlot = LiveSignalView
