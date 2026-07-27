"""Live scrolling multichannel plot for MuseStudio.

One stacked panel per LSL stream (EEG, fNIRS, ...), one trace per channel.
Data arrives via :meth:`add_samples` (driven by ``LSLReaderThread.samples_ready``)
and is buffered in ring buffers; a QTimer repaints at a fixed rate decoupled
from the (much higher) data rate.

Traces are robustly auto-scaled per channel (mean-subtracted, std-normalized)
and stacked with fixed spacing so EEG (µV) and fNIRS (arbitrary units) are both
readable without manual gain tuning.
"""

from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QVBoxLayout, QWidget

# Per-stream window length in seconds and an assumed upper-bound sample rate
# used only to size ring buffers. Oversizing is harmless.
WINDOW_SECONDS = 5.0
_MAX_RATE_HZ = 300  # > EEG 256 Hz; buffers hold WINDOW_SECONDS * this
_LANE_SPACING = 1.0  # vertical spacing between stacked channel lanes


class _StreamPanel:
    """Holds the pyqtgraph PlotItem, curves and ring buffers for one stream."""

    def __init__(self, glw, name, channel_names, row):
        self.name = name
        self.channel_names = channel_names
        n = len(channel_names)
        maxlen = int(WINDOW_SECONDS * _MAX_RATE_HZ)

        self.t = deque(maxlen=maxlen)
        self.buffers = [deque(maxlen=maxlen) for _ in range(n)]

        self.plot = glw.addPlot(row=row, col=0)
        self.plot.setTitle(name, color="#cccccc", size="9pt")
        self.plot.showGrid(x=True, y=False, alpha=0.15)
        self.plot.getAxis("left").setStyle(showValues=False)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()

        # Lane labels on the left axis.
        ticks = [
            (((n - 1 - i) * _LANE_SPACING), channel_names[i]) for i in range(n)
        ]
        self.plot.getAxis("left").setStyle(showValues=True)
        self.plot.getAxis("left").setTicks([ticks])
        self.plot.getAxis("left").setTextPen("#999999")
        self.plot.getAxis("bottom").setTextPen("#999999")
        self.plot.setYRange(-_LANE_SPACING, n * _LANE_SPACING)

        palette = ["#4fc3f7", "#81c784", "#ffb74d", "#e57373",
                   "#ba68c8", "#f06292", "#a1887f", "#90a4ae"]
        self.curves = []
        for i in range(n):
            pen = pg.mkPen(palette[i % len(palette)], width=1)
            self.curves.append(self.plot.plot(pen=pen))

    def add(self, timestamps, data):
        for i in range(len(timestamps)):
            self.t.append(timestamps[i])
        n_ch = len(self.buffers)
        for ch in range(n_ch):
            col = data[:, ch] if data.ndim == 2 and data.shape[1] > ch else data
            self.buffers[ch].extend(col.tolist())

    def refresh(self):
        if len(self.t) < 2:
            return
        t = np.fromiter(self.t, dtype=float, count=len(self.t))
        t = t - t[-1]  # show relative time, newest at 0
        n_ch = len(self.buffers)
        for ch in range(n_ch):
            y = np.fromiter(self.buffers[ch], dtype=float, count=len(self.buffers[ch]))
            if len(y) != len(t):
                m = min(len(y), len(t))
                y = y[-m:]
                tt = t[-m:]
            else:
                tt = t
            lane = (n_ch - 1 - ch) * _LANE_SPACING
            self.curves[ch].setData(tt, lane + _normalize(y))


def _normalize(y):
    """Mean-subtract and scale a channel into roughly +/-0.4 lane units."""
    if len(y) == 0:
        return y
    mu = np.mean(y)
    sd = np.std(y)
    if sd < 1e-9:
        return y - mu
    return (y - mu) / sd * 0.3 * _LANE_SPACING


class MultiChannelScrollPlot(QWidget):
    """Container that lazily creates one stacked panel per stream."""

    def __init__(self, parent=None, refresh_hz=30):
        super().__init__(parent)
        pg.setConfigOptions(antialias=False, background="#1e1e1e", foreground="#cccccc")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self.glw)

        self._panels = {}  # stream_name -> _StreamPanel
        self._channel_names = {}  # provided by the window when a stream connects

        self._timer = pg.QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh_all)
        self._timer.start(int(1000 / refresh_hz))

    def set_channel_names(self, mapping):
        """Provide {stream_name: [channel names]} before samples arrive."""
        self._channel_names.update(mapping)

    def add_samples(self, stream_name, timestamps, data):
        panel = self._panels.get(stream_name)
        if panel is None:
            names = self._channel_names.get(stream_name)
            if not names:
                n = data.shape[1] if data.ndim == 2 else 1
                names = [f"{stream_name}[{i}]" for i in range(n)]
            panel = _StreamPanel(self.glw, stream_name, names, row=len(self._panels))
            self._panels[stream_name] = panel
        panel.add(timestamps, data)

    def _refresh_all(self):
        for panel in self._panels.values():
            panel.refresh()

    def clear(self):
        self.glw.clear()
        self._panels.clear()
