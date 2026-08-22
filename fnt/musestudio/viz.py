"""Frequency-domain and laterality views for MuseStudio.

Every band keeps the same colour here that it has in the band selector and the
history plot (see :mod:`fnt.musestudio.theme`), so "alpha" reads the same way
across the whole app.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from fnt.musestudio import theme
from fnt.musestudio.dsp import BAND_ORDER, BANDS


class BandPowerBars(QWidget):
    """Relative power per band, with the dominant band called out."""

    def __init__(self, parent=None):
        super().__init__(parent)
        theme.apply_pyqtgraph_defaults()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        head = QHBoxLayout()
        head.setContentsMargins(6, 0, 6, 0)
        title = QLabel("Band power")
        title.setStyleSheet(f"color: {theme.TEXT_DIM}; font-weight: 600;")
        head.addWidget(title)
        head.addStretch()
        self.dominant = QLabel("—")
        self.dominant.setStyleSheet(
            f"color: {theme.TEXT}; font-weight: 700; font-size: 14px;")
        head.addWidget(self.dominant)
        root.addLayout(head)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(theme.PLOT_BG)
        self.plot = self.glw.addPlot()
        theme.style_plot(self.plot, y_label="relative power")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setYRange(-0.03, 1.0, padding=0)   # breathing room under the bars
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        bottom = self.plot.getAxis("bottom")
        bottom.setTicks([[(i, b) for i, b in enumerate(BAND_ORDER)]])
        bottom.setHeight(30)
        self.plot.setXRange(-0.6, len(BAND_ORDER) - 0.4, padding=0)

        self.bars = pg.BarGraphItem(
            x=list(range(len(BAND_ORDER))), height=[0] * len(BAND_ORDER),
            width=0.62, brushes=[QColor(theme.band_color(b)) for b in BAND_ORDER],
            pen=pg.mkPen(None),
        )
        self.plot.addItem(self.bars)
        root.addWidget(self.glw, stretch=1)

    def update_metrics(self, m):
        heights = [m.mean_relative.get(b, 0.0) for b in BAND_ORDER]
        self.bars.setOpts(height=heights)
        if m.dominant:
            self.dominant.setText(m.dominant.upper())
            self.dominant.setStyleSheet(
                f"color: {theme.band_color(m.dominant)}; "
                "font-weight: 700; font-size: 14px;")


class BandHistoryPlot(QWidget):
    """Relative band power over the last couple of minutes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        theme.apply_pyqtgraph_defaults()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        legend = QHBoxLayout()
        legend.setContentsMargins(6, 0, 6, 0)
        lbl = QLabel("History")
        lbl.setStyleSheet(f"color: {theme.TEXT_DIM}; font-weight: 600;")
        legend.addWidget(lbl)
        legend.addStretch()
        for b in BAND_ORDER:
            chip = QLabel(f"● {b}")
            chip.setStyleSheet(f"color: {theme.band_color(b)}; font-weight: 600;")
            legend.addWidget(chip)
        root.addLayout(legend)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(theme.PLOT_BG)
        self.plot = self.glw.addPlot()
        theme.style_plot(self.plot, x_label="seconds ago", y_label="relative power")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setYRange(0, 1, padding=0)
        # Without this pyqtgraph rescales the axis to "x0.001" style SI units.
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.curves = {
            b: self.plot.plot(pen=pg.mkPen(theme.band_color(b), width=2))
            for b in BAND_ORDER
        }
        root.addWidget(self.glw, stretch=1)

    def update_history(self, times, series):
        if times is None or len(times) < 2:
            return
        rel = np.asarray(times, dtype=float)
        rel = rel - rel[-1]          # newest at 0, older negative
        for b, curve in self.curves.items():
            y = series.get(b)
            if y is not None and len(y) == len(rel):
                curve.setData(rel, y)
        self.plot.setXRange(rel[0], 0, padding=0)


class SpectrogramView(QWidget):
    """Rolling time-frequency heatmap for one channel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        theme.apply_pyqtgraph_defaults()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        bar = QHBoxLayout()
        bar.setContentsMargins(6, 0, 6, 0)
        bar.addWidget(QLabel("Channel"))
        self.channel_combo = QComboBox()
        self.channel_combo.setToolTip("Which electrode the spectrogram follows.")
        bar.addWidget(self.channel_combo)
        bar.addStretch()
        self.scale_hint = QLabel("power dB · brighter = stronger")
        self.scale_hint.setStyleSheet(f"color: {theme.TEXT_FAINT};")
        bar.addWidget(self.scale_hint)
        root.addLayout(bar)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(theme.PLOT_BG)
        self.plot = self.glw.addPlot()
        theme.style_plot(self.plot, x_label="seconds ago", y_label="frequency (Hz)")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self._apply_colormap()

        # Band boundaries, labelled on the right edge so they don't sit on the data.
        self._band_labels = []
        for name, (lo, hi) in BANDS.items():
            self.plot.addItem(pg.InfiniteLine(
                pos=hi, angle=0,
                pen=pg.mkPen(theme.band_color(name), width=1, style=Qt.DashLine)))
            # Label at the band centre (not the edge) so the low bands, whose
            # boundaries are only ~4 Hz apart, don't collide. Anchored to the
            # left (oldest) edge so the text can't be clipped off the right.
            text = pg.TextItem(name, color=theme.band_color(name), anchor=(0, 0.5))
            self.plot.addItem(text)
            self._band_labels.append((text, (lo + hi) / 2.0))
        root.addWidget(self.glw, stretch=1)

    def _apply_colormap(self):
        for name in ("viridis", "inferno", "magma", "CET-L9"):
            try:
                cmap = pg.colormap.get(name)
                if cmap is not None:
                    self.image.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
                    return
            except Exception:
                continue

    def set_channels(self, names):
        current = self.channel_combo.currentIndex()
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        for i, n in enumerate(names):
            self.channel_combo.addItem(str(n), i)
        if 0 <= current < self.channel_combo.count():
            self.channel_combo.setCurrentIndex(current)
        self.channel_combo.blockSignals(False)

    def update_spectrogram(self, sg):
        img = sg.filled_image()
        if img is None:
            return
        lo, hi = sg.levels()
        if hi - lo < 1e-6:
            hi = lo + 1.0
        self.image.setImage(img, levels=(lo, hi), autoLevels=False)
        fmax = float(sg.freqs[-1]) if sg.freqs is not None and len(sg.freqs) else 50.0
        span = sg.span_seconds()
        # Draw only what exists, anchored so "now" sits at x=0.
        self.image.setRect(QRectF(-span, 0, span, fmax))
        self.plot.setXRange(-span, 0, padding=0)
        self.plot.setYRange(0, fmax, padding=0)
        for text, hz in self._band_labels:
            text.setPos(-span, hz)


class LateralityView(QWidget):
    """Left vs right hemisphere comparison (EEG alpha asymmetry + fNIRS ΔOD)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._asym = 0.0
        self._hemo = None
        self._contact = False

    def update_bands(self, m):
        self._asym = m.alpha_asym
        self._contact = m.contact_ok
        self.update()

    def update_hemo(self, h):
        self._hemo = h
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(theme.PLOT_BG))

        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        p.setPen(QPen(QColor(theme.TEXT_DIM)))
        p.drawText(QRectF(0, 4, w / 2, 18), Qt.AlignCenter, "LEFT")
        p.drawText(QRectF(w / 2, 4, w / 2, 18), Qt.AlignCenter, "RIGHT")

        self._diverging(p, 28, w, "EEG alpha asymmetry", self._asym, 1.5,
                        enabled=self._contact,
                        hint="more right alpha →" if self._asym > 0 else "← more left alpha")
        lat = self._hemo.laterality if self._hemo and self._hemo.ready else None
        self._diverging(p, 88, w, "fNIRS ΔOD (proxy)", lat, 0.05,
                        enabled=lat is not None,
                        hint="waiting for optics…" if lat is None else "more right absorption →"
                        if lat > 0 else "← more left absorption")
        p.end()

    def _diverging(self, p, y, w, label, value, full_scale, enabled=True, hint=""):
        pad = 12
        track_w = w - 2 * pad
        cx = pad + track_w / 2

        p.setPen(QPen(QColor(theme.TEXT_DIM)))
        p.drawText(QRectF(pad, y, track_w, 16), Qt.AlignLeft, label)

        track_y = y + 20
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(theme.SURFACE_HI)))
        p.drawRoundedRect(QRectF(pad, track_y, track_w, 14), 7, 7)

        if enabled and value is not None:
            frac = float(np.clip(value / full_scale, -1.0, 1.0))
            bar_w = abs(frac) * (track_w / 2)
            colour = QColor(theme.RIGHT_COLOR if frac > 0 else theme.LEFT_COLOR)
            p.setBrush(QBrush(colour))
            x = cx if frac > 0 else cx - bar_w
            p.drawRoundedRect(QRectF(x, track_y, max(bar_w, 2), 14), 7, 7)

        p.setPen(QPen(QColor(theme.BORDER_HI), 1))
        p.drawLine(int(cx), int(track_y - 2), int(cx), int(track_y + 16))

        p.setPen(QPen(QColor(theme.TEXT_FAINT)))
        value_txt = "—" if (value is None or not enabled) else f"{value:+.3f}"
        p.drawText(QRectF(pad, track_y + 16, track_w, 16), Qt.AlignRight, value_txt)
        p.drawText(QRectF(pad, track_y + 16, track_w, 16), Qt.AlignLeft, hint)
