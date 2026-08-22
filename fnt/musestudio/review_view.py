"""Post-session review — load a recording and see whether the protocol worked.

Three x-linked timelines (spectrogram, band power, synchrony) share one axis
with the protocol phases shaded behind them, so a change in the signal can be
read directly against the stimulus that was playing at that moment. The table
underneath reduces that to numbers: each phase compared against baseline.
"""

import os

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QBrush, QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from fnt.musestudio import theme
from fnt.musestudio.dsp import BAND_ORDER, BANDS
from fnt.musestudio.review import (
    analyze_eeg, audio_intervals, compare_to_baseline, load_session,
    phase_intervals, summarize_phases,
)

# Alternating tints behind protocol phases.
_PHASE_TINTS = ["#4CC2FF", "#7C5CFF", "#16C79A", "#FFB020", "#FF6B6B"]


class ReviewPanel(QWidget):
    """Open a recording folder and inspect it against the protocol timeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        theme.apply_pyqtgraph_defaults()
        self.data = None
        self._band_times = None
        self._band_series = None
        self._spec_fmax = 50.0
        self._spec = None            # (n_freqs, n_times) dB, for hover
        self._spec_times = np.array([])
        self._spec_freqs = np.array([])

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 8, 6, 6)
        root.setSpacing(6)

        bar = QHBoxLayout()
        self.open_btn = QPushButton("Open recording…")
        self.open_btn.setProperty("accent", True)
        self.open_btn.clicked.connect(self.on_open)
        self.open_btn.setToolTip(
            "Choose a *_FNT_MuseStudio_recording folder to analyse.")
        bar.addWidget(self.open_btn)

        self.name_label = QLabel("No recording loaded")
        self.name_label.setStyleSheet(f"color: {theme.TEXT}; font-weight: 600;")
        bar.addWidget(self.name_label)
        bar.addStretch()

        bar.addWidget(QLabel("Channel"))
        self.channel_combo = QComboBox()
        self.channel_combo.setToolTip("Electrode used for the spectrogram and band series.")
        self.channel_combo.activated.connect(lambda _i: self._recompute())
        bar.addWidget(self.channel_combo)
        root.addLayout(bar)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        self.summary_label.setWordWrap(True)
        root.addWidget(self.summary_label)

        split = QSplitter(Qt.Vertical)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(theme.PLOT_BG)

        self.spec_plot = self.glw.addPlot(row=0, col=0)
        theme.style_plot(self.spec_plot, y_label="Hz", title="Spectrogram")
        self.spec_image = pg.ImageItem()
        self.spec_plot.addItem(self.spec_image)
        self._apply_colormap()

        self.band_plot = self.glw.addPlot(row=1, col=0)
        theme.style_plot(self.band_plot, y_label="rel. power", title="Band power")
        self.band_curves = {
            b: self.band_plot.plot(pen=pg.mkPen(theme.band_color(b), width=2))
            for b in BAND_ORDER
        }
        self.band_plot.setYRange(0, 1, padding=0)

        self.sync_plot = self.glw.addPlot(row=2, col=0)
        theme.style_plot(self.sync_plot, x_label="seconds from start",
                         y_label="PLV", title="Interhemispheric synchrony")
        self.plv_curve = self.sync_plot.plot(pen=pg.mkPen(theme.ACCENT, width=2))
        self.sync_plot.setYRange(0, 1, padding=0)

        for p in (self.spec_plot, self.band_plot, self.sync_plot):
            p.setMouseEnabled(y=False)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.getAxis("left").enableAutoSIPrefix(False)
            p.setMinimumHeight(120)
            p.getAxis("left").setWidth(48)
        # The spectrogram carries the most information — give it the most room.
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 2)
        self.glw.ci.layout.setRowStretchFactor(2, 2)
        # One shared time axis: scrubbing one plot scrubs all three.
        self.band_plot.setXLink(self.spec_plot)
        self.sync_plot.setXLink(self.spec_plot)
        # Shared crosshair + numeric readout. The three plots are x-linked, so
        # one cursor position describes all of them at once: the point of the
        # readout is to turn "the lines moved around here" into actual numbers.
        self._cursor_lines = []
        for plot in (self.spec_plot, self.band_plot, self.sync_plot):
            line = pg.InfiniteLine(angle=90, movable=False,
                                   pen=pg.mkPen(theme.TEXT_FAINT, width=1))
            line.setZValue(100)
            line.hide()
            plot.addItem(line, ignoreBounds=True)
            self._cursor_lines.append(line)
        self.glw.scene().sigMouseMoved.connect(self._on_hover)

        self.readout = QLabel("Hover a plot to read values at that moment")
        self.readout.setStyleSheet(
            f"color: {theme.TEXT}; background: {theme.SURFACE_HI}; "
            f"border: 1px solid {theme.BORDER}; border-radius: 6px; padding: 6px;")
        self.readout.setWordWrap(True)
        self.readout.setToolTip(
            "Values under the cursor.\n\n"
            "Spectrogram: power in dB at that time and frequency — brighter is "
            "stronger, and a steady horizontal band near 10 Hz is an alpha rhythm.\n"
            "Bands: each band's share of total power (they sum to 1).\n"
            "PLV: interhemispheric phase locking, 0 = independent, 1 = locked."
        )

        split.addWidget(self.glw)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Phase", "Duration", "PLV", "Δ PLV", "Alpha", "Δ Alpha", "Contact"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 7):
            self.table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.setToolTip(
            "Per-phase means. Δ columns compare each phase against the baseline "
            "phase — that is the number that says whether the protocol moved you."
        )
        split.addWidget(self.readout)
        split.addWidget(self.table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([540, 200])
        root.addWidget(split, stretch=1)

        self._phase_items = []

    def _on_hover(self, pos):
        """Report every series' value at the hovered instant."""
        if self.data is None:
            return
        plot = next((p for p in (self.spec_plot, self.band_plot, self.sync_plot)
                     if p.sceneBoundingRect().contains(pos)), None)
        if plot is None:
            for line in self._cursor_lines:
                line.hide()
            return
        point = plot.vb.mapSceneToView(pos)
        t = float(point.x())
        for line in self._cursor_lines:
            line.setPos(t)
            line.show()

        bits = [f"<b>t = {t:.1f}s</b>"]
        phase = next((n for n, a, b in phase_intervals(self.data) if a <= t < b), None)
        if phase:
            bits.append(f"phase <b>{phase}</b>")

        # Spectrogram: report the frequency actually under the pointer.
        if plot is self.spec_plot and self._spec is not None and len(self._spec_times):
            hz = float(point.y())
            ti = int(np.clip(np.searchsorted(self._spec_times, t), 0,
                             self._spec.shape[1] - 1))
            fi = int(np.clip(np.searchsorted(self._spec_freqs, hz), 0,
                             self._spec.shape[0] - 1))
            bits.append(f"{self._spec_freqs[fi]:.1f} Hz = "
                        f"{self._spec[fi, ti]:.1f} dB")

        if self._band_times is not None and len(self._band_times):
            i = int(np.clip(np.searchsorted(self._band_times, t), 0,
                            len(self._band_times) - 1))
            bands = "  ".join(
                f"<span style='color:{theme.band_color(b)}'>{b} "
                f"{self._band_series[b][i]:.2f}</span>" for b in BAND_ORDER)
            bits.append(bands)

        d = self.data
        if d.synchrony is not None and "plv_combined" in d.synchrony and len(d.synchrony):
            st = d.synchrony["lsl_timestamp"].to_numpy(dtype=float) - d.t0
            i = int(np.clip(np.searchsorted(st, t), 0, len(st) - 1))
            bits.append(f"PLV {d.synchrony['plv_combined'].to_numpy(float)[i]:.3f}")

        self.readout.setText("&nbsp;&nbsp;·&nbsp;&nbsp;".join(bits))

    def _apply_colormap(self):
        for name in ("viridis", "inferno", "magma", "CET-L9"):
            try:
                cmap = pg.colormap.get(name)
                if cmap is not None:
                    self.spec_image.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
                    return
            except Exception:
                continue

    # --- loading ----------------------------------------------------------
    def on_open(self, _checked=False, path=None):
        if path is None:
            path = QFileDialog.getExistingDirectory(
                self, "Open a MuseStudio recording folder")
        if not path:
            return
        self.load(path)

    def load(self, path):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.data = load_session(path)
            self.channel_combo.clear()
            for i, ch in enumerate(self.data.eeg_channels):
                self.channel_combo.addItem(str(ch), i)
            self.name_label.setText(self.data.name)
            self._recompute()
        finally:
            QApplication.restoreOverrideCursor()

    def _recompute(self):
        d = self.data
        if d is None:
            return
        idx = self.channel_combo.currentData() or 0
        times, freqs, spec, series = analyze_eeg(d, idx)
        self._band_times, self._band_series = times, series
        self._spec, self._spec_times, self._spec_freqs = spec, times, freqs
        # Phase labels are placed relative to the spectrogram's top, so the
        # frequency range has to be known before the phases are drawn.
        self._spec_fmax = float(freqs[-1]) if len(freqs) else 50.0

        self._draw_phases()
        if spec is not None and len(times):
            lo = float(np.percentile(spec, 5))
            hi = float(np.percentile(spec, 99))
            if hi - lo < 1e-6:
                hi = lo + 1.0
            self.spec_image.setImage(spec, levels=(lo, hi), autoLevels=False)
            self.spec_image.setRect(
                pg.QtCore.QRectF(times[0], 0, times[-1] - times[0], float(freqs[-1])))
            self.spec_plot.setYRange(0, float(freqs[-1]), padding=0)
            for b, curve in self.band_curves.items():
                curve.setData(times, series[b])
            self.spec_plot.setXRange(times[0], times[-1], padding=0)

        if d.synchrony is not None and "plv_combined" in d.synchrony:
            st = d.synchrony["lsl_timestamp"].to_numpy(dtype=float) - d.t0
            self.plv_curve.setData(st, d.synchrony["plv_combined"].to_numpy(dtype=float))
        else:
            self.plv_curve.setData([], [])

        rows = summarize_phases(d, times, series)
        self._fill_table(rows)
        self._set_summary(d, rows)

    def _draw_phases(self):
        """Shade protocol phases behind every timeline."""
        for plot, item in self._phase_items:
            plot.removeItem(item)
        self._phase_items = []
        phases = phase_intervals(self.data)
        # Inset labels slightly so the first one isn't clipped by the axis.
        span = max((p[2] for p in phases), default=1.0) or 1.0
        inset = span * 0.004
        label_y = self._spec_fmax * 0.94
        for i, (label, t0, t1) in enumerate(phases):
            tint = QColor(_PHASE_TINTS[i % len(_PHASE_TINTS)])
            tint.setAlpha(28)
            for plot in (self.spec_plot, self.band_plot, self.sync_plot):
                region = pg.LinearRegionItem(values=(t0, t1), movable=False,
                                             brush=QBrush(tint))
                region.setZValue(-100)
                for line in region.lines:
                    line.setPen(pg.mkPen(theme.BORDER_HI, width=1))
                plot.addItem(region)
                self._phase_items.append((plot, region))
            text = pg.TextItem(label, color=theme.TEXT, anchor=(0, 0))
            text.setPos(t0 + inset, label_y)
            self.spec_plot.addItem(text)
            self._phase_items.append((self.spec_plot, text))

        # Mark when a tone was actually audible.
        for t0, t1, beat in audio_intervals(self.data):
            for plot in (self.band_plot, self.sync_plot):
                line = pg.InfiniteLine(pos=t0, angle=90,
                                       pen=pg.mkPen(theme.GOOD, width=1,
                                                    style=Qt.DashLine))
                plot.addItem(line)
                self._phase_items.append((plot, line))

    def _fill_table(self, rows):
        base, deltas = compare_to_baseline(rows)
        delta_by_phase = {d["phase"]: d for d in deltas}
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            is_base = base is not None and row is base
            d = delta_by_phase.get(row["phase"], {})
            cells = [
                row["phase"] + ("  (baseline)" if is_base else ""),
                f"{row['duration']:.0f} s",
                _fmt(row.get("plv")),
                "—" if is_base else _fmt(d.get("d_plv"), signed=True),
                _fmt(row.get("alpha")),
                "—" if is_base else _fmt(d.get("d_alpha"), signed=True),
                _fmt_pct(row.get("contact")),
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c > 0:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if is_base:
                    f = QFont()
                    f.setBold(True)
                    item.setFont(f)
                # Colour the deltas: green = went up, red = went down.
                if c in (3, 5) and not is_base:
                    val = d.get("d_plv" if c == 3 else "d_alpha")
                    if val is not None and not np.isnan(val):
                        item.setForeground(QBrush(QColor(
                            theme.GOOD if val > 0 else theme.DANGER)))
                # Flag phases whose contact was poor — their means are unreliable.
                if c == 6:
                    contact = row.get("contact")
                    if contact is not None and not np.isnan(contact) and contact < 0.8:
                        item.setForeground(QBrush(QColor(theme.WARN)))
                self.table.setItem(r, c, item)

    def _set_summary(self, d, rows):
        bits = []
        subj = (d.config.get("subject") or {})
        if subj.get("id"):
            bits.append(f"subject {subj['id']}")
        proto = d.config.get("protocol") or d.config.get("mode") or "free recording"
        bits.append(f"{proto}")
        if d.duration:
            bits.append(f"{d.duration / 60:.1f} min")
        if d.eeg_channels:
            bits.append(f"{len(d.eeg_channels)} EEG ch @ {d.fs:.0f} Hz")
        if not phase_intervals(d):
            bits.append("no protocol phases logged — showing the whole recording")
        base, deltas = compare_to_baseline(rows)
        if base and deltas:
            best = max(deltas, key=lambda x: (x["d_plv"]
                                              if not np.isnan(x["d_plv"]) else -9))
            if not np.isnan(best["d_plv"]):
                arrow = "rose" if best["d_plv"] > 0 else "fell"
                bits.append(f"PLV {arrow} {abs(best['d_plv']):+.3f} in "
                            f"{best['phase']} vs baseline")
        for note in d.notes:
            bits.append(f"⚠ {note}")
        self.summary_label.setText("   ·   ".join(bits))


def _fmt(v, signed=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def _fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.0f}%"
