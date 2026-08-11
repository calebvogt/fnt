"""Visual feedback widgets for MuseStudio closed-loop biofeedback.

- HeadMapWidget: top-down head schematic with the four Athena electrodes
  heat-mapped by band power and arcs between the homologous left/right pairs
  whose thickness/colour encode the pair PLV.
- SynchronyMeter: a vertical desync->sync gauge.
- NeuroPanel: groups the two, plus a band selector and baseline-calibrate
  button; emits band_changed / calibrate_requested for the host to wire to the
  SynchronyAnalyzer.
"""

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from fnt.musestudio import theme
from fnt.musestudio.synchrony import BANDS, PAIRS

# Electrode positions in a top-down unit head (nose up); (x, y) in 0..1.
ELECTRODE_POS = {
    "AF7": (0.36, 0.24), "AF8": (0.64, 0.24),
    "TP9": (0.22, 0.72), "TP10": (0.78, 0.72),
}


def _heat_color(v):
    """Band-power heatmap: 0 -> blue (cool), 1 -> red (hot)."""
    v = float(np.clip(v, 0, 1))
    stops = [(0.0, (55, 138, 221)), (0.5, (151, 196, 89)),
             (0.75, (239, 159, 39)), (1.0, (226, 75, 74))]
    return QColor(*_interp(v, stops))


def _sync_color(v):
    """Synchrony scale: 0 -> muted red, 1 -> teal/green."""
    v = float(np.clip(v, 0, 1))
    stops = [(0.0, (163, 90, 90)), (0.5, (186, 117, 23)), (1.0, (29, 158, 117))]
    return QColor(*_interp(v, stops))


def _interp(v, stops):
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            f = (v - a) / (b - a) if b > a else 0.0
            return tuple(int(ca[i] + f * (cb[i] - ca[i])) for i in range(3))
    return stops[-1][1]


class HeadMapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self._metrics = None

    def update_metrics(self, metrics):
        self._metrics = metrics
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        side = min(w, h) - 20
        ox, oy = (w - side) / 2, (h - side) / 2

        def pt(nx, ny):
            return QPointF(ox + nx * side, oy + ny * side)

        m = self._metrics
        dim = m is None or not m.contact_ok

        # Head outline + nose + ears.
        p.setPen(QPen(QColor("#888880"), 2))
        p.setBrush(QBrush(QColor(40, 40, 40)))
        cx, cy, r = ox + 0.5 * side, oy + 0.5 * side, 0.42 * side
        p.drawEllipse(QPointF(cx, cy), r, r)
        nose = QPainterPath()
        nose.moveTo(pt(0.5, 0.05))
        nose.lineTo(pt(0.44, 0.11)); nose.lineTo(pt(0.56, 0.11)); nose.closeSubpath()
        p.drawPath(nose)
        p.drawEllipse(pt(0.06, 0.5), 8, 16)
        p.drawEllipse(pt(0.94, 0.5), 8, 16)

        # Synchrony arcs between homologous pairs.
        plv = (m.plv if m else {})
        for label, left, right in PAIRS:
            lv = 0.0 if dim else float(plv.get(label, 0.0))
            a, b = ELECTRODE_POS[left], ELECTRODE_POS[right]
            path = QPainterPath()
            path.moveTo(pt(*a))
            bow = -0.10 if a[1] < 0.5 else 0.10
            path.quadTo(pt(0.5, (a[1] + b[1]) / 2 + bow), pt(*b))
            col = QColor("#555550") if dim else _sync_color(lv)
            p.setPen(QPen(col, 2 + lv * 10, Qt.SolidLine, Qt.RoundCap))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        # Electrodes coloured by (normalized) band power.
        powers = (m.band_power if m else {})
        vals = np.array([powers.get(e, 0.0) for e in ELECTRODE_POS]) if powers else None
        norm = {}
        if vals is not None and vals.max() > vals.min():
            for e, v in zip(ELECTRODE_POS, (vals - vals.min()) / (vals.max() - vals.min())):
                norm[e] = v
        for e, (nx, ny) in ELECTRODE_POS.items():
            col = QColor("#3f3f3f") if dim else _heat_color(norm.get(e, 0.5))
            p.setPen(QPen(QColor("#1e1e1e"), 2))
            p.setBrush(QBrush(col))
            c = pt(nx, ny)
            p.drawEllipse(c, 18, 18)
            p.setPen(QPen(QColor("#eeeeee")))
            p.drawText(QRectF(c.x() - 20, c.y() - 8, 40, 16), Qt.AlignCenter, e)

        if dim:
            p.setPen(QPen(QColor("#e0a030")))
            p.drawText(self.rect().adjusted(0, 6, 0, 0), Qt.AlignHCenter | Qt.AlignTop,
                       "poor contact" if m else "waiting for EEG…")
        p.end()


class SynchronyMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(96, 220)
        self.setMaximumWidth(120)
        self._metrics = None

    def update_metrics(self, metrics):
        self._metrics = metrics
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        bar_w = 34
        x = (w - bar_w) / 2
        top, bot = 26, h - 40
        span = bot - top

        # Zones: green (top), amber (mid), red (bottom).
        for frac0, frac1, col in [(0.66, 1.0, "#97C459"),
                                   (0.33, 0.66, "#EF9F27"),
                                   (0.0, 0.33, "#E24B4A")]:
            y0 = bot - frac1 * span
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(col))
            p.drawRect(QRectF(x, y0, bar_w, (frac1 - frac0) * span))

        m = self._metrics
        level = 0.0 if m is None else float(m.level)
        y = bot - level * span
        contact = m is not None and m.contact_ok
        marker = QColor("#f0f0f0") if contact else QColor("#777777")
        p.setPen(QPen(marker, 4))
        p.drawLine(int(x - 6), int(y), int(x + bar_w + 6), int(y))

        p.setPen(QPen(QColor("#cccccc")))
        p.drawText(QRectF(0, 2, w, 20), Qt.AlignHCenter, "sync")
        p.drawText(QRectF(0, h - 20, w, 18), Qt.AlignHCenter,
                   f"{int(level * 100)}%")
        if m is not None and not m.calibrated:
            p.setPen(QPen(QColor("#e0a030")))
            p.drawText(QRectF(0, h - 38, w, 16), Qt.AlignHCenter, "raw")
        p.end()


class NeuroControls(QGroupBox):
    """Left-column controls: band selector, calibrate button, status."""

    band_changed = pyqtSignal(str)
    calibrate_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Neurofeedback", parent)
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Band"))
        self.band_combo = QComboBox()
        for name in BANDS:
            label = name + (" (EMG-prone)" if name == "gamma" else "")
            self.band_combo.addItem(label, name)
        self.band_combo.setCurrentText("alpha")
        self.band_combo.activated.connect(
            lambda _i: self.band_changed.emit(self.band_combo.currentData())
        )
        self.band_combo.setToolTip(
            "Frequency band to measure synchrony in. Alpha (8–12 Hz) is most "
            "reliable on Muse; gamma is easily contaminated by muscle activity."
        )
        row.addWidget(self.band_combo, stretch=1)
        root.addLayout(row)
        self.calibrate_btn = QPushButton("Calibrate baseline (30s)")
        self.calibrate_btn.clicked.connect(self.calibrate_requested.emit)
        self.calibrate_btn.setToolTip(
            "Sit still for 30 s to record your resting synchrony. The meter is "
            "then shown relative to this baseline."
        )
        root.addWidget(self.calibrate_btn)
        self.status = QLabel("Connect the Muse to begin.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #999999;")
        root.addWidget(self.status)

    def set_status(self, text):
        self.status.setText(text)


class NeuroView(QGroupBox):
    """Right-side data view: head map + synchrony meter."""

    def __init__(self, parent=None):
        super().__init__("Hemisphere synchrony", parent)
        self.headmap = HeadMapWidget()
        self.headmap.setToolTip(
            "Top-down head. Disc colour = band power at each electrode; the arcs "
            "between left/right pairs brighten and thicken as their phases lock."
        )
        self.meter = SynchronyMeter()
        self.meter.setToolTip(
            "Interhemispheric phase-locking (PLV): bottom = desynchronized, "
            "top = synchronized. 'raw' means no baseline set yet."
        )
        root = QVBoxLayout(self)
        body = QHBoxLayout()
        body.addWidget(self.headmap, stretch=1)
        body.addWidget(self.meter)
        root.addLayout(body, stretch=1)

        # Numeric readouts — the head map shows shape, these give the values.
        stats = QHBoxLayout()
        stats.setSpacing(18)
        self._stat_labels = {}
        for key, title, tip in (
            ("frontal", "Frontal AF7↔AF8",
             "Phase-locking between the two forehead electrodes (0–1).\n"
             "Most sensitive to frontal-midline activity, but also the pair\n"
             "most affected by blinks and brow movement."),
            ("temporal", "Temporal TP9↔TP10",
             "Phase-locking between the two behind-the-ear electrodes (0–1).\n"
             "Usually the steadier pair — less blink contamination, but more\n"
             "affected by jaw tension."),
            ("drift", "Heterodyne drift",
             "How fast the left/right phase difference is rotating, in Hz.\n"
             "Near zero means the hemispheres hold a fixed phase relationship.\n"
             "A sustained non-zero value during the heterodyne protocol is the\n"
             "effect that protocol is trying to induce."),
            ("state", "Signal",
             "Whether the current reading is trustworthy. 'poor contact' means\n"
             "an electrode is loose or you moved; 'raw' means no resting\n"
             "baseline has been recorded yet, so the meter shows absolute PLV\n"
             "rather than change from your own baseline."),
        ):
            box = QVBoxLayout()
            cap = QLabel(title)
            cap.setStyleSheet(f"color: {theme.TEXT_FAINT}; font-size: 10px;")
            cap.setToolTip(tip)
            val = QLabel("—")
            val.setStyleSheet(f"color: {theme.TEXT}; font-size: 15px; font-weight: 700;")
            val.setToolTip(tip)
            box.addWidget(cap)
            box.addWidget(val)
            stats.addLayout(box)
            self._stat_labels[key] = val
        stats.addStretch()
        root.addLayout(stats)

    def update_metrics(self, metrics):
        self.headmap.update_metrics(metrics)
        self.meter.update_metrics(metrics)
        plv = metrics.plv or {}
        self._stat_labels["frontal"].setText(f"{plv.get('frontal', 0.0):.2f}")
        self._stat_labels["temporal"].setText(f"{plv.get('temporal', 0.0):.2f}")
        self._stat_labels["drift"].setText(f"{metrics.drift_hz:+.2f} Hz")
        if not metrics.contact_ok:
            state, colour = "poor contact", theme.WARN
        elif not metrics.calibrated:
            state, colour = "raw (uncalibrated)", theme.TEXT_DIM
        else:
            state, colour = "good", theme.GOOD
        self._stat_labels["state"].setText(state)
        self._stat_labels["state"].setStyleSheet(
            f"color: {colour}; font-size: 15px; font-weight: 700;")
