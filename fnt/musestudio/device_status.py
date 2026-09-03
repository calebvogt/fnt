"""Always-visible headband status strip.

Answers, at a glance and without changing tabs: is the headband connected, is
data actually arriving, how much battery is left, and is each electrode making
contact.

The **data rate** field matters more than it looks. "Connected" only means an
LSL inlet was opened — it stays true while a browned-out or out-of-range
headband sends nothing at all, which is how a protocol can run to completion
and record an empty file. A live samples-per-second number makes that failure
visible immediately.
"""

from collections import deque

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget,
)

from fnt.musestudio import theme
from fnt.musestudio.dsp import EEG_ELECTRODES


class _LiveTrace(QWidget):
    """A tiny always-visible EEG trace. Movement is the connection indicator.

    A coloured dot and a "256 Hz" label both failed the only test that matters:
    an operator wearing the headband could not tell at a glance whether anything
    was arriving. Both encode liveness as something you must read and interpret.
    A trace that visibly moves does not need interpreting — if it is scrolling,
    data is arriving, and if it flatlines you know instantly.
    """

    def __init__(self, parent=None, width=120):
        super().__init__(parent)
        self.setFixedSize(width, 26)
        self._buf = deque(maxlen=width)
        self._live = False
        self.setToolTip(
            "Live EEG from the best-contact electrode.\n\n"
            "If this is moving, data is arriving. A flat line means the stream "
            "has stopped even if the headband still says connected.")

    def push(self, values):
        for v in values:
            self._buf.append(float(v))
        self._live = True
        self.update()

    def set_live(self, live):
        if not live:
            self._buf.clear()
        self._live = live
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2.0
        if len(self._buf) < 2:
            p.setPen(QColor(theme.TEXT_FAINT))
            p.drawLine(0, int(mid), w, int(mid))
            p.end()
            return
        y = np.fromiter(self._buf, dtype=float, count=len(self._buf))
        y = y - np.mean(y)
        span = float(np.percentile(np.abs(y), 98)) or 1.0
        y = np.clip(y / span, -1.0, 1.0) * (h * 0.42)
        p.setPen(QPen(QColor(theme.GOOD if self._live else theme.TEXT_FAINT), 1.3))
        n = len(y)
        step = w / max(n - 1, 1)
        path = QPainterPath()
        path.moveTo(0.0, mid - y[0])
        for i in range(1, n):
            path.lineTo(i * step, mid - y[i])
        p.drawPath(path)
        p.end()


class _Dot(QWidget):
    """Small status light with a label underneath."""

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.label = label
        self.colour = theme.TEXT_FAINT
        self.setFixedSize(34, 30)

    def set_colour(self, colour):
        if colour != self.colour:
            self.colour = colour
            self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.colour))
        p.drawEllipse(int(self.width() / 2) - 4, 2, 8, 8)
        p.setPen(QColor(theme.TEXT_FAINT))
        f = QFont()
        f.setPointSize(9)
        p.setFont(f)
        p.drawText(0, 12, self.width(), 16, Qt.AlignHCenter, self.label)
        p.end()


class DeviceStatusBar(QWidget):
    """Connection · data rate · battery · per-electrode contact — and the
    connect control itself.

    The connect button lives here rather than in a panel of its own because
    connecting is normally automatic: the window remembers the last headband and
    reconnects on open. What remains is a status readout plus a single fallback
    for the times that fails, and both belong in the same always-visible strip.
    A whole left-column group for three controls the operator no longer touches
    was costing the most valuable space in the window.
    """

    connect_clicked = pyqtSignal()

    def __init__(self, parent=None, compact=False):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(38)
        self._compact = compact

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 2, 10, 2)
        row.setSpacing(14)

        self.link = _Dot("muse")
        self.link.setToolTip("Green when a Muse is connected and streaming.")
        row.addWidget(self.link)

        self._name = ""
        self.state_label = QLabel("Not connected")
        self.state_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        row.addWidget(self.state_label)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setFixedHeight(24)
        self.connect_btn.clicked.connect(self.connect_clicked.emit)
        self.connect_btn.setToolTip(
            "Connect to the headband.\n\nNormally unnecessary — MuseStudio "
            "looks for the headband it used last as soon as you open it. Use "
            "this if that fails, or to disconnect.")
        row.addWidget(self.connect_btn)

        self.trace = _LiveTrace()
        row.addWidget(self.trace)

        self.rate_label = QLabel("")
        self.rate_label.setToolTip(
            "EEG samples arriving per second.\n\n"
            "This is the field to trust: 'connected' only means a stream was "
            "opened, but a headband that has browned out or gone out of range "
            "keeps that status while sending nothing. If this reads 0, you are "
            "recording an empty file."
        )
        self.rate_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        row.addWidget(self.rate_label)

        self.battery_label = QLabel("battery —")
        self.battery_label.setToolTip("Headband battery. Below ~15% the "
                                      "Bluetooth radio browns out and streams "
                                      "start dropping.")
        self.battery_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        row.addWidget(self.battery_label)

        row.addStretch()

        contact_label = QLabel("contact")
        contact_label.setStyleSheet(f"color: {theme.TEXT_FAINT};")
        row.addWidget(contact_label)
        self.contacts = {}
        for name in EEG_ELECTRODES:
            dot = _Dot(name)
            dot.setToolTip(
                f"{name} electrode contact.\n"
                "Green = usable, amber = marginal, red = not making contact.\n"
                "Judged on the high-passed signal, so a large DC offset alone "
                "does not count as bad contact."
            )
            self.contacts[name] = dot
            row.addWidget(dot)

        self.rec_dot = _Dot("rec")
        self.rec_dot.setToolTip("Red while a recording is being written to disk.")
        row.addWidget(self.rec_dot)

        self.setStyleSheet(
            f"background-color: {theme.SURFACE}; border: 1px solid {theme.BORDER};"
            " border-radius: 8px;")
        self.set_connected(False)

    # --- updates ----------------------------------------------------------
    def push_signal(self, values):
        """Feed the live trace (a decimated slice of the newest EEG chunk)."""
        self.trace.push(values)

    def set_busy(self, text):
        """Show an in-progress state (scanning, connecting)."""
        self.state_label.setText(text)
        self.link.set_colour(theme.WARN)
        self.connect_btn.setEnabled(False)

    def set_connect_enabled(self, enabled):
        self.connect_btn.setEnabled(enabled)

    def set_connected(self, connected, name=""):
        self._name = name if connected else ""
        self.link.set_colour(theme.GOOD if connected else theme.DANGER)
        self.state_label.setText(name if connected and name else
                                 ("Connected" if connected else "Not connected"))
        self.connect_btn.setText("Disconnect" if connected else "Connect")
        self.connect_btn.setEnabled(True)
        self.trace.set_live(connected)
        if not connected:
            self.rate_label.setText("")
            self.battery_label.setText("battery —")
            for dot in self.contacts.values():
                dot.set_colour(theme.TEXT_FAINT)

    def set_rate(self, samples_per_sec, expected=None):
        """Live EEG throughput; red when nothing is arriving."""
        if samples_per_sec is None:
            self.rate_label.setText("")
            return
        self.rate_label.setText(f"{samples_per_sec:.0f} Hz")
        if samples_per_sec <= 0:
            colour = theme.DANGER
            self.rate_label.setText("NO DATA")
            self.trace.set_live(False)
            # Say it in the state field too. Showing the device name next to
            # "NO DATA" still reads as a working headband, and that ambiguity is
            # exactly what let a protocol run to completion recording nothing.
            self.state_label.setText("NOT STREAMING")
            self.state_label.setStyleSheet(
                f"color: {theme.DANGER}; font-weight: 700;")
        else:
            if self._name:
                self.state_label.setText(self._name)
            self.state_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
            if expected and samples_per_sec < expected * 0.6:
                colour = theme.WARN
            else:
                colour = theme.TEXT_DIM
        self.rate_label.setStyleSheet(f"color: {colour}; font-weight: 600;")
        # The link light follows real data, not just an open socket.
        if samples_per_sec <= 0:
            self.link.set_colour(theme.DANGER)
        else:
            self.link.set_colour(theme.GOOD)

    def set_battery(self, pct):
        if pct is None:
            self.battery_label.setText("battery —")
            return
        self.battery_label.setText(f"battery {pct:.0f}%")
        colour = (theme.DANGER if pct < 10 else
                  theme.WARN if pct < 20 else theme.TEXT_DIM)
        self.battery_label.setStyleSheet(f"color: {colour}; font-weight: 600;")

    def set_contacts(self, per_channel):
        """``{channel_name: bool}`` — matched loosely so EEG_AF7 finds AF7."""
        for name, dot in self.contacts.items():
            state = None
            for ch, ok in (per_channel or {}).items():
                if name in str(ch).upper():
                    state = ok
                    break
            if state is None:
                dot.set_colour(theme.TEXT_FAINT)
            else:
                dot.set_colour(theme.GOOD if state else theme.DANGER)

    def set_recording(self, recording):
        self.rec_dot.set_colour(theme.DANGER if recording else theme.TEXT_FAINT)
