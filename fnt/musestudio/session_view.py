"""Full-screen session view — what the subject looks at during a protocol.

Two things here are measurement decisions, not decoration:

**A fixation target during eyes-open blocks.** "Rest your gaze on one spot" with
no spot provided means the subject picks a different one each session and their
eyes wander. Wandering gaze means saccades and eye-movement artifact in exactly
the block used as the eyes-open control, which weakens the alpha comparison the
whole probe depends on. A fixed cross is the standard remedy.

**A near-black screen during eyes-closed blocks.** Light still reaches the
retina through closed eyelids, and retinal illumination suppresses alpha. A
bright window a foot from your face during the "eyes closed" blocks works
directly against the state being measured, so those phases dim to almost
nothing.

Escape returns to the normal window at any time; the session keeps running.
"""

import math
import time

from PyQt5.QtCore import Qt, QPointF, QRectF, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from fnt.musestudio import theme

# Backgrounds per gaze mode. "closed" is deliberately near-black.
_BG = {"fixate": "#0B0E12", "pursuit": "#0B0E12", "closed": "#000000",
       "": "#0B0E12"}
_RING_BG = "#1E2630"
_RING_FG = "#2E9BFF"
# Smooth-pursuit sweep: one full left-right-left cycle every PURSUIT_PERIOD_S.
# Slow enough to be followed smoothly rather than in saccades, which is the
# whole point — saccades produce spiky EOG, smooth pursuit a clean slow sweep.
PURSUIT_PERIOD_S = 6.0
PURSUIT_SPAN = 0.62          # fraction of screen width travelled
_DIM_TEXT = "#2A3038"        # barely-visible text during eyes-closed blocks


class FullscreenSessionView(QWidget):
    """Distraction-free phase display with a fixation target."""

    exited = pyqtSignal()          # Escape pressed
    continue_pressed = pyqtSignal()  # Space/Enter on a wait-for-user phase

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("MuseStudio — session")
        self.setStyleSheet(f"background-color: {_BG['']};")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.BlankCursor)

        self._phase = None
        self._index = 0
        self._total = 0
        self._waiting = False
        self._remaining = None
        self._duration = None
        self._tick_at = time.monotonic()
        self._pursuit_timer = QTimer(self)
        self._pursuit_timer.timeout.connect(self.update)

    # --- state ------------------------------------------------------------
    def _sync_pursuit_timer(self):
        """Repaint at 30 Hz during pursuit blocks only.

        The protocol runner ticks at 10 Hz, and a dot that jumps in 10 Hz steps
        is tracked by saccades rather than smooth pursuit — which would defeat
        the point, since the block exists to capture a clean slow eye sweep and
        not a train of spikes. The timer runs only while a pursuit phase is on
        screen, so it costs nothing the rest of the time.
        """
        if self._pursuit():
            if not self._pursuit_timer.isActive():
                self._pursuit_timer.start(33)
        elif self._pursuit_timer.isActive():
            self._pursuit_timer.stop()

    def show_phase(self, phase, index, total, waiting):
        self._phase, self._index, self._total = phase, index, total
        self._waiting = waiting
        self._remaining = None if waiting else (phase.duration or 0.0)
        self._duration = phase.duration or 0.0
        self._sync_pursuit_timer()
        self.update()

    def set_countdown(self, remaining):
        self._tick_at = time.monotonic()
        self._remaining = remaining
        self.update()

    def gaze_mode(self):
        return self._phase.gaze_mode() if self._phase else ""

    # --- input ------------------------------------------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.exited.emit()
        elif event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            if self._waiting:
                self.continue_pressed.emit()
        else:
            super().keyPressEvent(event)

    # --- painting ---------------------------------------------------------
    def paintEvent(self, _event):
        mode = self.gaze_mode()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(_BG.get(mode, _BG[""])))
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2

        if mode == "closed":
            # Eyes are shut: show almost nothing so the screen doesn't light
            # the retina and suppress the alpha we're trying to measure.
            self._paint_dim(p, w, h)
            p.end()
            return

        if mode in ("fixate", "pursuit"):
            # A moving target during a *pursuit* block; a static cross otherwise.
            if self._pursuit():
                self._paint_pursuit_target(p, w, h)
            else:
                self._paint_fixation(p, cx, cy)
                self._paint_countdown_ring(p, cx, cy)

        self._paint_text(p, w, h, mode)
        p.end()

    def _paint_dim(self, p, w, h):
        if self._phase is None:
            return
        p.setPen(QPen(QColor(_DIM_TEXT)))
        f = QFont()
        f.setPointSize(13)
        p.setFont(f)
        label = self._phase.name
        if self._remaining is not None:
            label += f"   ·   {_fmt(self._remaining)}"
        p.drawText(0, h - 60, w, 30, Qt.AlignHCenter, label)
        p.drawText(0, h - 34, w, 24, Qt.AlignHCenter,
                   "eyes closed   ·   Esc ends the session")

    def _pursuit(self):
        """Is this a smooth-pursuit block?"""
        return bool(self._phase is not None and self._phase.gaze == "pursuit")

    def _paint_pursuit_target(self, p, w, h):
        """A dot sweeping horizontally for the eye-movement artifact block.

        Deliberately NOT used for the eyes-open baseline. AF7 and AF8 sit
        directly over the eyes, so moving the gaze drags the corneo-retinal
        dipole across them and injects large slow deflections into the two
        electrodes the whole control law depends on. The baseline has to be the
        cleanest eyes-open reference available; contaminating it would bias
        every z-score computed against it afterwards.

        As a *labelled artifact* block it is exactly right, though — eye
        movement is the contaminant most likely to masquerade as frontal signal,
        and it was the one the artifact section was missing.
        """
        # Interpolate from the wall clock between runner ticks; using the 10 Hz
        # countdown alone would quantise the sweep into visible steps.
        t = 0.0
        if self._remaining is not None and self._duration:
            t = self._duration - self._remaining
        t += max(0.0, time.monotonic() - self._tick_at)
        phase = 2 * math.pi * (t / PURSUIT_PERIOD_S)
        x = w / 2.0 + math.sin(phase) * (w * PURSUIT_SPAN / 2.0)
        y = h / 2.0
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(_RING_FG))
        p.drawEllipse(QPointF(x, y), 14, 14)
        p.setBrush(QColor("#0B0E12"))
        p.drawEllipse(QPointF(x, y), 4, 4)

    def _paint_countdown_ring(self, p, cx, cy):
        """Ring around the cross that drains as the block runs.

        Staring at a static cross with no idea how long is left is what makes a
        45-second block feel like two minutes. Showing the remaining time as a
        shrinking arc costs no extra eye movement — it is concentric with the
        thing they are already fixating — and turns an open-ended wait into a
        visibly finite one.
        """
        if self._remaining is None or not self._duration:
            return
        frac = max(0.0, min(1.0, self._remaining / float(self._duration)))
        r = 54
        rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(_RING_BG), 3))
        p.drawEllipse(rect)
        p.setPen(QPen(QColor(_RING_FG), 3, Qt.SolidLine, Qt.RoundCap))
        # Qt angles are 1/16 degree, counter-clockwise from 3 o'clock.
        p.drawArc(rect, 90 * 16, -int(360 * 16 * frac))

    def _paint_fixation(self, p, cx, cy):
        """Standard fixation cross with a centre dot."""
        arm = 18
        p.setPen(QPen(QColor("#D8E2EC"), 3, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(int(cx - arm), int(cy), int(cx + arm), int(cy))
        p.drawLine(int(cx), int(cy - arm), int(cx), int(cy + arm))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(theme.ACCENT))
        p.drawEllipse(int(cx - 3), int(cy - 3), 6, 6)

    def _paint_text(self, p, w, h, mode):
        if self._phase is None:
            return
        # Phase label, top.
        p.setPen(QPen(QColor(theme.TEXT_DIM)))
        f = QFont()
        f.setPointSize(14)
        f.setBold(True)
        p.setFont(f)
        header = f"Phase {self._index + 1}/{self._total} · {self._phase.name}"
        p.drawText(0, 48, w, 32, Qt.AlignHCenter, header)

        # Countdown, top-right — big enough to read at a glance.
        if self._remaining is not None:
            f2 = QFont()
            f2.setPointSize(26)
            f2.setBold(True)
            p.setFont(f2)
            p.setPen(QPen(QColor(theme.TEXT)))
            p.drawText(w - 200, 40, 160, 44, Qt.AlignRight, _fmt(self._remaining))

        # Instruction. During a fixate block keep it well clear of the cross so
        # it doesn't pull the eyes off target.
        f3 = QFont()
        f3.setPointSize(16 if mode in ("fixate", "pursuit") else 20)
        p.setFont(f3)
        p.setPen(QPen(QColor(theme.TEXT_DIM if mode in ("fixate", "pursuit")
                             else theme.TEXT)))
        text = self._phase.instruction.replace("\n\n", "\n")
        if mode in ("fixate", "pursuit"):
            p.drawText(int(w * 0.15), int(h * 0.72), int(w * 0.7), 120,
                       Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, text)
        else:
            p.drawText(int(w * 0.15), int(h * 0.35), int(w * 0.7), int(h * 0.4),
                       Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, text)

        # Progress bar across the bottom.
        if self._remaining is not None and self._duration:
            done = 1.0 - max(0.0, min(1.0, self._remaining / self._duration))
            bar_w, bar_h = int(w * 0.6), 4
            x, y = int((w - bar_w) / 2), h - 80
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.BORDER))
            p.drawRect(x, y, bar_w, bar_h)
            p.setBrush(QColor(theme.ACCENT))
            p.drawRect(x, y, int(bar_w * done), bar_h)

        # Footer hints.
        p.setPen(QPen(QColor(theme.TEXT_FAINT)))
        f4 = QFont()
        f4.setPointSize(11)
        p.setFont(f4)
        hint = ("press Space to continue   ·   Esc to leave full screen"
                if self._waiting else "Esc ends the session")
        p.drawText(0, h - 40, w, 24, Qt.AlignHCenter, hint)


def _fmt(seconds):
    seconds = max(0, int(seconds + 0.999))     # ceil, so it ticks evenly
    return f"{seconds // 60:d}:{seconds % 60:02d}"
