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

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from fnt.musestudio import theme

# Backgrounds per gaze mode. "closed" is deliberately near-black.
_BG = {"fixate": "#0B0E12", "closed": "#000000", "": "#0B0E12"}
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

    # --- state ------------------------------------------------------------
    def show_phase(self, phase, index, total, waiting):
        self._phase, self._index, self._total = phase, index, total
        self._waiting = waiting
        self._remaining = None if waiting else (phase.duration or 0.0)
        self._duration = phase.duration or 0.0
        self.update()

    def set_countdown(self, remaining):
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

        if mode == "fixate":
            self._paint_fixation(p, cx, cy)

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
                   "eyes closed   ·   Esc to leave full screen")

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
        f3.setPointSize(16 if mode == "fixate" else 20)
        p.setFont(f3)
        p.setPen(QPen(QColor(theme.TEXT if mode != "fixate" else theme.TEXT_DIM)))
        text = self._phase.instruction.replace("\n\n", "\n")
        if mode == "fixate":
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
                if self._waiting else "Esc to leave full screen")
        p.drawText(0, h - 40, w, 24, Qt.AlignHCenter, hint)


def _fmt(seconds):
    seconds = max(0, int(seconds + 0.999))     # ceil, so it ticks evenly
    return f"{seconds // 60:d}:{seconds % 60:02d}"
