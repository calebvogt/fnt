"""Mindball board: a ball on a rail between two ends.

Drawn with QPainter, like the flight view, so MuseStudio still needs no OpenGL
and stays independent of the other FNT tools.

FIRST PERSON, across the table
------------------------------
Drawn from the player's own seat: a table receding to a vanishing point, the
opponent seated at the far end, and the ball on a rail between you. The overhead
side-on view that came first was legible but read as a diagram — you were
looking AT the game rather than sitting in it, which is most of what made the
original compelling to play.

Depth cues do the work: the table narrows with distance, the ball shrinks as it
rolls away from you, and the opponent sits at the horizon. A ball moving toward
the vanishing point is unmistakably going into their end without needing a label.

The two calm bars stay as prominent as the ball. Seeing WHY it is moving is the
point of the game — the original's appeal was spectators watching someone lose
by trying too hard, and a bare ball hides that mechanism.
"""

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PyQt5.QtWidgets import QWidget

from fnt.musestudio import theme

ME = "#16C79A"
THEM = "#FFB020"


def _c(col, a=255):
    q = QColor(col)
    q.setAlpha(a)
    return q


class MindballView(QWidget):
    """First-person Mindball table."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 300)
        self._state = None
        self._status = ""
        self._opponent = "opponent"
        self._history = []

    def set_frame(self, state, opponent_label="", history=None):
        self._state = state
        if opponent_label:
            self._opponent = opponent_label
        if history is not None:
            self._history = history
        self.update()

    def set_status(self, text):
        self._status = text or ""
        self.update()

    # ------------------------------------------------------------ geometry
    def _table(self, w, h):
        """Near and far edges of the table, in screen coords."""
        horizon = h * 0.34
        near_y = h * 0.80
        near_half = w * 0.42
        far_half = w * 0.10
        return horizon, near_y, near_half, far_half

    def _rail_point(self, frac, w, h):
        """frac 0 = my edge, 1 = opponent's edge. Returns (x, y, scale)."""
        horizon, near_y, near_half, far_half = self._table(w, h)
        # Perspective foreshortening: equal steps along the rail cover less
        # screen distance as they recede, which is what sells the depth.
        t = frac ** 0.62
        y = near_y + (horizon - near_y) * t
        half = near_half + (far_half - near_half) * t
        scale = half / max(near_half, 1e-6)
        return w / 2.0, y, scale

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        s = self._state
        horizon, near_y, near_half, far_half = self._table(w, h)

        # Room behind the table
        sky = QLinearGradient(0, 0, 0, horizon)
        sky.setColorAt(0.0, _c("#070A0E"))
        sky.setColorAt(1.0, _c("#121A22"))
        p.fillRect(QRectF(0, 0, w, horizon), QBrush(sky))
        p.fillRect(QRectF(0, horizon, w, h - horizon), _c("#0B0E12"))

        # Table surface
        path = QPainterPath()
        path.moveTo(w / 2 - near_half, near_y)
        path.lineTo(w / 2 + near_half, near_y)
        path.lineTo(w / 2 + far_half, horizon)
        path.lineTo(w / 2 - far_half, horizon)
        path.closeSubpath()
        grad = QLinearGradient(0, horizon, 0, near_y)
        grad.setColorAt(0.0, _c("#16222C"))
        grad.setColorAt(1.0, _c("#0D141A"))
        p.setPen(QPen(_c(theme.BORDER_HI), 1.5))
        p.setBrush(QBrush(grad))
        p.drawPath(path)

        # End zones: mine near, theirs far
        for frac, col in ((0.06, ME), (0.94, THEM)):
            _, y, sc = self._rail_point(frac, w, h)
            half = near_half + (far_half - near_half) * (frac ** 0.62)
            p.setPen(Qt.NoPen)
            p.setBrush(_c(col, 55))
            p.drawRect(QRectF(w / 2 - half, y - 10 * sc, half * 2, 20 * sc))

        # Rail
        p.setPen(QPen(_c(theme.BORDER_HI), 2))
        p.drawLine(QPointF(w / 2, near_y), QPointF(w / 2, horizon))

        # Opponent, seated at the far edge
        oy = horizon - 6
        p.setPen(Qt.NoPen)
        p.setBrush(_c(THEM, 150))
        p.drawEllipse(QPointF(w / 2, oy - 26), 15, 15)                 # head
        shoulders = QPainterPath()
        shoulders.moveTo(w / 2 - 34, oy + 6)
        shoulders.quadTo(w / 2, oy - 20, w / 2 + 34, oy + 6)
        p.setBrush(_c(THEM, 90))
        p.drawPath(shoulders)
        f = QFont(); f.setPointSize(9); f.setBold(True); p.setFont(f)
        p.setPen(_c(THEM, 210))
        p.drawText(QRectF(0, horizon - 62, w, 16), Qt.AlignCenter,
                   self._opponent.upper())

        if s is None:
            p.setPen(_c(theme.TEXT_FAINT))
            p.drawText(self.rect(), Qt.AlignCenter, "waiting for signal…")
            p.end()
            return

        # Ball: position -1 (my end, near) .. +1 (their end, far)
        frac = (s.position + 1.0) / 2.0
        bx, by, sc = self._rail_point(frac, w, h)
        lead = s.my_calm - s.their_calm
        col = ME if lead > 0.02 else (THEM if lead < -0.02 else theme.TEXT_DIM)
        r = max(5.0, 26.0 * sc)
        p.setPen(Qt.NoPen)
        p.setBrush(_c("#000000", 110))
        p.drawEllipse(QPointF(bx, by + r * 0.55), r * 1.05, r * 0.34)   # shadow
        p.setBrush(_c(col, 70))
        p.drawEllipse(QPointF(bx, by), r * 1.5, r * 1.5)                # glow
        p.setBrush(_c(col))
        p.drawEllipse(QPointF(bx, by), r, r)

        self._paint_traces(p, w, h)
        self._paint_hud(p, w, h, s, lead)
        p.end()

    def _paint_traces(self, p, w, h):
        """Both players' calm over time, side by side.

        The bars say who is calmer now; the traces say who has been holding it,
        which is what actually decides the match. Watching your own line drop
        the moment you start trying is the whole lesson of the game, and a
        single instantaneous number cannot show that.
        """
        hist = self._history
        if len(hist) < 4:
            return
        gh = 46
        gy = h - 96
        gx0, gx1 = 114, w - 62
        p.setPen(Qt.NoPen)
        p.setBrush(_c("#0A0D12", 190))
        p.drawRect(QRectF(gx0, gy, gx1 - gx0, gh))
        p.setPen(QPen(_c(theme.BORDER), 1))
        p.drawRect(QRectF(gx0, gy, gx1 - gx0, gh))

        span = max(1.0, hist[-1][0] - hist[0][0])
        for idx, col in ((1, ME), (2, THEM)):
            path = QPainterPath()
            started = False
            for row in hist:
                x = gx0 + (gx1 - gx0) * ((row[0] - hist[0][0]) / span)
                y = gy + gh - gh * max(0.0, min(1.0, row[idx]))
                if not started:
                    path.moveTo(x, y); started = True
                else:
                    path.lineTo(x, y)
            p.setPen(QPen(_c(col, 220), 1.6))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
        f = QFont(); f.setPointSize(8); p.setFont(f)
        p.setPen(_c(theme.TEXT_FAINT))
        p.drawText(QRectF(gx0 - 104, gy + gh / 2 - 8, 98, 16), Qt.AlignRight,
                   "calm over time")
        p.drawText(QRectF(gx1 + 4, gy - 2, 56, 14), Qt.AlignLeft, "1.0")
        p.drawText(QRectF(gx1 + 4, gy + gh - 12, 56, 14), Qt.AlignLeft, "0.0")

    def _paint_hud(self, p, w, h, s, lead):
        f = QFont(); f.setPointSize(9); p.setFont(f)
        for i, (lab, val, col2) in enumerate((
                ("you", s.my_calm, ME), (self._opponent, s.their_calm, THEM))):
            by = h - 40 + i * 18
            p.setPen(_c(theme.TEXT_FAINT))
            p.drawText(QRectF(8, by - 2, 100, 16), Qt.AlignRight, f"{lab[:14]} calm")
            p.setPen(Qt.NoPen)
            p.setBrush(_c(theme.SURFACE_HI))
            p.drawRect(QRectF(114, by, w - 176, 11))
            p.setBrush(_c(col2))
            p.drawRect(QRectF(114, by, (w - 176) * max(0.0, min(1.0, val)), 11))
            p.setPen(_c(theme.TEXT_DIM))
            p.drawText(QRectF(w - 56, by - 3, 48, 16), Qt.AlignLeft, f"{val:.2f}")

        big = QFont(); big.setPointSize(15); big.setBold(True); p.setFont(big)
        if s.winner == "player":
            p.setPen(_c(ME)); msg = "YOU WIN"
        elif s.winner == "opponent":
            p.setPen(_c(THEM)); msg = f"{self._opponent.upper()} WINS"
        elif s.winner == "draw":
            p.setPen(_c(theme.TEXT_DIM)); msg = "DRAW"
        else:
            p.setPen(_c(theme.TEXT_DIM))
            msg = ("you are calmer" if lead > 0.02
                   else "they are calmer" if lead < -0.02 else "level")
        p.drawText(QRectF(0, 14, w, 26), Qt.AlignCenter, msg)

        if s.wraps:
            p.setPen(_c(theme.WARN))
            f2 = QFont(); f2.setPointSize(8); p.setFont(f2)
            p.drawText(QRectF(0, 56, w, 14), Qt.AlignCenter,
                       f"ghost recording looped {s.wraps}x — shorter than this match")
        if self._status:
            p.setPen(_c(theme.TEXT_FAINT))
            f2 = QFont(); f2.setPointSize(9); p.setFont(f2)
            p.drawText(QRectF(0, 42, w, 14), Qt.AlignCenter, self._status)
