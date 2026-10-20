"""The FED3 drawn as it looks, with its counts painted on the pokes and well.

A count flashes when it goes up, so a poke or a pellet is visible from across
the room. The drawing dims while the device isn't connected.
"""
import os
import sys

from PyQt5.QtCore import QRect, QRectF, QSize, Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter
from PyQt5.QtWidgets import QSizePolicy, QWidget

try:
    from PyQt5.QtSvg import QSvgRenderer
except ImportError:          # the counts still draw without the artwork
    QSvgRenderer = None

ASPECT = 163.67577 / 116.04688      # the SVG's viewBox
FLASH_MS = 250
FLASH = {"left": QColor(76, 175, 80, 200), "right": QColor(76, 175, 80, 200),
         "pellets": QColor(0, 120, 212, 220)}
# Where the pokes and the well are, as fractions of the drawing.
POKE_RADIUS = 0.0953
LEFT_POKE = (0.2501, 0.6444)
RIGHT_POKE = (0.7488, 0.6426)
WELL = (0.4108, 0.6095, 0.1772, 0.1677)


def _svg_path():
    base = getattr(sys, "_MEIPASS", None)      # a PyInstaller bundle
    if base:
        return os.path.join(base, "fnt", "fed3", "ui", "fed3.svg")
    return os.path.join(os.path.dirname(__file__), "fed3.svg")


class FedGraphic(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.renderer = QSvgRenderer(_svg_path()) if QSvgRenderer else None
        self.counts = None              # None: nothing known yet
        self.connected = False
        self.flashing = {}
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setFixedHeight(118)
        self.setMinimumWidth(150)

    def set_state(self, counts, connected):
        """``counts``: {"left", "right", "pellets"} as ints, or None."""
        if counts and self.counts:
            for key, value in counts.items():
                if value > self.counts.get(key, value):
                    self._flash(key)
        changed = counts != self.counts or connected != self.connected
        self.counts, self.connected = counts, connected
        if changed:
            self.update()

    def _flash(self, key):
        self.flashing[key] = FLASH[key]
        QTimer.singleShot(FLASH_MS, lambda: (self.flashing.pop(key, None), self.update()))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if not self.connected:
            painter.setOpacity(0.45)
        width, height = self.width(), self.height()
        if width / max(height, 1) > ASPECT:
            draw_h, draw_w = height, int(height * ASPECT)
        else:
            draw_w, draw_h = width, int(width / ASPECT)
        x0, y0 = (width - draw_w) // 2, (height - draw_h) // 2
        if self.renderer is not None and self.renderer.isValid():
            self.renderer.render(painter, QRectF(x0, y0, draw_w, draw_h))

        radius = int(draw_w * POKE_RADIUS)
        font = QFont()
        font.setBold(True)
        font.setPixelSize(max(9, int(radius * 0.8)))
        painter.setFont(font)
        for key, (fx, fy) in (("left", LEFT_POKE), ("right", RIGHT_POKE)):
            rect = QRect(x0 + int(draw_w * fx) - radius, y0 + int(draw_h * fy) - radius,
                         radius * 2, radius * 2)
            self._badge(painter, rect, key, circle=True)
        fx, fy, fw, fh = WELL
        rect = QRect(x0 + int(draw_w * fx), y0 + int(draw_h * fy),
                     int(draw_w * fw), int(draw_h * fh))
        self._badge(painter, rect, "pellets", circle=False)
        painter.end()

    def _badge(self, painter, rect, key, circle):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.flashing.get(key, QColor(0, 0, 0, 170))))
        if circle:
            painter.drawEllipse(rect)
        else:
            painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QColor(255, 255, 255))
        text = "–" if self.counts is None else str(self.counts.get(key, 0))
        # Large counts shrink to fit rather than spill past the poke.
        font = painter.font()
        base = size = font.pixelSize()
        while size > 7 and painter.fontMetrics().horizontalAdvance(text) > rect.width() - 2:
            size -= 1
            font.setPixelSize(size)
            painter.setFont(font)
        painter.drawText(rect, Qt.AlignCenter, text)
        font.setPixelSize(base)
        painter.setFont(font)

    def sizeHint(self):
        return QSize(int(118 * ASPECT), 118)
