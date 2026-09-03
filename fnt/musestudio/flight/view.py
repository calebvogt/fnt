"""First-person flight view, drawn with QPainter. No OpenGL.

WHY NOT OPENGL
--------------
FNT's tools are wholly independent of one another, and the pyqtgraph
compiled-shader cache is process-global: a second ``GLViewWidget`` in the same
process reuses program ids compiled in a different GL context and every draw
fails with GLError 1281. The workaround for that lives inside ABMA, so using GL
here would have meant importing another tool. Benchmarking also put a QPainter
2.5D scene *ahead* of GLViewWidget for exactly this kind of view, and it drops
the PyOpenGL requirement entirely, so nothing was traded away.

WHO IS THIS FOR
---------------
Not the pilot. Flight Mode is flown with the eyes closed -- that is where the
alpha the craft runs on comes from -- so this view exists for an experimenter
watching over their shoulder, and later for the pilot reviewing the flight.
That inverts the usual priorities: legibility from across a room matters more
than immersion, and the instrument panel matters more than the scenery.

The single most important element is the control-signal strip along the bottom.
A BCI display that shows only the outcome teaches the pilot nothing about why it
moved; showing z, the dead zone and the veto state side by side is what lets
someone connect a mental state to an outcome.

WHAT IS DRAWN
-------------
A ground plane in perspective, receding to a horizon, with two cues that make
speed and altitude readable at a glance:

* **Aperiodic near-field texture.** A regular grid alone is a poor speed cue --
  every row looks like the last, so motion reads as a shimmer rather than travel.
  Scattered ground features that do not repeat give the eye something to track.
* **A horizon that rises and falls with altitude**, plus a numeric readout,
  because judging height from a perspective grid alone is unreliable.
"""

import math

from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen,
)
from PyQt5.QtWidgets import QWidget

from fnt.musestudio import theme
from fnt.musestudio.flight.sim import FlightPhase

# World scale. These three are coupled and were set by looking at rendered
# frames, not by taste: a ground grid projects sensibly only when the line
# spacing is comparable to the camera height. The first attempt paired a 2-unit
# camera with 20-unit spacing, and the whole grid collapsed into a 20-pixel band
# under the horizon with an empty screen below it. The rule of thumb that fixed
# it is that the nearest line wants to land near the bottom of the frame, i.e.
# z_near ~ 2 * CAM_HEIGHT * FOV_SCALE.
HORIZON_FRAC = 0.44
FOV_SCALE = 0.85
CAM_HEIGHT = 6.0             # eye height above the craft, world units
GRID_SPACING = 10.0
GRID_LINES = 40
FEATURE_COUNT = 110

# Altitude is deliberately compressed for the *view* only. Geometrically honest
# altitude makes the ground vanish almost immediately -- at the 120-unit ceiling
# a true projection leaves nothing on screen but a dark band, which tells an
# observer nothing. The HUD carries the exact number, so the picture can afford
# to be a legible impression of height while the readout stays literal.
ALT_VISUAL_SCALE = 0.15
RAIL_X = 12.0


def _c(hex_or_color, alpha=255):
    col = QColor(hex_or_color)
    col.setAlpha(alpha)
    return col


class FlightView(QWidget):
    """Renders craft state. Purely a view -- it owns no simulation state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 240)
        self.setAutoFillBackground(False)
        self._state = None          # CraftState
        self._trace = None          # PipelineTrace
        self._dead_zone = 0.8
        self._full_scale = 3.0
        self._status = ""
        # Deterministic scatter: a fixed pseudo-random field, so the ground looks
        # irregular but is identical every run. Reproducibility matters here --
        # a replayed flight must render the same world as the live one did.
        self._features = self._make_features()

    @staticmethod
    def _make_features():
        feats = []
        seed = 12345
        for i in range(FEATURE_COUNT):
            seed = (1103515245 * seed + 12345) % (2 ** 31)
            x = (seed / (2 ** 31) - 0.5) * 120.0
            seed = (1103515245 * seed + 12345) % (2 ** 31)
            z = (seed / (2 ** 31)) * (GRID_SPACING * GRID_LINES)
            seed = (1103515245 * seed + 12345) % (2 ** 31)
            size = 0.5 + (seed / (2 ** 31)) * 1.4
            feats.append((x, z, size))
        return feats

    # -------------------------------------------------------------- updating
    def set_frame(self, state, trace=None, dead_zone=None, full_scale=None):
        self._state = state
        self._trace = trace
        if dead_zone is not None:
            self._dead_zone = dead_zone
        if full_scale is not None:
            self._full_scale = full_scale
        self.update()

    def set_status(self, text):
        self._status = text or ""
        self.update()

    # ------------------------------------------------------------ projection
    def _project(self, x, z, y, w, h, horizon_y, heading=0.0):
        """World (x right, z forward, y up) -> screen. None when behind the eye.

        ``heading`` rotates the world about the eye, which is what makes head-tilt
        steering visible: the craft stays centred and the ground swings past it.
        """
        if heading:
            ch, sh = math.cos(heading), math.sin(heading)
            x, z = x * ch - z * sh, x * sh + z * ch
        if z <= 0.5:
            return None
        f = (h * FOV_SCALE) / z
        return QPointF(w / 2.0 + x * f, horizon_y + (y * f))

    # --------------------------------------------------------------- painting
    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        st = self._state

        altitude = st.altitude if st else 0.0
        distance = st.distance if st else 0.0
        # The horizon is FIXED at eye level. A camera looking level does not see
        # the horizon move when it climbs -- it sees more ground. Sliding the
        # horizon with altitude (the first attempt here) reads as the craft
        # pitching, which it never does.
        horizon_y = h * HORIZON_FRAC

        self._paint_sky(p, w, h, horizon_y)
        self._paint_ground(p, w, h, horizon_y, distance, altitude)
        self._paint_craft(p, w, h, horizon_y)
        self._paint_hud(p, w, h)
        p.end()

    def _paint_sky(self, p, w, h, horizon_y):
        grad = QLinearGradient(0, 0, 0, max(horizon_y, 1))
        grad.setColorAt(0.0, _c("#070A0E"))
        grad.setColorAt(1.0, _c("#16222E"))
        p.fillRect(QRectF(0, 0, w, max(horizon_y, 0)), QBrush(grad))

    def _paint_ground(self, p, w, h, horizon_y, distance, altitude):
        if horizon_y < h:
            grad = QLinearGradient(0, horizon_y, 0, h)
            grad.setColorAt(0.0, _c("#0C1A16"))
            grad.setColorAt(1.0, _c("#071009"))
            p.fillRect(QRectF(0, horizon_y, w, h - horizon_y), QBrush(grad))

        # Horizon line: without it the sky/ground boundary is a soft gradient
        # edge and the eye has nothing to level against.
        p.setPen(QPen(_c(theme.ACCENT_DIM, 120), 1.2))
        p.drawLine(QPointF(0, horizon_y), QPointF(w, horizon_y))

        p.save()
        p.setClipRect(QRectF(0, max(horizon_y, 0), w, h))
        eye = CAM_HEIGHT + max(altitude, 0.0) * ALT_VISUAL_SCALE
        hdg = -(self._state.heading if self._state else 0.0)
        offset = distance % GRID_SPACING

        # Transverse lines: these carry the sense of forward motion.
        for i in range(GRID_LINES):
            z = i * GRID_SPACING - offset + 4.0
            if z <= 0.5:
                continue
            a = self._depth_alpha(z)
            if a <= 3:
                continue
            left = self._project(-90.0, z, eye, w, h, horizon_y, hdg)
            right = self._project(90.0, z, eye, w, h, horizon_y, hdg)
            if left is None or right is None:
                continue
            p.setPen(QPen(_c(theme.GRID, a), 1.0))
            p.drawLine(left, right)

        # Rails: two longitudinal lines give a heading reference.
        for x in (-RAIL_X, RAIL_X):
            path = QPainterPath()
            started = False
            for i in range(GRID_LINES):
                z = i * GRID_SPACING - offset + 4.0
                pt = self._project(x, z, eye, w, h, horizon_y, hdg)
                if pt is None:
                    continue
                if not started:
                    path.moveTo(pt)
                    started = True
                else:
                    path.lineTo(pt)
            if started:
                p.setPen(QPen(_c(theme.ACCENT_DIM, 150), 1.4))
                p.drawPath(path)

        # Aperiodic ground features -- the actual speed cue.
        span = GRID_SPACING * GRID_LINES
        for fx, fz, fsize in self._features:
            z = (fz - distance) % span
            if z <= 1.0:
                continue
            a = self._depth_alpha(z)
            if a <= 4:
                continue
            pt = self._project(fx, z, eye, w, h, horizon_y, hdg)
            if pt is None:
                continue
            r = max(1.0, fsize * (h * FOV_SCALE) / z * 0.5)
            p.setPen(Qt.NoPen)
            p.setBrush(_c("#1E4C3C", a))
            p.drawEllipse(pt, r, r * 0.45)
        p.restore()

    @staticmethod
    def _depth_alpha(z):
        """Atmospheric fade. pyqtgraph has no fog; this is the cheap equivalent."""
        far = GRID_SPACING * GRID_LINES
        return int(max(0.0, 1.0 - (z / far) ** 0.8) * 190)

    def _paint_craft(self, p, w, h, horizon_y):
        """The craft's nose, seen from inside it.

        A black equilateral triangle with the tip pointing away toward the
        horizon. Drawn low and centred so it reads as the hull ahead of the
        pilot rather than a cursor: it sits in the same place every frame while
        the world moves past it, which is what sells a first-person view.

        It banks slightly with vertical speed — a few degrees of pitch is enough
        to make climbing and sinking legible from the craft alone, without
        reading the altimeter.
        """
        strip_h = 56
        cx = w / 2.0
        base_y = h - strip_h - 10
        # Roughly 2.5x the first pass — the craft reads as a vehicle you are
        # sitting in rather than a cursor, which is the whole point of a
        # first-person view.
        size = max(140.0, min(w, h) * 0.28)
        halfw = size * 0.5

        # Pitch: nose lifts when climbing, drops when sinking. Clamped so a
        # burst of thrust never flips it into something unreadable.
        vs = self._state.vertical_speed if self._state else 0.0
        pitch = max(-1.0, min(1.0, vs / 6.0)) * (size * 0.16)
        # Bank into the turn — the clearest read that steering is responding.
        bank = max(-1.0, min(1.0, (self._state.turn_rate if self._state else 0.0) / 1.0))
        p.save()
        p.translate(cx, base_y)
        p.rotate(bank * 22.0)
        p.translate(-cx, -base_y)

        tip = QPointF(cx, base_y - size * 0.72 - pitch)
        left = QPointF(cx - halfw, base_y + size * 0.14 + pitch * 0.4)
        right = QPointF(cx + halfw, base_y + size * 0.14 + pitch * 0.4)

        path = QPainterPath()
        path.moveTo(tip)
        path.lineTo(right)
        path.lineTo(left)
        path.closeSubpath()

        # Soft shadow so the silhouette separates from the dark ground.
        p.setPen(Qt.NoPen)
        p.setBrush(_c("#000000", 120))
        p.drawEllipse(QPointF(cx, base_y + size * 0.22), halfw * 0.9, size * 0.10)

        p.setBrush(_c("#05070A"))
        p.setPen(QPen(_c(theme.ACCENT, 190), 1.6))
        p.drawPath(path)

        # Centre spine — gives the flat black shape some read of orientation.
        p.setPen(QPen(_c(theme.ACCENT_DIM, 150), 1.2))
        p.drawLine(tip, QPointF(cx, base_y + size * 0.14 + pitch * 0.4))
        p.restore()

    # ------------------------------------------------------------------- HUD
    def _paint_hud(self, p, w, h):
        st, tr = self._state, self._trace
        small = QFont(); small.setPointSize(8)
        big = QFont(); big.setPointSize(20); big.setBold(True)

        # --- altitude + phase, top-left
        p.setFont(big)
        p.setPen(_c(theme.TEXT))
        alt = st.altitude if st else 0.0
        p.drawText(QRectF(14, 10, 200, 30), Qt.AlignLeft | Qt.AlignVCenter,
                   f"{alt:6.1f}")
        p.setFont(small)
        p.setPen(_c(theme.TEXT_FAINT))
        p.drawText(QRectF(14, 38, 200, 16), Qt.AlignLeft, "ALTITUDE")

        phase = st.phase if st else FlightPhase.GROUNDED
        colour = {FlightPhase.GROUNDED: theme.TEXT_FAINT,
                  FlightPhase.ARMING: theme.WARN,
                  FlightPhase.AIRBORNE: theme.GOOD,
                  FlightPhase.LANDED: theme.ACCENT}.get(phase, theme.TEXT_DIM)
        p.setPen(_c(colour))
        p.drawText(QRectF(w - 214, 12, 200, 18), Qt.AlignRight,
                   phase.value.upper())
        if st:
            p.setPen(_c(theme.TEXT_FAINT))
            p.drawText(QRectF(w - 214, 30, 200, 16), Qt.AlignRight,
                       f"{st.distance:.0f} u   v {st.vertical_speed:+.1f}")

        # --- arming progress: the pilot's first success, so it gets real estate
        if phase is FlightPhase.ARMING and st:
            bw = min(260, w - 40)
            x0 = (w - bw) / 2.0
            y0 = h * 0.30
            p.setPen(QPen(_c(theme.BORDER_HI), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x0, y0, bw, 8))
            p.setPen(Qt.NoPen)
            p.setBrush(_c(theme.WARN))
            p.drawRect(QRectF(x0 + 1, y0 + 1, (bw - 2) * st.arm_progress, 6))
            p.setPen(_c(theme.WARN))
            p.setFont(small)
            p.drawText(QRectF(x0, y0 - 18, bw, 16), Qt.AlignCenter,
                       "HOLD FOR LIFT-OFF")

        self._paint_signal_strip(p, w, h, tr)

        if self._status:
            p.setFont(small)
            p.setPen(_c(theme.TEXT_DIM))
            p.drawText(QRectF(14, h - 78, w - 28, 16), Qt.AlignLeft, self._status)

    def _paint_signal_strip(self, p, w, h, tr):
        """The pilot's own control signal, shown honestly.

        This strip is the difference between a game and an instrument. It shows
        z against the dead zone and the resulting thrust, and it says plainly
        when a tick was vetoed -- so an experimenter can tell "they relaxed and
        it climbed" from "they clenched and it froze", which are the two things
        most easily confused from the outside.
        """
        small = QFont(); small.setPointSize(8)
        strip_h = 56
        y0 = h - strip_h
        p.fillRect(QRectF(0, y0, w, strip_h), _c("#0A0D12", 220))
        p.setPen(QPen(_c(theme.BORDER), 1))
        p.drawLine(QPointF(0, y0), QPointF(w, y0))

        if tr is None:
            p.setFont(small)
            p.setPen(_c(theme.TEXT_FAINT))
            p.drawText(QRectF(14, y0 + 18, 300, 16), Qt.AlignLeft,
                       "no signal — headband not streaming")
            return

        pad = 14
        bar_x = pad
        bar_w = w - pad * 2 - 150
        bar_y = y0 + 24
        lo, hi = -1.0, max(self._full_scale, self._dead_zone + 1.0)

        def zx(z):
            return bar_x + bar_w * (min(max(z, lo), hi) - lo) / (hi - lo)

        # scale + dead-zone band
        p.setPen(Qt.NoPen)
        p.setBrush(_c(theme.SURFACE_HI, 200))
        p.drawRect(QRectF(bar_x, bar_y, bar_w, 10))
        p.setBrush(_c(theme.BORDER_HI, 160))
        p.drawRect(QRectF(zx(lo), bar_y, zx(self._dead_zone) - zx(lo), 10))

        # the live value
        vetoed = tr.vetoed
        val_col = theme.DANGER if vetoed else (
            theme.GOOD if tr.z > self._dead_zone else theme.ACCENT)
        x = zx(tr.z)
        p.setBrush(_c(val_col))
        p.drawRect(QRectF(min(x, zx(0.0)) if tr.z < 0 else zx(max(lo, 0.0)),
                          bar_y, max(2.0, abs(x - zx(max(lo, 0.0)))), 10))
        p.setPen(QPen(_c(theme.TEXT), 1.5))
        p.drawLine(QPointF(x, bar_y - 3), QPointF(x, bar_y + 13))

        p.setFont(small)
        p.setPen(_c(theme.TEXT_FAINT))
        p.drawText(QRectF(bar_x, y0 + 6, max(bar_w, 220), 14), Qt.AlignLeft,
                   "CONTROL SIGNAL  (z vs your own baseline)")
        p.drawText(QRectF(zx(self._dead_zone) - 30, bar_y + 14, 60, 14),
                   Qt.AlignCenter, "dead zone")

        # numeric block on the right
        rx = w - 150
        p.setPen(_c(theme.TEXT))
        p.drawText(QRectF(rx, y0 + 6, 140, 14), Qt.AlignLeft,
                   f"z {tr.z:+.2f}   thrust {tr.thrust:+.2f}")
        if vetoed:
            p.setPen(_c(theme.DANGER))
            reason = (tr.veto_reason or "artifact")[:34]
            p.drawText(QRectF(rx, y0 + 22, 140, 14), Qt.AlignLeft, "VETOED")
            p.setPen(_c(theme.TEXT_FAINT))
            p.drawText(QRectF(rx, y0 + 36, 140, 14), Qt.AlignLeft, reason)
        else:
            p.setPen(_c(theme.TEXT_FAINT))
            ok = tr.n_accepted
            names = [c for c, t in tr.channels.items() if t.accepted]
            p.drawText(QRectF(rx, y0 + 22, 140, 14), Qt.AlignLeft,
                       f"{ok} electrode{'s' if ok != 1 else ''} clean")
            p.drawText(QRectF(rx, y0 + 36, 140, 14), Qt.AlignLeft,
                       ", ".join(names)[:22])
