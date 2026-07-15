"""pyqtgraph OpenGL 3D live view for ABMA runs.

A GPU-rendered, orbitable scene used during playback: a floor grid, optional 3D
walls (e.g. the 50x50x50 cm OFT box), resource markers, agents drawn as oriented
bodies (a box body + a sphere head) so heading is visible, fading trails,
day/night background, a selection highlight, and click-to-inspect picking.

Requires ``pyqtgraph.opengl`` (PyOpenGL). The GUI falls back to the 2D canvas if
unavailable, so importing this module is always guarded by the caller.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import Vector
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QVector4D, QFont

from ..core.config import ResourceObject

_MALE = (0.29, 0.56, 0.85, 1.0)
_FEMALE = (0.88, 0.33, 0.60, 1.0)
_DEAD = (0.40, 0.40, 0.40, 1.0)
_KIND_RGBA = {
    "nest": (0.71, 0.47, 0.23, 1.0),
    "food": (0.30, 0.69, 0.31, 1.0),
    "water": (0.23, 0.63, 0.81, 1.0),
}
_DAY_BG = (0.13, 0.15, 0.18, 1.0)
_NIGHT_BG = (0.04, 0.05, 0.07, 1.0)
_ZONE_RGBA = {
    "center": (0.95, 0.76, 0.31, 1.0),
    "periphery": (0.50, 0.53, 0.56, 1.0),
    "roi": (0.35, 0.66, 0.90, 1.0),
}
_WALL_RGBA = (0.60, 0.70, 0.82, 0.10)   # translucent acrylic-like walls

# mouse body geometry (metres): body box L×W×H, head sphere radius
_BODY_L, _BODY_W, _BODY_H = 0.09, 0.035, 0.03
_HEAD_R = 0.022
_HEAD_OFF = _BODY_L / 2 + _HEAD_R * 0.6


def _nice_step(extent, target=5):
    """A 'nice' round grid step (1/2/2.5/5 × 10^k) giving ~``target`` divisions."""
    import math
    if extent <= 0:
        return 1.0
    raw = extent / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def _slack_cable(nodes, sag_k=0.09, seg_pts=10):
    """Polyline through ``nodes`` ([x,y,z]) with a downward parabolic sag."""
    out = []
    for a, b in zip(nodes, nodes[1:]):
        span = math.hypot(b[0] - a[0], b[1] - a[1])
        sag = min(0.3, sag_k * span)
        for i in range(seg_pts + 1):
            t = i / seg_pts
            out.append([a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t,
                        a[2] + (b[2] - a[2]) * t - sag * 4 * t * (1 - t)])
    return np.array(out, float)


def _box_meshdata(sx, sy, sz):
    """MeshData for a box of the given size centred at the origin."""
    x, y, z = sx / 2.0, sy / 2.0, sz / 2.0
    verts = np.array([
        [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
        [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], dtype=float)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=int)
    return gl.MeshData(vertexes=verts, faces=faces)


class Arena3DView(gl.GLViewWidget):
    """3D playback view. Drop-in for the run canvas (subset of ArenaCanvas API)."""

    object_added = pyqtSignal(object)   # emitted when placing on the floor
    agent_picked = pyqtSignal(int)      # click: agent index, or -1 for empty
    agent_hovered = pyqtSignal(int)     # hover: agent index, or -1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setBackgroundColor(pg.mkColor(*[int(c * 255) for c in _NIGHT_BG]))
        self.arena = None
        self._selected = None
        self._last_pos = None
        self._history = deque(maxlen=10)
        self._press_xy = None
        self._n = 0
        self._arm_kind = None

        self._chamber_items = []
        self._chambers = [(0.0, 0.0)]
        self._grass_items = []
        self._grass_on = False
        self._antenna_items = []
        self._antenna_idx = -1          # -1 = off; else index into layouts
        self._compass_items = []
        self._measure_items = []
        self._measure_mode = None
        self._bodies = []
        self._heads = []
        self._body_md = _box_meshdata(_BODY_L, _BODY_W, _BODY_H)
        self._head_md = gl.MeshData.sphere(rows=8, cols=8, radius=_HEAD_R)

        self._resources = gl.GLScatterPlotItem()
        self._trail = gl.GLScatterPlotItem()
        self._sel = gl.GLScatterPlotItem()
        for it in (self._resources, self._trail, self._sel):
            it.setGLOptions("translucent")
            self.addItem(it)

    # ------------------------------------------------------------------ #
    def set_arena(self, arena, chambers=None):
        """Draw the arena; ``chambers`` is a list of (dx, dy) replicate offsets."""
        self.arena = arena
        self._chambers = chambers or [(0.0, 0.0)]
        w, h = arena.width, arena.height

        for it in self._chamber_items:
            self.removeItem(it)
        self._chamber_items = []
        for dx, dy in self._chambers:
            self._draw_chamber(arena, dx, dy)
        self._build_grass()
        self._build_antennas()
        self._build_compass()
        self._build_measure()

        # resource markers across all chambers (one scatter)
        rp, rc = [], []
        for dx, dy in self._chambers:
            for o in arena.objects:
                rp.append([dx + o.x, dy + o.y, 0.02])
                rc.append(_KIND_RGBA.get(o.kind, _KIND_RGBA["nest"]))
        if rp:
            self._resources.setData(pos=np.array(rp, float),
                                    color=np.array(rc, float), size=18,
                                    pxMode=True)
        else:
            self._resources.setData(pos=np.zeros((0, 3)))

        # frame the whole grid of chambers
        xs = [c[0] for c in self._chambers]
        ys = [c[1] for c in self._chambers]
        gw = (max(xs) + w) - min(xs)
        gh = (max(ys) + h) - min(ys)
        wh = getattr(arena, "wall_height", 0.0)
        self._cam_center = Vector(min(xs) + gw / 2, min(ys) + gh / 2,
                                  min(wh, h) / 3.0)
        self._cam_distance = float(np.hypot(gw, gh)) * 1.5
        self.reset_camera()
        self.clear_playback()

    def has_grass(self):
        return getattr(self.arena, "ground", "floor") == "grass"

    def set_grass_enabled(self, on):
        self._grass_on = bool(on)
        for it in self._grass_items:
            it.setVisible(self._grass_on)
        self.update()

    def _build_grass(self):
        """Scatter thin vertical blades over the floor; heights ~U(2\", 4\")."""
        for it in self._grass_items:
            self.removeItem(it)
        self._grass_items = []
        if not self.has_grass():
            return
        w, h = self.arena.width, self.arena.height
        rng = np.random.default_rng(1234)               # stable field each load
        density = 22.0                                  # blades per m^2
        for dx, dy in self._chambers:
            n = int(min(28000, max(200, w * h * density)))
            bx = dx + rng.random(n) * w
            by = dy + rng.random(n) * h
            bh = 0.0508 + rng.random(n) * 0.0508        # 2"–4" tall
            lean = 0.01                                 # small random tip sway
            tx = bx + (rng.random(n) - 0.5) * lean
            ty = by + (rng.random(n) - 0.5) * lean
            pos = np.empty((2 * n, 3), float)
            pos[0::2] = np.column_stack([bx, by, np.zeros(n)])
            pos[1::2] = np.column_stack([tx, ty, bh])
            base = np.tile([0.16, 0.32, 0.12, 0.85], (n, 1))   # dark green root
            tip = np.tile([0.42, 0.68, 0.28, 0.9], (n, 1))     # lighter tip
            tip[:, :3] *= (0.75 + 0.5 * rng.random(n))[:, None]  # per-blade variety
            col = np.empty((2 * n, 4), float)
            col[0::2] = base
            col[1::2] = np.clip(tip, 0, 1)
            blades = gl.GLLinePlotItem(pos=pos, color=col, width=1.0,
                                       mode="lines", antialias=True)
            blades.setGLOptions("translucent")
            blades.setVisible(self._grass_on)
            self.addItem(blades)
            self._grass_items.append(blades)

    def antenna_sets(self):
        return self.arena.antenna_sets() if self.arena is not None else []

    def set_antenna_layout(self, idx):
        """idx: -1 = off, else index into the arena's antenna layouts."""
        self._antenna_idx = idx
        self._build_antennas()

    def _build_antennas(self):
        """Antenna boxes + numbers + colour-coded PoE cables for the layout."""
        from ..core.poe import gateway_color_map
        for it in self._antenna_items:
            self.removeItem(it)
        self._antenna_items = []
        sets = self.antenna_sets()
        idx = self._antenna_idx
        if not (0 <= idx < len(sets)):
            return
        _, boxes, cables = sets[idx]
        cmap = gateway_color_map(cables)
        has_text = hasattr(gl, "GLTextItem")
        font = QFont("Helvetica", 20)
        font.setBold(True)
        for dx, dy in self._chambers:
            # PoE cables first (drawn under the boxes)
            for cab in cables:
                col = cmap.get(cab.gateway, (0.7, 0.7, 0.7, 1.0))
                pts = _slack_cable([[dx + n[0], dy + n[1], n[2]]
                                    for n in cab.nodes])
                line = gl.GLLinePlotItem(pos=pts, color=col, width=2.5,
                                         antialias=True)
                line.setGLOptions("translucent")
                self.addItem(line)
                self._antenna_items.append(line)
            for a in boxes:
                gw = a.label in cmap
                color = cmap[a.label] if gw else (0.80, 0.72, 0.55, 1.0)
                box = gl.GLMeshItem(meshdata=_box_meshdata(a.w, a.d, a.h),
                                    smooth=False, color=color,
                                    glOptions="opaque", drawEdges=True,
                                    edgeColor=(0.20, 0.20, 0.22, 1.0))
                box.translate(dx + a.x, dy + a.y, a.z)
                self.addItem(box)
                self._antenna_items.append(box)
                if has_text and a.label:
                    t = gl.GLTextItem(
                        pos=np.array([dx + a.x, dy + a.y,
                                      a.z + a.h / 2 + 0.35]),
                        text=a.label, color=pg.mkColor(255, 235, 140, 255),
                        font=font)
                    self.addItem(t)
                    self._antenna_items.append(t)

    def _build_compass(self):
        """N/E/S/W markers just outside the arena for geographically-aligned sites.

        Convention: +y is North, +x is East (origin corner = SW). Always shown
        (not toggled) when ``arena.oriented`` is set.
        """
        for it in self._compass_items:
            self.removeItem(it)
        self._compass_items = []
        if not getattr(self.arena, "oriented", False) or \
                not hasattr(gl, "GLTextItem"):
            return
        w, h = self.arena.width, self.arena.height
        xs = [c[0] for c in self._chambers]
        ys = [c[1] for c in self._chambers]
        x0, x1 = min(xs), max(xs) + w
        y0, y1 = min(ys), max(ys) + h
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        m = 0.06 * max(x1 - x0, y1 - y0)
        font = QFont("Helvetica", 30)
        font.setBold(True)
        _N = pg.mkColor(235, 90, 90, 255)     # red north
        _O = pg.mkColor(220, 224, 230, 255)   # others
        for txt, px, py, col in [("N", cx, y1 + m, _N), ("S", cx, y0 - m, _O),
                                 ("E", x1 + m, cy, _O), ("W", x0 - m, cy, _O)]:
            t = gl.GLTextItem(pos=np.array([px, py, 0.4]), text=txt,
                              color=col, font=font)
            self.addItem(t)
            self._compass_items.append(t)

    def set_measure(self, mode):
        """mode: None | 'metric' | 'imperial' — labelled measurement grid."""
        self._measure_mode = mode
        self._build_measure()

    def _build_measure(self):
        for it in self._measure_items:
            self.removeItem(it)
        self._measure_items = []
        mode = getattr(self, "_measure_mode", None)
        if not mode or self.arena is None:
            return
        w, h = self.arena.width, self.arena.height
        scale = (1.0 / 0.3048) if mode == "imperial" else 1.0
        unit = "ft" if mode == "imperial" else "m"
        step = _nice_step(max(w, h) * scale) / scale        # step in metres
        xs = np.arange(0, w + 1e-9, step)
        ys = np.arange(0, h + 1e-9, step)
        z = 0.02
        lab_off = 0.02 * max(w, h)
        col = pg.mkColor(120, 210, 230, 255)
        has_text = hasattr(gl, "GLTextItem")
        font = QFont("Helvetica", 13)
        for dx, dy in self._chambers:
            segs = []
            for x in xs:
                segs += [[dx + x, dy, z], [dx + x, dy + h, z]]
            for y in ys:
                segs += [[dx, dy + y, z], [dx + w, dy + y, z]]
            line = gl.GLLinePlotItem(pos=np.array(segs, float),
                                     color=(0.30, 0.82, 0.92, 0.5), width=1,
                                     mode="lines", antialias=True)
            line.setGLOptions("translucent")
            self.addItem(line)
            self._measure_items.append(line)
            if not has_text:
                continue
            for x in xs:
                if x <= 0:
                    continue
                t = gl.GLTextItem(pos=np.array([dx + x, dy - lab_off, z]),
                                  text=f"{x * scale:g}{unit}", color=col,
                                  font=font)
                self.addItem(t)
                self._measure_items.append(t)
            for y in ys:
                if y <= 0:
                    continue
                t = gl.GLTextItem(pos=np.array([dx - lab_off, dy + y, z]),
                                  text=f"{y * scale:g}{unit}", color=col,
                                  font=font)
                self.addItem(t)
                self._measure_items.append(t)

    def reset_camera(self):
        if getattr(self, "_cam_center", None) is None:
            return
        self.opts["center"] = self._cam_center
        self.setCameraPosition(distance=self._cam_distance, elevation=32,
                               azimuth=-60)

    def top_down(self):
        if getattr(self, "_cam_center", None) is None:
            return
        self.opts["center"] = self._cam_center
        self.setCameraPosition(distance=self._cam_distance, elevation=89,
                               azimuth=-90)

    def center_on(self, x, y):
        c = getattr(self, "_cam_center", None)
        z = c.z() if c is not None else 0.0
        self.opts["center"] = Vector(float(x), float(y), z)
        self.update()

    def _draw_chamber(self, arena, dx, dy):
        w, h = arena.width, arena.height
        ft = 0.01
        floor = gl.GLMeshItem(meshdata=_box_meshdata(w, h, ft), smooth=False,
                              color=(0.12, 0.14, 0.17, 1.0), glOptions="opaque")
        floor.translate(dx + w / 2, dy + h / 2, -ft / 2)
        self._add_chamber(floor)

        grid = gl.GLGridItem()
        grid.setSize(w, h)
        grid.setSpacing(max(0.05, w / 10.0), max(0.05, h / 10.0))
        grid.translate(dx + w / 2, dy + h / 2, 0.003)
        grid.setColor((1, 1, 1, 0.12))
        self._add_chamber(grid)

        rect = np.array([[dx, dy, 0], [dx + w, dy, 0], [dx + w, dy + h, 0],
                         [dx, dy + h, 0], [dx, dy, 0]], float)
        self._add_chamber(gl.GLLinePlotItem(pos=rect, color=(0.6, 0.6, 0.6, 1.0),
                                            width=2, antialias=True))

        wh = getattr(arena, "wall_height", 0.0)
        wt = getattr(arena, "wall_thickness", 0.005)
        if wh and wh > 0:
            for cx, cy, sx, sy in [(w / 2, 0.0, w + wt, wt),
                                   (w / 2, h, w + wt, wt),
                                   (0.0, h / 2, wt, h), (w, h / 2, wt, h)]:
                slab = gl.GLMeshItem(meshdata=_box_meshdata(sx, sy, wh),
                                     smooth=False, color=_WALL_RGBA,
                                     glOptions="translucent", drawEdges=True,
                                     edgeColor=(0.5, 0.6, 0.72, 0.35))
                slab.translate(dx + cx, dy + cy, wh / 2)
                self._add_chamber(slab)

        for p in getattr(arena, "poles", []):
            md = gl.MeshData.cylinder(rows=1, cols=16,
                                      radius=[p.radius, p.radius],
                                      length=p.height)
            pole = gl.GLMeshItem(meshdata=md, smooth=True,
                                 color=(0.46, 0.34, 0.23, 1.0),
                                 glOptions="opaque")
            pole.translate(dx + p.x, dy + p.y, 0.0)
            self._add_chamber(pole)

        for z in getattr(arena, "zones", []):
            col = _ZONE_RGBA.get(z.role, _ZONE_RGBA["roi"])
            zr = np.array([[dx + z.x, dy + z.y, 0.01],
                           [dx + z.x + z.w, dy + z.y, 0.01],
                           [dx + z.x + z.w, dy + z.y + z.h, 0.01],
                           [dx + z.x, dy + z.y + z.h, 0.01],
                           [dx + z.x, dy + z.y, 0.01]], float)
            self._add_chamber(gl.GLLinePlotItem(pos=zr, color=col, width=2,
                                                antialias=True))

    def _add_chamber(self, item):
        self.addItem(item)
        self._chamber_items.append(item)

    def _ensure_agents(self, n):
        if n == self._n:
            return
        for it in self._bodies + self._heads:
            self.removeItem(it)
        self._bodies, self._heads = [], []
        for _ in range(n):
            b = gl.GLMeshItem(meshdata=self._body_md, smooth=False,
                              color=_MALE, glOptions="opaque")
            hd = gl.GLMeshItem(meshdata=self._head_md, smooth=True,
                               color=_MALE, glOptions="opaque")
            self.addItem(b)
            self.addItem(hd)
            self._bodies.append(b)
            self._heads.append(hd)
        self._n = n

    def clear_playback(self):
        self._history.clear()
        self._last_pos = None
        self._selected = None
        self._trail.setData(pos=np.zeros((0, 3)))
        self._sel.setData(pos=np.zeros((0, 3)))
        for it in self._bodies + self._heads:
            self.removeItem(it)
        self._bodies, self._heads = [], []
        self._n = 0

    def set_selected(self, idx):
        self._selected = idx

    # ------------------------------------------------------------------ #
    def update_agents(self, x, y, sex_m, heading=None, day=None, hour=None,
                      is_day=None, alive=None, colors=None, sizes=None,
                      shapes=None):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = len(x)
        sex_m = np.asarray(sex_m)
        alive = np.ones(n, bool) if alive is None else np.asarray(alive, bool)
        heading = np.zeros(n) if heading is None else np.asarray(heading, float)
        scale = np.ones(n) if sizes is None else np.asarray(sizes, float)
        z = _BODY_H / 2.0
        self._last_pos = np.column_stack([x, y, np.full(n, z)])

        self._ensure_agents(n)
        for i in range(n):
            if not alive[i]:
                col = _DEAD
            elif colors is not None:
                col = tuple(colors[i])
            else:
                col = _MALE if sex_m[i] else _FEMALE
            ang = float(np.degrees(heading[i]))
            sc = float(scale[i])
            off = _HEAD_OFF * sc
            # local=True -> rotate/scale about the body's OWN centre, then
            # translate to position (otherwise it orbits the world origin).
            b = self._bodies[i]
            b.resetTransform()
            b.translate(float(x[i]), float(y[i]), z * sc)
            b.rotate(ang, 0, 0, 1, local=True)
            b.scale(sc, sc, sc, local=True)
            b.setColor(col)
            hd = self._heads[i]
            hd.resetTransform()
            hd.translate(float(x[i] + np.cos(heading[i]) * off),
                         float(y[i] + np.sin(heading[i]) * off), z * sc)
            hd.scale(sc, sc, sc, local=True)
            hd.setColor(col)

        # fading trails
        self._history.append(self._last_pos.copy())
        if len(self._history) > 1:
            tp, tc = [], []
            hist = list(self._history)[:-1]
            for k, hp in enumerate(hist):
                a = 0.05 + 0.25 * (k / len(hist))
                tp.append(hp)
                tc.append(np.tile((0.6, 0.6, 0.6, a), (len(hp), 1)))
            self._trail.setData(pos=np.vstack(tp), color=np.vstack(tc),
                                size=5, pxMode=True)

        if self._selected is not None and self._selected < n:
            sp = self._last_pos[self._selected].copy()
            sp[2] += _BODY_H
            self._sel.setData(pos=sp[None, :], color=(1.0, 0.82, 0.25, 1.0),
                              size=26, pxMode=True)
        else:
            self._sel.setData(pos=np.zeros((0, 3)))

        if is_day is not None:
            bg = _DAY_BG if is_day else _NIGHT_BG
            self.setBackgroundColor(pg.mkColor(*[int(c * 255) for c in bg]))

    # ------------------------------------------------------------------ #
    def mousePressEvent(self, ev):
        self._press_xy = (ev.pos().x(), ev.pos().y())
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)   # orbit/pan while dragging
        if (ev.buttons() == Qt.NoButton and not self._arm_kind
                and self._last_pos is not None):
            idx = self._pick(ev.pos().x(), ev.pos().y())
            self.agent_hovered.emit(idx if idx is not None else -1)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self._press_xy is None:
            return
        rx, ry = ev.pos().x(), ev.pos().y()
        if abs(rx - self._press_xy[0]) > 5 or abs(ry - self._press_xy[1]) > 5:
            return  # a drag (orbit/pan), not a click
        if self._arm_kind:                       # armed: place an object on floor
            pt = self._floor_point(rx, ry)
            if pt is not None and self.arena is not None:
                x = min(max(pt[0], 0.0), self.arena.width)
                y = min(max(pt[1], 0.0), self.arena.height)
                self.object_added.emit(ResourceObject(
                    kind=self._arm_kind, x=round(x, 3), y=round(y, 3),
                    radius=0.18 if self._arm_kind == "food" else 0.15,
                    label=f"{self._arm_kind}_{len(self.arena.objects) + 1}"))
            return
        if self._last_pos is not None:           # otherwise pick nearest agent
            idx = self._pick(rx, ry)
            if idx is not None:
                self._selected = idx
                self.agent_picked.emit(idx)
            else:
                self.agent_picked.emit(-1)       # empty click -> unpin

    def _floor_point(self, sx, sy):
        """Un-project a screen click onto the z=0 floor plane -> (x, y) world."""
        try:
            mvp = self.projectionMatrix() * self.viewMatrix()
            inv, ok = mvp.inverted()
        except Exception:
            return None
        if not ok:
            return None
        w, h = max(1, self.width()), max(1, self.height())
        nx, ny = sx / w * 2.0 - 1.0, 1.0 - sy / h * 2.0
        p0 = inv * QVector4D(nx, ny, -1.0, 1.0)
        p1 = inv * QVector4D(nx, ny, 1.0, 1.0)
        if abs(p0.w()) < 1e-9 or abs(p1.w()) < 1e-9:
            return None
        a = np.array([p0.x() / p0.w(), p0.y() / p0.w(), p0.z() / p0.w()])
        b = np.array([p1.x() / p1.w(), p1.y() / p1.w(), p1.z() / p1.w()])
        d = b - a
        if abs(d[2]) < 1e-9:
            return None
        t = -a[2] / d[2]
        hit = a + t * d
        return float(hit[0]), float(hit[1])

    def _pick(self, sx, sy):
        try:
            mvp = self.projectionMatrix() * self.viewMatrix()
        except Exception:
            return None
        w, h = max(1, self.width()), max(1, self.height())
        best, best_d = None, 30.0
        for i, (px, py, pz) in enumerate(self._last_pos):
            v = mvp * QVector4D(float(px), float(py), float(pz), 1.0)
            if abs(v.w()) < 1e-9:
                continue
            ex = (v.x() / v.w() * 0.5 + 0.5) * w
            ey = (0.5 - v.y() / v.w() * 0.5) * h
            d = np.hypot(ex - sx, ey - sy)
            if d < best_d:
                best, best_d = i, d
        return best

    def arm_add(self, kind):
        self._arm_kind = kind
