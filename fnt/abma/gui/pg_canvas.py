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
_ZONE_RGBA = {
    "center": (0.95, 0.76, 0.31, 1.0),
    "periphery": (0.50, 0.53, 0.56, 1.0),
    "roi": (0.35, 0.66, 0.90, 1.0),
}

# Light / dark palettes for the arena model (agents keep their own colours).
_THEMES = {
    "dark": dict(
        day_bg=(0.13, 0.15, 0.18, 1.0), night_bg=(0.04, 0.05, 0.07, 1.0),
        floor=(0.12, 0.14, 0.17, 1.0), grid=(1, 1, 1, 0.12),
        rect=(0.60, 0.60, 0.60, 1.0), wall=(0.60, 0.70, 0.82, 0.10),
        pole=(0.46, 0.34, 0.23, 1.0),
        compass_n=(235, 90, 90, 255), compass=(220, 224, 230, 255),
        ant_text=(255, 235, 140, 255),
        meas_line=(0.30, 0.82, 0.92, 0.5), meas_text=(120, 210, 230, 255)),
    "light": dict(
        day_bg=(0.95, 0.96, 0.97, 1.0), night_bg=(0.78, 0.81, 0.85, 1.0),
        floor=(0.86, 0.88, 0.90, 1.0), grid=(0, 0, 0, 0.18),
        rect=(0.25, 0.27, 0.30, 1.0), wall=(0.30, 0.42, 0.60, 0.16),
        pole=(0.42, 0.30, 0.19, 1.0),
        compass_n=(190, 45, 40, 255), compass=(45, 50, 58, 255),
        ant_text=(120, 78, 0, 255),
        meas_line=(0.05, 0.42, 0.55, 0.65), meas_text=(12, 95, 125, 255)),
}

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
        self._theme = "dark"
        self._pal = _THEMES["dark"]
        self.setBackgroundColor(
            pg.mkColor(*[int(c * 255) for c in self._pal["night_bg"]]))
        self.arena = None
        self._selected = None
        self._last_pos = None
        self._history = deque(maxlen=10)
        self._press_xy = None
        self._press_btn = Qt.LeftButton
        self._pan_last = None
        self._n = 0
        self._agents_visible = True
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
        self._resource_items = []
        self._resource_mode = -1     # -1 off, 0 lids-on, 1 lids-off (look inside)
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
        self._build_resources()

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
        spec = getattr(self.arena, "grass", None)
        density = getattr(spec, "density", 44.0)
        h_min = getattr(spec, "h_min", 0.0508)
        h_max = getattr(spec, "h_max", 0.1016)
        dry_f = getattr(spec, "dry_fraction", 0.0)
        patch = getattr(spec, "patchiness", 0.0)
        zones = getattr(self.arena, "resource_zones", [])
        towers = getattr(self.arena, "water_towers", [])
        for dx, dy in self._chambers:
            n = int(min(60000, max(200, w * h * density)))
            bx = dx + rng.random(n) * w
            by = dy + rng.random(n) * h
            bh = h_min + rng.random(n) * max(1e-4, h_max - h_min)
            lean = 0.01                                 # small random tip sway
            tx = bx + (rng.random(n) - 0.5) * lean
            ty = by + (rng.random(n) - 0.5) * lean
            var = rng.random(n)                         # per-blade tip variety
            # grass does not grow through the solid resource boxes / towers
            keep = np.ones(n, bool)
            cmap_ = getattr(spec, "cover_map", None)
            cmv = None
            if cmap_ is not None and len(cmap_):
                # site-scale cover pattern; rows S->N, cols W->E
                cm = np.asarray(cmap_, float)
                ci = np.clip(((by - dy) / h * cm.shape[0]).astype(int),
                             0, cm.shape[0] - 1)
                cj = np.clip(((bx - dx) / w * cm.shape[1]).astype(int),
                             0, cm.shape[1] - 1)
                cmv = cm[ci, cj]
                keep &= rng.random(n) < cmv
            if cmv is not None and cmv.mean() > 0:
                # green tufts follow the measured pattern; straw fills the rest
                p_green = np.clip((1.0 - dry_f) * cmv / cmv.mean(), 0.0, 1.0)
                dry = rng.random(n) >= p_green
            else:
                dry = rng.random(n) < dry_f             # straw vs green blades
            if patch > 0:      # clump into tufts, leaving bare ground between
                gsz = 26
                field = rng.random((gsz, gsz))
                for _ in range(2):                      # smooth the noise field
                    field = (field
                             + np.roll(field, 1, 0) + np.roll(field, -1, 0)
                             + np.roll(field, 1, 1) + np.roll(field, -1, 1)) / 5.0
                field = (field - field.min()) / max(1e-9, np.ptp(field))
                gi = np.clip(((by - dy) / h * gsz).astype(int), 0, gsz - 1)
                gj = np.clip(((bx - dx) / w * gsz).astype(int), 0, gsz - 1)
                keep &= rng.random(n) < (1.0 - patch) + patch * field[gi, gj]
            for z in zones:
                keep &= ~((np.abs(bx - (dx + z.x)) <= z.w / 2)
                          & (np.abs(by - (dy + z.y)) <= z.d / 2))
            for wt in towers:
                keep &= ((bx - (dx + wt.x)) ** 2
                         + (by - (dy + wt.y)) ** 2) > wt.radius ** 2
            bx, by, bh, tx, ty, var, dry = (bx[keep], by[keep], bh[keep],
                                            tx[keep], ty[keep], var[keep],
                                            dry[keep])
            m = len(bx)
            if m == 0:
                continue
            pos = np.empty((2 * m, 3), float)
            pos[0::2] = np.column_stack([bx, by, np.zeros(m)])
            pos[1::2] = np.column_stack([tx, ty, bh])
            base = np.tile([0.16, 0.32, 0.12, 0.85], (m, 1))   # dark green root
            tip = np.tile([0.42, 0.68, 0.28, 0.9], (m, 1))     # lighter tip
            if dry.any():                                      # straw / senesced
                base[dry] = [0.34, 0.28, 0.16, 0.85]
                tip[dry] = [0.72, 0.63, 0.38, 0.9]
            tip[:, :3] *= (0.75 + 0.5 * var)[:, None]          # per-blade variety
            col = np.empty((2 * m, 4), float)
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
                bare = getattr(a, "style", "box") == "bare"
                color = cmap[a.label] if gw else (
                    (0.16, 0.16, 0.18, 1.0) if bare else (0.80, 0.72, 0.55, 1.0))
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
                        text=a.label, color=pg.mkColor(*self._pal["ant_text"]),
                        font=font)
                    t.setDepthValue(10)          # draw text after all geometry
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
        _N = pg.mkColor(*self._pal["compass_n"])    # red north
        _O = pg.mkColor(*self._pal["compass"])      # others
        for txt, px, py, col in [("N", cx, y1 + m, _N), ("S", cx, y0 - m, _O),
                                 ("E", x1 + m, cy, _O), ("W", x0 - m, cy, _O)]:
            t = gl.GLTextItem(pos=np.array([px, py, 0.4]), text=txt,
                              color=col, font=font)
            t.setDepthValue(10)
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
        col = pg.mkColor(*self._pal["meas_text"])
        has_text = hasattr(gl, "GLTextItem")
        font = QFont("Helvetica", 13)
        for dx, dy in self._chambers:
            segs = []
            for x in xs:
                segs += [[dx + x, dy, z], [dx + x, dy + h, z]]
            for y in ys:
                segs += [[dx, dy + y, z], [dx + w, dy + y, z]]
            line = gl.GLLinePlotItem(pos=np.array(segs, float),
                                     color=self._pal["meas_line"], width=1,
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
                t.setDepthValue(10)
                self.addItem(t)
                self._measure_items.append(t)
            for y in ys:
                if y <= 0:
                    continue
                t = gl.GLTextItem(pos=np.array([dx - lab_off, dy + y, z]),
                                  text=f"{y * scale:g}{unit}", color=col,
                                  font=font)
                t.setDepthValue(10)
                self.addItem(t)
                self._measure_items.append(t)

    def has_resources(self):
        a = self.arena
        return bool(a and (getattr(a, "water_towers", [])
                           or getattr(a, "resource_zones", [])))

    def set_resources_mode(self, mode):
        """mode: -1 off, 0 lids-on, 1 lids-off (look inside)."""
        self._resource_mode = mode
        self._build_resources()

    def _slab(self, sx, sy, sz, cx, cy, cz, color, edge=(0.24, 0.30, 0.16, 1.0)):
        it = gl.GLMeshItem(meshdata=_box_meshdata(sx, sy, sz), smooth=False,
                           color=color, glOptions="opaque", drawEdges=True,
                           edgeColor=edge)
        it.translate(cx, cy, cz)
        self.addItem(it)
        self._resource_items.append(it)

    def _build_resources(self):
        """Water towers + walled resource zones (doorway, lid, interior)."""
        for it in self._resource_items:
            self.removeItem(it)
        self._resource_items = []
        mode = self._resource_mode
        if mode < 0 or not self.has_resources():
            return
        WALL = (0.13, 0.20, 0.46, 1.0)     # navy blue box material
        t = 0.012                           # wall / floor thickness
        for dx, dy in self._chambers:
            for wt in getattr(self.arena, "water_towers", []):
                md = gl.MeshData.cylinder(rows=1, cols=20,
                                          radius=[wt.radius, wt.radius],
                                          length=wt.height)
                cyl = gl.GLMeshItem(meshdata=md, smooth=True,
                                    color=(0.25, 0.55, 0.85, 0.92),
                                    glOptions="opaque")
                cyl.translate(dx + wt.x, dy + wt.y, 0.0)
                self.addItem(cyl)
                self._resource_items.append(cyl)
            for z in getattr(self.arena, "resource_zones", []):
                self._build_zone(dx + z.x, dy + z.y, z, WALL, t, mode)

    def _door_outline(self, loop):
        """Thin white outline tracing an entrance opening."""
        edge = gl.GLLinePlotItem(pos=np.array(loop, float), color=(1, 1, 1, 1),
                                 width=2.0, antialias=True)
        edge.setGLOptions("opaque")
        self.addItem(edge)
        self._resource_items.append(edge)

    def _build_zone(self, cx, cy, z, wall, t, mode):
        w, d, h = z.w, z.d, z.h
        hw = getattr(z, "hole", 0.0762)         # 3" doorway
        side = getattr(z, "entrance", "E")
        self._slab(w, d, t, cx, cy, t / 2, wall)                     # floor
        # north / south walls (run east-west); one may carry the doorway
        for sy, nm in ((d / 2, "N"), (-d / 2, "S")):
            wy = cy + sy
            if nm == side:                       # split around the hole (in x, z)
                seg = (w - hw) / 2
                xoff = (hw + seg) / 2
                self._slab(seg, t, h, cx + xoff, wy, h / 2, wall)
                self._slab(seg, t, h, cx - xoff, wy, h / 2, wall)
                self._slab(hw, t, h - hw, cx, wy, hw + (h - hw) / 2, wall)
                oy = wy + (1.0 if nm == "N" else -1.0) * (t / 2 + 0.004)
                x0, x1 = cx - hw / 2, cx + hw / 2
                self._door_outline([[x0, oy, 0.004], [x1, oy, 0.004],
                                    [x1, oy, hw], [x0, oy, hw],
                                    [x0, oy, 0.004]])
            else:
                self._slab(w, t, h, cx, wy, h / 2, wall)
        # east / west walls (run north-south)
        for sx, xw in ((w / 2, "E"), (-w / 2, "W")):
            wx = cx + sx
            if xw == side:                       # split around the hole (in y, z)
                seg = (d - hw) / 2
                yoff = (hw + seg) / 2
                self._slab(t, seg, h, wx, cy + yoff, h / 2, wall)
                self._slab(t, seg, h, wx, cy - yoff, h / 2, wall)
                self._slab(t, hw, h - hw, wx, cy, hw + (h - hw) / 2, wall)
                ox = wx + (1.0 if xw == "E" else -1.0) * (t / 2 + 0.004)
                y0, y1 = cy - hw / 2, cy + hw / 2
                self._door_outline([[ox, y0, 0.004], [ox, y1, 0.004],
                                    [ox, y1, hw], [ox, y0, hw],
                                    [ox, y0, 0.004]])
            else:
                self._slab(t, d, h, wx, cy, h / 2, wall)
        # tops are never drawn — zones stay open so you can see inside
        self._fill_zone(cx, cy, z, t)

    def _fill_zone(self, cx, cy, z, t):
        w, d = z.w, z.d
        iw, idp = w - 3 * t, d - 3 * t           # inset footprint
        rng = np.random.default_rng(int(abs(cx * 1000 + cy)) + 7)
        # aspen bedding: scattered short tan chips lying on the floor
        n = int(max(90, iw * idp * 2700))        # 3× density
        bx = cx + (rng.random(n) - 0.5) * iw
        by = cy + (rng.random(n) - 0.5) * idp
        ang = rng.random(n) * np.pi
        ln = 0.006 + rng.random(n) * 0.006
        segs = np.empty((2 * n, 3), float)
        segs[0::2] = np.column_stack([bx - np.cos(ang) * ln,
                                      by - np.sin(ang) * ln,
                                      np.full(n, t + 0.004)])
        segs[1::2] = np.column_stack([bx + np.cos(ang) * ln,
                                      by + np.sin(ang) * ln,
                                      np.full(n, t + 0.006)])
        chips = gl.GLLinePlotItem(pos=segs, color=(0.82, 0.72, 0.52, 1.0),
                                  width=1.2, mode="lines", antialias=True)
        chips.setGLOptions("opaque")
        self.addItem(chips)
        self._resource_items.append(chips)
        # ~6×6" chow pile mounded in a corner of the box
        px = cx + iw / 2 - 0.09
        py = cy - idp / 2 + 0.09
        m = 180                                  # 2× pellet density
        r = 0.076 * np.sqrt(rng.random(m))       # within ~3" radius
        a = rng.random(m) * 2 * np.pi
        gx = px + r * np.cos(a)
        gy = py + r * np.sin(a)
        gz = t + 0.05 * np.clip(1 - r / 0.08, 0, 1) + rng.random(m) * 0.01
        pel = gl.GLScatterPlotItem(pos=np.column_stack([gx, gy, gz]),
                                   color=(0.50, 0.40, 0.26, 1.0), size=6.0,
                                   pxMode=True)
        pel.setGLOptions("opaque")
        self.addItem(pel)
        self._resource_items.append(pel)

    def reset_camera(self):
        if getattr(self, "_cam_center", None) is None:
            return
        self.opts["center"] = self._cam_center
        self.setCameraPosition(distance=self._cam_distance, elevation=32,
                               azimuth=-120)

    def top_down(self):
        self.snap_view("top")

    # CAD-style snap views. Convention: +y = North, +x = East.
    _SNAP = {
        "iso": (32, -120),       # south-west vantage
        "top": (89.9, -90),      # straight down, North up
        "bottom": (-89.9, -90),  # straight up from below
        "north": (7, 90),        # camera on the N side, looking S
        "south": (7, -90),       # camera on the S side, looking N
        "east": (7, 0),          # camera on the E side, looking W
        "west": (7, 180),        # camera on the W side, looking E
    }

    def set_theme(self, name):
        """Switch the arena model between 'dark' and 'light' palettes."""
        self._theme = name if name in _THEMES else "dark"
        self._pal = _THEMES[self._theme]
        self.setBackgroundColor(
            pg.mkColor(*[int(c * 255) for c in self._pal["day_bg"]]))
        if self.arena is not None:
            cam = dict(center=self.opts["center"], distance=self.opts["distance"],
                       elevation=self.opts["elevation"],
                       azimuth=self.opts["azimuth"])
            self.set_arena(self.arena, self._chambers)   # rebuild with new colours
            self.opts["center"] = cam["center"]          # keep the camera put
            self.setCameraPosition(distance=cam["distance"],
                                   elevation=cam["elevation"],
                                   azimuth=cam["azimuth"])

    def snap_view(self, name):
        if getattr(self, "_cam_center", None) is None:
            return
        elev, azim = self._SNAP.get(name, self._SNAP["iso"])
        self.opts["center"] = self._cam_center
        self.setCameraPosition(distance=self._cam_distance,
                               elevation=elev, azimuth=azim)

    def center_on(self, x, y):
        c = getattr(self, "_cam_center", None)
        z = c.z() if c is not None else 0.0
        self.opts["center"] = Vector(float(x), float(y), z)
        self.update()

    def _draw_chamber(self, arena, dx, dy):
        w, h = arena.width, arena.height
        ft = 0.01
        pal = self._pal
        floor = gl.GLMeshItem(meshdata=_box_meshdata(w, h, ft), smooth=False,
                              color=pal["floor"], glOptions="opaque")
        floor.translate(dx + w / 2, dy + h / 2, -ft / 2)
        self._add_chamber(floor)

        grid = gl.GLGridItem()
        grid.setSize(w, h)
        grid.setSpacing(max(0.05, w / 10.0), max(0.05, h / 10.0))
        grid.translate(dx + w / 2, dy + h / 2, 0.003)
        grid.setColor(pal["grid"])
        self._add_chamber(grid)

        rect = np.array([[dx, dy, 0], [dx + w, dy, 0], [dx + w, dy + h, 0],
                         [dx, dy + h, 0], [dx, dy, 0]], float)
        self._add_chamber(gl.GLLinePlotItem(pos=rect, color=pal["rect"],
                                            width=2, antialias=True))

        wh = getattr(arena, "wall_height", 0.0)
        wt = getattr(arena, "wall_thickness", 0.005)
        if wh and wh > 0:
            for cx, cy, sx, sy in [(w / 2, 0.0, w + wt, wt),
                                   (w / 2, h, w + wt, wt),
                                   (0.0, h / 2, wt, h), (w, h / 2, wt, h)]:
                slab = gl.GLMeshItem(meshdata=_box_meshdata(sx, sy, wh),
                                     smooth=False, color=pal["wall"],
                                     glOptions="translucent", drawEdges=True,
                                     edgeColor=(0.5, 0.6, 0.72, 0.35))
                slab.translate(dx + cx, dy + cy, wh / 2)
                self._add_chamber(slab)

        for p in getattr(arena, "poles", []):
            md = gl.MeshData.cylinder(rows=1, cols=16,
                                      radius=[p.radius, p.radius],
                                      length=p.height)
            pole = gl.GLMeshItem(meshdata=md, smooth=True,
                                 color=pal["pole"], glOptions="opaque")
            pole.translate(dx + p.x, dy + p.y, 0.0)
            self._add_chamber(pole)

        for hut in getattr(arena, "huts", []):
            self._draw_hut(dx + hut.x, dy + hut.y, hut)

        for z in getattr(arena, "zones", []):
            col = _ZONE_RGBA.get(z.role, _ZONE_RGBA["roi"])
            zr = np.array([[dx + z.x, dy + z.y, 0.01],
                           [dx + z.x + z.w, dy + z.y, 0.01],
                           [dx + z.x + z.w, dy + z.y + z.h, 0.01],
                           [dx + z.x, dy + z.y + z.h, 0.01],
                           [dx + z.x, dy + z.y, 0.01]], float)
            self._add_chamber(gl.GLLinePlotItem(pos=zr, color=col, width=2,
                                                antialias=True))

    def _draw_hut(self, px, py, hut):
        """Red acrylic shelter: hollow tube (open both ends) or half dome."""
        red = (0.82, 0.16, 0.16, 0.75)
        if hut.kind == "dome":
            r = hut.w / 2.0
            md = gl.MeshData.sphere(rows=10, cols=18, radius=r)
            dome = gl.GLMeshItem(meshdata=md, smooth=True, color=red,
                                 glOptions="translucent")
            dome.translate(px, py, 0.0)     # lower half sits under the floor
            self._add_chamber(dome)
            return
        L, wd, ht, th = hut.w, hut.d, hut.h, hut.thickness
        ang = math.radians(hut.angle)
        ca, sa = math.cos(ang), math.sin(ang)
        # floor, ceiling and the two sides — ends stay open so mice run through
        for sx, sy, sz, ox, oy, oz in (
                (L, wd, th, 0.0, 0.0, th / 2),                  # floor
                (L, wd, th, 0.0, 0.0, ht - th / 2),             # roof
                (L, th, ht, 0.0, wd / 2 - th / 2, ht / 2),      # side
                (L, th, ht, 0.0, -(wd / 2 - th / 2), ht / 2)):  # side
            it = gl.GLMeshItem(meshdata=_box_meshdata(sx, sy, sz), smooth=False,
                               color=red, glOptions="translucent")
            it.translate(px + ox * ca - oy * sa, py + ox * sa + oy * ca, oz)
            it.rotate(hut.angle, 0, 0, 1, local=True)
            self._add_chamber(it)

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
            b.setVisible(self._agents_visible)
            hd.setVisible(self._agents_visible)
            self.addItem(b)
            self.addItem(hd)
            self._bodies.append(b)
            self._heads.append(hd)
        self._n = n

    def set_agents_visible(self, on):
        self._agents_visible = bool(on)
        for it in self._bodies + self._heads:
            it.setVisible(self._agents_visible)
        for it in (self._trail, self._sel):
            it.setVisible(self._agents_visible)
        self.update()

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

        # per-agent colour (used for bodies and their trails)
        if colors is not None:
            acol = np.asarray(colors, float)
            if acol.shape[1] == 3:
                acol = np.column_stack([acol, np.ones(n)])
        else:
            acol = np.tile(_FEMALE, (n, 1))
            acol[sex_m.astype(bool)] = _MALE
        acol = acol.copy()
        acol[~alive] = _DEAD
        self._agent_colors = acol

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

        # fading trails, tinted to each agent's colour so they read on the ground
        self._history.append(self._last_pos.copy())
        if len(self._history) > 1:
            tp, tc = [], []
            hist = list(self._history)[:-1]
            for k, hp in enumerate(hist):
                a = 0.20 + 0.60 * (k / len(hist))       # brighter, fades in
                tp.append(hp)
                c = self._agent_colors.copy()
                if len(c) == len(hp):
                    c[:, 3] = a
                    tc.append(c)
                else:                                    # agent count changed
                    tc.append(np.tile((0.7, 0.7, 0.7, a), (len(hp), 1)))
            pos = np.vstack(tp)
            pos[:, 2] = 0.022                            # sit just above the floor
            self._trail.setData(pos=pos, color=np.vstack(tc),
                                size=7, pxMode=True)

        if self._selected is not None and self._selected < n:
            sp = self._last_pos[self._selected].copy()
            sp[2] += _BODY_H
            self._sel.setData(pos=sp[None, :], color=(1.0, 0.82, 0.25, 1.0),
                              size=26, pxMode=True)
        else:
            self._sel.setData(pos=np.zeros((0, 3)))

        if is_day is not None:
            bg = self._pal["day_bg"] if is_day else self._pal["night_bg"]
            self.setBackgroundColor(pg.mkColor(*[int(c * 255) for c in bg]))

    # ------------------------------------------------------------------ #
    def _is_pan(self, ev):
        """Pan gesture: right-button drag, or Shift+left drag (trackpad-friendly)."""
        return bool((ev.buttons() & Qt.RightButton)
                    or ((ev.buttons() & Qt.LeftButton)
                        and (ev.modifiers() & Qt.ShiftModifier)))

    def mousePressEvent(self, ev):
        self._press_xy = (ev.pos().x(), ev.pos().y())
        self._press_btn = ev.button()
        pan = (ev.button() == Qt.RightButton
               or (ev.button() == Qt.LeftButton
                   and (ev.modifiers() & Qt.ShiftModifier)))
        self._pan_last = ev.pos() if pan else None
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._is_pan(ev):                 # slide the orbit centre in view plane
            lpos = ev.pos()
            if self._pan_last is not None:
                diff = lpos - self._pan_last
                self.pan(diff.x(), diff.y(), 0, relative="view-upright")
            self._pan_last = lpos
            ev.accept()
            return
        super().mouseMoveEvent(ev)   # orbit while left-dragging
        if (ev.buttons() == Qt.NoButton and not self._arm_kind
                and self._last_pos is not None):
            idx = self._pick(ev.pos().x(), ev.pos().y())
            self.agent_hovered.emit(idx if idx is not None else -1)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        self._pan_last = None
        if self._press_xy is None:
            return
        if getattr(self, "_press_btn", Qt.LeftButton) != Qt.LeftButton:
            self._press_xy = None
            return  # right-button (pan) release is never a click
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

    def wheelEvent(self, ev):
        """Smooth, clamped zoom (fixes trackpad jumpiness at zoom extremes)."""
        pd = ev.pixelDelta().y() or ev.pixelDelta().x()      # trackpad: fine-grained
        ad = ev.angleDelta().y() or ev.angleDelta().x()      # mouse wheel: ±120 notch
        steps = (pd / 90.0) if pd else (ad / 120.0)
        steps = max(-2.5, min(2.5, steps))                   # cap a single flick
        factor = max(0.55, min(1.8, 1.0 - 0.14 * steps))     # gentle per-event scale
        ref = getattr(self, "_cam_distance", 10.0) or 10.0
        dist = self.opts.get("distance", ref) * factor
        dist = max(0.12, min(40.0 * ref, dist))              # keep within sane range
        self.setCameraPosition(distance=dist)
        ev.accept()

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
