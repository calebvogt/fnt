"""Standalone preview canvases for the UWB PreProcessing Tool.

Two interchangeable render backends for scrubbing through a bounded window of
UWB tracking data:

``UWBPreview3D``
    GPU-rendered orbitable scene (pyqtgraph OpenGL): floor grid, translucent
    walls, support poles, real anchor/antenna boxes at their surveyed heights,
    tag markers and fading trails. Requires ``pyqtgraph.opengl`` (PyOpenGL);
    importing this module never fails if that is absent -- ``HAVE_GL`` reports
    availability and the caller falls back to the 2D canvas.

``UWBPreview2D``
    Matplotlib top-down view. Always available, and the only backend that
    draws the loaded background/floorplan image.

Both share the same small API so the GUI can swap them freely::

    set_arena(arena)                     # geometry, once per config change
    update_frame(x, y, colors, trails)   # per playback frame
    clear()
    set_theme("dark" | "light")
    top_down() / reset_camera()

Everything is drawn in **UWB world coordinates** (metres, corner origin, +x
east / +y north) -- the same convention the preprocessing pipeline already uses
for ``location_x`` / ``location_y`` and for background-image extents. The arena
rectangle carries its own origin, so registering a known enclosure against the
data is a matter of moving the arena, never of transforming the samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # 3D needs pyqtgraph.opengl (PyOpenGL); the 2D canvas covers the fallback
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph import Vector
    HAVE_GL = True
    GL_ERROR = ""
except Exception as _e:  # pragma: no cover - depends on optional PyOpenGL
    pg = gl = Vector = None
    HAVE_GL = False
    GL_ERROR = f"{type(_e).__name__}: {_e}"

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon, Rectangle

# Palettes deliberately mirror the ABMA arena views so the two tools read as
# one system, without coupling this module to fnt.abma.
_THEMES = {
    "dark": dict(
        bg=(0.13, 0.15, 0.18, 1.0), floor=(0.12, 0.14, 0.17, 1.0),
        grid=(1, 1, 1, 0.12), wall=(0.60, 0.70, 0.82, 0.10),
        pole=(0.46, 0.34, 0.23, 1.0), anchor=(0.95, 0.76, 0.31, 1.0),
        mpl_bg="#1e1e1e", mpl_fg="#cccccc", mpl_grid="#3f3f3f",
        mpl_floor="#26292e"),
    "light": dict(
        bg=(1.0, 1.0, 1.0, 1.0), floor=(0.89, 0.83, 0.70, 1.0),
        grid=(0, 0, 0, 0.15), wall=(0.30, 0.42, 0.60, 0.16),
        pole=(0.42, 0.30, 0.19, 1.0), anchor=(0.80, 0.55, 0.10, 1.0),
        # pure-white surround with a tan arena floor, mirroring the ABMA
        # VoleTerra 2D light model
        mpl_bg="#ffffff", mpl_fg="#33383f", mpl_grid="#b6bbc3",
        mpl_floor="#e3d5b0"),
}


@dataclass
class PreviewArena:
    """Enclosure geometry in UWB world coordinates (metres).

    ``origin_x`` / ``origin_y`` place the arena's lower-left corner, so a known
    enclosure can be registered against data whose origin sits elsewhere
    without touching the samples themselves.
    """
    width: float = 10.0
    height: float = 10.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    wall_height: float = 0.0
    label: str = "Auto (fit to data)"
    # poles: dicts of x, y, radius, height (metres, absolute world coords)
    poles: list = field(default_factory=list)
    # anchors: dicts of x, y, z, shortid (metres, absolute world coords)
    anchors: list = field(default_factory=list)
    # zones: dicts of name, color (hex), points -> (N,2) array in metres.
    # Populated from the site XML; empty for idealized arenas.
    zones: list = field(default_factory=list)
    # Site map decoded from the XML, with its (x0, x1, y0, y1) extent in metres
    map_image: object = None
    map_extent: tuple = None

    @property
    def center(self):
        return (self.origin_x + self.width / 2.0,
                self.origin_y + self.height / 2.0)

    @property
    def span(self):
        return max(self.width, self.height)


def marker_radius_for(arena):
    """Tag marker radius that stays visible across very different arena sizes.

    A 9 cm vole is a single pixel in a 23 m enclosure, so markers are scaled to
    the arena rather than to life size: ~1.2% of the smaller span, floored so
    small arenas (an open-field box) still get animal-scale dots.
    """
    small = max(0.5, min(arena.width, arena.height))
    return float(max(0.05, small * 0.012))


def fit_arena_to_data(x, y, anchors=None, margin=0.05, label="Auto (fit to data)"):
    """Build a PreviewArena bounding the samples (and anchors), plus a margin.

    ``margin`` is a fraction of the larger extent. Falls back to a 1 m box if
    the inputs are empty or degenerate, so the caller never has to special-case
    an empty selection.
    """
    xs, ys = [], []
    if x is not None and len(x):
        xs.append(np.asarray(x, float))
        ys.append(np.asarray(y, float))
    if anchors:
        xs.append(np.array([a["x"] for a in anchors], float))
        ys.append(np.array([a["y"] for a in anchors], float))
    if not xs:
        return PreviewArena(width=1.0, height=1.0, label=label)

    xa = np.concatenate(xs)
    ya = np.concatenate(ys)
    xa = xa[np.isfinite(xa)]
    ya = ya[np.isfinite(ya)]
    if not len(xa) or not len(ya):
        return PreviewArena(width=1.0, height=1.0, label=label)

    x0, x1 = float(xa.min()), float(xa.max())
    y0, y1 = float(ya.min()), float(ya.max())
    w = max(x1 - x0, 0.5)
    h = max(y1 - y0, 0.5)
    pad = max(w, h) * margin
    return PreviewArena(
        width=w + 2 * pad, height=h + 2 * pad,
        origin_x=x0 - pad, origin_y=y0 - pad,
        anchors=list(anchors or []), label=label)


FT = 0.3048
IN = 0.0254


def voleterra_arena(origin_x=0.0, origin_y=0.0):
    """VoleTerra semi-natural enclosure: 75x75 ft, 3 ft walls, 5x5 pole grid.

    Geometry is defined locally (rather than imported from the ABMA presets) to
    keep this module free of cross-tool dependencies. Perimeter poles are 6 in
    diameter and the inner 3x3 free-standing poles are 3 in.
    """
    w = h = 75 * FT                      # 22.86 m
    sp = w / 4.0                         # 5 poles per side -> 4 gaps
    poles = []
    for i in range(5):
        for j in range(5):
            perimeter = i in (0, 4) or j in (0, 4)
            poles.append(dict(x=origin_x + i * sp, y=origin_y + j * sp,
                              radius=(3 * IN) if perimeter else (1.5 * IN),
                              height=7 * FT))
    return PreviewArena(width=w, height=h, origin_x=origin_x, origin_y=origin_y,
                        wall_height=3 * FT, poles=poles, label="VoleTerra (75x75 ft)")


# name -> factory(origin_x, origin_y). "Auto (fit to data)" is handled by the
# caller via fit_arena_to_data, since it needs the samples.
BUILTIN_ARENAS = {
    "VoleTerra (75x75 ft)": voleterra_arena,
}


def _box_meshdata(sx, sy, sz):
    """MeshData for a box of the given size centred on the origin."""
    x, y, z = sx / 2.0, sy / 2.0, sz / 2.0
    verts = np.array([
        [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
        [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], dtype=float)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=int)
    return gl.MeshData(vertexes=verts, faces=faces)


class UWBPreview3D(gl.GLViewWidget if HAVE_GL else object):
    """Orbitable 3D preview of tag trajectories inside the enclosure."""

    def __init__(self, parent=None):
        if not HAVE_GL:  # pragma: no cover - guarded by caller
            raise RuntimeError(f"pyqtgraph.opengl unavailable: {GL_ERROR}")
        super().__init__(parent)
        self._theme = "dark"
        self._pal = _THEMES["dark"]
        self.setBackgroundColor(pg.mkColor(*[int(c * 255) for c in self._pal["bg"]]))

        self.arena = PreviewArena()
        self._arena_items = []     # rebuilt on set_arena
        self._tag_meshes = []      # one sphere per tag, reused across frames
        self._marker_r = 0.1
        self._sphere_md = None

        self._trail = gl.GLScatterPlotItem()
        self._trail.setGLOptions("translucent")
        self.addItem(self._trail)

    # -- geometry ---------------------------------------------------------- #
    def set_arena(self, arena):
        self.arena = arena
        for it in self._arena_items:
            self.removeItem(it)
        self._arena_items = []

        cx, cy = arena.center
        w, h = arena.width, arena.height

        floor = gl.GLMeshItem(meshdata=_box_meshdata(w, h, 0.01), smooth=False,
                              color=self._pal["floor"], drawEdges=False)
        floor.translate(cx, cy, -0.005)
        self._add(floor)

        grid = gl.GLGridItem()
        grid.setSize(w, h)
        # ~10 divisions across the arena, rounded to a legible step
        step = max(0.1, round(max(w, h) / 10.0, 1))
        grid.setSpacing(step, step)
        grid.setColor(pg.mkColor(*[int(c * 255) for c in self._pal["grid"]]))
        grid.translate(cx, cy, 0.0)
        self._add(grid)

        for z in arena.zones:
            self._build_zone(z)

        if arena.wall_height > 0:
            self._build_walls(arena)

        for p in arena.poles:
            md = gl.MeshData.cylinder(rows=1, cols=12,
                                      radius=[p.get("radius", 0.05)] * 2,
                                      length=p.get("height", 1.0))
            it = gl.GLMeshItem(meshdata=md, smooth=True, color=self._pal["pole"])
            it.translate(p["x"], p["y"], 0.0)
            self._add(it)

        for a in arena.anchors:
            it = gl.GLMeshItem(meshdata=_box_meshdata(0.20, 0.10, 0.20),
                               smooth=False, color=self._pal["anchor"])
            it.translate(a["x"], a["y"], a.get("z", 1.8))
            self._add(it)

        self._marker_r = marker_radius_for(arena)
        self._sphere_md = gl.MeshData.sphere(rows=8, cols=12, radius=self._marker_r)
        for m in self._tag_meshes:      # force rebuild at the new marker scale
            self.removeItem(m)
        self._tag_meshes = []

        self.reset_camera()

    @staticmethod
    def _hex_rgba(color, alpha):
        """'#rrggbb' -> (r, g, b, a) floats. Falls back to mid grey."""
        try:
            c = color.lstrip("#")
            return (int(c[0:2], 16) / 255.0, int(c[2:4], 16) / 255.0,
                    int(c[4:6], 16) / 255.0, alpha)
        except Exception:
            return (0.55, 0.55, 0.55, alpha)

    def _build_zone(self, zone):
        """Lay an XML zone polygon flat on the floor as a translucent patch.

        Uses fan triangulation, which is exact for the convex quads these site
        configs use and degrades gracefully (a slightly clipped fill) if a zone
        is ever concave.
        """
        pts = np.asarray(zone["points"], float)
        if len(pts) < 3:
            return
        is_bounds = zone.get("name", "").strip().lower() == "arena"
        z = 0.004 if is_bounds else 0.002      # keep fills off the floor plane
        verts = np.column_stack([pts, np.full(len(pts), z)])
        faces = np.array([[0, i, i + 1] for i in range(1, len(pts) - 1)], dtype=int)
        it = gl.GLMeshItem(
            meshdata=gl.MeshData(vertexes=verts, faces=faces), smooth=False,
            color=self._hex_rgba(zone.get("color", "#888888"),
                                 0.12 if is_bounds else 0.30),
            drawEdges=True, edgeColor=self._hex_rgba(zone.get("color", "#888888"), 0.9))
        it.setGLOptions("translucent")
        self._add(it)

    def _build_walls(self, arena):
        """Four translucent slabs standing on the arena perimeter."""
        w, h, wh = arena.width, arena.height, arena.wall_height
        cx, cy = arena.center
        t = max(0.01, min(w, h) * 0.004)
        for sx, sy, dx, dy in ((w, t, 0, -h / 2), (w, t, 0, h / 2),
                               (t, h, -w / 2, 0), (t, h, w / 2, 0)):
            slab = gl.GLMeshItem(meshdata=_box_meshdata(sx, sy, wh), smooth=False,
                                 color=self._pal["wall"], drawEdges=False)
            slab.setGLOptions("translucent")
            slab.translate(cx + dx, cy + dy, wh / 2.0)
            self._add(slab)

    def _add(self, item):
        self.addItem(item)
        self._arena_items.append(item)

    # -- per-frame --------------------------------------------------------- #
    def update_frame(self, x, y, colors, tracks=None, raw_pts=None):
        """Place one sphere per tag and draw the trailing tracks.

        ``tracks``: list of (xy Nx2, rgb) smoothed polylines per tag.
        ``raw_pts``: optional Mx2 raw fixes. Both are flattened into the GL
        scatter trail here (the 3D view keeps its point-cloud trail; the 2D
        view is the one that renders true connected lines).
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = len(x)

        while len(self._tag_meshes) < n:
            it = gl.GLMeshItem(meshdata=self._sphere_md, smooth=True,
                               color=(1, 1, 1, 1))
            self.addItem(it)
            self._tag_meshes.append(it)
        for i, it in enumerate(self._tag_meshes):
            it.setVisible(i < n)

        z = self._marker_r
        for i in range(n):
            it = self._tag_meshes[i]
            it.resetTransform()
            if np.isfinite(x[i]) and np.isfinite(y[i]):
                it.translate(float(x[i]), float(y[i]), z)
                it.setColor(tuple(colors[i]))
                it.setVisible(True)
            else:
                it.setVisible(False)   # tag has no fix at this instant

        pos_list, col_list = [], []
        for xy, rgb in (tracks or []):
            if len(xy) < 1:
                continue
            m = len(xy)
            pos_list.append(np.column_stack([xy[:, 0], xy[:, 1], np.zeros(m)]))
            c = np.empty((m, 4), float)
            c[:, :3] = np.array(rgb[:3])
            c[:, 3] = np.linspace(0.12, 0.9, m)
            col_list.append(c)
        if raw_pts is not None and len(raw_pts):
            m = len(raw_pts)
            pos_list.append(np.column_stack([raw_pts[:, 0], raw_pts[:, 1], np.zeros(m)]))
            col_list.append(np.tile((0.5, 0.5, 0.5, 0.3), (m, 1)))
        if pos_list:
            self._trail.setData(pos=np.vstack(pos_list), color=np.vstack(col_list),
                                size=max(2.0, self._marker_r * 40), pxMode=True)
        else:
            self._trail.setData(pos=np.zeros((0, 3)))

    def clear(self):
        for it in self._tag_meshes:
            it.setVisible(False)
        self._trail.setData(pos=np.zeros((0, 3)))

    # -- view -------------------------------------------------------------- #
    def set_theme(self, name):
        self._theme = name if name in _THEMES else "dark"
        self._pal = _THEMES[self._theme]
        self.setBackgroundColor(pg.mkColor(*[int(c * 255) for c in self._pal["bg"]]))
        self.set_arena(self.arena)

    def reset_camera(self):
        cx, cy = self.arena.center
        self.setCameraPosition(pos=Vector(cx, cy, 0),
                               distance=self.arena.span * 1.6,
                               elevation=32, azimuth=-60)

    def top_down(self):
        cx, cy = self.arena.center
        self.setCameraPosition(pos=Vector(cx, cy, 0),
                               distance=self.arena.span * 1.35,
                               elevation=89.9, azimuth=-90)


class UWBPreview2D(FigureCanvas):
    """Top-down matplotlib preview. The only backend that draws a background image."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 6))
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self._theme = "dark"
        self.arena = PreviewArena()
        self.background_image = None
        self.bg_extent = None       # (x0, x1, y0, y1) in metres
        self.show_anchors = True
        # Scroll-wheel zoom: when the user zooms, remember the view so it
        # persists across scrub frames (which otherwise reset to the arena).
        # Double-click resets to the full arena.
        self._user_zoom = None      # ((xmin, xmax), (ymin, ymax)) or None
        self.mpl_connect('scroll_event', self._on_scroll)
        self.mpl_connect('button_press_event', self._on_button)
        # Blitting: the arena / zones / anchors / background never move — only
        # the tags do. So the static scene is rendered once, cached, and each
        # scrub frame just restores it and blits the moving tracks + markers on
        # top. The cache is invalidated when the scene changes (arena, theme,
        # zoom, resize).
        self._static_dirty = True
        self._blit_bg = None
        self._last_frame = None     # (x, y, colors, tracks, raw_pts) for re-render
        self._apply_theme()
        self.show_placeholder()

    def _arena_view_bounds(self):
        """Full view bounds: the arena, expanded to include any background."""
        a = self.arena
        xmin, xmax = a.origin_x, a.origin_x + a.width
        ymin, ymax = a.origin_y, a.origin_y + a.height
        if self.background_image is not None and self.bg_extent is not None:
            bx0, bx1, by0, by1 = self.bg_extent
            xmin, xmax = min(xmin, bx0), max(xmax, bx1)
            ymin, ymax = min(ymin, by0), max(ymax, by1)
        return xmin, xmax, ymin, ymax

    def _on_scroll(self, event):
        """Zoom about the cursor (scroll up = in). Persists while scrubbing."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 0.8 if event.button == 'up' else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        self._user_zoom = ((xd - (xd - x0) * factor, xd + (x1 - xd) * factor),
                           (yd - (yd - y0) * factor, yd + (y1 - yd) * factor))
        self._static_dirty = True     # limits changed -> static cache stale
        self._rerender()

    def _on_button(self, event):
        """Double-click resets the zoom to the full arena."""
        if getattr(event, 'dblclick', False):
            self._user_zoom = None
            self._static_dirty = True
            self._rerender()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._static_dirty = True     # a resize invalidates the blit cache
        self._rerender()

    def _rerender(self):
        """Redraw the current frame (used after the static scene changes)."""
        if self._last_frame is not None:
            self.update_frame(*self._last_frame)
        else:
            self.draw_idle()

    def show_placeholder(self, msg="Load a database and select tags\nto preview tracking data"):
        """Neutral empty state: no bare 0–1 axes, just a centred hint.

        Shown at startup and whenever there is nothing to draw, so the pane
        reads as 'waiting' rather than 'broken'.
        """
        pal = _THEMES[self._theme]
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(pal["mpl_bg"])
        self.fig.patch.set_facecolor(pal["mpl_bg"])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_visible(False)
        self.ax.text(0.5, 0.5, msg, transform=self.ax.transAxes,
                     ha="center", va="center", color=pal["mpl_grid"],
                     fontsize=13, fontstyle="italic")
        # The axes were recreated; drop the blit cache so the next real frame
        # rebuilds the static scene from scratch.
        self._static_dirty = True
        self._blit_bg = None
        self._last_frame = None
        self.draw_idle()

    def _apply_theme(self):
        pal = _THEMES[self._theme]
        self.fig.patch.set_facecolor(pal["mpl_bg"])
        self.ax.set_facecolor(pal["mpl_bg"])
        for spine in self.ax.spines.values():
            spine.set_color(pal["mpl_grid"])
        self.ax.tick_params(colors=pal["mpl_fg"], labelsize=8)
        self.ax.xaxis.label.set_color(pal["mpl_fg"])
        self.ax.yaxis.label.set_color(pal["mpl_fg"])

    def set_theme(self, name):
        self._theme = name if name in _THEMES else "dark"
        self._apply_theme()
        self._static_dirty = True
        self._rerender()

    def set_arena(self, arena):
        self.arena = arena
        self._user_zoom = None    # new framing drops any scroll-zoom
        self._static_dirty = True

    def _draw_static(self):
        """Render the unchanging scene and cache it for blitting.

        Everything that does NOT move per frame lives here: the arena floor /
        background image, the zone polygons + labels, support poles and anchors,
        plus the axis limits/labels. This is the expensive draw, but it only
        runs when the scene changes — not on every scrub frame.
        """
        pal = _THEMES[self._theme]
        a = self.arena
        self.ax.clear()
        self._apply_theme()

        # Site map from the XML takes precedence; otherwise any separately
        # loaded floorplan image. Both use origin="upper" (drawn UPRIGHT, north
        # at top) with the bottom-left corner at world (0, 0) — the Wiser frame.
        if a.map_image is not None and a.map_extent is not None:
            self.ax.imshow(a.map_image, extent=list(a.map_extent),
                           origin="upper", zorder=0)
        elif self.background_image is not None and self.bg_extent is not None:
            self.ax.imshow(self.background_image, extent=list(self.bg_extent),
                           origin="upper", zorder=0)
        elif not a.zones:
            self.ax.add_patch(Rectangle(
                (a.origin_x, a.origin_y), a.width, a.height,
                facecolor=pal["mpl_floor"], edgecolor=pal["mpl_grid"],
                linewidth=1.0, zorder=0))

        if a.zones:
            for z in a.zones:
                pts = z["points"]
                if len(pts) < 3:
                    continue
                is_bounds = z.get("name", "").strip().lower() == "arena"
                self.ax.add_patch(Polygon(
                    pts, closed=True,
                    facecolor="none" if is_bounds else z.get("color", "#888888"),
                    edgecolor=z.get("color", "#888888"),
                    alpha=1.0 if is_bounds else 0.28,
                    linewidth=2.0 if is_bounds else 1.2, zorder=1))
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                self.ax.annotate(
                    z.get("name", ""), (cx, cy), color="#111111", fontsize=7,
                    fontweight="bold", ha="center", va="center", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

        for p in a.poles:
            self.ax.add_patch(Circle(
                (p["x"], p["y"]), p.get("radius", 0.05),
                facecolor="#75573b", edgecolor="none", zorder=2))

        if self.show_anchors and a.anchors:
            self.ax.scatter([q["x"] for q in a.anchors],
                            [q["y"] for q in a.anchors],
                            marker="^", s=110, c="#ffd21e",
                            edgecolors="#1a1a1a", linewidths=1.0, zorder=6)

        if self._user_zoom is not None:
            (xmin, xmax), (ymin, ymax) = self._user_zoom
        else:
            xmin, xmax, ymin, ymax = self._arena_view_bounds()
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.fig.tight_layout()
        self.draw()                                   # full render (synchronous)
        self._blit_bg = self.copy_from_bbox(self.ax.bbox)
        self._static_dirty = False

    def update_frame(self, x, y, colors, tracks=None, raw_pts=None,
                     labels=None, batteries=None):
        """Draw one frame by blitting the moving tags over the cached scene.

        ``tracks``: list of (xy Nx2 array, rgb tuple) — one fading polyline per
        tag. ``raw_pts``: optional Mx2 array of raw fixes drawn as faint dots.
        ``labels``: optional per-tag ID strings drawn above each marker.
        ``batteries``: optional per-tag battery voltages drawn under the label in
        small black font (only shown alongside ``labels``).
        Only these dynamic artists are drawn per frame; the arena/zones/anchors
        come from the cached static background.
        """
        self._last_frame = (x, y, colors, tracks, raw_pts, labels, batteries)
        if self._static_dirty or self._blit_bg is None:
            self._draw_static()

        self.restore_region(self._blit_bg)
        dynamic = []

        if raw_pts is not None and len(raw_pts):
            dynamic.append(self.ax.scatter(
                raw_pts[:, 0], raw_pts[:, 1], s=4,
                c=[(0.5, 0.5, 0.5, 0.30)], edgecolors="none", zorder=3))

        if tracks:
            for xy, rgb in tracks:
                if len(xy) < 2:
                    continue
                pts = xy.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                n = len(segs)
                seg_cols = np.empty((n, 4), float)
                seg_cols[:, :3] = np.array(rgb[:3])
                seg_cols[:, 3] = np.linspace(0.12, 0.95, n)
                lc = LineCollection(segs, colors=seg_cols, linewidths=1.7, zorder=4)
                self.ax.add_collection(lc)
                dynamic.append(lc)

        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.any():
            edge = "white" if self._theme == "dark" else "#333333"
            # ``tag_size`` is a marker *diameter* in points (matching the
            # animation's markersize); scatter wants an area in points**2, so
            # square it. Kept in sync with the export so the preview reads true.
            tag_size = getattr(self, "tag_size", 10)
            dynamic.append(self.ax.scatter(
                x[ok], y[ok], s=float(tag_size) ** 2, c=np.asarray(colors)[ok],
                edgecolors=edge, linewidths=0.6, zorder=5))

        # Optional per-tag ID label (above the marker) with the battery voltage
        # directly beneath it in small black font. Both anchor at y+off: the
        # label grows upward (va="bottom"), the voltage downward (va="top").
        if labels is not None and ok.any():
            ylim = self.ax.get_ylim()
            off = (ylim[1] - ylim[0]) * 0.035
            lab_color = "white" if self._theme == "dark" else "#111111"
            for i in range(len(x)):
                if not ok[i]:
                    continue
                lbl = labels[i] if i < len(labels) else None
                if lbl:
                    dynamic.append(self.ax.text(
                        x[i], y[i] + off, str(lbl), fontsize=8, ha="center",
                        va="bottom", color=lab_color, fontweight="bold", zorder=6))
                if batteries is not None and i < len(batteries):
                    bv = batteries[i]
                    if bv is not None and np.isfinite(bv):
                        dynamic.append(self.ax.text(
                            x[i], y[i] + off, f"{bv:.2f} V", fontsize=6,
                            ha="center", va="top", color="#000000", zorder=6))

        for art in dynamic:
            self.ax.draw_artist(art)
        self.blit(self.ax.bbox)
        # Remove the per-frame artists so they are not baked into the next
        # static cache (and do not accumulate).
        for art in dynamic:
            art.remove()

    def clear(self):
        self.show_placeholder()

    def reset_camera(self):
        self.draw_idle()

    def top_down(self):
        self.draw_idle()
