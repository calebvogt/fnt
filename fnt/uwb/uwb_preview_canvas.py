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

from PyQt5.QtCore import Qt, pyqtSignal   # cursor shapes; ROI draw signals
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


def label_halo(color, linewidth=2.0):
    """Path effects that keep ``color`` text legible on any background.

    The halo is the OPPOSITE of the text: light text gets a dark edge and
    dark text a light one. That is what makes a label survive an arbitrary
    floorplan image - a fixed white halo disappears the moment the text is
    drawn over something pale, which is the same failure as the text itself.

    Returns a list suitable for ``path_effects=``; empty if ``color`` cannot
    be interpreted, so a bad value degrades to plain text rather than raising
    inside a render loop.
    """
    import matplotlib.patheffects as pe
    from matplotlib.colors import to_rgb

    try:
        r, g, b = to_rgb(color)
    except (ValueError, TypeError):
        return []
    # Rec. 601 luma: how bright the text reads, not how bright its channels are.
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return [pe.withStroke(linewidth=linewidth,
                          foreground="#000000" if luma > 0.5 else "#ffffff")]


class UWBPreview2D(FigureCanvas):
    """Top-down matplotlib preview. The only backend that draws a background image."""

    # ROI drawing. Emitted as the user works so the panel can keep its
    # buttons and hint text in step without polling the canvas.
    roi_draft_changed = pyqtSignal(int)     # number of points placed so far
    roi_completed = pyqtSignal(object)      # list of (x, y) in metres
    roi_cancelled = pyqtSignal()
    # A finished region was right-clicked: the window pops the menu, because
    # the canvas has no business knowing what the options are.
    roi_context_requested = pyqtSignal(int)
    # A region's geometry changed in place (moved, or a corner dragged).
    roi_edited = pyqtSignal(int)
    # Move/edit mode ended, so the window can put its hint away.
    roi_edit_ended = pyqtSignal()

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
        self._pan = None            # drag-pan state while a button is held
        self.mpl_connect('scroll_event', self._on_scroll)
        self.mpl_connect('button_press_event', self._on_button)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('button_release_event', self._on_release)
        # Blitting: the arena / zones / anchors / background never move — only
        # the tags do. So the static scene is rendered once, cached, and each
        # scrub frame just restores it and blits the moving tracks + markers on
        # top. The cache is invalidated when the scene changes (arena, theme,
        # zoom, resize).
        self._static_dirty = True
        self._blit_bg = None
        self._last_frame = None     # (x, y, colors, tracks, raw_pts) for re-render
        # User-drawn regions of interest, in the same shape as XML zones:
        # dicts of name / color / linewidth / points (N,2) in metres. Finished
        # ROIs belong to the static scene; only the one being drawn moves.
        self.rois = []
        self.show_rois = True
        # Per-tag label styling, pushed in by the window alongside tag_size.
        # None means "follow the theme", which is only right over a plain
        # figure - hence the option.
        self.label_color = None
        self.label_outline = True
        self._roi_mode = False      # True while the user is placing points
        self._roi_draft = []        # points confirmed so far, in metres
        self._roi_style = {"color": "#e6194b", "linewidth": 2.0}
        self._roi_cursor = None     # live cursor position, for the rubber band
        # Editing a FINISHED region. Separate from draw mode: there is an
        # existing polygon on screen and the gesture changes it, rather than
        # building a new one.
        self._roi_action = None     # None | 'move' | 'edit'
        self._roi_active = None     # index into self.rois being acted on
        self._roi_drag = None       # in-flight drag state, or None
        self._roi_grab = None       # index of the corner being dragged
        self.setFocusPolicy(Qt.StrongFocus)   # so Enter/Esc reach key_press
        self.mpl_connect('key_press_event', self._on_roi_key)
        self._apply_theme()
        self.show_placeholder()

    # ── ROI drawing ──────────────────────────────────────────────────────
    def set_rois(self, rois, show=True):
        """Replace the finished ROI list and redraw the static scene."""
        self.rois = list(rois or [])
        self.show_rois = bool(show)
        # An edit in progress refers to a region by INDEX, and the list just
        # changed under it. Dropping out of the edit is the honest response:
        # the polygon it was holding may not be there any more.
        if self._roi_active is not None and self._roi_active >= len(self.rois):
            self._roi_action = None
            self._roi_active = None
            self._roi_drag = None
            self._roi_grab = None
            self.unsetCursor()
            self.roi_edit_ended.emit()
        self._static_dirty = True
        self._rerender()

    def begin_roi(self, color=None, linewidth=None):
        """Enter draw mode: left-click places corners until the user finishes."""
        self._roi_mode = True
        self._roi_draft = []
        self._roi_cursor = None
        if color:
            self._roi_style["color"] = color
        if linewidth:
            self._roi_style["linewidth"] = float(linewidth)
        # A closed hand or resize cursor would suggest dragging; a crosshair
        # says "this click lands somewhere exact".
        self.setCursor(Qt.CrossCursor)
        self.roi_draft_changed.emit(0)
        self._rerender()

    # Corner handles are grabbed within this many PIXELS, not metres, so the
    # grab feels the same at every zoom level - which is the whole point of
    # being able to zoom while adjusting one.
    ROI_GRAB_PX = 11.0

    def roi_at(self, x, y):
        """Index of the topmost region containing (x, y) in metres, else None.

        Searched last-drawn-first so the region on top is the one you get,
        matching what is actually under the cursor where two overlap.
        """
        from matplotlib.path import Path

        for i in range(len(self.rois) - 1, -1, -1):
            pts = np.asarray(self.rois[i].get("points"),
                             dtype=float).reshape(-1, 2)
            if len(pts) >= 3 and Path(pts).contains_point((x, y)):
                return i
        return None

    def _roi_vertex_at(self, event):
        """Index of the corner of the active region under the cursor, or None."""
        if self._roi_active is None or event.x is None:
            return None
        pts = np.asarray(self.rois[self._roi_active].get("points"),
                         dtype=float).reshape(-1, 2)
        if not len(pts):
            return None
        px = self.ax.transData.transform(pts)
        d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
        i = int(np.argmin(d))
        return i if d[i] <= self.ROI_GRAB_PX else None

    def begin_roi_move(self, index):
        """Enter move mode: drag the whole region to a new place."""
        self._begin_roi_action('move', index)

    def begin_roi_edit(self, index):
        """Enter edit mode: drag individual corners."""
        self._begin_roi_action('edit', index)

    def _begin_roi_action(self, action, index):
        if not (0 <= index < len(self.rois)):
            return
        if self._roi_mode:
            self.cancel_roi()
        self._roi_action = action
        self._roi_active = index
        self._roi_drag = None
        self._roi_grab = None
        # The active region leaves the static cache for the duration, so a
        # drag re-blits one polygon instead of redrawing the whole scene.
        self._static_dirty = True
        self.setCursor(Qt.SizeAllCursor if action == 'move' else Qt.CrossCursor)
        self.setFocus()
        self._rerender()

    def end_roi_edit(self):
        """Leave move/edit mode, keeping whatever the region now looks like."""
        if self._roi_action is None:
            return
        self._roi_action = None
        self._roi_active = None
        self._roi_drag = None
        self._roi_grab = None
        self._static_dirty = True
        self.unsetCursor()
        self.roi_edit_ended.emit()
        self._rerender()

    def finish_roi(self):
        """Close the polygon if it encloses anything, otherwise cancel."""
        pts = list(self._roi_draft)
        self._end_roi_mode()
        if len(pts) >= 3:
            self.roi_completed.emit(pts)
        else:
            self.roi_cancelled.emit()
        self._rerender()

    def cancel_roi(self):
        """Leave draw mode, discarding whatever was placed."""
        self._end_roi_mode()
        self.roi_cancelled.emit()
        self._rerender()

    def _end_roi_mode(self):
        self._roi_mode = False
        self._roi_draft = []
        self._roi_cursor = None
        self.unsetCursor()

    def undo_roi_point(self):
        """Drop the last corner placed (right-click, or the panel's button)."""
        if self._roi_mode and self._roi_draft:
            self._roi_draft.pop()
            self.roi_draft_changed.emit(len(self._roi_draft))
            self._rerender()

    def _on_roi_key(self, event):
        # The window also watches these keys application-wide, because a click
        # in the preview pane hands focus to the timeline slider; this handler
        # covers the case where the canvas does hold focus.
        if self._roi_action is not None:
            if event.key in ('enter', 'return', 'escape'):
                self.end_roi_edit()
            return
        if not self._roi_mode:
            return
        if event.key in ('enter', 'return'):
            self.finish_roi()
        elif event.key == 'escape':
            self.cancel_roi()
        elif event.key in ('backspace', 'delete'):
            self.undo_roi_point()

    def _roi_artists(self):
        """Artists for the in-progress polygon: edges, corners, rubber band.

        Kept out of the static cache because the rubber band follows the mouse;
        baking it in would mean a full redraw on every motion event.
        """
        if self._roi_action is not None and self._roi_active is not None:
            return self._roi_active_artists()
        if not self._roi_mode or not self._roi_draft:
            return []
        col = self._roi_style["color"]
        lw = self._roi_style["linewidth"]
        pts = np.asarray(self._roi_draft, dtype=float)
        out = []
        if len(pts) > 1:
            out.append(self.ax.plot(pts[:, 0], pts[:, 1], '-', color=col,
                                    linewidth=lw, zorder=8, animated=True)[0])
        out.append(self.ax.plot(pts[:, 0], pts[:, 1], 'o', color=col,
                                markersize=5, markeredgecolor='white',
                                markeredgewidth=0.8, zorder=9,
                                animated=True)[0])
        if self._roi_cursor is not None:
            cx, cy = self._roi_cursor
            # Two dashed leaders: one from the last corner to the cursor, one
            # back to the first, so the shape the click would close is visible
            # before committing to it.
            out.append(self.ax.plot([pts[-1, 0], cx], [pts[-1, 1], cy], '--',
                                    color=col, linewidth=lw, alpha=0.9,
                                    zorder=8, animated=True)[0])
            if len(pts) >= 2:
                out.append(self.ax.plot([cx, pts[0, 0]], [cy, pts[0, 1]], ':',
                                        color=col, linewidth=lw, alpha=0.5,
                                        zorder=8, animated=True)[0])
        return out

    def _roi_active_artists(self):
        """The region being moved or edited, with grab handles on its corners.

        Drawn brighter than a settled region and with its corners exposed, so
        it is obvious which one the next drag will affect.
        """
        r = self.rois[self._roi_active]
        pts = np.asarray(r.get("points"), dtype=float).reshape(-1, 2)
        if len(pts) < 2:
            return []
        col = r.get("color", "#e6194b")
        lw = float(r.get("linewidth", 2.0))
        loop = np.vstack([pts, pts[:1]])
        out = [self.ax.plot(loop[:, 0], loop[:, 1], '-', color=col,
                            linewidth=lw + 1.0, zorder=8, animated=True)[0]]
        if self._roi_action == 'edit':
            # Handles only in edit mode: in move mode the corners are not
            # individually grabbable, and showing them would say they were.
            out.append(self.ax.plot(pts[:, 0], pts[:, 1], 'o', color=col,
                                    markersize=7, markeredgecolor='white',
                                    markeredgewidth=1.2, zorder=9,
                                    animated=True)[0])
        return out

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

    def _clamp_view(self, x0, x1, y0, y1):
        """Keep a view within the arena/background extent (plus a small margin).

        Bounds how far the user can zoom OUT (never past the whole window) and
        keeps pans from leaving the scene."""
        axmin, axmax, aymin, aymax = self._arena_view_bounds()
        mx = max((axmax - axmin) * 0.05, 0.25)
        my = max((aymax - aymin) * 0.05, 0.25)
        axmin, axmax, aymin, aymax = axmin - mx, axmax + mx, aymin - my, aymax + my
        # Never wider/taller than the full scene.
        w = min(x1 - x0, axmax - axmin)
        h = min(y1 - y0, aymax - aymin)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        # Shift back inside the bounds.
        if x0 < axmin:
            x1 += axmin - x0; x0 = axmin
        if x1 > axmax:
            x0 -= x1 - axmax; x1 = axmax
        if y0 < aymin:
            y1 += aymin - y0; y0 = aymin
        if y1 > aymax:
            y0 -= y1 - aymax; y1 = aymax
        return (max(x0, axmin), min(x1, axmax), max(y0, aymin), min(y1, aymax))

    def _on_scroll(self, event):
        """Zoom about the cursor (scroll up = in). Persists while scrubbing.
        Zoom-out is capped so you cannot zoom past the whole window."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        factor = 0.8 if event.button == 'up' else 1.25
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        xd, yd = event.xdata, event.ydata
        nx0, nx1 = xd - (xd - x0) * factor, xd + (x1 - xd) * factor
        ny0, ny1 = yd - (yd - y0) * factor, yd + (y1 - yd) * factor
        nx0, nx1, ny0, ny1 = self._clamp_view(nx0, nx1, ny0, ny1)
        axmin, axmax, aymin, aymax = self._arena_view_bounds()
        # If clamped back to (approximately) the whole scene, drop to the
        # natural full-arena framing instead of pinning an odd rectangle.
        if (nx1 - nx0) >= (axmax - axmin) and (ny1 - ny0) >= (aymax - aymin):
            self._user_zoom = None
        else:
            self._user_zoom = ((nx0, nx1), (ny0, ny1))
        self._static_dirty = True     # limits changed -> static cache stale
        self._rerender()

    def _on_button(self, event):
        """Double-click resets zoom; left-press starts a drag-pan (when zoomed).

        While an ROI is being drawn the left button belongs to the polygon, so
        panning moves to the middle button for the duration. Scroll-to-zoom is
        untouched either way - placing a corner accurately is the whole reason
        to zoom in, so it has to keep working mid-polygon.
        """
        if self._roi_mode and event.inaxes is self.ax:
            if event.button == 1 and event.xdata is not None:
                if getattr(event, 'dblclick', False):
                    self.finish_roi()       # double-click closes the polygon
                else:
                    self._roi_draft.append((float(event.xdata),
                                            float(event.ydata)))
                    self.roi_draft_changed.emit(len(self._roi_draft))
                    self._rerender()
                return
            if event.button == 3:
                self.undo_roi_point()       # right-click takes one back
                return
            if event.button == 2:           # middle-drag pans while drawing
                self._pan = (event.x, event.y,
                             self.ax.get_xlim(), self.ax.get_ylim())
                return

        # Moving or reshaping a finished region. Like draw mode, this owns the
        # left button and leaves the middle button to pan.
        if self._roi_action is not None and event.inaxes is self.ax:
            if event.button == 1 and event.xdata is not None:
                pts = np.asarray(self.rois[self._roi_active].get("points"),
                                 dtype=float).reshape(-1, 2)
                if self._roi_action == 'move':
                    self._roi_drag = (float(event.xdata), float(event.ydata),
                                      pts.copy())
                else:
                    self._roi_grab = self._roi_vertex_at(event)
                    if self._roi_grab is not None:
                        self._roi_drag = (float(event.xdata),
                                          float(event.ydata), pts.copy())
                return
            if event.button == 3:
                self.end_roi_edit()         # right-click finishes the edit
                return
            if event.button == 2:
                self._pan = (event.x, event.y,
                             self.ax.get_xlim(), self.ax.get_ylim())
                return

        # Right-click on a finished region asks the window for its menu. Only
        # when nothing else is in progress, so it can never interrupt a
        # polygon halfway through being placed.
        if (event.button == 3 and event.inaxes is self.ax
                and event.xdata is not None and self.show_rois):
            hit = self.roi_at(float(event.xdata), float(event.ydata))
            if hit is not None:
                self.roi_context_requested.emit(hit)
            return

        if getattr(event, 'dblclick', False):
            self._user_zoom = None
            self._pan = None
            self._static_dirty = True
            self._rerender()
        elif (event.button == 1 and event.inaxes is self.ax
              and self._user_zoom is not None):
            # Pan only makes sense when zoomed in.
            self._pan = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())
            # Closed-hand cursor while dragging, so the pan reads as grabbing
            # the view rather than clicking on it.
            self.setCursor(Qt.ClosedHandCursor)

    def _on_motion(self, event):
        """Drag-pan: translate the view by the pixel delta since button-press."""
        if self._roi_mode and self._pan is None:
            # Redraw only once at least one corner exists; before that there
            # is no rubber band to follow the cursor with.
            inside = event.inaxes is self.ax and event.xdata is not None
            self._roi_cursor = ((float(event.xdata), float(event.ydata))
                                if inside else None)
            if self._roi_draft:
                self._rerender()
            return
        if self._roi_action is not None and self._pan is None:
            if self._roi_drag is not None and event.xdata is not None:
                x0, y0, pts = self._roi_drag
                dx, dy = float(event.xdata) - x0, float(event.ydata) - y0
                moved = pts.copy()
                if self._roi_action == 'move':
                    moved += (dx, dy)
                else:
                    moved[self._roi_grab] = pts[self._roi_grab] + (dx, dy)
                self.rois[self._roi_active]["points"] = moved
                self._rerender()
            elif self._roi_action == 'edit':
                # Say when a corner is grabbable before the click, rather than
                # leaving the user to discover the tolerance by missing it.
                over = self._roi_vertex_at(event)
                self.setCursor(Qt.PointingHandCursor if over is not None
                               else Qt.CrossCursor)
            return

        if self._pan is None:
            # Advertise that the view can be grabbed, but only where it can.
            if event.inaxes is self.ax and self._user_zoom is not None:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.unsetCursor()
        if self._pan is None or event.x is None or event.y is None:
            return
        px, py, (x0, x1), (y0, y1) = self._pan
        bbox = self.ax.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0:
            return
        dx = (x1 - x0) / bbox.width * (px - event.x)
        dy = (y1 - y0) / bbox.height * (py - event.y)
        nx0, nx1, ny0, ny1 = self._clamp_view(x0 + dx, x1 + dx, y0 + dy, y1 + dy)
        self._user_zoom = ((nx0, nx1), (ny0, ny1))
        self._static_dirty = True
        self._rerender()

    def _on_release(self, event):
        if self._roi_drag is not None:
            # Announce the new geometry once per drag, not once per motion
            # event: the listener rewrites the config file.
            self._roi_drag = None
            self._roi_grab = None
            self.roi_edited.emit(self._roi_active)
            return
        if self._pan is not None:
            self.unsetCursor()
        self._pan = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._static_dirty = True     # a resize invalidates the blit cache
        self._rerender()

    def _rerender(self):
        """Redraw the current frame (used after the static scene changes)."""
        if self._last_frame is not None:
            self.update_frame(*self._last_frame)
        elif self._roi_mode or self._roi_action is not None:
            # No tracking frame yet - a region can be drawn or adjusted against
            # a bare site map, so it still needs a blit pass of its own.
            if self._static_dirty or self._blit_bg is None:
                self._draw_static()
            self.restore_region(self._blit_bg)
            arts = self._roi_artists()
            for a in arts:
                self.ax.draw_artist(a)
            self.blit(self.ax.bbox)
            for a in arts:
                a.remove()
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
        """Install the arena, keeping the scroll-zoom if the framing is the same.

        A genuine re-framing - a different view mode, a moved origin, a fit to
        new data - invalidates a zoom expressed in the old coordinates, so it
        is dropped. An arena that lands on the SAME bounds is not a re-framing
        even though it is a new object, and dropping the zoom there threw away
        the user's view every time the scene was rebuilt for an unrelated
        reason: toggling zones, or nudging the tag icon size.
        """
        before = self._arena_view_bounds() if self.arena is not None else None
        self.arena = arena
        after = self._arena_view_bounds()
        if before is None or any(abs(a - b) > 1e-9
                                 for a, b in zip(before, after)):
            self._user_zoom = None
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
                    path_effects=label_halo(z.get("color", "#888888"), 2.2))

        # User-drawn ROIs sit above the zones and the map but below the tags.
        # Outline only, with a faint wash: an ROI is a boundary the user placed
        # to read tracks against, so it must not hide the tracks.
        if self.show_rois:
            for i, r in enumerate(self.rois or []):
                if i == self._roi_active:
                    continue        # drawn as a dynamic artist while edited
                pts = np.asarray(r.get("points"), dtype=float).reshape(-1, 2)
                if len(pts) < 3:
                    continue
                col = r.get("color", "#e6194b")
                self.ax.add_patch(Polygon(
                    pts, closed=True, facecolor=col, alpha=0.12,
                    edgecolor="none", zorder=3))
                self.ax.add_patch(Polygon(
                    pts, closed=True, facecolor="none", edgecolor=col,
                    linewidth=float(r.get("linewidth", 2.0)), zorder=4))
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                self.ax.annotate(
                    r.get("name", ""), (cx, cy), color=col, fontsize=8,
                    fontweight="bold", ha="center", va="center", zorder=7,
                    path_effects=label_halo(col, 2.4))

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
                     labels=None, batteries=None, readouts=None, behavior=None):
        """Draw one frame by blitting the moving tags over the cached scene.

        ``tracks``: list of (xy Nx2 array, rgb tuple) — one fading polyline per
        tag. ``raw_pts``: optional Mx2 array of raw fixes drawn as faint dots.
        ``labels``: optional per-tag ID strings drawn above each marker.
        ``batteries``: optional per-tag battery voltages drawn under the label in
        small black font (only shown alongside ``labels``).
        ``readouts``: optional per-tag strings (speed / step distance) drawn
        under the voltage, or None for a tag with nothing to report.
        Only these dynamic artists are drawn per frame; the arena/zones/anchors
        come from the cached static background.
        """
        self._last_frame = (x, y, colors, tracks, raw_pts, labels, batteries,
                            readouts, behavior)
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

        # --- behaviour overlays, drawn beneath the markers ------------------
        # ``behavior`` = {"radius": social radius in metres or None,
        #                 "links": [(i, j, colour)] pairs to join,
        #                 "states": [(text, colour)] per tag}
        if behavior:
            radius = behavior.get("radius")
            if radius:
                edge_col = behavior.get("circle_color") or "#9fb3c8"
                for i in range(len(x)):
                    if not ok[i]:
                        continue
                    circ = Circle((x[i], y[i]), float(radius), fill=False,
                                  linestyle=(0, (2, 2)), linewidth=1.2,
                                  edgecolor=edge_col, alpha=0.9, zorder=4.5)
                    self.ax.add_patch(circ)
                    dynamic.append(circ)
            # Dotted like the social circles they connect, but heavier so the
            # link reads as a connection rather than another radius outline.
            for i, j, col in behavior.get("links") or ():
                if i < len(x) and j < len(x) and ok[i] and ok[j]:
                    ln = self.ax.plot([x[i], x[j]], [y[i], y[j]], color=col,
                                      linewidth=2.2, alpha=0.95,
                                      linestyle=(0, (3, 2)), zorder=4.6)[0]
                    dynamic.append(ln)

        if ok.any():
            edge = "white" if self._theme == "dark" else "#333333"
            # ``tag_size`` is a marker *diameter* in points (matching the
            # animation's markersize); scatter wants an area in points**2, so
            # square it. Kept in sync with the export so the preview reads true.
            tag_size = getattr(self, "tag_size", 10)
            dynamic.append(self.ax.scatter(
                x[ok], y[ok], s=float(tag_size) ** 2, c=np.asarray(colors)[ok],
                edgecolors=edge, linewidths=0.6, zorder=5))

        # Per-tag text block, anchored in SCREEN space just outside the marker
        # so it tracks the Tag Icon Size and never drifts as you zoom (a fixed
        # fraction of the y-range left the label stranded far from its dot):
        #
        #     M9429        <- ID
        #    (inactive)    <- behaviour state, smaller, under the ID
        #       o          <- marker
        #     3.91 V       <- battery, below so nothing collides
        #  0.083 m/s · 0.09 m   <- raw speed / step distance, below that
        #
        states = (behavior or {}).get("states") or []
        state_parts = (behavior or {}).get("state_parts") or []
        if (labels is not None or batteries is not None or readouts is not None
                or states) and ok.any():
            pal = _THEMES[self._theme]
            colarr = np.asarray(colors)
            # One colour for the plain readouts; the ID and the state keep
            # their own, because those colours mean something.
            read_col = self.label_color or pal["mpl_fg"]
            halo = (lambda c: label_halo(c)) if self.label_outline else (
                lambda c: [])
            # Clear whichever circle is larger. The icon size is already in
            # points, but the social radius is in metres, so convert it through
            # the data transform - which also means the label keeps its
            # clearance as you zoom, since the circle grows on screen.
            pad = float(tag_size) / 2.0 + 2.0
            radius_m = (behavior or {}).get("radius")
            if radius_m:
                try:
                    origin = self.ax.transData.transform((0.0, 0.0))
                    edge_px = self.ax.transData.transform((float(radius_m), 0.0))
                    radius_pts = (abs(edge_px[0] - origin[0]) * 72.0
                                  / float(self.figure.dpi))
                    pad = max(pad, radius_pts + 2.0)
                except Exception:
                    pass
            for i in range(len(x)):
                if not ok[i]:
                    continue
                lbl = labels[i] if (labels is not None and i < len(labels)) else None
                state_text, state_col = (states[i] if i < len(states) else ("", None))

                # State sits directly beneath the ID, so it takes the inner slot
                # and pushes the ID up by one line only when both are shown.
                parts = state_parts[i] if i < len(state_parts) else None
                if parts:
                    # Each half in its own colour with a white separator, so
                    # locomotion and social state read as two separate facts.
                    from matplotlib.offsetbox import (TextArea, HPacker,
                                                      AnnotationBbox)
                    props = dict(size=6.5, weight="bold")
                    chunks = [TextArea("(", textprops=dict(color="white", **props))]
                    for n_part, (ptxt, pcol) in enumerate(parts):
                        if n_part:
                            chunks.append(TextArea(
                                " - ", textprops=dict(color="white", **props)))
                        chunks.append(TextArea(
                            ptxt, textprops=dict(color=pcol, **props)))
                    chunks.append(TextArea(")", textprops=dict(color="white", **props)))
                    if self.label_outline:
                        for chunk in chunks:
                            t = chunk.get_children()[0]
                            t.set_path_effects(label_halo(t.get_color(), 1.8))
                    packed = HPacker(children=chunks, align="baseline", pad=0, sep=0)
                    abox = AnnotationBbox(
                        packed, (x[i], y[i]), xybox=(0, pad),
                        xycoords="data", boxcoords="offset points",
                        box_alignment=(0.5, 0.0), frameon=False, pad=0.0,
                        annotation_clip=False, zorder=6)
                    self.ax.add_artist(abox)
                    dynamic.append(abox)
                elif state_text:
                    _c = state_col or "#cccccc"
                    dynamic.append(self.ax.annotate(
                        f"({state_text})", (x[i], y[i]),
                        textcoords="offset points", xytext=(0, pad),
                        fontsize=6.5, ha="center", va="bottom",
                        color=_c, zorder=6, path_effects=halo(_c)))
                if lbl:
                    # ID matches the marker colour, so 'Color by: Sex' tints it
                    # blue=M / red=F exactly like the export animation.
                    dynamic.append(self.ax.annotate(
                        str(lbl), (x[i], y[i]),
                        textcoords="offset points",
                        xytext=(0, pad + (8.0 if state_text else 0.0)),
                        fontsize=8, ha="center", va="bottom",
                        color=tuple(colarr[i]), fontweight="bold", zorder=6,
                        path_effects=halo(tuple(colarr[i]))))
                below = pad
                if batteries is not None and i < len(batteries):
                    bv = batteries[i]
                    if bv is not None and np.isfinite(bv):
                        dynamic.append(self.ax.annotate(
                            f"{bv:.2f} V", (x[i], y[i]),
                            textcoords="offset points", xytext=(0, -below),
                            fontsize=6, ha="center", va="top",
                            color=read_col, zorder=6,
                            path_effects=halo(read_col)))
                        below += 8.0
                if readouts is not None and i < len(readouts) and readouts[i]:
                    dynamic.append(self.ax.annotate(
                        readouts[i], (x[i], y[i]),
                        textcoords="offset points", xytext=(0, -below),
                        fontsize=6, ha="center", va="top",
                        color=read_col, zorder=6,
                        path_effects=halo(read_col)))

        dynamic.extend(self._roi_artists())

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
