"""Matplotlib canvas for arena design and live agent playback.

Renders the arena, its resource objects, and (during/after a run) the moving
agents. Click-to-add is supported when an object type is armed via
:meth:`arm_add`.
"""
from __future__ import annotations

from collections import deque

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
from PyQt5.QtCore import pyqtSignal

from ..core.config import ArenaConfig, ResourceObject


def _bow_xy(a, b, k=0.08, n=14):
    """Quadratic-bezier points from a to b, bowed sideways to look like slack."""
    import math
    dx, dy = b[0] - a[0], b[1] - a[1]
    ln = math.hypot(dx, dy) or 1.0
    px, py = -dy / ln, dx / ln              # unit perpendicular
    off = k * ln
    mx, my = (a[0] + b[0]) / 2 + px * off, (a[1] + b[1]) / 2 + py * off
    xs, ys = [], []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        xs.append(u * u * a[0] + 2 * u * t * mx + t * t * b[0])
        ys.append(u * u * a[1] + 2 * u * t * my + t * t * b[1])
    return xs, ys


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

_ZONE_COLOR = {"center": "#f2c14e", "periphery": "#7f8790", "roi": "#5aa9e6"}

_KIND_STYLE = {
    "nest": dict(marker="s", color="#b5793a", size=140, label="nest"),
    "food": dict(marker="o", color="#4caf50", size=120, label="food"),
    "water": dict(marker="o", color="#3aa0d0", size=120, label="water"),
}
_MALE = "#4a90d9"
_FEMALE = "#e0559a"
_MALE_RGBA = (0.29, 0.56, 0.85, 1.0)
_FEMALE_RGBA = (0.88, 0.33, 0.60, 1.0)


class ArenaCanvas(FigureCanvas):
    object_added = pyqtSignal(object)  # emits a ResourceObject on click-add
    agent_picked = pyqtSignal(int)     # emits agent index on click, or -1 empty
    agent_hovered = pyqtSignal(int)    # emits agent index on hover, or -1

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), facecolor="#191b1f")
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self.arena = ArenaConfig()
        self._arm_kind = None
        self._agent_artist = None
        self._agent_scatters = []
        self._sel_artist = None
        self._heading_artist = None
        self._selected = None
        self._last_xy = None
        self._chambers = [(0.0, 0.0)]
        self._trail_artists = []
        self._history = deque(maxlen=8)
        self._daynight_text = None
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.draw_arena()

    def set_selected(self, idx):
        self._selected = idx

    # 2D is already top-down; camera controls are no-ops for interface parity
    def reset_camera(self):
        pass

    def top_down(self):
        pass

    def center_on(self, x, y):
        pass

    # ------------------------------------------------------------------ #
    def set_arena(self, arena: ArenaConfig, chambers=None):
        self.arena = arena
        self._chambers = chambers or [(0.0, 0.0)]
        self.draw_arena()

    def arm_add(self, kind: str | None):
        """Arm click-to-add for ('nest'/'food'/'water') or disable (None)."""
        self._arm_kind = kind

    def set_grass_enabled(self, on):
        """Tint the ground green for outdoor field sites (toggle)."""
        self._grass_on = bool(on)
        if self.arena is not None:
            self.draw_arena()

    def set_antenna_layout(self, idx):
        """Select which UWB layout to draw (-1 = off, else layout index)."""
        self._antenna_idx = idx
        if self.arena is not None:
            self.draw_arena()

    def set_measure(self, mode):
        """Measurement grid: None | 'metric' | 'imperial'."""
        self._measure_mode = mode
        if self.arena is not None:
            self.draw_arena()

    def draw_arena(self):
        ax = self.ax
        ax.clear()
        a = self.arena
        chambers = getattr(self, "_chambers", [(0.0, 0.0)])
        ax.set_facecolor("#14171a")
        xs = [c[0] for c in chambers]
        ys = [c[1] for c in chambers]
        # wider margin when a compass / measurement labels sit outside the arena
        gm = 0.05
        if getattr(a, "oriented", False):
            gm = max(gm, 0.09 * max(a.width, a.height))
        if getattr(self, "_measure_mode", None):
            gm = max(gm, 0.06 * max(a.width, a.height))
        ax.set_xlim(min(xs) - gm, max(xs) + a.width + gm)
        ax.set_ylim(min(ys) - gm, max(ys) + a.height + gm)
        ax.set_aspect("equal")
        ax.set_title("Arena", color="#8a9099", fontsize=10)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color("#303338")
        ax.tick_params(colors="#6f767e", labelsize=8)
        ax.grid(True, color="#22262b", lw=0.6)
        # measurement grid overlay (toggle: metric / imperial)
        mmode = getattr(self, "_measure_mode", None)
        if mmode:
            import numpy as np
            mscale = (1.0 / 0.3048) if mmode == "imperial" else 1.0
            munit = "ft" if mmode == "imperial" else "m"
            mstep = _nice_step(max(a.width, a.height) * mscale) / mscale
            gx = np.arange(0, a.width + 1e-9, mstep)
            gy = np.arange(0, a.height + 1e-9, mstep)
            loff = 0.012 * max(a.width, a.height)
            for dx, dy in chambers:
                for x in gx:
                    ax.plot([dx + x, dx + x], [dy, dy + a.height],
                            color="#4dd0e1", lw=0.6, alpha=0.5, zorder=1)
                    if x > 0:
                        ax.text(dx + x, dy - loff, f"{x * mscale:g}{munit}",
                                color="#9fe3ec", fontsize=6, ha="center",
                                va="top", zorder=6)
                for y in gy:
                    ax.plot([dx, dx + a.width], [dy + y, dy + y],
                            color="#4dd0e1", lw=0.6, alpha=0.5, zorder=1)
                    if y > 0:
                        ax.text(dx - loff, dy + y, f"{y * mscale:g}{munit}",
                                color="#9fe3ec", fontsize=6, ha="right",
                                va="center", zorder=6)
        seen = set()
        for dx, dy in chambers:
            # grass ground tint (outdoor field sites, toggleable)
            if getattr(self, "_grass_on", False) and \
                    getattr(a, "ground", "floor") == "grass":
                ax.add_patch(Rectangle(
                    (dx, dy), a.width, a.height, facecolor="#2f5d2a",
                    alpha=0.4, edgecolor="none", zorder=0))
            # boundary
            ax.plot([dx, dx + a.width, dx + a.width, dx, dx],
                    [dy, dy, dy + a.height, dy + a.height, dy],
                    color="#5a6069", lw=1.5)
            # zones (regions of interest, e.g. OFT centre)
            for z in getattr(a, "zones", []):
                col = _ZONE_COLOR.get(z.role, _ZONE_COLOR["roi"])
                ax.add_patch(Rectangle(
                    (dx + z.x, dy + z.y), z.w, z.h, facecolor=col, alpha=0.12,
                    edgecolor=col, lw=1.2, ls="--", zorder=1))
                ax.text(dx + z.x + z.w / 2, dy + z.y + z.h / 2, z.name,
                        color=col, fontsize=8, ha="center", va="center",
                        alpha=0.8, zorder=2)
            # poles (vertical posts, drawn top-down as filled circles)
            for p in getattr(a, "poles", []):
                ax.add_patch(Circle(
                    (dx + p.x, dy + p.y), p.radius, facecolor="#75542f",
                    edgecolor="#c9a06a", lw=0.8, zorder=2))
            # UWB antenna boxes + colour-coded PoE cables (cyclable layout)
            asets = a.antenna_sets() if hasattr(a, "antenna_sets") else []
            aidx = getattr(self, "_antenna_idx", -1)
            if 0 <= aidx < len(asets):
                from ..core.poe import gateway_color_map
                _, boxes, cables = asets[aidx]
                cmap = gateway_color_map(cables)
                off = 0.02 * max(a.width, a.height)
                for cab in cables:                    # cables under the boxes
                    col = cmap.get(cab.gateway, (0.7, 0.7, 0.7, 1.0))
                    for p, q in zip(cab.nodes, cab.nodes[1:]):
                        bx, by = _bow_xy((dx + p[0], dy + p[1]),
                                         (dx + q[0], dy + q[1]))
                        ax.plot(bx, by, color=col, lw=1.3, alpha=0.9, zorder=3)
                for an in boxes:
                    fc = cmap.get(an.label)
                    face = ("#%02x%02x%02x" % tuple(int(c * 255) for c in fc[:3])
                            ) if fc else "#ccb787"
                    ax.add_patch(Rectangle(
                        (dx + an.x - an.w / 2, dy + an.y - an.d / 2), an.w, an.d,
                        facecolor=face, edgecolor="#20201f", lw=0.7, zorder=4))
                    if an.label:
                        ax.text(dx + an.x, dy + an.y + off, an.label,
                                color="#ffeb8c", fontsize=8, fontweight="bold",
                                ha="center", va="bottom", zorder=5)
            # objects
            for o in a.objects:
                st = _KIND_STYLE.get(o.kind, _KIND_STYLE["nest"])
                lbl = st["label"] if st["label"] not in seen else None
                seen.add(st["label"])
                ax.scatter([dx + o.x], [dy + o.y], marker=st["marker"],
                           s=st["size"], c=st["color"], edgecolors="white",
                           linewidths=0.6, label=lbl, zorder=3)
        if a.objects:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.3,
                      labelcolor="#dddddd")
        # compass for geographically-aligned sites (+y=N, +x=E)
        if getattr(a, "oriented", False):
            x0, x1 = min(xs), max(xs) + a.width
            y0, y1 = min(ys), max(ys) + a.height
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            m = 0.045 * max(a.width, a.height)
            for txt, px, py, col in [("N", cx, y1 + m, "#eb5a5a"),
                                     ("S", cx, y0 - m, "#dce0e6"),
                                     ("E", x1 + m, cy, "#dce0e6"),
                                     ("W", x0 - m, cy, "#dce0e6")]:
                ax.text(px, py, txt, color=col, fontsize=13, fontweight="bold",
                        ha="center", va="center", zorder=6)
        self._agent_artist = None
        self._agent_scatters = []
        self._sel_artist = None
        self._heading_artist = None
        self._trail_artists = []
        self._history.clear()
        self._daynight_text = None
        self.fig.tight_layout()
        self.draw_idle()

    def clear_playback(self):
        """Reset agent markers and trails (call before a new run)."""
        self.draw_arena()

    def update_agents(self, x, y, sex_m, heading=None, day=None, hour=None,
                      is_day=None, alive=None, colors=None, sizes=None,
                      shapes=None):
        """Redraw agent positions (with fading trails) over the static arena."""
        import numpy as np
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        sex_m = np.asarray(sex_m)
        if alive is not None:
            alive = np.asarray(alive, bool)
        else:
            alive = np.ones(len(x), bool)
        self._last_xy = np.column_stack([x, y])

        # day/night tint
        if is_day is not None:
            self.ax.set_facecolor("#20242b" if is_day else "#0c0d11")

        # fading trails from recent history
        for artist in self._trail_artists:
            artist.remove()
        self._trail_artists = []
        self._history.append((x.copy(), y.copy()))
        n_hist = len(self._history)
        for k, (hx, hy) in enumerate(list(self._history)[:-1]):
            alpha = 0.06 + 0.18 * (k / max(1, n_hist))
            art = self.ax.scatter(hx, hy, c="#888888", s=10, alpha=alpha,
                                  linewidths=0, zorder=4)
            self._trail_artists.append(art)

        # current positions — per-agent colour/size, marker by shape
        if colors is not None:
            rgba = np.asarray(colors, float).copy()
        else:
            rgba = np.array([_MALE_RGBA if m else _FEMALE_RGBA for m in sex_m])
        rgba[~alive] = (0.35, 0.35, 0.35, 1.0)          # dead = grey
        s = (70.0 * np.asarray(sizes, float)) if sizes is not None \
            else np.full(len(x), 70.0)
        codes = np.asarray(shapes, int) if shapes is not None \
            else np.zeros(len(x), int)
        for a in self._agent_scatters:
            a.remove()
        self._agent_scatters = []
        for code, mk in ((0, "o"), (1, "D"), (2, "^")):
            m = codes == code
            if m.any():
                self._agent_scatters.append(self.ax.scatter(
                    x[m], y[m], c=rgba[m], s=s[m], marker=mk,
                    edgecolors="black", linewidths=0.5, zorder=6))

        # heading ticks (short line in each agent's facing direction)
        if heading is not None:
            heading = np.asarray(heading, float)
            L = 0.04 * max(self.arena.width, self.arena.height)
            segs = [[(x[i], y[i]),
                     (x[i] + np.cos(heading[i]) * L, y[i] + np.sin(heading[i]) * L)]
                    for i in range(len(x))]
            if self._heading_artist is None:
                self._heading_artist = LineCollection(
                    segs, colors="#e8e8e8", linewidths=1.4, zorder=5)
                self.ax.add_collection(self._heading_artist)
            else:
                self._heading_artist.set_segments(segs)

        # selection ring
        if self._selected is not None and self._selected < len(x):
            sx, sy = x[self._selected], y[self._selected]
            if self._sel_artist is None:
                self._sel_artist = self.ax.scatter(
                    [sx], [sy], s=220, facecolors="none", edgecolors="#ffd23f",
                    linewidths=2.0, zorder=7)
            else:
                self._sel_artist.set_offsets([[sx, sy]])
                self._sel_artist.set_visible(True)
        elif self._sel_artist is not None:
            self._sel_artist.set_visible(False)

        title = "Arena"
        if day is not None:
            title += f" — day {day}"
        if hour is not None:
            icon = "☀" if is_day else "☾"  # sun / moon
            title += f"  {icon} {int(hour):02d}:00"
        n_alive = int(alive.sum())
        if n_alive < len(alive):
            title += f"  ({n_alive}/{len(alive)} alive)"
        self.ax.set_title(title, color="#cccccc", fontsize=10)
        self.draw_idle()

    # ------------------------------------------------------------------ #
    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        # playback mode (no object armed): pick the nearest agent
        if self._arm_kind is None:
            if self._last_xy is None or len(self._last_xy) == 0:
                return
            import numpy as np
            d = np.hypot(self._last_xy[:, 0] - event.xdata,
                         self._last_xy[:, 1] - event.ydata)
            i = int(np.argmin(d))
            if d[i] < 0.2:  # within 20 cm of a marker
                self._selected = i
                self.agent_picked.emit(i)
            else:
                self.agent_picked.emit(-1)   # empty click -> unpin
            return
        obj = ResourceObject(
            kind=self._arm_kind, x=round(event.xdata, 3),
            y=round(event.ydata, 3),
            radius=0.15 if self._arm_kind != "food" else 0.18,
            label=f"{self._arm_kind}_{len(self.arena.objects) + 1}",
        )
        # The window owns object state; just propose the object.
        self.object_added.emit(obj)

    def _on_motion(self, event):
        """Emit the agent under the cursor (or -1) for the hover popup."""
        import numpy as np
        if (self._arm_kind is not None or event.inaxes != self.ax
                or event.xdata is None or self._last_xy is None
                or len(self._last_xy) == 0):
            self.agent_hovered.emit(-1)
            return
        d = np.hypot(self._last_xy[:, 0] - event.xdata,
                     self._last_xy[:, 1] - event.ydata)
        i = int(np.argmin(d))
        self.agent_hovered.emit(i if d[i] < 0.15 else -1)
