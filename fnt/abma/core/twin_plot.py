"""One figure that shows two lives apart.

Left, the enclosure with the changed animal's two paths over the same hours —
the life as it was in black, the life with the tweak in red. Right, how far
every animal drifted from its other self, and what the change did to the
changed animal's body. Black against red throughout, never red against green.
"""
from __future__ import annotations

import numpy as np

from .twin import TwinResult

AS_IT_WAS = "black"
WITH_TWEAK = "red"


def figure(res: TwinResult, cfg=None, title: str | None = None):
    """Build the matplotlib figure for a twin result (no display needed)."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12.0, 6.4), dpi=110)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], hspace=0.42,
                          wspace=0.2, left=0.05, right=0.97, top=0.9,
                          bottom=0.09)
    ax = fig.add_subplot(gs[:, 0])
    w, h = res.arena
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    if cfg is not None:
        marks = {"food": ("s", "goldenrod", "food"),
                 "water": ("o", "dodgerblue", "water"),
                 "nest": ("^", "gray", "nest")}
        seen = set()
        for o in cfg.arena.objects:
            mk, col, lab = marks.get(o.kind, ("x", "gray", o.kind))
            ax.scatter([o.x], [o.y], marker=mk, s=110, color=col, alpha=0.55,
                       edgecolors="none", zorder=1,
                       label=None if lab in seen else lab)
            seen.add(lab)
        for z in getattr(cfg.arena, "resource_zones", []):
            ax.add_patch(_rect(z.x - z.w / 2, z.y - z.d / 2, z.w, z.d,
                               "goldenrod"))
        for t in getattr(cfg.arena, "water_towers", []):
            ax.scatter([t.x], [t.y], marker="o", s=60, color="dodgerblue",
                       alpha=0.55, zorder=1)
    # everyone else, faintly, so the changed animal is read against its world
    others = np.setdiff1d(np.arange(res.xy_a.shape[1]), res.rows)
    for i in others:
        ax.plot(res.xy_a[:, i, 0], res.xy_a[:, i, 1], color="gray", lw=0.4,
                alpha=0.25, zorder=2)
    for i in res.rows:
        ax.plot(res.xy_a[:, i, 0], res.xy_a[:, i, 1], color=AS_IT_WAS, lw=1.1,
                alpha=0.85, zorder=3, label="as it was")
        ax.plot(res.xy_b[:, i, 0], res.xy_b[:, i, 1], color=WITH_TWEAK, lw=1.1,
                alpha=0.85, zorder=4, linestyle="--", label="with the change")
        ax.scatter(*res.xy_a[0, i], marker="*", s=220, color="white",
                   edgecolors="black", zorder=6, label="the fork")
        ax.scatter(*res.xy_a[-1, i], marker="o", s=70, color=AS_IT_WAS,
                   zorder=6)
        ax.scatter(*res.xy_b[-1, i], marker="o", s=70, color=WITH_TWEAK,
                   zorder=6)
    hnd, lab = ax.get_legend_handles_labels()
    keep = dict(zip(lab, hnd))
    ax.legend(keep.values(), keep.keys(), loc="upper right", fontsize=8,
              framealpha=0.9)
    who = ", ".join(res.sexids[i] for i in res.rows)
    ax.set_title(f"{who}: two versions of the next "
                 f"{res.t[-1] / 3600.0:.1f} h", fontsize=11)

    hours = res.t / 3600.0
    d = res.divergence()
    ax2 = fig.add_subplot(gs[0, 1])
    for i in others:
        ax2.plot(hours, d[:, i], color="gray", lw=0.6, alpha=0.6)
    for i in res.rows:
        ax2.plot(hours, d[:, i], color=WITH_TWEAK, lw=1.6)
    ax2.set_ylabel("distance from\nother self (m)")
    ax2.set_title("red: the changed animal · gray: the others", fontsize=9)

    row = res.rows[0]
    for k, (bar, lab) in enumerate((("hunger", "hunger"),
                                    ("thirst", "thirst"))):
        axb = fig.add_subplot(gs[k + 1, 1], sharex=ax2)
        axb.plot(hours, res.bars_a[bar][:, row], color=AS_IT_WAS, lw=1.3,
                 label="as it was")
        axb.plot(hours, res.bars_b[bar][:, row], color=WITH_TWEAK, lw=1.3,
                 linestyle="--", label="with the change")
        axb.set_ylabel(lab)
        axb.set_ylim(-3, 103)
        if k == 0:
            axb.legend(fontsize=8, loc="upper right")
        if k == 1:
            axb.set_xlabel("hours after the fork")
    fig.suptitle(title or ("Twin experiment — "
                           + "; ".join(t.describe() for t in res.tweaks)),
                 fontsize=12)
    return fig


def _rect(x, y, w, h, colour):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, facecolor=colour, alpha=0.18,
                     edgecolor=colour, lw=1.0, zorder=1)


def save_png(res: TwinResult, path: str, cfg=None,
             title: str | None = None) -> str:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = figure(res, cfg, title)
    FigureCanvasAgg(fig)
    fig.savefig(path)
    return path
