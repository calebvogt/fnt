"""A run as a short animation — a GIF or MP4 to watch, send, or put on a slide.

The live window (``--watch``) is the place to explore a run. A movie is for
showing someone else: the enclosure from above, every animal with a fading
trail, and under it each animal's hunger and thirst. Two or more runs can be
put side by side, which is how a treatment is best seen — the same animals,
the same release, one thing different.

    from fnt.abma.core import movie
    clip = movie.film(cfg, seconds=1800)
    movie.render([clip], "run.gif")

Nothing here needs Qt; frames are drawn off screen with matplotlib.
"""
from __future__ import annotations

import copy
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig, Treatment, zone_doors, zone_sides
from .simulation import Simulation

MALE, FEMALE = "dodgerblue", "tomato"
HUNGER, THIRST = "goldenrod", "lightskyblue"


@dataclass
class Clip:
    """What one run looked like, sampled for drawing."""
    title: str
    cfg: ExperimentConfig
    t: np.ndarray            # (T,) seconds since release
    xy: np.ndarray           # (T, n, 2); NaN where an animal is not present
    heading: np.ndarray      # (T, n)
    hunger: np.ndarray       # (T, n)
    thirst: np.ndarray       # (T, n)
    sex: list
    sexid: list


def arrive_deprived(level: float = 60.0):
    """A ``prepare`` for :func:`film`: every animal is put down this hungry
    and this thirsty (0-100), as after a fast, so foraging starts at once
    instead of hours into the clip."""
    level = float(np.clip(level, 0.0, 100.0))

    def prepare(sim: Simulation) -> None:
        sim.energy[:] = 100.0 - level
        sim.hunger[:] = level
        sim.thirst[:] = level
    return prepare


def anosmic(cfg: ExperimentConfig, dose: float = 1.0) -> ExperimentConfig:
    """A copy of ``cfg`` in which every animal had methimazole before release."""
    out = copy.deepcopy(cfg)
    for g in out.groups:
        g.treatment = Treatment("methimazole", float(dose), -5.0)
    return out


def film(cfg: ExperimentConfig, seconds: float, start_s: float = 0.0,
         sample_s: float = 4.0, title: str | None = None, prepare=None,
         trial_index: int = 0, progress_cb=None) -> Clip:
    """Run ``cfg`` and keep what a camera above the enclosure would have seen
    between ``start_s`` and ``start_s + seconds``.

    ``prepare(sim)`` is called once before the first step, for a state the
    config cannot express (see :func:`arrive_deprived`). The run is trial
    ``trial_index`` of the experiment, so a clip of trial 0 shows the same
    animals doing the same things as trial 0 of the real run.
    """
    sim = Simulation(copy.deepcopy(cfg), trial_index=trial_index)
    if prepare is not None:
        prepare(sim)
    dt = float(cfg.dt)
    every = max(1, int(round(sample_s / dt)))
    first = int(round(start_s / dt))
    n_steps = first + int(round(seconds / dt))
    t, rows = [], []
    for k in range(n_steps):
        sim.step(k * dt, dt)
        done = k + 1
        if done >= first and (done - first) % every == 0:
            t.append(done * dt)
            here = sim.alive & ~np.array([a.removed for a in sim.agents])
            rows.append((np.where(here[:, None], sim.P, np.nan),
                         sim.H.copy(), sim.hunger.copy(), sim.thirst.copy()))
        if progress_cb is not None and done % 500 == 0:
            progress_cb(done / n_steps)
    n = sim.n                                   # animals may have been added

    def pad(i, width=None):
        shape = (len(rows), n) + ((width,) if width else ())
        out = np.full(shape, np.nan)
        for f, r in enumerate(rows):
            out[f, :len(r[i])] = r[i]
        return out

    return Clip(title=title or cfg.name, cfg=cfg, t=np.asarray(t, float),
                xy=pad(0, 2), heading=pad(1), hunger=pad(2), thirst=pad(3),
                sex=[a.sex for a in sim.agents],
                sexid=[a.sexid for a in sim.agents])


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _draw_arena(ax, arena) -> None:
    from matplotlib.patches import Circle, Rectangle

    w, h = float(arena.width), float(arena.height)
    pad = 0.03 * max(w, h)
    ax.set_xlim(-pad, w + pad)
    ax.set_ylim(-pad, h + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.add_patch(Rectangle((0, 0), w, h, fill=False, edgecolor="black",
                           lw=1.6, zorder=1))
    for z in getattr(arena, "resource_zones", []):
        has_water = bool(getattr(z, "water", False))
        ax.add_patch(Rectangle((z.x - z.w / 2, z.y - z.d / 2), z.w, z.d,
                               facecolor=HUNGER, alpha=0.22,
                               edgecolor="none", zorder=1))
        ax.add_patch(Rectangle((z.x - z.w / 2, z.y - z.d / 2), z.w, z.d,
                               fill=False, edgecolor="black", lw=1.2,
                               zorder=2))
        # doorways: a gap in the wall, drawn by painting over it
        gap = max(float(getattr(z, "hole", 0.0762)), 0.02 * max(w, h))
        for side, (px, py) in zip(zone_sides(z), zone_doors(z)):
            if side in ("N", "S"):
                ax.plot([px - gap / 2, px + gap / 2], [py, py], color="white",
                        lw=3.0, solid_capstyle="butt", zorder=2.5)
            else:
                ax.plot([px, px], [py - gap / 2, py + gap / 2], color="white",
                        lw=3.0, solid_capstyle="butt", zorder=2.5)
        if has_water:
            ax.add_patch(Circle((z.x, z.y), 0.12 * min(z.w, z.d),
                                facecolor=THIRST, edgecolor="black", lw=0.6,
                                zorder=2))
        ax.text(z.x, z.y - 0.17 * z.d, "food + water" if has_water else "food",
                ha="center", va="top", fontsize=6, color="0.25", zorder=2)
    look = {"food": (HUNGER, 0.75, "food"),
            "water": (THIRST, 0.9, "water"),
            "nest": ("0.88", 1.0, "nest")}
    small = 0.012 * max(w, h)
    for o in arena.objects:
        col, alpha, word = look.get(o.kind, ("0.8", 1.0, o.kind))
        r = max(float(o.radius), small)
        ax.add_patch(Circle((o.x, o.y), r, facecolor=col, alpha=alpha,
                            edgecolor="black", lw=0.8, zorder=2))
        ax.text(o.x, o.y - r - 0.012 * h, word, ha="center", va="top",
                fontsize=7, color="0.25", zorder=2)
    for tw in getattr(arena, "water_towers", []):
        r = max(float(tw.radius), small)
        ax.add_patch(Circle((tw.x, tw.y), r, facecolor=THIRST,
                            edgecolor="black", lw=0.8, zorder=2))
        ax.text(tw.x, tw.y - r - 0.012 * h, "water", ha="center", va="top",
                fontsize=7, color="0.25", zorder=2)


class _Panel:
    """One clip's two axes and the artists that move."""

    def __init__(self, fig, gs_arena, gs_bars, clip: Clip, trail_frames: int):
        from matplotlib.collections import LineCollection

        self.clip = clip
        self.trail = trail_frames
        n = clip.xy.shape[1]
        cols = [MALE if s == "M" else FEMALE for s in clip.sex]
        arena = clip.cfg.arena
        self.tick = 0.045 * max(arena.width, arena.height)

        ax = fig.add_subplot(gs_arena)
        _draw_arena(ax, arena)
        ax.set_title(clip.title, fontsize=12, pad=6)
        self.trails = LineCollection([], colors=cols, linewidths=1.1,
                                     alpha=0.4, zorder=3)
        self.noses = LineCollection([], colors="black", linewidths=1.2,
                                    zorder=5)
        ax.add_collection(self.trails)
        ax.add_collection(self.noses)
        self.dots = ax.scatter(np.zeros(n), np.zeros(n), s=70, c=cols,
                               edgecolors="black", linewidths=0.8, zorder=4)

        bx = fig.add_subplot(gs_bars)
        x = np.arange(n)
        self.b_h = bx.bar(x - 0.2, np.zeros(n), 0.38, color=HUNGER,
                          edgecolor="black", linewidth=0.5, label="hunger")
        self.b_t = bx.bar(x + 0.2, np.zeros(n), 0.38, color=THIRST,
                          edgecolor="black", linewidth=0.5, label="thirst")
        bx.set_ylim(0, 100)
        bx.set_xlim(-0.6, n - 0.4)
        bx.set_yticks([0, 50, 100])
        bx.tick_params(axis="y", labelsize=7)
        bx.set_xticks(x)
        bx.set_xticklabels(clip.sexid, fontsize=7)
        for lab, c in zip(bx.get_xticklabels(), cols):
            lab.set_color(c)
        for side in ("top", "right"):
            bx.spines[side].set_visible(False)
        bx.legend(loc="upper right", fontsize=7, frameon=False, ncol=2,
                  bbox_to_anchor=(1.0, 1.32), handlelength=1.0,
                  columnspacing=0.8)

    def show(self, f: int) -> None:
        c = self.clip
        f = min(f, len(c.t) - 1)
        p = c.xy[f]
        self.dots.set_offsets(np.where(np.isnan(p), -1e6, p))
        lo = max(0, f - self.trail)
        self.trails.set_segments(
            [c.xy[lo:f + 1, i] for i in range(p.shape[0])
             if not np.isnan(c.xy[lo:f + 1, i]).any()] or [])
        tip = p + self.tick * np.stack([np.cos(c.heading[f]),
                                        np.sin(c.heading[f])], axis=1)
        self.noses.set_segments([np.stack([p[i], tip[i]])
                                 for i in range(p.shape[0])
                                 if not np.isnan(p[i]).any()])
        for bars, vals in ((self.b_h, c.hunger[f]), (self.b_t, c.thirst[f])):
            for rect, v in zip(bars, np.nan_to_num(vals)):
                rect.set_height(float(v))


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min after release"
    return f"{m} min {s:02d} s after release"


def render(clips: list, path: str, fps: int = 15, trail_s: float = 90.0,
           caption: str | None = None, dpi: int = 100) -> str:
    """Write ``clips`` side by side to ``path`` (``.gif``, or ``.mp4`` when
    ffmpeg is installed). Returns the path actually written.

    Clips are shown frame for frame, so they should have been filmed with the
    same ``sample_s``; the shorter ones hold their last frame.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PIL import Image

    if not clips:
        raise ValueError("nothing to render: no clips")
    k = len(clips)
    step = float(np.median(np.diff(clips[0].t))) if len(clips[0].t) > 1 else 1.0
    fig = Figure(figsize=(4.7 * k + 0.3, 6.3), dpi=dpi, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, k, height_ratios=[4.6, 1.0], hspace=0.16,
                          wspace=0.08, left=0.06 / k + 0.02, right=0.99,
                          top=0.875, bottom=0.05)
    panels = [_Panel(fig, gs[0, j], gs[1, j], c,
                     max(1, int(round(trail_s / step))))
              for j, c in enumerate(clips)]
    speed = step * fps
    fig.text(0.5, 0.975, caption or clips[0].cfg.name, ha="center", va="top",
             fontsize=13)
    clock = fig.text(0.5, 0.932, "", ha="center", va="top", fontsize=10,
                     color="0.3")
    fig.text(0.99, 0.932, f"{speed:.0f}× real time", ha="right", va="top",
             fontsize=8, color="0.5")

    n_frames = max(len(c.t) for c in clips)
    longest = max(clips, key=lambda c: len(c.t))

    def frames():
        for f in range(n_frames):
            for p in panels:
                p.show(f)
            clock.set_text(_clock(longest.t[f]))
            canvas.draw()
            yield np.asarray(canvas.buffer_rgba())

    if path.lower().endswith(".mp4") and shutil.which("ffmpeg"):
        w, h = canvas.get_width_height()
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
             "-vcodec", "libx264", path], stdin=subprocess.PIPE)
        for buf in frames():
            proc.stdin.write(buf.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg could not write " + path)
        return path
    if not path.lower().endswith(".gif"):
        path = path.rsplit(".", 1)[0] + ".gif"
    # one palette for the whole film, taken from a busy frame, so colours do
    # not shimmer from frame to frame
    imgs = [Image.fromarray(buf[..., :3].copy()) for buf in frames()]
    palette = imgs[len(imgs) // 2].quantize(colors=96)
    # The page must come out pure white. The quantiser keeps several
    # near-white entries and maps white pixels to whichever it finds first,
    # which leaves a faint grey page — visible on a white slide. Make every
    # near-white entry white, so it cannot pick a wrong one.
    pal = np.array(palette.getpalette(), np.uint8).reshape(-1, 3)
    pal[pal.min(axis=1) >= 248] = 255
    palette.putpalette(pal.ravel().tolist())
    out = [im.quantize(palette=palette, dither=Image.Dither.NONE)
           for im in imgs]
    out[0].save(path, save_all=True, append_images=out[1:],
                duration=int(round(1000 / fps)), loop=0, optimize=False)
    return path
