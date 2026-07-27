"""Cumulative-pellet plot for the FED3 tab.

Two problems with the previous rendering are fixed here.

The x-axis never advanced: limits were pinned to the first and last *event*, so
between pellets the axis froze and a device that stopped responding looked
identical to one that was simply idle. The right edge now tracks the current
time whenever a device is live, so elapsed time is always visible and a flat
line reads unambiguously as "no pellets since".

Styling was applied ad hoc on every redraw, with a legend and tick colours
re-set each frame and a hard-coded 5-minute floor on the range. Style now lives
in one place, the axis is configured once, and only the data changes between
frames.
"""

from datetime import datetime, timedelta

import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

BACKGROUND = "#2b2b2b"
AXES_FACE = "#1e1e1e"
FOREGROUND = "#e0e0e0"
MUTED = "#8a8a8a"
GRID = "#3a3a3a"
SPINE = "#4a4a4a"

# Qualitative palette chosen to stay distinguishable on the dark axes and to
# survive being printed in greyscale.
SERIES_COLORS = [
    "#4fc3f7", "#81c784", "#ffb74d", "#e57373",
    "#ba68c8", "#4db6ac", "#fff176", "#f06292",
]

# Time windows offered in the toolbar: (label, hours or None for everything).
WINDOWS = [("Last hour", 1), ("Last 6 h", 6), ("Last 24 h", 24),
           ("Last 3 days", 72), ("Entire session", None)]

DEFAULT_DARK_CYCLE = (19, 7)        # lights off at 19:00, on at 07:00


class PlotSeries:
    """One device's contribution to the plot."""

    __slots__ = ("name", "times", "is_live", "start_time")

    def __init__(self, name, times, is_live=False, start_time=None):
        self.name = name
        self.times = times                  # datetimes of pellet events
        self.is_live = is_live
        self.start_time = start_time


class FedPlotManager:
    """Renders cumulative pellet counts against wall-clock time."""

    def __init__(self, canvas, ax, placeholder):
        self.canvas = canvas
        self.ax = ax
        self.placeholder = placeholder
        self.window_hours = None
        self.dark_cycle = DEFAULT_DARK_CYCLE
        self.show_dark_cycle = True
        self._style_axes()

    # --- configuration ----------------------------------------------------

    def set_window(self, hours):
        self.window_hours = hours

    def set_dark_cycle(self, start_hour, end_hour, enabled=True):
        self.dark_cycle = (start_hour, end_hour)
        self.show_dark_cycle = enabled

    # --- rendering --------------------------------------------------------

    def update(self, series):
        """Redraw from a list of :class:`PlotSeries`."""
        drawable = [s for s in series if s.times or s.is_live]
        self.placeholder.setVisible(not drawable)
        self.canvas.setVisible(bool(drawable))
        if not drawable:
            return

        self.ax.clear()
        self._style_axes()

        now = datetime.now()
        start, end = self._time_range(drawable, now)

        if self.show_dark_cycle:
            self._shade_dark_cycle(start, end)

        max_count = 0
        for index, item in enumerate(drawable):
            color = SERIES_COLORS[index % len(SERIES_COLORS)]
            max_count = max(max_count, self._draw_series(item, color, start, end, now))

        self.ax.set_xlim(mdates.date2num(start), mdates.date2num(end))
        self.ax.set_ylim(0, max(max_count * 1.1, 5))
        self.ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

        locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
        self.ax.xaxis.set_major_locator(locator)
        self.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        legend = self.ax.legend(
            loc="upper left", frameon=True, fontsize=8,
            facecolor=AXES_FACE, edgecolor=SPINE, labelcolor=FOREGROUND)
        legend.get_frame().set_alpha(0.85)

        self.canvas.figure.tight_layout(pad=1.2)
        self.canvas.draw_idle()

    def _draw_series(self, item, color, start, end, now):
        """Draw one device; returns its highest cumulative count in view."""
        visible = [t for t in item.times if start <= t <= end]
        # Pellets before the window still count toward the cumulative total.
        base = sum(1 for t in item.times if t < start)

        if not visible:
            # A live device with no pellets in view is drawn as a flat line at
            # its carried-over total, so it is visibly present rather than absent.
            if item.is_live:
                anchor = max(start, item.start_time or start)
                self.ax.plot([mdates.date2num(anchor), mdates.date2num(now)],
                             [base, base], "-", color=color, linewidth=1.6,
                             alpha=0.85, label=item.name)
            return base

        counts = list(range(base + 1, base + len(visible) + 1))
        times = [mdates.date2num(t) for t in visible]

        self.ax.plot(times, counts, drawstyle="steps-post", color=color,
                     linewidth=1.8, label=item.name)
        self.ax.plot(times, counts, linestyle="none", marker="o", markersize=3.5,
                     color=color, alpha=0.9)

        # Extend the trace to the present so the gap since the last pellet is
        # legible rather than implied by empty space.
        if item.is_live and visible[-1] < now:
            self.ax.plot([times[-1], mdates.date2num(now)], [counts[-1], counts[-1]],
                         "-", color=color, linewidth=1.8, alpha=0.5)
        return counts[-1]

    def _time_range(self, series, now):
        """Window to display, always ending at *now* while anything is live."""
        earliest = None
        for item in series:
            candidates = [t for t in (item.start_time,) if t is not None]
            if item.times:
                candidates.append(item.times[0])
            for candidate in candidates:
                if earliest is None or candidate < earliest:
                    earliest = candidate
        if earliest is None:
            earliest = now

        latest = now if any(s.is_live for s in series) else max(
            (item.times[-1] for item in series if item.times), default=now)

        if self.window_hours is not None:
            earliest = max(earliest, latest - timedelta(hours=self.window_hours))

        # Never hand matplotlib a zero-width range: date formatting degenerates.
        if (latest - earliest).total_seconds() < 60:
            latest = earliest + timedelta(minutes=1)
        pad = (latest - earliest) * 0.02
        return earliest - pad, latest + pad

    def _shade_dark_cycle(self, start, end):
        """Shade lights-off periods behind the traces."""
        lights_off, lights_on = self.dark_cycle
        cursor = start.replace(hour=lights_off, minute=0, second=0, microsecond=0)
        if cursor > start:
            cursor -= timedelta(days=1)

        # Dark phase may wrap midnight; its length is the forward distance from
        # lights-off to lights-on.
        hours = (lights_on - lights_off) % 24 or 24

        while cursor < end:
            self.ax.axvspan(
                mdates.date2num(max(cursor, start)),
                mdates.date2num(min(cursor + timedelta(hours=hours), end)),
                color="#000000", alpha=0.28, zorder=0, linewidth=0)
            cursor += timedelta(days=1)

    def _style_axes(self):
        self.ax.set_facecolor(AXES_FACE)
        self.ax.set_title("Cumulative pellets retrieved", color=FOREGROUND,
                          fontsize=11, pad=8)
        self.ax.set_xlabel("Time", color=MUTED, fontsize=9)
        self.ax.set_ylabel("Pellets", color=MUTED, fontsize=9)
        self.ax.tick_params(colors=MUTED, labelsize=8, length=3)
        self.ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
        self.ax.set_axisbelow(True)
        for side, spine in self.ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_edgecolor(SPINE)
