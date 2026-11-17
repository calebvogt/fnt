"""Cumulative pellets per device, read from the run folder's data copies.

The time axis can be shown in standard time (the devices' clocks), local time
with daylight saving, or UTC; see ZONES.
"""
import os
import time
from datetime import timedelta, timezone, tzinfo

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure

from ..schedule import parse_iso
from . import theme

REFRESH_SECONDS = 10
# Phase changes this close together on different devices draw as one line.
SAME_CHANGE = timedelta(minutes=10)
# How much of the run the plot shows: (label, hours back from now; None for all of it).
SPANS = (("the whole run", None), ("the last 7 days", 7 * 24), ("the last 24 hours", 24))


class LocalTime(tzinfo):
    """The PC's local time, daylight saving included, for each moment."""

    def _local(self, dt):
        return time.localtime(time.mktime(dt.replace(tzinfo=None).timetuple()))

    def utcoffset(self, dt):
        return timedelta(seconds=self._local(dt).tm_gmtoff)

    def dst(self, dt):
        return timedelta(hours=1) if self._local(dt).tm_isdst > 0 else timedelta(0)

    def tzname(self, dt):
        return self._local(dt).tm_zone

    def fromutc(self, dt):
        stamp = (dt.replace(tzinfo=timezone.utc)).timestamp()
        offset = timedelta(seconds=time.localtime(stamp).tm_gmtoff)
        return (dt + offset).replace(tzinfo=self)


def offset_text(zone, moment):
    minutes = int(moment.astimezone(zone).utcoffset().total_seconds() // 60)
    sign = "+" if minutes >= 0 else "−"
    return f"UTC{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def zones(standard, moment):
    """(key, label, tzinfo) for the plot's time axis. Standard time first: it's
    what the devices' clocks and every schedule time use."""
    local = LocalTime()
    return [
        ("standard", f"Standard time ({offset_text(standard, moment)}), as the devices keep it",
         standard),
        ("local", f"Local time ({offset_text(local, moment)} now), with daylight saving", local),
        ("utc", "UTC", timezone.utc),
    ]


class PelletPlot(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(6, 3.5), facecolor=theme.BACKGROUND)
        super().__init__(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.run_id = None
        self._reset()

    def _reset(self):
        self.sizes = {}
        self.series = {}
        self.phase_marks = []       # (moment, phase index, phase name, logical)
        self.stop_mark = None
        self.last_draw = 0.0

    def _read(self, logical, path, mark_stop):
        """New rows since the last read: dispense times and phase changes."""
        try:
            size = os.path.getsize(path)
        except OSError:
            return
        offset = self.sizes.get(path, 0)
        if size == offset:
            return
        with open(path, encoding="ascii", errors="replace") as handle:
            handle.seek(offset)
            text = handle.read()
        # Only whole lines; a partly written row is read next time.
        end = text.rfind("\n") + 1
        self.sizes[path] = offset + len(text[:end].encode("ascii", "replace"))
        times = self.series.setdefault(logical, [])
        for line in text[:end].splitlines():
            parts = line.split(",", 6)
            if len(parts) < 6 or parts[0] == "time":
                continue
            moment = parse_iso(parts[0])
            if moment is None:
                continue
            if parts[4] == "dispense":
                times.append(moment)
            elif parts[4] == "phase":
                # detail is the last column: "name[;reason][;start=<when>]". A
                # catch-up row is written late; start= is when the phase began.
                detail = line.rsplit(",", 1)[-1].strip().split(";")
                for item in detail[1:]:
                    if item.startswith("start="):
                        moment = parse_iso(item[6:]) or moment
                self.phase_marks.append((moment, int(parts[3] or 0), detail[0], logical))
            elif parts[4] == "run_end" and mark_stop and self.stop_mark is None:
                self.stop_mark = moment

    def update_plot(self, run, zone, now, span_hours=None):
        if run["run_id"] != self.run_id:    # a new run: same device names, new data
            self.run_id = run["run_id"]
            self._reset()
        if time.monotonic() - self.last_draw < REFRESH_SECONDS and self.series:
            return
        self.last_draw = time.monotonic()
        for n, (logical, path) in enumerate(run["copies"].items()):
            self._read(logical, path, mark_stop=n == 0)
        ax = self.axes
        ax.clear()
        ax.set_facecolor(theme.PANEL)
        for spine in ax.spines.values():
            spine.set_color(theme.BORDER)
        ax.tick_params(colors=theme.TEXT, labelsize=8)
        # The axis is the run, t0 to now, whatever the data: with one pellet or
        # none, matplotlib would otherwise pick a week, or some other day.
        t0 = run["t0"].astimezone(zone)
        end = max(now.astimezone(zone), t0 + timedelta(hours=1))
        # A span shows the end of the run, with the count axis starting where
        # the lowest line enters it, so a day's pellets aren't a flat line on
        # top of three weeks' worth.
        start = max(t0, end - timedelta(hours=span_hours)) if span_hours else t0
        ax.set_xlim(start, end)
        most = max((len(times) for times in self.series.values()), default=0)
        least = min((sum(t < start for t in times) for times in self.series.values()),
                    default=0) if start > t0 else 0
        # A little headroom over the top count, more when phase names sit there.
        headroom = 1.18 if self.phase_marks else 1.08
        ax.set_ylim(least, least + max(most - least, 1) * headroom)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        for n, logical in enumerate(run["binding"]):
            times = [t.astimezone(zone) for t in self.series.get(logical, [])]
            if not times:
                continue
            # From 0 at t0, a step per pellet, then level until now.
            xs = [t0, *times, max(end, times[-1])]
            ys = [0, *range(1, len(times) + 1), len(times)]
            ax.step(xs, ys, where="post", color=theme.SERIES[n % len(theme.SERIES)],
                    label=logical, linewidth=1.5)
        self._draw_phases(ax, run, zone, start, end)
        if self.stop_mark:
            # The devices keep feeding after Stop, so data carries on past it.
            ax.axvline(self.stop_mark.astimezone(zone), color=theme.BAD, linewidth=1,
                       linestyle="--", label="Stop")
        if not any(self.series.values()):
            ax.set_yticks([])
            ax.text(0.5, 0.5, "No pellets yet", transform=ax.transAxes,
                    ha="center", va="center", color=theme.MUTED)
        ax.set_ylabel("pellets", color=theme.TEXT, fontsize=9)
        ax.xaxis.set_major_formatter(DateFormatter("%a %H:%M", tz=zone))
        if any(self.series.values()) or self.stop_mark:
            # Beside the plot, not on it: inside, it covers the lines it labels.
            ax.legend(facecolor=theme.PANEL, edgecolor=theme.BORDER, labelcolor=theme.TEXT,
                      fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        self.figure.autofmt_xdate()
        self.figure.tight_layout()
        self.draw_idle()

    def _draw_phases(self, ax, run, zone, start, end):
        """A line where the protocol moves on to another phase, named at the top.
        Devices that change together share one line; one that changes on its
        own gets a line in its colour."""
        colours = {logical: theme.SERIES[n % len(theme.SERIES)]
                   for n, logical in enumerate(run["binding"])}
        changes = []        # [moment, phase, name, {logical}]
        for moment, phase, name, logical in sorted(self.phase_marks, key=lambda m: m[0]):
            for change in changes:
                if change[1] == phase and abs(moment - change[0]) <= SAME_CHANGE:
                    change[3].add(logical)
                    break
            else:
                changes.append([moment, phase, name, {logical}])
        named = set()
        for moment, phase, name, logicals in changes:
            x = moment.astimezone(zone)
            if not start <= x <= end:
                continue
            alone = len(logicals) == 1 and len(colours) > 1
            colour = colours.get(next(iter(logicals)), theme.TEXT) if alone else theme.TEXT
            # Solid, so it isn't mistaken for Stop's dashed line.
            ax.axvline(x, color=colour, linewidth=1.2, alpha=0.6)
            if phase in named:      # name each phase once, at its first line
                continue
            named.add(phase)
            ax.text(x, 0.98, f" {phase + 1}: {name}", transform=ax.get_xaxis_transform(),
                    ha="left", va="top", fontsize=8, color=colour, clip_on=True,
                    bbox={"facecolor": theme.PANEL, "edgecolor": "none", "pad": 1})
