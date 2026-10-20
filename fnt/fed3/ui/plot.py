"""Cumulative pellets per device, read from the run folder's data copies.

The time axis can be shown in standard time (the devices' clocks), local time
with daylight saving, or UTC; see ZONES.
"""
import os
import time
from datetime import timedelta, timezone, tzinfo

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure

from ..device_sim import parse_iso
from . import theme

REFRESH_SECONDS = 10


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
        self.phase_marks = []
        self.stop_mark = None
        self.last_draw = 0.0

    def _read(self, logical, path, mark_phases):
        """New rows since the last read: (dispense times, phase-change times)."""
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
            elif parts[4] == "phase" and mark_phases:
                self.phase_marks.append(moment)
            elif parts[4] == "run_end" and mark_phases and self.stop_mark is None:
                self.stop_mark = moment

    def update_plot(self, run, zone):
        if run["run_id"] != self.run_id:    # a new run: same device names, new data
            self.run_id = run["run_id"]
            self._reset()
        if time.monotonic() - self.last_draw < REFRESH_SECONDS and self.series:
            return
        self.last_draw = time.monotonic()
        for n, (logical, path) in enumerate(run["copies"].items()):
            self._read(logical, path, mark_phases=n == 0)
        ax = self.axes
        ax.clear()
        ax.set_facecolor(theme.PANEL)
        for spine in ax.spines.values():
            spine.set_color(theme.BORDER)
        ax.tick_params(colors=theme.TEXT, labelsize=8)
        for n, logical in enumerate(run["binding"]):
            times = [t.astimezone(zone) for t in self.series.get(logical, [])]
            if not times:
                continue
            ax.step(times, range(1, len(times) + 1), where="post",
                    color=theme.SERIES[n % len(theme.SERIES)], label=logical, linewidth=1.5)
        for mark in self.phase_marks:
            ax.axvline(mark.astimezone(zone), color=theme.MUTED, linewidth=0.8, linestyle=":")
        if self.stop_mark:
            # The devices keep feeding after Stop, so data carries on past it.
            ax.axvline(self.stop_mark.astimezone(zone), color=theme.BAD, linewidth=1,
                       linestyle="--", label="Stop")
        ax.set_ylabel("pellets", color=theme.TEXT, fontsize=9)
        ax.xaxis.set_major_formatter(DateFormatter("%a %H:%M", tz=zone))
        if any(self.series.values()) or self.stop_mark:
            ax.legend(facecolor=theme.PANEL, edgecolor=theme.BORDER, labelcolor=theme.TEXT,
                      fontsize=8, loc="upper left")
        self.figure.autofmt_xdate()
        self.figure.tight_layout()
        self.draw_idle()
