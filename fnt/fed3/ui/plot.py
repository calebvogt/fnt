"""Cumulative pellets per device, read from the run folder's data copies."""
import os
import time

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure

from ..device_sim import parse_iso
from . import theme

REFRESH_SECONDS = 10


class PelletPlot(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(6, 3.5), facecolor=theme.BACKGROUND)
        super().__init__(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.sizes = {}
        self.series = {}
        self.phase_marks = []
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

    def update_plot(self, run, zone):
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
        ax.set_ylabel("pellets", color=theme.TEXT, fontsize=9)
        ax.set_title(f"Cumulative pellets ({run['t0']:%Z}, standard time)",
                     color=theme.TEXT, fontsize=9)
        ax.xaxis.set_major_formatter(DateFormatter("%a %H:%M", tz=zone))
        if any(self.series.values()):
            ax.legend(facecolor=theme.PANEL, edgecolor=theme.BORDER, labelcolor=theme.TEXT,
                      fontsize=8, loc="upper left")
        self.figure.autofmt_xdate()
        self.figure.tight_layout()
        self.draw_idle()
