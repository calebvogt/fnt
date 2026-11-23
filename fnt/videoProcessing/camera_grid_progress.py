"""A window showing camera-grid encodes as they run.

The command-line grid (:mod:`fnt.videoProcessing.camera_grid_cli`) runs without
a window of its own; this one sits beside it so a run in the background still
shows on screen what it is doing. It reads only what the runs write anyway --
each trial's ``grid/clock/<trial>_grid_cli.log``, the day's ffmpeg log and the
finished video -- so it can be opened on runs already going, closed and opened
again at any time, and never slows them down.

One row per day: its stage (waiting, reading clocks, encoding, checking, done,
failed, stopped), a progress bar and the detail (video time encoded, speed,
time left; the clock check's result when done), and a bar for all days.

    python -m fnt.videoProcessing.camera_grid_progress <trial folder> --day 2026-02-08 [--day ...]
    python -m fnt.videoProcessing.camera_grid_progress <trial folder> --all-days
    python -m fnt.videoProcessing.camera_grid_progress --queue days.csv --root <folder of trial folders>

A ``--queue`` CSV has ``trial,date`` columns (``VC-T005,2026-02-08``), the
trial being the start of a folder name under ``--root``. The grid CLI opens
this window itself unless given ``--no-window``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
import sys
import time

STAGES = ("waiting", "reading clocks", "encoding", "checking", "done", "failed", "stopped")
_TIME = re.compile(r"time=(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_SPEED = re.compile(r"speed=\s*([\d.]+)x")
_STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)$")
STALE_S = 300.0            # encoded time not advancing this long, with no result: the run stopped


class Job:
    """One day's grid: where its files are and what the logs say about it."""

    def __init__(self, trial_folder, trial, date):
        self.folder = trial_folder
        self.trial = trial
        self.date = date                                   # datetime.date
        self.tag = f"{trial}_{date:%Y-%m-%d}"
        self.clock = os.path.join(trial_folder, "grid", "clock")
        self.stage = "waiting"
        self.done_s = 0.0                                  # video time encoded
        self.total_s = None                                # video time to encode
        self.speed = None
        self.readings = (0, 0)
        self.result = ""                                   # the clock check, or why it failed
        self.final = None
        self.started = None

    @property
    def fraction(self):
        if self.stage == "done":
            return 1.0
        if self.stage == "reading clocks" and self.readings[1]:
            return 0.0
        if self.stage in ("encoding", "checking", "stopped") and self.total_s:
            return min(1.0, self.done_s / self.total_s)
        return 0.0


def _hms(s):
    s = max(0, int(round(s)))
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"


def apply_log(job, lines):
    """Update `job` from its trial's grid CLI log lines (all days mixed; the last event wins)."""
    tag = job.tag
    for line in lines:
        m = _STAMP.match(line.rstrip())
        if not m:
            continue
        when, msg = m.groups()
        if not msg.strip().startswith(tag):
            continue
        body = msg.strip()[len(tag):]
        if "reading" in body and "clocks every" in body:
            job.stage, job.readings = "reading clocks", (0, 0)
            job.started = job.started or when
        elif "readings (" in body:
            r = re.search(r"(\d+)/(\d+) readings", body)
            if r and job.stage == "reading clocks":
                job.readings = (int(r.group(1)), int(r.group(2)))
        elif re.search(r"encoding (\(|\d\d:\d\d)", body):
            job.stage, job.done_s, job.result = "encoding", 0.0, ""
            job.started = when
            f = re.search(r"-> (.+?\.mp4)", body)
            if f:
                job.final = f.group(1).strip()
            w = re.search(r"(\d\d:\d\d) to (\d\d:\d\d)", body)
            if w:
                job.total_s = _span_s(w.group(1), w.group(2))
        elif "footage" in body and " to " in body and "left out" in body:
            w = re.search(r"footage (\d\d:\d\d) to (\d\d:\d\d)", body)
            if w:
                job.total_s = _span_s(w.group(1), w.group(2))
        elif "encoded in" in body or "grid exists" in body:
            job.stage, job.done_s = "checking", job.total_s or job.done_s
        elif "clocks checked" in body or "no clock could be read back" in body:
            job.stage = "done"
            job.result = body.split(":", 1)[-1].strip()
        elif "ffmpeg failed" in body or "failed:" in body:
            job.stage, job.result = "failed", body.split(":", 1)[-1].strip()


def _span_s(a, b):
    h0, m0 = map(int, a.split(":"))
    h1, m1 = map(int, b.split(":"))
    s = (h1 * 60 + m1 - h0 * 60 - m0) * 60
    return s if s > 0 else s + 86400                       # "to 00:00" is the next midnight


def parse_ffmpeg_tail(text):
    """(seconds encoded, speed) from the end of an ffmpeg log, or (None, None)."""
    times, speeds = _TIME.findall(text), _SPEED.findall(text)
    t = None
    if times:
        h, m, s = times[-1]
        t = int(h) * 3600 + int(m) * 60 + float(s)
    return t, (float(speeds[-1]) if speeds else None)


def _tail(path, n=4096):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - n))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def refresh(jobs, now=None):
    """Re-read every job's logs."""
    now = now or time.time()
    logs = {}
    for job in jobs:
        # the day's own log (one writer) when the run wrote one, else the trial's shared log
        path = os.path.join(job.clock, f"{job.tag}_progress.log")
        if not os.path.exists(path):
            path = os.path.join(job.clock, f"{job.trial}_grid_cli.log")
        if path not in logs:
            try:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    logs[path] = fh.readlines()
            except OSError:
                logs[path] = []
        job.stage, job.result = "waiting", ""
        apply_log(job, logs[path])
        if job.total_s is None:
            job.total_s = _footage_seconds(job)
        if job.stage in ("encoding", "checking"):
            ff = os.path.join(job.clock, f"{job.tag}_ffmpeg.log")
            t, sp = parse_ffmpeg_tail(_tail(ff))
            if t is not None and job.stage == "encoding":
                job.done_s, job.speed = t, sp
            # A file's modified time on a network share is not updated while
            # ffmpeg holds it open, so a stall is judged by the encoded time
            # no longer advancing between refreshes.
            seen = getattr(job, "_seen", None)
            if seen is None or seen[0] != (job.stage, t):
                job._seen = ((job.stage, t), now)
            elif job.stage == "encoding" and now - seen[1] > STALE_S:
                job.stage = "stopped"
    return jobs


def _footage_seconds(job):
    """The day's video length from its clock readings (black left out), when the log has not said."""
    try:
        from .camera_grid_cli import _read_probes, footage_window
        probes = _read_probes(os.path.join(job.clock, f"{job.tag}_clock_readings.csv"))
    except Exception:
        return None
    start = datetime.datetime.combine(job.date, datetime.time.min)
    s, e = footage_window(probes, start, start + datetime.timedelta(days=1))
    return (e - s).total_seconds()


def detail(job):
    if job.stage == "reading clocks":
        d, n = job.readings
        return f"{d}/{n} readings" if n else "starting"
    if job.stage in ("encoding", "stopped"):
        txt = _hms(job.done_s) + (f" of {_hms(job.total_s)}" if job.total_s else "")
        if job.stage == "encoding" and job.speed:
            txt += f" · {job.speed:.2f}x"
            if job.total_s and job.speed > 0:
                txt += f" · {_hms((job.total_s - job.done_s) / job.speed)} left"
        return txt
    if job.stage == "checking":
        return "reading the clocks back off the grid"
    return job.result


# ------------------------------------------------------------------------------------------------ jobs

def jobs_from_queue(csv_path, root):
    folders = {d: os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    out = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            folder = next((p for d, p in sorted(folders.items()) if d.startswith(r["trial"] + "_")
                           or d == r["trial"]), None)
            if folder:
                out.append(Job(folder, r["trial"], datetime.date.fromisoformat(r["date"])))
    return out


def jobs_for_trial(folder, days=(), all_days=False):
    folder = os.path.abspath(folder)
    m = re.match(r"(VC-T\d+|T\d+)", os.path.basename(folder))
    trial = m.group(1) if m else os.path.basename(folder)
    dates = [datetime.date.fromisoformat(d) for d in days]
    if all_days or not dates:
        clock = os.path.join(folder, "grid", "clock")
        found = set()
        if os.path.isdir(clock):
            for f in os.listdir(clock):
                r = re.match(re.escape(trial) + r"_(\d{4}-\d\d-\d\d)_clock_readings\.csv$", f)
                if r:
                    found.add(datetime.date.fromisoformat(r.group(1)))
        dates = sorted(set(dates) | found)
    return [Job(folder, trial, d) for d in dates]


# ------------------------------------------------------------------------------------------------ window

def run_window(jobs, title, watch_pid=None, interval_ms=10000):
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (QApplication, QHeaderView, QLabel, QProgressBar, QTableWidget,
                                 QTableWidgetItem, QVBoxLayout, QWidget)
    try:
        from fnt import tool_host as th
        th._prepare_windows_taskbar()
        th._enable_hidpi()
    except Exception:
        pass
    app = QApplication.instance() or QApplication(sys.argv[:1])
    try:
        from fnt.theme import apply_dark_theme
        apply_dark_theme(app)
    except Exception:
        pass

    win = QWidget()
    win.setWindowTitle(title)
    lay = QVBoxLayout(win)
    head = QLabel()
    head.setStyleSheet("font-size: 14px; font-weight: bold;")
    total = QProgressBar()
    total.setRange(0, 1000)
    total.setTextVisible(True)
    table = QTableWidget(len(jobs), 4)
    table.setHorizontalHeaderLabels(["Day", "Stage", "Progress", "Detail"])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionMode(QTableWidget.NoSelection)
    hdr = table.horizontalHeader()
    hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
    hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
    hdr.setSectionResizeMode(2, QHeaderView.Fixed)
    table.setColumnWidth(2, 220)
    hdr.setSectionResizeMode(3, QHeaderView.Stretch)
    bars = []
    for i, job in enumerate(jobs):
        name = os.path.basename(job.final) if job.final else job.tag
        table.setItem(i, 0, QTableWidgetItem(name))
        table.setItem(i, 1, QTableWidgetItem(""))
        b = QProgressBar()
        b.setRange(0, 1000)
        bars.append(b)
        table.setCellWidget(i, 2, b)
        table.setItem(i, 3, QTableWidgetItem(""))
    foot = QLabel()
    foot.setStyleSheet("color: #999999;")
    lay.addWidget(head)
    lay.addWidget(total)
    lay.addWidget(table, 1)
    lay.addWidget(foot)
    colours = {"done": "#4caf50", "failed": "#ef5350", "stopped": "#ef5350", "encoding": "#0078d4",
               "checking": "#26a69a", "reading clocks": "#ffb74d", "waiting": "#777777"}

    def tick():
        refresh(jobs)
        n_done = sum(j.stage == "done" for j in jobs)
        running = [j for j in jobs if j.stage in ("reading clocks", "encoding", "checking")]
        frac = sum(j.fraction for j in jobs) / max(1, len(jobs))
        total.setValue(int(frac * 1000))
        total.setFormat(f"{n_done} of {len(jobs)} days done · {frac:.0%} of the video")
        speeds = [j.speed for j in running if j.stage == "encoding" and j.speed]
        alive = watch_pid is None or _pid_alive(watch_pid)
        state = "running" if running and alive else ("finished" if n_done == len(jobs) else
                                                      ("stopped" if not alive else "waiting"))
        head.setText(f"{len(running)} running, {n_done} done, "
                     f"{sum(j.stage == 'waiting' for j in jobs)} waiting"
                     + (f" · {sum(speeds):.1f}x real time in all" if speeds else "")
                     + ("" if state == "running" else f" · {state}"))
        win.setWindowTitle(f"{title} — {n_done}/{len(jobs)} done" + ("" if state == "running" else f" ({state})"))
        for i, job in enumerate(jobs):
            if job.final:
                table.item(i, 0).setText(os.path.basename(job.final))
            it = table.item(i, 1)
            it.setText(job.stage)
            from PyQt5.QtGui import QColor
            it.setForeground(QColor(colours.get(job.stage, "#dddddd")))
            bars[i].setValue(int(job.fraction * 1000))
            bars[i].setFormat(f"{job.fraction:.0%}" if job.stage not in ("waiting", "reading clocks")
                              else (f"{job.readings[0]}/{job.readings[1]}" if job.readings[1] else ""))
            if job.stage == "reading clocks" and job.readings[1]:
                bars[i].setValue(int(1000 * job.readings[0] / job.readings[1]))
            table.item(i, 3).setText(detail(job))
        # A finished run closes its window by itself (after a short look); one with a
        # failed or stopped day stays up so the failure is seen.
        failed = any(j.stage in ("failed", "stopped") for j in jobs)
        if not failed and (state == "finished" or (watch_pid is not None and not alive and not running)):
            if not getattr(win, "_closing", False):
                win._closing = True
                QTimer.singleShot(15000, win.close)
        foot.setText(f"updated {datetime.datetime.now():%H:%M:%S} · reads each trial's grid/clock logs "
                     f"every {interval_ms // 1000} s; closing this window does not stop the encodes")

    tick()
    timer = QTimer(win)
    timer.timeout.connect(tick)
    timer.start(interval_ms)
    win.resize(1100, min(140 + 32 * len(jobs), 900))
    win.show()
    return app.exec_()


def _pid_alive(pid):
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                         # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def launch(trial_folder=None, days=(), queue=None, root=None, watch_pid=None, title=None):
    """Open the window in its own process (the grid CLI calls this). Never raises."""
    import subprocess
    exe = sys.executable
    w = os.path.join(os.path.dirname(exe), "pythonw.exe")
    if os.name == "nt" and os.path.exists(w):
        exe = w
    cmd = [exe, "-m", "fnt.videoProcessing.camera_grid_progress"]
    if queue:
        cmd += ["--queue", queue, "--root", root]
    else:
        cmd += [trial_folder] + [a for d in days for a in ("--day", str(d))]
    if watch_pid:
        cmd += ["--pid", str(watch_pid)]
    if title:
        cmd += ["--title", title]
    try:
        return subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trial", nargs="?", help="trial folder (with grid/clock/)")
    ap.add_argument("--day", action="append", default=[], help="YYYY-MM-DD, repeatable")
    ap.add_argument("--all-days", action="store_true", help="every day with clock readings")
    ap.add_argument("--queue", help="CSV with trial,date columns (several trials)")
    ap.add_argument("--root", help="folder holding the trial folders (with --queue)")
    ap.add_argument("--pid", type=int, help="the run's process: the window says when it has stopped")
    ap.add_argument("--title", help="window title")
    a = ap.parse_args(argv)
    if a.queue:
        root = a.root or os.path.dirname(os.path.abspath(a.queue))
        jobs = jobs_from_queue(a.queue, root)
        title = a.title or f"Camera grids — {os.path.basename(a.queue)}"
    elif a.trial:
        jobs = jobs_for_trial(a.trial, a.day, a.all_days)
        title = a.title or f"Camera grids — {os.path.basename(os.path.abspath(a.trial))}"
    else:
        ap.error("give a trial folder or --queue")
    if not jobs:
        ap.error("no days to show")
    return run_window(jobs, title, watch_pid=a.pid)


if __name__ == "__main__":
    sys.exit(main())
