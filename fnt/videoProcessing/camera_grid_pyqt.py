"""Camera Grid Tool - build one synchronised multi-camera video for scoring.

Takes several folders of NVR/DVR segments -- one per camera -- reconstructs
each camera's wall-clock timeline, and renders them into a single grid video
where every cell shows the same instant, exactly as a live NVR view would.
A camera that dropped out holds its cell black instead of sliding its later
footage earlier.

The synchronisation and FFmpeg logic live in camera_timeline.py and
camera_grid_encode.py, both Qt-free and unit tested; this module is the UI over
them.
"""

import cv2
import datetime
import time
import numpy as np
import os
import shutil
import subprocess
import sys
import tempfile

from PyQt5.QtCore import QMimeData, QThread, QTimer, Qt, QUrl, pyqtSignal
from PyQt5.QtGui import (QColor, QDrag, QFont, QImage, QPainter, QPen,
                         QPixmap)
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QFileDialog, QFrame, QGridLayout, QGroupBox,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QScrollArea, QSizePolicy,
                             QSlider, QSplitter, QTextEdit, QVBoxLayout,
                             QWidget)

from fnt.paths import describe_location, is_network_path
from fnt.theme import BLUE_BUTTON_STYLE as BLUE_BUTTON, apply_dark_theme
from fnt.videoProcessing.camera_grid_encode import (
    EncodeSettings, GridLayout, best_grid, build_chunk_command,
    build_concat_command, chunk_filename, estimate, human_size, human_time)
from PyQt5.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem
from fnt.videoProcessing.camera_timeline import (
    DEFAULT_PROFILE, CameraTrack, TrialTimeline, discover_cameras)

PROFILE_CHOICES = [
    ("ViewTron / DVR (chronological)", "viewtron"),
    ("Alphabetical (no gap detection)", "alphabetical"),
]

GRID_CHOICES = [("1 x 1", 1, 1), ("2 x 1", 1, 2), ("2 x 2", 2, 2),
                ("3 x 2", 2, 3), ("3 x 3", 3, 3), ("4 x 3", 3, 4),
                ("4 x 4", 4, 4), ("5 x 5", 5, 5)]

FPS_CHOICES = [10, 15, 20, 30]

RESOLUTIONS = [("1080p (1920x1080)", 1920, 1080),
               ("720p (1280x720)", 1280, 720),
               ("1440p (2560x1440)", 2560, 1440),
               ("4K (3840x2160)", 3840, 2160)]

QUALITY = [("High (CRF 20)", 20), ("Good (CRF 23)", 23),
           ("Medium (CRF 26)", 26), ("Small (CRF 28)", 28)]


def _no_window():
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------

class ScanWorker(QThread):
    """Probe a camera folder off the GUI thread.

    Every segment needs an ffprobe call to get its duration, and a trial folder
    can hold hundreds of files on a network share, so this would freeze the
    window if run inline.
    """
    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, folder, profile=DEFAULT_PROFILE):
        super().__init__()
        self.folder = folder
        self.profile = profile

    def run(self):
        try:
            def tick(done, total):
                self.progress.emit(f"Probing {os.path.basename(self.folder)}: "
                                   f"{done}/{total} segments")
            track = CameraTrack.from_folder(self.folder, progress=tick,
                                            profile=self.profile)
            # One camera's folder was picked, so the trial is its parent.
            track.trial_root = os.path.dirname(os.path.normpath(self.folder))
            self.done.emit(track)
        except Exception as exc:
            self.failed.emit(str(exc))


class DiscoverWorker(QThread):
    """Find every camera under a trial folder and build all their timelines.

    Groups files by the Camera<N> label they carry, so it copes with per-camera
    subfolders, a flat trial folder, or a mixture -- the footage does not have
    to be sorted by hand first.
    """
    progress = pyqtSignal(str)
    found = pyqtSignal(object)      # one CameraTrack at a time
    done = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, folder, profile=DEFAULT_PROFILE):
        super().__init__()
        self.folder = folder
        self.profile = profile

    def run(self):
        try:
            groups = discover_cameras(self.folder)
            if not groups:
                self.failed.emit("No recognisable camera segments found.")
                return
            self.progress.emit(
                f"Found {len(groups)} camera(s): {', '.join(groups)}")
            for name, paths in groups.items():
                def tick(d, t, n=name):
                    self.progress.emit(f"Probing {n}: {d}/{t} segments")
                track = CameraTrack.from_paths(
                    name, paths, folder=self.folder, progress=tick,
                    profile=self.profile)
                # The folder the user picked IS the trial, whatever the
                # footage inside it is arranged like.
                track.trial_root = os.path.normpath(self.folder)
                if track.segments:
                    self.found.emit(track)
            self.done.emit(len(groups))
        except Exception as exc:
            self.failed.emit(str(exc))


# --------------------------------------------------------------------------
# Grid designer
# --------------------------------------------------------------------------

class GridCell(QFrame):
    """One drop target in the grid designer."""

    assigned = pyqtSignal(int, int, str)
    cleared = pyqtSignal(int, int)

    def __init__(self, row, col, parent=None):
        super().__init__(parent)
        self.row, self.col = row, col
        self.camera = None
        self.thumbnail = None
        self.setAcceptDrops(True)
        self.setMinimumSize(90, 56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFrameShape(QFrame.Box)
        self._refresh_style()

    def _refresh_style(self):
        colour = "#0078d4" if self.camera else "#3f3f3f"
        self.setStyleSheet(f"background-color:#141414; border:2px solid {colour};")

    def set_camera(self, name):
        self.camera = name
        self._refresh_style()
        self.update()

    def set_thumbnail(self, pixmap):
        self.thumbnail = pixmap
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        if self.thumbnail:
            scaled = self.thumbnail.scaled(self.rect().size(), Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        if self.camera:
            p.setPen(QPen(QColor("#ffd400")))
            f = p.font(); f.setBold(True); f.setPointSize(9); p.setFont(f)
            p.drawText(self.rect().adjusted(6, 4, -6, -4),
                       Qt.AlignTop | Qt.AlignLeft, self.camera)
        else:
            p.setPen(QPen(QColor("#555")))
            p.drawText(self.rect(), Qt.AlignCenter, "empty")
        p.end()

    # drag and drop -------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
            self.setStyleSheet("background-color:#1d3550; border:2px solid #4da3ff;")

    def dragLeaveEvent(self, event):
        self._refresh_style()

    def dropEvent(self, event):
        name = event.mimeData().text()
        self._refresh_style()
        if name:
            self.assigned.emit(self.row, self.col, name)
            event.acceptProposedAction()

    def mousePressEvent(self, event):
        # Right-click empties a cell; dragging moves the camera elsewhere.
        if event.button() == Qt.RightButton and self.camera:
            self.cleared.emit(self.row, self.col)
            return
        if event.button() == Qt.LeftButton and self.camera:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(self.camera)
            drag.setMimeData(mime)
            drag.exec_(Qt.MoveAction)


class GridDesigner(QWidget):
    """The rows x cols board of GridCells."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Starts small and grows to fit whatever cameras are added; the
        # combo box in the window shows the same default.
        self.layout_model = GridLayout(2, 2)
        self._grid = QGridLayout(self)
        self._grid.setSpacing(3)
        self.cells = {}
        self.rebuild(2, 2)

    def rebuild(self, rows, cols):
        for cell in self.cells.values():
            cell.setParent(None)
        self.cells.clear()
        self.layout_model.resize(rows, cols)
        for r in range(rows):
            for c in range(cols):
                cell = GridCell(r, c, self)
                cell.assigned.connect(self._on_assigned)
                cell.cleared.connect(self._on_cleared)
                self._grid.addWidget(cell, r, c)
                self.cells[(r, c)] = cell
        self._sync_cells()
        self.changed.emit()

    def _on_assigned(self, row, col, name):
        self.layout_model.place(row, col, name)
        self._sync_cells()
        self.changed.emit()

    def _on_cleared(self, row, col):
        self.layout_model.clear_cell(row, col)
        self._sync_cells()
        self.changed.emit()

    def _sync_cells(self):
        for (r, c), cell in self.cells.items():
            cell.set_camera(self.layout_model.assignments.get((r, c)))

    def apply_layout(self, layout):
        self.layout_model = layout
        self.rebuild(layout.rows, layout.cols)

    def set_thumbnail(self, camera, pixmap):
        cell = self.layout_model.cell_of(camera)
        if cell and cell in self.cells:
            self.cells[cell].set_thumbnail(pixmap)

    def clear_thumbnails(self):
        for cell in self.cells.values():
            cell.set_thumbnail(None)


class CameraList(QListWidget):
    """Source list; items are dragged onto the grid."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setMinimumHeight(110)

    def startDrag(self, actions):
        item = self.currentItem()
        if not item:
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText(item.data(Qt.UserRole))
        drag.setMimeData(mime)
        drag.exec_(Qt.CopyAction)


class CoverageStrip(QWidget):
    """Per-camera footage coverage across the trial, so dropouts are visible
    before committing to a long encode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tracks = []
        self.window = None
        self.setMinimumHeight(70)

    def set_tracks(self, tracks, window):
        self.tracks = tracks
        self.window = window
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1b1b1b"))
        if not self.tracks or not self.window:
            p.setPen(QPen(QColor("#777")))
            p.drawText(self.rect(), Qt.AlignCenter, "Add cameras to see coverage")
            p.end()
            return
        start, end = self.window
        label_w = 62
        row_h = max(10, (self.height() - 6) // max(1, len(self.tracks)))
        p.setFont(QFont("Arial", 7))
        for i, track in enumerate(self.tracks):
            y = 3 + i * row_h
            p.setPen(QPen(QColor("#bbb")))
            p.drawText(2, y + row_h - 3, track.name[:9])
            usable = max(1, self.width() - label_w - 6)
            buckets = track.coverage(start, end, buckets=usable)
            for x, frac in enumerate(buckets):
                if frac > 0.98:
                    col = QColor("#2f9e44")
                elif frac > 0.01:
                    col = QColor("#c9a227")
                else:
                    col = QColor("#b03030")
                p.fillRect(label_w + x, y, 1, row_h - 4, col)
        p.end()


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

class QueuedTrial:
    """One trial captured with the settings it was queued under.

    Everything is snapshotted at add time -- the cameras, the grid and the
    encode settings -- so the controls can be retargeted at the next trial
    without disturbing jobs already waiting. A queue whose jobs read live UI
    state would silently re-interpret them at run time.
    """

    def __init__(self, label, tracks, layout, settings, chunks, out_dir, prefix):
        self.label = label
        # Copied so a later re-calibration cannot rewrite a job already queued.
        self.tracks = {n: t.copy() for n, t in tracks.items()}
        self.layout = layout.copy()
        self.settings = settings.copy()
        self.chunks = list(chunks)
        self.out_dir = out_dir
        self.prefix = prefix

    @property
    def cameras(self):
        return sorted(set(self.layout.assignments.values()))

    @property
    def total_seconds(self):
        return sum((b - a).total_seconds() for a, b in self.chunks)

    def estimate(self):
        return estimate(self.settings, self.layout, self.total_seconds,
                        len(self.layout.assignments))


class GridEncodeWorker(QThread):
    """Encode every queued trial in order, one ffmpeg process at a time.

    A single "Build" is just a one-job queue, so both paths share this code.
    A trial that fails does not stop the ones behind it -- an overnight batch
    should not be lost to one bad trial -- and the summary names whatever
    failed.
    """

    progress = pyqtSignal(str)
    chunk_progress = pyqtSignal(int, int, float)   # index, total, fraction
    job_started = pyqtSignal(int, str)             # job index, label
    job_finished = pyqtSignal(int, bool)           # job index, ok
    ffmpeg_line = pyqtSignal(str)
    finished_all = pyqtSignal(bool, str)

    def __init__(self, jobs):
        super().__init__()
        self.jobs = list(jobs)
        self.cancelled = False
        self._proc = None

    def _join_chunks(self, job, files, work_dir):
        """Stream-copy the finished chunks into one full-trial file."""
        name = chunk_filename(job.prefix, job.chunks[0][0], "continuous", 0, 1)
        final = os.path.join(job.out_dir, name)
        staging = job.settings.stage_output_locally and is_network_path(final)
        out = os.path.join(work_dir, name) if staging else final

        self.progress.emit(f"  joining {len(files)} day(s) into {name} "
                           f"(stream copy, no re-encode)")
        cmd = build_concat_command(files, out,
                                   os.path.join(work_dir, "join.txt"))
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                universal_newlines=True, **_no_window())
        except Exception as exc:
            self.progress.emit(f"      join failed to start: {exc}")
            return None
        for line in iter(self._proc.stdout.readline, ""):
            if self.cancelled:
                self._proc.terminate()
                break
            if line.strip():
                self.ffmpeg_line.emit(line.rstrip() + "\n")
        self._proc.wait()

        if self._proc.returncode != 0 or self.cancelled:
            try:
                if os.path.exists(out):
                    os.remove(out)
            except OSError:
                pass
            return None
        if staging:
            try:
                shutil.move(out, final)
            except Exception as exc:
                self.progress.emit(f"      joined but could not move it: {exc}")
                return None
        self.progress.emit(f"      wrote {name} "
                           f"({human_size(os.path.getsize(final))})")
        return final

    def cancel(self):
        self.cancelled = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def run(self):
        # Concat list files for runs spanning several segments live here and
        # are thrown away with the directory when the batch ends.
        work_dir = tempfile.mkdtemp(prefix="fnt_grid_")
        total_chunks = sum(len(j.chunks) for j in self.jobs) or 1
        done_chunks = 0
        made, failed = [], []
        # Output folders this run brought into existence, so a cancel can take
        # them away again rather than leaving empty directories behind.
        created_dirs = []
        partial = None          # the file currently being written

        for job_index, job in enumerate(self.jobs):
            if self.cancelled:
                break
            self.job_started.emit(job_index, job.label)
            self.progress.emit(
                f"=== {job.label}  ({len(job.chunks)} file(s), "
                f"{len(job.layout.assignments)} camera(s)) ===")
            if not os.path.isdir(job.out_dir):
                created_dirs.append(job.out_dir)
            os.makedirs(job.out_dir, exist_ok=True)
            job_ok = True
            job_files = []          # this trial's finished chunks, in order

            for i, (w0, w1) in enumerate(job.chunks):
                if self.cancelled:
                    break
                name = chunk_filename(job.prefix, w0, job.settings.chunk_mode,
                                      i, len(job.chunks))
                final = os.path.join(job.out_dir, name)
                # Encode locally when the destination is a share, then move.
                staging = (job.settings.stage_output_locally
                           and is_network_path(final))
                out = os.path.join(work_dir, name) if staging else final
                partial = out
                span = (w1 - w0).total_seconds()
                self.progress.emit(
                    f"  [{i + 1}/{len(job.chunks)}] {name}  "
                    f"({w0:%Y-%m-%d %H:%M} -> {w1:%H:%M}, {span / 3600:.1f}h)")

                cmd, n_runs = build_chunk_command(
                    job.tracks, job.layout, job.settings, w0, w1, out,
                    work_dir=work_dir)
                if n_runs == 0:
                    self.progress.emit("      no footage in this window - skipped")
                    done_chunks += 1
                    continue

                self.ffmpeg_line.emit(" ".join(cmd[:14]) + " ...\n")
                try:
                    self._proc = subprocess.Popen(
                        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1,
                        universal_newlines=True, **_no_window())
                except Exception as exc:
                    failed.append(f"{job.label}/{name}: {exc}")
                    job_ok = False
                    done_chunks += 1
                    continue

                for line in iter(self._proc.stdout.readline, ""):
                    if self.cancelled:
                        self._proc.terminate()
                        break
                    line = line.rstrip()
                    if not line:
                        continue
                    self.ffmpeg_line.emit(line + "\n")
                    if "time=" in line:
                        secs = _parse_time(line)
                        if secs is not None and span:
                            self.chunk_progress.emit(
                                done_chunks, total_chunks,
                                min(1.0, secs / span))
                self._proc.wait()
                done_chunks += 1

                if self._proc.returncode != 0 or self.cancelled:
                    # Cancelled or failed: whatever ffmpeg wrote is truncated
                    # and unplayable, and would otherwise be taken for a
                    # finished chunk by the next run's skip logic.
                    try:
                        if os.path.exists(out):
                            os.remove(out)
                    except OSError:
                        pass
                    partial = None
                    if not self.cancelled:
                        failed.append(f"{job.label}/{name}: "
                                      f"ffmpeg exit {self._proc.returncode}")
                        job_ok = False
                        self.progress.emit(
                            f"      FAILED ({self._proc.returncode})")
                    continue

                partial = None
                if staging:
                    try:
                        self.progress.emit(
                            f"      moving {human_size(os.path.getsize(out))}"
                            f" to {job.out_dir}")
                        shutil.move(out, final)
                    except Exception as exc:
                        failed.append(f"{job.label}/{name}: "
                                      f"encoded but could not be moved: {exc}")
                        job_ok = False
                        continue
                made.append(final)
                job_files.append(final)
                self.progress.emit(
                    f"      wrote {name} "
                    f"({human_size(os.path.getsize(final))})")

            # "both": join the finished dailies instead of encoding the trial
            # a second time. Skipped if anything went wrong, since a joined
            # file with a day missing would look complete but not be.
            if (job.settings.chunk_mode == "both" and job_ok
                    and not self.cancelled and len(job_files) > 1):
                joined = self._join_chunks(job, job_files, work_dir)
                if joined:
                    made.append(joined)
                else:
                    failed.append(f"{job.label}: could not join the daily files")
                    job_ok = False
            elif job.settings.chunk_mode == "both" and len(job_files) == 1:
                self.progress.emit(
                    "  only one day of footage - no separate trial file needed")

            self.job_finished.emit(job_index, job_ok and not self.cancelled)

        # A terminated ffmpeg leaves a truncated file behind; it is unplayable
        # and would be mistaken for a finished chunk on the next run.
        if partial and os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)

        # Drop folders this run created that ended up with nothing in them.
        # Directories holding completed chunks are left alone -- those are
        # hours of work, and throwing them away on a cancel is not ours to do.
        removed_dirs = []
        for path in created_dirs:
            try:
                if os.path.isdir(path) and not os.listdir(path):
                    os.rmdir(path)
                    removed_dirs.append(path)
            except OSError:
                pass

        if self.cancelled:
            note = f"Cancelled after {len(made)} file(s)."
            if removed_dirs:
                note += "  Removed the empty output folder."
            if made:
                note += ("  Files already finished were kept: "
                         + ", ".join(os.path.basename(m) for m in made[:3]))
                if len(made) > 3:
                    note += f" (and {len(made) - 3} more)"
            self.finished_all.emit(False, note)
            return
        msg = f"Wrote {len(made)} file(s) across {len(self.jobs)} trial(s)."
        if failed:
            msg += f"  {len(failed)} failed: " + "; ".join(failed[:3])
            if len(failed) > 3:
                msg += f" (and {len(failed) - 3} more)"
        self.finished_all.emit(not failed, msg)


def _parse_time(line):
    """Seconds encoded so far, from an ffmpeg 'time=HH:MM:SS.xx' progress line."""
    try:
        chunk = line.split("time=")[1].split()[0]
        h, m, s = chunk.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (IndexError, ValueError):
        return None



class SamplePreviewDialog(QDialog):
    """Play a freshly rendered sample, then let the user keep or discard it.

    The sample lives in a temp folder rather than the output folder: it exists
    to check a layout, and a scoring folder should not accumulate throwaway
    clips the user then has to hunt down and delete. Closing this window
    deletes it.

    Playback steps frames with OpenCV rather than QtMultimedia. QMediaPlayer's
    Windows backends report themselves as available but fail to load media in
    common conda/pip Qt installs -- silently, with an empty errorString -- so
    the pane just sits black. OpenCV reuses the decode path the rest of the
    tool already relies on. Trade-off: video only, no audio.
    """

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self.saved_to = None
        self.setWindowTitle("Sample Preview")
        self.resize(980, 760)

        self._cap = cv2.VideoCapture(path)
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        layout = QVBoxLayout(self)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 360)
        self.view.setStyleSheet("background-color:#101010;")
        layout.addWidget(self.view, 1)

        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, max(0, self._frames - 1))
        self.scrub.setToolTip("Scrub through the sample.")
        self.scrub.sliderMoved.connect(self.seek)
        layout.addWidget(self.scrub)

        controls = QHBoxLayout()
        self.btn_play = QPushButton("Pause")
        self.btn_play.setToolTip(
            "Play or pause the sample. Video only -- the preview has no audio "
            "even when the output does.")
        self.btn_play.clicked.connect(self.toggle)
        controls.addWidget(self.btn_play)
        self.lbl_time = QLabel("")
        self.lbl_time.setStyleSheet("color:#999999;")
        controls.addWidget(self.lbl_time)
        controls.addStretch()
        layout.addLayout(controls)

        actions = QHBoxLayout()
        btn_copy = QPushButton("Copy to Clipboard")
        btn_copy.setToolTip(
            "Put the sample FILE on the clipboard, so it can be pasted "
            "straight into a folder, an email or a chat window.")
        btn_copy.clicked.connect(self.copy_to_clipboard)
        actions.addWidget(btn_copy)

        btn_save = QPushButton("Save As...")
        btn_save.setStyleSheet(BLUE_BUTTON)
        btn_save.setToolTip(
            "Keep this sample by saving it somewhere of your choosing.")
        btn_save.clicked.connect(self.save_as)
        actions.addWidget(btn_save)

        actions.addStretch()
        btn_close = QPushButton("Close and Discard")
        btn_close.setToolTip(
            "Close without keeping the sample. The temporary file is deleted, "
            "so nothing is left behind in your output folder.")
        btn_close.clicked.connect(self.close)
        actions.addWidget(btn_close)
        layout.addLayout(actions)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.timer.start(self._interval())

    def _interval(self):
        return max(10, int(round(1000.0 / (self._fps or 30.0))))

    # -- playback ---------------------------------------------------------

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start(self._interval())
            self.btn_play.setText("Pause")

    def seek(self, frame):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        self.next_frame()

    def next_frame(self):
        ok, frame = self._cap.read()
        if not ok:
            # Loop, so a short sample can be watched over without fiddling.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                self.timer.stop()
                return
        # POS_FRAMES points at the NEXT frame to read, so the one just drawn
        # is the previous index.
        pos = max(0, int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.scrub.blockSignals(True)
        self.scrub.setValue(min(pos, self.scrub.maximum()))
        self.scrub.blockSignals(False)
        total = self._frames / self._fps if self._fps else 0
        self.lbl_time.setText(f"{pos / self._fps:0.1f}s / {total:0.1f}s")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.view.setPixmap(QPixmap.fromImage(img).scaled(
            self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # -- keep or discard --------------------------------------------------

    def copy_to_clipboard(self):
        """Put the FILE on the clipboard, so it can be pasted into a folder.

        Copies a file reference rather than the current frame -- pasting the
        clip somewhere is what is actually wanted here.
        """
        full = os.path.abspath(self.path)
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(full)])
        mime.setText(full)
        QApplication.clipboard().setMimeData(mime)
        self.lbl_time.setText("copied to clipboard")

    def save_as(self):
        target, _ = QFileDialog.getSaveFileName(
            self, "Save sample", os.path.basename(self.path),
            "MP4 video (*.mp4)")
        if not target:
            return
        try:
            shutil.copy2(self.path, target)
            self.saved_to = target
            QMessageBox.information(self, "Saved", f"Sample saved to:\n{target}")
        except Exception as exc:
            QMessageBox.warning(self, "Could not save", str(exc))

    def closeEvent(self, event):
        self.timer.stop()
        try:
            self._cap.release()
        except Exception:
            pass
        event.accept()


class ClockCalibrationDialog(QDialog):
    """Nudge each camera frame by frame until their burnt-in clocks agree.

    Reconstructing time from the day stamp plus accumulated duration lands the
    cameras within about a second of each other, but each carries its own
    small, stable offset from that anchor. This is the manual correction: the
    user steps a camera until it shows the first frame of a chosen second, does
    the same for the rest, and every cell then starts that second together.

    Aim for the SAME named second on every camera. Two cameras can each sit on
    the first frame of a tick and still be a whole second apart, which would
    look aligned here and be wrong in the export -- so each row shows the clock
    VALUE it is currently sitting on, not merely that a tick happened.

    Offsets are stable across a trial, so this is a one-time step whose cost is
    spread over the whole encode.
    """

    STEP_LABELS = [("<<", -5), ("<", -1), (">", 1), (">>", 5)]

    def __init__(self, tracks, when, parent=None):
        super().__init__(parent)
        self.tracks = tracks
        self.when = when
        self.fps = 30.0
        self.rows = {}
        self.original = {n: t.clock_offset for n, t in tracks.items()}

        self.setWindowTitle("Calibrate Camera Clocks")
        self.resize(1150, 240 + 150 * len(tracks))

        outer = QVBoxLayout(self)
        blurb = QLabel(
            "Step each camera until it shows the FIRST frame of the same "
            "second - e.g. the first frame reading :30 on every camera. "
            "The crop on the right is that camera's burnt-in clock. "
            "Your alignment overrides the computed one.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet("color:#bbbbbb;")
        outer.addWidget(blurb)

        trow = QHBoxLayout()
        trow.addWidget(QLabel("Calibrating at:"))
        self.time_label = QLabel(f"{when:%Y-%m-%d %H:%M:%S}")
        self.time_label.setStyleSheet("color:#ffd400; font-weight:bold;")
        trow.addWidget(self.time_label)
        trow.addStretch()
        outer.addLayout(trow)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        holder = QWidget()
        vbox = QVBoxLayout(holder)

        for name in sorted(tracks):
            vbox.addWidget(self._build_row(name))
        vbox.addStretch()
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)

        actions = QHBoxLayout()
        reset = QPushButton("Reset All")
        reset.setToolTip("Return every camera to the computed alignment.")
        reset.clicked.connect(self.reset_all)
        actions.addWidget(reset)
        actions.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setToolTip("Close without changing any offsets.")
        cancel.clicked.connect(self.reject)
        actions.addWidget(cancel)
        apply = QPushButton("Apply Calibration")
        apply.setStyleSheet(BLUE_BUTTON)
        apply.setToolTip(
            "Keep these offsets. They are applied to previews and to every "
            "exported video, including trials added to the queue afterwards.")
        apply.clicked.connect(self.accept)
        actions.addWidget(apply)
        outer.addLayout(actions)

        self.refresh_all()

    def _build_row(self, name):
        box = QGroupBox(name)
        row = QHBoxLayout()

        frame = QLabel()
        frame.setFixedSize(280, 158)
        frame.setAlignment(Qt.AlignCenter)
        frame.setStyleSheet("background-color:#101010;")
        row.addWidget(frame)

        clock = QLabel()
        clock.setFixedHeight(72)
        clock.setMinimumWidth(430)
        clock.setAlignment(Qt.AlignCenter)
        clock.setStyleSheet("background-color:#101010;")
        clock.setToolTip(
            "This camera's burnt-in clock, enlarged. Step until it shows the "
            "first frame of the second you are aligning everything to.")
        row.addWidget(clock)

        buttons = QVBoxLayout()
        nudge = QHBoxLayout()
        for label, delta in self.STEP_LABELS:
            b = QPushButton(label)
            b.setFixedWidth(46)
            b.setToolTip(f"Shift this camera {abs(delta)} frame"
                         f"{'s' if abs(delta) > 1 else ''} "
                         f"{'later' if delta > 0 else 'earlier'}.")
            b.clicked.connect(lambda _, n=name, d=delta: self.nudge(n, d))
            nudge.addWidget(b)
        buttons.addLayout(nudge)

        offset = QLabel()
        offset.setStyleSheet("color:#cccccc;")
        offset.setToolTip("Current correction for this camera.")
        buttons.addWidget(offset)

        zero = QPushButton("Reset")
        zero.setToolTip("Return this camera to the computed alignment.")
        zero.clicked.connect(lambda _, n=name: self.reset_one(n))
        buttons.addWidget(zero)
        buttons.addStretch()
        row.addLayout(buttons)

        box.setLayout(row)
        self.rows[name] = {"frame": frame, "clock": clock, "offset": offset}
        return box

    # -- interaction ------------------------------------------------------

    def nudge(self, name, frames):
        self.tracks[name].clock_offset += frames / self.fps
        self.refresh_one(name)

    def reset_one(self, name):
        self.tracks[name].clock_offset = 0.0
        self.refresh_one(name)

    def reset_all(self):
        for name in self.tracks:
            self.tracks[name].clock_offset = 0.0
        self.refresh_all()

    def restore(self):
        """Put back the offsets that were in force when the dialog opened."""
        for name, value in self.original.items():
            self.tracks[name].clock_offset = value

    def reject(self):
        self.restore()
        super().reject()

    # -- rendering --------------------------------------------------------

    def refresh_all(self):
        for name in self.tracks:
            self.refresh_one(name)

    def refresh_one(self, name):
        widgets = self.rows[name]
        track = self.tracks[name]
        off = track.clock_offset
        widgets["offset"].setText(
            f"{off:+.3f}s  ({int(round(off * self.fps)):+d} frames)")

        path, seek = track.locate(self.when)
        if not path:
            widgets["frame"].setText("no footage")
            widgets["clock"].clear()
            return

        frame = self._grab(path, seek)
        if frame is None:
            widgets["frame"].setText("decode failed")
            widgets["clock"].clear()
            return

        widgets["frame"].setPixmap(self._to_pixmap(frame, widgets["frame"].size()))
        widgets["clock"].setPixmap(
            self._to_pixmap(self._clock_crop(frame), widgets["clock"].size()))

    def _grab(self, path, seek):
        """Decode one frame, seeking precisely enough to trust a frame count."""
        pre = max(0.0, seek - 3.0)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-ss", f"{pre:.3f}", "-i", path, "-ss", f"{seek - pre:.3f}",
               "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60, **_no_window())
            if r.returncode or not r.stdout:
                return None
            arr = np.frombuffer(r.stdout, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _clock_crop(frame):
        """The bottom-right corner, where these recorders burn the clock."""
        h, w = frame.shape[:2]
        return frame[int(h * 0.92):, int(w * 0.62):]

    @staticmethod
    def _to_pixmap(frame, size):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(img).scaled(
            size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class CameraGridWindow(QMainWindow):
    """Camera Grid Tool."""

    def __init__(self):
        super().__init__()
        # Match the rest of FNT. Tools run in their own process, so each one
        # must set its own look -- otherwise it inherits the platform theme,
        # which is light on Windows.
        apply_dark_theme()
        self.tracks = {}
        self.scan_workers = []
        self.encode_worker = None
        # Set while a sample is rendering, so the finish handler knows to open
        # the preview instead of reporting a completed batch.
        self._sample_dir = None
        self.queue = []
        self._run_started = None

        self.preview_time = None
        self.setWindowTitle("Camera Grid")
        self.resize(1400, 900)
        self._build_ui()
        self._update_estimate()

    # -- construction ----------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        title = QLabel("Camera Grid")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color:#0078d4;")
        title.setAlignment(Qt.AlignCenter)
        outer.addWidget(title)

        splitter = QSplitter(Qt.Horizontal)
        left = self._build_left()
        left.setMinimumWidth(330)
        splitter.addWidget(left)
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)

        self.status = QLabel("Add a folder for each camera to begin.")
        self.status.setStyleSheet("color:#999;")
        outer.addWidget(self.status)

    def _build_left(self):
        panel = QWidget()
        col = QVBoxLayout(panel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        # --- sources ---
        src = QGroupBox("1. Camera Sources")
        sl = QVBoxLayout()
        row = QHBoxLayout()
        auto = QPushButton("Add Trial Folder (auto-detect)")
        auto.setToolTip(
            "Point at a whole trial folder and let the tool find the cameras "
            "itself, grouping files by the Camera<N> label in their names.\n\n"
            "Works whether each camera has its own subfolder, all the footage "
            "sits loose in one folder, or it is a mixture - no need to sort "
            "files by hand first.\n\n"
            "Use 'Add Camera Folder' instead when you want to pick cameras "
            "one at a time, or when the filenames carry no camera label.")
        auto.setStyleSheet(BLUE_BUTTON)
        auto.clicked.connect(self.add_trial_folder)
        sl.addWidget(auto)

        add = QPushButton("Add Camera Folder")
        add.setToolTip("Pick the folder holding ONE camera's segment files. Add a folder per camera. Durations are probed to rebuild that camera's timeline, which can take a moment over the network.")
        add.clicked.connect(self.add_camera_folder)
        row.addWidget(add)
        rm = QPushButton("Remove")
        rm.setToolTip("Remove the selected camera and clear it from the grid.")
        rm.clicked.connect(self.remove_camera)
        row.addWidget(rm)
        sl.addLayout(row)

        prow = QHBoxLayout()
        plab = QLabel("Recorder:")
        self.profile_combo = QComboBox()
        self.profile_combo.addItems([p[0] for p in PROFILE_CHOICES])
        profile_tip = (
            "How to read this recorder's filenames and reconstruct time.\n\n"
            "ViewTron / DVR: understands <timestamp>(<sequence>) naming. The "
            "timestamp marks an offload window and the sequence counts splits "
            "within it, which is what lets the tool place footage on the real "
            "wall clock AND spot dropouts (a missing sequence number). Use this "
            "for mesocosm and VoleTerra footage.\n\n"
            "Alphabetical: for recorders whose naming we do not model. Files "
            "are chained in name order, so playback is continuous and correctly "
            "ordered, but a dropout CANNOT be detected -- if one camera lost "
            "footage its cell will drift out of step with the others.")
        plab.setToolTip(profile_tip)
        self.profile_combo.setToolTip(profile_tip)
        prow.addWidget(plab)
        prow.addWidget(self.profile_combo, 1)
        sl.addLayout(prow)

        self.camera_list = CameraList()
        self.camera_list.setToolTip("Drag a camera onto a grid cell")
        sl.addWidget(self.camera_list)
        hint = QLabel("Drag a camera into a cell. Right-click a cell to clear it.")
        hint.setStyleSheet("color:#888; font-style:italic;")
        hint.setWordWrap(True)
        sl.addWidget(hint)
        src.setLayout(sl)
        layout.addWidget(src)

        # --- layout ---
        lay = QGroupBox("2. Grid Layout")
        ll = QVBoxLayout()
        grow = QHBoxLayout()
        grow.addWidget(QLabel("Grid:"))
        self.grid_combo = QComboBox()
        for label, _, _ in GRID_CHOICES:
            self.grid_combo.addItem(label)
        self.grid_combo.setCurrentText("2 x 2")
        self.grid_combo.setToolTip(
            "Rows x columns of the output. Sized automatically to fit the "
            "cameras you add, but you can override it: empty cells stay black, "
            "and picking a grid smaller than the camera count exports only the "
            "cameras that fit.")
        self.grid_combo.currentIndexChanged.connect(self.on_grid_changed)
        grow.addWidget(self.grid_combo, 1)
        ll.addLayout(grow)
        self.btn_calibrate = QPushButton("Calibrate Camera Clocks...")
        self.btn_calibrate.setToolTip(
            "Fine-tune synchronisation by hand. Step each camera frame by "
            "frame until every one shows the FIRST frame of the same second, "
            "read off their burnt-in clocks.\n\n"
            "The computed alignment is already within about a second; this "
            "closes the remaining gap to a single frame. Your calibration "
            "takes precedence and is applied to previews and exports.")
        self.btn_calibrate.clicked.connect(self.calibrate_clocks)
        ll.addWidget(self.btn_calibrate)

        hint = QLabel("Sized automatically when cameras are detected. "
                      "Choosing a smaller grid keeps only the cameras that fit.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888888; font-style:italic;")
        ll.addWidget(hint)
        lay.setLayout(ll)
        layout.addWidget(lay)

        # --- output ---
        out = QGroupBox("3. Output Settings")
        ol = QVBoxLayout()

        def combo_row(label, items, current=None, handler=None, tip=None):
            r = QHBoxLayout()
            lab = QLabel(label)
            box = QComboBox()
            box.addItems(items)
            if current:
                box.setCurrentText(current)
            if handler:
                box.currentIndexChanged.connect(handler)
            if tip:
                # Tip on the label too, so hovering either half explains it.
                lab.setToolTip(tip)
                box.setToolTip(tip)
            r.addWidget(lab)
            r.addWidget(box, 1)
            ol.addLayout(r)
            return box

        self.res_combo = combo_row(
            "Resolution:", [r[0] for r in RESOLUTIONS], RESOLUTIONS[0][0],
            self._update_estimate,
            "Size of the WHOLE grid, not one cell. Each camera gets "
            "resolution / grid size, so 1080p on a 3x3 gives 640x360 per "
            "camera. Raise this if animals are hard to see in a cell -- it "
            "costs file size and encode time roughly in proportion to pixels.")
        self.fps_combo = combo_row(
            "Frame rate:", [f"{f} fps" for f in FPS_CHOICES], "30 fps",
            self._update_estimate,
            "Frames per second in the output. 30 matches the cameras and keeps "
            "every frame -- best for brief behaviours. Lower values shrink the "
            "file and cut encode time, but very short events may fall between "
            "frames.")
        self.quality_combo = combo_row(
            "Quality:", [q[0] for q in QUALITY], QUALITY[1][0],
            self._update_estimate,
            "Compression level. Lower CRF = better quality and bigger files; "
            "each step of ~6 roughly doubles or halves size. Static "
            "surveillance footage compresses very well, so 'Good' is usually "
            "indistinguishable from 'High' for scoring.")
        self.codec_combo = combo_row(
            "Encoder:", ["libx264 (CPU)", "h264_nvenc (GPU)"],
            handler=self._update_estimate,
            tip="libx264 uses the CPU and gives the best size-for-quality. "
            "h264_nvenc offloads to an NVIDIA GPU: noticeably faster and it "
            "frees the CPU, at somewhat larger files. Both play anywhere.")
        self.chunk_combo = combo_row(
            "Split output:", ["One file per calendar day",
                              "One continuous file",
                              "Both (daily files + joined trial video)"],
            handler=self._update_estimate,
            tip="Daily files are named by date, seek quickly, and a failure costs "
            "only that day. One continuous file is simpler to reference but "
            "can be very large and slow to scrub through.")

        self.chk_labels = QCheckBox("Camera label in each cell")
        self.chk_labels.setChecked(True)
        self.chk_labels.setToolTip(
            "Burn each camera's name into the corner of its cell. The feeds "
            "carry their own label, but it becomes tiny once scaled into a "
            "grid cell.")
        self.chk_nosignal = QCheckBox("NO SIGNAL marker on dropouts")
        self.chk_nosignal.setChecked(True)
        self.chk_nosignal.setToolTip(
            "Write 'NO SIGNAL' across a cell while that camera has no footage. "
            "Without it a black cell is ambiguous -- a scorer cannot tell a "
            "camera outage from an empty arena.")
        self.chk_grayscale = QCheckBox("Convert to grayscale")
        self.chk_grayscale.setToolTip(
            "Drop colour from the finished grid. IR/night footage is already "
            "effectively grey, so this usually costs nothing visually and "
            "trims a little file size.")
        self.chk_stage = QCheckBox("Encode locally, then move to output")
        self.chk_stage.setChecked(True)
        self.chk_stage.setToolTip(
            "When the output folder is on a network drive, write each file to "
            "local disk first and move it across once finished.\n\n"
            "This does NOT speed the encode up - the source footage is read "
            "over the network either way. What it changes is what a share "
            "dropout costs: a single file copy at the end rather than hours of "
            "encoding. Ignored when the output is already local.")
        self.chk_remove_audio = QCheckBox("Remove audio")
        self.chk_remove_audio.setChecked(True)
        self.chk_remove_audio.setToolTip(
            "Drop all sound. A grid has several soundtracks and no sensible "
            "way to merge them, so when audio is kept it comes from ONE camera "
            "(chosen below). Preprocessed footage often has no audio at all, "
            "in which case this makes no difference.")
        self.chk_remove_audio.stateChanged.connect(self._on_audio_toggled)
        self.audio_combo = QComboBox()
        self.audio_combo.setEnabled(False)
        self.audio_combo.setToolTip(
            "Which camera's soundtrack to keep. It is delayed to match the "
            "video, so it stays in step and goes silent during that camera's "
            "dropouts.")

        self.chk_clock = QCheckBox("Large wall-clock timestamp")
        self.chk_clock.setToolTip(
            "Burn one large ticking clock along the bottom, driven by the "
            "reconstructed timeline. Easier to read than the small clocks each "
            "camera burns in, and useful for logging event times.")
        for box in (self.chk_labels, self.chk_nosignal, self.chk_clock,
                    self.chk_grayscale, self.chk_stage, self.chk_remove_audio):
            ol.addWidget(box)
        arow = QHBoxLayout()
        alab = QLabel("Audio from:")
        alab.setToolTip(self.audio_combo.toolTip())
        arow.addWidget(alab)
        arow.addWidget(self.audio_combo, 1)
        ol.addLayout(arow)

        out.setLayout(ol)
        layout.addWidget(out)

        # --- estimate ---
        est = QGroupBox("4. Estimate")
        el = QVBoxLayout()
        self.estimate_label = QLabel("-")
        self.estimate_label.setWordWrap(True)
        self.estimate_label.setStyleSheet(
            "color:#ddd; background:#1e1e1e; padding:8px;")
        el.addWidget(self.estimate_label)
        note = QLabel("Estimates from measured surveillance footage; actual size "
                      "varies with how much the scene moves.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888; font-style:italic;")
        el.addWidget(note)
        est.setLayout(el)
        layout.addWidget(est)

        # --- run ---
        run = QGroupBox("5. Build")
        rl = QVBoxLayout()
        dst = QHBoxLayout()
        dst.addWidget(QLabel("Output:"))
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("choose an output folder")
        self.out_dir_edit.setToolTip("Where the grid video(s) are written.")
        dst.addWidget(self.out_dir_edit, 1)
        browse = QPushButton("...")
        browse.setToolTip("Browse for the output folder.")
        browse.setFixedWidth(30)
        browse.clicked.connect(self.choose_output_dir)
        dst.addWidget(browse)
        rl.addLayout(dst)

        pre = QHBoxLayout()
        pre.addWidget(QLabel("Prefix:"))
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("trial name")
        self.prefix_edit.setToolTip("Start of each output filename, e.g. a trial ID. Daily files become <prefix>_YYYYMMDD_grid.mp4.")
        pre.addWidget(self.prefix_edit, 1)
        rl.addLayout(pre)

        self.btn_sample = QPushButton("Render 60-second Sample")
        self.btn_sample.setToolTip("Encode one minute at the previewed time so "
                                   "you can check the layout before a long run")
        self.btn_sample.clicked.connect(self.render_sample)
        rl.addWidget(self.btn_sample)

        self.btn_queue_add = QPushButton("Add This Trial to Queue")
        self.btn_queue_add.setToolTip(
            "Capture the loaded trial with the settings above as a queued job, "
            "then load the next trial folder and repeat.\n\n"
            "Each job keeps the cameras, grid and settings it was added with, "
            "so changing the controls afterwards will not alter jobs already "
            "waiting.")
        self.btn_queue_add.clicked.connect(self.add_to_queue)
        rl.addWidget(self.btn_queue_add)

        self.btn_build = QPushButton("Build Grid Video")
        self.btn_build.setStyleSheet(BLUE_BUTTON)
        self.btn_build.setToolTip("Encode the full trial with the settings above. This can run for many hours -- render a sample first.")
        self.btn_build.clicked.connect(self.start_build)
        rl.addWidget(self.btn_build)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("Stop after the current chunk. Files already written are kept.")
        self.btn_cancel.clicked.connect(self.cancel_build)
        rl.addWidget(self.btn_cancel)

        self.progress = QProgressBar()
        # Scaled to 100000 rather than 100: a whole-percent bar sits on 0 for
        # the first hour of a multi-day encode, which reads as "nothing is
        # happening".
        self.progress.setRange(0, 100_000)
        self.progress.setFormat("0.000%")
        self.progress.setToolTip(
            "Progress across every chunk in this run, to three decimals.")
        rl.addWidget(self.progress)

        self.lbl_eta = QLabel("")
        self.lbl_eta.setStyleSheet("color:#999999;")
        self.lbl_eta.setToolTip(
            "Elapsed time and an estimate of what remains, from the rate so "
            "far. It settles once the first chunk is underway.")
        rl.addWidget(self.lbl_eta)
        run.setLayout(rl)
        layout.addWidget(run)

        # --- batch queue ---
        q = QGroupBox("6. Trial Queue")
        ql = QVBoxLayout()
        self.queue_table = QTableWidget(0, 5)
        self.queue_table.setHorizontalHeaderLabels(
            ["", "Trial", "Cameras", "Files", "Estimate"])
        self.queue_table.setMinimumHeight(130)
        self.queue_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectRows)
        hh = self.queue_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.Stretch)
        self.queue_table.setColumnWidth(0, 40)
        self.queue_table.setToolTip(
            "Trials waiting to be processed, in the order they were added. "
            "Each row keeps its own captured settings.")
        ql.addWidget(self.queue_table)

        qrow = QHBoxLayout()
        self.btn_queue_remove = QPushButton("Remove Selected")
        self.btn_queue_remove.setToolTip("Drop the selected trial from the queue.")
        self.btn_queue_remove.clicked.connect(self.remove_from_queue)
        qrow.addWidget(self.btn_queue_remove)
        self.btn_queue_clear = QPushButton("Clear")
        self.btn_queue_clear.setToolTip("Empty the queue.")
        self.btn_queue_clear.clicked.connect(self.clear_queue)
        qrow.addWidget(self.btn_queue_clear)
        ql.addLayout(qrow)

        self.btn_queue_run = QPushButton("Process Queue")
        self.btn_queue_run.setStyleSheet(BLUE_BUTTON)
        self.btn_queue_run.setToolTip(
            "Encode every queued trial in the order listed. A trial that fails "
            "does not stop the ones behind it, and the summary names whatever "
            "failed.")
        self.btn_queue_run.clicked.connect(self.process_queue)
        ql.addWidget(self.btn_queue_run)

        self.queue_summary = QLabel("Queue empty.")
        self.queue_summary.setWordWrap(True)
        self.queue_summary.setStyleSheet("color:#999999;")
        ql.addWidget(self.queue_summary)
        q.setLayout(ql)
        layout.addWidget(q)

        layout.addStretch()
        scroll.setWidget(inner)
        col.addWidget(scroll)
        return panel

    def _build_right(self):
        panel = QWidget()
        col = QVBoxLayout(panel)

        design = QGroupBox("Grid Designer  -  drag cameras into cells")
        dl = QVBoxLayout()
        self.designer = GridDesigner()
        self.designer.changed.connect(self.on_layout_changed)
        dl.addWidget(self.designer, 1)
        design.setLayout(dl)
        col.addWidget(design, 3)

        prev = QGroupBox("Preview")
        pl = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel("Time:"))
        self.preview_combo = QComboBox()
        self.preview_combo.setToolTip(
            "A moment in the trial to preview. The list samples across the "
            "recording and adds one entry inside each detected dropout, so you "
            "can check what a scorer sees when a camera goes down.")
        self.preview_combo.currentIndexChanged.connect(self.refresh_preview)
        row.addWidget(self.preview_combo, 1)
        btn = QPushButton("Refresh")
        btn.setToolTip("Re-decode one frame per camera at the selected time. "
                       "Use it to confirm the layout and that the cameras are "
                       "in step before starting a long encode.")
        btn.clicked.connect(self.refresh_preview)
        row.addWidget(btn)
        pl.addLayout(row)
        prev.setLayout(pl)
        col.addWidget(prev)

        cov = QGroupBox("Coverage  (green = footage, red = dropout)")
        cl = QVBoxLayout()
        self.coverage = CoverageStrip()
        cl.addWidget(self.coverage)
        cov.setLayout(cl)
        col.addWidget(cov)

        log = QGroupBox("Log")
        gl = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        self.log_text.setStyleSheet(
            "QTextEdit { background:#1e1e1e; color:#d4d4d4;"
            " font-family:Consolas,monospace; font-size:9pt; }")
        gl.addWidget(self.log_text)
        log.setLayout(gl)
        col.addWidget(log, 2)
        return panel

    # -- logging ---------------------------------------------------------

    def log(self, message):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{stamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())

    # -- sources ---------------------------------------------------------

    def add_camera_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a camera folder")
        if not folder:
            return
        profile = PROFILE_CHOICES[self.profile_combo.currentIndex()][1]
        self.log(f"Scanning {folder}  (profile: {profile})")
        worker = ScanWorker(folder, profile=profile)
        worker.progress.connect(self.status.setText)
        worker.done.connect(self.on_scan_done)
        worker.failed.connect(lambda e: self.log(f"Scan failed: {e}"))
        self.scan_workers.append(worker)
        worker.start()

    def add_trial_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select a trial folder (cameras detected automatically)")
        if not folder:
            return
        profile = PROFILE_CHOICES[self.profile_combo.currentIndex()][1]
        self.log(f"Auto-detecting cameras in {folder}  (profile: {profile})")
        worker = DiscoverWorker(folder, profile=profile)
        worker.progress.connect(self.status.setText)
        worker.found.connect(self.on_scan_done)
        worker.done.connect(self._on_discovery_done)
        worker.failed.connect(lambda e: self.log(f"Auto-detect failed: {e}"))
        self.scan_workers.append(worker)
        worker.start()

    def on_scan_done(self, track):
        if not track.segments:
            self.log(f"No NVR segments found in {track.folder}")
            QMessageBox.warning(
                self, "No segments",
                "No recognisable segment files were found in that folder.\n\n"
                "Expected names ending like '_20260224000000(007).mp4'.")
            return
        name = track.name
        suffix = 2
        while name in self.tracks:
            name = f"{track.name}_{suffix}"
            suffix += 1
        track.name = name
        self.tracks[name] = track

        item = QListWidgetItem(
            f"{name}   {len(track.segments)} segs   "
            f"{track.footage_seconds / 3600:.1f}h"
            + (f"   {len(track.gaps)} gap(s)" if track.gaps else ""))
        item.setData(Qt.UserRole, name)
        item.setToolTip(
            f"{track.folder}\n{track.start} -> {track.end}\n"
            f"footage {track.footage_seconds / 3600:.2f}h of "
            f"{track.span_seconds / 3600:.2f}h span")
        self.camera_list.addItem(item)

        self.log(f"{name}: {len(track.segments)} segments, "
                 f"{track.start:%Y-%m-%d %H:%M} -> {track.end:%Y-%m-%d %H:%M}, "
                 f"{track.footage_seconds / 3600:.2f}h footage")
        for gap in track.gaps:
            self.log(f"    gap {gap.start:%Y-%m-%d %H:%M:%S} "
                     f"{gap.duration / 60:.1f} min"
                     + ("  (position uncertain)" if gap.unlocated else ""))
        # Default the output INSIDE the trial folder, and name it after the
        # trial. Deriving these from the camera folder's parent put them beside
        # the trial rather than in it whenever the trial folder itself was the
        # one picked, which is what auto-detect always does.
        trial_root = getattr(track, "trial_root", None) or os.path.dirname(
            os.path.normpath(track.folder))
        if not self.prefix_edit.text():
            self.prefix_edit.setText(os.path.basename(trial_root))
        if not self.out_dir_edit.text():
            self.out_dir_edit.setText(os.path.join(trial_root, "grid"))
        self._auto_assign(name)
        self._refresh_trial()

    def _auto_assign(self, name):
        """Fit the grid to the cameras and drop this one in the next free cell.

        The grid grows to the smallest listed layout that holds every camera,
        so detecting 16 cameras lands on 4x4 without the user resizing first.
        Cells fill left to right, top to bottom, in camera-name order.
        Shrinking the grid afterwards is a legitimate way to export only some
        of the cameras -- assignments that no longer fit are simply dropped.
        """
        wanted = len(self.tracks)
        model = self.designer.layout_model
        if model.rows * model.cols < wanted:
            rows, cols = best_grid(wanted)
            for label, r, c in GRID_CHOICES:
                if (r, c) == (rows, cols):
                    self.grid_combo.blockSignals(True)
                    self.grid_combo.setCurrentText(label)
                    self.grid_combo.blockSignals(False)
                    break
            self.designer.rebuild(rows, cols)
            model = self.designer.layout_model

        for r in range(model.rows):
            for c in range(model.cols):
                if (r, c) not in model.assignments:
                    model.place(r, c, name)
                    self.designer._sync_cells()
                    self.on_layout_changed()
                    return

    def remove_camera(self):
        item = self.camera_list.currentItem()
        if not item:
            return
        name = item.data(Qt.UserRole)
        self.tracks.pop(name, None)
        self.designer.layout_model.clear_camera(name)
        self.designer._sync_cells()
        self.camera_list.takeItem(self.camera_list.row(item))
        self._refresh_trial()

    # -- layout / preview -------------------------------------------------

    def _on_audio_toggled(self):
        keep = not self.chk_remove_audio.isChecked()
        self.audio_combo.setEnabled(keep)
        if keep:
            self._refresh_audio_sources()

    def _refresh_audio_sources(self):
        """Offer the cameras currently placed on the grid as audio sources."""
        current = self.audio_combo.currentText()
        names = sorted(self.designer.layout_model.assignments.values())
        self.audio_combo.blockSignals(True)
        self.audio_combo.clear()
        self.audio_combo.addItems(names)
        if current in names:
            self.audio_combo.setCurrentText(current)
        self.audio_combo.blockSignals(False)

    def on_grid_changed(self):
        _, rows, cols = GRID_CHOICES[self.grid_combo.currentIndex()]
        self.designer.rebuild(rows, cols)

    def on_layout_changed(self):
        self._refresh_audio_sources()
        self._update_estimate()
        self.refresh_preview()

    def _refresh_trial(self):
        tracks = list(self.tracks.values())
        if not tracks:
            self.coverage.set_tracks([], None)
            self.preview_combo.clear()
            self._update_estimate()
            return
        trial = TrialTimeline(tracks)
        self.coverage.set_tracks(tracks, (trial.start, trial.end))

        self.preview_combo.blockSignals(True)
        self.preview_combo.clear()
        span = trial.duration_seconds
        for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
            when = trial.start + datetime.timedelta(seconds=span * frac)
            self.preview_combo.addItem(when.strftime("%Y-%m-%d %H:%M:%S"), when)
        # Offer a moment inside each dropout: the most useful thing to eyeball.
        for track in tracks:
            for gap in track.gaps:
                when = gap.start + datetime.timedelta(seconds=gap.duration / 2)
                self.preview_combo.addItem(
                    f"{when:%Y-%m-%d %H:%M:%S}  ({track.name} dropout)", when)
        self.preview_combo.blockSignals(False)
        self._update_estimate()
        self.refresh_preview()

    def refresh_preview(self):
        when = self.preview_combo.currentData()
        if when is None:
            return
        self.designer.clear_thumbnails()
        for name, track in self.tracks.items():
            if self.designer.layout_model.cell_of(name) is None:
                continue
            path, offset = track.locate(when)
            if not path:
                continue                      # cell stays black: a real dropout
            pix = self._grab_thumbnail(path, offset)
            if pix:
                self.designer.set_thumbnail(name, pix)
        self.status.setText(f"Preview at {when:%Y-%m-%d %H:%M:%S}")

    def _grab_thumbnail(self, path, offset, width=320):
        """One frame as a QPixmap, decoded straight from ffmpeg's stdout."""
        pre = max(0.0, offset - 3.0)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-ss", f"{pre:.3f}", "-i", path, "-ss", f"{offset - pre:.3f}",
               "-frames:v", "1", "-vf", f"scale={width}:-2",
               "-f", "image2pipe", "-vcodec", "png", "-"]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60, **_no_window())
            if r.returncode or not r.stdout:
                return None
            img = QImage.fromData(r.stdout, "PNG")
            return QPixmap.fromImage(img) if not img.isNull() else None
        except Exception:
            return None

    # -- settings / estimate ---------------------------------------------

    def current_settings(self):
        _, w, h = RESOLUTIONS[self.res_combo.currentIndex()]
        return EncodeSettings(
            width=w, height=h,
            fps=FPS_CHOICES[self.fps_combo.currentIndex()],
            codec=("h264_nvenc" if self.codec_combo.currentIndex() == 1
                   else "libx264"),
            crf=QUALITY[self.quality_combo.currentIndex()][1],
            preset="veryfast",
            chunk_mode={0: "daily", 1: "continuous", 2: "both"}[
                self.chunk_combo.currentIndex()],
            show_camera_labels=self.chk_labels.isChecked(),
            show_no_signal=self.chk_nosignal.isChecked(),
            show_clock=self.chk_clock.isChecked(),
            grayscale=self.chk_grayscale.isChecked(),
            stage_output_locally=self.chk_stage.isChecked(),
            remove_audio=self.chk_remove_audio.isChecked(),
            audio_source=(None if self.chk_remove_audio.isChecked()
                          else self.audio_combo.currentText() or None))

    def _update_estimate(self, *_):
        if not self.tracks:
            self.estimate_label.setText("Add cameras to see an estimate.")
            return
        settings = self.current_settings()
        trial = TrialTimeline(list(self.tracks.values()))
        total = trial.duration_seconds
        n_cams = len(self.designer.layout_model.assignments)
        if not n_cams or not total:
            self.estimate_label.setText("Assign cameras to grid cells.")
            return
        est = estimate(settings, self.designer.layout_model, total, n_cams)
        chunks = trial.chunks(settings.chunk_mode)
        cw, ch = settings.cell_size(self.designer.layout_model)
        per_chunk = est["size_bytes"] / max(1, len(chunks))
        self.estimate_label.setText(
            f"<b>{total / 3600:.1f} h</b> of footage &middot; {n_cams} camera(s)"
            f"<br>{len(chunks)} output file(s), each ~{human_size(per_chunk)}"
            f"<br>total size <b>~{human_size(est['size_bytes'])}</b>"
            f"<br>encode time <b>~{human_time(est['encode_seconds'])}</b>"
            f" ({est['realtime_factor']:.1f}x realtime)"
            f"<br>cell size {cw}x{ch} px")

    # -- build -----------------------------------------------------------

    def _ready(self):
        if not self.tracks:
            QMessageBox.warning(self, "No cameras", "Add at least one camera folder.")
            return False
        if not self.designer.layout_model.assignments:
            QMessageBox.warning(self, "Empty grid",
                                "Drag at least one camera into a grid cell.")
            return False
        if not self.out_dir_edit.text().strip():
            QMessageBox.warning(self, "No output folder",
                                "Choose an output folder.")
            return False
        return True

    def choose_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            self.out_dir_edit.setText(folder)

    def render_sample(self):
        """Encode 60 seconds at the previewed time, to check before a long run.

        Written to a temp folder, not the output folder: a sample is for
        looking at, and a scoring folder should not fill up with throwaway
        clips. The preview window offers to save it if it is worth keeping.
        """
        if not self.tracks:
            QMessageBox.warning(self, "No cameras", "Add at least one camera folder.")
            return
        if not self.designer.layout_model.assignments:
            QMessageBox.warning(self, "Empty grid",
                                "Drag at least one camera into a grid cell.")
            return
        when = self.preview_combo.currentData()
        if when is None:
            return

        self._discard_sample()               # clear any previous one
        self._sample_dir = tempfile.mkdtemp(prefix="fnt_grid_sample_")
        settings = self.current_settings()
        settings.chunk_mode = "continuous"
        self._start_worker([self._current_job(
            label="sample", chunks=[(when, when + datetime.timedelta(seconds=60))],
            settings=settings, out_dir=self._sample_dir,
            prefix=f"sample_{when:%Y%m%d_%H%M%S}")])

    # -- queue ------------------------------------------------------------

    def add_to_queue(self):
        if not self._ready():
            return
        label = self.prefix_edit.text().strip() or f"trial {len(self.queue) + 1}"
        job = self._current_job(label=label)
        if not job.chunks:
            QMessageBox.warning(self, "Nothing to queue",
                                "This trial produced no output windows.")
            return
        self.queue.append(job)
        self._refresh_queue_table()
        self.log(f"Queued {label}: {len(job.chunks)} file(s), "
                 f"{len(job.layout.assignments)} camera(s)")

    def remove_from_queue(self):
        row = self.queue_table.currentRow()
        if 0 <= row < len(self.queue):
            self.log(f"Removed {self.queue.pop(row).label} from the queue")
            self._refresh_queue_table()

    def clear_queue(self):
        self.queue.clear()
        self._refresh_queue_table()

    def _refresh_queue_table(self):
        self.queue_table.setRowCount(len(self.queue))
        total_size = total_time = 0.0
        for i, job in enumerate(self.queue):
            est = job.estimate()
            total_size += est["size_bytes"]
            total_time += est["encode_seconds"]
            cells = ["", job.label, ", ".join(job.cameras),
                     str(len(job.chunks)),
                     f"{human_size(est['size_bytes'])} / "
                     f"{human_time(est['encode_seconds'])}"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.queue_table.setItem(i, col, item)
        if self.queue:
            self.queue_summary.setText(
                f"{len(self.queue)} trial(s) queued - about "
                f"{human_size(total_size)} and {human_time(total_time)} in total.")
        else:
            self.queue_summary.setText("Queue empty.")

    def process_queue(self):
        if not self.queue:
            QMessageBox.information(
                self, "Queue empty",
                "Add at least one trial with 'Add This Trial to Queue'.")
            return
        total_size = sum(j.estimate()["size_bytes"] for j in self.queue)
        total_time = sum(j.estimate()["encode_seconds"] for j in self.queue)
        reply = QMessageBox.question(
            self, "Process queue",
            f"{len(self.queue)} trial(s), "
            f"{sum(len(j.chunks) for j in self.queue)} output file(s)\n"
            f"Estimated total size: {human_size(total_size)}\n"
            f"Estimated total time: {human_time(total_time)}\n\nStart now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply != QMessageBox.Yes:
            return
        self._start_worker(list(self.queue))

    def calibrate_clocks(self):
        """Open the manual clock calibration for the previewed instant."""
        if not self.tracks:
            QMessageBox.warning(self, "No cameras",
                                "Add at least one camera folder first.")
            return
        when = self.preview_combo.currentData()
        if when is None:
            return
        dlg = ClockCalibrationDialog(self.tracks, when, self)
        if dlg.exec_() == QDialog.Accepted:
            applied = {n: t.clock_offset for n, t in self.tracks.items()
                       if t.clock_offset}
            if applied:
                self.log("Clock calibration applied: " + ", ".join(
                    f"{n} {v:+.3f}s" for n, v in sorted(applied.items())))
            else:
                self.log("Clock calibration cleared.")
            self.refresh_preview()

    def _on_discovery_done(self, n_found):
        """Lay every detected camera out in name order once they are all in.

        Cameras arrive one at a time and the grid grows as they do, so a
        camera detected late can land in a cell freed by a resize -- leaving
        the order looking shuffled. Re-filling once at the end makes the
        layout deterministic and strictly left to right.
        """
        self.log(f"Auto-detect finished: {n_found} camera(s)")
        if not self.tracks:
            return
        names = sorted(self.tracks)
        rows, cols = best_grid(len(names))
        for label, r, c in GRID_CHOICES:
            if (r, c) == (rows, cols):
                self.grid_combo.blockSignals(True)
                self.grid_combo.setCurrentText(label)
                self.grid_combo.blockSignals(False)
                break
        self.designer.apply_layout(GridLayout.filled(names, rows, cols))
        self.on_layout_changed()

    def _discard_sample(self):
        """Delete the temp folder holding the last sample, if any."""
        if self._sample_dir:
            shutil.rmtree(self._sample_dir, ignore_errors=True)
            self._sample_dir = None
        self.queue = []
        self._run_started = None


    def _show_sample(self):
        """Open the rendered sample, then bin it unless the user saved it."""
        clips = sorted(f for f in os.listdir(self._sample_dir)
                       if f.lower().endswith(".mp4"))
        if not clips:
            self.log("Sample produced no output.")
            self._discard_sample()
            return
        path = os.path.join(self._sample_dir, clips[0])
        dlg = SamplePreviewDialog(path, self)
        dlg.exec_()
        if dlg.saved_to:
            self.log(f"Sample saved to {dlg.saved_to}")
        else:
            self.log("Sample discarded.")
        self._discard_sample()

    def start_build(self):
        if not self._ready():
            return
        settings = self.current_settings()
        trial = TrialTimeline(list(self.tracks.values()))
        chunks = trial.chunks(settings.chunk_mode)
        est = estimate(settings, self.designer.layout_model,
                       trial.duration_seconds,
                       len(self.designer.layout_model.assignments))
        reply = QMessageBox.question(
            self, "Build grid video",
            f"{len(chunks)} output file(s)\n"
            f"Estimated total size: {human_size(est['size_bytes'])}\n"
            f"Estimated encode time: {human_time(est['encode_seconds'])}\n\n"
            f"Start now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply != QMessageBox.Yes:
            return
        self._start_worker([self._current_job(chunks=chunks, settings=settings)])

    def _current_job(self, label=None, chunks=None, settings=None,
                     out_dir=None, prefix=None):
        """Snapshot the loaded trial and current controls as one job."""
        settings = settings or self.current_settings()
        if chunks is None:
            chunks = TrialTimeline(list(self.tracks.values())).chunks(
                settings.chunk_mode)
        prefix = prefix if prefix is not None else self.prefix_edit.text().strip()
        return QueuedTrial(
            label or prefix or "trial", self.tracks,
            self.designer.layout_model, settings, chunks,
            out_dir or self.out_dir_edit.text().strip(), prefix or "grid")

    def _start_worker(self, jobs):
        self.btn_build.setEnabled(False)
        self.btn_sample.setEnabled(False)
        self.btn_queue_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setFormat("0.000%")
        self._run_started = time.time()
        self.lbl_eta.setText("starting...")
        self.encode_worker = GridEncodeWorker(jobs)
        self.encode_worker.progress.connect(self.log)
        self.encode_worker.job_started.connect(self.on_job_started)
        self.encode_worker.job_finished.connect(self.on_job_finished)
        self.encode_worker.chunk_progress.connect(self.on_chunk_progress)
        self.encode_worker.ffmpeg_line.connect(self.on_ffmpeg_line)
        self.encode_worker.finished_all.connect(self.on_build_finished)
        self.encode_worker.start()

    def on_job_started(self, index, label):
        self._mark_queue_row(index, "running")

    def on_job_finished(self, index, ok):
        self._mark_queue_row(index, "done" if ok else "failed")

    def _mark_queue_row(self, index, state):
        """Colour a queue row so an overnight batch can be read at a glance."""
        if index >= self.queue_table.rowCount():
            return
        colours = {"running": "#c9a227", "done": "#2f9e44", "failed": "#b03030"}
        item = self.queue_table.item(index, 0)
        if item:
            item.setText({"running": "...", "done": "OK",
                          "failed": "FAIL"}.get(state, ""))
            item.setForeground(QColor(colours.get(state, "#cccccc")))

    def on_chunk_progress(self, index, total, fraction):
        overall = (index + fraction) / max(1, total)
        self.progress.setValue(int(round(overall * 100_000)))
        self.progress.setFormat(f"{overall * 100:.3f}%")

        if self._run_started is None or overall <= 0:
            return
        elapsed = time.time() - self._run_started
        remaining = elapsed / overall - elapsed
        self.lbl_eta.setText(
            f"chunk {min(index + 1, total)}/{total}  -  "
            f"{human_time(elapsed)} elapsed, about {human_time(remaining)} left")

    def on_ffmpeg_line(self, line):
        # Progress lines would flood the log; keep only real messages.
        if "frame=" in line or "time=" in line:
            self.status.setText(line.strip()[:140])
        else:
            self.log_text.moveCursor(self.log_text.textCursor().End)

    def cancel_build(self):
        if self.encode_worker:
            self.encode_worker.cancel()
            self.log("Cancelling...")

    def on_build_finished(self, ok, message):
        self.btn_build.setEnabled(True)
        self.btn_sample.setEnabled(True)
        self.btn_queue_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(100_000 if ok else 0)
        self.progress.setFormat("100.000%" if ok else "0.000%")
        if self._run_started:
            self.lbl_eta.setText(
                f"finished in {human_time(time.time() - self._run_started)}")
        self._run_started = None
        self.log(message)

        if self._sample_dir:
            # A sample opens straight into the preview rather than announcing
            # a file the user did not ask to keep.
            if ok:
                self._show_sample()
            else:
                QMessageBox.warning(self, "Sample failed", message)
                self._discard_sample()
            return

        (QMessageBox.information if ok else QMessageBox.warning)(
            self, "Camera Grid", message)

    def closeEvent(self, event):
        if self.encode_worker and self.encode_worker.isRunning():
            reply = QMessageBox.question(
                self, "Close", "An encode is running. Cancel it and close?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.encode_worker.cancel()
            self.encode_worker.wait(5000)
        self._discard_sample()
        event.accept()


def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    win = CameraGridWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
