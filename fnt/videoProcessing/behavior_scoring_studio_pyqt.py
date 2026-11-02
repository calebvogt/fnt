"""
FNT Behavior Scoring Studio
Manual behavioral annotation tool with ethogram definition, video playback,
timeline visualization, and CSV/JSON data export.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import (
    Qt, QEvent, QTimer, QRectF, pyqtSignal
)
from PyQt5.QtGui import (
    QColor, QFont, QImage, QPainter, QPen, QPixmap, QBrush, QFontMetrics
)
from PyQt5.QtWidgets import (
    QApplication, QDialogButtonBox, QCheckBox, QColorDialog, QComboBox, QDialog,
    QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSpinBox, QStatusBar, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget, QHeaderView, QAbstractItemView
)

from fnt.theme import BLUE_BUTTON_STYLE, apply_dark_theme
from fnt.videoProcessing.ethogram import (
    POINT, STATE, BehaviorDefinition, ScoringEvent, ScoringSession, Subject,
    format_time,
)
from fnt.videoProcessing.ethogram_dialogs import (
    BehaviorEditDialog, ModifierPromptDialog, SubjectEditDialog,
)



class EventEditDialog(QDialog):
    """Change one already-scored event.

    Mistimed or misattributed events are normal in manual scoring, and without
    this the only remedy is undo-last -- which means discarding everything
    scored since. Time is edited in seconds because that is what the exported
    data carries; the frame is derived from it.
    """

    def __init__(self, parent=None, event=None, session=None):
        super().__init__(parent)
        self.event = event
        self.session = session
        self.setWindowTitle("Edit Event")
        self.resize(420, 340)

        v = QVBoxLayout(self)

        v.addWidget(QLabel("Time (seconds):"))
        self.spin_time = QDoubleSpinBox()
        self.spin_time.setDecimals(3)
        self.spin_time.setRange(0.0, 24 * 3600.0)
        self.spin_time.setSingleStep(1.0 / max(1.0, session.fps))
        self.spin_time.setValue(event.time_seconds)
        self.spin_time.setToolTip(
            "Wall position of the event. The frame number is recomputed from "
            "this and the video's frame rate.")
        v.addWidget(self.spin_time)

        v.addWidget(QLabel("Subject:"))
        self.combo_subject = QComboBox()
        self.combo_subject.setEditable(True)
        self.combo_subject.addItem("")
        for s in session.subjects:
            self.combo_subject.addItem(s.name)
        self.combo_subject.setCurrentText(event.subject)
        self.combo_subject.setToolTip("Which animal this event belongs to.")
        v.addWidget(self.combo_subject)

        v.addWidget(QLabel("Behavior:"))
        self.combo_behavior = QComboBox()
        for b in session.ethogram:
            self.combo_behavior.addItem(b.name)
        self.combo_behavior.setCurrentText(event.behavior)
        self.combo_behavior.setToolTip(
            "Changing this does not convert between point and state events.")
        v.addWidget(self.combo_behavior)

        v.addWidget(QLabel("Modifiers:"))
        self.edit_modifiers = QLineEdit(
            "; ".join(f"{k}={v}" for k, v in sorted(event.modifiers.items())))
        self.edit_modifiers.setPlaceholderText("Set=value; Set=value")
        self.edit_modifiers.setToolTip(
            "One entry per modifier set, written as Set=value and separated "
            "by semicolons.")
        v.addWidget(self.edit_modifiers)

        v.addWidget(QLabel("Comment:"))
        self.edit_comment = QLineEdit(event.comment)
        v.addWidget(self.edit_comment)
        v.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setStyleSheet(BLUE_BUTTON_STYLE)
        v.addWidget(buttons)

    def apply_to_event(self):
        ev = self.event
        ev.time_seconds = self.spin_time.value()
        ev.frame = int(round(ev.time_seconds * self.session.fps))
        ev.subject = self.combo_subject.currentText().strip()
        ev.behavior = self.combo_behavior.currentText().strip()
        ev.comment = self.edit_comment.text().strip()
        mods = {}
        for part in self.edit_modifiers.text().split(";"):
            if "=" in part:
                name, _, value = part.partition("=")
                if name.strip():
                    mods[name.strip()] = value.strip()
        ev.modifiers = mods
        return ev


class TimeBudgetDialog(QDialog):
    """Occurrences and durations per subject and behavior.

    Only state events accumulate duration; point events are counted. An
    unclosed state contributes an occurrence but no time, so a forgotten stop
    cannot quietly inflate a total -- those are listed separately instead.
    """

    def __init__(self, parent=None, session=None):
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Time Budget")
        self.resize(760, 480)

        v = QVBoxLayout(self)

        unpaired = session.unpaired_states()
        if unpaired:
            warn = QLabel(
                "Unclosed state events: "
                + ", ".join(f"{e.behavior} ({e.subject})" for e in unpaired)
                + "\nThese contribute no duration below.")
            warn.setWordWrap(True)
            warn.setStyleSheet(
                "color:#ffb84d; background:#3a2f1a; padding:6px;")
            v.addWidget(warn)

        rows = session.time_budget()
        self.table = QTableWidget(len(rows), 7)
        self.table.setHorizontalHeaderLabels(
            ["Subject", "Behavior", "Category", "Type", "N",
             "Total (s)", "Mean (s)"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i, r in enumerate(rows):
            cells = [r["subject"], r["behavior"], r["category"], r["type"],
                     str(r["n"]), f"{r['total_seconds']:.2f}",
                     f"{r['mean_seconds']:.2f}"]
            for c, text in enumerate(cells):
                self.table.setItem(i, c, QTableWidgetItem(text))
        v.addWidget(self.table, 1)

        row = QHBoxLayout()
        export = QPushButton("Export Time Budget CSV")
        export.setStyleSheet(BLUE_BUTTON_STYLE)
        export.setToolTip(
            "Write these figures next to the scoring file, as time_budget.csv.")
        export.clicked.connect(self._export)
        row.addWidget(export)
        row.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        v.addLayout(row)

    def _export(self):
        folder = self.session.output_folder()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "time_budget.csv"
        pd.DataFrame(self.session.time_budget()).to_csv(path, index=False)
        QMessageBox.information(self, "Exported", f"Written to:\n{path}")

# =============================================================================
# Timeline Widget
# =============================================================================

class TimelineWidget(QWidget):
    """Custom-painted Gantt-style timeline for behavioral scoring visualization."""

    frame_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMaximumHeight(350)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMouseTracking(True)

        self.behaviors = []
        self.events = []
        self.active_states = {}
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame = 0
        self.visible_duration_seconds = 30.0

        self.MARGIN_LEFT = 110
        self.MARGIN_RIGHT = 10
        self.MARGIN_TOP = 24
        self.MARGIN_BOTTOM = 6
        self.ROW_HEIGHT = 22
        self.ROW_GAP = 2

    def set_data(self, behaviors, events, active_states, total_frames, fps):
        self.behaviors = behaviors
        self.events = events
        self.active_states = active_states
        self.total_frames = total_frames
        self.fps = fps if fps > 0 else 30.0
        self.update()

    def set_current_frame(self, frame):
        self.current_frame = frame
        self.update()

    def set_zoom(self, seconds):
        self.visible_duration_seconds = max(2.0, seconds)
        self.update()

    def _get_timeline_width(self):
        return max(1, self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT)

    def _frame_to_x(self, frame):
        center_time = self.current_frame / self.fps
        view_start = center_time - self.visible_duration_seconds / 2
        view_end = center_time + self.visible_duration_seconds / 2
        if view_end <= view_start:
            return self.MARGIN_LEFT
        t = (frame / self.fps - view_start) / (view_end - view_start)
        return self.MARGIN_LEFT + t * self._get_timeline_width()

    def _x_to_frame(self, x):
        center_time = self.current_frame / self.fps
        view_start = center_time - self.visible_duration_seconds / 2
        view_end = center_time + self.visible_duration_seconds / 2
        t = (x - self.MARGIN_LEFT) / self._get_timeline_width()
        time_s = view_start + t * (view_end - view_start)
        return max(0, min(self.total_frames - 1, int(time_s * self.fps)))

    def _behavior_row_y(self, idx):
        return self.MARGIN_TOP + idx * (self.ROW_HEIGHT + self.ROW_GAP)

    def sizeHint(self):
        from PyQt5.QtCore import QSize
        n = max(len(self.behaviors), 3)
        h = self.MARGIN_TOP + n * (self.ROW_HEIGHT + self.ROW_GAP) + self.MARGIN_BOTTOM
        return QSize(400, max(120, h))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor("#1e1e1e"))

        if not self.behaviors:
            painter.setPen(QColor("#666666"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Define behaviors in the ethogram to see the timeline")
            painter.end()
            return

        center_time = self.current_frame / self.fps
        view_start = center_time - self.visible_duration_seconds / 2
        view_end = center_time + self.visible_duration_seconds / 2

        # Draw time axis
        self._draw_time_axis(painter, view_start, view_end)

        # Draw behavior rows
        for i, beh in enumerate(self.behaviors):
            y = self._behavior_row_y(i)

            # Alternating row background
            row_color = QColor("#222222") if i % 2 == 0 else QColor("#282828")
            painter.fillRect(self.MARGIN_LEFT, y, self._get_timeline_width(), self.ROW_HEIGHT, row_color)

            # Row label
            painter.setPen(QColor(beh.color))
            fm = QFontMetrics(painter.font())
            label_rect = QRectF(4, y, self.MARGIN_LEFT - 8, self.ROW_HEIGHT)
            painter.drawText(label_rect, Qt.AlignVCenter | Qt.AlignRight, beh.name)

            # Color swatch
            painter.fillRect(int(self.MARGIN_LEFT - 16), int(y + 6), 10, 10, QColor(beh.color))

            # Draw events for this behavior
            self._draw_behavior_events(painter, beh, i, view_start, view_end)

        # Draw current frame indicator (center line)
        cx = self.MARGIN_LEFT + self._get_timeline_width() / 2
        pen = QPen(QColor("#0078d4"), 2)
        painter.setPen(pen)
        painter.drawLine(int(cx), 0, int(cx), h)

        # Border
        painter.setPen(QColor("#3f3f3f"))
        painter.drawRect(0, 0, w - 1, h - 1)

        painter.end()

    def _draw_time_axis(self, painter, view_start, view_end):
        duration = view_end - view_start
        if duration <= 0:
            return

        # Choose tick interval
        intervals = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
        tick_interval = 1.0
        for iv in intervals:
            if duration / iv <= 15:
                tick_interval = iv
                break

        painter.setPen(QColor("#666666"))
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)

        t = (int(view_start / tick_interval)) * tick_interval
        while t <= view_end:
            if t >= view_start:
                x = self._frame_to_x(int(t * self.fps))
                painter.drawLine(int(x), self.MARGIN_TOP - 4, int(x), self.MARGIN_TOP)

                # Label
                if t >= 0:
                    mins = int(t // 60)
                    secs = t % 60
                    if tick_interval >= 60:
                        label = f"{mins}:{int(secs):02d}"
                    elif tick_interval >= 1:
                        label = f"{mins}:{secs:04.1f}" if mins > 0 else f"{secs:.1f}s"
                    else:
                        label = f"{secs:.2f}s"
                    painter.drawText(int(x) - 25, 2, 50, self.MARGIN_TOP - 6, Qt.AlignCenter, label)
            t += tick_interval

        # Axis line
        painter.drawLine(self.MARGIN_LEFT, self.MARGIN_TOP, self.width() - self.MARGIN_RIGHT, self.MARGIN_TOP)

    def _draw_behavior_events(self, painter, beh, row_idx, view_start, view_end):
        y = self._behavior_row_y(row_idx)
        color = QColor(beh.color)

        view_start_frame = max(0, int(view_start * self.fps) - 1)
        view_end_frame = min(self.total_frames, int(view_end * self.fps) + 1)

        if beh.event_type == "state":
            # Pair START/STOP events
            pairs = []
            open_start = None
            for ev in self.events:
                if ev.behavior != beh.name:
                    continue
                if ev.status == "START":
                    open_start = ev
                elif ev.status == "STOP" and open_start is not None:
                    pairs.append((open_start.frame, ev.frame))
                    open_start = None
            # Active (unclosed) state
            open_for_beh = [ev for (subj, name), ev in self.active_states.items()
                            if name == beh.name]
            if open_for_beh:
                start_ev = open_for_beh[0]
                pairs.append((start_ev.frame, None))  # None = still open

            for start_f, stop_f in pairs:
                if stop_f is not None and stop_f < view_start_frame:
                    continue
                if start_f > view_end_frame:
                    continue

                x1 = max(self.MARGIN_LEFT, self._frame_to_x(start_f))
                if stop_f is not None:
                    x2 = min(self.width() - self.MARGIN_RIGHT, self._frame_to_x(stop_f))
                else:
                    # Draw to current frame with dashed right edge
                    x2 = min(self.width() - self.MARGIN_RIGHT, self._frame_to_x(self.current_frame))

                bar_w = max(2, x2 - x1)
                bar_color = QColor(color)
                bar_color.setAlpha(160)
                painter.fillRect(int(x1), int(y + 2), int(bar_w), self.ROW_HEIGHT - 4, bar_color)

                # Border
                pen = QPen(color, 1)
                painter.setPen(pen)
                painter.drawRect(int(x1), int(y + 2), int(bar_w), self.ROW_HEIGHT - 4)

                # Dashed right edge for active states
                if stop_f is None:
                    pen = QPen(color, 1, Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawLine(int(x2), int(y + 2), int(x2), int(y + self.ROW_HEIGHT - 2))

        elif beh.event_type == "point":
            for ev in self.events:
                if ev.behavior != beh.name or ev.status != "POINT":
                    continue
                if ev.frame < view_start_frame or ev.frame > view_end_frame:
                    continue
                x = self._frame_to_x(ev.frame)
                pen = QPen(color, 2)
                painter.setPen(pen)
                painter.drawLine(int(x), int(y + 2), int(x), int(y + self.ROW_HEIGHT - 2))
                # Small diamond marker
                painter.setBrush(QBrush(color))
                mid_y = y + self.ROW_HEIGHT / 2
                painter.drawEllipse(int(x - 3), int(mid_y - 3), 6, 6)
                painter.setBrush(Qt.NoBrush)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.x() >= self.MARGIN_LEFT:
            frame = self._x_to_frame(event.x())
            self.frame_clicked.emit(frame)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 0.8 if delta > 0 else 1.25
        new_dur = self.visible_duration_seconds * factor
        total_dur = self.total_frames / self.fps if self.fps > 0 else 60
        new_dur = max(2.0, min(total_dur, new_dur))
        self.visible_duration_seconds = new_dur
        self.update()


# =============================================================================
# Behavior Edit Dialog
# =============================================================================

class BehaviorScoringStudioWindow(QMainWindow):
    """Main window for Behavior Scoring Studio."""

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    def __init__(self):
        super().__init__()
        # Match the other FNT tools; each runs in its own process, so it must
        # set its own look rather than inheriting the launcher's.
        apply_dark_theme()
        self.setWindowTitle("FNT Behavior Scoring Studio")
        self.setMinimumSize(1100, 750)
        self.resize(1500, 900)

        # State
        self.video_files = []
        self.current_file_idx = 0
        self.sessions = {}  # video_path -> ScoringSession
        self.current_session = None

        # Video playback
        self.cap = None
        self.current_frame_idx = 0
        self.current_frame = None
        self.is_playing = False
        self.playback_speed = 1.0
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._on_play_tick)

        # Scoring
        self.scoring_enabled = False

        # Feedback timer
        self._feedback_timer = QTimer()
        self._feedback_timer.setSingleShot(True)
        self._feedback_timer.timeout.connect(self._clear_feedback)

        self._setup_ui()
        self._apply_styles()

        QApplication.instance().installEventFilter(self)

    # =========================================================================
    # Event Filter (keyboard shortcuts)
    # =========================================================================

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return super().eventFilter(obj, event)

        # Don't capture keys when typing in text fields
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QSpinBox, QDoubleSpinBox)):
            return super().eventFilter(obj, event)

        key = event.key()
        mods = event.modifiers()

        # Ctrl+Z: undo
        if key == Qt.Key_Z and mods & Qt.ControlModifier:
            self._undo_last_event()
            return True

        # Space: play/pause
        if key == Qt.Key_Space:
            self._toggle_play_pause()
            return True

        # Arrow keys: frame stepping
        if key == Qt.Key_Right:
            step = 10 if mods & Qt.ShiftModifier else 1
            self._step_frames(step)
            return True
        if key == Qt.Key_Left:
            step = -10 if mods & Qt.ShiftModifier else -1
            self._step_frames(step)
            return True

        # Subject and behavior keys
        if self.scoring_enabled and self.current_session and self.cap:
            key_text = event.text()
            if key_text and len(key_text) == 1:
                # Subjects first: switching focus must not also fire a
                # behavior bound to the same character.
                sub = self.current_session.subject_for_key(key_text)
                if sub is not None:
                    self.current_session.current_subject = sub.name
                    self._refresh_subject_table()
                    self.status_bar.showMessage(f"Focus: {sub.name}")
                    return True
                beh = self.current_session.behavior_for_key(key_text)
                if beh is not None:
                    self._score_behavior(beh)
                    return True

        return super().eventFilter(obj, event)

    # =========================================================================
    # UI Setup
    # =========================================================================

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left panel (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFixedWidth(360)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)

        self._create_input_section(left_layout)
        self._create_subject_section(left_layout)
        self._create_ethogram_section(left_layout)
        self._create_scoring_section(left_layout)
        self._create_export_section(left_layout)

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        # Video info bar
        self.lbl_video_info = QLabel("Load a video to begin")
        self.lbl_video_info.setObjectName("video_info")
        self.lbl_video_info.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_video_info)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #3f3f3f;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setText("Load a video to begin")
        right_layout.addWidget(self.video_label, 1)

        # Scoring feedback overlay label
        self.lbl_feedback = QLabel("")
        self.lbl_feedback.setObjectName("scoring_feedback")
        self.lbl_feedback.setAlignment(Qt.AlignCenter)
        self.lbl_feedback.setFixedHeight(30)
        self.lbl_feedback.hide()
        right_layout.addWidget(self.lbl_feedback)

        # Playback controls
        self._create_playback_controls(right_layout)

        # Timeline
        self.timeline_widget = TimelineWidget()
        self.timeline_widget.frame_clicked.connect(self._seek_to_frame)
        right_layout.addWidget(self.timeline_widget)

        self._create_event_table(right_layout)

        main_layout.addWidget(left_scroll)
        main_layout.addWidget(right_panel, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Welcome to Behavior Scoring Studio - Load video files to begin")

    def _create_event_table(self, layout):
        group = QGroupBox("Events")
        v = QVBoxLayout()
        v.setSpacing(3)

        self.event_table = QTableWidget(0, 7)
        self.event_table.setHorizontalHeaderLabels(
            ["Time", "Subject", "Behavior", "Modifiers", "Status",
             "Comment", "Frame"])
        self.event_table.setMinimumHeight(150)
        self.event_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.event_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.event_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.event_table.horizontalHeader()
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setSectionResizeMode(5, QHeaderView.Stretch)
        self.event_table.setToolTip(
            "Every scored event. Double-click a row to jump the video there; "
            "Edit fixes a mistimed or misattributed event without discarding "
            "everything scored since.")
        self.event_table.itemDoubleClicked.connect(self._seek_to_event)
        v.addWidget(self.event_table)

        row = QHBoxLayout()
        row.setSpacing(2)
        for label, slot, tip in (
                ("Go To", self._seek_to_event,
                 "Move the video to the selected event."),
                ("Edit", self._edit_event,
                 "Change the selected event's time, subject, behavior, "
                 "modifiers or comment."),
                ("Delete", self._delete_event,
                 "Remove the selected event. Open states are recomputed, so "
                 "deleting a stop reopens its state.")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch()

        self.btn_time_budget = QPushButton("Time Budget...")
        self.btn_time_budget.setToolTip(
            "Occurrences and durations per subject and behavior, with any "
            "unclosed states listed.")
        self.btn_time_budget.clicked.connect(self._show_time_budget)
        row.addWidget(self.btn_time_budget)
        v.addLayout(row)

        group.setLayout(v)
        layout.addWidget(group)

    # -- event table -------------------------------------------------------

    def _sorted_events(self):
        """Events in time order, which is how a scorer reads them."""
        if not self.current_session:
            return []
        return sorted(self.current_session.events,
                      key=lambda e: (e.time_seconds, e.frame))

    def _refresh_event_table(self):
        events = self._sorted_events()
        self.event_table.setRowCount(len(events))
        for i, ev in enumerate(events):
            cells = [format_time(ev.time_seconds), ev.subject, ev.behavior,
                     ev.modifier_text, ev.status, ev.comment, str(ev.frame)]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 2:
                    beh = self.current_session.behavior_named(ev.behavior)
                    if beh:
                        item.setForeground(QColor(beh.color))
                self.event_table.setItem(i, c, item)

    def _selected_event(self):
        row = self.event_table.currentRow()
        events = self._sorted_events()
        return events[row] if 0 <= row < len(events) else None

    def _seek_to_event(self):
        ev = self._selected_event()
        if ev is not None:
            self._seek_to_frame(ev.frame)

    def _edit_event(self):
        ev = self._selected_event()
        if ev is None or not self.current_session:
            return
        dlg = EventEditDialog(self, ev, self.current_session)
        if dlg.exec_() == QDialog.Accepted:
            dlg.apply_to_event()
            # A changed time or subject can re-pair states differently.
            self.current_session.rebuild_active_states()
            self._refresh_all_views()
            self._autosave()

    def _delete_event(self):
        ev = self._selected_event()
        if ev is None or not self.current_session:
            return
        reply = QMessageBox.question(
            self, "Delete event",
            f"Delete {ev.status} {ev.behavior}"
            f"{f' ({ev.subject})' if ev.subject else ''} at "
            f"{format_time(ev.time_seconds)}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self.current_session.remove_event(ev)
        self._refresh_all_views()
        self._autosave()

    def _refresh_all_views(self):
        self._refresh_event_table()
        self._update_active_states_list()
        self._update_timeline()
        self._update_last_event_label()

    def _show_time_budget(self):
        if not self.current_session or not self.current_session.events:
            QMessageBox.information(self, "No events",
                                    "Score some events first.")
            return
        TimeBudgetDialog(self, self.current_session).exec_()

    def _create_input_section(self, layout):
        group = QGroupBox("1. Input")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.setToolTip(
            "Add every video in a folder to the scoring list.")
        self.btn_add_folder.clicked.connect(self._add_folder)
        btn_row.addWidget(self.btn_add_folder)

        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.setToolTip(
            "Add individual video files to the scoring list.")
        self.btn_add_files.clicked.connect(self._add_files)
        btn_row.addWidget(self.btn_add_files)

        self.btn_clear_files = QPushButton("Clear")
        self.btn_clear_files.setToolTip(
            "Empty the video list. Scoring already saved to disk is untouched.")
        self.btn_clear_files.clicked.connect(self._clear_files)
        btn_row.addWidget(self.btn_clear_files)

        group_layout.addLayout(btn_row)

        self.file_list = QListWidget()
        self.file_list.setToolTip(
            "Videos queued for scoring. A tick marks one that already has "
            "saved scoring alongside it.")
        self.file_list.setMaximumHeight(120)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        group_layout.addWidget(self.file_list)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)

        self.btn_prev_file = QPushButton("< Prev")
        self.btn_prev_file.setToolTip(
            "Load the previous video. The ethogram and subjects carry over.")
        self.btn_prev_file.setObjectName("small_btn")
        self.btn_prev_file.clicked.connect(self._prev_file)
        self.btn_prev_file.setEnabled(False)
        nav_row.addWidget(self.btn_prev_file)

        self.lbl_file_num = QLabel("File 0/0")
        self.lbl_file_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_file_num, 1)

        self.btn_next_file = QPushButton("Next >")
        self.btn_next_file.setToolTip(
            "Load the next video. The ethogram and subjects carry over.")
        self.btn_next_file.setObjectName("small_btn")
        self.btn_next_file.clicked.connect(self._next_file)
        self.btn_next_file.setEnabled(False)
        nav_row.addWidget(self.btn_next_file)

        group_layout.addLayout(nav_row)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_subject_section(self, layout):
        group = QGroupBox("2. Subjects")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        self.subject_table = QTableWidget(0, 2)
        self.subject_table.setHorizontalHeaderLabels(["Subject", "Key"])
        self.subject_table.setMaximumHeight(110)
        self.subject_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.subject_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.subject_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.subject_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        self.subject_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents)
        self.subject_table.setToolTip(
            "Animals that events are attributed to. Press a subject's key "
            "while scoring to move focus to it; behaviours are then recorded "
            "against that subject.")
        self.subject_table.itemSelectionChanged.connect(
            self._on_subject_row_selected)
        group_layout.addWidget(self.subject_table)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)
        for label, slot, tip in (
                ("Add", self._add_subject, "Define a new subject."),
                ("Edit", self._edit_subject, "Edit the selected subject."),
                ("Remove", self._remove_subject, "Delete the selected subject.")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        group_layout.addLayout(btn_row)

        self.lbl_focus = QLabel("Focus: (none)")
        self.lbl_focus.setToolTip(
            "The subject that scored events are attributed to right now.")
        self.lbl_focus.setStyleSheet("color:#ffd400; font-weight:bold;")
        group_layout.addWidget(self.lbl_focus)

        group.setLayout(group_layout)
        layout.addWidget(group)

    # -- subjects ----------------------------------------------------------

    def _all_bound_keys(self, skip_subject=None, skip_behavior=None):
        """Every key already in use. Behaviours and subjects share a keyboard,
        so both lists have to be checked together."""
        keys = []
        if not self.current_session:
            return keys
        for b in self.current_session.ethogram:
            if b.key and b.name != skip_behavior:
                keys.append(b.key)
        for sub in self.current_session.subjects:
            if sub.key and sub.name != skip_subject:
                keys.append(sub.key)
        return keys

    def _refresh_subject_table(self):
        if not self.current_session:
            return
        subs = self.current_session.subjects
        self.subject_table.blockSignals(True)
        self.subject_table.setRowCount(len(subs))
        for i, sub in enumerate(subs):
            self.subject_table.setItem(i, 0, QTableWidgetItem(sub.name))
            self.subject_table.setItem(i, 1, QTableWidgetItem(sub.key or "--"))
            if sub.name == self.current_session.current_subject:
                self.subject_table.selectRow(i)
        self.subject_table.blockSignals(False)
        self._update_focus_label()

    def _update_focus_label(self):
        current = (self.current_session.current_subject
                   if self.current_session else "")
        self.lbl_focus.setText(f"Focus: {current or '(none)'}")

    def _on_subject_row_selected(self):
        row = self.subject_table.currentRow()
        if self.current_session and 0 <= row < len(self.current_session.subjects):
            self.current_session.current_subject =                 self.current_session.subjects[row].name
            self._update_focus_label()

    def _add_subject(self):
        if not self.current_session:
            return
        dlg = SubjectEditDialog(self, existing_keys=self._all_bound_keys())
        if dlg.exec_() == QDialog.Accepted:
            sub = dlg.get_subject()
            self.current_session.subjects.append(sub)
            if not self.current_session.current_subject:
                self.current_session.current_subject = sub.name
            self._refresh_subject_table()
            self._autosave()

    def _edit_subject(self):
        row = self.subject_table.currentRow()
        if not self.current_session or row < 0:
            return
        old = self.current_session.subjects[row]
        dlg = SubjectEditDialog(self, old,
                                self._all_bound_keys(skip_subject=old.name))
        if dlg.exec_() == QDialog.Accepted:
            new = dlg.get_subject()
            # Events already scored keep pointing at the old name unless it
            # follows the rename.
            if new.name != old.name:
                for ev in self.current_session.events:
                    if ev.subject == old.name:
                        ev.subject = new.name
                if self.current_session.current_subject == old.name:
                    self.current_session.current_subject = new.name
                self.current_session.rebuild_active_states()
            self.current_session.subjects[row] = new
            self._refresh_subject_table()
            self._autosave()

    def _remove_subject(self):
        row = self.subject_table.currentRow()
        if not self.current_session or row < 0:
            return
        sub = self.current_session.subjects[row]
        scored = sum(1 for e in self.current_session.events if e.subject == sub.name)
        if scored:
            reply = QMessageBox.question(
                self, "Remove subject",
                f"'{sub.name}' has {scored} scored event(s).\n\n"
                f"Removing the subject leaves those events in place, still "
                f"labelled with its name. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
        del self.current_session.subjects[row]
        if self.current_session.current_subject == sub.name:
            self.current_session.current_subject = (
                self.current_session.subjects[0].name
                if self.current_session.subjects else "")
        self._refresh_subject_table()
        self._autosave()

    def _create_ethogram_section(self, layout):
        group = QGroupBox("3. Ethogram")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Table
        self.ethogram_table = QTableWidget(0, 7)
        self.ethogram_table.setHorizontalHeaderLabels(
            ["Name", "Key", "Type", "Color", "Modifiers", "Category", "Excludes"])
        self.ethogram_table.setMaximumHeight(160)
        self.ethogram_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ethogram_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.ethogram_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.ethogram_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        self.ethogram_table.setToolTip(
            "Behaviours in this ethogram. Hover the Modifiers cell to see each "
            "set's values.")
        group_layout.addWidget(self.ethogram_table)

        # Add/Edit/Remove buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_add_behavior = QPushButton("Add")
        self.btn_add_behavior.setToolTip(
            "Define a new behavior, with its key, modifiers and exclusions.")
        self.btn_add_behavior.clicked.connect(self._add_behavior)
        btn_row.addWidget(self.btn_add_behavior)

        self.btn_edit_behavior = QPushButton("Edit")
        self.btn_edit_behavior.setToolTip(
            "Edit the selected behavior.")
        self.btn_edit_behavior.clicked.connect(self._edit_behavior)
        btn_row.addWidget(self.btn_edit_behavior)

        self.btn_remove_behavior = QPushButton("Remove")
        self.btn_remove_behavior.setToolTip(
            "Delete the selected behavior. Events already scored against it remain.")
        # Red only for the destructive action, as in the other tools.
        self.btn_remove_behavior.setStyleSheet(
            "QPushButton { background-color:#a02b2b; color:#fff; padding:5px 10px;"
            " border:none; font-weight:bold; }"
            "QPushButton:hover { background-color:#b83b3b; }")
        self.btn_remove_behavior.clicked.connect(self._remove_behavior)
        btn_row.addWidget(self.btn_remove_behavior)

        group_layout.addLayout(btn_row)

        # Load/Save ethogram
        io_row = QHBoxLayout()
        io_row.setSpacing(2)

        self.btn_load_ethogram = QPushButton("Load Ethogram")
        self.btn_load_ethogram.setToolTip(
            "Load a saved ethogram (behaviors and subjects) from a JSON file, so one definition can be reused across trials.")
        self.btn_load_ethogram.clicked.connect(self._load_ethogram_file)
        io_row.addWidget(self.btn_load_ethogram)

        self.btn_save_ethogram = QPushButton("Save Ethogram")
        self.btn_save_ethogram.setToolTip(
            "Save this ethogram and subject list to a JSON file for reuse.")
        self.btn_save_ethogram.clicked.connect(self._save_ethogram_file)
        io_row.addWidget(self.btn_save_ethogram)

        group_layout.addLayout(io_row)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_scoring_section(self, layout):
        group = QGroupBox("4. Scoring")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Enable scoring
        self.chk_scoring = QCheckBox("Enable Scoring (Capture Keys)")
        self.chk_scoring.setToolTip(
            "When on, behavior and subject keys are captured by this window. Turn it off to type freely elsewhere.")
        self.chk_scoring.toggled.connect(self._on_scoring_toggled)
        group_layout.addWidget(self.chk_scoring)

        # Modifier selector
        mod_row = QHBoxLayout()
        mod_row.addWidget(QLabel("Modifier:"))
        # Modifiers are asked per behavior at scoring time, so there is
        # no global modifier selector any more.
        group_layout.addLayout(mod_row)

        # Active states list
        group_layout.addWidget(QLabel("Active State Events:"))
        self.active_states_list = QListWidget()
        self.active_states_list.setToolTip(
            "State events currently open, with the subject holding each one. "
            "Anything left here when you finish is an unclosed state.")
        self.active_states_list.setMaximumHeight(80)
        group_layout.addWidget(self.active_states_list)

        # Last event label
        self.lbl_last_event = QLabel("Last event: --")
        self.lbl_last_event.setStyleSheet("color: #999999; font-size: 10px;")
        group_layout.addWidget(self.lbl_last_event)

        # Undo button
        self.btn_undo = QPushButton("Undo Last (Ctrl+Z)")
        self.btn_undo.setToolTip(
            "Remove the most recent event. Undoing a stop reopens its state.")
        self.btn_undo.clicked.connect(self._undo_last_event)
        group_layout.addWidget(self.btn_undo)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_export_section(self, layout):
        group = QGroupBox("5. Export")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        self.chk_autosave = QCheckBox("Auto-save on event")
        self.chk_autosave.setToolTip(
            "Save after every scored event, so an interrupted session loses nothing.")
        self.chk_autosave.setChecked(True)
        group_layout.addWidget(self.chk_autosave)

        self.btn_export_csv = QPushButton("Export Scoring CSV")
        self.btn_export_csv.setToolTip(
            "Write scoring.csv and ethogram_config.json to the output folder.")
        self.btn_export_csv.clicked.connect(self._export_csv)
        group_layout.addWidget(self.btn_export_csv)

        self.btn_open_folder = QPushButton("Open Output Folder")
        self.btn_open_folder.setToolTip(
            "Open the folder holding this video's scoring files.")
        self.btn_open_folder.clicked.connect(self._open_output_folder)
        group_layout.addWidget(self.btn_open_folder)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_playback_controls(self, layout):
        bar = QWidget()
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(5, 2, 5, 2)
        bar_layout.setSpacing(4)

        self.btn_step_back_big = QPushButton("|<")
        self.btn_step_back_big.setObjectName("small_btn")
        self.btn_step_back_big.setFixedWidth(30)
        self.btn_step_back_big.setToolTip("Back 10 frames")
        self.btn_step_back_big.clicked.connect(lambda: self._step_frames(-10))
        bar_layout.addWidget(self.btn_step_back_big)

        self.btn_step_back = QPushButton("<")
        self.btn_step_back.setObjectName("small_btn")
        self.btn_step_back.setFixedWidth(24)
        self.btn_step_back.setToolTip("Back 1 frame")
        self.btn_step_back.clicked.connect(lambda: self._step_frames(-1))
        bar_layout.addWidget(self.btn_step_back)

        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.setToolTip(
            "Play or pause. Space does the same; arrow keys step frames.")
        self.btn_play_pause.setFixedWidth(60)
        self.btn_play_pause.clicked.connect(self._toggle_play_pause)
        bar_layout.addWidget(self.btn_play_pause)

        self.btn_step_fwd = QPushButton(">")
        self.btn_step_fwd.setObjectName("small_btn")
        self.btn_step_fwd.setFixedWidth(24)
        self.btn_step_fwd.setToolTip("Forward 1 frame")
        self.btn_step_fwd.clicked.connect(lambda: self._step_frames(1))
        bar_layout.addWidget(self.btn_step_fwd)

        self.btn_step_fwd_big = QPushButton(">|")
        self.btn_step_fwd_big.setObjectName("small_btn")
        self.btn_step_fwd_big.setFixedWidth(30)
        self.btn_step_fwd_big.setToolTip("Forward 10 frames")
        self.btn_step_fwd_big.clicked.connect(lambda: self._step_frames(10))
        bar_layout.addWidget(self.btn_step_fwd_big)

        bar_layout.addWidget(QLabel("Speed:"))
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.combo_speed.setCurrentText("1x")
        self.combo_speed.setFixedWidth(70)
        self.combo_speed.setToolTip(
            "Playback speed. Slower speeds make brief behaviours easier to time.")
        self.combo_speed.currentTextChanged.connect(self._on_speed_changed)
        bar_layout.addWidget(self.combo_speed)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(0)
        self.seek_slider.sliderMoved.connect(self._on_seek_slider_moved)
        bar_layout.addWidget(self.seek_slider, 1)

        self.lbl_time = QLabel("00:00:00 / 00:00:00")
        self.lbl_time.setFixedWidth(140)
        bar_layout.addWidget(self.lbl_time)

        layout.addWidget(bar)

    # =========================================================================
    # Styles
    # =========================================================================

    def _apply_styles(self):
        """Match the other FNT tools.

        The base look comes from fnt.theme (Fusion plus the shared palette), so
        only the pieces specific to this window are styled here. The previous
        sheet painted EVERY button blue, which left nothing to mark the actions
        that actually commit something.
        """
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: #0078d4;
            }
            QLabel#video_info {
                background-color: #1e1e1e;
                color: #cccccc;
                padding: 4px;
                border: 1px solid #3f3f3f;
            }
            QLabel#scoring_feedback {
                font-size: 13px;
                font-weight: bold;
                border-radius: 4px;
            }
            QTableWidget, QListWidget {
                background-color: #1e1e1e;
                alternate-background-color: #262626;
                gridline-color: #3f3f3f;
            }
            QHeaderView::section {
                background-color: #333333;
                color: #dddddd;
                padding: 3px;
                border: none;
            }
        """)

        # Blue marks the actions that commit something, as in the other tools.
        for widget in (getattr(self, name, None) for name in
                       ("btn_export_csv", "btn_time_budget")):
            if widget is not None:
                widget.setStyleSheet(BLUE_BUTTON_STYLE)

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Folder")
        if not folder:
            return
        files = []
        for f in sorted(Path(folder).iterdir()):
            if f.suffix.lower() in self.VIDEO_EXTENSIONS:
                files.append(str(f))
        if not files:
            QMessageBox.information(self, "No Videos", "No video files found in the selected folder.")
            return
        self._add_video_files(files)

    def _add_files(self):
        exts = " ".join(f"*{e}" for e in self.VIDEO_EXTENSIONS)
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "", f"Video Files ({exts});;All Files (*)"
        )
        if files:
            self._add_video_files(files)

    def _add_video_files(self, files):
        for f in files:
            if f not in self.video_files:
                self.video_files.append(f)
                self._scan_for_existing_scoring(f)
        self._refresh_file_list()
        if self.video_files and self.current_session is None:
            self.file_list.setCurrentRow(0)

    def _scan_for_existing_scoring(self, video_path):
        output_folder = Path(video_path).parent / f"{Path(video_path).stem}_fntScoring"
        config_path = output_folder / "ethogram_config.json"
        csv_path = output_folder / "scoring.csv"

        if config_path.exists():
            session = ScoringSession(video_path)
            session.load_config(config_path)
            if csv_path.exists():
                session.load_events(
                    pd.read_csv(csv_path).to_dict('records'))
            self.sessions[video_path] = session
            return True
        return False

    def _refresh_file_list(self):
        self.file_list.clear()
        for f in self.video_files:
            name = Path(f).name
            has_data = f in self.sessions
            item = QListWidgetItem(f"{'[S] ' if has_data else ''}{name}")
            if has_data:
                item.setForeground(QColor("#82e0aa"))
            self.file_list.addItem(item)
        n = len(self.video_files)
        idx = self.current_file_idx + 1 if n > 0 else 0
        self.lbl_file_num.setText(f"File {idx}/{n}")
        self.btn_prev_file.setEnabled(n > 1)
        self.btn_next_file.setEnabled(n > 1)

    def _clear_files(self):
        self._release_video()
        self.video_files = []
        self.current_file_idx = 0
        self.sessions = {}
        self.current_session = None
        self._refresh_file_list()
        self._refresh_ethogram_table()
        self._update_active_states_list()
        self.video_label.clear()
        self.video_label.setText("Load a video to begin")
        self.lbl_video_info.setText("Load a video to begin")
        self.timeline_widget.set_data([], [], {}, 0, 30)
        self.seek_slider.setMaximum(0)
        self.status_bar.showMessage("Files cleared")

    def _prev_file(self):
        if len(self.video_files) > 1:
            idx = (self.current_file_idx - 1) % len(self.video_files)
            self.file_list.setCurrentRow(idx)

    def _next_file(self):
        if len(self.video_files) > 1:
            idx = (self.current_file_idx + 1) % len(self.video_files)
            self.file_list.setCurrentRow(idx)

    def _on_file_selected(self, row):
        if row < 0 or row >= len(self.video_files):
            return
        # Save current session before switching
        if self.current_session and self.chk_autosave.isChecked() and self.current_session.events:
            self._save_session()

        self.current_file_idx = row
        video_path = self.video_files[row]

        # Get or create session
        if video_path not in self.sessions:
            session = ScoringSession(video_path)
            # Copy ethogram from current session if available
            if self.current_session and self.current_session.ethogram:
                session.ethogram = [
                    BehaviorDefinition.from_dict(b.to_dict()) for b in self.current_session.ethogram
                ]
                session.subjects = list(self.current_session.subjects)
                session.current_subject = self.current_session.current_subject
            self.sessions[video_path] = session

        self.current_session = self.sessions[video_path]
        self._load_video(video_path)
        self._refresh_ethogram_table()
        self._refresh_subject_table()
        self._refresh_all_views()
        self._refresh_file_list()
        self.status_bar.showMessage(f"Loaded: {Path(video_path).name}")

    # =========================================================================
    # Video Playback
    # =========================================================================

    def _load_video(self, video_path):
        self._release_video()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video:\n{video_path}")
            self.cap = None
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_session.fps = fps if fps > 0 else 30.0
        self.current_session.total_frames = total
        self.current_session.width = w
        self.current_session.height = h
        self.current_session.duration_seconds = total / self.current_session.fps

        self.current_frame_idx = 0
        self.seek_slider.setMaximum(max(0, total - 1))
        self.seek_slider.setValue(0)

        # Read first frame
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            self._display_frame(frame)

        self._update_position_ui()
        self._update_timeline()

    def _release_video(self):
        self.play_timer.stop()
        self.is_playing = False
        self.btn_play_pause.setText("Play")
        if self.cap:
            self.cap.release()
            self.cap = None

    def _display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def _toggle_play_pause(self):
        if self.is_playing:
            self._pause_video()
        else:
            self._play_video()

    def _play_video(self):
        if not self.cap:
            return
        fps = self.current_session.fps if self.current_session else 30.0
        interval_ms = int(1000.0 / (fps * self.playback_speed))
        interval_ms = max(1, interval_ms)
        self.play_timer.start(interval_ms)
        self.is_playing = True
        self.btn_play_pause.setText("Pause")

    def _pause_video(self):
        self.play_timer.stop()
        self.is_playing = False
        self.btn_play_pause.setText("Play")

    def _on_play_tick(self):
        if not self.cap or not self.is_playing:
            return
        ret, frame = self.cap.read()
        if not ret:
            self._pause_video()
            return
        self.current_frame_idx += 1
        self.current_frame = frame
        self._display_frame(frame)
        self._update_position_ui()
        self.timeline_widget.set_current_frame(self.current_frame_idx)

    def _step_frames(self, delta):
        if not self.cap or not self.current_session:
            return
        was_playing = self.is_playing
        if was_playing:
            self._pause_video()
        target = max(0, min(self.current_session.total_frames - 1, self.current_frame_idx + delta))
        self._seek_to_frame(target)

    def _seek_to_frame(self, frame_idx):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = frame_idx
            self.current_frame = frame
            self._display_frame(frame)
            self._update_position_ui()
            self.timeline_widget.set_current_frame(frame_idx)

    def _on_seek_slider_moved(self, value):
        self._seek_to_frame(value)

    def _on_speed_changed(self, text):
        speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
        self.playback_speed = speed_map.get(text, 1.0)
        if self.is_playing:
            self._play_video()  # Restart timer with new interval

    def _update_position_ui(self):
        if not self.current_session:
            return
        total = self.current_session.total_frames
        fps = self.current_session.fps
        current_time = self.current_frame_idx / fps if fps > 0 else 0
        total_time = total / fps if fps > 0 else 0

        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(self.current_frame_idx)
        self.seek_slider.blockSignals(False)

        ct = format_time(current_time)
        tt = format_time(total_time)
        self.lbl_time.setText(f"{ct} / {tt}")

        name = Path(self.current_session.video_path).name
        self.lbl_video_info.setText(
            f"{name}  |  Frame: {self.current_frame_idx}/{total}  |  "
            f"Time: {ct}  |  FPS: {fps:.1f}  |  "
            f"{self.current_session.width}x{self.current_session.height}"
        )

    # =========================================================================
    # Subject
    # =========================================================================

    def _refresh_ethogram_table(self):
        self.ethogram_table.setRowCount(0)
        if not self.current_session:
            return
        for beh in self.current_session.ethogram:
            row = self.ethogram_table.rowCount()
            self.ethogram_table.insertRow(row)
            self.ethogram_table.setItem(row, 0, QTableWidgetItem(beh.name))
            self.ethogram_table.setItem(row, 1, QTableWidgetItem(beh.key.upper()))
            self.ethogram_table.setItem(row, 2, QTableWidgetItem(beh.event_type.capitalize()))

            # Color swatch
            color_item = QTableWidgetItem("")
            color_item.setBackground(QColor(beh.color))
            self.ethogram_table.setItem(row, 3, color_item)

            mods = ", ".join(f"{ms.name}" for ms in beh.modifier_sets) or "--"
            mod_item = QTableWidgetItem(mods)
            if beh.modifier_sets:
                mod_item.setToolTip("\n".join(
                    f"{ms.name} [{ms.type}]: " + ", ".join(ms.value_names)
                    for ms in beh.modifier_sets))
            self.ethogram_table.setItem(row, 4, mod_item)
            self.ethogram_table.setItem(row, 5, QTableWidgetItem(beh.category or "--"))
            excl = ", ".join(beh.exclusions) or "--"
            self.ethogram_table.setItem(row, 6, QTableWidgetItem(excl))

        self._update_timeline()

    def _get_existing_keys(self, skip_behavior=None):
        """Keys already taken, including subject keys -- one keyboard."""
        if not self.current_session:
            return set()
        return set(self._all_bound_keys(skip_behavior=skip_behavior))

    def _add_behavior(self):
        if not self.current_session:
            QMessageBox.information(self, "No Video", "Load a video first.")
            return
        dlg = BehaviorEditDialog(
            self, existing_keys=self._get_existing_keys(),
            all_behaviors=self.current_session.ethogram,
            categories=self.current_session.categories)
        if dlg.exec_() == QDialog.Accepted:
            beh = dlg.get_behavior()
            self.current_session.ethogram.append(beh)
            self._refresh_ethogram_table()
            self._save_session()
            self.status_bar.showMessage(f"Added behavior: {beh.name} [{beh.key.upper()}]")

    def _edit_behavior(self):
        row = self.ethogram_table.currentRow()
        if row < 0 or not self.current_session:
            return
        beh = self.current_session.ethogram[row]
        dlg = BehaviorEditDialog(
            self, behavior=beh,
            existing_keys=self._get_existing_keys(skip_behavior=beh.name),
            all_behaviors=self.current_session.ethogram,
            categories=self.current_session.categories)
        if dlg.exec_() == QDialog.Accepted:
            new_beh = dlg.get_behavior()
            old_name = beh.name
            # Update event references if name changed
            if old_name != new_beh.name:
                for ev in self.current_session.events:
                    if ev.behavior == old_name:
                        ev.behavior = new_beh.name
                for (subj, name) in list(self.current_session.active_states):
                    if name == old_name:
                        self.current_session.active_states[(subj, new_beh.name)] = (
                            self.current_session.active_states.pop((subj, name)))
            self.current_session.ethogram[row] = new_beh
            self._refresh_ethogram_table()
            self._refresh_all_views()
            self._save_session()
            self.status_bar.showMessage(f"Updated behavior: {new_beh.name}")

    def _remove_behavior(self):
        row = self.ethogram_table.currentRow()
        if row < 0 or not self.current_session:
            return
        beh = self.current_session.ethogram[row]
        existing_events = [e for e in self.current_session.events if e.behavior == beh.name]
        if existing_events:
            reply = QMessageBox.warning(
                self, "Behavior Has Events",
                f"'{beh.name}' has {len(existing_events)} scored event(s).\n\n"
                "Removing it will delete all associated events.\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            self.current_session.events = [
                e for e in self.current_session.events if e.behavior != beh.name
            ]
            for key in [k for k in self.current_session.active_states
                        if k[1] == beh.name]:
                self.current_session.active_states.pop(key, None)

        self.current_session.ethogram.pop(row)
        self._refresh_ethogram_table()
        self._update_active_states_list()
        self._save_session()
        self.status_bar.showMessage(f"Removed behavior: {beh.name}")

    def _load_ethogram_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ethogram", "", "JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        if not self.current_session:
            QMessageBox.information(self, "No Video", "Load a video first.")
            return
        # Check for existing events
        if self.current_session.events:
            reply = QMessageBox.warning(
                self, "Replace Ethogram?",
                "Loading a new ethogram may affect existing scored events.\n"
                "Events for behaviors not in the new ethogram will be kept but orphaned.\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            behaviors = [BehaviorDefinition.from_dict(b) for b in data.get("behaviors", [])]
            self.current_session.ethogram = behaviors
            self._refresh_ethogram_table()
            self._save_session()
            self.status_bar.showMessage(f"Loaded ethogram from {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load ethogram:\n{e}")

    def _save_ethogram_file(self):
        if not self.current_session or not self.current_session.ethogram:
            QMessageBox.information(self, "No Ethogram", "Define behaviors first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Ethogram", "ethogram.json", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            data = {
                "version": "1.0",
                "behaviors": [b.to_dict() for b in self.current_session.ethogram],
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            self.status_bar.showMessage(f"Ethogram saved to {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save ethogram:\n{e}")

    def _on_scoring_toggled(self, checked):
        self.scoring_enabled = checked
        state = "enabled" if checked else "disabled"
        self.status_bar.showMessage(f"Scoring {state}")

    def _score_behavior(self, behavior):
        if not self.current_session or self.cap is None:
            return
        if not self.current_session.current_subject and self.current_session.subjects:
            self.current_session.current_subject =                 self.current_session.subjects[0].name

        frame = self.current_frame_idx
        time_s = frame / self.current_session.fps

        # Ask this behavior's own modifier sets. Asked before the event is
        # recorded so a cancelled prompt scores nothing at all.
        modifiers = {}
        if behavior.has_modifiers:
            was_playing = self.is_playing
            if was_playing:
                self._pause_video()
            dlg = ModifierPromptDialog(self, behavior)
            if dlg.exec_() != QDialog.Accepted:
                self.status_bar.showMessage("Cancelled - nothing scored")
                return
            modifiers = dlg.get_modifiers()

        if behavior.event_type == POINT:
            self.current_session.add_point_event(
                frame, time_s, behavior.name, modifiers)
            self._show_feedback(f"POINT: {behavior.name}", behavior.color)
        else:
            if self.current_session.is_state_active(behavior.name):
                self.current_session.stop_state_event(frame, time_s, behavior.name)
                self._show_feedback(f"STOP: {behavior.name}", behavior.color)
            else:
                _, closed = self.current_session.start_state_event(
                    frame, time_s, behavior.name, modifiers)
                label = f"START: {behavior.name}"
                if closed:
                    # Say what was auto-stopped; a silent close looks like the
                    # scorer forgot to end it.
                    label += f"  (stopped {', '.join(closed)})"
                self._show_feedback(label, behavior.color)

        self._refresh_all_views()
        self._autosave()

    def _autosave(self):
        if getattr(self, "chk_autosave", None) and self.chk_autosave.isChecked():
            self._save_session()

    def _save_session(self):
        if not self.current_session:
            return
        folder = self.current_session.output_folder()
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.current_session.event_rows()).to_csv(
            folder / "scoring.csv", index=False)
        with open(folder / "ethogram_config.json", "w", encoding="utf-8") as fh:
            json.dump(self.current_session.config_dict(), fh, indent=2)

    def _undo_last_event(self):
        if not self.current_session or not self.current_session.events:
            return
        ev = self.current_session.undo_last()
        if ev:
            self.status_bar.showMessage(
                f"Undone: {ev.status} {ev.behavior} at frame {ev.frame}"
            )
            self._refresh_all_views()
            if self.chk_autosave.isChecked():
                self._save_session()

    def _show_feedback(self, text, color):
        self.lbl_feedback.setText(text)
        self.lbl_feedback.setStyleSheet(
            f"font-size: 13px; font-weight: bold; padding: 4px 10px; "
            f"border-radius: 4px; background-color: {color}; color: white;"
        )
        self.lbl_feedback.show()
        self._feedback_timer.start(1500)

    def _clear_feedback(self):
        self.lbl_feedback.hide()

    def _update_active_states_list(self):
        self.active_states_list.clear()
        if not self.current_session:
            return
        for (subj, beh_name), start_ev in self.current_session.active_states.items():
            # Find behavior color
            color = "#cccccc"
            for beh in self.current_session.ethogram:
                if beh.name == beh_name:
                    color = beh.color
                    break
            t = format_time(start_ev.time_seconds)
            label = f"{beh_name} ({subj}) since {t}" if subj else                 f"{beh_name} since {t}"
            item = QListWidgetItem(label)
            item.setForeground(QColor(color))
            self.active_states_list.addItem(item)

    def _update_last_event_label(self):
        if not self.current_session or not self.current_session.events:
            self.lbl_last_event.setText("Last event: --")
            return
        ev = self.current_session.events[-1]
        t = format_time(ev.time_seconds)
        self.lbl_last_event.setText(f"Last: {ev.status} {ev.behavior} @ {t}")

    def _update_timeline(self):
        if not self.current_session:
            self.timeline_widget.set_data([], [], {}, 0, 30)
            return
        self.timeline_widget.set_data(
            self.current_session.ethogram,
            self.current_session.events,
            self.current_session.active_states,
            self.current_session.total_frames,
            self.current_session.fps,
        )

    # =========================================================================
    # Export
    # =========================================================================

    def _warn_unpaired(self):
        """Tell the user about states with no stop before they rely on the data."""
        if not self.current_session:
            return
        open_states = self.current_session.unpaired_states()
        if not open_states:
            return
        names = ", ".join(f"{e.behavior} ({e.subject})" if e.subject else e.behavior
                          for e in open_states)
        QMessageBox.warning(
            self, "Unclosed state events",
            f"These state events were never stopped:\n\n{names}\n\n"
            f"They are exported as they stand, but contribute no duration to "
            f"a time budget. Close them, or delete the stray starts, if the "
            f"durations matter.")

    def _export_csv(self):
        if not self.current_session or not self.current_session.events:
            QMessageBox.information(self, "No Data", "No scored events to export.")
            return
        self._save_session()
        self._warn_unpaired()
        folder = self.current_session.output_folder()
        self.status_bar.showMessage(f"Saved to {folder}")
        QMessageBox.information(
            self, "Export Complete",
            f"Scoring data saved to:\n{folder / 'scoring.csv'}\n\n"
            f"Ethogram config saved to:\n{folder / 'ethogram_config.json'}",
        )

    def _open_output_folder(self):
        if not self.current_session:
            return
        folder = self.current_session.output_folder()
        folder.mkdir(parents=True, exist_ok=True)
        import subprocess
        import sys
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        elif sys.platform == "win32":
            os.startfile(str(folder))
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    # =========================================================================
    # Cleanup
    # =========================================================================

    def closeEvent(self, event):
        # Auto-save all sessions
        for path, session in self.sessions.items():
            if session.events:
                folder = session.output_folder()
                folder.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(session.event_rows()).to_csv(
                    folder / "scoring.csv", index=False)
                with open(folder / "ethogram_config.json", "w",
                          encoding="utf-8") as fh:
                    json.dump(session.config_dict(), fh, indent=2)

        # Close open state events warning
        if self.current_session and self.current_session.active_states:
            names = ", ".join(f"{b} ({s})" if s else b
                              for (s, b) in self.current_session.active_states)
            reply = QMessageBox.warning(
                self, "Open State Events",
                f"The following state events are still open:\n{names}\n\n"
                "Close them at the current frame before exiting?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.Yes:
                frame = self.current_frame_idx
                fps = self.current_session.fps
                time_s = frame / fps if fps > 0 else 0
                for subj, beh_name in list(self.current_session.active_states):
                    self.current_session.stop_state_event(frame, time_s, beh_name)
                self._save_session()
                self._save_session()

        self._release_video()
        QApplication.instance().removeEventFilter(self)
        event.accept()
