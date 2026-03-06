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
    QApplication, QCheckBox, QColorDialog, QComboBox, QDialog,
    QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSpinBox, QStatusBar, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget, QHeaderView, QAbstractItemView
)


# =============================================================================
# Data Classes
# =============================================================================

class BehaviorDefinition:
    """Single behavior in the ethogram."""

    DEFAULT_COLORS = [
        "#ff6b35", "#4ecdc4", "#ffe66d", "#a8e6cf", "#ff8b94",
        "#7ec8e3", "#c9b1ff", "#f7dc6f", "#82e0aa", "#f1948a",
        "#85c1e9", "#d7bde2", "#f0b27a", "#76d7c4", "#f9e79f",
    ]
    _color_idx = 0

    def __init__(self, name="", key="", event_type="point", color="", modifiers=None):
        self.name = name
        self.key = key
        self.event_type = event_type  # "point" or "state"
        if not color:
            color = BehaviorDefinition.DEFAULT_COLORS[
                BehaviorDefinition._color_idx % len(BehaviorDefinition.DEFAULT_COLORS)
            ]
            BehaviorDefinition._color_idx += 1
        self.color = color
        self.modifiers = modifiers if modifiers is not None else []

    def to_dict(self):
        return {
            "name": self.name,
            "key": self.key,
            "event_type": self.event_type,
            "color": self.color,
            "modifiers": self.modifiers,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d.get("name", ""),
            key=d.get("key", ""),
            event_type=d.get("event_type", "point"),
            color=d.get("color", ""),
            modifiers=d.get("modifiers", []),
        )


class ScoringEvent:
    """A single scored behavioral event."""

    def __init__(self, frame, time_seconds, subject, behavior, modifier, event_type, status):
        self.frame = frame
        self.time_seconds = time_seconds
        self.subject = subject
        self.behavior = behavior
        self.modifier = modifier
        self.event_type = event_type  # "point" or "state"
        self.status = status  # "POINT", "START", "STOP"


class ScoringSession:
    """All scoring data for one video file."""

    def __init__(self, video_path):
        self.video_path = video_path
        self.ethogram = []  # List[BehaviorDefinition]
        self.events = []  # List[ScoringEvent]
        self.active_states = {}  # behavior_name -> ScoringEvent (start)
        self.subject = ""

        # Video metadata
        self.total_frames = 0
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self.duration_seconds = 0.0

    def output_folder(self):
        stem = Path(self.video_path).stem
        parent = Path(self.video_path).parent
        return parent / f"{stem}_fntScoring"

    def add_point_event(self, frame, time_s, behavior, modifier=""):
        ev = ScoringEvent(frame, time_s, self.subject, behavior, modifier, "point", "POINT")
        self.events.append(ev)
        return ev

    def start_state_event(self, frame, time_s, behavior, modifier=""):
        ev = ScoringEvent(frame, time_s, self.subject, behavior, modifier, "state", "START")
        self.events.append(ev)
        self.active_states[behavior] = ev
        return ev

    def stop_state_event(self, frame, time_s, behavior):
        start_ev = self.active_states.pop(behavior, None)
        modifier = start_ev.modifier if start_ev else ""
        ev = ScoringEvent(frame, time_s, self.subject, behavior, modifier, "state", "STOP")
        self.events.append(ev)
        return ev

    def is_state_active(self, behavior_name):
        return behavior_name in self.active_states

    def undo_last(self):
        if not self.events:
            return None
        ev = self.events.pop()
        if ev.status == "START":
            self.active_states.pop(ev.behavior, None)
        elif ev.status == "STOP":
            # Re-find the matching START and restore it as active
            for prev in reversed(self.events):
                if prev.behavior == ev.behavior and prev.status == "START":
                    self.active_states[ev.behavior] = prev
                    break
        return ev

    @staticmethod
    def _format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    def to_dataframe(self):
        rows = []
        for ev in self.events:
            rows.append({
                "frame": ev.frame,
                "time": self._format_time(ev.time_seconds),
                "time_seconds": round(ev.time_seconds, 4),
                "subject": ev.subject,
                "behavior": ev.behavior,
                "modifier": ev.modifier,
                "type": ev.event_type,
                "status": ev.status,
            })
        return pd.DataFrame(rows)

    def save_csv(self):
        folder = self.output_folder()
        folder.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(folder / "scoring.csv", index=False)

    def save_config(self):
        folder = self.output_folder()
        folder.mkdir(parents=True, exist_ok=True)
        config = {
            "version": "1.0",
            "video_path": str(self.video_path),
            "subject": self.subject,
            "behaviors": [b.to_dict() for b in self.ethogram],
        }
        with open(folder / "ethogram_config.json", "w") as f:
            json.dump(config, f, indent=2)

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.subject = config.get("subject", "")
        self.ethogram = [BehaviorDefinition.from_dict(b) for b in config.get("behaviors", [])]

    def load_existing_scoring(self, csv_path):
        df = pd.read_csv(csv_path)
        self.events = []
        self.active_states = {}
        for _, row in df.iterrows():
            ev = ScoringEvent(
                frame=int(row.get("frame", 0)),
                time_seconds=float(row.get("time_seconds", 0.0)),
                subject=str(row.get("subject", "")),
                behavior=str(row.get("behavior", "")),
                modifier=str(row.get("modifier", "")),
                event_type=str(row.get("type", "point")),
                status=str(row.get("status", "POINT")),
            )
            self.events.append(ev)
        # Rebuild active states: find STARTs without matching STOPs
        start_stack = {}
        for ev in self.events:
            if ev.status == "START":
                start_stack[ev.behavior] = ev
            elif ev.status == "STOP":
                start_stack.pop(ev.behavior, None)
        self.active_states = start_stack


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
            if beh.name in self.active_states:
                start_ev = self.active_states[beh.name]
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

class BehaviorEditDialog(QDialog):
    """Dialog for adding/editing a behavior definition."""

    def __init__(self, parent=None, behavior=None, existing_keys=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Behavior" if behavior else "Add Behavior")
        self.setMinimumWidth(380)
        self.existing_keys = existing_keys or set()
        self.selected_color = behavior.color if behavior else ""

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Name
        layout.addWidget(QLabel("Behavior Name:"))
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("e.g., Grooming")
        layout.addWidget(self.edit_name)

        # Key
        layout.addWidget(QLabel("Shortcut Key (single character):"))
        self.edit_key = QLineEdit()
        self.edit_key.setMaxLength(1)
        self.edit_key.setPlaceholderText("e.g., g")
        layout.addWidget(self.edit_key)

        # Type
        layout.addWidget(QLabel("Event Type:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(["point", "state"])
        layout.addWidget(self.combo_type)

        # Color
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color:"))
        self.btn_color = QPushButton("Pick Color")
        self.btn_color.clicked.connect(self._pick_color)
        color_row.addWidget(self.btn_color)
        self.lbl_color_preview = QLabel()
        self.lbl_color_preview.setFixedSize(24, 24)
        color_row.addWidget(self.lbl_color_preview)
        color_row.addStretch()
        layout.addLayout(color_row)

        # Modifiers
        layout.addWidget(QLabel("Modifiers (comma-separated, optional):"))
        self.edit_modifiers = QLineEdit()
        self.edit_modifiers.setPlaceholderText("e.g., self, allo")
        layout.addWidget(self.edit_modifiers)

        # Buttons
        btn_row = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self._validate_and_accept)
        btn_row.addWidget(btn_ok)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background-color: #5c5c5c;")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        # Pre-fill if editing
        if behavior:
            self.edit_name.setText(behavior.name)
            self.edit_key.setText(behavior.key)
            self.combo_type.setCurrentText(behavior.event_type)
            self.selected_color = behavior.color
            self.edit_modifiers.setText(", ".join(behavior.modifiers))
            # Don't count the behavior's own key as a conflict
            self.existing_keys.discard(behavior.key.lower())

        self._update_color_preview()
        self._apply_styles()

    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: #cccccc; }
            QLabel { color: #cccccc; background-color: transparent; }
            QLineEdit, QComboBox {
                padding: 4px; border: 1px solid #3f3f3f; border-radius: 3px;
                background-color: #1e1e1e; color: #cccccc;
            }
            QPushButton {
                background-color: #0078d4; color: white; border: none;
                padding: 6px 12px; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background-color: #106ebe; }
        """)

    def _pick_color(self):
        color = QColorDialog.getColor(QColor(self.selected_color or "#0078d4"), self)
        if color.isValid():
            self.selected_color = color.name()
            self._update_color_preview()

    def _update_color_preview(self):
        if self.selected_color:
            self.lbl_color_preview.setStyleSheet(
                f"background-color: {self.selected_color}; border: 1px solid #3f3f3f; border-radius: 3px;"
            )

    def _validate_and_accept(self):
        name = self.edit_name.text().strip()
        key = self.edit_key.text().strip().lower()
        if not name:
            QMessageBox.warning(self, "Validation", "Behavior name is required.")
            return
        if not key:
            QMessageBox.warning(self, "Validation", "Shortcut key is required.")
            return
        if key in self.existing_keys:
            QMessageBox.warning(self, "Validation", f"Key '{key}' is already assigned to another behavior.")
            return
        self.accept()

    def get_behavior(self):
        mods_text = self.edit_modifiers.text().strip()
        modifiers = [m.strip() for m in mods_text.split(",") if m.strip()] if mods_text else []
        return BehaviorDefinition(
            name=self.edit_name.text().strip(),
            key=self.edit_key.text().strip().lower(),
            event_type=self.combo_type.currentText(),
            color=self.selected_color or "#0078d4",
            modifiers=modifiers,
        )


# =============================================================================
# Main Window
# =============================================================================

class BehaviorScoringStudioWindow(QMainWindow):
    """Main window for Behavior Scoring Studio."""

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    def __init__(self):
        super().__init__()
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

        # Behavior scoring keys
        if self.scoring_enabled and self.current_session and self.cap:
            key_text = event.text().lower()
            if key_text and len(key_text) == 1:
                for beh in self.current_session.ethogram:
                    if beh.key.lower() == key_text:
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

        main_layout.addWidget(left_scroll)
        main_layout.addWidget(right_panel, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Welcome to Behavior Scoring Studio - Load video files to begin")

    def _create_input_section(self, layout):
        group = QGroupBox("1. Input")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.clicked.connect(self._add_folder)
        btn_row.addWidget(self.btn_add_folder)

        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.clicked.connect(self._add_files)
        btn_row.addWidget(self.btn_add_files)

        self.btn_clear_files = QPushButton("Clear")
        self.btn_clear_files.setStyleSheet("background-color: #5c5c5c;")
        self.btn_clear_files.clicked.connect(self._clear_files)
        btn_row.addWidget(self.btn_clear_files)

        group_layout.addLayout(btn_row)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(120)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        group_layout.addWidget(self.file_list)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)

        self.btn_prev_file = QPushButton("< Prev")
        self.btn_prev_file.setObjectName("small_btn")
        self.btn_prev_file.clicked.connect(self._prev_file)
        self.btn_prev_file.setEnabled(False)
        nav_row.addWidget(self.btn_prev_file)

        self.lbl_file_num = QLabel("File 0/0")
        self.lbl_file_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_file_num, 1)

        self.btn_next_file = QPushButton("Next >")
        self.btn_next_file.setObjectName("small_btn")
        self.btn_next_file.clicked.connect(self._next_file)
        self.btn_next_file.setEnabled(False)
        nav_row.addWidget(self.btn_next_file)

        group_layout.addLayout(nav_row)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_subject_section(self, layout):
        group = QGroupBox("2. Subject")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        self.edit_subject = QLineEdit()
        self.edit_subject.setPlaceholderText("Subject ID (optional)")
        self.edit_subject.textChanged.connect(self._on_subject_changed)
        group_layout.addWidget(self.edit_subject)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_ethogram_section(self, layout):
        group = QGroupBox("3. Ethogram")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Table
        self.ethogram_table = QTableWidget(0, 5)
        self.ethogram_table.setHorizontalHeaderLabels(["Name", "Key", "Type", "Color", "Modifiers"])
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
        group_layout.addWidget(self.ethogram_table)

        # Add/Edit/Remove buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_add_behavior = QPushButton("Add")
        self.btn_add_behavior.clicked.connect(self._add_behavior)
        btn_row.addWidget(self.btn_add_behavior)

        self.btn_edit_behavior = QPushButton("Edit")
        self.btn_edit_behavior.clicked.connect(self._edit_behavior)
        btn_row.addWidget(self.btn_edit_behavior)

        self.btn_remove_behavior = QPushButton("Remove")
        self.btn_remove_behavior.setStyleSheet("background-color: #d13438;")
        self.btn_remove_behavior.clicked.connect(self._remove_behavior)
        btn_row.addWidget(self.btn_remove_behavior)

        group_layout.addLayout(btn_row)

        # Load/Save ethogram
        io_row = QHBoxLayout()
        io_row.setSpacing(2)

        self.btn_load_ethogram = QPushButton("Load Ethogram")
        self.btn_load_ethogram.setStyleSheet("background-color: #5c5c5c;")
        self.btn_load_ethogram.clicked.connect(self._load_ethogram_file)
        io_row.addWidget(self.btn_load_ethogram)

        self.btn_save_ethogram = QPushButton("Save Ethogram")
        self.btn_save_ethogram.setStyleSheet("background-color: #5c5c5c;")
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
        self.chk_scoring.toggled.connect(self._on_scoring_toggled)
        group_layout.addWidget(self.chk_scoring)

        # Modifier selector
        mod_row = QHBoxLayout()
        mod_row.addWidget(QLabel("Modifier:"))
        self.combo_modifier = QComboBox()
        self.combo_modifier.addItem("None")
        mod_row.addWidget(self.combo_modifier, 1)
        group_layout.addLayout(mod_row)

        # Active states list
        group_layout.addWidget(QLabel("Active State Events:"))
        self.active_states_list = QListWidget()
        self.active_states_list.setMaximumHeight(80)
        group_layout.addWidget(self.active_states_list)

        # Last event label
        self.lbl_last_event = QLabel("Last event: --")
        self.lbl_last_event.setStyleSheet("color: #999999; font-size: 10px;")
        group_layout.addWidget(self.lbl_last_event)

        # Undo button
        self.btn_undo = QPushButton("Undo Last (Ctrl+Z)")
        self.btn_undo.setStyleSheet("background-color: #5c5c5c;")
        self.btn_undo.clicked.connect(self._undo_last_event)
        group_layout.addWidget(self.btn_undo)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_export_section(self, layout):
        group = QGroupBox("5. Export")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        self.chk_autosave = QCheckBox("Auto-save on event")
        self.chk_autosave.setChecked(True)
        group_layout.addWidget(self.chk_autosave)

        self.btn_export_csv = QPushButton("Export Scoring CSV")
        self.btn_export_csv.clicked.connect(self._export_csv)
        group_layout.addWidget(self.btn_export_csv)

        self.btn_open_folder = QPushButton("Open Output Folder")
        self.btn_open_folder.setStyleSheet("background-color: #5c5c5c;")
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
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
                font-family: Arial;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
                min-height: 18px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
            QPushButton#small_btn {
                padding: 4px 8px;
                min-height: 16px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 6px;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 4px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QListWidget, QTableWidget {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QListWidget::item {
                padding: 2px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QTableWidget::item {
                padding: 2px;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            QHeaderView::section {
                background-color: #333333;
                color: #cccccc;
                padding: 4px;
                border: 1px solid #3f3f3f;
                font-weight: bold;
            }
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #0078d4;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #106ebe;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                background-color: #2b2b2b;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #0078d4;
                border-radius: 4px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #106ebe;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3f3f3f;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #106ebe;
            }
            QLabel#video_info {
                font-size: 10px;
                color: #999999;
                padding: 4px;
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 2px;
            }
            QLabel#scoring_feedback {
                font-size: 13px;
                font-weight: bold;
                padding: 4px 10px;
                border-radius: 4px;
                background-color: rgba(0, 0, 0, 180);
            }
        """)

    # =========================================================================
    # File Management
    # =========================================================================

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
                session.load_existing_scoring(csv_path)
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
            self.current_session.save_csv()
            self.current_session.save_config()

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
                session.subject = self.current_session.subject
            self.sessions[video_path] = session

        self.current_session = self.sessions[video_path]
        self._load_video(video_path)
        self._refresh_ethogram_table()
        self._update_modifier_combo()
        self._update_active_states_list()
        self.edit_subject.setText(self.current_session.subject)
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

        ct = ScoringSession._format_time(current_time)
        tt = ScoringSession._format_time(total_time)
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

    def _on_subject_changed(self, text):
        if self.current_session:
            self.current_session.subject = text.strip()

    # =========================================================================
    # Ethogram Management
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

            mods = ", ".join(beh.modifiers) if beh.modifiers else "--"
            self.ethogram_table.setItem(row, 4, QTableWidgetItem(mods))

        self._update_timeline()

    def _get_existing_keys(self):
        if not self.current_session:
            return set()
        return {b.key.lower() for b in self.current_session.ethogram}

    def _add_behavior(self):
        if not self.current_session:
            QMessageBox.information(self, "No Video", "Load a video first.")
            return
        dlg = BehaviorEditDialog(self, existing_keys=self._get_existing_keys())
        if dlg.exec_() == QDialog.Accepted:
            beh = dlg.get_behavior()
            self.current_session.ethogram.append(beh)
            self._refresh_ethogram_table()
            self._update_modifier_combo()
            self.current_session.save_config()
            self.status_bar.showMessage(f"Added behavior: {beh.name} [{beh.key.upper()}]")

    def _edit_behavior(self):
        row = self.ethogram_table.currentRow()
        if row < 0 or not self.current_session:
            return
        beh = self.current_session.ethogram[row]
        dlg = BehaviorEditDialog(self, behavior=beh, existing_keys=self._get_existing_keys())
        if dlg.exec_() == QDialog.Accepted:
            new_beh = dlg.get_behavior()
            old_name = beh.name
            # Update event references if name changed
            if old_name != new_beh.name:
                for ev in self.current_session.events:
                    if ev.behavior == old_name:
                        ev.behavior = new_beh.name
                if old_name in self.current_session.active_states:
                    self.current_session.active_states[new_beh.name] = (
                        self.current_session.active_states.pop(old_name)
                    )
            self.current_session.ethogram[row] = new_beh
            self._refresh_ethogram_table()
            self._update_modifier_combo()
            self._update_active_states_list()
            self.current_session.save_config()
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
            self.current_session.active_states.pop(beh.name, None)

        self.current_session.ethogram.pop(row)
        self._refresh_ethogram_table()
        self._update_modifier_combo()
        self._update_active_states_list()
        self.current_session.save_config()
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
            self._update_modifier_combo()
            self.current_session.save_config()
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

    def _update_modifier_combo(self):
        self.combo_modifier.clear()
        self.combo_modifier.addItem("None")
        if not self.current_session:
            return
        # Collect all unique modifiers across ethogram
        all_mods = set()
        for beh in self.current_session.ethogram:
            for m in beh.modifiers:
                all_mods.add(m)
        for m in sorted(all_mods):
            self.combo_modifier.addItem(m)

    # =========================================================================
    # Scoring
    # =========================================================================

    def _on_scoring_toggled(self, checked):
        self.scoring_enabled = checked
        state = "enabled" if checked else "disabled"
        self.status_bar.showMessage(f"Scoring {state}")

    def _score_behavior(self, behavior):
        if not self.current_session or self.cap is None:
            return

        frame = self.current_frame_idx
        time_s = frame / self.current_session.fps

        modifier = self.combo_modifier.currentText()
        if modifier == "None":
            modifier = ""

        if behavior.event_type == "point":
            self.current_session.add_point_event(frame, time_s, behavior.name, modifier)
            self._show_feedback(f"POINT: {behavior.name}", behavior.color)
        elif behavior.event_type == "state":
            if self.current_session.is_state_active(behavior.name):
                self.current_session.stop_state_event(frame, time_s, behavior.name)
                self._show_feedback(f"STOP: {behavior.name}", behavior.color)
            else:
                self.current_session.start_state_event(frame, time_s, behavior.name, modifier)
                self._show_feedback(f"START: {behavior.name}", behavior.color)

        self._update_active_states_list()
        self._update_timeline()
        self._update_last_event_label()

        if self.chk_autosave.isChecked():
            self.current_session.save_csv()
            self.current_session.save_config()

    def _undo_last_event(self):
        if not self.current_session or not self.current_session.events:
            return
        ev = self.current_session.undo_last()
        if ev:
            self.status_bar.showMessage(
                f"Undone: {ev.status} {ev.behavior} at frame {ev.frame}"
            )
            self._update_active_states_list()
            self._update_timeline()
            self._update_last_event_label()
            if self.chk_autosave.isChecked():
                self.current_session.save_csv()

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
        for beh_name, start_ev in self.current_session.active_states.items():
            # Find behavior color
            color = "#cccccc"
            for beh in self.current_session.ethogram:
                if beh.name == beh_name:
                    color = beh.color
                    break
            t = ScoringSession._format_time(start_ev.time_seconds)
            item = QListWidgetItem(f"{beh_name} (started at {t})")
            item.setForeground(QColor(color))
            self.active_states_list.addItem(item)

    def _update_last_event_label(self):
        if not self.current_session or not self.current_session.events:
            self.lbl_last_event.setText("Last event: --")
            return
        ev = self.current_session.events[-1]
        t = ScoringSession._format_time(ev.time_seconds)
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

    def _export_csv(self):
        if not self.current_session or not self.current_session.events:
            QMessageBox.information(self, "No Data", "No scored events to export.")
            return
        self.current_session.save_csv()
        self.current_session.save_config()
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
                session.save_csv()
                session.save_config()

        # Close open state events warning
        if self.current_session and self.current_session.active_states:
            names = ", ".join(self.current_session.active_states.keys())
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
                for beh_name in list(self.current_session.active_states.keys()):
                    self.current_session.stop_state_event(frame, time_s, beh_name)
                self.current_session.save_csv()
                self.current_session.save_config()

        self._release_video()
        QApplication.instance().removeEventFilter(self)
        event.accept()
