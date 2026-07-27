"""FED3 monitoring and control tab.

Ties together the serial links (:mod:`fed_serial`), the SD-card mirror
(:mod:`fed_mirror`), the recording session (:mod:`fed_session`), the webcam
(:mod:`fed_webcam`), the scheduler (:mod:`fed_scheduler`) and the plot
(:mod:`fed_plot`).

Design notes for the parts that changed most:

*Connections.* Each device owns exactly one :class:`~fnt.fed3.fed_serial.Fed3Link`
for the lifetime of its port. Commands are queued onto that link rather than
opening the port again, which is what previously produced port-busy lockouts.

*Recording.* Behavioural events, camera frames and every user action share one
host time base, so a pellet and a video frame can be compared without alignment.
Session state is persisted continuously, so a crash is resumable.

*Plot refresh.* Events mark the plot dirty; a 1 Hz timer redraws. Redrawing per
event made a busy rig unresponsive during pellet bursts.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

from PyQt5.QtCore import Qt, QTime, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressDialog,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTableWidget,
    QTableWidgetItem, QTimeEdit, QVBoxLayout, QWidget,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from . import fed_protocol as proto
from . import fed_scheduler as sched
from . import fed_session as session_mod
from .fed_device import FedDevice
from .fed_export import Fed3Transfer
from .fed_mirror import DeviceMirror
from .fed_plot import WINDOWS, FedPlotManager, PlotSeries
from .fed_serial import Fed3Link, PortScannerWorker
from .fed_ui import (
    CollapsibleLogBox, FEDSvgView, FileSelectorDialog, FlowLayout,
    ResumeSessionDialog,
)
from .fed_webcam import WebcamRecorder, list_cameras

UI_TICK_MS = 1000               # scheduler countdowns and plot refresh
RECONNECT_TICK_MS = 5000        # auto-reconnect sweep
STATE_SAVE_MS = 15000           # session state persistence
NARROW_WIDTH = 1120             # below this the panels stack into one column

# Reconnect backoff: give up automatic retries after this many consecutive
# failures so a permanently unplugged device stops churning the port list.
MAX_RECONNECT_ATTEMPTS = 20

PORT_PLACEHOLDERS = ("Scanning...", "No FED3 found", "")


class FEDTabWidget(QWidget):
    """Live monitoring, control and recording for a bank of FED3 devices."""

    scan_finished = pyqtSignal(list, list)

    def __init__(self, parent=None, worker_class=None):
        super().__init__(parent)
        self.main_window = parent
        self._worker_class = worker_class    # retained for API compatibility

        self.devices = []
        self.session = None
        self.logger = session_mod.SessionLogger()
        self.scheduler = sched.Scheduler()
        self.webcam = None
        self.sessions_dir = session_mod.default_session_root()
        self.removed_ports = set()
        self._port_info = {}                 # port -> {"id":.., "firmware":..}
        self._all_ports = []                 # every system port from the last scan
        self._scanner = None
        self._plot_dirty = True
        self._sched_rows = {}                # event id -> table row

        self.scan_finished.connect(self._on_scan_finished)
        self._build_ui()
        self._start_timers()

        self.logger.log("FED3 tab opened", source="system")
        self._offer_resume()
        self.refresh_ports()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(10, 10, 10, 10)
        self.content_layout.setSpacing(12)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        self.content_layout.addWidget(self._build_session_group())

        self.columns = QWidget()
        self.columns_layout = QGridLayout(self.columns)
        self.columns_layout.setContentsMargins(0, 0, 0, 0)
        self.columns_layout.setSpacing(12)
        self.content_layout.addWidget(self.columns)

        self.control_group = self._build_control_group()
        self.scheduler_group = self._build_scheduler_group()
        self.devices_group = self._build_devices_group()

        self.left_column = QWidget()
        self.left_column_layout = QVBoxLayout(self.left_column)
        self.left_column_layout.setContentsMargins(0, 0, 0, 0)
        self.left_column_layout.setSpacing(12)

        self.right_column = QWidget()
        self.right_column_layout = QVBoxLayout(self.right_column)
        self.right_column_layout.setContentsMargins(0, 0, 0, 0)
        self.right_column_layout.setSpacing(12)

        self._layout_is_narrow = None
        self._apply_responsive_layout()

        self.content_layout.addWidget(self._build_plot_group())
        self.content_layout.addWidget(self._build_device_view_group())
        self.content_layout.addStretch()

        self.log = CollapsibleLogBox("Serial Monitor")
        self.log.command_submitted.connect(self._on_raw_command)
        outer.addWidget(self.log)

    # --- session / recording ---------------------------------------------

    def _build_session_group(self):
        group = QGroupBox("Recording Session")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        row = QHBoxLayout()
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setStyleSheet("""
            QPushButton { font-weight: bold; min-height: 26px; }
            QPushButton:checked { background-color: #c0392b; color: white; }
        """)
        self.record_btn.setCheckable(True)
        self.record_btn.setToolTip(
            "Start a timestamped session folder. Behavioural events, the SD-card "
            "mirror, webcam video and every user action are recorded into it on a "
            "shared clock.")
        self.record_btn.toggled.connect(self._on_record_toggled)
        row.addWidget(self.record_btn)

        self.session_label = QLabel("No session recording")
        self.session_label.setStyleSheet("color: #999999;")
        row.addWidget(self.session_label, stretch=1)

        self.open_folder_btn = QPushButton("Open Folder")
        self.open_folder_btn.setEnabled(False)
        self.open_folder_btn.clicked.connect(self._open_session_folder)
        row.addWidget(self.open_folder_btn)

        choose_btn = QPushButton("Change Location...")
        choose_btn.clicked.connect(self._choose_sessions_dir)
        row.addWidget(choose_btn)
        layout.addLayout(row)

        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera:"))
        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip(
            "Detected cameras. While a session is recording, video is written "
            "with a per-frame timestamp CSV on the same clock as FED events.")
        camera_row.addWidget(self.camera_combo)

        self.camera_btn = QPushButton("Start Camera")
        self.camera_btn.clicked.connect(self._toggle_camera)
        camera_row.addWidget(self.camera_btn)

        rescan_btn = QPushButton("Rescan")
        rescan_btn.clicked.connect(self._populate_cameras)
        camera_row.addWidget(rescan_btn)

        self.camera_status = QLabel("Camera off")
        self.camera_status.setStyleSheet("color: #999999;")
        camera_row.addWidget(self.camera_status, stretch=1)
        layout.addLayout(camera_row)

        self.camera_view = QLabel("Camera off")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumHeight(240)
        self.camera_view.setStyleSheet(
            "background-color: #1e1e1e; color: #777777; "
            "border: 1px dashed #444444; border-radius: 6px;")
        self.camera_view.setVisible(False)
        layout.addWidget(self.camera_view)

        self._populate_cameras()
        return group

    # --- global control ---------------------------------------------------

    def _build_control_group(self):
        group = QGroupBox("FED Control Panel")
        layout = QHBoxLayout(group)
        layout.setSpacing(15)

        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(_section_label("Device configuration & commands"))

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Global mode:"))
        self.global_mode_combo = QComboBox()
        self.global_mode_combo.addItems(proto.MODE_LABELS)
        self.global_mode_combo.currentTextChanged.connect(
            lambda: _sync_mode_params(self.global_mode_combo, self.global_params))
        mode_row.addWidget(self.global_mode_combo)
        mode_row.addStretch()
        left.addLayout(mode_row)

        self.global_params = _ModeParams()
        left.addWidget(self.global_params)

        actions = QHBoxLayout()
        apply_btn = QPushButton("Apply to All")
        apply_btn.setStyleSheet("font-weight: bold; min-height: 22px;")
        apply_btn.clicked.connect(self._apply_global_mode)
        dispense_btn = QPushButton("Dispense All")
        dispense_btn.setStyleSheet("font-weight: bold; min-height: 22px;")
        dispense_btn.clicked.connect(self._dispense_all)
        self.global_lights_btn = QPushButton("Lights: OFF")
        self.global_lights_btn.setCheckable(True)
        self.global_lights_btn.setStyleSheet("""
            QPushButton:checked { background-color: #f1c40f; color: black; }
            QPushButton { min-height: 22px; font-weight: bold; }
        """)
        self.global_lights_btn.clicked.connect(self._toggle_global_lights)
        for widget in (apply_btn, dispense_btn, self.global_lights_btn):
            actions.addWidget(widget)
        actions.addStretch()
        left.addLayout(actions)
        left.addStretch()

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setStyleSheet("background-color: #333333; margin: 0px 5px;")

        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(_section_label("Clock sync & data"))

        sync_row = QHBoxLayout()
        sync_row.addWidget(QLabel("Auto sync every:"))
        self.sync_interval_spin = QSpinBox()
        self.sync_interval_spin.setRange(1, 99999)
        self.sync_interval_spin.setValue(6)
        self.sync_interval_spin.setFixedWidth(70)
        self.sync_unit_combo = QComboBox()
        self.sync_unit_combo.addItems(["Minutes", "Hours", "Days"])
        self.sync_unit_combo.setCurrentText("Hours")
        self.sync_interval_spin.valueChanged.connect(self._restart_sync_timer)
        self.sync_unit_combo.currentTextChanged.connect(self._restart_sync_timer)
        sync_row.addWidget(self.sync_interval_spin)
        sync_row.addWidget(self.sync_unit_combo)
        sync_row.addStretch()
        right.addLayout(sync_row)

        sync_actions = QHBoxLayout()
        self.auto_sync_btn = QPushButton("Auto Sync: ON")
        self.auto_sync_btn.setCheckable(True)
        self.auto_sync_btn.setChecked(True)
        self.auto_sync_btn.setStyleSheet("""
            QPushButton:checked { background-color: #4caf50; color: white; }
            QPushButton { font-weight: bold; }
        """)
        self.auto_sync_btn.toggled.connect(self._on_auto_sync_toggled)
        sync_now_btn = QPushButton("Sync Now")
        sync_now_btn.setStyleSheet("font-weight: bold;")
        sync_now_btn.clicked.connect(self._sync_all)
        sync_actions.addWidget(self.auto_sync_btn)
        sync_actions.addWidget(sync_now_btn)
        sync_actions.addStretch()
        right.addLayout(sync_actions)

        data_row = QHBoxLayout()
        export_btn = QPushButton("Export SD Logs...")
        export_btn.setStyleSheet("font-weight: bold;")
        export_btn.clicked.connect(self._export_all)
        pull_btn = QPushButton("Pull Data Now")
        pull_btn.setToolTip(
            "Force an immediate mirror pull from every connected device.")
        pull_btn.clicked.connect(self._force_mirror_sync)
        reset_btn = QPushButton("New Trial (All)")
        reset_btn.setStyleSheet(
            "font-weight: bold; background-color: #c0392b; color: white;")
        reset_btn.clicked.connect(self._new_trial_all)
        for widget in (export_btn, pull_btn, reset_btn):
            data_row.addWidget(widget)
        data_row.addStretch()
        right.addLayout(data_row)

        self.mirror_status = QLabel("Mirror idle")
        self.mirror_status.setStyleSheet("color: #999999; font-size: 11px;")
        right.addWidget(self.mirror_status)
        right.addStretch()

        layout.addLayout(left, stretch=1)
        layout.addWidget(divider)
        layout.addLayout(right, stretch=1)
        _sync_mode_params(self.global_mode_combo, self.global_params)
        return group

    # --- devices ----------------------------------------------------------

    def _build_devices_group(self):
        group = QGroupBox("Connected Devices")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        add_btn = QPushButton("Add Device")
        add_btn.clicked.connect(lambda: self.add_device_slot())
        self.refresh_btn = QPushButton("Refresh Ports")
        self.refresh_btn.setToolTip(
            "Scan for FED3 devices. Ports already held by a connected device are "
            "never reopened, so this is safe to run mid-experiment.")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        controls.addWidget(add_btn)
        controls.addStretch()
        controls.addWidget(self.refresh_btn)
        layout.addLayout(controls)

        self.devices_container = QWidget()
        self.devices_flow = FlowLayout(margin=4, spacing=8)
        self.devices_container.setLayout(self.devices_flow)
        layout.addWidget(self.devices_container)
        return group

    # --- plot -------------------------------------------------------------

    def _build_plot_group(self):
        group = QGroupBox("Activity")
        layout = QVBoxLayout(group)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("Show:"))
        self.plot_filter_combo = QComboBox()
        self.plot_filter_combo.addItem("All Devices")
        self.plot_filter_combo.currentIndexChanged.connect(self._mark_plot_dirty)
        toolbar.addWidget(self.plot_filter_combo)

        toolbar.addWidget(QLabel("Window:"))
        self.plot_window_combo = QComboBox()
        for label, hours in WINDOWS:
            self.plot_window_combo.addItem(label, hours)
        self.plot_window_combo.setCurrentIndex(len(WINDOWS) - 1)
        self.plot_window_combo.currentIndexChanged.connect(self._on_plot_window_changed)
        toolbar.addWidget(self.plot_window_combo)

        self.dark_cycle_check = QCheckBox("Shade dark cycle")
        self.dark_cycle_check.setChecked(True)
        self.dark_cycle_check.toggled.connect(self._on_dark_cycle_changed)
        toolbar.addWidget(self.dark_cycle_check)

        self.lights_off_spin = QSpinBox()
        self.lights_off_spin.setRange(0, 23)
        self.lights_off_spin.setValue(19)
        self.lights_off_spin.setSuffix(":00 off")
        self.lights_off_spin.valueChanged.connect(self._on_dark_cycle_changed)
        self.lights_on_spin = QSpinBox()
        self.lights_on_spin.setRange(0, 23)
        self.lights_on_spin.setValue(7)
        self.lights_on_spin.setSuffix(":00 on")
        self.lights_on_spin.valueChanged.connect(self._on_dark_cycle_changed)
        toolbar.addWidget(self.lights_off_spin)
        toolbar.addWidget(self.lights_on_spin)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        self.plot_placeholder = QLabel(
            "No pellet data yet.\nThe activity graph appears once a device "
            "reports pellet delivery.")
        self.plot_placeholder.setAlignment(Qt.AlignCenter)
        self.plot_placeholder.setFont(QFont("Arial", 11))
        self.plot_placeholder.setStyleSheet(
            "color: #888888; padding: 40px; background-color: #1e1e1e; "
            "border: 1px dashed #444444; border-radius: 6px;")
        layout.addWidget(self.plot_placeholder)

        figure = Figure(figsize=(6, 3.2))
        figure.patch.set_facecolor("#2b2b2b")
        self.canvas = FigureCanvas(figure)
        self.canvas.setMinimumHeight(300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setVisible(False)
        layout.addWidget(self.canvas)

        self.plot_manager = FedPlotManager(
            self.canvas, figure.add_subplot(111), self.plot_placeholder)
        return group

    def _build_device_view_group(self):
        group = QGroupBox("Device View")
        layout = QVBoxLayout(group)
        container = QWidget()
        self.device_view_flow = FlowLayout(margin=10, spacing=20)
        container.setLayout(self.device_view_flow)
        layout.addWidget(container)
        return group

    # ==================================================================
    # Scheduler UI
    # ==================================================================

    def _build_scheduler_group(self):
        group = QGroupBox("Protocol Event Scheduler")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.sched_table = QTableWidget(0, 7)
        self.sched_table.setHorizontalHeaderLabels(
            ["On", "Fires at", "Countdown", "Target", "Action", "Status", ""])
        self.sched_table.verticalHeader().setVisible(False)
        self.sched_table.setSelectionBehavior(QTableWidget.SelectRows)
        header = self.sched_table.horizontalHeader()
        for column in range(7):
            header.setSectionResizeMode(
                column, QHeaderView.Stretch if column == 4 else QHeaderView.ResizeToContents)
        self.sched_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e; color: #ffffff;
                gridline-color: #333333; border: 1px solid #444444;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2b2b2b; color: #ffffff; padding: 4px;
                border: 1px solid #333333; font-weight: bold;
            }
        """)
        self.sched_table.setMinimumHeight(140)
        self.sched_table.setMaximumHeight(260)
        layout.addWidget(self.sched_table)

        layout.addWidget(self._build_scheduler_form())

        table_actions = QHBoxLayout()
        clear_btn = QPushButton("Clear Finished")
        clear_btn.setToolTip("Remove events that have run, failed or been missed.")
        clear_btn.clicked.connect(self._clear_finished_events)
        table_actions.addStretch()
        table_actions.addWidget(clear_btn)
        layout.addLayout(table_actions)
        return group

    def _build_scheduler_form(self):
        panel = QGroupBox("Schedule an event")
        grid = QGridLayout(panel)
        grid.setSpacing(8)

        grid.addWidget(QLabel("Target:"), 0, 0)
        self.sched_target_combo = QComboBox()
        self.sched_target_combo.addItem(sched.ALL_DEVICES)
        grid.addWidget(self.sched_target_combo, 0, 1)

        grid.addWidget(QLabel("Trigger:"), 0, 2)
        trigger = QWidget()
        trigger_layout = QHBoxLayout(trigger)
        trigger_layout.setContentsMargins(0, 0, 0, 0)
        trigger_layout.setSpacing(6)

        self.sched_kind_combo = QComboBox()
        self.sched_kind_combo.addItems(["At a time", "After a delay"])
        self.sched_kind_combo.currentIndexChanged.connect(self._on_sched_kind_changed)

        self.sched_day_combo = QComboBox()
        self.sched_day_combo.addItems(sched.DAY_CHOICES)
        self.sched_alarm_time = QTimeEdit()
        self.sched_alarm_time.setDisplayFormat("HH:mm:ss")
        self.sched_alarm_time.setTime(QTime(12, 0, 0))

        self.sched_delay_days = QSpinBox()
        self.sched_delay_days.setRange(0, 999)
        self.sched_delay_days.setSuffix(" d")
        self.sched_delay_time = QTimeEdit()
        self.sched_delay_time.setDisplayFormat("HH:mm:ss")
        self.sched_delay_time.setTime(QTime(1, 0, 0))

        for widget in (self.sched_kind_combo, self.sched_day_combo,
                       self.sched_alarm_time, self.sched_delay_days,
                       self.sched_delay_time):
            trigger_layout.addWidget(widget)
        trigger_layout.addStretch()
        grid.addWidget(trigger, 0, 3)

        grid.addWidget(QLabel("Repeat:"), 0, 4)
        self.sched_repeat_combo = QComboBox()
        self.sched_repeat_combo.addItems(sched.REPEATS)
        grid.addWidget(self.sched_repeat_combo, 0, 5)

        grid.addWidget(QLabel("Action:"), 1, 0)
        self.sched_action_combo = QComboBox()
        self.sched_action_combo.addItems(sched.ACTIONS)
        self.sched_action_combo.currentTextChanged.connect(self._on_sched_action_changed)
        grid.addWidget(self.sched_action_combo, 1, 1)

        grid.addWidget(QLabel("Parameters:"), 1, 2)
        params = QWidget()
        params_layout = QHBoxLayout(params)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(6)
        self.sched_mode_combo = QComboBox()
        self.sched_mode_combo.addItems(proto.MODE_LABELS)
        self.sched_mode_combo.currentTextChanged.connect(self._on_sched_action_changed)
        self.sched_params = _ModeParams()
        self.sched_lights_combo = QComboBox()
        self.sched_lights_combo.addItems(["Lights ON", "Lights OFF"])
        params_layout.addWidget(self.sched_mode_combo)
        params_layout.addWidget(self.sched_params)
        params_layout.addWidget(self.sched_lights_combo)
        params_layout.addStretch()
        grid.addWidget(params, 1, 3, 1, 3)

        # A live preview of the resolved fire time removes the guesswork that
        # made "Today at 09:00" silently mean tomorrow.
        self.sched_preview = QLabel("")
        self.sched_preview.setStyleSheet("color: #4fc3f7; font-size: 11px;")
        grid.addWidget(self.sched_preview, 2, 0, 1, 4)

        add_btn = QPushButton("Add Event")
        add_btn.setStyleSheet("font-weight: bold; min-height: 24px;")
        add_btn.clicked.connect(self._add_scheduled_event)
        grid.addWidget(add_btn, 2, 5, Qt.AlignRight)

        for widget in (self.sched_day_combo, self.sched_alarm_time,
                       self.sched_delay_days, self.sched_delay_time,
                       self.sched_repeat_combo):
            _connect_any_change(widget, self._update_sched_preview)

        self._on_sched_kind_changed(0)
        self._on_sched_action_changed()
        return panel

    def _on_sched_kind_changed(self, index):
        is_alarm = index == 0
        self.sched_day_combo.setVisible(is_alarm)
        self.sched_alarm_time.setVisible(is_alarm)
        self.sched_delay_days.setVisible(not is_alarm)
        self.sched_delay_time.setVisible(not is_alarm)
        self._update_sched_preview()

    def _on_sched_action_changed(self):
        action = self.sched_action_combo.currentText()
        is_mode = action == sched.ACTION_SET_MODE
        self.sched_mode_combo.setVisible(is_mode)
        self.sched_params.setVisible(is_mode)
        self.sched_lights_combo.setVisible(action == sched.ACTION_LIGHTS)
        if is_mode:
            _sync_mode_params(self.sched_mode_combo, self.sched_params)
        self._update_sched_preview()

    def _resolve_sched_time(self):
        """Fire time for the current form values, plus a delay in seconds."""
        if self.sched_kind_combo.currentIndex() == 0:
            when = sched.time_from_qtime(self.sched_alarm_time.time())
            return sched.resolve_alarm(self.sched_day_combo.currentText(), when), 0
        return sched.resolve_timer(
            self.sched_delay_days.value(),
            sched.time_from_qtime(self.sched_delay_time.time()))

    def _update_sched_preview(self):
        target_time, _ = self._resolve_sched_time()
        if target_time is None:
            self.sched_preview.setText("Set a delay greater than zero.")
            return
        delta = target_time - datetime.now()
        hours, remainder = divmod(max(int(delta.total_seconds()), 0), 3600)
        repeat = self.sched_repeat_combo.currentText()
        suffix = "" if repeat == sched.REPEAT_NONE else f", repeating {repeat.lower()}"
        self.sched_preview.setText(
            f"Will fire {target_time:%a %d %b %H:%M:%S} "
            f"(in {hours}h {remainder // 60:02d}m){suffix}")

    def _add_scheduled_event(self):
        target_time, offset = self._resolve_sched_time()
        if target_time is None:
            QMessageBox.warning(self, "Scheduler",
                                "Set a delay greater than 00:00:00.")
            return

        action = self.sched_action_combo.currentText()
        params = {}
        if action == sched.ACTION_SET_MODE:
            params = {
                "mode": self.sched_mode_combo.currentText(),
                "ratio": self.sched_params.ratio(),
                "timeout": self.sched_params.timeout(),
            }
        elif action == sched.ACTION_LIGHTS:
            params = {"lights": self.sched_lights_combo.currentText() == "Lights ON"}

        event = sched.ScheduledEvent(
            target=self.sched_target_combo.currentText(),
            action=action,
            params=params,
            target_time=target_time,
            kind="alarm" if self.sched_kind_combo.currentIndex() == 0 else "timer",
            repeat=self.sched_repeat_combo.currentText(),
            offset_seconds=offset,
        )
        self.scheduler.add(event)
        self._rebuild_scheduler_table()
        self._save_state()
        self._log_action("Scheduled event added", detail=(
            f"{event.target} · {event.describe_action()} · "
            f"{target_time:%Y-%m-%d %H:%M:%S} · repeat {event.repeat}"))

    def _rebuild_scheduler_table(self):
        """Rebuild rows from scratch. Called on structural change only.

        Countdowns are refreshed in place by :meth:`_tick_scheduler_display`;
        rebuilding every second (as the previous version did) destroyed the
        selection and re-created a QPushButton per row on every tick.
        """
        self.scheduler.sort()
        self.sched_table.setRowCount(0)
        self._sched_rows = {}

        for event in self.scheduler.events:
            row = self.sched_table.rowCount()
            self.sched_table.insertRow(row)
            self._sched_rows[event.id] = row

            toggle = QCheckBox()
            toggle.setChecked(event.enabled)
            toggle.setToolTip("Disable to keep the event without running it.")
            toggle.toggled.connect(
                lambda checked, eid=event.id: self._set_event_enabled(eid, checked))
            holder = QWidget()
            holder_layout = QHBoxLayout(holder)
            holder_layout.setContentsMargins(0, 0, 0, 0)
            holder_layout.setAlignment(Qt.AlignCenter)
            holder_layout.addWidget(toggle)
            self.sched_table.setCellWidget(row, 0, holder)

            self.sched_table.setItem(row, 1, QTableWidgetItem(event.describe_when()))
            self.sched_table.setItem(row, 2, QTableWidgetItem(event.describe_trigger()))
            self.sched_table.setItem(row, 3, QTableWidgetItem(event.target))
            self.sched_table.setItem(row, 4, QTableWidgetItem(event.describe_action()))
            self.sched_table.setItem(row, 5, _status_item(event))

            delete_btn = QPushButton("Delete")
            delete_btn.setStyleSheet(
                "background-color: #c0392b; color: white; "
                "max-height: 20px; font-size: 10px;")
            # Bound to the event id, not the row index: the table is re-sorted by
            # fire time, so a captured index would delete the wrong event.
            delete_btn.clicked.connect(
                lambda _, eid=event.id: self._delete_event(eid))
            self.sched_table.setCellWidget(row, 6, delete_btn)

    def _tick_scheduler_display(self):
        now = datetime.now()
        for event in self.scheduler.events:
            row = self._sched_rows.get(event.id)
            if row is None:
                continue
            countdown = self.sched_table.item(row, 2)
            if countdown is not None:
                countdown.setText(event.describe_trigger(now))
            self.sched_table.setItem(row, 5, _status_item(event))

    def _set_event_enabled(self, event_id, enabled):
        event = self.scheduler.get(event_id)
        if event is None:
            return
        event.enabled = enabled
        if enabled and event.status == sched.STATUS_MISSED:
            event.status = sched.STATUS_PENDING
        self._save_state()
        self._log_action("Scheduled event " + ("enabled" if enabled else "disabled"),
                         detail=event.describe_action())

    def _delete_event(self, event_id):
        event = self.scheduler.remove(event_id)
        if event is None:
            return
        self._rebuild_scheduler_table()
        self._save_state()
        self._log_action("Scheduled event deleted", detail=event.describe_action())

    def _clear_finished_events(self):
        removed = self.scheduler.clear_finished()
        if removed:
            self._rebuild_scheduler_table()
            self._save_state()
            self._log_action("Cleared finished events", detail=f"{len(removed)} removed")

    def _run_due_events(self):
        due = self.scheduler.due()
        if not due:
            return
        for event in due:
            targets = self._resolve_targets(event.target)
            if not targets:
                self.scheduler.complete(event, False, "target device not connected")
                self._log_action("Scheduled event failed", source="scheduler",
                                 detail=f"{event.target} not available",
                                 result="failed")
                continue
            ok = all(self._execute_action(device, event.action, event.params,
                                          source="scheduler")
                     for device in targets)
            self.scheduler.complete(
                event, ok, "sent" if ok else "one or more devices rejected the command")
            self._log_action(
                f"Scheduled event {'executed' if ok else 'failed'}",
                source="scheduler",
                detail=f"{event.target} · {event.describe_action()}",
                result="ok" if ok else "failed")
        self._rebuild_scheduler_table()
        self._save_state()

    def _resolve_targets(self, target_name):
        if target_name == sched.ALL_DEVICES:
            return [d for d in self.devices if d.is_connected]
        return [d for d in self.devices if d.name == target_name and d.is_connected]

    # ==================================================================
    # Device slots
    # ==================================================================

    def add_device_slot(self, slot_num=None, refresh=True):
        """Create a device slot. ``slot_num`` matches the on-board device ID."""
        if slot_num is None:
            used = {d.slot_num for d in self.devices}
            slot_num = next(n for n in range(1, 1000) if n not in used)

        box = QGroupBox(f"Device {slot_num}")
        grid = QGridLayout(box)

        name_edit = QLineEdit()
        name_edit.setPlaceholderText("Optional device name")
        port_combo = QComboBox()
        port_combo.setEditable(True)
        remove_btn = QPushButton("Remove")
        mode_combo = QComboBox()
        mode_combo.addItems(proto.MODE_LABELS)
        apply_btn = QPushButton("Apply Mode")
        params = _ModeParams()
        status_label = QLabel("Not connected")
        status_label.setStyleSheet("color: #999999; font-size: 11px;")

        feed_btn = QPushButton("Dispense")
        lights_btn = QPushButton("Lights: OFF")
        lights_btn.setCheckable(True)
        lights_btn.setStyleSheet("""
            QPushButton:checked { background-color: #f1c40f; color: black; }
            QPushButton { min-height: 22px; font-weight: bold; }
        """)
        new_trial_btn = QPushButton("New Trial")
        new_trial_btn.setStyleSheet(
            "font-weight: bold; background-color: #c0392b; color: white; "
            "min-height: 22px;")
        export_btn = QPushButton("Export Logs...")

        manual = _row(QLabel("Manual:"), feed_btn, lights_btn)
        data = _row(QLabel("Data:"), new_trial_btn, export_btn)

        grid.addWidget(QLabel("Port:"), 0, 0)
        grid.addWidget(port_combo, 0, 1, 1, 2)
        grid.addWidget(remove_btn, 0, 3, Qt.AlignRight)
        grid.addWidget(QLabel("Name:"), 1, 0)
        grid.addWidget(name_edit, 1, 1, 1, 3)
        grid.addWidget(QLabel("Mode:"), 2, 0)
        grid.addWidget(mode_combo, 2, 1, 1, 2)
        grid.addWidget(apply_btn, 2, 3)
        grid.addWidget(params, 3, 0, 1, 4)
        grid.addWidget(manual, 4, 0, 1, 4)
        grid.addWidget(data, 5, 0, 1, 4)
        grid.addWidget(status_label, 6, 0, 1, 4)
        grid.setColumnStretch(1, 1)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        view_container = QWidget()
        view_layout = QVBoxLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_title = QLabel(f"Device {slot_num}")
        view_title.setAlignment(Qt.AlignCenter)
        view_title.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")
        svg_view = FEDSvgView()
        svg_view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        view_layout.addWidget(view_title)
        view_layout.addWidget(svg_view)

        device = FedDevice(slot_num, {
            "box": box, "name_edit": name_edit, "port_combo": port_combo,
            "mode_combo": mode_combo, "params": params,
            "ratio_spin": params.ratio_spin, "timeout_spin": params.timeout_spin,
            "status_label": status_label, "lights_btn": lights_btn,
            "svg_view": svg_view, "view_container": view_container,
            "view_title": view_title, "remove_btn": remove_btn,
        })

        # The title tracks each keystroke, but the combo boxes only rebuild once
        # editing finishes — refreshing them per keystroke reset the user's plot
        # filter to "All Devices" mid-word.
        name_edit.textChanged.connect(lambda: device.view_title.setText(device.name))
        name_edit.editingFinished.connect(lambda: self._on_device_renamed(device))
        remove_btn.clicked.connect(lambda: self._remove_device(device))
        apply_btn.clicked.connect(lambda: self._apply_device_mode(device))
        mode_combo.currentTextChanged.connect(
            lambda: _sync_mode_params(mode_combo, params))
        feed_btn.clicked.connect(
            lambda: self._execute_action(device, sched.ACTION_DISPENSE, {}))
        lights_btn.clicked.connect(
            lambda: self._execute_action(device, sched.ACTION_LIGHTS,
                                         {"lights": lights_btn.isChecked()}))
        new_trial_btn.clicked.connect(lambda: self._new_trial(device))
        export_btn.clicked.connect(lambda: self._export_device(device))
        port_combo.activated.connect(lambda: self._on_port_selected(device))
        port_combo.lineEdit().editingFinished.connect(
            lambda: self._on_port_selected(device))

        _sync_mode_params(mode_combo, params)
        port_combo.addItem("Scanning...")
        port_combo.setEnabled(False)

        self.devices_flow.addWidget(box)
        self.device_view_flow.addWidget(view_container)
        self.devices.append(device)
        self._reorder_devices()
        self._refresh_device_combos()
        if refresh:
            self.refresh_ports()
        return device

    def _on_device_renamed(self, device):
        if device.last_known_name == device.name:
            return
        # Scheduled events reference devices by name, so a rename has to carry
        # them across or they silently stop matching a target.
        self.scheduler.rename_target(device.last_known_name, device.name)
        device.last_known_name = device.name
        device.view_title.setText(device.name)
        if device.link is not None:
            device.link.owner = device.name
        self._refresh_device_combos()
        self._rebuild_scheduler_table()

    def _remove_device(self, device):
        if len(self.devices) <= 1:
            QMessageBox.warning(self, "Cannot remove device",
                                "At least one device slot must remain.")
            return
        if device.is_tracking and not self._confirm(
                f"Remove {device.name}? It is currently connected and tracking."):
            return

        if device.port:
            self.removed_ports.add(device.port)
        self._disconnect_device(device)
        self.devices_flow.removeWidget(device.box)
        device.box.deleteLater()
        self.device_view_flow.removeWidget(device.view_container)
        device.view_container.deleteLater()
        self.devices.remove(device)

        self._reorder_devices()
        self._refresh_device_combos()
        self._mark_plot_dirty()
        self._log_action("Device removed", device=device.name)

    def _reorder_devices(self):
        self.devices.sort(key=lambda d: d.slot_num)
        for device in self.devices:
            device.box.setTitle(f"Device {device.slot_num}")
            device.view_title.setText(device.name)
            self.devices_flow.removeWidget(device.box)
            self.device_view_flow.removeWidget(device.view_container)
        for device in self.devices:
            self.devices_flow.addWidget(device.box)
            self.device_view_flow.addWidget(device.view_container)
        for device in self.devices:
            device.remove_btn.setEnabled(len(self.devices) > 1)

    def _refresh_device_combos(self):
        """Keep the plot filter and scheduler target lists in step with names."""
        for combo, first in ((self.plot_filter_combo, "All Devices"),
                             (self.sched_target_combo, sched.ALL_DEVICES)):
            previous = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(first)
            for device in self.devices:
                combo.addItem(device.name)
            index = combo.findText(previous)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    # ==================================================================
    # Port scanning and connection
    # ==================================================================

    def refresh_ports(self):
        if self._scanner is not None and self._scanner.isRunning():
            return
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("Scanning...")
        self._set_status("Scanning ports for FED3 devices...")

        for device in self.devices:
            if device.port:
                device.saved_port = device.port

        self._scanner = PortScannerWorker()
        self._scanner.finished_scan.connect(self.scan_finished.emit)
        self._scanner.finished.connect(self._scanner.deleteLater)
        self._scanner.start()

    def _on_scan_finished(self, results, all_ports):
        self._scanner = None
        self.refresh_btn.setEnabled(True)
        self.refresh_btn.setText("Refresh Ports")
        self._set_status("Ready")

        active = []
        for port, status, device_id, firmware in results:
            if status == "FED3 Active":
                active.append(port)
                self._port_info[port] = {"id": device_id, "firmware": firmware}

        self.log.append_log(f"[Scan] Active FED3 ports: {active or 'none'}")
        self._populate_port_combos(all_ports)
        self._assign_discovered_ports(active)

        # Leave one empty slot so the panel is never blank and a port can be
        # assigned by hand when auto-discovery finds nothing.
        if not self.devices:
            self.add_device_slot(refresh=False)
            self._populate_port_combos(all_ports)

        self._connect_assigned_ports()

    def _populate_port_combos(self, all_ports):
        self._all_ports = list(all_ports)
        for device in self.devices:
            combo = device.port_combo
            combo.blockSignals(True)
            current = device.saved_port or device.port
            combo.clear()
            for port in all_ports:
                combo.addItem(port, port)
            if current and combo.findText(current) < 0:
                combo.addItem(current, current)
            combo.setCurrentIndex(combo.findText(current) if current else -1)
            combo.setEnabled(True)
            combo.blockSignals(False)

    def _assign_discovered_ports(self, active_ports):
        """Give each discovered device a slot, preferring its on-board ID.

        A FED3 reporting ID 3 lands in the "Device 3" slot, so slot numbers keep
        matching the physical labels across replugs and reboots.
        """
        assigned = {d.port for d in self.devices if d.port}
        for port in active_ports:
            if port in assigned or port in self.removed_ports:
                continue

            info = self._port_info.get(port, {})
            device = None
            if info.get("id") and str(info["id"]).isdigit():
                slot = int(info["id"])
                device = next((d for d in self.devices if d.slot_num == slot), None)
                if device is None:
                    device = self.add_device_slot(slot_num=slot, refresh=False)
                    self._populate_port_combos_for(device, port)
                elif device.port:
                    device = None       # slot taken by another port; fall through

            if device is None:
                device = next((d for d in self.devices if not d.port), None)
                if device is None:
                    device = self.add_device_slot(refresh=False)
                    self._populate_port_combos_for(device, port)

            self._set_device_port(device, port, info)
            assigned.add(port)

    def _populate_port_combos_for(self, device, port):
        """Fill a slot created mid-scan with the full port list.

        Populating it with only the discovered port would leave the user unable
        to reassign the slot until the next refresh.
        """
        combo = device.port_combo
        combo.blockSignals(True)
        combo.clear()
        for known in self._all_ports or [port]:
            combo.addItem(known, known)
        if combo.findData(port) < 0:
            combo.addItem(port, port)
        combo.setEnabled(True)
        combo.blockSignals(False)

    def _set_device_port(self, device, port, info=None):
        combo = device.port_combo
        combo.blockSignals(True)
        index = combo.findData(port)
        if index < 0:
            combo.addItem(port, port)
            index = combo.findData(port)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

        device.saved_port = port
        info = info or self._port_info.get(port, {})
        device.device_id = info.get("id")
        device.firmware = info.get("firmware")
        if device.device_id and not device.name_edit.text().strip():
            device.name_edit.setText(f"FED {device.device_id}")
            self._on_device_renamed(device)     # setText alone skips editingFinished

    def _on_port_selected(self, device):
        """User picked a port by hand."""
        port = device.port
        if not port:
            self._disconnect_device(device)
            return

        clash = next((d for d in self.devices if d is not device and d.port == port), None)
        if clash is not None:
            QMessageBox.warning(
                self, "Port already assigned",
                f"{port} is already assigned to {clash.name}.")
            if device.saved_port:
                self._set_device_port(device, device.saved_port)
            else:
                device.port_combo.setCurrentIndex(-1)
            return

        info = self._port_info.get(port, {})
        onboard = info.get("id")
        if onboard and str(onboard) != str(device.slot_num) and not self._confirm(
                f"The device on {port} reports on-board ID {onboard}, but this is "
                f"the Device {device.slot_num} slot.\n\nAssign it anyway?"):
            device.port_combo.setCurrentIndex(-1)
            return

        self.removed_ports.discard(port)
        device.saved_port = port
        device.device_id = onboard
        device.firmware = info.get("firmware")
        self._connect_device(device)

    def _connect_assigned_ports(self):
        for device in self.devices:
            if device.port and not device.is_connected:
                self._connect_device(device)

    def _connect_device(self, device):
        port = device.port
        if not port:
            return
        if device.link is not None and device.link.port == port and device.link.is_live():
            return

        self._disconnect_device(device)
        device.status_label.setText(f"Connecting to {port}...")

        link = Fed3Link(port, owner=device.name)
        link.line_received.connect(lambda line, d=device: self._on_line(d, line))
        link.connected.connect(lambda d=device: self._on_link_connected(d))
        link.disconnected.connect(
            lambda reason, d=device: self._on_link_disconnected(d, reason))
        link.command_sent.connect(
            lambda cmd, ok, detail, d=device: self._on_command_sent(d, cmd, ok, detail))

        device.link = link
        device.transfer = Fed3Transfer(link.send, parent=self)
        link.start()

    def _on_link_connected(self, device):
        device.has_connected = True
        device.connect_attempts = 0
        device.is_tracking = True
        device.svg_view.is_tracking = True
        device.svg_view.is_stale = False
        device.box.setStyleSheet("")
        device.box.setToolTip("")
        if device.tracking_start_time is None:
            device.tracking_start_time = datetime.now()
        device.status_label.setText(f"Connected on {device.port}")
        device.status_label.setStyleSheet("color: #4caf50; font-size: 11px;")

        self._log_action("Connected", device=device.name,
                         detail=device.port, source="system")
        self._sync_device(device)
        device.link.send(proto.CMD_STATUS)
        if self.session is not None:
            self._attach_session_to_device(device)
        self._mark_plot_dirty()

    def _on_link_disconnected(self, device, reason):
        was_tracking = device.is_tracking
        device.is_tracking = False
        device.svg_view.is_tracking = False
        device.svg_view.is_stale = True
        device.svg_view.update()
        if device.transfer is not None:
            device.transfer.cancel("connection lost")
        device.link = None
        device.status_label.setText(f"Disconnected — {reason}")
        device.status_label.setStyleSheet("color: #e57373; font-size: 11px;")

        if was_tracking and device.has_connected:
            device.box.setStyleSheet("QGroupBox { border: 2px solid #ff4d4d; }")
            device.box.setToolTip(f"Disconnected: {reason}")
            self.log.append_log(f"[{device.name}] Disconnected: {reason}", False)
            self._set_status(f"{device.name} disconnected: {reason}")
        self._log_action("Disconnected", device=device.name, detail=reason,
                         source="system", result="warn")
        self._mark_plot_dirty()

    def _disconnect_device(self, device):
        if device.mirror is not None:
            device.mirror.stop()
            device.mirror = None
        if device.transfer is not None:
            device.transfer.cancel("disconnecting")
            device.transfer = None
        if device.link is not None:
            link, device.link = device.link, None
            link.stop()
        device.is_tracking = False
        device.svg_view.is_tracking = False

    def _check_reconnects(self):
        """Reopen ports that came back, without ever touching a live one."""
        if self._scanner is not None and self._scanner.isRunning():
            return
        try:
            from serial.tools import list_ports
            available = {p.device for p in list_ports.comports()}
        except Exception:
            return

        for device in self.devices:
            if device.is_connected or not device.has_connected:
                continue
            if device.connect_attempts >= MAX_RECONNECT_ATTEMPTS:
                continue
            port = device.port
            if port and port in available:
                device.connect_attempts += 1
                self.log.append_log(
                    f"[{device.name}] Reconnecting to {port} "
                    f"(attempt {device.connect_attempts})")
                self._connect_device(device)

    # ==================================================================
    # Incoming serial lines
    # ==================================================================

    def _on_line(self, device, line):
        # Transfers claim their own lines; a file's contents must never be
        # mistaken for device chatter.
        if device.transfer is not None and device.transfer.handle_line(line):
            return

        stripped = line.strip()
        if not stripped:
            return

        is_error = proto.is_error(stripped)
        self.log.append_log(f"[{device.name}] {stripped}", not is_error)
        if is_error:
            self._log_action("Device error", device=device.name,
                             detail=stripped, source="device", result="error")
            return

        status = proto.parse_status(stripped)
        if status is not None:
            self._apply_status(device, status)
            return

        if stripped.startswith("SYNCED,"):
            self._record_clock_sync(device, stripped.split(",", 1)[1])
            return

        event = proto.parse_event(stripped)
        if event is not None:
            self._on_device_event(device, event)

    def _apply_status(self, device, status):
        device.firmware = status.get("fw") or device.firmware
        device.device_id = status.get("id") or device.device_id
        counts = {}
        for key, field in (("l", "left"), ("r", "right"), ("p", "pellet")):
            if key in status:
                try:
                    counts[field] = int(status[key])
                except ValueError:
                    pass
        if counts:
            device.apply_counts(counts)
            device.svg_view.set_counts(device.stats)
        if status.get("session"):
            device.status_label.setText(
                f"Connected on {device.port} — {status['session']} "
                f"· {status.get('file', 'no file')}")

    def _on_device_event(self, device, event):
        host_ts = session_mod.host_now()
        device.apply_counts(event.counts)

        key = {proto.EVENT_LEFT: "left", proto.EVENT_RIGHT: "right",
               proto.EVENT_PELLET: "pellet"}[event.kind]
        device.svg_view.flash(key)
        device.svg_view.set_counts(device.stats)

        if event.kind == proto.EVENT_PELLET:
            device.events.append(datetime.fromtimestamp(host_ts))
            self._mark_plot_dirty()

        if device.event_log is not None:
            device.event_log.append(event, host_ts)
        if device.mirror is not None:
            device.mirror.note_event()

    def _record_clock_sync(self, device, iso_text):
        device_time = proto._parse_iso(iso_text)
        device.last_sync_time = datetime.now()
        device.last_device_time = device_time
        if self.session is not None:
            self.session.log_clock_sync(device.name, device_time)
        drift = ""
        if device_time is not None:
            offset = (device_time - device.last_sync_time).total_seconds()
            drift = f" (device was {offset:+.0f}s off)"
        self._log_action("Clock synced", device=device.name,
                         detail=f"{iso_text}{drift}", source="system")

    def _on_command_sent(self, device, command, ok, detail):
        if not ok:
            self.log.append_log(f"[{device.name}] {command}: {detail}", False)
            self._log_action("Command failed", device=device.name,
                             detail=f"{command}: {detail}", result="failed")

    # ==================================================================
    # Commands
    # ==================================================================

    def _send(self, device, command, description, source="user"):
        """Queue a command on a device's link and record it."""
        if not device.is_connected:
            self.log.append_log(f"[{device.name}] not connected; {command} skipped",
                                False)
            self._log_action(description, device=device.name, detail=command,
                             source=source, result="not connected")
            return False
        device.link.send(command)
        self._log_action(description, device=device.name, detail=command,
                         source=source)
        return True

    def _execute_action(self, device, action, params, source="user"):
        """Run one scheduler-style action against a device."""
        if action == sched.ACTION_DISPENSE:
            return self._send(device, proto.CMD_FEED, "Dispense pellet", source)
        if action == sched.ACTION_LIGHTS:
            on = bool(params.get("lights"))
            device.lights_btn.blockSignals(True)
            device.lights_btn.setChecked(on)
            device.lights_btn.setText(f"Lights: {'ON' if on else 'OFF'}")
            device.lights_btn.blockSignals(False)
            return self._send(device,
                              proto.CMD_LIGHTS_ON if on else proto.CMD_LIGHTS_OFF,
                              f"Lights {'on' if on else 'off'}", source)
        if action == sched.ACTION_NEW_TRIAL:
            return self._start_new_trial(device, source)
        if action == sched.ACTION_SET_MODE:
            command, description = proto.mode_command(
                params.get("mode", ""), params.get("ratio", 1),
                params.get("timeout", 30))
            if command is None:
                return False
            device.mode_combo.blockSignals(True)
            device.mode_combo.setCurrentText(params.get("mode", ""))
            device.mode_combo.blockSignals(False)
            device.params.set_values(params.get("ratio", 1), params.get("timeout", 30))
            _sync_mode_params(device.mode_combo, device.params)
            return self._send(device, command, f"Set mode: {description}", source)
        return False

    def _apply_device_mode(self, device):
        mode = device.mode_combo.currentText()
        if device.is_tracking and not self._confirm(
                f"Apply {mode} to {device.name}? This changes the protocol mid-session."):
            return
        self._execute_action(device, sched.ACTION_SET_MODE, {
            "mode": mode,
            "ratio": device.params.ratio(),
            "timeout": device.params.timeout(),
        })

    def _apply_global_mode(self):
        mode = self.global_mode_combo.currentText()
        connected = [d for d in self.devices if d.is_connected]
        if not connected:
            QMessageBox.warning(self, "Apply mode", "No connected devices.")
            return
        if not self._confirm(
                f"Apply {mode} to all {len(connected)} connected device(s)?"):
            return
        params = {"mode": mode, "ratio": self.global_params.ratio(),
                  "timeout": self.global_params.timeout()}
        for device in connected:
            self._execute_action(device, sched.ACTION_SET_MODE, params)

    def _dispense_all(self):
        for device in self.devices:
            if device.is_connected:
                self._execute_action(device, sched.ACTION_DISPENSE, {})

    def _toggle_global_lights(self):
        on = self.global_lights_btn.isChecked()
        self.global_lights_btn.setText(f"Lights: {'ON' if on else 'OFF'}")
        for device in self.devices:
            if device.is_connected:
                self._execute_action(device, sched.ACTION_LIGHTS, {"lights": on})

    def _on_raw_command(self, command):
        connected = [d for d in self.devices if d.is_connected]
        if not connected:
            self.log.append_log("No connected devices.", False)
            return
        if not self._confirm(f"Send '{command}' to all {len(connected)} device(s)?"):
            return
        for device in connected:
            self._send(device, command, "Raw command")

    def _new_trial(self, device):
        if not self._confirm(
                f"Start a new trial on {device.name}?\n\n"
                "This zeroes the device counters and starts a new CSV on its SD "
                "card. Data already mirrored to the session folder is kept."):
            return
        self._start_new_trial(device)

    def _start_new_trial(self, device, source="user"):
        ok = self._send(device, proto.CMD_NEW_TRIAL, "Start new trial", source)
        if ok:
            device.reset_counters()
            device.svg_view.set_counts(device.stats)
            self._mark_plot_dirty()
        return ok

    def _new_trial_all(self):
        connected = [d for d in self.devices if d.is_connected]
        if not connected:
            QMessageBox.warning(self, "New trial", "No connected devices.")
            return
        if self._confirm(f"Start a new trial on all {len(connected)} device(s)?"):
            for device in connected:
                self._start_new_trial(device)

    # --- clock sync --------------------------------------------------------

    def _sync_device(self, device):
        return self._send(device, proto.cmd_sync(), "Sync clock", source="system")

    def _sync_all(self):
        for device in self.devices:
            if device.is_connected:
                self._sync_device(device)

    def _sync_interval_ms(self):
        multiplier = {"Minutes": 60, "Hours": 3600, "Days": 86400}[
            self.sync_unit_combo.currentText()]
        return max(60, self.sync_interval_spin.value() * multiplier) * 1000

    def _restart_sync_timer(self):
        if self.auto_sync_btn.isChecked():
            self.sync_timer.start(self._sync_interval_ms())

    def _on_auto_sync_toggled(self, checked):
        self.auto_sync_btn.setText(f"Auto Sync: {'ON' if checked else 'OFF'}")
        if checked:
            self.sync_timer.start(self._sync_interval_ms())
            self._sync_all()
        else:
            self.sync_timer.stop()
        self._log_action(f"Auto sync {'enabled' if checked else 'disabled'}")

    # ==================================================================
    # Recording session
    # ==================================================================

    def _on_record_toggled(self, checked):
        if checked:
            if not self._start_session():
                self.record_btn.blockSignals(True)
                self.record_btn.setChecked(False)
                self.record_btn.blockSignals(False)
        else:
            self._stop_session()

    def _start_session(self, resume_root=None):
        try:
            self.session = session_mod.RecordingSession(self.sessions_dir,
                                                        root=resume_root)
        except OSError as exc:
            QMessageBox.critical(self, "Recording",
                                 f"Could not create the session folder:\n{exc}")
            return False

        self.logger.attach(self.session)
        self.session.write_config({
            "started_at": session_mod.host_iso(),
            "resumed": self.session.resumed,
            "fnt_module": "fed3",
            "time_base": "host wall clock (Unix epoch seconds)",
            "sync_interval_ms": self._sync_interval_ms(),
            "devices": [d.to_state() for d in self.devices],
        })

        for device in self.devices:
            self._attach_session_to_device(device)

        if self.webcam is not None:
            self._arm_camera_recording()

        self.record_btn.setText("Stop Recording")
        self.session_label.setText(f"Recording to {self.session.name}")
        self.session_label.setStyleSheet("color: #4caf50;")
        self.open_folder_btn.setEnabled(True)
        self._log_action("Recording started" if not resume_root else "Recording resumed",
                         detail=self.session.root, source="system")
        self._save_state()
        return True

    def _attach_session_to_device(self, device):
        """Give a device its event log and SD mirror for the active session."""
        if self.session is None:
            return
        device.event_log = session_mod.DeviceEventLog(
            self.session.device_events_path(device.name), device.name)
        if device.transfer is not None and device.mirror is None:
            device.mirror = DeviceMirror(
                device.transfer, self.session.device_mirror_dir(device.name),
                parent=self)
            device.mirror.progress.connect(self.mirror_status.setText)
            device.mirror.failed.connect(
                lambda message, d=device: self._on_mirror_failed(d, message))
            device.mirror.updated.connect(
                lambda name, added, total, d=device:
                self._log_action("Mirrored SD data", device=d.name,
                                 detail=f"{name}: +{added} bytes ({total} total)",
                                 source="system"))
            device.mirror.sync_now(force=True)

    def _on_mirror_failed(self, device, message):
        self.mirror_status.setText(message)
        self.log.append_log(f"[{device.name}] {message}", False)
        self._log_action("Mirror pull failed", device=device.name,
                         detail=message, source="system", result="failed")

    def _stop_session(self):
        if self.session is None:
            return
        frames = 0
        if self.webcam is not None:
            frames = self.webcam.stop_recording()

        for device in self.devices:
            if device.mirror is not None:
                device.mirror.sync_now(force=True)   # last pull before closing
                device.mirror.stop()
                device.mirror = None
            device.event_log = None

        self._log_action("Recording stopped",
                         detail=f"{frames} video frames" if frames else "",
                         source="system")
        self.session.mark_closed()
        self.logger.detach()

        self.record_btn.setText("Start Recording")
        self.session_label.setText(f"Last session: {self.session.name}")
        self.session_label.setStyleSheet("color: #999999;")
        self.session = None

    def _save_state(self):
        """Persist everything needed to resume after a crash."""
        if self.session is None:
            return
        self.session.write_state({
            "status": session_mod.STATUS_RUNNING,
            "devices": [d.to_state() for d in self.devices],
            "scheduled_events": self.scheduler.to_list(),
            "camera_index": self.camera_combo.currentData(),
            "camera_recording": self.webcam is not None and self.webcam.is_recording(),
            "auto_sync": self.auto_sync_btn.isChecked(),
        })

    def _offer_resume(self):
        sessions = session_mod.find_resumable_sessions(self.sessions_dir)
        if not sessions:
            return
        dialog = ResumeSessionDialog(sessions, self)
        if dialog.exec_() != QDialog.Accepted or not dialog.choice:
            # Declining leaves the folders on disk but stops them being offered
            # again on every launch.
            for root, _state in sessions:
                session_mod.RecordingSession(self.sessions_dir, root=root).mark_closed()
            self._log_action("Declined to resume interrupted sessions",
                             detail=f"{len(sessions)} found", source="system")
            return
        self._resume_session(dialog.choice)

    def _resume_session(self, root):
        state = {}
        try:
            with open(os.path.join(root, "session_state.json"), encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, ValueError):
            pass

        for entry in state.get("devices") or []:
            device = self.add_device_slot(slot_num=entry.get("slot_num"), refresh=False)
            device.name_edit.setText(entry.get("name", ""))
            device.last_known_name = device.name
            device.view_title.setText(device.name)
            device.saved_port = entry.get("port", "")
            device.device_id = entry.get("device_id")
            device.firmware = entry.get("firmware")
            device.mode_combo.setCurrentText(entry.get("mode", proto.MODE_LABELS[0]))
            device.params.set_values(entry.get("ratio", 1), entry.get("timeout", 30))
            device.stats = dict(entry.get("stats") or device.stats)
            device.svg_view.set_counts(device.stats)
            started = entry.get("tracking_start_time")
            if started:
                try:
                    device.tracking_start_time = datetime.fromisoformat(started)
                except ValueError:
                    pass

        self.scheduler.load(state.get("scheduled_events"))
        # Anything that fell due while FNT was down is marked missed rather than
        # replayed; see Scheduler.due().
        self.scheduler.due()
        self._rebuild_scheduler_table()

        if state.get("auto_sync") is False:
            self.auto_sync_btn.setChecked(False)

        if not self._start_session(resume_root=root):
            return

        # Rebuild the plot from the events already on disk so the graph is
        # continuous across the interruption.
        for device in self.devices:
            log = session_mod.DeviceEventLog(
                self.session.device_events_path(device.name), device.name)
            device.events = log.read_event_times(proto.EVENT_PELLET)
            if device.events and device.tracking_start_time is None:
                device.tracking_start_time = device.events[0]

        self.record_btn.blockSignals(True)
        self.record_btn.setChecked(True)
        self.record_btn.blockSignals(False)
        self.record_btn.setText("Stop Recording")
        self._refresh_device_combos()
        self._mark_plot_dirty()
        self.log.append_log(f"Resumed interrupted session: {os.path.basename(root)}")

    def _choose_sessions_dir(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose where sessions are saved", self.sessions_dir)
        if chosen:
            self.sessions_dir = chosen
            self._log_action("Session location changed", detail=chosen)

    def _open_session_folder(self):
        if self.session is None:
            return
        path = self.session.root
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)      # noqa: S606 - platform API
        else:
            subprocess.Popen(["xdg-open", path])

    # ==================================================================
    # Camera
    # ==================================================================

    def _populate_cameras(self):
        try:
            cameras = list_cameras()
        except Exception as exc:  # noqa: BLE001
            self.log.append_log(f"Camera scan failed: {exc}", False)
            cameras = []
        self.camera_combo.clear()
        for index in cameras:
            self.camera_combo.addItem(f"Camera {index}", index)
        if not cameras:
            self.camera_combo.addItem("No camera detected", None)

    def _toggle_camera(self):
        if self.webcam is not None:
            self._stop_camera()
            return
        index = self.camera_combo.currentData()
        if index is None:
            QMessageBox.warning(self, "Camera", "No camera was detected.")
            return

        self.webcam = WebcamRecorder(camera_index=index,
                                     label=self.camera_combo.currentText())
        self.webcam.frame_ready.connect(self._on_camera_frame)
        self.webcam.opened.connect(self._on_camera_opened)
        self.webcam.error.connect(self._on_camera_error)
        self.webcam.start()

        self.camera_btn.setText("Stop Camera")
        self.camera_combo.setEnabled(False)
        self.camera_view.setVisible(True)
        self._log_action("Camera started", detail=self.camera_combo.currentText())

        if self.session is not None:
            self._arm_camera_recording()

    def _arm_camera_recording(self):
        label = self.webcam.label
        self.webcam.start_recording(self.session.video_path(label),
                                    self.session.video_frames_path(label))
        self._log_action("Camera recording armed", detail=label, source="system")

    def _stop_camera(self):
        if self.webcam is None:
            return
        frames = self.webcam.stop_recording()
        self.webcam.stop()
        if not self.webcam.wait(3000):      # a blocked camera read needs forcing
            self.webcam.terminate()
            self.webcam.wait(1000)
        self.webcam = None

        self.camera_btn.setText("Start Camera")
        self.camera_combo.setEnabled(True)
        self.camera_view.setPixmap(QPixmap())
        self.camera_view.setText("Camera off")
        self.camera_view.setVisible(False)
        self.camera_status.setText("Camera off")
        self._log_action("Camera stopped", detail=f"{frames} frames recorded")

    def _on_camera_frame(self, image):
        self.camera_view.setPixmap(QPixmap.fromImage(image).scaled(
            self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_camera_opened(self, width, height, fps):
        self.camera_status.setText(f"Camera on: {width}x{height} @ {fps:.0f} fps")

    def _on_camera_error(self, message):
        self._stop_camera()
        QMessageBox.critical(self, "Camera error", message)

    # ==================================================================
    # Export
    # ==================================================================

    def _force_mirror_sync(self):
        pulled = 0
        for device in self.devices:
            if device.mirror is not None:
                device.mirror.sync_now(force=True)
                pulled += 1
        if pulled:
            self._log_action("Manual mirror pull", detail=f"{pulled} device(s)")
        else:
            QMessageBox.information(
                self, "Pull data",
                "Mirroring runs while a session is recording. Start a recording "
                "to keep a continuous copy of the SD cards on this computer.")

    def _export_device(self, device):
        self._export([device])

    def _export_all(self):
        connected = [d for d in self.devices if d.is_connected]
        if not connected:
            QMessageBox.warning(self, "Export", "No connected devices.")
            return
        self._export(connected)

    def _export(self, devices):
        """List files on each device, then copy the chosen ones to a folder."""
        available = [d for d in devices if d.is_connected and d.transfer is not None]
        if not available:
            QMessageBox.warning(self, "Export",
                                "The device must be connected to list its SD card.")
            return

        progress = QProgressDialog("Requesting file lists...", "Cancel", 0,
                                   len(available), self)
        progress.setWindowTitle("Reading SD cards")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        listings = {}
        state = {"cancelled": False, "index": 0}

        def cancel():
            state["cancelled"] = True
            for device in available:
                device.transfer.cancel("export cancelled")

        progress.canceled.connect(cancel)

        def query(index):
            if state["cancelled"]:
                return
            if index >= len(available):
                progress.close()
                self._choose_and_download(listings)
                return
            device = available[index]
            progress.setLabelText(f"Listing files on {device.name}...")
            progress.setValue(index)

            def received(ok, data, _offset=None, d=device):
                if state["cancelled"]:
                    return
                if ok:
                    listings[d] = data
                else:
                    self.log.append_log(f"[{d.name}] file list failed: {data}", False)
                query(index + 1)

            device.transfer.list_files(received)

        query(0)

    def _choose_and_download(self, listings):
        listings = {device: files for device, files in listings.items() if files}
        if not listings:
            QMessageBox.information(self, "Export",
                                    "No CSV logs were found on the SD cards.")
            return

        dialog = FileSelectorDialog(listings, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        selected = dialog.selected()
        if not selected:
            return

        destination = QFileDialog.getExistingDirectory(
            self, "Choose where to save the logs",
            self.session.root if self.session else self.sessions_dir)
        if not destination:
            return

        progress = QProgressDialog("", "Cancel", 0, len(selected), self)
        progress.setWindowTitle("Downloading")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        state = {"cancelled": False, "written": 0}

        def cancel():
            state["cancelled"] = True
            for device, _ in selected:
                device.transfer.cancel("download cancelled")

        progress.canceled.connect(cancel)

        def download(index):
            if state["cancelled"]:
                return
            if index >= len(selected):
                progress.close()
                QMessageBox.information(
                    self, "Export complete",
                    f"Saved {state['written']} file(s) to:\n{destination}")
                self._log_action("Exported SD logs",
                                 detail=f"{state['written']} file(s) -> {destination}")
                return

            device, filename = selected[index]
            progress.setLabelText(
                f"{filename} from {device.name} ({index + 1}/{len(selected)})")
            progress.setValue(index)

            def finished(ok, payload, _offset=0, d=device, name=filename):
                if state["cancelled"]:
                    return
                if not ok:
                    progress.close()
                    QMessageBox.critical(
                        self, "Download failed",
                        f"{name} from {d.name}:\n{payload}")
                    return
                path = os.path.join(destination, f"{_safe(d.name)}_{name}")
                try:
                    with open(path, "wb") as f:
                        f.write(payload)
                except OSError as exc:
                    progress.close()
                    QMessageBox.critical(self, "Write failed",
                                         f"Could not write {path}:\n{exc}")
                    return
                state["written"] += 1
                self.log.append_log(f"[{d.name}] exported {name}")
                download(index + 1)

            # Offset 0: an explicit export is a full copy, independent of the
            # incremental mirror's progress.
            device.transfer.download(filename, 0, finished)

        download(0)

    # ==================================================================
    # Timers, plot, layout, teardown
    # ==================================================================

    def _start_timers(self):
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._on_ui_tick)
        self.ui_timer.start(UI_TICK_MS)

        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.timeout.connect(self._check_reconnects)
        self.reconnect_timer.start(RECONNECT_TICK_MS)

        self.state_timer = QTimer(self)
        self.state_timer.timeout.connect(self._save_state)
        self.state_timer.start(STATE_SAVE_MS)

        self.sync_timer = QTimer(self)
        self.sync_timer.timeout.connect(self._sync_all)
        self.sync_timer.start(self._sync_interval_ms())

    def _on_ui_tick(self):
        self._run_due_events()
        self._tick_scheduler_display()
        self._update_sched_preview()
        # Redraw on a timer rather than per event: a pellet burst would otherwise
        # queue a full matplotlib redraw per pellet.
        if self._plot_dirty or any(d.is_tracking for d in self.devices):
            self._redraw_plot()

    def _mark_plot_dirty(self):
        self._plot_dirty = True

    def _redraw_plot(self):
        self._plot_dirty = False
        index = self.plot_filter_combo.currentIndex()
        devices = (self.devices if index <= 0
                   else [d for d in self.devices if d.name == self.plot_filter_combo.currentText()])
        self.plot_manager.update([
            PlotSeries(d.name, d.events, d.is_tracking, d.tracking_start_time)
            for d in devices
        ])

    def _on_plot_window_changed(self):
        self.plot_manager.set_window(self.plot_window_combo.currentData())
        self._mark_plot_dirty()

    def _on_dark_cycle_changed(self):
        self.plot_manager.set_dark_cycle(
            self.lights_off_spin.value(), self.lights_on_spin.value(),
            self.dark_cycle_check.isChecked())
        self._mark_plot_dirty()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _apply_responsive_layout(self):
        """Two columns when wide, a single stack when narrow."""
        is_narrow = self.width() < NARROW_WIDTH
        if is_narrow == self._layout_is_narrow:
            return
        self._layout_is_narrow = is_narrow

        for widget in (self.control_group, self.scheduler_group, self.devices_group,
                       self.left_column, self.right_column):
            self.columns_layout.removeWidget(widget)
        for layout, widget in ((self.left_column_layout, self.control_group),
                               (self.left_column_layout, self.scheduler_group),
                               (self.right_column_layout, self.devices_group)):
            layout.removeWidget(widget)

        if is_narrow:
            self.left_column.hide()
            self.right_column.hide()
            for row, widget in enumerate(
                    (self.control_group, self.scheduler_group, self.devices_group)):
                self.columns_layout.addWidget(widget, row, 0)
                widget.show()
            self.columns_layout.setColumnStretch(0, 1)
            self.columns_layout.setColumnStretch(1, 0)
        else:
            self.left_column_layout.addWidget(self.control_group)
            self.left_column_layout.addWidget(self.scheduler_group)
            self.right_column_layout.addWidget(self.devices_group)
            self.columns_layout.addWidget(self.left_column, 0, 0)
            self.columns_layout.addWidget(self.right_column, 0, 1)
            for widget in (self.left_column, self.right_column, self.control_group,
                           self.scheduler_group, self.devices_group):
                widget.show()
            self.columns_layout.setColumnStretch(0, 1)
            self.columns_layout.setColumnStretch(1, 1)

    def cleanup(self):
        """Close every connection, timer and file before the window goes away."""
        self.log.append_log("Shutting down FED3 tab...")
        for name in ("ui_timer", "reconnect_timer", "state_timer", "sync_timer"):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.stop()

        if self.webcam is not None:
            self._stop_camera()
        if self.session is not None:
            self._stop_session()

        for device in self.devices:
            self._disconnect_device(device)

        if self._scanner is not None and self._scanner.isRunning():
            try:
                self._scanner.finished_scan.disconnect()
            except TypeError:
                pass
            self._scanner.terminate()
            self._scanner.wait(2000)
        self._scanner = None
        self.logger.detach()

    # --- small helpers ----------------------------------------------------

    def _confirm(self, message):
        return QMessageBox.question(
            self, "Confirm", message, QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No) == QMessageBox.Yes

    def _log_action(self, action, device="", detail="", source="user", result="ok"):
        line = self.logger.log(action, device=device, detail=detail,
                               source=source, result=result)
        if result != "ok":
            self.log.append_log(line, False)

    def _set_status(self, message):
        if self.main_window is not None:
            try:
                self.main_window.statusBar().showMessage(message, 5000)
            except AttributeError:
                pass


# ======================================================================
# Small shared widgets
# ======================================================================

class _ModeParams(QWidget):
    """Ratio and timeout spinners, shown only for modes that use them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.ratio_label = QLabel("Ratio:")
        self.ratio_spin = QSpinBox()
        self.ratio_spin.setRange(1, 999)
        self.ratio_spin.setFixedWidth(60)
        self.ratio_spin.setToolTip("Pokes required per pellet (average, for RR)")

        self.timeout_label = QLabel("Timeout:")
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(0, 9999)
        self.timeout_spin.setValue(30)
        self.timeout_spin.setFixedWidth(60)
        self.timeout_spin.setToolTip("Lockout after pellet delivery, in seconds")
        self.timeout_unit = QLabel("s")

        for widget in (self.ratio_label, self.ratio_spin, self.timeout_label,
                       self.timeout_spin, self.timeout_unit):
            layout.addWidget(widget)
        layout.addStretch()

    def ratio(self):
        return self.ratio_spin.value()

    def timeout(self):
        return self.timeout_spin.value()

    def set_values(self, ratio, timeout):
        self.ratio_spin.setValue(int(ratio))
        self.timeout_spin.setValue(int(timeout))

    def show_fields(self, fields):
        for widget in (self.ratio_label, self.ratio_spin):
            widget.setVisible("ratio" in fields)
        for widget in (self.timeout_label, self.timeout_spin, self.timeout_unit):
            widget.setVisible("timeout" in fields)


def _sync_mode_params(mode_combo, params):
    """Show only the parameters the selected mode actually uses."""
    params.show_fields(proto.mode_fields(mode_combo.currentText()))


def _status_item(event):
    item = QTableWidgetItem(event.status if event.enabled else sched.STATUS_DISABLED)
    colors = {
        sched.STATUS_DONE: "#2ecc71",
        sched.STATUS_FAILED: "#e74c3c",
        sched.STATUS_MISSED: "#e67e22",
        sched.STATUS_PENDING: "#bbbbbb",
    }
    item.setForeground(QBrush(QColor(
        colors.get(event.status, "#888888") if event.enabled else "#666666")))
    return item


def _section_label(text):
    label = QLabel(text)
    label.setStyleSheet(
        "font-weight: bold; font-size: 11px; color: #888888; "
        "text-transform: uppercase; padding-bottom: 2px;")
    return label


def _row(*widgets):
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch()
    return container


def _connect_any_change(widget, slot):
    """Attach ``slot`` to whichever change signal a form widget exposes."""
    for signal_name in ("currentIndexChanged", "timeChanged", "valueChanged"):
        signal = getattr(widget, signal_name, None)
        if signal is not None:
            signal.connect(lambda *_: slot())
            return


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
