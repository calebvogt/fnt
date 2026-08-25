"""FED3 monitoring and control tab.

Ties together the serial links (:mod:`fed_serial`), the SD-card mirror
(:mod:`fed_mirror`), the recording session (:mod:`fed_session`), the scheduler
(:mod:`fed_scheduler`) and the plot (:mod:`fed_plot`).

Design notes for the parts that changed most:

*Connections.* Each device owns exactly one :class:`~fnt.fed3.fed_serial.Fed3Link`
for the lifetime of its port. Commands are queued onto that link rather than
opening the port again, which is what previously produced port-busy lockouts.

*Recording.* Behavioural events and every user action share one
host time base. Session state is persisted continuously, so a crash is resumable.

*Plot refresh.* Events mark the plot dirty; a 1 Hz timer redraws. Redrawing per
event made a busy rig unresponsive during pellet bursts.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

from PyQt5.QtCore import QEventLoop, Qt, QTime, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont
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

UI_TICK_MS = 1000               # scheduler countdowns
RECONNECT_TICK_MS = 5000        # auto-reconnect sweep
STATE_SAVE_MS = 15000           # session state persistence
PLOT_TICK_MS = 5000             # activity plot refresh
HEARTBEAT_TICK_MS = 10000       # device liveness probe
NARROW_WIDTH = 1120             # below this the panels stack into one column

# Reconnect backoff: give up automatic retries after this many consecutive
# failures so a permanently unplugged device stops churning the port list.
MAX_RECONNECT_ATTEMPTS = 20

# Liveness. A FED3 is silent between pokes, so silence alone proves nothing;
# after this long without a byte the tab sends PING, which every firmware answers.
# A device that does not answer within the grace period has wedged its USB stack
# and the link is recycled rather than left looking connected forever.
HEARTBEAT_IDLE_S = 30
HEARTBEAT_GRACE_S = 20

# How long to wait for a final SD pull when a recording stops.
FINAL_PULL_TIMEOUT_MS = 30000

# Grace period between telling devices to start a new trial and the session's
# first mirror pull, so the pull is scoped to the log they just rolled onto.
NEW_TRIAL_SETTLE_MS = 2500


class FEDTabWidget(QWidget):
    """Live monitoring, control and recording for a bank of FED3 devices."""

    scan_finished = pyqtSignal(list, list)

    def __init__(self, parent=None, worker_class=None):
        super().__init__(parent)
        self.main_window = parent
        # worker_class is accepted and ignored: this tab owns QThreads directly
        # (Fed3Link, PortScannerWorker) and never used the caller's pool.
        del worker_class

        self.devices = []
        self.session = None
        self.logger = session_mod.SessionLogger()
        self.scheduler = sched.Scheduler()
        self.sessions_dir = session_mod.default_session_root()
        self.device_names = session_mod.DeviceNames()
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
        # The plot shares the left column rather than sitting under both, so the
        # space the scheduler does not need goes to the graph instead of staying
        # blank while the device column runs on past it.
        self.plot_group = self._build_plot_group()

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
            "mirror and every user action are recorded into it on a shared "
            "clock.\n\n"
            "Every connected device also rolls onto a fresh SD log and zeroes "
            "its counters, so the session owns a clean boundary on the card as "
            "well as on the host. Resuming an interrupted session does not.")
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

        return group

    # --- global control ---------------------------------------------------

    def _build_control_group(self):
        # Named for its scope, not its contents. Every control in here acts on
        # every connected device at once, and each one is duplicated per-device
        # on the cards opposite; titled "FED Control Panel" there was nothing to
        # say which of the two a given button was.
        group = QGroupBox("All Connected Devices")
        layout = QHBoxLayout(group)
        layout.setSpacing(15)

        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(_section_label(
            "Configuration & commands — sent to every connected device"))

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
        right.addWidget(_section_label("Clock sync & data — all devices"))

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
        export_btn.setToolTip(
            "Save a copy of the SD logs to a folder you choose. This is for "
            "taking data elsewhere — while a session is recording, the mirror "
            "is already keeping a complete copy in the session folder.")
        export_btn.clicked.connect(self._export_all)
        pull_btn = QPushButton("Pull Data Now")
        pull_btn.setToolTip(
            "Force the mirror to pull immediately instead of waiting for the "
            "next event or tick. It pulls into the current session folder, so "
            "it does nothing useful when no session is recording.")
        pull_btn.clicked.connect(self._force_mirror_sync)
        for widget in (export_btn, pull_btn):
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
        # Sized to its contents. Left to expand, it grew to match the scheduler
        # table beside it and left a few hundred pixels of empty panel below
        # three rows of buttons.
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
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

        # Shown instead of the cards when nothing is plugged in. The previous
        # version kept one permanently empty card here, titled "Device 1", which
        # read as a device that existed and had failed rather than as an absence.
        self.devices_empty = QLabel(
            "No FED3 devices found.\n\n"
            "Plug one in and press Refresh Ports. If a device is connected but "
            "not listed, press Add Device to assign its port by hand.")
        self.devices_empty.setWordWrap(True)
        self.devices_empty.setAlignment(Qt.AlignCenter)
        self.devices_empty.setStyleSheet(
            "color: #999999; font-size: 12px; padding: 24px;")
        layout.addWidget(self.devices_empty)

        self.devices_container = QWidget()
        self.devices_flow = FlowLayout(margin=4, spacing=8)
        self.devices_container.setLayout(self.devices_flow)
        layout.addWidget(self.devices_container)
        # Kept in step by _reorder_devices from here on.
        self.devices_container.setVisible(False)
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
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
        self.sched_table.setMinimumHeight(70)
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
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
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
        """Refresh countdowns in place.

        Only the text changes. Replacing the status cell every second — as this
        did — allocated a QTableWidgetItem per row per tick and defeated the
        in-place refresh the rebuild logic exists to allow.
        """
        now = datetime.now()
        for event in self.scheduler.events:
            row = self._sched_rows.get(event.id)
            if row is None:
                continue
            countdown = self.sched_table.item(row, 2)
            if countdown is not None:
                countdown.setText(event.describe_trigger(now))
            status = self.sched_table.item(row, 5)
            label = event.status if event.enabled else sched.STATUS_DISABLED
            if status is not None and status.text() != label:
                _style_status_item(status, event)

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

        box = QGroupBox(f"Slot {slot_num}")
        grid = QGridLayout(box)

        name_edit = QLineEdit()
        # The placeholder shows the name that will be used if this is left blank,
        # so the field reads as an override rather than as something that must be
        # filled in. It is re-set whenever the device identifies itself.
        name_edit.setPlaceholderText(f"Slot {slot_num}")
        port_combo = QComboBox()
        port_combo.setEditable(True)
        remove_btn = QPushButton("Remove")
        mode_combo = QComboBox()
        mode_combo.addItems(proto.MODE_LABELS)
        apply_btn = QPushButton("Apply Mode")
        params = _ModeParams()
        status_label = QLabel("Not connected")
        status_label.setStyleSheet("color: #999999; font-size: 11px;")
        # Connection and recording are different states, and the card used to
        # show only the first: a connected device ticked its counters and drew a
        # live plot whether or not anything was being written to disk.
        rec_label = QLabel("")
        rec_label.setStyleSheet(
            "color: #e74c3c; font-size: 11px; font-weight: bold;")
        rec_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        feed_btn = QPushButton("Dispense")
        lights_btn = QPushButton("Lights: OFF")
        lights_btn.setCheckable(True)
        lights_btn.setStyleSheet("""
            QPushButton:checked { background-color: #f1c40f; color: black; }
            QPushButton { min-height: 22px; font-weight: bold; }
        """)
        export_btn = QPushButton("Export Logs...")

        manual = _row(QLabel("Manual:"), feed_btn, lights_btn)
        data = _row(QLabel("Data:"), export_btn)

        # The live poke/pellet counters sit on the card that controls the
        # device. They used to live in a separate "Device View" group at the
        # bottom of the page, which put a device's readout and its controls in
        # different columns, hundreds of pixels apart.
        svg_view = FEDSvgView()
        svg_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        svg_view.setMinimumHeight(84)
        svg_view.setMaximumHeight(110)

        grid.addWidget(QLabel("Port:"), 0, 0)
        grid.addWidget(port_combo, 0, 1, 1, 2)
        grid.addWidget(remove_btn, 0, 3, Qt.AlignRight)
        grid.addWidget(QLabel("Name:"), 1, 0)
        grid.addWidget(name_edit, 1, 1, 1, 3)
        grid.addWidget(QLabel("Mode:"), 2, 0)
        grid.addWidget(mode_combo, 2, 1, 1, 2)
        grid.addWidget(apply_btn, 2, 3)
        grid.addWidget(params, 3, 0, 1, 4)
        grid.addWidget(svg_view, 4, 0, 1, 4)
        grid.addWidget(manual, 5, 0, 1, 4)
        grid.addWidget(data, 6, 0, 1, 4)
        grid.addWidget(status_label, 7, 0, 1, 3)
        grid.addWidget(rec_label, 7, 3)
        grid.setColumnStretch(1, 1)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        device = FedDevice(slot_num, {
            "box": box, "name_edit": name_edit, "port_combo": port_combo,
            "mode_combo": mode_combo, "params": params,
            "ratio_spin": params.ratio_spin, "timeout_spin": params.timeout_spin,
            "status_label": status_label, "rec_label": rec_label,
            "lights_btn": lights_btn,
            "svg_view": svg_view, "remove_btn": remove_btn,
        })

        # The title tracks each keystroke, but the combo boxes only rebuild once
        # editing finishes — refreshing them per keystroke reset the user's plot
        # filter to "All Devices" mid-word.
        name_edit.textChanged.connect(lambda: device.box.setTitle(device.name))
        name_edit.editingFinished.connect(lambda: self._on_label_edited(device))
        remove_btn.clicked.connect(lambda: self._remove_device(device))
        apply_btn.clicked.connect(lambda: self._apply_device_mode(device))
        mode_combo.currentTextChanged.connect(
            lambda: _sync_mode_params(mode_combo, params))
        feed_btn.clicked.connect(
            lambda: self._execute_action(device, sched.ACTION_DISPENSE, {}))
        lights_btn.clicked.connect(
            lambda: self._execute_action(device, sched.ACTION_LIGHTS,
                                         {"lights": lights_btn.isChecked()}))
        export_btn.clicked.connect(lambda: self._export_device(device))
        port_combo.activated.connect(lambda: self._on_port_selected(device))
        port_combo.lineEdit().editingFinished.connect(
            lambda: self._on_port_selected(device))

        _sync_mode_params(mode_combo, params)
        port_combo.addItem("Scanning...")
        port_combo.setEnabled(False)

        self.devices_flow.addWidget(box)
        self.devices.append(device)
        self._reorder_devices()
        self._refresh_device_combos()
        if refresh:
            self.refresh_ports()
        return device

    def _on_label_edited(self, device):
        """The user finished typing in a slot's name field."""
        # Remembered against the on-board ID so the label comes back the next
        # time this physical device is plugged in, on whatever port and slot.
        self.device_names.set(device.device_id, device.label)
        self._rename_device(device)

    def _rename_device(self, device):
        """Propagate a change to anything that feeds ``device.name``.

        Called for an edited label *and* for a newly identified device, because
        identification changes the default name ("Slot 2" becomes "FED 4") for
        every slot the user has not labelled by hand.
        """
        device.name_edit.setPlaceholderText(device.default_name)
        if device.last_known_name == device.name:
            device.box.setTitle(device.name)
            return
        # Scheduled events reference devices by name, so a rename has to carry
        # them across or they silently stop matching a target.
        self.scheduler.rename_target(device.last_known_name, device.name)
        device.last_known_name = device.name
        device.box.setTitle(device.name)
        if device.link is not None:
            device.link.owner = device.name
        self._refresh_device_combos()
        self._rebuild_scheduler_table()

    def _adopt_identity(self, device, device_id, firmware=None, replace=False):
        """Record what a device says it is, and rename the slot to match.

        ``replace`` distinguishes the two ways identity arrives. A device
        speaking for itself (PING, STATUS) can only ever add information, so an
        empty ID is ignored. Pointing a slot at a port replaces its identity
        outright, empty included: a slot moved to a port that reports no ID must
        stop calling itself "FED 4".
        """
        # Compared as text: the ID arrives as a string from PING and STATUS but
        # as whatever json.load produced when a session is resumed, and an int/str
        # mismatch would re-adopt the same identity on every STATUS reply.
        known = device_id not in (None, "")
        changed = str(device_id or "") != str(device.device_id or "")
        if known or replace:
            device.device_id = device_id or None
        if firmware or replace:
            device.firmware = firmware
        if changed and not known:
            # Identity withdrawn: the label went with the old device, not the slot.
            device.name_edit.setText("")
        changed = changed and known
        if changed:
            # A label the user gave this device on a previous run outranks the
            # "FED n" default, but never overwrites one they typed just now.
            remembered = self.device_names.get(device.device_id)
            if remembered and not device.label:
                device.name_edit.setText(remembered)
        self._rename_device(device)

    def _remove_device(self, device):
        if device.is_tracking and not self._confirm(
                f"Remove {device.name}? It is currently connected and tracking.\n\n"
                f"{device.port} will be ignored by future scans until you pick it "
                f"from a slot's port list again."):
            return

        # Removing a slot is how the user says "leave this device alone", so the
        # port is skipped by later scans. Without this, the next Refresh Ports —
        # or the automatic rescan — would put it straight back.
        port = device.port
        if port:
            self.removed_ports.add(port)
        self._disconnect_device(device)
        self.devices_flow.removeWidget(device.box)
        device.box.deleteLater()
        self.devices.remove(device)

        self._reorder_devices()
        self._refresh_device_combos()
        self._mark_plot_dirty()
        self._log_action("Device removed", device=device.name,
                         detail=(f"{port} ignored by future scans" if port
                                 else "empty slot"))

    def _reorder_devices(self):
        self.devices.sort(key=lambda d: d.slot_num)
        for device in self.devices:
            device.box.setTitle(device.name)
            device.name_edit.setPlaceholderText(device.default_name)
            self.devices_flow.removeWidget(device.box)
        for device in self.devices:
            self.devices_flow.addWidget(device.box)
        self.devices_container.setVisible(bool(self.devices))
        self.devices_empty.setVisible(not self.devices)
        if not self.devices:
            self.devices_empty.setText(self._empty_devices_message())

    def _empty_devices_message(self):
        """What to say in place of the cards when no slot exists.

        A dismissed port is called out by name, because otherwise "press Refresh
        Ports" is advice that cannot work: a removed device is deliberately
        skipped by the scan, and nothing on screen would say so.
        """
        message = ("No FED3 devices found.\n\n"
                   "Plug one in and press Refresh Ports. If a device is connected "
                   "but not listed, press Add Device to assign its port by hand.")
        if self.removed_ports:
            ports = ", ".join(sorted(self.removed_ports))
            message += ("\n\nScans are currently ignoring " + ports +
                        ", removed earlier this session. Add Device and pick the "
                        "port to bring it back.")
        return message

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

        # A rescan is the documented remedy after reflashing, so it clears the
        # refusal and lets the device be tried again.
        for device in self.devices:
            device.refused = False
            device.box.setStyleSheet("")

        active = []
        for port, status, device_id, firmware in results:
            if status == "FED3 Active":
                active.append(port)
                self._port_info[port] = {"id": device_id, "firmware": firmware}

        self.log.append_log(f"[Scan] Active FED3 ports: {active or 'none'}")
        self._populate_port_combos(all_ports)
        self._assign_discovered_ports(active)
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
        self._adopt_identity(device, info.get("id"), info.get("firmware"),
                             replace=True)

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
                f"slot {device.slot_num}.\n\nSlots normally match the number on "
                f"the device, so FED {onboard} would usually belong in slot "
                f"{onboard}.\n\nAssign it here anyway?"):
            device.port_combo.setCurrentIndex(-1)
            return

        self.removed_ports.discard(port)
        device.refused = False
        device.box.setStyleSheet("")
        device.saved_port = port
        self._adopt_identity(device, onboard, info.get("firmware"), replace=True)
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
        device.reconnect_gave_up = False
        device.awaiting_pong_since = None
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
        # PING first, and nothing else until it is answered: the reply is what
        # confirms the device is running firmware FNT can actually drive.
        device.link.send(proto.CMD_PING)
        if self.session is not None:
            self._attach_session_to_device(device)
        self._refresh_recording_indicators()
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
        self._refresh_recording_indicators()

        if device.refused:
            # The panel already explains why, in terms the user can act on.
            return

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
            mirror, device.mirror = device.mirror, None
            mirror.stop()
            mirror.deleteLater()
        if device.transfer is not None:
            transfer, device.transfer = device.transfer, None
            transfer.cancel("disconnecting")
            # Parented to the tab, so it outlives the device slot unless it is
            # explicitly dropped: 20 reconnect attempts would otherwise leave 20
            # live Fed3Transfer objects and their timers behind.
            transfer.deleteLater()
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
            # Reconnecting a refused device would loop: connect, PING, refuse,
            # repeat every few seconds. Refresh Ports is the way back.
            if device.refused:
                continue
            if device.connect_attempts >= MAX_RECONNECT_ATTEMPTS:
                if not device.reconnect_gave_up:
                    # Silently giving up looked identical to still trying, which
                    # is how a device could sit disconnected for a whole run
                    # without anyone noticing.
                    device.reconnect_gave_up = True
                    self.log.append_log(
                        f"[{device.name}] gave up after {MAX_RECONNECT_ATTEMPTS} "
                        f"reconnect attempts — reselect its port to retry", False)
                    self._log_action(
                        "Reconnect abandoned", device=device.name,
                        detail=f"{MAX_RECONNECT_ATTEMPTS} consecutive failures",
                        source="system", result="failed")
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

        if stripped.startswith("PONG_FED3"):
            self._apply_handshake(device, stripped)
            return

        status = proto.parse_status(stripped)
        if status is not None:
            self._apply_status(device, status)
            return

        if stripped.startswith("SYNCED,"):
            self._record_clock_sync(device, stripped.split(",", 1)[1])
            return

        if stripped.startswith(proto.REPLY_NEW_TRIAL):
            self._adopt_current_file(
                device, stripped[len(proto.REPLY_NEW_TRIAL):])
            self._log_action("New trial started", device=device.name,
                             detail=device.current_file or "", source="device")
            return

        event = proto.parse_event(stripped)
        if event is not None:
            self._on_device_event(device, event)

    def _apply_handshake(self, device, line):
        """Accept or refuse a device based on the firmware its PING reports.

        A board older than the supported firmware is dropped rather than driven
        in a reduced mode. It parses ``GET_FILE:<name>,<offset>`` as a request
        for a file literally named ``"<name>,0"``, so every download and every
        mirror pull would fail — and it would fail quietly, halfway through an
        experiment, which is exactly when nobody is watching the log.
        """
        device_id, firmware = proto.parse_pong(line)
        device.firmware = firmware
        self._adopt_identity(device, device_id)

        if not proto.is_supported(firmware):
            self._refuse_device(device, firmware)
            return

        device.box.setToolTip("")
        device.status_label.setText(f"Connected on {device.port}")
        device.status_label.setStyleSheet("color: #4caf50; font-size: 11px;")
        self._log_action("Firmware accepted", device=device.name,
                         detail=f"FW {firmware}", source="device")

        # Deferred until the handshake so nothing is sent to a device that turns
        # out to be unusable. The link may already be gone: lines from the serial
        # thread are queued, so a PONG can arrive after the link was torn down.
        if device.link is not None:
            self._sync_device(device)
            device.link.send(proto.CMD_STATUS)

    def _refuse_device(self, device, firmware):
        """Disconnect a device running firmware FNT cannot drive, and say so."""
        required = proto.firmware_requirement()
        reported = firmware or "none reported"
        detail = (f"firmware {reported}; {required} or newer required")

        # Set before tearing the link down: the disconnected signal is queued and
        # arrives after this returns, and _on_link_disconnected must not overwrite
        # the explanation below with a generic "Disconnected" line.
        device.refused = True
        self._disconnect_device(device)
        device.status_label.setText(f"Refused — firmware {reported}, needs {required}")
        device.status_label.setStyleSheet("color: #e57373; font-size: 11px;")
        device.box.setStyleSheet("QGroupBox { border: 2px solid #e67e22; }")
        device.box.setToolTip(
            f"This device reports firmware {reported}, but FNT requires "
            f"{required} or newer.\n\n"
            "Older firmware cannot do ranged SD transfers, so mirroring and "
            "exports fail, and it reports pokes without a device timestamp.\n\n"
            "Reflash it from fnt-fed3/ClassicFed3withTimeSync, then press "
            "Refresh Ports.")
        self.log.append_log(
            f"[{device.name}] refused: {detail}. Reflash from fnt-fed3 and "
            f"press Refresh Ports.", False)
        self._log_action("Device refused", device=device.name, detail=detail,
                         source="device", result="failed")
        self._set_status(f"{device.name} needs reflashing to firmware {required}")

    def _apply_status(self, device, status):
        device.firmware = status.get("fw") or device.firmware
        self._adopt_identity(device, status.get("id"))
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
        self._adopt_current_file(device, status.get("file"))
        if status.get("session"):
            device.status_label.setText(
                f"Connected on {device.port} — {status['session']} "
                f"· {status.get('file', 'no file')}")

    def _adopt_current_file(self, device, filename):
        """Note which SD log the device is writing to, and tell its mirror.

        This is what scopes a session's mirror. Without it the mirror would have
        to guess which of the files on the card belongs to the running
        experiment, and the only honest guess — all of them — is what used to
        make the live data arrive last.
        """
        filename = (filename or "").strip()
        if not filename:
            return
        device.current_file = filename
        # Forwarded unconditionally, not only when the name changes. NEW_TRIAL
        # can hand back the *same* name — FED3 recycles the filename of any log
        # with fewer than three lines — and an early return on "unchanged" left
        # the mirror having never been told which file the session owns at all.
        if device.mirror is not None:
            device.mirror.adopt_current_file(filename)

    def _on_device_event(self, device, event):
        host_ts = session_mod.host_now()
        device.apply_counts(event.counts)

        if event.kind == proto.EVENT_JAM:
            self._on_device_jam(device, event, host_ts)
            return

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

    def _on_device_jam(self, device, event, host_ts):
        """The device gave up trying to dispense.

        Reported loudly because it is the one thing an unattended cage cannot
        recover from on its own: the hopper is empty or the disk is jammed, and
        until someone attends to it the animal is on extinction. Older firmware
        could not report this at all — it spun in the dispense loop forever and
        simply stopped answering, which read as a dead device.
        """
        device.status_label.setText(f"{device.name}: JAMMED — check the hopper")
        device.status_label.setStyleSheet(
            "color: #e67e22; font-size: 11px; font-weight: bold;")
        device.box.setStyleSheet("QGroupBox { border: 2px solid #e67e22; }")
        device.box.setToolTip(
            "The device tried to dispense and could not. Refill the hopper or "
            "clear the pellet disk; it keeps running and logging in the "
            "meantime.")
        if device.event_log is not None:
            device.event_log.append(event, host_ts)
        self.log.append_log(f"[{device.name}] JAM — could not dispense", False)
        self._log_action("Dispense failed", device=device.name,
                         detail="hopper empty or disk jammed",
                         source="device", result="failed")
        self._set_status(f"{device.name} could not dispense — check the hopper")

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

    def _start_new_trial(self, device, source="user"):
        """Roll the device onto a fresh SD log and zero its counters.

        No longer a button. Starting a recording does this, so a session owns a
        file boundary on the card without anyone having to remember a second
        step; the scheduler can still do it mid-run for a daily boundary. It used
        to be a red button in three places, competing with Start Recording to
        mean "this is where my experiment begins" and agreeing with it only if
        the user pressed both.
        """
        ok = self._send(device, proto.CMD_NEW_TRIAL, "Start new trial", source)
        if ok:
            device.reset_counters()
            device.svg_view.set_counts(device.stats)
            self._mark_plot_dirty()
        return ok

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

        # A resumed session is continuing an experiment that is already running:
        # rolling its file and zeroing its counters mid-run is exactly the wrong
        # thing. Only a genuinely new recording claims a fresh trial.
        rolling = resume_root is None
        for device in self.devices:
            self._attach_session_to_device(device, adopt_current=not rolling)

        if rolling:
            for device in self.devices:
                if device.is_connected:
                    self._start_new_trial(device, source="system")
                    # The reply names the new log; STATUS is the fallback for a
                    # device that was mid-stream when NEW_TRIAL arrived.
                    device.link.send(proto.CMD_STATUS)
            # Give those replies time to land before the first pull, so it is
            # scoped to the session's own file rather than the one it replaced.
            QTimer.singleShot(NEW_TRIAL_SETTLE_MS, self._force_mirror_sync)

        self.record_btn.setText("Stop Recording")
        self._refresh_recording_indicators()
        self.session_label.setText(f"Recording to {self.session.name}")
        self.session_label.setStyleSheet("color: #4caf50;")
        self.open_folder_btn.setEnabled(True)
        self._log_action("Recording started" if not resume_root else "Recording resumed",
                         detail=self.session.root, source="system")
        self._save_state()
        return True

    def _refresh_recording_indicators(self):
        """Show on each card whether its data is actually being written."""
        for device in self.devices:
            recording = (self.session is not None
                         and device.is_connected
                         and device.event_log is not None)
            device.rec_label.setText("\u25cf RECORDING" if recording else "")

    def _attach_session_to_device(self, device, adopt_current=True):
        """Give a device its event log and SD mirror for the active session.

        ``adopt_current`` is False when the caller is about to roll the device
        onto a new log. The file it is writing *now* belongs to whatever ran
        before this session, and claiming it here would both put pre-session data
        in the session folder and spend the first pull on it.
        """
        if self.session is None:
            return
        device.event_log = session_mod.DeviceEventLog(
            self.session.device_events_path(device.name), device.name)
        if device.transfer is not None and device.mirror is None:
            device.mirror = DeviceMirror(
                device.transfer,
                self.session.device_mirror_dir(device.name),
                session_mod.device_archive_dir(self.sessions_dir, device.name),
                parent=self)
            if adopt_current:
                device.mirror.adopt_current_file(device.current_file)
            device.mirror.progress.connect(self.mirror_status.setText)
            device.mirror.failed.connect(
                lambda message, d=device: self._on_mirror_failed(d, message))
            device.mirror.updated.connect(
                lambda name, added, total, d=device:
                self._log_action("Mirrored SD data", device=d.name,
                                 detail=f"{name}: +{added} bytes ({total} total)",
                                 source="system"))
            if adopt_current:
                device.mirror.sync_now(force=True)

    def _on_mirror_failed(self, device, message):
        self.mirror_status.setText(message)
        self.log.append_log(f"[{device.name}] {message}", False)
        self._log_action("Mirror pull failed", device=device.name,
                         detail=message, source="system", result="failed")

    def _stop_session(self):
        if self.session is None:
            return

        # Wait for the final pull instead of firing and forgetting it. The
        # previous version requested a sync and immediately dropped the mirror,
        # so the last few kilobytes written to the SD card between the previous
        # pull and the stop button were never copied to the session folder.
        self._drain_mirrors()

        for device in self.devices:
            if device.mirror is not None:
                mirror, device.mirror = device.mirror, None
                mirror.stop()
                mirror.deleteLater()
            device.event_log = None
        self._refresh_recording_indicators()

        self._log_action("Recording stopped", source="system")
        self.session.mark_closed()
        self.logger.detach()

        self.record_btn.setText("Start Recording")
        self.session_label.setText(f"Last session: {self.session.name}")
        self.session_label.setStyleSheet("color: #999999;")
        self.session = None

    def _drain_mirrors(self, timeout_ms=FINAL_PULL_TIMEOUT_MS):
        """Run one last SD pull on every mirror and wait for it to land.

        Bounded: a device that has gone unresponsive must not be able to block
        the stop button. Anything not pulled stays on the SD card and is picked
        up by the next session's mirror, which resumes from the recorded offset.
        """
        mirrors = [d.mirror for d in self.devices if d.mirror is not None]
        if not mirrors:
            return

        self.mirror_status.setText("Finishing final SD pull...")
        for mirror in mirrors:
            mirror.sync_now(force=True)

        loop = QEventLoop()
        deadline = QTimer(self)
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        poll = QTimer(self)
        poll.timeout.connect(
            lambda: loop.quit() if not any(m.busy for m in mirrors) else None)
        deadline.start(timeout_ms)
        poll.start(200)
        try:
            loop.exec_()
        finally:
            poll.stop()
            deadline.stop()

        if any(m.busy for m in mirrors):
            self._log_action(
                "Final SD pull timed out", source="system", result="warn",
                detail=f"still running after {timeout_ms // 1000}s; "
                       f"remaining data stays on the SD card")

    def _save_state(self):
        """Persist everything needed to resume after a crash."""
        if self.session is None:
            return
        self.session.write_state({
            "status": session_mod.STATUS_RUNNING,
            "devices": [d.to_state() for d in self.devices],
            "scheduled_events": self.scheduler.to_list(),
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
            device.device_id = entry.get("device_id")
            device.firmware = entry.get("firmware")
            # The label, not the resolved name: restoring "FED 4" into the
            # override field would freeze the default in place as a user label.
            device.name_edit.setText(entry.get("label", ""))
            device.last_known_name = device.name
            device.box.setTitle(device.name)
            device.name_edit.setPlaceholderText(device.default_name)
            device.saved_port = entry.get("port", "")
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

        # The plot is expensive and slow-moving; it gets its own, slower timer
        # rather than riding the 1 Hz countdown tick.
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self._on_plot_tick)
        self.plot_timer.start(PLOT_TICK_MS)

        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self._check_heartbeats)
        self.heartbeat_timer.start(HEARTBEAT_TICK_MS)

    def _on_ui_tick(self):
        self._run_due_events()
        self._tick_scheduler_display()
        # Cheap, and genuinely clock-dependent: a delay-based trigger resolves
        # relative to now, so a frozen preview would misstate when it fires.
        self._update_sched_preview()

    def _on_plot_tick(self):
        """Redraw only when something changed, or when a live trace must advance.

        The previous version redrew whenever *any* device was tracking, which
        meant a full matplotlib re-render every second for the whole length of a
        multi-day session — the single biggest reason the tab felt sluggish.
        A live trace only needs redrawing to extend its "time since last pellet"
        line, which is legible at this cadence.
        """
        if self._plot_dirty or any(d.is_tracking for d in self.devices):
            self._redraw_plot()

    # ------------------------------------------------------------ liveness

    def _check_heartbeats(self):
        """Detect a device that has stopped talking without dropping the port.

        A wedged SAMD21 USB stack leaves the port open and readable, so nothing
        in the serial layer errors: FNT goes on showing "Connected" while the
        device answers nothing. PING is answered by every firmware, so an
        unanswered one is unambiguous.
        """
        for device in self.devices:
            if not device.is_connected:
                device.awaiting_pong_since = None
                continue
            # Activity of any kind, including an SD transfer, proves liveness.
            if device.transfer is not None and device.transfer.busy:
                device.awaiting_pong_since = None
                continue

            idle = device.link.seconds_since_rx()
            if device.awaiting_pong_since is not None:
                if idle < HEARTBEAT_IDLE_S:
                    device.awaiting_pong_since = None       # it answered
                elif (session_mod.host_now() - device.awaiting_pong_since
                        > HEARTBEAT_GRACE_S):
                    device.awaiting_pong_since = None
                    self.log.append_log(
                        f"[{device.name}] no response to PING after "
                        f"{HEARTBEAT_GRACE_S}s — recycling the connection", False)
                    self._log_action(
                        "Device unresponsive", device=device.name,
                        detail=f"silent for {idle:.0f}s, PING unanswered",
                        source="system", result="failed")
                    # Reconnecting reopens the port, which resets the device's
                    # USB session and is what actually clears the wedge.
                    self._connect_device(device)
            elif idle > HEARTBEAT_IDLE_S:
                device.awaiting_pong_since = session_mod.host_now()
                device.link.send(proto.CMD_PING)

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
                       self.plot_group, self.left_column, self.right_column):
            self.columns_layout.removeWidget(widget)
        for layout, widget in ((self.left_column_layout, self.control_group),
                               (self.left_column_layout, self.scheduler_group),
                               (self.left_column_layout, self.plot_group),
                               (self.right_column_layout, self.devices_group)):
            layout.removeWidget(widget)

        for layout in (self.left_column_layout, self.right_column_layout):
            _drop_spacers(layout)

        if is_narrow:
            self.left_column.hide()
            self.right_column.hide()
            for row, widget in enumerate(
                    (self.control_group, self.scheduler_group, self.devices_group,
                     self.plot_group)):
                self.columns_layout.addWidget(widget, row, 0)
                widget.show()
            self.columns_layout.setColumnStretch(0, 1)
            self.columns_layout.setColumnStretch(1, 0)
        else:
            self.left_column_layout.addWidget(self.control_group)
            self.left_column_layout.addWidget(self.scheduler_group)
            self.left_column_layout.addWidget(self.plot_group, stretch=1)
            self.right_column_layout.addWidget(self.devices_group)
            self.right_column_layout.addStretch()
            self.columns_layout.addWidget(self.left_column, 0, 0)
            self.columns_layout.addWidget(self.right_column, 0, 1)
            for widget in (self.left_column, self.right_column, self.control_group,
                           self.scheduler_group, self.devices_group, self.plot_group):
                widget.show()
            self.columns_layout.setColumnStretch(0, 1)
            self.columns_layout.setColumnStretch(1, 1)

    def cleanup(self):  # noqa: D401
        """Close every connection, timer and file before the window goes away."""
        self.log.append_log("Shutting down FED3 tab...")
        for name in ("ui_timer", "reconnect_timer", "state_timer", "sync_timer",
                     "plot_timer", "heartbeat_timer"):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.stop()

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


STATUS_COLORS = {
    sched.STATUS_DONE: "#2ecc71",
    sched.STATUS_FAILED: "#e74c3c",
    sched.STATUS_MISSED: "#e67e22",
    sched.STATUS_PENDING: "#bbbbbb",
}


def _style_status_item(item, event):
    """Set a status cell's text and colour from an event."""
    item.setText(event.status if event.enabled else sched.STATUS_DISABLED)
    item.setForeground(QBrush(QColor(
        STATUS_COLORS.get(event.status, "#888888") if event.enabled else "#666666")))
    return item


def _status_item(event):
    return _style_status_item(QTableWidgetItem(), event)


def _section_label(text):
    label = QLabel(text)
    label.setStyleSheet(
        "font-weight: bold; font-size: 11px; color: #888888; "
        "text-transform: uppercase; padding-bottom: 2px;")
    return label


def _drop_spacers(layout):
    """Remove trailing stretch items so a re-layout does not accumulate them."""
    for index in reversed(range(layout.count())):
        item = layout.itemAt(index)
        if item is not None and item.widget() is None and item.layout() is None:
            layout.takeAt(index)


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
