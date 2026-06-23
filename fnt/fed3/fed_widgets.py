import os
import sys
import csv
from datetime import datetime, time, timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLabel, QGroupBox, QTextEdit, QScrollArea, QSizePolicy, 
    QComboBox, QSpinBox, QLineEdit, QFrame, QFileDialog, QTabWidget,
    QLayout, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QTimeEdit, QStackedWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, QRect, QRectF, QPoint, QSize, QTime
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter, QBrush

try:
    from PyQt5.QtSvg import QSvgRenderer
except ImportError:
    QSvgRenderer = None

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

from . import fed_comms


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=-1, spacing=-1):
        super(FlowLayout, self).__init__(parent)
        if margin is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self.itemList = []

    def __del__(self):
        item = self.takeAt(0)
        while item:
            item = self.takeAt(0)

    def addItem(self, item):
        self.itemList.append(item)

    def removeWidget(self, widget):
        for i in reversed(range(len(self.itemList))):
            item = self.itemList[i]
            if item.widget() == widget:
                self.takeAt(i)
                break
        super(FlowLayout, self).removeWidget(widget)

    def count(self):
        return len(self.itemList)

    def itemAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList[index]
        return None

    def takeAt(self, index):
        if index >= 0 and index < len(self.itemList):
            return self.itemList.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        height = self.doLayout(QRect(0, 0, width, 0), True)
        return height

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self.doLayout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self.itemList:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        size += QSize(left + right, top + bottom)
        return size

    def doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0
        
        spacing = self.spacing()

        for item in self.itemList:
            wid = item.widget()
            spaceX = spacing
            spaceY = spacing
            
            nextX = x + item.sizeHint().width() + spaceX
            if nextX - spaceX > rect.right() and lineHeight > 0:
                x = rect.x()
                y = y + lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()

class _CounterData:
    """Lightweight data holder that mimics the old OverlayLabel API."""
    def __init__(self, text="0"):
        self._text = text

    def text(self):
        return self._text

    def setText(self, text):
        self._text = str(text)


class FEDSvgView(QWidget):
    """Renders the FED3 SVG and paints overlay counters directly — no child
    widgets, so there are no background-fill / white-corner issues."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Load SVG via QSvgRenderer
        self._renderer = None
        if QSvgRenderer is not None:
            try:
                base_path = sys._MEIPASS
                svg_path = os.path.join(base_path, "fnt", "fed3", "fed3_image.svg")
            except Exception:
                svg_path = os.path.join(os.path.dirname(__file__), "fed3_image.svg")
            self._renderer = QSvgRenderer(svg_path)

        # Counter data objects (public API matches old OverlayLabel)
        self.left_counter = _CounterData("0")
        self.right_counter = _CounterData("0")
        self.pellet_counter = _CounterData("0")

        # Tracking status (controls count text display)
        self.is_tracking = False

        # Flash state
        self._flash_colors = {}  # counter_name -> QColor

    # ------------------------------------------------------------------
    def flash_counter(self, counter_name):
        self._flash_colors[counter_name] = QColor(76, 175, 80, 180)
        self.update()
        QTimer.singleShot(200, lambda: self._end_flash(counter_name))

    def _end_flash(self, counter_name):
        self._flash_colors.pop(counter_name, None)
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # --- SVG aspect-fit ---
        svg_aspect = 163.67577 / 116.04688
        widget_aspect = w / h if h > 0 else 1

        if widget_aspect > svg_aspect:
            new_h = h
            new_w = int(h * svg_aspect)
        else:
            new_w = w
            new_h = int(w / svg_aspect)

        x_off = (w - new_w) // 2
        y_off = (h - new_h) // 2
        svg_rect = QRectF(x_off, y_off, new_w, new_h)

        if self._renderer:
            self._renderer.render(painter, svg_rect)

        # --- Overlay sizes ---
        poke_radius = int(new_w * 0.0953)
        poke_diam = poke_radius * 2
        pellet_w = int(new_w * 0.1772)
        pellet_h = int(new_h * 0.1677)

        default_bg = QColor(0, 0, 0, 150)

        # Helper to draw one overlay
        def draw_overlay(cx, cy, ow, oh, is_circle, counter_name, counter_data, radius=0):
            bg = self._flash_colors.get(counter_name, default_bg)
            rect = QRect(x_off + cx, y_off + cy, ow, oh)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(bg))
            if is_circle:
                painter.drawEllipse(rect)
            else:
                painter.drawRoundedRect(rect, radius, radius)
            
            # Only draw text if tracking has begun
            if self.is_tracking:
                painter.setPen(QColor(255, 255, 255))
                font_size = max(10, int(poke_radius * 0.65))
                painter.setFont(QFont("Arial", font_size, QFont.Bold))
                painter.drawText(rect, Qt.AlignCenter, counter_data.text())

        # Left poke
        lx = int(new_w * 0.2501) - poke_radius
        ly = int(new_h * 0.6444) - poke_radius
        draw_overlay(lx, ly, poke_diam, poke_diam, True, "left", self.left_counter)

        # Right poke
        rx = int(new_w * 0.7488) - poke_radius
        ry = int(new_h * 0.6426) - poke_radius
        draw_overlay(rx, ry, poke_diam, poke_diam, True, "right", self.right_counter)

        # Pellet
        px = int(new_w * 0.4108)
        py = int(new_h * 0.6095)
        draw_overlay(px, py, pellet_w, pellet_h, False, "pellet", self.pellet_counter, radius=4)

        painter.end()

    # ------------------------------------------------------------------
    def sizeHint(self):
        return QSize(350, 200)

    def minimumSizeHint(self):
        return QSize(250, 150)



class Fed3TrackerWorker(QThread):
    line_received = pyqtSignal(str, str) # dev_name, line
    error_received = pyqtSignal(str, str) # dev_name, error

    def __init__(self, port, dev_name):
        super().__init__()
        self.port = port
        self.dev_name = dev_name
        self.tracker = fed_comms.Fed3Tracker(self.port)

    def run(self):
        def callback(line):
            if line.startswith("ERROR:"):
                self.error_received.emit(self.dev_name, line)
            else:
                self.line_received.emit(self.dev_name, line)
        self.tracker.start(callback)

    def stop(self):
        self.tracker.stop()
        self.quit()
        self.wait()


class PortScannerWorker(QThread):
    finished_scan = pyqtSignal(list, list)

    def __init__(self, active_ports=None):
        super().__init__()
        self.active_ports = active_ports or []

    def run(self):
        import serial
        from serial.tools import list_ports
        import time
        from concurrent.futures import ThreadPoolExecutor
        from . import fed_comms

        ports = list(list_ports.comports())
        # We only check candidate ports for active FED3 to prevent hangs on legacy/virtual ports
        ping_ports = [p for p in ports if fed_comms.is_candidate_port(p)]

        def check_port(p):
            dev = p.device
            desc = getattr(p, "description", "") or ""
            mfg = getattr(p, "manufacturer", "") or ""

            if dev in self.active_ports:
                return (dev, "Active", None)

            try:
                ser = serial.Serial()
                ser.port = dev
                ser.baudrate = 115200
                ser.timeout = 1.0
                ser.open()
                # Wait for board to process and reboot upon connection
                time.sleep(2.0)
                ser.write(b"PING\n")
                time.sleep(0.2)
                response = ser.read_all().decode('utf-8', errors='ignore')
                ser.close()
                if "PONG_FED3" in response:
                    device_id = None
                    if "ID:" in response:
                        try:
                            parts = response.split("ID:")
                            if len(parts) > 1:
                                raw_id = parts[1].strip().split()[0]
                                device_id = "".join([c for c in raw_id if c.isdigit()])
                        except Exception:
                            pass
                    return (dev, "FED3 Active", device_id)
            except Exception as e:
                err_str = str(e)
                if "busy" in err_str.lower() or "already open" in err_str.lower() or "permission denied" in err_str.lower():
                    return (dev, "Busy/In Use", None)

            return (dev, "Unresponsive", None)

        # Check ports in parallel to avoid long UI hangs
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(check_port, ping_ports)

        valid_ports = [r for r in results if r is not None]
        # Return all system ports to allow selection of non-FED ports
        all_devs = [p.device for p in ports]
        self.finished_scan.emit(valid_ports, all_devs)



class CollapsibleLogBox(QWidget):
    """A collapsible log box with a command input for serial communication."""
    command_submitted = pyqtSignal(str)

    def __init__(self, title="FED Log", parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Toggle button
        self.toggle_button = QPushButton(f"▶ {title}")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left; 
                padding: 6px; 
                font-weight: bold; 
                background-color: #333333; 
                color: #ffffff; 
                border: none;
                border-top: 1px solid #444444;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        self.toggle_button.toggled.connect(self.on_toggle)

        # Log and Input container
        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(0)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(150)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #cccccc; border: none; font-family: monospace;")

        self.input_layout = QHBoxLayout()
        self.input_layout.setContentsMargins(0, 0, 0, 0)
        self.input_layout.setSpacing(0)
        
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Message")
        self.command_input.setStyleSheet("background-color: #2b2b2b; color: #ffffff; border: 1px solid #3f3f3f; padding: 4px;")
        self.command_input.returnPressed.connect(self.submit_command)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedWidth(60)
        self.send_btn.clicked.connect(self.submit_command)
        
        self.input_layout.addWidget(self.command_input)
        self.input_layout.addWidget(self.send_btn)

        self.container_layout.addLayout(self.input_layout)
        self.container_layout.addWidget(self.log_text)

        self.layout.addWidget(self.toggle_button)
        self.layout.addWidget(self.container)
        
        self.container.hide()

    def on_toggle(self, checked):
        if checked:
            self.toggle_button.setText(f"▼ {self.toggle_button.text()[2:]}")
            self.container.show()
        else:
            self.toggle_button.setText(f"▶ {self.toggle_button.text()[2:]}")
            self.container.hide()

    def append_log(self, text, success=True):
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = "[OK] " if success else "[ERR] "
        lines = text.splitlines()
        for line in lines:
            if line.strip():
                self.log_text.append(f"<span style='color: #888888;'>[{ts}]</span> <span style='color: {'#4caf50' if success else '#f44336'};'>{prefix}</span> {line}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def submit_command(self):
        command = self.command_input.text().strip()
        if command:
            self.command_submitted.emit(command)
            self.command_input.clear()


class FEDTabWidget(QWidget):
    """Modular widget for the FED processing tab."""
    scan_finished_signal = pyqtSignal(list, list)
    
    def __init__(self, parent=None, worker_class=None):
        super().__init__(parent)
        self.main_window = parent
        self.WorkerThread = worker_class # Pass the WorkerThread class from main
        self._active_workers = []
        self.fed_devices = []
        self.removed_ports = set() # Track removed ports to avoid auto-adding them back
        self._port_to_id = {} # Track discovered on-board device IDs
        self.scan_finished_signal.connect(self.handle_scan_finished)
        
        # Setup the default directory for FED3 logs
        self.default_log_dir = os.path.expanduser("~/Documents/FED3_Logs")
        if not os.path.exists(self.default_log_dir):
            try:
                os.makedirs(self.default_log_dir)
            except Exception:
                self.default_log_dir = os.path.expanduser("~/FED3_Logs")
                os.makedirs(self.default_log_dir, exist_ok=True)
                
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self._scroll_content = QWidget()
        self.layout = QVBoxLayout(self._scroll_content)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        self._scroll.setWidget(self._scroll_content)
        main_layout.addWidget(self._scroll)

        # --- Section 1: Plot View ---
        self.plot_tab = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_tab)
        self.plot_layout.setContentsMargins(10, 10, 10, 10)

        self.plot_filter_combo = QComboBox()
        self.plot_filter_combo.addItem("All Devices", userData=None)
        self.plot_filter_combo.currentTextChanged.connect(lambda: self.update_plot())
        self.plot_layout.addWidget(self.plot_filter_combo)

        # Placeholder shown when no data has been collected
        self.plot_placeholder = QLabel(
            "No pellet data collected yet.\n"
            "The activity graph will appear here once pellet data is received."
        )
        self.plot_placeholder.setAlignment(Qt.AlignCenter)
        self.plot_placeholder.setFont(QFont("Arial", 11))
        self.plot_placeholder.setStyleSheet(
            "color: #888888; padding: 40px; "
            "background-color: #1e1e1e; border: 1px dashed #444444; border-radius: 6px;"
        )
        self.plot_layout.addWidget(self.plot_placeholder)

        # Matplotlib visualization
        self.figure = Figure(figsize=(5, 3))
        self.figure.patch.set_facecolor('#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.set_title("Pellets retrieved")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Cumulative Pellets")
        self.figure.tight_layout(pad=1.5)
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#444444')
        self.plot_layout.addWidget(self.canvas)
        self.canvas.setVisible(False)
        self.plot_layout.addStretch()

        # --- Section 2: FED3 View ---
        self.fed_view_tab = QWidget()
        self.fed_view_layout = FlowLayout(margin=10, spacing=20)
        self.fed_view_tab.setLayout(self.fed_view_layout)

        # --- Columns Layout for Controls and Scheduler ---
        columns_widget = QWidget()
        columns_layout = QGridLayout(columns_widget)
        columns_layout.setContentsMargins(10, 10, 10, 10)
        columns_layout.setSpacing(15)

        # FED Control Panel (Group Box)
        self.control_group = QGroupBox("FED Control Panel")
        control_group_layout = QHBoxLayout(self.control_group)
        control_group_layout.setSpacing(15)
        control_group_layout.setContentsMargins(15, 15, 15, 12)
 
        # --- LEFT COLUMN: Device Modes & Overrides ---
        left_layout = QVBoxLayout()
        left_layout.setSpacing(10)
        
        left_header = QLabel("Device Configuration & Commands")
        left_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #888888; text-transform: uppercase; padding-bottom: 2px;")
        left_layout.addWidget(left_header)
        
        # Mode selector row
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Global Mode:"))
        self.global_mode_combo = QComboBox()
        self.global_mode_combo.addItems([
            "Fixed Ratio (FR)", 
            "Progressive Ratio (PR)", 
            "Random Ratio (RR)", 
            "FR with Timeout", 
            "Free Feeding",
            "Extinction",
            "Light Tracking",
            "FR Reversed",
            "PR Reversed",
            "Opto Stimulation",
            "Opto Reversed",
            "Timed Feeding"
        ])
        mode_row.addWidget(self.global_mode_combo)
        mode_row.addStretch()
        left_layout.addLayout(mode_row)
        
        # Mode Parameters row
        param_row = QHBoxLayout()
        self.global_fr_label = QLabel("Ratio:")
        self.global_ratio_spin = QSpinBox()
        self.global_ratio_spin.setRange(1, 999)
        self.global_ratio_spin.setValue(1)
        self.global_ratio_spin.setFixedWidth(60)
 
        self.global_timeout_label = QLabel("Timeout:")
        self.global_timeout_spin = QSpinBox()
        self.global_timeout_spin.setRange(0, 9999)
        self.global_timeout_spin.setValue(30)
        self.global_timeout_spin.setFixedWidth(60)
        self.global_timeout_unit_label = QLabel("s")
        
        param_row.addWidget(self.global_fr_label)
        param_row.addWidget(self.global_ratio_spin)
        param_row.addWidget(self.global_timeout_label)
        param_row.addWidget(self.global_timeout_spin)
        param_row.addWidget(self.global_timeout_unit_label)
        param_row.addStretch()
        left_layout.addLayout(param_row)
        
        # Actions row (Apply, Dispense, Lights)
        action_row = QHBoxLayout()
        self.global_apply_btn = QPushButton("Apply Mode")
        self.global_apply_btn.setToolTip("Apply selected mode and settings to all devices")
        self.global_apply_btn.setStyleSheet("font-weight: bold; min-height: 22px;")
 
        self.global_dispense_btn = QPushButton("Dispense All")
        self.global_dispense_btn.setToolTip("Manually dispense a pellet on all devices")
        self.global_dispense_btn.setStyleSheet("font-weight: bold; min-height: 22px;")
 
        self.global_lights_toggle_btn = QPushButton("Lights: OFF")
        self.global_lights_toggle_btn.setCheckable(True)
        self.global_lights_toggle_btn.setToolTip("Manually toggle all device LEDs ON/OFF globally")
        self.global_lights_toggle_btn.setStyleSheet("""
            QPushButton:checked { background-color: #f1c40f; color: black; }
            QPushButton { min-height: 22px; font-weight: bold; }
        """)
        
        action_row.addWidget(self.global_apply_btn)
        action_row.addWidget(self.global_dispense_btn)
        action_row.addWidget(self.global_lights_toggle_btn)
        action_row.addStretch()
        left_layout.addLayout(action_row)
        left_layout.addStretch()
 
        # --- MIDDLE COLUMN: Vertical Line ---
        v_line = QFrame()
        v_line.setFrameShape(QFrame.VLine)
        v_line.setFrameShadow(QFrame.Sunken)
        v_line.setStyleSheet("background-color: #333333; margin: 0px 5px;")
 
        # --- RIGHT COLUMN: Sync & Session Logging ---
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)
        
        right_header = QLabel("Sync & Session Data Logging")
        right_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #888888; text-transform: uppercase; padding-bottom: 2px;")
        right_layout.addWidget(right_header)
        
        # Sync interval settings row
        sync_interval_row = QHBoxLayout()
        sync_interval_row.addWidget(QLabel("Auto Sync Every:"))
        self.global_interval_spin = QSpinBox()
        self.global_interval_spin.setRange(1, 99999)
        self.global_interval_spin.setValue(1)
        self.global_interval_spin.setFixedWidth(80)
        self.global_unit_combo = QComboBox()
        self.global_unit_combo.addItems(["Seconds", "Minutes", "Hours", "Days"])
        self.global_unit_combo.setCurrentText("Days")
        self.global_unit_combo.setFixedWidth(100)
        
        sync_interval_row.addWidget(self.global_interval_spin)
        sync_interval_row.addWidget(self.global_unit_combo)
        sync_interval_row.addStretch()
        right_layout.addLayout(sync_interval_row)
        
        # Sync action row
        sync_action_row = QHBoxLayout()
        self.start_all_btn = QPushButton("Start Auto Sync")
        self.start_all_btn.setCheckable(True)
        self.start_all_btn.setStyleSheet("""
            QPushButton:checked { background-color: #4caf50; color: white; }
            QPushButton { font-weight: bold; }
        """)
        self.sync_now_btn = QPushButton("Sync Now")
        self.sync_now_btn.setStyleSheet("font-weight: bold;")
        
        sync_action_row.addWidget(self.start_all_btn)
        sync_action_row.addWidget(self.sync_now_btn)
        sync_action_row.addStretch()
        right_layout.addLayout(sync_action_row)
        
        # Data export/reset row
        data_row = QHBoxLayout()
        self.reset_all_btn = QPushButton("Reset All Counters")
        self.reset_all_btn.setStyleSheet("font-weight: bold; background-color: #c0392b; color: white;")
        self.reset_all_btn.clicked.connect(self.reset_all_counters)
        
        self.export_all_btn = QPushButton("Export All Logs...")
        self.export_all_btn.setStyleSheet("font-weight: bold;")
        self.export_all_btn.clicked.connect(self.export_all_logs)
        
        data_row.addWidget(self.export_all_btn)
        data_row.addWidget(self.reset_all_btn)
        data_row.addStretch()
        right_layout.addLayout(data_row)
        right_layout.addStretch()
 
        # Add columns to main Control Panel layout
        control_group_layout.addLayout(left_layout, stretch=1)
        control_group_layout.addWidget(v_line)
        control_group_layout.addLayout(right_layout, stretch=1)
        self.control_group.setLayout(control_group_layout)

        # Connected Devices (Group Box)
        self.devices_group = QGroupBox("Connected Devices")
        devices_group_layout = QVBoxLayout()
        devices_group_layout.setSpacing(8)
        devices_group_layout.setContentsMargins(10, 15, 10, 10)

        # Management layout (Add Device & Refresh Ports buttons)
        mgmt_layout = QHBoxLayout()
        self.add_device_btn = QPushButton("Add Device")
        self.refresh_ports_btn = QPushButton("Refresh Ports")
        mgmt_layout.addWidget(self.add_device_btn)
        mgmt_layout.addStretch()
        mgmt_layout.addWidget(self.refresh_ports_btn)
        devices_group_layout.addLayout(mgmt_layout)

        # Devices list
        self.devices_container = QWidget()
        self.devices_layout = FlowLayout(margin=4, spacing=8)
        self.devices_container.setLayout(self.devices_layout)
        devices_group_layout.addWidget(self.devices_container)
        self.devices_group.setLayout(devices_group_layout)

        # Store columns_layout as self.columns_layout
        self.columns_layout = columns_layout

        # Initialize the scheduler group box
        self.init_scheduler()

        # Add the columns widget to the main layout (components added dynamically via adjust_responsive_layout)
        self.layout.addWidget(columns_widget)
        
        # Add the unifying sections sequentially
        self.layout.addWidget(self.plot_tab)
        
        self.plot_divider = QFrame()
        self.plot_divider.setFrameShape(QFrame.HLine)
        self.plot_divider.setFrameShadow(QFrame.Sunken)
        self.plot_divider.setStyleSheet("background-color: #444444; margin: 10px 0px;")
        self.layout.addWidget(self.plot_divider)
        
        self.layout.addWidget(self.fed_view_tab)
        self.layout.addStretch()

        # Log box at bottom
        self.fed_log = CollapsibleLogBox("Serial Monitor")
        self.layout.addWidget(self.fed_log)

        # Connections
        self.add_device_btn.clicked.connect(self.create_device_widget)
        self.refresh_ports_btn.clicked.connect(self.refresh_all_ports)
        self.start_all_btn.toggled.connect(self.toggle_auto_sync)
        self.sync_now_btn.clicked.connect(self.sync_all)
        self.fed_log.command_submitted.connect(self.handle_fed_command)
        self.global_mode_combo.currentTextChanged.connect(self.update_global_mode_ui)
        self.global_apply_btn.clicked.connect(self.apply_global_mode)
        self.global_dispense_btn.clicked.connect(self.dispense_all)
        self.global_lights_toggle_btn.toggled.connect(self.toggle_global_lights)

        # Initial global mode UI state
        self.update_global_mode_ui()
        self.update_control_panels_enabled_state()
        self.adjust_responsive_layout()
        
        # Start Auto Sync by default
        self.start_all_btn.setChecked(True)
 
        # Initial port scan on startup (auto-creates widgets for active devices when finished)
        self.refresh_all_ports()


    def get_global_interval_ms(self):
        unit = self.global_unit_combo.currentText()
        value = self.global_interval_spin.value()
        mult = 1
        if unit == 'Seconds': mult = 1
        elif unit == 'Minutes': mult = 60
        elif unit == 'Hours': mult = 3600
        elif unit == 'Days': mult = 86400
        seconds = max(1, value * mult)
        return int(seconds * 1000)

    def update_global_mode_ui(self):
        mode = self.global_mode_combo.currentText()
        is_fr = (mode == "Fixed Ratio (FR)")
        is_rr = (mode == "Random Ratio (RR)")
        is_timeout = (mode == "FR with Timeout")
        is_fr_rev = (mode == "FR Reversed")

        self.global_fr_label.setVisible(is_fr or is_rr or is_timeout or is_fr_rev)
        self.global_ratio_spin.setVisible(is_fr or is_rr or is_timeout or is_fr_rev)

        self.global_timeout_label.setVisible(is_timeout)
        self.global_timeout_spin.setVisible(is_timeout)
        self.global_timeout_unit_label.setVisible(is_timeout)

    def apply_global_mode(self):
        if not self.confirm_action_if_tracking("Are you sure you want to apply mode settings to all devices?"):
            return
        mode = self.global_mode_combo.currentText()
        ratio = self.global_ratio_spin.value()
        timeout = self.global_timeout_spin.value()

        for device in self.fed_devices:
            device['mode_combo'].setCurrentText(mode)
            device['ratio_spin'].setValue(ratio)
            device['timeout_spin'].setValue(timeout)
            self.apply_device_mode(device, confirm=False)

    def on_sched_time_type_changed(self, idx):
        self.sched_time_edit.setText("00d 00:00:00")

    def init_scheduler(self):
        self.scheduler_group = QGroupBox("Protocol Event Scheduler")
        layout = QVBoxLayout(self.scheduler_group)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 15, 10, 10)


        # 2. Table widget
        self.scheduler_table = QTableWidget(0, 5)
        self.scheduler_table.setHorizontalHeaderLabels(["Target Device", "Trigger / Countdown", "Command Details", "Status", "Actions"])
        self.scheduler_table.verticalHeader().setVisible(True)
        header = self.scheduler_table.horizontalHeader()
        for i in range(5):
            if i == 2:
                header.setSectionResizeMode(i, QHeaderView.Stretch)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.scheduler_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                gridline-color: #333333;
                border: 1px solid #444444;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #2b2b2b;
                color: #ffffff;
                padding: 4px;
                border: 1px solid #333333;
                font-weight: bold;
            }
        """)
        self.scheduler_table.setMinimumHeight(150)
        self.scheduler_table.setMaximumHeight(250)
        layout.addWidget(self.scheduler_table)

        # 3. Add Event Panel
        add_panel = QGroupBox("Schedule New Event")
        add_layout = QGridLayout(add_panel)
        add_layout.setSpacing(8)

        # Target Selector
        add_layout.addWidget(QLabel("Target FED:"), 0, 0)
        self.sched_target_combo = QComboBox()
        self.sched_target_combo.addItem("All Devices")
        add_layout.addWidget(self.sched_target_combo, 0, 1)

        # Trigger Input
        add_layout.addWidget(QLabel("Trigger Time:"), 0, 2)
        
        time_input_container = QWidget()
        time_input_layout = QHBoxLayout(time_input_container)
        time_input_layout.setContentsMargins(0, 0, 0, 0)
        time_input_layout.setSpacing(6)
        
        self.sched_time_type_combo = QComboBox()
        self.sched_time_type_combo.addItems(["Alarm", "Timer"])
        
        self.sched_time_edit = QLineEdit()
        self.sched_time_edit.setText("00d 00:00:00")
        self.sched_time_edit.setFixedWidth(90)
        self.sched_time_edit.setAlignment(Qt.AlignCenter)
        self.sched_time_edit.setToolTip("Type digits to format time (e.g. 4123015 -> 04d 12:30:15)")
        
        def on_text_edited(text):
            digits = "".join([c for c in text if c.isdigit()])
            if not digits:
                digits = "0"
            digits = digits[-8:]
            digits = digits.zfill(8)
            formatted = f"{digits[0:2]}d {digits[2:4]}:{digits[4:6]}:{digits[6:8]}"
            
            self.sched_time_edit.blockSignals(True)
            self.sched_time_edit.setText(formatted)
            self.sched_time_edit.setCursorPosition(len(formatted))
            self.sched_time_edit.blockSignals(False)
            
        self.sched_time_edit.textEdited.connect(on_text_edited)
        
        time_input_layout.addWidget(self.sched_time_type_combo)
        time_input_layout.addWidget(self.sched_time_edit)
        time_input_layout.addStretch()
        
        add_layout.addWidget(time_input_container, 0, 3)

        # Action Selector
        add_layout.addWidget(QLabel("Action:"), 1, 0)
        self.sched_action_combo = QComboBox()
        self.sched_action_combo.addItems(["Set Mode", "Dispense Pellet", "Toggle Lights"])
        self.sched_action_combo.currentTextChanged.connect(self.update_sched_action_parameters_visibility)
        add_layout.addWidget(self.sched_action_combo, 1, 1)

        # Parameters panel
        self.sched_params_container = QWidget()
        self.sched_params_layout = QHBoxLayout(self.sched_params_container)
        self.sched_params_layout.setContentsMargins(0, 0, 0, 0)
        self.sched_params_layout.setSpacing(6)

        # Parameter: Mode
        self.sched_mode_combo = QComboBox()
        self.sched_mode_combo.addItems([
            "Fixed Ratio (FR)", 
            "Progressive Ratio (PR)", 
            "Random Ratio (RR)", 
            "FR with Timeout", 
            "Free Feeding",
            "Extinction",
            "Light Tracking",
            "FR Reversed",
            "PR Reversed",
            "Opto Stimulation",
            "Opto Reversed",
            "Timed Feeding"
        ])
        self.sched_mode_combo.currentTextChanged.connect(self.update_sched_mode_parameters_visibility)
        
        self.sched_ratio_label = QLabel("Ratio:")
        self.sched_ratio_spin = QSpinBox()
        self.sched_ratio_spin.setRange(1, 999)
        self.sched_ratio_spin.setValue(1)
        self.sched_ratio_spin.setFixedWidth(60)

        self.sched_timeout_label = QLabel("Timeout:")
        self.sched_timeout_spin = QSpinBox()
        self.sched_timeout_spin.setRange(0, 9999)
        self.sched_timeout_spin.setValue(30)
        self.sched_timeout_spin.setFixedWidth(60)
        self.sched_timeout_unit_label = QLabel("s")

        # Parameter: Lights
        self.sched_lights_combo = QComboBox()
        self.sched_lights_combo.addItems(["Lights ON", "Lights OFF"])

        self.sched_params_layout.addWidget(self.sched_mode_combo)
        self.sched_params_layout.addWidget(self.sched_ratio_label)
        self.sched_params_layout.addWidget(self.sched_ratio_spin)
        self.sched_params_layout.addWidget(self.sched_timeout_label)
        self.sched_params_layout.addWidget(self.sched_timeout_spin)
        self.sched_params_layout.addWidget(self.sched_timeout_unit_label)
        self.sched_params_layout.addWidget(self.sched_lights_combo)
        self.sched_params_layout.addStretch()

        add_layout.addWidget(QLabel("Parameters:"), 1, 2)
        add_layout.addWidget(self.sched_params_container, 1, 3)

        self.add_sched_event_btn = QPushButton("Add Event")
        self.add_sched_event_btn.setStyleSheet("font-weight: bold; min-height: 24px;")
        self.add_sched_event_btn.clicked.connect(self.add_sched_event)
        add_layout.addWidget(self.add_sched_event_btn, 2, 3, 1, 1, Qt.AlignRight)

        layout.addWidget(add_panel)

        self.scheduled_events = []
        self.scheduler_timer = QTimer(self)
        self.scheduler_timer.timeout.connect(self.tick_scheduler)
        self.scheduler_timer.start(1000)
        self.scheduler_start_time = None

        self.sched_time_type_combo.currentIndexChanged.connect(self.on_sched_time_type_changed)
        self.on_sched_time_type_changed(0)

        self.update_sched_action_parameters_visibility()

    def update_sched_action_parameters_visibility(self):
        action = self.sched_action_combo.currentText()
        is_mode = (action == "Set Mode")
        is_lights = (action == "Toggle Lights")

        self.sched_mode_combo.setVisible(is_mode)
        self.sched_ratio_label.setVisible(is_mode)
        self.sched_ratio_spin.setVisible(is_mode)
        self.sched_timeout_label.setVisible(is_mode)
        self.sched_timeout_spin.setVisible(is_mode)
        self.sched_timeout_unit_label.setVisible(is_mode)
        self.sched_lights_combo.setVisible(is_lights)

        if is_mode:
            self.update_sched_mode_parameters_visibility()

    def update_sched_mode_parameters_visibility(self):
        if not self.sched_mode_combo.isVisible():
            return
        mode = self.sched_mode_combo.currentText()
        is_fr = (mode == "Fixed Ratio (FR)")
        is_rr = (mode == "Random Ratio (RR)")
        is_timeout = (mode == "FR with Timeout")
        is_fr_rev = (mode == "FR Reversed")

        self.sched_ratio_label.setVisible(is_fr or is_rr or is_timeout or is_fr_rev)
        self.sched_ratio_spin.setVisible(is_fr or is_rr or is_timeout or is_fr_rev)
        self.sched_timeout_label.setVisible(is_timeout)
        self.sched_timeout_spin.setVisible(is_timeout)
        self.sched_timeout_unit_label.setVisible(is_timeout)

    def add_sched_event(self):
        target_name = self.sched_target_combo.currentText()
        action = self.sched_action_combo.currentText()
        time_text = self.sched_time_edit.text().strip()
        is_relative = (self.sched_time_type_combo.currentIndex() == 1)

        parts_d = time_text.split("d ")
        try:
            days_val = int(parts_d[0]) if len(parts_d) > 0 else 0
            rest = parts_d[1] if len(parts_d) > 1 else "00:00:00"
        except ValueError:
            days_val = 0
            rest = time_text

        parts = rest.split(":")
        try:
            h = int(parts[0]) if len(parts) > 0 else 0
            m = int(parts[1]) if len(parts) > 1 else 0
            s = int(parts[2]) if len(parts) > 2 else 0
        except ValueError:
            h, m, s = 0, 0, 0

        if is_relative and days_val == 0 and h == 0 and m == 0 and s == 0:
            QMessageBox.warning(self, "Validation Error", "Please specify a timer delay greater than 0 days and 00:00:00.")
            return

        params = {}
        param_desc = ""
        if action == "Set Mode":
            mode = self.sched_mode_combo.currentText()
            ratio = self.sched_ratio_spin.value()
            timeout = self.sched_timeout_spin.value()
            params = {'mode': mode, 'ratio': ratio, 'timeout': timeout}
            
            if mode == "Fixed Ratio (FR)":
                param_desc = f"{mode} (R:{ratio})"
            elif mode == "Random Ratio (RR)":
                param_desc = f"{mode} (Avg R:{ratio})"
            elif mode == "FR with Timeout":
                param_desc = f"{mode} (R:{ratio}, TO:{timeout}s)"
            elif mode == "FR Reversed":
                param_desc = f"{mode} (R:{ratio})"
            else:
                param_desc = mode
        elif action == "Toggle Lights":
            lights_state = self.sched_lights_combo.currentText()
            params = {'lights': (lights_state == "Lights ON")}
            param_desc = lights_state
        else:
            param_desc = "None"

        total_offset_sec = (days_val * 86400 + h * 3600 + m * 60 + s) if is_relative else None

        event = {
            'target_name': target_name,
            'trigger_type': 'Relative' if is_relative else 'Absolute',
            'trigger_val': f"+{time_text}" if is_relative else (time_text if days_val > 0 else rest),
            'relative_offset_sec': total_offset_sec,
            'days_offset': days_val,
            'time_text': time_text,
            'action': action,
            'params': params,
            'param_desc': param_desc,
            'status': 'Pending',
            'target_time': None
        }

        now = datetime.now()
        if event['trigger_type'] == 'Relative':
            event['target_time'] = now + timedelta(seconds=event['relative_offset_sec'])
        else:
            target_tod = time(h, m, s)
            if days_val > 0:
                event['target_time'] = datetime.combine(now.date() + timedelta(days=days_val), target_tod)
            else:
                target_dt = datetime.combine(now.date(), target_tod)
                if target_dt < now:
                    target_dt += timedelta(days=1)
                event['target_time'] = target_dt

        self.scheduled_events.append(event)
        self.update_scheduler_table()
        # Reset input values
        self.sched_time_edit.setText("00d 00:00:00")

    def update_scheduler_table(self):
        # Sort events by target_time before display
        self.scheduled_events.sort(key=lambda x: x['target_time'] if x['target_time'] is not None else datetime.max)

        self.scheduler_table.setRowCount(0)
        now = datetime.now()
        vertical_labels = []

        for idx, event in enumerate(self.scheduled_events):
            row = self.scheduler_table.rowCount()
            self.scheduler_table.insertRow(row)

            # Set vertical header label (Alarm time or estimated target time for Timer)
            if event['target_time']:
                vertical_labels.append(event['target_time'].strftime('%m-%d %H:%M:%S'))
            else:
                vertical_labels.append("Pending")

            # Column 0: Target Device
            self.scheduler_table.setItem(row, 0, QTableWidgetItem(event['target_name']))

            # Column 1: Trigger / Countdown
            if event['trigger_type'] == 'Relative':
                if event['status'] == 'Pending' and event['target_time']:
                    remaining_sec = int((event['target_time'] - now).total_seconds())
                    if remaining_sec < 0:
                        remaining_sec = 0
                    d_rem = remaining_sec // 86400
                    h_rem = (remaining_sec % 86400) // 3600
                    m_rem = (remaining_sec % 3600) // 60
                    s_rem = remaining_sec % 60
                    if d_rem > 0:
                        trigger_display = f"Timer ({d_rem}d {h_rem:02d}:{m_rem:02d}:{s_rem:02d})"
                    else:
                        trigger_display = f"Timer ({h_rem:02d}:{m_rem:02d}:{s_rem:02d})"
                else:
                    trigger_display = "Timer"
            else:
                trigger_display = "Alarm"
            self.scheduler_table.setItem(row, 1, QTableWidgetItem(trigger_display))

            # Column 2: Command Details
            action = event['action']
            param_desc = event['param_desc']
            if action == "Set Mode":
                details_display = f"Set Mode to {param_desc}"
            elif action == "Toggle Lights":
                details_display = f"Toggle Lights ({param_desc})"
            else:
                details_display = action
            self.scheduler_table.setItem(row, 2, QTableWidgetItem(details_display))

            # Column 3: Status
            status_item = QTableWidgetItem(event['status'])
            if event['status'] == 'Executed':
                status_item.setForeground(QBrush(QColor("#2ecc71")))
            elif event['status'] == 'Failed':
                status_item.setForeground(QBrush(QColor("#e74c3c")))
            elif event['status'] == 'Running':
                status_item.setForeground(QBrush(QColor("#f1c40f")))
            else:
                status_item.setForeground(QBrush(QColor("#888888")))
            self.scheduler_table.setItem(row, 3, status_item)

            # Column 4: Actions
            remove_btn = QPushButton("Delete")
            remove_btn.setStyleSheet("background-color: #c0392b; color: white; max-height: 20px; font-size: 10px;")
            remove_btn.clicked.connect(lambda _, r_idx=idx: self.remove_scheduler_event(r_idx))
            self.scheduler_table.setCellWidget(row, 4, remove_btn)

        self.scheduler_table.setVerticalHeaderLabels(vertical_labels)

    def remove_scheduler_event(self, idx):
        if idx < len(self.scheduled_events):
            self.scheduled_events.pop(idx)
            self.update_scheduler_table()

    def tick_scheduler(self):
        now = datetime.now()
        updated = False

        for event in self.scheduled_events:
            if event['status'] == 'Pending' and event['target_time'] and now >= event['target_time']:
                event['status'] = 'Running'
                self.update_scheduler_table()
                self.execute_scheduled_event(event)
                updated = True

        # Always update the table to refresh countdowns if there are pending timers
        has_pending_timers = any(e['status'] == 'Pending' and e['trigger_type'] == 'Relative' for e in self.scheduled_events)
        if has_pending_timers or updated:
            self.update_scheduler_table()

    def execute_scheduled_event(self, event):
        target_name = event['target_name']
        action = event['action']
        params = event['params']

        devices_to_trigger = []
        if target_name == "All Devices":
            devices_to_trigger = list(self.fed_devices)
        else:
            for dev in self.fed_devices:
                dev_name = dev['name_edit'].text().strip() or dev['box'].title()
                if dev_name == target_name:
                    devices_to_trigger.append(dev)
                    break

        if not devices_to_trigger:
            event['status'] = 'Failed'
            self.fed_log.append_log(f"Scheduler failed: Target device '{target_name}' not found.", False)
            return

        success_count = 0
        for dev in devices_to_trigger:
            try:
                if action == "Set Mode":
                    dev['mode_combo'].setCurrentText(params['mode'])
                    dev['ratio_spin'].setValue(params['ratio'])
                    dev['timeout_spin'].setValue(params['timeout'])
                    self.apply_device_mode(dev, confirm=False)
                    success_count += 1
                elif action == "Dispense Pellet":
                    self.send_command_to_device(dev, "FEED", "Dispense Pellet", confirm=False)
                    success_count += 1
                elif action == "Toggle Lights":
                    btn = dev['lights_toggle_btn']
                    btn.blockSignals(True)
                    btn.setChecked(params['lights'])
                    if params['lights']:
                        btn.setText("Lights: ON")
                    else:
                        btn.setText("Lights: OFF")
                    btn.blockSignals(False)
                    cmd = "LIGHTS:ON" if params['lights'] else "LIGHTS:OFF"
                    self.send_command_to_device(dev, cmd, "Toggle Lights", confirm=False)
                    success_count += 1
            except Exception as e:
                self.fed_log.append_log(f"Scheduler error executing on device: {e}", False)

        if success_count > 0:
            event['status'] = 'Executed'
        else:
            event['status'] = 'Failed'

    def dispense_all(self):
        if not self.confirm_action_if_tracking("Are you sure you want to dispense a pellet on all devices?"):
            return
        for device in self.fed_devices:
            self.send_command_to_device(device, "FEED", "Dispense Pellet", confirm=False)

    def toggle_global_lights(self, checked):
        if not self.confirm_action_if_tracking("Are you sure you want to toggle lights on all devices?"):
            btn = self.global_lights_toggle_btn
            btn.blockSignals(True)
            btn.setChecked(not checked)
            btn.blockSignals(False)
            if checked:
                btn.setText("Lights: OFF")
            else:
                btn.setText("Lights: ON")
            return

        btn = self.global_lights_toggle_btn
        if checked:
            btn.setText("Lights: ON")
        else:
            btn.setText("Lights: OFF")
            
        for device in self.fed_devices:
            btn_dev = device['lights_toggle_btn']
            btn_dev.blockSignals(True)
            btn_dev.setChecked(checked)
            if checked:
                btn_dev.setText("Lights: ON")
            else:
                btn_dev.setText("Lights: OFF")
            btn_dev.blockSignals(False)
            cmd = "LIGHTS:ON" if checked else "LIGHTS:OFF"
            self.send_command_to_device(device, cmd, "Turn Lights ON" if checked else "Turn Lights OFF", confirm=False)

    def confirm_action_if_tracking(self, message="Are you sure you want to perform this action?"):
        any_tracking = any(dev.get('is_tracking', False) for dev in self.fed_devices)
        if not any_tracking:
            return True
            
        reply = QMessageBox.question(
            self,
            "Confirm Action",
            message + "\n\nWarning: Tracking is currently active. This may interrupt the ongoing experiment.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        return reply == QMessageBox.Yes

    def start_scanning_animation(self):
        from PyQt5.QtWidgets import QGraphicsOpacityEffect
        from PyQt5.QtCore import QPropertyAnimation
        
        # Ensure effects are set up
        if not hasattr(self, '_control_opacity_effect'):
            self._control_opacity_effect = QGraphicsOpacityEffect(self.control_group)
            self.control_group.setGraphicsEffect(self._control_opacity_effect)
        if not hasattr(self, '_sched_opacity_effect'):
            self._sched_opacity_effect = QGraphicsOpacityEffect(self.scheduler_group)
            self.scheduler_group.setGraphicsEffect(self._sched_opacity_effect)
            
        # Enable the effects
        self._control_opacity_effect.setEnabled(True)
        self._sched_opacity_effect.setEnabled(True)
        
        # Set up control group animation
        if not hasattr(self, '_control_anim'):
            self._control_anim = QPropertyAnimation(self._control_opacity_effect, b"opacity")
            self._control_anim.setDuration(1500)
            self._control_anim.setLoopCount(-1)
            self._control_anim.setKeyValueAt(0.0, 0.4)
            self._control_anim.setKeyValueAt(0.5, 0.8)
            self._control_anim.setKeyValueAt(1.0, 0.4)
            
        # Set up scheduler group animation
        if not hasattr(self, '_sched_anim'):
            self._sched_anim = QPropertyAnimation(self._sched_opacity_effect, b"opacity")
            self._sched_anim.setDuration(1500)
            self._sched_anim.setLoopCount(-1)
            self._sched_anim.setKeyValueAt(0.0, 0.4)
            self._sched_anim.setKeyValueAt(0.5, 0.8)
            self._sched_anim.setKeyValueAt(1.0, 0.4)
            
        self._control_anim.start()
        self._sched_anim.start()

    def stop_scanning_animation(self):
        if hasattr(self, '_control_anim'):
            self._control_anim.stop()
        if hasattr(self, '_sched_anim'):
            self._sched_anim.stop()
            
        # Disable the effects to restore full opacity (1.0) and prevent rendering performance impact
        if hasattr(self, '_control_opacity_effect'):
            self._control_opacity_effect.setEnabled(False)
        if hasattr(self, '_sched_opacity_effect'):
            self._sched_opacity_effect.setEnabled(False)

    def update_control_panels_enabled_state(self):
        is_scanning = hasattr(self, '_global_scanner') and self._global_scanner is not None and self._global_scanner.isRunning()
        enabled = not is_scanning
        
        self.control_group.setEnabled(enabled)
        self.scheduler_group.setEnabled(enabled)
        
        if is_scanning:
            self.start_scanning_animation()
        else:
            self.stop_scanning_animation()

    def create_device_widget(self, checked=False, refresh=True):
        idx = len(self.fed_devices) + 1
        box = QGroupBox(f"Device {idx}")
        box_layout = QGridLayout()

        # Define device fields
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("Optional device name")
        remove_btn = QPushButton("Remove")
        port_combo = QComboBox()
        port_combo.setEditable(True)

        # Feeding mode controls
        mode_combo = QComboBox()
        mode_combo.addItems([
            "Fixed Ratio (FR)", 
            "Progressive Ratio (PR)", 
            "Random Ratio (RR)", 
            "FR with Timeout", 
            "Free Feeding",
            "Extinction",
            "Light Tracking",
            "FR Reversed",
            "PR Reversed",
            "Opto Stimulation",
            "Opto Reversed",
            "Timed Feeding"
        ])
        mode_combo.setToolTip("Select the feeding program structure")
        apply_btn = QPushButton("Apply Mode")
        apply_btn.setToolTip("Apply selected mode and settings to device")

        # Dynamic mode parameter widgets
        fr_label = QLabel("Ratio:")
        ratio_spin = QSpinBox()
        ratio_spin.setRange(1, 999)
        ratio_spin.setValue(1)
        ratio_spin.setToolTip("Set feed ratio (number of pokes required per pellet / RR average)")

        timeout_label = QLabel("Timeout:")
        timeout_spin = QSpinBox()
        timeout_spin.setRange(0, 9999)
        timeout_spin.setValue(30)
        timeout_spin.setToolTip("Lockout timeout in seconds after pellet delivery")
        timeout_unit_label = QLabel("s")

        params_container = QWidget()
        params_layout = QHBoxLayout(params_container)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(6)
        params_layout.addWidget(fr_label)
        params_layout.addWidget(ratio_spin)
        params_layout.addWidget(timeout_label)
        params_layout.addWidget(timeout_spin)
        params_layout.addWidget(timeout_unit_label)
        params_layout.addStretch()

        last_sync_label = QLabel("Last Sync: Never")

        # Manual overrides container
        manual_container = QWidget()
        manual_layout = QHBoxLayout(manual_container)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(6)

        feed_btn = QPushButton("Dispense")
        feed_btn.setToolTip("Manually dispense a pellet")
        feed_btn.setStyleSheet("font-weight: bold; min-height: 22px;")

        lights_toggle_btn = QPushButton("Lights: OFF")
        lights_toggle_btn.setCheckable(True)
        lights_toggle_btn.setToolTip("Manually toggle all device LEDs ON/OFF")
        lights_toggle_btn.setStyleSheet("""
            QPushButton:checked { background-color: #f1c40f; color: black; }
            QPushButton { min-height: 22px; font-weight: bold; }
        """)

        manual_layout.addWidget(QLabel("Manual:"))
        manual_layout.addWidget(feed_btn)
        manual_layout.addWidget(lights_toggle_btn)
        manual_layout.addStretch()

        # Individual Actions container (Reset / Export)
        actions_container = QWidget()
        actions_layout = QHBoxLayout(actions_container)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(6)
        
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset counters and plot for this device")
        reset_btn.setStyleSheet("font-weight: bold; background-color: #c0392b; color: white; min-height: 22px;")
        
        export_btn = QPushButton("Export Log...")
        export_btn.setToolTip("Export CSV log for this device")
        export_btn.setStyleSheet("font-weight: bold; min-height: 22px;")
        
        actions_layout.addWidget(QLabel("Data:"))
        actions_layout.addWidget(reset_btn)
        actions_layout.addWidget(export_btn)
        actions_layout.addStretch()

        box_layout.addWidget(QLabel("Port:"), 0, 0)
        box_layout.addWidget(port_combo, 0, 1, 1, 2)
        box_layout.addWidget(remove_btn, 0, 3, 1, 1, Qt.AlignRight)

        box_layout.addWidget(QLabel("Mode:"), 1, 0)
        box_layout.addWidget(mode_combo, 1, 1, 1, 2)
        box_layout.addWidget(apply_btn, 1, 3, 1, 1)

        box_layout.addWidget(params_container, 2, 0, 1, 4)
        box_layout.addWidget(manual_container, 3, 0, 1, 4)
        box_layout.addWidget(actions_container, 4, 0, 1, 4)
        box_layout.addWidget(last_sync_label, 5, 0, 1, 4)
        box_layout.setColumnStretch(1, 1)
        box.setLayout(box_layout)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        timer = QTimer(self)
        
        svg_container = QWidget()
        svg_layout = QVBoxLayout(svg_container)
        svg_layout.setContentsMargins(0, 0, 0, 0)
        svg_title = QLabel(f"Device {idx}")
        svg_title.setAlignment(Qt.AlignCenter)
        svg_title.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")
        svg_view = FEDSvgView()
        svg_view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        svg_layout.addWidget(svg_title)
        svg_layout.addWidget(svg_view)
        
        self.fed_view_layout.addWidget(svg_container)

        device = {
            'box': box,
            'name_edit': name_edit,
            'port_combo': port_combo,
            'remove_btn': remove_btn,
            'mode_combo': mode_combo,
            'apply_btn': apply_btn,
            'fr_label': fr_label,
            'ratio_spin': ratio_spin,
            'timeout_label': timeout_label,
            'timeout_spin': timeout_spin,
            'timeout_unit_label': timeout_unit_label,
            'last_sync_label': last_sync_label,
            'feed_btn': feed_btn,
            'lights_toggle_btn': lights_toggle_btn,
            'reset_btn': reset_btn,
            'export_btn': export_btn,
            'timer': timer,
            'svg_container': svg_container,
            'svg_title': svg_title,
            'svg_view': svg_view,
            'is_syncing': False,
            'is_tracking': False,
            'tracker_worker': None,
            'log_file': None,
            'events': [],
            'stats': {'left': 0, 'right': 0, 'pellet': 0}
        }
        
        name_edit.textChanged.connect(
            lambda t, d=device: d['svg_title'].setText(
                t.strip() or f"Device {self.fed_devices.index(d) + 1 if d in self.fed_devices else len(self.fed_devices) + 1}"
            )
        )
        name_edit.textChanged.connect(lambda _: self.update_fed_view_combo())
        
        remove_btn.clicked.connect(lambda: self.remove_device(device))
        timer.timeout.connect(lambda: self.do_device_sync(device))
        apply_btn.clicked.connect(lambda _, d=device: self.apply_device_mode(d))
        mode_combo.currentTextChanged.connect(lambda _, d=device: self.update_mode_ui(d))
        
        feed_btn.clicked.connect(lambda _, d=device: self.send_command_to_device(d, "FEED", "Dispense Pellet"))
        lights_toggle_btn.toggled.connect(
            lambda checked, d=device: self.toggle_lights(d, checked)
        )
        reset_btn.clicked.connect(lambda _, d=device: self.reset_device_counters(d))
        export_btn.clicked.connect(lambda _, d=device: self.export_device_log(d))
        
        # Trigger port changed on item activation or manual edit finished, to avoid triggering on every keystroke
        port_combo.activated.connect(
            lambda _, d=device: self.on_port_changed(d, d['port_combo'].currentText())
        )
        port_combo.lineEdit().editingFinished.connect(
            lambda d=device: self.on_port_changed(d, d['port_combo'].currentText())
        )
        
        # Initialize feeding mode layout visibility
        self.update_mode_ui(device)
        
        port_combo.addItem("Scanning...")
        port_combo.setEnabled(False)
        self.devices_layout.addWidget(box)
        self.fed_devices.append(device)
        self.update_remove_buttons()
        self.update_fed_view_combo()
        self.update_control_panels_enabled_state()
        if refresh:
            self.refresh_all_ports()
        return device

    def start_device_acquisition(self, device, port):
        # Stop existing worker if running
        worker = device.get('tracker_worker')
        if worker:
            try:
                worker.stop()
            except Exception:
                pass
            device['tracker_worker'] = None
            
        dev_name = device['name_edit'].text().strip() or device['box'].title()
        
        # Initialize tracking stats and file path as soon as connection is opened
        device['is_tracking'] = True
        device['svg_view'].is_tracking = True
        device['events'] = []
        device['stats'] = {'left': 0, 'right': 0, 'pellet': 0}
        device['tracking_start_time'] = datetime.now()
        
        # Generate unique log file path in self.default_log_dir
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        clean_dev_name = "".join(c for c in dev_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        log_filename = f"{clean_dev_name}_{date_str}.csv"
        device['log_file'] = os.path.join(self.default_log_dir, log_filename)
        
        # Write CSV header
        try:
            with open(device['log_file'], 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Device", "EventType", "LeftPokes", "RightPokes", "Pellets", "RawData"])
        except Exception as e:
            self.fed_log.append_log(f"Error creating default log file for {dev_name}: {e}", False)
            
        new_worker = Fed3TrackerWorker(port, dev_name)
        new_worker.line_received.connect(lambda d, l: self.on_tracker_line(d, l, device))
        new_worker.error_received.connect(lambda d, l: self.fed_log.append_log(f"{d}: {l}", False))
        new_worker.finished.connect(lambda d=device: self.handle_tracker_finished(d))
        device['tracker_worker'] = new_worker
        
        device['port_combo'].setEnabled(False)
        new_worker.start()
        
        # Automatically start auto sync if checked
        if self.start_all_btn.isChecked():
            ms = self.get_global_interval_ms()
            device['timer'].start(ms)
            self.do_device_sync(device)
        
        self.update_fed_view_counts(device)
        self.update_plot()
        self.fed_log.append_log(f"Opened live visual monitoring on {port} for {dev_name}")

    def handle_tracker_finished(self, device):
        device['is_tracking'] = False
        device['svg_view'].is_tracking = False
        device['port_combo'].setEnabled(True)
        worker = device.get('tracker_worker')
        if worker:
            try:
                worker.stop()
            except Exception:
                pass
            device['tracker_worker'] = None
        self.update_fed_view_counts(device)
        self.update_plot()
        self.fed_log.append_log(f"Stopped live monitoring/worker for {device['box'].title()}")

    def get_current_plot_device(self):
        idx = self.plot_filter_combo.currentIndex()
        if idx > 0 and idx - 1 < len(self.fed_devices):
            return self.fed_devices[idx - 1]
        return None

    def on_tracker_line(self, dev_name, line, device):
        self.fed_log.append_log(f"[{dev_name}] {line}")
        
        event_type = "Other"
        updated_plot = False
        parts = [p.strip().upper() for p in line.split(',')]
        
        event_val = None
        for p in parts:
            if p in ("PELLET", "LEFT", "RIGHT", "LEFT POKE", "RIGHT POKE"):
                event_val = p
                break
                
        if not event_val:
            line_upper = line.upper()
            if "COUNT" not in line_upper:
                if "PELLET" in line_upper:
                    event_val = "PELLET"
                elif "LEFT POKE" in line_upper or "LEFT_POKE" in line_upper or "LEFT" in line_upper:
                    event_val = "LEFT"
                elif "RIGHT POKE" in line_upper or "RIGHT_POKE" in line_upper or "RIGHT" in line_upper:
                    event_val = "RIGHT"

        if event_val == "PELLET":
            event_type = "Pellet"
            device['svg_view'].flash_counter('pellet')
            if device.get('is_tracking'):
                device['events'].append(datetime.now())
                device['stats']['pellet'] += 1
                self.update_fed_view_counts(device)
                self.update_plot()
                updated_plot = True
            
        elif event_val in ("LEFT", "LEFT POKE", "LEFT_POKE"):
            event_type = "LeftPoke"
            device['svg_view'].flash_counter('left')
            if device.get('is_tracking'):
                device['stats']['left'] += 1
                self.update_fed_view_counts(device)
                
        elif event_val in ("RIGHT", "RIGHT POKE", "RIGHT_POKE"):
            event_type = "RightPoke"
            device['svg_view'].flash_counter('right')
            if device.get('is_tracking'):
                device['stats']['right'] += 1
                self.update_fed_view_counts(device)
                
                
        # Log to CSV
        log_file = device.get('log_file')
        if log_file and device.get('is_tracking'):
            try:
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        dev_name,
                        event_type,
                        device['stats']['left'],
                        device['stats']['right'],
                        device['stats']['pellet'],
                        line
                    ])
            except Exception as e:
                self.fed_log.append_log(f"CSV Write Error: {e}", False)

    def update_plot(self):
        from datetime import timedelta
        from matplotlib.ticker import MaxNLocator

        self.ax.clear()
        self.ax.set_title("Pellets retrieved", color='white')
        self.ax.set_xlabel("Time", color='white')
        self.ax.set_ylabel("Cumulative Pellets", color='white')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#444444')
        
        plotted_something = False
        min_time = None
        max_time = None
        
        target_dev = self.get_current_plot_device()
        devices_to_plot = [target_dev] if target_dev else self.fed_devices
        
        for dev in devices_to_plot:
            events = dev.get('events', [])
            dev_name = dev['name_edit'].text().strip() or dev['box'].title()
            
            if events:
                times = mdates.date2num(events)
                counts = list(range(1, len(events) + 1))
                
                if min_time is None or events[0] < min_time:
                    min_time = events[0]
                if max_time is None or events[-1] > max_time:
                    max_time = events[-1]
                
                self.ax.plot_date(times, counts, '-', label=dev_name, linewidth=2, marker='o', markersize=6, drawstyle='steps-post')
                plotted_something = True
            elif dev.get('is_tracking') and dev.get('tracking_start_time'):
                start_t = dev['tracking_start_time']
                end_t = max(datetime.now(), start_t + timedelta(minutes=5))
                times = mdates.date2num([start_t, end_t])
                counts = [0, 0]
                
                if min_time is None or start_t < min_time:
                    min_time = start_t
                if max_time is None or end_t > max_time:
                    max_time = end_t
                
                self.ax.plot_date(times, counts, '-', label=dev_name, linewidth=2, marker='o', markersize=6, drawstyle='steps-post')
                plotted_something = True
        
        # Toggle between placeholder and canvas
        self.plot_placeholder.setVisible(not plotted_something)
        self.canvas.setVisible(plotted_something)
                
        if plotted_something:
            self.ax.legend(facecolor='#2b2b2b', edgecolor='#444444', labelcolor='white')
            
            # Shade dark cycle (19:00 to 07:00)
            if min_time and max_time:
                current_shade_start = min_time.replace(hour=19, minute=0, second=0, microsecond=0)
                if min_time.hour < 19:
                    current_shade_start -= timedelta(days=1)
                
                while current_shade_start < max_time:
                    shade_end = current_shade_start + timedelta(hours=12)
                    self.ax.axvspan(mdates.date2num(current_shade_start), 
                                    mdates.date2num(shade_end), 
                                    color='gray', alpha=0.3, zorder=0)
                    current_shade_start += timedelta(days=1)
            
            # Auto-scale x-axis so ticks adapt as the time range grows
            locator = mdates.AutoDateLocator()
            self.ax.xaxis.set_major_locator(locator)
            self.ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
            self.ax.autoscale_view()
            
            # Overwrite autoscaled limits with safer bounds to prevent zero-range formatting errors
            if min_time and max_time:
                if (max_time - min_time).total_seconds() < 300:
                    mid = min_time + (max_time - min_time) / 2
                    min_t_plot = mid - timedelta(minutes=2.5)
                    max_t_plot = mid + timedelta(minutes=2.5)
                else:
                    min_t_plot = min_time
                    max_t_plot = max_time
                self.ax.set_xlim(mdates.date2num(min_t_plot), mdates.date2num(max_t_plot))
            
            # Configure integer ticks on y-axis
            self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
            max_y = 0
            for dev in devices_to_plot:
                events = dev.get('events', [])
                if events:
                    max_y = max(max_y, len(events))
            if max_y == 0:
                self.ax.set_ylim(0, 10)
            else:
                self.ax.set_ylim(bottom=0)
                
            self.figure.autofmt_xdate()
            self.figure.tight_layout(pad=1.5)
            self.canvas.draw()

    def update_fed_view_combo(self):
        current_plot_idx = self.plot_filter_combo.currentIndex()
        
        current_sched_idx = -1
        if hasattr(self, 'sched_target_combo'):
            current_sched_idx = self.sched_target_combo.currentIndex()
            self.sched_target_combo.clear()
            self.sched_target_combo.addItem("All Devices")
        
        self.plot_filter_combo.clear()
        self.plot_filter_combo.addItem("All Devices")
        
        for dev in self.fed_devices:
            dev_name = dev['name_edit'].text().strip() or dev['box'].title()
            self.plot_filter_combo.addItem(dev_name)
            if hasattr(self, 'sched_target_combo'):
                self.sched_target_combo.addItem(dev_name)
            
        if current_plot_idx > 0 and current_plot_idx <= len(self.fed_devices):
            self.plot_filter_combo.setCurrentIndex(current_plot_idx)
        else:
            self.plot_filter_combo.setCurrentIndex(0)

        if hasattr(self, 'sched_target_combo'):
            if current_sched_idx > 0 and current_sched_idx <= len(self.fed_devices):
                self.sched_target_combo.setCurrentIndex(current_sched_idx)
            else:
                self.sched_target_combo.setCurrentIndex(0)

    def update_fed_view_counts(self, device):
        device['svg_view'].left_counter.setText(str(device['stats']['left']))
        device['svg_view'].right_counter.setText(str(device['stats']['right']))
        device['svg_view'].pellet_counter.setText(str(device['stats']['pellet']))
        device['svg_view'].update()  # trigger repaint since counters are painted directly

    def handle_scan_finished(self, valid_ports, candidate_ports=None):
        # Stop spinner animation
        if hasattr(self, '_spinner_timer'):
            self._spinner_timer.stop()
        self.refresh_ports_btn.setText("Refresh Ports")
        self.refresh_ports_btn.setEnabled(True)
        if self.main_window:
            self.main_window.statusBar().showMessage("Ready")

        # Clear any auto-populated names first so they can be freshly generated and unique
        for dev in self.fed_devices:
            name = dev['name_edit'].text().strip()
            if name.startswith("FED "):
                dev['name_edit'].clear()

        # 1. Gather all active ports that are FED3 Active and extract device IDs
        active_fed_ports = []
        all_port_displays = []
        port_to_id = {}
        added_devs = set()
        
        # Process valid ports (detected active FEDs, busy ports, etc.)
        for p in (valid_ports or []):
            dev_port = p[0]
            status = p[1]
            dev_id = p[2] if len(p) > 2 else None
            
            all_port_displays.append((dev_port, dev_port))
            added_devs.add(dev_port)
            if status == "FED3 Active":
                active_fed_ports.append(dev_port)
                if dev_id:
                    port_to_id[dev_port] = dev_id
                    
        # Process all other system ports
        for p in (candidate_ports or []):
            if p not in added_devs:
                all_port_displays.append((p, p))
                added_devs.add(p)

        # Check for duplicate device IDs and warn
        id_counts = {}
        for dev_id in port_to_id.values():
            if dev_id:
                id_counts[dev_id] = id_counts.get(dev_id, 0) + 1
        
        warned_ids = set()
        for dev_id, count in id_counts.items():
            if count > 1 and dev_id not in warned_ids:
                warned_ids.add(dev_id)
                msg = f"Warning: Multiple devices detected with ID {dev_id}!"
                self.fed_log.append_log(msg, False)
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Duplicate Device ID", msg)

        assigned_names = {}
        def get_unique_name(device_id):
            base_name = f"FED {device_id}"
            if base_name not in assigned_names:
                assigned_names[base_name] = 1
                return base_name
            else:
                assigned_names[base_name] += 1
                return f"{base_name} ({assigned_names[base_name]})"

        # 2. Populate port_to_id mapping
        self._port_to_id = port_to_id

        # 3. Populate combo boxes for all existing devices
        for dev in self.fed_devices:
            combo = dev['port_combo']
            combo.blockSignals(True)
            combo.clear()
            if all_port_displays:
                for port_val, display_val in all_port_displays:
                    combo.addItem(display_val, port_val)
            else:
                combo.addItem("No FED3 found")
            combo.setEnabled(True)
            combo.blockSignals(False)

        # 4. Restore selections for devices that have a saved port
        assigned_ports = []
        for dev in self.fed_devices:
            combo = dev['port_combo']
            curr = dev.get('saved_port', '')
            combo.blockSignals(True)
            if curr and curr not in ("Scanning...", "No FED3 found", ""):
                idx = combo.findText(curr)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    assigned_ports.append(curr)
                else:
                    combo.addItem(curr, curr)
                    combo.setCurrentText(curr)
                    assigned_ports.append(curr)
                
                # If name is empty, auto-populate with the device ID if we discovered it
                if curr in port_to_id:
                    device_id = port_to_id[curr]
                    if device_id and not dev['name_edit'].text().strip():
                        dev['name_edit'].setText(get_unique_name(device_id))
            else:
                combo.setCurrentIndex(-1)
            combo.blockSignals(False)

        # 5. Auto-discover active FED3 devices and assign them to matching slots
        unassigned_fed_ports = [p for p in active_fed_ports if p not in assigned_ports]
        unassigned_fed_ports = [p for p in unassigned_fed_ports if p not in self.removed_ports]

        # First pass: try to assign ports to their matching slot indices (based on on-board ID)
        still_unassigned = []
        for port in unassigned_fed_ports:
            dev_id = self._port_to_id.get(port)
            matched = False
            if dev_id:
                try:
                    slot_idx = int(dev_id) - 1
                    if 0 <= slot_idx < len(self.fed_devices):
                        target_dev = self.fed_devices[slot_idx]
                        combo = target_dev['port_combo']
                        curr = target_dev.get('saved_port', '')
                        if not curr or curr in ("Scanning...", "No FED3 found", ""):
                            # Slot is empty/unassigned, assign it!
                            combo.blockSignals(True)
                            idx = combo.findData(port)
                            if idx >= 0:
                                combo.setCurrentIndex(idx)
                                assigned_ports.append(port)
                                target_dev['saved_port'] = port
                                matched = True
                                # Auto-populate name with the device ID
                                if not target_dev['name_edit'].text().strip():
                                    target_dev['name_edit'].setText(get_unique_name(dev_id))
                            combo.blockSignals(False)
                except ValueError:
                    pass
            if not matched:
                still_unassigned.append(port)

        # Second pass: assign remaining ports to any available empty slots
        for port in still_unassigned:
            assigned = False
            for dev in self.fed_devices:
                combo = dev['port_combo']
                curr = dev.get('saved_port', '')
                if not curr or curr in ("Scanning...", "No FED3 found", ""):
                    combo.blockSignals(True)
                    idx = combo.findData(port)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                        assigned_ports.append(port)
                        dev['saved_port'] = port
                        assigned = True
                        dev_id = self._port_to_id.get(port)
                        if dev_id and not dev['name_edit'].text().strip():
                            dev['name_edit'].setText(get_unique_name(dev_id))
                    combo.blockSignals(False)
                    break
            
            if not assigned:
                # Create a new device widget
                new_dev = self.create_device_widget(refresh=False)
                combo = new_dev['port_combo']
                combo.blockSignals(True)
                combo.clear()
                if all_port_displays:
                    for port_val, display_val in all_port_displays:
                        combo.addItem(display_val, port_val)
                else:
                    combo.addItem("No FED3 found")
                combo.setEnabled(True)
                
                idx = combo.findData(port)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    assigned_ports.append(port)
                    new_dev['saved_port'] = port
                    dev_id = self._port_to_id.get(port)
                    if dev_id and not new_dev['name_edit'].text().strip():
                        new_dev['name_edit'].setText(get_unique_name(dev_id))
                combo.blockSignals(False)

        # 6. Trigger on_port_changed for all devices to initialize/restore serial workers
        for dev in self.fed_devices:
            self.on_port_changed(dev, dev['port_combo'].currentText(), warn_id_mismatch=False)

        self.update_control_panels_enabled_state()

    def revert_port_combo(self, device):
        prev_port = device['port_combo'].property("previous_port")
        device['port_combo'].blockSignals(True)
        if prev_port is not None:
            idx = device['port_combo'].findText(prev_port)
            if idx >= 0:
                device['port_combo'].setCurrentIndex(idx)
            else:
                device['port_combo'].setCurrentText(str(prev_port))
        else:
            device['port_combo'].setCurrentIndex(-1)
        device['port_combo'].blockSignals(False)
        
        # Ensure tracker is running on the reverted port
        reverted_text = device['port_combo'].currentText()
        reverted_port = reverted_text.strip()
        if reverted_port and reverted_port not in ("Scanning...", "No FED3 found", ""):
            worker = device.get('tracker_worker')
            if worker and worker.port == reverted_port and worker.isRunning():
                return
            self.start_device_acquisition(device, reverted_port)

    def on_port_changed(self, device, text, warn_id_mismatch=True):
        port = text.strip()
        if not port or port in ("Scanning...", "No FED3 found"):
            # Stop existing worker if running
            worker = device.get('tracker_worker')
            if worker:
                worker.stop()
                device['tracker_worker'] = None
            return

        # Check if current worker is already running on the same port
        worker = device.get('tracker_worker')
        if worker and worker.port == port and worker.isRunning():
            # Update previous port property and return
            device['port_combo'].setProperty("previous_port", text)
            return

        # 1. Check if port is already assigned to another device
        duplicate = False
        for dev in self.fed_devices:
            if dev is not device:
                dev_port = dev['port_combo'].currentText().strip()
                if dev_port == port:
                    duplicate = True
                    break
        
        if duplicate:
            QMessageBox.warning(
                self,
                "Duplicate Port Selection",
                f"The port '{port}' is already selected by another device. Please choose a different port."
            )
            self.revert_port_combo(device)
            return

        # 2. Check if on-board ID matches the slot index
        if warn_id_mismatch and hasattr(self, '_port_to_id'):
            onboard_id = self._port_to_id.get(port)
            slot_num = self.fed_devices.index(device) + 1
            if onboard_id and str(onboard_id) != str(slot_num):
                reply = QMessageBox.question(
                    self,
                    "Device ID Mismatch",
                    f"Warning: The physical device on port {port} has on-board ID {onboard_id}, "
                    f"but you are assigning it to the Device {slot_num} slot.\n\n"
                    f"Do you want to proceed?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    self.revert_port_combo(device)
                    return

        # If the user selected this port manually, remove it from the removed_ports blacklist
        if port in self.removed_ports:
            self.removed_ports.remove(port)
            
        # Ensure the manually entered port is in the dropdown list for all devices
        for dev in self.fed_devices:
            c = dev['port_combo']
            if c.findText(port) < 0:
                c.blockSignals(True)
                curr_sel = c.currentText()
                c.addItem(port, port)
                c.setCurrentText(curr_sel)
                c.blockSignals(False)

        # Stop existing worker if running on a different port
        if worker:
            worker.stop()
            device['tracker_worker'] = None

        dev_name = device['name_edit'].text().strip() or device['box'].title()
        new_worker = Fed3TrackerWorker(port, dev_name)
        new_worker.line_received.connect(lambda d, l: self.on_tracker_line(d, l, device))
        new_worker.error_received.connect(lambda d, l: self.fed_log.append_log(f"{d}: {l}", False))
        new_worker.finished.connect(lambda d=device: self.handle_tracker_finished(d))
        device['tracker_worker'] = new_worker
        new_worker.start()
        self.fed_log.append_log(f"Opened live visual monitoring on {port} for {dev_name}")

        # Update previous port property
        device['port_combo'].setProperty("previous_port", text)


    def remove_device(self, device):
        if len(self.fed_devices) <= 1: return
        
        if not self.confirm_action_if_tracking("Are you sure you want to remove this device?"):
            return
        
        # Track removed port to avoid auto-adding it back
        port = device['port_combo'].currentData() or device['port_combo'].currentText()
        if port and port not in ("Scanning...", "No FED3 found", ""):
            self.removed_ports.add(port)
            
        device['is_tracking'] = False
        device['timer'].stop()
        worker = device.get('tracker_worker')
        if worker:
            worker.stop()
        self.devices_layout.removeWidget(device['box'])
        device['box'].deleteLater()
        if 'svg_container' in device:
            self.fed_view_layout.removeWidget(device['svg_container'])
            device['svg_container'].deleteLater()
            
        if device in self.fed_devices:
            self.fed_devices.remove(device)
        self.reorganize_device_indices()
        self.update_remove_buttons()
        self.update_fed_view_combo()
        self.update_control_panels_enabled_state()
        self.update_plot()

    def reorganize_device_indices(self):
        for i, dev in enumerate(self.fed_devices, start=1):
            dev['box'].setTitle(f"Device {i}")
            dev['svg_title'].setText(dev['name_edit'].text().strip() or f"Device {i}")

    def update_remove_buttons(self):
        enable = len(self.fed_devices) > 1
        for dev in self.fed_devices:
            dev['remove_btn'].setEnabled(enable)

    def do_device_sync(self, device):
        if device.get('is_syncing'):
            return
            
        port = device['port_combo'].currentData() or device['port_combo'].currentText() or None
        dev_name = device['name_edit'].text().strip() or device['box'].title()
        
        if self.main_window:
            self.main_window.statusBar().showMessage(f"Syncing {dev_name}...")

        worker = device.get('tracker_worker')
        if worker and worker.isRunning():
            now = datetime.now()
            sync_string = now.strftime("SYNC:%Y,%m,%d,%H,%M,%S\n")
            success, msg = worker.tracker.send_command(sync_string)
            self.fed_log.append_log(f"{dev_name} Sync: {msg}", success)
            if success:
                device['last_sync_label'].setText(f"Last Sync: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            if self.main_window: self.main_window.statusBar().showMessage("Ready")
            return

        device['is_syncing'] = True

        def task():
            return fed_comms.sync_time(port=port)

        thread_worker = self.WorkerThread(task, f"Sync {dev_name}")
        
        def on_finished(success, message):
            prefixed = "\n".join([f"{dev_name}: {l}" for l in message.splitlines()])
            self.fed_log.append_log(prefixed, success)
            device['last_sync_label'].setText(f"Last Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            device['is_syncing'] = False
            if self.main_window: self.main_window.statusBar().showMessage("Ready")
            if thread_worker in self._active_workers: self._active_workers.remove(thread_worker)
            thread_worker.deleteLater()

        thread_worker.finished.connect(on_finished)
        self._active_workers.append(thread_worker)
        thread_worker.start()

    def toggle_lights(self, device, checked, confirm=True):
        if device.get('is_syncing'):
            self.fed_log.append_log(f"Device {device['name_edit'].text().strip() or device['box'].title()} is busy.", False)
            btn = device['lights_toggle_btn']
            btn.blockSignals(True)
            btn.setChecked(not checked)
            btn.blockSignals(False)
            return

        if confirm and not self.confirm_action_if_tracking("Are you sure you want to toggle lights on this device?"):
            btn = device['lights_toggle_btn']
            btn.blockSignals(True)
            btn.setChecked(not checked)
            btn.blockSignals(False)
            if checked:
                btn.setText("Lights: OFF")
            else:
                btn.setText("Lights: ON")
            return

        btn = device['lights_toggle_btn']
        if checked:
            btn.setText("Lights: ON")
            self.send_command_to_device(device, "LIGHTS:ON", "Turn Lights ON", confirm=False)
        else:
            btn.setText("Lights: OFF")
            self.send_command_to_device(device, "LIGHTS:OFF", "Turn Lights OFF", confirm=False)

    def send_command_to_device(self, device, command, action_name, confirm=True):
        if confirm and not self.confirm_action_if_tracking(f"Are you sure you want to send command '{action_name}' to this device?"):
            return

        if device.get('is_syncing'):
            self.fed_log.append_log(f"Device {device['name_edit'].text().strip() or device['box'].title()} is busy.", False)
            return

        port = device['port_combo'].currentData() or device['port_combo'].currentText() or None
        dev_name = device['name_edit'].text().strip() or device['box'].title()
        
        if not port:
            self.fed_log.append_log(f"[{dev_name}] No port selected.", False)
            return

        worker = device.get('tracker_worker')
        if worker and worker.isRunning():
            success, msg = worker.tracker.send_command(command)
            self.fed_log.append_log(f"[{dev_name}] {msg}", success)
            return

        device['is_syncing'] = True
        if self.main_window:
            self.main_window.statusBar().showMessage(f"Sending {action_name} to {dev_name}...")

        def task():
            return fed_comms.send_custom_command(command, port=port)

        thread_worker = self.WorkerThread(task, f"{action_name} {dev_name}")
        
        def on_finished(success, message, d=device, w=thread_worker):
            prefixed = "\n".join([f"{dev_name}: {l}" for l in message.splitlines()])
            self.fed_log.append_log(prefixed, success)
            d['is_syncing'] = False
            if self.main_window: self.main_window.statusBar().showMessage("Ready")
            if w in self._active_workers: self._active_workers.remove(w)
            w.deleteLater()

        thread_worker.finished.connect(on_finished)
        self._active_workers.append(thread_worker)
        thread_worker.start()

    def update_mode_ui(self, device):
        mode = device['mode_combo'].currentText()
        is_fr = (mode == "Fixed Ratio (FR)")
        is_rr = (mode == "Random Ratio (RR)")
        is_timeout = (mode == "FR with Timeout")
        is_fr_rev = (mode == "FR Reversed")
        
        device['fr_label'].setVisible(is_fr or is_rr or is_timeout or is_fr_rev)
        device['ratio_spin'].setVisible(is_fr or is_rr or is_timeout or is_fr_rev)
        
        device['timeout_label'].setVisible(is_timeout)
        device['timeout_spin'].setVisible(is_timeout)
        device['timeout_unit_label'].setVisible(is_timeout)

    def apply_device_mode(self, device, confirm=True):
        if confirm and not self.confirm_action_if_tracking("Are you sure you want to apply mode settings to this device?"):
            return

        mode = device['mode_combo'].currentText()
        if mode == "Fixed Ratio (FR)":
            ratio = device['ratio_spin'].value()
            if ratio == 1:
                command = "MODE:FR1"
                action_name = "Set Mode to FR1"
            elif ratio == 3:
                command = "MODE:FR3"
                action_name = "Set Mode to FR3"
            elif ratio == 5:
                command = "MODE:FR5"
                action_name = "Set Mode to FR5"
            else:
                command = f"MODE:FR,{ratio}"
                action_name = f"Set Mode to FR (Ratio {ratio})"
        elif mode == "Progressive Ratio (PR)":
            command = "MODE:PR"
            action_name = "Set Mode to PR"
        elif mode == "Random Ratio (RR)":
            ratio = device['ratio_spin'].value()
            command = f"MODE:RR,{ratio}"
            action_name = f"Set Mode to RR (Avg Ratio {ratio})"
        elif mode == "FR with Timeout":
            ratio = device['ratio_spin'].value()
            timeout_s = device['timeout_spin'].value()
            command = f"MODE:FRTO,{ratio},{timeout_s}"
            action_name = f"Set Mode to FR-Timeout (Ratio {ratio}, Timeout {timeout_s}s)"
        elif mode == "Free Feeding":
            command = "MODE:FREE"
            action_name = "Set Mode to Free Feed"
        elif mode == "Extinction":
            command = "MODE:EXTINCT"
            action_name = "Set Mode to Extinction"
        elif mode == "Light Tracking":
            command = "MODE:LIGHTTRK"
            action_name = "Set Mode to Light Tracking"
        elif mode == "FR Reversed":
            ratio = device['ratio_spin'].value()
            if ratio == 1:
                command = "MODE:FR1_R"
                action_name = "Set Mode to FR1 Reversed"
            else:
                command = f"MODE:FR_R,{ratio}"
                action_name = f"Set Mode to FR Reversed (Ratio {ratio})"
        elif mode == "PR Reversed":
            command = "MODE:PR_R"
            action_name = "Set Mode to PR Reversed"
        elif mode == "Opto Stimulation":
            command = "MODE:OPTO"
            action_name = "Set Mode to Opto Stimulation"
        elif mode == "Opto Reversed":
            command = "MODE:OPTO_R"
            action_name = "Set Mode to Opto Reversed"
        elif mode == "Timed Feeding":
            command = "MODE:TIMED"
            action_name = "Set Mode to Timed Feeding"
        else:
            return
            
        self.send_command_to_device(device, command, action_name)

    def handle_fed_command(self, command):
        if not self.confirm_action_if_tracking(f"Are you sure you want to send command '{command}' to all devices?"):
            return
        if not self.fed_devices:
            self.fed_log.append_log("No devices added.", False)
            return
        
        self.fed_log.append_log(f"Sending '{command}' to all devices...")
        for device in self.fed_devices:
            if device.get('is_syncing'):
                continue
                
            port = device['port_combo'].currentData() or device['port_combo'].currentText() or None
            dev_name = device['name_edit'].text().strip() or device['box'].title()
            
            worker = device.get('tracker_worker')
            if worker and worker.isRunning():
                success, msg = worker.tracker.send_command(command)
                self.fed_log.append_log(f"[{dev_name}] {msg}", success)
                continue
            
            device['is_syncing'] = True

            def make_task(p, cmd):
                return lambda: fed_comms.send_custom_command(cmd, port=p)

            thread_worker = self.WorkerThread(make_task(port, command), f"Cmd {dev_name}")
            
            def on_cmd_finished(success, msg, d=device, w=thread_worker):
                prefixed = "\n".join([f"{d['name_edit'].text().strip() or d['box'].title()}: {l}" for l in msg.splitlines()])
                self.fed_log.append_log(prefixed, success)
                d['is_syncing'] = False
                if w in self._active_workers: self._active_workers.remove(w)
                w.deleteLater()

            thread_worker.finished.connect(on_cmd_finished)
            self._active_workers.append(thread_worker)
            thread_worker.start()

    def _update_refresh_spinner(self):
        frame = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
        self.refresh_ports_btn.setText(f"Refreshing {frame}")
        self._spinner_idx += 1

    def refresh_all_ports(self):
        if hasattr(self, '_global_scanner') and self._global_scanner is not None and self._global_scanner.isRunning():
            return
            
        # Start spinner animation
        self.refresh_ports_btn.setEnabled(False)
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_idx = 0
        if not hasattr(self, '_spinner_timer'):
            self._spinner_timer = QTimer(self)
            self._spinner_timer.timeout.connect(self._update_refresh_spinner)
        self._spinner_timer.start(100)
        if self.main_window:
            self.main_window.statusBar().showMessage("Scanning ports for FED3 devices...")
            
        active_ports = []
        for dev in self.fed_devices:
            p = dev['port_combo'].currentData() or dev['port_combo'].currentText()
            if p:
                if dev.get('is_tracking') or dev.get('is_syncing'):
                    active_ports.append(p)
            
        # Save current selections before clearing
        for dev in self.fed_devices:
            curr = dev['port_combo'].currentText().strip()
            if curr and curr not in ("Scanning...", "No FED3 found", ""):
                dev['saved_port'] = curr
            elif 'saved_port' not in dev:
                dev['saved_port'] = ""

        for dev in self.fed_devices:
            dev['port_combo'].clear()
            dev['port_combo'].addItem("Scanning...")
            dev['port_combo'].setEnabled(False)
            
        self._global_scanner = PortScannerWorker(active_ports)
        self._global_scanner.finished_scan.connect(self.scan_finished_signal.emit)
        self._global_scanner.start()
        self.update_control_panels_enabled_state()

    def toggle_auto_sync(self, checked):
        if checked:
            self.start_all_btn.setText("Stop Auto Sync")
            ms = self.get_global_interval_ms()
            for dev in self.fed_devices:
                if not dev['timer'].isActive():
                    dev['timer'].start(ms)
                    self.do_device_sync(dev)
        else:
            self.start_all_btn.setText("Start Auto Sync")
            for dev in self.fed_devices:
                dev['timer'].stop()

    def reset_device_counters(self, device, confirm=True):
        dev_name = device['name_edit'].text().strip() or device['box'].title()
        if confirm:
            reply = QMessageBox.question(
                self,
                "Reset Counters",
                f"Are you sure you want to reset counters and clear the plot for {dev_name}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
                
        device['stats'] = {'left': 0, 'right': 0, 'pellet': 0}
        device['events'] = []
        self.update_fed_view_counts(device)
        self.update_plot()
        
        # Log RESET event to CSV
        log_file = device.get('log_file')
        if log_file and os.path.exists(log_file):
            try:
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                        dev_name,
                        "RESET",
                        0, 0, 0,
                        "COUNTERS RESET BY USER"
                    ])
            except Exception as e:
                self.fed_log.append_log(f"Error logging reset to CSV: {e}", False)
                
        self.fed_log.append_log(f"Reset counters for {dev_name}")

    def reset_all_counters(self):
        reply = QMessageBox.question(
            self,
            "Reset All Counters",
            "Are you sure you want to reset counters and clear plots for ALL connected devices?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            for dev in self.fed_devices:
                self.reset_device_counters(dev, confirm=False)
            self.fed_log.append_log("Reset counters on all devices.", True)

    def export_device_log(self, device):
        log_file = device.get('log_file')
        if not log_file or not os.path.exists(log_file):
            QMessageBox.warning(self, "Export Log", "No log file found for this device yet.")
            return
            
        dev_name = device['name_edit'].text().strip() or device['box'].title()
        default_name = f"{dev_name}_export.csv"
        dest_path, _ = QFileDialog.getSaveFileName(self, "Export Device Log", default_name, "CSV Files (*.csv)")
        if dest_path:
            try:
                import shutil
                shutil.copy2(log_file, dest_path)
                self.fed_log.append_log(f"Exported log for {dev_name} to {dest_path}", True)
                QMessageBox.information(self, "Export Successful", f"Successfully exported log to {dest_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export log: {e}")

    def export_all_logs(self):
        active_logs = [d for d in self.fed_devices if d.get('log_file') and os.path.exists(d['log_file'])]
        if not active_logs:
            QMessageBox.warning(self, "Export Logs", "No active log files found to export.")
            return
            
        dest_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Export All Logs")
        if dest_dir:
            import shutil
            success_count = 0
            for d in active_logs:
                try:
                    src = d['log_file']
                    filename = os.path.basename(src)
                    dest = os.path.join(dest_dir, filename)
                    shutil.copy2(src, dest)
                    success_count += 1
                except Exception as e:
                    self.fed_log.append_log(f"Failed to export log for {d['box'].title()}: {e}", False)
            if success_count > 0:
                QMessageBox.information(self, "Export Successful", f"Successfully exported {success_count} log file(s) to {dest_dir}")

    def sync_all(self):
        for dev in self.fed_devices:
            self.do_device_sync(dev)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.adjust_responsive_layout()

    def adjust_responsive_layout(self):
        if not hasattr(self, 'columns_layout') or not hasattr(self, 'control_group') or not hasattr(self, 'scheduler_group') or not hasattr(self, 'devices_group'):
            return
            
        width = self.width()
        is_narrow = width < 1120
        
        current_state = self.columns_layout.property("is_narrow")
        if current_state == is_narrow:
            return
            
        self.columns_layout.setProperty("is_narrow", is_narrow)
        
        self.columns_layout.removeWidget(self.control_group)
        self.columns_layout.removeWidget(self.scheduler_group)
        self.columns_layout.removeWidget(self.devices_group)
        
        if is_narrow:
            # Stacked single-column layout
            self.columns_layout.addWidget(self.control_group, 0, 0)
            self.columns_layout.addWidget(self.scheduler_group, 1, 0)
            self.columns_layout.addWidget(self.devices_group, 2, 0)
            
            self.columns_layout.setColumnStretch(0, 1)
            self.columns_layout.setColumnStretch(1, 0)
            self.columns_layout.setRowStretch(0, 0)
            self.columns_layout.setRowStretch(1, 0)
            self.columns_layout.setRowStretch(2, 0)
        else:
            # Side-by-side two-column layout
            self.columns_layout.addWidget(self.control_group, 0, 0)
            self.columns_layout.addWidget(self.scheduler_group, 1, 0)
            self.columns_layout.addWidget(self.devices_group, 0, 1, 2, 1)
            
            self.columns_layout.setColumnStretch(0, 1)
            self.columns_layout.setColumnStretch(1, 1)
            self.columns_layout.setRowStretch(0, 0)
            self.columns_layout.setRowStretch(1, 1)
            self.columns_layout.setRowStretch(2, 0)
            
        self.columns_layout.invalidate()
