import os
import csv
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLabel, QGroupBox, QTextEdit, QScrollArea, QSizePolicy, 
    QComboBox, QSpinBox, QLineEdit, QFrame, QFileDialog, QTabWidget,
    QLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, QRect, QPoint, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter, QBrush

try:
    from PyQt5.QtSvg import QSvgWidget
except ImportError:
    QSvgWidget = None

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

class OverlayLabel(QLabel):
    def __init__(self, text, parent=None, is_circle=False):
        super().__init__(text, parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._bg_color = QColor(0, 0, 0, 150)
        self._radius = 0
        self.is_circle = is_circle

    def set_bg_color(self, r, g, b, a):
        self._bg_color = QColor(r, g, b, a)
        self.update()

    def set_radius(self, r):
        self._radius = r
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._bg_color))
        rect = self.rect()
        if self.is_circle:
            painter.drawEllipse(rect)
        else:
            painter.drawRoundedRect(rect, self._radius, self._radius)
        
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(self.font())
        painter.drawText(rect, Qt.AlignCenter, self.text())
        painter.end()

class FEDSvgView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: transparent;")
        if QSvgWidget:
            svg_path = os.path.join(os.path.dirname(__file__), "fed3_image.svg")
            self.svg_widget = QSvgWidget(svg_path, self)
        else:
            self.svg_widget = QLabel("SVG Support Missing", self)
            
        self.left_counter = OverlayLabel("0", self.svg_widget, is_circle=True)
        self.right_counter = OverlayLabel("0", self.svg_widget, is_circle=True)
        self.pellet_counter = OverlayLabel("0", self.svg_widget, is_circle=False)
        
        for counter in (self.left_counter, self.right_counter, self.pellet_counter):
            counter.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        
        # SVG aspect ratio from fed3_image.svg viewBox
        svg_aspect = 163.67577 / 116.04688
        widget_aspect = w / h if h > 0 else 1
        
        if widget_aspect > svg_aspect:
            # Widget is wider than SVG. Height is limiting.
            new_h = h
            new_w = int(h * svg_aspect)
        else:
            # Widget is taller than SVG. Width is limiting.
            new_w = w
            new_h = int(w / svg_aspect)
            
        x_offset = (w - new_w) // 2
        y_offset = (h - new_h) // 2
        
        self.svg_widget.setGeometry(x_offset, y_offset, new_w, new_h)
        
        # Dynamic sizes based on SVG proportions
        poke_radius = int(new_w * 0.0953)
        poke_diam = poke_radius * 2
        pellet_w = int(new_w * 0.1772)
        pellet_h = int(new_h * 0.1677)

        self.left_counter.setFixedSize(poke_diam, poke_diam)
        self.right_counter.setFixedSize(poke_diam, poke_diam)
        self.pellet_counter.setFixedSize(pellet_w, pellet_h)

        self.left_counter.set_radius(poke_radius)
        self.right_counter.set_radius(poke_radius)
        self.pellet_counter.set_radius(4)

        left_right_style = f"background-color: transparent; font-size: {max(10, int(poke_radius*0.8))}px; font-weight: bold;"
        pellet_style = f"background-color: transparent; font-size: {max(10, int(pellet_h*0.5))}px; font-weight: bold;color: white;"
        
        self.left_counter.setStyleSheet(left_right_style)
        self.right_counter.setStyleSheet(left_right_style)
        self.pellet_counter.setStyleSheet(pellet_style)
        
        # Positions based on raw SVG geometry centers/corners.
        # Coordinates are relative to svg_widget.
        self.left_counter.move(int(new_w * 0.2501) - poke_radius, int(new_h * 0.6444) - poke_radius)
        self.right_counter.move(int(new_w * 0.7488) - poke_radius, int(new_h * 0.6426) - poke_radius)
        self.pellet_counter.move(int(new_w * 0.4108), int(new_h * 0.6095))
        
    def flash_counter(self, counter_name):
        counter = getattr(self, f"{counter_name}_counter")
        counter.set_bg_color(76, 175, 80, 180)
        QTimer.singleShot(200, lambda: counter.set_bg_color(0, 0, 0, 150))
        
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
    finished_scan = pyqtSignal(list)

    def run(self):
        import serial
        from serial.tools import list_ports
        import time
        from concurrent.futures import ThreadPoolExecutor

        ports = list(list_ports.comports())

        def check_port(p):
            dev = p.device
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
                    return dev
            except Exception:
                pass
            return None

        # Check ports in parallel to avoid long UI hangs
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(check_port, ports)

        valid_ports = [r for r in results if r is not None]
        self.finished_scan.emit(valid_ports)


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
    scan_finished_signal = pyqtSignal(list)
    
    def __init__(self, parent=None, worker_class=None):
        super().__init__(parent)
        self.main_window = parent
        self.WorkerThread = worker_class # Pass the WorkerThread class from main
        self._active_workers = []
        self.fed_devices = []
        self.scan_finished_signal.connect(self.handle_scan_finished)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        main_widget = QWidget()
        self.layout = QVBoxLayout(main_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        scroll.setWidget(main_widget)
        main_layout.addWidget(scroll)

        # --- Section 1: Plot View ---
        self.plot_tab = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_tab)
        self.plot_layout.setContentsMargins(10, 10, 10, 10)

        self.plot_filter_combo = QComboBox()
        self.plot_filter_combo.addItem("All Devices", userData=None)
        self.plot_filter_combo.currentTextChanged.connect(lambda: self.update_plot())
        self.plot_layout.addWidget(self.plot_filter_combo)

        # Matplotlib visualization
        self.figure = Figure(figsize=(5, 3))
        self.figure.patch.set_facecolor('#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(300)
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
        self.plot_layout.addStretch()

        # --- Section 2: FED3 View ---
        self.fed_view_tab = QWidget()
        self.fed_view_layout = FlowLayout(margin=10, spacing=20)
        self.fed_view_tab.setLayout(self.fed_view_layout)

        # FED Sync group
        sync_group = QGroupBox("FED Sync Tools & Tracking")
        sync_group_layout = QVBoxLayout()
        sync_group_layout.setSpacing(12)
        sync_group_layout.setContentsMargins(10, 15, 10, 10)

        # Top controls
        mgmt_layout = QHBoxLayout()
        self.add_device_btn = QPushButton("Add Device")
        self.refresh_ports_btn = QPushButton("Refresh Ports")
        mgmt_layout.addWidget(self.add_device_btn)
        mgmt_layout.addStretch()
        mgmt_layout.addWidget(self.refresh_ports_btn)
        sync_group_layout.addLayout(mgmt_layout)

        sync_settings_layout = QHBoxLayout()
        sync_settings_layout.addWidget(QLabel("Auto Sync:"))
        self.global_interval_spin = QSpinBox()
        self.global_interval_spin.setRange(1, 99999)
        self.global_interval_spin.setValue(1)
        self.global_interval_spin.setFixedWidth(80)
        self.global_unit_combo = QComboBox()
        self.global_unit_combo.addItems(["Seconds", "Minutes", "Hours", "Days"])
        self.global_unit_combo.setCurrentText("Days")
        self.global_unit_combo.setFixedWidth(100)
        
        self.track_all_btn = QPushButton("Start Tracking")
        self.track_all_btn.setCheckable(True)
        self.track_all_btn.setStyleSheet("""
            QPushButton:checked { background-color: #4caf50; color: white; }
            QPushButton { font-weight: bold; }
        """)
        
        self.start_all_btn = QPushButton("Start Auto Sync")
        self.start_all_btn.setCheckable(True)
        self.start_all_btn.setStyleSheet("""
            QPushButton:checked { background-color: #4caf50; color: white; }
            QPushButton { font-weight: bold; }
        """)
        self.sync_now_btn = QPushButton("Sync Now")
        
        sync_settings_layout.addWidget(self.global_interval_spin)
        sync_settings_layout.addWidget(self.global_unit_combo)
        sync_settings_layout.addSpacing(20)
        sync_settings_layout.addWidget(self.track_all_btn)
        sync_settings_layout.addWidget(self.start_all_btn)
        sync_settings_layout.addWidget(self.sync_now_btn)
        sync_settings_layout.addStretch()
        sync_group_layout.addLayout(sync_settings_layout)

        # Devices list
        self.devices_container = QWidget()
        self.devices_layout = FlowLayout(margin=4, spacing=8)
        self.devices_container.setLayout(self.devices_layout)
        sync_group_layout.addWidget(self.devices_container)

        sync_group.setLayout(sync_group_layout)
        self.layout.addWidget(sync_group)
        
        # Add the unifying sections sequentially
        self.layout.addWidget(self.plot_tab)
        
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        divider.setStyleSheet("background-color: #444444; margin: 10px 0px;")
        self.layout.addWidget(divider)
        
        self.layout.addWidget(self.fed_view_tab)
        self.layout.addStretch()

        # Log box at bottom
        self.fed_log = CollapsibleLogBox("Serial Monitor")
        self.layout.addWidget(self.fed_log)

        # Connections
        self.add_device_btn.clicked.connect(self.create_device_widget)
        self.refresh_ports_btn.clicked.connect(self.refresh_all_ports)
        self.track_all_btn.toggled.connect(self.toggle_track_all)
        self.start_all_btn.toggled.connect(self.toggle_auto_sync)
        self.sync_now_btn.clicked.connect(self.sync_all)
        self.fed_log.command_submitted.connect(self.handle_fed_command)

        # Initial device
        self.create_device_widget()

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

    def create_device_widget(self):
        idx = len(self.fed_devices) + 1
        box = QGroupBox(f"Device {idx}")
        box_layout = QGridLayout()

        name_edit = QLineEdit()
        name_edit.setPlaceholderText("Optional device name")
        remove_btn = QPushButton("Remove")
        port_combo = QComboBox()
        port_combo.setEditable(True)
        last_sync_label = QLabel("Last Sync: Never")

        box_layout.addWidget(QLabel("Name:"), 0, 0)
        box_layout.addWidget(name_edit, 0, 1, 1, 2)
        box_layout.addWidget(remove_btn, 0, 3, 1, 1, Qt.AlignRight)
        
        box_layout.addWidget(QLabel("Port:"), 1, 0)
        box_layout.addWidget(port_combo, 1, 1, 1, 3)

        box_layout.addWidget(last_sync_label, 2, 0, 1, 4)
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
        
        name_edit.textChanged.connect(lambda t: svg_title.setText(t.strip() or f"Device {idx}"))
        name_edit.textChanged.connect(lambda _: self.update_fed_view_combo())

        device = {
            'box': box,
            'name_edit': name_edit,
            'port_combo': port_combo,
            'remove_btn': remove_btn,
            'last_sync_label': last_sync_label,
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
        
        remove_btn.clicked.connect(lambda: self.remove_device(device))
        timer.timeout.connect(lambda: self.do_device_sync(device))
        port_combo.addItem("Scanning...")
        port_combo.setEnabled(False)
        self.devices_layout.addWidget(box)
        self.fed_devices.append(device)
        self.update_remove_buttons()
        self.update_fed_view_combo()
        self.refresh_all_ports()
        return device

    def toggle_tracking(self, checked, device):
        if checked:
            port = device['port_combo'].currentText()
            if not port:
                self.fed_log.append_log("No port selected for tracking.", False)
                device['is_tracking'] = False
                return
                
            if not device.get('log_file'):
                device['is_tracking'] = False
                return

            dev_name = device['name_edit'].text().strip() or device['box'].title()
            device['events'] = [] # Reset events for fresh plot
            
            worker = Fed3TrackerWorker(port, dev_name)
            worker.line_received.connect(lambda d, l: self.on_tracker_line(d, l, device))
            worker.error_received.connect(lambda d, l: self.fed_log.append_log(f"{d}: {l}", False))
            device['tracker_worker'] = worker
            worker.start()
            
            device['is_tracking'] = True
            self.fed_log.append_log(f"Started tracking on {port} for {dev_name}")
        else:
            worker = device.get('tracker_worker')
            if worker:
                worker.stop()
                device['tracker_worker'] = None
            device['is_tracking'] = False
            self.fed_log.append_log(f"Stopped tracking for {device['box'].title()}")

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
            device['events'].append(datetime.now())
            device['stats']['pellet'] += 1
            device['svg_view'].flash_counter('pellet')
            self.update_fed_view_counts(device)
            self.update_plot()
            updated_plot = True
            
        elif event_val in ("LEFT", "LEFT POKE", "LEFT_POKE"):
            event_type = "LeftPoke"
            device['stats']['left'] += 1
            device['svg_view'].flash_counter('left')
            self.update_fed_view_counts(device)
                
        elif event_val in ("RIGHT", "RIGHT POKE", "RIGHT_POKE"):
            event_type = "RightPoke"
            device['stats']['right'] += 1
            device['svg_view'].flash_counter('right')
            self.update_fed_view_counts(device)
                
                
        # Log to CSV
        log_file = device.get('log_file')
        if log_file:
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
        self.ax.clear()
        self.ax.set_title("Pellets retrieved", color='white')
        self.ax.set_xlabel("Time", color='white')
        self.ax.set_ylabel("Cumulative Pellets", color='white')
        self.ax.tick_params(colors='white')
        
        plotted_something = False
        min_time = None
        max_time = None
        
        target_dev = self.get_current_plot_device()
        devices_to_plot = [target_dev] if target_dev else self.fed_devices
        
        for dev in devices_to_plot:
            events = dev.get('events', [])
            if events:
                dev_name = dev['name_edit'].text().strip() or dev['box'].title()
                times = mdates.date2num(events)
                counts = list(range(1, len(events) + 1))
                
                if min_time is None or events[0] < min_time:
                    min_time = events[0]
                if max_time is None or events[-1] > max_time:
                    max_time = events[-1]
                self.ax.plot_date(times, counts, '-', label=dev_name, linewidth=2, marker='o', markersize=6, drawstyle='steps-post')
                plotted_something = True
                
        if plotted_something:
            self.ax.legend(facecolor='#2b2b2b', edgecolor='#444444', labelcolor='white')
            
            # Shade dark cycle (19:00 to 07:00)
            if min_time and max_time:
                from datetime import timedelta
                current_shade_start = min_time.replace(hour=19, minute=0, second=0, microsecond=0)
                if min_time.hour < 19:
                    current_shade_start -= timedelta(days=1)
                
                while current_shade_start < max_time:
                    shade_end = current_shade_start + timedelta(hours=12)
                    self.ax.axvspan(mdates.date2num(current_shade_start), 
                                    mdates.date2num(shade_end), 
                                    color='gray', alpha=0.3, zorder=0)
                    current_shade_start += timedelta(days=1)
            
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.figure.autofmt_xdate()
        self.figure.tight_layout(pad=1.5)
        self.canvas.draw()

    def update_fed_view_combo(self):
        current_plot_idx = self.plot_filter_combo.currentIndex()
        
        self.plot_filter_combo.clear()
        self.plot_filter_combo.addItem("All Devices")
        
        for dev in self.fed_devices:
            dev_name = dev['name_edit'].text().strip() or dev['box'].title()
            self.plot_filter_combo.addItem(dev_name)
            
        if current_plot_idx > 0 and current_plot_idx <= len(self.fed_devices):
            self.plot_filter_combo.setCurrentIndex(current_plot_idx)
        else:
            self.plot_filter_combo.setCurrentIndex(0)

    def update_fed_view_counts(self, device):
        device['svg_view'].left_counter.setText(str(device['stats']['left']))
        device['svg_view'].right_counter.setText(str(device['stats']['right']))
        device['svg_view'].pellet_counter.setText(str(device['stats']['pellet']))

    def handle_scan_finished(self, valid_ports):
        for dev in self.fed_devices:
            combo = dev['port_combo']
            current = combo.currentText()
            combo.clear()
            if valid_ports:
                for p in valid_ports:
                    combo.addItem(p)
                idx = combo.findText(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            else:
                # Fallback or just empty
                combo.addItem("No FED3 found")
            combo.setEnabled(True)

    def remove_device(self, device):
        if len(self.fed_devices) <= 1: return
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
        self.update_remove_buttons()
        self.update_fed_view_combo()

    def update_remove_buttons(self):
        enable = len(self.fed_devices) > 1
        for dev in self.fed_devices:
            dev['remove_btn'].setEnabled(enable)

    def do_device_sync(self, device):
        if device.get('is_syncing'):
            return
            
        port = device['port_combo'].currentText() or None
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

    def handle_fed_command(self, command):
        if not self.fed_devices:
            self.fed_log.append_log("No devices added.", False)
            return
        
        self.fed_log.append_log(f"Sending '{command}' to all devices...")
        for device in self.fed_devices:
            if device.get('is_syncing'):
                continue
                
            port = device['port_combo'].currentText() or None
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

    def refresh_all_ports(self):
        if hasattr(self, '_global_scanner') and self._global_scanner is not None and self._global_scanner.isRunning():
            return
            
        for dev in self.fed_devices:
            dev['port_combo'].clear()
            dev['port_combo'].addItem("Scanning...")
            dev['port_combo'].setEnabled(False)
            
        self._global_scanner = PortScannerWorker()
        self._global_scanner.finished_scan.connect(self.scan_finished_signal.emit)
        self._global_scanner.start()

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

    def toggle_track_all(self, checked):
        if checked:
            if not self.fed_devices:
                self.track_all_btn.setChecked(False)
                return
                
            file_path, _ = QFileDialog.getSaveFileName(self, "Select Global CSV Log File", "", "CSV Files (*.csv)")
            if not file_path:
                self.track_all_btn.setChecked(False)
                return
                
            if not os.path.exists(file_path):
                try:
                    with open(file_path, 'w', newline='') as f:
                        f.write("Timestamp,Device,EventType,LeftPokes,RightPokes,Pellets,RawData\n")
                except Exception as e:
                    self.fed_log.append_log(f"Could not create CSV: {e}", False)
                    self.track_all_btn.setChecked(False)
                    return
            
            self.track_all_btn.setText("Stop Tracking")
            
            for dev in self.fed_devices:
                dev['log_file'] = file_path
                if not dev.get('is_tracking', False):
                    self.toggle_tracking(True, dev)
        else:
            self.track_all_btn.setText("Start Tracking")
            for dev in self.fed_devices:
                if dev.get('is_tracking', False):
                    self.toggle_tracking(False, dev)

    def sync_all(self):
        for dev in self.fed_devices:
            self.do_device_sync(dev)
