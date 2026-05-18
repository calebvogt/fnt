import os
import csv
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLabel, QGroupBox, QTextEdit, QScrollArea, QSizePolicy, 
    QComboBox, QSpinBox, QLineEdit, QFrame, QFileDialog
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates

from . import fed_comms


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
    
    def __init__(self, parent=None, worker_class=None):
        super().__init__(parent)
        self.main_window = parent
        self.WorkerThread = worker_class # Pass the WorkerThread class from main
        self._active_workers = []
        self.fed_devices = []
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # Scroll area for main content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_content)
        self.scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(scroll_content)
        
        self.layout.addWidget(scroll)

        # Description
        desc = QLabel("Feeding Experimentation Device (FED) data analysis & tracking")
        desc.setFont(QFont("Arial", 10, QFont.Bold))
        desc.setStyleSheet("color: #cccccc; margin: 10px;")
        self.scroll_layout.addWidget(desc)

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
        self.ax.set_title("Real-Time Events")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Cumulative Count")
        self.figure.tight_layout(pad=1.5)
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#444444')
        self.scroll_layout.addWidget(self.canvas)

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
        
        self.start_all_btn = QPushButton("Start Auto Sync")
        self.stop_all_btn = QPushButton("Stop Auto Sync")
        self.sync_now_btn = QPushButton("Sync Now")
        
        sync_settings_layout.addWidget(self.global_interval_spin)
        sync_settings_layout.addWidget(self.global_unit_combo)
        sync_settings_layout.addSpacing(20)
        sync_settings_layout.addWidget(self.start_all_btn)
        sync_settings_layout.addWidget(self.stop_all_btn)
        sync_settings_layout.addWidget(self.sync_now_btn)
        sync_settings_layout.addStretch()
        sync_group_layout.addLayout(sync_settings_layout)

        # Devices list
        self.devices_container = QWidget()
        self.devices_layout = QVBoxLayout(self.devices_container)
        self.devices_layout.setContentsMargins(4, 4, 4, 4)
        self.devices_layout.setSpacing(8)
        self.devices_layout.setAlignment(Qt.AlignTop)
        sync_group_layout.addWidget(self.devices_container)

        sync_group.setLayout(sync_group_layout)
        self.scroll_layout.addWidget(sync_group)
        self.scroll_layout.addStretch()

        # Log box at bottom
        self.fed_log = CollapsibleLogBox("Serial Monitor")
        self.layout.addWidget(self.fed_log)

        # Connections
        self.add_device_btn.clicked.connect(self.create_device_widget)
        self.refresh_ports_btn.clicked.connect(self.refresh_all_ports)
        self.start_all_btn.clicked.connect(self.start_all)
        self.stop_all_btn.clicked.connect(self.stop_all)
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

        # Tracking controls
        track_btn = QPushButton("Start Tracking")
        track_btn.setCheckable(True)
        track_btn.setStyleSheet("""
            QPushButton:checked { background-color: #4caf50; color: white; }
            QPushButton { font-weight: bold; }
        """)

        box_layout.addWidget(QLabel("Name:"), 0, 0)
        box_layout.addWidget(name_edit, 0, 1, 1, 2)
        box_layout.addWidget(remove_btn, 0, 3, 1, 1, Qt.AlignRight)
        
        box_layout.addWidget(QLabel("Port:"), 1, 0)
        box_layout.addWidget(port_combo, 1, 1, 1, 2)
        box_layout.addWidget(track_btn, 1, 3, 1, 1)

        box_layout.addWidget(last_sync_label, 2, 0, 1, 4)
        box_layout.setColumnStretch(1, 1)
        box.setLayout(box_layout)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        timer = QTimer(self)
        device = {
            'box': box,
            'name_edit': name_edit,
            'port_combo': port_combo,
            'remove_btn': remove_btn,
            'track_btn': track_btn,
            'last_sync_label': last_sync_label,
            'timer': timer,
            'is_syncing': False,
            'tracker_worker': None,
            'log_file': None,
            'events': []
        }
        
        remove_btn.clicked.connect(lambda: self.remove_device(device))
        timer.timeout.connect(lambda: self.do_device_sync(device))
        track_btn.toggled.connect(lambda checked: self.toggle_tracking(checked, device))
        
        self.populate_port_combo(port_combo)
        self.devices_layout.addWidget(box)
        self.fed_devices.append(device)
        self.update_remove_buttons()
        return device

    def toggle_tracking(self, checked, device):
        if checked:
            port = device['port_combo'].currentText()
            if not port:
                self.fed_log.append_log("No port selected for tracking.", False)
                device['track_btn'].setChecked(False)
                return
                
            if not device.get('log_file'):
                file_path, _ = QFileDialog.getSaveFileName(self, "Select CSV Log File", "", "CSV Files (*.csv)")
                if not file_path:
                    device['track_btn'].setChecked(False)
                    return
                device['log_file'] = file_path
                # Write header if new
                if not os.path.exists(file_path):
                    try:
                        with open(file_path, 'w', newline='') as f:
                            f.write("Timestamp,Device,EventData\n")
                    except Exception as e:
                        self.fed_log.append_log(f"Could not create CSV: {e}", False)
                        device['track_btn'].setChecked(False)
                        return

            dev_name = device['name_edit'].text().strip() or device['box'].title()
            device['events'] = [] # Reset events for fresh plot
            
            worker = Fed3TrackerWorker(port, dev_name)
            worker.line_received.connect(lambda d, l: self.on_tracker_line(d, l, device))
            worker.error_received.connect(lambda d, l: self.fed_log.append_log(f"{d}: {l}", False))
            device['tracker_worker'] = worker
            worker.start()
            
            device['track_btn'].setText("Stop Tracking")
            self.fed_log.append_log(f"Started tracking on {port} for {dev_name}")
        else:
            worker = device.get('tracker_worker')
            if worker:
                worker.stop()
                device['tracker_worker'] = None
            device['track_btn'].setText("Start Tracking")
            self.fed_log.append_log(f"Stopped tracking for {device['box'].title()}")

    def on_tracker_line(self, dev_name, line, device):
        self.fed_log.append_log(f"[{dev_name}] {line}")
        
        # Log to CSV
        log_file = device.get('log_file')
        if log_file:
            try:
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), dev_name, line])
            except Exception as e:
                self.fed_log.append_log(f"CSV Write Error: {e}", False)
        
        # Parse for visualization. FED3 lines often contain keywords like "Pellet" or "Poke"
        if "Pellet" in line or "Poke" in line:
            device['events'].append(datetime.now())
            self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.set_title("Real-Time Events")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Cumulative Count")
        
        plotted_something = False
        for dev in self.fed_devices:
            events = dev.get('events', [])
            if events:
                dev_name = dev['name_edit'].text().strip() or dev['box'].title()
                times = mdates.date2num(events)
                counts = list(range(1, len(events) + 1))
                self.ax.plot_date(times, counts, '-', label=dev_name, linewidth=2, marker='o')
                plotted_something = True
                
        if plotted_something:
            self.ax.legend(facecolor='#2b2b2b', edgecolor='#444444', labelcolor='white')
            
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.figure.autofmt_xdate()
        self.canvas.draw()

    def populate_port_combo(self, combo):
        combo.clear()
        try:
            from serial.tools import list_ports
            ports = list(list_ports.comports())
            
            candidates = []
            for p in ports:
                dev = getattr(p, "device", "")
                desc = getattr(p, "description", "")
                hwid = getattr(p, "hwid", "")
                
                if any(k in dev or k in desc or k in hwid for k in ("ACM", "USB", "Arduino", "Feather", "Adafruit", "CDC")):
                    candidates.append(dev)
            
            if not candidates:
                candidates = [p.device for p in ports]
                
            for p in candidates:
                combo.addItem(p)
                
        except Exception:
            pass
        combo.setEditable(True)

    def remove_device(self, device):
        if len(self.fed_devices) <= 1: return
        device['timer'].stop()
        worker = device.get('tracker_worker')
        if worker:
            worker.stop()
        self.devices_layout.removeWidget(device['box'])
        device['box'].deleteLater()
        if device in self.fed_devices:
            self.fed_devices.remove(device)
        self.update_remove_buttons()

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
        for dev in self.fed_devices:
            self.populate_port_combo(dev['port_combo'])

    def start_all(self):
        ms = self.get_global_interval_ms()
        for dev in self.fed_devices:
            if not dev['timer'].isActive():
                dev['timer'].start(ms)
                self.do_device_sync(dev)

    def stop_all(self):
        for dev in self.fed_devices:
            dev['timer'].stop()

    def sync_all(self):
        for dev in self.fed_devices:
            self.do_device_sync(dev)
