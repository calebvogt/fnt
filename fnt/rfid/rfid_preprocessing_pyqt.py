"""
RFID Preprocessing Tool (PyQt5 GUI)

Universal RFID preprocessing pipeline with configurable parameters:
- Raw RFID data processing
- Movement bout detection
- GBI matrix generation
- Social network analysis
- Edgelist creation
- Displacement detection
- Hinde index calculation

Standalone usage:
    python rfid_preprocessing_pyqt.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QFileDialog, QGroupBox, QCheckBox, QComboBox,
    QMessageBox, QSpinBox, QDoubleSpinBox, QProgressBar, QGridLayout,
    QLineEdit, QFormLayout, QScrollArea, QTableWidget, QTableWidgetItem,
    QDialog, QDialogButtonBox, QListWidget
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .config import RFIDConfig, get_default_config, get_available_templates, ConfigManager
from .rfid_worker import RFIDProcessingWorker


class RFIDPreprocessingWindow(QWidget):
    """Main window for RFID preprocessing tool."""

    def __init__(self):
        super().__init__()
        self.config = get_default_config("8_zone_paddock")
        self.worker = None

        self.init_ui()
        self.apply_dark_theme()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("RFID Preprocessing Tool")
        self.setMinimumSize(900, 800)

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title = QLabel("RFID Preprocessing Pipeline")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        # Create scroll area for main content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Configuration section
        scroll_layout.addWidget(self.create_config_section())

        # File selection section
        scroll_layout.addWidget(self.create_file_section())

        # Pipeline steps section
        scroll_layout.addWidget(self.create_pipeline_section())

        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        # Action buttons
        main_layout.addWidget(self.create_action_section())

        # Progress section
        main_layout.addWidget(self.create_progress_section())

        # Log output
        main_layout.addWidget(self.create_log_section())

        self.setLayout(main_layout)

    def create_config_section(self) -> QGroupBox:
        """Create configuration section."""
        group = QGroupBox("Configuration")
        layout = QVBoxLayout()

        # Profile selection
        profile_layout = QHBoxLayout()
        profile_layout.addWidget(QLabel("Template:"))

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(get_available_templates())
        self.profile_combo.currentTextChanged.connect(self.load_template)
        profile_layout.addWidget(self.profile_combo)

        self.load_btn = QPushButton("Load Profile...")
        self.load_btn.clicked.connect(self.load_config_file)
        profile_layout.addWidget(self.load_btn)

        self.save_btn = QPushButton("Save Profile...")
        self.save_btn.clicked.connect(self.save_config_file)
        profile_layout.addWidget(self.save_btn)

        profile_layout.addStretch()
        layout.addLayout(profile_layout)

        # Temporal parameters
        temp_group = QGroupBox("Temporal Parameters")
        temp_layout = QFormLayout()

        self.bout_threshold_spin = QDoubleSpinBox()
        self.bout_threshold_spin.setRange(1, 300)
        self.bout_threshold_spin.setValue(self.config.bout_threshold_sec)
        self.bout_threshold_spin.setSuffix(" sec")
        temp_layout.addRow("Bout Threshold:", self.bout_threshold_spin)

        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.1, 10)
        self.min_duration_spin.setValue(self.config.min_duration_sec)
        self.min_duration_spin.setSuffix(" sec")
        temp_layout.addRow("Min Duration:", self.min_duration_spin)

        self.day_origin_edit = QLineEdit(self.config.day_origin_time)
        temp_layout.addRow("Day Origin Time:", self.day_origin_edit)

        day_layout = QHBoxLayout()
        self.day_start_spin = QSpinBox()
        self.day_start_spin.setRange(1, 365)
        self.day_start_spin.setValue(self.config.analysis_days[0])
        day_layout.addWidget(QLabel("From:"))
        day_layout.addWidget(self.day_start_spin)

        self.day_end_spin = QSpinBox()
        self.day_end_spin.setRange(1, 365)
        self.day_end_spin.setValue(self.config.analysis_days[1])
        day_layout.addWidget(QLabel("To:"))
        day_layout.addWidget(self.day_end_spin)
        day_layout.addStretch()

        temp_layout.addRow("Analysis Days:", day_layout)
        temp_group.setLayout(temp_layout)
        layout.addWidget(temp_group)

        # Spatial parameters
        spatial_group = QGroupBox("Spatial Configuration")
        spatial_layout = QFormLayout()

        self.num_zones_spin = QSpinBox()
        self.num_zones_spin.setRange(1, 100)
        self.num_zones_spin.setValue(self.config.num_zones)
        spatial_layout.addRow("Number of Zones:", self.num_zones_spin)

        self.num_antennas_spin = QSpinBox()
        self.num_antennas_spin.setRange(1, 200)
        self.num_antennas_spin.setValue(self.config.num_antennas)
        spatial_layout.addRow("Number of Antennas:", self.num_antennas_spin)

        self.edit_antenna_btn = QPushButton("Edit Antenna-Zone Map...")
        self.edit_antenna_btn.clicked.connect(self.edit_antenna_map)
        spatial_layout.addRow("", self.edit_antenna_btn)

        self.edit_coords_btn = QPushButton("Edit Zone Coordinates...")
        self.edit_coords_btn.clicked.connect(self.edit_zone_coords)
        spatial_layout.addRow("", self.edit_coords_btn)

        spatial_group.setLayout(spatial_layout)
        layout.addWidget(spatial_group)

        # Trial configuration
        trial_group = QGroupBox("Trial Configuration")
        trial_layout = QVBoxLayout()

        trial_list_layout = QHBoxLayout()
        self.trial_list = QListWidget()
        self.trial_list.addItems(self.config.trial_ids)
        trial_list_layout.addWidget(self.trial_list)

        trial_btn_layout = QVBoxLayout()
        add_trial_btn = QPushButton("Add Trial...")
        add_trial_btn.clicked.connect(self.add_trial)
        trial_btn_layout.addWidget(add_trial_btn)

        remove_trial_btn = QPushButton("Remove Trial")
        remove_trial_btn.clicked.connect(self.remove_trial)
        trial_btn_layout.addWidget(remove_trial_btn)

        trial_btn_layout.addStretch()
        trial_list_layout.addLayout(trial_btn_layout)

        trial_layout.addLayout(trial_list_layout)

        self.edit_readers_btn = QPushButton("Edit Trial-Reader Map...")
        self.edit_readers_btn.clicked.connect(self.edit_trial_readers)
        trial_layout.addWidget(self.edit_readers_btn)

        trial_group.setLayout(trial_layout)
        layout.addWidget(trial_group)

        group.setLayout(layout)
        return group

    def create_file_section(self) -> QGroupBox:
        """Create file selection section."""
        group = QGroupBox("Input/Output Files")
        layout = QFormLayout()

        # RFID data directory
        rfid_layout = QHBoxLayout()
        self.rfid_dir_edit = QLineEdit(self.config.input_dir)
        rfid_layout.addWidget(self.rfid_dir_edit)
        rfid_browse_btn = QPushButton("Browse...")
        rfid_browse_btn.clicked.connect(lambda: self.browse_directory(self.rfid_dir_edit))
        rfid_layout.addWidget(rfid_browse_btn)
        layout.addRow("RFID Data Directory:", rfid_layout)

        # Metadata file
        meta_layout = QHBoxLayout()
        self.metadata_edit = QLineEdit(self.config.metadata_file_path)
        meta_layout.addWidget(self.metadata_edit)
        meta_browse_btn = QPushButton("Browse...")
        meta_browse_btn.clicked.connect(lambda: self.browse_file(self.metadata_edit))
        meta_layout.addWidget(meta_browse_btn)
        layout.addRow("Metadata File:", meta_layout)

        # Output directory
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit(self.config.output_dir)
        output_layout.addWidget(self.output_dir_edit)
        output_browse_btn = QPushButton("Browse...")
        output_browse_btn.clicked.connect(lambda: self.browse_directory(self.output_dir_edit))
        output_layout.addWidget(output_browse_btn)
        layout.addRow("Output Directory:", output_layout)

        group.setLayout(layout)
        return group

    def create_pipeline_section(self) -> QGroupBox:
        """Create pipeline steps section."""
        group = QGroupBox("Pipeline Steps")
        layout = QVBoxLayout()

        self.step_checks = {}
        steps = [
            ("raw_rfid", "1. Raw RFID Processing", True),
            ("bout_detection", "2. Movement Bout Detection", True),
            ("gbi", "3. GBI Matrix Generation", True),
            ("social_network", "4. Social Network Analysis", True),
            ("edgelist", "5. Edgelist Creation", True),
            ("displacement", "6. Displacement Detection", True),
            ("hinde_index", "7. Hinde Index Calculation", True)
        ]

        for key, label, default in steps:
            checkbox = QCheckBox(label)
            checkbox.setChecked(default)
            self.step_checks[key] = checkbox
            layout.addWidget(checkbox)

        group.setLayout(layout)
        return group

    def create_action_section(self) -> QWidget:
        """Create action buttons section."""
        widget = QWidget()
        layout = QHBoxLayout()

        self.validate_btn = QPushButton("Validate Configuration")
        self.validate_btn.clicked.connect(self.validate_config)
        layout.addWidget(self.validate_btn)

        self.process_btn = QPushButton("Process RFID Data")
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_progress_section(self) -> QWidget:
        """Create progress section."""
        widget = QWidget()
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        widget.setLayout(layout)
        return widget

    def create_log_section(self) -> QGroupBox:
        """Create log output section."""
        group = QGroupBox("Log Output")
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)

        group.setLayout(layout)
        return group

    def log(self, message: str):
        """Add message to log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def browse_directory(self, line_edit: QLineEdit):
        """Browse for directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            line_edit.setText(directory)

    def browse_file(self, line_edit: QLineEdit):
        """Browse for file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "",
            "Excel/CSV Files (*.xlsx *.xls *.csv);;All Files (*)"
        )
        if file_path:
            line_edit.setText(file_path)

    def load_template(self, template_name: str):
        """Load configuration template."""
        try:
            self.config = get_default_config(template_name)
            self.update_ui_from_config()
            self.log(f"Loaded template: {template_name}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load template: {e}")

    def load_config_file(self):
        """Load configuration from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                self.config = ConfigManager.load_config(file_path)
                self.update_ui_from_config()
                self.log(f"Loaded configuration from: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load configuration: {e}")

    def save_config_file(self):
        """Save configuration to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", "", "JSON Files (*.json)"
        )
        if file_path:
            try:
                self.update_config_from_ui()
                ConfigManager.save_config(self.config, file_path)
                self.log(f"Saved configuration to: {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save configuration: {e}")

    def update_ui_from_config(self):
        """Update UI elements from config."""
        self.bout_threshold_spin.setValue(self.config.bout_threshold_sec)
        self.min_duration_spin.setValue(self.config.min_duration_sec)
        self.day_origin_edit.setText(self.config.day_origin_time)
        self.day_start_spin.setValue(self.config.analysis_days[0])
        self.day_end_spin.setValue(self.config.analysis_days[1])

        self.num_zones_spin.setValue(self.config.num_zones)
        self.num_antennas_spin.setValue(self.config.num_antennas)

        self.trial_list.clear()
        self.trial_list.addItems(self.config.trial_ids)

        self.rfid_dir_edit.setText(self.config.input_dir)
        self.metadata_edit.setText(self.config.metadata_file_path)
        self.output_dir_edit.setText(self.config.output_dir)

    def update_config_from_ui(self):
        """Update config from UI elements."""
        self.config.bout_threshold_sec = self.bout_threshold_spin.value()
        self.config.min_duration_sec = self.min_duration_spin.value()
        self.config.day_origin_time = self.day_origin_edit.text()
        self.config.analysis_days = (
            self.day_start_spin.value(),
            self.day_end_spin.value()
        )

        self.config.num_zones = self.num_zones_spin.value()
        self.config.num_antennas = self.num_antennas_spin.value()

        self.config.trial_ids = [
            self.trial_list.item(i).text()
            for i in range(self.trial_list.count())
        ]

        self.config.input_dir = self.rfid_dir_edit.text()
        self.config.metadata_file_path = self.metadata_edit.text()
        self.config.output_dir = self.output_dir_edit.text()

    def add_trial(self):
        """Add a new trial."""
        from PyQt5.QtWidgets import QInputDialog
        trial_id, ok = QInputDialog.getText(self, "Add Trial", "Enter trial ID:")
        if ok and trial_id:
            self.trial_list.addItem(trial_id)

    def remove_trial(self):
        """Remove selected trial."""
        current_row = self.trial_list.currentRow()
        if current_row >= 0:
            self.trial_list.takeItem(current_row)

    def edit_antenna_map(self):
        """Edit antenna-zone mapping."""
        dialog = AntennaMapDialog(self.config.antenna_zone_map, self)
        if dialog.exec_() == QDialog.Accepted:
            self.config.antenna_zone_map = dialog.get_mapping()
            self.log("Updated antenna-zone map")

    def edit_zone_coords(self):
        """Edit zone coordinates."""
        dialog = ZoneCoordsDialog(self.config.zone_coordinates, self)
        if dialog.exec_() == QDialog.Accepted:
            self.config.zone_coordinates = dialog.get_coordinates()
            self.log("Updated zone coordinates")

    def edit_trial_readers(self):
        """Edit trial-reader mapping."""
        dialog = TrialReaderDialog(self.config.trial_reader_map, self)
        if dialog.exec_() == QDialog.Accepted:
            self.config.trial_reader_map = dialog.get_mapping()
            self.log("Updated trial-reader map")

    def validate_config(self):
        """Validate current configuration."""
        self.update_config_from_ui()

        # Validate configuration
        is_valid, errors = ConfigManager.validate_config(self.config)

        if not is_valid:
            error_msg = "Configuration errors:\n\n" + "\n".join(f"• {e}" for e in errors)
            QMessageBox.warning(self, "Invalid Configuration", error_msg)
            self.log("Configuration validation failed")
            return False

        # Validate paths
        is_valid, errors = ConfigManager.validate_paths(self.config)

        if not is_valid:
            error_msg = "Path errors:\n\n" + "\n".join(f"• {e}" for e in errors)
            QMessageBox.warning(self, "Invalid Paths", error_msg)
            self.log("Path validation failed")
            return False

        QMessageBox.information(self, "Success", "Configuration is valid!")
        self.log("Configuration validated successfully")
        return True

    def start_processing(self):
        """Start RFID processing."""
        # Validate first
        if not self.validate_config():
            return

        # Get enabled steps
        enabled_steps = {
            key: checkbox.isChecked()
            for key, checkbox in self.step_checks.items()
        }

        if not any(enabled_steps.values()):
            QMessageBox.warning(self, "Error", "Please enable at least one pipeline step")
            return

        # Create and start worker
        self.worker = RFIDProcessingWorker(self.config, enabled_steps)
        self.worker.progress_updated.connect(self.on_progress_updated)
        self.worker.step_completed.connect(self.on_step_completed)
        self.worker.processing_complete.connect(self.on_processing_complete)
        self.worker.error_occurred.connect(self.on_error)

        # Disable buttons
        self.process_btn.setEnabled(False)
        self.validate_btn.setEnabled(False)

        self.log("Starting RFID processing pipeline...")
        self.worker.start()

    def on_progress_updated(self, percentage: int, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        self.log(message)

    def on_step_completed(self, step_name: str, output_file: str):
        """Handle step completion."""
        self.log(f"✓ Completed: {step_name} → {output_file}")

    def on_processing_complete(self, success: bool, message: str):
        """Handle processing completion."""
        self.process_btn.setEnabled(True)
        self.validate_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
            self.log("✓ Pipeline completed successfully!")
        else:
            QMessageBox.critical(self, "Error", message)
            self.log("✗ Pipeline failed")

    def on_error(self, error_msg: str):
        """Handle error."""
        self.log(f"ERROR: {error_msg}")

    def apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
                font-family: Arial;
                font-size: 10pt;
            }
            QGroupBox {
                border: 1px solid #3f3f3f;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbe;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3f3f3f;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
            }
            QTextEdit, QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)


class AntennaMapDialog(QDialog):
    """Dialog for editing antenna-zone mapping."""

    def __init__(self, antenna_map: dict, parent=None):
        super().__init__(parent)
        self.antenna_map = antenna_map.copy()
        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        self.setWindowTitle("Edit Antenna-Zone Map")
        self.setMinimumSize(400, 500)

        layout = QVBoxLayout()

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Antenna ID", "Zone ID"])

        # Populate table
        self.table.setRowCount(len(self.antenna_map))
        for i, (antenna, zone) in enumerate(sorted(self.antenna_map.items())):
            self.table.setItem(i, 0, QTableWidgetItem(str(antenna)))
            self.table.setItem(i, 1, QTableWidgetItem(str(zone)))

        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()
        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(self.add_row)
        button_layout.addWidget(add_btn)

        remove_btn = QPushButton("Remove Row")
        remove_btn.clicked.connect(self.remove_row)
        button_layout.addWidget(remove_btn)

        layout.addLayout(button_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def add_row(self):
        """Add new row."""
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)

    def remove_row(self):
        """Remove selected row."""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)

    def get_mapping(self) -> dict:
        """Get mapping from table."""
        mapping = {}
        for i in range(self.table.rowCount()):
            antenna_item = self.table.item(i, 0)
            zone_item = self.table.item(i, 1)

            if antenna_item and zone_item:
                try:
                    antenna = int(antenna_item.text())
                    zone = int(zone_item.text())
                    mapping[antenna] = zone
                except ValueError:
                    pass

        return mapping


class ZoneCoordsDialog(QDialog):
    """Dialog for editing zone coordinates."""

    def __init__(self, zone_coords: list, parent=None):
        super().__init__(parent)
        self.zone_coords = zone_coords.copy()
        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        self.setWindowTitle("Edit Zone Coordinates")
        self.setMinimumSize(500, 500)

        layout = QVBoxLayout()

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Zone", "X", "Y", "Location"])

        # Populate table
        self.table.setRowCount(len(self.zone_coords))
        for i, coord in enumerate(self.zone_coords):
            self.table.setItem(i, 0, QTableWidgetItem(str(coord['zone'])))
            self.table.setItem(i, 1, QTableWidgetItem(str(coord['x'])))
            self.table.setItem(i, 2, QTableWidgetItem(str(coord['y'])))
            self.table.setItem(i, 3, QTableWidgetItem(str(coord['location'])))

        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()
        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(self.add_row)
        button_layout.addWidget(add_btn)

        remove_btn = QPushButton("Remove Row")
        remove_btn.clicked.connect(self.remove_row)
        button_layout.addWidget(remove_btn)

        layout.addLayout(button_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def add_row(self):
        """Add new row."""
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)

    def remove_row(self):
        """Remove selected row."""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)

    def get_coordinates(self) -> list:
        """Get coordinates from table."""
        coords = []
        for i in range(self.table.rowCount()):
            zone_item = self.table.item(i, 0)
            x_item = self.table.item(i, 1)
            y_item = self.table.item(i, 2)
            loc_item = self.table.item(i, 3)

            if all([zone_item, x_item, y_item, loc_item]):
                try:
                    coord = {
                        'zone': int(zone_item.text()),
                        'x': float(x_item.text()),
                        'y': float(y_item.text()),
                        'location': loc_item.text()
                    }
                    coords.append(coord)
                except ValueError:
                    pass

        return coords


class TrialReaderDialog(QDialog):
    """Dialog for editing trial-reader mapping."""

    def __init__(self, trial_reader_map: dict, parent=None):
        super().__init__(parent)
        self.trial_reader_map = trial_reader_map.copy()
        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        self.setWindowTitle("Edit Trial-Reader Map")
        self.setMinimumSize(400, 400)

        layout = QVBoxLayout()

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Trial ID", "Reader ID"])

        # Populate table
        self.table.setRowCount(len(self.trial_reader_map))
        for i, (trial, reader) in enumerate(sorted(self.trial_reader_map.items())):
            self.table.setItem(i, 0, QTableWidgetItem(str(trial)))
            self.table.setItem(i, 1, QTableWidgetItem(str(reader)))

        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()
        add_btn = QPushButton("Add Row")
        add_btn.clicked.connect(self.add_row)
        button_layout.addWidget(add_btn)

        remove_btn = QPushButton("Remove Row")
        remove_btn.clicked.connect(self.remove_row)
        button_layout.addWidget(remove_btn)

        layout.addLayout(button_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def add_row(self):
        """Add new row."""
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)

    def remove_row(self):
        """Remove selected row."""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)

    def get_mapping(self) -> dict:
        """Get mapping from table."""
        mapping = {}
        for i in range(self.table.rowCount()):
            trial_item = self.table.item(i, 0)
            reader_item = self.table.item(i, 1)

            if trial_item and reader_item:
                try:
                    trial = trial_item.text()
                    reader = int(reader_item.text())
                    mapping[trial] = reader
                except ValueError:
                    pass

        return mapping


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RFIDPreprocessingWindow()
    window.show()
    sys.exit(app.exec_())
