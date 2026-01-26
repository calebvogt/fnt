"""
PyQt5 GUI for USV Detection.

Provides a user-friendly interface for batch processing audio files
to detect ultrasonic vocalizations.
"""

import os
import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QGroupBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QTextEdit, QComboBox, QCheckBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from .usv_detector import (
    USVDetectorConfig, DSPDetector, batch_process,
    get_prairie_vole_config, generate_summary
)


class DetectionWorker(QThread):
    """Worker thread for running detection without blocking the GUI."""

    progress = pyqtSignal(str, int, int)  # filename, current, total
    finished = pyqtSignal(list)  # results
    error = pyqtSignal(str)  # error message

    def __init__(self, input_folder: str, output_folder: str, config: USVDetectorConfig):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.config = config
        self._is_cancelled = False

    def run(self):
        """Run the detection process."""
        try:
            def progress_callback(filename, current, total):
                if self._is_cancelled:
                    raise InterruptedError("Processing cancelled by user")
                self.progress.emit(filename, current, total)

            results = batch_process(
                self.input_folder,
                config=self.config,
                output_folder=self.output_folder,
                progress_callback=progress_callback
            )

            if not self._is_cancelled:
                self.finished.emit(results)

        except InterruptedError:
            pass  # Cancelled by user
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        """Cancel the processing."""
        self._is_cancelled = True


class USVDetectorWindow(QWidget):
    """Main window for USV detection GUI."""

    def __init__(self):
        super().__init__()
        self.input_folder = None
        self.output_folder = None
        self.worker = None
        self.results = []
        self.initUI()

    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle("USV Detector - FNT")
        self.setGeometry(100, 100, 900, 700)

        # Apply dark theme styling (matching FNT style)
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
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-height: 20px;
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
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 8px;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                font-family: Consolas, Monaco, monospace;
            }
            QProgressBar {
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            QTableWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                gridline-color: #3f3f3f;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #2b2b2b;
                color: #cccccc;
                padding: 5px;
                border: 1px solid #3f3f3f;
            }
            QTabWidget::pane {
                border: 1px solid #3f3f3f;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #2b2b2b;
                color: #cccccc;
                padding: 8px 16px;
                border: 1px solid #3f3f3f;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
                color: white;
            }
        """)

        layout = QVBoxLayout()

        # Title
        title = QLabel("USV Detector")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        layout.addWidget(title)

        # Description
        desc = QLabel("Automatic detection of ultrasonic vocalizations from audio files")
        desc.setFont(QFont("Arial", 10))
        desc.setStyleSheet("color: #999999; margin-bottom: 10px; font-style: italic;")
        layout.addWidget(desc)

        # Tab widget for main content
        tabs = QTabWidget()

        # === Processing Tab ===
        processing_tab = QWidget()
        processing_layout = QVBoxLayout()

        # Folder selection group
        folder_group = QGroupBox("Folders")
        folder_layout = QVBoxLayout()

        # Input folder
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Folder:"))
        self.lbl_input = QLabel("No folder selected")
        self.lbl_input.setStyleSheet("color: #999999; font-style: italic;")
        input_layout.addWidget(self.lbl_input, 1)
        self.btn_input = QPushButton("Browse...")
        self.btn_input.clicked.connect(self.select_input_folder)
        input_layout.addWidget(self.btn_input)
        folder_layout.addLayout(input_layout)

        # Output folder
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.lbl_output = QLabel("Same as input")
        self.lbl_output.setStyleSheet("color: #999999; font-style: italic;")
        output_layout.addWidget(self.lbl_output, 1)
        self.btn_output = QPushButton("Browse...")
        self.btn_output.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.btn_output)
        folder_layout.addLayout(output_layout)

        folder_group.setLayout(folder_layout)
        processing_layout.addWidget(folder_group)

        # Parameters group
        params_group = QGroupBox("Detection Parameters")
        params_layout = QVBoxLayout()

        # Species preset
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Species Preset:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(["Prairie Vole", "Mouse", "Rat", "Custom"])
        self.combo_preset.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.combo_preset)
        preset_layout.addStretch()
        params_layout.addLayout(preset_layout)

        # Frequency range
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Min Frequency (Hz):"))
        self.spin_min_freq = QSpinBox()
        self.spin_min_freq.setRange(1000, 150000)
        self.spin_min_freq.setSingleStep(1000)
        self.spin_min_freq.setValue(25000)
        freq_layout.addWidget(self.spin_min_freq)

        freq_layout.addWidget(QLabel("Max Frequency (Hz):"))
        self.spin_max_freq = QSpinBox()
        self.spin_max_freq.setRange(1000, 150000)
        self.spin_max_freq.setSingleStep(1000)
        self.spin_max_freq.setValue(65000)
        freq_layout.addWidget(self.spin_max_freq)
        freq_layout.addStretch()
        params_layout.addLayout(freq_layout)

        # Detection threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Energy Threshold (dB above noise):"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(1.0, 30.0)
        self.spin_threshold.setSingleStep(0.5)
        self.spin_threshold.setValue(10.0)
        thresh_layout.addWidget(self.spin_threshold)
        thresh_layout.addStretch()
        params_layout.addLayout(thresh_layout)

        # Duration limits
        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("Min Duration (ms):"))
        self.spin_min_dur = QDoubleSpinBox()
        self.spin_min_dur.setRange(1.0, 100.0)
        self.spin_min_dur.setValue(5.0)
        dur_layout.addWidget(self.spin_min_dur)

        dur_layout.addWidget(QLabel("Max Duration (ms):"))
        self.spin_max_dur = QDoubleSpinBox()
        self.spin_max_dur.setRange(10.0, 5000.0)
        self.spin_max_dur.setValue(300.0)
        dur_layout.addWidget(self.spin_max_dur)
        dur_layout.addStretch()
        params_layout.addLayout(dur_layout)

        # Classify calls checkbox
        self.chk_classify = QCheckBox("Classify call types (sweep_up, inverted_u, flat, etc.)")
        self.chk_classify.setChecked(True)
        params_layout.addWidget(self.chk_classify)

        params_group.setLayout(params_layout)
        processing_layout.addWidget(params_group)

        # Progress group
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #999999;")
        progress_layout.addWidget(self.lbl_status)

        progress_group.setLayout(progress_layout)
        processing_layout.addWidget(progress_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_process = QPushButton("Start Detection")
        self.btn_process.clicked.connect(self.start_processing)
        self.btn_process.setEnabled(False)
        btn_layout.addWidget(self.btn_process)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_processing)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background-color: #d41a1a;
            }
            QPushButton:hover {
                background-color: #a01515;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
        """)
        btn_layout.addWidget(self.btn_cancel)

        btn_layout.addStretch()
        processing_layout.addLayout(btn_layout)

        # Log output
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(150)
        log_layout.addWidget(self.txt_log)
        log_group.setLayout(log_layout)
        processing_layout.addWidget(log_group)

        processing_tab.setLayout(processing_layout)
        tabs.addTab(processing_tab, "Processing")

        # === Results Tab ===
        results_tab = QWidget()
        results_layout = QVBoxLayout()

        # Summary labels
        summary_layout = QHBoxLayout()
        self.lbl_total_files = QLabel("Files: 0")
        self.lbl_total_calls = QLabel("Total Calls: 0")
        self.lbl_avg_calls = QLabel("Avg Calls/File: 0")
        summary_layout.addWidget(self.lbl_total_files)
        summary_layout.addWidget(self.lbl_total_calls)
        summary_layout.addWidget(self.lbl_avg_calls)
        summary_layout.addStretch()
        results_layout.addLayout(summary_layout)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Filename", "Status", "Calls", "Duration (s)", "Calls/min"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)

        # Export button
        export_layout = QHBoxLayout()
        self.btn_open_folder = QPushButton("Open Output Folder")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setEnabled(False)
        export_layout.addWidget(self.btn_open_folder)
        export_layout.addStretch()
        results_layout.addLayout(export_layout)

        results_tab.setLayout(results_layout)
        tabs.addTab(results_tab, "Results")

        layout.addWidget(tabs)
        self.setLayout(layout)

    def log(self, message: str):
        """Add message to log."""
        self.txt_log.append(message)
        # Auto-scroll to bottom
        scrollbar = self.txt_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def select_input_folder(self):
        """Select input folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", ""
        )
        if folder:
            self.input_folder = folder
            self.lbl_input.setText(folder)
            self.lbl_input.setStyleSheet("color: #cccccc;")
            self.btn_process.setEnabled(True)

            # Count WAV files
            wav_count = len(list(Path(folder).glob("*.wav")))
            self.log(f"Selected input folder: {folder}")
            self.log(f"Found {wav_count} WAV files")

    def select_output_folder(self):
        """Select output folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ""
        )
        if folder:
            self.output_folder = folder
            self.lbl_output.setText(folder)
            self.lbl_output.setStyleSheet("color: #cccccc;")
            self.log(f"Selected output folder: {folder}")

    def on_preset_changed(self, preset: str):
        """Handle species preset change."""
        if preset == "Prairie Vole":
            self.spin_min_freq.setValue(25000)
            self.spin_max_freq.setValue(65000)
            self.spin_threshold.setValue(10.0)
            self.spin_min_dur.setValue(5.0)
            self.spin_max_dur.setValue(300.0)
        elif preset == "Mouse":
            self.spin_min_freq.setValue(30000)
            self.spin_max_freq.setValue(100000)
            self.spin_threshold.setValue(12.0)
            self.spin_min_dur.setValue(3.0)
            self.spin_max_dur.setValue(500.0)
        elif preset == "Rat":
            self.spin_min_freq.setValue(18000)
            self.spin_max_freq.setValue(80000)
            self.spin_threshold.setValue(10.0)
            self.spin_min_dur.setValue(10.0)
            self.spin_max_dur.setValue(2000.0)
        # Custom: leave values as-is

    def get_config(self) -> USVDetectorConfig:
        """Get configuration from GUI values."""
        return USVDetectorConfig(
            min_freq_hz=self.spin_min_freq.value(),
            max_freq_hz=self.spin_max_freq.value(),
            energy_threshold_db=self.spin_threshold.value(),
            min_duration_ms=self.spin_min_dur.value(),
            max_duration_ms=self.spin_max_dur.value(),
            classify_calls=self.chk_classify.isChecked(),
        )

    def start_processing(self):
        """Start the detection process."""
        if not self.input_folder:
            QMessageBox.warning(self, "No Input", "Please select an input folder.")
            return

        # Validate parameters
        config = self.get_config()
        errors = config.validate()
        if errors:
            QMessageBox.warning(self, "Invalid Parameters", "\n".join(errors))
            return

        # Update UI
        self.btn_process.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Starting...")
        self.results = []
        self.results_table.setRowCount(0)

        output = self.output_folder or self.input_folder
        self.log(f"Starting detection with config: {config.min_freq_hz}-{config.max_freq_hz} Hz")

        # Start worker thread
        self.worker = DetectionWorker(self.input_folder, output, config)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def cancel_processing(self):
        """Cancel the current processing."""
        if self.worker:
            self.worker.cancel()
            self.lbl_status.setText("Cancelling...")
            self.btn_cancel.setEnabled(False)

    def on_progress(self, filename: str, current: int, total: int):
        """Handle progress update."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.lbl_status.setText(f"Processing {current}/{total}: {filename}")

    def on_finished(self, results: list):
        """Handle processing completion."""
        self.results = results
        self.worker = None

        # Update UI
        self.btn_process.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_open_folder.setEnabled(True)
        self.progress_bar.setValue(self.progress_bar.maximum())

        # Calculate summary
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        total_calls = sum(r.get('num_calls', 0) for r in successful)

        self.lbl_status.setText(f"Complete: {len(successful)} files processed, {total_calls} calls detected")
        self.log(f"Processing complete!")
        self.log(f"  Files processed: {len(successful)}")
        self.log(f"  Files failed: {len(failed)}")
        self.log(f"  Total calls: {total_calls}")

        # Update results table
        self.update_results_table(results)

        # Update summary labels
        self.lbl_total_files.setText(f"Files: {len(successful)}")
        self.lbl_total_calls.setText(f"Total Calls: {total_calls}")
        if successful:
            avg_calls = total_calls / len(successful)
            self.lbl_avg_calls.setText(f"Avg Calls/File: {avg_calls:.1f}")

        # Show completion message
        QMessageBox.information(
            self, "Complete",
            f"Detection complete!\n\n"
            f"Files processed: {len(successful)}\n"
            f"Total calls detected: {total_calls}\n\n"
            f"Results saved to:\n{self.output_folder or self.input_folder}"
        )

    def on_error(self, error_msg: str):
        """Handle processing error."""
        self.worker = None
        self.btn_process.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("Error")
        self.log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", f"Processing failed:\n{error_msg}")

    def update_results_table(self, results: list):
        """Update the results table."""
        self.results_table.setRowCount(len(results))

        for i, result in enumerate(results):
            filename = os.path.basename(result.get('input_file', 'unknown'))
            self.results_table.setItem(i, 0, QTableWidgetItem(filename))

            if 'error' in result:
                self.results_table.setItem(i, 1, QTableWidgetItem("Error"))
                self.results_table.setItem(i, 2, QTableWidgetItem("-"))
                self.results_table.setItem(i, 3, QTableWidgetItem("-"))
                self.results_table.setItem(i, 4, QTableWidgetItem("-"))
            else:
                self.results_table.setItem(i, 1, QTableWidgetItem("Success"))
                self.results_table.setItem(i, 2, QTableWidgetItem(str(result.get('num_calls', 0))))

                summary = result.get('summary', {})
                duration = summary.get('file_duration_s', 0)
                calls_per_min = summary.get('calls_per_minute', 0)

                self.results_table.setItem(i, 3, QTableWidgetItem(f"{duration:.1f}"))
                self.results_table.setItem(i, 4, QTableWidgetItem(f"{calls_per_min:.2f}"))

    def open_output_folder(self):
        """Open the output folder in file explorer."""
        folder = self.output_folder or self.input_folder
        if folder and os.path.exists(folder):
            import subprocess
            import platform
            if platform.system() == "Darwin":  # macOS
                subprocess.run(["open", folder])
            elif platform.system() == "Windows":
                subprocess.run(["explorer", folder])
            else:  # Linux
                subprocess.run(["xdg-open", folder])


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = USVDetectorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
