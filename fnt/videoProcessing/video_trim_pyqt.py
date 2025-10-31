"""
Video Trimming Tool - PyQt5 Implementation

Interactive video trimming with preview and duration selection.
Allows users to:
- Select start frame using slider and adjustment buttons
- Choose duration from preset options or custom duration
- Preview the start frame before trimming
"""

import os
import cv2
import subprocess
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QPushButton, QFileDialog, QMessageBox, QGroupBox,
    QComboBox, QSpinBox, QGridLayout, QFrame, QApplication,
    QLineEdit, QTextEdit, QScrollArea, QWidget
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont


class TrimWorker(QThread):
    """Worker thread for running FFmpeg trim operations"""
    progress = pyqtSignal(str)  # For log output
    finished = pyqtSignal(bool, str)  # Success, message
    
    def __init__(self, command, output_file):
        super().__init__()
        self.command = command
        self.output_file = output_file
    
    def run(self):
        """Run FFmpeg command in background thread"""
        try:
            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            
            # Stream output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.progress.emit(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                self.finished.emit(True, f"✅ Video trimmed successfully!\n\nSaved as:\n{self.output_file}")
            else:
                self.finished.emit(False, f"❌ FFmpeg failed with error code {process.returncode}")
                
        except Exception as e:
            self.finished.emit(False, f"❌ An error occurred:\n{str(e)}")


class VideoTrimDialog(QDialog):
    """PyQt5 dialog for interactive video trimming with duration selection"""
    
    def __init__(self, video_path=None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.cap = None
        
        # Video properties (will be set when video is loaded)
        self.total_frames = 0
        self.fps = 0
        self.duration = 0
        self.width = 0
        self.height = 0
        
        # Current frame position (in seconds)
        self.current_position = 0.0
        
        # Duration presets (in seconds)
        self.duration_presets = {
            "1 minute": 60,
            "5 minutes": 300,
            "10 minutes": 600,
            "20 minutes": 1200,
            "30 minutes": 1800,
            "1 hour": 3600,
            "Custom": -1
        }
        
        # Worker thread for background processing
        self.worker = None
        
        self.init_ui()
        
        # Load video if provided
        if video_path:
            self.load_video(video_path)
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Video Trimming Tool")
        self.setMinimumSize(900, 700)
        
        # Enable window resize with maximize button, remove help button
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        
        # Apply dark mode styling
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
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
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3f3f3f;
                height: 8px;
                background-color: #1e1e1e;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background-color: #0078d4;
                border: 1px solid #0078d4;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background-color: #106ebe;
            }
            QComboBox, QSpinBox, QLineEdit {
                padding: 5px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QComboBox::drop-down {
                border: none;
                background-color: #3f3f3f;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #cccccc;
                margin-right: 5px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
                border: 1px solid #3f3f3f;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #3f3f3f;
                border: 1px solid #3f3f3f;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #0078d4;
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("Video Trimming Tool")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        main_layout.addWidget(title)
        
        # Select Video button (smaller)
        select_button_layout = QHBoxLayout()
        select_button_layout.addStretch()
        
        self.select_video_btn = QPushButton("Select Video")
        self.select_video_btn.setFixedSize(150, 35)
        self.select_video_btn.clicked.connect(self.select_video_file)
        self.select_video_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-size: 12px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        select_button_layout.addWidget(self.select_video_btn)
        select_button_layout.addStretch()
        main_layout.addLayout(select_button_layout)
        
        # Video info
        self.info_label = QLabel("No video selected")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("color: #999999; padding: 5px; background-color: transparent;")
        main_layout.addWidget(self.info_label)
        
        # Create scrollable area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
        """)
        scroll_widget = QWidget()
        layout = QVBoxLayout()
        scroll_widget.setLayout(layout)
        
        # Preview section (larger)
        preview_group = QGroupBox("Start Frame Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black; border: 2px solid #0078d4;")
        self.preview_label.setMinimumSize(800, 450)
        preview_layout.addWidget(self.preview_label)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Position control section
        position_group = QGroupBox("Select Start Position")
        position_layout = QVBoxLayout()
        
        # Time display
        self.time_label = QLabel(f"Start Time: {self.format_time(0)}")
        self.time_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.time_label.setAlignment(Qt.AlignCenter)
        position_layout.addWidget(self.time_label)
        
        # Slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(100)  # Will be updated when video loads
        self.position_slider.setValue(0)
        self.position_slider.setEnabled(False)  # Disabled until video is loaded
        self.position_slider.valueChanged.connect(self.on_slider_changed)
        position_layout.addWidget(self.position_slider)
        
        # Adjustment buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        adjustments = [
            ("-60s", -60),
            ("-30s", -30),
            ("-10s", -10),
            ("-1s", -1),
            ("+1s", +1),
            ("+10s", +10),
            ("+30s", +30),
            ("+60s", +60),
        ]
        
        for label, seconds in adjustments:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, s=seconds: self.adjust_position(s))
            btn.setFixedWidth(70)
            button_layout.addWidget(btn)
        
        button_layout.addStretch()
        position_layout.addLayout(button_layout)
        
        position_group.setLayout(position_layout)
        layout.addWidget(position_group)
        
        # Duration selection section
        duration_group = QGroupBox("Select Duration")
        duration_layout = QHBoxLayout()
        
        duration_layout.addWidget(QLabel("Duration:"))
        
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(list(self.duration_presets.keys()))
        self.duration_combo.setCurrentText("10 minutes")  # Default
        self.duration_combo.setEnabled(False)  # Disabled until video is loaded
        self.duration_combo.currentTextChanged.connect(self.on_duration_changed)
        self.duration_combo.setMinimumWidth(150)
        duration_layout.addWidget(self.duration_combo)
        
        # Custom duration spinbox (initially hidden)
        self.custom_duration_label = QLabel("Custom Duration (seconds):")
        self.custom_duration_spinbox = QSpinBox()
        self.custom_duration_spinbox.setMinimum(1)
        self.custom_duration_spinbox.setMaximum(int(self.duration))
        self.custom_duration_spinbox.setValue(600)  # Default 10 minutes
        self.custom_duration_spinbox.setSuffix(" seconds")
        self.custom_duration_spinbox.setMinimumWidth(150)
        
        self.custom_duration_label.setVisible(False)
        self.custom_duration_spinbox.setVisible(False)
        
        duration_layout.addWidget(self.custom_duration_label)
        duration_layout.addWidget(self.custom_duration_spinbox)
        
        # End time display
        duration_layout.addStretch()
        self.end_time_label = QLabel("End Time: -- (Duration: --)")
        self.end_time_label.setFont(QFont("Arial", 10, QFont.Bold))
        duration_layout.addWidget(self.end_time_label)
        
        duration_group.setLayout(duration_layout)
        layout.addWidget(duration_group)
        
        # Output filename section
        output_group = QGroupBox("Output File")
        output_layout = QHBoxLayout()
        
        output_layout.addWidget(QLabel("Output filename:"))
        
        # Output filename field (will be populated when video is loaded)
        self.output_filename = QLineEdit()
        self.output_filename.setText("")
        self.output_filename.setEnabled(False)  # Disabled until video is loaded
        self.output_filename.setMinimumWidth(300)
        self.output_filename.setPlaceholderText("Enter output filename...")
        output_layout.addWidget(self.output_filename)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Progress and log section
        progress_group = QGroupBox("Processing Log")
        progress_group.setStyleSheet("QGroupBox { font-weight: bold; color: #0078d4; }")
        progress_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #dcdcdc;
                border: 1px solid #404040;
                padding: 5px;
            }
        """)
        self.log_text.setPlaceholderText("Processing output will appear here...")
        progress_layout.addWidget(self.log_text)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.addStretch()
        
        self.trim_button = QPushButton("Trim Video")
        self.trim_button.clicked.connect(self.trim_video)
        self.trim_button.setFixedSize(150, 40)
        self.trim_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        action_layout.addWidget(self.trim_button)
        
        self.cancel_button = QPushButton("Cancel Trim")
        self.cancel_button.clicked.connect(self.cancel_trim)
        self.cancel_button.setFixedSize(150, 40)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
        """)
        action_layout.addWidget(self.cancel_button)
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        close_button.setFixedSize(150, 40)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #3f3f3f;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4f4f4f;
            }
            QPushButton:pressed {
                background-color: #2f2f2f;
            }
        """)
        action_layout.addWidget(close_button)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
        
        # Set scroll widget and add to main layout
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)
        
        # Set main layout on dialog
        self.setLayout(main_layout)
    
    def format_time(self, seconds):
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def select_video_file(self):
        """Open file dialog to select a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, video_path):
        """Load a video file and update the UI"""
        # Release previous video if any
        if self.cap:
            self.cap.release()
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video file.")
            return
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Reset position
        self.current_position = 0.0
        
        # Update UI
        info_text = f"Video: {os.path.basename(self.video_path)}\n"
        info_text += f"Duration: {self.format_time(self.duration)} | "
        info_text += f"Resolution: {self.width}x{self.height} | "
        info_text += f"FPS: {self.fps:.2f}"
        self.info_label.setText(info_text)
        
        # Update slider range
        self.position_slider.setMaximum(int(self.duration * 10))
        self.position_slider.setValue(0)
        
        # Update custom duration spinbox max
        self.custom_duration_spinbox.setMaximum(int(self.duration))
        
        # Update output filename
        base_name, ext = os.path.splitext(os.path.basename(self.video_path))
        default_output = f"{base_name}_trimmed.mp4"
        self.output_filename.setText(default_output)
        
        # Enable controls
        self.position_slider.setEnabled(True)
        self.duration_combo.setEnabled(True)
        self.output_filename.setEnabled(True)
        
        # Update preview
        self.update_preview()
        self.update_end_time_display()
    
    def on_slider_changed(self, value):
        """Handle slider value change"""
        if not self.cap or not self.cap.isOpened():
            return
        
        self.current_position = value / 10.0  # Convert back to seconds
        self.update_preview()
        self.time_label.setText(f"Start Time: {self.format_time(self.current_position)}")
        self.update_end_time_display()
    
    def adjust_position(self, seconds):
        """Adjust the current position by the given seconds"""
        if not self.cap or not self.cap.isOpened() or self.duration == 0:
            return
        
        new_position = max(0, min(self.duration, self.current_position + seconds))
        slider_value = int(new_position * 10)
        self.position_slider.setValue(slider_value)
    
    def on_duration_changed(self, text):
        """Handle duration selection change"""
        is_custom = text == "Custom"
        self.custom_duration_label.setVisible(is_custom)
        self.custom_duration_spinbox.setVisible(is_custom)
        self.update_end_time_display()
    
    def get_selected_duration(self):
        """Get the currently selected duration in seconds"""
        selected = self.duration_combo.currentText()
        if selected == "Custom":
            return self.custom_duration_spinbox.value()
        else:
            return self.duration_presets[selected]
    
    def update_end_time_display(self):
        """Update the end time display label"""
        if not self.cap or not self.cap.isOpened() or self.duration == 0:
            self.end_time_label.setText("End Time: -- (Duration: --)")
            return
        
        duration = self.get_selected_duration()
        end_time = min(self.current_position + duration, self.duration)
        self.end_time_label.setText(
            f"End Time: {self.format_time(end_time)} "
            f"(Duration: {self.format_time(duration)})"
        )
    
    def update_preview(self):
        """Update the video preview frame"""
        if not self.cap or not self.cap.isOpened():
            # Show placeholder text when no video is loaded
            self.preview_label.setText("No video loaded\n\nClick 'Select Video' to begin")
            self.preview_label.setStyleSheet("background-color: #1e1e1e; border: 2px solid #3f3f3f; color: #999999; font-size: 14px;")
            return
        
        frame_number = int(self.current_position * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            # Convert frame to RGB for Qt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit preview (maintain aspect ratio)
            preview_height = 360
            preview_width = int(preview_height * self.width / self.height)
            frame_resized = cv2.resize(frame_rgb, (preview_width, preview_height))
            
            # Convert to QImage and display
            h, w, ch = frame_resized.shape
            bytes_per_line = ch * w
            q_image = QImage(frame_resized.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setStyleSheet("background-color: black; border: 2px solid #0078d4;")
    
    def trim_video(self):
        """Execute FFmpeg to trim the video using worker thread"""
        # Check if video is loaded
        if not self.video_path or not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "No Video", 
                              "Please select a video file first.")
            return
        
        # Check if already processing
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Processing", 
                              "Video trimming is already in progress.")
            return
        
        start_time = self.current_position
        duration = self.get_selected_duration()
        end_time = min(start_time + duration, self.duration)
        
        if end_time <= start_time:
            QMessageBox.warning(self, "Invalid Range", 
                              "End time must be greater than start time.")
            return
        
        # Get output filename from the text field
        output_filename = self.output_filename.text().strip()
        if not output_filename:
            QMessageBox.warning(self, "Invalid Filename", 
                              "Please enter a valid output filename.")
            return
        
        # Ensure .mp4 extension
        if not output_filename.lower().endswith('.mp4'):
            output_filename += '.mp4'
        
        # Construct full output path (same directory as input)
        output_dir = os.path.dirname(self.video_path)
        output_file = os.path.join(output_dir, output_filename)
        
        # Check if output file exists
        if os.path.exists(output_file):
            reply = QMessageBox.question(
                self, "File Exists",
                f"Output file already exists:\n{output_file}\n\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # SLEAP-compatible FFmpeg command for precise frame handling
        command = [
            "ffmpeg",
            "-y",  # Overwrite without asking
            "-i", self.video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            # SLEAP-compatible encoding instead of copy
            "-c:v", "libx264",                   # Re-encode video for frame accuracy
            "-preset", "medium",                 # Balance speed/quality
            "-crf", "18",                        # High quality
            "-pix_fmt", "yuv420p",              # Standard pixel format
            "-movflags", "+faststart",           # Move moov atom to beginning
            "-avoid_negative_ts", "make_zero",   # Fix timestamp issues
            "-fflags", "+genpts",                # Generate presentation timestamps
            "-vsync", "vfr",                     # Variable frame rate (preserves timing)
            "-an",                               # Remove audio to avoid sync issues
            output_file
        ]
        
        # Clear log and prepare UI for processing
        self.log_text.clear()
        self.log_text.append(f"🎬 Starting video trim...\n")
        self.log_text.append(f"📁 Input: {os.path.basename(self.video_path)}")
        self.log_text.append(f"📁 Output: {output_filename}")
        self.log_text.append(f"⏱️  Start: {self.format_time(start_time)}")
        self.log_text.append(f"⏱️  End: {self.format_time(end_time)}")
        self.log_text.append(f"⏱️  Duration: {self.format_time(end_time - start_time)}\n")
        self.log_text.append("Running FFmpeg command...")
        
        # Update button states
        self.trim_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        
        # Create and start worker thread
        self.worker = TrimWorker(command, output_file)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
    
    def cancel_trim(self):
        """Cancel the current trim operation"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.log_text.append("\n❌ Trim operation cancelled by user.")
            
            # Reset button states
            self.trim_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
    
    def on_worker_progress(self, message):
        """Handle progress messages from worker thread"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_worker_finished(self, success, message):
        """Handle worker thread completion"""
        # Reset button states
        self.trim_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        # Show completion message in log
        self.log_text.append(f"\n{message}")
        
        if success:
            # Show success dialog but don't close the window
            QMessageBox.information(self, "Success", message)
        else:
            # Show error dialog
            QMessageBox.critical(self, "Error", message)
    
    def closeEvent(self, event):
        """Clean up when dialog is closed"""
        if self.cap:
            self.cap.release()
        event.accept()


def video_trim():
    """Launches the PyQt5 trimming GUI"""
    from PyQt5.QtWidgets import QApplication
    import sys
    
    # Create QApplication if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Open the dialog (user will select video from within the GUI)
    dialog = VideoTrimDialog()
    dialog.exec_()


# For backward compatibility and standalone testing
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    video_trim()
    sys.exit(app.exec_())
