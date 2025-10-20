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
    QLineEdit
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont


class VideoTrimDialog(QDialog):
    """PyQt5 dialog for interactive video trimming with duration selection"""
    
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video file.")
            return
        
        # Video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.total_frames / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
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
        
        self.init_ui()
        self.update_preview()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Video Trimming Tool")
        self.setMinimumSize(900, 700)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Video Trimming Tool")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Video info
        info_text = f"Video: {os.path.basename(self.video_path)}\n"
        info_text += f"Duration: {self.format_time(self.duration)} | "
        info_text += f"Resolution: {self.width}x{self.height} | "
        info_text += f"FPS: {self.fps:.2f}"
        
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #666666; padding: 10px;")
        layout.addWidget(info_label)
        
        # Preview section
        preview_group = QGroupBox("Start Frame Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black; border: 2px solid #007acc;")
        self.preview_label.setMinimumSize(640, 360)
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
        self.position_slider.setMaximum(int(self.duration * 10))  # 0.1 second resolution
        self.position_slider.setValue(0)
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
        self.end_time_label = QLabel()
        self.end_time_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.update_end_time_display()
        duration_layout.addWidget(self.end_time_label)
        
        duration_group.setLayout(duration_layout)
        layout.addWidget(duration_group)
        
        # Output filename section
        output_group = QGroupBox("Output File")
        output_layout = QHBoxLayout()
        
        output_layout.addWidget(QLabel("Output filename:"))
        
        # Generate default output filename
        base_name, ext = os.path.splitext(os.path.basename(self.video_path))
        default_output = f"{base_name}_trimmed.mp4"
        
        self.output_filename = QLineEdit()
        self.output_filename.setText(default_output)
        self.output_filename.setMinimumWidth(300)
        self.output_filename.setPlaceholderText("Enter output filename...")
        output_layout.addWidget(self.output_filename)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.addStretch()
        
        trim_button = QPushButton("Trim Video")
        trim_button.clicked.connect(self.trim_video)
        trim_button.setFixedSize(150, 40)
        trim_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        action_layout.addWidget(trim_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setFixedSize(150, 40)
        cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        action_layout.addWidget(cancel_button)
        
        action_layout.addStretch()
        layout.addLayout(action_layout)
        
        self.setLayout(layout)
    
    def format_time(self, seconds):
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def on_slider_changed(self, value):
        """Handle slider value change"""
        self.current_position = value / 10.0  # Convert back to seconds
        self.update_preview()
        self.time_label.setText(f"Start Time: {self.format_time(self.current_position)}")
        self.update_end_time_display()
    
    def adjust_position(self, seconds):
        """Adjust the current position by the given seconds"""
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
        duration = self.get_selected_duration()
        end_time = min(self.current_position + duration, self.duration)
        self.end_time_label.setText(
            f"End Time: {self.format_time(end_time)} "
            f"(Duration: {self.format_time(duration)})"
        )
    
    def update_preview(self):
        """Update the video preview frame"""
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
    
    def trim_video(self):
        """Execute FFmpeg to trim the video"""
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
        
        # FFmpeg command
        command = [
            "ffmpeg",
            "-y",  # Overwrite without asking
            "-i", self.video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c", "copy",
            output_file
        ]
        
        try:
            # Run FFmpeg in background (non-blocking)
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Wait for completion
            output, _ = process.communicate()
            
            if process.returncode == 0:
                QMessageBox.information(
                    self, "Success",
                    f"Video trimmed successfully!\n\nSaved as:\n{output_file}"
                )
                self.accept()
            else:
                QMessageBox.critical(
                    self, "Error",
                    f"FFmpeg failed with error code {process.returncode}\n\n{output[:500]}"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
    
    def closeEvent(self, event):
        """Clean up when dialog is closed"""
        if self.cap:
            self.cap.release()
        event.accept()


def video_trim():
    """Opens a file dialog to select a video file and launches the PyQt5 trimming GUI"""
    from PyQt5.QtWidgets import QApplication
    import sys
    
    # Create QApplication if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # File dialog
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select Video File",
        "",
        "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
    )
    
    if file_path:
        dialog = VideoTrimDialog(file_path)
        dialog.exec_()


# For backward compatibility and standalone testing
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    video_trim()
    sys.exit(app.exec_())
