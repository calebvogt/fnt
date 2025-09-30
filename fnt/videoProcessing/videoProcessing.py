#!/usr/bin/env python3
"""
Combined Video Processing Tool for FieldNeuroToolbox

Combines functionality f            # Get filename without extension
            video_filename = os.path.basename(video_file)
            video_filename_no_ext = re.sub(r'\.(avi|mp4|mov|mkv|webm|flv|wmv|m4v)$', '', video_filename, flags=re.IGNORECASE) videoDownsample.py and video_reencode.py with a modern PyQt interface.
Allows users to process videos with customizable frame rate and grayscale options.
"""

import os
import sys
import subprocess
import glob
import re
from pathlib import Path

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QGridLayout, QPushButton, QLabel, QSpinBox, QCheckBox, QComboBox,
        QFileDialog, QMessageBox, QProgressBar, QTextEdit, QGroupBox,
        QFrame, QSizePolicy, QScrollArea
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 not available. Please install with: pip install PyQt5")
    sys.exit(1)


class VideoProcessorWorker(QThread):
    """Worker thread for video processing to avoid blocking the GUI"""
    progress_update = pyqtSignal(str)  # status message
    file_progress = pyqtSignal(int, int)  # current file, total files
    ffmpeg_output = pyqtSignal(str)  # FFmpeg output lines
    finished = pyqtSignal(bool, str)  # success, final message
    
    def __init__(self, input_dirs, frame_rate, grayscale, gpu_acceleration, apply_clahe):
        super().__init__()
        self.input_dirs = input_dirs
        self.frame_rate = frame_rate
        self.grayscale = grayscale
        self.gpu_acceleration = gpu_acceleration
        self.apply_clahe = apply_clahe
        self.should_stop = False
    
    def stop(self):
        """Stop the processing"""
        self.should_stop = True
    
    def run(self):
        """Main processing function"""
        try:
            total_files = 0
            processed_files = 0
            
            # Count total files first
            video_extensions = ["*.avi", "*.mp4", "*.mov", "*.mkv", "*.webm", "*.flv", "*.wmv", "*.m4v"]
            for input_dir in self.input_dirs:
                for ext in video_extensions:
                    total_files += len(glob.glob(os.path.join(input_dir, ext)))
            
            if total_files == 0:
                self.finished.emit(False, "No video files found in selected directories.")
                return
            
            self.progress_update.emit(f"Found {total_files} video files to process...")
            
            # Process each directory
            for input_dir in self.input_dirs:
                if self.should_stop:
                    break
                    
                # Create output directory
                out_dir = os.path.join(input_dir, "proc")
                os.makedirs(out_dir, exist_ok=True)
                
                self.progress_update.emit(f"Processing directory: {input_dir}")
                
                # Process each video type
                for video_extension in video_extensions:
                    if self.should_stop:
                        break
                        
                    video_files = glob.glob(os.path.join(input_dir, video_extension))
                    
                    for video_file in video_files:
                        if self.should_stop:
                            break
                            
                        processed_files += 1
                        self.file_progress.emit(processed_files, total_files)
                        
                        # Process individual file
                        success = self.process_single_file(video_file, out_dir)
                        if not success and not self.should_stop:
                            self.finished.emit(False, f"Failed to process: {os.path.basename(video_file)}")
                            return
            
            if not self.should_stop:
                self.finished.emit(True, f"Successfully processed {processed_files} video files!")
            else:
                self.finished.emit(False, "Processing stopped by user.")
                
        except Exception as e:
            self.finished.emit(False, f"Error during processing: {str(e)}")
    
    def process_single_file(self, video_file, out_dir):
        """Process a single video file"""
        try:
            # Get filename without extension
            video_filename = os.path.basename(video_file)
            video_filename_no_ext = re.sub(r'\\.(avi|mp4|mov)$', '', video_filename, flags=re.IGNORECASE)
            
            # Output file path
            output_file = os.path.join(out_dir, video_filename_no_ext + '_processed.mp4')
            
            self.progress_update.emit(f"Processing: {video_filename}")
            
            # Build FFmpeg command based on settings
            cmd = self.build_ffmpeg_command(video_file, output_file)
            
            # Run FFmpeg
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                universal_newlines=True,
                bufsize=1
            )
            
            # Monitor process output and stream to GUI
            for line in process.stdout:
                if self.should_stop:
                    process.terminate()
                    return False
                
                # Send FFmpeg output to GUI
                line = line.strip()
                if line:  # Only send non-empty lines
                    self.ffmpeg_output.emit(line)
            
            process.wait()
            
            if process.returncode == 0:
                self.progress_update.emit(f"✅ Completed: {video_filename}")
                return True
            else:
                self.progress_update.emit(f"❌ Failed: {video_filename}")
                return False
                
        except Exception as e:
            self.progress_update.emit(f"❌ Error processing {video_filename}: {str(e)}")
            return False
    
    def build_ffmpeg_command(self, input_file, output_file):
        """Build the FFmpeg command based on user settings"""
        
        if self.gpu_acceleration:
            # GPU-accelerated command (similar to videoDownsample.py GPU mode)
            cmd = [
                "ffmpeg", "-y",  # -y to overwrite output files
                "-hwaccel", "cuda", 
                "-i", input_file,
                "-vcodec", "hevc_nvenc",  # GPU acceleration
                "-preset", "hq",
                "-rc:v", "vbr",           # variable bitrate mode
                "-cq:v", "30",            # quality (15-32): lower is better
                "-b:v", "0.8M",           # target average bitrate
                "-maxrate", "0.8M",       # maximum bitrate
                "-bufsize", "1.6M",       # buffer size
                "-pix_fmt", "yuv420p",
                "-r", str(self.frame_rate),
                "-vsync", "cfr",
                "-an",                    # Remove audio
                "-movflags", "+faststart",
                "-max_muxing_queue_size", "10000000"
            ]
        else:
            # CPU command (similar to video_reencode.py)
            cmd = [
                "ffmpeg", "-y",  # -y to overwrite output files
                "-i", input_file,
                "-vcodec", "libx265",
                "-preset", "fast",
                "-crf", "25",             # Good quality compromise
                "-pix_fmt", "yuv420p",
                "-r", str(self.frame_rate),
                "-an",                    # Remove audio
                "-movflags", "+faststart",
                "-max_muxing_queue_size", "10000000"
            ]
        
        # Add video filter for scaling, grayscale, and contrast enhancement
        video_filters = []
        
        # Base scaling and padding
        video_filters.append("scale=1920:1080:force_original_aspect_ratio=decrease:eval=frame")
        video_filters.append("pad=1920:1080:-1:-1:color=black")
        
        # Add contrast enhancement if requested (works with both color and grayscale)
        if self.apply_clahe:
            if self.grayscale:
                video_filters.append("format=gray")
                # Contrast enhancement for grayscale
                video_filters.append("eq=contrast=1.3:brightness=0.05")
            else:
                # Contrast enhancement for color videos
                video_filters.append("eq=contrast=1.2:brightness=0.03:saturation=1.1")
        elif self.grayscale:
            video_filters.append("format=gray")
        
        video_filter = ",".join(video_filters)
        cmd.extend(["-vf", video_filter, output_file])
        
        return cmd


class VideoProcessingGUI(QMainWindow):
    """Main GUI window for combined video processing"""
    
    def __init__(self):
        super().__init__()
        self.selected_dirs = []
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Video Processing Tool - FieldNeuroToolbox")
        self.setGeometry(200, 200, 800, 600)
        self.setMinimumSize(600, 500)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #004578;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #c0c0c0;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #333333;
            }
            QSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Header
        self.create_header(layout)
        
        # Main content in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        
        # Directory selection
        self.create_directory_selection(scroll_layout)
        
        # Processing options
        self.create_processing_options(scroll_layout)
        
        # Control buttons
        self.create_control_buttons(scroll_layout)
        
        # Progress section
        self.create_progress_section(scroll_layout)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
    
    def create_header(self, layout):
        """Create header section"""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        header_frame.setStyleSheet("background-color: white; padding: 15px;")
        
        header_layout = QVBoxLayout()
        header_frame.setLayout(header_layout)
        
        title = QLabel("Video Processing Tool")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #007acc;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("Combined video downsampling and re-encoding with customizable options")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet("color: #666666; font-style: italic;")
        header_layout.addWidget(subtitle)
        
        layout.addWidget(header_frame)
    
    def create_directory_selection(self, layout):
        """Create directory selection section"""
        group = QGroupBox("Input Directories")
        group_layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Select directories containing video files (.avi, .mp4, .mov, .mkv, .webm, .flv, .wmv, .m4v)")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666666; margin-bottom: 10px;")
        group_layout.addWidget(instructions)
        
        # Directory list display
        self.dir_list_label = QLabel("No directories selected")
        self.dir_list_label.setStyleSheet("border: 1px solid #cccccc; padding: 10px; background-color: white; min-height: 60px;")
        self.dir_list_label.setWordWrap(True)
        group_layout.addWidget(self.dir_list_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.add_dir_btn = QPushButton("Add Directory")
        self.add_dir_btn.clicked.connect(self.add_directory)
        button_layout.addWidget(self.add_dir_btn)
        
        self.clear_dirs_btn = QPushButton("Clear All")
        self.clear_dirs_btn.clicked.connect(self.clear_directories)
        button_layout.addWidget(self.clear_dirs_btn)
        
        button_layout.addStretch()
        group_layout.addLayout(button_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def create_processing_options(self, layout):
        """Create processing options section"""
        group = QGroupBox("Processing Options")
        group_layout = QGridLayout()
        
        # Frame rate option
        group_layout.addWidget(QLabel("Frame Rate (fps):"), 0, 0)
        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setRange(1, 120)
        self.frame_rate_spin.setValue(30)
        self.frame_rate_spin.setToolTip("Target frame rate for output videos")
        group_layout.addWidget(self.frame_rate_spin, 0, 1)
        
        # Grayscale option
        self.grayscale_check = QCheckBox("Convert to Grayscale")
        self.grayscale_check.setChecked(True)
        self.grayscale_check.setToolTip("Convert videos to grayscale to reduce file size")
        group_layout.addWidget(self.grayscale_check, 1, 0, 1, 2)
        
        # GPU acceleration option
        self.gpu_check = QCheckBox("Use GPU Acceleration (NVIDIA CUDA)")
        self.gpu_check.setChecked(False)
        self.gpu_check.setToolTip("Use NVIDIA GPU for faster encoding (requires CUDA-capable GPU)")
        group_layout.addWidget(self.gpu_check, 2, 0, 1, 2)
        
        # CLAHE contrast enhancement option
        self.clahe_check = QCheckBox("Apply Contrast Enhancement")
        self.clahe_check.setChecked(False)
        self.clahe_check.setToolTip("Apply contrast and brightness enhancement for better visibility (works with both color and grayscale)")
        group_layout.addWidget(self.clahe_check, 3, 0, 1, 2)
        
        # Output format info
        info_label = QLabel("Output: 1920x1080 MP4, audio removed, saved to 'proc' subfolder")
        info_label.setStyleSheet("color: #666666; font-style: italic; margin-top: 10px;")
        info_label.setWordWrap(True)
        group_layout.addWidget(info_label, 4, 0, 1, 2)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def create_control_buttons(self, layout):
        """Create control buttons section"""
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Processing")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_progress_section(self, layout):
        """Create progress section"""
        group = QGroupBox("Progress")
        group_layout = QVBoxLayout()
        
        # File progress
        self.file_progress_label = QLabel("Ready to start...")
        group_layout.addWidget(self.file_progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        group_layout.addWidget(self.progress_bar)
        
        # Status log
        status_label = QLabel("Status Log:")
        status_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        group_layout.addWidget(status_label)
        
        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(100)
        self.status_log.setReadOnly(True)
        self.status_log.setStyleSheet("background-color: #f8f8f8; border: 1px solid #cccccc;")
        group_layout.addWidget(self.status_log)
        
        # FFmpeg output log
        ffmpeg_label = QLabel("FFmpeg Output:")
        ffmpeg_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        group_layout.addWidget(ffmpeg_label)
        
        self.ffmpeg_log = QTextEdit()
        self.ffmpeg_log.setMaximumHeight(150)
        self.ffmpeg_log.setReadOnly(True)
        self.ffmpeg_log.setStyleSheet("background-color: #f0f0f0; border: 1px solid #cccccc; font-family: 'Courier New', monospace; font-size: 9px;")
        group_layout.addWidget(self.ffmpeg_log)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def add_directory(self):
        """Add a directory to the processing list"""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Directory with Video Files",
            "", 
            QFileDialog.ShowDirsOnly
        )
        
        if directory and directory not in self.selected_dirs:
            self.selected_dirs.append(directory)
            self.update_directory_display()
            self.start_btn.setEnabled(len(self.selected_dirs) > 0)
    
    def clear_directories(self):
        """Clear all selected directories"""
        self.selected_dirs.clear()
        self.update_directory_display()
        self.start_btn.setEnabled(False)
    
    def update_directory_display(self):
        """Update the directory list display"""
        if not self.selected_dirs:
            self.dir_list_label.setText("No directories selected")
        else:
            dir_list = "\\n".join([f"• {d}" for d in self.selected_dirs])
            self.dir_list_label.setText(f"Selected directories ({len(self.selected_dirs)}): \\n{dir_list}")
    
    def start_processing(self):
        """Start video processing"""
        if not self.selected_dirs:
            QMessageBox.warning(self, "Warning", "Please select at least one directory.")
            return
        
        # Get user settings
        frame_rate = self.frame_rate_spin.value()
        grayscale = self.grayscale_check.isChecked()
        gpu_acceleration = self.gpu_check.isChecked()
        apply_clahe = self.clahe_check.isChecked()
        
        # Disable controls
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.add_dir_btn.setEnabled(False)
        self.clear_dirs_btn.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Clear logs
        self.status_log.clear()
        self.ffmpeg_log.clear()
        self.log_message("Starting video processing...")
        self.log_message(f"Settings: {frame_rate} fps, Grayscale: {grayscale}, GPU: {gpu_acceleration}, Contrast Enhancement: {apply_clahe}")
        
        # Start worker thread
        self.worker = VideoProcessorWorker(
            self.selected_dirs, 
            frame_rate, 
            grayscale, 
            gpu_acceleration,
            apply_clahe
        )
        self.worker.progress_update.connect(self.log_message)
        self.worker.file_progress.connect(self.update_file_progress)
        self.worker.ffmpeg_output.connect(self.log_ffmpeg_output)
        self.worker.finished.connect(self.processing_finished)
        self.worker.start()
    
    def stop_processing(self):
        """Stop video processing"""
        if self.worker:
            self.worker.stop()
            self.log_message("Stopping processing...")
    
    def update_file_progress(self, current, total):
        """Update file progress"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.file_progress_label.setText(f"Processing file {current} of {total}")
    
    def log_message(self, message):
        """Add message to status log"""
        self.status_log.append(message)
        # Auto-scroll to bottom
        cursor = self.status_log.textCursor()
        cursor.movePosition(cursor.End)
        self.status_log.setTextCursor(cursor)
    
    def log_ffmpeg_output(self, output_line):
        """Add FFmpeg output to the FFmpeg log"""
        self.ffmpeg_log.append(output_line)
        # Auto-scroll to bottom
        cursor = self.ffmpeg_log.textCursor()
        cursor.movePosition(cursor.End)
        self.ffmpeg_log.setTextCursor(cursor)
    
    def processing_finished(self, success, message):
        """Handle processing completion"""
        # Re-enable controls
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.add_dir_btn.setEnabled(True)
        self.clear_dirs_btn.setEnabled(True)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.file_progress_label.setText("Ready")
        
        # Log final message
        self.log_message(f"\\n{'✅' if success else '❌'} {message}")
        
        # Show completion dialog
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)
        
        self.worker = None
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, 
                "Confirm Close", 
                "Video processing is still running. Do you want to stop it and close?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait()  # Wait for thread to finish
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point"""
    if not PYQT_AVAILABLE:
        print("PyQt5 is required for this tool. Please install it with: pip install PyQt5")
        return
    
    app = QApplication(sys.argv)
    app.setApplicationName("Video Processing Tool")
    app.setApplicationVersion("1.0")
    
    window = VideoProcessingGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()