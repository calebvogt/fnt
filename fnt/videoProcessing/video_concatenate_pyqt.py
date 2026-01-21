#!/usr/bin/env python3
"""
Video Concatenation Tool - PyQt5 Implementation

Concatenate multiple video files within directories using FFmpeg.
Features batch processing with progress tracking and FFmpeg output display.
"""

import os
import sys
import subprocess
import glob
from pathlib import Path

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QGridLayout, QPushButton, QLabel, QFileDialog, QMessageBox, 
        QProgressBar, QTextEdit, QGroupBox, QFrame, QScrollArea, QLineEdit,
        QComboBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 not available. Please install with: pip install PyQt5")
    sys.exit(1)


class ConcatenationWorker(QThread):
    """Worker thread for video concatenation to avoid blocking the GUI"""
    progress_update = pyqtSignal(str)  # status message
    folder_progress = pyqtSignal(int, int)  # current folder, total folders
    ffmpeg_output = pyqtSignal(str)  # FFmpeg output lines
    finished = pyqtSignal(bool, str)  # success, final message
    
    def __init__(self, input_dirs, output_filename, sort_order="default", instance_id=1):
        super().__init__()
        self.input_dirs = input_dirs
        self.output_filename = output_filename
        self.sort_order = sort_order
        self.instance_id = instance_id
        self.should_stop = False
    
    def stop(self):
        """Stop the processing"""
        self.should_stop = True
    
    def sort_viewtron_files(self, video_files):
        """
        Sort ViewTron DVR files in chronological order.
        ViewTron naming: Base_YYYYMMDDHHMMSS.ext, then Base_YYYYMMDDHHMMSS(001).ext, etc.
        """
        import re
        
        def viewtron_sort_key(filepath):
            filename = os.path.basename(filepath)
            
            # Pattern: extract timestamp and sequence number
            # Matches: YYYYMMDDHHMMSS_suffix.ext or YYYYMMDDHHMMSS(NNN)_suffix.ext
            match = re.search(r'_(\d{14})(?:\((\d+)\))?', filename)
            
            if match:
                timestamp = match.group(1)  # YYYYMMDDHHMMSS
                sequence = match.group(2)    # Optional (NNN)
                
                # Files without (NNN) come before files with (NNN)
                if sequence is None:
                    sequence = -1  # Base file comes first
                else:
                    sequence = int(sequence)
                
                return (timestamp, sequence)
            else:
                # Fallback for files that don't match pattern
                return (filename, 0)
        
        return sorted(video_files, key=viewtron_sort_key)
    
    def run(self):
        """Main processing function"""
        try:
            total_folders = len(self.input_dirs)
            successful = 0
            failed = 0
            
            for idx, input_dir in enumerate(self.input_dirs, 1):
                if self.should_stop:
                    self.finished.emit(False, "Processing stopped by user.")
                    return
                
                self.folder_progress.emit(idx, total_folders)
                self.progress_update.emit(f"Processing folder {idx}/{total_folders}: {input_dir}")
                
                # Process the folder
                success = self.concatenate_folder(input_dir)
                
                if success:
                    successful += 1
                else:
                    failed += 1
            
            if self.should_stop:
                self.finished.emit(False, "Processing stopped by user.")
            else:
                msg = f"Concatenation complete! Processed {successful} folder(s)"
                if failed > 0:
                    msg += f", {failed} failed"
                self.finished.emit(True, msg)
                
        except Exception as e:
            self.finished.emit(False, f"Error during processing: {str(e)}")
    
    def concatenate_folder(self, folder_path):
        """Concatenate all videos in a single folder"""
        try:
            # Video extensions to look for
            VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".MP4", ".mkv", ".flv", ".wmv", ".m4v")
            
            # Find all video files (use set to avoid duplicates from case-insensitive filesystems)
            video_files_set = set()
            for ext in VIDEO_EXTENSIONS:
                found_files = glob.glob(os.path.join(folder_path, f"*{ext}"))
                video_files_set.update(found_files)
            
            # Convert back to list
            video_files = list(video_files_set)
            
            # Sort files based on selected order
            if self.sort_order == "viewtron":
                video_files = self.sort_viewtron_files(video_files)
            else:
                video_files = sorted(video_files)
            
            if not video_files:
                self.progress_update.emit(f"⚠️ No video files found in {os.path.basename(folder_path)}")
                return False
            
            self.progress_update.emit(f"Found {len(video_files)} video file(s)")
            
            # Create concat list file
            list_file = os.path.join(folder_path, "concat_list.txt")
            with open(list_file, "w") as fp:
                for video in video_files:
                    # Use relative path and escape special characters
                    rel_path = os.path.basename(video)
                    fp.write(f"file '{rel_path}'\n")
            
            # Output file path
            output_file = os.path.join(folder_path, self.output_filename)
            
            # Check if output file already exists
            if os.path.exists(output_file):
                counter = 1
                base_name, ext = os.path.splitext(self.output_filename)
                while os.path.exists(output_file):
                    output_file = os.path.join(folder_path, f"{base_name}_{counter}{ext}")
                    counter += 1
                self.progress_update.emit(f"Output file exists, using: {os.path.basename(output_file)}")
            
            # SLEAP-compatible FFmpeg command for precise frame handling
            command = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                # SLEAP-compatible encoding instead of copy
                "-c:v", "libx264",                   # Re-encode for consistency
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
            
            self.progress_update.emit(f"Concatenating videos...")
            
            # Run FFmpeg with output capture
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=folder_path
            )
            
            # Stream FFmpeg output
            for line in process.stdout:
                if self.should_stop:
                    process.terminate()
                    return False
                
                line = line.strip()
                if line:
                    self.ffmpeg_output.emit(line)
            
            process.wait()
            
            # Clean up list file
            try:
                os.remove(list_file)
            except:
                pass
            
            if process.returncode == 0:
                self.progress_update.emit(f"✅ Successfully created: {os.path.basename(output_file)}")
                return True
            else:
                self.progress_update.emit(f"❌ Failed to concatenate videos in {os.path.basename(folder_path)}")
                return False
                
        except Exception as e:
            self.progress_update.emit(f"❌ Error processing {folder_path}: {str(e)}")
            return False


class VideoConcatenationGUI(QMainWindow):
    """Main GUI window for video concatenation"""
    
    # Class variable to track instance count
    instance_count = 0
    
    def __init__(self):
        super().__init__()
        
        # Increment instance counter and set unique ID
        VideoConcatenationGUI.instance_count += 1
        self.instance_id = VideoConcatenationGUI.instance_count
        
        self.selected_dirs = []
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"Video Concatenation Tool #{self.instance_id} - FieldNeuroToolbox")
        self.setGeometry(200 + (self.instance_id - 1) * 50, 200 + (self.instance_id - 1) * 50, 900, 700)
        self.setMinimumSize(700, 600)
        
        # Set application style - Dark Mode
        self.setStyleSheet("""
            QMainWindow {
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
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                color: #cccccc;
            }
            QProgressBar {
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                text-align: center;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
            QFrame {
                background-color: #2b2b2b;
                border-color: #3f3f3f;
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
        
        # Output options
        self.create_output_options(scroll_layout)
        
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
        header_frame.setStyleSheet("background-color: #1e1e1e; padding: 15px; border: 1px solid #3f3f3f;")
        
        header_layout = QVBoxLayout()
        header_frame.setLayout(header_layout)
        
        title = QLabel(f"Video Concatenation Tool #{self.instance_id}")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("Join multiple video files together using FFmpeg concat")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet("color: #999999; font-style: italic; background-color: transparent;")
        header_layout.addWidget(subtitle)
        
        layout.addWidget(header_frame)
    
    def create_directory_selection(self, layout):
        """Create directory selection section"""
        group = QGroupBox("Input Directories")
        group_layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Select directories containing video files to concatenate (.mp4, .avi, .mov, .mkv, etc.)\nVideos in each directory will be concatenated into a single output file.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #999999; margin-bottom: 10px;")
        group_layout.addWidget(instructions)
        
        # Directory list display
        self.dir_list_label = QLabel("No directories selected")
        self.dir_list_label.setStyleSheet("border: 1px solid #3f3f3f; padding: 10px; background-color: #1e1e1e; min-height: 60px; color: #cccccc;")
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
    
    def create_output_options(self, layout):
        """Create output filename options section"""
        group = QGroupBox("Output Options")
        group_layout = QGridLayout()
        
        # Row 0: Output filename
        group_layout.addWidget(QLabel("Output Filename:"), 0, 0)
        
        self.output_filename_edit = QLineEdit()
        self.output_filename_edit.setText("concatenated_output.mp4")
        self.output_filename_edit.setPlaceholderText("Enter output filename...")
        self.output_filename_edit.setToolTip("Filename for the concatenated video (saved in each directory)")
        group_layout.addWidget(self.output_filename_edit, 0, 1)
        
        info_label = QLabel("💡 Files saved in each selected directory")
        info_label.setStyleSheet("color: #999999; font-style: italic;")
        group_layout.addWidget(info_label, 0, 2)
        
        # Row 1: Sort order
        group_layout.addWidget(QLabel("Sort Order:"), 1, 0)
        
        self.sort_order_combo = QComboBox()
        self.sort_order_combo.addItems(["Default (Alphabetical)", "ViewTron DVR (Chronological)"])
        self.sort_order_combo.setCurrentIndex(0)
        self.sort_order_combo.setToolTip("How to order videos for concatenation:\n• Default: Standard alphabetical sorting\n• ViewTron DVR: Handles ViewTron naming (YYYYMMDDHHMMSS, then YYYYMMDDHHMMSS(001), etc.)")
        group_layout.addWidget(self.sort_order_combo, 1, 1)
        
        sort_info_label = QLabel("ℹ️ Choose ViewTron for DVR recordings")
        sort_info_label.setStyleSheet("color: #999999; font-style: italic;")
        group_layout.addWidget(sort_info_label, 1, 2)
        
        group_layout.setColumnStretch(1, 1)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def create_control_buttons(self, layout):
        """Create control buttons section"""
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Concatenation")
        self.start_btn.clicked.connect(self.start_concatenation)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_concatenation)
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
        
        # Folder progress
        self.folder_progress_label = QLabel("Ready to start...")
        group_layout.addWidget(self.folder_progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        group_layout.addWidget(self.progress_bar)
        
        # Status log
        status_label = QLabel("Status Log:")
        status_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #cccccc;")
        group_layout.addWidget(status_label)
        
        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(100)
        self.status_log.setReadOnly(True)
        self.status_log.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3f3f3f; color: #cccccc;")
        group_layout.addWidget(self.status_log)
        
        # FFmpeg output log
        ffmpeg_label = QLabel("FFmpeg Output:")
        ffmpeg_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #cccccc;")
        group_layout.addWidget(ffmpeg_label)
        
        self.ffmpeg_log = QTextEdit()
        self.ffmpeg_log.setMaximumHeight(150)
        self.ffmpeg_log.setReadOnly(True)
        self.ffmpeg_log.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3f3f3f; color: #cccccc; font-family: 'Courier New', monospace; font-size: 9px;")
        group_layout.addWidget(self.ffmpeg_log)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def add_directory(self):
        """Add a directory to the concatenation list"""
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
            dir_list = "\n".join([f"• {d}" for d in self.selected_dirs])
            self.dir_list_label.setText(f"Selected directories ({len(self.selected_dirs)}):\n{dir_list}")
    
    def start_concatenation(self):
        """Start video concatenation"""
        if not self.selected_dirs:
            QMessageBox.warning(self, "Warning", "Please select at least one directory.")
            return
        
        # Get output filename
        output_filename = self.output_filename_edit.text().strip()
        if not output_filename:
            QMessageBox.warning(self, "Warning", "Please enter an output filename.")
            return
        
        # Ensure .mp4 extension
        if not output_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_filename += '.mp4'
        
        # Disable controls
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.add_dir_btn.setEnabled(False)
        self.clear_dirs_btn.setEnabled(False)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Get sort order
        sort_order = "viewtron" if self.sort_order_combo.currentIndex() == 1 else "default"
        
        # Clear logs
        self.status_log.clear()
        self.ffmpeg_log.clear()
        self.log_message("Starting video concatenation...")
        self.log_message(f"Output filename: {output_filename}")
        self.log_message(f"Sort order: {self.sort_order_combo.currentText()}")
        
        # Start worker thread
        self.worker = ConcatenationWorker(self.selected_dirs, output_filename, sort_order, self.instance_id)
        self.worker.progress_update.connect(self.log_message)
        self.worker.folder_progress.connect(self.update_folder_progress)
        self.worker.ffmpeg_output.connect(self.log_ffmpeg_output)
        self.worker.finished.connect(self.concatenation_finished)
        self.worker.start()
    
    def stop_concatenation(self):
        """Stop video concatenation"""
        if self.worker:
            self.log_message("Stopping concatenation...")
            self.worker.stop()
            self.stop_btn.setEnabled(False)
    
    def update_folder_progress(self, current, total):
        """Update folder progress"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.folder_progress_label.setText(f"Processing folder {current} of {total}")
    
    def log_message(self, message):
        """Add message to status log"""
        self.status_log.append(message)
        # Auto-scroll to bottom
        cursor = self.status_log.textCursor()
        cursor.movePosition(cursor.End)
        self.status_log.setTextCursor(cursor)
    
    def log_ffmpeg_output(self, output_line):
        """Add FFmpeg output to log"""
        self.ffmpeg_log.append(output_line)
        # Auto-scroll to bottom
        cursor = self.ffmpeg_log.textCursor()
        cursor.movePosition(cursor.End)
        self.ffmpeg_log.setTextCursor(cursor)
    
    def concatenation_finished(self, success, message):
        """Handle concatenation completion"""
        self.log_message(message)
        
        # Re-enable controls
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.add_dir_btn.setEnabled(True)
        self.clear_dirs_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Processing Stopped", message)
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Concatenation in Progress",
                "Concatenation is still running. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def video_concatenate():
    """Launch the PyQt5 video concatenation GUI"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = VideoConcatenationGUI()
    window.show()
    
    return window


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoConcatenationGUI()
    window.show()
    sys.exit(app.exec_())
