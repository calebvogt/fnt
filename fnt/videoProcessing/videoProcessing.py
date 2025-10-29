#!/usr/bin/env python3
"""
Video PreProcessing Tool for FieldNeuroToolbox

Comprehensive video preprocessing combining downsampling, re-encoding, and format conversion.
Allows users to batch process videos with customizable quality, resolution, and encoding options.
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
    
    def __init__(self, input_dirs, frame_rate, grayscale, apply_clahe, 
                 remove_audio, output_format, crf_quality, resolution, codec, preset, instance_id=1):
        super().__init__()
        self.input_dirs = input_dirs
        self.frame_rate = frame_rate
        self.grayscale = grayscale
        self.apply_clahe = apply_clahe
        self.remove_audio = remove_audio
        self.output_format = output_format
        self.crf_quality = crf_quality
        self.resolution = resolution
        self.codec = codec
        self.preset = preset
        self.instance_id = instance_id
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
                        success = self.process_single_file(video_file, out_dir, processed_files)
                        if not success and not self.should_stop:
                            self.finished.emit(False, f"Failed to process: {os.path.basename(video_file)}")
                            return
            
            if not self.should_stop:
                self.finished.emit(True, f"Successfully processed {processed_files} video files!")
            else:
                self.finished.emit(False, "Processing stopped by user.")
                
        except Exception as e:
            self.finished.emit(False, f"Error during processing: {str(e)}")
    
    def process_single_file(self, video_file, out_dir, file_index):
        """Process a single video file"""
        try:
            # Get filename without extension
            video_filename = os.path.basename(video_file)
            video_filename_no_ext = re.sub(r'\.(avi|mp4|mov|mkv|webm|flv|wmv|m4v)$', '', video_filename, flags=re.IGNORECASE)
            
            # Output file path with selected format
            output_file = os.path.join(out_dir, f"{video_filename_no_ext}_processed.{self.output_format}")
            
            self.progress_update.emit(f"Processing: {video_filename}")
            
            # Build FFmpeg command based on settings
            cmd = self.build_ffmpeg_command(video_file, output_file)
            
            # Run FFmpeg and capture output for GUI display
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
            success = process.returncode == 0
            
            if success:
                self.progress_update.emit(f"✅ Completed: {video_filename}")
                return True
            else:
                self.progress_update.emit(f"❌ Failed: {video_filename}")
                return False
                
        except Exception as e:
            self.progress_update.emit(f"❌ Error processing {video_filename}: {str(e)}")
            return False
    
    def build_ffmpeg_command(self, input_file, output_file):
        """Build the FFmpeg command based on user settings with SLEAP-compatible frame handling"""
        
        # Get resolution dimensions
        if self.resolution == "1080p":
            width, height = 1920, 1080
        else:  # 720p
            width, height = 1280, 720
        
        # Build FFmpeg command with SLEAP-compatible settings
        cmd = [
            "ffmpeg", "-y",                      # Overwrite output files
            "-i", input_file,
            "-vcodec", self.codec,               # User-selected codec (libx265 or libx264)
            "-preset", self.preset,              # User-selected speed preset
            "-crf", str(self.crf_quality),       # Quality (0-51): lower is better
            "-pix_fmt", "yuv420p",
        ]
        
        # SLEAP-compatible frame rate handling
        if self.frame_rate < 30:  # Only downsample if requested rate is lower
            # Use fps filter for precise frame selection instead of -r
            video_filters = [f"fps={self.frame_rate}"]
        else:
            # Keep original frame rate for high frame rates
            video_filters = []
        
        # Add audio option
        if self.remove_audio:
            cmd.append("-an")  # Remove audio
        else:
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])  # Keep audio with AAC codec
        
        # SLEAP-compatible settings for better frame handling
        cmd.extend([
            "-movflags", "+faststart",           # Move moov atom to beginning
            "-avoid_negative_ts", "make_zero",   # Fix timestamp issues
            "-max_muxing_queue_size", "10000000",
            "-fflags", "+genpts",                # Generate presentation timestamps
            "-vsync", "vfr"                      # Variable frame rate (preserves timing)
        ])
        
        # Scaling and padding
        video_filters.extend([
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:eval=frame",
            f"pad={width}:{height}:-1:-1:color=black"
        ])
        
        # CONTRAST ENHANCEMENT - COMMENTED OUT FOR NOW
        # Can be re-enabled later if needed
        # Add contrast enhancement if requested (works with both color and grayscale)
        # if self.apply_clahe:
        #     if self.grayscale:
        #         video_filters.append("format=gray")
        #         # Contrast enhancement for grayscale
        #         video_filters.append("eq=contrast=1.3:brightness=0.05")
        #     else:
        #         # Contrast enhancement for color videos
        #         video_filters.append("eq=contrast=1.2:brightness=0.03:saturation=1.1")
        # elif self.grayscale:
        #     video_filters.append("format=gray")
        
        # Grayscale conversion (without contrast enhancement)
        if self.grayscale:
            video_filters.append("format=gray")
        
        video_filter = ",".join(video_filters)
        cmd.extend(["-vf", video_filter, output_file])
        
        return cmd


class VideoProcessingGUI(QMainWindow):
    """Main GUI window for combined video processing"""
    
    # Class variable to track instance count
    instance_count = 0
    
    def __init__(self):
        super().__init__()
        
        # Increment instance counter and set unique ID
        VideoProcessingGUI.instance_count += 1
        self.instance_id = VideoProcessingGUI.instance_count
        
        self.selected_dirs = []
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"Video PreProcessing Tool #{self.instance_id} - FieldNeuroToolbox")
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
            QSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #3f3f3f;
                border: 1px solid #3f3f3f;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #0078d4;
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
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
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
        header_frame.setStyleSheet("background-color: #1e1e1e; padding: 15px; border: 1px solid #3f3f3f;")
        
        header_layout = QVBoxLayout()
        header_frame.setLayout(header_layout)
        
        title = QLabel(f"Video PreProcessing Tool #{self.instance_id}")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("Comprehensive video preprocessing with downsampling, re-encoding, and format conversion")
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
        instructions = QLabel("Select directories containing video files (.avi, .mp4, .mov, .mkv, .webm, .flv, .wmv, .m4v)")
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
        
        self.add_dir_btn = QPushButton("Add Folder")
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
        
        row = 0
        
        # Frame rate option
        group_layout.addWidget(QLabel("Frame Rate (fps):"), row, 0)
        self.frame_rate_spin = QSpinBox()
        self.frame_rate_spin.setRange(1, 120)
        self.frame_rate_spin.setValue(30)
        self.frame_rate_spin.setToolTip("Target frame rate for output videos")
        group_layout.addWidget(self.frame_rate_spin, row, 1)
        row += 1
        
        # Output format option
        group_layout.addWidget(QLabel("Output Format:"), row, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["mp4", "avi"])
        self.format_combo.setCurrentText("mp4")
        self.format_combo.setToolTip("Output video file format")
        group_layout.addWidget(self.format_combo, row, 1)
        row += 1
        
        # Grayscale option
        self.grayscale_check = QCheckBox("Convert to Grayscale")
        self.grayscale_check.setChecked(True)
        self.grayscale_check.setToolTip("Convert videos to grayscale to reduce file size")
        group_layout.addWidget(self.grayscale_check, row, 0, 1, 2)
        row += 1
        
        # Remove audio option
        self.remove_audio_check = QCheckBox("Remove Audio")
        self.remove_audio_check.setChecked(True)
        self.remove_audio_check.setToolTip("Remove audio track from videos to reduce file size")
        group_layout.addWidget(self.remove_audio_check, row, 0, 1, 2)
        row += 1
        
        # CLAHE contrast enhancement option - COMMENTED OUT FOR NOW
        # Can be re-enabled later if needed
        # self.clahe_check = QCheckBox("Apply Contrast Enhancement")
        # self.clahe_check.setChecked(False)
        # self.clahe_check.setToolTip("Apply contrast and brightness enhancement for better visibility (works with both color and grayscale)")
        # group_layout.addWidget(self.clahe_check, row, 0, 1, 2)
        # row += 1
        
        # Show/Hide Advanced Options Button
        self.advanced_btn = QPushButton("Show Advanced Options ▼")
        self.advanced_btn.clicked.connect(self.toggle_advanced_options)
        self.advanced_btn.setStyleSheet("""
            QPushButton {
                background-color: #3f3f3f;
                color: #cccccc;
                text-align: left;
                padding-left: 10px;
            }
            QPushButton:hover {
                background-color: #4f4f4f;
            }
        """)
        group_layout.addWidget(self.advanced_btn, row, 0, 1, 2)
        row += 1
        
        # Advanced options frame (initially hidden)
        self.advanced_frame = QFrame()
        self.advanced_frame.setVisible(False)
        self.advanced_frame.setStyleSheet("QFrame { border: 1px solid #3f3f3f; background-color: #1e1e1e; padding: 10px; }")
        advanced_layout = QGridLayout()
        self.advanced_frame.setLayout(advanced_layout)
        
        # Video Codec option
        advanced_layout.addWidget(QLabel("Video Codec:"), 0, 0)
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(["libx265 (H.265/HEVC)", "libx264 (H.264/AVC)"])
        self.codec_combo.setCurrentText("libx265 (H.265/HEVC)")
        self.codec_combo.setToolTip("Video codec: H.265 offers better compression but slower encoding; H.264 is more compatible")
        advanced_layout.addWidget(self.codec_combo, 0, 1)
        
        # Speed Preset option
        advanced_layout.addWidget(QLabel("Speed Preset:"), 1, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "ultrafast", "superfast", "veryfast", "faster", 
            "fast", "medium", "slow", "slower", "veryslow"
        ])
        self.preset_combo.setCurrentText("ultrafast")
        self.preset_combo.setToolTip("Encoding speed: ultrafast = fastest encoding but larger files; slower = better compression but takes longer")
        advanced_layout.addWidget(self.preset_combo, 1, 1)
        
        # CRF Quality option
        advanced_layout.addWidget(QLabel("CRF Quality:"), 2, 0)
        self.crf_combo = QComboBox()
        self.crf_combo.addItems(["10 (Best)", "15 (High)", "20 (Good)", "25 (Medium)", "30 (Low)"])
        self.crf_combo.setCurrentText("25 (Medium)")
        self.crf_combo.setToolTip("Lower values = better quality but larger file size. CRF 15 is near-lossless.")
        advanced_layout.addWidget(self.crf_combo, 2, 1)
        
        # Resolution option
        advanced_layout.addWidget(QLabel("Resolution:"), 3, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1080p (1920x1080)", "720p (1280x720)"])
        self.resolution_combo.setCurrentText("1080p (1920x1080)")
        self.resolution_combo.setToolTip("Target output resolution")
        advanced_layout.addWidget(self.resolution_combo, 3, 1)
        
        group_layout.addWidget(self.advanced_frame, row, 0, 1, 2)
        row += 1
        
        # Output info
        info_label = QLabel("Videos saved to 'proc' subfolder in each input directory")
        info_label.setStyleSheet("color: #999999; font-style: italic; margin-top: 10px;")
        info_label.setWordWrap(True)
        group_layout.addWidget(info_label, row, 0, 1, 2)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def toggle_advanced_options(self):
        """Toggle visibility of advanced options"""
        is_visible = self.advanced_frame.isVisible()
        self.advanced_frame.setVisible(not is_visible)
        
        if is_visible:
            self.advanced_btn.setText("Show Advanced Options ▼")
        else:
            self.advanced_btn.setText("Hide Advanced Options ▲")
    
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
        # apply_clahe = self.clahe_check.isChecked()  # COMMENTED OUT - contrast enhancement disabled
        apply_clahe = False  # Set to False since feature is disabled
        remove_audio = self.remove_audio_check.isChecked()
        output_format = self.format_combo.currentText()
        
        # Get codec (extract first word from selection)
        codec_text = self.codec_combo.currentText()
        codec = codec_text.split()[0]  # Extract "libx265" or "libx264"
        
        # Get speed preset
        preset = self.preset_combo.currentText()
        
        # Get CRF quality value (extract number from selection)
        crf_text = self.crf_combo.currentText()
        crf_quality = int(crf_text.split()[0])  # Extract number from "15 (High)"
        
        # Get resolution (extract value from selection)
        resolution_text = self.resolution_combo.currentText()
        resolution = resolution_text.split()[0]  # Extract "1080p" or "720p"
        
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
        self.log_message("Starting video preprocessing...")
        self.log_message(f"Settings: {frame_rate} fps, {resolution}, {codec}, Preset: {preset}, CRF: {crf_quality}, Format: {output_format}")
        self.log_message(f"Options: Grayscale: {grayscale}, Remove Audio: {remove_audio}, Contrast: {apply_clahe}")
        
        # Start worker thread
        self.worker = VideoProcessorWorker(
            self.selected_dirs, 
            frame_rate, 
            grayscale,
            apply_clahe,
            remove_audio,
            output_format,
            crf_quality,
            resolution,
            codec,
            preset,
            self.instance_id
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