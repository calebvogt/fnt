"""
Video Trim and Crop Tool - PyQt5 Implementation (Batch Processing Version)

Interactive video trimming and cropping with batch processing support.
Allows users to:
- Add multiple videos for batch processing
- Set start position, duration, and crop region per video
- Process videos in batch with FFmpeg

Matches SLEAP ROI Tool styling and workflow.
"""

import os
import sys
import cv2
import subprocess
import numpy as np
import tempfile
from typing import List, Tuple, Optional
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QFileDialog, QMessageBox, QGroupBox, QComboBox,
    QLineEdit, QTextEdit, QListWidget, QListWidgetItem, QApplication,
    QProgressBar, QCheckBox, QScrollArea, QFrame, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont


class VideoTrimConfig:
    """Configuration for a single video's trim/crop settings."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.start_time = 0.0  # seconds
        self.duration = 300  # seconds (5 minutes default)
        self.crop_polygon = []  # List of (x, y) points
        self.output_filename = ""
        self.configured = False
        
        # Video properties
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_duration = 0
        self.first_frame = None


class BatchTrimWorker(QThread):
    """Worker thread for batch processing videos"""
    progress = pyqtSignal(int, int)  # video_idx, progress_percent
    status = pyqtSignal(str)
    video_finished = pyqtSignal(int, bool, str)  # video_idx, success, message
    all_finished = pyqtSignal(bool, str)
    output_message = pyqtSignal(str)  # For FFmpeg output
    
    def __init__(self, video_configs: List[VideoTrimConfig]):
        super().__init__()
        self.video_configs = video_configs
        self.cancelled = False
    
    def run(self):
        """Process all videos in the batch"""
        total_videos = len(self.video_configs)
        successful = 0
        failed = 0
        
        for video_idx, config in enumerate(self.video_configs):
            if self.cancelled:
                break
            
            self.status.emit(f"Processing video {video_idx + 1}/{total_videos}: {os.path.basename(config.video_path)}")
            
            try:
                success = self.process_video(config, video_idx)
                if success:
                    successful += 1
                    self.video_finished.emit(video_idx, True, f"✅ Completed: {os.path.basename(config.video_path)}")
                else:
                    failed += 1
                    self.video_finished.emit(video_idx, False, f"❌ Failed: {os.path.basename(config.video_path)}")
            except Exception as e:
                failed += 1
                self.video_finished.emit(video_idx, False, f"❌ Error: {str(e)}")
        
        summary = f"Batch complete: {successful} successful, {failed} failed"
        self.all_finished.emit(failed == 0, summary)
    
    def process_video(self, config: VideoTrimConfig, video_idx: int) -> bool:
        """Process a single video"""
        try:
            # Calculate end time and expected frames
            end_time = min(config.start_time + config.duration, config.total_duration)
            trim_duration = end_time - config.start_time
            expected_frames = int(trim_duration * config.fps)
            
            # Diagnostic output
            self.output_message.emit(f"\n{'='*80}\n")
            self.output_message.emit(f"📊 VIDEO DIAGNOSTICS:\n")
            self.output_message.emit(f"  Input file: {os.path.basename(config.video_path)}\n")
            self.output_message.emit(f"  Video FPS: {config.fps}\n")
            self.output_message.emit(f"  Video dimensions: {config.width}x{config.height}\n")
            self.output_message.emit(f"  Total duration: {config.total_duration:.2f}s\n")
            self.output_message.emit(f"  Trim start: {config.start_time:.2f}s\n")
            self.output_message.emit(f"  Trim end: {end_time:.2f}s\n")
            self.output_message.emit(f"  Trim duration: {trim_duration:.2f}s\n")
            self.output_message.emit(f"  ⚠️ EXPECTED OUTPUT FRAMES: {expected_frames}\n")
            self.output_message.emit(f"{'='*80}\n\n")
            
            # Build output path
            output_dir = os.path.dirname(config.video_path)
            output_file = os.path.join(output_dir, config.output_filename)
            
            # Build FFmpeg command
            command = [
                "ffmpeg",
                "-y",
                "-i", config.video_path,
                "-ss", str(config.start_time),
                "-to", str(end_time)
            ]
            
            # Add crop filter if defined
            temp_mask_file = None
            if len(config.crop_polygon) >= 3:
                temp_mask_file = self.create_mask(config)
                
                # Use movie filter to load mask and apply to every video frame
                # Convert Windows path to forward slashes and escape properly for FFmpeg
                mask_path = temp_mask_file.replace('\\', '/')
                # On Windows, escape the colon in drive letters for movie filter
                if len(mask_path) > 1 and mask_path[1] == ':':
                    mask_path = mask_path[0] + '\\:' + mask_path[2:]
                
                filter_complex = (
                    f"movie='{mask_path}',scale={config.width}:{config.height}[mask];"
                    f"[0:v][mask]alphamerge[vid_with_alpha];"
                    f"color=black:s={config.width}x{config.height}[black];"
                    f"[black][vid_with_alpha]overlay=format=auto"
                )
                command.extend(["-filter_complex", filter_complex])
            
            # Add encoding parameters
            command.extend([
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-avoid_negative_ts", "make_zero",
                "-fflags", "+genpts",
                "-an"
            ])
            
            # CRITICAL: Force exact frame count AND framerate if using mask
            if len(config.crop_polygon) >= 3:
                command.extend([
                    "-r", str(config.fps),  # Force output framerate to match input
                    "-frames:v", str(expected_frames)
                ])
            
            command.append(output_file)
            
            # Emit the FFmpeg command
            cmd_str = " ".join(command)
            self.output_message.emit(f"🎬 PROCESSING VIDEO {video_idx + 1}/{len(self.video_configs)}\n")
            self.output_message.emit(f"FFmpeg Command:\n{cmd_str}\n")
            self.output_message.emit(f"{'-'*80}\n")
            
            # Run FFmpeg
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output in real-time and track frame count
            final_frame_count = 0
            for line in iter(process.stdout.readline, ''):
                if self.cancelled:
                    process.terminate()
                    break
                if line.strip():
                    self.output_message.emit(line.rstrip() + "\n")
                    # Extract frame count from FFmpeg output
                    if 'frame=' in line:
                        try:
                            parts = line.split('frame=')
                            if len(parts) > 1:
                                frame_str = parts[1].split()[0]
                                final_frame_count = int(frame_str)
                        except:
                            pass
            
            process.wait()
            
            # Cleanup temp mask file
            if temp_mask_file and os.path.exists(temp_mask_file):
                try:
                    os.remove(temp_mask_file)
                except:
                    pass
            
            success = process.returncode == 0
            
            # Output final diagnostics
            self.output_message.emit(f"\n{'='*80}\n")
            self.output_message.emit(f"📊 PROCESSING RESULTS:\n")
            self.output_message.emit(f"  Expected frames: {expected_frames}\n")
            self.output_message.emit(f"  Actual frames: {final_frame_count}\n")
            self.output_message.emit(f"  Expected duration: {trim_duration:.2f}s @ {config.fps}fps\n")
            self.output_message.emit(f"  Calculated output duration: {final_frame_count / config.fps:.2f}s\n")
            
            if success:
                if final_frame_count == expected_frames:
                    self.output_message.emit(f"  ✅ Frame count matches expected!\n")
                    self.output_message.emit(f"✅ Successfully processed: {os.path.basename(config.video_path)}\n")
                else:
                    frame_diff = final_frame_count - expected_frames
                    self.output_message.emit(f"  ⚠️ WARNING: Frame count mismatch by {frame_diff} frames!\n")
                    self.output_message.emit(f"  ⚠️ Processed with unexpected frame count: {os.path.basename(config.video_path)}\n")
            else:
                self.output_message.emit(f"  ❌ FFmpeg returned error code: {process.returncode}\n")
                self.output_message.emit(f"❌ Failed to process: {os.path.basename(config.video_path)}\n")
            
            self.output_message.emit(f"{'='*80}\n\n")
            
            return success
            
        except Exception as e:
            self.status.emit(f"Error: {str(e)}")
            self.output_message.emit(f"❌ Exception: {str(e)}\n")
            return False
    
    def create_mask(self, config: VideoTrimConfig) -> str:
        """Create a temporary mask file for cropping"""
        mask = np.zeros((config.height, config.width, 3), dtype=np.uint8)
        pts = np.array(config.crop_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], (255, 255, 255))
        
        temp_mask = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        temp_mask_path = temp_mask.name
        temp_mask.close()
        cv2.imwrite(temp_mask_path, mask)
        
        return temp_mask_path
    
    def cancel(self):
        """Cancel processing"""
        self.cancelled = True


class VideoTrimTool(QMainWindow):
    """Main window for batch video trimming and cropping"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Trim and Crop - FieldNeuroToolbox")
        self.setGeometry(100, 100, 1600, 950)  # Increased size to prevent cutoff
        
        # State
        self.video_configs = []
        self.current_config_idx = 0
        self.preview_cap = None
        self.current_frame = None
        
        # Drawing state
        self.drawing_crop = False
        
        # Processing queue
        self.processing_queue = []
        
        # Worker thread
        self.processor = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Apply dark theme matching ROI tool
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
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
                color: #0078d4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #cccccc;
            }
            QLineEdit, QComboBox {
                padding: 5px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QListWidget {
                background-color: #2b2b2b;
                border: 1px solid #3f3f3f;
                color: #cccccc;
            }
            QProgressBar {
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                text-align: center;
                background-color: #2b2b2b;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)
        
        # Main container
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Left panel (video list and settings)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel (preview)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
    
    def create_left_panel(self):
        """Create the left panel with video list and settings"""
        panel = QWidget()
        panel.setMaximumWidth(450)  # Constrain the width
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Title
        title = QLabel("Video Trim and Crop")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #0078d4;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create scroll area for ALL sections
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # No horizontal scroll
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_content.setLayout(scroll_layout)
        
        # Video selection section
        video_section = self.create_video_selection_section()
        scroll_layout.addWidget(video_section)
        
        # Settings section
        settings_section = self.create_settings_section()
        scroll_layout.addWidget(settings_section)
        
        # Processing section
        processing_section = self.create_processing_section()
        scroll_layout.addWidget(processing_section)
        
        # Add scroll area to main layout
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        return panel
    
    def create_video_selection_section(self):
        """Create video selection UI"""
        group = QGroupBox("1. Video Selection")
        layout = QVBoxLayout()
        
        # Buttons
        button_layout = QHBoxLayout()
        
        btn_add_folder = QPushButton("📁 Add Folder")
        btn_add_folder.clicked.connect(self.add_folder)
        button_layout.addWidget(btn_add_folder)
        
        btn_add_videos = QPushButton("🎬 Add Video(s)")
        btn_add_videos.clicked.connect(self.add_videos)
        button_layout.addWidget(btn_add_videos)
        
        btn_clear = QPushButton("🗑️ Clear All")
        btn_clear.clicked.connect(self.clear_videos)
        button_layout.addWidget(btn_clear)
        
        layout.addLayout(button_layout)
        
        # Video list
        self.video_list = QListWidget()
        self.video_list.setMinimumHeight(150)
        self.video_list.currentRowChanged.connect(self.on_video_selected)
        layout.addWidget(self.video_list)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.btn_previous = QPushButton("⬅️ Previous")
        self.btn_previous.clicked.connect(self.previous_video)
        self.btn_previous.setEnabled(False)
        nav_layout.addWidget(self.btn_previous)
        
        self.btn_next = QPushButton("Next ➡️")
        self.btn_next.clicked.connect(self.next_video)
        self.btn_next.setEnabled(False)
        nav_layout.addWidget(self.btn_next)
        
        layout.addLayout(nav_layout)
        
        group.setLayout(layout)
        return group
    
    def create_settings_section(self):
        """Create settings UI"""
        group = QGroupBox("2. Configure Trim & Crop Settings")
        layout = QVBoxLayout()
        
        # Start position
        start_label = QLabel("Start Position:")
        start_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(start_label)
        
        self.start_time_input = QLineEdit()
        self.start_time_input.setPlaceholderText("00:00:00 (HH:MM:SS)")
        self.start_time_input.setEnabled(False)
        layout.addWidget(self.start_time_input)
        
        # Duration
        duration_label = QLabel("Duration:")
        duration_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(duration_label)
        
        duration_layout = QHBoxLayout()
        
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["Original Video Duration", "1 minute", "5 minutes", "10 minutes", "20 minutes", "30 minutes", "1 hour", "Custom"])
        self.duration_combo.setCurrentText("Original Video Duration")
        self.duration_combo.setEnabled(False)
        self.duration_combo.currentTextChanged.connect(self.on_duration_changed)
        duration_layout.addWidget(self.duration_combo)
        
        # Custom duration inputs
        self.custom_duration_input = QLineEdit()
        self.custom_duration_input.setPlaceholderText("Value")
        self.custom_duration_input.setMaximumWidth(80)
        self.custom_duration_input.setText("5")
        self.custom_duration_input.setVisible(False)
        duration_layout.addWidget(self.custom_duration_input)
        
        self.custom_duration_unit = QComboBox()
        self.custom_duration_unit.addItems(["seconds", "minutes", "hours"])
        self.custom_duration_unit.setCurrentText("minutes")
        self.custom_duration_unit.setMaximumWidth(100)
        self.custom_duration_unit.setVisible(False)
        duration_layout.addWidget(self.custom_duration_unit)
        
        layout.addLayout(duration_layout)
        
        # Crop ROI
        crop_label = QLabel("Crop ROI (Optional):")
        crop_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(crop_label)
        
        crop_button_layout = QHBoxLayout()
        
        self.btn_draw_crop = QPushButton("✏️ Select Crop ROI")
        self.btn_draw_crop.clicked.connect(self.start_drawing_crop)
        self.btn_draw_crop.setEnabled(False)
        crop_button_layout.addWidget(self.btn_draw_crop)
        
        self.btn_clear_crop = QPushButton("🗑️ Clear Crop")
        self.btn_clear_crop.clicked.connect(self.clear_crop)
        self.btn_clear_crop.setEnabled(False)
        crop_button_layout.addWidget(self.btn_clear_crop)
        
        layout.addLayout(crop_button_layout)
        
        self.crop_status_label = QLabel("No crop region defined")
        self.crop_status_label.setStyleSheet("color: #999999;")
        layout.addWidget(self.crop_status_label)
        
        # Output filename
        output_label = QLabel("Output Filename:")
        output_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(output_label)
        
        self.output_filename = QLineEdit()
        self.output_filename.setPlaceholderText("video_trimmed.mp4")
        self.output_filename.setEnabled(False)
        layout.addWidget(self.output_filename)
        
        # Add to queue button
        layout.addSpacing(10)
        self.btn_add_to_queue = QPushButton("➕ Add Video to Processing Queue")
        self.btn_add_to_queue.clicked.connect(self.add_to_queue)
        self.btn_add_to_queue.setEnabled(False)
        self.btn_add_to_queue.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                padding: 10px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
            }
        """)
        layout.addWidget(self.btn_add_to_queue)
        
        layout.addStretch()
        
        group.setLayout(layout)
        return group
    
    def create_processing_section(self):
        """Create processing controls"""
        group = QGroupBox("3. Batch Processing")
        layout = QVBoxLayout()
        
        # Queue status
        self.queue_status_label = QLabel("Clips in queue: 0")
        self.queue_status_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(self.queue_status_label)
        
        # Queue list widget
        queue_label = QLabel("Queued Clips:")
        queue_label.setFont(QFont("Arial", 9, QFont.Bold))
        layout.addWidget(queue_label)
        
        self.queue_table = QTableWidget()
        self.queue_table.setColumnCount(4)  # Added Remove column
        self.queue_table.setHorizontalHeaderLabels(["", "Clip Name", "Duration", "Origin"])
        self.queue_table.setMinimumHeight(150)  # Show at least 5 rows
        self.queue_table.setEditTriggers(QTableWidget.DoubleClicked)  # Allow double-click editing on clip name
        self.queue_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.queue_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.queue_table.horizontalHeader().setStretchLastSection(False)
        self.queue_table.setColumnWidth(0, 30)   # Remove button
        self.queue_table.setColumnWidth(1, 140)  # Clip Name
        self.queue_table.setColumnWidth(2, 70)   # Duration
        self.queue_table.setColumnWidth(3, 120)  # Origin Video
        self.queue_table.verticalHeader().setDefaultSectionSize(22)  # Smaller row height
        self.queue_table.setStyleSheet("""
            QTableWidget {
                background-color: #2b2b2b;
                border: 1px solid #3f3f3f;
                color: #cccccc;
                font-size: 8pt;
                gridline-color: #3f3f3f;
            }
            QTableWidget::item {
                padding: 1px;
            }
            QHeaderView::section {
                background-color: #3f3f3f;
                color: #cccccc;
                padding: 3px;
                border: 1px solid #2b2b2b;
                font-weight: bold;
                font-size: 8pt;
            }
        """)
        self.queue_table.cellChanged.connect(self.on_clip_name_changed)
        layout.addWidget(self.queue_table)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #999999;")
        layout.addWidget(self.progress_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.btn_start_batch = QPushButton("▶️ Start Batch Processing")
        self.btn_start_batch.clicked.connect(self.start_batch_processing)
        self.btn_start_batch.setEnabled(False)
        self.btn_start_batch.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
            }
        """)
        button_layout.addWidget(self.btn_start_batch)
        
        self.btn_cancel_batch = QPushButton("⏹️ Cancel")
        self.btn_cancel_batch.clicked.connect(self.cancel_batch_processing)
        self.btn_cancel_batch.setEnabled(False)
        button_layout.addWidget(self.btn_cancel_batch)
        
        layout.addLayout(button_layout)
        
        # Output window
        output_label = QLabel("FFmpeg Output:")
        output_label.setFont(QFont("Arial", 9, QFont.Bold))
        layout.addWidget(output_label)
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(250)  # Increased from 150
        self.output_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 9pt;
            }
        """)
        layout.addWidget(self.output_text)
        
        group.setLayout(layout)
        return group
    
    def create_right_panel(self):
        """Create the right panel with preview"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Preview section
        preview_group = QGroupBox("Video Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black; border: 2px solid #0078d4;")
        self.preview_label.setMinimumSize(960, 540)  # Larger preview (16:9 aspect ratio)
        self.preview_label.setText("Select a video to begin")
        preview_layout.addWidget(self.preview_label, stretch=1)
        
        # Start position controls (moved inside preview group, closer to video)
        position_controls = QVBoxLayout()
        position_controls.setSpacing(5)
        
        # Time display
        self.time_label = QLabel("Start Time: 00:00:00")
        self.time_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.time_label.setAlignment(Qt.AlignCenter)
        position_controls.addWidget(self.time_label)
        
        # Slider
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(100)
        self.position_slider.setValue(0)
        self.position_slider.setEnabled(False)
        self.position_slider.valueChanged.connect(self.on_slider_changed)
        position_controls.addWidget(self.position_slider)
        
        # Adjustment buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        adjustments = [("-60s", -60), ("-30s", -30), ("-10s", -10), ("-1s", -1),
                      ("+1s", +1), ("+10s", +10), ("+30s", +30), ("+60s", +60)]
        
        self.adjustment_buttons = []
        for label, seconds in adjustments:
            btn = QPushButton(label)
            btn.setFixedWidth(60)
            btn.clicked.connect(lambda checked, s=seconds: self.adjust_position(s))
            btn.setEnabled(False)
            button_layout.addWidget(btn)
            self.adjustment_buttons.append(btn)
        
        button_layout.addStretch()
        position_controls.addLayout(button_layout)
        
        preview_layout.addLayout(position_controls)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group, stretch=1)
        
        return panel
    
    # Video selection methods
    def add_folder(self):
        """Add all videos from a folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
            for filename in os.listdir(folder):
                if filename.lower().endswith(video_extensions):
                    video_path = os.path.join(folder, filename)
                    self.add_video_to_list(video_path)
    
    def add_videos(self):
        """Add individual videos"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Videos",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;All Files (*)"
        )
        for file_path in files:
            self.add_video_to_list(file_path)
    
    def add_video_to_list(self, video_path: str):
        """Add a video to the list"""
        # Check if already in list
        for config in self.video_configs:
            if config.video_path == video_path:
                return
        
        # Create config
        config = VideoTrimConfig(video_path)
        self.video_configs.append(config)
        
        # Add to list widget
        item = QListWidgetItem(f"📹 {os.path.basename(video_path)}")
        self.video_list.addItem(item)
        
        # Enable navigation
        self.btn_previous.setEnabled(len(self.video_configs) > 1)
        self.btn_next.setEnabled(len(self.video_configs) > 1)
        
        # Load first video automatically
        if len(self.video_configs) == 1:
            self.video_list.setCurrentRow(0)
    
    def clear_videos(self):
        """Clear all videos"""
        self.video_configs = []
        self.video_list.clear()
        self.processing_queue = []
        self.queue_table.setRowCount(0)  # Clear all rows from table
        self.update_queue_status()
        self.btn_previous.setEnabled(False)
        self.btn_next.setEnabled(False)
    
    def on_video_selected(self, row):
        """Handle video selection"""
        if row >= 0 and row < len(self.video_configs):
            self.current_config_idx = row
            self.load_current_video()
    
    def previous_video(self):
        """Go to previous video"""
        if self.current_config_idx > 0:
            self.video_list.setCurrentRow(self.current_config_idx - 1)
    
    def next_video(self):
        """Go to next video"""
        if self.current_config_idx < len(self.video_configs) - 1:
            self.video_list.setCurrentRow(self.current_config_idx + 1)
    
    def load_current_video(self):
        """Load the currently selected video"""
        if self.current_config_idx >= len(self.video_configs):
            return
        
        config = self.video_configs[self.current_config_idx]
        
        # Release previous video
        if self.preview_cap:
            self.preview_cap.release()
        
        # Open video
        self.preview_cap = cv2.VideoCapture(config.video_path)
        if not self.preview_cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video file.")
            return
        
        # Get properties
        config.width = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        config.height = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        config.fps = self.preview_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.preview_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        config.total_duration = total_frames / config.fps
        
        # Set default duration to full video if not configured
        if not config.configured:
            config.duration = config.total_duration
        
        # Read first frame
        ret, frame = self.preview_cap.read()
        if ret:
            config.first_frame = frame.copy()
            self.current_frame = frame.copy()
        
        # Set default output filename
        if not config.output_filename:
            base_name = os.path.splitext(os.path.basename(config.video_path))[0]
            config.output_filename = f"{base_name}_trimmed.mp4"
        
        # Update UI
        self.output_filename.setText(config.output_filename)
        self.output_filename.setEnabled(True)
        
        # Enable controls
        self.position_slider.setEnabled(True)
        self.position_slider.setMaximum(int(config.total_duration * 10))
        self.position_slider.setValue(int(config.start_time * 10))
        
        for btn in self.adjustment_buttons:
            btn.setEnabled(True)
        
        self.duration_combo.setEnabled(True)
        self.btn_draw_crop.setEnabled(True)
        self.start_time_input.setEnabled(True)
        self.btn_add_to_queue.setEnabled(True)
        
        # Restore settings if configured
        self.restore_video_settings(config)
        
        # Update preview
        self.update_preview()
    
    def restore_video_settings(self, config: VideoTrimConfig):
        """Restore settings for a video"""
        # Update start time display
        self.start_time_input.setText(self.format_time(config.start_time))
        self.time_label.setText(f"Start Time: {self.format_time(config.start_time)}")
        
        # Update crop status
        if len(config.crop_polygon) >= 3:
            self.crop_status_label.setText(f"✓ Crop region set with {len(config.crop_polygon)} points")
            self.crop_status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
            self.btn_clear_crop.setEnabled(True)
        else:
            self.crop_status_label.setText("No crop region defined")
            self.crop_status_label.setStyleSheet("color: #999999;")
            self.btn_clear_crop.setEnabled(False)
    
    def on_slider_changed(self, value):
        """Handle slider value change"""
        if self.current_config_idx >= len(self.video_configs):
            return
        
        config = self.video_configs[self.current_config_idx]
        config.start_time = value / 10.0
        
        self.start_time_input.setText(self.format_time(config.start_time))
        self.time_label.setText(f"Start Time: {self.format_time(config.start_time)}")
        self.update_preview()
    
    def adjust_position(self, seconds):
        """Adjust position by seconds"""
        if self.current_config_idx >= len(self.video_configs):
            return
        
        config = self.video_configs[self.current_config_idx]
        new_position = max(0, min(config.total_duration, config.start_time + seconds))
        self.position_slider.setValue(int(new_position * 10))
    
    def on_duration_changed(self, text):
        """Handle duration selection change"""
        is_custom = text == "Custom"
        self.custom_duration_input.setVisible(is_custom)
        self.custom_duration_unit.setVisible(is_custom)
        
        # Update config
        if self.current_config_idx < len(self.video_configs):
            config = self.video_configs[self.current_config_idx]
            config.duration = self.get_duration_from_ui()
    
    def get_duration_from_ui(self) -> float:
        """Get duration in seconds from UI"""
        selected = self.duration_combo.currentText()
        
        # Handle original video duration
        if selected == "Original Video Duration":
            if self.current_config_idx < len(self.video_configs):
                config = self.video_configs[self.current_config_idx]
                return config.total_duration
            return 300  # Fallback default
        
        duration_map = {
            "1 minute": 60,
            "5 minutes": 300,
            "10 minutes": 600,
            "20 minutes": 1200,
            "30 minutes": 1800,
            "1 hour": 3600
        }
        
        if selected == "Custom":
            try:
                value = float(self.custom_duration_input.text())
                unit = self.custom_duration_unit.currentText()
                
                if unit == "seconds":
                    return value
                elif unit == "minutes":
                    return value * 60
                elif unit == "hours":
                    return value * 3600
            except:
                return 300
        
        return duration_map.get(selected, 300)
    
    def start_drawing_crop(self):
        """Start drawing crop region"""
        if self.current_config_idx >= len(self.video_configs):
            return
        
        self.drawing_crop = True
        config = self.video_configs[self.current_config_idx]
        config.crop_polygon = []
        
        self.btn_draw_crop.setEnabled(False)
        self.crop_status_label.setText("Drawing... Click to add points. Press ENTER when done.")
        self.crop_status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        
        self.preview_label.setMouseTracking(True)
        self.preview_label.mousePressEvent = self.on_preview_click
        self.preview_label.setFocusPolicy(Qt.StrongFocus)
        self.preview_label.setFocus()
        self.preview_label.keyPressEvent = self.on_preview_key_press
    
    def on_preview_click(self, event):
        """Handle preview click for crop drawing"""
        if not self.drawing_crop or self.current_config_idx >= len(self.video_configs):
            return
        
        config = self.video_configs[self.current_config_idx]
        if self.current_frame is None:
            return
        
        # Get click coordinates
        label_width = self.preview_label.width()
        label_height = self.preview_label.height()
        
        frame_height, frame_width = self.current_frame.shape[:2]
        scale = min(label_width / frame_width, label_height / frame_height)
        
        x = int((event.x() - (label_width - frame_width * scale) / 2) / scale)
        y = int((event.y() - (label_height - frame_height * scale) / 2) / scale)
        
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        
        config.crop_polygon.append((x, y))
        self.update_preview()
        
        self.crop_status_label.setText(f"{len(config.crop_polygon)} points defined. Press ENTER when done.")
    
    def on_preview_key_press(self, event):
        """Handle key press on preview"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.drawing_crop:
                self.finish_drawing_crop()
    
    def finish_drawing_crop(self):
        """Finish drawing crop region"""
        if self.current_config_idx >= len(self.video_configs):
            return
        
        config = self.video_configs[self.current_config_idx]
        
        if len(config.crop_polygon) < 3:
            QMessageBox.warning(self, "Invalid Crop", "Please define at least 3 points for the crop region.")
            return
        
        self.drawing_crop = False
        self.btn_draw_crop.setEnabled(True)
        self.btn_clear_crop.setEnabled(True)
        
        self.crop_status_label.setText(f"✓ Crop region set with {len(config.crop_polygon)} points")
        self.crop_status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        
        self.preview_label.setMouseTracking(False)
        self.preview_label.setFocusPolicy(Qt.NoFocus)
        
        self.update_preview()
    
    def clear_crop(self):
        """Clear crop region"""
        if self.current_config_idx >= len(self.video_configs):
            return
        
        config = self.video_configs[self.current_config_idx]
        config.crop_polygon = []
        
        self.drawing_crop = False
        self.btn_draw_crop.setEnabled(True)
        self.btn_clear_crop.setEnabled(False)
        
        self.crop_status_label.setText("No crop region defined")
        self.crop_status_label.setStyleSheet("color: #999999;")
        
        self.preview_label.setMouseTracking(False)
        self.update_preview()
    
    def update_preview(self):
        """Update video preview"""
        if self.current_config_idx >= len(self.video_configs) or not self.preview_cap:
            return
        
        config = self.video_configs[self.current_config_idx]
        
        # Seek to position
        frame_number = int(config.start_time * config.fps)
        self.preview_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.preview_cap.read()
        
        if not ret:
            return
        
        self.current_frame = frame.copy()
        frame_display = frame.copy()
        
        # Draw crop polygon if defined
        if len(config.crop_polygon) > 0:
            for i in range(len(config.crop_polygon)):
                pt1 = config.crop_polygon[i]
                pt2 = config.crop_polygon[(i + 1) % len(config.crop_polygon)]
                cv2.line(frame_display, pt1, pt2, (0, 255, 0), 2)
            
            for pt in config.crop_polygon:
                cv2.circle(frame_display, pt, 5, (0, 255, 0), -1)
        
        # Convert to Qt format
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.preview_label.setPixmap(scaled_pixmap)
    
    def add_to_queue(self):
        """Add current video clip to processing queue"""
        if self.current_config_idx >= len(self.video_configs):
            return
        
        config = self.video_configs[self.current_config_idx]
        
        # Get current UI values
        output_filename = self.output_filename.text().strip()
        
        if not output_filename:
            QMessageBox.warning(self, "Invalid Filename", "Please enter an output filename.")
            return
        
        if not output_filename.lower().endswith('.mp4'):
            output_filename += '.mp4'
        
        # Check for duplicate filenames in queue
        for queued_config in self.processing_queue:
            if queued_config.output_filename == output_filename:
                reply = QMessageBox.warning(
                    self, 
                    "Duplicate Filename", 
                    f"The filename '{output_filename}' is already in the processing queue.\n\nDo you want to add it anyway? (This will overwrite the previous file)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
        
        # Create a copy of the config for this specific clip
        clip_config = VideoTrimConfig(config.video_path)
        clip_config.video_path = config.video_path
        clip_config.start_time = config.start_time
        clip_config.duration = self.get_duration_from_ui()
        clip_config.crop_polygon = config.crop_polygon.copy()  # Keep the crop
        clip_config.output_filename = output_filename
        clip_config.configured = True
        clip_config.width = config.width
        clip_config.height = config.height
        clip_config.fps = config.fps
        clip_config.total_duration = config.total_duration
        
        # Add to queue
        self.processing_queue.append(clip_config)
        
        # Add to queue table
        row_position = self.queue_table.rowCount()
        self.queue_table.insertRow(row_position)
        
        # Block signals to prevent cellChanged from firing during setup
        self.queue_table.blockSignals(True)
        
        # Remove button (red minus sign)
        remove_btn = QPushButton("🗑")
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 2px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        remove_btn.setMaximumSize(25, 20)
        remove_btn.clicked.connect(lambda checked, row=row_position: self.remove_clip_from_queue(row))
        self.queue_table.setCellWidget(row_position, 0, remove_btn)
        
        # Clip Name (editable by double-click)
        clip_name_item = QTableWidgetItem(output_filename)
        clip_name_item.setFlags(clip_name_item.flags() | Qt.ItemIsEditable)
        self.queue_table.setItem(row_position, 1, clip_name_item)
        
        # Duration (read-only)
        duration_str = self.format_time(clip_config.duration)
        duration_item = QTableWidgetItem(duration_str)
        duration_item.setFlags(duration_item.flags() & ~Qt.ItemIsEditable)
        self.queue_table.setItem(row_position, 2, duration_item)
        
        # Origin Video (read-only)
        origin_video = os.path.basename(config.video_path)
        origin_item = QTableWidgetItem(origin_video)
        origin_item.setFlags(origin_item.flags() & ~Qt.ItemIsEditable)
        self.queue_table.setItem(row_position, 3, origin_item)
        
        # Re-enable signals
        self.queue_table.blockSignals(False)
        
        self.update_queue_status()
        
        # Reset output filename to default (indicates new clip from same video)
        base_name = os.path.splitext(os.path.basename(config.video_path))[0]
        config.output_filename = f"{base_name}_trimmed.mp4"
        self.output_filename.setText(config.output_filename)
        
        # DON'T auto-advance to next video - user can make multiple clips from same video
        # Crop stays applied for next clip
    
    def remove_clip_from_queue(self, row):
        """Remove a clip from the queue by row index"""
        if 0 <= row < len(self.processing_queue):
            # Remove from processing queue
            self.processing_queue.pop(row)
            
            # Remove from table
            self.queue_table.removeRow(row)
            
            # Update remaining remove button connections (row indices have shifted)
            for i in range(row, self.queue_table.rowCount()):
                widget = self.queue_table.cellWidget(i, 0)
                if widget:
                    # Reconnect with updated row index
                    widget.clicked.disconnect()
                    widget.clicked.connect(lambda checked, r=i: self.remove_clip_from_queue(r))
            
            self.update_queue_status()
    
    def on_clip_name_changed(self, row, column):
        """Handle clip name changes in the table"""
        # Only process changes to column 1 (Clip Name)
        if column == 1 and 0 <= row < len(self.processing_queue):
            new_name = self.queue_table.item(row, column).text().strip()
            
            if new_name:
                # Ensure .mp4 extension
                if not new_name.lower().endswith('.mp4'):
                    new_name += '.mp4'
                
                # Update the config
                self.processing_queue[row].output_filename = new_name
                
                # Update the table cell (in case we added .mp4)
                self.queue_table.blockSignals(True)
                self.queue_table.item(row, column).setText(new_name)
                self.queue_table.blockSignals(False)
    
    def update_queue_status(self):
        """Update queue status label"""
        self.queue_status_label.setText(f"Clips in queue: {len(self.processing_queue)}")
        self.btn_start_batch.setEnabled(len(self.processing_queue) > 0)
    
    def start_batch_processing(self):
        """Start batch processing"""
        if len(self.processing_queue) == 0:
            return
        
        # Clear output window
        self.output_text.clear()
        
        # processing_queue now contains VideoTrimConfig objects directly
        configs_to_process = self.processing_queue
        
        # Start worker
        self.processor = BatchTrimWorker(configs_to_process)
        self.processor.progress.connect(self.on_batch_progress)
        self.processor.status.connect(self.on_batch_status)
        self.processor.video_finished.connect(self.on_video_finished)
        self.processor.all_finished.connect(self.on_batch_finished)
        self.processor.output_message.connect(self.on_output_message)
        self.processor.start()
        
        # Update UI
        self.btn_start_batch.setEnabled(False)
        self.btn_cancel_batch.setEnabled(True)
        self.progress_bar.setValue(0)
    
    def cancel_batch_processing(self):
        """Cancel batch processing"""
        if self.processor and self.processor.isRunning():
            self.processor.cancel()
    
    def on_batch_progress(self, video_idx, percent):
        """Handle batch progress update"""
        total = len(self.processing_queue)
        overall_progress = int((video_idx * 100 + percent) / total)
        self.progress_bar.setValue(overall_progress)
    
    def on_batch_status(self, message):
        """Handle batch status update"""
        self.progress_label.setText(message)
    
    def on_output_message(self, message):
        """Handle FFmpeg output message"""
        self.output_text.insertPlainText(message)
        # Auto-scroll to bottom
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )
    
    def on_video_finished(self, video_idx, success, message):
        """Handle individual video completion"""
        self.progress_label.setText(message)
    
    def on_batch_finished(self, success, message):
        """Handle batch completion"""
        self.btn_start_batch.setEnabled(True)
        self.btn_cancel_batch.setEnabled(False)
        self.progress_bar.setValue(100)
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Batch Complete", message)
        
        # Clear queue
        self.processing_queue = []
        self.queue_table.setRowCount(0)  # Clear all rows from table
        self.update_queue_status()
    
    def format_time(self, seconds):
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.preview_cap:
            self.preview_cap.release()
        event.accept()


def video_trim():
    """Launch the video trim tool"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = VideoTrimTool()
    window.show()
    
    if app:
        app.exec_()


if __name__ == "__main__":
    video_trim()
