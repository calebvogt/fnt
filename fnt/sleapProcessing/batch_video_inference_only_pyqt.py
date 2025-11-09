"""
SLEAP Video Inference Only - PyQt5 Implementation

Run SLEAP inference without tracking on video files.
Supports both top-down (centroid + centered instance) and bottom-up models.
Automatically converts output to CSV format.
"""

import os
import subprocess
import glob
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QMessageBox, QGroupBox, QTextEdit, QCheckBox,
    QListWidget, QGridLayout, QFrame, QApplication, QComboBox, QDoubleSpinBox, QSpinBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont


class InferenceWorker(QThread):
    """Worker thread for running SLEAP inference"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, video_folders, model_paths, overwrite_existing, create_csv=True, create_video=False, add_tracking=False, 
                 tracker_method="simple", similarity_method="instance", match_method="greedy", 
                 max_tracks=0, track_window=5, robust_quantile=1.0, conda_env=None):
        super().__init__()
        self.video_folders = video_folders
        self.model_paths = model_paths
        self.overwrite_existing = overwrite_existing
        self.create_csv = create_csv
        self.create_video = create_video
        self.add_tracking = add_tracking
        self.tracker_method = tracker_method
        self.similarity_method = similarity_method
        self.match_method = match_method
        self.max_tracks = max_tracks
        self.track_window = track_window
        self.robust_quantile = robust_quantile
        self.conda_env = conda_env
        self._stop_requested = False
    
    def request_stop(self):
        """Request the worker to stop processing"""
        self._stop_requested = True
        self.progress.emit("\n⚠️ Stop requested by user...")
        
    def run(self):
        try:
            total_processed = 0
            total_skipped = 0
            
            for folder in self.video_folders:
                if self._stop_requested:
                    break
                    
                video_files = [f for f in os.listdir(folder)
                              if f.lower().endswith((".mp4", ".avi", ".mov"))
                              and not f.endswith("_roiTracked.mp4")]  # Ignore ROI tracked videos
                
                if not video_files:
                    self.progress.emit(f"⚠️ No video files found in: {folder}\n")
                    continue
                
                self.progress.emit(f"\n📁 Processing folder: {folder}")
                self.progress.emit(f"Found {len(video_files)} video file(s)\n")
                
                for video_file in video_files:
                    if self._stop_requested:
                        break
                        
                    full_path = os.path.join(folder, video_file)
                    
                    # Check for existing prediction files with any timestamp
                    existing_files = self.find_existing_predictions(full_path)
                    
                    if existing_files and not self.overwrite_existing:
                        self.progress.emit(f"⏭️ Skipping {video_file} (existing predictions detected)")
                        total_skipped += 1
                        continue
                    
                    # Delete existing files if overwrite is enabled
                    if existing_files and self.overwrite_existing:
                        self.progress.emit(f"🗑️ Deleting {len(existing_files)} existing prediction file(s) for {video_file}")
                        for existing_file in existing_files:
                            try:
                                os.remove(existing_file)
                                self.progress.emit(f"   Deleted: {os.path.basename(existing_file)}")
                            except Exception as e:
                                self.progress.emit(f"   ⚠️ Failed to delete {os.path.basename(existing_file)}: {str(e)}")
                    
                    # Run inference
                    slp_path = self.get_output_path(full_path)
                    success = self.run_inference_on_video(full_path)
                    if success:
                        total_processed += 1
                        # Convert to CSV
                        self.convert_to_csv(slp_path)
                    else:
                        self.progress.emit(f"❌ Failed to process {video_file}")
            
            summary = f"\n{'='*60}\n"
            if self._stop_requested:
                summary += f"⚠️ Inference stopped by user!\n"
            else:
                summary += f"✅ Inference complete!\n"
            summary += f"Videos processed: {total_processed}\n"
            summary += f"Videos skipped: {total_skipped}\n"
            summary += f"Total videos: {total_processed + total_skipped}\n"
            
            self.progress.emit(summary)
            
            if self._stop_requested:
                self.finished.emit(False, "Inference stopped by user")
            else:
                self.finished.emit(True, "Inference completed successfully!")
            
        except Exception as e:
            self.finished.emit(False, f"Error during inference: {str(e)}")
    
    def find_existing_predictions(self, video_path):
        """Find all existing prediction files for a video (with any timestamp)"""
        base = os.path.basename(video_path)
        parent = os.path.dirname(video_path)
        
        # Search patterns for .slp, .csv, and .mp4 prediction files
        patterns = [
            os.path.join(parent, f"{base}.*.predictions.slp"),
            os.path.join(parent, f"{base}.*.predictions.analysis.csv"),
            os.path.join(parent, f"{base}.*.predictions.mp4")
        ]
        
        existing_files = []
        for pattern in patterns:
            existing_files.extend(glob.glob(pattern))
        
        return existing_files
    
    def get_output_path(self, video_path):
        """Generate output file path with timestamp"""
        base = os.path.basename(video_path)
        parent = os.path.dirname(video_path)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = f"{base}.{timestamp}.predictions.slp"
        return os.path.join(parent, filename)
    
    def run_inference_on_video(self, video_file):
        """Run SLEAP inference on a single video"""
        cmd = ["sleap-track", video_file]
        
        # In SLEAP v1.5.1, -m expects the model directory, not the training_config.json file
        for model_path in self.model_paths:
            cmd += ["-m", model_path]  # Pass the directory directly
        
        output_file = self.get_output_path(video_file)
        
        # Basic inference parameters for SLEAP v1.5.1
        cmd += [
            "-o", output_file,
            "--batch_size", "4",
            "--gpu", "auto",
            "--peak_threshold", "0.2"  # This IS available in v1.5.1!
        ]
        
        # Add tracking parameters if enabled (using correct v1.5.1 syntax)
        if self.add_tracking:
            cmd += [
                "--tracking.tracker", self.tracker_method,
                "--tracking.similarity", self.similarity_method,
                "--tracking.match", self.match_method
            ]
            
            # Add advanced tracking parameters if provided
            if hasattr(self, 'max_tracks') and self.max_tracks > 0:
                cmd += ["--tracking.max_tracks", str(self.max_tracks)]
            
            if hasattr(self, 'track_window') and self.track_window != 5:  # 5 is default
                cmd += ["--tracking.track_window", str(self.track_window)]
                
            if hasattr(self, 'robust_quantile') and self.robust_quantile != 1.0:  # 1.0 is default
                cmd += ["--tracking.robust", str(self.robust_quantile)]
        
        self.progress.emit(f"\n🔁 Running inference on: {os.path.basename(video_file)}")
        
        # Build full command with conda run
        if self.conda_env:
            full_cmd = ["conda", "run", "-n", self.conda_env] + cmd
            self.progress.emit(f"Environment: {self.conda_env}")
        else:
            full_cmd = cmd
            self.progress.emit("Warning: No conda environment specified")
        
        self.progress.emit(f"Command: {' '.join(full_cmd)}\n")
        
        try:
            # Execute command using conda run
            # Note: SLEAP doesn't stream progress, so we'll show a generic progress indicator
            self.progress.emit("🔄 Running SLEAP inference... (this may take several minutes)")
            self.progress.emit("⏳ SLEAP is processing frames - progress will appear when complete")
            
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                shell=True
            )
            
            # Show all output after completion
            if result.stdout:
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.strip():
                        self.progress.emit(line)
            
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines:
                    if line.strip():
                        self.progress.emit(f"⚠️ {line}")
            
            if result.returncode == 0:
                self.progress.emit(f"✅ Inference completed: {os.path.basename(output_file)}")
                
                # Convert to CSV if requested
                if self.create_csv:
                    self.convert_to_csv(output_file)
                
                # Render tracked video if requested
                if self.create_video:
                    self.render_video(output_file, video_file)
                    
                return True
            else:
                self.progress.emit(f"❌ Inference failed with return code {result.returncode}")
                return False
                
        except Exception as e:
            self.progress.emit(f"❌ Error running inference: {str(e)}")
            return False
    
    def convert_to_csv(self, slp_file):
        """Convert SLP file to CSV format using SLEAP v1.5.1 syntax"""
        csv_file = slp_file.replace(".predictions.slp", ".predictions.analysis.csv")
        
        if os.path.exists(csv_file) and not self.overwrite_existing:
            self.progress.emit(f"⏭️ Skipping CSV conversion: {os.path.basename(csv_file)} already exists")
            return
        
        self.progress.emit(f"� Converting to CSV: {os.path.basename(slp_file)}")
        
        try:
            # Build conversion command using conda run
            cmd = ["sleap-convert", "--format", "analysis.csv", "-o", csv_file, slp_file]
            
            if self.conda_env:
                full_cmd = ["conda", "run", "-n", self.conda_env] + cmd
            else:
                full_cmd = cmd
                
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                shell=True
            )
            
            if result.returncode == 0:
                self.progress.emit(f"✅ CSV created: {os.path.basename(csv_file)}")
            else:
                self.progress.emit(f"⚠️ CSV conversion failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
                
        except Exception as e:
            self.progress.emit(f"⚠️ CSV conversion error: {str(e)}")
    
    def render_video(self, slp_file, video_file):
        """Render tracked video with predictions overlay using sleap-render"""
        # Generate output video path
        video_output = slp_file.replace(".predictions.slp", ".predictions.mp4")
        
        if os.path.exists(video_output) and not self.overwrite_existing:
            self.progress.emit(f"⏭️ Skipping video rendering: {os.path.basename(video_output)} already exists")
            return
        
        self.progress.emit(f"🎬 Rendering tracked video: {os.path.basename(slp_file)}")
        
        try:
            # Get frame count from original video to render all frames
            frame_count = self.get_video_frame_count(video_file)
            
            # Build sleap-render command
            cmd = ["sleap-render", slp_file, "-o", video_output]
            
            # Add frame range to render all frames (even those without predictions)
            if frame_count > 0:
                cmd.extend(["--frames", f"1-{frame_count}"])
                self.progress.emit(f"   Rendering all {frame_count} frames (1-{frame_count})")
            
            if self.conda_env:
                full_cmd = ["conda", "run", "-n", self.conda_env] + cmd
            else:
                full_cmd = cmd
            
            # Log the command being executed
            cmd_str = " ".join(full_cmd)
            self.progress.emit(f"   Command: {cmd_str}")
            self.progress.emit(f"   Output: {video_output}")
                
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                shell=True
            )
            
            if result.returncode == 0:
                self.progress.emit(f"✅ Video rendered: {os.path.basename(video_output)}")
            else:
                self.progress.emit(f"⚠️ Video rendering failed (return code {result.returncode})")
                if result.stderr:
                    self.progress.emit(f"   stderr: {result.stderr[:500]}")
                if result.stdout:
                    self.progress.emit(f"   stdout: {result.stdout[:500]}")
                
        except Exception as e:
            self.progress.emit(f"⚠️ Video rendering error: {str(e)}")
    
    def get_video_frame_count(self, video_file):
        """Get the total number of frames in a video using ffprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                video_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.strip())
        except:
            pass
        
        return 0


class VideoInferenceWindow(QWidget):
    """PyQt5 window for SLEAP video inference configuration and execution"""
    
    def __init__(self):
        super().__init__()
        self.video_folders = []
        self.model_paths = []
        self.is_top_down = False
        self.worker = None
        self.init_ui()
        
        # Auto-detect conda environments after UI is ready
        QTimer.singleShot(500, self.detect_conda_environments)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("SLEAP Video Inference Only")
        self.setGeometry(100, 100, 1000, 800)  # Increased from 900x700 to 1000x800
        self.setMinimumSize(900, 750)  # Increased from 800x600 to 900x750
        
        # Apply dark theme styling
        self.setStyleSheet("""
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
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                color: #cccccc;
                padding: 5px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                color: #cccccc;
                font-family: Consolas, Courier New, monospace;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
                font-family: Consolas, Courier New, monospace;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("SLEAP Video Inference")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Run SLEAP inference with optional tracking\nAutomatically converts output to CSV format")
        desc.setFont(QFont("Arial", 10))
        desc.setStyleSheet("color: #999999; font-style: italic; background-color: transparent; margin-bottom: 10px;")
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)
        
        # Video Folders Group
        video_group = QGroupBox("1. Select Video Folders")
        video_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        btn_add_folder = QPushButton("Add Folder")
        btn_add_folder.clicked.connect(self.add_video_folder)
        btn_layout.addWidget(btn_add_folder)
        
        btn_clear_folders = QPushButton("Clear All")
        btn_clear_folders.clicked.connect(self.clear_video_folders)
        btn_layout.addWidget(btn_clear_folders)
        btn_layout.addStretch()
        video_layout.addLayout(btn_layout)
        
        self.folder_list = QListWidget()
        self.folder_list.setMaximumHeight(100)
        video_layout.addWidget(self.folder_list)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Model Selection Group
        model_group = QGroupBox("2. Select SLEAP Model(s)")
        model_layout = QVBoxLayout()
        
        # Model type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Model Type:"))
        
        btn_topdown = QPushButton("Top-Down (Centroid + Centered)")
        btn_topdown.clicked.connect(self.select_topdown_models)
        type_layout.addWidget(btn_topdown)
        
        btn_bottomup = QPushButton("Bottom-Up")
        btn_bottomup.clicked.connect(self.select_bottomup_models)
        type_layout.addWidget(btn_bottomup)
        
        type_layout.addStretch()
        model_layout.addLayout(type_layout)
        
        # Model paths display
        self.model_list = QListWidget()
        self.model_list.setMaximumHeight(80)
        model_layout.addWidget(self.model_list)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # SLEAP Environment Group
        env_group = QGroupBox("3. Select SLEAP Environment")
        env_layout = QVBoxLayout()
        
        # Environment selection
        env_select_layout = QHBoxLayout()
        env_select_layout.addWidget(QLabel("Conda Environment:"))
        
        self.combo_environment = QComboBox()
        self.combo_environment.setMinimumWidth(200)
        self.combo_environment.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
            }
        """)
        env_select_layout.addWidget(self.combo_environment)
        
        btn_refresh_envs = QPushButton("Refresh")
        btn_refresh_envs.clicked.connect(self.detect_conda_environments)
        btn_refresh_envs.setMaximumWidth(80)
        env_select_layout.addWidget(btn_refresh_envs)
        
        btn_test_sleap = QPushButton("Test SLEAP")
        btn_test_sleap.clicked.connect(self.test_sleap_in_environment)
        btn_test_sleap.setMaximumWidth(100)
        env_select_layout.addWidget(btn_test_sleap)
        
        env_select_layout.addStretch()
        env_layout.addLayout(env_select_layout)
        
        # Environment status
        self.lbl_env_status = QLabel("Click 'Refresh' to detect conda environments")
        self.lbl_env_status.setStyleSheet("color: #999999; font-style: italic;")
        env_layout.addWidget(self.lbl_env_status)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)
        
        # Options Group
        options_group = QGroupBox("4. Processing Options")
        options_layout = QVBoxLayout()
        
        # CSV Creation Option (first and checked by default)
        self.chk_create_csv = QCheckBox("Create CSV prediction file (in addition to .slp)")
        self.chk_create_csv.setChecked(True)
        options_layout.addWidget(self.chk_create_csv)
        
        # Rendered Video Option
        self.chk_create_video = QCheckBox("Create tracked video with predictions overlay")
        self.chk_create_video.setChecked(False)
        options_layout.addWidget(self.chk_create_video)
        
        self.chk_overwrite = QCheckBox("Overwrite existing prediction files")
        self.chk_overwrite.setChecked(True)
        options_layout.addWidget(self.chk_overwrite)
        
        # Tracking Options
        self.chk_tracking = QCheckBox("Add Tracking (assign identities across frames)")
        self.chk_tracking.setChecked(False)
        self.chk_tracking.stateChanged.connect(self.toggle_tracking_options)
        options_layout.addWidget(self.chk_tracking)
        
        # Tracking Parameters Container
        self.tracking_widget = QWidget()
        tracking_layout = QVBoxLayout()
        tracking_layout.setContentsMargins(20, 5, 0, 5)  # Indent tracking options
        
        # Create a grid layout for better organization
        grid_layout = QGridLayout()
        
        # Row 1: Tracker Method and Similarity Method
        grid_layout.addWidget(QLabel("Tracker Method:"), 0, 0)
        self.combo_tracker = QComboBox()
        self.combo_tracker.addItems(["simple", "flow", "simplemaxtracks", "flowmaxtracks", "None"])
        self.combo_tracker.setCurrentText("simple")
        self.combo_tracker.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
            }
        """)
        grid_layout.addWidget(self.combo_tracker, 0, 1)
        
        grid_layout.addWidget(QLabel("Similarity Method:"), 0, 2)
        self.combo_similarity = QComboBox()
        self.combo_similarity.addItems(["instance", "normalized_instance", "object_keypoint", "centroid", "iou"])
        self.combo_similarity.setCurrentText("instance")
        self.combo_similarity.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
            }
        """)
        grid_layout.addWidget(self.combo_similarity, 0, 3)
        
        # Row 2: Match Method and Max Tracks
        grid_layout.addWidget(QLabel("Match Method:"), 1, 0)
        self.combo_match = QComboBox()
        self.combo_match.addItems(["greedy", "hungarian"])
        self.combo_match.setCurrentText("greedy")
        self.combo_match.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
            }
        """)
        grid_layout.addWidget(self.combo_match, 1, 1)
        
        grid_layout.addWidget(QLabel("Max Tracks:"), 1, 2)
        self.spin_max_tracks = QSpinBox()
        self.spin_max_tracks.setMinimum(0)
        self.spin_max_tracks.setMaximum(50)
        self.spin_max_tracks.setValue(0)  # 0 = no limit
        self.spin_max_tracks.setSpecialValueText("No limit")
        self.spin_max_tracks.setStyleSheet("""
            QSpinBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        grid_layout.addWidget(self.spin_max_tracks, 1, 3)
        
        # Row 3: Track Window and Robust Quantile
        grid_layout.addWidget(QLabel("Track Window:"), 2, 0)
        self.spin_track_window = QSpinBox()
        self.spin_track_window.setMinimum(1)
        self.spin_track_window.setMaximum(20)
        self.spin_track_window.setValue(5)
        self.spin_track_window.setToolTip("How many frames back to look for matches")
        self.spin_track_window.setStyleSheet("""
            QSpinBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        grid_layout.addWidget(self.spin_track_window, 2, 1)
        
        grid_layout.addWidget(QLabel("Robust Quantile:"), 2, 2)
        self.spin_robust = QDoubleSpinBox()
        self.spin_robust.setMinimum(0.0)
        self.spin_robust.setMaximum(1.0)
        self.spin_robust.setSingleStep(0.05)
        self.spin_robust.setValue(1.0)
        self.spin_robust.setToolTip("Robust quantile of similarity score (1.0 = non-robust)")
        self.spin_robust.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        grid_layout.addWidget(self.spin_robust, 2, 3)
        
        tracking_layout.addLayout(grid_layout)
        self.tracking_widget.setLayout(tracking_layout)
        self.tracking_widget.setVisible(False)  # Hidden by default
        options_layout.addWidget(self.tracking_widget)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Run and Stop Buttons
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶️ Run Inference")
        self.btn_run.clicked.connect(self.run_inference)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #16825d;
                padding: 10px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1a9667;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
            }
        """)
        btn_layout.addWidget(self.btn_run)
        
        self.btn_stop = QPushButton("⏹️ Stop Processing")
        self.btn_stop.clicked.connect(self.stop_inference)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #c42b1c;
                padding: 10px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #d83b2c;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
            }
        """)
        btn_layout.addWidget(self.btn_stop)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Output Log Group
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_output)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        self.setLayout(layout)
        self.log("Ready to configure SLEAP inference...")
        self.log("1. Add video folder(s)")
        self.log("2. Select model type and model folder(s)")
        self.log("3. Select SLEAP conda environment")
        self.log("4. Click 'Run Inference' to start\n")
    
    def log(self, message):
        """Add message to log output"""
        self.log_output.append(message)
        # Auto-scroll to bottom
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )
    
    def add_video_folder(self):
        """Add a video folder to the processing list"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select folder containing video files for inference"
        )
        
        if folder:
            if folder not in self.video_folders:
                self.video_folders.append(folder)
                self.folder_list.addItem(folder)
                self.log(f"✅ Added video folder: {folder}")
                self.update_run_button()
            else:
                QMessageBox.information(self, "Duplicate Folder", "This folder has already been added.")
    
    def clear_video_folders(self):
        """Clear all video folders"""
        self.video_folders.clear()
        self.folder_list.clear()
        self.log("🗑️ Cleared all video folders")
        self.update_run_button()
    
    def toggle_tracking_options(self, state):
        """Show/hide tracking options based on checkbox state"""
        self.tracking_widget.setVisible(state == Qt.Checked)
    
    def detect_conda_environments(self):
        """Detect all conda environments and populate the dropdown"""
        self.combo_environment.clear()
        self.combo_environment.addItem("Detecting environments...", None)
        self.lbl_env_status.setText("Detecting conda environments...")
        self.lbl_env_status.setStyleSheet("color: #0078d4;")
        
        try:
            # Get list of conda environments
            result = subprocess.run(
                ['conda', 'env', 'list'],
                capture_output=True,
                text=True,
                shell=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.combo_environment.clear()
                env_lines = result.stdout.strip().split('\n')
                environments = []
                
                for line in env_lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract environment name (first word)
                        parts = line.split()
                        if parts:
                            env_name = parts[0]
                            environments.append(env_name)
                
                # Sort environments, but put common SLEAP names first
                sleap_priority = ['sleap', 'SLEAP', 'sleap-env', 'sleap_env']
                priority_envs = [env for env in environments if env in sleap_priority]
                other_envs = [env for env in environments if env not in sleap_priority]
                
                # Add environments to dropdown
                for env in priority_envs + sorted(other_envs):
                    self.combo_environment.addItem(env, env)
                
                self.lbl_env_status.setText(f"Found {len(environments)} conda environments")
                self.lbl_env_status.setStyleSheet("color: #cccccc;")
                self.log(f"✅ Detected {len(environments)} conda environments")
                
                # Auto-select if there's a likely SLEAP environment
                if priority_envs:
                    index = self.combo_environment.findText(priority_envs[0])
                    if index >= 0:
                        self.combo_environment.setCurrentIndex(index)
                        self.log(f"🎯 Auto-selected likely SLEAP environment: {priority_envs[0]}")
                
            else:
                self.combo_environment.clear()
                self.combo_environment.addItem("Error detecting environments", None)
                self.lbl_env_status.setText("Error: Could not detect conda environments")
                self.lbl_env_status.setStyleSheet("color: #ff6b6b;")
                self.log("❌ Error detecting conda environments")
                
        except subprocess.TimeoutExpired:
            self.combo_environment.clear()
            self.combo_environment.addItem("Timeout", None)
            self.lbl_env_status.setText("Timeout: conda env list took too long")
            self.lbl_env_status.setStyleSheet("color: #ff6b6b;")
            self.log("❌ Timeout while detecting conda environments")
            
        except Exception as e:
            self.combo_environment.clear()
            self.combo_environment.addItem("Error", None)
            self.lbl_env_status.setText(f"Error: {str(e)}")
            self.lbl_env_status.setStyleSheet("color: #ff6b6b;")
            self.log(f"❌ Error detecting conda environments: {str(e)}")
    
    def test_sleap_in_environment(self):
        """Test if SLEAP is available in the selected environment"""
        current_env = self.combo_environment.currentData()
        if not current_env:
            QMessageBox.warning(self, "No Environment", "Please select a conda environment first.")
            return
        
        self.lbl_env_status.setText(f"Testing SLEAP in '{current_env}'...")
        self.lbl_env_status.setStyleSheet("color: #0078d4;")
        self.log(f"🔍 Testing SLEAP in environment: {current_env}")
        
        try:
            # Test if sleap-track command is available and get help
            result = subprocess.run(
                ['conda', 'run', '-n', current_env, 'sleap-track', '--help'],
                capture_output=True,
                text=True,
                shell=True,
                timeout=15
            )
            
            if result.returncode == 0:
                # Get SLEAP version
                version_result = subprocess.run(
                    ['conda', 'run', '-n', current_env, 'python', '-c', 'import sleap; print(sleap.__version__)'],
                    capture_output=True,
                    text=True,
                    shell=True,
                    timeout=10
                )
                
                version = "unknown"
                if version_result.returncode == 0:
                    version = version_result.stdout.strip()
                
                # Log the help output for debugging
                self.log(f"📋 SLEAP v{version} help output:")
                help_lines = result.stdout.split('\n')  # Show ALL lines to find verbose options
                for line in help_lines:
                    if line.strip():
                        self.log(f"  {line}")
                
                self.lbl_env_status.setText(f"✅ SLEAP v{version} is available in '{current_env}'")
                self.lbl_env_status.setStyleSheet("color: #4caf50;")
                self.log(f"✅ SLEAP v{version} is working in environment: {current_env}")
                self.update_run_button()
                QMessageBox.information(
                    self, 
                    "SLEAP Test Successful", 
                    f"SLEAP v{version} is properly installed and working in environment '{current_env}'"
                )
            else:
                self.lbl_env_status.setText(f"❌ SLEAP not found in '{current_env}'")
                self.lbl_env_status.setStyleSheet("color: #ff6b6b;")
                self.log(f"❌ SLEAP not found in environment: {current_env}")
                QMessageBox.warning(
                    self, 
                    "SLEAP Not Found", 
                    f"SLEAP is not installed or not working in environment '{current_env}'\n\n"
                    f"Error: {result.stderr[:200] if result.stderr else 'Command failed'}"
                )
                
        except subprocess.TimeoutExpired:
            self.lbl_env_status.setText(f"❌ Timeout testing '{current_env}'")
            self.lbl_env_status.setStyleSheet("color: #ff6b6b;")
            self.log(f"❌ Timeout testing SLEAP in environment: {current_env}")
            QMessageBox.warning(self, "Timeout", f"Testing SLEAP in '{current_env}' timed out.")
            
        except Exception as e:
            self.lbl_env_status.setText(f"❌ Error testing '{current_env}'")
            self.lbl_env_status.setStyleSheet("color: #ff6b6b;")
            self.log(f"❌ Error testing SLEAP: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error testing SLEAP: {str(e)}")
    
    def select_topdown_models(self):
        """Select top-down models (centroid + centered instance)"""
        self.is_top_down = True
        self.model_paths.clear()
        self.model_list.clear()
        
        # Select centroid model
        centroid_folder = QFileDialog.getExistingDirectory(
            self,
            "Select CENTROID model folder"
        )
        if not centroid_folder:
            return
        
        # Select centered instance model
        centered_folder = QFileDialog.getExistingDirectory(
            self,
            "Select CENTERED INSTANCE model folder"
        )
        if not centered_folder:
            return
        
        self.model_paths = [centroid_folder, centered_folder]
        self.model_list.addItem(f"Centroid: {centroid_folder}")
        self.model_list.addItem(f"Centered: {centered_folder}")
        
        self.log(f"✅ Selected TOP-DOWN models:")
        self.log(f"   Centroid: {centroid_folder}")
        self.log(f"   Centered Instance: {centered_folder}")
        
        self.update_run_button()
    
    def select_bottomup_models(self):
        """Select bottom-up model"""
        self.is_top_down = False
        self.model_paths.clear()
        self.model_list.clear()
        
        # Select bottom-up model
        model_folder = QFileDialog.getExistingDirectory(
            self,
            "Select BOTTOM-UP model folder"
        )
        if not model_folder:
            return
        
        self.model_paths = [model_folder]
        self.model_list.addItem(f"Bottom-Up: {model_folder}")
        
        self.log(f"✅ Selected BOTTOM-UP model: {model_folder}")
        
        self.update_run_button()
    
    def update_run_button(self):
        """Enable/disable run button based on configuration"""
        has_videos = len(self.video_folders) > 0
        has_models = len(self.model_paths) > 0
        has_environment = self.combo_environment.currentData() is not None
        
        can_run = has_videos and has_models and has_environment
        self.btn_run.setEnabled(can_run)
    
    def run_inference(self):
        """Start the inference process"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Already Running",
                "Inference is already in progress. Please wait for it to complete."
            )
            return
        
        # Confirm with user
        video_count = 0
        for folder in self.video_folders:
            video_files = [f for f in os.listdir(folder)
                          if f.lower().endswith((".mp4", ".avi", ".mov"))
                          and not f.endswith("_roiTracked.mp4")]  # Ignore ROI tracked videos
            video_count += len(video_files)
        
        model_type = "Top-Down (Centroid + Centered)" if self.is_top_down else "Bottom-Up"
        
        msg = f"Ready to run inference:\n\n"
        msg += f"Video folders: {len(self.video_folders)}\n"
        msg += f"Total videos: {video_count}\n"
        msg += f"Model type: {model_type}\n"
        msg += f"Overwrite existing: {'Yes' if self.chk_overwrite.isChecked() else 'No'}\n"
        msg += f"Create CSV files: {'Yes' if self.chk_create_csv.isChecked() else 'No'}\n"
        msg += f"Tracking enabled: {'Yes' if self.chk_tracking.isChecked() else 'No'}\n"
        if self.chk_tracking.isChecked():
            msg += f"  - Tracker: {self.combo_tracker.currentText()}\n"
            msg += f"  - Similarity: {self.combo_similarity.currentText()}\n"
            msg += f"  - Match: {self.combo_match.currentText()}\n"
            msg += f"  - Max Tracks: {self.spin_max_tracks.value() if self.spin_max_tracks.value() > 0 else 'No limit'}\n"
        msg += "\nContinue?"
        
        reply = QMessageBox.question(
            self,
            "Confirm Inference",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Disable run button, enable stop button during processing
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log("\n" + "="*60)
        self.log("🚀 Starting SLEAP inference...")
        self.log(f"Model type: {model_type}")
        self.log(f"Video folders: {len(self.video_folders)}")
        self.log(f"Total videos: {video_count}")
        self.log("="*60 + "\n")
        
        # Start worker thread
        self.worker = InferenceWorker(
            self.video_folders,
            self.model_paths,
            self.chk_overwrite.isChecked(),
            self.chk_create_csv.isChecked(),
            self.chk_create_video.isChecked(),
            self.chk_tracking.isChecked(),
            self.combo_tracker.currentText(),
            self.combo_similarity.currentText(),
            self.combo_match.currentText(),
            self.spin_max_tracks.value() if hasattr(self, 'spin_max_tracks') else 0,
            self.spin_track_window.value() if hasattr(self, 'spin_track_window') else 5,
            self.spin_robust.value() if hasattr(self, 'spin_robust') else 1.0,
            self.combo_environment.currentData()
        )
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.start()
    
    def stop_inference(self):
        """Stop the running inference process"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Stop Processing",
                "Are you sure you want to stop the inference process?\n\nThe current video will finish processing.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.request_stop()
                self.btn_stop.setEnabled(False)
    
    def on_inference_finished(self, success, message):
        """Handle inference completion"""
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        if success:
            QMessageBox.information(
                self,
                "Inference Complete",
                message
            )
        else:
            QMessageBox.critical(
                self,
                "Inference Failed",
                message
            )


def main():
    """Main entry point for standalone execution"""
    import sys
    app = QApplication(sys.argv)
    window = VideoInferenceWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
