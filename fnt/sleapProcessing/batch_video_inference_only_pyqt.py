"""
SLEAP Video Inference Only - PyQt5 Implementation

Run SLEAP inference without tracking on video files.
Supports both top-down (centroid + centered instance) and bottom-up models.
Automatically converts output to CSV format.
"""

import os
import subprocess
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QMessageBox, QGroupBox, QTextEdit, QCheckBox,
    QListWidget, QGridLayout, QFrame, QApplication
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont


class InferenceWorker(QThread):
    """Worker thread for running SLEAP inference"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, video_folders, model_paths, overwrite_existing):
        super().__init__()
        self.video_folders = video_folders
        self.model_paths = model_paths
        self.overwrite_existing = overwrite_existing
        
    def run(self):
        try:
            total_processed = 0
            total_skipped = 0
            
            for folder in self.video_folders:
                video_files = [f for f in os.listdir(folder)
                              if f.lower().endswith((".mp4", ".avi", ".mov"))]
                
                if not video_files:
                    self.progress.emit(f"⚠️ No video files found in: {folder}\n")
                    continue
                
                self.progress.emit(f"\n📁 Processing folder: {folder}")
                self.progress.emit(f"Found {len(video_files)} video file(s)\n")
                
                for video_file in video_files:
                    full_path = os.path.join(folder, video_file)
                    slp_path = self.get_output_path(full_path)
                    csv_path = slp_path.replace(".predictions.slp", ".predictions.analysis.csv")
                    
                    if not self.overwrite_existing and (os.path.exists(slp_path) or os.path.exists(csv_path)):
                        self.progress.emit(f"⏭️ Skipping {video_file} (existing predictions detected)")
                        total_skipped += 1
                        continue
                    
                    # Run inference
                    success = self.run_inference_on_video(full_path)
                    if success:
                        total_processed += 1
                        # Convert to CSV
                        self.convert_to_csv(slp_path)
                    else:
                        self.progress.emit(f"❌ Failed to process {video_file}")
            
            summary = f"\n{'='*60}\n"
            summary += f"✅ Inference complete!\n"
            summary += f"Videos processed: {total_processed}\n"
            summary += f"Videos skipped: {total_skipped}\n"
            summary += f"Total videos: {total_processed + total_skipped}\n"
            
            self.progress.emit(summary)
            self.finished.emit(True, "Inference completed successfully!")
            
        except Exception as e:
            self.finished.emit(False, f"Error during inference: {str(e)}")
    
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
        
        for model_path in self.model_paths:
            cmd += ["-m", os.path.join(model_path, "training_config.json")]
        
        output_file = self.get_output_path(video_file)
        cmd += [
            "--only-suggested-frames",
            "--no-empty-frames",
            "--verbosity", "json",
            "--video.input_format", "channels_last",
            "--gpu", "auto",
            "--batch_size", "4",
            "--peak_threshold", "0.2",
            "--tracking.tracker", "none",
            "--controller_port", "9000",
            "--publish_port", "9001",
            "-o", output_file
        ]
        
        self.progress.emit(f"\n🔁 Running inference on: {os.path.basename(video_file)}")
        self.progress.emit(f"Command: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Show output
            if result.stdout:
                self.progress.emit(result.stdout)
            if result.stderr:
                # Filter out verbose JSON output, show only errors
                stderr_lines = result.stderr.split('\n')
                for line in stderr_lines:
                    if line and not line.strip().startswith('{'):
                        self.progress.emit(line)
            
            if result.returncode == 0:
                self.progress.emit(f"✅ Inference completed: {os.path.basename(output_file)}")
                return True
            else:
                self.progress.emit(f"❌ Inference failed with return code {result.returncode}")
                return False
                
        except Exception as e:
            self.progress.emit(f"❌ Error running inference: {str(e)}")
            return False
    
    def convert_to_csv(self, slp_file):
        """Convert SLP file to CSV format"""
        csv_file = slp_file.replace(".predictions.slp", ".predictions.analysis.csv")
        cmd = ["sleap-convert", "--format", "analysis.csv", "-o", csv_file, slp_file]
        
        self.progress.emit(f"📄 Converting to CSV: {os.path.basename(csv_file)}")
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0:
                self.progress.emit(f"✅ CSV conversion completed")
            else:
                self.progress.emit(f"⚠️ CSV conversion warning: {result.stderr}")
                
        except Exception as e:
            self.progress.emit(f"⚠️ Error converting to CSV: {str(e)}")


class VideoInferenceWindow(QWidget):
    """PyQt5 window for SLEAP video inference configuration and execution"""
    
    def __init__(self):
        super().__init__()
        self.video_folders = []
        self.model_paths = []
        self.is_top_down = False
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("SLEAP Video Inference Only")
        self.setGeometry(100, 100, 900, 700)
        self.setMinimumSize(800, 600)
        
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
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("SLEAP Video Inference Only")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Run SLEAP inference without tracking\nAutomatically converts output to CSV format")
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
        
        # Options Group
        options_group = QGroupBox("3. Processing Options")
        options_layout = QVBoxLayout()
        
        self.chk_overwrite = QCheckBox("Overwrite existing prediction files")
        self.chk_overwrite.setChecked(True)
        options_layout.addWidget(self.chk_overwrite)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Run Button
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Inference")
        self.btn_run.clicked.connect(self.run_inference)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("padding: 10px; font-size: 13px;")
        btn_layout.addWidget(self.btn_run)
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
        self.log("3. Click 'Run Inference' to start\n")
    
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
        can_run = len(self.video_folders) > 0 and len(self.model_paths) > 0
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
                          if f.lower().endswith((".mp4", ".avi", ".mov"))]
            video_count += len(video_files)
        
        model_type = "Top-Down (Centroid + Centered)" if self.is_top_down else "Bottom-Up"
        
        msg = f"Ready to run inference:\n\n"
        msg += f"Video folders: {len(self.video_folders)}\n"
        msg += f"Total videos: {video_count}\n"
        msg += f"Model type: {model_type}\n"
        msg += f"Overwrite existing: {'Yes' if self.chk_overwrite.isChecked() else 'No'}\n\n"
        msg += "Continue?"
        
        reply = QMessageBox.question(
            self,
            "Confirm Inference",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Disable controls during processing
        self.btn_run.setEnabled(False)
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
            self.chk_overwrite.isChecked()
        )
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.start()
    
    def on_inference_finished(self, success, message):
        """Handle inference completion"""
        self.btn_run.setEnabled(True)
        
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
