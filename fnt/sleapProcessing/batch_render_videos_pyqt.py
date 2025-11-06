#!/usr/bin/env python3
"""
SLEAP Batch Video Rendering Tool
Renders tracked videos from existing .slp prediction files without re-running inference.
"""

import os
import sys
import re
import subprocess
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QFileDialog, QGroupBox, QCheckBox, QComboBox,
    QMessageBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont


class RenderWorker(QThread):
    """Worker thread for rendering SLEAP videos"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, slp_files, overwrite_existing, conda_env=None):
        super().__init__()
        self.slp_files = slp_files
        self.overwrite_existing = overwrite_existing
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
            
            for slp_file in self.slp_files:
                if self._stop_requested:
                    break
                video_output = slp_file.replace(".predictions.slp", ".predictions.mp4")
                
                if not self.overwrite_existing and os.path.exists(video_output):
                    self.progress.emit(f"⏭️ Skipping: {os.path.basename(video_output)} (already exists)")
                    total_skipped += 1
                    continue
                
                # Render video
                success = self.render_video(slp_file, video_output)
                if success:
                    total_processed += 1
                else:
                    self.progress.emit(f"❌ Failed to render: {os.path.basename(slp_file)}")
            
            summary = f"\n{'='*60}\n"
            if self._stop_requested:
                summary += f"⚠️ Video rendering stopped by user!\n"
            else:
                summary += f"✅ Video rendering complete!\n"
            summary += f"Videos rendered: {total_processed}\n"
            summary += f"Videos skipped: {total_skipped}\n"
            summary += f"Total videos: {total_processed + total_skipped}\n"
            
            self.progress.emit(summary)
            
            if self._stop_requested:
                self.finished.emit(False, "Video rendering stopped by user")
            else:
                self.finished.emit(True, "Video rendering completed successfully!")
            
        except Exception as e:
            self.finished.emit(False, f"Error during video rendering: {str(e)}")
    
    def render_video(self, slp_file, video_output):
        """Render tracked video with predictions overlay using sleap-render"""
        self.progress.emit(f"🎬 Rendering: {os.path.basename(slp_file)}")
        
        try:
            # Find the original video file by removing the timestamp and .predictions.slp
            # Example: F9039_PreLDB.mp4.251104_202452.predictions.slp -> F9039_PreLDB.mp4
            video_file = self.find_original_video(slp_file)
            
            if not video_file:
                self.progress.emit(f"⚠️ Could not find original video for {os.path.basename(slp_file)}")
                self.progress.emit(f"   Rendering without frame range specification...")
            
            # Build sleap-render command
            cmd = ["sleap-render", slp_file, "-o", video_output]
            
            # Get frame count and add frame range to render all frames
            if video_file:
                frame_count = self.get_video_frame_count(video_file)
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
                return True
            else:
                self.progress.emit(f"⚠️ Rendering failed (return code {result.returncode})")
                if result.stderr:
                    self.progress.emit(f"   stderr: {result.stderr[:500]}")
                if result.stdout:
                    self.progress.emit(f"   stdout: {result.stdout[:500]}")
                return False
                
        except Exception as e:
            self.progress.emit(f"⚠️ Rendering error: {str(e)}")
            return False
    
    def find_original_video(self, slp_file):
        """Find the original video file from the .slp filename"""
        # Remove .predictions.slp and timestamp to get original video name
        # Example: F9039_PreLDB.mp4.251104_202452.predictions.slp -> F9039_PreLDB.mp4
        import re
        basename = os.path.basename(slp_file)
        # Pattern: {videoname}.{timestamp}.predictions.slp
        match = re.match(r'(.+)\.\d{6}_\d{6}\.predictions\.slp$', basename)
        if match:
            video_name = match.group(1)
            video_dir = os.path.dirname(slp_file)
            video_path = os.path.join(video_dir, video_name)
            if os.path.exists(video_path):
                return video_path
        return None
    
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


class RenderVideosWindow(QWidget):
    """PyQt5 window for batch rendering SLEAP tracked videos"""
    
    def __init__(self):
        super().__init__()
        self.slp_files = []
        self.worker = None
        self.init_ui()
        
        # Auto-detect conda environments after UI is ready
        QTimer.singleShot(500, self.detect_conda_environments)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("SLEAP Batch Video Rendering")
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
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cc1;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #666666;
            }
            QGroupBox {
                border: 2px solid #3f3f3f;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #cccccc;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QListWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #cccccc;
                margin-right: 5px;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("SLEAP Batch Video Rendering")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #0078d4; margin: 10px;")
        main_layout.addWidget(title)
        
        # Description
        desc = QLabel("Render tracked videos from existing .slp prediction files")
        desc.setStyleSheet("color: #999999; margin-bottom: 10px;")
        main_layout.addWidget(desc)
        
        # File Selection Section
        file_group = QGroupBox("1. Select .slp Files")
        file_layout = QVBoxLayout()
        
        # File selection buttons
        btn_layout = QHBoxLayout()
        self.btn_select_folder = QPushButton("📁 Select Folder(s)")
        self.btn_select_folder.clicked.connect(self.select_folders)
        self.btn_select_files = QPushButton("📄 Select Individual Files")
        self.btn_select_files.clicked.connect(self.select_files)
        self.btn_clear = QPushButton("🗑️ Clear Selection")
        self.btn_clear.clicked.connect(self.clear_selection)
        
        btn_layout.addWidget(self.btn_select_folder)
        btn_layout.addWidget(self.btn_select_files)
        btn_layout.addWidget(self.btn_clear)
        file_layout.addLayout(btn_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        file_layout.addWidget(QLabel("Selected .slp files:"))
        file_layout.addWidget(self.file_list)
        
        file_group.setLayout(file_layout)
        main_layout.addWidget(file_group)
        
        # Options Section
        options_group = QGroupBox("2. Rendering Options")
        options_layout = QVBoxLayout()
        
        # Overwrite option
        self.chk_overwrite = QCheckBox("Overwrite existing .predictions.mp4 files")
        self.chk_overwrite.setChecked(False)
        options_layout.addWidget(self.chk_overwrite)
        
        # Conda environment selection
        env_layout = QHBoxLayout()
        env_layout.addWidget(QLabel("Conda Environment:"))
        self.combo_environment = QComboBox()
        self.combo_environment.setMinimumWidth(200)
        env_layout.addWidget(self.combo_environment)
        env_layout.addStretch()
        options_layout.addLayout(env_layout)
        
        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)
        
        # Run and Stop Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_run = QPushButton("▶️ Start Rendering")
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #16825d;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1a9667;
            }
            QPushButton:pressed {
                background-color: #127352;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
            }
        """)
        self.btn_run.clicked.connect(self.start_rendering)
        btn_layout.addWidget(self.btn_run)
        
        self.btn_stop = QPushButton("⏹️ Stop Rendering")
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #c42b1c;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d83b2c;
            }
            QPushButton:pressed {
                background-color: #a82010;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
            }
        """)
        self.btn_stop.clicked.connect(self.stop_rendering)
        btn_layout.addWidget(self.btn_stop)
        
        main_layout.addLayout(btn_layout)
        
        # Log Output
        log_group = QGroupBox("Rendering Log")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(200)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        self.setLayout(main_layout)
        
    def detect_conda_environments(self):
        """Detect available conda environments"""
        self.combo_environment.clear()
        self.combo_environment.addItem("System Python (no conda)", None)
        
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                shell=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if parts:
                            env_name = parts[0]
                            if env_name != "base":
                                self.combo_environment.addItem(f"conda: {env_name}", env_name)
                
                # Try to auto-select SLEAP environment
                for i in range(self.combo_environment.count()):
                    if "sleap" in self.combo_environment.itemText(i).lower():
                        self.combo_environment.setCurrentIndex(i)
                        break
                        
        except Exception as e:
            self.log(f"⚠️ Could not detect conda environments: {str(e)}")
    
    def select_folders(self):
        """Select folder(s) containing .slp files"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with .slp Files")
        
        if folder:
            slp_files = list(Path(folder).rglob("*.predictions.slp"))
            
            if slp_files:
                for slp_file in slp_files:
                    slp_path = str(slp_file)
                    if slp_path not in self.slp_files:
                        self.slp_files.append(slp_path)
                        self.file_list.addItem(os.path.basename(slp_path))
                
                self.log(f"✅ Added {len(slp_files)} .slp file(s) from {folder}")
            else:
                QMessageBox.warning(self, "No Files Found", 
                                  f"No .predictions.slp files found in {folder}")
    
    def select_files(self):
        """Select individual .slp files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .slp Files",
            "",
            "SLEAP Prediction Files (*.predictions.slp);;All Files (*.*)"
        )
        
        if files:
            for file_path in files:
                if file_path not in self.slp_files:
                    self.slp_files.append(file_path)
                    self.file_list.addItem(os.path.basename(file_path))
            
            self.log(f"✅ Added {len(files)} .slp file(s)")
    
    def clear_selection(self):
        """Clear all selected files"""
        self.slp_files.clear()
        self.file_list.clear()
        self.log("🗑️ Cleared all file selections")
    
    def start_rendering(self):
        """Start the video rendering process"""
        # Validation
        if not self.slp_files:
            QMessageBox.warning(self, "No Files Selected", 
                              "Please select .slp files to render.")
            return
        
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Already Running", 
                              "Video rendering is already in progress.")
            return
        
        # Disable run button, enable stop button during processing
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        self.log("="*60)
        self.log("Starting SLEAP video rendering...")
        self.log(f"Files to process: {len(self.slp_files)}")
        self.log(f"Overwrite existing: {self.chk_overwrite.isChecked()}")
        self.log(f"Conda environment: {self.combo_environment.currentText()}")
        self.log("="*60 + "\n")
        
        # Start worker thread
        self.worker = RenderWorker(
            self.slp_files,
            self.chk_overwrite.isChecked(),
            self.combo_environment.currentData()
        )
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.on_rendering_finished)
        self.worker.start()
    
    def stop_rendering(self):
        """Stop the rendering process"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Stop Rendering",
                "Are you sure you want to stop the rendering process?\n\nThe current video will finish rendering.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.request_stop()
                self.btn_stop.setEnabled(False)
    
    def on_rendering_finished(self, success, message):
        """Handle rendering completion"""
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        if success:
            QMessageBox.information(self, "Complete", message)
        else:
            QMessageBox.critical(self, "Error", message)
    
    def log(self, message):
        """Append message to log output"""
        self.log_output.append(message)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )


def main():
    """Main entry point for standalone execution"""
    app = QApplication(sys.argv)
    window = RenderVideosWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
