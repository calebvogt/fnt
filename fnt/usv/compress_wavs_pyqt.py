import os
import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTextEdit, QFileDialog, 
                             QMessageBox, QGroupBox, QLineEdit, QProgressBar, QFrame,
                             QScrollArea)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

class CompressWorker(QThread):
    """Worker thread for compressing WAV files"""
    progress = pyqtSignal(str)
    file_progress = pyqtSignal(int, int)  # current file, total files
    finished = pyqtSignal(bool, str)
    
    def __init__(self, folder_path, output_folder):
        super().__init__()
        self.folder_path = folder_path
        self.output_folder = output_folder
        self._stop_requested = False
    
    def stop(self):
        """Request the worker to stop processing"""
        self._stop_requested = True
        
    def run(self):
        try:
            # Create output folder
            os.makedirs(self.output_folder, exist_ok=True)
            self.progress.emit(f"Output folder: {self.output_folder}\n")
            
            # List all .wav files
            wav_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(".wav")]
            
            if not wav_files:
                self.finished.emit(False, "No .wav files found in the selected folder.")
                return
            
            self.progress.emit(f"Found {len(wav_files)} WAV file(s)\n")
            
            # Process each file
            for idx, wav_file in enumerate(wav_files, 1):
                # Check if stop was requested
                if self._stop_requested:
                    self.progress.emit("\n" + "="*50)
                    self.progress.emit("⚠️ Processing stopped by user")
                    self.finished.emit(False, "Processing stopped by user")
                    return
                
                input_path = os.path.join(self.folder_path, wav_file)
                output_path = os.path.join(self.output_folder, os.path.splitext(wav_file)[0] + ".wav")
                
                # Update progress
                self.file_progress.emit(idx, len(wav_files))
                self.progress.emit(f"\n[{idx}/{len(wav_files)}] Processing: {wav_file}")
                
                # ffmpeg command: downsample to 250kHz, mono, PCM 16-bit
                # Using PCM instead of ADPCM for better compatibility with Audacity
                cmd = [
                    "ffmpeg",
                    "-i", input_path,
                    "-ar", "250000",  # 250kHz sample rate
                    "-ac", "1",       # mono
                    "-c:a", "pcm_s16le",  # PCM 16-bit little-endian (standard WAV)
                    "-y",  # overwrite output
                    output_path
                ]
                
                try:
                    # Run ffmpeg and capture output
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
                    
                    # Show stderr (ffmpeg progress info)
                    if result.stderr:
                        # Only show last few lines to avoid clutter
                        stderr_lines = result.stderr.strip().split('\n')
                        relevant_lines = [line for line in stderr_lines if 'time=' in line or 'size=' in line]
                        if relevant_lines:
                            self.progress.emit(relevant_lines[-1])
                    
                    self.progress.emit(f"✓ Completed: {wav_file}")
                    
                except subprocess.CalledProcessError as e:
                    self.progress.emit(f"✗ Error processing {wav_file}: {e}")
                    self.progress.emit(f"stderr: {e.stderr}")
            
            self.progress.emit("\n" + "="*50)
            self.progress.emit("✅ All files compressed successfully!")
            self.finished.emit(True, "Compression completed successfully!")
            
        except Exception as e:
            self.finished.emit(False, f"Error during compression: {str(e)}")


class CompressWavsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.folder_path = None
        self.worker = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Compress WAV Files - FieldNeuroToolbox")
        self.setGeometry(200, 200, 900, 700)
        self.setMinimumSize(700, 600)
        
        # Apply dark mode styling matching video preprocessing
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
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        header_frame.setStyleSheet("background-color: #1e1e1e; padding: 15px; border: 1px solid #3f3f3f;")
        
        header_layout = QVBoxLayout()
        header_frame.setLayout(header_layout)
        
        title = QLabel("Compress WAV Files")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("Downsample WAV files to 250kHz mono PCM format")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet("color: #999999; font-style: italic; background-color: transparent;")
        header_layout.addWidget(subtitle)
        
        layout.addWidget(header_frame)
        
        # Folder selection group
        folder_group = QGroupBox("Input Folder")
        folder_layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Select a folder containing WAV files to compress")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #999999; margin-bottom: 10px;")
        folder_layout.addWidget(instructions)
        
        # Selected folder display
        self.lbl_folder = QLabel("No folder selected")
        self.lbl_folder.setStyleSheet("border: 1px solid #3f3f3f; padding: 10px; background-color: #1e1e1e; min-height: 40px; color: #cccccc;")
        self.lbl_folder.setWordWrap(True)
        folder_layout.addWidget(self.lbl_folder)
        
        # Folder selection button
        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("Select Folder")
        self.btn_select.clicked.connect(self.select_folder)
        btn_layout.addWidget(self.btn_select)
        btn_layout.addStretch()
        folder_layout.addLayout(btn_layout)
        
        # Output folder name input
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output subfolder:"))
        self.txt_output = QLineEdit("proc")
        self.txt_output.setMaximumWidth(200)
        output_layout.addWidget(self.txt_output)
        output_layout.addStretch()
        folder_layout.addLayout(output_layout)
        
        info_label = QLabel("Compressed files saved to output subfolder in input directory")
        info_label.setStyleSheet("color: #999999; font-style: italic; margin-top: 10px;")
        info_label.setWordWrap(True)
        folder_layout.addWidget(info_label)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.btn_process = QPushButton("Start Compression")
        self.btn_process.clicked.connect(self.start_compression)
        self.btn_process.setEnabled(False)
        btn_layout.addWidget(self.btn_process)
        
        self.btn_stop = QPushButton("Stop Processing")
        self.btn_stop.clicked.connect(self.stop_compression)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #c42b1c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #a52314;
            }
            QPushButton:pressed {
                background-color: #8b1d10;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
        """)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        
        # File progress
        self.file_progress_label = QLabel("Ready to start...")
        progress_layout.addWidget(self.file_progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        # Status log
        status_label = QLabel("Status Log:")
        status_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #cccccc;")
        progress_layout.addWidget(status_label)
        
        self.txt_output_display = QTextEdit()
        self.txt_output_display.setReadOnly(True)
        self.txt_output_display.setFont(QFont("Courier New", 9))
        self.txt_output_display.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3f3f3f; color: #cccccc;")
        progress_layout.addWidget(self.txt_output_display)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
    def select_folder(self):
        """Select folder containing WAV files"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing WAV Files",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if folder:
            self.folder_path = folder
            self.lbl_folder.setText(f"• {folder}")
            self.lbl_folder.setStyleSheet("border: 1px solid #3f3f3f; padding: 10px; background-color: #1e1e1e; min-height: 40px; color: #cccccc;")
            self.btn_process.setEnabled(True)
            self.txt_output_display.clear()
            self.txt_output_display.append(f"Selected folder: {folder}\n")
            
            # Count WAV files
            wav_files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]
            self.txt_output_display.append(f"Found {len(wav_files)} WAV file(s)\n")
            
    def start_compression(self):
        """Start the compression process"""
        if not self.folder_path:
            QMessageBox.warning(self, "No Folder", "Please select a folder first.")
            return
        
        # Get output folder name
        output_name = self.txt_output.text().strip() or "proc"
        output_folder = os.path.join(self.folder_path, output_name)
        
        # Disable buttons during processing
        self.btn_select.setEnabled(False)
        self.btn_process.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Clear log
        self.txt_output_display.clear()
        self.txt_output_display.append("Starting compression...\n")
        
        # Create and start worker thread
        self.worker = CompressWorker(self.folder_path, output_folder)
        self.worker.progress.connect(self.update_progress)
        self.worker.file_progress.connect(self.update_file_progress)
        self.worker.finished.connect(self.compression_finished)
        self.worker.start()
    
    def update_file_progress(self, current, total):
        """Update file progress bar"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.file_progress_label.setText(f"Processing file {current} of {total}")
        
    def update_progress(self, message):
        """Update the progress text area"""
        self.txt_output_display.append(message)
        # Auto-scroll to bottom
        self.txt_output_display.verticalScrollBar().setValue(
            self.txt_output_display.verticalScrollBar().maximum()
        )
        
    def stop_compression(self):
        """Stop the compression process"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.btn_stop.setEnabled(False)
            self.txt_output_display.append("\nStopping processing...")
    
    def compression_finished(self, success, message):
        """Handle compression completion"""
        # Re-enable buttons
        self.btn_select.setEnabled(True)
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        self.file_progress_label.setText("Ready")
        
        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)


def main():
    app = QApplication(sys.argv)
    window = CompressWavsWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
