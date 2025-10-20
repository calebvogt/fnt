import os
import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTextEdit, QFileDialog, 
                             QMessageBox, QGroupBox, QLineEdit)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont

class CompressWorker(QThread):
    """Worker thread for compressing WAV files"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, folder_path, output_folder):
        super().__init__()
        self.folder_path = folder_path
        self.output_folder = output_folder
        
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
                input_path = os.path.join(self.folder_path, wav_file)
                output_path = os.path.join(self.output_folder, os.path.splitext(wav_file)[0] + ".wav")
                
                self.progress.emit(f"\n[{idx}/{len(wav_files)}] Processing: {wav_file}")
                
                # ffmpeg command: downsample to 250kHz, mono, ADPCM compression
                cmd = [
                    "ffmpeg",
                    "-i", input_path,
                    "-ar", "250000",  # 250kHz sample rate
                    "-ac", "1",       # mono
                    "-c:a", "adpcm_ima_wav",  # ADPCM compression
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


class CompressWavsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.folder_path = None
        self.worker = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Compress WAV Files")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Compress WAV Files")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Compress WAV files using ADPCM encoding (250kHz, mono)\nOutputs to 'proc' subfolder")
        desc.setFont(QFont("Arial", 10))
        desc.setStyleSheet("color: #666666; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # Folder selection group
        folder_group = QGroupBox("Input Folder")
        folder_layout = QVBoxLayout()
        
        # Folder selection button
        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("Select Folder")
        self.btn_select.clicked.connect(self.select_folder)
        btn_layout.addWidget(self.btn_select)
        btn_layout.addStretch()
        folder_layout.addLayout(btn_layout)
        
        # Selected folder display
        self.lbl_folder = QLabel("No folder selected")
        self.lbl_folder.setStyleSheet("color: #666666; font-style: italic;")
        folder_layout.addWidget(self.lbl_folder)
        
        # Output folder name input
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output subfolder:"))
        self.txt_output = QLineEdit("proc")
        self.txt_output.setMaximumWidth(200)
        output_layout.addWidget(self.txt_output)
        output_layout.addStretch()
        folder_layout.addLayout(output_layout)
        
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)
        
        # Process button
        btn_layout = QHBoxLayout()
        self.btn_process = QPushButton("Compress Files")
        self.btn_process.clicked.connect(self.start_compression)
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet("padding: 8px; font-size: 12px;")
        btn_layout.addWidget(self.btn_process)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Output text area
        self.txt_output_display = QTextEdit()
        self.txt_output_display.setReadOnly(True)
        self.txt_output_display.setFont(QFont("Courier", 9))
        layout.addWidget(self.txt_output_display)
        
        self.setLayout(layout)
        
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
            self.lbl_folder.setText(folder)
            self.lbl_folder.setStyleSheet("color: black;")
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
        self.txt_output_display.clear()
        self.txt_output_display.append("Starting compression...\n")
        
        # Create and start worker thread
        self.worker = CompressWorker(self.folder_path, output_folder)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.compression_finished)
        self.worker.start()
        
    def update_progress(self, message):
        """Update the progress text area"""
        self.txt_output_display.append(message)
        # Auto-scroll to bottom
        self.txt_output_display.verticalScrollBar().setValue(
            self.txt_output_display.verticalScrollBar().maximum()
        )
        
    def compression_finished(self, success, message):
        """Handle compression completion"""
        # Re-enable buttons
        self.btn_select.setEnabled(True)
        self.btn_process.setEnabled(True)
        
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
