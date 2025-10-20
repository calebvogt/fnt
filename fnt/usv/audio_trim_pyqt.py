import os
import sys
import numpy as np
import soundfile as sf
from scipy import signal
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QMessageBox, 
                             QGroupBox, QLineEdit, QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class AudioTrimWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.audio_file = None
        self.audio_data = None
        self.sample_rate = None
        self.start_time = 0.0
        self.end_time = 0.0
        self.min_freq = 30000  # 30 kHz default
        self.max_freq = 100000  # 100 kHz default
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Trim Audio File - USV Processing")
        self.setGeometry(100, 100, 1200, 800)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Trim Audio File with Frequency Filter")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Load an audio file, visualize spectrogram, select time range, and filter frequency range")
        desc.setFont(QFont("Arial", 10))
        desc.setStyleSheet("color: #666666; margin-bottom: 10px;")
        layout.addWidget(desc)
        
        # File selection group
        file_group = QGroupBox("Input File")
        file_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Audio File")
        self.btn_load.clicked.connect(self.load_audio)
        btn_layout.addWidget(self.btn_load)
        btn_layout.addStretch()
        file_layout.addLayout(btn_layout)
        
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setStyleSheet("color: #666666; font-style: italic;")
        file_layout.addWidget(self.lbl_file)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Spectrogram display
        spec_group = QGroupBox("Spectrogram Visualization")
        spec_layout = QVBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        spec_layout.addWidget(self.toolbar)
        spec_layout.addWidget(self.canvas)
        
        # Info label
        self.lbl_info = QLabel("Use the pan/zoom tools to navigate the spectrogram")
        self.lbl_info.setStyleSheet("color: #666666; font-style: italic; margin-top: 5px;")
        spec_layout.addWidget(self.lbl_info)
        
        spec_group.setLayout(spec_layout)
        layout.addWidget(spec_group)
        
        # Trim parameters group
        trim_group = QGroupBox("Trim Parameters")
        trim_layout = QVBoxLayout()
        
        # Time range
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Start Time (s):"))
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setDecimals(3)
        self.spin_start.setMinimum(0.0)
        self.spin_start.setMaximum(9999.0)
        self.spin_start.setValue(0.0)
        self.spin_start.valueChanged.connect(self.update_selection)
        time_layout.addWidget(self.spin_start)
        
        time_layout.addWidget(QLabel("End Time (s):"))
        self.spin_end = QDoubleSpinBox()
        self.spin_end.setDecimals(3)
        self.spin_end.setMinimum(0.0)
        self.spin_end.setMaximum(9999.0)
        self.spin_end.setValue(0.0)
        self.spin_end.valueChanged.connect(self.update_selection)
        time_layout.addWidget(self.spin_end)
        time_layout.addStretch()
        trim_layout.addLayout(time_layout)
        
        # Frequency range
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Min Frequency (Hz):"))
        self.spin_min_freq = QSpinBox()
        self.spin_min_freq.setMinimum(0)
        self.spin_min_freq.setMaximum(500000)
        self.spin_min_freq.setSingleStep(1000)
        self.spin_min_freq.setValue(30000)
        freq_layout.addWidget(self.spin_min_freq)
        
        freq_layout.addWidget(QLabel("Max Frequency (Hz):"))
        self.spin_max_freq = QSpinBox()
        self.spin_max_freq.setMinimum(0)
        self.spin_max_freq.setMaximum(500000)
        self.spin_max_freq.setSingleStep(1000)
        self.spin_max_freq.setValue(100000)
        freq_layout.addWidget(self.spin_max_freq)
        freq_layout.addStretch()
        trim_layout.addLayout(freq_layout)
        
        # Duration display
        self.lbl_duration = QLabel("Selected duration: 0.000 s")
        self.lbl_duration.setStyleSheet("font-weight: bold; margin-top: 5px;")
        trim_layout.addWidget(self.lbl_duration)
        
        trim_group.setLayout(trim_layout)
        layout.addWidget(trim_group)
        
        # Output group
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        
        output_name_layout = QHBoxLayout()
        output_name_layout.addWidget(QLabel("Output filename:"))
        self.txt_output = QLineEdit("trimmed_audio.wav")
        output_name_layout.addWidget(self.txt_output)
        output_layout.addLayout(output_name_layout)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Process button
        btn_layout = QHBoxLayout()
        self.btn_process = QPushButton("Trim and Save Audio")
        self.btn_process.clicked.connect(self.trim_audio)
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet("padding: 10px; font-size: 12px; font-weight: bold;")
        btn_layout.addWidget(self.btn_process)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.setLayout(layout)
        
    def load_audio(self):
        """Load an audio file and display spectrogram"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.flac *.mp3);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # Load audio file
            self.audio_file = file_path
            self.audio_data, self.sample_rate = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(self.audio_data.shape) > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)
            
            # Update UI
            self.lbl_file.setText(f"{os.path.basename(file_path)}")
            self.lbl_file.setStyleSheet("color: black;")
            
            # Update time spinbox maximum
            duration = len(self.audio_data) / self.sample_rate
            self.spin_start.setMaximum(duration)
            self.spin_end.setMaximum(duration)
            self.spin_end.setValue(duration)
            
            # Update info
            self.lbl_info.setText(
                f"Sample rate: {self.sample_rate} Hz | Duration: {duration:.3f} s | "
                f"Samples: {len(self.audio_data)}"
            )
            
            # Generate and display spectrogram
            self.display_spectrogram()
            
            # Enable process button
            self.btn_process.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio file:\n{str(e)}")
            
    def display_spectrogram(self):
        """Generate and display the spectrogram"""
        if self.audio_data is None:
            return
        
        try:
            # Clear previous plot
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Compute spectrogram
            # Use shorter window for better time resolution with USVs
            nperseg = min(1024, len(self.audio_data) // 4)
            frequencies, times, Sxx = signal.spectrogram(
                self.audio_data,
                self.sample_rate,
                nperseg=nperseg,
                noverlap=nperseg // 2
            )
            
            # Convert to dB scale
            Sxx_dB = 10 * np.log10(Sxx + 1e-10)
            
            # Plot spectrogram
            im = ax.pcolormesh(times, frequencies / 1000, Sxx_dB, 
                              shading='gouraud', cmap='viridis')
            
            ax.set_ylabel('Frequency (kHz)', fontsize=10)
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_title('Audio Spectrogram', fontsize=12, fontweight='bold')
            
            # Add colorbar
            cbar = self.figure.colorbar(im, ax=ax, label='Power (dB)')
            
            # Set y-axis limit to show high frequencies better
            # If sample rate supports it, show up to 125 kHz
            max_freq_khz = min(self.sample_rate / 2000, 125)
            ax.set_ylim([0, max_freq_khz])
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate spectrogram:\n{str(e)}")
            
    def update_selection(self):
        """Update the duration display when time selection changes"""
        start = self.spin_start.value()
        end = self.spin_end.value()
        duration = max(0, end - start)
        self.lbl_duration.setText(f"Selected duration: {duration:.3f} s")
        
    def trim_audio(self):
        """Trim the audio file with time and frequency filtering"""
        if self.audio_data is None:
            QMessageBox.warning(self, "No Audio", "Please load an audio file first.")
            return
        
        try:
            # Get parameters
            start_time = self.spin_start.value()
            end_time = self.spin_end.value()
            min_freq = self.spin_min_freq.value()
            max_freq = self.spin_max_freq.value()
            
            # Validate
            if start_time >= end_time:
                QMessageBox.warning(self, "Invalid Range", "Start time must be less than end time.")
                return
            
            if min_freq >= max_freq:
                QMessageBox.warning(self, "Invalid Range", "Min frequency must be less than max frequency.")
                return
            
            # Convert time to samples
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Trim time range
            trimmed_audio = self.audio_data[start_sample:end_sample]
            
            # Apply frequency filtering using bandpass filter
            nyquist = self.sample_rate / 2
            
            # Only filter if frequencies are within valid range
            if min_freq < nyquist and max_freq <= nyquist:
                # Design bandpass filter
                sos = signal.butter(4, [min_freq, max_freq], btype='bandpass', 
                                   fs=self.sample_rate, output='sos')
                trimmed_audio = signal.sosfilt(sos, trimmed_audio)
            else:
                QMessageBox.warning(
                    self,
                    "Frequency Warning",
                    f"Frequency range exceeds Nyquist frequency ({nyquist:.0f} Hz).\n"
                    "Skipping frequency filter."
                )
            
            # Get output path
            output_name = self.txt_output.text().strip()
            if not output_name:
                output_name = "trimmed_audio.wav"
            
            # Save to same directory as input
            input_dir = os.path.dirname(self.audio_file)
            output_path = os.path.join(input_dir, output_name)
            
            # Save audio file
            sf.write(output_path, trimmed_audio, self.sample_rate)
            
            QMessageBox.information(
                self,
                "Success",
                f"Audio trimmed and saved successfully!\n\n"
                f"Output: {output_path}\n"
                f"Duration: {len(trimmed_audio) / self.sample_rate:.3f} s\n"
                f"Sample rate: {self.sample_rate} Hz"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to trim audio:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    window = AudioTrimWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
