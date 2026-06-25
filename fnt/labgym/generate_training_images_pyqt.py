import os
import sys
import subprocess
import random
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox,
    QGroupBox, QLineEdit, QProgressBar, QFrame, QSpinBox,
    QComboBox, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}


def get_video_frame_count(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass

    cmd_fallback = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        result = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip() and result.stdout.strip() != "N/A":
            return int(result.stdout.strip())
    except Exception:
        pass

    cmd_duration = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration,r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        result = subprocess.run(cmd_duration, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                fps_str = parts[0]
                duration = float(parts[1])
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    fps = float(num) / float(den)
                else:
                    fps = float(fps_str)
                return int(fps * duration)
    except Exception:
        pass

    return None


class ExtractWorker(QThread):
    progress = pyqtSignal(str)
    file_progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool, str)

    def __init__(self, video_paths, output_dir, num_frames, sampling_method):
        super().__init__()
        self.video_paths = video_paths
        self.output_dir = output_dir
        self.num_frames = num_frames
        self.sampling_method = sampling_method
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            self.progress.emit(f"Output directory: {self.output_dir}")
            self.progress.emit(f"Sampling method: {self.sampling_method}")
            self.progress.emit(f"Frames per video: {self.num_frames}\n")

            total_extracted = 0

            for idx, video_path in enumerate(self.video_paths, 1):
                if self._stop_requested:
                    self.progress.emit("\nStopped by user")
                    self.finished.emit(False, "Stopped by user")
                    return

                self.file_progress.emit(idx, len(self.video_paths))
                video_name = os.path.basename(video_path)
                stem = Path(video_path).stem
                self.progress.emit(f"[{idx}/{len(self.video_paths)}] {video_name}")

                frame_count = get_video_frame_count(video_path)
                if frame_count is None or frame_count == 0:
                    self.progress.emit(f"  Could not determine frame count, skipping")
                    continue

                n = min(self.num_frames, frame_count)
                self.progress.emit(f"  Total frames: {frame_count}, extracting {n}")

                if self.sampling_method == "stride":
                    if n >= frame_count:
                        frame_indices = list(range(frame_count))
                    else:
                        step = frame_count / n
                        frame_indices = [int(round(i * step)) for i in range(n)]
                        frame_indices = [min(f, frame_count - 1) for f in frame_indices]
                else:
                    if n >= frame_count:
                        frame_indices = list(range(frame_count))
                    else:
                        frame_indices = sorted(random.sample(range(frame_count), n))

                for i, frame_idx in enumerate(frame_indices):
                    if self._stop_requested:
                        self.progress.emit("\nStopped by user")
                        self.finished.emit(False, "Stopped by user")
                        return

                    out_filename = f"{stem}_frame{frame_idx:06d}.png"
                    out_path = os.path.join(self.output_dir, out_filename)

                    cmd = [
                        "ffmpeg", "-v", "error",
                        "-i", video_path,
                        "-vf", f"select=eq(n\\,{frame_idx})",
                        "-vframes", "1",
                        "-y",
                        out_path
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0 and os.path.exists(out_path):
                        total_extracted += 1
                    else:
                        self.progress.emit(f"  Failed to extract frame {frame_idx}")
                        if result.stderr:
                            self.progress.emit(f"    {result.stderr.strip()}")

                self.progress.emit(f"  Extracted {min(len(frame_indices), n)} frames")

            self.progress.emit(f"\nDone! Extracted {total_extracted} images total")
            self.finished.emit(True, f"Extracted {total_extracted} images to {self.output_dir}")

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")


class GenerateTrainingImagesWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_paths = []
        self.worker = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Generate Training Images - FieldNeuroethologyToolbox")
        self.setGeometry(200, 200, 900, 750)
        self.setMinimumSize(700, 600)

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
            QSpinBox {
                padding: 5px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QRadioButton {
                color: #cccccc;
                background-color: transparent;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
        """)

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

        title = QLabel("Generate Training Images")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        header_layout.addWidget(title)

        subtitle = QLabel("Extract frames from videos for LabGym training datasets")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet("color: #999999; font-style: italic; background-color: transparent;")
        header_layout.addWidget(subtitle)

        layout.addWidget(header_frame)

        # Input selection group
        input_group = QGroupBox("Input Videos")
        input_layout = QVBoxLayout()

        instructions = QLabel("Select individual video files or folders containing videos")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #999999; margin-bottom: 10px;")
        input_layout.addWidget(instructions)

        self.lbl_input = QLabel("No videos selected")
        self.lbl_input.setStyleSheet("border: 1px solid #3f3f3f; padding: 10px; background-color: #1e1e1e; min-height: 40px; color: #cccccc;")
        self.lbl_input.setWordWrap(True)
        input_layout.addWidget(self.lbl_input)

        btn_row = QHBoxLayout()
        self.btn_add_videos = QPushButton("Add Videos")
        self.btn_add_videos.clicked.connect(self.add_videos)
        btn_row.addWidget(self.btn_add_videos)

        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.clicked.connect(self.add_folder)
        btn_row.addWidget(self.btn_add_folder)

        self.btn_clear = QPushButton("Clear All")
        self.btn_clear.clicked.connect(self.clear_inputs)
        self.btn_clear.setEnabled(False)
        btn_row.addWidget(self.btn_clear)

        btn_row.addStretch()
        input_layout.addLayout(btn_row)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Settings group
        settings_group = QGroupBox("Extraction Settings")
        settings_layout = QVBoxLayout()

        # Frames per video
        frames_row = QHBoxLayout()
        frames_row.addWidget(QLabel("Frames per video:"))
        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(1, 10000)
        self.spin_frames.setValue(20)
        self.spin_frames.setMaximumWidth(120)
        frames_row.addWidget(self.spin_frames)
        frames_row.addStretch()
        settings_layout.addLayout(frames_row)

        # Sampling method
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Sampling method:"))

        self.radio_stride = QRadioButton("Stride (evenly spaced)")
        self.radio_random = QRadioButton("Random")
        self.radio_stride.setChecked(True)

        self.sampling_group = QButtonGroup()
        self.sampling_group.addButton(self.radio_stride)
        self.sampling_group.addButton(self.radio_random)

        method_row.addWidget(self.radio_stride)
        method_row.addWidget(self.radio_random)
        method_row.addStretch()
        settings_layout.addLayout(method_row)

        # Output directory
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output folder:"))
        self.txt_output_dir = QLineEdit("")
        self.txt_output_dir.setPlaceholderText("Click 'Choose...' to select output location")
        self.txt_output_dir.setReadOnly(True)
        output_row.addWidget(self.txt_output_dir)

        self.btn_choose_output = QPushButton("Choose...")
        self.btn_choose_output.clicked.connect(self.choose_output_dir)
        self.btn_choose_output.setMaximumWidth(100)
        output_row.addWidget(self.btn_choose_output)
        settings_layout.addLayout(output_row)

        output_info = QLabel("A 'training_images' subfolder will be created in the chosen directory")
        output_info.setStyleSheet("color: #999999; font-style: italic; margin-top: 5px;")
        output_info.setWordWrap(True)
        settings_layout.addWidget(output_info)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Control buttons
        ctrl_row = QHBoxLayout()
        self.btn_start = QPushButton("Generate Images")
        self.btn_start.clicked.connect(self.start_extraction)
        self.btn_start.setEnabled(False)
        ctrl_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_extraction)
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
        ctrl_row.addWidget(self.btn_stop)
        ctrl_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        ctrl_row.addWidget(close_btn)

        layout.addLayout(ctrl_row)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        self.file_progress_label = QLabel("Ready to start...")
        progress_layout.addWidget(self.file_progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        status_label = QLabel("Status Log:")
        status_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #cccccc;")
        progress_layout.addWidget(status_label)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Courier New", 9))
        self.txt_log.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3f3f3f; color: #cccccc;")
        progress_layout.addWidget(self.txt_log)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

    def add_videos(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.m4v);;All Files (*.*)"
        )
        if files:
            for f in files:
                if f not in self.video_paths:
                    self.video_paths.append(f)
            self.update_input_display()

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing Videos", "",
            QFileDialog.ShowDirsOnly
        )
        if folder:
            found = 0
            for f in sorted(os.listdir(folder)):
                full = os.path.join(folder, f)
                if os.path.isfile(full) and Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                    if full not in self.video_paths:
                        self.video_paths.append(full)
                        found += 1
            if found == 0:
                QMessageBox.information(self, "No Videos", "No video files found in the selected folder.")
            self.update_input_display()

    def clear_inputs(self):
        self.video_paths.clear()
        self.update_input_display()

    def update_input_display(self):
        if not self.video_paths:
            self.lbl_input.setText("No videos selected")
            self.btn_clear.setEnabled(False)
            self.btn_start.setEnabled(False)
            self.txt_log.clear()
        else:
            if len(self.video_paths) <= 10:
                text = "\n".join([f"  {os.path.basename(v)}" for v in self.video_paths])
            else:
                text = "\n".join([f"  {os.path.basename(v)}" for v in self.video_paths[:10]])
                text += f"\n  ... and {len(self.video_paths) - 10} more"
            self.lbl_input.setText(f"{len(self.video_paths)} video(s) selected:\n{text}")
            self.btn_clear.setEnabled(True)
            self.btn_start.setEnabled(bool(self.txt_output_dir.text()))
            self.txt_log.clear()
            self.txt_log.append(f"{len(self.video_paths)} video(s) ready for processing\n")

    def choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", ""
        )
        if directory:
            self.txt_output_dir.setText(directory)
            self.btn_start.setEnabled(bool(self.video_paths))

    def start_extraction(self):
        if not self.video_paths:
            QMessageBox.warning(self, "No Videos", "Please add video files first.")
            return

        output_base = self.txt_output_dir.text().strip()
        if not output_base:
            QMessageBox.warning(self, "No Output Directory", "Please choose an output directory.")
            return

        output_dir = os.path.join(output_base, "training_images")

        num_frames = self.spin_frames.value()
        sampling = "stride" if self.radio_stride.isChecked() else "random"

        self.btn_add_videos.setEnabled(False)
        self.btn_add_folder.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_choose_output.setEnabled(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.txt_log.clear()

        self.worker = ExtractWorker(self.video_paths, output_dir, num_frames, sampling)
        self.worker.progress.connect(self.update_log)
        self.worker.file_progress.connect(self.update_file_progress)
        self.worker.finished.connect(self.extraction_finished)
        self.worker.start()

    def stop_extraction(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.btn_stop.setEnabled(False)
            self.txt_log.append("\nStopping...")

    def update_log(self, message):
        self.txt_log.append(message)
        self.txt_log.verticalScrollBar().setValue(
            self.txt_log.verticalScrollBar().maximum()
        )

    def update_file_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.file_progress_label.setText(f"Processing video {current} of {total}")

    def extraction_finished(self, success, message):
        self.btn_add_videos.setEnabled(True)
        self.btn_add_folder.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_choose_output.setEnabled(True)

        self.progress_bar.setVisible(False)
        self.file_progress_label.setText("Ready")

        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Stopped", message)


def main():
    app = QApplication(sys.argv)
    window = GenerateTrainingImagesWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
