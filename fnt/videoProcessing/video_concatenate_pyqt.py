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
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition
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
    # Signal to ask user whether to skip corrupt file(s) or cancel.
    # Emits (message, file_list) — GUI must call worker.respond_to_prompt(True/False)
    user_decision_needed = pyqtSignal(str, list)  # message, list of filenames

    def __init__(self, input_dirs, output_filename, sort_order="default", instance_id=1):
        super().__init__()
        self.input_dirs = input_dirs
        self.output_filename = output_filename
        self.sort_order = sort_order
        self.instance_id = instance_id
        self.should_stop = False
        # Mutex/condition for blocking the worker until the user responds
        self._decision_mutex = QMutex()
        self._decision_cond = QWaitCondition()
        self._decision_result = None  # True = skip & continue, False = cancel

    def stop(self):
        """Stop the processing"""
        self.should_stop = True
        # Wake up the worker if it's waiting on a user decision so it can exit
        self._decision_mutex.lock()
        self._decision_result = False  # treat as cancel
        self._decision_cond.wakeAll()
        self._decision_mutex.unlock()

    # ------------------------------------------------------------------
    # Cross-thread user decision helpers
    # ------------------------------------------------------------------
    def respond_to_prompt(self, skip: bool):
        """Called from the GUI thread after the user clicks Skip or Cancel.
        *skip=True* means skip the bad file(s) and continue;
        *skip=False* means cancel the entire concatenation."""
        self._decision_mutex.lock()
        self._decision_result = skip
        self._decision_cond.wakeAll()
        self._decision_mutex.unlock()

    def _ask_user_skip_or_cancel(self, message: str, file_list: list) -> bool:
        """Emit a signal so the GUI can show a dialog, then block until the
        user responds.  Returns True (skip & continue) or False (cancel)."""
        self._decision_mutex.lock()
        self._decision_result = None
        self._decision_mutex.unlock()

        # Tell the GUI to show the dialog (runs on the main thread)
        self.user_decision_needed.emit(message, file_list)

        # Block until respond_to_prompt() is called
        self._decision_mutex.lock()
        while self._decision_result is None:
            self._decision_cond.wait(self._decision_mutex)
        result = self._decision_result
        self._decision_mutex.unlock()
        return result

    def sort_viewtron_files(self, video_files):
        """
        Sort ViewTron DVR files in chronological order.
        ViewTron naming: Base_YYYYMMDDHHMMSS.ext, then Base_YYYYMMDDHHMMSS(001).ext, etc.
        """
        import re

        def viewtron_sort_key(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'_(\d{14})(?:\((\d+)\))?', filename)
            if match:
                timestamp = match.group(1)
                sequence = match.group(2)
                if sequence is None:
                    sequence = -1
                else:
                    sequence = int(sequence)
                return (timestamp, sequence)
            else:
                return (filename, 0)

        return sorted(video_files, key=viewtron_sort_key)

    # ------------------------------------------------------------------
    # ffprobe helpers
    # ------------------------------------------------------------------
    def _probe_video(self, filepath):
        """Validate a single video file with ffprobe.
        Returns (ok: bool, info_or_error: str)."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,duration,nb_frames",
                "-show_entries", "format=duration,size",
                "-of", "default=noprint_wrappers=1",
                filepath
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return False, result.stderr.strip()
            return True, result.stdout.strip()
        except FileNotFoundError:
            return True, "ffprobe not found — skipping validation"
        except subprocess.TimeoutExpired:
            return False, "ffprobe timed out (file may be corrupt)"
        except Exception as e:
            return False, str(e)

    def _try_repair_video(self, filepath, folder_path):
        """Attempt to repair a corrupt video by re-muxing it with ffmpeg.
        Returns the path to the repaired file, or None on failure."""
        base = os.path.basename(filepath)
        name, ext = os.path.splitext(base)
        repaired_path = os.path.join(folder_path, f"{name}_repaired{ext}")

        self.progress_update.emit(f"🔧 Attempting to repair: {base}")
        cmd = [
            "ffmpeg", "-y",
            "-err_detect", "ignore_err",
            "-i", filepath,
            "-c", "copy",
            "-movflags", "+faststart",
            repaired_path
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0 and os.path.exists(repaired_path) and os.path.getsize(repaired_path) > 0:
                self.progress_update.emit(f"✅ Repair succeeded: {os.path.basename(repaired_path)}")
                return repaired_path
            else:
                self.progress_update.emit(f"❌ Repair failed for {base}")
                if proc.stderr:
                    for err_line in proc.stderr.strip().splitlines()[-3:]:
                        self.ffmpeg_output.emit(f"  repair stderr: {err_line}")
                # Clean up failed repair
                try:
                    os.remove(repaired_path)
                except Exception:
                    pass
                return None
        except Exception as e:
            self.progress_update.emit(f"❌ Repair error for {base}: {e}")
            try:
                os.remove(repaired_path)
            except Exception:
                pass
            return None

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
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

                success = self.concatenate_folder(input_dir)

                if success:
                    successful += 1
                else:
                    failed += 1

            if self.should_stop:
                self.finished.emit(False, "Processing stopped by user.")
            elif failed > 0:
                msg = f"Concatenation finished with errors. {successful} folder(s) succeeded, {failed} failed."
                self.finished.emit(False, msg)
            else:
                msg = f"Concatenation complete! Processed {successful} folder(s) successfully."
                self.finished.emit(True, msg)

        except Exception as e:
            self.finished.emit(False, f"Error during processing: {str(e)}")

    # ------------------------------------------------------------------
    # Core concat logic
    # ------------------------------------------------------------------
    def _run_ffmpeg_concat(self, video_files, folder_path, output_file):
        """Run the ffmpeg concat command on a list of video files.
        Returns (success: bool, last_stderr_lines: list[str])."""
        list_file = os.path.join(folder_path, "concat_list.txt")
        with open(list_file, "w") as fp:
            for video in video_files:
                rel_path = os.path.basename(video)
                fp.write(f"file '{rel_path}'\n")

        command = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",
            "-vsync", "vfr",
            "-an",
            output_file
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=folder_path
        )

        last_lines = []
        for line in process.stdout:
            if self.should_stop:
                process.terminate()
                try:
                    os.remove(list_file)
                except Exception:
                    pass
                return False, ["Stopped by user"]
            line = line.strip()
            if line:
                self.ffmpeg_output.emit(line)
                last_lines.append(line)
                if len(last_lines) > 20:
                    last_lines.pop(0)

        process.wait()

        try:
            os.remove(list_file)
        except Exception:
            pass

        return process.returncode == 0, last_lines

    def _find_bad_video_bisect(self, video_files, folder_path):
        """Use binary search to find the first video that causes concat failure.
        Returns the index of the problematic file, or -1 if not found."""
        self.progress_update.emit("🔍 Searching for the problematic video file...")

        # Try progressively larger slices from the start
        # Binary search: find the smallest N where concat of files[0:N] fails
        lo, hi = 1, len(video_files)
        bad_idx = -1
        temp_out = os.path.join(folder_path, "_bisect_test.mp4")

        while lo <= hi:
            if self.should_stop:
                break
            mid = (lo + hi) // 2
            self.progress_update.emit(f"  Testing first {mid} of {len(video_files)} files...")
            ok, _ = self._run_ffmpeg_concat(video_files[:mid], folder_path, temp_out)

            # Clean up test output
            try:
                os.remove(temp_out)
            except Exception:
                pass

            if ok:
                lo = mid + 1
            else:
                bad_idx = mid - 1  # 0-indexed: the last file in the failing slice
                hi = mid - 1

        return bad_idx

    def concatenate_folder(self, folder_path):
        """Concatenate all videos in a single folder with validation and auto-repair."""
        try:
            VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".MP4", ".mkv", ".flv", ".wmv", ".m4v")

            video_files_set = set()
            for ext in VIDEO_EXTENSIONS:
                found_files = glob.glob(os.path.join(folder_path, f"*{ext}"))
                video_files_set.update(found_files)

            video_files = list(video_files_set)

            if self.sort_order == "viewtron":
                video_files = self.sort_viewtron_files(video_files)
            else:
                video_files = sorted(video_files)

            if not video_files:
                self.progress_update.emit(f"⚠️ No video files found in {os.path.basename(folder_path)}")
                return False

            self.progress_update.emit(f"Found {len(video_files)} video file(s)")

            # --- Phase 1: Validate each video with ffprobe ---
            self.progress_update.emit("Validating video files with ffprobe...")
            bad_files = []
            for i, vf in enumerate(video_files):
                if self.should_stop:
                    return False
                ok, info = self._probe_video(vf)
                if not ok:
                    bad_files.append((i, vf, info))
                    self.progress_update.emit(
                        f"  ⚠️ [{i+1}/{len(video_files)}] PROBLEM: {os.path.basename(vf)}")
                    self.progress_update.emit(f"     Error: {info}")
                elif i % 50 == 0 or i == len(video_files) - 1:
                    self.progress_update.emit(
                        f"  ✓ Validated {i+1}/{len(video_files)} files...")

            # Attempt to repair any bad files found by ffprobe
            repaired_map = {}  # original_path -> repaired_path
            if bad_files:
                self.progress_update.emit(
                    f"Found {len(bad_files)} problematic file(s). Attempting repairs...")
                for idx, filepath, error_info in bad_files:
                    if self.should_stop:
                        return False
                    repaired = self._try_repair_video(filepath, folder_path)
                    if repaired:
                        repaired_map[filepath] = repaired

                # Swap repaired files into the list
                for i, vf in enumerate(video_files):
                    if vf in repaired_map:
                        video_files[i] = repaired_map[vf]

                # Any files that couldn't be repaired — ask user what to do
                unrepairable = [vf for _, vf, _ in bad_files if vf not in repaired_map]
                if unrepairable:
                    names = [os.path.basename(uf) for uf in unrepairable]
                    skip = self._ask_user_skip_or_cancel(
                        f"{len(unrepairable)} file(s) could not be repaired.",
                        names
                    )
                    if not skip or self.should_stop:
                        self.progress_update.emit("❌ Concatenation cancelled by user.")
                        self._cleanup_repaired(repaired_map)
                        return False
                    # User chose to skip — remove unrepairable files
                    for uf in unrepairable:
                        self.progress_update.emit(f"   Skipping: {os.path.basename(uf)}")
                    video_files = [vf for vf in video_files if vf not in unrepairable]

                if not video_files:
                    self.progress_update.emit("❌ No valid video files remain after validation.")
                    self._cleanup_repaired(repaired_map)
                    return False
            else:
                self.progress_update.emit(f"✅ All {len(video_files)} files passed validation.")

            # --- Phase 2: Attempt full concatenation ---
            output_file = os.path.join(folder_path, self.output_filename)
            if os.path.exists(output_file):
                counter = 1
                base_name, ext = os.path.splitext(self.output_filename)
                while os.path.exists(output_file):
                    output_file = os.path.join(folder_path, f"{base_name}_{counter}{ext}")
                    counter += 1
                self.progress_update.emit(f"Output file exists, using: {os.path.basename(output_file)}")

            self.progress_update.emit(f"Concatenating {len(video_files)} videos...")
            ok, last_lines = self._run_ffmpeg_concat(video_files, folder_path, output_file)

            if ok:
                self.progress_update.emit(f"✅ Successfully created: {os.path.basename(output_file)}")
                self._cleanup_repaired(repaired_map)
                return True

            # --- Phase 3: Concat failed — find the culprit via binary search ---
            self.progress_update.emit("❌ Full concatenation failed. Identifying problematic file...")

            # Show last few lines of FFmpeg output for context
            if last_lines:
                self.progress_update.emit("Last FFmpeg output before failure:")
                for ln in last_lines[-5:]:
                    self.progress_update.emit(f"  {ln}")

            bad_idx = self._find_bad_video_bisect(video_files, folder_path)

            if bad_idx >= 0 and bad_idx < len(video_files):
                bad_file = video_files[bad_idx]
                self.progress_update.emit(
                    f"🔴 IDENTIFIED problematic file #{bad_idx + 1}: {os.path.basename(bad_file)}")
                self.progress_update.emit(
                    f"   Full path: {bad_file}")

                # Try to repair this file too
                repaired = self._try_repair_video(bad_file, folder_path)
                if repaired:
                    video_files[bad_idx] = repaired
                    repaired_map[bad_file] = repaired
                    self.progress_update.emit("Retrying concatenation with repaired file...")
                    ok2, _ = self._run_ffmpeg_concat(video_files, folder_path, output_file)
                    if ok2:
                        self.progress_update.emit(
                            f"✅ Successfully created (after repair): {os.path.basename(output_file)}")
                        self._cleanup_repaired(repaired_map)
                        return True
                    else:
                        self.progress_update.emit(
                            "❌ Concatenation still failed after repair. "
                            "Trying without the problematic file...")

                # Last resort: ask user whether to skip the problematic file
                skipped_name = os.path.basename(bad_file)
                skip = self._ask_user_skip_or_cancel(
                    "A problematic video file was identified that could not be repaired.",
                    [skipped_name]
                )
                if not skip or self.should_stop:
                    self.progress_update.emit("❌ Concatenation cancelled by user.")
                    self._cleanup_repaired(repaired_map)
                    return False

                # User chose to skip — retry without the bad file
                video_files_without = [vf for vf in video_files if vf != bad_file and vf != repaired]
                if video_files_without:
                    self.progress_update.emit(
                        f"Retrying concatenation without: {skipped_name} "
                        f"({len(video_files_without)} files remaining)...")
                    ok3, _ = self._run_ffmpeg_concat(video_files_without, folder_path, output_file)
                    if ok3:
                        self.progress_update.emit(
                            f"✅ Created (skipped {skipped_name}): {os.path.basename(output_file)}")
                        self.progress_update.emit(
                            f"⚠️ NOTE: The output is missing footage from: {skipped_name}")
                        self._cleanup_repaired(repaired_map)
                        return True

            self.progress_update.emit(
                f"❌ Failed to concatenate videos in {os.path.basename(folder_path)}")
            self._cleanup_repaired(repaired_map)
            return False

        except Exception as e:
            self.progress_update.emit(f"❌ Error processing {folder_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _cleanup_repaired(self, repaired_map):
        """Remove any temporary repaired video files."""
        for original, repaired in repaired_map.items():
            try:
                if os.path.exists(repaired):
                    os.remove(repaired)
            except Exception:
                pass


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
        self.worker.user_decision_needed.connect(self._handle_user_decision)
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
    
    def _handle_user_decision(self, message, file_list):
        """Show a dialog asking the user whether to skip corrupt file(s) or cancel.
        Called on the GUI thread via the worker's user_decision_needed signal."""
        file_names = "\n".join([f"  • {f}" for f in file_list])
        full_message = (
            f"{message}\n\n"
            f"Affected file(s):\n{file_names}\n\n"
            "Would you like to skip these file(s) and continue concatenation, "
            "or cancel the process?"
        )
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Corrupt Video File")
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(full_message)
        skip_btn = msg_box.addButton("Skip && Continue", QMessageBox.AcceptRole)
        cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
        msg_box.setDefaultButton(cancel_btn)
        msg_box.exec_()

        skip = msg_box.clickedButton() == skip_btn
        self.worker.respond_to_prompt(skip)

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
            QMessageBox.warning(self, "Concatenation Failed", message)
    
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
