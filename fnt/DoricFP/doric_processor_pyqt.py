#!/usr/bin/env python3
"""
Doric WiFP Fiber Photometry Processing Tool (PyQt5 GUI)

Provides batch processing of .doric files:
- Reads photometry signals (470nm signal, 405/415nm isosbestic)
- Calculates ΔF/F with isosbestic correction
- Synchronizes with behavior video timestamps
- Exports CSV (Frame, Time_sec, DeltaF_F) and combined video

Standalone usage:
    python doric_processor_pyqt.py
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTextEdit, QFileDialog, QGroupBox, QCheckBox, QComboBox,
    QMessageBox, QListWidget, QListWidgetItem, QSpinBox, QDoubleSpinBox,
    QProgressBar, QSplitter, QFrame, QGridLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

# Import the processing classes
from fnt.DoricFP.doric_processor import (
    DoricFileReader, DFFCalculator, TraditionalDFFCalculator, VideoSynchronizer,
    DoricFileData, DoricChannelInfo, find_doric_video_pairs
)


class DoricProcessWorker(QThread):
    """Worker thread for batch processing .doric files"""
    progress = pyqtSignal(str)
    file_started = pyqtSignal(str)
    file_finished = pyqtSignal(str, bool, str)
    all_finished = pyqtSignal(bool, str)
    
    def __init__(
        self,
        files: List[Path],
        # Pipeline selection
        pipeline_type: str = 'traditional',  # 'traditional' or 'doric'
        # Traditional pipeline parameters
        trad_apply_filter: bool = True,
        trad_filter_cutoff: float = 20.0,
        trad_filter_order: int = 100,
        trad_clean_dropouts: bool = True,
        trad_dropout_threshold: float = 0.5,
        # Doric-style processing parameters
        smooth_algorithm: str = 'butterworth',  # 'none', 'butterworth', 'running_average'
        filter_cutoff: float = 2.0,
        baseline_lambda: float = 10.0,
        onset_trim: float = 0.0,
        offset_trim: float = 0.0,
        fit_max_threshold: float = 3.0,
        # Output options
        create_video: bool = True,
        show_raw_data: bool = False,
        overwrite: bool = False,
        # Channel overrides
        signal_channel_override: Optional[str] = None,
        isosbestic_channel_override: Optional[str] = None
    ):
        super().__init__()
        self.files = files
        # Pipeline type
        self.pipeline_type = pipeline_type
        # Traditional parameters
        self.trad_apply_filter = trad_apply_filter
        self.trad_filter_cutoff = trad_filter_cutoff
        self.trad_filter_order = trad_filter_order
        self.trad_clean_dropouts = trad_clean_dropouts
        self.trad_dropout_threshold = trad_dropout_threshold
        # Doric parameters
        self.smooth_algorithm = smooth_algorithm
        self.filter_cutoff = filter_cutoff
        self.baseline_lambda = baseline_lambda
        self.onset_trim = onset_trim
        self.offset_trim = offset_trim
        self.fit_max_threshold = fit_max_threshold
        # Output options
        self.create_video = create_video
        self.show_raw_data = show_raw_data
        self.overwrite = overwrite
        # Channel overrides
        self.signal_channel_override = signal_channel_override
        self.isosbestic_channel_override = isosbestic_channel_override
        self._stop_requested = False
    
    def request_stop(self):
        """Request the worker to stop processing"""
        self._stop_requested = True
        self.progress.emit("\n⚠️ Stop requested by user...")
    
    def run(self):
        """Process all files"""
        try:
            total_success = 0
            total_failed = 0
            total_skipped = 0
            
            for i, doric_file in enumerate(self.files):
                if self._stop_requested:
                    break
                
                self.file_started.emit(str(doric_file))
                self.progress.emit(f"\n{'='*60}")
                self.progress.emit(f"Processing [{i+1}/{len(self.files)}]: {doric_file.name}")
                self.progress.emit(f"{'='*60}")
                
                success, message = self.process_single_file(doric_file)
                
                if success:
                    total_success += 1
                    self.file_finished.emit(str(doric_file), True, message)
                elif "already exists" in message.lower():
                    total_skipped += 1
                    self.file_finished.emit(str(doric_file), False, message)
                else:
                    total_failed += 1
                    self.file_finished.emit(str(doric_file), False, message)
            
            # Summary
            summary = f"\n{'='*60}\n"
            if self._stop_requested:
                summary += "⚠️ Processing stopped by user!\n"
            else:
                summary += "✅ Processing complete!\n"
            summary += f"Files processed: {total_success}\n"
            summary += f"Files skipped: {total_skipped}\n"
            summary += f"Files failed: {total_failed}\n"
            summary += f"Total files: {len(self.files)}\n"
            
            self.progress.emit(summary)
            self.all_finished.emit(not self._stop_requested, summary)
            
        except Exception as e:
            self.all_finished.emit(False, f"Error during processing: {str(e)}")
    
    def process_single_file(self, doric_path: Path) -> Tuple[bool, str]:
        """Process a single .doric file"""
        try:
            # Check output files
            csv_path = doric_path.parent / f"{doric_path.stem}_dff.csv"
            video_path = doric_path.parent / f"{doric_path.stem}_combined.mp4"
            
            if not self.overwrite:
                if csv_path.exists():
                    self.progress.emit(f"⏭️ CSV already exists: {csv_path.name}")
                    return False, "Output already exists"
            
            # Read file
            self.progress.emit("📂 Reading .doric file...")
            reader = DoricFileReader(doric_path)
            file_data = reader.scan_file()
            
            # Report detected channels
            if file_data.signal_channel:
                self.progress.emit(f"   Signal (470nm): {file_data.signal_channel.name}")
                self.progress.emit(f"     - {file_data.signal_channel.n_samples} samples at {file_data.signal_channel.sampling_rate:.1f} Hz")
            else:
                self.progress.emit("   ⚠️ No signal channel detected!")
            
            if file_data.isosbestic_channel:
                self.progress.emit(f"   Isosbestic (415nm): {file_data.isosbestic_channel.name}")
            else:
                self.progress.emit("   ⚠️ No isosbestic channel detected (using baseline method)")
            
            if file_data.video_info:
                self.progress.emit(f"   Video: {file_data.video_info.n_frames} frames at {file_data.video_info.fps:.1f} fps")
                if file_data.video_info.absolute_path:
                    self.progress.emit(f"   Video file: {Path(file_data.video_info.absolute_path).name}")
            
            # Apply channel overrides if specified
            if self.signal_channel_override:
                for ch in file_data.all_channels:
                    if ch.path == self.signal_channel_override:
                        file_data.signal_channel = ch
                        break
            
            if self.isosbestic_channel_override:
                for ch in file_data.all_channels:
                    if ch.path == self.isosbestic_channel_override:
                        file_data.isosbestic_channel = ch
                        break
            
            # Validate
            if file_data.signal_channel is None:
                return False, "No signal channel available"
            
            # Load data
            self.progress.emit("📊 Loading signal data...")
            file_data = reader.load_data(file_data)
            
            # Get sampling rate
            srate = file_data.signal_channel.sampling_rate
            self.progress.emit(f"   Sampling rate: {srate:.1f} Hz")
            
            # Calculate ΔF/F based on pipeline type
            if self.pipeline_type == 'traditional':
                self.progress.emit(f"🔬 Calculating ΔF/F (Traditional MATLAB-style)...")
                self.progress.emit(f"   • Filter: {'FIR ' + str(self.trad_filter_cutoff) + ' Hz' if self.trad_apply_filter else 'None'}")
                self.progress.emit(f"   • Dropout cleaning: {'Enabled (threshold: ' + str(self.trad_dropout_threshold) + ')' if self.trad_clean_dropouts else 'Disabled'}")
                
                calc = TraditionalDFFCalculator(
                    apply_filter=self.trad_apply_filter,
                    filter_cutoff=self.trad_filter_cutoff,
                    filter_order=self.trad_filter_order,
                    clean_dropouts=self.trad_clean_dropouts,
                    dropout_threshold=self.trad_dropout_threshold
                )
                
                if file_data.isosbestic_channel is not None and file_data.isosbestic_data is not None:
                    dff, proc_signal, proc_iso, fitted = calc.calculate(
                        file_data.signal_data,
                        file_data.isosbestic_data,
                        srate
                    )
                    if calc.last_dropout_count > 0:
                        self.progress.emit(f"   ⚠️ Cleaned {calc.last_dropout_count} signal dropouts (wireless transmission issues)")
                    raw_signal = file_data.signal_data.copy()
                    raw_iso = file_data.isosbestic_data.copy()
                else:
                    dff, proc_signal = calc.calculate_simple(file_data.signal_data, srate)
                    if calc.last_dropout_count > 0:
                        self.progress.emit(f"   ⚠️ Cleaned {calc.last_dropout_count} signal dropouts (wireless transmission issues)")
                    raw_signal = file_data.signal_data.copy()
                    raw_iso = None
                    proc_iso = None
                    fitted = None
            else:
                # Doric-style pipeline
                self.progress.emit(f"🔬 Calculating ΔF/F (Doric-style)...")
                self.progress.emit(f"   • Smooth: {self.smooth_algorithm} (cutoff: {self.filter_cutoff} Hz)")
                self.progress.emit(f"   • Baseline λ: {self.baseline_lambda:.0f}")
                self.progress.emit(f"   • Trim: onset={self.onset_trim}s, offset={self.offset_trim}s")
                self.progress.emit(f"   • Fit threshold: {self.fit_max_threshold}σ")
                
                calc = DFFCalculator(
                    smooth_algorithm=self.smooth_algorithm,
                    filter_cutoff=self.filter_cutoff,
                    baseline_lambda=self.baseline_lambda,
                    onset_trim=self.onset_trim,
                    offset_trim=self.offset_trim,
                    fit_max_threshold=self.fit_max_threshold
                )
                
                if file_data.isosbestic_channel is not None and file_data.isosbestic_data is not None:
                    dff, proc_signal, proc_iso, fitted, valid_mask = calc.calculate(
                        file_data.signal_data,
                        file_data.isosbestic_data,
                        srate
                    )
                    raw_signal = file_data.signal_data.copy()
                    raw_iso = file_data.isosbestic_data.copy()
                else:
                    dff, proc_signal = calc.calculate_simple(file_data.signal_data, srate)
                    raw_signal = file_data.signal_data.copy()
                    raw_iso = None
                    proc_iso = None
                    fitted = None
            
            self.progress.emit(f"   ΔF/F range: [{np.min(dff):.2f}%, {np.max(dff):.2f}%]")
            
            # Synchronize with video
            if file_data.video_info is not None and file_data.video_timestamps is not None:
                self.progress.emit("🎬 Synchronizing with video frames...")
                sync = VideoSynchronizer(method='average')
                frame_dff, frame_nums = sync.synchronize(
                    dff, file_data.signal_time, file_data.video_timestamps
                )
                
                df = sync.create_aligned_dataframe(
                    frame_dff, frame_nums, file_data.video_timestamps
                )
                self.progress.emit(f"   Synchronized {len(frame_nums)} video frames")
            else:
                self.progress.emit("📝 No video sync available, exporting raw timestamps...")
                df = pd.DataFrame({
                    'Sample': np.arange(1, len(dff) + 1),
                    'Time_sec': file_data.signal_time,
                    'DeltaF_F': dff
                })
            
            # Save CSV
            self.progress.emit(f"💾 Saving CSV: {csv_path.name}")
            df.to_csv(csv_path, index=False)
            
            # Create processed video and combined video (if requested)
            if self.create_video and file_data.video_info and file_data.video_info.absolute_path:
                source_video = file_data.video_info.absolute_path
                
                # Put _proc.mp4 in the _Videos folder (grayscale, re-encoded)
                video_folder = doric_path.parent / f"{doric_path.stem}_Videos"
                if video_folder.exists():
                    proc_video_path = video_folder / f"{doric_path.stem}_proc.mp4"
                else:
                    proc_video_path = doric_path.parent / f"{doric_path.stem}_proc.mp4"
                
                # Step 1: Create _proc.mp4 (grayscale, re-encoded for tracking compatibility)
                self.progress.emit("🎬 Creating processed video (_proc.mp4, grayscale)...")
                proc_success = self.create_processed_video(source_video, proc_video_path, grayscale=True)
                
                if proc_success:
                    self.progress.emit(f"✅ Processed video saved: {proc_video_path.name}")
                    
                    # Step 2: Create combined video with ΔF/F overlay
                    self.progress.emit("🎬 Creating combined video with ΔF/F overlay...")
                    
                    # Prepare trace data for visualization
                    trace_data = {
                        'time': file_data.signal_time,
                        'dff': dff,
                        'raw_signal': raw_signal if self.show_raw_data else None,
                        'raw_iso': raw_iso if self.show_raw_data else None,
                        'proc_signal': proc_signal if self.show_raw_data else None,
                        'imu_time': file_data.imu_time if self.show_raw_data else None,
                        'imu_x': file_data.imu_accel_x if self.show_raw_data else None,
                        'imu_y': file_data.imu_accel_y if self.show_raw_data else None,
                        'imu_z': file_data.imu_accel_z if self.show_raw_data else None,
                    }
                    
                    try:
                        video_success = self.create_combined_video(
                            str(proc_video_path),
                            df,
                            video_path,
                            file_data.video_info.fps,
                            trace_data=trace_data if self.show_raw_data else None
                        )
                        if video_success:
                            self.progress.emit(f"✅ Combined video saved: {video_path.name}")
                        else:
                            self.progress.emit("⚠️ Combined video creation failed")
                    except Exception as e:
                        self.progress.emit(f"⚠️ Combined video error: {str(e)}")
                else:
                    self.progress.emit("⚠️ Processed video creation failed, skipping combined video")
            
            self.progress.emit(f"✅ Successfully processed: {doric_path.name}")
            return True, f"CSV saved: {csv_path.name}"
            
        except Exception as e:
            self.progress.emit(f"❌ Error: {str(e)}")
            return False, str(e)
    
    def create_processed_video(self, source_path: str, output_path: Path, grayscale: bool = True) -> bool:
        """Re-encode raw video to H.264 (optionally grayscale) for tracking compatibility using FFmpeg"""
        import subprocess
        
        try:
            # Build FFmpeg command with optional grayscale filter
            if grayscale:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", source_path,
                    "-vf", "format=gray",  # Convert to grayscale
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-an",  # Remove audio
                    "-movflags", "+faststart",
                    str(output_path)
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", source_path,
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-an",  # Remove audio
                    "-movflags", "+faststart",
                    str(output_path)
            ]
            
            self.progress.emit(f"   Running FFmpeg to create processed video...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.progress.emit(f"   FFmpeg error: {result.stderr[:300]}")
                return False
            
            return True
            
        except FileNotFoundError:
            self.progress.emit("   ❌ FFmpeg not found! Please install FFmpeg.")
            return False
        except Exception as e:
            self.progress.emit(f"   FFmpeg error: {str(e)}")
            return False
    
    def create_combined_video(
        self,
        video_path: str,
        df: pd.DataFrame,
        output_path: Path,
        fps: float,
        trace_data: Optional[dict] = None
    ) -> bool:
        """Create combined video with ΔF/F trace using FFmpeg"""
        import subprocess
        import tempfile
        import cv2
        
        try:
            # Get video dimensions
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.progress.emit(f"   Could not open video: {video_path}")
                return False
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # For horizontal layout (video left, traces right):
            # Trace panel is same height as video, width ~60% of video width for good proportions
            trace_width = int(width * 0.6) if trace_data else int(width * 0.4)
            trace_height = height
            
            # Get ΔF/F data
            if 'Frame' in df.columns:
                dff_values = df['DeltaF_F'].values
                time_values = df['Time_sec'].values
            else:
                dff_values = df['DeltaF_F'].values[:total_frames]
                time_values = df['Time_sec'].values[:total_frames]
            
            dff_min, dff_max = np.min(dff_values), np.max(dff_values)
            n_frames = min(len(time_values), total_frames)
            
            # Create trace video frames in temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                self.progress.emit(f"   Generating {n_frames} trace frames...")
                
                # Generate trace frames as PNGs
                for i in range(n_frames):
                    if self._stop_requested:
                        return False
                    
                    trace_img = self.create_trace_frame(
                        time_values, dff_values, i,
                        trace_width, trace_height, dff_min, dff_max,
                        trace_data=trace_data
                    )
                    
                    frame_path = Path(tmpdir) / f"trace_{i:06d}.png"
                    cv2.imwrite(str(frame_path), trace_img)
                    
                    if i % 500 == 0:
                        self.progress.emit(f"   Trace frames: {i}/{n_frames}")
                
                # Use FFmpeg to stack video and trace
                self.progress.emit("   Compositing video with FFmpeg...")
                
                trace_pattern = str(Path(tmpdir) / "trace_%06d.png")
                
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-framerate", str(fps),
                    "-i", trace_pattern,
                    "-filter_complex", "[0:v][1:v]hstack=inputs=2",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    "-movflags", "+faststart",
                    str(output_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.progress.emit(f"   FFmpeg composite error: {result.stderr[:300]}")
                    return False
                
                return True
            
        except FileNotFoundError:
            self.progress.emit("   ❌ FFmpeg not found! Please install FFmpeg.")
            return False
        except Exception as e:
            self.progress.emit(f"   Video error: {str(e)}")
            return False
    
    def create_trace_frame(
        self,
        time_values: np.ndarray,
        dff_values: np.ndarray,
        current_frame: int,
        width: int,
        height: int,
        dff_min: float,
        dff_max: float,
        trace_data: Optional[dict] = None
    ) -> np.ndarray:
        """Create a single frame of the ΔF/F trace plot
        
        If trace_data is provided, shows multiple traces in order:
        - Row 1: 470nm calcium-dependent signal (green)
        - Row 2: 415nm calcium-independent/isosbestic (magenta)
        - Row 3: ΔF/F trace (green)
        - Row 4: IMU accelerometer magnitude (cyan, if available)
        """
        import cv2
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        # Create figure with dark background
        dpi = 100
        fig_width = width / dpi
        fig_height = height / dpi
        
        # Determine number of subplots based on available data
        if trace_data and trace_data.get('raw_signal') is not None:
            has_iso = trace_data.get('raw_iso') is not None
            has_imu = trace_data.get('imu_x') is not None
            # Count: 470nm + (optional 415nm) + ΔF/F + (optional IMU)
            n_rows = 1 + (1 if has_iso else 0) + 1 + (1 if has_imu else 0)
            fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), dpi=dpi, 
                                     height_ratios=[1] * n_rows)
            if n_rows == 1:
                axes = [axes]
        else:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            axes = [ax]
            n_rows = 1
        
        fig.patch.set_facecolor('#1e1e1e')
        for ax in axes:
            ax.set_facecolor('#1e1e1e')
        
        current_time = time_values[current_frame] if current_frame < len(time_values) else time_values[-1]
        
        ax_idx = 0  # Track which axis we're on
        
        if trace_data and trace_data.get('raw_signal') is not None:
            raw_time = trace_data['time']
            raw_signal = trace_data['raw_signal']
            raw_iso = trace_data['raw_iso']
            
            # Find current index in trace data
            trace_idx = np.searchsorted(raw_time, current_time)
            trace_idx = min(trace_idx, len(raw_time) - 1)
            
            # Row 1: 470nm calcium-dependent signal
            ax_470 = axes[ax_idx]
            ax_idx += 1
            
            # Plot full trace dimmed
            ax_470.plot(raw_time, raw_signal, color='#1a4a1a', linewidth=0.5, alpha=0.5)
            # Plot up to current time
            ax_470.plot(raw_time[:trace_idx+1], raw_signal[:trace_idx+1], 
                       color='#00ff00', linewidth=1)
            ax_470.axvline(x=current_time, color='#ff0000', linewidth=1, alpha=0.7)
            ax_470.set_xlim(raw_time[0], raw_time[-1])
            ax_470.set_ylabel('LockIn01', color='#00ff00', fontsize=11)
            ax_470.tick_params(colors='#cccccc', labelsize=9)
            ax_470.set_title('Headstage01LockIn01 (470nm)', color='#00ff00', fontsize=10, loc='left')
            for spine in ax_470.spines.values():
                spine.set_color('#3f3f3f')
            
            # Row 2: 415nm calcium-independent (isosbestic) - if available
            if raw_iso is not None:
                ax_415 = axes[ax_idx]
                ax_idx += 1
                
                # Plot full trace dimmed
                ax_415.plot(raw_time, raw_iso, color='#4a1a4a', linewidth=0.5, alpha=0.5)
                # Plot up to current time
                ax_415.plot(raw_time[:trace_idx+1], raw_iso[:trace_idx+1], 
                           color='#ff00ff', linewidth=1)
                ax_415.axvline(x=current_time, color='#ff0000', linewidth=1, alpha=0.7)
                ax_415.set_xlim(raw_time[0], raw_time[-1])
                ax_415.set_ylabel('LockIn02', color='#ff00ff', fontsize=11)
                ax_415.tick_params(colors='#cccccc', labelsize=9)
                ax_415.set_title('Headstage01LockIn02 (415nm)', color='#ff00ff', fontsize=10, loc='left')
                for spine in ax_415.spines.values():
                    spine.set_color('#3f3f3f')
            
            # Row 3: ΔF/F
            ax_dff = axes[ax_idx]
            ax_idx += 1
        else:
            ax_dff = axes[0]
            ax_idx = 1
        
        # Plot ΔF/F trace
        if current_frame < len(time_values):
            # Plot full trace dimmed
            ax_dff.plot(time_values, dff_values, color='#3f3f3f', linewidth=0.5)
            
            # Plot up to current position
            ax_dff.plot(time_values[:current_frame+1], dff_values[:current_frame+1], 
                       color='#00ff00', linewidth=1)
            
            # Mark current position
            ax_dff.axvline(x=current_time, color='#ff0000', linewidth=1, alpha=0.7)
            ax_dff.scatter([current_time], [dff_values[current_frame]], 
                          color='#ff0000', s=20, zorder=5)
        
        ax_dff.set_xlim(time_values[0], time_values[-1])
        margin = max(abs(dff_min), abs(dff_max)) * 0.1
        ax_dff.set_ylim(dff_min - margin, dff_max + margin)
        ax_dff.set_ylabel('ΔF/F (%)', color='#00ff00', fontsize=11)
        ax_dff.tick_params(colors='#cccccc', labelsize=9)
        ax_dff.set_title('ΔF/F (Corrected)', color='#00ff00', fontsize=10, loc='left')
        for spine in ax_dff.spines.values():
            spine.set_color('#3f3f3f')
        
        # Row 4: IMU (if available) - Plot X, Y, Z separately like Doric software
        if trace_data and trace_data.get('imu_x') is not None:
            ax_imu = axes[ax_idx]
            imu_time = trace_data['imu_time']
            imu_x = trace_data['imu_x']
            imu_y = trace_data['imu_y']
            imu_z = trace_data['imu_z']
            
            # Find current index
            imu_idx = np.searchsorted(imu_time, current_time)
            imu_idx = min(imu_idx, len(imu_time) - 1)
            
            # Plot X (red), Y (green), Z (blue) - matching Doric colors
            # Dimmed full traces
            ax_imu.plot(imu_time, imu_x, color='#4a1a1a', linewidth=0.4, alpha=0.4)
            ax_imu.plot(imu_time, imu_y, color='#1a4a1a', linewidth=0.4, alpha=0.4)
            ax_imu.plot(imu_time, imu_z, color='#1a1a4a', linewidth=0.4, alpha=0.4)
            
            # Bright traces up to current time
            ax_imu.plot(imu_time[:imu_idx+1], imu_x[:imu_idx+1], 
                       color='#ff0000', linewidth=0.8, label='X')
            ax_imu.plot(imu_time[:imu_idx+1], imu_y[:imu_idx+1], 
                       color='#00ff00', linewidth=0.8, label='Y')
            ax_imu.plot(imu_time[:imu_idx+1], imu_z[:imu_idx+1], 
                       color='#0088ff', linewidth=0.8, label='Z')
            
            ax_imu.axvline(x=current_time, color='#ff0000', linewidth=1, alpha=0.5)
            ax_imu.set_xlim(imu_time[0], imu_time[-1])
            ax_imu.set_ylabel('Accel (m/s²)', color='#00ffff', fontsize=11)
            ax_imu.set_xlabel('Time (s)', color='#cccccc', fontsize=11)
            ax_imu.tick_params(colors='#cccccc', labelsize=9)
            ax_imu.set_title('Acceleration', color='#00ffff', fontsize=10, loc='left')
            ax_imu.legend(loc='upper right', fontsize=8, framealpha=0.5,
                         labelcolor=['#ff0000', '#00ff00', '#0088ff'])
            for spine in ax_imu.spines.values():
                spine.set_color('#3f3f3f')
        else:
            # Add x-label to bottom plot
            axes[-1].set_xlabel('Time (s)', color='#cccccc', fontsize=11)
        
        plt.tight_layout(pad=0.3)
        
        # Convert to numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.asarray(buf)
        plt.close(fig)
        
        # Convert RGBA to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Resize to exact dimensions if needed
        if img_bgr.shape[1] != width or img_bgr.shape[0] != height:
            img_bgr = cv2.resize(img_bgr, (width, height))
        
        return img_bgr


class DoricProcessorWindow(QWidget):
    """Main window for Doric WiFP processing tool"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Doric WiFP Fiber Photometry Processor")
        self.setMinimumSize(900, 700)
        
        self.files_to_process: List[Path] = []
        self.worker: Optional[DoricProcessWorker] = None
        
        self.setup_ui()
        self.apply_dark_theme()
    
    def setup_ui(self):
        """Set up the user interface"""
        main_layout = QVBoxLayout()
        
        # Title
        title = QLabel("Doric WiFP Fiber Photometry Processor")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: #cccccc; margin: 10px;")
        main_layout.addWidget(title)
        
        # Create splitter for file list and log
        splitter = QSplitter(Qt.Vertical)
        
        # Top section: File selection and options
        top_widget = QWidget()
        top_layout = QVBoxLayout()
        top_widget.setLayout(top_layout)
        
        # File selection group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        
        # Buttons row
        btn_layout = QHBoxLayout()
        
        self.btn_add_folder = QPushButton("📁 Add Folder")
        self.btn_add_folder.setToolTip("Scan folder for .doric files with matching _Videos folders")
        self.btn_add_folder.clicked.connect(self.add_folder)
        btn_layout.addWidget(self.btn_add_folder)
        
        self.btn_add_files = QPushButton("📄 Add Files")
        self.btn_add_files.setToolTip("Select individual .doric files")
        self.btn_add_files.clicked.connect(self.add_files)
        btn_layout.addWidget(self.btn_add_files)
        
        self.btn_remove_selected = QPushButton("❌ Remove Selected")
        self.btn_remove_selected.clicked.connect(self.remove_selected)
        btn_layout.addWidget(self.btn_remove_selected)
        
        self.btn_clear_all = QPushButton("🗑️ Clear All")
        self.btn_clear_all.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.btn_clear_all)
        
        file_layout.addLayout(btn_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(150)
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        file_layout.addWidget(self.file_list)
        
        file_group.setLayout(file_layout)
        top_layout.addWidget(file_group)
        
        # ====== Pipeline Selection ======
        pipeline_group = QGroupBox("ΔF/F Processing Pipeline")
        pipeline_layout = QVBoxLayout()
        
        # Pipeline selector row
        pipeline_row = QHBoxLayout()
        pipeline_row.addWidget(QLabel("Pipeline:"))
        self.pipeline_selector = QComboBox()
        self.pipeline_selector.addItems(["Traditional (MATLAB-style)", "Doric-Style"])
        self.pipeline_selector.setCurrentIndex(0)
        self.pipeline_selector.setToolTip(
            "Traditional: Replicates MATLAB fitcaldff.m approach (FIR filter + polyfit/NNLS)\n"
            "Doric-Style: Butterworth filter + ALS baseline + robust regression"
        )
        self.pipeline_selector.currentIndexChanged.connect(self._on_pipeline_changed)
        pipeline_row.addWidget(self.pipeline_selector)
        pipeline_row.addStretch()
        
        # Preview button
        self.btn_preview = QPushButton("🔍 Preview First File")
        self.btn_preview.setToolTip("Generate a quick preview plot of the first file's ΔF/F before batch processing")
        self.btn_preview.clicked.connect(self.preview_first_file)
        pipeline_row.addWidget(self.btn_preview)
        
        pipeline_layout.addLayout(pipeline_row)
        pipeline_group.setLayout(pipeline_layout)
        top_layout.addWidget(pipeline_group)
        
        # ====== Traditional Pipeline Options ======
        self.trad_options_group = QGroupBox("Traditional Pipeline Options")
        trad_layout = QGridLayout()
        
        self.chk_trad_filter = QCheckBox("Apply Finite Impulse Response (FIR) Low-pass Filter")
        self.chk_trad_filter.setChecked(True)  # Enabled by default to match MATLAB
        self.chk_trad_filter.setToolTip("Low-pass filter as in MATLAB filteredfitcaldff.m (vs unfiltered fitcaldff.m)")
        trad_layout.addWidget(self.chk_trad_filter, 0, 0, 1, 2)
        
        trad_layout.addWidget(QLabel("Cutoff (Hz):"), 0, 2)
        self.trad_filter_cutoff = QDoubleSpinBox()
        self.trad_filter_cutoff.setRange(1.0, 100.0)
        self.trad_filter_cutoff.setValue(20.0)  # MATLAB default
        self.trad_filter_cutoff.setSingleStep(1.0)
        self.trad_filter_cutoff.setDecimals(1)
        self.trad_filter_cutoff.setToolTip("FIR filter cutoff frequency (Hz). MATLAB default: 20 Hz")
        trad_layout.addWidget(self.trad_filter_cutoff, 0, 3)
        
        trad_layout.addWidget(QLabel("Filter Order:"), 0, 4)
        self.trad_filter_order = QSpinBox()
        self.trad_filter_order.setRange(10, 500)
        self.trad_filter_order.setValue(100)  # MATLAB default
        self.trad_filter_order.setToolTip("FIR filter order. MATLAB default: 100")
        trad_layout.addWidget(self.trad_filter_order, 0, 5)
        
        # Row 2: Dropout cleaning
        self.chk_trad_dropouts = QCheckBox("Clean Wireless Signal Dropouts")
        self.chk_trad_dropouts.setChecked(True)  # Enabled by default
        self.chk_trad_dropouts.setToolTip("Interpolate over wireless transmission dropouts (values that drop near zero)")
        trad_layout.addWidget(self.chk_trad_dropouts, 1, 0, 1, 2)
        
        trad_layout.addWidget(QLabel("Threshold (×median):"), 1, 2)
        self.trad_dropout_threshold = QDoubleSpinBox()
        self.trad_dropout_threshold.setRange(0.1, 0.9)
        self.trad_dropout_threshold.setValue(0.5)
        self.trad_dropout_threshold.setSingleStep(0.1)
        self.trad_dropout_threshold.setDecimals(2)
        self.trad_dropout_threshold.setToolTip("Values below (median × threshold) are treated as dropouts. Default: 0.5")
        trad_layout.addWidget(self.trad_dropout_threshold, 1, 3)
        
        # Sampling rate info
        self.trad_srate_label = QLabel("Sampling Rate: (auto-detected from file)")
        self.trad_srate_label.setStyleSheet("color: #888888; font-style: italic;")
        trad_layout.addWidget(self.trad_srate_label, 2, 0, 1, 6)
        
        self.trad_options_group.setLayout(trad_layout)
        top_layout.addWidget(self.trad_options_group)
        
        # ====== Doric-Style Processing Options ======
        self.doric_options_group = QGroupBox("Doric-Style Pipeline Options")
        doric_layout = QGridLayout()
        doric_layout.setColumnStretch(1, 1)
        doric_layout.setColumnStretch(3, 1)
        doric_layout.setColumnStretch(5, 1)
        
        row = 0
        
        # --- Smooth Signal Section ---
        smooth_label = QLabel("▶ Smooth Signal")
        smooth_label.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        doric_layout.addWidget(smooth_label, row, 0, 1, 6)
        row += 1
        
        doric_layout.addWidget(QLabel("Algorithm:"), row, 0)
        self.smooth_algorithm = QComboBox()
        self.smooth_algorithm.addItems(["Low-pass Butterworth", "Running Average", "None"])
        self.smooth_algorithm.setCurrentIndex(0)
        self.smooth_algorithm.setToolTip("Smoothing algorithm to reduce high-frequency noise")
        doric_layout.addWidget(self.smooth_algorithm, row, 1)
        
        doric_layout.addWidget(QLabel("Cutoff (Hz):"), row, 2)
        self.filter_cutoff = QDoubleSpinBox()
        self.filter_cutoff.setRange(0.1, 50.0)
        self.filter_cutoff.setValue(2.0)  # Doric default
        self.filter_cutoff.setSingleStep(0.5)
        self.filter_cutoff.setDecimals(1)
        self.filter_cutoff.setToolTip("Low-pass filter cutoff frequency (Hz). Doric default: 2 Hz")
        doric_layout.addWidget(self.filter_cutoff, row, 3)
        row += 1
        
        # --- Correct Baseline Section ---
        baseline_label = QLabel("▶ Correct Baseline (ALS)")
        baseline_label.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        doric_layout.addWidget(baseline_label, row, 0, 1, 6)
        row += 1
        
        doric_layout.addWidget(QLabel("Lambda (λ):"), row, 0)
        self.baseline_lambda = QDoubleSpinBox()
        self.baseline_lambda.setRange(1, 100000)
        self.baseline_lambda.setValue(10)  # Corrected Doric default
        self.baseline_lambda.setDecimals(0)
        self.baseline_lambda.setSingleStep(10)
        self.baseline_lambda.setToolTip("ALS baseline correction smoothness. Higher = smoother baseline. Doric default: 10")
        doric_layout.addWidget(self.baseline_lambda, row, 1)
        
        # Lambda presets (corrected values)
        doric_layout.addWidget(QLabel("Presets:"), row, 2)
        lambda_preset_layout = QHBoxLayout()
        for val, label in [(1, "1"), (10, "10"), (100, "100"), (1000, "1000")]:
            btn = QPushButton(label)
            btn.setMaximumWidth(50)
            btn.clicked.connect(lambda checked, v=val: self.baseline_lambda.setValue(v))
            lambda_preset_layout.addWidget(btn)
        doric_layout.addLayout(lambda_preset_layout, row, 3, 1, 3)
        row += 1
        
        # --- Discard Signal Section (Collapsible) ---
        self.chk_discard_signal = QCheckBox("▶ Discard Signal (Trim onset/offset)")
        self.chk_discard_signal.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        self.chk_discard_signal.setChecked(False)
        self.chk_discard_signal.toggled.connect(self._on_discard_toggled)
        doric_layout.addWidget(self.chk_discard_signal, row, 0, 1, 6)
        row += 1
        
        # Trim controls (initially hidden)
        self.lbl_onset = QLabel("Onset (sec):")
        doric_layout.addWidget(self.lbl_onset, row, 0)
        self.onset_trim = QDoubleSpinBox()
        self.onset_trim.setRange(0.0, 300.0)
        self.onset_trim.setValue(0.0)
        self.onset_trim.setSingleStep(1.0)
        self.onset_trim.setDecimals(1)
        self.onset_trim.setToolTip("Seconds to discard from start of recording")
        doric_layout.addWidget(self.onset_trim, row, 1)
        
        self.lbl_offset = QLabel("Offset (sec):")
        doric_layout.addWidget(self.lbl_offset, row, 2)
        self.offset_trim = QDoubleSpinBox()
        self.offset_trim.setRange(0.0, 300.0)
        self.offset_trim.setValue(0.0)
        self.offset_trim.setSingleStep(1.0)
        self.offset_trim.setDecimals(1)
        self.offset_trim.setToolTip("Seconds to discard from end of recording")
        doric_layout.addWidget(self.offset_trim, row, 3)
        row += 1
        
        # Hide trim controls initially
        self.lbl_onset.setVisible(False)
        self.onset_trim.setVisible(False)
        self.lbl_offset.setVisible(False)
        self.offset_trim.setVisible(False)
        
        # --- Fit Signals Section ---
        fit_label = QLabel("▶ Fit Signals (Robust Regression)")
        fit_label.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        doric_layout.addWidget(fit_label, row, 0, 1, 6)
        row += 1
        
        doric_layout.addWidget(QLabel("Max Threshold (σ):"), row, 0)
        self.fit_max_threshold = QDoubleSpinBox()
        self.fit_max_threshold.setRange(0.5, 10.0)
        self.fit_max_threshold.setValue(3.0)  # Doric default
        self.fit_max_threshold.setSingleStep(0.5)
        self.fit_max_threshold.setDecimals(1)
        self.fit_max_threshold.setToolTip("Outlier threshold for robust regression. Points > threshold*σ are treated as transients.")
        doric_layout.addWidget(self.fit_max_threshold, row, 1)
        
        self.doric_options_group.setLayout(doric_layout)
        self.doric_options_group.setVisible(False)  # Hidden by default (Traditional is selected)
        top_layout.addWidget(self.doric_options_group)
        
        # ====== Output Options (shared) ======
        output_group = QGroupBox("Output Options")
        output_layout = QHBoxLayout()
        
        self.chk_create_video = QCheckBox("Create combined video with ΔF/F overlay")
        self.chk_create_video.setChecked(True)
        output_layout.addWidget(self.chk_create_video)
        
        self.chk_show_raw_data = QCheckBox("Show raw traces (470nm, 415nm, IMU)")
        self.chk_show_raw_data.setChecked(False)
        self.chk_show_raw_data.setToolTip("Display raw signal traces in video. Makes trace panel taller (400px)")
        output_layout.addWidget(self.chk_show_raw_data)
        
        self.chk_overwrite = QCheckBox("Overwrite existing outputs")
        self.chk_overwrite.setChecked(False)
        output_layout.addWidget(self.chk_overwrite)
        
        output_group.setLayout(output_layout)
        top_layout.addWidget(output_group)
        
        splitter.addWidget(top_widget)
        
        # Bottom section: Log output
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 9))
        self.log_output.setMinimumHeight(200)
        log_layout.addWidget(self.log_output)
        
        log_group.setLayout(log_layout)
        splitter.addWidget(log_group)
        
        # Set splitter proportions
        splitter.setSizes([400, 300])
        main_layout.addWidget(splitter)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.btn_process = QPushButton("▶️ Process Files")
        self.btn_process.setMinimumHeight(40)
        self.btn_process.clicked.connect(self.start_processing)
        control_layout.addWidget(self.btn_process)
        
        self.btn_stop = QPushButton("⏹️ Stop")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_processing)
        control_layout.addWidget(self.btn_stop)
        
        main_layout.addLayout(control_layout)
        
        self.setLayout(main_layout)
    
    def apply_dark_theme(self):
        """Apply dark theme styling"""
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QGroupBox {
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3f3f3f;
                border: 1px solid #5f5f5f;
                border-radius: 4px;
                padding: 8px 16px;
                color: #cccccc;
            }
            QPushButton:hover {
                background-color: #4f4f4f;
            }
            QPushButton:pressed {
                background-color: #2f2f2f;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
            }
            QListWidget {
                background-color: #252526;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px;
            }
            QListWidget::item:selected {
                background-color: #094771;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #252526;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                padding: 4px;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QProgressBar {
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
            }
        """)
    
    def _on_pipeline_changed(self, index: int):
        """Handle pipeline selector change - show/hide appropriate options"""
        is_traditional = (index == 0)
        self.trad_options_group.setVisible(is_traditional)
        self.doric_options_group.setVisible(not is_traditional)
    
    def _on_discard_toggled(self, checked: bool):
        """Show/hide trim controls based on checkbox"""
        self.lbl_onset.setVisible(checked)
        self.onset_trim.setVisible(checked)
        self.lbl_offset.setVisible(checked)
        self.offset_trim.setVisible(checked)
    
    def preview_first_file(self):
        """Generate a preview plot of the first file's ΔF/F"""
        if not self.files_to_process:
            QMessageBox.warning(self, "No Files", "Please add .doric files first.")
            return
        
        first_file = self.files_to_process[0]
        self.log(f"\n{'='*60}")
        self.log(f"🔍 Generating preview for: {first_file.name}")
        self.log(f"{'='*60}")
        
        try:
            # Read file
            reader = DoricFileReader(first_file)
            file_data = reader.scan_file()
            
            if file_data.signal_channel is None:
                self.log("❌ No signal channel detected!")
                return
            
            # Report sampling rate
            srate = file_data.signal_channel.sampling_rate
            self.log(f"📊 Detected sampling rate: {srate:.1f} Hz")
            self.trad_srate_label.setText(f"Sampling Rate: {srate:.1f} Hz (auto-detected)")
            
            # Load data
            file_data = reader.load_data(file_data)
            
            if file_data.signal_data is None:
                self.log("❌ Failed to load signal data!")
                return
            
            # Calculate ΔF/F based on selected pipeline
            is_traditional = (self.pipeline_selector.currentIndex() == 0)
            
            if is_traditional:
                self.log(f"📈 Using Traditional (MATLAB-style) pipeline...")
                calc = TraditionalDFFCalculator(
                    apply_filter=self.chk_trad_filter.isChecked(),
                    filter_cutoff=self.trad_filter_cutoff.value(),
                    filter_order=self.trad_filter_order.value(),
                    clean_dropouts=self.chk_trad_dropouts.isChecked(),
                    dropout_threshold=self.trad_dropout_threshold.value()
                )
                
                if file_data.isosbestic_data is not None:
                    dff, filt_sig, filt_iso, fitted = calc.calculate(
                        file_data.signal_data,
                        file_data.isosbestic_data,
                        srate
                    )
                    if calc.last_dropout_count > 0:
                        self.log(f"   ⚠️ Cleaned {calc.last_dropout_count} signal dropouts")
                else:
                    dff, filt_sig = calc.calculate_simple(file_data.signal_data, srate)
                    if calc.last_dropout_count > 0:
                        self.log(f"   ⚠️ Cleaned {calc.last_dropout_count} signal dropouts")
                    filt_iso = None
                    fitted = None
            else:
                self.log(f"📈 Using Doric-Style pipeline...")
                algo_map = {
                    "Low-pass Butterworth": "butterworth",
                    "Running Average": "running_average",
                    "None": "none"
                }
                smooth_algo = algo_map.get(self.smooth_algorithm.currentText(), "butterworth")
                
                calc = DFFCalculator(
                    smooth_algorithm=smooth_algo,
                    filter_cutoff=self.filter_cutoff.value(),
                    baseline_lambda=self.baseline_lambda.value(),
                    onset_trim=self.onset_trim.value() if self.chk_discard_signal.isChecked() else 0,
                    offset_trim=self.offset_trim.value() if self.chk_discard_signal.isChecked() else 0,
                    fit_max_threshold=self.fit_max_threshold.value()
                )
                
                if file_data.isosbestic_data is not None:
                    dff, filt_sig, filt_iso, fitted, _ = calc.calculate(
                        file_data.signal_data,
                        file_data.isosbestic_data,
                        srate
                    )
                else:
                    dff, filt_sig = calc.calculate_simple(file_data.signal_data, srate)
                    filt_iso = None
                    fitted = None
            
            self.log(f"   ΔF/F range: [{np.min(dff):.2f}%, {np.max(dff):.2f}%]")
            self.log(f"   ΔF/F mean: {np.mean(dff):.2f}%, std: {np.std(dff):.2f}%")
            
            # Create preview plot
            self._show_preview_plot(file_data, dff, filt_sig, filt_iso, fitted, is_traditional)
            
        except Exception as e:
            self.log(f"❌ Preview error: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
    
    def _show_preview_plot(self, file_data, dff, filt_sig, filt_iso, fitted, is_traditional):
        """Display the preview plot in a popup window"""
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        
        # Create time vector
        n_samples = len(dff)
        srate = file_data.signal_channel.sampling_rate
        time = np.arange(n_samples) / srate
        
        # Determine number of subplots
        n_rows = 3 if filt_iso is not None else 2
        
        # Create figure
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 8), sharex=True)
        fig.patch.set_facecolor('#1e1e1e')
        
        pipeline_name = "Traditional (MATLAB)" if is_traditional else "Doric-Style"
        fig.suptitle(f"Preview: {file_data.filepath.name}\\n{pipeline_name} Pipeline", 
                     color='white', fontsize=12)
        
        # Plot 1: Raw/Filtered signals
        ax1 = axes[0]
        ax1.set_facecolor('#1e1e1e')
        ax1.plot(time, file_data.signal_data[:n_samples], color='#00ff00', alpha=0.3, 
                linewidth=0.5, label='Raw 470nm')
        ax1.plot(time, filt_sig, color='#00ff00', linewidth=1, label='Filtered 470nm')
        if filt_iso is not None:
            ax1.plot(time, file_data.isosbestic_data[:n_samples], color='#ff00ff', 
                    alpha=0.3, linewidth=0.5, label='Raw 415nm')
            ax1.plot(time, filt_iso, color='#ff00ff', linewidth=1, label='Filtered 415nm')
        ax1.set_ylabel('Signal (mV)', color='white')
        ax1.tick_params(colors='white')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_title('Photometry Signals', color='white', fontsize=10)
        for spine in ax1.spines.values():
            spine.set_color('#3f3f3f')
        
        # Plot 2: Fitted baseline (if available)
        if fitted is not None and filt_iso is not None:
            ax2 = axes[1]
            ax2.set_facecolor('#1e1e1e')
            ax2.plot(time, filt_sig, color='#00ff00', linewidth=1, label='Signal (470nm)')
            ax2.plot(time, fitted, color='#ff6600', linewidth=1, label='Fitted (from 415nm)')
            ax2.set_ylabel('Signal (mV)', color='white')
            ax2.tick_params(colors='white')
            ax2.legend(loc='upper right', fontsize=8)
            ax2.set_title('Isosbestic Fit', color='white', fontsize=10)
            for spine in ax2.spines.values():
                spine.set_color('#3f3f3f')
            ax_dff = axes[2]
        else:
            ax_dff = axes[1]
        
        # Plot 3: ΔF/F
        ax_dff.set_facecolor('#1e1e1e')
        ax_dff.plot(time, dff, color='#00ff00', linewidth=1)
        ax_dff.axhline(y=0, color='#666666', linestyle='--', linewidth=0.5)
        ax_dff.set_ylabel('ΔF/F (%)', color='white')
        ax_dff.set_xlabel('Time (s)', color='white')
        ax_dff.tick_params(colors='white')
        ax_dff.set_title('ΔF/F Result', color='white', fontsize=10)
        for spine in ax_dff.spines.values():
            spine.set_color('#3f3f3f')
        
        plt.tight_layout()
        
        # Show in a dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Preview: {file_data.filepath.name}")
        dialog.setMinimumSize(1000, 700)
        dialog.setStyleSheet("background-color: #1e1e1e;")
        
        layout = QVBoxLayout()
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        dialog.setLayout(layout)
        
        self.log("✅ Preview generated - displaying plot...")
        dialog.exec_()
        plt.close(fig)
    
    def log(self, message: str):
        """Add message to log output"""
        self.log_output.append(message)
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def add_folder(self):
        """Add all .doric files from a folder"""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder with .doric Files"
        )
        
        if not folder:
            return
        
        folder_path = Path(folder)
        pairs = find_doric_video_pairs(folder_path)
        
        added = 0
        for doric_file, video_folder in pairs:
            if doric_file not in self.files_to_process:
                self.files_to_process.append(doric_file)
                
                # Create list item with video info
                video_status = "✅ Video found" if video_folder else "⚠️ No video folder"
                item = QListWidgetItem(f"{doric_file.name} [{video_status}]")
                item.setData(Qt.UserRole, str(doric_file))
                self.file_list.addItem(item)
                added += 1
        
        self.log(f"📁 Added {added} .doric files from {folder_path.name}")
        self.update_button_states()
    
    def add_files(self):
        """Add individual .doric files"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select .doric Files", "",
            "Doric Files (*.doric);;All Files (*.*)"
        )
        
        if not files:
            return
        
        added = 0
        for file_path in files:
            path = Path(file_path)
            if path not in self.files_to_process:
                self.files_to_process.append(path)
                
                # Check for video folder
                video_folder = path.parent / f"{path.stem}_Videos"
                video_status = "✅ Video found" if video_folder.exists() else "⚠️ No video folder"
                
                item = QListWidgetItem(f"{path.name} [{video_status}]")
                item.setData(Qt.UserRole, str(path))
                self.file_list.addItem(item)
                added += 1
        
        self.log(f"📄 Added {added} .doric files")
        self.update_button_states()
    
    def remove_selected(self):
        """Remove selected files from the list"""
        selected_items = self.file_list.selectedItems()
        
        for item in selected_items:
            file_path = Path(item.data(Qt.UserRole))
            if file_path in self.files_to_process:
                self.files_to_process.remove(file_path)
            self.file_list.takeItem(self.file_list.row(item))
        
        self.update_button_states()
    
    def clear_all(self):
        """Clear all files from the list"""
        self.files_to_process.clear()
        self.file_list.clear()
        self.update_button_states()
    
    def update_button_states(self):
        """Update button enabled states based on file list"""
        has_files = len(self.files_to_process) > 0
        self.btn_process.setEnabled(has_files)
        self.btn_remove_selected.setEnabled(has_files)
        self.btn_clear_all.setEnabled(has_files)
    
    def start_processing(self):
        """Start batch processing"""
        if not self.files_to_process:
            QMessageBox.warning(self, "No Files", "Please add .doric files to process.")
            return
        
        # Check for existing outputs if not overwriting
        if not self.chk_overwrite.isChecked():
            existing = []
            for f in self.files_to_process:
                csv_path = f.parent / f"{f.stem}_dff.csv"
                if csv_path.exists():
                    existing.append(f.name)
            
            if existing:
                reply = QMessageBox.question(
                    self, "Existing Outputs",
                    f"{len(existing)} file(s) already have outputs.\n\n"
                    "Do you want to:\n"
                    "• Yes = Overwrite all\n"
                    "• No = Skip existing\n"
                    "• Cancel = Abort",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                
                if reply == QMessageBox.Cancel:
                    return
                elif reply == QMessageBox.Yes:
                    self.chk_overwrite.setChecked(True)
        
        # Disable UI
        self.btn_process.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_add_folder.setEnabled(False)
        self.btn_add_files.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(self.files_to_process))
        self.progress_bar.setValue(0)
        
        # Start worker with appropriate pipeline parameters
        is_traditional = (self.pipeline_selector.currentIndex() == 0)
        pipeline_type = 'traditional' if is_traditional else 'doric'
        
        # Map combo box to algorithm name for Doric pipeline
        algo_map = {
            "Low-pass Butterworth": "butterworth",
            "Running Average": "running_average",
            "None": "none"
        }
        smooth_algo = algo_map.get(self.smooth_algorithm.currentText(), "butterworth")
        
        self.worker = DoricProcessWorker(
            files=self.files_to_process.copy(),
            # Pipeline type
            pipeline_type=pipeline_type,
            # Traditional parameters
            trad_apply_filter=self.chk_trad_filter.isChecked(),
            trad_filter_cutoff=self.trad_filter_cutoff.value(),
            trad_filter_order=self.trad_filter_order.value(),
            trad_clean_dropouts=self.chk_trad_dropouts.isChecked(),
            trad_dropout_threshold=self.trad_dropout_threshold.value(),
            # Doric parameters
            smooth_algorithm=smooth_algo,
            filter_cutoff=self.filter_cutoff.value(),
            baseline_lambda=self.baseline_lambda.value(),
            onset_trim=self.onset_trim.value() if self.chk_discard_signal.isChecked() else 0.0,
            offset_trim=self.offset_trim.value() if self.chk_discard_signal.isChecked() else 0.0,
            fit_max_threshold=self.fit_max_threshold.value(),
            # Output options
            create_video=self.chk_create_video.isChecked(),
            show_raw_data=self.chk_show_raw_data.isChecked(),
            overwrite=self.chk_overwrite.isChecked()
        )
        
        self.worker.progress.connect(self.log)
        self.worker.file_started.connect(self.on_file_started)
        self.worker.file_finished.connect(self.on_file_finished)
        self.worker.all_finished.connect(self.on_processing_finished)
        
        pipeline_name = "Traditional (MATLAB-style)" if is_traditional else "Doric-Style"
        self.log("\n" + "="*60)
        self.log(f"🚀 Starting batch processing ({pipeline_name} pipeline)...")
        self.log(f"   Processing {len(self.files_to_process)} files...")
        self.log("="*60)
        
        self.worker.start()
    
    def stop_processing(self):
        """Stop processing"""
        if self.worker:
            self.worker.request_stop()
    
    def on_file_started(self, file_path: str):
        """Handle file processing started"""
        # Update progress bar
        current = self.progress_bar.value()
        self.progress_bar.setValue(current)
    
    def on_file_finished(self, file_path: str, success: bool, message: str):
        """Handle file processing finished"""
        # Update progress bar
        current = self.progress_bar.value()
        self.progress_bar.setValue(current + 1)
        
        # Update list item status
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.UserRole) == file_path:
                status = "✅ Done" if success else "⚠️ " + message[:20]
                item.setText(f"{Path(file_path).name} [{status}]")
                break
    
    def on_processing_finished(self, success: bool, message: str):
        """Handle all processing finished"""
        # Re-enable UI
        self.btn_process.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_add_folder.setEnabled(True)
        self.btn_add_files.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.worker = None
        
        if success:
            QMessageBox.information(self, "Complete", "Batch processing completed!")
        else:
            QMessageBox.warning(self, "Stopped", message)


def main():
    """Standalone entry point"""
    app = QApplication(sys.argv)
    window = DoricProcessorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
