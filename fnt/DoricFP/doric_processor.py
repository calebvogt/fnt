#!/usr/bin/env python3
"""
Doric WiFP Fiber Photometry Processor

This module provides classes for:
- Reading .doric (HDF5) files from Doric wireless fiber photometry systems
- Calculating ΔF/F using isosbestic correction
- Synchronizing photometry data with behavior video frames
- Exporting aligned CSV and combined video outputs

Based on analysis of NC500 WiFP system data structure.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import h5py
from scipy.signal import firwin, filtfilt
from scipy.optimize import nnls


@dataclass
class DoricChannelInfo:
    """Information about a detected photometry channel."""
    path: str
    time_path: str
    wavelength: Optional[int] = None
    name: str = ""
    sampling_rate: float = 0.0
    n_samples: int = 0


@dataclass  
class DoricVideoInfo:
    """Information about behavior video in .doric file."""
    time_path: str
    relative_path: str
    n_frames: int = 0
    fps: float = 0.0
    absolute_path: Optional[str] = None


@dataclass
class DoricFileData:
    """Container for all extracted data from a .doric file."""
    filepath: Path
    signal_channel: Optional[DoricChannelInfo] = None
    isosbestic_channel: Optional[DoricChannelInfo] = None
    video_info: Optional[DoricVideoInfo] = None
    all_channels: List[DoricChannelInfo] = field(default_factory=list)
    
    # Raw data arrays (loaded on demand)
    signal_data: Optional[np.ndarray] = None
    signal_time: Optional[np.ndarray] = None
    isosbestic_data: Optional[np.ndarray] = None
    isosbestic_time: Optional[np.ndarray] = None
    video_timestamps: Optional[np.ndarray] = None
    
    # IMU data (accelerometer)
    imu_accel_x: Optional[np.ndarray] = None
    imu_accel_y: Optional[np.ndarray] = None
    imu_accel_z: Optional[np.ndarray] = None
    imu_time: Optional[np.ndarray] = None


class DoricFileReader:
    """
    Reader for Doric .doric (HDF5) files from WiFP systems.
    
    Handles auto-detection of photometry channels based on LED wavelength
    configuration and extracts video synchronization information.
    """
    
    # Known paths in Doric file structure
    DATA_ROOT = "DataAcquisition"
    CONFIG_ROOT = "Configurations"
    
    def __init__(self, filepath: str | Path):
        """
        Initialize reader with path to .doric file.
        
        Args:
            filepath: Path to the .doric file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        if not self.filepath.suffix.lower() == '.doric':
            raise ValueError(f"Expected .doric file, got: {self.filepath.suffix}")
    
    def scan_file(self) -> DoricFileData:
        """
        Scan the .doric file and detect all available channels and video info.
        
        Returns:
            DoricFileData with detected channels and video information
        """
        data = DoricFileData(filepath=self.filepath)
        
        with h5py.File(self.filepath, 'r') as f:
            # Find all LockIn channels (demodulated photometry signals)
            data.all_channels = self._find_lockin_channels(f)
            
            # Auto-detect signal vs isosbestic based on wavelength
            signal_ch, iso_ch = self._auto_detect_channels(f, data.all_channels)
            data.signal_channel = signal_ch
            data.isosbestic_channel = iso_ch
            
            # Find video information
            data.video_info = self._find_video_info(f)
            
            # Resolve video absolute path
            if data.video_info:
                data.video_info.absolute_path = self._resolve_video_path(
                    data.video_info.relative_path
                )
        
        return data
    
    def load_data(self, data: DoricFileData) -> DoricFileData:
        """
        Load actual data arrays for the detected channels.
        
        Args:
            data: DoricFileData from scan_file()
            
        Returns:
            Same DoricFileData with data arrays populated
        """
        with h5py.File(self.filepath, 'r') as f:
            # Load signal channel
            if data.signal_channel:
                data.signal_data = f[data.signal_channel.path][:]
                data.signal_time = f[data.signal_channel.time_path][:]
            
            # Load isosbestic channel
            if data.isosbestic_channel:
                data.isosbestic_data = f[data.isosbestic_channel.path][:]
                data.isosbestic_time = f[data.isosbestic_channel.time_path][:]
            
            # Load video timestamps
            if data.video_info:
                data.video_timestamps = f[data.video_info.time_path][:]
            
            # Load IMU data (accelerometer)
            data = self._load_imu_data(f, data)
        
        return data
    
    def _load_imu_data(self, f: h5py.File, data: DoricFileData) -> DoricFileData:
        """Load IMU accelerometer data if available."""
        # Search for IMU datasets
        imu_paths = []
        
        def find_imu(name: str, obj):
            if 'IMU' in name and isinstance(obj, h5py.Group):
                imu_paths.append(name)
        
        f.visititems(find_imu)
        
        # Look for the main headstage IMU (not Headstage256)
        for imu_path in imu_paths:
            if 'Headstage01IMU' in imu_path and 'Headstage256' not in imu_path:
                try:
                    accel_x_path = f"{imu_path}/AccelerometerX"
                    accel_y_path = f"{imu_path}/AccelerometerY"
                    accel_z_path = f"{imu_path}/AccelerometerZ"
                    time_path = f"{imu_path}/Time"
                    
                    if all(p in f for p in [accel_x_path, accel_y_path, accel_z_path, time_path]):
                        data.imu_accel_x = f[accel_x_path][:]
                        data.imu_accel_y = f[accel_y_path][:]
                        data.imu_accel_z = f[accel_z_path][:]
                        data.imu_time = f[time_path][:]
                        break
                except Exception:
                    pass
        
        return data
    
    def _find_lockin_channels(self, f: h5py.File) -> List[DoricChannelInfo]:
        """Find all LockIn channels in the file."""
        channels = []
        
        def search(name: str, obj):
            # Look for LockIn data groups
            if 'LockIn' in name and isinstance(obj, h5py.Group):
                # Find the data dataset (same name as group)
                group_name = name.split('/')[-1]
                data_path = f"{name}/{group_name}"
                time_path = f"{name}/Time"
                
                if data_path in f and time_path in f:
                    data_ds = f[data_path]
                    time_ds = f[time_path]
                    
                    # Skip empty datasets
                    if data_ds.size == 0:
                        return
                    
                    # Calculate sampling rate
                    if time_ds.size >= 2:
                        time_arr = time_ds[:]
                        dt = np.mean(np.diff(time_arr))
                        sr = 1.0 / dt if dt > 0 else 0.0
                    else:
                        sr = 0.0
                    
                    channels.append(DoricChannelInfo(
                        path=data_path,
                        time_path=time_path,
                        name=group_name,
                        sampling_rate=sr,
                        n_samples=data_ds.size
                    ))
        
        f.visititems(search)
        return channels
    
    def _auto_detect_channels(
        self, 
        f: h5py.File, 
        channels: List[DoricChannelInfo]
    ) -> Tuple[Optional[DoricChannelInfo], Optional[DoricChannelInfo]]:
        """
        Auto-detect signal (470nm) and isosbestic (405/415nm) channels.
        
        Uses LED wavelength configuration from file attributes.
        """
        signal_ch = None
        iso_ch = None
        
        # Try to read wavelength configuration
        wavelengths = self._read_led_wavelengths(f)
        
        for ch in channels:
            # Extract LockIn number from name (e.g., "Headstage01LockIn01" -> "01")
            match = re.search(r'LockIn(\d+)', ch.name)
            if not match:
                continue
            
            lockin_num = match.group(1)
            
            # Check if we have wavelength info for this LED
            if lockin_num in wavelengths:
                wl = wavelengths[lockin_num]
                ch.wavelength = wl
                
                # 465-475nm range = signal (GCaMP/dLight excitation)
                if 465 <= wl <= 475:
                    signal_ch = ch
                # 405-420nm range = isosbestic
                elif 400 <= wl <= 420:
                    iso_ch = ch
        
        # Fallback: assume LockIn01 = signal, LockIn02 = isosbestic
        if signal_ch is None and iso_ch is None:
            for ch in channels:
                if 'LockIn01' in ch.name and signal_ch is None:
                    signal_ch = ch
                elif 'LockIn02' in ch.name and iso_ch is None:
                    iso_ch = ch
        
        return signal_ch, iso_ch
    
    def _read_led_wavelengths(self, f: h5py.File) -> Dict[str, int]:
        """Read LED wavelength configurations from file attributes."""
        wavelengths = {}
        
        def search(name: str, obj):
            if isinstance(obj, h5py.Group) and 'LED' in name:
                # Extract LED number
                match = re.search(r'LED(\d+)', name)
                if match and 'WaveLength' in obj.attrs:
                    led_num = match.group(1)
                    wl = int(obj.attrs['WaveLength'])
                    # LED1 -> LockIn01, LED2 -> LockIn02
                    lockin_num = f"{int(led_num):02d}"
                    wavelengths[lockin_num] = wl
        
        f.visititems(search)
        return wavelengths
    
    def _find_video_info(self, f: h5py.File) -> Optional[DoricVideoInfo]:
        """Find behavior video information in the file."""
        video_info = None
        
        def search(name: str, obj):
            nonlocal video_info
            if video_info is not None:
                return  # Already found
            
            if isinstance(obj, h5py.Group) and 'Video' in name and 'Series' in name:
                # Look for camera subgroups
                for cam_name in obj.keys():
                    cam_group = obj[cam_name]
                    if isinstance(cam_group, h5py.Group):
                        # Check for Time dataset and video path attribute
                        if 'Time' in cam_group and 'RelativeFilePath' in cam_group.attrs:
                            time_path = f"{name}/{cam_name}/Time"
                            rel_path = cam_group.attrs['RelativeFilePath']
                            if isinstance(rel_path, bytes):
                                rel_path = rel_path.decode('utf-8')
                            
                            time_ds = f[time_path]
                            n_frames = time_ds.size
                            
                            # Calculate FPS
                            if n_frames >= 2:
                                time_arr = time_ds[:]
                                dt = np.mean(np.diff(time_arr))
                                fps = 1.0 / dt if dt > 0 else 0.0
                            else:
                                fps = 0.0
                            
                            video_info = DoricVideoInfo(
                                time_path=time_path,
                                relative_path=rel_path,
                                n_frames=n_frames,
                                fps=fps
                            )
                            return
        
        f.visititems(search)
        return video_info
    
    def _resolve_video_path(self, relative_path: str) -> Optional[str]:
        """
        Resolve the video file's absolute path.
        
        The relative path is relative to the .doric file's directory.
        """
        # Remove leading slash if present
        rel_path = relative_path.lstrip('/')
        
        # Construct absolute path relative to .doric file
        video_path = self.filepath.parent / rel_path
        
        if video_path.exists():
            return str(video_path)
        
        # Try looking in a Videos folder next to the .doric file
        basename = self.filepath.stem
        videos_folder = self.filepath.parent / f"{basename}_Videos"
        if videos_folder.exists():
            # Look for any .mp4 file
            mp4_files = list(videos_folder.glob("*.mp4"))
            if mp4_files:
                return str(mp4_files[0])
        
        return None


class DFFCalculator:
    """
    Calculate ΔF/F from fiber photometry signals using isosbestic correction.
    
    Implements Doric Danse-style processing pipeline:
    1. Smooth Signal - Low-pass Butterworth filter or running average
    2. Correct Baseline - Asymmetric Least Squares (ALS) baseline correction
    3. Discard Signal Onset/Offset - Trim beginning and end of recording
    4. Fit Signals - Robust regression with outlier detection (calcium transients are outliers)
    """
    
    # Smoothing algorithm options
    SMOOTH_NONE = 'none'
    SMOOTH_BUTTERWORTH = 'butterworth'
    SMOOTH_RUNNING_AVG = 'running_average'
    
    def __init__(
        self,
        smooth_algorithm: str = 'butterworth',
        filter_cutoff: float = 2.0,  # Hz - Doric default is 2 Hz
        running_avg_window: float = 0.5,  # seconds
        baseline_lambda: float = 10.0,  # ALS lambda parameter
        onset_trim: float = 0.0,  # seconds to trim from start
        offset_trim: float = 0.0,  # seconds to trim from end
        fit_max_threshold: float = 1.0,  # RANSAC inlier threshold multiplier
        output_as_percentage: bool = True
    ):
        """
        Initialize the ΔF/F calculator with Doric-style parameters.
        
        Args:
            smooth_algorithm: 'none', 'butterworth', or 'running_average'
            filter_cutoff: Butterworth filter cutoff frequency in Hz
            running_avg_window: Running average window in seconds
            baseline_lambda: ALS baseline correction lambda (higher = smoother)
            onset_trim: Seconds to discard from start of recording
            offset_trim: Seconds to discard from end of recording  
            fit_max_threshold: Threshold for robust regression outlier detection
            output_as_percentage: If True, multiply ΔF/F by 100
        """
        self.smooth_algorithm = smooth_algorithm
        self.filter_cutoff = filter_cutoff
        self.running_avg_window = running_avg_window
        self.baseline_lambda = baseline_lambda
        self.onset_trim = onset_trim
        self.offset_trim = offset_trim
        self.fit_max_threshold = fit_max_threshold
        self.output_as_percentage = output_as_percentage
    
    def calculate(
        self,
        signal: np.ndarray,
        isosbestic: np.ndarray,
        sampling_rate: float,
        time_vector: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ΔF/F with isosbestic correction using Doric-style pipeline.
        
        Args:
            signal: Raw signal trace (470nm / calcium-dependent)
            isosbestic: Raw isosbestic trace (405/415nm / calcium-independent)
            sampling_rate: Sampling rate in Hz
            time_vector: Optional time vector for trimming
            
        Returns:
            Tuple of (dff, processed_signal, processed_isosbestic, fitted_baseline, valid_mask)
        """
        # Ensure same length
        min_len = min(len(signal), len(isosbestic))
        signal = signal[:min_len].copy()
        isosbestic = isosbestic[:min_len].copy()
        
        if time_vector is None:
            time_vector = np.arange(min_len) / sampling_rate
        else:
            time_vector = time_vector[:min_len].copy()
        
        # Create valid mask (for trimming)
        valid_mask = np.ones(min_len, dtype=bool)
        
        # Step 3: Discard onset/offset (do this first to inform other steps)
        if self.onset_trim > 0:
            valid_mask[time_vector < self.onset_trim] = False
        if self.offset_trim > 0:
            max_time = time_vector[-1]
            valid_mask[time_vector > (max_time - self.offset_trim)] = False
        
        # Step 1: Smooth Signal
        proc_signal = self._smooth_signal(signal, sampling_rate)
        proc_iso = self._smooth_signal(isosbestic, sampling_rate)
        
        # Step 2: Correct Baseline (ALS)
        proc_signal = self._correct_baseline(proc_signal, sampling_rate)
        proc_iso = self._correct_baseline(proc_iso, sampling_rate)
        
        # Step 4: Fit Signals with robust regression
        fitted = self._robust_fit(proc_iso, proc_signal, valid_mask)
        
        # Calculate ΔF/F = (signal - fitted) / fitted
        # Using fitted as F0 is more standard for fiber photometry
        with np.errstate(divide='ignore', invalid='ignore'):
            dff = (proc_signal - fitted) / np.abs(fitted)
            dff = np.nan_to_num(dff, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.output_as_percentage:
            dff = dff * 100
        
        return dff, proc_signal, proc_iso, fitted, valid_mask
    
    def _smooth_signal(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Apply smoothing algorithm to signal."""
        if self.smooth_algorithm == self.SMOOTH_NONE:
            return signal.copy()
        
        elif self.smooth_algorithm == self.SMOOTH_BUTTERWORTH:
            from scipy.signal import butter, filtfilt
            
            nyquist = sampling_rate / 2
            normalized_cutoff = self.filter_cutoff / nyquist
            
            # Ensure cutoff is valid
            if normalized_cutoff >= 1.0:
                normalized_cutoff = 0.99
            if normalized_cutoff <= 0:
                normalized_cutoff = 0.01
            
            # 2nd order Butterworth (Doric uses this)
            b, a = butter(2, normalized_cutoff, btype='low')
            
            # Pad signal to avoid edge effects
            pad_len = min(3 * max(len(a), len(b)), len(signal) - 1)
            if pad_len > 0:
                return filtfilt(b, a, signal, padlen=pad_len)
            else:
                return filtfilt(b, a, signal)
        
        elif self.smooth_algorithm == self.SMOOTH_RUNNING_AVG:
            # Running average with specified window
            window_samples = int(self.running_avg_window * sampling_rate)
            if window_samples < 1:
                window_samples = 1
            if window_samples > len(signal):
                window_samples = len(signal)
            
            # Use uniform filter (running average)
            from scipy.ndimage import uniform_filter1d
            return uniform_filter1d(signal, size=window_samples, mode='nearest')
        
        else:
            return signal.copy()
    
    def _correct_baseline(self, signal: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        Apply Asymmetric Least Squares (ALS) baseline correction.
        
        This removes slow drift/slope while preserving peaks and transients.
        """
        if self.baseline_lambda <= 0:
            return signal
        
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        
        # ALS parameters - use lambda directly as provided by user
        lam = self.baseline_lambda
        p = 0.01  # Asymmetry parameter (small = baseline follows lower envelope)
        n_iter = 10
        
        L = len(signal)
        
        # Second derivative matrix
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
        D = D.T @ D
        
        w = np.ones(L)
        
        for _ in range(n_iter):
            W = sparse.diags(w, 0, shape=(L, L))
            Z = W + lam * D
            baseline = spsolve(Z, w * signal)
            w = p * (signal > baseline) + (1 - p) * (signal <= baseline)
        
        return signal - baseline
    
    def _robust_fit(
        self, 
        isosbestic: np.ndarray, 
        signal: np.ndarray, 
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Fit isosbestic to signal using robust regression.
        
        Calcium transients are treated as outliers (they don't follow the motion artifact pattern).
        Uses RANSAC-like approach with configurable threshold.
        """
        # Use only valid (non-trimmed) data for fitting
        iso_valid = isosbestic[valid_mask]
        sig_valid = signal[valid_mask]
        
        if len(iso_valid) < 10:
            # Not enough data, use simple fit
            return self._simple_fit(isosbestic, signal)
        
        # Initial linear fit
        try:
            coeffs = np.polyfit(iso_valid, sig_valid, 1)
        except:
            return self._simple_fit(isosbestic, signal)
        
        # Calculate residuals
        predicted = np.polyval(coeffs, iso_valid)
        residuals = np.abs(sig_valid - predicted)
        
        # Determine inlier threshold based on MAD (median absolute deviation)
        mad = np.median(residuals)
        threshold = self.fit_max_threshold * mad * 1.4826  # Scale to approximate std
        
        # Identify inliers (motion artifacts follow the fit line)
        inliers = residuals < threshold
        
        if np.sum(inliers) < 10:
            # Not enough inliers, use all data
            fitted = np.polyval(coeffs, isosbestic)
        else:
            # Refit using only inliers
            try:
                coeffs_robust = np.polyfit(iso_valid[inliers], sig_valid[inliers], 1)
                
                # Handle negative slope with NNLS
                if coeffs_robust[0] <= 0:
                    A = np.column_stack([iso_valid[inliers], np.ones(np.sum(inliers))])
                    x, _ = nnls(A, sig_valid[inliers])
                    fitted = x[0] * isosbestic + x[1]
                else:
                    fitted = np.polyval(coeffs_robust, isosbestic)
            except:
                fitted = np.polyval(coeffs, isosbestic)
        
        return fitted
    
    def _simple_fit(self, isosbestic: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Simple linear fit fallback."""
        try:
            coeffs = np.polyfit(isosbestic, signal, 1)
            if coeffs[0] <= 0:
                A = np.column_stack([isosbestic, np.ones_like(isosbestic)])
                x, _ = nnls(A, signal)
                return x[0] * isosbestic + x[1]
            return np.polyval(coeffs, isosbestic)
        except:
            return np.mean(signal) * np.ones_like(signal)
    
    def calculate_simple(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        baseline_percentile: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ΔF/F without isosbestic correction (simple baseline method).
        
        Uses a percentile-based baseline estimation.
        
        Args:
            signal: Raw signal trace
            sampling_rate: Sampling rate in Hz
            baseline_percentile: Percentile for baseline estimation
            
        Returns:
            Tuple of (dff, smoothed_signal)
        """
        # Apply smoothing
        proc_signal = self._smooth_signal(signal, sampling_rate)
        
        # Apply baseline correction
        proc_signal = self._correct_baseline(proc_signal, sampling_rate)
        
        # Estimate baseline using percentile
        baseline = np.percentile(proc_signal, baseline_percentile)
        
        if baseline == 0:
            baseline = 1e-10
        
        dff = (proc_signal - baseline) / np.abs(baseline)
        
        if self.output_as_percentage:
            dff = dff * 100
        
        return dff, proc_signal


class TraditionalDFFCalculator:
    """
    Calculate ΔF/F using the traditional MATLAB approach.
    
    This replicates the pipeline from fitcaldff.m / filteredfitcaldff.m:
    1. Optional Finite Impulse Response (FIR) low-pass filter (20 Hz cutoff, order 100)
    2. Linear regression: polyfit(iso, signal, 1) to fit isosbestic to signal
    3. Fallback to NNLS if slope <= 0 (ensures positive scaling)
    4. ΔF/F = 100 * (signal - fitted) / mean(signal)
    
    The key insight: isosbestic (405/415nm) is used to PREDICT the signal (470nm),
    and anything that deviates from this prediction is calcium-related activity.
    """
    
    def __init__(
        self,
        apply_filter: bool = True,
        filter_cutoff: float = 20.0,  # Hz - MATLAB default
        filter_order: int = 100,  # Finite Impulse Response filter order
        sampling_rate_override: Optional[float] = None,  # If None, use detected rate
        clean_dropouts: bool = True,  # Clean wireless signal dropouts
        dropout_threshold: float = 0.5  # Values below median * threshold are dropouts
    ):
        """
        Initialize the traditional ΔF/F calculator.
        
        Args:
            apply_filter: Whether to apply low-pass Finite Impulse Response filter
            filter_cutoff: Filter cutoff frequency in Hz (MATLAB default: 20)
            filter_order: Finite Impulse Response filter order (MATLAB default: 100)
            sampling_rate_override: Override detected sampling rate if specified
            clean_dropouts: Whether to detect and interpolate wireless signal dropouts
            dropout_threshold: Values below median * this factor are considered dropouts
        """
        self.apply_filter = apply_filter
        self.filter_cutoff = filter_cutoff
        self.filter_order = filter_order
        self.sampling_rate_override = sampling_rate_override
        self.clean_dropouts = clean_dropouts
        self.dropout_threshold = dropout_threshold
        self.last_dropout_count = 0  # Track dropouts for reporting
    
    def calculate(
        self,
        signal: np.ndarray,
        isosbestic: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ΔF/F using traditional MATLAB approach.
        
        Replicates filteredfitcaldff.m exactly:
        - FIR low-pass filter at 20 Hz
        - polyfit for linear regression, lsqnonneg fallback
        - ΔF/F = 100 * (filt473 - yfit1) / mean(filt473)
        
        Args:
            signal: Raw signal trace (470nm / calcium-dependent)
            isosbestic: Raw isosbestic trace (405/415nm / calcium-independent)
            sampling_rate: Sampling rate in Hz (auto-detected from file)
            
        Returns:
            Tuple of (dff, filtered_signal, filtered_isosbestic, fitted_baseline)
        """
        # Use override if provided
        srate = self.sampling_rate_override if self.sampling_rate_override else sampling_rate
        
        # Ensure same length
        min_len = min(len(signal), len(isosbestic))
        trace473 = signal[:min_len].copy()
        trace405 = isosbestic[:min_len].copy()
        
        # Step 0: Clean wireless dropouts (if enabled)
        if self.clean_dropouts:
            trace473, dropout_mask_473 = self._clean_signal_dropouts(trace473)
            trace405, dropout_mask_405 = self._clean_signal_dropouts(trace405)
            # Track total dropouts (either channel)
            combined_mask = dropout_mask_473 | dropout_mask_405
            self.last_dropout_count = int(np.sum(combined_mask))
        else:
            self.last_dropout_count = 0
        
        # Step 1: Apply FIR low-pass filter (if enabled)
        if self.apply_filter:
            filt473, filt405 = self._apply_fir_filter(trace473, trace405, srate)
        else:
            filt473 = trace473
            filt405 = trace405
        
        # Step 2: Linear regression - fit isosbestic to signal
        # MATLAB: bls1 = polyfit(trace405, trace473, 1)
        bls1 = np.polyfit(filt405, filt473, 1)  # Returns [slope, intercept]
        
        # Step 3: Non-negative least squares fallback
        # MATLAB: x2 = lsqnonneg(trace405, trace473)
        # In MATLAB, lsqnonneg with vector inputs returns a scalar multiplier
        x2, _ = nnls(filt405.reshape(-1, 1), filt473)
        x2 = x2[0]  # Extract scalar
        
        # Step 4: Choose fit based on slope
        # MATLAB: if bls1(1,1) <= 0, use lsqnonneg result
        if bls1[0] <= 0:
            # Negative or zero slope - use non-negative scaling
            yfit1 = x2 * filt405
        else:
            # Positive slope - use linear fit
            # MATLAB: yfit1 = polyval(bls1, trace405)
            yfit1 = np.polyval(bls1, filt405)
        
        # Step 5: Calculate ΔF/F
        # MATLAB: dff = 100 * (filt473 - yfit1) / mean(filt473)
        mean_signal = np.mean(filt473)
        if mean_signal == 0:
            mean_signal = 1e-10
        
        dff = 100 * (filt473 - yfit1) / mean_signal
        
        return dff, filt473, filt405, yfit1
    
    def _apply_fir_filter(
        self,
        signal: np.ndarray,
        isosbestic: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply FIR low-pass filter to both channels.
        
        Replicates MATLAB:
            Nyquist = sRate / 2;
            b = fir1(order, cutoff/Nyquist, 'low');
            filt = filtfilt(b, 1, trace);
        """
        nyquist = sampling_rate / 2
        normalized_cutoff = self.filter_cutoff / nyquist
        
        # Ensure valid cutoff
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.99
        if normalized_cutoff <= 0:
            normalized_cutoff = 0.01
        
        # Design FIR filter
        # MATLAB fir1(order, Wn) creates order+1 taps
        # scipy.signal.firwin(numtaps, cutoff) where numtaps = order + 1
        numtaps = self.filter_order + 1
        b = firwin(numtaps, normalized_cutoff)
        
        # Apply zero-phase filtering (same as MATLAB filtfilt)
        # Pad to avoid edge effects
        pad_len = min(3 * numtaps, len(signal) - 1)
        
        filt_signal = filtfilt(b, [1.0], signal, padlen=pad_len)
        filt_isosbestic = filtfilt(b, [1.0], isosbestic, padlen=pad_len)
        
        return filt_signal, filt_isosbestic
    
    def _clean_signal_dropouts(
        self,
        signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and interpolate wireless signal dropouts.
        
        Dropouts are identified as values below threshold_factor * median(signal).
        These are typically caused by wireless transmission issues in the WiFP system.
        
        Args:
            signal: Input signal array
            
        Returns:
            cleaned_signal: Signal with dropouts interpolated
            dropout_mask: Boolean array where True = dropout was detected
        """
        median_val = np.median(signal)
        threshold = median_val * self.dropout_threshold
        
        # Detect dropouts - values significantly below median
        dropout_mask = signal < threshold
        n_dropouts = int(np.sum(dropout_mask))
        
        if n_dropouts == 0:
            return signal.copy(), dropout_mask
        
        cleaned = signal.copy()
        
        # Linear interpolation across dropout regions
        good_indices = np.where(~dropout_mask)[0]
        bad_indices = np.where(dropout_mask)[0]
        
        if len(good_indices) > 1:
            cleaned[bad_indices] = np.interp(
                bad_indices, 
                good_indices, 
                signal[good_indices]
            )
        elif len(good_indices) == 1:
            # Only one good value - fill with that
            cleaned[bad_indices] = signal[good_indices[0]]
        # else: all dropouts - leave as is (shouldn't happen in real data)
        
        return cleaned, dropout_mask

    def calculate_simple(
        self,
        signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate ΔF/F without isosbestic (simple baseline method).
        
        Uses mean of signal as baseline.
        
        Args:
            signal: Raw signal trace
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (dff, filtered_signal)
        """
        srate = self.sampling_rate_override if self.sampling_rate_override else sampling_rate
        
        trace = signal.copy()
        
        # Clean dropouts if enabled
        if self.clean_dropouts:
            trace, dropout_mask = self._clean_signal_dropouts(trace)
            self.last_dropout_count = int(np.sum(dropout_mask))
        else:
            self.last_dropout_count = 0
        
        if self.apply_filter:
            nyquist = srate / 2
            normalized_cutoff = min(self.filter_cutoff / nyquist, 0.99)
            numtaps = self.filter_order + 1
            b = firwin(numtaps, normalized_cutoff)
            pad_len = min(3 * numtaps, len(trace) - 1)
            filt_signal = filtfilt(b, [1.0], trace, padlen=pad_len)
        else:
            filt_signal = trace
        
        mean_signal = np.mean(filt_signal)
        if mean_signal == 0:
            mean_signal = 1e-10
        
        dff = 100 * (filt_signal - mean_signal) / mean_signal
        
        return dff, filt_signal


class VideoSynchronizer:
    """
    Synchronize photometry data with behavior video frames.
    
    Uses video timestamps from .doric file to average photometry
    data within each video frame's time window.
    """
    
    def __init__(self, method: str = 'average'):
        """
        Initialize synchronizer.
        
        Args:
            method: Aggregation method ('average', 'max', 'min', 'median')
        """
        self.method = method
        self._agg_funcs = {
            'average': np.mean,
            'max': np.max,
            'min': np.min,
            'median': np.median
        }
        if method not in self._agg_funcs:
            raise ValueError(f"Unknown method: {method}. Use one of {list(self._agg_funcs.keys())}")
    
    def synchronize(
        self,
        photometry_data: np.ndarray,
        photometry_time: np.ndarray,
        video_timestamps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Synchronize photometry data to video frame timestamps.
        
        Args:
            photometry_data: Photometry signal (e.g., ΔF/F)
            photometry_time: Timestamps for photometry data
            video_timestamps: Timestamps for each video frame
            
        Returns:
            Tuple of (frame_aligned_data, frame_numbers)
        """
        n_frames = len(video_timestamps)
        frame_data = np.zeros(n_frames)
        frame_numbers = np.arange(1, n_frames + 1)  # 1-indexed frame numbers
        
        agg_func = self._agg_funcs[self.method]
        
        for i in range(n_frames):
            # Determine time window for this frame
            t_start = video_timestamps[i]
            if i < n_frames - 1:
                t_end = video_timestamps[i + 1]
            else:
                # Last frame: use same duration as previous
                if i > 0:
                    t_end = t_start + (video_timestamps[i] - video_timestamps[i - 1])
                else:
                    t_end = t_start + 0.033  # Assume ~30fps
            
            # Find photometry samples in this window
            mask = (photometry_time >= t_start) & (photometry_time < t_end)
            
            if np.any(mask):
                frame_data[i] = agg_func(photometry_data[mask])
            else:
                # No samples in window, use nearest
                idx = np.argmin(np.abs(photometry_time - t_start))
                frame_data[i] = photometry_data[idx]
        
        return frame_data, frame_numbers
    
    def create_aligned_dataframe(
        self,
        frame_data: np.ndarray,
        frame_numbers: np.ndarray,
        video_timestamps: np.ndarray
    ) -> pd.DataFrame:
        """
        Create a DataFrame with aligned photometry and video data.
        
        Args:
            frame_data: Frame-aligned photometry data
            frame_numbers: Frame numbers (1-indexed)
            video_timestamps: Video frame timestamps
            
        Returns:
            DataFrame with columns: Frame, Time_sec, DeltaF_F
        """
        return pd.DataFrame({
            'Frame': frame_numbers.astype(int),
            'Time_sec': video_timestamps,
            'DeltaF_F': frame_data
        })


def find_doric_video_pairs(folder: str | Path) -> List[Tuple[Path, Optional[Path]]]:
    """
    Find all .doric files and their associated video folders in a directory.
    
    Args:
        folder: Directory to search
        
    Returns:
        List of (doric_path, video_folder_path) tuples
    """
    folder = Path(folder)
    pairs = []
    
    for doric_file in folder.glob("*.doric"):
        # Look for matching _Videos folder
        videos_folder = folder / f"{doric_file.stem}_Videos"
        
        if videos_folder.exists() and videos_folder.is_dir():
            pairs.append((doric_file, videos_folder))
        else:
            pairs.append((doric_file, None))
    
    return pairs


def process_doric_file(
    doric_path: str | Path,
    output_dir: Optional[str | Path] = None,
    filter_cutoff: float = 20.0,
    signal_channel_path: Optional[str] = None,
    isosbestic_channel_path: Optional[str] = None,
    create_video: bool = True,
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Process a single .doric file and generate outputs.
    
    Args:
        doric_path: Path to .doric file
        output_dir: Output directory (default: same as .doric file)
        filter_cutoff: Low-pass filter cutoff in Hz
        signal_channel_path: Override auto-detected signal channel
        isosbestic_channel_path: Override auto-detected isosbestic channel
        create_video: Whether to create combined video output
        overwrite: Whether to overwrite existing outputs
        
    Returns:
        Dict with processing results and output paths
    """
    doric_path = Path(doric_path)
    
    if output_dir is None:
        output_dir = doric_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file paths
    csv_path = output_dir / f"{doric_path.stem}_dff.csv"
    video_path = output_dir / f"{doric_path.stem}_combined.mp4"
    
    # Check for existing outputs
    if not overwrite:
        if csv_path.exists():
            raise FileExistsError(f"Output already exists: {csv_path}")
        if create_video and video_path.exists():
            raise FileExistsError(f"Output already exists: {video_path}")
    
    results = {
        'doric_file': str(doric_path),
        'csv_output': None,
        'video_output': None,
        'success': False,
        'error': None
    }
    
    try:
        # Read .doric file
        reader = DoricFileReader(doric_path)
        file_data = reader.scan_file()
        
        # Override channels if specified
        if signal_channel_path:
            for ch in file_data.all_channels:
                if ch.path == signal_channel_path:
                    file_data.signal_channel = ch
                    break
        
        if isosbestic_channel_path:
            for ch in file_data.all_channels:
                if ch.path == isosbestic_channel_path:
                    file_data.isosbestic_channel = ch
                    break
        
        # Validate required channels
        if file_data.signal_channel is None:
            raise ValueError("No signal channel detected. Please specify manually.")
        
        # Load data
        file_data = reader.load_data(file_data)
        
        # Calculate ΔF/F
        calc = DFFCalculator(filter_cutoff=filter_cutoff)
        
        if file_data.isosbestic_channel is not None:
            dff, filt_sig, filt_iso, fitted = calc.calculate(
                file_data.signal_data,
                file_data.isosbestic_data,
                file_data.signal_channel.sampling_rate
            )
            photometry_time = file_data.signal_time
        else:
            # No isosbestic, use simple baseline method
            dff, filt_sig = calc.calculate_simple(
                file_data.signal_data,
                file_data.signal_channel.sampling_rate
            )
            photometry_time = file_data.signal_time
        
        # Synchronize with video
        if file_data.video_info is not None and file_data.video_timestamps is not None:
            sync = VideoSynchronizer(method='average')
            frame_dff, frame_nums = sync.synchronize(
                dff, photometry_time, file_data.video_timestamps
            )
            
            # Create aligned DataFrame
            df = sync.create_aligned_dataframe(
                frame_dff, frame_nums, file_data.video_timestamps
            )
        else:
            # No video sync, export photometry data directly
            df = pd.DataFrame({
                'Sample': np.arange(1, len(dff) + 1),
                'Time_sec': photometry_time,
                'DeltaF_F': dff
            })
        
        # Save CSV
        df.to_csv(csv_path, index=False)
        results['csv_output'] = str(csv_path)
        
        # Create combined video (if requested and video exists)
        if create_video and file_data.video_info and file_data.video_info.absolute_path:
            # Video creation will be handled by a separate function
            # using moviepy (to be implemented in GUI integration)
            results['video_output'] = str(video_path)
            results['video_source'] = file_data.video_info.absolute_path
        
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
    
    return results


# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        doric_file = sys.argv[1]
    else:
        # Use sample data
        script_dir = Path(__file__).parent
        doric_file = script_dir / "SampleData" / "NC500_Acq_0004.doric"
    
    print(f"Processing: {doric_file}")
    
    # Test file reading
    reader = DoricFileReader(doric_file)
    data = reader.scan_file()
    
    print(f"\nDetected Signal Channel: {data.signal_channel}")
    print(f"Detected Isosbestic Channel: {data.isosbestic_channel}")
    print(f"Video Info: {data.video_info}")
    print(f"\nAll Channels:")
    for ch in data.all_channels:
        print(f"  - {ch.name}: {ch.path} ({ch.sampling_rate:.1f} Hz, {ch.n_samples} samples)")
    
    # Load data and calculate ΔF/F
    data = reader.load_data(data)
    
    if data.signal_channel and data.isosbestic_channel:
        calc = DFFCalculator()
        dff, filt_sig, filt_iso, fitted = calc.calculate(
            data.signal_data,
            data.isosbestic_data,
            data.signal_channel.sampling_rate
        )
        
        print(f"\nΔF/F Statistics:")
        print(f"  Min: {np.min(dff):.2f}%")
        print(f"  Max: {np.max(dff):.2f}%")
        print(f"  Mean: {np.mean(dff):.2f}%")
        print(f"  Std: {np.std(dff):.2f}%")
        
        # Test video synchronization
        if data.video_timestamps is not None:
            sync = VideoSynchronizer()
            frame_dff, frame_nums = sync.synchronize(
                dff, data.signal_time, data.video_timestamps
            )
            
            print(f"\nVideo Synchronization:")
            print(f"  Video frames: {len(frame_nums)}")
            print(f"  Frame-aligned ΔF/F range: [{np.min(frame_dff):.2f}%, {np.max(frame_dff):.2f}%]")
