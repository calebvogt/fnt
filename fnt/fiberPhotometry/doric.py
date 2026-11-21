"""Reading Doric .doric recordings (HDF5) from the Doric wireless (WiFP / NC500) system.

This is the Doric-specific part of the fiber photometry module: it finds the demodulated
(lock-in) channels, tells the signal from the isosbestic channel by LED wavelength, and reads
the behaviour-video timestamps and the headstage IMU. Everything after reading -- dropouts,
the control fit, dF/F, z-scores, video alignment -- is system-independent and lives in
``processing.py`` and ``sync.py``.
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import h5py


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
    # True when video_timestamps hold one hardware time per camera frame (LabJack camera TTL):
    # frames are then timed by them, not by k / fps
    video_frame_clock: bool = False
    
    # IMU data (accelerometer)
    imu_accel_x: Optional[np.ndarray] = None
    imu_accel_y: Optional[np.ndarray] = None
    imu_accel_z: Optional[np.ndarray] = None
    imu_time: Optional[np.ndarray] = None

    # Headstage state (wireless link quality and battery), when the file has it
    signal_strength: Optional[np.ndarray] = None     # dBm; -127 = no link
    battery_level: Optional[np.ndarray] = None       # %
    state_time: Optional[np.ndarray] = None


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
            data = self._load_state(f, data)
        
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
    
    def _load_state(self, f: h5py.File, data: DoricFileData) -> DoricFileData:
        """Wireless signal strength and battery level of the recording's headstage (the one
        whose signal channel was chosen), from its ``<headstage>State`` group."""
        if data.signal_channel is None:
            return data
        group = data.signal_channel.path.rsplit("/", 2)[0]          # .../Series0001
        head = re.match(r"(Headstage\d+)", data.signal_channel.name)
        if not head:
            return data
        base = f"{group}/{head.group(1)}State"
        try:
            if f"{base}/Time" in f:
                data.state_time = f[f"{base}/Time"][:]
                if f"{base}/SignalStrength" in f:
                    data.signal_strength = f[f"{base}/SignalStrength"][:]
                if f"{base}/BatteryLevel" in f:
                    data.battery_level = f[f"{base}/BatteryLevel"][:]
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
                        # median, not mean: wireless gaps would drag a mean interval up
                        dt = np.median(np.diff(time_arr))
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
                                # the logged rate; frames are timed from the video itself (sync.py)
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
