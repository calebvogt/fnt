#!/usr/bin/env python3
"""
Explore the HDF5 structure of .doric files from Doric WiFP system.

This script recursively walks the HDF5 tree and prints:
- Group/dataset hierarchy
- Dataset shapes, dtypes, and sample values
- Attributes at each level

Usage:
    python explore_doric_structure.py <path_to_doric_file>
    
Or run directly to use the sample data file.
"""

import sys
import os
from pathlib import Path

try:
    import h5py
except ImportError:
    print("ERROR: h5py not installed. Run: pip install h5py")
    sys.exit(1)

import numpy as np


def explore_hdf5(name: str, obj, indent: int = 0) -> None:
    """Recursively explore and print HDF5 structure."""
    prefix = "  " * indent
    
    if isinstance(obj, h5py.Group):
        print(f"{prefix}📁 GROUP: {name}/")
        
        # Print attributes if any
        if len(obj.attrs) > 0:
            print(f"{prefix}   Attributes:")
            for attr_name, attr_val in obj.attrs.items():
                # Handle byte strings
                if isinstance(attr_val, bytes):
                    attr_val = attr_val.decode('utf-8', errors='replace')
                elif isinstance(attr_val, np.ndarray) and attr_val.dtype.kind == 'S':
                    attr_val = [v.decode('utf-8', errors='replace') for v in attr_val.flat]
                print(f"{prefix}     - {attr_name}: {attr_val}")
        
        # Recurse into children
        for child_name in obj.keys():
            child_path = f"{name}/{child_name}" if name else child_name
            explore_hdf5(child_path, obj[child_name], indent + 1)
            
    elif isinstance(obj, h5py.Dataset):
        shape_str = str(obj.shape)
        dtype_str = str(obj.dtype)
        
        # Get sample values
        try:
            if obj.size == 0:
                sample = "(empty)"
            elif obj.size <= 5:
                sample = str(obj[()])
            else:
                # Get first and last few values
                flat = obj[()].flatten()
                first_vals = flat[:3]
                last_vals = flat[-2:]
                sample = f"[{first_vals[0]:.6g}, {first_vals[1]:.6g}, {first_vals[2]:.6g} ... {last_vals[0]:.6g}, {last_vals[1]:.6g}]"
        except Exception as e:
            sample = f"(error reading: {e})"
        
        print(f"{prefix}📊 DATASET: {name}")
        print(f"{prefix}     Shape: {shape_str}, Dtype: {dtype_str}")
        print(f"{prefix}     Sample: {sample}")
        
        # Print attributes if any
        if len(obj.attrs) > 0:
            print(f"{prefix}     Attributes:")
            for attr_name, attr_val in obj.attrs.items():
                if isinstance(attr_val, bytes):
                    attr_val = attr_val.decode('utf-8', errors='replace')
                elif isinstance(attr_val, np.ndarray) and attr_val.dtype.kind == 'S':
                    attr_val = [v.decode('utf-8', errors='replace') for v in attr_val.flat]
                print(f"{prefix}       - {attr_name}: {attr_val}")


def find_photometry_channels(f: h5py.File) -> dict:
    """
    Search for potential photometry signal channels.
    Returns dict with detected channels and their paths.
    """
    channels = {
        'isosbestic_405': [],
        'signal_470_473': [],
        'ttl_digital': [],
        'time': [],
        'video_related': []
    }
    
    def search(name, obj):
        name_lower = name.lower()
        
        # Look for isosbestic (405nm)
        if any(x in name_lower for x in ['405', 'isosbestic', 'iso']):
            if isinstance(obj, h5py.Dataset):
                channels['isosbestic_405'].append(name)
        
        # Look for signal (470/473nm, GCaMP, dLight)
        if any(x in name_lower for x in ['470', '473', 'gcamp', 'dlight', 'signal']):
            if isinstance(obj, h5py.Dataset):
                channels['signal_470_473'].append(name)
        
        # Look for TTL/digital signals
        if any(x in name_lower for x in ['ttl', 'digital', 'dio', 'din']):
            if isinstance(obj, h5py.Dataset):
                channels['ttl_digital'].append(name)
        
        # Look for time vectors
        if any(x in name_lower for x in ['time', 'timestamp']):
            if isinstance(obj, h5py.Dataset):
                channels['time'].append(name)
        
        # Look for video-related data
        if any(x in name_lower for x in ['video', 'frame', 'camera']):
            channels['video_related'].append(name)
    
    f.visititems(search)
    return channels


def analyze_sampling_rate(f: h5py.File, time_path: str) -> float:
    """Calculate sampling rate from a time dataset."""
    try:
        time_data = f[time_path][:]
        if len(time_data) < 2:
            return 0.0
        
        # Calculate average time step
        dt = np.diff(time_data)
        avg_dt = np.mean(dt)
        
        # Sampling rate = 1 / dt
        if avg_dt > 0:
            return 1.0 / avg_dt
        return 0.0
    except Exception:
        return 0.0


def main():
    # Determine file path
    if len(sys.argv) > 1:
        doric_path = sys.argv[1]
    else:
        # Default to sample data
        script_dir = Path(__file__).parent
        doric_path = script_dir / "SampleData" / "NC500_Acq_0004.doric"
    
    doric_path = Path(doric_path)
    
    if not doric_path.exists():
        print(f"ERROR: File not found: {doric_path}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"EXPLORING DORIC FILE: {doric_path.name}")
    print(f"Full path: {doric_path}")
    print(f"File size: {doric_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 80)
    
    with h5py.File(doric_path, 'r') as f:
        print("\n" + "=" * 80)
        print("FULL HDF5 STRUCTURE")
        print("=" * 80 + "\n")
        
        # Explore root level
        for key in f.keys():
            explore_hdf5(key, f[key], indent=0)
        
        print("\n" + "=" * 80)
        print("AUTO-DETECTED CHANNELS")
        print("=" * 80 + "\n")
        
        channels = find_photometry_channels(f)
        
        for channel_type, paths in channels.items():
            print(f"\n{channel_type.upper()}:")
            if paths:
                for p in paths:
                    print(f"  - {p}")
            else:
                print("  (none found)")
        
        print("\n" + "=" * 80)
        print("SAMPLING RATE ANALYSIS")
        print("=" * 80 + "\n")
        
        for time_path in channels['time']:
            rate = analyze_sampling_rate(f, time_path)
            if rate > 0:
                print(f"  {time_path}: {rate:.2f} Hz")
    
    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
