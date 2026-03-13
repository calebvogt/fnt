"""
GPU detection utilities for USV processing.

Detects available CUDA (NVIDIA) and MPS (Apple Silicon) devices
and provides device selection helpers.
"""

from typing import List, Dict, Optional


def detect_available_devices() -> List[Dict]:
    """
    Detect all available compute devices.

    Returns:
        List of dicts with keys:
            - name: Human-readable device name
            - device: PyTorch device string (e.g. "cuda:0", "mps", "cpu")
            - type: Device type ("cuda", "mps", "cpu")
            - vram_mb: VRAM in MB (None if unavailable)
    """
    devices = []

    try:
        import torch

        # Check CUDA (NVIDIA) devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    'name': props.name,
                    'device': f'cuda:{i}',
                    'type': 'cuda',
                    'vram_mb': props.total_mem // (1024 * 1024),
                })

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append({
                'name': 'Apple Silicon GPU',
                'device': 'mps',
                'type': 'mps',
                'vram_mb': None,  # MPS doesn't expose VRAM info
            })

    except ImportError:
        pass  # torch not installed

    # Always include CPU
    devices.append({
        'name': 'CPU',
        'device': 'cpu',
        'type': 'cpu',
        'vram_mb': None,
    })

    return devices


def get_best_device() -> str:
    """
    Get the best available PyTorch device string.

    Prefers CUDA > MPS > CPU.

    Returns:
        PyTorch device string (e.g. "cuda:0", "mps", "cpu")
    """
    devices = detect_available_devices()
    for dev in devices:
        if dev['type'] == 'cuda':
            return dev['device']
    for dev in devices:
        if dev['type'] == 'mps':
            return dev['device']
    return 'cpu'


def is_gpu_available() -> bool:
    """Check if any GPU (CUDA or MPS) is available."""
    devices = detect_available_devices()
    return any(d['type'] in ('cuda', 'mps') for d in devices)
