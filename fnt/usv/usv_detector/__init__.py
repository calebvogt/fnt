"""
USV Detector Module for FNT

This module provides automatic detection and classification of Ultrasonic Vocalizations (USVs)
from audio files, specifically designed for prairie vole recordings.

Main components:
- USVDetectorConfig: Configuration dataclass for detection parameters
- DSPDetector: Signal processing-based USV detection
- batch_process: Batch processing of multiple WAV files

Example usage:
    from fnt.usv.usv_detector import USVDetectorConfig, DSPDetector, batch_process

    # Process a single file
    config = USVDetectorConfig()
    detector = DSPDetector(config)
    calls = detector.detect_file("recording.wav")

    # Batch process a folder
    results = batch_process("/path/to/wav/files", config)
"""

from .config import USVDetectorConfig, get_prairie_vole_config
from .dsp_detector import DSPDetector
from .batch import batch_process, process_single_file
from .io import save_das_format, load_das_annotations, generate_summary

__all__ = [
    'USVDetectorConfig',
    'get_prairie_vole_config',
    'DSPDetector',
    'batch_process',
    'process_single_file',
    'save_das_format',
    'load_das_annotations',
    'generate_summary',
]
