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

# Lazy exports (PEP 562): the DSP detector / batch / io submodules pull in
# heavy deps (scipy, pandas). Importing them eagerly here made EVERY import of
# a sibling submodule (mad_labels, mad_project, spectrogram — used by the Mask
# Audio Detector, which never touches the DSP stack) drag pandas + the DSP
# detector in, adding seconds to MAD's startup. Load each name on first access
# instead, so `from fnt.usv.usv_detector import DSPDetector` still works but
# `from fnt.usv.usv_detector.mad_labels import ...` stays lightweight.
_LAZY_EXPORTS = {
    'USVDetectorConfig': '.config',
    'get_prairie_vole_config': '.config',
    'DSPDetector': '.dsp_detector',
    'batch_process': '.batch',
    'process_single_file': '.batch',
    'save_das_format': '.io',
    'load_das_annotations': '.io',
    'generate_summary': '.io',
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    mod = _LAZY_EXPORTS.get(name)
    if mod is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value  # cache so subsequent access skips __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
