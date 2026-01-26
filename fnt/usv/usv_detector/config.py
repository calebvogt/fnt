"""
Configuration module for USV detection.

Provides a dataclass-based configuration system with sensible defaults
for prairie vole ultrasonic vocalization detection.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import json


@dataclass
class USVDetectorConfig:
    """
    Configuration for USV detection and classification.

    Attributes:
        min_freq_hz: Lower bound of USV frequency range (Hz)
        max_freq_hz: Upper bound of USV frequency range (Hz)
        sample_rate: Expected sample rate of audio files (Hz)
        nperseg: FFT window size in samples
        noverlap: Overlap between FFT windows in samples
        nfft: FFT size (zero-padded)
        window_type: Window function for spectrogram
        energy_threshold_db: Energy threshold relative to background (dB)
        min_duration_ms: Minimum call duration (ms)
        max_duration_ms: Maximum call duration (ms)
        min_gap_ms: Minimum gap between calls to merge (ms)
        chunk_duration_s: Duration of chunks for processing long files (s)
        chunk_overlap_s: Overlap between chunks (s)
        call_types: List of call type labels for classification
        output_suffix: Suffix for output annotation files
    """

    # Frequency parameters (prairie vole specific: 30-60 kHz typical)
    min_freq_hz: int = 25000        # Lower bound with buffer
    max_freq_hz: int = 65000        # Upper bound with buffer

    # Audio parameters
    sample_rate: int = 250000       # Expected sample rate

    # Spectrogram parameters
    nperseg: int = 512              # ~2ms window at 250kHz
    noverlap: int = 384             # 75% overlap for better time resolution
    nfft: int = 1024                # Zero-pad for frequency resolution
    window_type: str = "hann"       # Window function

    # Detection parameters
    energy_threshold_db: float = 10.0   # dB above background noise
    min_duration_ms: float = 5.0        # Minimum call duration
    max_duration_ms: float = 300.0      # Maximum call duration
    min_gap_ms: float = 5.0             # Minimum gap to consider separate calls

    # Adaptive threshold parameters
    noise_percentile: float = 25.0      # Percentile for background noise estimate

    # Batch processing
    chunk_duration_s: float = 60.0      # Process in 60-second chunks
    chunk_overlap_s: float = 0.5        # Small overlap to avoid boundary issues

    # Classification parameters
    classify_calls: bool = True
    call_types: List[str] = field(default_factory=lambda: [
        "sweep_up", "sweep_down", "inverted_u", "u_shape", "flat", "complex", "unknown"
    ])

    # Classification thresholds
    freq_modulation_threshold: float = 5000.0   # Hz change to count as modulated
    slope_threshold: float = 50000.0            # Hz/s to distinguish sweep direction

    # Output
    output_suffix: str = "_usv_detections"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'USVDetectorConfig':
        """Create config from dictionary."""
        # Filter out any keys that aren't valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'USVDetectorConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self) -> List[str]:
        """
        Validate configuration parameters.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if self.min_freq_hz >= self.max_freq_hz:
            errors.append("min_freq_hz must be less than max_freq_hz")

        if self.max_freq_hz > self.sample_rate / 2:
            errors.append(f"max_freq_hz ({self.max_freq_hz}) exceeds Nyquist frequency ({self.sample_rate / 2})")

        if self.min_duration_ms >= self.max_duration_ms:
            errors.append("min_duration_ms must be less than max_duration_ms")

        if self.noverlap >= self.nperseg:
            errors.append("noverlap must be less than nperseg")

        if self.energy_threshold_db < 0:
            errors.append("energy_threshold_db should be positive (dB above noise floor)")

        return errors


def get_prairie_vole_config() -> USVDetectorConfig:
    """
    Get default configuration optimized for prairie vole USVs.

    Prairie voles typically produce calls in the 30-60 kHz range,
    with sweep-up and inverted-U being common call types.
    """
    return USVDetectorConfig(
        min_freq_hz=25000,
        max_freq_hz=65000,
        energy_threshold_db=10.0,
        min_duration_ms=5.0,
        max_duration_ms=300.0,
        min_gap_ms=5.0,
    )


def get_mouse_config() -> USVDetectorConfig:
    """
    Get default configuration for mouse USVs.

    Mice typically produce calls in a wider range (30-100+ kHz).
    """
    return USVDetectorConfig(
        min_freq_hz=30000,
        max_freq_hz=100000,
        energy_threshold_db=12.0,
        min_duration_ms=3.0,
        max_duration_ms=500.0,
        min_gap_ms=3.0,
    )


def get_rat_config() -> USVDetectorConfig:
    """
    Get default configuration for rat USVs.

    Rats produce 22 kHz and 50 kHz calls with different characteristics.
    """
    return USVDetectorConfig(
        min_freq_hz=18000,
        max_freq_hz=80000,
        energy_threshold_db=10.0,
        min_duration_ms=10.0,
        max_duration_ms=2000.0,  # 22 kHz calls can be long
        min_gap_ms=10.0,
    )
