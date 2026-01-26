"""
Spectrogram computation utilities for USV detection.

Handles loading audio files (including ADPCM format) and computing spectrograms
optimized for ultrasonic vocalization detection.
"""

import os
import subprocess
import tempfile
import numpy as np
from scipy import signal
from typing import Tuple, Optional
import warnings

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def load_audio(filepath: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio file, handling various formats including ADPCM.

    Uses soundfile if available, falls back to ffmpeg for formats
    like ADPCM IMA WAV that soundfile can't handle directly.

    Args:
        filepath: Path to audio file
        target_sr: Optional target sample rate for resampling

    Returns:
        Tuple of (audio_data, sample_rate)
        audio_data is a 1D numpy array of float32 samples
    """
    audio = None
    sr = None

    # Try soundfile first (fastest, handles most formats)
    if HAS_SOUNDFILE:
        try:
            audio, sr = sf.read(filepath, dtype='float32')
        except Exception:
            pass  # Fall through to ffmpeg

    # Fallback to ffmpeg for ADPCM and other formats
    if audio is None:
        audio, sr = _load_with_ffmpeg(filepath)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample if requested
    if target_sr is not None and sr != target_sr:
        audio = _resample(audio, sr, target_sr)
        sr = target_sr

    return audio, sr


def _load_with_ffmpeg(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load audio using ffmpeg (handles ADPCM and other exotic formats).

    Args:
        filepath: Path to audio file

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    # First, get the sample rate from the input file
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filepath
    ]

    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        sr = int(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Failed to probe audio file: {e}")

    # Convert to raw PCM float32 using ffmpeg
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp_file:
        temp_path = tmp_file.name

    try:
        # Convert to raw 32-bit float PCM
        convert_cmd = [
            'ffmpeg',
            '-i', filepath,
            '-f', 'f32le',          # 32-bit float, little-endian
            '-acodec', 'pcm_f32le',
            '-ac', '1',              # Convert to mono
            '-y',                    # Overwrite output
            temp_path
        ]

        subprocess.run(convert_cmd, capture_output=True, check=True)

        # Read raw PCM data
        audio = np.fromfile(temp_path, dtype=np.float32)

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    return audio, sr


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    num_samples = int(len(audio) * target_sr / orig_sr)
    return signal.resample(audio, num_samples)


def compute_spectrogram(
    audio: np.ndarray,
    sr: int,
    nperseg: int = 512,
    noverlap: int = 384,
    nfft: int = 1024,
    window: str = 'hann',
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram optimized for USV detection.

    Args:
        audio: Audio signal array
        sr: Sample rate
        nperseg: Length of each segment (FFT window size)
        noverlap: Number of points to overlap between segments
        nfft: Length of FFT (zero-padded)
        window: Window function
        min_freq: Minimum frequency to include (Hz)
        max_freq: Maximum frequency to include (Hz)

    Returns:
        Tuple of (frequencies, times, Sxx_db)
        - frequencies: Array of sample frequencies (Hz)
        - times: Array of segment times (s)
        - Sxx_db: Spectrogram in dB scale
    """
    # Compute spectrogram
    frequencies, times, Sxx = signal.spectrogram(
        audio,
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window=window,
        scaling='density'
    )

    # Convert to dB scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Apply frequency mask if specified
    if min_freq is not None or max_freq is not None:
        min_f = min_freq if min_freq is not None else 0
        max_f = max_freq if max_freq is not None else sr / 2
        freq_mask = (frequencies >= min_f) & (frequencies <= max_f)
        frequencies = frequencies[freq_mask]
        Sxx_db = Sxx_db[freq_mask, :]

    return frequencies, times, Sxx_db


def bandpass_filter(
    audio: np.ndarray,
    sr: int,
    lowcut: float,
    highcut: float,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to audio.

    Args:
        audio: Input audio array
        sr: Sample rate
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order

    Returns:
        Filtered audio array
    """
    nyquist = sr / 2

    # Ensure frequencies are within valid range
    lowcut = max(lowcut, 1)  # Avoid 0 Hz
    highcut = min(highcut, nyquist - 1)

    if lowcut >= highcut:
        raise ValueError(f"Invalid frequency range: {lowcut} - {highcut} Hz")

    # Design filter
    sos = signal.butter(
        order,
        [lowcut / nyquist, highcut / nyquist],
        btype='bandpass',
        output='sos'
    )

    # Apply filter
    return signal.sosfilt(sos, audio)


def estimate_noise_floor(
    Sxx_db: np.ndarray,
    percentile: float = 25.0
) -> np.ndarray:
    """
    Estimate background noise floor from spectrogram.

    Uses a low percentile of power values per frequency bin
    to estimate the noise floor.

    Args:
        Sxx_db: Spectrogram in dB scale (freq x time)
        percentile: Percentile for noise estimate (default 25)

    Returns:
        Noise floor estimate per frequency bin (1D array)
    """
    return np.percentile(Sxx_db, percentile, axis=1)


def compute_power_envelope(
    Sxx_db: np.ndarray,
    frequencies: np.ndarray,
    min_freq: float,
    max_freq: float
) -> np.ndarray:
    """
    Compute power envelope over time in specified frequency range.

    Args:
        Sxx_db: Spectrogram in dB scale
        frequencies: Frequency array
        min_freq: Minimum frequency
        max_freq: Maximum frequency

    Returns:
        Power envelope array (1D, one value per time bin)
    """
    freq_mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    return np.max(Sxx_db[freq_mask, :], axis=0)


def get_audio_info(filepath: str) -> dict:
    """
    Get information about an audio file.

    Args:
        filepath: Path to audio file

    Returns:
        Dictionary with audio file information
    """
    probe_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration,size:stream=codec_name,sample_rate,channels',
        '-of', 'json',
        filepath
    ]

    try:
        import json
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        stream = data.get('streams', [{}])[0]
        fmt = data.get('format', {})

        return {
            'filepath': filepath,
            'codec': stream.get('codec_name', 'unknown'),
            'sample_rate': int(stream.get('sample_rate', 0)),
            'channels': int(stream.get('channels', 0)),
            'duration': float(fmt.get('duration', 0)),
            'size_bytes': int(fmt.get('size', 0)),
        }
    except Exception as e:
        return {
            'filepath': filepath,
            'error': str(e)
        }
