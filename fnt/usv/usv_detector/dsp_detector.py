"""
DSP-based USV detector using spectral energy thresholding.

This module implements a signal processing approach to USV detection
without requiring machine learning training data.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import label, binary_dilation, binary_erosion
from typing import List, Dict, Optional, Tuple
from .config import USVDetectorConfig
from .spectrogram import (
    load_audio, compute_spectrogram, bandpass_filter,
    estimate_noise_floor, get_audio_info
)


class DSPDetector:
    """
    DSP-based USV detector using spectral energy thresholding.

    This detector works by:
    1. Computing a spectrogram of the audio
    2. Estimating the noise floor adaptively
    3. Finding regions where energy exceeds the threshold
    4. Filtering by duration and merging close calls
    5. Classifying call types based on frequency trajectory

    Attributes:
        config: USVDetectorConfig with detection parameters
    """

    def __init__(self, config: Optional[USVDetectorConfig] = None):
        """
        Initialize the detector.

        Args:
            config: Configuration object. Uses default prairie vole config if None.
        """
        if config is None:
            from .config import get_prairie_vole_config
            config = get_prairie_vole_config()
        self.config = config

    def detect_file(self, filepath: str) -> List[Dict]:
        """
        Detect USV calls in an audio file.

        Args:
            filepath: Path to audio file

        Returns:
            List of detected calls, each as a dictionary with:
                - start_seconds: Start time
                - stop_seconds: End time
                - name: Call type classification
                - peak_freq_hz: Peak frequency
                - mean_freq_hz: Mean frequency
                - freq_bandwidth_hz: Frequency bandwidth
                - duration_ms: Duration in milliseconds
                - mean_power_db: Mean power in dB
                - max_power_db: Maximum power in dB
        """
        # Load audio
        audio, sr = load_audio(filepath)

        # Detect calls
        return self.detect(audio, sr)

    def detect(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Detect USV calls in audio data.

        Args:
            audio: Audio signal array (1D, float)
            sr: Sample rate in Hz

        Returns:
            List of detected calls as dictionaries
        """
        # Validate sample rate
        if sr != self.config.sample_rate:
            # Could resample here, but for now just use the actual rate
            pass

        # Process in chunks for long files
        duration = len(audio) / sr
        if duration > self.config.chunk_duration_s * 2:
            return self._detect_chunked(audio, sr)

        # Apply bandpass filter
        filtered = bandpass_filter(
            audio, sr,
            self.config.min_freq_hz,
            self.config.max_freq_hz
        )

        # Compute spectrogram
        frequencies, times, Sxx_db = compute_spectrogram(
            filtered, sr,
            nperseg=self.config.nperseg,
            noverlap=self.config.noverlap,
            nfft=self.config.nfft,
            window=self.config.window_type,
            min_freq=self.config.min_freq_hz,
            max_freq=self.config.max_freq_hz
        )

        # Detect calls from spectrogram
        calls = self._detect_from_spectrogram(frequencies, times, Sxx_db)

        # Filter by duration
        calls = self._filter_by_duration(calls)

        # Merge close calls
        calls = self._merge_close_calls(calls)

        # Sample peak frequencies across each call
        calls = self._sample_peak_frequencies(calls, frequencies, times, Sxx_db)

        return calls

    def _detect_chunked(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """
        Process long audio files in chunks.

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            List of all detected calls
        """
        chunk_samples = int(self.config.chunk_duration_s * sr)
        overlap_samples = int(self.config.chunk_overlap_s * sr)

        all_calls = []
        offset = 0

        while offset < len(audio):
            chunk_end = min(offset + chunk_samples, len(audio))
            chunk = audio[offset:chunk_end]

            # Detect in this chunk
            calls = self.detect(chunk, sr)

            # Adjust times for offset
            time_offset = offset / sr
            for call in calls:
                call['start_seconds'] += time_offset
                call['stop_seconds'] += time_offset

            all_calls.extend(calls)
            offset += chunk_samples - overlap_samples

        # Remove duplicates from overlap regions
        all_calls = self._deduplicate_calls(all_calls)

        return all_calls

    def _detect_from_spectrogram(
        self,
        frequencies: np.ndarray,
        times: np.ndarray,
        Sxx_db: np.ndarray
    ) -> List[Dict]:
        """
        Detect calls from spectrogram using adaptive thresholding.

        Args:
            frequencies: Frequency array
            times: Time array
            Sxx_db: Spectrogram in dB

        Returns:
            List of raw detections
        """
        # Estimate noise floor per frequency bin
        noise_floor = estimate_noise_floor(Sxx_db, self.config.noise_percentile)

        # Create threshold mask
        threshold = noise_floor[:, np.newaxis] + self.config.energy_threshold_db
        mask = Sxx_db > threshold

        # Apply morphological operations to clean up mask
        # Dilate to connect nearby regions, then erode to remove small artifacts
        struct = np.ones((3, 3))
        mask = binary_dilation(mask, structure=struct, iterations=1)
        mask = binary_erosion(mask, structure=struct, iterations=1)

        # Find connected regions
        labeled, num_features = label(mask)

        calls = []
        for i in range(1, num_features + 1):
            # Get region mask
            region_mask = labeled == i

            # Get time span
            time_indices = np.any(region_mask, axis=0)
            if not np.any(time_indices):
                continue

            time_idx = np.where(time_indices)[0]
            start_idx, end_idx = time_idx[0], time_idx[-1]

            start_s = times[start_idx]
            stop_s = times[end_idx]

            # Get frequency span
            freq_indices = np.any(region_mask, axis=1)
            freq_idx = np.where(freq_indices)[0]
            min_freq_idx, max_freq_idx = freq_idx[0], freq_idx[-1]

            # Extract region for feature computation
            region_spec = Sxx_db[:, start_idx:end_idx+1].copy()
            region_spec[~region_mask[:, start_idx:end_idx+1]] = -100  # Mask out non-call

            # Compute features
            peak_freq_hz = frequencies[np.unravel_index(
                np.argmax(region_spec), region_spec.shape
            )[0]]

            # Weighted mean frequency
            power_linear = 10 ** (region_spec / 10)
            power_linear[region_spec < -99] = 0
            total_power = np.sum(power_linear)
            if total_power > 0:
                mean_freq_hz = np.sum(frequencies[:, np.newaxis] * power_linear) / total_power
            else:
                mean_freq_hz = peak_freq_hz

            # Actual frequency bounds of this call
            call_min_freq_hz = frequencies[min_freq_idx]
            call_max_freq_hz = frequencies[max_freq_idx]
            freq_bandwidth_hz = call_max_freq_hz - call_min_freq_hz
            duration_ms = (stop_s - start_s) * 1000
            mean_power_db = np.mean(region_spec[region_spec > -99])
            max_power_db = np.max(region_spec)

            calls.append({
                'start_seconds': float(start_s),
                'stop_seconds': float(stop_s),
                'min_freq_hz': float(call_min_freq_hz),
                'max_freq_hz': float(call_max_freq_hz),
                'peak_freq_hz': float(peak_freq_hz),
                'mean_freq_hz': float(mean_freq_hz),
                'freq_bandwidth_hz': float(freq_bandwidth_hz),
                'duration_ms': float(duration_ms),
                'mean_power_db': float(mean_power_db),
                'max_power_db': float(max_power_db),
                '_start_idx': start_idx,
                '_end_idx': end_idx,
            })

        return calls

    def _filter_by_duration(self, calls: List[Dict]) -> List[Dict]:
        """Filter calls by minimum and maximum duration."""
        return [
            call for call in calls
            if (self.config.min_duration_ms <= call['duration_ms'] <= self.config.max_duration_ms)
        ]

    def _merge_close_calls(self, calls: List[Dict]) -> List[Dict]:
        """Merge calls that are closer than min_gap_ms."""
        if len(calls) < 2:
            return calls

        # Sort by start time
        calls = sorted(calls, key=lambda x: x['start_seconds'])

        min_gap_s = self.config.min_gap_ms / 1000
        merged = [calls[0]]

        for call in calls[1:]:
            gap = call['start_seconds'] - merged[-1]['stop_seconds']

            if gap < min_gap_s:
                # Merge with previous call
                prev = merged[-1]
                prev['stop_seconds'] = call['stop_seconds']
                prev['duration_ms'] = (prev['stop_seconds'] - prev['start_seconds']) * 1000

                # Update aggregate features - expand frequency bounds to encompass both calls
                prev['peak_freq_hz'] = max(prev['peak_freq_hz'], call['peak_freq_hz'])
                prev['max_power_db'] = max(prev['max_power_db'], call['max_power_db'])
                prev['mean_power_db'] = (prev['mean_power_db'] + call['mean_power_db']) / 2
                prev['min_freq_hz'] = min(prev.get('min_freq_hz', call['min_freq_hz']), call['min_freq_hz'])
                prev['max_freq_hz'] = max(prev.get('max_freq_hz', call['max_freq_hz']), call['max_freq_hz'])
                prev['freq_bandwidth_hz'] = prev['max_freq_hz'] - prev['min_freq_hz']

                # Update internal indices
                if '_end_idx' in call:
                    prev['_end_idx'] = call['_end_idx']
            else:
                merged.append(call)

        return merged

    def _deduplicate_calls(self, calls: List[Dict]) -> List[Dict]:
        """Remove duplicate calls from overlapping chunks."""
        if len(calls) < 2:
            return calls

        calls = sorted(calls, key=lambda x: x['start_seconds'])
        deduplicated = [calls[0]]

        for call in calls[1:]:
            # Check if overlaps with previous call
            prev = deduplicated[-1]
            overlap_start = max(prev['start_seconds'], call['start_seconds'])
            overlap_end = min(prev['stop_seconds'], call['stop_seconds'])

            if overlap_start < overlap_end:
                # Overlapping - keep the one with higher power
                if call['max_power_db'] > prev['max_power_db']:
                    deduplicated[-1] = call
            else:
                deduplicated.append(call)

        return deduplicated

    def _sample_peak_frequencies(
        self,
        calls: List[Dict],
        frequencies: np.ndarray,
        times: np.ndarray,
        Sxx_db: np.ndarray
    ) -> List[Dict]:
        """
        Sample peak frequencies at evenly-spaced points across each call.

        Adds peak_freq_1 ... peak_freq_N columns where N = config.freq_samples.
        Each value is the frequency of maximum amplitude at that time sample.
        """
        n_samples = getattr(self.config, 'freq_samples', 5)

        for call in calls:
            start_idx = call.get('_start_idx', 0)
            end_idx = call.get('_end_idx', Sxx_db.shape[1] - 1)

            # Clean up internal indices
            call.pop('_start_idx', None)
            call.pop('_end_idx', None)

            n_frames = end_idx - start_idx + 1
            if n_frames < 1:
                for i in range(1, n_samples + 1):
                    call[f'peak_freq_{i}'] = float('nan')
                continue

            # Evenly-spaced indices across the call duration
            sample_indices = np.linspace(start_idx, end_idx, n_samples, dtype=int)

            for i, col_idx in enumerate(sample_indices, 1):
                col_idx = min(col_idx, Sxx_db.shape[1] - 1)
                peak_freq = frequencies[np.argmax(Sxx_db[:, col_idx])]
                call[f'peak_freq_{i}'] = float(peak_freq)

        return calls

    def get_spectrogram(
        self,
        filepath: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get spectrogram for visualization.

        Args:
            filepath: Path to audio file

        Returns:
            Tuple of (frequencies, times, Sxx_db)
        """
        audio, sr = load_audio(filepath)

        # Apply bandpass filter
        filtered = bandpass_filter(
            audio, sr,
            self.config.min_freq_hz,
            self.config.max_freq_hz
        )

        return compute_spectrogram(
            filtered, sr,
            nperseg=self.config.nperseg,
            noverlap=self.config.noverlap,
            nfft=self.config.nfft,
            window=self.config.window_type,
            min_freq=self.config.min_freq_hz,
            max_freq=self.config.max_freq_hz
        )
