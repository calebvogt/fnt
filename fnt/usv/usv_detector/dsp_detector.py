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
    load_audio, compute_spectrogram, compute_spectrogram_auto,
    bandpass_filter, estimate_noise_floor, get_audio_info
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

    def detect_file(self, filepath: str, progress_callback=None) -> List[Dict]:
        """
        Detect USV calls in an audio file.

        Args:
            filepath: Path to audio file
            progress_callback: Optional callable(fraction) where fraction is 0.0-1.0

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
        return self.detect(audio, sr, progress_callback=progress_callback)

    def detect(self, audio: np.ndarray, sr: int, progress_callback=None) -> List[Dict]:
        """
        Detect USV calls in audio data.

        Args:
            audio: Audio signal array (1D, float)
            sr: Sample rate in Hz
            progress_callback: Optional callable(fraction) where fraction is 0.0-1.0

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
            return self._detect_chunked(audio, sr, progress_callback=progress_callback)

        # For short files, report complete immediately
        if progress_callback:
            progress_callback(1.0)

        # Apply bandpass filter
        filtered = bandpass_filter(
            audio, sr,
            self.config.min_freq_hz,
            self.config.max_freq_hz
        )

        # Compute spectrogram (GPU-accelerated if enabled)
        frequencies, times, Sxx_db = compute_spectrogram_auto(
            filtered, sr,
            nperseg=self.config.nperseg,
            noverlap=self.config.noverlap,
            nfft=self.config.nfft,
            window=self.config.window_type,
            min_freq=self.config.min_freq_hz,
            max_freq=self.config.max_freq_hz,
            gpu_enabled=self.config.gpu_enabled,
            gpu_device=self.config.gpu_device,
        )

        # Detect calls from spectrogram
        calls = self._detect_from_spectrogram(frequencies, times, Sxx_db)

        # Filter by duration
        calls = self._filter_by_duration(calls)

        # Merge close calls
        calls = self._merge_close_calls(calls)

        # Noise rejection filters
        calls = self._filter_by_bandwidth(calls)
        calls = self._filter_by_tonality(calls, frequencies, times, Sxx_db)
        calls = self._filter_by_min_freq(calls)
        calls = self._filter_harmonics(calls)

        # Sample peak frequencies across each call (optional)
        n_samples = getattr(self.config, 'freq_samples', 5)
        if n_samples and n_samples > 0:
            calls = self._sample_peak_frequencies(calls, frequencies, times, Sxx_db)
        else:
            # Clean up internal indices
            for call in calls:
                call.pop('_start_idx', None)
                call.pop('_end_idx', None)

        return calls

    def _detect_chunked(self, audio: np.ndarray, sr: int, progress_callback=None) -> List[Dict]:
        """
        Process long audio files in chunks.

        Args:
            audio: Audio signal
            sr: Sample rate
            progress_callback: Optional callable(fraction) where fraction is 0.0-1.0

        Returns:
            List of all detected calls
        """
        chunk_samples = int(self.config.chunk_duration_s * sr)
        overlap_samples = int(self.config.chunk_overlap_s * sr)
        total_samples = len(audio)

        all_calls = []
        offset = 0

        while offset < total_samples:
            chunk_end = min(offset + chunk_samples, total_samples)
            chunk = audio[offset:chunk_end]

            # Detect in this chunk (no recursive progress callback)
            calls = self.detect(chunk, sr)

            # Adjust times for offset
            time_offset = offset / sr
            for call in calls:
                call['start_seconds'] += time_offset
                call['stop_seconds'] += time_offset

            all_calls.extend(calls)
            offset += chunk_samples - overlap_samples

            # Report per-chunk progress
            if progress_callback:
                progress_callback(min(1.0, offset / total_samples))

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

        # Split connected regions that have vertical frequency gaps
        min_freq_gap_hz = getattr(self.config, 'min_freq_gap_hz', 0)
        if min_freq_gap_hz > 0 and len(frequencies) > 1:
            freq_resolution = float(frequencies[1] - frequencies[0])
            min_gap_bins = max(1, int(min_freq_gap_hz / freq_resolution))
            labeled, num_features = self._split_by_freq_gap(
                labeled, num_features, min_gap_bins
            )

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

    def _split_by_freq_gap(
        self,
        labeled: np.ndarray,
        num_features: int,
        min_gap_bins: int
    ) -> tuple:
        """Split connected regions that have vertical frequency gaps.

        When a fundamental and its harmonic are connected in the binary mask
        (e.g. via a noise bridge), they form one region. This method detects
        vertical gaps within each region and splits them into separate labels.

        Args:
            labeled: 2D label array from scipy.ndimage.label
            num_features: Number of regions found
            min_gap_bins: Minimum gap in frequency bins to trigger a split

        Returns:
            (new_labeled, new_num_features) with split regions relabeled
        """
        new_labeled = labeled.copy()
        next_label = num_features + 1

        for i in range(1, num_features + 1):
            region_mask = labeled == i

            # Get occupied frequency bins for this region
            freq_occupied = np.any(region_mask, axis=1)  # shape: (n_freq,)
            occupied_bins = np.where(freq_occupied)[0]

            if len(occupied_bins) < 2:
                continue

            # Find gaps between occupied bins
            diffs = np.diff(occupied_bins)
            gap_positions = np.where(diffs >= min_gap_bins)[0]

            if len(gap_positions) == 0:
                continue  # No gaps to split on

            # Split into sub-regions at each gap
            # gap_positions[k] means there's a gap between
            # occupied_bins[gap_positions[k]] and occupied_bins[gap_positions[k]+1]
            split_boundaries = []
            for gp in gap_positions:
                # Midpoint of the gap (in frequency bin space)
                split_at = (occupied_bins[gp] + occupied_bins[gp + 1]) // 2
                split_boundaries.append(split_at)

            # Relabel: first segment keeps original label, subsequent get new labels
            # Sort boundaries ascending
            split_boundaries.sort()

            for sb in split_boundaries:
                # Everything above this boundary in this region gets a new label
                upper_mask = region_mask.copy()
                upper_mask[:sb + 1, :] = False  # Zero out below boundary

                if np.any(upper_mask):
                    new_labeled[upper_mask] = next_label
                    next_label += 1

                    # Update region_mask to only contain the lower portion
                    region_mask[sb + 1:, :] = False

        return new_labeled, next_label - 1

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

    def _filter_by_bandwidth(self, calls: List[Dict]) -> List[Dict]:
        """Discard detections whose frequency bandwidth exceeds max_bandwidth_hz.

        Broadband noise events (cage bumps, movement artifacts) typically span
        a much wider frequency range than real USV calls. This filter removes
        detections that are too wide spectrally.
        """
        max_bw = getattr(self.config, 'max_bandwidth_hz', 0)
        if max_bw <= 0:
            return calls  # Filter disabled
        return [c for c in calls if c.get('freq_bandwidth_hz', 0) <= max_bw]

    def _filter_by_tonality(
        self,
        calls: List[Dict],
        frequencies: np.ndarray,
        times: np.ndarray,
        Sxx_db: np.ndarray
    ) -> List[Dict]:
        """Filter detections by spectral purity (tonality score).

        Tonality measures how concentrated the energy is vs. spread across
        frequencies. A pure tone has tonality ~1.0; broadband noise has ~0.0.

        For each detection, at each time step we find the peak frequency bin
        and measure what fraction of the total column energy is within ±2 bins
        of that peak. The mean of this ratio across all time steps is the
        tonality score. Real USV calls are tonal (energy concentrated in a
        narrow band); noise events are diffuse.
        """
        min_tonality = getattr(self.config, 'min_tonality', 0)
        if min_tonality <= 0:
            return calls  # Filter disabled

        filtered = []
        for call in calls:
            start_idx = call.get('_start_idx')
            end_idx = call.get('_end_idx')
            if start_idx is None or end_idx is None:
                # No indices available — keep the call
                filtered.append(call)
                continue

            # Extract the detection's spectrogram columns
            region = Sxx_db[:, start_idx:end_idx + 1]
            # Convert to linear power for ratio computation
            power = 10 ** (region / 10)

            n_cols = region.shape[1]
            if n_cols == 0:
                filtered.append(call)
                continue

            ratios = np.zeros(n_cols)
            half_width = 2  # ±2 bins around peak
            for t in range(n_cols):
                col = power[:, t]
                total = col.sum()
                if total <= 0:
                    continue
                peak_bin = np.argmax(col)
                lo = max(0, peak_bin - half_width)
                hi = min(len(col), peak_bin + half_width + 1)
                ratios[t] = col[lo:hi].sum() / total

            tonality = float(np.mean(ratios))
            call['tonality'] = tonality

            if tonality >= min_tonality:
                filtered.append(call)

        return filtered

    def _filter_by_min_freq(self, calls: List[Dict]) -> List[Dict]:
        """Discard detections whose actual minimum frequency is too low.

        Broadband noise events often have strong energy extending down to very
        low frequencies (near the detection bandpass floor). Real USV calls
        typically start well above the lower bound. Setting min_call_freq_hz
        higher than min_freq_hz discards detections that reach too low.
        """
        min_f = getattr(self.config, 'min_call_freq_hz', 0)
        if min_f <= 0:
            return calls  # Filter disabled
        return [c for c in calls if c.get('min_freq_hz', min_f) >= min_f]

    def _filter_harmonics(self, calls: List[Dict]) -> List[Dict]:
        """Remove harmonic duplicates by merging temporally overlapping detections.

        When a USV call produces harmonics, the detector may find separate
        detections at the fundamental and at 2x, 3x, etc. These overlap
        heavily in time but sit at different frequency bands.

        For each group of detections that overlap by >=80% in time, only the
        detection with the lowest min_freq_hz (the fundamental) is kept.
        """
        if not getattr(self.config, 'harmonic_filter', False):
            return calls
        if len(calls) < 2:
            return calls

        calls = sorted(calls, key=lambda c: c['start_seconds'])
        kept = []
        used = set()

        for i, call_a in enumerate(calls):
            if i in used:
                continue

            # Collect all calls that overlap >=80% with call_a
            group = [i]
            dur_a = call_a['stop_seconds'] - call_a['start_seconds']
            if dur_a <= 0:
                kept.append(call_a)
                used.add(i)
                continue

            for j in range(i + 1, len(calls)):
                if j in used:
                    continue
                call_b = calls[j]
                # If call_b starts well after call_a ends, no more overlaps
                if call_b['start_seconds'] > call_a['stop_seconds']:
                    break

                overlap_start = max(call_a['start_seconds'], call_b['start_seconds'])
                overlap_end = min(call_a['stop_seconds'], call_b['stop_seconds'])
                overlap = max(0, overlap_end - overlap_start)

                dur_b = call_b['stop_seconds'] - call_b['start_seconds']
                min_dur = min(dur_a, dur_b) if dur_b > 0 else dur_a

                if overlap / min_dur >= 0.80:
                    group.append(j)

            # Keep only the lowest-frequency detection (the fundamental)
            best_idx = min(group, key=lambda k: calls[k].get('min_freq_hz', 0))
            kept.append(calls[best_idx])
            for idx in group:
                used.add(idx)

        return kept

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

        return compute_spectrogram_auto(
            filtered, sr,
            nperseg=self.config.nperseg,
            noverlap=self.config.noverlap,
            nfft=self.config.nfft,
            window=self.config.window_type,
            min_freq=self.config.min_freq_hz,
            max_freq=self.config.max_freq_hz,
            gpu_enabled=self.config.gpu_enabled,
            gpu_device=self.config.gpu_device,
        )
