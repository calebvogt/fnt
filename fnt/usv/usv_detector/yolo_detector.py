"""
YOLO-based visual USV detection module.

Provides a SLEAP-inspired human-in-the-loop workflow:
1. User labels a few USV calls + negative (background) regions
2. Export labeled data as spectrogram tiles with YOLO annotations
3. Train YOLOv8-nano model on the tiles
4. Run inference on new files
5. User corrects predictions, retrains for improved accuracy

This module owns the deterministic spectrogram tile renderer used for both
training and inference, ensuring pixel-level consistency.
"""

import json
import os
import random
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .spectrogram import compute_spectrogram, load_audio


# =============================================================================
# Project Configuration
# =============================================================================

PROJECT_INFO_FILENAME = 'project_info.json'
LEGACY_CONFIG_FILENAME = 'project_config.json'


@dataclass
class YOLOProjectConfig:
    """Configuration for a YOLO USV detection project (SLEAP-style).

    A project lives in its own directory (default ``~/fnt_projects/<name>/``)
    and points at one or more *source_folders* of .wav files. The project owns
    the trained model, dataset tiles, and training runs; **labels do not live
    here** — they are sibling CSVs next to each wav (see ``labels_io``).
    """
    project_dir: str = ""
    project_name: str = ""
    source_folders: List[str] = field(default_factory=list)
    last_opened_file: Optional[str] = None

    tile_size: Tuple[int, int] = (1280, 1280)
    tile_time_window_s: float = 0.5       # Time span per tile in seconds
    tile_overlap_fraction: float = 0.25   # Overlap between inference tiles
    tile_padding_s: float = 0.05          # Padding around labeled regions

    # Spectrogram parameters (must match between train and inference)
    nperseg: int = 512
    noverlap: int = 384
    nfft: int = 1024

    # Fixed dB normalization range (NOT percentile-based)
    db_min: float = -100.0
    db_max: float = -20.0

    # Colormap
    colormap: str = 'viridis'

    # Model + training
    model_variant: str = 'yolo11s.pt'     # Ultralytics auto-downloads; yolo11n/s/m/l/x or yolov8n
    imgsz: int = 1280                     # Training/inference image size; 1280 is SOTA for small objects
    val_fraction: float = 0.20            # Fraction of tiles held out for validation (stratified per source file)

    # Inference
    confidence_threshold: float = 0.15    # Low default favors recall; user accepts/rejects downstream
    iou_threshold: float = 0.5            # NMS IoU threshold
    use_sahi: bool = True                 # Slicing Aided Hyper Inference for small-object recall

    # Model history: list of {name, n_positive, n_negative, path, date}
    models: List[Dict] = field(default_factory=list)

    # Schema version; bump when incompatible changes are made.
    schema_version: int = 2

    def save(self, path: Optional[str] = None):
        """Save config to the canonical ``project_info.json``.

        If ``path`` is omitted, writes to ``<project_dir>/project_info.json``.
        """
        if path is None:
            path = os.path.join(self.project_dir, PROJECT_INFO_FILENAME)
        # Keep the derived project name in sync with the folder basename on
        # every save so renames-on-disk are picked up automatically.
        if self.project_dir and not self.project_name:
            self.project_name = os.path.basename(os.path.normpath(self.project_dir))
        data = asdict(self)
        data['tile_size'] = list(data['tile_size'])
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'YOLOProjectConfig':
        """Load config from JSON, tolerating missing newer fields for back-compat.

        ``path`` may point at a ``project_info.json`` file *or* at the project
        directory itself; in the latter case we look for either the canonical
        name or the legacy ``project_config.json``.
        """
        if os.path.isdir(path):
            candidate = os.path.join(path, PROJECT_INFO_FILENAME)
            if not os.path.exists(candidate):
                candidate = os.path.join(path, LEGACY_CONFIG_FILENAME)
            path = candidate
        with open(path) as f:
            data = json.load(f)
        data['tile_size'] = tuple(data.get('tile_size', [1280, 1280]))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {k: v for k, v in data.items() if k in known}
        cfg = cls(**data)
        # Backfill project_dir / project_name if the file was loaded from disk
        # at a path we recognize — surprising absences are common in older v1
        # configs.
        if not cfg.project_dir:
            cfg.project_dir = os.path.dirname(os.path.abspath(path))
        if not cfg.project_name and cfg.project_dir:
            cfg.project_name = os.path.basename(os.path.normpath(cfg.project_dir))
        return cfg


# =============================================================================
# Colormap LUT
# =============================================================================

_COLORMAP_ANCHORS = {
    'viridis': [
        [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
        [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
        [180, 222, 44], [253, 231, 37],
    ],
    'magma': [
        [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
        [181, 54, 122], [229, 89, 100], [251, 135, 97], [254, 194, 140],
        [254, 237, 176], [252, 253, 191],
    ],
    'inferno': [
        [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110],
        [147, 38, 103], [188, 55, 84], [221, 81, 58], [243, 120, 25],
        [249, 173, 10], [252, 255, 164],
    ],
}


def create_colormap_lut(name: str = 'viridis') -> np.ndarray:
    """Create a 256-entry RGB colormap lookup table.

    Matches the SpectrogramWidget._create_colormap_lut() implementation
    in usv_studio_pyqt.py to ensure visual consistency.

    Returns:
        np.ndarray of shape (256, 3) dtype uint8
    """
    anchors = _COLORMAP_ANCHORS.get(name, _COLORMAP_ANCHORS['viridis'])
    lut = np.zeros((256, 3), dtype=np.uint8)
    n = len(anchors)
    for i in range(256):
        frac = i / 255.0 * (n - 1)
        lo = int(frac)
        hi = min(lo + 1, n - 1)
        t = frac - lo
        for c in range(3):
            lut[i, c] = int(anchors[lo][c] * (1 - t) + anchors[hi][c] * t)
    return lut


# =============================================================================
# Deterministic Spectrogram Tile Renderer
# =============================================================================

def render_spectrogram_tile(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    tile_size: Tuple[int, int] = (640, 640),
    nperseg: int = 512,
    noverlap: int = 384,
    nfft: int = 1024,
    db_min: float = -100.0,
    db_max: float = -20.0,
    colormap_lut: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render a deterministic spectrogram tile for a time window.

    This is the SOLE renderer for both training tile export and inference.
    It uses fixed parameters (no adaptive normalization) to guarantee
    pixel-level consistency between training and inference.

    Args:
        audio: Full audio signal array (1D float)
        sr: Sample rate in Hz
        start_s: Start time of tile in seconds
        end_s: End time of tile in seconds
        tile_size: Output image dimensions (width, height)
        nperseg: FFT window size
        noverlap: FFT overlap
        nfft: FFT length (zero-padded)
        db_min: Minimum dB value for normalization (clipped below)
        db_max: Maximum dB value for normalization (clipped above)
        colormap_lut: 256x3 uint8 colormap LUT (default: viridis)

    Returns:
        Tuple of (image_rgb, frequencies, times):
        - image_rgb: np.ndarray shape (tile_h, tile_w, 3) uint8 RGB image
        - frequencies: 1D array of frequency values (Hz), low to high
        - times: 1D array of time values (seconds) relative to audio start
    """
    from PIL import Image

    if colormap_lut is None:
        colormap_lut = create_colormap_lut('viridis')

    tile_w, tile_h = tile_size

    # Extract audio segment
    start_sample = max(0, int(start_s * sr))
    end_sample = min(len(audio), int(end_s * sr))
    segment = audio[start_sample:end_sample]

    if len(segment) < nperseg:
        # Pad if segment is too short
        segment = np.pad(segment, (0, nperseg - len(segment)))

    # Compute spectrogram — full frequency range, no bandpass
    frequencies, times, Sxx_db = compute_spectrogram(
        segment, sr,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        min_freq=None, max_freq=None
    )

    # Offset times to absolute position
    times = times + start_s

    # Fixed dB normalization (NOT percentile-based)
    Sxx_normalized = np.clip(Sxx_db, db_min, db_max)
    Sxx_normalized = (Sxx_normalized - db_min) / (db_max - db_min)  # 0 to 1
    indices = (Sxx_normalized * 255).astype(np.uint8)

    # Apply colormap: indices shape is (n_freqs, n_times)
    # Flip vertically so high frequencies are at top (row 0)
    indices_flipped = indices[::-1, :]
    rgb_data = colormap_lut[indices_flipped]  # (n_freqs, n_times, 3)

    # Resize to tile dimensions using PIL for quality
    img = Image.fromarray(rgb_data, 'RGB')
    img = img.resize((tile_w, tile_h), Image.BILINEAR)
    image_rgb = np.array(img)

    return image_rgb, frequencies, times


# =============================================================================
# Coordinate Transforms
# =============================================================================

def detection_to_yolo_box(
    det_start_s: float,
    det_stop_s: float,
    det_min_freq: float,
    det_max_freq: float,
    tile_start_s: float,
    tile_end_s: float,
    freq_min: float,
    freq_max: float,
    class_id: int = 0,
) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Convert a detection in time/frequency space to YOLO normalized box.

    YOLO format: (class_id, x_center, y_center, width, height)
    All coordinates normalized to [0, 1] relative to the tile.

    Image convention: y=0 is top (high freq), y=1 is bottom (low freq).

    Args:
        det_start_s: Detection start time (seconds)
        det_stop_s: Detection stop time (seconds)
        det_min_freq: Detection minimum frequency (Hz)
        det_max_freq: Detection maximum frequency (Hz)
        tile_start_s: Tile start time (seconds)
        tile_end_s: Tile end time (seconds)
        freq_min: Tile minimum frequency (Hz), typically 0
        freq_max: Tile maximum frequency (Hz), typically Nyquist
        class_id: YOLO class ID (0 for 'usv_call')

    Returns:
        (class_id, x_center, y_center, width, height) or None if box outside tile
    """
    tile_duration = tile_end_s - tile_start_s
    freq_range = freq_max - freq_min

    if tile_duration <= 0 or freq_range <= 0:
        return None

    # Clip detection to tile bounds
    clipped_start = max(det_start_s, tile_start_s)
    clipped_stop = min(det_stop_s, tile_end_s)
    clipped_min_freq = max(det_min_freq, freq_min)
    clipped_max_freq = min(det_max_freq, freq_max)

    if clipped_start >= clipped_stop or clipped_min_freq >= clipped_max_freq:
        return None

    # Normalize to [0, 1] in image coordinates
    x1 = (clipped_start - tile_start_s) / tile_duration
    x2 = (clipped_stop - tile_start_s) / tile_duration

    # y axis: image top = high freq, bottom = low freq
    y1 = 1.0 - (clipped_max_freq - freq_min) / freq_range  # top of box
    y2 = 1.0 - (clipped_min_freq - freq_min) / freq_range  # bottom of box

    # YOLO center format
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1

    # Sanity: clamp to valid range
    x_center = np.clip(x_center, 0, 1)
    y_center = np.clip(y_center, 0, 1)
    width = np.clip(width, 0, 1)
    height = np.clip(height, 0, 1)

    if width < 0.001 or height < 0.001:
        return None

    return (class_id, float(x_center), float(y_center), float(width), float(height))


def yolo_box_to_detection(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    confidence: float,
    tile_start_s: float,
    tile_end_s: float,
    freq_min: float,
    freq_max: float,
) -> Dict:
    """
    Convert a YOLO predicted box back to time/frequency coordinates.

    Args:
        x_center, y_center, width, height: YOLO normalized coords [0, 1]
        confidence: Model confidence score
        tile_start_s: Tile start time (seconds)
        tile_end_s: Tile end time (seconds)
        freq_min: Tile minimum frequency (Hz)
        freq_max: Tile maximum frequency (Hz)

    Returns:
        Detection dict with start_seconds, stop_seconds, min_freq_hz, etc.
    """
    tile_duration = tile_end_s - tile_start_s
    freq_range = freq_max - freq_min

    start_s = tile_start_s + (x_center - width / 2) * tile_duration
    stop_s = tile_start_s + (x_center + width / 2) * tile_duration

    # Invert y: image y=0 is top (high freq)
    max_freq = freq_max - (y_center - height / 2) * freq_range
    min_freq = freq_max - (y_center + height / 2) * freq_range

    return {
        'start_seconds': round(start_s, 6),
        'stop_seconds': round(stop_s, 6),
        'duration_ms': round((stop_s - start_s) * 1000, 3),
        'min_freq_hz': round(max(min_freq, freq_min), 1),
        'max_freq_hz': round(min(max_freq, freq_max), 1),
        'peak_freq_hz': round((min_freq + max_freq) / 2, 1),
        'freq_bandwidth_hz': round(max_freq - min_freq, 1),
        'max_power_db': 0.0,
        'mean_power_db': 0.0,
        'confidence': round(confidence, 4),
        'status': 'pending',
        'source': 'ml',
    }


# =============================================================================
# Training Data Export
# =============================================================================

def _cluster_detections_by_time(
    detections: List[Dict],
    padding_s: float = 0.05,
) -> List[List[Dict]]:
    """
    Group detections that overlap or are close in time into clusters.

    Each cluster will become one training tile containing all its detections.

    Args:
        detections: List of detection dicts (must have start_seconds, stop_seconds)
        padding_s: Time padding around each detection

    Returns:
        List of clusters, each cluster is a list of detection dicts
    """
    if not detections:
        return []

    # Sort by start time
    sorted_dets = sorted(detections, key=lambda d: d['start_seconds'])

    clusters = []
    current_cluster = [sorted_dets[0]]
    cluster_end = sorted_dets[0]['stop_seconds'] + padding_s

    for det in sorted_dets[1:]:
        det_start = det['start_seconds'] - padding_s
        if det_start <= cluster_end:
            # Overlaps with current cluster
            current_cluster.append(det)
            cluster_end = max(cluster_end, det['stop_seconds'] + padding_s)
        else:
            # New cluster
            clusters.append(current_cluster)
            current_cluster = [det]
            cluster_end = det['stop_seconds'] + padding_s

    clusters.append(current_cluster)
    return clusters


def export_training_data(
    audio_files: List[str],
    all_detections: Dict[str, pd.DataFrame],
    output_dir: str,
    config: YOLOProjectConfig,
    progress_callback=None,
) -> Dict[str, int]:
    """
    Export labeled detections as YOLO-format training data.

    Only generates tiles from time regions where the user has actively labeled.
    Accepted detections become positive annotations; negative-marked regions
    become pure background tiles. Unlabeled regions are never included.

    Args:
        audio_files: List of audio file paths
        all_detections: Dict mapping filepath -> detections DataFrame
        output_dir: Base directory for dataset (will contain images/ and labels/)
        config: Project configuration
        progress_callback: Optional callback(message: str, current: int, total: int)

    Returns:
        Dict with counts: {n_positive, n_negative, n_tiles, n_files}
    """
    from PIL import Image

    val_fraction = float(getattr(config, 'val_fraction', 0.20))
    val_fraction = max(0.0, min(0.5, val_fraction))

    split_dirs = {
        'train': {
            'images': os.path.join(output_dir, 'train', 'images'),
            'labels': os.path.join(output_dir, 'train', 'labels'),
        },
        'val': {
            'images': os.path.join(output_dir, 'val', 'images'),
            'labels': os.path.join(output_dir, 'val', 'labels'),
        },
    }
    for split in split_dirs.values():
        os.makedirs(split['images'], exist_ok=True)
        os.makedirs(split['labels'], exist_ok=True)

    colormap_lut = create_colormap_lut(config.colormap)

    tile_idx = 0
    total_positive = 0
    total_negative = 0
    files_with_labels = 0
    n_train = 0
    n_val = 0

    total_files = len(audio_files)

    for file_i, filepath in enumerate(audio_files):
        if filepath not in all_detections:
            continue

        df = all_detections[filepath]
        if df is None or len(df) == 0:
            continue

        # Get accepted (positive) and negative/rejected detections
        # Rejected = DSP false positives (hard negatives), Negative = user-marked background
        positive_mask = df['status'] == 'accepted'
        negative_mask = df['status'].isin(['negative', 'rejected'])

        positive_dets = df[positive_mask].to_dict('records')
        negative_dets = df[negative_mask].to_dict('records')

        if not positive_dets and not negative_dets:
            continue

        files_with_labels += 1

        if progress_callback:
            progress_callback(
                f"Loading {os.path.basename(filepath)}...",
                file_i, total_files
            )

        # Load audio
        try:
            audio, sr = load_audio(filepath)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

        nyquist = sr / 2.0
        total_duration = len(audio) / sr

        # --- Build the full list of tile work-items for this file ---
        # Each item: (kind, tile_start, tile_end, cluster_or_none)
        #   kind='pos' with cluster, or kind='neg' with cluster=None
        tile_items: List[Tuple[str, float, float, Optional[List[Dict]]]] = []

        if positive_dets:
            clusters = _cluster_detections_by_time(positive_dets, config.tile_padding_s)

            for cluster in clusters:
                cluster_start = min(d['start_seconds'] for d in cluster) - config.tile_padding_s
                cluster_stop = max(d['stop_seconds'] for d in cluster) + config.tile_padding_s

                cluster_duration = cluster_stop - cluster_start
                if cluster_duration < config.tile_time_window_s:
                    center = (cluster_start + cluster_stop) / 2
                    cluster_start = center - config.tile_time_window_s / 2
                    cluster_stop = center + config.tile_time_window_s / 2

                cluster_start = max(0, cluster_start)
                cluster_stop = min(total_duration, cluster_stop)

                tile_starts = []
                if cluster_stop - cluster_start <= config.tile_time_window_s * 1.5:
                    tile_starts.append(cluster_start)
                else:
                    step = config.tile_time_window_s * (1 - config.tile_overlap_fraction)
                    t = cluster_start
                    while t < cluster_stop - config.tile_time_window_s * 0.5:
                        tile_starts.append(t)
                        t += step

                for tile_start in tile_starts:
                    tile_end = min(tile_start + config.tile_time_window_s, total_duration)
                    tile_items.append(('pos', tile_start, tile_end, cluster))

        for neg_det in negative_dets:
            neg_start = neg_det['start_seconds'] - config.tile_padding_s
            neg_stop = neg_det['stop_seconds'] + config.tile_padding_s

            neg_duration = neg_stop - neg_start
            if neg_duration < config.tile_time_window_s:
                center = (neg_start + neg_stop) / 2
                neg_start = center - config.tile_time_window_s / 2
                neg_stop = center + config.tile_time_window_s / 2

            neg_start = max(0, neg_start)
            neg_stop = min(total_duration, neg_stop)
            tile_items.append(('neg', neg_start, neg_stop, None))

        # --- Stratified per-file train/val split (reproducible) ---
        n_items = len(tile_items)
        val_indices: set = set()
        if val_fraction > 0 and n_items >= 2:
            n_val_for_file = max(1, int(round(n_items * val_fraction)))
            n_val_for_file = min(n_val_for_file, n_items - 1)  # always keep at least one train
            rng = random.Random(hash(filepath) & 0xFFFFFFFF)
            shuffled = list(range(n_items))
            rng.shuffle(shuffled)
            val_indices = set(shuffled[:n_val_for_file])

        # --- Render + write each tile ---
        for local_idx, (kind, tile_start, tile_end, cluster) in enumerate(tile_items):
            split = 'val' if local_idx in val_indices else 'train'

            image_rgb, freqs, times = render_spectrogram_tile(
                audio, sr, tile_start, tile_end,
                tile_size=config.tile_size,
                nperseg=config.nperseg, noverlap=config.noverlap,
                nfft=config.nfft,
                db_min=config.db_min, db_max=config.db_max,
                colormap_lut=colormap_lut,
            )

            annotations = []
            if kind == 'pos' and cluster is not None:
                for det in cluster:
                    box = detection_to_yolo_box(
                        det['start_seconds'], det['stop_seconds'],
                        det['min_freq_hz'], det['max_freq_hz'],
                        tile_start, tile_end, 0, nyquist,
                        class_id=0,
                    )
                    if box is not None:
                        annotations.append(box)
                        total_positive += 1
            else:
                total_negative += 1

            tile_name = f"tile_{tile_idx:06d}"
            img = Image.fromarray(image_rgb, 'RGB')
            img.save(os.path.join(split_dirs[split]['images'], f"{tile_name}.png"))

            label_path = os.path.join(split_dirs[split]['labels'], f"{tile_name}.txt")
            with open(label_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

            tile_idx += 1
            if split == 'val':
                n_val += 1
            else:
                n_train += 1

    if progress_callback:
        progress_callback("Export complete", total_files, total_files)

    return {
        'n_positive': total_positive,
        'n_negative': total_negative,
        'n_tiles': tile_idx,
        'n_train': n_train,
        'n_val': n_val,
        'n_files': files_with_labels,
    }


def write_yolo_dataset_yaml(project_dir: str, dataset_dir: str) -> str:
    """
    Write the data.yaml file required by ultralytics for training.

    Args:
        project_dir: Root project directory
        dataset_dir: Path to dataset directory containing images/ and labels/

    Returns:
        Path to the written data.yaml file
    """
    yaml_path = os.path.join(project_dir, 'data.yaml')

    # Prefer new split layout (train/images, val/images). Fall back to the
    # legacy flat layout (images/) when that's all that exists so old projects
    # keep working without a re-export.
    train_images = os.path.join(dataset_dir, 'train', 'images')
    val_images = os.path.join(dataset_dir, 'val', 'images')
    if os.path.isdir(train_images) and os.path.isdir(val_images):
        train_rel = 'train/images'
        val_rel = 'val/images'
    else:
        train_rel = 'images'
        val_rel = 'images'

    content = {
        'path': os.path.abspath(dataset_dir),
        'train': train_rel,
        'val': val_rel,
        'nc': 1,
        'names': ['usv_call'],
    }
    # Write as YAML manually (avoid pyyaml dependency)
    with open(yaml_path, 'w') as f:
        for key, value in content.items():
            if isinstance(value, list):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, int):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")

    return yaml_path


# =============================================================================
# YOLO Training
# =============================================================================

def train_yolo_model(
    dataset_yaml: str,
    output_dir: str,
    model_name: str,
    device: str = 'auto',
    pretrained_weights: Optional[str] = None,
    progress_callback=None,
    epoch_callback=None,
    model_variant: str = 'yolo11s.pt',
    imgsz: int = 1280,
) -> str:
    """
    Train a YOLO model for USV detection.

    Uses automatic early stopping (SLEAP-style): training runs up to a high
    epoch ceiling but stops when validation loss plateaus for `patience` epochs.

    Augmentations are tuned for spectrograms: frequency-axis flip (flipud) and
    mosaic are disabled because the vertical axis has physical meaning and
    mosaic splices distort call boundaries. Time-axis flip (fliplr) is also
    disabled because FM-sweep direction is semantically meaningful.

    Args:
        dataset_yaml: Path to data.yaml
        output_dir: Directory to save model (e.g., models/fntUSVStudioModel_n=30/)
        model_name: Name for this training run
        device: PyTorch device ('auto', 'cuda:0', 'mps', 'cpu')
        pretrained_weights: Path to previous best.pt for fine-tuning,
                           or None to start from `model_variant` (ultralytics
                           auto-downloads if not cached)
        progress_callback: Optional callback(epoch: int, total_epochs: int, metrics: dict)
        model_variant: Ultralytics weights identifier (e.g. 'yolo11n.pt',
                       'yolo11s.pt', 'yolo11m.pt', 'yolov8n.pt')
        imgsz: Training image size; 1280 is the community standard for
               small-object detection on spectrograms

    Returns:
        Path to best.pt weights file
    """
    from ultralytics import YOLO

    # Determine device
    if device == 'auto':
        from .gpu_utils import get_best_device
        device = get_best_device()

    # Load model
    if pretrained_weights and os.path.exists(pretrained_weights):
        model = YOLO(pretrained_weights)
    else:
        model = YOLO(model_variant)

    # Register an epoch-end callback so the GUI can plot live metrics.
    # `trainer.metrics` includes box/cls/dfl losses and mAP50, mAP50-95, P, R.
    # `trainer.loss_items` adds raw per-epoch train losses we can surface.
    if epoch_callback is not None:
        def _on_fit_epoch_end(trainer):
            try:
                metrics = dict(getattr(trainer, 'metrics', {}) or {})
                loss_items = getattr(trainer, 'loss_items', None)
                if loss_items is not None:
                    names = getattr(trainer, 'loss_names', None) or (
                        'box_loss', 'cls_loss', 'dfl_loss'
                    )
                    try:
                        vals = loss_items.detach().cpu().tolist()
                    except Exception:
                        vals = list(loss_items)
                    for name, val in zip(names, vals):
                        metrics.setdefault(f'train/{name}', float(val))
                epoch_callback(int(getattr(trainer, 'epoch', 0)), metrics)
            except Exception:
                # Never let a plotting glitch take down training.
                pass

        try:
            model.add_callback('on_fit_epoch_end', _on_fit_epoch_end)
        except Exception:
            pass

    # Train with high ceiling + early stopping on loss plateau
    os.makedirs(output_dir, exist_ok=True)
    results = model.train(
        data=dataset_yaml,
        epochs=500,           # High ceiling — early stopping controls actual duration
        imgsz=imgsz,
        device=device,
        project=output_dir,
        name='train',
        exist_ok=True,
        verbose=False,
        batch=-1,             # Auto batch size
        patience=20,          # Stop after 20 epochs with no improvement
        save=True,
        plots=False,
        # Spectrogram-appropriate augmentations
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
    )

    # Return path to best weights
    best_path = os.path.join(output_dir, 'train', 'weights', 'best.pt')
    if not os.path.exists(best_path):
        # Fallback to last.pt
        best_path = os.path.join(output_dir, 'train', 'weights', 'last.pt')

    return best_path


# =============================================================================
# Power Measurement
# =============================================================================

def _measure_detection_power(
    audio: np.ndarray,
    sr: int,
    start_s: float,
    stop_s: float,
    min_freq_hz: float,
    max_freq_hz: float,
    nperseg: int = 512,
    noverlap: int = 384,
    nfft: int = 1024,
) -> Tuple[float, float]:
    """
    Measure actual dB power within a detection's time/frequency bounds.

    Args:
        audio: Full audio signal array
        sr: Sample rate
        start_s, stop_s: Detection time bounds (seconds)
        min_freq_hz, max_freq_hz: Detection frequency bounds (Hz)
        nperseg, noverlap, nfft: Spectrogram parameters

    Returns:
        (max_power_db, mean_power_db)
    """
    start_sample = max(0, int(start_s * sr))
    end_sample = min(len(audio), int(stop_s * sr))
    segment = audio[start_sample:end_sample]

    if len(segment) < nperseg:
        segment = np.pad(segment, (0, nperseg - len(segment)))

    frequencies, times, Sxx_db = compute_spectrogram(
        segment, sr,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        min_freq=min_freq_hz, max_freq=max_freq_hz,
    )

    if Sxx_db.size == 0:
        return 0.0, 0.0

    return float(np.max(Sxx_db)), float(np.mean(Sxx_db))


# =============================================================================
# YOLO Inference
# =============================================================================

def _run_sahi_inference(
    model_path: str,
    audio: np.ndarray,
    sr: int,
    config: YOLOProjectConfig,
    conf: float,
    device: str,
    progress_callback=None,
) -> List[Dict]:
    """
    Run SAHI (Slicing Aided Hyper Inference) on a full audio recording.

    Renders the recording as one or more long spectrogram segments at the same
    pixels-per-second density as training tiles, then lets SAHI slice each
    segment into tile-sized windows for prediction and merge results.

    This typically boosts small-object recall vs. naive tile loops
    (SAHI paper: 31.8% -> 86.4% on COCO small-object benchmarks).
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
        image_size=max(config.tile_size),
    )

    nyquist = sr / 2.0
    total_duration = len(audio) / sr
    colormap_lut = create_colormap_lut(config.colormap)

    tile_w, tile_h = config.tile_size
    # Pixels-per-second derived from training tile density
    px_per_sec = tile_w / config.tile_time_window_s
    # Cap segment pixel width to keep peak RAM reasonable
    max_segment_px = 8192
    segment_duration = max(
        config.tile_time_window_s,
        min(30.0, max_segment_px / px_per_sec),
    )
    segment_px = max(tile_w, int(round(segment_duration * px_per_sec)))

    # Overlap between adjacent segments so calls that straddle a boundary
    # get picked up on at least one side
    overlap_s = config.tile_time_window_s * config.tile_overlap_fraction
    step_s = max(segment_duration - overlap_s, config.tile_time_window_s)

    seg_starts: List[float] = []
    t = 0.0
    while True:
        seg_starts.append(t)
        if t + segment_duration >= total_duration:
            break
        t += step_s

    all_detections: List[Dict] = []
    n_segments = len(seg_starts)

    for i, seg_start in enumerate(seg_starts):
        if progress_callback:
            progress_callback((i + 1) / n_segments)

        seg_end = min(seg_start + segment_duration, total_duration)
        seg_dur = seg_end - seg_start
        seg_w_px = max(tile_w, int(round(seg_dur * px_per_sec)))

        image_rgb, _freqs, _times = render_spectrogram_tile(
            audio, sr, seg_start, seg_end,
            tile_size=(seg_w_px, tile_h),
            nperseg=config.nperseg, noverlap=config.noverlap,
            nfft=config.nfft,
            db_min=config.db_min, db_max=config.db_max,
            colormap_lut=colormap_lut,
        )

        result = get_sliced_prediction(
            image_rgb,
            detection_model,
            slice_height=tile_h,
            slice_width=tile_w,
            overlap_height_ratio=0.0,  # Full frequency range covered by tile height
            overlap_width_ratio=config.tile_overlap_fraction,
            postprocess_type='NMS',
            postprocess_match_threshold=config.iou_threshold,
            verbose=0,
        )

        img_h, img_w = image_rgb.shape[:2]
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            x_center = ((bbox.minx + bbox.maxx) / 2.0) / img_w
            y_center = ((bbox.miny + bbox.maxy) / 2.0) / img_h
            width = (bbox.maxx - bbox.minx) / img_w
            height = (bbox.maxy - bbox.miny) / img_h

            det = yolo_box_to_detection(
                x_center, y_center, width, height,
                float(pred.score.value),
                seg_start, seg_end, 0, nyquist,
            )
            all_detections.append(det)

    return all_detections


def _run_tiled_inference(
    model_path: str,
    audio: np.ndarray,
    sr: int,
    config: YOLOProjectConfig,
    conf: float,
    device: str,
    progress_callback=None,
) -> List[Dict]:
    """Legacy tiled inference path (pre-SAHI). Kept as a fallback."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    nyquist = sr / 2.0
    total_duration = len(audio) / sr
    colormap_lut = create_colormap_lut(config.colormap)

    tile_window = config.tile_time_window_s
    step = tile_window * (1 - config.tile_overlap_fraction)

    tile_starts: List[float] = []
    t = 0.0
    while t < total_duration - tile_window * 0.25:
        tile_starts.append(t)
        t += step
    if not tile_starts:
        tile_starts.append(0.0)
    if total_duration - tile_starts[-1] > tile_window * 0.25:
        tile_starts.append(max(0.0, total_duration - tile_window))

    all_detections: List[Dict] = []
    n_tiles = len(tile_starts)

    for i, tile_start in enumerate(tile_starts):
        tile_end = min(tile_start + tile_window, total_duration)

        if progress_callback:
            progress_callback((i + 1) / n_tiles)

        image_rgb, _freqs, _times = render_spectrogram_tile(
            audio, sr, tile_start, tile_end,
            tile_size=config.tile_size,
            nperseg=config.nperseg, noverlap=config.noverlap,
            nfft=config.nfft,
            db_min=config.db_min, db_max=config.db_max,
            colormap_lut=colormap_lut,
        )

        results = model.predict(
            image_rgb,
            conf=conf,
            iou=config.iou_threshold,
            imgsz=max(config.tile_size),
            device=device,
            verbose=False,
        )

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxyn = box.xyxyn[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())

                    x_center = (xyxyn[0] + xyxyn[2]) / 2
                    y_center = (xyxyn[1] + xyxyn[3]) / 2
                    width = xyxyn[2] - xyxyn[0]
                    height = xyxyn[3] - xyxyn[1]

                    det = yolo_box_to_detection(
                        x_center, y_center, width, height, confidence,
                        tile_start, tile_end, 0, nyquist,
                    )
                    all_detections.append(det)

    return all_detections


def run_yolo_inference(
    model_path: str,
    audio_path: str,
    config: YOLOProjectConfig,
    confidence_threshold: Optional[float] = None,
    progress_callback=None,
) -> List[Dict]:
    """
    Run YOLO inference on an audio file.

    Uses SAHI tiled inference by default (config.use_sahi=True) which boosts
    small-object recall. Falls back to the legacy per-tile loop otherwise.

    Args:
        model_path: Path to trained YOLO weights (best.pt)
        audio_path: Path to audio file
        config: Project configuration (must match training config)
        confidence_threshold: Override config threshold if provided
        progress_callback: Optional callback(fraction: float)

    Returns:
        List of detection dicts ready for detections_df
    """
    from .gpu_utils import get_best_device

    conf = confidence_threshold if confidence_threshold is not None else config.confidence_threshold
    device = get_best_device()

    audio, sr = load_audio(audio_path)

    use_sahi = getattr(config, 'use_sahi', True)
    if use_sahi:
        try:
            all_detections = _run_sahi_inference(
                model_path, audio, sr, config, conf, device, progress_callback,
            )
        except ImportError:
            # SAHI not installed — fall back silently
            all_detections = _run_tiled_inference(
                model_path, audio, sr, config, conf, device, progress_callback,
            )
    else:
        all_detections = _run_tiled_inference(
            model_path, audio, sr, config, conf, device, progress_callback,
        )

    # Cross-segment/tile NMS: remove duplicates from overlapping windows
    all_detections = _nms_temporal(all_detections, iou_threshold=config.iou_threshold)

    # Add call numbers and measure real power
    all_detections.sort(key=lambda d: d['start_seconds'])
    for i, det in enumerate(all_detections):
        det['call_number'] = i + 1
        max_pwr, mean_pwr = _measure_detection_power(
            audio, sr,
            det['start_seconds'], det['stop_seconds'],
            det['min_freq_hz'], det['max_freq_hz'],
            nperseg=config.nperseg, noverlap=config.noverlap, nfft=config.nfft,
        )
        det['max_power_db'] = round(max_pwr, 2)
        det['mean_power_db'] = round(mean_pwr, 2)

    return all_detections


def _nms_temporal(detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
    """
    Non-maximum suppression based on temporal IoU.

    When overlapping inference tiles produce duplicate detections,
    this keeps the highest-confidence one from each overlapping group.

    Args:
        detections: List of detection dicts with start_seconds, stop_seconds, confidence
        iou_threshold: IoU threshold for merging

    Returns:
        Filtered list of detections
    """
    if not detections:
        return []

    # Sort by confidence (highest first)
    dets = sorted(detections, key=lambda d: d.get('confidence', 0), reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)

        remaining = []
        for det in dets:
            iou = _compute_iou(best, det)
            if iou < iou_threshold:
                remaining.append(det)
        dets = remaining

    return keep


def _compute_iou(det_a: Dict, det_b: Dict) -> float:
    """Compute 2D IoU between two detections (time x frequency)."""
    # Time overlap
    t_start = max(det_a['start_seconds'], det_b['start_seconds'])
    t_end = min(det_a['stop_seconds'], det_b['stop_seconds'])
    t_overlap = max(0, t_end - t_start)

    # Frequency overlap
    f_start = max(det_a['min_freq_hz'], det_b['min_freq_hz'])
    f_end = min(det_a['max_freq_hz'], det_b['max_freq_hz'])
    f_overlap = max(0, f_end - f_start)

    intersection = t_overlap * f_overlap

    area_a = (det_a['stop_seconds'] - det_a['start_seconds']) * \
             (det_a['max_freq_hz'] - det_a['min_freq_hz'])
    area_b = (det_b['stop_seconds'] - det_b['start_seconds']) * \
             (det_b['max_freq_hz'] - det_b['min_freq_hz'])

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# =============================================================================
# Project Management
# =============================================================================

def default_projects_root() -> str:
    """Return the default root directory for DAD projects (``~/fnt_projects``)."""
    return os.path.join(os.path.expanduser('~'), 'fnt_projects')


def create_project(
    project_dir: str,
    config: Optional[YOLOProjectConfig] = None,
    source_folders: Optional[List[str]] = None,
) -> YOLOProjectConfig:
    """
    Create a new DAD project at ``project_dir``.

    Layout:
        <project_dir>/
            project_info.json
            datasets/               # exported YOLO tiles (regenerated each train)
            models/                 # per-run ultralytics outputs

    Labels are NOT stored here — they live as sibling CSVs next to each wav
    (see :mod:`fnt.usv.usv_detector.labels_io`).

    Args:
        project_dir: Absolute directory path for the new project.
        config: Optional starting config; a default one is created otherwise.
        source_folders: Optional list of wav folders to seed the project with.
    """
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'datasets'), exist_ok=True)

    if config is None:
        config = YOLOProjectConfig()
    config.project_dir = project_dir
    config.project_name = os.path.basename(os.path.normpath(project_dir))
    if source_folders:
        for folder in source_folders:
            if folder and folder not in config.source_folders:
                config.source_folders.append(folder)
    config.save()
    return config


def collect_project_labels(
    config: YOLOProjectConfig,
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """Read every sibling-CSV under this project's source folders.

    Returns ``(audio_files, all_detections)`` suitable for passing to
    :func:`export_training_data` or :func:`get_training_data_counts`. Wavs
    without a sibling CSV are included in ``audio_files`` with an empty frame
    so the caller can still see them.
    """
    from .labels_io import load_labels, scan_folders_for_wavs

    wavs = scan_folders_for_wavs(config.source_folders)
    all_detections: Dict[str, pd.DataFrame] = {}
    audio_files: List[str] = []
    for wav in wavs:
        path = str(wav)
        audio_files.append(path)
        df = load_labels(wav)
        if df is not None and len(df) > 0:
            all_detections[path] = df
    return audio_files, all_detections


def get_training_data_counts(
    all_detections: Dict[str, pd.DataFrame],
) -> Dict[str, int]:
    """
    Count positive and negative labeled examples across all files.

    Args:
        all_detections: Dict mapping filepath -> detections DataFrame

    Returns:
        Dict with n_positive, n_negative, n_files_with_labels
    """
    n_positive = 0
    n_negative = 0
    n_files = 0

    for filepath, df in all_detections.items():
        if df is None or len(df) == 0:
            continue

        file_pos = (df['status'] == 'accepted').sum()
        file_neg = df['status'].isin(['negative', 'rejected']).sum()

        if file_pos > 0 or file_neg > 0:
            n_positive += file_pos
            n_negative += file_neg
            n_files += 1

    return {
        'n_positive': int(n_positive),
        'n_negative': int(n_negative),
        'n_files_with_labels': n_files,
    }
