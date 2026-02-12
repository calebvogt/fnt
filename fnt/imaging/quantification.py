"""
Image Quantification Engine for CZI microscopy images.

Provides automated cell counting, area measurement, and intensity analysis
using scikit-image segmentation. No Qt dependency - pure numpy/scikit-image.

Supports single-channel and multi-channel analysis with colocalization
metrics and ROI-based density measurement.
"""

from dataclasses import dataclass, field, asdict
from itertools import combinations
from typing import Dict, List, Optional, Tuple
import json

import numpy as np


@dataclass
class ParticleResult:
    """Measurements for a single detected particle/cell."""
    label: int
    area_px: float
    area_um2: Optional[float]
    centroid_y: float
    centroid_x: float
    mean_intensity: float
    integrated_intensity: float
    perimeter_px: float
    circularity: float  # 4*pi*area / perimeter^2
    bbox: Tuple[int, int, int, int]  # min_row, min_col, max_row, max_col


@dataclass
class QuantificationResult:
    """Complete results from a quantification analysis run."""
    # Summary metrics
    particle_count: int
    total_area_px: float
    total_area_um2: Optional[float]
    mean_area_px: float
    mean_area_um2: Optional[float]
    mean_intensity: float
    area_fraction: float
    integrated_density: float

    # Per-particle data
    particles: List[ParticleResult] = field(default_factory=list)

    # Analysis parameters (for reproducibility)
    threshold_method: str = ""
    threshold_value: float = 0.0
    min_area_um2: float = 0.0
    max_area_um2: float = float('inf')
    watershed_used: bool = False
    pixel_size_um: Optional[float] = None
    channel_index: int = 0
    channel_name: str = ""
    bg_subtract_applied: str = "none"

    # Intermediate data for overlay (not serialized)
    binary_mask: Optional[np.ndarray] = field(default=None, repr=False)
    label_image: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize to dict for export (excludes numpy arrays)."""
        d = {
            'particle_count': self.particle_count,
            'total_area_px': self.total_area_px,
            'total_area_um2': self.total_area_um2,
            'mean_area_px': self.mean_area_px,
            'mean_area_um2': self.mean_area_um2,
            'mean_intensity': self.mean_intensity,
            'area_fraction': self.area_fraction,
            'integrated_density': self.integrated_density,
            'threshold_method': self.threshold_method,
            'threshold_value': self.threshold_value,
            'min_area_um2': self.min_area_um2,
            'max_area_um2': self.max_area_um2,
            'watershed_used': self.watershed_used,
            'pixel_size_um': self.pixel_size_um,
            'channel_index': self.channel_index,
            'channel_name': self.channel_name,
            'bg_subtract_applied': self.bg_subtract_applied,
        }
        return d


@dataclass
class QuantificationConfig:
    """Configuration for a quantification run."""
    channel_index: int = 0
    threshold_method: str = "otsu"  # "otsu", "manual", "triangle", "li"
    manual_threshold: float = 0.5
    min_area_um2: float = 10.0
    max_area_um2: float = 10000.0
    use_watershed: bool = False
    apply_bg_subtraction: bool = True
    roi: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h or None for full image

    def to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'QuantificationConfig':
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Convert roi list back to tuple if present
        if data.get('roi') is not None:
            data['roi'] = tuple(data['roi'])
        return cls(**data)


@dataclass
class ROIDefinition:
    """A named rectangular ROI for density analysis."""
    label: str
    x: int
    y: int
    w: int
    h: int
    color: str = "white"


@dataclass
class MultiChannelConfig:
    """Configuration for multi-channel quantification."""
    channel_configs: Dict[int, QuantificationConfig] = field(default_factory=dict)
    roi_definitions: List[ROIDefinition] = field(default_factory=list)


@dataclass
class ROIChannelMetrics:
    """Per-channel metrics within an ROI."""
    channel_idx: int
    channel_name: str
    particle_count: int
    area_px: float
    area_um2: Optional[float]
    density_per_mm2: Optional[float]
    area_fraction: float
    mean_intensity: float


@dataclass
class ROIDensityResult:
    """Density metrics for a single ROI region."""
    roi: ROIDefinition
    channel_results: Dict[int, ROIChannelMetrics] = field(default_factory=dict)


@dataclass
class ColocalizationResult:
    """Colocalization metrics between two channels."""
    channel_a_idx: int
    channel_a_name: str
    channel_b_idx: int
    channel_b_name: str
    # Centroid-based overlap
    a_in_b_count: int
    a_in_b_percent: float
    b_in_a_count: int
    b_in_a_percent: float
    # Mask overlap
    overlap_area_px: float
    overlap_area_um2: Optional[float]
    dice_coefficient: float
    # Cross-channel intensity
    mean_a_intensity_in_b: float
    mean_b_intensity_in_a: float


@dataclass
class MultiChannelResult:
    """Complete results from multi-channel quantification."""
    channel_results: Dict[int, QuantificationResult] = field(default_factory=dict)
    colocalizations: List[ColocalizationResult] = field(default_factory=list)
    roi_densities: List[ROIDensityResult] = field(default_factory=list)
    pixel_size_um: Optional[float] = None


class ImageQuantifier:
    """
    Quantification engine for fluorescence microscopy images.

    Operates on normalized float32 images (0-1 range).
    All scikit-image operations are contained here.
    """

    def compute_threshold(self, image: np.ndarray,
                          method: str = "otsu") -> float:
        """
        Compute automatic threshold value.

        Args:
            image: Normalized float32 image (0-1)
            method: "otsu", "triangle", or "li"

        Returns:
            Threshold value in 0-1 range
        """
        from skimage.filters import threshold_otsu, threshold_triangle, threshold_li

        # Use nonzero pixels for better threshold estimation
        nonzero = image[image > 0]
        if len(nonzero) == 0:
            return 0.5

        if method == "triangle":
            return float(threshold_triangle(nonzero))
        elif method == "li":
            return float(threshold_li(nonzero))
        else:
            # Default to Otsu
            return float(threshold_otsu(nonzero))

    def threshold_image(self, image: np.ndarray,
                        threshold_value: float) -> np.ndarray:
        """
        Create binary mask from threshold.

        Args:
            image: Normalized float32 image (0-1)
            threshold_value: Threshold in 0-1 range

        Returns:
            Binary mask (bool ndarray)
        """
        return image > threshold_value

    def apply_watershed(self, binary_mask: np.ndarray,
                        image: np.ndarray) -> np.ndarray:
        """
        Apply watershed segmentation to separate touching objects.

        Args:
            binary_mask: Binary mask of foreground
            image: Original intensity image (for distance transform weighting)

        Returns:
            Label image where each connected component has a unique integer
        """
        from scipy import ndimage
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max

        distance = ndimage.distance_transform_edt(binary_mask)

        # Adaptive min_distance based on typical object size
        n_objects = max(1, ndimage.label(binary_mask)[1])
        avg_area = binary_mask.sum() / n_objects
        min_distance = max(5, int(np.sqrt(avg_area) * 0.3))

        coords = peak_local_max(
            distance, min_distance=min_distance, labels=binary_mask
        )
        if len(coords) == 0:
            # No peaks found, fall back to connected components
            labels, _ = ndimage.label(binary_mask)
            return labels

        mask_seeds = np.zeros(distance.shape, dtype=bool)
        mask_seeds[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask_seeds)

        labels = watershed(-distance, markers, mask=binary_mask)
        return labels

    def detect_and_measure(self, image: np.ndarray,
                           config: QuantificationConfig,
                           pixel_size_um: Optional[float] = None
                           ) -> QuantificationResult:
        """
        Full detection and measurement pipeline.

        Args:
            image: Normalized float32 image (0-1), optionally BG-subtracted
            config: Quantification parameters
            pixel_size_um: Pixel size for calibrated measurements

        Returns:
            QuantificationResult with all measurements
        """
        from skimage.measure import label, regionprops
        from scipy import ndimage

        full_shape = image.shape

        # Optionally crop to ROI
        roi_image = image
        roi_offset_y, roi_offset_x = 0, 0
        total_roi_area_px = image.shape[0] * image.shape[1]

        if config.roi is not None:
            x, y, w, h = config.roi
            # Clamp to image bounds
            x = max(0, min(x, full_shape[1] - 1))
            y = max(0, min(y, full_shape[0] - 1))
            w = min(w, full_shape[1] - x)
            h = min(h, full_shape[0] - y)
            if w <= 0 or h <= 0:
                return self._empty_result(config, pixel_size_um, full_shape)
            roi_image = image[y:y+h, x:x+w]
            roi_offset_y, roi_offset_x = y, x
            total_roi_area_px = w * h

        # Compute threshold
        if config.threshold_method == "manual":
            thresh_val = config.manual_threshold
        else:
            thresh_val = self.compute_threshold(roi_image, config.threshold_method)

        # Create binary mask
        binary_mask = self.threshold_image(roi_image, thresh_val)

        # Label connected components or apply watershed
        if config.use_watershed and binary_mask.any():
            label_image = self.apply_watershed(binary_mask, roi_image)
        else:
            label_image, _ = ndimage.label(binary_mask)

        # Pixel-to-um conversion
        px_to_um2 = (pixel_size_um ** 2) if pixel_size_um and pixel_size_um > 0 else None

        # Size filter bounds in pixels
        if px_to_um2:
            min_area_px = config.min_area_um2 / px_to_um2
            max_area_px = config.max_area_um2 / px_to_um2
        else:
            min_area_px = config.min_area_um2
            max_area_px = config.max_area_um2

        # Measure properties
        props = regionprops(label_image, intensity_image=roi_image)

        particles = []
        for prop in props:
            area_px = prop.area

            # Size filter
            if area_px < min_area_px or area_px > max_area_px:
                label_image[label_image == prop.label] = 0
                continue

            area_um2 = area_px * px_to_um2 if px_to_um2 else None
            perimeter = prop.perimeter if hasattr(prop, 'perimeter') else 0.0
            circularity = (4 * np.pi * area_px / (perimeter ** 2)) if perimeter > 0 else 0.0

            # Use intensity_mean (skimage >=0.26) with fallback to mean_intensity
            if hasattr(prop, 'intensity_mean'):
                mean_int = float(prop.intensity_mean)
            else:
                mean_int = float(prop.mean_intensity)

            particles.append(ParticleResult(
                label=prop.label,
                area_px=float(area_px),
                area_um2=float(area_um2) if area_um2 is not None else None,
                centroid_y=float(prop.centroid[0] + roi_offset_y),
                centroid_x=float(prop.centroid[1] + roi_offset_x),
                mean_intensity=mean_int,
                integrated_intensity=mean_int * area_px,
                perimeter_px=float(perimeter),
                circularity=float(circularity),
                bbox=(
                    int(prop.bbox[0] + roi_offset_y),
                    int(prop.bbox[1] + roi_offset_x),
                    int(prop.bbox[2] + roi_offset_y),
                    int(prop.bbox[3] + roi_offset_x),
                ),
            ))

        # Summary statistics
        count = len(particles)
        total_area_px = sum(p.area_px for p in particles)
        total_area_um2 = sum(p.area_um2 for p in particles) if px_to_um2 else None
        mean_area_px = total_area_px / count if count > 0 else 0.0
        mean_area_um2 = total_area_um2 / count if (count > 0 and total_area_um2 is not None) else None
        mean_intensity = float(np.mean([p.mean_intensity for p in particles])) if count > 0 else 0.0
        area_fraction = float(binary_mask.sum()) / total_roi_area_px if total_roi_area_px > 0 else 0.0
        integrated_density = sum(p.integrated_intensity for p in particles)

        # Place ROI results back into full-image-sized arrays
        if config.roi is not None:
            full_mask = np.zeros(full_shape, dtype=bool)
            full_labels = np.zeros(full_shape, dtype=label_image.dtype)
            full_mask[y:y+h, x:x+w] = binary_mask
            full_labels[y:y+h, x:x+w] = label_image
            binary_mask = full_mask
            label_image = full_labels

        return QuantificationResult(
            particle_count=count,
            total_area_px=total_area_px,
            total_area_um2=total_area_um2,
            mean_area_px=mean_area_px,
            mean_area_um2=mean_area_um2,
            mean_intensity=mean_intensity,
            area_fraction=area_fraction,
            integrated_density=integrated_density,
            particles=particles,
            threshold_method=config.threshold_method,
            threshold_value=thresh_val,
            min_area_um2=config.min_area_um2,
            max_area_um2=config.max_area_um2,
            watershed_used=config.use_watershed,
            pixel_size_um=pixel_size_um,
            channel_index=config.channel_index,
            binary_mask=binary_mask,
            label_image=label_image,
        )

    def _empty_result(self, config: QuantificationConfig,
                      pixel_size_um: Optional[float],
                      image_shape: tuple) -> QuantificationResult:
        """Return an empty result (e.g., when ROI is invalid)."""
        return QuantificationResult(
            particle_count=0,
            total_area_px=0.0,
            total_area_um2=0.0 if pixel_size_um else None,
            mean_area_px=0.0,
            mean_area_um2=0.0 if pixel_size_um else None,
            mean_intensity=0.0,
            area_fraction=0.0,
            integrated_density=0.0,
            particles=[],
            threshold_method=config.threshold_method,
            threshold_value=0.0,
            min_area_um2=config.min_area_um2,
            max_area_um2=config.max_area_um2,
            watershed_used=config.use_watershed,
            pixel_size_um=pixel_size_um,
            channel_index=config.channel_index,
            binary_mask=np.zeros(image_shape, dtype=bool),
            label_image=np.zeros(image_shape, dtype=np.int32),
        )

    # =========================================================================
    # Multi-Channel Analysis (Phase 2+3)
    # =========================================================================

    def detect_and_measure_multi(
        self,
        images: Dict[int, np.ndarray],
        config: MultiChannelConfig,
        pixel_size_um: Optional[float] = None,
        channel_names: Optional[Dict[int, str]] = None,
    ) -> MultiChannelResult:
        """
        Run detection on multiple channels, compute colocalization, and
        optionally compute ROI-based density.

        Args:
            images: Dict mapping channel index to normalized float32 image
            config: Multi-channel config with per-channel parameters
            pixel_size_um: Pixel size for calibrated measurements
            channel_names: Optional dict mapping channel index to name

        Returns:
            MultiChannelResult with per-channel results, colocalization,
            and ROI density
        """
        if channel_names is None:
            channel_names = {}

        # 1. Run per-channel detection
        channel_results: Dict[int, QuantificationResult] = {}
        for ch_idx, ch_config in config.channel_configs.items():
            if ch_idx not in images:
                continue
            result = self.detect_and_measure(images[ch_idx], ch_config, pixel_size_um)
            result.channel_name = channel_names.get(ch_idx, f"Channel {ch_idx}")
            channel_results[ch_idx] = result

        # 2. Compute pairwise colocalization (only when 2+ channels)
        colocalizations: List[ColocalizationResult] = []
        ch_indices = sorted(channel_results.keys())
        if len(ch_indices) >= 2:
            for ch_a, ch_b in combinations(ch_indices, 2):
                coloc = self.compute_colocalization(
                    channel_results[ch_a], channel_results[ch_b],
                    images[ch_a], images[ch_b],
                    pixel_size_um,
                    channel_names.get(ch_a, f"Channel {ch_a}"),
                    channel_names.get(ch_b, f"Channel {ch_b}"),
                )
                colocalizations.append(coloc)

        # 3. Compute ROI densities if ROIs are defined
        roi_densities: List[ROIDensityResult] = []
        if config.roi_definitions:
            roi_densities = self.compute_roi_densities(
                channel_results, config.roi_definitions,
                pixel_size_um, channel_names,
            )

        return MultiChannelResult(
            channel_results=channel_results,
            colocalizations=colocalizations,
            roi_densities=roi_densities,
            pixel_size_um=pixel_size_um,
        )

    def compute_colocalization(
        self,
        result_a: QuantificationResult,
        result_b: QuantificationResult,
        image_a: np.ndarray,
        image_b: np.ndarray,
        pixel_size_um: Optional[float] = None,
        name_a: str = "",
        name_b: str = "",
    ) -> ColocalizationResult:
        """
        Compute colocalization between two channel results.

        Uses centroid-based overlap (particle centroid in other channel's mask),
        mask overlap (Dice coefficient), and cross-channel intensity.
        """
        mask_a = result_a.binary_mask
        mask_b = result_b.binary_mask

        if mask_a is None or mask_b is None:
            return ColocalizationResult(
                channel_a_idx=result_a.channel_index,
                channel_a_name=name_a,
                channel_b_idx=result_b.channel_index,
                channel_b_name=name_b,
                a_in_b_count=0, a_in_b_percent=0.0,
                b_in_a_count=0, b_in_a_percent=0.0,
                overlap_area_px=0.0, overlap_area_um2=None,
                dice_coefficient=0.0,
                mean_a_intensity_in_b=0.0,
                mean_b_intensity_in_a=0.0,
            )

        # Centroid-based: A particles whose centroid is inside B's mask
        a_in_b = 0
        for p in result_a.particles:
            cy, cx = int(round(p.centroid_y)), int(round(p.centroid_x))
            if 0 <= cy < mask_b.shape[0] and 0 <= cx < mask_b.shape[1]:
                if mask_b[cy, cx]:
                    a_in_b += 1

        b_in_a = 0
        for p in result_b.particles:
            cy, cx = int(round(p.centroid_y)), int(round(p.centroid_x))
            if 0 <= cy < mask_a.shape[0] and 0 <= cx < mask_a.shape[1]:
                if mask_a[cy, cx]:
                    b_in_a += 1

        count_a = max(1, len(result_a.particles))
        count_b = max(1, len(result_b.particles))
        a_in_b_pct = (a_in_b / count_a) * 100.0
        b_in_a_pct = (b_in_a / count_b) * 100.0

        # Mask overlap
        overlap = mask_a & mask_b
        overlap_px = float(overlap.sum())
        sum_masks = float(mask_a.sum()) + float(mask_b.sum())
        dice = (2.0 * overlap_px / sum_masks) if sum_masks > 0 else 0.0

        px_to_um2 = (pixel_size_um ** 2) if pixel_size_um and pixel_size_um > 0 else None
        overlap_um2 = overlap_px * px_to_um2 if px_to_um2 else None

        # Cross-channel intensity
        # Mean of image_a intensity within B's positive mask
        mean_a_in_b = float(image_a[mask_b].mean()) if mask_b.any() else 0.0
        mean_b_in_a = float(image_b[mask_a].mean()) if mask_a.any() else 0.0

        return ColocalizationResult(
            channel_a_idx=result_a.channel_index,
            channel_a_name=name_a,
            channel_b_idx=result_b.channel_index,
            channel_b_name=name_b,
            a_in_b_count=a_in_b,
            a_in_b_percent=a_in_b_pct,
            b_in_a_count=b_in_a,
            b_in_a_percent=b_in_a_pct,
            overlap_area_px=overlap_px,
            overlap_area_um2=overlap_um2,
            dice_coefficient=dice,
            mean_a_intensity_in_b=mean_a_in_b,
            mean_b_intensity_in_a=mean_b_in_a,
        )

    def compute_roi_densities(
        self,
        channel_results: Dict[int, QuantificationResult],
        roi_definitions: List[ROIDefinition],
        pixel_size_um: Optional[float] = None,
        channel_names: Optional[Dict[int, str]] = None,
    ) -> List[ROIDensityResult]:
        """
        Compute per-ROI density metrics for each channel.

        Counts particles whose centroid falls within each ROI and computes
        density (cells/mm²) and area fraction within the ROI region.
        """
        if channel_names is None:
            channel_names = {}

        px_to_um2 = (pixel_size_um ** 2) if pixel_size_um and pixel_size_um > 0 else None

        results = []
        for roi in roi_definitions:
            roi_x, roi_y, roi_w, roi_h = roi.x, roi.y, roi.w, roi.h
            roi_area_px = roi_w * roi_h
            roi_area_mm2 = (roi_area_px * px_to_um2 / 1e6) if px_to_um2 else None

            ch_metrics: Dict[int, ROIChannelMetrics] = {}
            for ch_idx, ch_result in channel_results.items():
                # Count particles with centroid inside this ROI
                count = 0
                for p in ch_result.particles:
                    if (roi_x <= p.centroid_x < roi_x + roi_w and
                            roi_y <= p.centroid_y < roi_y + roi_h):
                        count += 1

                # Area fraction within ROI crop of binary mask
                af = 0.0
                if ch_result.binary_mask is not None and roi_area_px > 0:
                    mask = ch_result.binary_mask
                    y1 = max(0, roi_y)
                    y2 = min(mask.shape[0], roi_y + roi_h)
                    x1 = max(0, roi_x)
                    x2 = min(mask.shape[1], roi_x + roi_w)
                    if y2 > y1 and x2 > x1:
                        roi_crop = mask[y1:y2, x1:x2]
                        af = float(roi_crop.sum()) / roi_area_px

                # Mean intensity within ROI region (from raw channel data
                # is not available here - use area fraction as proxy)
                # We approximate by averaging particle intensities in ROI
                roi_intensities = []
                for p in ch_result.particles:
                    if (roi_x <= p.centroid_x < roi_x + roi_w and
                            roi_y <= p.centroid_y < roi_y + roi_h):
                        roi_intensities.append(p.mean_intensity)
                mean_int = float(np.mean(roi_intensities)) if roi_intensities else 0.0

                roi_area_um2 = roi_area_px * px_to_um2 if px_to_um2 else None
                density = (count / roi_area_mm2) if roi_area_mm2 and roi_area_mm2 > 0 else None

                ch_metrics[ch_idx] = ROIChannelMetrics(
                    channel_idx=ch_idx,
                    channel_name=channel_names.get(ch_idx, f"Channel {ch_idx}"),
                    particle_count=count,
                    area_px=float(roi_area_px),
                    area_um2=roi_area_um2,
                    density_per_mm2=density,
                    area_fraction=af,
                    mean_intensity=mean_int,
                )

            results.append(ROIDensityResult(roi=roi, channel_results=ch_metrics))

        return results
