"""
Image Processor - False coloring, adjustments, and channel merging for CZI images.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class ChannelDisplaySettings:
    """Display settings for a single channel."""
    enabled: bool = True
    color: str = "gray"
    brightness: float = 1.0
    contrast: float = 1.0
    gamma: float = 1.0
    min_display: float = 0.0
    max_display: float = 1.0


@dataclass
class TextAnnotation:
    """Text annotation to be rendered on the image."""
    text: str
    x: int
    y: int
    font_size: int = 24
    color: str = "white"
    rotation: float = 0.0  # Rotation in degrees
    # Computed bounding box (set after rendering)
    width: int = 0
    height: int = 0


class CZIImageProcessor:
    """
    Process and render CZI image channels for display.

    Handles:
    - False coloring (mapping grayscale to colored LUTs)
    - Brightness, contrast, gamma adjustments
    - Channel merging for composite view
    - Text annotation overlay
    """

    # Color name to RGB mapping for annotations
    ANNOTATION_COLORS = {
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
    }

    def __init__(self):
        self.channel_settings: Dict[int, ChannelDisplaySettings] = {}
        self.annotations: List[TextAnnotation] = []

    def set_channel_settings(self, channel_idx: int, settings: ChannelDisplaySettings):
        """Set display settings for a channel."""
        self.channel_settings[channel_idx] = settings

    def get_channel_settings(self, channel_idx: int) -> ChannelDisplaySettings:
        """Get display settings for a channel, creating defaults if needed."""
        if channel_idx not in self.channel_settings:
            self.channel_settings[channel_idx] = ChannelDisplaySettings()
        return self.channel_settings[channel_idx]

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-1 float range.

        Args:
            image: Input image (any dtype)

        Returns:
            Float32 array normalized to 0-1 range
        """
        if image.dtype == np.float32 or image.dtype == np.float64:
            return image.astype(np.float32)

        # Integer types - normalize by max value
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        elif image.dtype == np.uint16:
            return image.astype(np.float32) / 65535.0
        else:
            # Generic normalization
            img_min = float(image.min())
            img_max = float(image.max())
            if img_max - img_min < 1e-10:
                return np.zeros_like(image, dtype=np.float32)
            return ((image.astype(np.float32) - img_min) / (img_max - img_min))

    def apply_adjustments(self, image: np.ndarray,
                          settings: ChannelDisplaySettings) -> np.ndarray:
        """
        Apply brightness, contrast, gamma adjustments to a normalized image.

        Args:
            image: Normalized float image (0-1 range)
            settings: Display settings to apply

        Returns:
            Adjusted float image (0-1 range, clipped)
        """
        img = image.copy()

        # Apply min/max display range (window/level)
        if settings.min_display != 0.0 or settings.max_display != 1.0:
            range_width = settings.max_display - settings.min_display
            if range_width > 1e-10:
                img = (img - settings.min_display) / range_width
            else:
                img = np.zeros_like(img)

        # Apply gamma correction
        if settings.gamma != 1.0:
            # Avoid issues with negative values
            img = np.clip(img, 0, None)
            img = np.power(img, 1.0 / settings.gamma)

        # Apply brightness (multiplicative)
        if settings.brightness != 1.0:
            img = img * settings.brightness

        # Apply contrast (around 0.5 midpoint)
        if settings.contrast != 1.0:
            img = (img - 0.5) * settings.contrast + 0.5

        # Clip to valid range
        return np.clip(img, 0, 1)

    def colorize_channel(self, image: np.ndarray, color: str) -> np.ndarray:
        """
        Apply false color LUT to grayscale channel.

        Args:
            image: 2D float image (0-1 range)
            color: Color name ('green', 'magenta', 'cyan', 'red', 'blue', 'gray', 'yellow')

        Returns:
            RGB float image (H, W, 3) in 0-1 range
        """
        h, w = image.shape

        if color == 'green':
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[:, :, 1] = image  # Green channel only
        elif color == 'magenta':
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[:, :, 0] = image  # Red
            rgb[:, :, 2] = image  # Blue
        elif color == 'cyan':
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[:, :, 1] = image  # Green
            rgb[:, :, 2] = image  # Blue
        elif color == 'red':
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[:, :, 0] = image
        elif color == 'blue':
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[:, :, 2] = image
        elif color == 'yellow':
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[:, :, 0] = image  # Red
            rgb[:, :, 1] = image  # Green
        else:  # gray or unknown
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[:, :, 0] = image
            rgb[:, :, 1] = image
            rgb[:, :, 2] = image

        return rgb

    def process_channel(self, image: np.ndarray,
                        settings: ChannelDisplaySettings) -> np.ndarray:
        """
        Process a single channel: normalize, adjust, and colorize.

        Args:
            image: Raw channel image (any dtype)
            settings: Display settings

        Returns:
            RGB float image (H, W, 3) in 0-1 range
        """
        # Normalize to 0-1
        normalized = self.normalize_image(image)

        # Apply adjustments
        adjusted = self.apply_adjustments(normalized, settings)

        # Apply color
        colored = self.colorize_channel(adjusted, settings.color)

        return colored

    def merge_channels(self, channel_images: Dict[int, np.ndarray],
                       settings: Optional[Dict[int, ChannelDisplaySettings]] = None) -> np.ndarray:
        """
        Merge multiple channels into composite RGB using additive blending.

        Args:
            channel_images: Dict mapping channel index to raw channel image
            settings: Optional dict of settings per channel (uses self.channel_settings if None)

        Returns:
            Merged RGB float image (H, W, 3) in 0-1 range
        """
        if settings is None:
            settings = self.channel_settings

        # Get image dimensions from first channel
        first_key = next(iter(channel_images.keys()))
        h, w = channel_images[first_key].shape[:2]

        # Start with black
        merged = np.zeros((h, w, 3), dtype=np.float32)

        # Process and add each enabled channel
        for idx, raw_image in channel_images.items():
            channel_settings = settings.get(idx, ChannelDisplaySettings())

            if not channel_settings.enabled:
                continue

            # Process channel (normalize, adjust, colorize)
            colored = self.process_channel(raw_image, channel_settings)

            # Add to composite
            merged += colored

        # Clip final result
        return np.clip(merged, 0, 1)

    def render_single_channel(self, image: np.ndarray, channel_idx: int) -> np.ndarray:
        """
        Render a single channel with its current settings.

        Args:
            image: Raw channel image
            channel_idx: Channel index for settings lookup

        Returns:
            RGB float image (H, W, 3) in 0-1 range
        """
        settings = self.get_channel_settings(channel_idx)
        return self.process_channel(image, settings)

    def auto_levels(self, image: np.ndarray, percentile: float = 0.1) -> Tuple[float, float]:
        """
        Calculate auto-levels based on histogram percentiles.

        Args:
            image: Input grayscale image
            percentile: Percentile for min/max calculation (default 0.1%)

        Returns:
            (min_value, max_value) for display range (normalized to 0-1)
        """
        # Normalize first
        normalized = self.normalize_image(image)

        min_val = float(np.percentile(normalized, percentile))
        max_val = float(np.percentile(normalized, 100 - percentile))

        # Ensure valid range
        if max_val - min_val < 0.01:
            min_val = float(normalized.min())
            max_val = float(normalized.max())

        return min_val, max_val

    def add_annotation(self, text: str, x: int, y: int,
                       font_size: int = 24, color: str = "white"):
        """Add a text annotation to the list."""
        self.annotations.append(TextAnnotation(
            text=text, x=x, y=y, font_size=font_size, color=color
        ))

    def clear_annotations(self):
        """Clear all annotations."""
        self.annotations.clear()

    def remove_annotation(self, index: int):
        """Remove annotation by index."""
        if 0 <= index < len(self.annotations):
            self.annotations.pop(index)

    def render_with_annotations(self, image: np.ndarray) -> np.ndarray:
        """
        Render image with all text annotations.

        Args:
            image: RGB float image (H, W, 3) in 0-1 range

        Returns:
            RGB image with annotations rendered
        """
        if not self.annotations or not HAS_PIL:
            return image

        # Convert to PIL Image
        img_uint8 = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_uint8, mode='RGB')
        draw = ImageDraw.Draw(pil_image)

        for annotation in self.annotations:
            # Get color
            color = self.ANNOTATION_COLORS.get(annotation.color, (255, 255, 255))

            # Try to get a font
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",
                                          annotation.font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("arial.ttf", annotation.font_size)
                except Exception:
                    font = ImageFont.load_default()

            # Draw text
            draw.text((annotation.x, annotation.y), annotation.text,
                      fill=color, font=font)

        # Convert back to numpy
        return np.array(pil_image).astype(np.float32) / 255.0

    def to_uint8(self, image: np.ndarray) -> np.ndarray:
        """Convert float image to uint8 for display/export."""
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)

    def export_image(self, image: np.ndarray, filepath: str,
                     include_annotations: bool = True):
        """
        Export processed image to file.

        Args:
            image: RGB float image (H, W, 3) in 0-1 range
            filepath: Output file path (PNG or TIFF)
            include_annotations: Whether to include text annotations
        """
        if not HAS_PIL:
            raise ImportError("PIL/Pillow required for image export")

        # Render annotations if requested
        if include_annotations and self.annotations:
            image = self.render_with_annotations(image)

        # Convert to uint8
        img_uint8 = self.to_uint8(image)

        # Save with PIL
        pil_image = Image.fromarray(img_uint8, mode='RGB')
        pil_image.save(filepath)
