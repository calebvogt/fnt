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

try:
    from skimage.restoration import rolling_ball
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


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
    # Sharpness (beta) - unsharp mask amount, 0 = off
    sharpness: float = 0.0
    # Brightness thresholding
    threshold_enabled: bool = True
    threshold_low: float = 0.0
    threshold_high: float = 1.0
    # Background subtraction
    bg_subtract_method: str = "none"  # "none", "rolling_ball", "gaussian", "roi"
    bg_subtract_radius: float = 50.0
    bg_subtract_roi_value: float = 0.0


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


@dataclass
class ShapeAnnotation:
    """Shape annotation for drawing on the image."""
    shape_type: str  # "arrow", "line", "circle", "ellipse", "rectangle", "freehand"
    x: float = 0.0   # Start/anchor point
    y: float = 0.0
    x2: float = 0.0   # End point (line/arrow)
    y2: float = 0.0
    width: float = 0.0   # For rectangle/ellipse
    height: float = 0.0
    color: str = "white"
    line_width: float = 2.0
    line_style: str = "solid"  # "solid", "dashed", "dotted"
    rotation: float = 0.0
    arrow_head_size: float = 15.0
    points: List[Tuple[float, float]] = field(default_factory=list)  # For freehand


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
        self.shape_annotations: List[ShapeAnnotation] = []
        self._bg_cache: Dict[int, Tuple[str, float, np.ndarray]] = {}  # channel_idx -> (method, radius, result)

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

    def apply_threshold(self, image: np.ndarray,
                        settings: ChannelDisplaySettings) -> np.ndarray:
        """
        Apply brightness thresholding - zero out pixels outside [low, high].

        Args:
            image: Normalized float image (0-1 range)
            settings: Display settings with threshold parameters

        Returns:
            Thresholded image
        """
        if not settings.threshold_enabled:
            return image
        mask = (image >= settings.threshold_low) & (image <= settings.threshold_high)
        return image * mask.astype(np.float32)

    def apply_background_subtraction(self, image: np.ndarray,
                                     settings: ChannelDisplaySettings,
                                     channel_idx: int = -1) -> np.ndarray:
        """
        Apply background subtraction using the specified method.

        Args:
            image: Normalized float image (0-1 range)
            settings: Display settings with bg subtraction parameters
            channel_idx: Channel index for caching

        Returns:
            Background-subtracted image (clipped to 0-1)
        """
        if settings.bg_subtract_method == "none":
            return image

        # Check cache
        if channel_idx >= 0 and channel_idx in self._bg_cache:
            cached_method, cached_radius, cached_result = self._bg_cache[channel_idx]
            if cached_method == settings.bg_subtract_method and cached_radius == settings.bg_subtract_radius:
                return cached_result

        if settings.bg_subtract_method == "rolling_ball":
            result = self._bg_rolling_ball(image, settings.bg_subtract_radius)
        elif settings.bg_subtract_method == "gaussian":
            result = self._bg_gaussian(image, settings.bg_subtract_radius)
        elif settings.bg_subtract_method == "roi":
            result = self._bg_roi(image, settings.bg_subtract_roi_value)
        else:
            return image

        # Cache the result
        if channel_idx >= 0:
            self._bg_cache[channel_idx] = (settings.bg_subtract_method, settings.bg_subtract_radius, result)

        return result

    def _bg_rolling_ball(self, image: np.ndarray, radius: float) -> np.ndarray:
        """Rolling ball background subtraction using scikit-image, with downsampling for speed."""
        if not HAS_SKIMAGE:
            print("Warning: scikit-image not installed. Rolling ball not available.")
            return image
        from scipy.ndimage import zoom as ndimage_zoom
        h, w = image.shape[:2]
        downsample_factor = max(1, min(4, max(h, w) // 512))
        if downsample_factor > 1:
            small = ndimage_zoom(image, 1.0 / downsample_factor, order=1)
            bg_small = rolling_ball(small, radius=radius / downsample_factor)
            background = ndimage_zoom(bg_small, downsample_factor, order=1)[:h, :w]
        else:
            background = rolling_ball(image, radius=radius)
        return np.clip(image - background, 0, 1).astype(np.float32)

    def _bg_gaussian(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian blur background subtraction, with downsampling for speed."""
        from scipy.ndimage import gaussian_filter, zoom as ndimage_zoom
        h, w = image.shape[:2]
        downsample_factor = max(1, min(4, max(h, w) // 512))
        if downsample_factor > 1:
            small = ndimage_zoom(image, 1.0 / downsample_factor, order=1)
            bg_small = gaussian_filter(small, sigma=sigma / downsample_factor)
            background = ndimage_zoom(bg_small, downsample_factor, order=1)[:h, :w]
        else:
            background = gaussian_filter(image, sigma=sigma)
        return np.clip(image - background, 0, 1).astype(np.float32)

    def _bg_roi(self, image: np.ndarray, roi_value: float) -> np.ndarray:
        """ROI-based background subtraction (subtract constant value)."""
        return np.clip(image - roi_value, 0, 1).astype(np.float32)

    def clear_bg_cache(self):
        """Clear background subtraction cache (call on file load or settings change)."""
        self._bg_cache.clear()

    def invalidate_bg_cache(self, channel_idx: int):
        """Invalidate cache for a specific channel."""
        self._bg_cache.pop(channel_idx, None)

    def apply_sharpness(self, image: np.ndarray, amount: float) -> np.ndarray:
        """
        Apply unsharp mask sharpening.

        Args:
            image: Normalized float image (0-1 range)
            amount: Sharpness amount (0 = no sharpening)

        Returns:
            Sharpened image (clipped to 0-1)
        """
        if amount <= 0:
            return image
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(image, sigma=1.0)
        return np.clip(image + amount * (image - blurred), 0, 1).astype(np.float32)

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
                        settings: ChannelDisplaySettings,
                        channel_idx: int = -1) -> np.ndarray:
        """
        Process a single channel: normalize, threshold, bg subtract, adjust, sharpen, colorize.

        Args:
            image: Raw channel image (any dtype)
            settings: Display settings
            channel_idx: Channel index (for bg subtraction caching)

        Returns:
            RGB float image (H, W, 3) in 0-1 range
        """
        # Normalize to 0-1
        normalized = self.normalize_image(image)

        # Apply threshold (zero out pixels outside range)
        thresholded = self.apply_threshold(normalized, settings)

        # Apply background subtraction
        bg_corrected = self.apply_background_subtraction(thresholded, settings, channel_idx)

        # Apply brightness/contrast/gamma/min-max adjustments
        adjusted = self.apply_adjustments(bg_corrected, settings)

        # Apply sharpness (unsharp mask)
        sharpened = self.apply_sharpness(adjusted, settings.sharpness)

        # Apply color LUT
        colored = self.colorize_channel(sharpened, settings.color)

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

            # Process channel (normalize, threshold, bg subtract, adjust, sharpen, colorize)
            colored = self.process_channel(raw_image, channel_settings, channel_idx=idx)

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

    # --- Shape Annotation Methods ---

    def add_shape(self, shape: ShapeAnnotation):
        """Add a shape annotation."""
        self.shape_annotations.append(shape)

    def remove_shape(self, index: int):
        """Remove shape annotation by index."""
        if 0 <= index < len(self.shape_annotations):
            self.shape_annotations.pop(index)

    def clear_shapes(self):
        """Clear all shape annotations."""
        self.shape_annotations.clear()

    def render_with_shapes(self, image: np.ndarray) -> np.ndarray:
        """
        Render image with all shape annotations burned in for export.

        Args:
            image: RGB float image (H, W, 3) in 0-1 range

        Returns:
            RGB image with shapes rendered
        """
        if not self.shape_annotations or not HAS_PIL:
            return image

        import math
        img_uint8 = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_uint8, mode='RGB')
        draw = ImageDraw.Draw(pil_image)

        for shape in self.shape_annotations:
            color = self.ANNOTATION_COLORS.get(shape.color, (255, 255, 255))
            lw = max(1, int(shape.line_width))

            if shape.shape_type in ("line", "arrow"):
                draw.line([(shape.x, shape.y), (shape.x2, shape.y2)],
                          fill=color, width=lw)
                if shape.shape_type == "arrow":
                    # Draw arrowhead
                    dx = shape.x2 - shape.x
                    dy = shape.y2 - shape.y
                    length = math.sqrt(dx * dx + dy * dy)
                    if length > 0:
                        ux, uy = dx / length, dy / length
                        # Perpendicular
                        px, py = -uy, ux
                        hs = shape.arrow_head_size
                        tip_x, tip_y = shape.x2, shape.y2
                        left_x = tip_x - ux * hs + px * hs * 0.4
                        left_y = tip_y - uy * hs + py * hs * 0.4
                        right_x = tip_x - ux * hs - px * hs * 0.4
                        right_y = tip_y - uy * hs - py * hs * 0.4
                        draw.polygon([(tip_x, tip_y), (left_x, left_y), (right_x, right_y)],
                                     fill=color)

            elif shape.shape_type == "rectangle":
                draw.rectangle([(shape.x, shape.y),
                                (shape.x + shape.width, shape.y + shape.height)],
                               outline=color, width=lw)

            elif shape.shape_type in ("circle", "ellipse"):
                draw.ellipse([(shape.x - shape.width / 2, shape.y - shape.height / 2),
                              (shape.x + shape.width / 2, shape.y + shape.height / 2)],
                             outline=color, width=lw)

            elif shape.shape_type == "freehand" and shape.points:
                if len(shape.points) >= 2:
                    draw.line(shape.points, fill=color, width=lw)

        return np.array(pil_image).astype(np.float32) / 255.0

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

    def render_scale_bar(self, image: np.ndarray, pixel_size_um: float,
                         position: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Render a scale bar onto the exported image.

        Args:
            image: RGB float image (H, W, 3) in 0-1 range
            pixel_size_um: Pixel size in micrometers
            position: (x_frac, y_frac) as fraction of image dimensions (0.0-1.0),
                      or None for default bottom-right

        Returns:
            Image with scale bar burned in
        """
        if pixel_size_um <= 0 or not HAS_PIL:
            return image

        h, w = image.shape[:2]

        # Calculate bar length — target ~20% of image width
        target_um = w * pixel_size_um * 0.2
        nice_values = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
        bar_um = nice_values[0]
        for v in nice_values:
            if v <= target_um:
                bar_um = v
            else:
                break

        bar_px = int(bar_um / pixel_size_um)
        bar_height = max(3, h // 200)
        margin = max(15, w // 60)

        # Position
        if position is not None:
            x = int(position[0] * w)
            y = int(position[1] * h)
        else:
            x = w - margin - bar_px
            y = h - margin - bar_height - 20

        # Label text
        label = f"{bar_um / 1000:.0f} mm" if bar_um >= 1000 else f"{bar_um} \u00b5m"

        # Draw using PIL
        img_uint8 = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_uint8, mode='RGB')
        draw = ImageDraw.Draw(pil_image)

        # Black outline
        draw.rectangle([x - 1, y - 1, x + bar_px + 1, y + bar_height + 1],
                       fill=(0, 0, 0))
        # White bar
        draw.rectangle([x, y, x + bar_px, y + bar_height], fill=(255, 255, 255))

        # Label font
        font_size = max(12, h // 60)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x + (bar_px - text_w) // 2
        text_y = y + bar_height + 2

        # Text shadow for visibility, then white text
        draw.text((text_x + 1, text_y + 1), label, fill=(0, 0, 0), font=font)
        draw.text((text_x, text_y), label, fill=(255, 255, 255), font=font)

        return np.array(pil_image).astype(np.float32) / 255.0

    def to_uint8(self, image: np.ndarray) -> np.ndarray:
        """Convert float image to uint8 for display/export."""
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)

    def export_image(self, image: np.ndarray, filepath: str,
                     include_annotations: bool = True,
                     scale_bar_info: Optional[dict] = None):
        """
        Export processed image to file.

        Args:
            image: RGB float image (H, W, 3) in 0-1 range
            filepath: Output file path (PNG or TIFF)
            include_annotations: Whether to include text annotations
            scale_bar_info: Optional dict with 'pixel_size_um' and 'position' keys
        """
        if not HAS_PIL:
            raise ImportError("PIL/Pillow required for image export")

        # Render shapes if any
        if include_annotations and self.shape_annotations:
            image = self.render_with_shapes(image)

        # Render text annotations if requested
        if include_annotations and self.annotations:
            image = self.render_with_annotations(image)

        # Render scale bar if info provided
        if scale_bar_info:
            image = self.render_scale_bar(
                image,
                pixel_size_um=scale_bar_info['pixel_size_um'],
                position=scale_bar_info.get('position')
            )

        # Convert to uint8
        img_uint8 = self.to_uint8(image)

        # Save with PIL
        pil_image = Image.fromarray(img_uint8, mode='RGB')
        pil_image.save(filepath)
