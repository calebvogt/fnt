"""
CZI File Reader - Core functionality for reading Zeiss CZI microscopy files.

Uses aicspylibczi directly for reading CZI format (avoids heavy aicsimageio dependencies).
Falls back to aicsimageio if available.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

# Check for available CZI reading backends
HAS_AICSPYLIBCZI = False
HAS_AICSIMAGEIO = False

try:
    from aicspylibczi import CziFile
    HAS_AICSPYLIBCZI = True
except ImportError:
    pass

try:
    from aicsimageio import AICSImage
    HAS_AICSIMAGEIO = True
except ImportError:
    pass


@dataclass
class CZIChannelInfo:
    """Information about a single channel in a CZI file."""
    index: int
    name: str
    wavelength_nm: Optional[float] = None
    suggested_color: str = "gray"
    bit_depth: int = 16


@dataclass
class CZIMetadata:
    """Metadata extracted from CZI file."""
    filepath: str
    dimensions: Dict[str, int] = field(default_factory=dict)
    pixel_size_um: Optional[float] = None
    pixel_size_y_um: Optional[float] = None
    pixel_size_z_um: Optional[float] = None
    objective: Optional[str] = None
    magnification: Optional[float] = None
    numerical_aperture: Optional[float] = None
    acquisition_date: Optional[str] = None
    channels: List[CZIChannelInfo] = field(default_factory=list)
    raw_xml: Optional[str] = None

    @property
    def width(self) -> int:
        return self.dimensions.get('X', 0)

    @property
    def height(self) -> int:
        return self.dimensions.get('Y', 0)

    @property
    def n_channels(self) -> int:
        return self.dimensions.get('C', len(self.channels))

    @property
    def n_z_slices(self) -> int:
        return self.dimensions.get('Z', 1)

    @property
    def n_timepoints(self) -> int:
        return self.dimensions.get('T', 1)


@dataclass
class CZIImageData:
    """Container for loaded CZI image data."""
    metadata: CZIMetadata
    channel_data: Dict[int, np.ndarray] = field(default_factory=dict)

    def get_channel(self, index: int) -> Optional[np.ndarray]:
        """Get channel data by index."""
        return self.channel_data.get(index)

    @property
    def n_channels(self) -> int:
        return len(self.channel_data)


def _suggest_color_from_name(name: str) -> str:
    """Suggest a display color based on channel name."""
    name_lower = name.lower()

    # GFP/green fluorescent protein
    if any(x in name_lower for x in ['gfp', 'fitc', 'alexa488', '488', 'green']):
        return 'green'

    # Cy3/RFP/red fluorescent protein
    if any(x in name_lower for x in ['cy3', 'rfp', 'mcherry', 'dsred', '561', 'tritc', 'red']):
        return 'magenta'

    # DAPI/Hoechst/blue
    if any(x in name_lower for x in ['dapi', 'hoechst', '405', 'blue']):
        return 'blue'

    # Cy5/far-red
    if any(x in name_lower for x in ['cy5', 'alexa647', '647', '633']):
        return 'cyan'

    return 'gray'


def _suggest_color_from_index(index: int) -> str:
    """Suggest a display color based on channel index."""
    colors = ['green', 'magenta', 'cyan', 'blue', 'red', 'yellow']
    return colors[index % len(colors)]


class CZIFileReader:
    """
    Reader for Zeiss CZI microscopy files.

    Handles multi-channel fluorescence images with automatic channel detection.
    Uses aicspylibczi directly for reading (lighter dependency than aicsimageio).

    Example:
        reader = CZIFileReader("/path/to/image.czi")
        metadata = reader.scan_file()
        print(f"Channels: {metadata.n_channels}")

        data = reader.load_all_channels()
        for idx, channel in data.channel_data.items():
            print(f"Channel {idx}: {channel.shape}")
    """

    def __init__(self, filepath: str):
        """
        Initialize with path to .czi file.

        Args:
            filepath: Path to CZI file

        Raises:
            ImportError: If no CZI reading library is installed
            FileNotFoundError: If file does not exist
            ValueError: If file is not a CZI file
        """
        if not HAS_AICSPYLIBCZI and not HAS_AICSIMAGEIO:
            raise ImportError(
                "CZI file support requires aicspylibczi.\n"
                "Install with: pip install aicspylibczi fsspec Pillow"
            )

        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if self.filepath.suffix.lower() != '.czi':
            raise ValueError(f"Expected .czi file, got: {self.filepath.suffix}")

        self._czi = None
        self._metadata: Optional[CZIMetadata] = None
        self._use_aicsimageio = HAS_AICSIMAGEIO and not HAS_AICSPYLIBCZI

    def _ensure_loaded(self):
        """Ensure the CZI file is loaded (lazy loading)."""
        if self._czi is None:
            if self._use_aicsimageio:
                self._czi = AICSImage(str(self.filepath))
            else:
                self._czi = CziFile(str(self.filepath))

    def scan_file(self) -> CZIMetadata:
        """
        Scan CZI file and extract metadata without loading full image data.

        Returns:
            CZIMetadata containing file information and channel details
        """
        if self._metadata is not None:
            return self._metadata

        self._ensure_loaded()

        if self._use_aicsimageio:
            return self._scan_with_aicsimageio()
        else:
            return self._scan_with_aicspylibczi()

    def _scan_with_aicspylibczi(self) -> CZIMetadata:
        """Scan using aicspylibczi directly."""
        # Get dimensions from the CZI file
        dims_shape = self._czi.get_dims_shape()

        # dims_shape returns a list of dicts, one per scene
        # For most files, there's one scene at index 0
        if dims_shape:
            dim_dict = dims_shape[0]
        else:
            dim_dict = {}

        # Convert to our dimension format
        dims = {}
        for dim_name, (start, size) in dim_dict.items():
            dims[dim_name] = size

        # Ensure we have basic dimensions
        if 'X' not in dims or 'Y' not in dims:
            # Try to get from read_mosaic_size or first read
            try:
                bbox = self._czi.get_all_mosaic_tile_bounding_boxes()
                if bbox:
                    # Get overall bounding box
                    all_x = [b.x + b.w for b in bbox.values()]
                    all_y = [b.y + b.h for b in bbox.values()]
                    dims['X'] = max(all_x) if all_x else 0
                    dims['Y'] = max(all_y) if all_y else 0
            except Exception:
                pass

        n_channels = dims.get('C', 1)

        # Build channel info - aicspylibczi doesn't easily expose channel names
        # so we use generic names with color suggestions
        channels = []
        for i in range(n_channels):
            suggested_color = _suggest_color_from_index(i)
            channels.append(CZIChannelInfo(
                index=i,
                name=f"Channel {i + 1}",
                suggested_color=suggested_color
            ))

        self._metadata = CZIMetadata(
            filepath=str(self.filepath),
            dimensions=dims,
            channels=channels
        )

        # Parse XML metadata for pixel sizes, objective, channel names, etc.
        self._parse_xml_metadata()

        return self._metadata

    def _parse_xml_metadata(self):
        """Parse CZI XML metadata for physical dimensions, objective info, etc."""
        if self._metadata is None or self._czi is None:
            return

        try:
            import xml.etree.ElementTree as ET

            # Get raw XML from CZI file
            xml_str = self._czi.meta
            if xml_str is None:
                return

            # Store raw XML for debugging
            if isinstance(xml_str, str):
                self._metadata.raw_xml = xml_str
                root = ET.fromstring(xml_str)
            else:
                # Some versions return an ElementTree directly
                root = xml_str
                self._metadata.raw_xml = ET.tostring(root, encoding='unicode')

            # --- Physical pixel size ---
            # CZI stores scaling in meters
            scaling = root.find('.//Scaling/Items')
            if scaling is not None:
                for distance in scaling.findall('Distance'):
                    dim_id = distance.get('Id')
                    value_elem = distance.find('Value')
                    if value_elem is not None and value_elem.text:
                        try:
                            val_m = float(value_elem.text)
                            val_um = val_m * 1e6  # meters to micrometers
                            if dim_id == 'X':
                                self._metadata.pixel_size_um = val_um
                            elif dim_id == 'Y':
                                self._metadata.pixel_size_y_um = val_um
                            elif dim_id == 'Z':
                                self._metadata.pixel_size_z_um = val_um
                        except (ValueError, TypeError):
                            pass

            # --- Objective info ---
            objective = root.find('.//Instrument/Objectives/Objective')
            if objective is not None:
                name_elem = objective.find('Name')
                if name_elem is not None and name_elem.text:
                    self._metadata.objective = name_elem.text

                mag_elem = objective.find('NominalMagnification')
                if mag_elem is not None and mag_elem.text:
                    try:
                        self._metadata.magnification = float(mag_elem.text)
                    except (ValueError, TypeError):
                        pass

                na_elem = objective.find('LensNA')
                if na_elem is not None and na_elem.text:
                    try:
                        self._metadata.numerical_aperture = float(na_elem.text)
                    except (ValueError, TypeError):
                        pass

            # --- Acquisition date ---
            acq_date = root.find('.//AcquisitionDateAndTime')
            if acq_date is not None and acq_date.text:
                self._metadata.acquisition_date = acq_date.text

            # --- Channel names from XML (better than generic "Channel N") ---
            channels_elem = root.find('.//Dimensions/Channels')
            if channels_elem is not None:
                xml_channels = channels_elem.findall('Channel')
                for i, ch_elem in enumerate(xml_channels):
                    ch_name = ch_elem.get('Name') or ch_elem.get('Id')
                    if ch_name and i < len(self._metadata.channels):
                        self._metadata.channels[i].name = ch_name
                        self._metadata.channels[i].suggested_color = _suggest_color_from_name(ch_name)

                        # Try to get wavelength
                        exc_elem = ch_elem.find('.//ExcitationWavelength')
                        if exc_elem is not None and exc_elem.text:
                            try:
                                self._metadata.channels[i].wavelength_nm = float(exc_elem.text)
                            except (ValueError, TypeError):
                                pass

        except Exception as e:
            # Don't fail on metadata parsing errors - just skip
            print(f"Warning: Could not parse CZI XML metadata: {e}")

    def _scan_with_aicsimageio(self) -> CZIMetadata:
        """Scan using aicsimageio."""
        # Parse dimensions
        dims = {}
        dim_order = self._czi.dims.order
        shape = self._czi.shape

        for i, dim in enumerate(dim_order):
            dims[dim] = shape[i]

        # Get channel names
        channel_names = self._czi.channel_names or []
        if not channel_names:
            n_channels = dims.get('C', 1)
            channel_names = [f"Channel {i}" for i in range(n_channels)]

        # Build channel info
        channels = []
        for i, name in enumerate(channel_names):
            suggested_color = _suggest_color_from_name(name)
            channels.append(CZIChannelInfo(
                index=i,
                name=name,
                suggested_color=suggested_color
            ))

        # Try to get pixel size
        pixel_size = None
        try:
            physical_pixel_sizes = self._czi.physical_pixel_sizes
            if physical_pixel_sizes and physical_pixel_sizes.X:
                pixel_size = physical_pixel_sizes.X
        except Exception:
            pass

        self._metadata = CZIMetadata(
            filepath=str(self.filepath),
            dimensions=dims,
            pixel_size_um=pixel_size,
            channels=channels
        )

        return self._metadata

    def load_channel(self, channel_index: int, z_slice: int = 0, timepoint: int = 0) -> np.ndarray:
        """
        Load a specific channel's image data as 2D array.

        Args:
            channel_index: Channel index (0-based)
            z_slice: Z-slice index for 3D images (default 0)
            timepoint: Timepoint index for time-series (default 0)

        Returns:
            2D numpy array (Y, X) with channel data
        """
        self._ensure_loaded()

        if self._use_aicsimageio:
            return self._czi.get_image_data("YX", C=channel_index, Z=z_slice, T=timepoint)
        else:
            return self._load_channel_aicspylibczi(channel_index, z_slice, timepoint)

    def _load_channel_aicspylibczi(self, channel_index: int, z_slice: int = 0,
                                    timepoint: int = 0) -> np.ndarray:
        """Load channel using aicspylibczi directly."""
        metadata = self.scan_file()

        # Build the dimension selection
        # aicspylibczi uses read_image with specific dimension values
        try:
            # Try reading as mosaic first (common for microscopy)
            data, shape = self._czi.read_image(C=channel_index, Z=z_slice, T=timepoint)

            # data comes back as (1, 1, 1, 1, Y, X) or similar - squeeze to 2D
            data = np.squeeze(data)

            # If still more than 2D, take first slice of extra dims
            while data.ndim > 2:
                data = data[0]

            return data

        except Exception:
            # Fall back to reading mosaic
            try:
                data, shape = self._czi.read_mosaic(C=channel_index, Z=z_slice, T=timepoint,
                                                     scale_factor=1.0)
                data = np.squeeze(data)
                while data.ndim > 2:
                    data = data[0]
                return data
            except Exception as e:
                raise RuntimeError(f"Failed to read channel {channel_index}: {e}")

    def load_all_channels(self, z_slice: int = 0, timepoint: int = 0) -> CZIImageData:
        """
        Load all channels for a given Z-slice and timepoint.

        Args:
            z_slice: Z-slice index for 3D images (default 0)
            timepoint: Timepoint index for time-series (default 0)

        Returns:
            CZIImageData containing metadata and all channel arrays
        """
        metadata = self.scan_file()

        channel_data = {}
        for channel_info in metadata.channels:
            data = self.load_channel(channel_info.index, z_slice, timepoint)
            channel_data[channel_info.index] = data

        return CZIImageData(
            metadata=metadata,
            channel_data=channel_data
        )

    def get_z_projection(self, channel_index: int, method: str = 'max',
                         timepoint: int = 0) -> np.ndarray:
        """
        Create Z-projection for a channel.

        Args:
            channel_index: Channel index
            method: Projection method ('max', 'mean', 'sum')
            timepoint: Timepoint index

        Returns:
            2D projected image
        """
        self._ensure_loaded()
        metadata = self.scan_file()

        n_z = metadata.n_z_slices
        if n_z <= 1:
            return self.load_channel(channel_index, 0, timepoint)

        # Load all Z slices
        slices = []
        for z in range(n_z):
            slices.append(self.load_channel(channel_index, z, timepoint))

        stack = np.stack(slices, axis=0)

        if method == 'max':
            return np.max(stack, axis=0)
        elif method == 'mean':
            return np.mean(stack, axis=0)
        elif method == 'sum':
            return np.sum(stack, axis=0)
        else:
            raise ValueError(f"Unknown projection method: {method}")
