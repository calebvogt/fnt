"""
FNT Imaging Module - Zeiss CZI microscopy image viewing and processing.

This module provides tools for:
- Reading Zeiss CZI microscopy files
- False coloring and channel merging
- Image adjustments (brightness, contrast, gamma)
- Text annotations and export
"""

try:
    from .czi_reader import CZIFileReader, CZIChannelInfo, CZIMetadata, CZIImageData
    from .image_processor import CZIImageProcessor, ChannelDisplaySettings

    __all__ = [
        'CZIFileReader',
        'CZIChannelInfo',
        'CZIMetadata',
        'CZIImageData',
        'CZIImageProcessor',
        'ChannelDisplaySettings',
    ]
except ImportError:
    # Dependencies not installed
    __all__ = []
