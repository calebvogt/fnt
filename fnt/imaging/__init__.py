"""
FNT Imaging Module - Zeiss CZI microscopy image viewing and processing.

This module provides tools for:
- Reading Zeiss CZI microscopy files
- False coloring and channel merging
- Per-channel brightness, contrast, gamma, sharpness
- Brightness thresholding and background subtraction
- Scale bar overlay from CZI metadata
- Interactive text annotations and shape drawing
- Image export
"""

try:
    from .czi_reader import CZIFileReader, CZIChannelInfo, CZIMetadata, CZIImageData
    from .image_processor import (
        CZIImageProcessor, ChannelDisplaySettings, TextAnnotation, ShapeAnnotation
    )

    __all__ = [
        'CZIFileReader',
        'CZIChannelInfo',
        'CZIMetadata',
        'CZIImageData',
        'CZIImageProcessor',
        'ChannelDisplaySettings',
        'TextAnnotation',
        'ShapeAnnotation',
    ]
except ImportError:
    # Dependencies not installed
    __all__ = []
