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
- Automated cell/particle counting and quantification
"""

try:
    from .czi_reader import CZIFileReader, CZIChannelInfo, CZIMetadata, CZIImageData
    from .image_processor import (
        CZIImageProcessor, ChannelDisplaySettings, TextAnnotation, ShapeAnnotation
    )
    from .quantification import (
        ImageQuantifier, QuantificationResult, QuantificationConfig, ParticleResult,
        MultiChannelConfig, MultiChannelResult, ColocalizationResult,
        ROIDefinition, ROIDensityResult, ROIChannelMetrics,
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
        'ImageQuantifier',
        'QuantificationResult',
        'QuantificationConfig',
        'ParticleResult',
        'MultiChannelConfig',
        'MultiChannelResult',
        'ColocalizationResult',
        'ROIDefinition',
        'ROIDensityResult',
        'ROIChannelMetrics',
    ]
except ImportError:
    # Dependencies not installed
    __all__ = []
