"""
DoricFP - Doric Fiber Photometry Processing Module

This module provides tools for processing data from Doric wireless
fiber photometry (WiFP) systems, including:

- Reading .doric (HDF5) files
- ΔF/F calculation with isosbestic correction
- Video synchronization and alignment
- Batch processing with GUI
"""

from .doric_processor import (
    DoricFileReader,
    DFFCalculator,
    TraditionalDFFCalculator,
    VideoSynchronizer,
    DoricFileData,
    DoricChannelInfo,
    DoricVideoInfo,
    find_doric_video_pairs,
    process_doric_file
)

from .doric_processor_pyqt import (
    DoricProcessorWindow,
    DoricProcessWorker
)

__all__ = [
    'DoricFileReader',
    'DFFCalculator',
    'TraditionalDFFCalculator',
    'DFFCalculator', 
    'VideoSynchronizer',
    'DoricFileData',
    'DoricChannelInfo',
    'DoricVideoInfo',
    'find_doric_video_pairs',
    'process_doric_file',
    'DoricProcessorWindow',
    'DoricProcessWorker'
]
