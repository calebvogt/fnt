"""Fiber photometry: dF/F, z-scores, transients and video alignment.

The analysis (``processing``, ``sync``) works on any signal/control pair; readers for each
acquisition system sit beside it. Supported so far:

- Doric wireless (WiFP / NC500) ``.doric`` files: ``doric``.
- LabJack LJStreamUD ``.dat`` files (the Warden lab rig, 2019-2020): ``labjack``.

Methods adapted from GuPPy (Lerner lab, BSD-3-Clause; ``guppy_methods``) and PhAT (Donaldson lab,
MIT; ``phat_methods``) keep their licences and are cited in docs/tools/fiber-photometry.md.

The GUI (``fiber_photometry_pyqt``) is imported on demand so the analysis can be used without Qt.
"""

from .doric import (
    DoricChannelInfo,
    DoricFileData,
    DoricFileReader,
    DoricVideoInfo,
    find_doric_video_pairs,
)
from .labjack import LabJackFileReader, is_labjack_dat
from .processing import (
    PIPELINES,
    PIPELINE_LABELS,
    PhotometryResult,
    TraditionalDFFCalculator,
    find_dropouts,
    process,
)
from .sync import align_to_frames, check_video_clock, recording_frame_times, video_frame_times

__all__ = [
    "DoricChannelInfo", "DoricFileData", "DoricFileReader", "DoricVideoInfo",
    "find_doric_video_pairs", "LabJackFileReader", "is_labjack_dat",
    "PIPELINES", "PIPELINE_LABELS", "PhotometryResult", "TraditionalDFFCalculator",
    "find_dropouts", "process",
    "align_to_frames", "check_video_clock", "recording_frame_times", "video_frame_times",
]
