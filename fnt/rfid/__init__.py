"""RFID preprocessing for the FNT toolbox.

Turns Biomark reader downloads into analysis-ready tables: cleaned reads,
bouts, per animal-day summaries, zone holders and pairwise co-occupancy, with
QC reports on everything that was removed or looks wrong. One config, one
analysis folder - see :class:`RFIDConfig` and :func:`fnt.rfid.pipeline.run`.
The GUI is :class:`fnt.rfid.rfid_preprocessing_pyqt.RFIDPreprocessingWindow`.
"""

from .config import (RFIDConfig, Arena, Paddock, Zone, Antenna, MetadataSpec,
                     ConfigManager, get_default_config, get_available_presets,
                     liddell_8_zone, one_antenna_per_zone)

__all__ = [
    "RFIDConfig", "Arena", "Paddock", "Zone", "Antenna", "MetadataSpec",
    "ConfigManager", "get_default_config", "get_available_presets",
    "liddell_8_zone", "one_antenna_per_zone",
]
