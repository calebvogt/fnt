"""Configuration for RFID preprocessing."""

from .defaults import (RFIDConfig, Arena, Paddock, Zone, Antenna, MetadataSpec,
                       ARENA_PRESETS, OPTIONAL_LAYERS, METADATA_FIELDS,
                       BOUT_RULES, CROSS_PADDOCK_POLICIES,
                       animal_settings, get_default_config,
                       get_available_presets, get_fnt_version, normalise_id,
                       liddell_8_zone, one_antenna_per_zone)
from .config_manager import (ConfigManager, default_output_dir, default_name,
                             resolve_output_dir, read_antenna_csv,
                             write_antenna_csv, zones_for_antennas)

__all__ = [
    "RFIDConfig", "Arena", "Paddock", "Zone", "Antenna", "MetadataSpec",
    "ARENA_PRESETS", "OPTIONAL_LAYERS", "METADATA_FIELDS", "BOUT_RULES",
    "CROSS_PADDOCK_POLICIES", "animal_settings", "get_default_config",
    "get_available_presets", "get_fnt_version", "normalise_id",
    "liddell_8_zone", "one_antenna_per_zone",
    "ConfigManager", "default_output_dir", "default_name", "resolve_output_dir",
    "read_antenna_csv", "write_antenna_csv", "zones_for_antennas",
]
