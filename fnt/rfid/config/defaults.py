"""
Default configuration templates for RFID preprocessing.

This module provides preset configurations for common experimental setups.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
import json


@dataclass
class RFIDConfig:
    """Configuration for RFID preprocessing pipeline."""

    # Temporal parameters
    bout_threshold_sec: float = 50.0
    min_duration_sec: float = 1.0
    day_origin_time: str = "12:00:00"
    analysis_days: Tuple[int, int] = (1, 12)  # (start_day, end_day)

    # Spatial configuration
    num_zones: int = 8
    num_antennas: int = 16
    antenna_zone_map: Dict[int, int] = field(default_factory=dict)
    zone_coordinates: List[Dict[str, any]] = field(default_factory=list)

    # Trial configuration
    trial_ids: List[str] = field(default_factory=list)
    trial_reader_map: Dict[str, int] = field(default_factory=dict)

    # Metadata configuration
    metadata_file_path: str = ""
    tag_columns: List[str] = field(default_factory=lambda: ["tag_1", "tag_2"])
    strain_prefixes: List[str] = field(default_factory=lambda: ["OB-M-", "OB-F-"])
    phases: List[str] = field(default_factory=lambda: ["early", "late"])
    sex_values: List[str] = field(default_factory=lambda: ["M", "F"])

    # Path configuration
    input_dir: str = ""
    output_dir: str = ""

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'RFIDConfig':
        """Create config from dictionary."""
        # Convert tuple back from list if needed
        if 'analysis_days' in data and isinstance(data['analysis_days'], list):
            data['analysis_days'] = tuple(data['analysis_days'])
        return cls(**data)


def get_8_zone_paddock_config() -> RFIDConfig:
    """
    Get default configuration for 8-zone paddock setup.

    This matches the configuration used in the 2021_LID_TER project:
    - 8 zones arranged in 2 columns x 4 rows
    - 16 antennas (paired per zone: wall + floor)
    - 50-second bout threshold
    - Day boundaries at noon (12:00:00)

    Returns:
        RFIDConfig: Preset configuration for 8-zone paddock
    """
    config = RFIDConfig()

    # Temporal parameters (matching R pipeline)
    config.bout_threshold_sec = 50.0
    config.min_duration_sec = 1.0
    config.day_origin_time = "12:00:00"
    config.analysis_days = (1, 12)

    # Spatial configuration
    config.num_zones = 8
    config.num_antennas = 16

    # Antenna to zone mapping (antennas 1-8 = wall, 9-16 = floor, paired per zone)
    # Antennas 1 & 9 → Zone 1, Antennas 2 & 10 → Zone 2, etc.
    config.antenna_zone_map = {
        1: 1, 9: 1,   # Zone 1
        2: 2, 10: 2,  # Zone 2
        3: 3, 11: 3,  # Zone 3
        4: 4, 12: 4,  # Zone 4
        5: 5, 13: 5,  # Zone 5
        6: 6, 14: 6,  # Zone 6
        7: 7, 15: 7,  # Zone 7
        8: 8, 16: 8   # Zone 8
    }

    # Zone coordinates (x, y, location)
    # Layout: 2 columns (x = 3.75 or 11.25) x 4 rows (y = 7.6, 15.2, 22.8, 30.4)
    config.zone_coordinates = [
        {"zone": 1, "x": 3.75, "y": 7.6, "location": "wall"},
        {"zone": 2, "x": 11.25, "y": 7.6, "location": "wall"},
        {"zone": 3, "x": 3.75, "y": 15.2, "location": "wall"},
        {"zone": 4, "x": 11.25, "y": 15.2, "location": "wall"},
        {"zone": 5, "x": 3.75, "y": 22.8, "location": "wall"},
        {"zone": 6, "x": 11.25, "y": 22.8, "location": "wall"},
        {"zone": 7, "x": 3.75, "y": 30.4, "location": "wall"},
        {"zone": 8, "x": 11.25, "y": 30.4, "location": "wall"}
    ]

    # Trial configuration (example - user should customize)
    config.trial_ids = ["T001", "T002"]
    config.trial_reader_map = {"T001": 1, "T002": 2}

    # Metadata configuration
    config.tag_columns = ["tag_1", "tag_2"]
    config.strain_prefixes = ["OB-M-", "OB-F-"]
    config.phases = ["early", "late"]
    config.sex_values = ["M", "F"]

    return config


def get_default_config(template_name: str = "8_zone_paddock") -> RFIDConfig:
    """
    Get a default configuration template by name.

    Args:
        template_name: Name of the template ("8_zone_paddock" or "custom")

    Returns:
        RFIDConfig: Configuration object

    Raises:
        ValueError: If template name is not recognized
    """
    templates = {
        "8_zone_paddock": get_8_zone_paddock_config,
        "custom": RFIDConfig  # Empty config for custom setup
    }

    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")

    if template_name == "custom":
        return templates[template_name]()
    else:
        return templates[template_name]()


def get_available_templates() -> List[str]:
    """
    Get list of available configuration templates.

    Returns:
        List of template names
    """
    return ["8_zone_paddock", "custom"]
