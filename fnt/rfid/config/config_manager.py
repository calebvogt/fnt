"""
Configuration management for RFID preprocessing.

Handles loading, saving, and validating RFID configuration profiles.
"""

import json
from pathlib import Path
from typing import Optional
from .defaults import RFIDConfig


class ConfigManager:
    """Manager for RFID configuration profiles."""

    @staticmethod
    def save_config(config: RFIDConfig, filepath: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            config: RFIDConfig object to save
            filepath: Path to save the configuration file

        Raises:
            IOError: If unable to write file
        """
        filepath = Path(filepath)

        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to dictionary and save as JSON
        config_dict = config.to_dict()

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @staticmethod
    def load_config(filepath: str) -> RFIDConfig:
        """
        Load configuration from JSON file.

        Args:
            filepath: Path to configuration file

        Returns:
            RFIDConfig object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return RFIDConfig.from_dict(config_dict)

    @staticmethod
    def validate_config(config: RFIDConfig) -> tuple[bool, list[str]]:
        """
        Validate configuration for completeness and correctness.

        Args:
            config: RFIDConfig object to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check temporal parameters
        if config.bout_threshold_sec <= 0:
            errors.append("Bout threshold must be positive")

        if config.min_duration_sec <= 0:
            errors.append("Minimum duration must be positive")

        if not config.day_origin_time:
            errors.append("Day origin time is required")

        # Check spatial configuration
        if config.num_zones <= 0:
            errors.append("Number of zones must be positive")

        if config.num_antennas <= 0:
            errors.append("Number of antennas must be positive")

        if not config.antenna_zone_map:
            errors.append("Antenna-zone map is empty")
        else:
            # Validate antenna IDs are within range
            for antenna_id in config.antenna_zone_map.keys():
                if antenna_id < 1 or antenna_id > config.num_antennas:
                    errors.append(f"Antenna ID {antenna_id} out of range (1-{config.num_antennas})")

            # Validate zone IDs are within range
            for zone_id in config.antenna_zone_map.values():
                if zone_id < 1 or zone_id > config.num_zones:
                    errors.append(f"Zone ID {zone_id} out of range (1-{config.num_zones})")

        if not config.zone_coordinates:
            errors.append("Zone coordinates are empty")
        elif len(config.zone_coordinates) != config.num_zones:
            errors.append(f"Expected {config.num_zones} zone coordinates, got {len(config.zone_coordinates)}")

        # Check trial configuration
        if not config.trial_ids:
            errors.append("No trial IDs specified")

        if not config.trial_reader_map:
            errors.append("Trial-reader map is empty")
        else:
            # Validate all trial IDs have reader mappings
            for trial_id in config.trial_ids:
                if trial_id not in config.trial_reader_map:
                    errors.append(f"Trial {trial_id} missing from trial-reader map")

        # Check paths
        if not config.input_dir:
            errors.append("Input directory not specified")

        if not config.output_dir:
            errors.append("Output directory not specified")

        if not config.metadata_file_path:
            errors.append("Metadata file path not specified")

        # Check metadata configuration
        if not config.tag_columns:
            errors.append("Tag columns not specified")

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def validate_paths(config: RFIDConfig) -> tuple[bool, list[str]]:
        """
        Validate that paths in configuration exist.

        Args:
            config: RFIDConfig object to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check input directory exists
        if config.input_dir:
            input_path = Path(config.input_dir)
            if not input_path.exists():
                errors.append(f"Input directory does not exist: {config.input_dir}")
            elif not input_path.is_dir():
                errors.append(f"Input path is not a directory: {config.input_dir}")

        # Check metadata file exists
        if config.metadata_file_path:
            metadata_path = Path(config.metadata_file_path)
            if not metadata_path.exists():
                errors.append(f"Metadata file does not exist: {config.metadata_file_path}")
            elif not metadata_path.is_file():
                errors.append(f"Metadata path is not a file: {config.metadata_file_path}")

        # Check output directory (create if doesn't exist)
        if config.output_dir:
            output_path = Path(config.output_dir)
            if output_path.exists() and not output_path.is_dir():
                errors.append(f"Output path exists but is not a directory: {config.output_dir}")

        # Check trial directories exist
        if config.input_dir:
            input_path = Path(config.input_dir)
            for trial_id in config.trial_ids:
                trial_path = input_path / trial_id
                if not trial_path.exists():
                    errors.append(f"Trial directory does not exist: {trial_path}")

        is_valid = len(errors) == 0
        return is_valid, errors
