"""Configuration management for RFID preprocessing."""

from .defaults import RFIDConfig, get_default_config, get_available_templates
from .config_manager import ConfigManager

__all__ = [
    'RFIDConfig',
    'get_default_config',
    'get_available_templates',
    'ConfigManager'
]
