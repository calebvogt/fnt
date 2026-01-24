"""
RFID Preprocessing Module for FNT Toolbox.

Provides tools for processing raw RFID data into analysis-ready datasets
for social network analysis, behavioral analysis, and movement tracking.
"""

from .config import RFIDConfig, get_default_config, get_available_templates, ConfigManager
from .core import (
    RFIDPreprocessor,
    BoutDetector,
    GBIGenerator,
    SocialNetworkAnalyzer,
    EdgelistGenerator,
    DisplacementDetector,
    HindeIndexCalculator
)

__version__ = '1.0.0'

__all__ = [
    # Configuration
    'RFIDConfig',
    'get_default_config',
    'get_available_templates',
    'ConfigManager',

    # Pipeline components
    'RFIDPreprocessor',
    'BoutDetector',
    'GBIGenerator',
    'SocialNetworkAnalyzer',
    'EdgelistGenerator',
    'DisplacementDetector',
    'HindeIndexCalculator',
]
