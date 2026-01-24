"""Core RFID preprocessing modules."""

from .file_readers import read_rfid_file, read_rfid_directory, read_all_trials
from .utils import (
    map_antenna_to_zone,
    get_zone_coordinates,
    parse_metadata,
    create_tag_mapping,
    filter_cross_paddock_reads,
    create_temporal_variables,
    assign_zone_info
)
from .preprocessor import RFIDPreprocessor
from .bout_detector import BoutDetector
from .gbi_generator import GBIGenerator
from .social_network import SocialNetworkAnalyzer
from .edgelist import EdgelistGenerator
from .displacement import DisplacementDetector
from .hinde_index import HindeIndexCalculator

__all__ = [
    # File readers
    'read_rfid_file',
    'read_rfid_directory',
    'read_all_trials',

    # Utilities
    'map_antenna_to_zone',
    'get_zone_coordinates',
    'parse_metadata',
    'create_tag_mapping',
    'filter_cross_paddock_reads',
    'create_temporal_variables',
    'assign_zone_info',

    # Pipeline stages
    'RFIDPreprocessor',
    'BoutDetector',
    'GBIGenerator',
    'SocialNetworkAnalyzer',
    'EdgelistGenerator',
    'DisplacementDetector',
    'HindeIndexCalculator',
]
