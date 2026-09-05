"""Core RFID preprocessing stages.

The pipeline runs in this order, each stage taking the previous one's table:

    raw exports -> reads -> bouts -> GBI -> {ownership, contacts, networks}

Every stage below the GBI is an analysis layer: it reads the GBI and writes its
own table, and none of them feed each other except displacement, which needs
zone ownership to say whether a winner was at home.
"""

from .file_readers import (read_biomark_txt, read_rfid_file, read_download_dir,
                           summarise_downloads)
from .preprocessor import (RFIDPreprocessor, ReadsResult, load_metadata,
                           tag_lookup, pad_tag)
from .bout_detector import BoutDetector, detect_bouts
from .gbi_generator import GBIGenerator, create_gbi, melt_gbi
from .zone_ownership import (zone_ownership, daily_owners, zones_owned_per_day)
from .edgelist import EdgelistGenerator, co_presence_bouts, edgelist
from .displacement import (DisplacementDetector, detect_displacements,
                           annotate_ownership)
from .hinde_index import HindeIndexCalculator, hinde_index, hinde_summary
from .social_network import (SocialNetworkAnalyzer, social_networks,
                             sri_matrix, network_for_day)

__all__ = [
    # readers
    "read_biomark_txt", "read_rfid_file", "read_download_dir",
    "summarise_downloads",
    # stage 1
    "RFIDPreprocessor", "ReadsResult", "load_metadata", "tag_lookup", "pad_tag",
    # stage 2
    "BoutDetector", "detect_bouts",
    # stage 3
    "GBIGenerator", "create_gbi", "melt_gbi",
    # analysis layers
    "zone_ownership", "daily_owners", "zones_owned_per_day",
    "EdgelistGenerator", "co_presence_bouts", "edgelist",
    "DisplacementDetector", "detect_displacements", "annotate_ownership",
    "HindeIndexCalculator", "hinde_index", "hinde_summary",
    "SocialNetworkAnalyzer", "social_networks", "sri_matrix", "network_for_day",
]
