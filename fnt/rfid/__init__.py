"""RFID preprocessing for the FNT toolbox.

Turns raw RFID reader exports into a trial's analysis-ready tables: reads,
movement bouts, a group-by-individual matrix, and the zone-ownership, contact
and social-network layers derived from it.

One trial, one config, one output folder - see :class:`TrialConfig`.
"""

from .config import (TrialConfig, RFIDConfig, Arena, Zone, Antenna,
                     get_default_config, get_available_templates, ConfigManager)
from .core import (
    RFIDPreprocessor, ReadsResult,
    BoutDetector, detect_bouts,
    GBIGenerator, create_gbi, melt_gbi,
    zone_ownership, daily_owners, zones_owned_per_day,
    EdgelistGenerator, co_presence_bouts, edgelist,
    DisplacementDetector, detect_displacements, annotate_ownership,
    HindeIndexCalculator, hinde_index, hinde_summary,
    SocialNetworkAnalyzer, social_networks,
)

__all__ = [
    "TrialConfig", "RFIDConfig", "Arena", "Zone", "Antenna",
    "get_default_config", "get_available_templates", "ConfigManager",
    "RFIDPreprocessor", "ReadsResult",
    "BoutDetector", "detect_bouts",
    "GBIGenerator", "create_gbi", "melt_gbi",
    "zone_ownership", "daily_owners", "zones_owned_per_day",
    "EdgelistGenerator", "co_presence_bouts", "edgelist",
    "DisplacementDetector", "detect_displacements", "annotate_ownership",
    "HindeIndexCalculator", "hinde_index", "hinde_summary",
    "SocialNetworkAnalyzer", "social_networks",
]
