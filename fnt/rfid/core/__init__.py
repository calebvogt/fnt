"""Core RFID preprocessing stages.

    downloads -> reads -> cleaned reads -> bouts -> summaries, QC

``file_readers`` parses Biomark downloads, ``metadata`` the animal table,
``cleaning`` applies identity, place and every cleaning rule, ``bout_detector``
builds bouts, ``summaries`` the per-day tables and ``qc`` the reports.

The GBI, edgelist, displacement, Hinde and social-network modules are optional
layers carried over from the earlier module (see ``RFIDConfig.layers``).
"""

from .file_readers import (read_downloads, list_download_files, read_export,
                           read_biomark_txt, read_xlsx_text, normalise_tag_value,
                           normalise_tags, parse_scan_times, DownloadResult)
from .metadata import (load_animals, AnimalTable, read_grid, list_sheets,
                       detect_header_row, header_names, guess_mapping,
                       excel_date, excel_time, combine_date_time)
from .cleaning import clean_reads, CleanResult, day_index, resolve_origin
from .bout_detector import detect_bouts, BOUT_COLUMNS
from .summaries import animal_days, zone_holders, co_occupancy, overlapping_bouts
from . import qc

__all__ = [
    "read_downloads", "list_download_files", "read_export", "read_biomark_txt",
    "read_xlsx_text", "normalise_tag_value", "normalise_tags", "parse_scan_times",
    "DownloadResult",
    "load_animals", "AnimalTable", "read_grid", "list_sheets", "detect_header_row",
    "header_names", "guess_mapping", "excel_date", "excel_time", "combine_date_time",
    "clean_reads", "CleanResult", "day_index", "resolve_origin",
    "detect_bouts", "BOUT_COLUMNS",
    "animal_days", "zone_holders", "co_occupancy", "overlapping_bouts", "qc",
]
