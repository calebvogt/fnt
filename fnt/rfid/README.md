# RFID Preprocessing Module

Universal RFID preprocessing tool for the FNT (FieldNeuroToolbox) package.

## Overview

This module provides a comprehensive pipeline for processing raw RFID data into analysis-ready datasets for social network analysis, behavioral analysis, and movement tracking. It replaces the existing 8-script R pipeline with a unified, configurable Python interface.

## Features

### Complete Pipeline (7 Stages)

1. **Raw RFID Processing**: Consolidates raw RFID files, maps tags to animals, assigns zones and coordinates
2. **Movement Bout Detection**: Identifies continuous bouts of presence in zones using configurable time thresholds
3. **GBI Matrix Generation**: Creates Group-By-Individual matrices showing co-occurrence patterns
4. **Social Network Analysis**: Calculates node and network-level metrics using Simple Ratio Index (SRI)
5. **Edgelist Creation**: Generates dyadic interaction records
6. **Displacement Detection**: Identifies male displacement events
7. **Hinde Index Calculation**: Computes broad/narrow contact indices and summary statistics

### Key Capabilities

- **Multi-format Support**: Auto-detects and reads Excel (.xlsx), CSV (.csv), and plain text (.txt) files
- **Configurable Parameters**: GUI-based configuration with save/load profiles
- **Memory Optimization**: Chunked processing for large datasets (191M+ records)
- **Progress Tracking**: Real-time progress updates and detailed logging
- **Preset Templates**: Ships with "8-zone paddock" configuration matching existing R pipeline

## Installation

The RFID module is part of the FNT toolbox. Install required dependencies:

```bash
pip install networkx openpyxl
```

Or install the full FNT package:

```bash
cd /path/to/fnt
pip install -e .
```

## Usage

### From FNT GUI

1. Launch FNT toolbox: `python -m fnt.gui_pyqt`
2. Navigate to the **RFID** tab
3. Click **RFID Preprocessing Tool**
4. Configure parameters and select files
5. Click **Process RFID Data**

### Standalone

```bash
python -m fnt.rfid.rfid_preprocessing_pyqt
```

### Programmatic Usage

```python
from fnt.rfid import RFIDConfig, get_default_config, ConfigManager
from fnt.rfid import RFIDPreprocessor, BoutDetector, GBIGenerator

# Load preset configuration
config = get_default_config("8_zone_paddock")

# Customize for your experiment
config.input_dir = "/path/to/rfid/data"
config.metadata_file_path = "/path/to/metadata.xlsx"
config.output_dir = "/path/to/output"
config.trial_ids = ["T001", "T002"]
config.trial_reader_map = {"T001": 1, "T002": 2}

# Run pipeline
preprocessor = RFIDPreprocessor(config)
rfid_df = preprocessor.process_raw_rfid()

bout_detector = BoutDetector(config)
movebout_df = bout_detector.detect_bouts(rfid_df)

gbi_generator = GBIGenerator(config)
gbi_dict = gbi_generator.create_gbi_matrices(movebout_df, preprocessor.metadata_df)
```

## Configuration

### Templates

- **8_zone_paddock**: Default configuration matching 2021_LID_TER project
  - 8 zones (2 columns × 4 rows)
  - 16 antennas (paired: wall + floor per zone)
  - 50-second bout threshold
  - Noon day boundaries (12:00:00)

- **custom**: Empty template for custom experimental setups

### Parameters

#### Temporal
- `bout_threshold_sec`: Time threshold for bout detection (default: 50s)
- `min_duration_sec`: Minimum bout duration (default: 1s)
- `day_origin_time`: Time for day boundaries (default: "12:00:00")
- `analysis_days`: Day range to analyze (default: 1-12)

#### Spatial
- `num_zones`: Number of zones in enclosure
- `num_antennas`: Number of RFID antennas
- `antenna_zone_map`: Dictionary mapping antenna IDs to zone IDs
- `zone_coordinates`: List of zone coordinates (x, y, location)

#### Trial
- `trial_ids`: List of trial identifiers
- `trial_reader_map`: Dictionary mapping trial IDs to reader IDs

#### Metadata
- `tag_columns`: Tag ID column names (default: ["tag_1", "tag_2"])
- `strain_prefixes`: Prefixes to remove from animal names
- `phases`: Experimental phases (default: ["early", "late"])

## Input Files

### RFID Data
Expected structure:
```
input_dir/
  T001/
    file1.xlsx
    file2.csv
  T002/
    file1.txt
```

Required columns (auto-detected with flexible naming):
- `scan_date`: Date of RFID read
- `scan_time`: Time of RFID read
- `reader_id`: Reader/paddock identifier
- `antenna_id`: Antenna identifier
- `tag_id`: RFID tag ID

### Metadata File
Excel or CSV file with:
- `trial`: Trial identifier
- `name`: Animal name
- `sex`: M/F
- `tag_1`, `tag_2`: RFID tag IDs (dual-tag system)
- `phase`: Experimental phase (optional)

## Output Files

1. **ALLTRIAL_RFID_DATA.csv**: Raw RFID reads with metadata (may be chunked if >100MB)
2. **ALLTRIAL_MOVEBOUT.csv**: Movement bouts with START/STOP classifications
3. **{trial}_MOVEBOUT_GBI.csv**: Trial-specific GBI matrices
4. **ALLTRIAL_SNA_node_stats.csv**: Node-level network metrics
5. **ALLTRIAL_SNA_net_stats.csv**: Network-level metrics
6. **ALLTRIAL_MOVEBOUT_GBI_edgelist.csv**: Dyadic interactions
7. **ALLTRIAL_MOVEBOUT_GBI_displace.csv**: Displacement events
8. **ALLTRIAL_MOVEBOUT_GBI_hinde_broad.csv**: All contact events
9. **ALLTRIAL_MOVEBOUT_GBI_hinde_narrow.csv**: 2-individual contacts only
10. **ALLTRIAL_MOVEBOUT_GBI_summary.csv**: Individual movement statistics

## Module Structure

```
fnt/rfid/
├── __init__.py                          # Package exports
├── config/
│   ├── __init__.py
│   ├── config_manager.py                # Configuration I/O
│   ├── defaults.py                      # Preset templates
│   └── validators.py                    # Configuration validation
├── core/
│   ├── __init__.py
│   ├── file_readers.py                  # Multi-format RFID file reading
│   ├── preprocessor.py                  # Stage 1: Raw RFID processing
│   ├── bout_detector.py                 # Stage 2: Movement bouts
│   ├── gbi_generator.py                 # Stage 3a: GBI matrices
│   ├── social_network.py                # Stage 3b: Network analysis
│   ├── edgelist.py                      # Stage 3c: Edgelist
│   ├── displacement.py                  # Stage 3d: Displacement events
│   ├── hinde_index.py                   # Stage 3e-f: Hinde indices
│   └── utils.py                         # Shared utilities
├── rfid_preprocessing_pyqt.py           # Main GUI window
└── rfid_worker.py                       # Background processing thread
```

## Algorithm Details

### Bout Detection
Uses time-gap thresholding:
- New bout starts when: `time_gap > threshold` OR `zone_changed`
- Reads classified as: `START`, `STOP`, or `SINGLE_READ`
- Minimum duration applied to prevent zero-duration bouts

### GBI Generation
Uses center-time method for co-occurrence:
- Bouts overlap if their centers fall within each other's duration
- Creates binary presence matrix (1 = present, 0 = absent)
- Includes sex-based summaries (m_sum, f_sum, mf_sum)

### Social Network Metrics
**Node-level** (using NetworkX):
- Edge strength (sum of SRI weights)
- Degree, eigenvector, betweenness, closeness centrality
- PageRank, authority scores
- Opposite-sex edge strength

**Network-level**:
- Density, transitivity, centralization
- Mean distance, modularity
- Number of communities (greedy modularity)

## Comparison with R Pipeline

This Python implementation replicates the 8-script R pipeline:

| R Script | Python Module | Function |
|----------|---------------|----------|
| `1_create_ALLTRIAL_RFID.R` | `preprocessor.py` | Raw RFID processing |
| `2_create_ALLTRIAL_MOVEBOUT.R` | `bout_detector.py` | Movement bouts |
| `3_create_ALLTRIAL_MOVEBOUT_GBI.R` | `gbi_generator.py` | GBI matrices |
| `3a_create_MOVEBOUT_GBI_sn_node_net.R` | `social_network.py` | Network analysis |
| `3b_create_ALLTRIAL_MOVEBOUT_GBI_edgelist.R` | `edgelist.py` | Edgelist |
| `3c_create_ALLTRIAL_MOVEBOUT_GBI_displace.R` | `displacement.py` | Displacements |
| `3d, 3e, 3f` | `hinde_index.py` | Hinde indices + summary |

**Advantages over R pipeline:**
- Single unified interface (no script juggling)
- GUI-based configuration (no hardcoded paths)
- Multi-format file support (not just Excel)
- Real-time progress tracking
- Memory-efficient chunked processing
- Configuration profiles (reuse setups)

## Testing

Test with your existing 2021_LID_TER data:

```python
config = get_default_config("8_zone_paddock")
config.input_dir = "/Users/caleb/Box/1_projects/2021_LID_TER/data/rfid"
config.metadata_file_path = "/Users/caleb/Box/1_projects/2021_LID_TER/data/metadata.xlsx"
config.output_dir = "/Users/caleb/Box/1_projects/2021_LID_TER/data/output_python"

# Run full pipeline
# Compare outputs with existing R outputs for validation
```

## Troubleshooting

### "Missing required columns" error
- Check that your RFID files have columns matching: scan_date, scan_time, reader_id, antenna_id, tag_id
- The tool supports flexible column naming (e.g., "Scan Date", "scan date", "scandate" all work)

### "Metadata file missing required columns"
- Ensure metadata has: trial, name, sex, tag_1, tag_2 columns

### Memory issues with large datasets
- The tool automatically chunks files >100MB
- Reduce `analysis_days` range to process fewer days at once
- Process trials separately by adjusting `trial_ids`

### No displacement events detected
- Displacement detection requires male-only data
- Check that metadata `sex` column is correctly labeled ("M", "F")

## License

Part of the FieldNeuroToolbox (FNT) package.

## Author

Caleb Sankaran (with Claude Sonnet 4.5)
