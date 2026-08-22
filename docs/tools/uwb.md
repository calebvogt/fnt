# UWB Tracking

Preprocessing and analysis tools for **Ultra-Wideband (UWB)** positional tracking data. Designed for multi-animal tracking systems (e.g., Qorvo/Decawave DWM1001) used in field and laboratory neuroethology.

Launch from the FNT main GUI **UWB** tab.

## UWB PreProcessing Tool

Interactive GUI for loading, cleaning, and exporting UWB tracking data:

- **Load raw data** from `.db` (SQLite) or `.csv` files containing timestamped tag positions
- **Tag identity mapping** — assign animal IDs, sex, and metadata to each UWB tag
- **Coordinate smoothing** with configurable Savitzky-Golay filter parameters
- **Timezone handling** — convert UTC timestamps to local time
- **Day/night segmentation** based on a configurable light cycle
- **Visual preview** of smoothed trajectories per tag
- **Export** cleaned data to CSV with standardized column naming

## Proximity Detection

Batch analysis for detecting pairwise proximity events between tracked animals:

- Computes inter-animal distances at each timestamp
- Detects **proximity bouts** — contiguous periods where two animals are within a configurable distance threshold
- Configurable gap-bridging to merge bouts separated by brief interruptions
- Outputs **proximity_bouts** — summarized bouts with start/stop times, duration, mean distance, and observation count

    Bouts are specific to the chosen distance threshold; re-run the analysis with a different threshold to explore other distances. The raw per-timestamp pairwise distances are computed internally but not exported.

## Input data format

The tool expects UWB data with at minimum:

| Column | Description |
|--------|-------------|
| `Timestamp` | ISO 8601 timestamp (UTC or local) |
| `shortid` | UWB tag identifier |
| `location_x` | X position (metres) |
| `location_y` | Y position (metres) |

Additional columns (e.g., `location_z`, `location_quality`) are preserved during export.

## Output

Exported CSVs include smoothed coordinates (`smoothed_x`, `smoothed_y`), day/night labels, and animal identity metadata — ready for downstream analysis in R or Python.
