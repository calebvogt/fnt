# UWB Processing Module

Preprocessing and analysis tools for **Ultra-Wideband (UWB)** positional tracking data. Designed for multi-animal tracking systems (e.g., Qorvo/Decawave DWM1001) used in field and laboratory neuroethology experiments.

## Tools

### UWB PreProcessing Tool

Launch from FNT main GUI: **UWB** tab → **UWB PreProcessing Tool**

Interactive GUI for loading, cleaning, and exporting UWB tracking data:

- **Load raw data** from `.db` (SQLite) or `.csv` files containing timestamped tag positions
- **Tag identity mapping** — assign animal IDs, sex, and metadata to each UWB tag
- **Coordinate smoothing** with configurable Savitzky-Golay filter parameters
- **Timezone handling** — convert UTC timestamps to local time
- **Day/night segmentation** based on configurable light cycle
- **Visual preview** of smoothed trajectories per tag
- **Export** cleaned data to CSV with standardized column naming

### Proximity Detection

Batch analysis module for detecting pairwise proximity events between tracked animals:

- Computes inter-animal distances at each timestamp
- Detects **proximity bouts** — contiguous periods where two animals are within a configurable distance threshold
- Configurable gap-bridging to merge bouts separated by brief interruptions
- Outputs **proximity_bouts** — summarized bouts with start/stop times, duration, mean distance, and observation count

  Bouts are specific to the chosen distance threshold; re-run the analysis with a different threshold to explore other distances. The raw per-timestamp pairwise distances are computed internally but not exported.

## Input Data Format

The input is a **Wiser tracking server SQLite database**. The columns the
pipeline reads are:

| Column | Description |
|--------|-------------|
| `timestamp` | milliseconds since the Unix epoch (an instant, not a local time) |
| `shortid` | UWB tag identifier (decimal encoding of the hex Tag ID) |
| `location_x` | X position **in inches** — converted to metres on read |
| `location_y` | Y position **in inches** — converted to metres on read |
| `battery_voltage` | optional per-tag readout |

Everything else in the table is left alone. Two columns look like quality
metrics and are deliberately **not** used for filtering: `calculation_error`
is not anchor-count-neutral (a fix from 3-5 anchors reports a perfect zero
71% of the time against 35% for a fix from 21+, because there is too little
redundancy for few anchors to disagree), and gating on `anchors_used` can lengthen the very steps it is
meant to remove.

See **[WISER_SCHEMA.md](WISER_SCHEMA.md)** for the full column reference, the
reverse-engineered form of `calculation_error`, the clock's measured
behaviour, and how to read the exported timestamps downstream.

## Output

Exported CSVs include smoothed coordinates (`smoothed_x`, `smoothed_y`), day/night labels, and animal identity metadata. These are ready for downstream analysis in R or Python.
