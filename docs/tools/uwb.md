# UWB Tracking

Preprocessing and analysis tools for **Ultra-Wideband (UWB)** positional tracking data. Designed for multi-animal tracking systems (e.g., Qorvo/Decawave DWM1001) used in field and laboratory neuroethology.

Launch from the FNT main GUI **UWB** tab.

## UWB PreProcessing Tool

Interactive GUI for loading, cleaning, and exporting UWB tracking data:

- **Load raw data** from `.db` (SQLite) or `.csv` files containing timestamped tag positions
- **Tag identity mapping** — assign animal IDs, sex, and metadata to each UWB tag
- **Deployment windows and off-animal intervals** — a Start/Stop per tag, plus any number of intervals when a tag was reporting but not on its animal (e.g. a head cap found off and re-attached); fixes in either are dropped from the preview and every export
- **Coordinate smoothing** with configurable Savitzky-Golay filter parameters
- **Timezone handling** — convert UTC timestamps to local time
- **Day/night segmentation** based on a configurable light cycle
- **Visual preview** of smoothed trajectories per tag
- **Weather, sunlight and sky** for the site (latitude/longitude you enter or load from a site profile): a weather line, a daylight strip, the moon's phase, and a sky icon (clear, partly cloudy, overcast, fog, light/moderate/heavy rain, snow, thunderstorm) that follows the playhead in the preview and the video. Measured data win: rain comes from the weather station's gauge wherever it has a record and daytime cloud from measured sunlight; the Open-Meteo model fills only gaps and what nothing measures (night cloud, thunder, fog), and is labelled "model". Exported minute by minute to `{db}_sky.csv` with the evidence behind each minute
- **Regions of interest** — draw regions on the arena (or import the site XML's zones) for visit and occupancy tables
- **Contact and chases** — contact bouts between every pair of animals within a distance threshold, and directed chase bouts
- **Export** cleaned data, social and region tables, plots and animations, one trial at a time or as a batch queue

## Exports

Everything goes to `<database>_FNT_analysis/` beside the database, with the settings in `fnt_config.json`. The tables are in `csvs/`:

| File | One row per |
|------|-------------|
| `<db>_smoothed.csv` | fix of a configured tag inside its deployment window: raw and smoothed positions (m), local time, animal |
| `<db>_raw.csv` | the same fixes before smoothing (optional) |
| `<db>_SocialOverlapBouts.csv` | contact bout of a pair: `animal1`, `animal2`, `Day`, `Date`, `bout_start`, `bout_stop`, `duration_s`, `mean_distance`, `n_reads` |
| `<db>_behavior_events.csv` | directed chase: `behavior`, `actor`, `target`, sexes, `Day`, `bout_start`, `bout_stop`, `duration_s` |
| `<db>_ROI_bout_occupancy.csv` | visit to a drawn region: `SexID`, `Day`, `ROI`, `bout_start`, `bout_stop`, `duration_s`, `n_reads` |
| `<db>_ROI_DailySummary.csv` | animal, region and day: time in the region and its rank among the animals that entered |
| `<db>_sky.csv` | minute: weather, daylight and sky state, with the evidence for each |

Contact bouts are specific to the distance threshold and the bridging gap chosen at export; export again to try other values. Plots go to `plots/` and tracking or social-network animations to `animation_*/`.

## Input data format

The tool expects UWB data with at minimum:

| Column | Description |
|--------|-------------|
| `Timestamp` | ISO 8601 timestamp (UTC or local) |
| `shortid` | UWB tag identifier |
| `location_x` | X position (metres) |
| `location_y` | Y position (metres) |

Additional columns (e.g., `location_z`, `location_quality`) are preserved during export.

!!! tip "Positions for analysis"
    Use `smoothed_x` and `smoothed_y` from `<db>_smoothed.csv`, at 1 Hz or finer. Tag read rates can differ several-fold between animals, so time spent anywhere is a sum of per-fix durations, never a count of fixes.
