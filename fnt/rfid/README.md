# RFID Processing Module

Preprocessing and analysis for **passive RFID** tracking of animals moving
between antenna-instrumented resource zones. Built for field enclosure studies
where each zone carries one or more antennas and animals wear PIT tags.

## Design

**One trial, one config, one output folder** — the same shape as the UWB module.
A trial's analysis folder sits beside its raw exports and is self-describing:

```
T001_Alpha/
    LID_2021_ALPHA_RFID_DL_1.txt        raw exports, never modified
    LID_2021_ALPHA_RFID_DL_1.xlsx
    ...
    T001_FNT/
        fnt_config.json                 settings, provenance, run stats, warnings
        _messageLog.txt
        csvs/                           every table the run produced
        plots/
        animations/
```

This replaces an earlier workflow that concatenated every trial into single
`ALLTRIAL_*.csv` files. Per-trial folders make a run reproducible (the config
that produced a folder lives in it) and make re-running one trial cheap.

## Pipeline

```
raw exports → reads → bouts → GBI → { ownership, contacts, networks }
```

| Stage | Output | What it is |
|---|---|---|
| Reads | `<trial>_reads.csv` | One row per tag detection, with animal identity, zone, coordinates, day, and elapsed time |
| Bouts | `<trial>_bouts.csv` | One row per continuous stay in a zone |
| GBI | `<trial>_gbi.csv` | Group-by-individual matrix: one row per interval in which the set of animals present cannot change |
| Bout summary | `<trial>_bout_summary.csv` | The GBI in long form — one row per animal per interval, carrying who else was there |
| Zone ownership | `<trial>_zone_ownership.csv` | Each male's share of a zone's reads per day, and his rank |
| Co-presence | `<trial>_co_presence_bouts.csv`, `<trial>_edgelist.csv` | Continuous encounters per pair, and their aggregation |
| Displacement | `<trial>_displacement.csv` | Male 1→n→1 turnovers in a zone, typed by who owned it |
| Hinde index | `<trial>_hinde_{broad,narrow}.csv` | Who closes distance and who breaks it |
| Social networks | `<trial>_sna_{node,net}_stats.csv` | SRI association networks with igraph metrics, per day |

Reads, bouts and the GBI are the preprocessing itself and are always produced.
Everything below them is an analysis layer selected in `TrialConfig.exports`.

## Input

**Reader exports.** Biomark Device Manager `.txt` (fixed-width), `.xlsx`, and
CSV are all read. Every file in the trial folder is ingested and the union
deduplicated — downloads overlap by design, and in real data a `.txt` export can
be silently truncated while its `.xlsx` sibling is complete. Taking the union
makes that harmless instead of lossy.

**Metadata.** A CSV or Excel table with at minimum `name`, `sex`, and the tag
columns (`tag_1`, `tag_2` by default). `code`, `phase`, `group`, `strain` (or
`genotype`), and `trial` are used when present.

> Tag IDs are 15-significant-digit decimals. Anything that stores them as a
> number drops the trailing digit — `985.113004548310` becomes
> `985.11300454831`, which matches no animal. They are read as text throughout
> and re-padded on load.

## Arena

Zones and antennas are configured together, because assigning several antennas
to one zone is the normal case (a wall and a floor antenna per zone):

```python
from fnt.rfid import Arena, get_default_config

cfg = get_default_config("8_zone_paddock")   # 2x4 zones, 16 antennas
cfg.arena.antenna_zone_map()                 # {1: 1, ..., 9: 1, ..., 16: 8}
```

`Arena.grid()` lays out a regular grid; individual antennas can then be moved or
reassigned. Validation refuses a config with a zone that has no antenna (it
could never record an animal) or an antenna assigned to a zone that does not
exist, and a run reports antennas that produced no reads — a dead antenna
otherwise looks exactly like an unvisited zone.

## Usage

```python
from fnt.rfid import get_default_config
from fnt.rfid.pipeline import run_trial

cfg = get_default_config("8_zone_paddock")
cfg.trial_id = "T001"
cfg.raw_dir = r"...\Data\RFID\T001_Alpha"
cfg.metadata_path = r"...\Data\metadata.csv"
cfg.reader_ids = [1]                 # readers installed in THIS enclosure
cfg.exports.update({"displacement": True, "sna": True})

result = run_trial(cfg, progress=print)
result.tables["gbi"]                 # tables are returned as well as written
```

## Choices worth knowing about

**Bout detection** defaults to `segment`: a new bout begins whenever the gap
reaches the threshold or the zone changes, so no bout can contain a gap or a
zone change. An `r_compat` algorithm reproduces the original R pipeline's
control flow exactly, for regenerating published numbers — it pairs bout starts
to stops by position rather than adjacency, which mis-pairs about 2.3% of bouts.

**Timestamp resolution** defaults to `ms`. The R pipeline lost sub-second
precision by writing reads to CSV and reading them back, so its 50 s threshold
was applied to whole-second times; `time_resolution="s"` reproduces that.

**Foreign reader reads** — detections of this trial's animals on another
enclosure's reader — are dropped by default and counted in the run's warnings.
They are real events but not valid observations of this arena.

**Zero-length GBI intervals** occur when one animal's bout ends exactly as
another's begins. Both animals count as present: these are the clearest
hand-off events in the data, and an implementation that probes slightly later
loses all of them.

**Two network metrics were renamed** from the R pipeline, which mislabelled
them. `net_spectral_radius` (was `net_eigen_centrality`) is the leading
eigenvalue of the unweighted adjacency, not a centralisation score.
`net_mean_dist_weighted` (was `net_mean_dist`) is a weighted shortest-path
length, not a count of edges — with SRI weights it lands near 0.02.

## Dependencies

`pandas`, `numpy`, `openpyxl` (Excel input), and `python-igraph` for the social
network layer.

> On Windows the `python-igraph` wheel needs `VCOMP140.DLL`, the MSVC OpenMP
> runtime, which is absent unless the Visual C++ Redistributable is installed.
> The module falls back to a copy shipped inside scikit-learn if one is present;
> otherwise it raises with an explanation rather than a bare "DLL load failed".

## Validation

Every stage was validated row-for-row against the outputs of the R pipeline it
replaces, on the 2021_LID two-trial dataset: 1,402,867 reads, 70,796 bouts,
43,837 + 45,810 GBI intervals, 510 ownership rows, 1,355 displacement events,
17,589 broad contact events, and 508 node × 24 network statistics rows all
reproduce exactly. See `tests/rfid/` for the properties that are pinned.
