# RFID Processing Module

Preprocessing for **passive RFID** (PIT-tag) tracking with Biomark readers:
animals wear one or more tags, antennas sit at resource zones, and the reader
logs every pass. This module turns the reader downloads into analysis-ready
tables, with QC reports on everything it removed or found suspicious. It does
for RFID what `fnt/uwb` does for UWB, and like every FNT tool it is wholly
standalone (no imports from other tools).

User documentation: `docs/tools/rfid.md`.

## Layout

```
fnt/rfid/
    rfid_preprocessing_pyqt.py   RFIDPreprocessingWindow (the GUI)
    rfid_dialogs.py              metadata mapping, arena, identities, lost tags
    rfid_worker.py               QThread wrapper around pipeline.run
    pipeline.py                  run(config) -> RunResult; writes the analysis folder
    config/
        defaults.py              RFIDConfig, Arena (Paddock/Zone/Antenna), presets
        config_manager.py        fnt_config.json, provenance, antenna CSV, paths
    core/
        file_readers.py          Biomark .xlsx/.xls/.txt/.csv, twins, truncation
        metadata.py              animal table: header row, mapping, Excel dates, tags
        cleaning.py              identity, place, cross-paddock, windows, lost tags,
                                 found dead, days
        bout_detector.py         gap/zone/day bouts (+ vectorised legacy R rule)
        summaries.py             animal_days, zone_holders, co_occupancy
        qc.py                    roster, cross-paddock, two places, stationary tags
        gbi_generator.py ...     optional legacy layers (see below)
```

## Pipeline

```
downloads -> reads -> cleaned reads -> bouts -> summaries  (+ QC, + optional layers)
```

```python
from fnt.rfid.config import RFIDConfig, MetadataSpec, liddell_8_zone
from fnt.rfid.pipeline import run

cfg = RFIDConfig(name="LID_2021",
                 download_dirs=[r".../T001_Alpha", r".../T002_Bravo"],
                 timezone="America/New_York", day_origin="2021-05-07 12:00:00")
cfg.arena = liddell_8_zone(2)                      # A = reader 1, B = reader 2
cfg.arena.paddocks[0].trials = ["T001"]            # who lives where
cfg.arena.paddocks[1].trials = ["T002"]
cfg.metadata = MetadataSpec(path=r".../LID_2021_metadata.xlsx", sheet="metadata",
                            columns={"name": "name", "sex": "sex", "trial": "trial",
                                     "release_date": "drop_date",
                                     "release_time": "drop_time"},
                            tag_columns=["tag_1", "tag_2"])
cfg.animals["Calypso"] = {"lost_tags": [{"tag": "985.113004548427",
                                         "start": "2021-05-16 12:00:00"}]}
cfg.animals["Isac"] = {"dead_mode": "last_entry"}

result = run(cfg, progress=lambda pct, msg: print(pct, msg))
result.tables["bouts"]            # every table is returned as well as written
```

## Design decisions

**(reader, antenna) -> zone_id.** Zones are paddock + zone (`A3`), antennas are
keyed by reader and antenna. The earlier module keyed on antenna number alone,
which cannot tell reader 1's antenna 3 (Alpha) from reader 2's (Bravo).

**Other paddock's reader.** A read by another paddock's reader is kept in that
paddock with `away = True` (default), dropped, or folded into the animal's own
paddock as the original pipeline did. The home paddock comes from the
metadata trial value, matched against each paddock's `trials`, ID or name.

**Twins and truncation.** Same-stem `.txt`/`.csv` and `.xlsx`: the spreadsheet
is used; the twin is read and compared, and fewer records or records the
spreadsheet lacks are reported. Every file's last line is checked.

**Tags** are compared as `"%.12f"` strings of their numeric value; non-numeric
tags as trimmed upper-case text. DEC tag first, HEX as a fallback.

**Found dead, last entry.** Entry = a read on an entrance-location antenna
after a gap > bout gap (gap measured over the animal's reads, both tags). The
cut is last entry + one bout gap, reads strictly after it dropped - the same
rule as the R reference. An optional "found at" time bounds the search.

**Days** are `ceil((t - origin) / 24 h)` on the local wall clock: a read
exactly at the origin is the last moment of day 0. Auto origin = noon on the
date of the earliest metadata release (not the first read: readers are tested
before animals go in).

**Bouts** split on animal, zone_id, day, or a gap > `bout_gap_s` (a gap of
exactly the threshold continues). Gaps are compared after rounding to the
microsecond so a 50.000 s gap is not 50.0000000001 s. Ties in time break by
antenna then tag, as in the R reference. Duration = max(stop - start, 1 s).

**Co-occupancy** is an interval sweep per zone_id (sort by start;
`searchsorted` finds each bout's later-starting partners), so the work grows
with the number of overlaps. On LID it agrees to the millisecond with the
GBI-based edgelist on all 752 pair-days.

**Timezone** is optional: with one, gaps are measured on UTC so a DST change
does not add or remove an hour; without one, the reader's clock is taken as
never changing.

**Nothing is written into a download folder**, and nothing is written at all
until every stage has succeeded.

## Optional layers

`gbi_generator`, `edgelist`, `zone_ownership`, `displacement`, `hinde_index`
and `social_network` predate this rebuild. They now run on string zone IDs and
have tests (`tests/rfid/test_rfid_legacy_layers.py`), but they are
interpretive, so `RFIDConfig.layers` has them all off by default. The social
network layer needs `python-igraph`.

## Validation

`tests/rfid/test_rfid_lid_acceptance.py` runs the 2021 LID dataset (when
present) against an independent R reimplementation: 1,523,161 rows ->
1,404,510 unique reads, 1,643 unknown-tag reads, 1,670 away reads, 47,705
lost-tag and 89,459 found-dead reads removed, 65,473 bouts and 1,439
animal-hours in zones - all exact.

## Dependencies

`pandas`, `numpy`, `openpyxl`, `pytz`, PyQt5; `xlrd` only for legacy `.xls`
downloads; `python-igraph` only for the social-network layer.
