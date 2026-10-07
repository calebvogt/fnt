# RFID Tracking

Preprocessing for **passive RFID** (PIT-tag) tracking with Biomark readers: animals wear one or more tags, and antennas at resource zones record every pass. The tool turns the reader downloads into analysis-ready tables and tells you what it had to throw away and why.

Launch from the FNT main GUI **RFID** tab → **RFID Preprocessing Tool**.

## What it does

- **Loads Biomark downloads** ("Export All Records To File") as `.xlsx`, `.xls`, `.txt` (fixed-width) or `.csv`, from one or more folders. Overlapping downloads are deduplicated on scan time, reader, antenna and tag.
- **Guards against truncated exports.** When a `.txt` and an `.xlsx` of the same download sit side by side, the spreadsheet is used and the `.txt` is checked against it; a `.txt` with fewer records, or one whose last line is cut mid-record, is reported. (In one real dataset 8 of 32 `.txt` exports stopped at ~61,000 rows while their `.xlsx` twins ran to 137,000.)
- **Reads the animal metadata** from a spreadsheet as people keep it: header row detected (or chosen), columns mapped in a dialog, any number of tag columns, and Excel dates/times in whatever form they arrive.
- **Describes the arena** as paddocks, readers, zones and antennas. Zones are named by paddock and zone (`A1`…`A8`, `B1`…`B8`) and antennas are keyed by **(reader, antenna)**, so two paddocks whose readers both number their antennas 1–16 are kept apart. Several antennas per zone (e.g. an entrance "wall" antenna and a "floor" antenna) is normal; one per zone works too.
- **Cleans**: unknown tags, unmapped antennas, reads by another paddock's reader (kept as flagged visits by default), optional release/trap windows, **lost tags**, and **animals found dead**.
- **Builds bouts** and per-day summaries: zones entered, minimum distance, time in zones, zone holders, and pairwise co-occupancy.
- **Writes QC reports** for everything above, shown as tabs in the window and saved as CSVs.

## Workflow

1. **Reader Downloads** → *Add Folder…* for each download folder. Folders are only ever read.
2. **Animals (metadata)** → *Choose File…*, pick the sheet, then *Columns…* to check the header row and which columns hold name, sex, trial / home paddock, phase, code, tags, release and trap date/time.
3. **Arena** → *Edit Arena…* → pick a preset (*Liddell 8-zone paddock*, *Two Liddell 8-zone paddocks*, *One antenna per zone*) and adjust. On the *Paddocks* tab, list each paddock's readers and the metadata trial values of the animals that live there. The antenna table can be imported from / exported to CSV (`reader, antenna, zone, location, paddock, x, y`).
4. **Time**: timezone (for gaps across a DST change; leave blank if the reader clock never changes), and when day 1 starts (days run noon to noon by default).
5. **Check (no export)** runs everything and fills the QC tabs without writing anything.
6. Read the **Two places** and **Stationary tags** tabs. Right-click a row to mark a tag as lost from that day, or an animal as found dead; fine-tune under *Animals & Tags…*.
7. **Run and Export**.

## Cleaning rules

| Rule | What it does | Why |
|------|--------------|-----|
| Unknown tags | Reads of tags no animal wears are dropped and listed | Test tags, strays |
| Unmapped antennas | Reads on a (reader, antenna) the arena does not describe are dropped and listed | Usually an incomplete antenna table |
| Other paddock's reader | **Keep** (default, `away` = true), **drop**, or **fold** into the animal's own paddock (legacy) | A hole in a shared wall makes these real visits |
| Release / trap windows | Optional: trim each animal to its metadata release and/or trap time, or to manual times | Pre-release reader tests, post-trapping reads |
| Lost tags | A tag off its animal from a time (to a time): its reads inside are dropped | A tag lying in a tube keeps being read and places the animal there |
| Found dead | Cut at the **last entry** (+ one bout gap), or at a manual time | A body in a zone is read like a resident that never leaves |

An **entry** is a read on an entrance-type antenna (location `wall` by default, configurable) after a gap longer than the bout gap. A mouse cannot help making entries; a body cannot make one. When the time a body was found is given, entries after it are ignored.

Tags are compared as 12-decimal strings, because spreadsheets store tag numbers as numbers and drop trailing zeros (`985.11300454831` is tag `985.113004548310`).

## Bouts and days

Per animal, consecutive reads in the same zone no more than the **bout gap** apart (default 50 s) form a bout; a zone change, a longer gap, or the day boundary ends it. A lone read is a 1 s bout. Days run from the day-1 origin to the same time the next day (noon to noon by default), numbered so that day 1 is *(origin, origin + 24 h]*.

A **legacy** option reproduces the original R pipeline's bout rule for regenerating published numbers; it pairs bout starts and stops by position and occasionally joins reads across a zone change, so it is not the default.

## Outputs

Everything goes to `<name>_FNT_analysis` **beside** the download folders (never inside one), or a folder you choose:

```
<name>_FNT_analysis/
    fnt_config.json          every setting + provenance (FNT version, run time,
                             each input file's size and modification time)
    <name>_messageLog.txt
    csvs/
        <name>_reads.csv             animal, tag, reader, paddock, antenna, location,
                                     zone, zone_id, away, day, t
        <name>_bouts.csv             one row per bout
        <name>_animal_days.csv       reads, bouts, zones entered, minimum distance,
                                     time in zones, per animal and day
        <name>_zone_holders.csv      per paddock-zone-day: the male with > 50% of
                                     that zone's male reads, or 'contested'
        <name>_co_occupancy.csv      per day and pair: seconds in the same zone
                                     at the same time
        <name>_qc_*.csv              files, animals, unknown tags, unmapped antennas,
                                     cross-paddock visits, two places at once,
                                     stationary tags, found dead, dropped reads
```

Minimum distance is the sum of straight lines between the centres of consecutive distinct zones, within a paddock only (a move between paddocks adds nothing: the route is unknown). The holder share threshold and sex are configurable.

!!! tip "Reading the CSVs"
    Read the `tag` column as text. A tag number read as a float displays with fewer digits and can stop matching.

## QC reports

| Tab / file | Look for |
|---|---|
| Files | Truncated `.txt` twins, malformed last lines, files skipped as "not a download" |
| Animals | Reads, days, first/last read against release/trap; animals never read; reads removed by each rule |
| Unknown tags | Tags read that no animal wears |
| Cross-paddock | Visits through a shared wall, per animal |
| Two places | Animal-days whose two tags each live mostly (> 80%) in different zones: a lost tag |
| Stationary tags | Tag-days read by one antenna in most hours: with no entries a body or a dropped tag; with entries on the one antenna, a tag lying at an entrance |
| Found dead | Last entry, cut time and reads removed per animal |
| Dropped reads | Reads removed by each rule |

## Optional layers

Group-by-individual matrix (GBI), co-presence edgelist, male displacement events, Hinde contact indices and social networks (needs `python-igraph`) are carried over from the earlier module. They run on the new zone IDs and are tested, but they are interpretive, so they are off by default (*Bouts & Summaries → Optional layers*).
