"""Readers for raw RFID reader exports.

The Biomark Device Manager writes a fixed-width ``.txt`` and an ``.xlsx`` for
each download. Neither format is authoritative on its own:

* The ``.txt`` is the reader's native export and needs no Excel round-trip, but
  in the 2021_LID dataset 8 of 32 downloads are TRUNCATED at ~61k rows, losing
  ~17% of the reads. A truncated export looks perfectly well-formed.
* The ``.xlsx`` carried the full record there, but it is slow to parse and is
  the file most likely to have been opened and re-saved by a human (which
  silently degrades timestamps to minute resolution).

So :func:`read_download_dir` reads EVERY file of BOTH formats and deduplicates.
Downloads already overlap heavily - each one re-exports records the previous
one also carried - so deduplication is required regardless, and taking the
union simply makes a truncated file harmless instead of silently lossy.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

# The five columns the pipeline needs, and the names it uses internally.
RAW_COLUMNS = ["Scan Date", "Scan Time", "Reader ID", "Antenna ID", "DEC Tag ID"]
READ_COLUMNS = ["scan_date", "scan_time", "reader_id", "antenna_id", "tag_id"]

# Everything else the reader writes. Deliberately unused:
#   Download Date/Time - an artefact of when the data was collected, not of the
#     animal, and it is what makes downloads overlap in the first place.
#   HEX Tag ID - the same identity as DEC Tag ID in another base.
#   Signal,mV - almost always blank in these exports.
#   Is Duplicate - the reader's own flag for "same tag as the previous read".
#     ~99.9% of reads carry it, because an animal sitting on an antenna is read
#     about once a second, and those repeats are exactly what defines a bout's
#     duration. Dropping them would destroy the signal, not clean it.


def read_biomark_txt(path: str) -> pd.DataFrame:
    """Read a Biomark fixed-width ``.txt`` export.

    The file opens with a few lines of preamble, then a header row, then a
    ruler of dashes (``----  ------  ...``) whose runs give the exact column
    widths. Parsing by that ruler rather than by whitespace matters: the
    ``Signal,mV`` and ``Is Duplicate`` fields are usually blank, so splitting on
    runs of spaces silently shifts later columns into earlier ones.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    ruler = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and "-" in stripped and set(stripped) <= {"-", " "}:
            ruler = i
            break
    if ruler is None or ruler == 0:
        raise ValueError(f"{path}: no column ruler found; not a Biomark export?")

    colspecs, cursor = [], 0
    for token in lines[ruler].rstrip("\n").split():
        start = lines[ruler].index(token, cursor)
        colspecs.append((start, start + len(token)))
        cursor = start + len(token)

    names = [lines[ruler - 1][a:b].strip() for a, b in colspecs]
    return pd.read_fwf(path, colspecs=colspecs, names=names,
                       skiprows=ruler + 1, dtype=str)


def read_rfid_file(path: str) -> pd.DataFrame:
    """Read one export of any supported format, as strings.

    Everything stays ``str`` on the way in. Tag IDs are 15-significant-digit
    decimals that lose their trailing digit the moment they become floats
    (``985.113004548310`` reads back as ``985.11300454831``), and the scan time
    carries milliseconds that a float would round.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_biomark_txt(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    if ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        return pd.read_csv(path, dtype=str, sep=sep)
    raise ValueError(f"Unsupported RFID export format: {path}")


def _standardise(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Map a raw export's columns onto ``READ_COLUMNS``."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    aliases = {
        "scan_date": ("scan date", "scan_date", "date"),
        "scan_time": ("scan time", "scan_time", "time"),
        "reader_id": ("reader id", "reader_id", "reader"),
        "antenna_id": ("antenna id", "antenna_id", "antenna", "ant"),
        "tag_id": ("dec tag id", "dec_tag_id", "tag id", "tag_id", "tag"),
    }
    rename, missing = {}, []
    for want, options in aliases.items():
        for option in options:
            if option in lookup:
                rename[lookup[option]] = want
                break
        else:
            missing.append(want)
    if missing:
        raise ValueError(
            f"{os.path.basename(path)}: could not find column(s) {missing}. "
            f"Found: {list(df.columns)}")
    out = df.rename(columns=rename)[READ_COLUMNS].copy()
    for col in READ_COLUMNS:
        out[col] = out[col].str.strip()
    return out


def read_download_dir(directory: str, patterns: tuple[str, ...] = (
        "*.txt", "*.xlsx", "*.xls", "*.csv")) -> tuple[pd.DataFrame, list[dict]]:
    """Read every export in ``directory`` and return ``(reads, per_file_report)``.

    ``reads`` is the deduplicated union across all files. ``per_file_report``
    carries one entry per file so the caller can show what each contributed -
    which is how a truncated ``.txt`` sitting next to a complete ``.xlsx``
    becomes visible rather than silently halving a trial.
    """
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(directory, pattern)))
    paths = sorted(set(paths), key=lambda p: (os.path.basename(p).lower(), p))
    if not paths:
        raise FileNotFoundError(f"No RFID exports found in {directory}")

    frames, report = [], []
    for path in paths:
        try:
            raw = _standardise(read_rfid_file(path), path)
        except Exception as exc:                      # noqa: BLE001 - reported
            report.append({"file": os.path.basename(path), "rows": 0,
                           "error": str(exc)})
            continue
        frames.append(raw)
        report.append({"file": os.path.basename(path), "rows": len(raw),
                       "error": None})

    if not frames:
        reasons = "\n".join(f"  {e['file']}: {e['error']}" for e in report)
        raise ValueError(
            f"No readable RFID exports in {directory}. Each file failed:\n{reasons}")

    reads = pd.concat(frames, ignore_index=True)
    reads = reads[reads["tag_id"].notna() & (reads["tag_id"] != "")]
    before = len(reads)
    reads = reads.drop_duplicates().reset_index(drop=True)

    # Attribute the union back to each file, so a file that contributed nothing
    # new (a re-download) reads differently from one that was truncated.
    for entry in report:
        entry["unique_total"] = len(reads)
    return reads, report


def summarise_downloads(report: list[dict]) -> str:
    """One line per export, for the run log."""
    lines = []
    for entry in report:
        if entry["error"]:
            lines.append(f"  {entry['file']:<40} FAILED: {entry['error']}")
        else:
            lines.append(f"  {entry['file']:<40} {entry['rows']:>9,} rows")
    return "\n".join(lines)
