"""The animal table: who wears which tags, and when they were released/trapped.

Field metadata is a spreadsheet a person keeps, and it shows. The LID workbook
has storage notes on row 1 and its header on row 2, a leading space in
``'   trial'``, tag numbers stored as numbers (so ``985.113004548310`` became
``985.11300454831``), and date/time columns that arrive three different ways:

* as real dates and times (openpyxl gives ``datetime`` / ``time``),
* as times on 1899-12-30/31, Excel's zero date, when the cell is a time,
* as TEXT Excel serials (``"44336.0"`` for 2021-05-20, ``"0.40625"`` for
  09:45) when the column also holds a text cell such as ``NA`` and the whole
  column was read as text.

All three are handled here, so the rest of the pipeline sees one release and
one trap Timestamp per animal (local wall clock) and a clean tag table.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time

import numpy as np
import pandas as pd

from ..config.defaults import MetadataSpec
from .file_readers import normalise_tag_value

EXCEL_EPOCH = pd.Timestamp("1899-12-30")
MISSING = {"", "na", "nan", "none", "null", "nat", "-", "n/a"}

# Header spellings the column mapping guesses from, in order of preference.
FIELD_ALIASES = {
    "name": ("name", "animal", "animal_name", "mouse", "id_name", "animal id"),
    "sex": ("sex", "gender"),
    "trial": ("trial", "paddock", "home_paddock", "home", "enclosure", "arena"),
    "phase": ("phase", "status", "cohort", "treatment"),
    "code": ("code", "abbrev", "abbreviation", "short_name"),
    "release_date": ("drop_date", "release_date", "release", "drop", "start_date"),
    "release_time": ("drop_time", "release_time", "start_time"),
    "trap_date": ("trap_date", "removal_date", "end_date", "stop_date"),
    "trap_time": ("trap_time", "removal_time", "end_time", "stop_time"),
}
TAG_HEADER = re.compile(r"^(tag|pit|rfid|pit_tag|rfid_tag)[\s_\-]*\d*$|^tag[\s_\-]?id[\s_\-]*\d*$",
                        re.IGNORECASE)
KNOWN_WORDS = ("name", "sex", "tag", "trial", "code", "id", "date", "time",
               "phase", "strain", "group", "mass", "paddock")


def _is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if value is pd.NaT:
        return True
    return str(value).strip().lower() in MISSING


# --------------------------------------------------------------------------- #
# Excel dates and times
# --------------------------------------------------------------------------- #

def excel_date(value):
    """A calendar date from whatever the spreadsheet gave, or None."""
    if _is_missing(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        stamp = pd.Timestamp(value)
        # a bare time lands on Excel's zero date; it is not a date
        if stamp.year < 1901:
            return None
        return stamp.date()
    if isinstance(value, date):
        return value
    if isinstance(value, time):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        serial = float(value)
        return (EXCEL_EPOCH + pd.Timedelta(days=int(np.floor(serial)))).date() \
            if serial >= 1 else None
    text = str(value).strip()
    try:
        serial = float(text)
        return (EXCEL_EPOCH + pd.Timedelta(days=int(np.floor(serial)))).date() \
            if serial >= 1 else None
    except ValueError:
        pass
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed) or parsed.year < 1901:
        return None
    return parsed.date()


def excel_time(value):
    """A time of day from whatever the spreadsheet gave, or None.

    Serials are rounded to the second, because a time typed as 09:45 is
    stored as 0.40625 of a day and comes back as 09:44:59.99999 otherwise.
    """
    if _is_missing(value):
        return None
    if isinstance(value, time):
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        stamp = pd.Timestamp(value)
        return stamp.time()
    if isinstance(value, (int, float, np.integer, np.floating)):
        fraction = float(value) % 1.0
        seconds = int(round(fraction * 86400)) % 86400
        return (datetime.min + pd.Timedelta(seconds=seconds)).time()
    text = str(value).strip()
    try:
        fraction = float(text) % 1.0
        seconds = int(round(fraction * 86400)) % 86400
        return (datetime.min + pd.Timedelta(seconds=seconds)).time()
    except ValueError:
        pass
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.time()


def _full_datetime(value):
    """A date cell that also carries a time of day, as a Timestamp, or None."""
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value)
    try:
        serial = float(str(value).strip())
        return EXCEL_EPOCH + pd.Timedelta(seconds=round(serial * 86400))
    except ValueError:
        pass
    parsed = pd.to_datetime(str(value).strip(), errors="coerce")
    return None if pd.isna(parsed) else parsed


def combine_date_time(date_value, time_value=None, end_of_day: bool = False):
    """Release / trap moment as a Timestamp (local wall clock), or NaT.

    A date column that already carries a time of day (a full datetime) is
    used as is when the time column is empty. A date with no time at all is
    the START of that day for a release and the END of it for a trap
    (``end_of_day``): trimming to it must never cut into the day itself.
    In LID, Glaucus has a trap date of 15 May but no time - he was found dead
    that day after an 11:29 entry - and midnight would have cut his last day.
    """
    d = excel_date(date_value)
    if d is None:
        return pd.NaT
    t = excel_time(time_value) if not _is_missing(time_value) else None
    if t is None:
        full = _full_datetime(date_value)
        if full is not None and full.time() != time(0):
            return full
    if t is None:
        start = pd.Timestamp(datetime.combine(d, time(0)))
        return start + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1) \
            if end_of_day else start
    return pd.Timestamp(datetime.combine(d, t))


# --------------------------------------------------------------------------- #
# Reading the sheet
# --------------------------------------------------------------------------- #

def list_sheets(path: str) -> list[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True)
        try:
            return list(wb.sheetnames)
        finally:
            wb.close()
    if ext == ".xls":
        return list(pd.ExcelFile(path).sheet_names)
    return []


def read_grid(path: str, sheet: str = "", max_rows: int | None = None) -> list[list]:
    """The raw cell grid, values typed as the spreadsheet stored them."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb[sheet] if sheet and sheet in wb.sheetnames else wb.worksheets[0]
            rows = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if max_rows is not None and i >= max_rows:
                    break
                rows.append(list(row))
        finally:
            wb.close()
    elif ext == ".xls":
        frame = pd.read_excel(path, sheet_name=sheet or 0, header=None,
                              nrows=max_rows)
        rows = frame.astype(object).where(frame.notna(), None).values.tolist()
    else:
        with open(path, "r", newline="", encoding="utf-8-sig") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            except csv.Error:
                dialect = csv.excel
            rows = []
            for i, row in enumerate(csv.reader(fh, dialect)):
                if max_rows is not None and i >= max_rows:
                    break
                rows.append(row)
    # drop trailing empty rows (Excel keeps formatted-but-empty rows around)
    while rows and all(_is_missing(v) for v in rows[-1]):
        rows.pop()
    return rows


def detect_header_row(grid: list[list], scan: int = 20) -> int:
    """0-based index of the row that looks most like a header.

    Scores each of the first rows by how many of its cells are non-numeric
    text, with a bonus for words headers usually contain (name, sex, tag...).
    A data row has text too (names, sexes), but it also has numbers and
    dates; a notes row above the header has text in only a few cells.
    """
    best, best_score = 0, -1.0
    for i, row in enumerate(grid[:scan]):
        cells = [c for c in row if not _is_missing(c)]
        if not cells:
            continue
        texts = [str(c).strip() for c in cells
                 if isinstance(c, str) and not _numeric(c)]
        score = len(texts) - 0.5 * (len(cells) - len(texts))
        lowered = [t.lower() for t in texts]
        score += 2 * sum(any(w in t for w in KNOWN_WORDS) for t in lowered)
        if len(set(lowered)) < len(lowered):         # headers are unique
            score -= len(lowered) - len(set(lowered))
        if score > best_score:
            best, best_score = i, score
    return best


def _numeric(text: str) -> bool:
    try:
        float(str(text).strip())
        return True
    except ValueError:
        return False


def header_names(grid: list[list], header_index: int) -> list[str]:
    row = grid[header_index] if header_index < len(grid) else []
    names, seen = [], {}
    for i, c in enumerate(row):
        n = "" if _is_missing(c) else str(c).strip()
        n = n or f"column_{i + 1}"
        if n in seen:
            seen[n] += 1
            n = f"{n}.{seen[n]}"
        else:
            seen[n] = 0
        names.append(n)
    return names


def guess_mapping(headers: list[str]) -> tuple[dict[str, str], list[str]]:
    """Best-guess ``{field: header}`` and tag columns from the header names."""
    folded = {re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_"): h for h in headers}
    mapping = {}
    for fld, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            key = re.sub(r"[^a-z0-9]+", "_", alias.lower()).strip("_")
            if key in folded and folded[key] not in mapping.values():
                mapping[fld] = folded[key]
                break
    tags = [h for h in headers if TAG_HEADER.match(h.strip())]
    return mapping, tags


def grid_frame(spec: MetadataSpec) -> tuple[pd.DataFrame, int]:
    """The metadata as a DataFrame of raw (typed) values, and the header row used."""
    grid = read_grid(spec.path, spec.sheet)
    if not grid:
        raise ValueError(f"{os.path.basename(spec.path)} is empty")
    header_index = (spec.header_row - 1) if spec.header_row > 0 else detect_header_row(grid)
    names = header_names(grid, header_index)
    body = [list(r) + [None] * (len(names) - len(r)) for r in grid[header_index + 1:]]
    frame = pd.DataFrame([r[:len(names)] for r in body], columns=names, dtype=object)
    return frame, header_index + 1


# --------------------------------------------------------------------------- #
# Animals and tags
# --------------------------------------------------------------------------- #

ANIMAL_COLUMNS = ["animal", "name", "trial", "sex", "phase", "code", "release",
                  "trap", "tags"]


@dataclass
class AnimalTable:
    """The animals and their tags, as the pipeline needs them.

    ``animals``: one row per animal, keyed by ``animal`` - the name, or
    ``"<trial> <name>"`` when a name is reused across trials.
    ``tags``: one row per (tag, animal), tags normalised.
    """
    animals: pd.DataFrame
    tags: pd.DataFrame
    header_row: int = 0
    warnings: list[str] = field(default_factory=list)

    def tag_to_animal(self) -> dict[str, str]:
        return dict(zip(self.tags["tag"], self.tags["animal"]))

    def info(self, column: str) -> dict:
        return dict(zip(self.animals["animal"], self.animals[column]))


def _text(value) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _sex(value) -> str:
    text = _text(value).upper()
    if text.startswith("M"):
        return "M"
    if text.startswith("F"):
        return "F"
    return text


def load_animals(spec: MetadataSpec) -> AnimalTable:
    """Read the metadata into an :class:`AnimalTable`.

    Raises if a tag belongs to more than one animal: a read of that tag could
    then be either of them, and every table built on it would silently pick
    one.
    """
    frame, header_row = grid_frame(spec)
    cols = spec.columns or {}
    name_col = cols.get("name")
    if not name_col or name_col not in frame.columns:
        raise ValueError(f"Metadata name column {name_col!r} not found in "
                         f"{list(frame.columns)[:15]}")
    missing_tags = [c for c in spec.tag_columns if c not in frame.columns]
    if missing_tags:
        raise ValueError(f"Metadata tag column(s) not found: {missing_tags}")
    if not spec.tag_columns:
        raise ValueError("No tag column chosen in the metadata mapping")

    # rows without a name are notes, totals, or formatted-but-empty rows
    names = frame[name_col].map(_text)
    frame = frame[names != ""].reset_index(drop=True)
    names = names[names != ""].reset_index(drop=True)
    warnings = []

    def col(fld):
        c = cols.get(fld)
        if c and c in frame.columns:
            return frame[c]
        return pd.Series([None] * len(frame), dtype=object)

    animals = pd.DataFrame({
        "name": names,
        "trial": col("trial").map(_text),
        "sex": col("sex").map(_sex),
        "phase": col("phase").map(_text),
        "code": col("code").map(_text),
        "release": [combine_date_time(d, t) for d, t in
                    zip(col("release_date"), col("release_time"))],
        "trap": [combine_date_time(d, t, end_of_day=True) for d, t in
                 zip(col("trap_date"), col("trap_time"))],
    })
    if animals["name"].duplicated().any():
        animals["animal"] = (animals["trial"] + " " + animals["name"]).str.strip()
        warnings.append("Animal names repeat across rows, so animals are keyed "
                        "as '<trial> <name>'")
    else:
        animals["animal"] = animals["name"]
    dup = animals["animal"][animals["animal"].duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"Metadata lists the same animal more than once: {dup}")

    rows = []
    for tag_col in spec.tag_columns:
        for animal, value in zip(animals["animal"], frame[tag_col]):
            tag = normalise_tag_value(value)
            if tag:
                rows.append({"tag": tag, "animal": animal, "tag_column": tag_col})
    tags = pd.DataFrame(rows, columns=["tag", "animal", "tag_column"])
    clashes = (tags.groupby("tag")["animal"].nunique())
    clashes = clashes[clashes > 1]
    if len(clashes):
        detail = "; ".join(f"{t}: {sorted(set(tags.loc[tags['tag'] == t, 'animal']))}"
                           for t in clashes.index)
        raise ValueError(f"Tag(s) assigned to more than one animal: {detail}")
    tags = tags.drop_duplicates("tag").reset_index(drop=True)
    animals["tags"] = animals["animal"].map(
        tags.groupby("animal")["tag"].apply(lambda s: ";".join(s))).fillna("")
    untagged = animals.loc[animals["tags"] == "", "animal"].tolist()
    if untagged:
        warnings.append(f"{len(untagged)} animal(s) have no tag: "
                        + ", ".join(untagged[:10]) + ("..." if len(untagged) > 10 else ""))
    return AnimalTable(animals=animals[ANIMAL_COLUMNS], tags=tags,
                       header_row=header_row, warnings=warnings)
