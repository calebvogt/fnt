"""Readers for Biomark reader downloads ("Export All Records To File").

A download carries Scan Date, Scan Time, Download Date/Time, Reader ID, Antenna
ID, HEX Tag ID, DEC Tag ID, Signal mV and Is Duplicate. Biomark Device Manager
writes it as a fixed-width ``.txt``, an ``.xlsx``, or a ``.csv``, and a person
often keeps both a ``.txt`` and an ``.xlsx`` of the same download.

Those twins are not interchangeable. In the 2021 LID dataset 8 of the 32
``.txt`` exports stop at about 61,000 rows, the last line cut mid-record, while
the ``.xlsx`` of the same download runs to 137,000. A truncated ``.txt`` looks
well formed apart from that last line, so nothing downstream notices - about
17% of the reads are simply gone. Hence the rules here:

* When a ``.txt`` (or ``.csv``) and an ``.xlsx`` share a folder and a stem, the
  spreadsheet is used. The twin is still read, and a twin with FEWER records
  (or with records the spreadsheet lacks) is reported.
* Every file's last line is checked; a malformed one is dropped and reported.
* Everything is read as TEXT. Tag numbers are 15-significant-digit decimals
  that lose digits as floats, and a scan time carries milliseconds a float
  would round.

Only scan date/time, reader, antenna and the tag (DEC, HEX as a fallback) are
kept. Download Date/Time says when the data was collected, not where the
animal was; Signal mV is almost always blank; and Is Duplicate is the reader's
own "same tag as last time" flag, carried by nearly every read because an
animal sitting on an antenna is read about once a second - those repeats are
what gives a bout its duration, so dropping them would destroy the signal.

Reading 32 spreadsheets with openpyxl costs about a minute. Parsing the sheet
XML directly is several times faster: one regex pass for the Biomark layout
(one sheet, string cells, no references), an array-indexed pass for a file
Excel has re-saved (shared strings, references, empty cells left out), and
openpyxl only as the last resort.
"""

from __future__ import annotations

import html
import os
import re
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..config.defaults import normalise_id

SPREADSHEET_EXTS = (".xlsx", ".xlsm", ".xls")
TEXT_EXTS = (".txt", ".csv", ".tsv")
DOWNLOAD_EXTS = SPREADSHEET_EXTS + TEXT_EXTS
# Which member of a same-stem twin set is trusted: the spreadsheet first,
# because it is the one that was complete when the .txt was not.
FORMAT_PREFERENCE = {".xlsx": 0, ".xlsm": 1, ".xls": 2, ".csv": 3, ".tsv": 4,
                     ".txt": 5}

EXCEL_EPOCH = pd.Timestamp("1899-12-30")

# Canonical column -> the header spellings seen in the wild (lower-cased,
# punctuation folded to spaces: "Signal,mV" -> "signal mv").
COLUMN_ALIASES = {
    "scan_date": ("scan date", "scan_date", "date"),
    "scan_time": ("scan time", "scan_time", "time"),
    "reader": ("reader id", "reader_id", "reader"),
    "antenna": ("antenna id", "antenna_id", "antenna", "ant"),
    "tag_dec": ("dec tag id", "dec_tag_id", "tag id", "tag_id", "tag", "dec tag",
                "pit tag"),
    "tag_hex": ("hex tag id", "hex_tag_id", "hex tag", "hex id"),
    "signal_mv": ("signal mv", "signal_mv", "signal"),
    "is_duplicate": ("is duplicate", "is_duplicate", "duplicate"),
}
REQUIRED = ("scan_date", "scan_time", "reader", "antenna")

READ_COLUMNS = ["t", "reader", "antenna", "tag", "tag_hex", "file"]


class NotADownload(ValueError):
    """A file in a download folder that is not a reader export (e.g. an old
    pipeline's output CSV). Skipped and listed, never fatal."""


def _fold(text) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def natural_key(path: str):
    """Sort DL_2 before DL_10."""
    name = os.path.basename(path).lower()
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


# --------------------------------------------------------------------------- #
# Tag normalisation
# --------------------------------------------------------------------------- #

def normalise_tag_value(value) -> str:
    """One tag as a canonical string: numbers as 12-decimal text.

    ``985.11300454831`` (a spreadsheet stored the tag as a number and dropped
    the trailing zero) and ``985.113004548310`` (the reader's own text) are the
    same tag; ``sprintf("%.12f")`` of the numeric value makes them the same
    string. Anything that is not a finite number - a HEX ID, a free-text test
    tag - is kept as trimmed, upper-cased text.
    """
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in ("nan", "na", "none", "null"):
        return ""
    try:
        number = float(text)
    except ValueError:
        return text.upper()
    if not np.isfinite(number):
        return text.upper()
    return f"{number:.12f}"


def normalise_tags(values: pd.Series) -> pd.Series:
    """Vector form of :func:`normalise_tag_value`, computed once per distinct tag.

    A dataset has a few hundred distinct tags and over a million reads, so the
    per-value work is done on the uniques and mapped back.
    """
    uniques = pd.unique(values)
    lookup = {u: normalise_tag_value(u) for u in uniques}
    return values.map(lookup)


def normalise_ids(values: pd.Series) -> pd.Series:
    uniques = pd.unique(values)
    return values.map({u: normalise_id(u) for u in uniques})


# --------------------------------------------------------------------------- #
# Timestamps
# --------------------------------------------------------------------------- #

_DATE_FORMATS = {"MDY": ("%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y"),
                 "DMY": ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d.%m.%Y")}


def _parse_date_text(text, date_order: str = "MDY"):
    """A scan date as a midnight Timestamp, or NaT.

    Handles the reader's own ``MM/DD/YYYY``, ISO dates, and an Excel serial
    number (``44322``) left behind when someone opened and re-saved the
    spreadsheet and Excel turned the text into a date.
    """
    if text is None:
        return pd.NaT
    s = str(text).strip()
    if not s or s.lower() in ("nan", "none", "na"):
        return pd.NaT
    try:
        serial = float(s)
        if 1 < serial < 200000:
            return EXCEL_EPOCH + pd.Timedelta(days=int(np.floor(serial)))
        return pd.NaT
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d") + _DATE_FORMATS.get(date_order, ()):
        try:
            return pd.Timestamp(datetime.strptime(s.split(" ")[0], fmt))
        except ValueError:
            continue
    parsed = pd.to_datetime(s, errors="coerce", dayfirst=(date_order == "DMY"))
    return parsed.normalize() if not pd.isna(parsed) else pd.NaT


def _parse_time_text(text):
    """A scan time that ``to_timedelta`` refused: Excel fraction or AM/PM."""
    if text is None:
        return pd.NaT
    s = str(text).strip()
    if not s:
        return pd.NaT
    try:
        fraction = float(s)
        if 0 <= fraction < 1:
            return pd.Timedelta(seconds=round(fraction * 86400, 3))
        return pd.NaT
    except ValueError:
        pass
    parsed = pd.to_datetime(s, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT
    return parsed - parsed.normalize()


def parse_scan_times(dates: pd.Series, times: pd.Series,
                     date_order: str = "MDY") -> pd.Series:
    """Combine Scan Date and Scan Time text into datetimes (local wall clock).

    Vectorised the cheap way: dates repeat (one per day), so they are parsed
    once per distinct value; times go through ``to_timedelta``, which reads
    ``16:59:35.260`` natively. Only the values that fail are retried one by
    one - usually none.
    """
    dates = dates.astype(object)
    times = times.astype(object)
    day_lookup = {d: _parse_date_text(d, date_order) for d in pd.unique(dates)}
    base = pd.to_datetime(dates.map(day_lookup))
    # to_timedelta reads "4:59:35 PM" as 04:59:35 without complaint, and
    # "1899-12-30 16:59:35" not at all - both go to the careful parser
    text = times.astype(str)
    careful = text.str.contains(r"[AaPp]\.?[Mm]|\d{4}-\d", regex=True, na=False)
    offset = pd.to_timedelta(times.where(times.notna() & ~careful, None), errors="coerce")
    retry = offset.isna() & times.notna() & (text.str.strip() != "")
    if retry.any():
        lookup = {t: _parse_time_text(t) for t in pd.unique(times[retry])}
        offset = offset.where(~retry, pd.to_timedelta(times[retry].map(lookup)))
    return base + offset


# --------------------------------------------------------------------------- #
# Format readers - each returns (raw text DataFrame, info dict)
# --------------------------------------------------------------------------- #

_ROW = re.compile(r"<(?:\w+:)?row\b[^>]*?(?:/>|>(.*?)</(?:\w+:)?row>)", re.S)
_CELL = re.compile(r"<(?:\w+:)?c\b([^>]*?)(?:/>|>(.*?)</(?:\w+:)?c>)", re.S)
_ATTR_R = re.compile(r'\br="([A-Z]+)\d*"')
_ATTR_T = re.compile(r'\bt="(\w+)"')
_V = re.compile(r"<(?:\w+:)?v>(.*?)</(?:\w+:)?v>", re.S)
_T = re.compile(r"<(?:\w+:)?t\b[^>]*>(.*?)</(?:\w+:)?t>", re.S)
_SI = re.compile(r"<(?:\w+:)?si\b[^>]*>(.*?)</(?:\w+:)?si>", re.S)


def _unescape(text: str) -> str:
    return html.unescape(text) if text and "&" in text else text


def _col_index(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _first_sheet_path(z: zipfile.ZipFile) -> str:
    names = z.namelist()
    try:
        wb = z.read("xl/workbook.xml").decode("utf-8", "replace")
        rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8", "replace")
        rid = re.search(r"<(?:\w+:)?sheet\b[^>]*\br:id=\"([^\"]+)\"", wb).group(1)
        target = None
        for m in re.finditer(r"<Relationship\b([^>]*)/?>", rels):
            attrs = m.group(1)
            if re.search(rf'\bId="{re.escape(rid)}"', attrs):
                target = re.search(r'\bTarget="([^"]+)"', attrs).group(1)
                break
        if target:
            path = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
            path = os.path.normpath(path).replace("\\", "/")
            if path in names:
                return path
    except Exception:
        pass
    sheets = sorted(n for n in names if n.startswith("xl/worksheets/") and n.endswith(".xml"))
    if not sheets:
        raise ValueError("no worksheet in workbook")
    return sheets[0]


def _shared_strings(z: zipfile.ZipFile) -> list[str]:
    try:
        xml = z.read("xl/sharedStrings.xml").decode("utf-8", "replace")
    except KeyError:
        return []
    return [_unescape("".join(_T.findall(si))) for si in _SI.findall(xml)]


def _parse_sheet_xml(xml: str, shared: list[str]) -> list[list]:
    """Rows of cell text from sheet XML, honouring cell references if present."""
    has_refs = re.search(r"<(?:\w+:)?c\b[^>]*\br=\"", xml) is not None
    rows = []
    for m in _ROW.finditer(xml):
        body = m.group(1) or ""
        values: list = []
        for c in _CELL.finditer(body):
            attrs, inner = c.group(1), c.group(2) or ""
            kind = _ATTR_T.search(attrs)
            kind = kind.group(1) if kind else "n"
            if kind == "inlineStr":
                value = _unescape("".join(_T.findall(inner)))
            else:
                v = _V.search(inner)
                value = _unescape(v.group(1)) if v else ""
                if kind == "s" and value != "":
                    value = shared[int(value)]
            ref = _ATTR_R.search(attrs) if has_refs else None
            if ref:
                # a writer that skips empty cells says where the next one goes
                idx = _col_index(ref.group(1))
                while len(values) < idx:
                    values.append("")
                if idx < len(values):
                    values[idx] = value
                    continue
            values.append(value)
        rows.append(values)
    return rows


def _fast_sheet(xml: str, shared: list[str]):
    """The whole sheet in one regex pass, when every row is the same shape.

    Biomark sheets have no cell references and every row carries all ten
    cells, so the cells can be pulled out in one ``findall`` and reshaped.
    Returns None whenever that assumption does not hold, and the other parsers
    take over.
    """
    m = re.search(r"<(\w+:)?sheetData\b", xml)
    if m is None:
        return None
    p = m.group(1) or ""
    # Cell references are all-or-nothing within a writer, so the first cells
    # are enough to tell.
    head = xml[m.start():m.start() + 20000]
    if any(re.search(r'\br="', a) for a in re.findall(rf"<{p}c\b([^>]*)", head)):
        return None
    if "inlineStr" in xml:
        return None
    # str.count is a C-speed scan; the regexes below are the only full passes.
    n_rows = sum(xml.count(f"<{p}row{e}") for e in (">", " ", "/"))
    n_cells = sum(xml.count(f"<{p}c{e}") for e in (">", " ", "/"))
    if n_rows == 0 or n_cells % n_rows:
        return None
    if shared and 't="s"' in xml:
        cells = re.findall(
            rf"<{p}c\b([^>]*?)(?:/>|>(?:<{p}v>([^<]*)</{p}v>)?</{p}c>)", xml)
        if len(cells) != n_cells:
            return None
        values = [shared[int(v)] if 't="s"' in a and v != "" else v for a, v in cells]
    else:
        values = re.findall(rf"<{p}v>([^<]*)</{p}v>", xml)
        if len(values) != n_cells:
            return None
    if "&" in xml:
        values = [_unescape(v) for v in values]
    return np.array(values, dtype=object).reshape(n_rows, n_cells // n_rows)


def _referenced_sheet(xml: str, shared: list[str]):
    """A sheet whose every cell carries its reference (``r="C12"``), placed by
    array indexing rather than row by row.

    This is what Excel writes when someone opens a download and saves it:
    shared strings, a reference on every cell, and empty cells simply left
    out (openpyxl writes the same with inline strings). One regex pass pulls
    (column, row, attributes, value) for every cell; numpy does the rest.
    Returns None for anything else (a cell without a reference, rich text),
    and the row parser takes over.
    """
    m = re.search(r"<(\w+:)?sheetData\b", xml)
    if m is None:
        return None
    p = m.group(1) or ""
    n_cells = sum(xml.count(f"<{p}c{e}") for e in (">", " ", "/"))
    # value: <v>...</v> or a plain inline string <is><t>...</t></is>
    cells = re.findall(
        rf'<{p}c r="([A-Z]+)(\d+)"([^>]*?)(?:/>|>(?:<{p}f\b[^>]*>[^<]*</{p}f>|'
        rf'<{p}f\b[^>]*/>)?(?:<{p}v>([^<]*)</{p}v>|<{p}is><{p}t\b[^>]*>([^<]*)'
        rf'</{p}t></{p}is>)?</{p}c>)', xml)
    if not cells or len(cells) != n_cells:
        return None
    letters, digits, attrs, values, inline = (np.array(c, dtype=object) for c in zip(*cells))
    values = np.where(values == "", inline, values)
    if shared:
        is_shared = (np.char.find(attrs.astype(str), 't="s"') >= 0) & (values != "")
        if is_shared.any():
            lookup = np.array(shared, dtype=object)
            values[is_shared] = lookup[values[is_shared].astype(np.int64)]
    if "&" in xml:
        escaped = np.char.find(values.astype(str), "&") >= 0
        values[escaped] = [html.unescape(v) for v in values[escaped]]
    col_of = {c: _col_index(c) for c in set(letters)}
    col = np.array([col_of[c] for c in letters], dtype=np.int64)
    row_numbers, row = np.unique(digits.astype(np.int64), return_inverse=True)
    grid = np.full((len(row_numbers), int(col.max()) + 1), "", dtype=object)
    grid[row, col] = values
    return grid


def _typed_to_text(v) -> str:
    """openpyxl values back to the text the reader wrote."""
    if v is None:
        return ""
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S.%f")
    if hasattr(v, "hour") and hasattr(v, "minute"):
        return v.strftime("%H:%M:%S.%f")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def read_xlsx_text(path: str) -> pd.DataFrame:
    """The first sheet of a workbook as text, header row included as row 0."""
    try:
        with zipfile.ZipFile(path) as z:
            sheet = _first_sheet_path(z)
            shared = _shared_strings(z)
            xml = z.read(sheet).decode("utf-8", "replace")
        grid = _fast_sheet(xml, shared)
        if grid is None:
            grid = _referenced_sheet(xml, shared)
        if grid is None:
            rows = _parse_sheet_xml(xml, shared)
            width = max((len(r) for r in rows), default=0)
            grid = np.array([r + [""] * (width - len(r)) for r in rows], dtype=object)
        return pd.DataFrame(grid)
    except (zipfile.BadZipFile, KeyError, ValueError, AttributeError, IndexError):
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb.worksheets[0]
            rows = [[_typed_to_text(v) for v in row]
                    for row in ws.iter_rows(values_only=True)]
        finally:
            wb.close()
        width = max((len(r) for r in rows), default=0)
        return pd.DataFrame([r + [""] * (width - len(r)) for r in rows])


def _grid_to_frame(grid: pd.DataFrame, path: str) -> pd.DataFrame:
    """Promote the header row: the first row that names a Scan Date column."""
    for i in range(min(len(grid), 40)):
        folded = {_fold(v) for v in grid.iloc[i].tolist()}
        if "scan date" in folded or ("date" in folded and "reader id" in folded):
            header = [str(v).strip() for v in grid.iloc[i].tolist()]
            body = grid.iloc[i + 1:].reset_index(drop=True)
            body.columns = _dedupe_names(header)
            return body
    raise NotADownload(f"{os.path.basename(path)}: no 'Scan Date' header row")


def _dedupe_names(names: list[str]) -> list[str]:
    out, seen = [], {}
    for n in names:
        n = n or "unnamed"
        if n in seen:
            seen[n] += 1
            n = f"{n}.{seen[n]}"
        else:
            seen[n] = 0
        out.append(n)
    return out


def read_biomark_txt(path: str) -> tuple[pd.DataFrame, dict]:
    """Read a Biomark fixed-width ``.txt`` export.

    A preamble, a header row, then a ruler of dashes whose runs give the exact
    column widths. Parsing by that ruler rather than by whitespace matters:
    ``Signal,mV`` and ``Is Duplicate`` are usually blank, so splitting on runs
    of spaces shifts later columns into earlier ones on exactly those rows.

    ``info`` reports whether the file ends with a newline and whether its last
    line is a complete record; see :func:`check_last_row`.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        text = fh.read()
    lines = text.splitlines()
    ruler = None
    for i, line in enumerate(lines[:80]):
        stripped = line.strip()
        if stripped and "-" in stripped and set(stripped) <= {"-", " "} and i > 0:
            ruler = i
            break
    if ruler is None:
        return _read_delimited_text(path, text)

    rule = lines[ruler]
    colspecs = [(m.start(), m.end()) for m in re.finditer(r"-+", rule)]
    header = lines[ruler - 1]
    names = [header[a:b].strip() for a, b in colspecs]
    # The last column's header can run past its dashes; widen each column to
    # the start of the next so a long value is not clipped.
    widened = [(a, colspecs[k + 1][0] if k + 1 < len(colspecs) else None)
               for k, (a, _b) in enumerate(colspecs)]
    body = pd.Series([ln for ln in lines[ruler + 1:] if ln.strip()], dtype=object)
    frame = pd.DataFrame({name: body.str.slice(a, b).str.strip()
                          for name, (a, b) in zip(_dedupe_names(names), widened)})
    info = {"ends_with_newline": text.endswith(("\n", "\r")),
            "line_width": len(rule.rstrip())}
    return frame, info


def _read_delimited_text(path: str, text: str) -> tuple[pd.DataFrame, dict]:
    """A .txt/.csv that is delimited rather than fixed width."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines[:40]):
        if "scan date" in _fold(line):
            start = i
            break
    if start is None:
        raise NotADownload(f"{os.path.basename(path)}: no 'Scan Date' header "
                           f"(not a Biomark export?)")
    header_line = lines[start]
    sep = "\t" if header_line.count("\t") > header_line.count(",") else ","
    import io
    frame = pd.read_csv(io.StringIO("\n".join(lines[start:])), sep=sep, dtype=str,
                        keep_default_na=False, skipinitialspace=True,
                        on_bad_lines="skip", engine="python")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame, {"ends_with_newline": text.endswith(("\n", "\r"))}


def peek_is_download(path: str) -> bool:
    """Cheap check on a text file's first lines before reading all of it."""
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            head = [next(fh, "") for _ in range(40)]
    except OSError:
        return False
    return any("scan date" in _fold(line) for line in head)


def read_export(path: str) -> tuple[pd.DataFrame, dict]:
    """Any supported export as a text frame with its original headers."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        return _grid_to_frame(read_xlsx_text(path), path), {}
    if ext == ".xls":
        try:
            grid = pd.read_excel(path, header=None, dtype=str).fillna("")
        except ImportError as exc:
            raise ValueError("reading .xls needs the 'xlrd' package "
                             "(pip install xlrd), or save the file as .xlsx") from exc
        return _grid_to_frame(grid, path), {}
    if ext in TEXT_EXTS:
        if not peek_is_download(path):
            raise NotADownload(f"{os.path.basename(path)}: no 'Scan Date' header "
                               f"in its first lines (not a Biomark export)")
        if ext == ".txt":
            return read_biomark_txt(path)
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            return _read_delimited_text(path, fh.read())
    raise ValueError(f"unsupported file type {ext}")


def standardise(frame: pd.DataFrame, path: str) -> pd.DataFrame:
    """Map an export's headers onto the canonical names."""
    lookup = {_fold(c): c for c in frame.columns}
    rename = {}
    for want, options in COLUMN_ALIASES.items():
        for option in options:
            col = lookup.get(_fold(option))
            if col is not None and col not in rename:
                rename[col] = want
                break
    out = frame.rename(columns=rename)
    missing = [c for c in REQUIRED if c not in out.columns]
    if "tag_dec" not in out.columns and "tag_hex" not in out.columns:
        missing.append("tag")
    if missing:
        raise NotADownload(f"{os.path.basename(path)}: missing column(s) {missing}; "
                           f"found {list(frame.columns)[:12]}")
    for col in ("tag_dec", "tag_hex"):
        if col not in out.columns:
            out[col] = ""
    keep = ["scan_date", "scan_time", "reader", "antenna", "tag_dec", "tag_hex"]
    out = out[keep].copy()
    for col in keep:
        out[col] = out[col].fillna("").astype(str).str.strip()
    return out


def check_last_row(std: pd.DataFrame, t: pd.Series) -> bool:
    """Whether the last row of an export is a complete record.

    A truncated ``.txt`` ends mid-record: the last line's tag is shorter than
    everyone else's, or its date/time does not parse, or it stops before the
    reader and antenna. Any of those marks it.
    """
    if std.empty:
        return True
    last = std.iloc[-1]
    if pd.isna(t.iloc[-1]):
        return False
    if not last["reader"] or not last["antenna"]:
        return False
    tag = last["tag_dec"] or last["tag_hex"]
    if not tag:
        return False
    column = std["tag_dec"] if last["tag_dec"] else std["tag_hex"]
    lengths = column.str.len()
    lengths = lengths[lengths > 0]
    if len(lengths) > 3:
        modal = int(lengths.mode().iloc[0])
        if len(tag) < modal and "." in tag:
            return False
    return True


# --------------------------------------------------------------------------- #
# Folder scan and the union of every download
# --------------------------------------------------------------------------- #

def list_download_files(dirs: list[str], include_subfolders: bool = False) -> list[str]:
    """Candidate download files, skipping hidden files, Office lock files,
    and anything inside an FNT analysis folder."""
    out = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        if include_subfolders:
            walker = os.walk(d)
        else:
            walker = [(d, [], os.listdir(d))]
        for root, subdirs, files in walker:
            subdirs[:] = [s for s in subdirs if not s.endswith("_FNT_analysis")
                          and not s.startswith(".")]
            for f in files:
                if f.startswith((".", "~$")):
                    continue
                if os.path.splitext(f)[1].lower() in DOWNLOAD_EXTS:
                    out.append(os.path.join(root, f))
    return sorted(set(out), key=lambda p: (os.path.dirname(p), natural_key(p)))


def plan_twin_group(path: str) -> tuple:
    """Files that are the same download in different formats share this key."""
    return (os.path.dirname(path), os.path.splitext(os.path.basename(path))[0].lower())


def plan_twins(paths: list[str]) -> dict[str, str | None]:
    """For each file, the preferred twin it defers to (None if it is used)."""
    groups: dict[tuple, list[str]] = {}
    for p in paths:
        groups.setdefault(plan_twin_group(p), []).append(p)
    plan = {}
    for members in groups.values():
        members.sort(key=lambda p: FORMAT_PREFERENCE.get(os.path.splitext(p)[1].lower(), 9))
        plan[members[0]] = None
        for other in members[1:]:
            plan[other] = members[0]
    return plan


@dataclass
class DownloadResult:
    """The deduplicated reads plus one row per file saying what it gave."""
    reads: pd.DataFrame
    files: pd.DataFrame
    raw_rows: int = 0            # data rows in the files used
    records: int = 0             # of those, complete records
    warnings: list[str] = field(default_factory=list)


FILE_REPORT_COLUMNS = [
    "file", "folder", "format", "size_bytes", "modified", "status", "rows",
    "records", "bad_rows", "last_row_ok", "ends_with_newline", "first_read",
    "last_read", "new_unique_reads", "twin_of", "twin_records",
    "fewer_than_twin", "missing_from_twin", "note"]


def _compact(std: pd.DataFrame, t: pd.Series, file_code: int) -> pd.DataFrame:
    tag = std["tag_dec"].where(std["tag_dec"] != "", std["tag_hex"])
    return pd.DataFrame({
        "t": t.to_numpy(),
        "reader": normalise_ids(std["reader"]).to_numpy(),
        "antenna": normalise_ids(std["antenna"]).to_numpy(),
        "tag": normalise_tags(tag).to_numpy(),
        "tag_hex": normalise_tags(std["tag_hex"]).to_numpy(),
        "file": file_code})


def read_downloads(paths: list[str], date_order: str = "MDY",
                   time_resolution: str = "ms", progress=None,
                   cancelled=None) -> DownloadResult:
    """Read every export, keep the preferred member of each twin set, and
    return the union deduplicated on (scan time, reader, antenna, tag).

    Downloads overlap by design - each re-exports records the previous one
    carried - so deduplication is required regardless of twins.
    """
    say = progress or (lambda _m: None)
    plan = plan_twins(paths)
    report: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    order = sorted(paths, key=lambda p: (plan[p] is not None, os.path.dirname(p),
                                         natural_key(p)))
    for i, path in enumerate(order):
        if cancelled and cancelled():
            raise InterruptedError("cancelled")
        name = os.path.basename(path)
        stamp = os.stat(path)
        entry = {"file": name, "folder": os.path.dirname(path),
                 "format": os.path.splitext(path)[1].lower().lstrip("."),
                 "size_bytes": stamp.st_size,
                 "modified": datetime.fromtimestamp(stamp.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                 "twin_of": os.path.basename(plan[path]) if plan[path] else "",
                 "note": ""}
        report[path] = entry
        say(f"Reading downloads... {i + 1}/{len(order)} {name}")
        try:
            raw, info = read_export(path)
            std = standardise(raw, path)
        except NotADownload as exc:
            entry.update(status="not a download", note=str(exc).split(": ", 1)[-1])
            continue
        except Exception as exc:                          # noqa: BLE001 - reported
            entry.update(status="error", note=f"{type(exc).__name__}: {exc}")
            continue
        blank = (std == "").all(axis=1)
        std = std[~blank].reset_index(drop=True)
        t = parse_scan_times(std["scan_date"], std["scan_time"], date_order)
        last_ok = check_last_row(std, t)
        tag_missing = (std["tag_dec"] == "") & (std["tag_hex"] == "")
        bad = t.isna() | (std["reader"] == "") | (std["antenna"] == "") | tag_missing
        if not last_ok:
            bad.iloc[-1] = True
        if time_resolution == "s":
            t = t.dt.floor("s")
        good = ~bad
        compact = _compact(std[good].reset_index(drop=True),
                           t[good].reset_index(drop=True), i)
        entry.update(status="used" if plan[path] is None else "twin (not used)",
                     rows=len(std), records=int(good.sum()), bad_rows=int(bad.sum()),
                     last_row_ok=bool(last_ok),
                     ends_with_newline=info.get("ends_with_newline", ""),
                     first_read=compact["t"].min() if len(compact) else pd.NaT,
                     last_read=compact["t"].max() if len(compact) else pd.NaT)
        if not last_ok:
            entry["note"] = "last line is not a complete record (file truncated?)"
        frames[path] = compact

    key = ["t", "reader", "antenna", "tag"]
    warnings = []
    # A preferred file that could not be read hands over to its best readable
    # twin: a truncated .txt is still better than no download at all.
    for path, target in list(plan.items()):
        if target is not None and target not in frames and path in frames and \
                not any(plan.get(o) is None and o in frames and plan_twin_group(o) ==
                        plan_twin_group(path) for o in plan):
            plan[path] = None
            report[path]["status"] = "used"
            note = f"used because {os.path.basename(target)} could not be read"
            report[path]["note"] = (report[path]["note"] + "; " if report[path]["note"]
                                    else "") + note
            warnings.append(f"{report[path]['file']}: {note}")
    # Twins: compare against the file they defer to.
    for path, target in plan.items():
        if target is None or path not in frames:
            continue
        entry = report[path]
        mine = frames[path]
        if target not in frames:
            entry["note"] = (entry["note"] + "; " if entry["note"] else "") + \
                f"preferred twin {os.path.basename(target)} could not be read"
            continue
        theirs = frames[target]
        entry["twin_records"] = len(theirs)
        entry["fewer_than_twin"] = max(0, len(theirs) - len(mine))
        merged = mine[key].drop_duplicates().merge(
            theirs[key].drop_duplicates(), on=key, how="left", indicator=True)
        extra = int((merged["_merge"] == "left_only").sum())
        entry["missing_from_twin"] = extra
        if len(mine) < len(theirs):
            note = (f"{len(theirs) - len(mine):,} fewer records than "
                    f"{os.path.basename(target)} (truncated export)")
            entry["note"] = (entry["note"] + "; " if entry["note"] else "") + note
            warnings.append(f"{entry['file']}: {note}")
        if extra:
            note = (f"{extra:,} records not in {os.path.basename(target)}, which "
                    f"was used instead - check both files")
            entry["note"] = (entry["note"] + "; " if entry["note"] else "") + note
            warnings.append(f"{entry['file']}: {note}")

    used = [p for p in order if plan[p] is None and p in frames]
    for p in order:
        if report[p].get("last_row_ok") is False:
            warnings.append(f"{report[p]['file']}: last line is not a complete "
                            f"record and was dropped")
    if not used:
        reads = pd.DataFrame({c: pd.Series(dtype=object) for c in READ_COLUMNS})
        reads["t"] = pd.to_datetime(reads["t"])
        raw_rows = records = 0
    else:
        combined = pd.concat([frames[p] for p in used], ignore_index=True)
        raw_rows = int(sum(report[p]["rows"] for p in used))
        records = int(sum(report[p]["records"] for p in used))
        dup = combined.duplicated(subset=key, keep="first")
        # attribute each unique read to the first file (in download order)
        # that carried it, so a re-download reads differently from new data
        new_counts = combined.loc[~dup, "file"].value_counts()
        for p in used:
            code = order.index(p)
            report[p]["new_unique_reads"] = int(new_counts.get(code, 0))
        reads = combined[~dup].reset_index(drop=True)
        names = {order.index(p): os.path.basename(p) for p in used}
        reads["file"] = reads["file"].map(names).astype("category")
        for col in ("reader", "antenna", "tag", "tag_hex"):
            reads[col] = reads[col].astype("category")

    files = pd.DataFrame([report[p] for p in order])
    for col in FILE_REPORT_COLUMNS:
        if col not in files.columns:
            files[col] = np.nan
    files = files[FILE_REPORT_COLUMNS]
    return DownloadResult(reads=reads, files=files, raw_rows=raw_rows,
                          records=records, warnings=warnings)
