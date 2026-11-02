"""Synthetic Biomark downloads and metadata for the RFID tests.

The fixed-width layout is copied from a real Device Manager export (the same
column widths as the LID 2021 files), and the spreadsheet writer reproduces
the real Biomark .xlsx structure - an ``x:`` namespace, ``t="str"`` cells, no
cell references - so the fast sheet parser is exercised on the shape it was
written for. ``write_xlsx_openpyxl`` produces the ordinary Excel structure
(shared strings, cell references) for the general path.
"""

from __future__ import annotations

import zipfile
from datetime import datetime, time

import pandas as pd

PREAMBLE = ("Biomark Device Manager Version 1.2.10\n"
            "Export All Records To File\n"
            "Export Date/Time: 5/8/2021 4:33:56 PM\n\n")
HEADERS = ["Scan Date", "Scan Time", "Download Date", "Download Time", "Reader ID",
           "Antenna ID", "HEX Tag ID", "DEC Tag ID", "Signal,mV", "Is Duplicate"]
WIDTHS = [10, 12, 13, 13, 9, 10, 14, 16, 9, 12]

TAGS = {  # a few realistic tag numbers
    "M1a": "985.113004548266", "M1b": "985.113004548275",
    "F1a": "985.113004548436", "F1b": "985.113004548397",
    "M2a": "985.113004548446", "M2b": "985.113004548419",
    "M3a": "985.113004548310", "M3b": "985.113004548320",
    "test": "989.002011236362",
}


def record(t, reader, antenna, tag, dup="Yes"):
    """One read as the ten Biomark fields (text)."""
    t = pd.Timestamp(t)
    return [t.strftime("%m/%d/%Y"), t.strftime("%H:%M:%S.") + f"{t.microsecond // 1000:03d}",
            "05/20/2021", "16:30:31", f"{int(reader):03d}", f"{int(antenna):03d}",
            "3DD.0077E1080A", tag, "", dup]


def fixed_width_line(fields):
    return "".join(str(f).ljust(w) + "  " for f, w in zip(fields, WIDTHS))


def txt_text(records, truncate_last: bool = False) -> str:
    lines = [fixed_width_line(HEADERS), "".join("-" * w + "  " for w in WIDTHS)]
    lines += [fixed_width_line(r) for r in records]
    text = PREAMBLE + "\n".join(lines) + "\n"
    if truncate_last:
        # cut the last record in the middle of its DEC tag, no newline -
        # what the truncated LID .txt exports look like
        body = text.rstrip("\n")
        cut = body.rfind(records[-1][7]) + 7
        text = body[:cut]
    return text


def write_txt(path, records, truncate_last=False):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(txt_text(records, truncate_last))
    return str(path)


def write_biomark_xlsx(path, records):
    """A workbook with the same XML structure as a real Biomark .xlsx."""
    def row(values):
        return "<x:row>" + "".join(
            f'<x:c s="2" t="str"><x:v>{v}</x:v></x:c>' for v in values) + "</x:row>"
    sheet = ('﻿<?xml version="1.0" encoding="utf-8"?><x:worksheet xmlns:x="http://'
             'schemas.openxmlformats.org/spreadsheetml/2006/main"><x:sheetData>'
             + row(HEADERS) + "".join(row(r) for r in records)
             + "</x:sheetData></x:worksheet>")
    workbook = ('﻿<?xml version="1.0" encoding="utf-8"?><x:workbook xmlns:x="http://'
                'schemas.openxmlformats.org/spreadsheetml/2006/main"><x:sheets>'
                '<x:sheet name="Downloaded Tag IDs" sheetId="1" r:id="R1" xmlns:r="http://'
                'schemas.openxmlformats.org/officeDocument/2006/relationships" />'
                '</x:sheets></x:workbook>')
    rels = ('﻿<?xml version="1.0" encoding="utf-8"?><Relationships xmlns="http://'
            'schemas.openxmlformats.org/package/2006/relationships"><Relationship Type="'
            'http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
            'worksheet" Target="/xl/worksheets/sheet.xml" Id="R1" /></Relationships>')
    types = ('<?xml version="1.0" encoding="utf-8"?><Types xmlns="http://schemas.'
             'openxmlformats.org/package/2006/content-types"><Default Extension="xml" '
             'ContentType="application/xml" /><Default Extension="rels" ContentType="'
             'application/vnd.openxmlformats-package.relationships+xml" /></Types>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        z.writestr("xl/worksheets/sheet.xml", sheet)
    return str(path)


def write_excel_saved_xlsx(path, records):
    """What Excel writes when someone opens a download and saves it: shared
    strings, a reference on every cell, and empty cells left out entirely."""
    strings, index = [], {}

    def sid(text):
        if text not in index:
            index[text] = len(strings)
            strings.append(text)
        return index[text]

    def row(r, values):
        cells = "".join(f'<c r="{chr(65 + c)}{r}" t="s"><v>{sid(v)}</v></c>'
                        for c, v in enumerate(values) if v != "")
        return f'<row r="{r}">{cells}</row>'
    body = row(1, HEADERS) + "".join(row(i + 2, rec) for i, rec in enumerate(records))
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    sheet = f'<?xml version="1.0"?><worksheet {ns}><sheetData>{body}</sheetData></worksheet>'
    sst = (f'<?xml version="1.0"?><sst {ns} count="{len(strings)}">'
           + "".join(f"<si><t>{t}</t></si>" for t in strings) + "</sst>")
    workbook = (f'<?xml version="1.0"?><workbook {ns} xmlns:r="http://schemas.openxmlformats'
                f'.org/officeDocument/2006/relationships"><sheets><sheet name="S" '
                f'sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels = ('<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/'
            'package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", rels)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
        z.writestr("xl/sharedStrings.xml", sst)
    return str(path)


def write_xlsx_openpyxl(path, records):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(HEADERS)
    for r in records:
        ws.append(r)
    wb.save(path)
    return str(path)


def write_csv(path, records):
    pd.DataFrame(records, columns=HEADERS).to_csv(path, index=False)
    return str(path)


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #

META_HEADER = ["trial", "name", "sex", "phase", "code", "tag_1", "tag_2",
               "drop_date", "drop_time", "trap_date", "trap_time"]

ANIMALS = [
    # trial, name, sex, phase, code, tag_1, tag_2, release, trap
    ("T001", "Mars", "M", "early", "MRS", TAGS["M1a"], TAGS["M1b"],
     "2021-05-07 17:48", "2021-05-20 09:20"),
    ("T001", "Flora", "F", "early", "FLR", TAGS["F1a"], TAGS["F1b"],
     "2021-05-07 17:40", "2021-05-20 09:30"),
    ("T002", "Duke", "M", "early", "DKE", TAGS["M2a"], TAGS["M2b"],
     "2021-05-07 18:03", "2021-05-20 10:00"),
    ("T002", "Isac", "M", "late", "ISC", TAGS["M3a"], TAGS["M3b"],
     "2021-05-14 17:20", None),
]


def write_metadata_xlsx(path, animals=ANIMALS, notes_row=True, text_serials=False):
    """Metadata as field people keep it: a notes row above the header, tags
    stored as NUMBERS, dates as dates and times as times - or, with
    ``text_serials``, the trap columns as text Excel serials (what a column
    holding a text cell turns into)."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "metadata"
    if notes_row:
        ws.append([None, None, None, None, None, "stored -20C", None, None, None,
                   None, "field notes"])
    ws.append(["   " + META_HEADER[0]] + META_HEADER[1:])
    for trial, name, sex, phase, code, t1, t2, rel, trap in animals:
        rel = pd.Timestamp(rel)
        row = [trial, name, sex, phase, code, float(t1), float(t2),
               datetime(rel.year, rel.month, rel.day), time(rel.hour, rel.minute)]
        if trap is None:
            row += ["NA", "NA"]
        else:
            tp = pd.Timestamp(trap)
            if text_serials:
                serial = (tp.normalize() - pd.Timestamp("1899-12-30")).days
                fraction = (tp - tp.normalize()).total_seconds() / 86400
                row += [f"{serial}.0", f"{fraction:.5f}"]
            else:
                row += [datetime(tp.year, tp.month, tp.day), time(tp.hour, tp.minute)]
        ws.append(row)
    wb.save(path)
    return str(path)


def lid_like_arena():
    """Two Liddell paddocks, T001 living in A (reader 1), T002 in B (reader 2)."""
    from fnt.rfid.config import liddell_8_zone
    arena = liddell_8_zone(2)
    arena.paddocks[0].trials = ["T001"]
    arena.paddocks[0].name = "Alpha"
    arena.paddocks[1].trials = ["T002"]
    arena.paddocks[1].name = "Bravo"
    return arena


def animal_table(rows):
    """An AnimalTable built directly: rows of (animal, trial, sex, tags list,
    release, trap)."""
    from fnt.rfid.core.metadata import AnimalTable
    animals = pd.DataFrame([{
        "animal": a, "name": a, "trial": trial, "sex": sex, "phase": "", "code": a[:3].upper(),
        "release": pd.Timestamp(rel) if rel else pd.NaT,
        "trap": pd.Timestamp(trap) if trap else pd.NaT,
        "tags": ";".join(tags)} for a, trial, sex, tags, rel, trap in rows])
    tags = pd.DataFrame([{"tag": t, "animal": a, "tag_column": f"tag_{i + 1}"}
                         for a, _tr, _s, tl, _r, _t in rows for i, t in enumerate(tl)])
    return AnimalTable(animals=animals, tags=tags)


def raw_reads(rows):
    """A raw-reads frame as read_downloads returns it: rows of
    (time, reader, antenna, tag)."""
    frame = pd.DataFrame(rows, columns=["t", "reader", "antenna", "tag"])
    frame["t"] = pd.to_datetime(frame["t"])
    frame["reader"] = frame["reader"].astype(str)
    frame["antenna"] = frame["antenna"].astype(str)
    frame["tag_hex"] = ""
    frame["file"] = "synthetic"
    return frame
