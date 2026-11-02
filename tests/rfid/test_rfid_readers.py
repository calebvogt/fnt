"""Reading Biomark downloads: the ruler, twins, truncation, and deduplication.

The case that motivates most of this file: in the 2021 LID dataset 8 of the 32
.txt exports stop at ~61,000 rows with the last line cut mid-record, while the
.xlsx of the same download is complete. A reader that trusted the .txt lost
17% of the reads with no error anywhere.
"""
import os

import pandas as pd
import pytest

from _rfid_synth import (HEADERS, TAGS, record, txt_text, write_biomark_xlsx,
                         write_csv, write_excel_saved_xlsx, write_txt,
                         write_xlsx_openpyxl)
from fnt.rfid.core.file_readers import (list_download_files, parse_scan_times,
                                        read_biomark_txt, read_downloads,
                                        read_xlsx_text)

T0 = pd.Timestamp("2021-05-08 20:00:00.250")


def recs(n, start=T0, step=2.0, reader=1, antenna=3, tag=TAGS["M1a"]):
    return [record(start + pd.Timedelta(seconds=step * i), reader, antenna, tag)
            for i in range(n)]


def test_fixed_width_columns_come_from_the_ruler(tmp_path):
    """Blank Signal/Is Duplicate fields must not shift later columns."""
    rows = [record(T0, 1, 1, TAGS["M1a"], dup=""), record(T0, 2, 13, TAGS["F1a"])]
    frame, info = read_biomark_txt(write_txt(tmp_path / "dl.txt", rows))
    assert list(frame["Reader ID"]) == ["001", "002"]
    assert list(frame["Antenna ID"]) == ["001", "013"]
    assert list(frame["DEC Tag ID"]) == [TAGS["M1a"], TAGS["F1a"]]
    assert info["ends_with_newline"]


def test_tag_ids_stay_text(tmp_path):
    frame, _ = read_biomark_txt(write_txt(tmp_path / "dl.txt", recs(3)))
    assert not pd.api.types.is_numeric_dtype(frame["DEC Tag ID"])


def test_truncated_txt_last_line_is_flagged_and_dropped(tmp_path):
    path = write_txt(tmp_path / "DL_1.txt", recs(5), truncate_last=True)
    assert not open(path).read().endswith("\n")
    res = read_downloads([path])
    row = res.files.iloc[0]
    assert row["rows"] == 5 and row["records"] == 4 and row["bad_rows"] == 1
    assert row["last_row_ok"] == False                      # noqa: E712
    assert row["ends_with_newline"] == False                # noqa: E712
    assert len(res.reads) == 4
    assert any("last line" in w for w in res.warnings)


def test_xlsx_twin_is_preferred_and_a_short_txt_is_reported(tmp_path):
    full = recs(10)
    write_biomark_xlsx(tmp_path / "DL_1.xlsx", full)
    write_txt(tmp_path / "DL_1.txt", full[:6], truncate_last=True)
    res = read_downloads(list_download_files([str(tmp_path)]))
    files = res.files.set_index("file")
    assert files.loc["DL_1.xlsx", "status"] == "used"
    assert files.loc["DL_1.txt", "status"] == "twin (not used)"
    assert files.loc["DL_1.txt", "twin_of"] == "DL_1.xlsx"
    assert files.loc["DL_1.txt", "records"] == 5          # last line was cut
    assert files.loc["DL_1.txt", "fewer_than_twin"] == 5
    assert len(res.reads) == 10
    assert any("fewer records" in w for w in res.warnings)


def test_a_twin_with_reads_the_spreadsheet_lacks_is_reported(tmp_path):
    base = recs(4)
    extra = record(T0 + pd.Timedelta(hours=1), 1, 9, TAGS["M1b"])
    write_biomark_xlsx(tmp_path / "DL_2.xlsx", base)
    write_txt(tmp_path / "DL_2.txt", base + [extra])
    res = read_downloads(list_download_files([str(tmp_path)]))
    files = res.files.set_index("file")
    assert files.loc["DL_2.txt", "missing_from_twin"] == 1
    assert len(res.reads) == 4                              # the spreadsheet wins
    assert any("not in DL_2.xlsx" in w for w in res.warnings)


def test_a_readable_twin_stands_in_for_an_unreadable_spreadsheet(tmp_path):
    (tmp_path / "DL_4.xlsx").write_bytes(b"corrupt")
    write_txt(tmp_path / "DL_4.txt", recs(3))
    res = read_downloads(list_download_files([str(tmp_path)]))
    files = res.files.set_index("file")
    assert files.loc["DL_4.xlsx", "status"] == "error"
    assert files.loc["DL_4.txt", "status"] == "used"
    assert len(res.reads) == 3
    assert any("could not be read" in w for w in res.warnings)


def test_biomark_and_excel_workbooks_read_the_same(tmp_path):
    rows = recs(7, reader=2, antenna=11)
    fast = read_xlsx_text(write_biomark_xlsx(tmp_path / "a.xlsx", rows))
    general = read_xlsx_text(write_xlsx_openpyxl(tmp_path / "b.xlsx", rows))
    assert fast.shape == general.shape == (8, 10)
    assert fast.iloc[0].tolist() == HEADERS
    assert fast.values.tolist() == general.values.tolist()


def test_a_download_resaved_by_excel_reads_the_same(tmp_path):
    """Shared strings, cell references, and the blank Signal / Is Duplicate
    cells simply missing from some rows."""
    rows = recs(6)
    rows[2][9] = ""                                       # no 'Yes' on one row
    a = read_downloads([write_biomark_xlsx(tmp_path / "a.xlsx", rows)]).reads
    b = read_downloads([write_excel_saved_xlsx(tmp_path / "b.xlsx", rows)]).reads
    cols = ["t", "reader", "antenna", "tag"]
    assert len(a) == len(b) == 6
    assert a[cols].astype(str).values.tolist() == b[cols].astype(str).values.tolist()
    grid = read_xlsx_text(tmp_path / "b.xlsx")
    assert grid.shape == (7, 10) and grid.iloc[3, 9] == ""


def test_non_download_files_are_skipped_not_fatal(tmp_path):
    """LID's download folders also hold an old pipeline's T001_MOVEBOUT_GBI.csv."""
    write_biomark_xlsx(tmp_path / "DL_1.xlsx", recs(3))
    pd.DataFrame({"trial": ["T001"], "zone": [1]}).to_csv(tmp_path / "T001_GBI.csv",
                                                          index=False)
    res = read_downloads(list_download_files([str(tmp_path)]))
    files = res.files.set_index("file")
    assert files.loc["T001_GBI.csv", "status"] == "not a download"
    assert len(res.reads) == 3


def test_overlapping_downloads_are_deduplicated_on_the_normalised_tag(tmp_path):
    """The same read in two downloads, once with the tag stored as a number
    (trailing zero lost), counts once."""
    tag = TAGS["M3a"]                                       # ...548310
    first = recs(5, tag=tag)
    second = [r[:] for r in first[3:]] + recs(2, start=T0 + pd.Timedelta(minutes=5), tag=tag)
    for r in second:
        r[7] = str(float(r[7]))                             # 985.11300454831
    write_biomark_xlsx(tmp_path / "DL_1.xlsx", first)
    write_csv(tmp_path / "DL_2.csv", second)
    res = read_downloads(list_download_files([str(tmp_path)]))
    assert res.raw_rows == 9
    assert len(res.reads) == 7
    assert set(res.reads["tag"].astype(str)) == {tag}
    per_file = res.files.set_index("file")["new_unique_reads"]
    assert per_file["DL_1.xlsx"] == 5 and per_file["DL_2.csv"] == 2


def test_scan_times_parse_reader_text_and_excel_serials():
    dates = pd.Series(["05/06/2021", "44322", "2021-05-06"])
    times = pd.Series(["16:59:35.260", "0.5", "4:59:35 PM"])
    t = parse_scan_times(dates, times)
    assert t.iloc[0] == pd.Timestamp("2021-05-06 16:59:35.260")
    assert t.iloc[1] == pd.Timestamp("2021-05-06 12:00:00")
    assert t.iloc[2] == pd.Timestamp("2021-05-06 16:59:35")


def test_folder_scan_skips_hidden_lock_and_analysis_files(tmp_path):
    write_biomark_xlsx(tmp_path / "DL_1.xlsx", recs(2))
    write_biomark_xlsx(tmp_path / "._DL_1.xlsx", recs(2))
    (tmp_path / "~$DL_1.xlsx").write_text("lock")
    out = tmp_path / "sub" / "X_FNT_analysis"
    out.mkdir(parents=True)
    write_csv(out / "reads.csv", recs(2))
    write_biomark_xlsx(tmp_path / "sub" / "DL_2.xlsx", recs(2))
    flat = [os.path.basename(p) for p in list_download_files([str(tmp_path)])]
    assert flat == ["DL_1.xlsx"]
    deep = [os.path.basename(p) for p in list_download_files([str(tmp_path)], True)]
    assert sorted(deep) == ["DL_1.xlsx", "DL_2.xlsx"]


def test_downloads_sort_naturally():
    from fnt.rfid.core.file_readers import natural_key
    names = ["DL_10.xlsx", "DL_2.xlsx", "DL_1.xlsx"]
    assert sorted(names, key=natural_key) == ["DL_1.xlsx", "DL_2.xlsx", "DL_10.xlsx"]


def test_whole_second_resolution_is_available_for_legacy_runs(tmp_path):
    path = write_txt(tmp_path / "dl.txt", recs(2, step=0.4))
    ms = read_downloads([path]).reads["t"]
    s = read_downloads([path], time_resolution="s").reads["t"]
    assert ms.dt.microsecond.max() > 0
    assert (s.dt.microsecond == 0).all()


def test_unreadable_file_is_reported(tmp_path):
    bad = tmp_path / "broken.xlsx"
    bad.write_bytes(b"not a zip")
    write_txt(tmp_path / "ok.txt", recs(2))
    res = read_downloads(list_download_files([str(tmp_path)]))
    files = res.files.set_index("file")
    assert files.loc["broken.xlsx", "status"] == "error"
    assert len(res.reads) == 2


@pytest.mark.parametrize("text_twin", ["txt", "csv"])
def test_spreadsheet_beats_either_text_twin(tmp_path, text_twin):
    write_biomark_xlsx(tmp_path / "DL_3.xlsx", recs(3))
    if text_twin == "txt":
        (tmp_path / "DL_3.txt").write_text(txt_text(recs(2)))
    else:
        write_csv(tmp_path / "DL_3.csv", recs(2))
    res = read_downloads(list_download_files([str(tmp_path)]))
    assert res.files.set_index("file").loc["DL_3.xlsx", "status"] == "used"
    assert len(res.reads) == 3
