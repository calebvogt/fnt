"""The Biomark .txt reader must parse by the ruler, not by whitespace.

A real export leaves Signal,mV and Is Duplicate blank on most rows. Splitting
on runs of spaces then shifts later columns into earlier ones - silently, and
only on the rows where a field happens to be empty, which is most of them.
"""
import textwrap

import pandas as pd
import pytest

from fnt.rfid.core.file_readers import read_biomark_txt, read_download_dir

# Column widths here are taken from a real export; the last two fields are
# blank on the first row and populated on the second, which is the case that
# breaks a whitespace split.
EXPORT = (
    "Biomark Device Manager Version 1.2.10\n"
    "Export All Records To File\n"
    "Export Date/Time: 5/8/2021 4:33:56 PM\n"
    "\n"
    "Scan Date   Scan Time     Download Date  Download Time  Reader ID  "
    "Antenna ID  HEX Tag ID      DEC Tag ID        Signal,mV  Is Duplicate  \n"
    "----------  ------------  -------------  -------------  ---------  "
    "----------  --------------  ----------------  ---------  ------------  \n"
    "05/06/2021  16:59:35.260  05/08/2021     16:30:31       001        "
    "001         3DD.0077E1080A  989.002011236362                           \n"
    "05/06/2021  16:59:36.380  05/08/2021     16:30:31       002        "
    "013         3DD.0077E1080B  985.113004548266             Yes           \n"
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_blank_trailing_fields_do_not_shift_columns(tmp_path):
    df = read_biomark_txt(_write(tmp_path, "dl.txt", EXPORT))
    assert list(df["Reader ID"]) == ["001", "002"]
    assert list(df["Antenna ID"]) == ["001", "013"]
    # the row with no Signal,mV must still land its tag in the tag column
    assert df["DEC Tag ID"].iloc[0] == "989.002011236362"
    assert df["DEC Tag ID"].iloc[1] == "985.113004548266"


def test_tag_ids_are_never_numeric(tmp_path):
    """As floats these lose their trailing digit, which changes the identity.

    985.113004548310 read as a float comes back 985.11300454831 - a different
    tag, and one that matches no animal.
    """
    df = read_biomark_txt(_write(tmp_path, "dl.txt", EXPORT))
    assert not pd.api.types.is_numeric_dtype(df["DEC Tag ID"])
    assert df["DEC Tag ID"].tolist() == ["989.002011236362", "985.113004548266"]


def test_a_file_without_a_ruler_is_rejected(tmp_path):
    path = _write(tmp_path, "bad.txt", "just\nsome\nlines\n")
    with pytest.raises(ValueError, match="ruler"):
        read_biomark_txt(path)


def test_downloads_are_unioned_and_deduplicated(tmp_path):
    """Overlapping downloads are normal; a truncated one must not lose reads.

    The second file here is the first one truncated, plus one read the first
    does not have. Taking the union means the truncation costs nothing.
    """
    _write(tmp_path, "dl_1.txt", EXPORT)
    truncated = EXPORT.rsplit("\n", 2)[0] + "\n" + (
        "05/06/2021  17:02:11.000  05/08/2021     16:30:31       001        "
        "002         3DD.0077E1080C  985.113004548275             Yes           \n")
    _write(tmp_path, "dl_2.txt", truncated)

    reads, report = read_download_dir(str(tmp_path))
    assert len(report) == 2
    # 2 reads from the first file, 1 shared + 1 new from the second
    assert len(reads) == 3
    assert set(reads.columns) == {"scan_date", "scan_time", "reader_id",
                                  "antenna_id", "tag_id"}


def test_a_file_missing_columns_names_them(tmp_path):
    """A silently skipped file is a silently halved trial, so say what broke."""
    _write(tmp_path, "partial.csv",
           "Scan Date,Scan Time,Reader ID\n05/06/2021,16:59:35.260,001\n")
    with pytest.raises(ValueError) as excinfo:
        read_download_dir(str(tmp_path))
    message = str(excinfo.value)
    assert "partial.csv" in message
    assert "antenna_id" in message and "tag_id" in message


def test_one_bad_file_does_not_sink_the_good_ones(tmp_path):
    _write(tmp_path, "dl_1.txt", EXPORT)
    _write(tmp_path, "broken.csv", "nothing,useful\n1,2\n")
    reads, report = read_download_dir(str(tmp_path))
    assert len(reads) == 2
    failed = [e for e in report if e["error"]]
    assert len(failed) == 1 and failed[0]["file"] == "broken.csv"
