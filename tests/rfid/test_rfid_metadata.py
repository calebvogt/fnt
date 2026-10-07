"""Tags and the animal metadata: the spreadsheet as people actually keep it.

LID's workbook has storage notes on row 1 and the header on row 2, tag numbers
stored as numbers (so trailing zeros are gone), and date/time columns that
arrive as datetimes, as times, or as TEXT Excel serials when the column also
holds a text cell. All of that has to come out as one clean table.
"""
from datetime import datetime, time

import pandas as pd
import pytest

from _rfid_synth import TAGS, write_metadata_xlsx
from fnt.rfid.config import MetadataSpec
from fnt.rfid.core.file_readers import normalise_tag_value, normalise_tags
from fnt.rfid.core.metadata import (combine_date_time, detect_header_row,
                                    excel_date, excel_time, guess_mapping,
                                    header_names, load_animals, read_grid)


# ------------------------------------------------------------------ tags ----

def test_a_tag_stored_as_a_number_matches_the_reader_text():
    """985.11300454831 (spreadsheet) is 985.113004548310 (reader)."""
    assert normalise_tag_value("985.113004548310") == "985.113004548310"
    assert normalise_tag_value(985.11300454831) == "985.113004548310"
    assert normalise_tag_value("985.11300454831") == "985.113004548310"
    assert normalise_tag_value(" 985.113004548266 ") == "985.113004548266"


def test_tags_are_twelve_decimal_strings():
    assert normalise_tag_value("989.2") == "989.200000000000"
    assert normalise_tag_value("900228000123456") == "900228000123456.000000000000"


def test_non_numeric_tags_are_trimmed_text():
    assert normalise_tag_value(" 3dd.0077e1080a ") == "3DD.0077E1080A"
    assert normalise_tag_value("") == ""
    assert normalise_tag_value(None) == ""
    assert normalise_tag_value(float("nan")) == ""
    assert normalise_tag_value("NA") == ""
    assert normalise_tag_value("inf") == "INF"


def test_vector_normalisation_matches_the_scalar_one():
    values = pd.Series(["985.11300454831", "985.113004548310", "abc", ""])
    assert normalise_tags(values).tolist() == [
        "985.113004548310", "985.113004548310", "ABC", ""]


# ---------------------------------------------------------- excel values ----

def test_excel_dates_in_every_form():
    assert excel_date(datetime(2021, 5, 20)) == datetime(2021, 5, 20).date()
    assert excel_date("44336.0") == datetime(2021, 5, 20).date()
    assert excel_date(44336) == datetime(2021, 5, 20).date()
    assert excel_date("2021-05-20") == datetime(2021, 5, 20).date()
    # a time on Excel's zero date is not a date
    assert excel_date(datetime(1899, 12, 30, 9, 45)) is None
    assert excel_date("NA") is None
    assert excel_date(None) is None


def test_excel_times_in_every_form():
    assert excel_time(time(9, 45)) == time(9, 45)
    assert excel_time("0.40625") == time(9, 45)
    assert excel_time(0.40625) == time(9, 45)
    assert excel_time(datetime(1899, 12, 31, 9, 45)) == time(9, 45)
    assert excel_time("09:45") == time(9, 45)
    assert excel_time("NA") is None


def test_a_date_without_a_time_is_the_start_or_end_of_that_day():
    assert combine_date_time("44336.0", None) == pd.Timestamp("2021-05-20 00:00")
    end = combine_date_time("44336.0", "NA", end_of_day=True)
    assert end.date() == pd.Timestamp("2021-05-20").date() and end.hour == 23
    assert combine_date_time(datetime(2021, 5, 20, 9, 45), None) == \
        pd.Timestamp("2021-05-20 09:45")


# ------------------------------------------------------------- the sheet ----

def test_header_row_is_found_below_a_notes_row(tmp_path):
    path = write_metadata_xlsx(tmp_path / "meta.xlsx")
    grid = read_grid(path, "metadata")
    assert detect_header_row(grid) == 1                      # row 2
    headers = header_names(grid, 1)
    assert headers[0] == "trial"                             # leading spaces gone
    mapping, tags = guess_mapping(headers)
    assert tags == ["tag_1", "tag_2"]
    assert mapping["name"] == "name" and mapping["trial"] == "trial"
    assert mapping["release_date"] == "drop_date"
    assert mapping["trap_time"] == "trap_time"


def spec_for(path, header_row=0):
    grid = read_grid(path, "metadata")
    mapping, tags = guess_mapping(header_names(grid, detect_header_row(grid)))
    return MetadataSpec(path=path, sheet="metadata", header_row=header_row,
                        columns=mapping, tag_columns=tags)


def test_animals_and_tags_load_from_numbers_and_real_dates(tmp_path):
    path = write_metadata_xlsx(tmp_path / "meta.xlsx")
    table = load_animals(spec_for(path))
    assert table.header_row == 2
    assert len(table.animals) == 4 and len(table.tags) == 8
    assert TAGS["M3a"] in set(table.tags["tag"])             # ...548310 survives
    mars = table.animals.set_index("animal").loc["Mars"]
    assert mars["release"] == pd.Timestamp("2021-05-07 17:48")
    assert mars["trap"] == pd.Timestamp("2021-05-20 09:20")
    assert pd.isna(table.animals.set_index("animal").loc["Isac", "trap"])


def test_text_excel_serials_in_date_columns(tmp_path):
    """A trap_date column holding one 'NA' is read as text: '44336.0'."""
    path = write_metadata_xlsx(tmp_path / "meta.xlsx", text_serials=True)
    table = load_animals(spec_for(path))
    mars = table.animals.set_index("animal").loc["Mars"]
    assert mars["trap"] == pd.Timestamp("2021-05-20 09:20")


def test_an_explicit_header_row_overrides_detection(tmp_path):
    path = write_metadata_xlsx(tmp_path / "meta.xlsx", notes_row=False)
    table = load_animals(spec_for(path, header_row=1))
    assert table.header_row == 1 and len(table.animals) == 4


def test_a_tag_on_two_animals_is_an_error(tmp_path):
    from _rfid_synth import ANIMALS
    clash = list(ANIMALS)
    clash[1] = clash[1][:5] + (TAGS["M1a"],) + clash[1][6:]
    path = write_metadata_xlsx(tmp_path / "meta.xlsx", animals=clash)
    with pytest.raises(ValueError, match="more than one animal"):
        load_animals(spec_for(path))


def test_one_animal_may_wear_several_tags(tmp_path):
    path = write_metadata_xlsx(tmp_path / "meta.xlsx")
    table = load_animals(spec_for(path))
    per_animal = table.tags.groupby("animal").size()
    assert (per_animal == 2).all()
    assert table.tag_to_animal()[TAGS["M1b"]] == "Mars"


def test_csv_metadata(tmp_path):
    path = tmp_path / "meta.csv"
    pd.DataFrame({"name": ["A", "B"], "sex": ["male", "F"],
                  "tag": ["985.11300454831", "985.113004548266"],
                  "release": ["2021-05-07 17:00", ""]}).to_csv(path, index=False)
    spec = MetadataSpec(path=str(path), columns={"name": "name", "sex": "sex",
                                                 "release_date": "release"},
                        tag_columns=["tag"])
    table = load_animals(spec)
    assert table.animals["sex"].tolist() == ["M", "F"]
    assert table.tag_to_animal()["985.113004548310"] == "A"
    assert table.animals["release"].iloc[0] == pd.Timestamp("2021-05-07 17:00")


def test_repeated_names_are_keyed_by_trial(tmp_path):
    path = tmp_path / "meta.csv"
    pd.DataFrame({"trial": ["T1", "T2"], "name": ["Ann", "Ann"], "sex": ["F", "F"],
                  "tag": ["1.1", "1.2"]}).to_csv(path, index=False)
    spec = MetadataSpec(path=str(path), columns={"name": "name", "trial": "trial"},
                        tag_columns=["tag"])
    table = load_animals(spec)
    assert table.animals["animal"].tolist() == ["T1 Ann", "T2 Ann"]
