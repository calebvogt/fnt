"""Optional animal Name / Code, and how every tag label is resolved.

Animals now carry an optional Name (e.g. Hera) and Code (e.g. HER) alongside
their SexID (e.g. F9905). The SexID stays the key every analysis joins on; the
new fields are display labels selectable from Show Tag ID, and extra
smoothed-CSV columns.

Two things are worth guarding:

* the label resolver the preview, the rendered video and the tag list all
  share -- an animal without a Name must fall back to its SexID, not go blank,
  and a tag with no identity at all to its HexID;
* the save-time completeness check. It works per ANIMAL, not per tag: a
  replacement tag sharing its animal's SexID must not be reported as "missing"
  a name that was typed on the original tag. And a roster that uses no names
  at all is complete, not incomplete.

Runs under pytest, or directly (``python test_identities.py``).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fnt.uwb.identities import (  # noqa: E402
    ID_DISPLAY_TYPES, SEX_ID, describe_problems, identity_field_problems,
    normalize_id_type, tag_label)

ROSTER = {
    1: {'sex': 'F', 'identity': '9905', 'name': 'Hera', 'code': 'HER'},
    2: {'sex': 'M', 'identity': '9809', 'name': 'Zeus', 'code': 'ZEU'},
    3: {'sex': 'M', 'identity': '9809'},       # replacement tag for Zeus
}


def test_selector_order_and_legacy_value():
    assert ID_DISPLAY_TYPES == ("SexID", "Name", "Code", "HexID", "ShortID")
    # Configs written before the rename stored the SexID option as this.
    assert normalize_id_type("Display ID") == SEX_ID
    assert normalize_id_type("nonsense") == SEX_ID


def test_labels_and_fallbacks():
    assert tag_label(1, ROSTER[1], "SexID") == "F9905"
    assert tag_label(1, ROSTER[1], "Name") == "Hera"
    assert tag_label(1, ROSTER[1], "Code") == "HER"
    # No name/code on this tag -> SexID, never blank.
    assert tag_label(3, ROSTER[3], "Name") == "M9809"
    # No identity at all -> the hex address.
    assert tag_label(42, None, "Name") == "2A"
    assert tag_label(42, {}, "SexID") == "2A"
    assert tag_label(42, None, "HexID") == "2A"
    assert tag_label(42, None, "ShortID") == "42"


def test_replacement_tag_is_not_missing_its_animals_name():
    assert identity_field_problems(ROSTER) == {}


def test_unused_fields_are_not_reported():
    plain = {1: {'sex': 'F', 'identity': '1'}, 2: {'sex': 'M', 'identity': '2'}}
    assert identity_field_problems(plain) == {}


def test_partial_roster_is_reported_per_field():
    roster = dict(ROSTER)
    roster[4] = {'sex': 'F', 'identity': '9826', 'name': 'Athena'}   # no code
    roster[5] = {'sex': 'M', 'identity': '9901'}                     # neither
    report = identity_field_problems(roster)
    assert report['name']['missing'] == ['M9901']
    assert report['code']['missing'] == ['F9826', 'M9901']
    text = "\n".join(describe_problems(report))
    assert "No Name for: M9901" in text
    assert "No Code for: F9826, M9901" in text


def test_conflicts_and_case_insensitive_duplicates():
    roster = {
        1: {'sex': 'F', 'identity': '1', 'name': 'Hera', 'code': 'HER'},
        2: {'sex': 'F', 'identity': '1', 'name': 'Rhea', 'code': 'HER'},  # same animal
        3: {'sex': 'M', 'identity': '2', 'name': 'Zeus', 'code': 'her'},  # clash
    }
    report = identity_field_problems(roster)
    assert report['name']['conflicts'] == [('F1', ['Hera', 'Rhea'])]
    assert report['code']['duplicates'] == [('HER', ['F1', 'M2'])]
    assert report['name']['missing'] == [] and report['code']['missing'] == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all passed")
