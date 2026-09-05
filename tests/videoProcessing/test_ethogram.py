"""Tests for the Behavior Scoring Studio ethogram and scoring model.

Pure model tests -- no Qt, no video, no files unless a test writes its own.
"""

import json

import pytest

from fnt.videoProcessing.ethogram import (
    MULTIPLE, NUMERIC, POINT, SINGLE, STATE, BehaviorDefinition, ModifierSet,
    ModifierValue, ScoringSession, Subject, format_time,
)


def session(behaviors=(), subjects=(), current=""):
    s = ScoringSession("/v/trial.mp4")
    s.ethogram = list(behaviors)
    s.subjects = list(subjects)
    s.current_subject = current or (subjects[0].name if subjects else "")
    return s


def beh(name, key="", type=STATE, **kw):
    return BehaviorDefinition(name=name, key=key, event_type=type, **kw)


# --- modifier sets ---------------------------------------------------------

def test_modifier_set_keeps_values_and_their_keys():
    ms = ModifierSet("Partner", SINGLE,
                     [ModifierValue("M1", "1"), ModifierValue("F1", "2")])
    assert ms.value_names == ["M1", "F1"]
    assert ms.key_for("F1") == "2"
    assert ms.value_for_key("1") == "M1"


def test_modifier_value_lookup_is_case_insensitive():
    ms = ModifierSet("Partner", SINGLE, [ModifierValue("M1", "A")])
    assert ms.value_for_key("a") == "M1"


def test_unknown_modifier_key_returns_nothing():
    ms = ModifierSet("Partner", SINGLE, [ModifierValue("M1", "1")])
    assert ms.value_for_key("9") is None
    assert ms.value_for_key("") is None


def test_single_select_keeps_only_one_value():
    ms = ModifierSet("Partner", SINGLE, ["M1", "F1"])
    got = ms.normalise(["M1", "F1"])
    assert got in ("M1", "F1")
    assert "," not in got


def test_multiple_select_joins_in_ethogram_order():
    """Selection order must not leak into the data."""
    ms = ModifierSet("Region", MULTIPLE, ["head", "flank", "rump"])
    assert ms.normalise(["rump", "head"]) == "head,rump"


def test_numeric_and_text_sets_pass_the_value_through():
    assert ModifierSet("Count", NUMERIC).normalise(3) == "3"
    assert ModifierSet("Note", "text").normalise("odd posture") == "odd posture"


def test_only_pick_lists_need_values():
    assert ModifierSet("a", SINGLE).needs_values
    assert ModifierSet("a", MULTIPLE).needs_values
    assert not ModifierSet("a", NUMERIC).needs_values


def test_behavior_can_own_several_independent_sets():
    b = beh("sniff", modifier_sets=[
        ModifierSet("Partner", SINGLE, ["M1", "F1"]),
        ModifierSet("Region", SINGLE, ["head", "ano-genital"])])
    assert b.has_modifiers
    assert [s.name for s in b.modifier_sets] == ["Partner", "Region"]
    assert b.set_named("Region").value_names == ["head", "ano-genital"]


# --- serialisation ---------------------------------------------------------

def test_behavior_round_trips_through_json():
    b = beh("sniff", "s", category="Social", description="nose contact",
            exclusions=["rest"],
            modifier_sets=[ModifierSet("Partner", SINGLE,
                                       [ModifierValue("M1", "1")])])
    back = BehaviorDefinition.from_dict(json.loads(json.dumps(b.to_dict())))
    assert back.name == "sniff" and back.category == "Social"
    assert back.exclusions == ["rest"]
    assert back.modifier_sets[0].key_for("M1") == "1"


def test_legacy_flat_modifier_list_is_promoted_to_a_set():
    """v1 ethograms stored modifiers as a bare list of strings."""
    b = BehaviorDefinition.from_dict(
        {"name": "sniff", "modifiers": ["self", "allo"]})
    assert len(b.modifier_sets) == 1
    assert b.modifier_sets[0].type == SINGLE
    assert b.modifier_sets[0].value_names == ["self", "allo"]


def test_subject_round_trips():
    s = Subject.from_dict(Subject("M1", "1", "resident male").to_dict())
    assert (s.name, s.key, s.description) == ("M1", "1", "resident male")


def test_v1_config_with_a_single_subject_string_still_loads(tmp_path):
    cfg = tmp_path / "ethogram_config.json"
    cfg.write_text(json.dumps({
        "version": "1.0", "subject": "M1",
        "behaviors": [{"name": "sniff", "key": "s", "event_type": "point",
                       "modifiers": ["self"]}]}))
    s = ScoringSession("/v/x.mp4")
    s.load_config(str(cfg))
    assert [x.name for x in s.subjects] == ["M1"]
    assert s.current_subject == "M1"
    assert s.ethogram[0].modifier_sets[0].value_names == ["self"]


# --- state pairing per subject ---------------------------------------------

def test_two_subjects_can_hold_the_same_state():
    """Keying states on behavior alone lost the first subject's start."""
    s = session([beh("sniff")], [Subject("M1"), Subject("F1")], "M1")
    s.start_state_event(10, 1.0, "sniff", subject="M1")
    s.start_state_event(20, 2.0, "sniff", subject="F1")
    assert s.is_state_active("sniff", "M1")
    assert s.is_state_active("sniff", "F1")

    s.stop_state_event(30, 3.0, "sniff", subject="M1")
    assert not s.is_state_active("sniff", "M1")
    assert s.is_state_active("sniff", "F1")      # untouched


def test_stop_carries_the_starts_modifiers():
    s = session([beh("sniff")], [Subject("M1")], "M1")
    s.start_state_event(10, 1.0, "sniff", {"Partner": "F1"})
    stop = s.stop_state_event(20, 2.0, "sniff")
    assert stop.modifiers == {"Partner": "F1"}


def test_toggle_starts_then_stops_for_the_same_subject():
    s = session([beh("rest")], [Subject("M1")], "M1")
    s.toggle_state(10, 1.0, "rest")
    assert s.is_state_active("rest")
    s.toggle_state(20, 2.0, "rest")
    assert not s.is_state_active("rest")


def test_events_are_attributed_to_the_focal_subject():
    s = session([beh("rear", type=POINT)], [Subject("M1"), Subject("F1")], "F1")
    ev = s.add_point_event(5, 0.5, "rear")
    assert ev.subject == "F1"


# --- exclusions ------------------------------------------------------------

def test_starting_a_behavior_stops_what_it_excludes():
    s = session([beh("rest", exclusions=["run"]), beh("run")],
                [Subject("M1")], "M1")
    s.start_state_event(10, 1.0, "run")
    _, closed = s.start_state_event(20, 2.0, "rest")
    assert closed == ["run"]
    assert not s.is_state_active("run")
    assert s.is_state_active("rest")


def test_exclusion_is_mutual_without_declaring_it_twice():
    s = session([beh("rest", exclusions=["run"]), beh("run")],
                [Subject("M1")], "M1")
    s.start_state_event(10, 1.0, "rest")
    _, closed = s.start_state_event(20, 2.0, "run")   # only rest declares it
    assert closed == ["rest"]


def test_exclusions_do_not_reach_across_subjects():
    s = session([beh("rest", exclusions=["run"]), beh("run")],
                [Subject("M1"), Subject("F1")], "M1")
    s.start_state_event(10, 1.0, "run", subject="F1")
    _, closed = s.start_state_event(20, 2.0, "rest", subject="M1")
    assert closed == []
    assert s.is_state_active("run", "F1")


def test_excluded_stop_is_recorded_as_a_real_event():
    """The auto-stop must appear in the data, not just clear a flag."""
    s = session([beh("rest", exclusions=["run"]), beh("run")],
                [Subject("M1")], "M1")
    s.start_state_event(10, 1.0, "run")
    s.start_state_event(20, 2.0, "rest")
    stops = [e for e in s.events if e.status == "STOP" and e.behavior == "run"]
    assert len(stops) == 1
    assert stops[0].time_seconds == 2.0


def test_unrelated_states_are_left_running():
    s = session([beh("rest", exclusions=["run"]), beh("run"), beh("sniff")],
                [Subject("M1")], "M1")
    s.start_state_event(10, 1.0, "sniff")
    s.start_state_event(20, 2.0, "rest")
    assert s.is_state_active("sniff")


# --- undo and editing ------------------------------------------------------

def test_undo_reopens_a_state_after_removing_its_stop():
    s = session([beh("rest")], [Subject("M1")], "M1")
    s.start_state_event(10, 1.0, "rest")
    s.stop_state_event(20, 2.0, "rest")
    s.undo_last()
    assert s.is_state_active("rest")


def test_undo_of_a_start_closes_it_again():
    s = session([beh("rest")], [Subject("M1")], "M1")
    s.start_state_event(10, 1.0, "rest")
    s.undo_last()
    assert not s.is_state_active("rest")


def test_deleting_a_start_leaves_no_dangling_state():
    s = session([beh("rest")], [Subject("M1")], "M1")
    start, _ = s.start_state_event(10, 1.0, "rest")
    s.remove_event(start)
    assert not s.is_state_active("rest")


def test_deleting_a_stop_reopens_the_state():
    s = session([beh("rest")], [Subject("M1")], "M1")
    s.start_state_event(10, 1.0, "rest")
    stop = s.stop_state_event(20, 2.0, "rest")
    s.remove_event(stop)
    assert s.is_state_active("rest")


def test_states_are_rebuilt_in_time_order_not_insertion_order():
    """An event inserted late must still pair correctly."""
    s = session([beh("rest")], [Subject("M1")], "M1")
    s.stop_state_event(20, 2.0, "rest")          # recorded first, later in time
    s.start_state_event(10, 1.0, "rest")
    s.rebuild_active_states()
    assert not s.is_state_active("rest")


# --- validation ------------------------------------------------------------

def test_duplicate_key_between_two_behaviors_is_reported():
    s = session([beh("rest", "r"), beh("run", "r")])
    assert "r" in s.duplicate_keys()


def test_key_clash_between_a_behavior_and_a_subject_is_reported():
    """They share one keyboard, so this clash matters just as much."""
    s = session([beh("rest", "m")], [Subject("M1", "m")])
    clash = s.duplicate_keys()
    assert "m" in clash and len(clash["m"]) == 2


def test_distinct_keys_are_clean():
    s = session([beh("rest", "r"), beh("run", "u")], [Subject("M1", "1")])
    assert s.duplicate_keys() == {}


def test_unpaired_state_is_reported():
    s = session([beh("rest")], [Subject("M1")], "M1")
    s.start_state_event(10, 1.0, "rest")
    assert [e.behavior for e in s.unpaired_states()] == ["rest"]


def test_closed_states_are_not_reported_as_unpaired():
    s = session([beh("rest")], [Subject("M1")], "M1")
    s.start_state_event(10, 1.0, "rest")
    s.stop_state_event(20, 2.0, "rest")
    assert s.unpaired_states() == []


def test_exclusion_naming_a_deleted_behavior_is_reported():
    s = session([beh("rest", exclusions=["run"])])
    assert s.undefined_exclusions() == {"rest": ["run"]}


# --- time budget -----------------------------------------------------------

def test_time_budget_sums_state_durations_per_subject():
    s = session([beh("rest", category="Inactive")],
                [Subject("M1"), Subject("F1")], "M1")
    s.start_state_event(0, 0.0, "rest", subject="M1")
    s.stop_state_event(0, 10.0, "rest", subject="M1")
    s.start_state_event(0, 20.0, "rest", subject="M1")
    s.stop_state_event(0, 26.0, "rest", subject="M1")
    s.start_state_event(0, 0.0, "rest", subject="F1")
    s.stop_state_event(0, 4.0, "rest", subject="F1")

    budget = {(r["subject"], r["behavior"]): r for r in s.time_budget()}
    m1 = budget[("M1", "rest")]
    assert m1["n"] == 2
    assert m1["total_seconds"] == pytest.approx(16.0)
    assert m1["mean_seconds"] == pytest.approx(8.0)
    assert m1["sd_seconds"] == pytest.approx(2.828, abs=0.01)
    assert m1["category"] == "Inactive"
    assert budget[("F1", "rest")]["total_seconds"] == pytest.approx(4.0)


def test_point_events_are_counted_but_add_no_duration():
    s = session([beh("rear", type=POINT)], [Subject("M1")], "M1")
    s.add_point_event(0, 1.0, "rear")
    s.add_point_event(0, 2.0, "rear")
    row = s.time_budget()[0]
    assert row["n"] == 2
    assert row["total_seconds"] == 0.0


def test_unclosed_state_adds_no_duration():
    """A forgotten STOP must not silently inflate a total."""
    s = session([beh("rest")], [Subject("M1")], "M1")
    s.start_state_event(0, 0.0, "rest")
    s.stop_state_event(0, 5.0, "rest")
    s.start_state_event(0, 10.0, "rest")          # never closed
    row = s.time_budget()[0]
    assert row["n"] == 2
    assert row["total_seconds"] == pytest.approx(5.0)


def test_single_occurrence_has_zero_sd():
    s = session([beh("rest")], [Subject("M1")], "M1")
    s.start_state_event(0, 0.0, "rest")
    s.stop_state_event(0, 5.0, "rest")
    assert s.time_budget()[0]["sd_seconds"] == 0.0


# --- export ----------------------------------------------------------------

def test_export_gives_each_modifier_set_its_own_column():
    s = session([beh("sniff", modifier_sets=[
        ModifierSet("Partner", SINGLE, ["M1"]),
        ModifierSet("Region", SINGLE, ["head"])])], [Subject("M1")], "M1")
    s.add_point_event(1, 0.1, "sniff", {"Partner": "M1", "Region": "head"})
    row = s.event_rows()[0]
    assert row["modifier_Partner"] == "M1"
    assert row["modifier_Region"] == "head"


def test_export_keeps_a_flattened_modifier_column():
    s = session([beh("sniff", modifier_sets=[
        ModifierSet("Partner", SINGLE, ["M1"]),
        ModifierSet("Region", SINGLE, ["head"])])], [Subject("M1")], "M1")
    s.add_point_event(1, 0.1, "sniff", {"Partner": "M1", "Region": "head"})
    assert s.event_rows()[0]["modifier"] == "M1|head"


def test_export_carries_subject_and_category():
    s = session([beh("rest", category="Inactive")], [Subject("M1")], "M1")
    s.start_state_event(1, 0.1, "rest")
    row = s.event_rows()[0]
    assert row["subject"] == "M1" and row["category"] == "Inactive"


def test_rows_round_trip_back_into_events():
    s = session([beh("sniff", modifier_sets=[
        ModifierSet("Partner", SINGLE, ["F1"])])], [Subject("M1")], "M1")
    s.add_point_event(3, 0.3, "sniff", {"Partner": "F1"})
    rows = s.event_rows()

    s2 = session([beh("sniff", modifier_sets=[
        ModifierSet("Partner", SINGLE, ["F1"])])], [Subject("M1")], "M1")
    s2.load_events(rows)
    assert len(s2.events) == 1
    assert s2.events[0].modifiers == {"Partner": "F1"}
    assert s2.events[0].subject == "M1"


def test_reading_a_v1_row_keeps_its_single_modifier():
    s = session([beh("sniff")], [Subject("M1")], "M1")
    s.load_events([{"frame": 1, "time_seconds": 0.1, "subject": "M1",
                    "behavior": "sniff", "modifier": "self",
                    "type": "point", "status": "POINT"}])
    assert s.events[0].modifiers == {"Modifier": "self"}


def test_loading_events_restores_open_states():
    s = session([beh("rest")], [Subject("M1")], "M1")
    s.load_events([
        {"frame": 1, "time_seconds": 1.0, "subject": "M1", "behavior": "rest",
         "type": "state", "status": "START"}])
    assert s.is_state_active("rest", "M1")


# --- lookups ---------------------------------------------------------------

def test_key_lookup_finds_behaviors_and_subjects():
    s = session([beh("rest", "r")], [Subject("M1", "1")])
    assert s.behavior_for_key("R").name == "rest"
    assert s.subject_for_key("1").name == "M1"
    assert s.behavior_for_key("z") is None


def test_categories_are_listed_once_and_sorted():
    s = session([beh("a", category="Social"), beh("b", category="Inactive"),
                 beh("c", category="Social"), beh("d")])
    assert s.categories == ["Inactive", "Social"]


def test_modifier_set_names_are_collected_across_the_ethogram():
    s = session([beh("a", modifier_sets=[ModifierSet("Partner", SINGLE, ["x"])]),
                 beh("b", modifier_sets=[ModifierSet("Region", SINGLE, ["y"]),
                                         ModifierSet("Partner", SINGLE, ["z"])])])
    assert s.modifier_set_names == ["Partner", "Region"]


@pytest.mark.parametrize("seconds,expected", [
    (0, "00:00:00.000"), (61.5, "00:01:01.500"), (3661.25, "01:01:01.250"),
])
def test_time_formatting(seconds, expected):
    assert format_time(seconds) == expected
