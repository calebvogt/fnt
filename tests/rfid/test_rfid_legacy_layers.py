"""The optional layers carried over from the earlier module, on zone_id keys.

GBI, co-presence edgelist, displacement and Hinde indices were written when
zones were integers; they now run on ``A1`` / ``B3`` strings. They stay off by
default (they are interpretation, not preprocessing), and these tests are what
keeps them honest. The hand-off test is the one worth reading: when one
animal's bout ends exactly as another's begins, BOTH count as present in the
zero-length interval between.
"""
import pandas as pd
import pytest

from fnt.rfid.core.displacement import detect_displacements
from fnt.rfid.core.edgelist import co_presence_bouts, edgelist
from fnt.rfid.core.gbi_generator import create_gbi, melt_gbi
from fnt.rfid.core.hinde_index import hinde_index, hinde_summary
from fnt.rfid.core.summaries import co_occupancy

SEX = {"A": "M", "B": "F", "C": "M"}


def bout(name, zone, start, stop, day=1):
    return {"trial": "LID", "name": name, "sex": SEX[name], "zone": zone,
            "noon_day": day, "field_time": pd.Timestamp(start),
            "field_time_stop": pd.Timestamp(stop),
            "duration_s": (pd.Timestamp(stop) - pd.Timestamp(start)).total_seconds()}


def test_overlapping_bouts_split_into_three_intervals_on_a_string_zone():
    gbi = create_gbi(pd.DataFrame([bout("A", "B3", "2021-05-07 20:00:00", "2021-05-07 20:00:30"),
                                   bout("B", "B3", "2021-05-07 20:00:10", "2021-05-07 20:00:40")]),
                     ["A", "B"], SEX)
    assert gbi["zone"].tolist() == ["B3"] * 3
    assert gbi["mf_sum"].tolist() == [1, 2, 1]
    assert gbi["duration_s"].tolist() == [10, 20, 10]


def test_a_handoff_records_both_animals():
    gbi = create_gbi(pd.DataFrame([bout("B", "A1", "2021-05-07 20:00:00", "2021-05-07 20:00:30"),
                                   bout("A", "A1", "2021-05-07 20:00:30", "2021-05-07 20:01:00")]),
                     ["A", "B"], SEX)
    handoff = gbi[gbi["field_time_start"] == gbi["field_time_stop"]]
    assert len(handoff) == 1 and handoff["mf_sum"].iloc[0] == 2


def test_same_zone_number_in_two_paddocks_does_not_mix():
    gbi = create_gbi(pd.DataFrame([bout("A", "A1", "2021-05-07 20:00:00", "2021-05-07 20:00:30"),
                                   bout("B", "B1", "2021-05-07 20:00:00", "2021-05-07 20:00:30")]),
                     ["A", "B"], SEX)
    assert set(gbi["zone"]) == {"A1", "B1"} and (gbi["mf_sum"] == 1).all()
    assert len(melt_gbi(gbi)) == 2


def test_edgelist_agrees_with_the_new_co_occupancy():
    """Two independent methods - GBI intervals vs the per-zone sweep - must
    give the same seconds per pair (on LID they agree on all 752 pair-days)."""
    legacy = pd.DataFrame([
        bout("A", "A1", "2021-05-07 20:00:00", "2021-05-07 20:10:00"),
        bout("B", "A1", "2021-05-07 20:05:00", "2021-05-07 20:20:00"),
        bout("C", "A1", "2021-05-07 20:07:00", "2021-05-07 20:08:00"),
        bout("A", "A1", "2021-05-07 20:15:00", "2021-05-07 20:16:00"),
        bout("C", "B2", "2021-05-07 20:00:00", "2021-05-07 20:30:00")])
    gbi = create_gbi(legacy, ["A", "B", "C"], SEX)
    el = edgelist(co_presence_bouts(gbi, ["A", "B", "C"]))
    new = legacy.rename(columns={"name": "animal", "zone": "zone_id", "noon_day": "day",
                                 "field_time": "start", "field_time_stop": "stop"})
    co = co_occupancy(new)
    a = el.set_index(["id1", "id2"])["total_duration_s"].to_dict()
    b = co.set_index(["animal_a", "animal_b"])["seconds"].to_dict()
    assert a == b == {("A", "B"): 360.0, ("A", "C"): 60.0, ("B", "C"): 60.0}


def test_displacement_on_zone_ids():
    """A holds B3, C arrives, A leaves: C displaced A (1 -> 2 -> 1 males)."""
    gbi = create_gbi(pd.DataFrame([
        bout("A", "B3", "2021-05-07 20:00:00", "2021-05-07 20:10:00"),
        bout("C", "B3", "2021-05-07 20:05:00", "2021-05-07 20:20:00")]), ["A", "C"], SEX)
    events = detect_displacements(gbi, ["A", "C"])
    assert len(events) == 1
    e = events.iloc[0]
    assert e["zone"] == "B3" and e["winner"] == "C" and e["loser"] == "A"


def test_hinde_on_zone_ids():
    gbi = create_gbi(pd.DataFrame([
        bout("A", "A2", "2021-05-07 20:00:00", "2021-05-07 20:10:00"),
        bout("B", "A2", "2021-05-07 20:05:00", "2021-05-07 20:20:00")]), ["A", "B"], SEX)
    events = hinde_index(gbi, "broad", ["A", "B"])
    assert len(events) == 1
    assert events.iloc[0]["zone"] == "A2"
    assert events.iloc[0]["made_contact"] == "B" and events.iloc[0]["broke_contact"] == "A"
    s = hinde_summary(events).set_index("name")
    # B did all the approaching (+1), A all the leaving (-1)
    assert s.loc["B", "hinde_index"] == 1 and s.loc["A", "hinde_index"] == -1


def test_social_networks_run_when_igraph_is_installed():
    pytest.importorskip("igraph")
    from fnt.rfid.core.social_network import social_networks
    gbi = create_gbi(pd.DataFrame([
        bout("A", "A1", "2021-05-07 20:00:00", "2021-05-07 20:10:00"),
        bout("B", "A1", "2021-05-07 20:05:00", "2021-05-07 20:20:00"),
        bout("C", "A1", "2021-05-07 20:06:00", "2021-05-07 20:07:00")]), ["A", "B", "C"], SEX)
    meta = pd.DataFrame({"name": ["A", "B", "C"], "sex": ["M", "F", "M"]})
    nets = social_networks(gbi, meta, ["A", "B", "C"])
    assert len(nets["node_stats"]) == 3
