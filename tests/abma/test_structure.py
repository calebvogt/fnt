"""The hierarchy-or-territory readouts, on visit tables whose answer is known."""
import numpy as np
import pandas as pd
import pytest

from fnt.abma.core import structure as st


def _bouts(rows):
    return pd.DataFrame(rows, columns=["animal", "zone", "start", "stop", "day"])


def test_territories_one_owner_per_zone():
    # three animals, three zones, each alone in its own zone on two days
    rows = [(a, z, d * 1000 + 10, d * 1000 + 500, d + 1)
            for d in range(2) for a, z in (("A", 1), ("B", 2), ("C", 3))]
    s = st.structure_summary(_bouts(rows), animals=["A", "B", "C"], zones=[1, 2, 3])
    assert s["owner_share"] == pytest.approx(1.0)
    assert s["poles"] == pytest.approx(3.0)
    assert s["landless"] == pytest.approx(0.0)
    assert s["owner_stays"] == pytest.approx(1.0)
    assert s["shared"] == pytest.approx(0.0)
    assert s["displacements"] == 0


def test_one_animal_owning_everything_is_one_pole():
    rows = [("A", z, 10 + 1000 * z, 900 + 1000 * z, 1) for z in (1, 2, 3)]
    rows += [("B", z, 950 + 1000 * z, 990 + 1000 * z, 1) for z in (1, 2, 3)]
    s = st.structure_summary(_bouts(rows), animals=["A", "B", "C"], zones=[1, 2, 3])
    assert s["poles"] == pytest.approx(1.0)
    assert s["top_owner_zones"] == pytest.approx(1.0)
    assert s["landless"] == pytest.approx(2.0 / 3.0)      # B and C own nothing; C never came
    assert 0.9 < s["owner_share"] < 1.0


def test_displacement_is_alone_joined_alone():
    # A is there, B joins, A leaves: B displaced A, and A was the resident
    b = _bouts([("A", 1, 0, 100, 1), ("B", 1, 60, 300, 1)])
    d = st.displacements(b)
    assert len(d) == 1
    r = d.iloc[0]
    assert (r.winner, r.loser, r.resident) == ("B", "A", "A")
    assert r.together_s == pytest.approx(40.0)
    assert r.after_s == pytest.approx(200.0)
    # the joiner leaving again is the resident winning
    d2 = st.displacements(_bouts([("A", 1, 0, 300, 1), ("B", 1, 60, 100, 1)]))
    assert (d2.iloc[0].winner, d2.iloc[0].loser, d2.iloc[0].resident) == ("A", "B", "A")


def test_a_hand_over_is_not_a_meeting():
    # B arrives the instant A leaves: they were never there together
    assert st.displacements(_bouts([("A", 1, 0, 100, 1), ("B", 1, 100, 200, 1)])).empty
    # different zones never meet
    assert st.displacements(_bouts([("A", 1, 0, 100, 1), ("B", 2, 50, 200, 1)])).empty


def test_three_way_meetings_are_not_scored():
    b = _bouts([("A", 1, 0, 100, 1), ("B", 1, 20, 300, 1), ("C", 1, 40, 300, 1)])
    assert st.displacements(b).empty


def test_linear_and_cyclic_orders():
    ids = list("ABCD")
    lin = pd.DataFrame(0.0, index=ids, columns=ids)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            lin.loc[a, b] = 5
    assert st.landau_h(lin) == pytest.approx(1.0)
    assert st.triangle_transitivity(lin) == pytest.approx(1.0)
    assert st.dci(lin) == pytest.approx(1.0)
    ds = st.davids_score(lin)
    assert list(ds.sort_values(ascending=False).index) == ids
    cyc = pd.DataFrame(0.0, index=list("ABC"), columns=list("ABC"))
    cyc.loc["A", "B"] = cyc.loc["B", "C"] = cyc.loc["C", "A"] = 3
    assert st.landau_h(cyc) == pytest.approx(0.0)
    assert st.triangle_transitivity(cyc) == pytest.approx(-3.0)
    even = pd.DataFrame([[0, 2], [2, 0]], index=["A", "B"], columns=["A", "B"], dtype=float)
    assert st.dci(even) == pytest.approx(0.0)


def _meet(winner, loser, zone, t, day):
    """The loser is there first, the winner joins, the loser leaves."""
    return [(loser, zone, t, t + 60, day), (winner, zone, t + 30, t + 200, day)]


def test_hierarchy_rank_transfers_between_zones():
    # A beats B at both zones: a record made at one zone predicts the other
    rows = []
    for k in range(4):
        rows += _meet("A", "B", 1, 1000 * k, 1) + _meet("A", "B", 2, 1000 * k + 500, 1)
    s = st.structure_summary(_bouts(rows), animals=["A", "B"], zones=[1, 2])
    assert s["rank_transfer"] == pytest.approx(1.0)
    assert s["dci"] == pytest.approx(1.0)


def test_territory_owner_wins_and_rank_does_not_transfer():
    # A owns zone 1 and wins there, B owns zone 2 and wins there, on both days
    rows = []
    for day in (1, 2):
        t0 = 100000 * (day - 1)
        rows += [("A", 1, t0 + 50000, t0 + 90000, day), ("B", 2, t0 + 50000, t0 + 90000, day)]
        for k in range(4):
            rows += _meet("A", "B", 1, t0 + 1000 * k, day) + _meet("B", "A", 2, t0 + 1000 * k + 500, day)
    s = st.structure_summary(_bouts(rows), animals=["A", "B"], zones=[1, 2])
    assert s["poles"] == pytest.approx(2.0)
    assert s["rank_transfer"] == pytest.approx(0.0)       # the winner here is the loser elsewhere
    assert s["owner_win"] == pytest.approx(1.0)           # scored on day 2, with day 1's owners
    assert s["owner_win_n"] == 8
    assert s["dci"] == pytest.approx(0.0)                 # the dyad is even once place is ignored


def test_gini():
    assert st.gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert st.gini([0, 0, 0, 10]) == pytest.approx(0.75)
    assert np.isnan(st.gini([0, 0]))


def test_bouts_from_positions():
    class Z:
        x, y, w, d, label = 1.0, 1.0, 1.0, 1.0, "Z1"

    t = np.arange(0, 100, 10.0)
    # inside for samples 2-4, outside, inside again for sample 8
    x = np.array([5, 5, 1, 1, 1, 5, 5, 5, 1, 5], float)
    traj = pd.DataFrame({"sexid": "M1", "sex": "M", "time_sec": t + 1000.0, "smoothed_x": x,
                         "smoothed_y": 1.0})
    b = st.bouts_from_positions(traj, [Z()])
    assert len(b) == 2
    assert list(b["start"]) == [20.0, 80.0]
    assert list(b["stop"]) == [50.0, 90.0]            # last sample inside plus one interval
    assert list(b["zone"]) == ["Z1", "Z1"]
    assert list(b["day"]) == [1, 1]


def test_structure_table_and_cli(tmp_path):
    rows = []
    for trial in ("T1", "T2"):
        for a, z in (("A", 1), ("B", 2)):
            rows.append((trial, "M", f"{trial} {a}", z, 10.0, 500.0, 1))
        rows.append((trial, "M", f"{trial} C", None, None, None, None))      # on the roster, never at a zone
        rows.append((trial, "F", f"{trial} X", 3, 10.0, 500.0, 1))           # a zone only females used
    v = pd.DataFrame(rows, columns=["trial", "sex", "animal", "zone", "start", "stop", "day"])
    t = st.structure_table(v)
    m = t[t.sex == "M"].set_index("trial")
    assert list(m.index) == ["T1", "T2"]
    assert (m.n_animals == 3).all() and (m.n_zones == 3).all()
    assert m.poles.tolist() == pytest.approx([2.0, 2.0])
    assert m.landless.tolist() == pytest.approx([1 / 3, 1 / 3])
    # the command line writes the same table
    from fnt.abma.run_headless import main
    src = tmp_path / "visits.csv"
    v.to_csv(src, index=False)
    assert main(["--structure", str(src), "--structure-days", "1-1"]) == 0
    got = pd.read_csv(tmp_path / "visits_structure.csv")
    assert len(got) == 4 and set(got.sex) == {"M", "F"}
