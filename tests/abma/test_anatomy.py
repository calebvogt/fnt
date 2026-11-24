"""A brain wired from an anatomy spec: the wiring is the spec's, edits change
exactly what they name, and it drives an animal through the ordinary engine."""
import json
import os

import numpy as np
import pytest

from fnt.abma.core import anatomy, presets
from fnt.abma.core.config import ExperimentConfig
from fnt.abma.core.neural import NeuralPolicy, make_policy
from fnt.abma.core.senses import SENSORS
from fnt.abma.run_headless import main


@pytest.fixture(scope="module")
def spec():
    return anatomy.load_spec("c57_male")


def _oft(seed=0, **brain):
    cfg = presets.open_field_test()
    cfg.seed = seed
    cfg.brain.enabled, cfg.brain.anatomy = True, "c57_male"
    for k, v in brain.items():
        setattr(cfg.brain, k, v)
    return cfg


def test_the_shipped_spec_is_listed_and_valid(spec):
    assert "c57_male" in [n for n, _ in anatomy.list_anatomies()]
    assert "untrained" in anatomy.describe(spec)
    for m in anatomy.MUSCLES:
        assert m in [r["id"] for r in spec["regions"]]


def test_a_broken_spec_says_what_is_wrong(spec):
    bad = json.loads(json.dumps(spec))
    bad["projections"].append(["mPFC", "NOWHERE", "+", 1])
    with pytest.raises(ValueError, match="unknown area"):
        anatomy.check_spec(bad)
    bad = json.loads(json.dumps(spec))
    bad["regions"] = [r for r in bad["regions"] if r["id"] != "EUS"]
    with pytest.raises(ValueError):
        anatomy.check_spec(bad)


def test_units_are_wired_only_where_the_spec_has_a_projection(spec):
    b = anatomy.AnatomicalBrain(spec, seed=3)
    wired = {(p[0], p[1]): p[2] for p in spec["projections"]}
    for a in b.ids:
        for c in b.ids:
            blk = b.W[b.sl[c], b.sl[a]]
            if a == c:
                assert np.all(np.diag(blk) == 0.0)
            elif (a, c) not in wired:
                assert not blk.any(), f"{a}>{c} is wired but not in the spec"
            elif wired[(a, c)] == "+":
                assert blk.min() >= 0.0 and blk.max() > 0.0
            elif wired[(a, c)] == "-":
                assert blk.max() <= 0.0 and blk.min() < 0.0
            else:                      # mixed cell types: each source cell has one sign
                assert all((col >= 0).all() or (col <= 0).all() for col in blk.T)


def test_senses_enter_only_at_their_organ(spec):
    b = anatomy.AnatomicalBrain(spec)
    col = {s: i for i, s in enumerate(SENSORS)}
    rows = np.flatnonzero(np.abs(b.Win[:, col["wall_ahead"]]) > 0)
    assert set(rows) <= set(range(b.sl["WHK"].start, b.sl["WHK"].stop))
    # the hand-set dispositions are not wired to this brain at all
    for name in ("aggression", "boldness", "sociability", "exploration"):
        assert not b.Win[:, col[name]].any()
    # the vomeronasal organ gets the contact channels and nothing else
    vno = np.flatnonzero(np.abs(b.Win[b.sl["VNO"]]).sum(axis=0) > 0)
    assert vno.size == 13 and all(SENSORS[i].startswith("vno_") for i in vno)


def test_an_edit_changes_what_it_names_and_nothing_else(spec):
    base = anatomy.AnatomicalBrain(spec, seed=1)
    ed = anatomy.AnatomicalBrain(spec, seed=1, strengths={"mPFC>DRN": 2.0, "SC>TURN": 0.0},
                                 receptors={"OT@LS": 0})
    blk = (base.sl["DRN"], base.sl["mPFC"])
    np.testing.assert_allclose(ed.W[blk], 2.0 * base.W[blk])
    assert not ed.W[ed.sl["TURN"], ed.sl["SC"]].any()
    same = np.ones_like(base.W, bool)
    same[blk] = False
    same[base.sl["TURN"], base.sl["SC"]] = False
    np.testing.assert_array_equal(ed.W[same], base.W[same])
    ot = [m for m in ed.mods if m[0] == "OT"][0][2]
    assert not ot[ed.sl["LS"]].any() and ot[ed.sl["CA2"]].all()
    with pytest.raises(ValueError, match="no such projection"):
        anatomy.AnatomicalBrain(spec, strengths={"mPFC>MOE": 2.0})
    with pytest.raises(ValueError, match="no such modulator"):
        anatomy.AnatomicalBrain(spec, receptors={"XX@LS": 1})


def test_parsing_the_command_line_edits():
    assert anatomy.parse_strength("mPFC>DRN=1.5") == ("mPFC>DRN", 1.5)
    assert anatomy.parse_receptor("OT@LS=3") == ("OT@LS", 3)
    for bad in ("mPFC=1", "mPFC>DRN=9"):
        with pytest.raises(ValueError):
            anatomy.parse_strength(bad)
    with pytest.raises(ValueError):
        anatomy.parse_receptor("OT@LS=7")


def test_a_wall_ahead_reaches_the_whiskers_first(spec):
    b = anatomy.AnatomicalBrain(spec)
    rng = np.random.default_rng(0)
    col = {s: i for i, s in enumerate(SENSORS)}

    def settle(wall):
        obs = np.zeros((1, len(SENSORS)))
        obs[0, col["wall_ahead"]] = wall
        r = b.initial_state(1)
        for _ in range(40):
            r = b.step(r, obs, rng)
        return r

    clear, touching = settle(0.0), settle(1.0)
    # one whisker cell answers mainly to that channel; the area's own inhibition quiets its neighbours
    assert touching[0, b.sl["WHK"]].max() > clear[0, b.sl["WHK"]].max() + 0.3
    assert 0.0 <= touching.min() and touching.max() <= 1.0


def test_the_config_chooses_it_and_round_trips():
    cfg = _oft(strengths={"mPFC>DRN": 1.5}, receptors={"OT@LS": 3}, anatomy_seed=4)
    pol = make_policy(cfg)
    assert isinstance(pol, anatomy.AnatomicalPolicy) and not isinstance(pol, NeuralPolicy)
    back = ExperimentConfig.from_dict(cfg.to_dict())
    assert back.brain.anatomy == "c57_male" and back.brain.anatomy_seed == 4
    assert back.brain.strengths == {"mPFC>DRN": 1.5} and back.brain.receptors == {"OT@LS": 3}
    # an ordinary learned-brain config is untouched
    plain = presets.open_field_test()
    assert plain.brain.anatomy == "" and not plain.brain.enabled


def test_it_drives_an_animal_and_repeats_exactly():
    a = anatomy.record(_oft(seed=2), 30.0)
    b = anatomy.record(_oft(seed=2), 30.0)
    c = anatomy.record(_oft(seed=2, anatomy_seed=1), 30.0)
    np.testing.assert_array_equal(a["rates"], b["rates"])
    np.testing.assert_array_equal(a["xy"], b["xy"])
    assert not np.array_equal(a["rates"], c["rates"])
    w, h = a["meta"]["arena"]
    assert np.isfinite(a["xy"]).all()
    assert (a["xy"][..., 0] >= 0).all() and (a["xy"][..., 0] <= w).all()
    assert (a["xy"][..., 1] >= 0).all() and (a["xy"][..., 1] <= h).all()
    assert a["rates"].shape[2] == len(a["meta"]["units"]) and a["meta"]["trained"] is False
    assert anatomy.movement_summary(a)["path_m"] > 0.0


def test_the_live_view_and_the_run_record_get_its_activity():
    from fnt.abma.core.simulation import Simulation

    cfg = _oft()
    sim = Simulation(cfg, 0)
    assert isinstance(sim.policy, anatomy.AnatomicalPolicy)
    for k in range(4):
        sim.step(k * cfg.dt, cfg.dt)
    extra = sim.policy.frame_extra(sim.n)
    assert extra["brain_rnn"].shape == (sim.n, sim.policy.brain.n)
    assert extra["brain_obs"].shape == (sim.n, len(SENSORS))
    assert extra["brain_sensory"].shape[1] == len(sim.policy.brain.sense_units)


def test_a_coarse_engine_step_runs_several_network_ticks():
    from fnt.abma.core.simulation import Simulation

    cfg = _oft()
    cfg.dt = 2.0
    sim = Simulation(cfg, 0)
    sim.step(0.0, cfg.dt)
    first = sim.policy.r.copy()
    sim.step(cfg.dt, cfg.dt)
    assert not np.allclose(first, sim.policy.r)        # eight ticks later
    assert abs(sim.policy._clock) < 1e-6


def test_command_line(tmp_path, capsys):
    assert main(["--list-anatomies"]) == 0
    assert "c57_male" in capsys.readouterr().out
    out = str(tmp_path)
    assert main(["--anatomy", "c57_male", "--strength", "SC>TURN=0", "--receptor", "OT@LS=3",
                 "--brain-recording", "0.25", "--out", out]) == 0
    text = capsys.readouterr().out
    assert "untrained" in text and "SC>TURN x0" in text and "most active areas" in text
    run = os.path.join(out, "open_field_test_anatomy")
    rec = anatomy.load_recording(os.path.join(run, "brain_recording.npz"))
    assert rec["meta"]["strengths"] == {"SC>TURN": 0.0} and rec["rates"].ndim == 3
    with open(os.path.join(run, "brain_areas.csv"), encoding="utf-8") as f:
        assert f.readline().startswith("area,name,class,units,mean_pct")
    for bad in (["--anatomy", "nope"], ["--anatomy", "c57_male", "--strength", "mPFC>MOE=2"],
                ["--strength", "SC>TURN=0", "--preset", "open field"]):
        with pytest.raises(SystemExit):
            main(bad + ["--dry-run"])


def test_the_spec_ships_with_the_package():
    import tomllib

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(here, "pyproject.toml"), "rb") as f:
        data = tomllib.load(f)["tool"]["setuptools"]["package-data"]["fnt"]
    assert "abma/brains/anatomy/*.json" in data


def _scent_sim(n=1):
    from fnt.abma.core.simulation import Simulation
    cfg = _oft()
    cfg.scent.enabled = True
    cfg.groups[0].count = n
    return Simulation(cfg, 0)


def test_a_urine_barcode_per_animal():
    from fnt.abma.core.biology import INBRED_BARCODE, MUP_LOCI, mup_barcode
    a, b = mup_barcode("WT", 1, 7), mup_barcode("WT", 2, 7)
    assert a.shape == (MUP_LOCI,) and a.any() and not np.array_equal(a, b)
    np.testing.assert_array_equal(a, mup_barcode("WT", 1, 7))            # stable for an animal
    np.testing.assert_array_equal(mup_barcode("INBRED", 1, 7), mup_barcode("INBRED", 99, 3))
    assert tuple(mup_barcode("inbred", 5, 0)) == INBRED_BARCODE
    assert not mup_barcode("KO", 1, 7).any() and mup_barcode("HET", 1, 7).max() == 0.5


def test_urine_leaves_when_pressure_beats_the_sphincter():
    sim = _scent_sim()
    cost = np.full(sim.n, 5.0)
    sim.bladder[:] = 80.0
    sim._void_by_muscle(0.25, np.array([0.1]), np.array([0.9]), cost)      # held shut
    assert sim.bladder[0] == 80.0 and not sim.last_voids and len(sim.urine_spots["t"]) == 0
    sim._void_by_muscle(0.25, np.array([0.6]), np.array([0.1]), cost)      # wall pushes, sphincter lets go
    assert sim.bladder[0] < 80.0 and sim.marks_made[0] == 1 and len(sim.urine_spots["t"]) == 1
    for _ in range(40):                                                    # once started it runs on
        sim._void_by_muscle(0.25, np.array([0.0]), np.array([1.0]), cost)
    assert sim.bladder[0] <= 0.5 and sim.marks_made[0] == 1
    assert sim.scent.strength.max() > 0.0 and abs(sim.urine_spots["vol"].sum() - 0.8) < 0.01


def test_the_nose_reads_a_spot_only_when_it_is_on_it():
    sim = _scent_sim(2)
    sim.P[:] = [[0.10, 0.10], [0.40, 0.40]]
    sim.H[:] = 0.0
    sim._elapsed = 0.0
    sim._add_spots(np.array([1]), np.array([0.2]))
    spot = sim.urine_spots["xy"][0].copy()
    sim.P[1] = [0.45, 0.10]                              # its owner walks off
    level, code = sim.nose_contact(1.0)
    assert not level.any() and not code.any()            # 30 cm away: nothing, however strong the smell
    sim.P[0] = spot - [0.02, 0.0]                        # nose (2 cm ahead) on the spot
    level, code = sim.nose_contact(1.0)
    assert level[0] > 0.9 and level[1] == 0.0
    np.testing.assert_array_equal(code[0], sim.mup_barcodes()[1])
    from fnt.abma.core.senses import PlaceMemory, SenseParams, sense
    obs, _ = sense(sim, 1.0, PlaceMemory(sim.n), SenseParams())
    assert obs[0, SENSORS.index("vno_contact")] > 0.9 and obs[1, SENSORS.index("vno_contact")] == 0.0
    assert obs[0, SENSORS.index("vno_mup_0")] == pytest.approx(level[0] * sim.mup_barcodes()[1, 0])


def test_nose_on_another_animal_reads_its_barcode():
    sim = _scent_sim(2)
    sim.P[:] = [[0.20, 0.20], [0.225, 0.20]]
    sim.H[:] = 0.0
    level, code = sim.nose_contact(0.0)
    assert level[0] == 1.0 and level[1] == 0.0           # only the one facing the other
    np.testing.assert_array_equal(code[0], sim.mup_barcodes()[1])


def test_the_anatomy_brain_urinates_in_the_engine():
    cfg = _oft(seed=0, anatomy_seed=5)
    cfg.scent.enabled, cfg.dt = True, 0.25
    rec = anatomy.record(cfg, 600.0)
    assert len(rec["voids"]) > 0 and rec["bladder"].min() < 0.5
    assert rec["meta"]["unwired"] == []
