"""Names across cameras: a vole anchored alone in its box keeps its name after walking out of the doorway onto
the overhead camera's floor, where UWB alone could not tell it from another vole close by."""
import json

import numpy as np
import pandas as pd

from fnt.videoTracking.mtt_floor import FloorMap
from fnt.videoTracking.mtt_identity import Identity, lookup

FPS, N = 30, 600                                      # 20 s


def _setup(tmp_path):
    # floor: 3 x 3 m; box Z1 in the corner (0..0.5, 0..0.5) with its doorway on the right side
    pd.DataFrame([("Z1", 0.0, 0.0), ("Z1", 0.5, 0.0), ("Z1", 0.5, 0.5), ("Z1", 0.0, 0.5)],
                 columns=["zone_name", "x", "y"]).to_csv(tmp_path / "zones.csv", index=False)
    # overhead: 100 px per metre, image y down = floor y up, no lens (the homography takes the lens step's
    # normalised coordinates, (px - 960) / 1000 and (py - 540) / 1000)
    H = [[10.0, 0, 9.6], [0, -10.0, -2.4], [0, 0, 1]]
    FloorMap(lens=[0.0, 0.0, 960.0, 540.0], H=H).save(tmp_path / "floor.json")
    # zone camera: the box fills the frame; its exit (the doorway) is the right edge
    json.dump({"cam1": {"exits": [[[1800, 300], [1920, 300], [1920, 800], [1800, 800]]]}},
              open(tmp_path / "exits.json", "w"))
    t = np.arange(N) / FPS
    # A: in the box 0-8 s (alone; B far away), out of the doorway at 8 s, then along y = 0.4 to x = 1.5;
    # B: walks along y = 0.55 from x = 2.4 towards the box, ending beside A (the two 0.15 m apart)
    ax = np.where(t < 8, 0.25, 0.5 + (t - 8) * 0.085)
    ay = np.where(t < 8, 0.25, 0.4)
    bx = np.maximum(2.4 - t * 0.075, 0.9)
    by = np.full(N, 0.55)
    rows = []
    for k in range(0, N, 10):                       # UWB at 3 Hz, with 0.12 m noise once both are outside
        rng = np.random.default_rng(k)
        noise = 0.12 if t[k] > 9 else 0.0
        rows += [("A", 1000 + t[k], ax[k] + rng.normal(0, noise), ay[k] + rng.normal(0, noise)),
                 ("B", 1000 + t[k], bx[k] + rng.normal(0, noise), by[k] + rng.normal(0, noise))]
    pd.DataFrame(rows, columns=["SexID", "sec", "x", "y"]).to_csv(tmp_path / "uwb.csv", index=False)

    def traj(frames, xs, ys, oid):
        return pd.DataFrame(dict(frame=frames, object_id=oid, x=xs, y=ys, confidence=0.9))
    inbox = np.arange(0, 240)                        # zone camera: A until 8 s, leaving at the right edge
    z = traj(inbox, np.linspace(900, 1850, len(inbox)), np.full(len(inbox), 540), 1)
    z.to_csv(tmp_path / "cam1.csv", index=False)
    out = np.arange(250, N)                          # overhead: A from 8.3 s (a new track), B all along
    over = pd.concat([traj(out, ax[out] * 100, (3 - ay[out]) * 100, 7),
                      traj(np.arange(N), bx * 100, (3 - by) * 100, 3)])
    over.to_csv(tmp_path / "cam5.csv", index=False)
    return dict(start="1970-01-01T00:16:40+00:00", fps=FPS, frames=N, uwb_offset_s=0, uwb=str(tmp_path / "uwb.csv"),
                zones=str(tmp_path / "zones.csv"),
                cameras={"cam1": {"tracks": str(tmp_path / "cam1.csv"), "zone": "Z1", "exits": str(tmp_path / "exits.json")},
                         "cam5": {"tracks": str(tmp_path / "cam5.csv"), "floor_map": str(tmp_path / "floor.json")}},
                params={"min_frames": 5})


def test_a_name_anchored_in_the_box_is_carried_out_through_the_doorway(tmp_path):
    idn = Identity(_setup(tmp_path))
    res = idn.run(log=lambda *_: None)
    get = lookup(res)
    assert get("cam1", 1, 100)[0] == "A"                        # alone in its box: anchored
    assert get("cam5", 7, 300)[0] == "A"                        # handed off at the doorway
    assert get("cam5", 7, N - 1)[0] == "A"                      # and kept beside B at the end
    assert get("cam5", 3, N - 1)[0] == "B"
    assert idn.links_.kind.str.startswith("handoff").any()


def test_an_animal_is_not_named_before_its_first_uwb_fix(tmp_path):
    # C is released at 15 s at (2, 2); before that, UWB has no C, but interpolation alone would hold C at its first
    # fix, and something seen there early (a hand at the release point) would be anchored as C
    cfg = _setup(tmp_path)
    u = pd.read_csv(cfg["uwb"])
    late = [("C", 1000 + k / FPS, 2.0, 2.0) for k in range(450, N, 10)]
    pd.concat([u, pd.DataFrame(late, columns=u.columns)]).to_csv(cfg["uwb"], index=False)
    over = pd.read_csv(cfg["cameras"]["cam5"]["tracks"])
    early = np.arange(0, 300)
    hand = pd.DataFrame(dict(frame=early, object_id=9, x=200.0, y=100.0, confidence=0.9))
    pd.concat([over, hand]).to_csv(cfg["cameras"]["cam5"]["tracks"], index=False)
    get = lookup(Identity(cfg).run(log=lambda *_: None))
    assert get("cam5", 9, 100)[0] != "C"
    assert get("cam5", 7, N - 1)[0] == "A"


def _corrected(tmp_path, rows):
    from fnt.videoTracking.mtt_identity import read_corrections
    cfg = _setup(tmp_path)
    pd.DataFrame(rows, columns=["camera", "object_id", "frame_first", "frame_last", "animal"]).to_csv(
        tmp_path / "corrections.csv", index=False)
    idn = Identity(cfg, read_corrections(str(tmp_path / "corrections.csv")))
    res = idn.run(log=lambda *_: None)
    return idn, res, lookup(res)


def test_a_correction_overrides_the_evidence_and_moves_the_rest(tmp_path):
    # the reviewer says overhead track 3 is A for the last 5 s: A can't be on two tracks at once, so track 7 (which
    # the UWB and the handoff call A) gives up the name, and the rest of track 3 stays B
    idn, res, get = _corrected(tmp_path, [("cam5", 3, 450, N - 1, "A")])
    assert get("cam5", 3, N - 1)[0] == "A"
    assert get("cam5", 3, 100)[0] == "B"
    assert get("cam5", 7, N - 1)[0] != "A"
    assert res.fixed.sum() == 1 and not idn.broken_


def test_a_stretch_marked_not_an_animal_is_left_unnamed(tmp_path):
    idn, res, get = _corrected(tmp_path, [("cam1", 1, 0, 50, "none")])
    assert get("cam1", 1, 20)[0] == ""
    assert get("cam1", 1, 100)[0] == "A"                       # the rest of the track keeps its anchors
