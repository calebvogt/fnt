"""The skeleton built from published measurements (fnt.abma.core.skeleton)."""
import copy
import math

import numpy as np
import pytest

from fnt.abma.core import anatomy, brainmap, skeleton


@pytest.fixture(scope="module")
def sk():
    return skeleton.Skeleton("c57_male")


def test_shipped_spec_loads_and_names_its_sources(sk):
    spec = sk.spec
    assert ("c57_male", skeleton.find_skeleton("c57_male")) in skeleton.list_skeletons()
    n = {b: sum(1 for v in spec["lengths"].values() if v["basis"] == b) for b in skeleton.BASES}
    assert n == {"published": 10, "derived": 1, "layout": 4}
    for k, v in spec["lengths"].items():
        if v["basis"] == "published":
            assert v["source"].split(" ")[0].strip(";,") in spec["sources"], k
    # the published body length is what the spine and skull add up to
    along = sum(spec["lengths"][k]["mm"] for k in ("skull", "cervical", "thoracic", "lumbar", "sacrum"))
    assert along == pytest.approx(spec["body"]["snout_to_tail_base_mm"], abs=0.05)


def test_bones_keep_their_length_in_every_pose(sk):
    want = {"femur": "femur", "tibia": "tibia", "foot": "foot", "humerus": "humerus", "forearm": "forearm",
            "forepaw": "forepaw", "scapula": "scapula", "pelvis": "pelvis", "skull": "skull"}
    for phase, amp, bend, sniff in ((0.0, 0.0, 0.0, False), (1.3, 1.0, 0.0, False), (4.0, 0.6, 0.4, True),
                                    (2.2, 1.0, -0.45, False)):
        P = sk.pose(phase, amp, bend, sniff)["points"]
        for a, b, kind in sk.bones:
            if kind in want:
                assert np.linalg.norm(P[a] - P[b]) == pytest.approx(sk.len[kind], abs=1e-6), (kind, phase, bend)
        tail = sum(np.linalg.norm(P[a] - P[b]) for a, b, kind in sk.bones if kind == "tail")
        assert tail == pytest.approx(sk.len["tail"], abs=1e-6)


def test_a_walk_stays_inside_the_printed_joint_ranges(sk):
    for amp in (0.0, 0.5, 1.0):
        rows = sk.gait_check(amp, n=24)
        assert all(r["inside"] for r in rows), [r["joint"] for r in rows if not r["inside"]]
    used = {r["joint"]: r for r in sk.gait_check() if r["joint"] != "reach"}
    assert used["knee"]["used_max"] - used["knee"]["used_min"] > 20      # the legs do move


def test_feet_are_on_the_ground_in_stance_and_lift_in_swing(sk):
    low, high = [], []
    for i in range(24):
        P = sk.pose(2 * math.pi * i / 24, 1.0)["points"]
        assert min(p[1] for p in P.values()) > -1e-6            # nothing goes through the floor
        low.append(min(P["mtp_R"][1], P["mtp_L"][1]))
        high.append(max(P["mtp_R"][1], P["mtp_L"][1]))
    assert max(low) < 1e-6 and max(high) > 2.0
    stand = sk.pose()["points"]
    assert all(stand[k][1] == pytest.approx(0.0, abs=1e-6) for k in ("mtp_R", "mtp_L", "paw_R", "paw_L"))


def test_turning_bends_the_spine_within_its_printed_range(sk):
    per, tper = sk.spine_yaw(0.45)
    assert abs(math.degrees(per)) <= 5.0 + 1e-9 and abs(math.degrees(tper)) <= 33.8 + 1e-9
    straight, bent = sk.pose()["points"], sk.pose(bend=0.45)["points"]
    assert abs(straight["snout"][2]) < 1e-9 and abs(bent["snout"][2]) > 5.0
    for k in ("hip_R", "knee_L", "ls"):                         # the hind quarters are the fixed end
        assert np.allclose(straight[k], bent[k])


def test_check_spec_names_what_is_wrong(sk):
    bad = copy.deepcopy(sk.spec)
    bad["lengths"]["femur"]["source"] = "nobody2020"
    with pytest.raises(ValueError, match="femur"):
        skeleton.check_spec(bad)
    bad = copy.deepcopy(sk.spec)
    bad["spine"][1]["pitch_deg"][4] += 20.0
    with pytest.raises(ValueError, match="thoracic"):
        skeleton.check_spec(bad)
    with pytest.raises(FileNotFoundError):
        skeleton.find_skeleton("no_such_animal")


def test_page_payload_and_page(sk):
    pay = sk.page_payload(phases=12)
    n = len(pay["pts"])
    assert len(pay["rest"]) == n and all(max(b[0], b[1]) < n for b in pay["bones"])
    assert len(pay["frames"]) == 3 and len(pay["frames"][0]) == 12
    assert len(pay["frames"][0][0]) == 3 * len(pay["moving"])
    spec = anatomy.load_spec("c57_male")
    assert spec["body"] == "c57_male"
    html = brainmap.build_page(spec)
    assert "const SKEL={" in html and "__SKEL__" not in html
    assert "const SKEL=null;" in brainmap.build_page(spec, skeleton=None)


def test_describe_and_plot(sk, tmp_path):
    text = skeleton.describe(sk.spec)
    assert "skop2020" in text and "NO" not in text.split("What had")[0].split("inside")[-1]
    out = skeleton.plot(sk.spec, str(tmp_path / "skeleton.png"))
    assert (tmp_path / "skeleton.png").stat().st_size > 20000 and out.endswith("skeleton.png")
