"""Behavioural classification inputs for a window of video, on a synthetic two-mouse clip.

Mouse A walks right toward mouse B, which faces it, and stops nose to nose with it;
both then sit still. A stub detector (dark blobs) stands in for the mask model, so the
tests check tracking, keypoints, pair distances, the candidate rules and the files
written, not the model.

Runs under pytest, or directly.
"""
import csv
import os
import tempfile

import cv2
import numpy as np
import pytest

from fnt.videoTracking import behavior_features as bf

FPS, N, W, H = 15, 90, 400, 240


def draw_mouse(img, cx, cy, facing, bob=0):
    """Body 60 x 24 px, head toward ``facing`` (+1 right, -1 left), 40-px tail behind;
    ``bob`` moves the head up or down by that many px (sniffing)."""
    cv2.ellipse(img, (int(cx), int(cy)), (30, 12), 0, 0, 360, 30, -1)
    if bob:
        img[int(cy) - 14:int(cy) + 15, int(cx) + facing * 12:int(cx) + facing * 31:facing] = 200
        cv2.ellipse(img, (int(cx + facing * 20), int(cy + bob)), (11, 8), 0, 0, 360, 30, -1)
    rump = int(cx - facing * 28)
    cv2.line(img, (rump, int(cy)), (rump - facing * 40, int(cy) + 4), 30, 3)


def a_x(i):
    return min(60 + 3.5 * i, 165)            # walks for ~2 s, then stops (nose at 195)


def make_video(path):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(N):
        img = np.full((H, W), 200, np.uint8)
        draw_mouse(img, a_x(i), 120, +1, bob=(3 if i % 2 else -3) if i > 30 else 0)  # sniffs once there
        draw_mouse(img, 250, 120, -1)        # B's nose at 220: 25 px from A's
        w.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    w.release()


def blob_detector(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((grey < 100).astype(np.uint8), connectivity=8)
    return [(lbl == i, 0.9) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 200]


@pytest.fixture(scope="module")
def window():
    tmp = tempfile.mkdtemp()
    video = os.path.join(tmp, "clip.avi")
    make_video(video)
    paths = bf.run_window(video, 0, N, os.path.join(tmp, "out"), blob_detector, "stub")
    rows = list(csv.DictReader(open(paths["tracks_csv"])))
    pairs = list(csv.DictReader(open(paths["pairs_csv"])))
    bouts = list(csv.DictReader(open(paths["bouts_csv"])))
    return tmp, video, paths, rows, pairs, bouts


def track_of(rows, x_min, x_max):
    """The track whose mean centroid lies between x_min and x_max."""
    by = {}
    for r in rows:
        by.setdefault(r["track"], []).append(float(r["cx"]))
    return [k for k, xs in by.items() if x_min <= np.mean(xs) <= x_max][0]


def test_two_tracks_span_the_window(window):
    _, _, _, rows, _, _ = window
    counts = {}
    for r in rows:
        counts[r["track"]] = counts.get(r["track"], 0) + 1
    assert len(counts) == 2 and all(c >= N - 2 for c in counts.values())


def test_noses_point_where_the_mice_face(window):
    _, _, _, rows, _, _ = window
    a = track_of(rows, 100, 200)
    b = track_of(rows, 230, 270)
    for r in rows:
        if r["track"] == a:
            assert float(r["nose_x"]) > float(r["tb_x"])     # A faces right
        else:
            assert r["track"] == b and float(r["nose_x"]) < float(r["tb_x"])
    body = np.median([float(r["body_len"]) for r in rows])
    assert 45 < body < 70


def test_pairs_measure_the_meeting(window):
    _, _, _, rows, pairs, _ = window
    a = track_of(rows, 100, 200)
    end = [float(p["nose_nose"]) for p in pairs if p["focal"] == a and int(p["t"]) > 40]
    start = [float(p["nose_nose"]) for p in pairs if p["focal"] == a and int(p["t"]) < 5]
    assert np.median(end) < bf.NOSE_NEAR_BL < np.median(start)


def test_rules_propose_nose_nose_and_immobility(window):
    _, _, _, rows, _, bouts = window
    kinds = [b["kind"] for b in bouts]
    assert "nose_nose" in kinds
    nn = [b for b in bouts if b["kind"] == "nose_nose"][0]
    assert int(nn["frame0"]) >= 25 and float(nn["dur_s"]) >= 2  # from when A arrives
    b = track_of(rows, 230, 270)
    assert any(x["kind"] == "immobile" and x["focal"] == b for x in bouts)
    assert "chase" not in kinds


def test_files_are_ready_for_review(window):
    tmp, video, paths, _, _, bouts = window
    for p in ("tracks_h5", "tracks_csv", "pairs_csv", "bouts_csv"):
        assert os.path.exists(paths[p])
    assert all(b["tracks_h5"] == paths["tracks_h5"] and b["video"] == video for b in bouts)
    from fnt.videoTracking.track_store import TrackMaskReader
    with TrackMaskReader(paths["tracks_h5"]) as r:
        assert len(r.object_ids()) == 2 and len(r.detections_for_frame(10)) == 2
    session = os.path.join(tmp, "session.csv")
    n = bf.build_session([paths["bouts_csv"], paths["bouts_csv"]], session)
    ids = [r["bout_id"] for r in csv.DictReader(open(session))]
    assert n == 2 * len(bouts) and ids == [str(i) for i in range(n)]


def _pair_rows(n, head_ev=3.0, me_a=None, me_b=None):
    """Two animals nose to nose for n frames, as the CSVs hold them."""
    rows, pairs = [], []
    for t in range(n):
        for k, me in ((1, me_a), (2, me_b)):
            rows.append({"t": t, "frame": t, "track": k, "cx": 100 * k, "cy": 100, "vx": 0, "vy": 0,
                         "speed": 0.0, "heading": 0.0, "head_ev": head_ev, "merged": 0,
                         "me_head": (me[t] if me else 1.0), "me_mid": 1.0, "me_rear": 1.0})
        for f, o in ((1, 2), (2, 1)):
            pairs.append({"t": t, "frame": t, "focal": f, "other": o, "centroid_dist": 1.0,
                          "nose_nose": 0.2, "nose_tailbase": 1.2, "contact": 1})
    return rows, pairs


def test_resting_nose_to_nose_is_not_sniffing():
    rows, pairs = _pair_rows(30)
    assert not [b for b in bf.candidate_bouts(rows, pairs, 15, 30) if b["kind"] == "nose_nose"]


def test_a_busy_head_makes_it_a_sniff():
    rows, pairs = _pair_rows(30, me_a=[5.0 if t % 2 else 1.0 for t in range(30)])
    assert [b for b in bf.candidate_bouts(rows, pairs, 15, 30) if b["kind"] == "nose_nose"]


def test_a_guessed_head_end_is_not_used():
    rows, pairs = _pair_rows(30, head_ev=0.5, me_a=[5.0 if t % 2 else 1.0 for t in range(30)])
    assert not [b for b in bf.candidate_bouts(rows, pairs, 15, 30) if b["kind"] in ("nose_nose", "anogenital")]


def test_a_still_merged_blob_is_a_huddle():
    rows = [{"t": t, "frame": t, "track": 1, "cx": 50, "cy": 50, "vx": 0, "vy": 0, "speed": 0.0,
             "heading": 0.0, "head_ev": 0.0, "merged": 1, "me_head": 1.0, "me_mid": 1.0, "me_rear": 1.0}
            for t in range(60)]
    kinds = [b["kind"] for b in bf.candidate_bouts(rows, [], 15, 60)]
    assert kinds == ["huddle"]


def test_merged_is_judged_against_one_animal():
    """A pile that stays merged all window is still a pile."""
    rows = [{"area": 1000, "tail": 1} for _ in range(30)] + [{"area": 3000, "tail": 0} for _ in range(30)]
    bf.mark_merged(rows)
    assert [r["merged"] for r in rows] == [0] * 30 + [1] * 30


def test_bouts_rebuild_from_the_saved_csvs(window):
    tmp, _, paths, _, _, bouts = window
    rebuilt = bf.bouts_from_csv(paths["tracks_csv"], paths["pairs_csv"], FPS)
    assert sorted((b["kind"], str(b["focal"]), b["frame0"]) for b in rebuilt) ==         sorted((b["kind"], b["focal"], int(b["frame0"])) for b in bouts)


def test_a_session_caps_each_kind_across_windows(tmp_path):
    files = []
    for w in range(3):
        f = tmp_path / f"w{w}_bouts.csv"
        with open(f, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(bf.BOUT_COLS)
            for k in range(10):
                wr.writerow(["anogenital", 1, 2, k, k + 5, 1 + k, f"v{w}.mp4", "", ""])
            wr.writerow(["chase", 1, 2, 0, 9, 0.7, f"v{w}.mp4", "", ""])
        files.append(str(f))
    out = tmp_path / "s.csv"
    n = bf.build_session(files, str(out), per_kind=6)
    rows = list(csv.DictReader(open(out)))
    ano = [r for r in rows if r["kind"] == "anogenital"]
    assert n == 6 + 3 and len(ano) == 6
    assert {r["video"] for r in ano} == {"v0.mp4", "v1.mp4", "v2.mp4"}   # two from each window
    assert all(float(r["dur_s"]) >= 9 for r in ano)                    # each window's longest


def test_link_tracks_bridges_a_short_gap():
    m1 = np.zeros((50, 80), bool); m1[20:30, 10:30] = True
    det = bf._as_det(m1, 0.9)
    dets = [[det]] * 5 + [[]] * 3 + [[det]] * 20
    tracks = bf.link_tracks(dets, fps=15)
    assert len(tracks) == 1 and len(next(iter(tracks.values()))) == 25


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
