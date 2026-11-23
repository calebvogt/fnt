"""Mask Tracker's frame-to-frame linker (MultiObjectTracker, bytetrack by default) and
``mask_tracker_cli relink``, on synthetic scenes (no model or video needed).

Runs under pytest, or directly.
"""
import json
import os

import numpy as np
import pytest

from fnt.videoTracking.mask_tracker_inference import (SINGLETON_ID0, MaskInferenceConfig,
                                                      MultiObjectTracker, mask_crop)

H, W = 120, 200
HERE = os.path.dirname(os.path.abspath(__file__))


def disc(cx, cy, rx=8, ry=None):
    ry = rx if ry is None else ry
    yy, xx = np.mgrid[:H, :W]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def dets(*objs):
    """objs: (cx, cy, score[, rx, ry]) -> one frame's detections with masks."""
    masks = np.array([disc(o[0], o[1], *o[3:]) for o in objs]).reshape(-1, H, W)
    boxes = np.array([[o[0] - 8, o[1] - 8, o[0] + 8, o[1] + 8] for o in objs], float).reshape(-1, 4)
    return dict(masks=masks, boxes=boxes, scores=np.array([o[2] for o in objs], float),
                labels=np.ones(len(objs), int))


def run(frames, step=1, **kw):
    kw.setdefault("max_detections", 0)
    t = MultiObjectTracker(MaskInferenceConfig(use_masks=True, frame_step=step, **kw))
    for i, d in enumerate(frames):
        t.update(d, i * step, 30.0, frame_hw=(H, W))
    return t


def ids_by_frame(t):
    out = {}
    for r in t.records:
        out.setdefault(r["frame"], []).append((r["object_id"], r["x"], r["y"]))
    return out


# ----------------------------------------------------------------------
# the old linker is unchanged
# ----------------------------------------------------------------------
def _golden_scene(seed):
    rng = np.random.default_rng(seed)
    frames = []
    pos = rng.uniform([20, 20], [180, 100], size=(4, 2))
    vel = rng.normal(0, 3, size=(4, 2))
    for _ in range(40):
        pos = np.clip(pos + vel + rng.normal(0, 1, size=(4, 2)), 10, [190, 110])
        keep = [i for i in range(4) if rng.random() > 0.15]
        masks = np.array([disc(*pos[i]) for i in keep]) if keep else np.zeros((0, H, W), bool)
        boxes = np.array([[pos[i][0] - 8, pos[i][1] - 8, pos[i][0] + 8, pos[i][1] + 8]
                          for i in keep]).reshape(-1, 4)
        scores = np.round(rng.uniform(0.3, 0.95, size=len(keep)), 3)
        frames.append(dict(masks=masks, boxes=boxes, scores=scores, labels=np.ones(len(keep), int)))
    return frames


def test_the_centroid_linker_gives_exactly_what_it_gave_before():
    """Golden records from the tracker as it was before the bytetrack linker."""
    with open(os.path.join(HERE, "linker_golden_centroid.json")) as fh:
        gold = json.load(fh)
    for seed in (1, 2, 3):
        fr = _golden_scene(seed)
        for key, kw in ((f"s{seed}_never_md4", dict(max_detections=4)),
                        (f"s{seed}_md3_end5", dict(max_detections=3, max_disappeared_frames=5)),
                        (f"s{seed}_greedy", dict(max_detections=0, matching_algorithm="greedy",
                                                 max_disappeared_frames=2))):
            t = MultiObjectTracker(MaskInferenceConfig(use_masks=True, linker="centroid", **kw))
            for f, d in enumerate(fr):
                t.update(d, f, 30.0, frame_hw=(H, W))
            got = [[r["frame"], r["object_id"], r["x"], r["y"]] for r in t.records]
            assert got == gold[key], key


# ----------------------------------------------------------------------
# bytetrack
# ----------------------------------------------------------------------
def test_two_animals_crossing_keep_their_ids():
    """Kalman motion carries each through the crossing (nearest-centroid swaps here)."""
    frames = [dets((20 + 8 * i, 55, 0.9), (180 - 8 * i, 63, 0.9)) for i in range(21)]
    t = run(frames, max_detections=2, gate_px=20)
    by_id = {}
    for r in t.records:
        by_id.setdefault(r["object_id"], []).append(r["x"])
    assert len(by_id) == 2
    for xs in by_id.values():
        d = np.diff(xs)
        assert (d > 0).all() or (d < 0).all()             # each keeps going its own way


def test_a_huddle_of_two_touching_animals_keeps_each_on_its_side():
    rng = np.random.default_rng(0)
    frames = []
    for i in range(30):
        jx, jy = rng.normal(0, 1.5, 2)
        frames.append(dets((100 + jx, 50 + jy, 0.9, 14, 7), (100 - jx, 64 - jy, 0.9, 14, 7)))
    t = run(frames, max_detections=2, gate_px=10)
    side = {}
    for r in t.records:
        side.setdefault(r["object_id"], set()).add(r["y"] > 57)
    assert len(side) == 2 and all(len(v) == 1 for v in side.values())


def test_low_confidence_flicker_continues_a_track_but_never_starts_one():
    frames = []
    for i in range(12):
        objs = [(30 + 3 * i, 60, 0.9 if i % 2 == 0 else 0.3)]       # one animal, flickering
        if i == 5:
            objs.append((170, 20, 0.3))                               # a doubtful one, far off
        frames.append(dets(*objs))
    t = run(frames, max_detections=0, two_pass_conf=0.5, gate_px=10)
    ids = {r["object_id"] for r in t.records if r["x"] < 100}
    assert ids == {1}                                                 # one track through the flicker
    lone = [r["object_id"] for r in t.records if r["x"] > 150]
    assert len(lone) == 1 and lone[0] >= SINGLETON_ID0               # its own id, no track
    assert t.n_singletons == 1 and t.next_object_id == 2


def test_tracks_end_at_the_sixth_missed_analysed_frame_and_frame_step_counts_analysed_frames():
    # every 3rd video frame analysed; the animal vanishes for 5, then for 6 analysed frames
    seen = [True] * 5 + [False] * 5 + [True] * 3 + [False] * 6 + [True] * 2
    frames = [dets((50 + i, 60, 0.9)) if s else dets() for i, s in enumerate(seen)]
    t = run(frames, step=3, max_detections=0, gate_px=5)
    ids = [r["object_id"] for r in t.records]
    assert ids == [1] * 8 + [2] * 2                                  # survives 5, ends at the 6th
    assert t.resolved_settings()["max_disappeared_frames"] == 5


def test_one_animal_keeps_id_1_wherever_it_reappears():
    frames = [dets((20, 20, 0.9)), dets((180, 100, 0.9)), dets(), dets(), dets(),
              dets(), dets(), dets(), dets(), dets((30, 90, 0.3))]
    t = run(frames, max_detections=1)
    assert {r["object_id"] for r in t.records} == {1}
    s = t.resolved_settings()
    assert s["takeover"] is True and s["max_disappeared_frames"] == 0


def test_auto_gate_and_split_follow_the_animals():
    frames = [dets((20 + i, 30, 0.6, 10), (20 + i, 90, 0.4, 10)) for i in range(40)]
    t = run(frames, max_detections=0)
    s = t.resolved_settings()
    body = np.sqrt(disc(50, 50, 10).sum())
    assert s["median_animal_px"] == pytest.approx(body, abs=0.5)
    assert s["gate_px_per_analysed_frame"] == pytest.approx(max(0.5 * body, 0.012 * np.hypot(H, W)), abs=0.2)
    assert s["two_pass_conf"] == pytest.approx(min(0.5, 0.8 * 0.5), abs=1e-6)   # median confidence 0.5


# ----------------------------------------------------------------------
# relink
# ----------------------------------------------------------------------
def test_decode_crop_matches_the_full_mask():
    from fnt.videoTracking.mask_tracker_annotator import mask_to_rle
    from fnt.videoTracking.mtt_relink import decode_crop
    m = disc(60, 40, 12, 6) | disc(75, 45, 4)
    m[0:3, 0:2] = True                                                # a run wrapping a column end
    got = decode_crop(np.array(mask_to_rle(m)["counts"]), H)
    want = mask_crop(m)
    assert got[0] == want[0] and got[1] == want[1] and np.array_equal(got[2], want[2])
    assert decode_crop(np.array(mask_to_rle(np.zeros((H, W), bool))["counts"]), H) is None


def _write_run(folder, frames, step=1, **kw):
    """A finished run's folder as tracking leaves it, linked with ``kw``."""
    from dataclasses import asdict
    from fnt.videoTracking import mtt_runs
    from fnt.videoTracking.track_store import TrackMaskWriter
    os.makedirs(folder)
    cfg = MaskInferenceConfig(use_masks=True, frame_step=step, **kw)
    t = MultiObjectTracker(cfg)
    with TrackMaskWriter(os.path.join(folder, "v_tracks.h5"), video_path="v.mp4", fps=30.0,
                         frame_count=len(frames) * step, width=W, height=H) as w:
        for i, d in enumerate(frames):
            m = t.update(d, i * step, 30.0, frame_hw=(H, W))
            if m:
                w.add_frame(i * step, m)
    t.get_trajectories({"1": "vole"}).to_csv(os.path.join(folder, "trajectories.csv"), index=False)
    mtt_runs.write_run_record(folder, {"video": "v.mp4", "model_dir": "", "settings": asdict(cfg),
                                       "complete": True}, ["tracked"])


def test_relink_gives_what_tracking_with_the_new_linker_gives_and_keeps_the_old_files(tmp_path):
    import pandas as pd
    from fnt.videoTracking import mask_tracker_cli as C
    from fnt.videoTracking.mtt_relink import relink_folder
    frames = [dets((20 + 8 * i, 55, 0.9), (180 - 8 * i, 63, 0.9)) for i in range(21)]
    folder = str(tmp_path / "v_MaskTracker")
    _write_run(folder, frames, step=3, max_detections=2, linker="centroid")
    old = pd.read_csv(os.path.join(folder, "trajectories.csv"))
    assert C.relink_folders([str(tmp_path)]) == [os.path.normpath(folder)]

    res = relink_folder(folder, log=lambda m: None)
    new = pd.read_csv(os.path.join(folder, "trajectories.csv"))
    live = run(frames, step=3, max_detections=2)                       # tracked with bytetrack
    key = lambda df: sorted(zip(df["frame"], df["object_id"], df["x"].round(1)))
    assert key(new) == key(live.get_trajectories({"1": "vole"}))
    assert list(new.columns) == list(old.columns) and len(new) == len(old)
    assert set(new["object_name"]) == {"vole_1", "vole_2"}
    aside = res["aside"]
    assert aside == os.path.join(str(tmp_path), "_TO_DELETE_tracking", "v_MaskTracker")
    assert sorted(os.listdir(aside)) == ["inference.log", "run_settings.json", "trajectories.csv",
                                         "v_tracks.h5"]
    with open(os.path.join(folder, "run_settings.json")) as fh:
        rec = json.load(fh)
    assert rec["settings"]["linker"] == "bytetrack" and rec["settings"]["frame_step"] == 3
    assert rec["linker_resolved"]["linker"] == "bytetrack"
    assert rec["relinked"]["previous_settings"]["linker"] == "centroid"
    from fnt.videoTracking.track_store import TrackMaskReader
    with TrackMaskReader(os.path.join(folder, "v_tracks.h5")) as rd:
        assert sorted(set(rd.object_id.tolist())) == [1, 2]
    with open(os.path.join(folder, "inference.log")) as fh:
        log = fh.read()
    assert "tracked" in log and "relinked" in log


def test_relink_settings_keep_the_runs_detection_settings_and_take_given_linker_ones():
    from fnt.videoTracking.mtt_relink import relink_settings
    rec = {"settings": {"frame_step": 3, "max_detections": 6, "max_disappeared_frames": 0,
                        "linker": "centroid", "confidence_threshold": 0.25, "unknown_key": 1}}
    c = relink_settings(rec)
    assert (c.frame_step, c.max_detections, c.confidence_threshold) == (3, 6, 0.25)
    assert c.linker == "bytetrack" and c.max_disappeared_frames == -1     # the linker's defaults
    c = relink_settings(rec, {"linker": "centroid", "gate_px": 60.0})
    assert c.linker == "centroid" and c.gate_px == 60.0


def test_cli_linker_flags_and_an_old_run_repeats_with_the_centroid_linker(tmp_path):
    from fnt.videoTracking import mask_tracker_cli as C
    a = C._parser().parse_args(["track", "--videos", "x.mp4", "--linker", "bytetrack", "--gate-px", "60",
                                "--two-pass-conf", "-1", "--max-disappear", "auto", "--no-takeover"])
    s = C.inference_settings(a)
    assert s["linker"] == "bytetrack" and s["gate_px"] == 60.0 and s["two_pass_conf"] == -1.0
    assert s["max_disappeared_frames"] == -1 and s["takeover"] is False
    assert "--linker" in C.settings_argv(a) and "--no-takeover" in C.settings_argv(a)
    run_dir = tmp_path / "old_MaskTracker"
    run_dir.mkdir()
    (run_dir / "run_settings.json").write_text(json.dumps(
        {"model_dir": "m", "settings": {"max_disappeared_frames": 0, "max_detections": 6}}))
    a = C._parser().parse_args(["track", "--videos", "x.mp4", "--settings-from", str(run_dir)])
    assert C.inference_settings(a)["linker"] == "centroid"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
