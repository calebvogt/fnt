"""Counting animals through huddles (mtt_counts): the model's masks in a huddle
say little about how many animals lie there; the animals that walked in and out
through the camera's exits (exits.json, the Exits window) do. An animal lost
anywhere else is hiding, not gone.

Runs under pytest, or directly.
"""
import json

import numpy as np
import pandas as pd
import pytest

from fnt.videoTracking.frame_activity import camera_key
from fnt.videoTracking.mtt_counts import CountSettings, count_video
from fnt.videoTracking.mtt_exits import save_camera_exits

W, H, FPS = 400, 200, 10.0
EXIT = [(0, 80), (20, 80), (20, 120), (0, 120)]          # the entrance, left wall
VIDEO = "VoleCosm_Camera1_{68E5}_20260209000000_processed.mp4"


def _write(tmp_path, rows, n_frames, exits=True):
    video = tmp_path / VIDEO
    out = tmp_path / (video.stem + "_MaskTracker")
    out.mkdir()
    df = pd.DataFrame(rows, columns=["frame", "x", "y", "w", "h", "confidence"])
    df["object_id"] = 1
    df["bbox_x1"], df["bbox_x2"] = df.x - df.w / 2, df.x + df.w / 2
    df["bbox_y1"], df["bbox_y2"] = df.y - df.h / 2, df.y + df.h / 2
    df["mask_area"] = df.w * df.h * 0.7
    df.to_csv(out / "trajectories.csv", index=False)
    (out / "run_settings.json").write_text(json.dumps(
        {"fps": FPS, "width": W, "height": H, "frames": n_frames, "video": str(video),
         "settings": {"frame_step": 1}}))
    if exits:
        save_camera_exits(str(tmp_path), camera_key(str(video))[1], [EXIT], (W, H))
    return video


def _animal(frames, x0, x1, y=100):
    xs = np.linspace(x0, x1, len(frames))
    return [(f, x, y, 40, 20, 0.9) for f, x in zip(frames, xs)]


def _count(video, **kw):
    count_video(str(video), CountSettings(animals=4, gap_s=0.5, **kw), log=None)
    out = video.parent / (video.stem + "_MaskTracker")
    return (pd.read_csv(out / "counts.csv").set_index("frame"),
            pd.read_csv(out / "count_events.csv"))


def test_a_huddle_counts_the_animals_that_went_in_not_its_masks(tmp_path):
    rows = []
    rows += _animal(range(3, 20), 12, 200)                 # A comes in, walks to the middle
    rows += [(f, 200, 100, 40, 20, 0.9) for f in range(20, 45)]      # and waits there
    rows += _animal(range(25, 45), 12, 150)                # B comes in later, joins A
    for f in range(45, 120):                               # A and B huddle: the model draws 1-4 masks
        k = 1 + (f % 4)
        rows += [(f, 200 + 6 * i, 100 + 3 * i, 40, 20, 0.8) for i in range(k)]
    rows += _animal(range(120, 140), 200, 12)              # one leaves through the exit
    rows += [(f, 200, 100, 40, 20, 0.9) for f in range(120, 160)]   # the other stays
    c, ev = _count(_write(tmp_path, rows, 160))
    assert c.loc[10, "in_view"] == 1
    assert (c.loc[50:115, "in_view"] == 2).all()           # the huddle: 2, whatever the masks
    assert c.loc[c.index >= 145, "in_view"].eq(1).all()
    assert set(ev["at"]) == {"exit"}
    assert (ev.event == "arrive").sum() == 2 and (ev.event == "leave").sum() == 1


def test_a_huddle_there_from_the_start_is_counted_back_from_the_animals_leaving(tmp_path):
    rows = []
    for f in range(0, 60):                                 # 3 animals huddled from the first frame: 2 masks
        rows += [(f, 200 + 8 * i, 100, 40, 20, 0.8) for i in range(2)]
    for j, f0 in enumerate((60, 80, 100)):                 # each leaves in turn through the exit
        rows += _animal(range(f0, f0 + 15), 200, 12)
        rows += [(f, 200, 100, 40, 20, 0.8) for f in range(f0, f0 + 20) if j < 2]
    c, ev = _count(_write(tmp_path, rows, 120))
    assert (c.loc[5:55, "in_view"] == 3).all()


def test_an_animal_lost_away_from_the_exits_is_hiding_not_gone(tmp_path):
    rows = []
    rows += _animal(range(3, 20), 12, 300)                 # comes in, walks under the feeder
    rows += _animal(range(70, 90), 300, 12)                # comes out 5 s later, leaves
    c, ev = _count(_write(tmp_path, rows, 100))
    assert (c.loc[25:65, "in_view"] == 0).all()
    assert (c.loc[25:65, "hiding"] == 1).all() and (c.loc[25:65, "in_space"] == 1).all()
    assert c.loc[95, "in_space"] == 0
    assert list(ev.event) == ["arrive", "into hiding", "out of hiding", "leave"]


def test_animals_side_by_side_each_with_its_own_mask_count_apart(tmp_path):
    rows = _animal(range(0, 30), 12, 300, y=40)            # a walker: one animal's mask size
    for f in range(40, 100):                               # two curled up side by side: each
        rows += [(f, 200, 100, 26, 20, 0.9), (f, 227, 100, 26, 20, 0.9)]   # smaller than a walker
    c, ev = _count(_write(tmp_path, rows, 100, exits=False))
    assert (c.loc[45:95, "in_view"] == 2).all()            # not 1, though the pair's area is ~1.3


def test_without_exits_every_start_and_end_is_an_arrival_or_departure(tmp_path):
    rows = _animal(range(3, 20), 12, 300) + _animal(range(70, 90), 300, 12)
    c, ev = _count(_write(tmp_path, rows, 100, exits=False))
    assert (c.loc[25:65, "in_space"] == 0).all() and set(ev["at"]) == {"unknown"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
