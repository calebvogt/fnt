"""ROI Analysis Tool without the GUI: roi_cli runs saved roiConfig.json files (data first, then plots and
videos, in parallel), and the GUI writes a video's configuration as soon as it is queued."""
import json
import os
import subprocess
import types

import numpy as np
import pandas as pd

from fnt.sleapProcessing import roi_cli
from fnt.sleapProcessing.sleap_roi_tool_pyqt import ROIToolGUI, VideoROIConfig

FLOOR = [[20, 20], [140, 20], [140, 140], [20, 140]]
CENTER = [[38, 38], [122, 38], [122, 122], [38, 122]]


def _video_with_track(folder, stem, n=60):
    """A 2-s 160x160 video and a Mask Tracker track: the vole sits in the centre, then by the wall."""
    video = os.path.join(folder, f"{stem}.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=gray:s=160x160:r=30:d=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", video], check=True)
    mt = os.path.join(folder, f"{stem}_MaskTracker")
    os.makedirs(mt)
    x = np.where(np.arange(n) < n // 2, 80.0, 30.0)
    pd.DataFrame({"frame": np.arange(n), "time_s": np.arange(n) / 30, "object_id": 1, "object_name": "vole_1",
                  "label": 1, "x": x, "y": 80.0}).to_csv(os.path.join(mt, f"{stem}_trajectories.csv"), index=False)
    return video, os.path.join(mt, f"{stem}_trajectories.csv")


def _config(video, csv, **opts):
    c = VideoROIConfig(video, csv, "mtt")
    c.tracking_area, c.tracking_area_set = [[0, 0], [160, 0], [160, 160], [0, 160]], True
    c.rois = [("oft_center", CENTER), ("oft_outer", FLOOR)]
    c.scale_bar_set, c.scale_bar_pixels, c.scale_bar_cm, c.pixels_per_cm = True, 120.0, 39.5, 120 / 39.5
    c.selected_keypoints = ["vole_1"]
    for k, v in opts.items():
        setattr(c, k, v)
    path = os.path.join(os.path.dirname(csv), os.path.basename(video)[:-4] + "_roiConfig.json")
    with open(path, "w") as f:
        json.dump(c.to_config_dict(), f)
    return path


def test_data_first_then_visuals_in_parallel(tmp_path):
    configs = []
    for stem in ("T001_F1_PreOFT", "T001_M2_PreOFT"):
        video, csv = _video_with_track(str(tmp_path), stem)
        configs.append(_config(video, csv, save_data_view=False))
    before = [open(c).read() for c in configs]
    assert roi_cli.main([str(tmp_path), "--workers", "2"]) == 0
    for c, b in zip(configs, before):
        mt = os.path.dirname(c)
        stem = os.path.basename(c).replace("_roiConfig.json", "")
        s = pd.read_csv(os.path.join(mt, f"{stem}_roiSummary.csv")).iloc[0]
        assert abs(s.time_s_oft_center - 1.0) < 0.1          # half of 2 s in the centre
        assert os.path.exists(os.path.join(mt, f"{stem}_roiTracked.mp4"))
        assert open(c).read() == b                           # the drawn configuration is left as it was


def test_data_only_writes_no_video(tmp_path):
    video, csv = _video_with_track(str(tmp_path), "T001_F1_PreOFT")
    cfg = _config(video, csv)
    assert roi_cli.main([cfg, "--data-only", "--workers", "1"]) == 0
    mt = os.path.dirname(cfg)
    assert os.path.exists(os.path.join(mt, "T001_F1_PreOFT_roiSummary.csv"))
    assert not os.path.exists(os.path.join(mt, "T001_F1_PreOFT_roiTracked.mp4"))


def test_a_retracked_video_uses_the_new_track(tmp_path):
    """A re-track writes trajectories.csv and moves <video>_trajectories.csv aside; the drawing still names the old."""
    video, csv = _video_with_track(str(tmp_path), "T001_F1_PreOFT")
    cfg = _config(video, csv)
    os.replace(csv, os.path.join(os.path.dirname(csv), "trajectories.csv"))
    assert roi_cli.main([cfg, "--data-only", "--workers", "1"]) == 0
    mt = os.path.dirname(cfg)
    assert os.path.exists(os.path.join(mt, "T001_F1_PreOFT_roiSummary.csv"))
    assert json.load(open(cfg))["csv_path"] == csv                  # the drawing is left as it was


def test_queueing_saves_the_drawing(tmp_path):
    video, csv = _video_with_track(str(tmp_path), "T001_F1_PreOFT")
    c = VideoROIConfig(video, csv, "mtt")
    c.rois = [("oft_center", CENTER), ("oft_outer", FLOOR)]
    ROIToolGUI._save_config_file(types.SimpleNamespace(lbl_status=None), c)
    saved = json.load(open(os.path.join(os.path.dirname(csv), "T001_F1_PreOFT_roiConfig.json")))
    assert saved["rois"][1] == ["oft_outer", FLOOR]
