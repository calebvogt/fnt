"""Per-trial ROIs and the margin (mtt_roi): a camera bumped between trials frames
the arena differently, so a project can keep one outline per trial folder; an
animal seen at an angle stands out past the floor it is on, so an outline can be
grown by a margin. Training frames, videos and predictions each use their own
trial's outline.

Runs under pytest, or directly.
"""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

# Qt before anything that may load torch (torch first crashes Qt on Windows)
_APP = QApplication.instance() or QApplication([])

from fnt.videoTracking.mtt_roi import ROI, frame_source, frame_sources, trial_key  # noqa: E402

FLOOR_A = [(20, 10), (179, 10), (179, 89), (20, 89)]      # trial A's camera
FLOOR_B = [(40, 20), (199, 20), (199, 99), (40, 99)]      # trial B's, bumped 20 px right, 10 down
VA = "Z:/Data/Video/VC-T005_prairie_anosmic-saline/cam5/VoleCosm_Camera5_a.mp4"
VB = "Z:/Data/Video/VC-T009_prairie_anosmic-saline/VoleCosm_Camera5_b.mp4"   # flat trial folder


def _trials():
    return ROI(include=[FLOOR_A], frame_size=(200, 100),
               by_folder={"VC-T009_prairie_anosmic-saline": ROI(include=[FLOOR_B], frame_size=(200, 100))})


def test_trial_key_skips_camera_folders():
    assert trial_key(VA) == "VC-T005_prairie_anosmic-saline"
    assert trial_key(VB) == "VC-T009_prairie_anosmic-saline"
    assert trial_key(r"Z:\x\VC-T014_y\ch3_RZ3\a.wav") == "VC-T014_y"


def test_each_trial_gets_its_own_outline_and_the_rest_the_projects():
    r = _trials()
    assert r.for_source(VB).box(200, 100) == (40, 20, 200, 100)
    assert r.for_source(VA).box(200, 100) == (20, 10, 180, 90)        # no own outline: the project's
    assert r.for_source(None).box(200, 100) == (20, 10, 180, 90)
    back = ROI.from_dict(json.loads(json.dumps(r.to_dict())))
    assert set(back.by_folder) == {"VC-T009_prairie_anosmic-saline"}
    assert back.for_source(VB).include == r.for_source(VB).include
    only = ROI.from_dict({"by_folder": {"T1": {"include": [[[0, 0], [9, 0], [9, 9]]]}}})
    assert only is not None and only.for_source("x/T1/v.mp4") is not None
    assert only.for_source("x/T2/v.mp4") is None                      # whole frames elsewhere


def test_the_margin_grows_the_area_and_shrinks_the_holes():
    hole = [(80, 40), (120, 40), (120, 60), (80, 60)]
    plain = ROI(include=[FLOOR_A], exclude=[hole], frame_size=(200, 100))
    grown = ROI(include=[FLOOR_A], exclude=[hole], frame_size=(200, 100), margin_px=8)
    assert not plain.allows(14, 50, 100, 200) and grown.allows(14, 50, 100, 200)   # past the floor
    assert not grown.allows(5, 50, 100, 200)                                        # beyond the margin
    assert not plain.allows(84, 50, 100, 200) and grown.allows(84, 50, 100, 200)   # hole edge given back
    assert not grown.allows(100, 50, 100, 200)                                      # its middle still out
    assert grown.box(200, 100) == (12, 2, 188, 98)
    assert ROI.from_dict(grown.to_dict()).margin_px == 8
    assert grown.for_size(400, 200).margin_px == 16


def test_training_cuts_each_frame_with_its_trials_outline(tmp_path):
    from fnt.videoTracking.yolo_training import convert_coco_to_yolo
    proj = tmp_path / "proj"
    (proj / "training_frames").mkdir(parents=True)
    (proj / "project_config.json").write_text(json.dumps({"source_videos": [VA, VB]}))
    names = ["VoleCosm_Camera5_a_frame_000010.png", "VoleCosm_Camera5_b_frame_000020.png"]
    for n in names:
        cv2.imwrite(str(proj / "training_frames" / n), np.full((100, 200, 3), 200, np.uint8))
    assert frame_source(names[1], frame_sources(str(proj))) == VB
    coco = {"images": [{"id": k + 1, "file_name": n, "width": 200, "height": 100}
                       for k, n in enumerate(names)],
            "annotations": [], "categories": [{"id": 1, "name": "vole"}]}
    cj = tmp_path / "c.json"
    cj.write_text(json.dumps(coco))
    out = tmp_path / "yolo"
    convert_coco_to_yolo(str(cj), str(proj / "training_frames"), str(out), 0.5,
                         roi=_trials().to_dict())
    shapes = {}
    for s in ("train", "val"):
        for f in (out / "images" / s).glob("*.png"):
            shapes[f.name] = cv2.imread(str(f)).shape[:2]
    assert shapes[names[0]] == (80, 160) and shapes[names[1]] == (80, 160)
    # B's cut starts at (40, 20): its grey corner is where A's floor was not
    b = next(cv2.imread(str(f)) for s in ("train", "val")
             for f in (out / "images" / s).glob(names[1]))
    assert (b[0, 0] == 200).all()


def test_inference_switches_to_the_videos_trial():
    from fnt.videoTracking.yolo_inference import YOLOInference
    eng = YOLOInference("unused", device="cpu", use_masks=True)
    eng._roi_all = _trials()
    eng.roi = eng._roi_all.for_source(None)
    eng.use_source(VB)
    assert eng.roi.box(200, 100) == (40, 20, 200, 100)
    eng.use_source(VA)
    assert eng.roi.box(200, 100) == (20, 10, 180, 90)


def test_the_editor_saves_a_margin_and_a_trial_choice():
    from fnt.videoTracking.mtt_roi_editor import ROIEditorDialog
    dlg = ROIEditorDialog(np.zeros((100, 200, 3), np.uint8), _trials().for_source(VB),
                          trial="VC-T009_prairie_anosmic-saline", own=True)
    dlg.spin_margin.setValue(12)
    assert dlg.for_trial() and dlg.roi().margin_px == 12
    assert dlg.roi().include == [FLOOR_B] or np.allclose(dlg.roi().include[0], FLOOR_B)
    dlg2 = ROIEditorDialog(np.zeros((100, 200, 3), np.uint8), None)
    assert not dlg2.for_trial()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
