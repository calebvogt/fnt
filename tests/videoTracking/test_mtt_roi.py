"""A project's region of interest (mtt_roi): frames are cut to the box around
its included area and everything outside that area (and in its excluded holes)
filled grey, for training and for every prediction; masks come back in
full-frame coordinates, and detections centred outside the area are dropped.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fnt.videoTracking.mtt_roi import FILL, ROI, model_roi, project_roi  # noqa: E402

# A 200 x 100 frame: the arena x 20-179, y 10-89; a box lid excluded at x 140-180, y 10-50.
ARENA = [(20, 10), (179, 10), (179, 89), (20, 89)]
LID = [(140, 10), (180, 10), (180, 50), (140, 50)]


def _roi():
    return ROI(include=[ARENA], exclude=[LID], frame_size=(200, 100))


def _box_mask(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_round_trip_and_scaling():
    r = _roi()
    back = ROI.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back.include == r.include and back.exclude == r.exclude
    assert back.frame_size == (200, 100)
    assert ROI.from_dict({}) is None and ROI.from_dict({"exclude": [[[0, 0], [1, 1]]]}) is None
    assert r.box(200, 100) == (20, 10, 180, 90)
    big = r.for_size(400, 200)                       # the same camera at twice the size
    assert big.include[0][1] == (358, 20) and big.exclude[0][0] == (280, 20)
    assert r.box(400, 200) == (40, 20, 359, 179)
    old = ROI.from_dict({"crop": [20, 10, 180, 90], "frame_size": [200, 100]})   # first version
    assert old.box(200, 100) == (20, 10, 180, 90)


def test_an_outline_not_a_box():
    tri = ROI(include=[[(0, 0), (199, 0), (0, 99)]], frame_size=(200, 100))
    out, origin = tri.apply(np.full((100, 200, 3), 255, np.uint8))
    assert origin == (0, 0) and out.shape == (100, 200, 3)
    assert (out[5, 5] == 255).all() and (out[95, 190] == FILL).all()
    assert tri.allows(10, 10, 100, 200) and not tri.allows(190, 95, 100, 200)


def test_apply_cuts_and_fills():
    img = np.full((100, 200, 3), 255, np.uint8)
    out, origin = _roi().apply(img)
    assert origin == (20, 10) and out.shape == (80, 160, 3)
    assert (out[5, 140] == FILL).all()               # under the lid: grey
    assert (out[60, 140] == 255).all()               # the arena below it: untouched


def test_to_frame_maps_back_and_drops_detections_under_the_lid():
    r = _roi()
    on_crop = np.stack([
        _box_mask(80, 160, 10, 50, 30, 70),          # a mouse in the arena
        _box_mask(80, 160, 125, 5, 155, 35),         # one seen through the lid
        _box_mask(80, 160, 100, 25, 130, 45),        # one half under the lid edge: kept, cut
    ])
    full, kept = r.to_frame(on_crop, 100, 200)
    assert kept == [0, 2]
    ys, xs = np.nonzero(full[0])
    assert (xs.min(), ys.min()) == (30, 60)          # crop origin added back
    assert not full[2][10:50, 140:180].any()         # the part under the lid cut off
    assert full[2][51:55, 140:150].all()             # the part below it kept (the edge row is the lid's)


def test_export_cuts_frames_and_labels(tmp_path):
    from fnt.videoTracking.yolo_training import convert_coco_to_yolo
    imgs = tmp_path / "frames"
    imgs.mkdir()
    for k in range(3):
        cv2.imwrite(str(imgs / f"f{k}.png"), np.full((100, 200, 3), 200, np.uint8))
    poly = lambda x0, y0, x1, y1: [[x0, y0, x1, y0, x1, y1, x0, y1]]   # noqa: E731
    coco = {"images": [{"id": k + 1, "file_name": f"f{k}.png", "width": 200, "height": 100}
                       for k in range(3)],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "segmentation": poly(40, 60, 60, 80)},
                {"id": 2, "image_id": 1, "category_id": 1, "segmentation": poly(150, 15, 170, 40)},
                {"id": 3, "image_id": 2, "category_id": 1, "segmentation": poly(40, 60, 60, 80)},
                {"id": 4, "image_id": 3, "category_id": 1, "segmentation": poly(40, 60, 60, 80)}],
            "categories": [{"id": 1, "name": "mouse"}]}
    cj = tmp_path / "c.json"
    cj.write_text(json.dumps(coco))
    out = tmp_path / "yolo"
    convert_coco_to_yolo(str(cj), str(imgs), str(out), 0.34, roi=_roi().to_dict())
    lbl = next((out / "labels" / s / "f0.txt") for s in ("train", "val")
               if (out / "labels" / s / "f0.txt").exists())
    lines = lbl.read_text().splitlines()
    assert len(lines) == 1                            # the mouse under the lid dropped
    xy = np.array(lines[0].split()[1:], float).reshape(-1, 2)
    # (40..60, 60..80) on the frame = (20..40, 50..70) on the 160 x 80 cut
    assert np.allclose([xy[:, 0].min() * 160, xy[:, 1].min() * 80], [20, 50], atol=1)
    img = next((out / "images" / s / "f0.png") for s in ("train", "val")
               if (out / "images" / s / "f0.png").exists())
    cut = cv2.imread(str(img))
    assert cut.shape[:2] == (80, 160) and (cut[5, 140] == FILL).all()


class _T:
    """Just enough of a torch tensor: .bool().cpu().numpy()."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def bool(self):
        return _T(self.a.astype(bool))

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _FakeYOLO:
    """Finds the white squares of the frame it is given (at that frame's size)."""

    def __init__(self):
        self.seen = []

    def predict(self, img, **kw):
        self.seen.append(img.shape[:2])
        h, w = img.shape[:2]
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(
            (img[:, :, 0] == 255).astype(np.uint8))
        masks = np.stack([lbl == i for i in range(1, n)]) if n > 1 else np.zeros((0, h, w), bool)
        boxes = np.array([[s[0], s[1], s[0] + s[2], s[1] + s[3]] for s in stats[1:]], np.float32)

        class R:
            pass

        class Masks:
            data = _T(masks.astype(np.float32))

            def __len__(self):
                return len(masks)
        r = R()
        r.boxes = R()
        r.boxes.xyxy = _T(boxes.reshape(-1, 4))
        r.boxes.conf = _T(np.full(len(boxes), 0.9, np.float32))
        r.boxes.cls = _T(np.zeros(len(boxes)))
        r.masks = Masks() if len(masks) else None
        return [r]


def _frame():
    img = np.zeros((100, 200, 3), np.uint8)
    img[60:80, 40:60] = 255          # a mouse in the arena
    img[15:40, 150:170] = 255        # one seen through the lid
    img[85:95, 5:15] = 255           # something outside the arena
    return img


def test_yolo_inference_predicts_on_the_cut_frame():
    from fnt.videoTracking.yolo_inference import YOLOInference
    eng = YOLOInference("unused", device="cpu", use_masks=True)
    eng.model = _FakeYOLO()
    eng.roi = _roi()
    det = eng.predict(_frame()[:, :, ::-1].copy(), confidence_threshold=0.25)
    assert eng.model.seen == [(80, 160)]
    assert len(det["scores"]) == 1 and det["masks"].shape == (1, 100, 200)
    assert np.allclose(det["boxes"][0], [40, 60, 60, 80])


def test_batch_sidecar_in_full_frame_coordinates(tmp_path):
    import h5py
    from fnt.videoTracking.mask_tracker_batch import infer_video
    video = str(tmp_path / "v.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 10, (200, 100))
    for _ in range(3):
        vw.write(_frame())
    vw.release()
    m = _FakeYOLO()
    # MJPG blurs edges: threshold the fake on near-white instead of exactly 255
    orig = m.predict
    m.predict = lambda img, **kw: orig(np.where(img > 200, 255, 0).astype(np.uint8), **kw)
    infer_video(m, video, "fake", 1, 0.25, 640, roi=_roi())
    with h5py.File(str(tmp_path / "v_masks.h5")) as f:
        cx, cy = f["det/cx"][:], f["det/cy"][:]
        assert len(json.loads(f.attrs["roi"])["include"]) == 1
    assert len(cx) == 3 and np.allclose(cx, 49.5, atol=1) and np.allclose(cy, 69.5, atol=1)


def test_model_and_project_roi_files(tmp_path):
    (tmp_path / "project_config.json").write_text(json.dumps({"roi": _roi().to_dict()}))
    assert project_roi(str(tmp_path)).box(200, 100) == (20, 10, 180, 90)
    run = tmp_path / "models" / "r1"
    run.mkdir(parents=True)
    (run / "training_config.json").write_text(json.dumps({"imgsz": 640, "roi": None}))
    assert model_roi(str(run)) is None
    (run / "training_config.json").write_text(json.dumps({"roi": _roi().to_dict()}))
    assert model_roi(str(run / "weights_best.pt")).exclude == [LID]


def _click(canvas, x, y, button=None, double=False):
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    wx, wy = canvas.to_widget(x, y)
    b = button or Qt.LeftButton
    kinds = (QEvent.MouseButtonDblClick,) if double else (QEvent.MouseButtonPress,
                                                          QEvent.MouseButtonRelease)
    for kind in kinds:
        ev = QMouseEvent(kind, QPointF(wx, wy), b, b, Qt.NoModifier)
        {QEvent.MouseButtonPress: canvas.mousePressEvent,
         QEvent.MouseButtonRelease: canvas.mouseReleaseEvent,
         QEvent.MouseButtonDblClick: canvas.mouseDoubleClickEvent}[kind](ev)


def test_the_editor_draws_outlines_and_moves_corners():
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from fnt.videoTracking.mtt_roi_editor import ROIEditorDialog
    dlg = ROIEditorDialog(_frame())
    c = dlg.canvas
    c.resize(800, 400)
    for x, y in ARENA:                                # include: four corners, then the first again
        _click(c, x, y)
    _click(c, *ARENA[0])
    assert len(c.areas["include"]) == 1 and not c._points
    dlg.rb_exclude.setChecked(True)
    for x, y in LID[:3]:
        _click(c, x, y)
    _click(c, *LID[3])
    _click(c, *LID[3], double=True)                   # double-click closes
    assert len(c.areas["exclude"]) == 1 and len(c.areas["exclude"][0]) == 4
    assert dlg.list_areas.count() == 2
    # drag the arena's bottom-right corner
    wx, wy = c.to_widget(179, 89)
    tx, ty = c.to_widget(170, 80)
    c.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, QPointF(wx, wy), Qt.LeftButton,
                                  Qt.LeftButton, Qt.NoModifier))
    c.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, QPointF(tx, ty), Qt.NoButton,
                                 Qt.LeftButton, Qt.NoModifier))
    c.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, QPointF(tx, ty), Qt.LeftButton,
                                    Qt.NoButton, Qt.NoModifier))
    x, y = c.areas["include"][0][2]
    assert abs(x - 170) < 1 and abs(y - 80) < 1 and not c._points
    dlg.chk_model_view.setChecked(True)
    dlg.grab()                                        # paints both views without error
    dlg.chk_model_view.setChecked(False)
    dlg.grab()
    r = dlg.roi()
    assert r.frame_size == (200, 100) and len(r.include) == 1 and len(r.exclude) == 1


def test_mask_tracker_draws_and_saves_the_roi(monkeypatch):
    from PyQt5.QtWidgets import QApplication, QDialog
    app = QApplication.instance() or QApplication([])  # noqa: F841
    import fnt.videoTracking.mask_tracker_gui as G
    import fnt.videoTracking.mtt_roi_editor as E
    d = tempfile.mkdtemp(prefix="mtt_roi_")
    os.makedirs(os.path.join(d, "training_frames"))
    with open(os.path.join(d, "project_config.json"), "w") as fh:
        json.dump({"project_dir": d, "video_paths": [], "categories": ["mouse"],
                   "source_videos": []}, fh)
    w = G.MaskTrackerWindow()
    w._launched = True
    w._load_project(os.path.join(d, "project_config.json"))
    assert w.btn_roi.text() == "ROI (none)" and w.preview.roi is None
    w.preview.set_frame(_frame()[:, :, ::-1].copy())

    def drawn(dlg):
        c = dlg.canvas
        c.areas["include"].append(list(ARENA))
        c.set_mode("exclude")
        c._points = list(LID)
        c.close_outline()
        dlg.grab()                                   # paints without error
        return QDialog.Accepted
    monkeypatch.setattr(E.ROIEditorDialog, "exec_", drawn)
    w._edit_roi()
    with open(os.path.join(d, "project_config.json")) as fh:
        saved = json.load(fh)["roi"]
    assert len(saved["include"]) == 1 and len(saved["exclude"]) == 1
    assert saved["frame_size"] == [200, 100]
    assert w.btn_roi.text() == "ROI" and w.preview.roi is not None
    w.preview.resize(400, 200)
    w.preview.grab()
    w.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
