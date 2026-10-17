"""Empty frames and Space in the Mask Tracker.

A frame the reviewer marks empty (E, or the Empty Frame button) is reviewed and
holds no object: it trains as a background image, Enter and post-training
inference skip it, and marking it drops its pending masks. The training export
holds only reviewed images -- an image whose masks are all still pending used
to go in with no annotation and train as "no mouse here".

Space walks the current video's queued frames and, past the last one, carries
on to the next frame in the queue instead of wrapping back within the video.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.videoTracking.mask_tracker_gui as G  # noqa: E402
from fnt.videoTracking.mask_tracker_annotator import COCOAnnotationManager  # noqa: E402
from fnt.videoTracking.yolo_training import convert_coco_to_yolo  # noqa: E402


def square_mask(h=48, w=64):
    m = np.zeros((h, w), bool)
    m[10:30, 10:40] = True
    return m


def manager():
    c = COCOAnnotationManager()
    cat = c.add_category("mouse")
    return c, cat


def test_marking_empty_drops_pending_and_refuses_accepted():
    c, cat = manager()
    pend = c.get_or_add_image("pend.png", 64, 48)
    c.add_annotation_from_mask(pend, cat, square_mask(), inferred=True, score=0.4)
    assert c.set_image_empty("pend.png", 64, 48) == pend
    assert c.is_image_empty("pend.png")
    assert c.get_annotations_for_image(pend) == []          # pending mask dropped

    acc = c.get_or_add_image("acc.png", 64, 48)
    c.add_annotation_from_mask(acc, cat, square_mask())
    assert c.set_image_empty("acc.png", 64, 48) == -1       # has an accepted mask
    assert not c.is_image_empty("acc.png")

    c.set_image_empty("pend.png", 64, 48, empty=False)
    assert not c.is_image_empty("pend.png")


def test_a_failed_save_leaves_the_previous_file_whole(monkeypatch):
    """Labels are saved on every action; a save that dies mid-write must not
    truncate annotations.json."""
    c, cat = manager()
    c.add_annotation_from_mask(c.get_or_add_image("a.png", 64, 48), cat, square_mask())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "annotations.json")
        c.export(path)
        before = open(path).read()
        c.add_annotation_from_mask(c.get_or_add_image("b.png", 64, 48), cat, square_mask())

        def dies(*a, **k):
            raise OSError("network drive dropped")
        monkeypatch.setattr(json, "dump", dies)
        with pytest.raises(OSError):
            c.export(path)
        assert open(path).read() == before
        monkeypatch.undo()
        c.export(path)
        assert len(json.load(open(path))["images"]) == 2
        assert os.listdir(d) == ["annotations.json"]


def test_a_mask_drawn_later_wins_over_the_empty_flag():
    c, cat = manager()
    img = c.set_image_empty("f.png", 64, 48)
    c.add_annotation_from_mask(img, cat, square_mask())
    assert not c.is_image_empty("f.png")
    assert c.count_empty_images() == 0


def test_training_export_holds_only_reviewed_images(tmp_path):
    c, cat = manager()
    acc = c.get_or_add_image("acc.png", 64, 48)
    c.add_annotation_from_mask(acc, cat, square_mask())
    pend = c.get_or_add_image("pend.png", 64, 48)
    c.add_annotation_from_mask(pend, cat, square_mask(), inferred=True, score=0.9)
    c.set_image_empty("empty.png", 64, 48)
    out = tmp_path / "train.json"
    c.export(str(out), exclude_inferred=True)
    d = json.load(open(out))
    assert sorted(i["file_name"] for i in d["images"]) == ["acc.png", "empty.png"]
    assert all(not a.get("inferred") for a in d["annotations"])
    # The full (auto-save) export still keeps every image and pending mask.
    full = tmp_path / "all.json"
    c.export(str(full))
    assert len(json.load(open(full))["images"]) == 3

    # YOLO: the empty frame becomes a background image with an empty label file.
    imgs = tmp_path / "frames"
    imgs.mkdir()
    for name in ("acc.png", "pend.png", "empty.png"):
        cv2.imwrite(str(imgs / name), np.zeros((48, 64, 3), np.uint8))
    yolo = tmp_path / "yolo"
    convert_coco_to_yolo(str(out), str(imgs), str(yolo), val_fraction=0.5)
    labels = {p.name: p.read_text() for p in yolo.rglob("labels/*/*.txt")}
    assert set(labels) == {"acc.txt", "empty.txt"}
    assert labels["empty.txt"] == "" and labels["acc.txt"].startswith("0 ")


def test_a_mask_in_pieces_is_one_yolo_instance(tmp_path):
    """A mouse whose tail was cut off by erasing is one mask in two pieces;
    the YOLO labels made each piece its own instance (an extra "mouse")."""
    c, cat = manager()
    m = square_mask()
    m[35:40, 45:60] = True                        # a separate tail piece
    img = c.get_or_add_image("pieces.png", 64, 48)
    c.add_annotation_from_mask(img, cat, m)
    out = tmp_path / "train.json"
    c.export(str(out), exclude_inferred=True)
    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "pieces.png"), np.zeros((48, 64, 3), np.uint8))
    convert_coco_to_yolo(str(out), str(frames), str(tmp_path / "yolo"), val_fraction=0.5)
    label = next((tmp_path / "yolo").rglob("labels/*/pieces.txt")).read_text().strip()
    assert len(label.splitlines()) == 1           # one mouse
    xy = np.array(label.split()[1:], float).reshape(-1, 2) * (64, 48)
    assert xy[:, 0].max() > 55 and xy[:, 1].max() > 37   # the tail piece is in the outline


# ----------------------------------------------------------------------
# the window
# ----------------------------------------------------------------------
FRAMES = ["CAM13_a_frame_000001.png", "CAM13_a_frame_000002.png",
          "CAM14_b_frame_000003.png", "CAM14_b_frame_000004.png"]
_APP = None


def window():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    root = tempfile.mkdtemp(prefix="mtt_empty_")
    proj = os.path.join(root, "proj")
    for sub in ("training_frames", "annotations", "models"):
        os.makedirs(os.path.join(proj, sub))
    json.dump({"video_paths": [], "source_videos": [], "categories": ["mouse"],
               "sam2_model_path": None}, open(os.path.join(proj, "project_config.json"), "w"))
    for name in FRAMES:
        cv2.imwrite(os.path.join(proj, "training_frames", name), np.zeros((48, 64, 3), np.uint8))
    w = G.MaskTrackerWindow()
    w._launched = True
    w._load_project(os.path.join(proj, "project_config.json"))
    if "mouse" not in w._categories:
        w._coco.add_category("mouse")
        w._categories.append("mouse")
    return w


def row_text(w, name):
    for i in range(w.frame_list.topLevelItemCount()):
        it = w.frame_list.topLevelItem(i)
        if it.text(0) == name:
            return it.text(1)
    raise KeyError(name)


def test_n_marks_empty_advances_and_unmarks():
    w = window()
    assert w.windowTitle().startswith("Mask Tracker")
    w._select_frame_row(0)
    assert w.current_frame_idx == 0
    w._toggle_empty_frame()                       # what E and the button call
    assert w._coco.is_image_empty(FRAMES[0])
    assert row_text(w, FRAMES[0]) == "empty ✓"
    assert w.current_frame_idx == 1               # moved on to the next unreviewed frame

    w._select_frame_row(0)
    assert w.btn_empty_frame.isChecked()
    w._toggle_empty_frame()                       # unmark
    assert not w._coco.is_image_empty(FRAMES[0])
    assert not w.btn_empty_frame.isChecked()
    assert w.current_frame_idx == 0               # unmarking does not move
    w.close()


def test_enter_and_post_inference_skip_empty_frames(monkeypatch):
    w = window()
    w._select_frame_row(1)
    w._toggle_empty_frame()                       # frame 1 empty
    w._select_frame_row(0)
    w._next_frame()                               # Enter
    assert w.current_frame_idx == 2               # skips the empty frame 1

    captured = {}

    class FakeWorker:
        class _Sig:
            def connect(self, *_):
                pass

        def __init__(self, model_dir, frame_paths, **kw):
            captured["paths"] = [os.path.basename(p) for p in frame_paths]
            self.progress = self.frame_result = self.finished = self.error = self._Sig()

        def start(self):
            pass
    monkeypatch.setattr(G, "PostTrainInferenceWorker", FakeWorker)
    w._run_post_train_inference("model_dir")
    assert FRAMES[1] not in captured["paths"] and len(captured["paths"]) == 3
    w.close()


def test_space_continues_through_the_queue_instead_of_wrapping(monkeypatch):
    w = window()
    w._select_frame_row(1)                        # second (last) queued frame of video a
    monkeypatch.setattr(w, "_scrubber_source", lambda: ("a.mp4", 2))
    monkeypatch.setattr(w._scrubber, "next_mark", lambda pos: 1)   # would wrap to frame 1
    monkeypatch.setattr(w._scrubber, "prev_mark", lambda pos: 1)
    w._scrubber_rows = {1: 0, 2: 1}
    w._on_timeline_advance(1)
    assert w.current_frame_idx == 2               # first frame of the next video, not frame 0

    w._select_frame_row(3)
    w._step_queue(1)                              # end of the queue wraps to the start
    assert w.current_frame_idx == 0
    w._step_queue(-1)
    assert w.current_frame_idx == 3
    w.close()


def test_s_skips_to_the_next_frame_still_to_label():
    from PyQt5.QtCore import Qt
    w = window()
    w._select_frame_row(0)
    w._toggle_empty_frame()                         # frame 0 empty; now on frame 1
    w._select_frame_row(1)
    w.preview.add_sam_click(5, 5)                   # an unsaved SAM click on frame 1
    w.preview._drawing_active = True
    key(w.btn_skip_frame, Qt.Key_S, "s")            # S with a button focused
    assert w.current_frame_idx == 2                 # next frame still to label
    assert w.preview._ai_objects == []              # unsaved SAM work dropped
    assert not w._coco.is_image_empty(FRAMES[1])    # nothing recorded for the skipped frame
    w._select_frame_row(3)
    w._skip_frame()                                 # past the last one: back to the first
    assert w.current_frame_idx == 1                 # frame 0 is empty (done), so frame 1
    w.close()


def key(widget, k, text=""):
    from PyQt5.QtCore import QEvent, Qt  # noqa: F401
    from PyQt5.QtGui import QKeyEvent
    QApplication.sendEvent(widget, QKeyEvent(QEvent.KeyPress, k, Qt.NoModifier, text))


def test_labeling_keys_work_after_clicking_a_button_or_the_list(monkeypatch):
    """Space after clicking Mark as Empty Frame used to press that button again
    (unmarking the frame) instead of moving on; keys go to the preview now."""
    from PyQt5.QtCore import Qt
    w = window()
    w._select_frame_row(0)
    w._toggle_empty_frame()                         # mark frame 0 empty -> moves to 1
    w._select_frame_row(0)
    monkeypatch.setattr(w, "_scrubber_source", lambda: None)   # no timeline: Space walks the queue
    key(w.btn_empty_frame, Qt.Key_Space, " ")       # Space with the button focused
    assert w._coco.is_image_empty(FRAMES[0])        # not unmarked
    assert w.current_frame_idx != 0                 # moved on

    w._select_frame_row(2)
    key(w.frame_list, Qt.Key_E, "e")                # E with the frame list focused
    assert w._coco.is_image_empty(FRAMES[2])

    # N / B step through the queue row by row, from anywhere in the window.
    w._select_frame_row(1)
    key(w.frame_list, Qt.Key_N, "n")
    assert w.current_frame_idx == 2
    key(w.btn_empty_frame, Qt.Key_B, "b")
    key(w.btn_empty_frame, Qt.Key_B, "b")
    assert w.current_frame_idx == 0
    assert not w._coco.is_image_empty(FRAMES[1])    # N no longer marks empty

    # Text entry keeps its keys.
    before = w.spin_batch.value()
    key(w.spin_batch, Qt.Key_E, "e")
    assert w.spin_batch.value() == before
    w.close()


def test_eraser_on_a_saved_mask_edits_it_and_enter_esc_turn_the_tools_off():
    """Clicking a saved mask with the Eraser did nothing (it had to be brought
    back through a menu first); Enter left Paint on; Esc left the Eraser button
    lit after the tool went off."""
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    w = window()
    w._select_frame_row(0)
    pv = w.preview
    pv.resize(400, 300)
    w._set_object_tool("paint")
    pv.add_pending_mask(square_mask())
    key(pv, Qt.Key_Return)                        # save it
    assert len(pv.annotations) == 1 and not pv._ai_objects
    assert pv.drawing_mode == "navigate" and not w.btn_paint.isChecked()

    w._set_object_tool("erase")
    assert w.btn_erase.isChecked()
    x, y = pv._img_to_widget(25, 20)              # on the saved mask
    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
        buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
        QApplication.sendEvent(pv, QMouseEvent(kind, QPointF(x, y), Qt.LeftButton, buttons, Qt.NoModifier))
    assert not pv.annotations and len(pv._ai_objects) == 1      # back as a pending object
    assert int(pv._ai_objects[0]["mask"].sum()) < 600           # and erased
    key(pv, Qt.Key_Escape)                        # drops the pending object, tool stays
    assert pv.drawing_mode == "erase" and w.btn_erase.isChecked()
    key(pv, Qt.Key_Escape)                        # tool off, button too
    assert pv.drawing_mode == "navigate" and not w.btn_erase.isChecked()
    w.close()


def test_a_number_key_saves_the_pending_mask_as_that_class():
    """Click an object, press its class number: saved as that class (user request,
    objects round 1); with nothing pending the key only switches the class."""
    from PyQt5.QtCore import Qt
    w = window()
    for c in ("running wheel", "hut"):
        w._coco.add_category(c)
        w._categories.append(c)
    w._refresh_active_class_combo()
    assert "3  hut" in w.lbl_class_keys.text()
    w._select_frame_row(0)
    pv = w.preview
    pv.resize(400, 300)
    w._set_object_tool("paint")
    pv.add_pending_mask(square_mask())
    key(pv, Qt.Key_3)                             # saves it as "hut"
    assert len(pv.annotations) == 1 and pv.annotations[0].category == "hut"
    assert not pv._ai_objects
    key(pv, Qt.Key_2)                             # nothing pending: just switches
    assert w.combo_active_class.currentText() == "running wheel" and len(pv.annotations) == 1
    w.close()


def _multiclass_window():
    w = window()
    for c in ("running wheel", "hut"):
        w._coco.add_category(c)
        w._categories.append(c)
    w._refresh_active_class_combo()
    w._select_frame_row(0)
    w.preview.resize(400, 300)
    return w


def test_enter_asks_the_class_when_there_are_several(monkeypatch):
    """Objects round 1: Enter on a pending mask opens the class keypad; the pick is
    saved and the tool stays on for the next object. Cancel keeps the mask pending."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QDialog

    class Pick:
        answer = "hut"

        def __init__(self, cats, current, parent=None):
            self.selected = None

        def exec_(self):
            self.selected = Pick.answer
            return QDialog.Accepted if Pick.answer else QDialog.Rejected

    monkeypatch.setattr(G, "ClassPickerDialog", Pick)
    w = _multiclass_window()
    pv = w.preview
    w._set_object_tool("paint")
    pv.add_pending_mask(square_mask())
    key(pv, Qt.Key_Return)
    assert [a.category for a in pv.annotations] == ["hut"]
    assert pv.drawing_mode == "paint" and w.btn_paint.isChecked()        # still on
    Pick.answer = None                                                    # Esc in the keypad
    pv.add_pending_mask(square_mask())
    key(pv, Qt.Key_Return)
    assert len(pv.annotations) == 1 and len(pv._ai_objects) == 1         # still pending
    w.close()


def test_the_class_keypad_keys():
    from PyQt5.QtCore import QEvent, Qt
    from PyQt5.QtGui import QKeyEvent
    d = G.ClassPickerDialog(["mouse", "running wheel", "hut"], "mouse")
    d.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_3, Qt.NoModifier, "3"))
    assert d.selected == "hut"
    d = G.ClassPickerDialog(["mouse", "running wheel", "hut"], "running wheel")
    d.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Return, Qt.NoModifier, ""))
    assert d.selected == "running wheel"                                 # Enter keeps it


def test_one_class_still_saves_silently_and_turns_the_tool_off():
    from PyQt5.QtCore import Qt
    w = window()
    w._select_frame_row(0)
    pv = w.preview
    pv.resize(400, 300)
    w._set_object_tool("paint")
    pv.add_pending_mask(square_mask())
    key(pv, Qt.Key_Return)
    assert [a.category for a in pv.annotations] == ["mouse"] and pv.drawing_mode == "navigate"
    w.close()


def test_train_only_narrows_the_training_export(tmp_path):
    """New classes labelled on a subset: --train-only keeps the other (mouse-only)
    frames out of training, so they don't teach the new objects as background."""
    c, cat = manager()
    for name in ("a.png", "b.png", "c.png"):
        img = c.get_or_add_image(name, 64, 48)
        c.add_annotation_from_mask(img, cat, square_mask())
    out = tmp_path / "train.json"
    c.export(str(out), exclude_inferred=True, only={"a.png", "c.png"})
    d = json.load(open(out))
    assert sorted(i["file_name"] for i in d["images"]) == ["a.png", "c.png"]
    assert len(d["annotations"]) == 2
    lst = tmp_path / "frames.txt"
    lst.write_text(chr(10).join(["a.png", "# note", "sub/c.png"]))
    assert G.read_frame_names([str(lst), "x/b.png"]) == {"a.png", "b.png", "c.png"}


def test_review_accepts_rejects_and_advances_as_in_mad():
    """A accepts the prediction under review, R rejects it, Shift+A accepts the
    frame's rest; a frame with none left opens the next with predictions."""
    from PyQt5.QtCore import Qt
    w = window()
    cat = w._coco.categories[0]["id"] if w._coco.categories else w._coco.add_category("mouse")

    def predict(name, *boxes):
        img = w._coco.get_or_add_image(name, 64, 48)
        for (x0, y0, x1, y1) in boxes:
            m = np.zeros((48, 64), bool)
            m[y0:y1, x0:x1] = True
            w._coco.add_annotation_from_mask(img, cat, m, inferred=True, score=0.7)
    predict(FRAMES[0], (5, 5, 20, 20), (30, 5, 45, 20), (5, 25, 20, 40))
    predict(FRAMES[2], (10, 10, 30, 30))
    w._select_frame_row(0)
    pv = w.preview
    assert pv.current_prediction() == 0
    assert "1 of 3" in w.lbl_review.text() and "2 frame(s) to review" in w.lbl_review.text()
    key(pv, Qt.Key_A, "a")                        # accept the first
    assert not pv.annotations[0].inferred and pv.current_prediction() == 1
    key(pv, Qt.Key_R, "r")                        # reject the second: kept, marked rejected
    assert len(pv.annotations) == 3 and pv.annotations[1].rejected
    assert pv.current_prediction() == 2
    key(w.btn_accept_pred, Qt.Key_R, "r")         # R with a button focused
    assert w.current_frame_idx == 2               # nothing left on frame 0: next to review
    assert pv.current_prediction() == 0
    QApplication.sendEvent(pv, __import__("PyQt5.QtGui", fromlist=["QKeyEvent"]).QKeyEvent(
        __import__("PyQt5.QtCore", fromlist=["QEvent"]).QEvent.KeyPress, Qt.Key_A,
        Qt.ShiftModifier, "A"))                   # Shift+A: accept all on frame 2
    assert not any(a.inferred for a in pv.annotations)
    assert w.lbl_review.text() == "No predictions to review"
    w.close()


def test_a_sam_click_on_a_prediction_replaces_it():
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    w = window()
    cat = w._coco.categories[0]["id"]
    img = w._coco.get_or_add_image(FRAMES[0], 64, 48)
    m = np.zeros((48, 64), bool)
    m[10:30, 10:40] = True
    w._coco.add_annotation_from_mask(img, cat, m, inferred=True, score=0.3)
    w._select_frame_row(0)
    pv = w.preview
    pv.resize(400, 300)
    pv.drawing_mode = "ai"
    x, y = pv._img_to_widget(25, 20)
    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
        buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
        QApplication.sendEvent(pv, QMouseEvent(kind, QPointF(x, y), Qt.LeftButton, buttons, Qt.NoModifier))
    assert not pv.annotations                     # the prediction is gone
    assert len(pv._ai_objects) == 1               # a SAM object from the click
    w.close()


def test_overlapping_predictions_keep_the_best_and_labels_pick_and_d_rejects():
    """The first zone model stacked masks on masks; each prediction now carries
    its own score in its label, a click on the label picks it, D rejects it."""
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from fnt.videoTracking.mask_tracker_annotator import mask_to_rle
    a = np.zeros((48, 64), bool)
    a[10:30, 10:40] = True
    b = np.zeros((48, 64), bool)
    b[12:30, 12:40] = True                         # mostly inside a
    c = np.zeros((48, 64), bool)
    c[35:45, 45:60] = True
    dets = [{"segmentation": mask_to_rle(m), "score": sc} for m, sc in ((b, 0.3), (a, 0.6), (c, 0.4))]
    kept = G._drop_overlapping(dets, (48, 64))
    assert [d["score"] for d in kept] == [0.6, 0.4]

    w = window()
    cat = w._coco.categories[0]["id"]
    img = w._coco.get_or_add_image(FRAMES[0], 64, 48)
    w._coco.add_annotation_from_mask(img, cat, a, inferred=True, score=0.6)
    w._coco.add_annotation_from_mask(img, cat, c, inferred=True, score=0.4)
    w._select_frame_row(0)
    pv = w.preview
    pv.resize(400, 300)
    pv.grab()                                     # paint: places the labels
    assert pv.annotations[1].score == 0.4
    rect = pv._label_rects[1]
    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
        buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
        QApplication.sendEvent(pv, QMouseEvent(kind, rect.center(), Qt.LeftButton, buttons, Qt.NoModifier))
    assert pv.current_prediction() == 1           # picked by its label
    key(pv, Qt.Key_D, "d")
    assert pv.annotations[1].rejected and not pv.annotations[0].rejected
    assert w._coco.get_annotation(pv.annotations[1].ann_id)["rejected"]
    w._undo()                                     # Ctrl+Z: back to pending
    assert not w._coco.get_annotation(pv.annotations[1].ann_id).get("rejected")
    w.close()


def test_rejections_survive_empty_and_new_predictions_and_specks_drop():
    """A rejected mask stays when the frame is marked empty and when the queue is
    predicted again, and a new prediction on the rejected spot is not offered."""
    from fnt.videoTracking.mask_tracker_annotator import mask_to_rle
    w = window()
    cat = w._coco.categories[0]["id"]
    a = np.zeros((48, 64), bool)
    a[10:30, 10:40] = True
    img = w._coco.get_or_add_image(FRAMES[0], 64, 48)
    rid = w._coco.add_annotation_from_mask(img, cat, a, inferred=True, score=0.5)
    w._coco.reject_annotation(rid)
    w._select_frame_row(0)
    w._on_post_infer_frame(FRAMES[0], [{"segmentation": mask_to_rle(a), "bbox": [10, 10, 30, 20],
                                        "area": 600, "category_id": 1, "score": 0.9}])
    anns = w._coco.get_annotations_for_image(img)
    assert len(anns) == 1 and anns[0]["rejected"]  # the same spot is not offered again
    w._coco.set_image_empty(FRAMES[0], 64, 48)
    assert w._coco.get_annotation(rid) is not None and w._coco.is_image_empty(FRAMES[0])
    speck = a.copy()
    speck[40:42, 50:52] = True                    # 4 px against 600
    assert int(G.drop_fragments(speck).sum()) == 600
    w.close()


def test_a_mask_in_pieces_shows_its_largest_piece():
    w = window()
    cat = w._coco.categories[0]["id"]
    m = np.zeros((48, 64), bool)
    m[2:5, 2:6] = True                            # a small piece first in scan order
    m[20:40, 20:50] = True                        # the body
    img = w._coco.get_or_add_image(FRAMES[0], 64, 48)
    w._coco.add_annotation_from_mask(img, cat, m)
    w._select_frame_row(0)
    xs = [p[0] for p in w.preview.annotations[0].points]
    assert min(xs) >= 19                          # the outline is the body's
    w.close()


def test_esc_on_an_edit_mask_keeps_erasures_and_drops_paint():
    """As in MAD: Esc after Edit Mask drops paint not yet accepted (Enter) but
    keeps what the Eraser removed; the mask is never thrown away."""
    from PyQt5.QtCore import QEvent, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    w = window()
    w._select_frame_row(0)
    pv = w.preview
    pv.resize(400, 300)
    img = w._coco._image_id_map

    def click(ix, iy):
        x, y = pv._img_to_widget(ix, iy)
        for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            b = Qt.NoButton if kind == QEvent.MouseButtonRelease else Qt.LeftButton
            QApplication.sendEvent(pv, QMouseEvent(kind, QPointF(x, y), Qt.LeftButton, b, Qt.NoModifier))

    w._set_object_tool("paint")
    pv.add_pending_mask(square_mask())            # rows 10-30, cols 10-40: 600 px
    key(pv, Qt.Key_Return)                        # saved
    w._edit_as_pending(0)                         # Edit Mask (paint tool)
    pv.brush_radius = 2
    click(50, 40)                                 # paint off the mask: dropped on Esc
    w._set_object_tool("erase")
    click(20, 20)                                 # erase a bit of it: kept
    key(pv, Qt.Key_Escape)
    anns = w._coco.get_annotations_for_image(img[FRAMES[0]])
    assert len(anns) == 1 and not pv._ai_objects
    from fnt.videoTracking.mask_tracker_annotator import segmentation_to_mask
    m = segmentation_to_mask(anns[0]["segmentation"], 48, 64).astype(bool)
    assert not m[20, 20] and m[12, 38]            # erased spot gone, rest intact
    assert not m[40, 50]                          # the paint did not stay
    assert int(m.sum()) < 600
    w.close()


def test_skip_is_recorded_until_the_frame_is_finished():
    """S on a frame counts a skip (the frames the model got too wrong to fix);
    finishing the frame ends it."""
    w = window()
    w._select_frame_row(0)
    w._skip_frame()
    assert w._coco.skip_count(FRAMES[0]) == 1
    assert w._coco.skipped_images() == [FRAMES[0]]
    w._select_frame_row(0)
    w._skip_frame()
    assert w._coco.skip_count(FRAMES[0]) == 2
    w._select_frame_row(0)
    w._toggle_empty_frame()                       # finished: no longer skipped
    assert w._coco.skip_count(FRAMES[0]) == 0 and w._coco.skipped_images() == []
    w.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
