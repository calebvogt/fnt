"""Fusion View: the scene draws the named animals and finds the one clicked, masks decoded only inside their
boxes match the stored masks, and a correction made in the window is saved and held when the names are re-solved.

The synthetic run of test_mtt_identity (vole A anchored in box Z1, out through the doorway onto the overhead floor
beside vole B) gets Mask Tracker folders, short clips and track stores with round masks.
"""
import json
import os
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from test_mtt_identity import _setup  # noqa: E402

from fnt.videoTracking.track_store import TrackMaskWriter  # noqa: E402

FRAMES = 120            # frames given clips and masks (the run is 600)


def _disc(x, y, r=20):
    m = np.zeros((1080, 1920), bool)
    cv2.circle(m.view(np.uint8), (int(x), int(y)), r, 1, -1)
    return m


def _run(tmp_path):
    cfg = _setup(tmp_path)
    for cam in ("cam1", "cam5"):
        d = tmp_path / cam / f"{cam}_t_MaskTracker"
        d.mkdir(parents=True)
        tj = pd.read_csv(cfg["cameras"][cam]["tracks"])
        tj.to_csv(d / "trajectories.csv", index=False)
        w = cv2.VideoWriter(str(tmp_path / cam / f"{cam}_t.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (192, 108))
        for _ in range(FRAMES):
            w.write(np.full((108, 192, 3), 128, np.uint8))
        w.release()
        with TrackMaskWriter(str(d / f"{cam}_t_tracks.h5"), fps=30, frame_count=600, width=1920, height=1080) as st:
            for f, g in tj[tj.frame < FRAMES].groupby("frame"):
                st.add_frame(int(f), {int(r.object_id): dict(mask=_disc(r.x, r.y), score=0.9,
                                                            bbox=[r.x - 20, r.y - 20, r.x + 20, r.y + 20],
                                                            centroid=(r.x, r.y)) for r in g.itertuples()})
        cfg["cameras"][cam]["tracks"] = str(d)
    cfg.update(out=str(tmp_path / "names"), layout="cam1,,cam5", frame_step=1)
    p = tmp_path / "identity.json"
    json.dump(cfg, open(p, "w", encoding="utf-8"))
    from fnt.videoTracking.mtt_identity import main
    main([str(p)])
    return str(p)


def test_the_scene_draws_names_and_finds_the_animal_clicked(tmp_path):
    from fnt.videoTracking.identity_view import Overlays, Scene
    sc = Scene(_run(tmp_path))
    det = [d for d in sc.detections("cam1", 100)][0]
    full = sc.readers["cam1"].detections_for_frame(100)[0]["mask"]
    x0, y0 = det["origin"]
    m = det["mask"]
    assert np.array_equal(full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]], m) and m.sum() == full.sum()
    img = sc.draw(sc.compose({}), 100, Overlays(uwb=True, uwb_tails=True, track_tails=True, floor=True))
    assert img.shape == (1080, 1920, 3) and img.any()
    x, y = sc.to_canvas("cam1", [det["centroid"]])[0]
    hit = sc.hit(100, x, y)
    assert (hit["cam"], hit["obj"], hit["animal"]) == ("cam1", 1, "A")
    d = sc.uwb_distances("cam1", 1, *hit["tracklet"])
    assert d["A"] < d["B"]


def test_a_correction_in_the_window_is_saved_and_held(tmp_path):
    from PyQt5.QtWidgets import QApplication
    from fnt.videoTracking.fusion_view import ReviewWindow
    app = QApplication.instance() or QApplication([])
    w = ReviewWindow(_run(tmp_path))
    w.show_frame(100)
    for _ in range(3):
        w.step(1)
    assert w.n == 103
    w.add_correction("cam1", 1, 0, 239, "B")
    assert os.path.exists(os.path.join(tmp_path, "names", "corrections.csv"))
    w.resolve()
    w.solver.wait()
    app.processEvents()
    assert w.scene.names_at.get("cam1", 1, 100) == ("B", False, True)
    assert "correction" in w.targets and w.targets["correction"] == [0]
    # zoomed in on one camera: drawn at 4K, only that camera decoded; moving onto the other draws it too
    sc = w.scene
    for cam in ("cam1", "cam5"):
        x, y, cw, ch = sc.cells[cam]
        w.view._set_view(4.0, ((x + cw / 2) / sc.W, (y + ch / 2) / sc.H))
        assert sc.size == "4k" and set(w._cell_sizes()) == {cam} and cam in w._shown
    w.view.fit()
    assert set(w._cell_sizes()) == set(sc.cells)
    w.dirty = False
    w.close()
    shutil.rmtree(tmp_path / "names", ignore_errors=True)


def test_zooming_keeps_the_point_under_the_pointer_and_clicks_land_on_the_canvas():
    from PyQt5.QtCore import QPoint, QPointF
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QApplication
    from fnt.videoTracking.fusion_view import GridView
    app = QApplication.instance() or QApplication([])  # noqa: F841
    v = GridView()
    v.resize(960, 540)
    v.set_image(QImage(1920, 1080, QImage.Format_RGB888))
    p = QPointF(700, 120)
    before = v._to_canvas(p)
    v.zoom_by(3.0, p)
    assert abs(v.zoom - 3.0) < 1e-9
    after = v._to_canvas(p)
    assert np.allclose(before, after, atol=1e-6)        # the point under the pointer stays put
    v.pan_pixels(-100, 0)
    assert v._to_canvas(p)[0] > after[0]                 # dragged left: further right on the canvas
    got = []
    v.clicked.connect(lambda x, y, g: got.append((x, y)))
    v.mouseReleaseEvent(type("E", (), {"pos": lambda s: QPoint(480, 270), "button": lambda s: 1,
                                       "globalPos": lambda s: QPoint(0, 0)})())
    v._press = QPoint(480, 270)
    from PyQt5.QtCore import Qt
    v.mouseReleaseEvent(type("E", (), {"pos": lambda s: QPoint(480, 270), "button": lambda s: Qt.LeftButton,
                                       "globalPos": lambda s: QPoint(0, 0)})())
    assert len(got) == 1 and np.allclose(got[0], v._to_canvas(QPointF(480, 270)))
    v.fit()
    assert v.zoom == 1.0 and np.allclose(v._to_canvas(QPointF(480, 270)), (960, 540))



class _Counting:
    """A VideoCapture that notes what it was asked to do."""

    def __init__(self, path):
        self.cap, self.calls = cv2.VideoCapture(path), []

    def set(self, *a):
        self.calls.append("seek")
        return self.cap.set(*a)

    def read(self):
        self.calls.append("read")
        return self.cap.read()

    def grab(self):
        self.calls.append("grab")
        return self.cap.grab()

    def release(self):
        self.cap.release()


def test_the_player_reads_the_right_frames_from_its_cache_and_after_a_step_back(tmp_path):
    """Frames from the cache, from a step back (decoded with the frames before it) and at a new cell size are the
    frames a plain seek gives; a redraw, further steps back and a zoom decode nothing."""
    from fnt.videoTracking.fusion_view import Player
    p = str(tmp_path / "count.avi")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 90))
    for i in range(200):
        fr = np.zeros((90, 160, 3), np.uint8)
        fr[:, : i % 160] = 255                       # the frame number, as a bar's length
        fr[:, :, 2] = i
        w.write(fr)
    w.release()

    def truth(n, size):
        cap = cv2.VideoCapture(p)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, fr = cap.read()
        cap.release()
        return cv2.resize(fr, size, interpolation=cv2.INTER_AREA)

    pl = Player({"cam1": p}, hold=3)
    pl.caps["cam1"].release()
    cap = pl.caps["cam1"] = _Counting(p)
    small, big = {"cam1": (80, 45)}, {"cam1": (120, 68)}
    for n in (0, 3, 6, 150):                                          # in order, then a jump
        assert np.array_equal(pl.read(n, small)["cam1"], truth(n, (80, 45)))
    cap.calls.clear()
    assert np.array_equal(pl.read(150, small)["cam1"], truth(150, (80, 45)))      # a redraw
    assert np.array_equal(pl.read(150, big)["cam1"], truth(150, (120, 68)))       # zoomed: a resize
    assert not cap.calls
    assert np.array_equal(pl.read(147, small)["cam1"], truth(147, (80, 45)))      # a step back
    assert cap.calls.count("seek") == 1
    cap.calls.clear()
    for n in range(144, 147 - 3 * Player.FILL - 1, -3):                           # further steps back
        assert np.array_equal(pl.read(n, small)["cam1"], truth(n, (80, 45)))
    assert np.array_equal(pl.read(141, big)["cam1"], truth(141, (120, 68)))       # and zooming there
    assert not cap.calls
    pl.prefetch(30, small)                                                        # decoded ahead
    assert np.array_equal(pl.read(30, small)["cam1"], truth(30, (80, 45)))
    pl.close()
