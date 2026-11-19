"""Midline keypoints from masks: the spine of an elongated blob, its head end from a bright marker, and orientation kept
through frames."""
import numpy as np

from fnt.videoTracking import mtt_midline as M


def _blob(cx, cy, angle_deg, L=120, W=40, shape=(200, 260), marker=True):
    """An elongated ellipse (a body) with a bright disc near one end (the head marker, at +angle)."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    a = np.radians(angle_deg)
    u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
    v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
    mask = (u / (L / 2)) ** 2 + (v / (W / 2)) ** 2 <= 1
    gray = np.where(mask, 40.0, 200.0)
    if marker:
        hx, hy = cx + 0.33 * L * np.cos(a), cy + 0.33 * L * np.sin(a)
        gray[(xx - hx) ** 2 + (yy - hy) ** 2 <= 9 ** 2] = 230.0
    return mask, gray, np.array([cx + 0.5 * L * np.cos(a), cy + 0.5 * L * np.sin(a)])


def test_the_nose_is_at_the_marked_end():
    mask, gray, nose_true = _blob(130, 100, 20)
    r = M.analyse_frame(mask, gray)
    assert r is not None and r["marker"]["area"] >= M.MARKER_MIN_AREA
    r["off"] = np.zeros(2)
    o = M.orient([r], np.array([r["ev_marker"]]))
    kp = M.keypoints(r, o[0])
    assert np.hypot(kp["nose_x"] - nose_true[0], kp["nose_y"] - nose_true[1]) < 15
    assert 90 < kp["body_len"] < 130                      # about the blob's length
    assert abs(kp["bend"]) < 20                           # a straight body


def test_orientation_is_kept_when_the_marker_is_missing():
    res, ev = [], []
    for t in range(10):
        mask, gray, _ = _blob(130, 100, 20 + t, marker=(t < 3))   # the marker only in the first 3 frames
        r = M.analyse_frame(mask, gray)
        r["off"] = np.zeros(2)
        res.append(r)
        ev.append(r["ev_marker"])
    o = M.orient(res, np.array(ev))
    noses = [M.keypoints(r, oo) for r, oo in zip(res, o)]
    # the nose stays at the +angle end in every frame
    for t, kp in enumerate(noses):
        a = np.radians(20 + t)
        assert (kp["nose_x"] - 130) * np.cos(a) + (kp["nose_y"] - 100) * np.sin(a) > 30


def test_small_masks_are_skipped():
    mask = np.zeros((50, 50), bool)
    mask[20:25, 20:25] = True
    assert M.analyse_frame(mask, np.zeros((50, 50))) is None


class _Store:
    """Two animals, analysed on every 3rd frame (a frame step), with masks as a mask store gives them."""
    has_masks = True

    def __init__(self, *_):
        self.items = []
        for f in range(0, 30, 3):
            for oid, (cx, ang) in ((1, (70, 0)), (2, (190, 180))):
                mask, gray, _ = _blob(cx, 100, ang, shape=(200, 260))
                self.items.append((f, oid, mask))
        self.frame = np.array([f for f, _, _ in self.items])
        self.object_id = np.array([o for _, o, _ in self.items])
        self.bbox = np.array([[np.where(m)[1].min(), np.where(m)[0].min(), np.where(m)[1].max(), np.where(m)[0].max()]
                              for _, _, m in self.items])

    def mask_for_row(self, i):
        return self.items[i][2]


class _Cap:
    """Frames whose grey levels hold both bodies with their head markers (at each body's +angle end)."""
    def __init__(self, *_):
        self.i = 0
        g = np.full((200, 260), 200.0)
        for cx, ang in ((70, 0), (190, 180)):
            mask, gray, _ = _blob(cx, 100, ang, shape=(200, 260))
            g = np.where(mask | (gray > 220), gray, g)
        self.img = np.repeat(g[:, :, None], 3, axis=2).astype(np.uint8)

    def get(self, _):
        return 30

    def grab(self):
        self.i += 1
        return self.i <= 30

    def read(self):
        self.i += 1
        return (self.i <= 30), self.img.copy()

    def set(self, _prop, value):               # seek: the next read is frame `value`
        self.i = int(value)

    def release(self):
        pass


def test_every_animal_gets_its_own_keypoints(tmp_path, monkeypatch):
    import cv2
    import pandas as pd
    from fnt.videoTracking import track_store
    monkeypatch.setattr(track_store, "TrackMaskReader", _Store)
    monkeypatch.setattr(cv2, "VideoCapture", _Cap)
    monkeypatch.setattr(M, "store_path_for", lambda v: str(tmp_path / "v_tracks.h5"))
    out = pd.read_csv(M.midline_for_video("v.mp4", objects="all"))
    assert sorted(out.object_id.unique()) == [1, 2]
    assert sorted(out.frame.unique()) == list(range(0, 30, 3))      # only the analysed frames
    assert out.ok.all()
    a, b = out[out.object_id == 1], out[out.object_id == 2]
    assert (a.nose_x > 70 + 30).all()          # animal 1 faces right
    assert (b.nose_x < 190 - 30).all()         # animal 2 faces left


def test_splitting_a_videos_frames_gives_the_same_keypoints(tmp_path, monkeypatch):
    import cv2
    import pandas as pd
    from fnt.videoTracking import track_store

    class _InProcess:                          # the fakes don't reach other processes: run the shares here
        def __init__(self, *_, **__):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def map(self, fn, jobs):
            return [fn(j) for j in jobs]

    monkeypatch.setattr(track_store, "TrackMaskReader", _Store)
    monkeypatch.setattr(cv2, "VideoCapture", _Cap)
    monkeypatch.setattr(M, "ProcessPoolExecutor", _InProcess)
    monkeypatch.setattr(M, "store_path_for", lambda v: str(tmp_path / "v_tracks.h5"))
    one = pd.read_csv(M.midline_for_video("v.mp4", objects="all", out_csv=str(tmp_path / "one.csv")))
    split = pd.read_csv(M.midline_for_video("v.mp4", objects="all", out_csv=str(tmp_path / "split.csv"),
                                            frame_workers=4))
    pd.testing.assert_frame_equal(one, split)
    assert sorted(split.frame.unique()) == list(range(0, 30, 3))


def _small_blob(cx, cy, ang, shape=(200, 260)):
    """A mesocosm overhead-sized vole (~35 px long, ~500 px mask) with a ~6 px bright hat near its head."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    a = np.radians(ang)
    u = (xx - cx) * np.cos(a) + (yy - cy) * np.sin(a)
    v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
    mask = (u / 18) ** 2 + (v / 9) ** 2 <= 1
    gray = np.where(mask, 40.0, 200.0)
    hx, hy = cx + 0.33 * 36 * np.cos(a), cy + 0.33 * 36 * np.sin(a)
    gray[(xx - hx) ** 2 + (yy - hy) ** 2 <= 3 ** 2] = 230.0
    return mask, gray


def test_small_animals_are_analysed_enlarged(tmp_path, monkeypatch):
    """Overhead voles are ~13x smaller in area than open-field ones: without enlarging, their masks fall under
    MIN_AREA and the hat under MARKER_MIN_AREA. Enlarged, they get keypoints and a head, written in frame px."""
    import cv2
    import pandas as pd
    from fnt.videoTracking import track_store

    class Store(_Store):
        def __init__(self, *_):
            self.items = []
            for f in range(0, 30, 3):
                for oid, (cx, ang) in ((1, (70, 0)), (2, (190, 180))):
                    self.items.append((f, oid, _small_blob(cx, 100, ang)[0]))
            self.frame = np.array([f for f, _, _ in self.items])
            self.object_id = np.array([o for _, o, _ in self.items])
            self.area = np.array([m.sum() for _, _, m in self.items])
            self.bbox = np.array([[np.where(m)[1].min(), np.where(m)[0].min(), np.where(m)[1].max(),
                                   np.where(m)[0].max()] for _, _, m in self.items])

    class Cap(_Cap):
        def __init__(self, *_):
            self.i = 0
            g = np.full((200, 260), 200.0)
            for cx, ang in ((70, 0), (190, 180)):
                mask, gray = _small_blob(cx, 100, ang)
                g = np.where(mask | (gray > 220), gray, g)
            self.img = np.repeat(g[:, :, None], 3, axis=2).astype(np.uint8)

    assert M.analysis_scale(Store().area) > 3
    assert M.analysis_scale([8000, 7000]) == 1.0                   # open-field size: unchanged
    monkeypatch.setattr(track_store, "TrackMaskReader", Store)
    monkeypatch.setattr(cv2, "VideoCapture", Cap)
    monkeypatch.setattr(M, "store_path_for", lambda v: str(tmp_path / "v_tracks.h5"))
    out = pd.read_csv(M.midline_for_video("v.mp4", objects="all"))
    assert out.ok.all()
    assert (out.head_evidence.abs() > 0).all()                      # the hat is found
    a, b = out[out.object_id == 1], out[out.object_id == 2]
    assert (a.nose_x > 70 + 8).all() and (b.nose_x < 190 - 8).all()
    assert out.body_len.between(25, 40).all()                       # frame px, not enlarged px
    assert out.mask_area.between(400, 600).all()
