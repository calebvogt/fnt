"""Behaviour review window: decisions per key, saved after every press, resumable; actor and
recipient roles with both animals' labels; masks drawn from a track store.

A synthetic video (two dark discs), tracks CSV and track store stand in for a real
session. Runs under pytest, or directly.
"""
import csv
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtCore import QEvent, Qt  # noqa: E402
from PyQt5.QtGui import QKeyEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fnt.videoTracking import behavior_review as br  # noqa: E402

app = QApplication.instance() or QApplication([])
W, H = 320, 240


def positions(i):
    """Track 1 moves right along y = 100; track 2 follows 40 px behind on y = 140."""
    return {1: (60 + 2 * i, 100), 2: (20 + 2 * i, 140)}


def make_session(tmp, n_bouts=4, with_store=False, other=""):
    video = os.path.join(tmp, "clip.avi")
    w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 15, (W, H))
    for i in range(90):
        img = np.full((H, W, 3), 200, np.uint8)
        for (x, y) in positions(i).values():
            cv2.circle(img, (x, y), 12, (20, 20, 20), -1)
        w.write(img)
    w.release()
    tracks = os.path.join(tmp, "tracks.csv")
    with open(tracks, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "track", "cx", "cy", "nose_x", "nose_y", "tb_x", "tb_y"])
        for i in range(90):
            for k, (x, y) in positions(i).items():
                wr.writerow([i, k, x, y, x + 12, y, x - 12, y])
    store = ""
    if with_store:
        from fnt.videoTracking.track_store import TrackMaskWriter
        store = os.path.join(tmp, "tracks.h5")
        tw = TrackMaskWriter(store, video_path=video, fps=15, frame_count=90, width=W, height=H)
        for i in range(90):
            matched = {}
            for k, (x, y) in positions(i).items():
                m = np.zeros((H, W), np.uint8)
                cv2.ellipse(m, (x, y), (14, 8), 0, 0, 360, 1, -1)
                matched[k] = {"mask": m.astype(bool), "bbox": [x - 14, y - 8, x + 14, y + 8],
                              "centroid": (x, y), "score": 0.9, "area": int(m.sum())}
            tw.add_frame(i, matched)
        tw.close()
    bouts = os.path.join(tmp, "bouts.csv")
    kinds = ["nose_nose", "chase", "immobile", "head_only"]
    with open(bouts, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "frame0", "frame1", "kind", "focal", "other", "tracks_csv", "tracks_h5"])
        for k in range(n_bouts):
            wr.writerow([video, 10 + 15 * k, 20 + 15 * k, kinds[k % 4], 1, other, tracks, store])
    return bouts


def press(win, key):
    win.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))


def read_review(bouts):
    with open(br.review_path(bouts), newline="") as fh:
        return {r["bout_id"]: r for r in csv.DictReader(fh)}


def test_keys_record_decisions_and_save_each_time():
    with tempfile.TemporaryDirectory() as tmp:
        bouts = make_session(tmp)
        win = br.BehaviorReviewWindow(bouts)
        assert win._frames, "frames of the first bout were not read"
        press(win, Qt.Key_A)                       # bout 0 accepted as proposed
        assert read_review(bouts)["0"]["label"] == "nose-nose sniff"
        press(win, Qt.Key_8)                       # bout 1 relabelled: key 8 = 8th behaviour
        press(win, Qt.Key_R)                       # bout 2 rejected
        press(win, Qt.Key_F)                       # bout 3: right behaviour, head/tail swapped
        r = read_review(bouts)
        assert r["1"]["decision"] == "relabel" and r["1"]["label"] == br.ETHOGRAM[7]
        assert r["2"]["decision"] == "reject" and r["2"]["label"] == ""
        assert r["3"]["decision"] == "accept" and r["3"]["flipped"] == "1"
        assert win.lbl_progress.text() == "4 of 4 reviewed"
        win.close()


def test_a_reopened_session_resumes_at_the_first_unreviewed_bout():
    with tempfile.TemporaryDirectory() as tmp:
        bouts = make_session(tmp)
        win = br.BehaviorReviewWindow(bouts)
        press(win, Qt.Key_A)
        press(win, Qt.Key_S)
        win.close()
        again = br.BehaviorReviewWindow(bouts)
        assert again.idx == 2
        assert again.table.item(0, 3).text() == "✓"
        assert again.table.item(1, 3).text() == "skipped"
        again.close()


def test_both_animals_get_their_side_of_a_social_behaviour():
    with tempfile.TemporaryDirectory() as tmp:
        bouts = make_session(tmp, n_bouts=3, other=2)
        win = br.BehaviorReviewWindow(bouts)
        press(win, Qt.Key_A)                       # nose-nose: mutual
        press(win, Qt.Key_W)                       # chase, but track 2 is the chaser
        press(win, Qt.Key_A)
        press(win, Qt.Key_5)                       # relabel bout 2: anogenital sniff
        r = read_review(bouts)
        assert (r["0"]["actor"], r["0"]["recipient"]) == ("1", "2")
        assert r["0"]["recipient_label"] == "nose-nose sniff"
        assert (r["1"]["actor"], r["1"]["recipient"]) == ("2", "1")
        assert r["1"]["actor_label"] == "chase" and r["1"]["recipient_label"] == "chased / flees"
        assert r["2"]["actor_label"] == "anogenital sniff"
        assert r["2"]["recipient_label"] == "receives anogenital sniff"
        win.close()


def test_a_solo_behaviour_has_no_recipient_label():
    assert br.recipient_label("self-groom", "2") == ""
    assert br.recipient_label("chase", "") == ""


def test_masks_are_drawn_in_role_colours():
    with tempfile.TemporaryDirectory() as tmp:
        plain = br.BehaviorReviewWindow(make_session(tmp, n_bouts=1, other=2))
        f_plain = plain._frames[8].astype(int)
        plain.close()
    with tempfile.TemporaryDirectory() as tmp:
        win = br.BehaviorReviewWindow(make_session(tmp, n_bouts=1, other=2, with_store=True))
        f_mask = win._frames[8].astype(int)
        win.close()
    diff = np.abs(f_mask - f_plain).sum(-1)
    assert (diff > 30).mean() > 0.005, "no mask overlay drawn"
    # The actor is tinted orange (red channel up relative to blue) somewhere.
    tinted = (f_mask[..., 2] - f_mask[..., 0]) > 40
    assert tinted.any()


def test_the_view_follows_the_animals():
    with tempfile.TemporaryDirectory() as tmp:
        bouts = make_session(tmp, n_bouts=1)
        win = br.BehaviorReviewWindow(bouts)
        # 0.5 s (8 frames at 15 fps) of context either side of an 11-frame bout.
        assert len(win._frames) == 11 + 2 * 8
        assert sum(win._in_bout) == 11
        win.close()


def test_bouts_need_the_core_columns():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "bad.csv")
        with open(p, "w", newline="") as fh:
            csv.writer(fh).writerows([["video", "frame0", "kind"], ["v.mp4", "1", "chase"]])
        with pytest.raises(ValueError):
            br.load_bouts(p)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
