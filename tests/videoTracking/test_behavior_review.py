"""Behaviour review window: decisions per key, saved after every press, resumable.

A synthetic video and bouts CSV stand in for a real session. Runs under pytest, or
directly.
"""
import csv
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtGui import QKeyEvent  # noqa: E402
from PyQt5.QtCore import QEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fnt.videoTracking import behavior_review as br  # noqa: E402

app = QApplication.instance() or QApplication([])


def make_session(tmp, n_bouts=4):
    video = os.path.join(tmp, "clip.avi")
    w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 240))
    for i in range(90):
        img = np.full((240, 320, 3), 80, np.uint8)
        cv2.circle(img, (40 + 2 * i, 120), 12, (20, 20, 20), -1)
        w.write(img)
    w.release()
    tracks = os.path.join(tmp, "tracks.csv")
    with open(tracks, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "track", "cx", "cy", "nose_x", "nose_y", "tb_x", "tb_y"])
        for i in range(90):
            wr.writerow([i, 1, 40 + 2 * i, 120, 52 + 2 * i, 120, 28 + 2 * i, 120])
    bouts = os.path.join(tmp, "bouts.csv")
    kinds = ["nose_nose", "chase", "immobile", "head_only"]
    with open(bouts, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["video", "frame0", "frame1", "kind", "focal", "other", "tracks_csv"])
        for k in range(n_bouts):
            wr.writerow([video, 10 + 15 * k, 20 + 15 * k, kinds[k % 4], 1, "", tracks])
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
