"""Play starts just before the selected call when it is on screen.

A wide view centred on a call used to play from its left edge, so at slowed
speed the reviewer waited through half the window before hearing the call.
With no call selected, or one off screen, Play still starts at the view's left
edge.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.usv.mad_pyqt as M  # noqa: E402

SR, HOP = 250_000, 128
DT = HOP / SR

_W = None
_APP = None


def win():
    global _W, _APP
    if _W is None:
        _APP = QApplication.instance() or QApplication([])     # kept: a collected app crashes Qt
        _W = M.MADMainWindow()
    w = _W
    w.sample_rate = SR
    w.spectrogram.hop = HOP
    # a call from 2.0 to 2.05 s
    w.spectrogram.annotations = [{'id': 'a', 't0': int(2.0 / DT), 't1': int(2.05 / DT)}]
    return w


def test_starts_just_before_the_selected_call():
    w = win()
    w.spectrogram._selected_ann_idx = 0
    t0 = w.spectrogram.annotations[0]['t0'] * DT
    w.playback_speed = 1.0
    assert abs(w._play_from_selected_s(0.5, 3.5, 'slow') - (t0 - 0.1)) < 1e-9
    w.playback_speed = 0.03                    # 0.5 s of listening = 15 ms of audio
    assert abs(w._play_from_selected_s(0.5, 3.5, 'slow') - (t0 - 0.015)) < 1e-9
    assert abs(w._play_from_selected_s(0.5, 3.5, 'het') - (t0 - 0.1)) < 1e-9
    # never before the view's left edge
    assert w._play_from_selected_s(1.99, 3.5, 'het') == 1.99


def test_view_edge_without_an_on_screen_selection():
    w = win()
    w.playback_speed = 1.0
    w.spectrogram._selected_ann_idx = None
    assert w._play_from_selected_s(0.5, 3.5, 'slow') == 0.5
    w.spectrogram._selected_ann_idx = 0
    assert w._play_from_selected_s(2.5, 5.5, 'slow') == 2.5      # call left of the view
    assert w._play_from_selected_s(0.0, 1.5, 'slow') == 0.0      # call right of the view


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
