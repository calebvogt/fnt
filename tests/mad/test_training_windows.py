"""The review overlay shades exactly what an accept or reject puts in training.

Confirming a call saves a patch of every frequency from EXAMPLE_CONTEXT_FRAMES
before it to as many after, and training supervises the whole patch. Inside it,
anything that is not a confirmed call -- a pending detection, an unmarked call
-- is taught as background. Nothing showed that, so the natural assumption
("only the call's own columns are used") left pending detections beside
accepted calls to be trained as noise. W now shades those spans and the header
counts the pending detections still inside one.

Runs under pytest, or directly.
"""
import inspect
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication
from scipy.io import wavfile

import fnt.usv.mad_pyqt as M
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR, NPERSEG, NOVERLAP, NFFT = 250_000, 512, 384, 1024
CTX = M.EXAMPLE_CONTEXT_FRAMES
_APP = None
_SG = None
_GUI = None


def widget():
    """A bare spectrogram widget over 1 s of noise."""
    global _APP, _SG
    if _SG is None:
        _APP = QApplication.instance() or QApplication([])
        M.MADMainWindow._apply_dark_theme()
        sg = M.MADSpectrogramWidget()
        sg.resize(1100, 700)
        audio = np.random.default_rng(0).normal(0, .05, SR).astype(np.float32)
        sg.set_audio_data(audio, SR, preserve_view=False)
        sg.init_mask(audio_len=len(audio), sample_rate=SR, nperseg=NPERSEG,
                     noverlap=NOVERLAP, nfft=NFFT)
        sg.min_freq, sg.max_freq = 0.0, SR / 2.0
        sg.view_mode = 'overlay'
        _SG = sg
    _SG.show_training_windows = True
    return _SG


def ann(t0, t1, status, i=0, f0=120, f1=200):
    m = np.ones((f1 - f0, t1 - t0), dtype=bool)
    return {'id': f'a{i}_{status}', 'status': status, 'category': 'USV',
            'blob_id': i, 't0': t0, 't1': t1, 'f0': f0, 'f1': f1, 'mask': m}


def put(sg, anns):
    sg.set_annotations(anns)
    return sg


# ----------------------------------------------------------------------
# What a window is
# ----------------------------------------------------------------------
def test_an_accepted_call_shades_its_saved_patch_span():
    sg = put(widget(), [ann(1000, 1050, 'accepted')])
    assert sg.training_windows() == [(1000 - CTX, 1050 + CTX)]


def test_a_rejection_shades_too_and_a_pending_detection_does_not():
    sg = put(widget(), [ann(600, 640, 'rejected', 0),
                        ann(1300, 1340, 'prediction', 1)])
    assert sg.training_windows() == [(600 - CTX, 640 + CTX)]


def test_a_hand_label_with_no_status_counts_as_confirmed():
    sg = put(widget(), [ann(1000, 1050, None)])
    assert sg.training_windows() == [(1000 - CTX, 1050 + CTX)]


def test_overlapping_windows_merge_and_edges_clip_to_the_file():
    sg = widget()
    n = sg.n_time_frames
    put(sg, [ann(50, 80, 'accepted', 0),          # starts inside the margin
             ann(150, 170, 'accepted', 1),        # overlaps the first window
             ann(n - 40, n - 10, 'rejected', 2)])  # ends inside the margin
    assert sg.training_windows() == [(0, 170 + CTX), (n - 40 - CTX, n)]


def test_the_save_path_uses_the_same_context_as_the_overlay():
    """If these drift apart the shading lies about what trains."""
    src = inspect.getsource(M.MADMainWindow._save_component_example)
    assert "margin = EXAMPLE_CONTEXT_FRAMES" in src


# ----------------------------------------------------------------------
# Pending detections left inside a window
# ----------------------------------------------------------------------
def test_pending_inside_and_straddling_a_window_are_flagged_outside_is_not():
    lo, hi = 1000 - CTX, 1050 + CTX
    sg = put(widget(), [
        ann(1000, 1050, 'accepted', 0),
        ann(1060, 1080, 'prediction', 1),          # inside
        ann(hi - 5, hi + 30, 'prediction', 2),     # straddles the right edge
        ann(lo - 40, lo, 'prediction', 3),         # ends exactly at the edge
        ann(hi + 200, hi + 240, 'prediction', 4),  # well outside
    ])
    flagged = {sg.annotations[i]['id'] for i in sg.undecided_in_training_windows()}
    assert flagged == {'a1_prediction', 'a2_prediction'}


def test_deciding_the_pending_call_clears_the_flag():
    sg = put(widget(), [ann(1000, 1050, 'accepted', 0),
                        ann(1060, 1080, 'prediction', 1)])
    assert len(sg.undecided_in_training_windows()) == 1
    sg.annotations[1]['status'] = 'rejected'
    assert sg.undecided_in_training_windows() == []


def test_no_decisions_means_no_windows_and_nothing_flagged():
    sg = put(widget(), [ann(1060, 1080, 'prediction', 1)])
    assert sg.training_windows() == []
    assert sg.undecided_in_training_windows() == []


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def _column_brightness(sg, frame):
    """Mean brightness of one screen column at spectrogram ``frame``, taken
    from a stretch of the canvas well away from any mask."""
    img = sg.grab().toImage()
    rect = sg._get_spec_rect()
    t_start, t_end, _f0, _f1 = sg._visible_spec_bounds()
    x = int(rect.left() + (frame + 0.5 - t_start) / (t_end - t_start)
            * rect.width())
    ys = range(int(rect.top()) + 5, int(rect.top()) + 60)
    return np.mean([sum(img.pixelColor(x, y).getRgb()[:3]) for y in ys])


def test_the_band_lifts_the_spectrogram_only_where_training_reaches():
    sg = put(widget(), [ann(1000, 1050, 'accepted')])
    sg.view_start, sg.view_end = 0.40, 0.70        # frames ~779 .. ~1365
    inside, outside = 1050 + CTX // 2, 1050 + CTX + 60
    sg.show_training_windows = False
    off_in, off_out = _column_brightness(sg, inside), _column_brightness(sg, outside)
    sg.show_training_windows = True
    on_in, on_out = _column_brightness(sg, inside), _column_brightness(sg, outside)
    assert on_in > off_in + 3, (off_in, on_in)
    assert abs(on_out - off_out) < 1e-6, (off_out, on_out)


def test_spec_only_view_draws_no_band():
    sg = put(widget(), [ann(1000, 1050, 'accepted')])
    sg.view_start, sg.view_end = 0.40, 0.70
    frame = 1050 + CTX // 2
    sg.view_mode = 'spec'
    try:
        sg.show_training_windows = False
        off = _column_brightness(sg, frame)
        sg.show_training_windows = True
        on = _column_brightness(sg, frame)
    finally:
        sg.view_mode = 'overlay'
    assert abs(on - off) < 1e-6


# ----------------------------------------------------------------------
# The window: header count, toggle, persistence
# ----------------------------------------------------------------------
def gui():
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_trainwin_")
        wav = os.path.join(root, "r0.wav")
        wavfile.write(wav, SR, (np.random.default_rng(0).normal(0, .05, SR * 2)
                                * 32767).astype(np.int16))
        proj = os.path.join(root, "p")
        create_mad_project(proj)
        app = QApplication.instance() or QApplication([])
        M.MADMainWindow._apply_dark_theme()
        w = M.MADMainWindow()
        w._activate_project(MADProjectConfig.load(proj))
        w._register_audio_files([wav])
        w._append_audio_paths([wav])
        w._wait_for_audio_load()
        w.resize(1400, 900)
        w.show()
        w._auto_advance = False
        if hasattr(w, 'chk_auto_advance'):
            w.chk_auto_advance.setChecked(False)
        for _ in range(50):
            app.processEvents()
        _GUI = w
    return _GUI


def test_the_header_counts_pending_detections_inside_a_window():
    w = gui()
    w.chk_train_windows.setChecked(True)
    sg = w.spectrogram
    sg.annotations = [ann(1000, 1050, 'accepted', 0),
                      ann(1060, 1080, 'prediction', 1),
                      ann(1700, 1740, 'prediction', 2)]
    w._update_view_header()
    assert "1 pending inside training windows" in w.lbl_view_header.text()


def test_the_count_is_absent_when_nothing_is_left_inside():
    w = gui()
    w.chk_train_windows.setChecked(True)
    w.spectrogram.annotations = [ann(1000, 1050, 'accepted', 0),
                                 ann(1700, 1740, 'prediction', 2)]
    w._update_view_header()
    assert "inside training windows" not in w.lbl_view_header.text()


def test_unticking_hides_the_bands_and_the_count_and_is_remembered():
    w = gui()
    w.spectrogram.annotations = [ann(1000, 1050, 'accepted', 0),
                                 ann(1060, 1080, 'prediction', 1)]
    w.chk_train_windows.setChecked(True)
    w.chk_train_windows.setChecked(False)
    assert w.spectrogram.show_training_windows is False
    assert "inside training windows" not in w.lbl_view_header.text()
    assert str(w._settings.value(M.TRAIN_WINDOWS_KEY)).lower() in ("false", "0")
    w._shortcut_training_windows()          # W
    assert w.chk_train_windows.isChecked()
    assert w.spectrogram.show_training_windows is True
    assert "1 pending inside training windows" in w.lbl_view_header.text()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
