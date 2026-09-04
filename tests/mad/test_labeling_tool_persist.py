"""Confirming a call must not disarm the labelling tool.

Enter used to switch SAM / Paint / Eraser off, so labelling a file meant
re-arming the tool between every single call. Labelling is a long run of the
same gesture -- Enter ends a call, not the session.

The SAM prompt points still have to be dropped: leaving them would make the
next click extend the prompt for the call just saved rather than starting a new
one, which produces a mask spanning two calls. That distinction (clear the
prompts, keep the mode) is the whole content of the change, so both halves are
pinned here.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication
from scipy.io import wavfile

import fnt.usv.mad_pyqt as M
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR = 250_000
_GUI = None


def gui():
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_tool_")
        wav = os.path.join(root, "a.wav")
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
        # The class prompt and the one-time dialogs are modal; headless they
        # would block forever.
        w._ask_class_for_confirm = lambda n, classes, last: 'USV'
        w._show_first_label_info = lambda: None
        _GUI = (w, wav, app)
    return _GUI[:2]


def _pending_call(sg, f0=120, t0=900):
    sg._pending = np.zeros((sg.n_freq_bins, sg.n_time_frames), dtype=np.uint8)
    sg._pending[f0:f0 + 20, t0:t0 + 40] = 1


def _arm(w, sg, tool):
    for b in (w.btn_paint, w.btn_erase, w.btn_sam):
        b.setChecked(False)
    {'sam': w.btn_sam, 'brush': w.btn_paint,
     'eraser': w.btn_erase}[tool].setChecked(True)
    sg.set_paint_mode(tool)


# ----------------------------------------------------------------------
def test_sam_survives_confirming_a_call():
    w, _wav = gui()
    sg = w.spectrogram
    _arm(w, sg, 'sam')
    sg._sam_pos_pts = [(900, 120)]
    _pending_call(sg)
    before = len(sg.annotations)

    w._confirm_pending()

    assert len(sg.annotations) == before + 1, "the call was not saved"
    assert w.btn_sam.isChecked(), "SAM was switched off by Enter"
    assert sg.paint_mode == 'sam'


def test_paint_survives_confirming_a_call():
    w, _wav = gui()
    sg = w.spectrogram
    _arm(w, sg, 'brush')
    _pending_call(sg, f0=200, t0=1200)

    w._confirm_pending()

    assert w.btn_paint.isChecked()
    assert sg.paint_mode == 'brush'


def test_the_sam_prompt_points_are_still_cleared():
    """Kept points would make the next click extend the prompt for the call
    just saved, producing one mask over two calls."""
    w, _wav = gui()
    sg = w.spectrogram
    _arm(w, sg, 'sam')
    sg._sam_pos_pts = [(900, 120)]
    sg._sam_neg_pts = [(950, 130)]
    _pending_call(sg, f0=260, t0=1500)

    w._confirm_pending()

    assert sg._sam_pos_pts == []
    assert sg._sam_neg_pts == []


def test_the_pending_mask_is_still_cleared():
    w, _wav = gui()
    sg = w.spectrogram
    _arm(w, sg, 'sam')
    _pending_call(sg, f0=300, t0=1800)

    w._confirm_pending()

    assert not sg.has_pending(), "the confirmed mask stayed pending"


def test_calls_can_be_labelled_back_to_back_without_rearming():
    """The point of the change, stated as the workflow it enables."""
    w, _wav = gui()
    sg = w.spectrogram
    _arm(w, sg, 'sam')
    before = len(sg.annotations)
    for i in range(3):
        _pending_call(sg, f0=120 + 40 * i, t0=2000 + 200 * i)
        w._confirm_pending()
        assert sg.paint_mode == 'sam', f"tool dropped after call {i + 1}"
    assert len(sg.annotations) == before + 3


def test_confirming_nothing_leaves_the_tool_alone():
    w, _wav = gui()
    sg = w.spectrogram
    _arm(w, sg, 'sam')
    sg.clear_pending()

    w._confirm_pending()

    assert w.btn_sam.isChecked()
    assert sg.paint_mode == 'sam'


def test_the_explicit_deactivate_still_turns_everything_off():
    """Kept for the paths that genuinely want it — losing audio, for one."""
    w, _wav = gui()
    sg = w.spectrogram
    _arm(w, sg, 'sam')

    w._deactivate_labeling_tools()

    assert not any(b.isChecked()
                   for b in (w.btn_paint, w.btn_erase, w.btn_sam))
    assert sg.paint_mode is None


if __name__ == "__main__":
    import sys
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("  OK   " + name, flush=True)
        except Exception:
            fails += 1
            print("  FAIL " + name, flush=True)
            traceback.print_exc()
    print("")
    print("ALL OK" if not fails else str(fails) + " FAILURE(S)", flush=True)
    sys.stdout.flush()
    os._exit(1 if fails else 0)
