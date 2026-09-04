"""Box-selected detections must answer the review keys, whatever their status.

Reported from the field: three confirmed calls rubber-banded on a fully
reviewed file, R pressed, nothing happened. There were two independent causes,
and either one alone was enough to make the keys look dead:

* the A/R shortcut bailed out before it ever consulted the box -- a box drag
  leaves no single selection index, and the file had nothing left pending;
* the bulk action then filtered its targets down to ``status == 'prediction'``,
  which a confirmed label is not, so it "succeeded" on zero items.

Bulk-rejecting *confirmed* labels is the main reason to draw a box at all --
culling harmonics out of a hand-labeled file -- so both had to go.

Runs under pytest, or directly (``python test_box_selection_review.py``) since
the lab environments do not all carry pytest.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication
from scipy.io import wavfile

import fnt.usv.mad_pyqt as M
from fnt.usv.usv_detector.fnt_mask_store import (
    masks_sibling_path, td_delete, td_list_ids)
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR = 250_000
_GUI = None


def gui():
    """One offscreen window, one project, one recording -- built once."""
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_box_")
        wav = os.path.join(root, "rec", "a.wav")
        os.makedirs(os.path.dirname(wav), exist_ok=True)
        wavfile.write(wav, SR, (np.random.default_rng(0).normal(0, .05, SR * 2)
                                * 32767).astype(np.int16))
        proj = os.path.join(root, "proj")
        create_mad_project(proj)
        # Keep a reference: a QApplication that gets garbage-collected takes
        # the process down with it, with no Python traceback to show for it.
        app = QApplication.instance() or QApplication([])
        M.MADMainWindow._apply_dark_theme()
        w = M.MADMainWindow()
        w._activate_project(MADProjectConfig.load(proj))
        w._register_audio_files([wav])
        w._append_audio_paths([wav])
        # Audio is read on a worker thread now; the window is not
        # usable until it lands.
        w._wait_for_audio_load()
        _GUI = (w, wav, app)
    return _GUI[:2]


def _crop():
    m = np.zeros((12, 20), dtype=bool)
    m[3:9, 4:16] = True
    return m


def seed(status, n=5, box=(0, 1, 2)):
    """``n`` detections of one status, with the first three rubber-banded."""
    w, wav = gui()
    h5 = masks_sibling_path(wav)
    for i in td_list_ids(h5):
        td_delete(h5, i)
    sg = w.spectrogram
    sg.annotations = [{
        'id': M._new_ann_id('pred'), 'blob_id': i, 'category': 'USV',
        'status': status, 'score': 0.9,
        'f0': 100 + i * 30, 'f1': 112 + i * 30,
        't0': 300 + i * 90, 't1': 320 + i * 90, 'mask': _crop()}
        for i in range(n)]
    sg._rebuild_confirmed_mask()
    w._undo_stack = []
    w._refresh_annotation_list()
    # A rubber-band drag exactly as the canvas reports it: a set of ids, and
    # NO single selection index. That absence is what broke the shortcut.
    sg._selected_ann_idx = None
    w._on_box_selection(list(box))
    return w, sg


def _statuses(sg):
    return [a.get('status') for a in sg.annotations]


# ----------------------------------------------------------------------
# The reported bug, driven through the real key path
# ----------------------------------------------------------------------
def test_r_rejects_every_boxed_confirmed_label():
    """The field report: 3 confirmed calls boxed, R pressed, nothing happened."""
    w, sg = seed('accepted')
    assert not w._pred_indices(), "file must have nothing pending to reproduce"

    w._shortcut_review('rejected')

    assert _statuses(sg) == ['rejected'] * 3 + ['accepted'] * 2
    assert w._box_sel_ids == [], "the box should be released once applied"


def test_a_accepts_every_boxed_rejected_label():
    w, sg = seed('rejected')

    w._shortcut_review('accepted')

    assert _statuses(sg) == ['accepted'] * 3 + ['rejected'] * 2


def test_hand_drawn_labels_are_targets_too():
    """A hand-drawn label carries status None, not 'accepted'."""
    w, sg = seed(None)

    w._shortcut_review('rejected')

    assert _statuses(sg) == ['rejected'] * 3 + [None] * 2


def test_pending_predictions_still_work():
    """The original behaviour must survive the widening."""
    w, sg = seed('prediction')

    w._shortcut_review('rejected')

    assert _statuses(sg) == ['rejected'] * 3 + ['prediction'] * 2


# ----------------------------------------------------------------------
# Rejecting a confirmed label has to stop it training the model
# ----------------------------------------------------------------------
def test_rejecting_confirmed_demotes_its_training_example():
    """A box-rejected label must stop training as a call — but keep its mask.

    Rejecting used to delete the example outright. That stopped it training,
    correctly, but destroyed the traced shape, so reopening showed a bounding
    box and re-accepting saved that box as the label. Now it is demoted to
    kind='rejected': pixels kept, supervised as a hard negative, reversible.
    """
    from fnt.usv.usv_detector.fnt_mask_store import example_kind, td_read_example
    w, sg = seed('accepted')
    h5 = masks_sibling_path(gui()[1])
    # Give every label a saved example, as accepting one would.
    for a in sg.annotations:
        a['id'] = w._save_component_example(
            'USV', (a['f0'], a['f1'], a['t0'], a['t1'], a['mask']),
            blob_id=a['blob_id'])
    w._refresh_annotation_list()
    sg._selected_ann_idx = None
    w._on_box_selection([0, 1, 2])
    boxed = [a['id'] for a in sg.annotations[:3]]
    untouched = [a['id'] for a in sg.annotations[3:]]
    px_before = {i: int((td_read_example(h5, i)['mask'] > 0).sum())
                 for i in boxed}

    w._shortcut_review('rejected')

    left = set(td_list_ids(h5))
    for i in boxed:
        assert i in left, "the example was deleted instead of demoted"
        ex = td_read_example(h5, i)
        assert example_kind(ex['meta']) == 'rejected', \
            "a rejected label still trains as a call"
        assert int((ex['mask'] > 0).sum()) == px_before[i], \
            "the traced mask was damaged"
    for i in untouched:
        ex = td_read_example(h5, i)
        assert example_kind(ex['meta']) == 'label', \
            "a label outside the box was disturbed"


# ----------------------------------------------------------------------
# Counting, skipping, deleting
# ----------------------------------------------------------------------
def test_already_settled_items_are_left_alone():
    """Rejecting an already-rejected label is not a change worth reporting."""
    w, sg = seed('accepted')
    sg.annotations[1]['status'] = 'rejected'
    w._refresh_annotation_list()
    sg._selected_ann_idx = None
    w._on_box_selection([0, 1, 2])

    w._shortcut_review('rejected')

    assert _statuses(sg) == ['rejected'] * 3 + ['accepted'] * 2
    assert "2 detection" in w.status_bar.currentMessage(), \
        w.status_bar.currentMessage()


def test_skip_releases_the_box_without_deciding():
    """S is the way out of a box drawn by mistake."""
    w, sg = seed('accepted')

    w._shortcut_skip()

    assert _statuses(sg) == ['accepted'] * 5, "skip changed a label"
    assert w._box_sel_ids == []


def test_delete_removes_boxed_labels_of_any_status():
    w, sg = seed('accepted')

    w._delete_selected_annotation()

    assert len(sg.annotations) == 2
    assert w._box_sel_ids == []


if __name__ == "__main__":
    import sys
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        print("  .... " + name, flush=True)
        try:
            fn()
            print("  OK   " + name, flush=True)
        except Exception:
            fails += 1
            print("  FAIL " + name, flush=True)
            traceback.print_exc()
    print("")
    print("ALL OK" if not fails else str(fails) + " FAILURE(S)", flush=True)
    # Qt's offscreen teardown aborts the process on this platform, which would
    # mask the result; the report is already written, so leave now.
    sys.stdout.flush()
    os._exit(1 if fails else 0)
