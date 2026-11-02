"""Harmonic grouping in the review GUI: detect, correct, persist.

The grouping rule itself is covered in test_harmonics.py. What matters here is
the review loop around it -- because the loop is the point. Deciding harmonics
in code instead of at labeling time only pays off if a wrong grouping is cheap
to see and cheap to fix, so: links have to reach the canvas, a right-click
correction has to stick, and re-running with different tolerances must not
quietly undo a decision the reviewer already made.

Runs under pytest, or directly, since the lab environments do not all carry
pytest.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication
from scipy.io import wavfile

import fnt.usv.mad_pyqt as M
from fnt.usv.usv_detector.mad_inference import CSV_FIELDNAMES
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR = 250_000
# Spectrogram bins for ~25 / ~50 / ~75 kHz at nfft 1024, sr 250 kHz.
BIN_F0, BIN_H2, BIN_H3 = 102, 205, 307
_GUI = None


def gui():
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_hgui_")
        wav = os.path.join(root, "a.wav")
        wavfile.write(wav, SR, (np.random.default_rng(0).normal(0, .05, SR * 2)
                                * 32767).astype(np.int16))
        proj = os.path.join(root, "p")
        create_mad_project(proj)
        # Keep the QApplication referenced: if it is collected the process dies
        # with no traceback to explain it.
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


def _ann(bin_lo, t0, thick=4, nt=30, status='accepted'):
    return {'id': M._new_ann_id('p'), 'blob_id': None, 'category': 'USV',
            'status': status, 'score': 1.0, 'f0': bin_lo, 'f1': bin_lo + thick,
            't0': t0, 't1': t0 + nt, 'mask': np.ones((thick, nt), bool)}


def seed():
    """A three-element stack at ~25/50/75 kHz, plus an unrelated later call."""
    w, _ = gui()
    sg = w.spectrogram
    sg.annotations = [_ann(BIN_F0, 1000), _ann(BIN_H2, 1000),
                      _ann(BIN_H3, 1000), _ann(150, 3000)]
    sg._rebuild_confirmed_mask()
    w._forced_harmonics.clear()
    w._refresh_annotation_list()
    return w, sg


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------
def test_the_csv_schema_carries_the_grouping():
    for col in ('harmonic_call_id', 'harmonic_n', 'f0_hz'):
        assert col in CSV_FIELDNAMES


def test_detecting_groups_a_stack_and_leaves_the_rest_alone():
    w, sg = seed()
    w._detect_harmonics()
    res = w._harmonic_result
    assert res is not None and res.n_calls == 2
    assert [a.get('harmonic_n') for a in sg.annotations[:3]] == [1, 2, 3]
    assert sg.annotations[3].get('harmonic_n') in (1, None)


def test_links_reach_the_canvas_and_start_at_the_fundamental():
    w, sg = seed()
    w._detect_harmonics()
    f0_id = str(sg.annotations[0]['id'])
    assert len(sg.harmonic_links) == 2
    assert all(lo == f0_id for lo, _up, _n in sg.harmonic_links)
    assert sorted(n for _l, _u, n in sg.harmonic_links) == [2, 3]


def test_links_are_keyed_by_id_so_deleting_a_call_cannot_mislink():
    """Indices shift when a detection is deleted; ids do not. A link that
    silently repoints at a different call is worse than no link."""
    w, sg = seed()
    w._detect_harmonics()
    survivor = str(sg.annotations[2]['id'])
    del sg.annotations[0]                 # drop the fundamental
    by_id = {a.get('id'): a for a in sg.annotations}
    live = [(lo, up) for lo, up, _n in sg.harmonic_links
            if lo in by_id and up in by_id]
    assert live == [], "a link outlived its fundamental"
    assert survivor in by_id
    assert sg.grab().toImage().width() > 0, "painting stale links must not crash"


def test_detection_changes_no_review_decision():
    """Grouping is an annotation, not a verdict -- it must not accept, reject
    or delete anything."""
    w, sg = seed()
    before = [(a['id'], a['status']) for a in sg.annotations]
    w._detect_harmonics()
    assert [(a['id'], a['status']) for a in sg.annotations] == before


def test_a_rejected_detection_never_joins_a_stack():
    """The reviewer already said it is not a call; grouping must not reopen it."""
    w, sg = seed()
    sg.annotations[1]['status'] = 'rejected'
    w._detect_harmonics()
    assert sg.annotations[1].get('harmonic_n') is None


def test_grouping_a_file_with_nothing_loaded_is_harmless():
    w, sg = seed()
    sg.annotations = []
    w._detect_harmonics()
    assert sg.harmonic_links == []


# ----------------------------------------------------------------------
# Correction
# ----------------------------------------------------------------------
def test_a_detection_can_be_pinned_out_of_its_stack():
    w, sg = seed()
    w._detect_harmonics()
    w._set_forced_harmonic(2, None)
    assert w._harmonic_result.n_calls == 3
    assert sg.annotations[2].get('harmonic_n') == 1


def test_a_detection_can_be_reassigned_to_another_fundamental():
    w, sg = seed()
    w._detect_harmonics()
    target = str(sg.annotations[3]['id'])
    w._set_forced_harmonic(1, target)
    assert w._harmonic_result.call_of[str(sg.annotations[1]['id'])] == target


def test_a_correction_survives_a_re_run():
    """The whole argument for grouping in code is that you can re-run it. That
    only holds if re-running does not throw away the corrections."""
    w, sg = seed()
    w._detect_harmonics()
    pinned = str(sg.annotations[2]['id'])
    w._set_forced_harmonic(2, None)
    w._detect_harmonics()
    assert sg.annotations[2].get('harmonic_n') == 1
    assert w._forced_harmonics_for_file()[pinned] is None


def test_a_correction_can_be_cleared():
    w, sg = seed()
    w._detect_harmonics()
    w._set_forced_harmonic(2, None)
    w._set_forced_harmonic(2, '__auto__')
    assert sg.annotations[2].get('harmonic_n') == 3
    assert str(sg.annotations[2]['id']) not in w._forced_harmonics_for_file()


def test_corrections_are_kept_per_recording():
    w, sg = seed()
    w._detect_harmonics()
    w._set_forced_harmonic(2, None)
    assert len(w._forced_harmonics) == 1
    key = next(iter(w._forced_harmonics))
    assert key == os.path.normcase(w._active_review_wav_path())


def test_the_menu_only_offers_plausible_fundamentals():
    """Lower in frequency and overlapping in time -- anything else cannot be
    this element's fundamental."""
    w, sg = seed()
    cands = w._harmonic_menu_candidates(2)
    assert 3 not in cands, "a call at another time was offered"
    assert 2 not in cands, "itself was offered"
    for i in cands:
        assert sg.annotations[i]['f1'] <= sg.annotations[2]['f1']
        assert sg.annotations[i]['t0'] < sg.annotations[2]['t1']


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def test_the_grouping_is_written_onto_the_calls():
    """The assignment lives on the call, not in a CSV column.

    A side file would have to be kept in step with the labels; metadata on the
    example travels with it, survives a close, and comes back out on export.
    """
    from fnt.usv.usv_detector.fnt_mask_store import masks_sibling_path, td_iter_meta
    w, sg = seed()
    wav = gui()[1]
    # Give each annotation a real stored example to carry the assignment.
    for a in sg.annotations:
        a['id'] = w._save_component_example(
            'USV', (a['f0'], a['f1'], a['t0'], a['t1'], a['mask']),
            blob_id=a['blob_id'])
    w._refresh_annotation_list()

    w._detect_harmonics()

    meta = {m['id']: m for m in td_iter_meta(masks_sibling_path(wav))}
    ns = {a['id']: meta.get(a['id'], {}).get('harmonic_n')
          for a in sg.annotations}
    assert [ns[a['id']] for a in sg.annotations[:3]] == [1, 2, 3], ns
    call_ids = {meta[a['id']].get('harmonic_call_id')
                for a in sg.annotations[:3]}
    assert len(call_ids) == 1, f"a stack should share one call id: {call_ids}"


def test_a_correction_survives_a_restart():
    """The corrections are the only part of the grouping that cannot be
    recomputed, so they have to live on the call rather than in the session."""
    from fnt.usv.usv_detector.fnt_mask_store import masks_sibling_path, td_iter_meta
    w, sg = seed()
    wav = gui()[1]
    for a in sg.annotations:
        a['id'] = w._save_component_example(
            'USV', (a['f0'], a['f1'], a['t0'], a['t1'], a['mask']),
            blob_id=a['blob_id'])
    w._refresh_annotation_list()
    w._detect_harmonics()
    pinned = str(sg.annotations[2]['id'])

    w._set_forced_harmonic(2, None)          # "not a harmonic"

    meta = {m['id']: m for m in td_iter_meta(masks_sibling_path(wav))}
    assert 'harmonic_forced' in meta[pinned], "the correction was session-only"

    # A fresh read of the store, as a restart would do.
    w._forced_harmonics.clear()
    w._load_forced_harmonics(wav)
    assert w._forced_harmonics_for_file().get(pinned, '__x__') is None


def test_clearing_a_correction_removes_it_from_the_call():
    """"No correction" is the absence of the key, not a sentinel — so clearing
    has to delete it, or a restart would read it back as a decision."""
    from fnt.usv.usv_detector.fnt_mask_store import masks_sibling_path, td_iter_meta
    w, sg = seed()
    wav = gui()[1]
    for a in sg.annotations:
        a['id'] = w._save_component_example(
            'USV', (a['f0'], a['f1'], a['t0'], a['t1'], a['mask']),
            blob_id=a['blob_id'])
    w._refresh_annotation_list()
    w._detect_harmonics()
    pinned = str(sg.annotations[2]['id'])
    w._set_forced_harmonic(2, None)

    w._set_forced_harmonic(2, '__auto__')

    meta = {m['id']: m for m in td_iter_meta(masks_sibling_path(wav))}
    assert 'harmonic_forced' not in meta[pinned]
    # Assert about THIS call, not the whole file: corrections persist now, so
    # earlier tests in this module have left their own in the shared store.
    w._forced_harmonics.clear()
    w._load_forced_harmonics(wav)
    assert pinned not in w._forced_harmonics_for_file()


def test_the_grouping_reaches_the_exported_csv():
    from fnt.usv.usv_detector.mad_csv_rebuild import rebuild_annotations_csv
    from fnt.usv.usv_detector.mad_inference import read_blob_csv
    from fnt.usv.usv_detector.mad_labels import annotations_csv_sibling_path
    w, sg = seed()
    wav = gui()[1]
    for a in sg.annotations:
        a['id'] = w._save_component_example(
            'USV', (a['f0'], a['f1'], a['t0'], a['t1'], a['mask']),
            blob_id=a['blob_id'])
    w._refresh_annotation_list()
    w._detect_harmonics()

    rebuild_annotations_csv(wav)

    rows = {str(r['blob_id']): r
            for r in read_blob_csv(annotations_csv_sibling_path(wav))}
    got = [rows[str(a['id'])].get('harmonic_n') for a in sg.annotations[:3]]
    assert [str(x) for x in got] == ['1', '2', '3'], got


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
