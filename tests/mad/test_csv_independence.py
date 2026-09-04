"""MAD reads and writes the .mad store; the CSV is an export it never needs.

The property this pins: delete every CSV, reopen the file, and nothing about
the review state changes. That was not true before — a rejection lived only in
the CSV, so deleting it made the rejection vanish from the recording while its
pixels sat in the store, unread.

Three separate guards had to move for this to hold, each of which returned
early before the store was ever consulted:

* the "nothing to load" short-circuit, which asked only about CSV rows and
  prediction crops;
* ``if not rows: return`` further down, reached before any store-derived row
  was seeded;
* ``_example_store_paths`` resolving through ``_active_review_wav_path``, which
  still names the *previous* recording while a file is loading.

Runs under pytest, or directly.
"""
import glob
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication
from scipy.io import wavfile

import fnt.usv.mad_pyqt as M
import fnt.usv.usv_detector.fnt_mask_store as MS
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR = 250_000
_APP = None


def _session(n_labels=3, reject_index=1, with_predictions=False):
    """A project with labels, one rejection, optionally some predictions."""
    root = tempfile.mkdtemp(prefix="mad_indep_")
    wav = os.path.join(root, "rec.wav")
    wavfile.write(wav, SR, (np.random.default_rng(0).normal(0, .05, SR * 2)
                            * 32767).astype(np.int16))
    proj = os.path.join(root, "p")
    create_mad_project(proj)
    # Keep the reference: a collected QApplication takes the process with it,
    # with no traceback to say why.
    global _APP
    _APP = QApplication.instance() or QApplication([])
    M.MADMainWindow._apply_dark_theme()
    w = M.MADMainWindow()
    w._activate_project(MADProjectConfig.load(proj))
    w._register_audio_files([wav])
    w._append_audio_paths([wav])
    w._wait_for_audio_load()
    w._ask_class_for_confirm = lambda k, c, l: 'USV'
    w._show_first_label_info = lambda: None
    sg = w.spectrogram
    for i in range(n_labels):
        sg._pending = np.zeros((sg.n_freq_bins, sg.n_time_frames), np.uint8)
        sg._pending[120 + 30 * i:140 + 30 * i, 700 + 300 * i:740 + 300 * i] = 1
        w._confirm_pending()
    if reject_index is not None:
        sg._selected_ann_idx = reject_index
        w._reject_current_pred()
    if with_predictions:
        h5 = MS.masks_sibling_path(wav)
        crops = []
        for b in (901, 902):
            m = np.zeros((10, 12), np.uint8)
            m[2:8, 3:9] = 1
            crops.append({'blob_id': b, 'mask': m, 'f_off': 300,
                          't_off': 1500 + b, 'score': 0.8, 'class': 'USV'})
        MS.write_pred_masks(h5, crops)
    return w, wav, root


def _reopen(w):
    w._loaded_wav_path = None
    w.audio_data = None
    w.current_file_idx = 0
    w._load_current_file()
    w._wait_for_audio_load()
    out = {}
    for a in w.spectrogram.annotations:
        k = a.get('status') or 'confirmed'
        out[k] = out.get(k, 0) + 1
    return out


def _delete_csvs(root):
    removed = glob.glob(os.path.join(root, "*_FNT_MAD_annotations.csv"))
    removed += glob.glob(os.path.join(root, "*_FNT_MAD_predictions.csv"))
    for c in removed:
        os.remove(c)
    return len(removed)


# ----------------------------------------------------------------------
def test_a_new_recording_gets_a_mad_store():
    _w, wav, _root = _session(n_labels=1, reject_index=None)
    assert MS.masks_sibling_path(wav).endswith(MS.MAD_SUFFIX)
    assert MS.read_format(MS.masks_sibling_path(wav)).get("format") == "FNT-MAD"


def test_labelling_creates_no_csv_at_all():
    """A review session touches the store and nothing else. The CSV is an
    export; producing one continuously is what let the two disagree."""
    _w, _wav, root = _session()
    assert glob.glob(os.path.join(root, "*.csv")) == [],         "labelling wrote a CSV"
    assert glob.glob(os.path.join(root, "*.mad")), "no store was written"


def test_review_state_is_identical_with_and_without_the_csv():
    """The headline property. An export is made and then removed, so this
    covers both directions: a CSV present must not change anything, and a CSV
    absent must not lose anything."""
    from fnt.usv.usv_detector.mad_csv_rebuild import rebuild_annotations_csv
    w, wav, root = _session()
    before = _reopen(w)
    assert before == {'confirmed': 2, 'rejected': 1}, before
    rebuild_annotations_csv(wav)
    assert glob.glob(os.path.join(root, "*.csv")), "export produced nothing"
    assert _reopen(w) == before, "an exported CSV changed what MAD shows"
    assert _delete_csvs(root) >= 1
    after = _reopen(w)
    assert after == before, f"{before} became {after} once the CSV was gone"


def test_a_rejection_survives_with_no_csv():
    """The specific case that used to disappear."""
    w, _wav, root = _session()
    _delete_csvs(root)
    assert _reopen(w).get('rejected') == 1


def test_pending_predictions_survive_with_no_csv():
    w, _wav, root = _session(with_predictions=True)
    _delete_csvs(root)
    got = _reopen(w)
    assert got.get('prediction') == 2, got


def test_a_file_whose_only_state_is_a_rejection_still_loads():
    """No CSV, no crops, one demoted example — every early-out had to learn
    about the store for this to survive."""
    w, _wav, root = _session(n_labels=1, reject_index=0)
    _delete_csvs(root)
    got = _reopen(w)
    assert got == {'rejected': 1}, got


def test_the_count_badges_come_from_the_store():
    w, wav, root = _session()
    _delete_csvs(root)
    assert w._csv_status_counts(wav) == (2, 0, 1)


def test_the_store_alone_can_regenerate_the_csv():
    from fnt.usv.usv_detector.mad_csv_rebuild import rebuild_annotations_csv
    w, wav, root = _session()
    _delete_csvs(root)
    rep = rebuild_annotations_csv(wav, recompute_features=True)
    assert (rep["accepted"], rep["rejected"]) == (2, 1), rep
    assert rep["kept_without_h5"] == 0
    assert len(glob.glob(os.path.join(root, "*_annotations.csv"))) == 1


def test_repacking_does_not_disturb_the_state():
    w, wav, root = _session()
    before = _reopen(w)
    rep = MS.repack_store(MS.masks_sibling_path(wav))
    assert rep["ok"], rep
    assert _reopen(w) == before


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
