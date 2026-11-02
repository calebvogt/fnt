"""The .mad store is what MAD reads and writes; the CSV is an export of it.

Three things had to be true before the store could be the only copy of a
recording's labels, and each failed quietly rather than loudly:

* **A failed write looked like a successful one.** 28 call sites wrap store
  calls in ``except Exception: pass``, which is survivable while a parallel CSV
  holds the same facts and is not once the store is the record. HDF5 takes a
  whole-file lock, so any other program holding the file open makes every write
  fail -- a real, recoverable, and previously invisible condition.
* **Deleting reclaimed nothing.** Measured: all 200 examples removed from a
  3.7 MB store left it at 3.7 MB. Fine for a cache, not for a document.
* **Review state lived in the CSV.** A rejection is recorded in the store as a
  demoted example, but nothing read it back, so deleting the CSV made
  rejections vanish from the file.

Runs under pytest, or directly.
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

import fnt.usv.usv_detector.fnt_mask_store as MS

META = {'class': 'USV', 'source_wav': 'a.wav', 'sample_rate': 250_000,
        'nperseg': 512, 'noverlap': 384, 'nfft': 1024}


def _store(n=20):
    d = tempfile.mkdtemp(prefix="mad_can_")
    p = os.path.join(d, "s.h5")
    for i in range(n):
        _add(p, 'e%d' % i)
    return p


def _add(p, eid, **meta):
    m = dict(META)
    m.update(meta)
    return MS.td_save_example(
        p, np.random.rand(48, 48).astype(np.float32),
        (np.random.rand(48, 48) > 0.5).astype(np.uint8), m, eid)


# ----------------------------------------------------------------------
# A failed write is reported even when the caller swallows it
# ----------------------------------------------------------------------
def test_a_successful_write_reports_nothing():
    seen = []
    MS.set_write_failure_hook(lambda *a: seen.append(a))
    try:
        _store(3)
    finally:
        MS.set_write_failure_hook(None)
    assert seen == []


def test_a_locked_file_reports_through_the_hook():
    """The store reports before the exception propagates, so a call site that
    swallows the error cannot swallow the notification with it."""
    p = _store(3)
    holder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_h5_holder.py")
    with open(holder, "w") as f:
        f.write("import sys, time, h5py\n"
                "with h5py.File(sys.argv[1], 'r') as f:\n"
                "    print('open', flush=True)\n"
                "    time.sleep(5)\n")
    seen = []
    MS.set_write_failure_hook(lambda op, path, exc: seen.append((op, exc)))
    proc = subprocess.Popen([sys.executable, holder, p],
                            stdout=subprocess.PIPE, text=True)
    try:
        proc.stdout.readline()
        time.sleep(0.3)
        try:
            _add(p, 'blocked')          # what a GUI call site does...
        except Exception:
            pass                        # ...and then swallows
    finally:
        proc.wait()
        MS.set_write_failure_hook(None)
        os.remove(holder)
    assert len(seen) == 1, "a failed write was invisible"
    assert seen[0][0] == 'td_save_example'
    assert MS.is_locked_error(seen[0][1]), seen[0][1]


def test_a_broken_hook_does_not_mask_the_real_error():
    """Reporting is a courtesy; the caller must still see what went wrong."""
    d = tempfile.mkdtemp()
    blocker = os.path.join(d, "notadir")
    open(blocker, "wb").close()
    doomed = os.path.join(blocker, "s.h5")     # parent is a file, not a dir

    def bad(*_a):
        raise RuntimeError("hook is broken")

    MS.set_write_failure_hook(bad)
    try:
        raised = None
        try:
            _add(doomed, 'nope')
        except Exception as e:
            raised = e
        assert raised is not None, "the write should have failed"
        assert "hook is broken" not in str(raised), \
            "the hook's own error replaced the real one"
    finally:
        MS.set_write_failure_hook(None)


# ----------------------------------------------------------------------
# Durability
# ----------------------------------------------------------------------
def test_deleting_alone_reclaims_nothing():
    """The premise for repack existing at all."""
    p = _store(40)
    before = os.path.getsize(p)
    for i in range(40):
        MS.td_delete(p, 'e%d' % i)
    assert os.path.getsize(p) >= before - 2048


def test_repack_shrinks_the_file_and_keeps_the_data():
    p = _store(40)
    for i in range(0, 40, 2):
        MS.td_delete(p, 'e%d' % i)
    MS.set_grid_attrs(p, sample_rate=250_000, nperseg=512, noverlap=384,
                      nfft=1024)
    kept = set(MS.td_list_ids(p))
    rep = MS.repack_store(p)
    assert rep["ok"] and rep["reclaimed"] > 0, rep
    assert set(MS.td_list_ids(p)) == kept
    assert MS.get_grid_attrs(p).get("nperseg") == 512


def test_repack_leaves_a_backup():
    p = _store(5)
    MS.repack_store(p)
    assert os.path.isfile(p + MS.BACKUP_SUFFIX)


def test_repack_of_a_missing_file_is_not_an_error():
    rep = MS.repack_store(os.path.join(tempfile.mkdtemp(), "nope.h5"))
    assert rep["ok"] is False and rep["before"] == 0


def test_backup_is_a_readable_store():
    p = _store(4)
    b = MS.backup_store(p)
    assert b and os.path.isfile(b)
    assert set(MS.td_list_ids(b)) == set(MS.td_list_ids(p))


# ----------------------------------------------------------------------
# The .mad container
# ----------------------------------------------------------------------
def _wav(d, name):
    p = os.path.join(d, name + ".wav")
    open(p, "wb").close()
    return p


def test_a_new_recording_gets_a_mad_store():
    d = tempfile.mkdtemp()
    p = MS.masks_sibling_path(_wav(d, "fresh"))
    assert os.path.basename(p) == "fresh_FNT.mad", p


def test_every_older_naming_keeps_being_used():
    """Existing projects hold thousands; starting a second store beside a
    populated one would split a recording's labels across two files. ``.mad``
    is in here because it was briefly the naming — files exist with it."""
    for suffix in MS.LEGACY_MAD_SUFFIXES:
        d = tempfile.mkdtemp()
        w = _wav(d, "old")
        older = os.path.join(d, "old" + suffix)
        MS.set_grid_attrs(older, sample_rate=250_000, nperseg=512)
        assert MS.masks_sibling_path(w) == older, suffix


def test_the_newest_naming_wins_when_several_exist():
    d = tempfile.mkdtemp()
    w = _wav(d, "many")
    for suffix in (MS.MAD_SUFFIX,) + MS.LEGACY_MAD_SUFFIXES:
        MS.set_grid_attrs(os.path.join(d, "many" + suffix), sample_rate=250_000)
    assert MS.masks_sibling_path(w).endswith(MS.MAD_SUFFIX)


def test_migration_renames_rather_than_rewrites():
    """A rename, not a rewrite — the bytes are already in the right format."""
    for suffix in MS.LEGACY_MAD_SUFFIXES:
        d = tempfile.mkdtemp()
        w = _wav(d, "old")
        older = os.path.join(d, "old" + suffix)
        MS.set_grid_attrs(older, sample_rate=250_000, nperseg=512)
        _add(older, 'keepme')
        new = MS.migrate_to_mad_suffix(w)
        assert new and new.endswith(MS.MAD_SUFFIX), suffix
        assert not os.path.isfile(older)
        assert 'keepme' in MS.td_list_ids(new)
        assert MS.migrate_to_mad_suffix(w) is None      # idempotent


def test_migration_refuses_to_clobber_an_existing_mad_file():
    d = tempfile.mkdtemp()
    w = _wav(d, "both")
    MS.set_grid_attrs(os.path.join(d, "both" + MS.LEGACY_MASKS_SUFFIX),
                      sample_rate=250_000)
    MS.set_grid_attrs(os.path.join(d, "both" + MS.MAD_SUFFIX),
                      sample_rate=250_000)
    assert MS.migrate_to_mad_suffix(w) is None
    assert os.path.isfile(os.path.join(d, "both" + MS.LEGACY_MASKS_SUFFIX))


def test_a_store_says_what_it_is():
    p = _store(2)
    MS.set_grid_attrs(p, sample_rate=250_000, nperseg=512)
    fmt = MS.read_format(p)
    assert fmt.get("format") == MS.FORMAT_NAME
    assert fmt.get("format_version") == MS.FORMAT_VERSION


def test_every_naming_is_recognised():
    assert MS.is_mad_store("x_FNT.mad")
    assert MS.is_mad_store("x.mad")
    assert MS.is_mad_store("x_FNT_masks.h5")
    assert not MS.is_mad_store("x.csv")


# ----------------------------------------------------------------------
# Cheap listings, used by the count badges
# ----------------------------------------------------------------------
def test_prediction_ids_list_without_reading_arrays():
    p = _store(2)
    crops = [{"blob_id": i, "mask": np.ones((6, 8), np.uint8),
              "f_off": 100, "t_off": 200 + i, "score": 0.5} for i in range(5)]
    MS.write_pred_masks(p, crops)
    assert sorted(MS.list_pred_ids(p), key=int) == ['0', '1', '2', '3', '4']


def test_prediction_ids_of_a_store_with_none():
    assert MS.list_pred_ids(_store(2)) == []


if __name__ == "__main__":
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
