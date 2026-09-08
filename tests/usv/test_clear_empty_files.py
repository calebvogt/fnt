"""What "no detections" means when clearing the Audio list.

This exists because getting it wrong is silent. The first version treated a
recording holding only *rejections* as empty — the reviewer looked and said no,
so surely there is nothing there — and clearing dropped 40 of one project's 61
hard negatives, including a recording carrying 35 of them. Nothing was deleted
and no error appeared; the loss only surfaced when a training report disagreed
with the confirmed-mask gallery.

Rejections are supervision. They are the mistakes the model actually made, and
they train it as hard negatives.
"""
import os

import pytest

pytest.importorskip("PyQt5")


@pytest.fixture(scope="module")
def classifier():
    from PyQt5.QtWidgets import QApplication
    if QApplication.instance() is None:
        QApplication([])
    from fnt.usv.mad_pyqt import MADMainWindow
    return MADMainWindow._files_without_detections


def _window(counts_by_name, names):
    class W:
        audio_files = [f"/rec/{n}" for n in names]
        _file_count_cache = dict(counts_by_name)
    return W()


def _clearable(classifier, counts, names):
    return sorted(os.path.basename(p)
                  for p in classifier(_window(counts, names)))


def test_a_recording_with_only_rejections_is_kept(classifier):
    """The regression: 35 curated rejections must not read as 'empty'."""
    got = _clearable(classifier, {"r.wav": (0, 0, 35)}, ["r.wav"])
    assert got == []


def test_never_analyzed_is_clearable(classifier):
    """No cache entry at all — nothing has ever looked at this recording."""
    got = _clearable(classifier, {}, ["fresh.wav"])
    assert got == ["fresh.wav"]


def test_analyzed_and_silent_is_clearable(classifier):
    """An all-zero tuple means inference ran and found nothing."""
    got = _clearable(classifier, {"quiet.wav": (0, 0, 0)}, ["quiet.wav"])
    assert got == ["quiet.wav"]


def test_accepted_or_pending_calls_are_kept(classifier):
    counts = {"acc.wav": (3, 0, 0), "pend.wav": (0, 5, 0)}
    got = _clearable(classifier, counts, ["acc.wav", "pend.wav"])
    assert got == []


def test_the_mixed_case_end_to_end(classifier):
    counts = {
        "accepted.wav": (3, 0, 0),
        "pending.wav": (0, 5, 0),
        "rejections_only.wav": (0, 0, 35),
        "silent.wav": (0, 0, 0),
        # never_analyzed.wav deliberately absent from the cache
    }
    names = ["accepted.wav", "pending.wav", "rejections_only.wav",
             "silent.wav", "never_analyzed.wav"]
    assert _clearable(classifier, counts, names) == [
        "never_analyzed.wav", "silent.wav"]


def test_anything_a_person_touched_survives(classifier):
    """The invariant, stated directly: a non-zero count in ANY column stays."""
    for i in range(3):
        counts = [0, 0, 0]
        counts[i] = 1
        got = _clearable(classifier, {"x.wav": tuple(counts)}, ["x.wav"])
        assert got == [], f"column {i} was treated as empty"
