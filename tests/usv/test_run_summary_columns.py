"""The run summary: two columns, and "accepted" rather than "confirmed".

Terminology first. "Confirmed" read as "I made a decision about this one",
which is true of accepts AND rejects, while the code only ever meant accepted.
Anything a reviewer sees now says accepted or rejected.

Layout second. Training and inference answer different questions — "is the
model any good?" and "how much work did this make for me?" — and as one prose
block the second was found only by reading past the first.
"""
import inspect

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QLabel  # noqa: E402

from fnt.usv.mad_pyqt import MADMainWindow, MADRunSummaryTable  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _texts(dlg):
    return " ".join(w.text() for w in dlg.findChildren(QLabel))


# ------------------------------------------------------ terminology
def test_the_training_summary_says_accepted_and_rejected():
    """The line above Training settings — the one that read '289 confirmed
    call(s) + 188 rejected as negatives'. Both decisions, both named."""
    src = inspect.getsource(MADMainWindow._split_preview_text)
    assert "accepted call(s)" in src
    assert "rejected call(s)" in src
    # Comments explain why the word was dropped; they are not shown to anyone.
    code = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    assert not any("confirmed" in ln for ln in code)


def test_no_user_facing_string_still_says_confirmed():
    """Docstrings and the internal 'confirmed' status key are exempt; text a
    reviewer can actually see is not."""
    import re
    import fnt.usv.mad_pyqt as M
    src = open(M.__file__, encoding="utf-8").read()
    # Strings that reach a widget: anything passed to setText/setToolTip/QLabel
    bad = []
    for m in re.finditer(r'(setToolTip|setText|QLabel)\(\s*("(?:[^"\\]|\\.)*")',
                         src):
        if "confirmed" in m.group(2).lower():
            bad.append(m.group(2)[:70])
    assert not bad, bad


def test_the_word_accepted_reaches_the_summary(qapp):
    dlg = MADRunSummaryTable(
        None, "Run complete", "Training complete",
        ["Trained on 289 accepted call(s).  ·  188 rejected call(s) as negatives"],
        [], infer_lines=[])
    assert "accepted call(s)" in _texts(dlg)
    assert "confirmed" not in _texts(dlg).lower()
    dlg.close()


# ---------------------------------------------------------- columns
def test_both_columns_appear_when_inference_ran(qapp):
    dlg = MADRunSummaryTable(
        None, "Run complete", "Training and Inference complete",
        ["Trained on 289 accepted call(s)."],
        [{'wav_path': 'a.wav', 'n_blobs': 3}],
        infer_lines=["Analyzed 314 recording(s) — 38496 pending detection(s)."])
    t = _texts(dlg)
    assert "TRAINING" in t and "INFERENCE" in t
    dlg.close()


def test_the_inference_column_is_absent_without_inference(qapp):
    """An empty heading is worse than no heading."""
    dlg = MADRunSummaryTable(
        None, "Run complete", "Training complete",
        ["Trained on 289 accepted call(s)."], [], infer_lines=[])
    t = _texts(dlg)
    assert "TRAINING" in t and "INFERENCE" not in t
    dlg.close()


def test_the_dialog_is_wider_when_it_has_two_columns(qapp):
    one = MADRunSummaryTable(None, "t", "h", ["x"], [], infer_lines=[])
    two = MADRunSummaryTable(None, "t", "h", ["x"], [], infer_lines=["y"])
    assert two.width() > one.width()
    one.close(); two.close()


def test_infer_lines_defaults_to_none_for_older_callers(qapp):
    dlg = MADRunSummaryTable(None, "t", "h", ["x"], [])
    assert "INFERENCE" not in _texts(dlg)
    dlg.close()


# ------------------------------------------------------- auto-eval off
def test_the_second_window_no_longer_opens_by_default():
    """A sweep nobody asked for, scored on the recordings just trained on."""
    src = inspect.getsource(MADMainWindow._maybe_auto_evaluate)
    assert 'value("mad/train/auto_eval", False, type=bool)' in src


def test_the_evaluate_dialog_checkbox_matches_that_default():
    src = inspect.getsource(MADMainWindow)
    import fnt.usv.mad_pyqt as M
    dsrc = inspect.getsource(M.MADEvalDialog)
    assert 'value("mad/train/auto_eval", False, type=bool)' in dsrc


# ------------------------------------------- count display order
"""Accepted, rejected, PENDING — pending last.

The stored tuple stays (accepted, pending, rejected); only the display order
changed. Pending is the number that says a recording still needs work, so it
reads last, where the eye stops.
"""


def test_the_file_row_text_puts_pending_last(qapp):
    from PyQt5.QtWidgets import QListWidgetItem
    it = QListWidgetItem()
    MADMainWindow._apply_file_row(
        _Stub(), it, "rec.wav", (7, 99, 3))      # (accepted, pending, rejected)
    assert it.text() == "rec.wav  (7, 3, 99)"


def test_a_file_with_no_detections_is_unchanged(qapp):
    from PyQt5.QtWidgets import QListWidgetItem
    it = QListWidgetItem()
    MADMainWindow._apply_file_row(_Stub(), it, "rec.wav", (0, 0, 0))
    assert it.text() == "rec.wav  (0 detections)"


def test_the_delegate_paints_in_the_same_order():
    """The row string is only the sizing fallback; the delegate draws the
    coloured numbers, and the two must not disagree."""
    import fnt.usv.mad_pyqt as M
    psrc = inspect.getsource(M.FileCountDelegate.paint)
    body = psrc[psrc.index("if has_counts"):]
    # The draw calls, not the tuple unpacking that precedes them.
    assert body.index("draw(str(r)") < body.index("draw(str(p)"), \
        "rejected must be drawn before pending"


def test_the_preview_header_puts_pending_last():
    src = inspect.getsource(MADMainWindow._update_view_header)
    assert src.index("rejected</span>") < src.index("pending</span>")


def test_the_tooltip_states_the_displayed_order():
    src = inspect.getsource(MADMainWindow)
    assert "(accepted, rejected, pending)" in src


class _Stub:
    """Minimum for _apply_file_row: no errors, nothing marked reviewed."""
    _file_errors = {}
    _review_done_cache = set()
