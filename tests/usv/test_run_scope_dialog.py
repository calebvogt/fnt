"""Re-analysing a recording is one decision, so it is one dialog.

Starting a batch used to raise two prompts back to back:

* "Re-run inference on these files?" — N of M files still hold pending
  (unreviewed) detections that a re-run would throw away;
* "Resume batch run?" — N of M files already have detections from this exact
  model at these settings, so re-analysing them burns hours to reproduce
  output that already exists.

Both opened with "N of M file(s) already have detections", both offered
proceed/cancel, and nothing distinguished *losing review work* from *repeating
compute*. Answering the first told you nothing about the second, and the
second arrived after you thought you had already confirmed.

``_confirm_run_scope`` states both costs once and offers the three answers
that exist: Resume, Redo all, Cancel.
"""
import os

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QCheckBox  # noqa: E402

from fnt.usv import mad_pyqt  # noqa: E402
from fnt.usv.mad_pyqt import MADMainWindow  # noqa: E402
from fnt.usv.usv_detector import mad_batch  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# --------------------------------------------------------------- fakes
class _Btn:
    def __init__(self, label, role):
        self.label, self.role = label, role

    def text(self):
        return self.label


class FakeBox:
    """Enough QMessageBox to record what the user was shown and told."""
    Question = 4
    Warning = 2
    AcceptRole = 0
    RejectRole = 1
    Yes = 0x4000
    No = 0x10000

    instances = []
    warn_calls = []
    warn_reply = 0x4000
    choose = staticmethod(lambda box: box.buttons[0])

    def __init__(self, parent=None):
        self.title = self.text_ = self.informative = self.detailed = ""
        self.buttons = []
        self.default = None
        self._clicked = None
        FakeBox.instances.append(self)

    # -- setters
    def setIcon(self, icon):
        self.icon = icon

    def setWindowTitle(self, t):
        self.title = t

    def setText(self, t):
        self.text_ = t

    def setInformativeText(self, t):
        self.informative = t

    def setDetailedText(self, t):
        self.detailed = t

    def addButton(self, label, role):
        b = _Btn(label, role)
        self.buttons.append(b)
        return b

    def setDefaultButton(self, b):
        self.default = b

    # -- interaction
    def exec_(self):
        self._clicked = FakeBox.choose(self)

    def clickedButton(self):
        return self._clicked

    def buttonRole(self, b):
        return None if b is None else b.role

    # -- the static form, used by the destructive branch
    @classmethod
    def warning(cls, parent, title, body, buttons=None, default=None):
        cls.warn_calls.append((title, body))
        return cls.warn_reply

    # -- helpers for tests
    def button(self, prefix):
        for b in self.buttons:
            if b.label.startswith(prefix):
                return b
        raise AssertionError(f"no {prefix!r} button in {self.labels()}")

    def labels(self):
        return [b.label for b in self.buttons]

    def whole(self):
        return "\n".join([self.title, self.text_, self.informative])


def pick(prefix):
    """Make the fake dialog answer with the button whose label starts here."""
    FakeBox.choose = staticmethod(lambda box: box.button(prefix))


class _Bar:
    def showMessage(self, m, *a):
        pass

    def clearMessage(self):
        pass


class Cfg:
    model_path = r"C:\models\20260913_150205_unet_n=413\weights.pt"
    threshold = 0.5
    min_blob_pixels = 40
    merge_consecutive = False
    merge_max_gap_s = 0.01
    merge_require_freq_overlap = True


@pytest.fixture
def win(qapp, monkeypatch):
    monkeypatch.setattr(mad_pyqt, "QMessageBox", FakeBox)
    FakeBox.instances = []
    FakeBox.warn_calls = []
    FakeBox.warn_reply = FakeBox.Yes
    pick("Resume")

    class W:
        _confirm_run_scope = MADMainWindow._confirm_run_scope
        _scan_predictions = MADMainWindow._scan_predictions
        _confirm_redetect = MADMainWindow._confirm_redetect
        _source_tag = MADMainWindow._source_tag
        _model_display_name = staticmethod(MADMainWindow._model_display_name)
        _confirm_overwrite_predictions = \
            MADMainWindow._confirm_overwrite_predictions

        def __init__(self, pending=(), done=()):
            self.pending = {os.path.normcase(p) for p in pending}
            self.done = {os.path.normcase(p) for p in done}
            self.chk_infer_redetect = QCheckBox()
            self.status_bar = _Bar()
            self.logged = []
            monkeypatch.setattr(
                mad_batch, "file_already_done",
                lambda p, s: os.path.normcase(p) in self.done)

        # predictions that exist at all (reviewed or not)
        def _file_has_predictions(self, w):
            return (os.path.normcase(w) in self.pending
                    or os.path.normcase(w) in self.done)

        def _file_has_pending_predictions(self, w):
            return os.path.normcase(w) in self.pending

        def _file_source_label(self, w):
            return ""

        def _batch_run_root(self):
            return None

        def _log(self, m):
            self.logged.append(m)

    return W


WAVS = [f"C:/rec/{i:03d}.wav" for i in range(10)]


def box():
    assert len(FakeBox.instances) == 1, \
        f"expected exactly one dialog, got {len(FakeBox.instances)}"
    return FakeBox.instances[0]


# ------------------------------------------------- the silent case
def test_a_clean_batch_asks_nothing(win):
    w = win()
    assert w._confirm_run_scope(Cfg(), WAVS) == WAVS
    assert FakeBox.instances == []


def test_an_empty_list_asks_nothing(win):
    w = win()
    assert w._confirm_run_scope(Cfg(), []) == []
    assert FakeBox.instances == []


# ------------------------------------------------- one dialog, both facts
def test_both_costs_are_stated_in_a_single_dialog(win):
    """The whole point: two prompts became one."""
    w = win(pending=WAVS[:4], done=WAVS[:2])
    w._confirm_run_scope(Cfg(), WAVS)
    b = box()
    assert "2 of 10" in b.text_          # already analyzed
    assert "4 of 10" in b.text_          # unreviewed detections
    assert "unet_n=413" in b.text_
    assert "0.5" in b.text_ and "40px" in b.text_


def test_the_dialog_says_what_survives(win):
    w = win(pending=WAVS[:4], done=WAVS[:2])
    w._confirm_run_scope(Cfg(), WAVS)
    info = box().informative
    assert "Accepted and Rejected" in info
    assert "painted / SAM labels" in info


def test_all_three_answers_are_offered(win):
    w = win(pending=WAVS[:4], done=WAVS[:2])
    w._confirm_run_scope(Cfg(), WAVS)
    assert box().labels() == ["Resume (8)", "Redo all (10)", "Cancel"]


def test_resume_is_the_default(win):
    """Skipping work already done is the answer that cannot lose anything."""
    w = win(pending=WAVS[:4], done=WAVS[:2])
    w._confirm_run_scope(Cfg(), WAVS)
    assert box().default.label.startswith("Resume")


# ------------------------------------------------- what each answer returns
def test_resume_returns_only_the_unanalyzed(win):
    w = win(done=WAVS[:3])
    pick("Resume")
    assert w._confirm_run_scope(Cfg(), WAVS) == WAVS[3:]
    assert any("skipping 3" in m for m in w.logged)


def test_redo_all_returns_everything(win):
    w = win(done=WAVS[:3])
    pick("Redo all")
    assert w._confirm_run_scope(Cfg(), WAVS) == WAVS
    assert any("Redoing all" in m for m in w.logged)


def test_cancel_returns_nothing(win):
    w = win(pending=WAVS[:2], done=WAVS[:1])
    pick("Cancel")
    assert w._confirm_run_scope(Cfg(), WAVS) == []
    assert any("cancelled" in m for m in w.logged)


def test_closing_the_dialog_counts_as_cancel(win):
    w = win(done=WAVS[:1])
    FakeBox.choose = staticmethod(lambda b: None)
    assert w._confirm_run_scope(Cfg(), WAVS) == []


# ------------------------------------------------- only one of the two facts
def test_nothing_analyzed_yet_means_no_resume_question(win):
    """Different model, or a first pass: 'resume' would be meaningless."""
    w = win(pending=WAVS[:3])
    pick("Analyze")
    assert w._confirm_run_scope(Cfg(), WAVS) == WAVS
    b = box()
    assert b.labels() == ["Analyze 10 file(s)", "Cancel"]
    assert "already have detections" not in b.text_


def test_fully_reviewed_files_say_nothing_is_at_risk(win):
    """Done but no pending: every detection on them has been judged."""
    w = win(done=WAVS[:3])
    w._confirm_run_scope(Cfg(), WAVS)
    assert "No unreviewed work would be lost" in box().informative
    assert "Accepted and Rejected" not in box().informative


def test_a_single_file_still_warns_about_unreviewed_work(win):
    """The resume scan is skipped for one file — the overwrite warning is
    not. That asymmetry was the reason for two separate prompts."""
    w = win(pending=WAVS[:1], done=WAVS[:1])
    pick("Analyze")
    assert w._confirm_run_scope(Cfg(), WAVS[:1]) == WAVS[:1]
    assert "already have detections" not in box().text_
    assert "pending (unreviewed)" in box().text_


# ------------------------------------------------- the file list
def test_every_affected_file_is_listed_not_just_eight(win):
    """Behind Details a full list costs nothing, and 'which files?' is the
    first question a number this size raises."""
    w = win(pending=WAVS)
    pick("Analyze")
    w._confirm_run_scope(Cfg(), WAVS)
    for p in WAVS:
        assert os.path.basename(p) in box().detailed


def test_the_list_names_the_files_at_risk_when_there_are_any(win):
    w = win(pending=WAVS[:2], done=WAVS[:5])
    w._confirm_run_scope(Cfg(), WAVS)
    assert "unreviewed" in box().detailed.splitlines()[0]


# ------------------------------------------------- naming the model
"""``RunSettings.model_name`` is ``Path(model_path).stem``, and every run
writes ``weights.pt`` — so the dialog offered to skip files "already analyzed
by 'weights'", which is true of every model ever trained here. The matching key
has to stay the stem (it is written into each prediction's attributes), so only
the wording changes: the run directory is the part that identifies a model."""


def test_the_model_is_named_by_its_run_not_weights_pt(win):
    w = win(done=WAVS[:2])
    w._confirm_run_scope(Cfg(), WAVS)
    assert "20260913_150205_unet_n=413" in box().text_
    assert "'weights'" not in box().text_


def test_a_checkpoint_with_its_own_name_keeps_it():
    assert MADMainWindow._model_display_name(
        r"C:\m\best_so_far.pt") == "best_so_far"


def test_no_model_path_is_not_guessed_at():
    assert MADMainWindow._model_display_name("") == "(unknown)"


# ------------------------------------------------- re-detect from scratch
def test_redetect_asks_only_the_destructive_question(win):
    """'Redo everything' is the point of the tick-box, so 'already analyzed'
    is not a reason to skip anything — there is no resume question to merge."""
    w = win(pending=WAVS[:2], done=WAVS[:5])
    w.chk_infer_redetect.setChecked(True)
    assert w._confirm_run_scope(Cfg(), WAVS) == WAVS
    assert FakeBox.instances == []
    title, body = FakeBox.warn_calls[-1]
    assert title == "Re-detect from scratch?"
    assert "Accepted and Rejected decision" in body


def test_declining_the_redetect_warning_cancels(win):
    w = win(done=WAVS[:5])
    w.chk_infer_redetect.setChecked(True)
    FakeBox.warn_reply = FakeBox.No
    assert w._confirm_run_scope(Cfg(), WAVS) == []


def test_redetect_with_nothing_to_lose_asks_nothing(win):
    w = win()
    w.chk_infer_redetect.setChecked(True)
    assert w._confirm_run_scope(Cfg(), WAVS) == WAVS
    assert FakeBox.warn_calls == []


# ------------------------------------------------- the pre-training prompt
"""Train + Inference has to ask before the model it will run exists, so it
keeps its own smaller prompt. It shares the destructive branch rather than
carrying a second copy of that wording."""


def test_the_training_path_still_asks_before_it_starts(win):
    w = win(pending=WAVS[:2])
    assert w._confirm_overwrite_predictions(WAVS) is True
    title, body = FakeBox.warn_calls[-1]
    assert title == "Re-run inference on these files?"
    assert "2 of 10" in body


def test_the_training_path_is_silent_when_nothing_is_pending(win):
    w = win(done=WAVS)
    assert w._confirm_overwrite_predictions(WAVS) is True
    assert FakeBox.warn_calls == []


def test_the_training_path_shares_the_redetect_wording(win):
    w = win(pending=WAVS[:3])
    w.chk_infer_redetect.setChecked(True)
    assert w._confirm_overwrite_predictions(WAVS) is True
    assert FakeBox.warn_calls[-1][0] == "Re-detect from scratch?"


def test_declining_the_training_prompt_says_no(win):
    w = win(pending=WAVS[:2])
    FakeBox.warn_reply = FakeBox.No
    assert w._confirm_overwrite_predictions(WAVS) is False


# ------------------------------------------------- wiring
def _code(fn):
    import inspect
    src = inspect.getsource(fn)
    body = src.split('"""')[-1]
    return "\n".join(ln for ln in body.splitlines()
                     if not ln.strip().startswith("#"))


def test_the_old_two_step_gate_is_gone():
    import inspect
    src = inspect.getsource(MADMainWindow)
    assert "def _resume_filter" not in src


def test_starting_a_run_goes_through_the_merged_gate():
    assert "self._confirm_run_scope(cfg, wav_paths)" in \
        _code(MADMainWindow._start_inference)


def test_run_inference_no_longer_confirms_twice():
    """The second dialog came from _on_deploy_infer asking before
    _start_inference asked again."""
    assert "_confirm_overwrite_predictions" not in \
        _code(MADMainWindow._on_deploy_infer)


def test_a_chained_post_training_run_still_asks_nothing():
    """One button, then the user walks away."""
    assert "skip_scope_prompt=True" in \
        _code(MADMainWindow._run_post_training_inference)
