"""Accept/reject must not repeat work the advance already does.

Two costs removed, both measured on a 2,000-3,000 detection file:

* ``_accept_prediction`` ended with a full ``_refresh_annotation_list()`` —
  104 ms at 2,000 rows, 140 ms at 3,000 — while its only caller's tail
  (``_after_review_decision``) already restyles the single row that changed.
  The fast path existed and was being paid around.
* Advancing re-centred the view and invalidated the spectrogram cache
  unconditionally. Consecutive calls in a dense file are usually inside the
  view already, so that recomputed an identical image: an STFT, two
  percentiles and a LUT map per keystroke for no visible change.

The timer stays because the remaining cost is machine-dependent — the store
lives on an SMB share, and guessing a bottleneck from a developer machine is
how the previous round of this work missed it.
"""
import inspect

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fnt.usv.mad_pyqt import MADMainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ------------------------------------------------- redundant rebuild
def _code_lines(fn):
    """Source with comment-only lines dropped, so a note explaining why a call
    was removed does not read as the call still being there."""
    return [ln for ln in inspect.getsource(fn).splitlines()
            if not ln.strip().startswith("#")]


def test_accept_does_not_rebuild_the_whole_list():
    """The single-row fast path in the tail is what should update the list."""
    assert not any("_refresh_annotation_list" in ln
                   for ln in _code_lines(MADMainWindow._accept_prediction))


def test_the_tail_still_updates_the_row():
    """Dropping the rebuild must not leave the row stale."""
    src = inspect.getsource(MADMainWindow._after_review_decision)
    assert "_touch_annotation_rows" in src
    assert "_refresh_annotation_list" in src      # the fallback survives


def test_accept_still_redraws_the_canvas():
    """The mask colour has to change even though the list is not rebuilt."""
    src = inspect.getsource(MADMainWindow._accept_prediction)
    assert "_rebuild_confirmed_mask" in src and "sg.update()" in src


# ------------------------------------------------ spectrogram cache
@pytest.fixture
def view(qapp):
    """A stand-in with just the view state _show_time_in_view touches."""
    class Spec:
        view_start = 100.0
        view_end = 200.0          # a 100 s window
        total_duration = 1800.0

    class W:
        VIEW_EDGE_MARGIN = MADMainWindow.VIEW_EDGE_MARGIN
        _show_time_in_view = MADMainWindow._show_time_in_view

        def __init__(self):
            self.spectrogram = Spec()
            self.renders = 0
            self.syncs = 0

        def _invalidate_spec_cache(self):
            self.renders += 1

        def _sync_scrollbar_from_view(self):
            self.syncs += 1

    return W()


def test_a_call_already_on_screen_does_not_re_render(view):
    """The fix: advancing between visible calls must cost nothing.

    Re-rendering here is what made a decision take 611 ms on a 58 s window
    and 3.8 s on a wide one.
    """
    assert view._show_time_in_view(150.0) is False
    assert view.renders == 0 and view.syncs == 0
    assert (view.spectrogram.view_start, view.spectrogram.view_end) == (100.0, 200.0)


def test_a_call_off_screen_does_re_render(view):
    assert view._show_time_in_view(900.0) is True
    assert view.renders == 1 and view.syncs == 1
    assert view.spectrogram.view_start < 900.0 < view.spectrogram.view_end


def test_a_call_near_the_edge_is_recentred(view):
    """Otherwise you read a call with no context on one side of it."""
    assert view._show_time_in_view(197.0) is True       # inside, but 3% from the edge
    assert view.renders == 1


def test_the_margin_band_is_respected_on_both_sides(view):
    # 15% of a 100 s window = 15 s, so the clear band is 115..185.
    assert view._show_time_in_view(115.5) is False
    assert view._show_time_in_view(184.5) is False
    assert view._show_time_in_view(114.5) is True


def test_the_window_keeps_its_width_at_the_end_of_the_file(view):
    """Clamping at the tail must not silently zoom in."""
    view._show_time_in_view(1799.0)
    sg = view.spectrogram
    assert sg.view_end == pytest.approx(1800.0)
    assert sg.view_end - sg.view_start == pytest.approx(100.0)


def test_the_window_keeps_its_width_at_the_start(view):
    view.spectrogram.view_start, view.spectrogram.view_end = 500.0, 600.0
    view._show_time_in_view(1.0)
    sg = view.spectrogram
    assert sg.view_start == pytest.approx(0.0)
    assert sg.view_end - sg.view_start == pytest.approx(100.0)


def test_a_degenerate_window_is_not_divided_by(view):
    view.spectrogram.view_start = view.spectrogram.view_end = 5.0
    assert view._show_time_in_view(900.0) is False
    assert view.renders == 0


def test_selection_routes_through_the_guard():
    src = inspect.getsource(MADMainWindow._on_annotation_list_selected)
    assert "_show_time_in_view" in src
    assert "_invalidate_spec_cache" not in src


# ------------------------------------------------------------ timer
@pytest.fixture
def win(qapp):
    class W:
        SLOW_DECISION_MS = MADMainWindow.SLOW_DECISION_MS
        _decision_timer = MADMainWindow._decision_timer
        _mark = MADMainWindow._mark

        def __init__(self):
            self.logged = []

        def _log(self, m):
            self.logged.append(m)

    return W()


def test_a_fast_decision_logs_nothing(win):
    """Silence below the threshold keeps the Session Logs readable."""
    with win._decision_timer("Accept") as mark:
        mark("snapshot")
        mark("save")
    assert win.logged == []


def test_a_slow_decision_logs_a_phase_breakdown(win):
    win.SLOW_DECISION_MS = 0          # force it
    with win._decision_timer("Accept") as mark:
        mark("snapshot")
        mark("save")
        mark("advance")
    assert len(win.logged) == 1
    msg = win.logged[0]
    assert msg.startswith("Accept took ")
    for phase in ("snapshot", "save", "advance", "tail"):
        assert phase in msg


def test_phases_are_reported_as_durations_not_timestamps(win):
    """Cumulative numbers would make the last phase look like the whole cost."""
    import time
    win.SLOW_DECISION_MS = 0
    with win._decision_timer("Accept") as mark:
        time.sleep(0.02)
        mark("snapshot")
        mark("save")
    msg = win.logged[0]
    # 'save' happened immediately after 'snapshot', so its own slice is ~0 ms
    # even though the elapsed time at that point was ~20 ms.
    save_ms = float(msg.split("save ")[1].split(" ")[0].rstrip("]·").strip())
    assert save_ms < 10, msg


def test_the_timer_reports_even_when_the_body_raises(win):
    """A decision that blows up mid-way is exactly when the timing matters."""
    win.SLOW_DECISION_MS = 0
    with pytest.raises(ValueError):
        with win._decision_timer("Reject") as mark:
            mark("snapshot")
            raise ValueError("boom")
    assert win.logged and "Reject took" in win.logged[0]


def test_reject_is_timed_too():
    src = inspect.getsource(MADMainWindow._reject_current_pred)
    assert "_decision_timer" in src


# ------------------------------------------- deep-stack instrumentation
def test_marks_can_come_from_deep_in_the_call_stack(win):
    """The expensive steps are several frames below the handler.

    A timer that could only mark the handler's own statements reported
    "advance took 3842 ms" without saying which part of the advance, which
    cost a whole round trip to the user.
    """
    win.SLOW_DECISION_MS = 0

    def deep():
        win._mark("render[600s]")

    def middle():
        deep()

    with win._decision_timer("Accept"):
        middle()
    assert "render[600s]" in win.logged[0]


def test_marking_outside_a_timed_action_is_a_no_op(win):
    win._mark("stray")                     # must not raise or accumulate
    win.SLOW_DECISION_MS = 0
    with win._decision_timer("Accept") as mark:
        mark("real")
    assert "stray" not in win.logged[0]


def test_a_nested_timer_does_not_start_a_second_report(win):
    """Skip is timed, and an Accept must not produce two overlapping lines."""
    win.SLOW_DECISION_MS = 0
    with win._decision_timer("Accept") as mark:
        mark("one")
        with win._decision_timer("Skip") as inner:
            inner("two")
    assert len(win.logged) == 1
    assert win.logged[0].startswith("Accept took")
    assert "one" in win.logged[0] and "two" in win.logged[0]


def test_the_timer_state_is_cleared_after_each_action(win):
    win.SLOW_DECISION_MS = 0
    with win._decision_timer("Accept") as mark:
        mark("ZZfirst")
    assert getattr(win, '_t_marks', None) is None
    with win._decision_timer("Reject") as mark:
        mark("ZZsecond")
    assert len(win.logged) == 2
    # Distinctive names: single letters collide with 'took' and 'tail'.
    assert "ZZfirst" not in win.logged[1]      # no leakage between actions
    assert "ZZsecond" in win.logged[1]


def test_skip_is_timed_for_comparison():
    """Skip being instant is the observation that disproved the previous
    diagnosis; it has to be measurable in the same units."""
    src = inspect.getsource(MADMainWindow._shortcut_skip)
    assert "_decision_timer" in src


def test_the_render_mark_carries_the_window_width():
    """Cost scales with the window, not the call, so the number is the point."""
    src = inspect.getsource(MADMainWindow._invalidate_spec_cache)
    assert "render[" in src and "view_end - self.spectrogram.view_start" in src


def test_the_advance_reports_which_list_path_it_took():
    """A full rebuild and a single-row restyle differ by ~100 ms; the log must
    say which one happened rather than leaving it to be inferred."""
    src = inspect.getsource(MADMainWindow._after_review_decision)
    assert "list:REBUILD" in src and "list:fast" in src


def test_centring_uses_the_same_visibility_guard():
    """Skip advances through here too, so both paths must share the rule."""
    src = inspect.getsource(MADMainWindow._center_and_select_ann)
    assert "_show_time_in_view" in src
    assert "_invalidate_spec_cache" not in src


# ----------------------------------------- the actual cause: label count
"""``_touch_annotation_rows``'s tail re-read EVERY recording's store to relabel
the Run Training button. On a 230-file project over SMB that is ~230 network
opens per keystroke: 0.8-3.2 s measured, independent of how many detections the
file held, and absent from Skip (which never calls the tail). That was the
entire review lag."""


class _CountingStore:
    """Stands in for td_count, recording which stores were opened."""

    def __init__(self):
        self.opened = []

    def __call__(self, h5):
        self.opened.append(h5)
        return 1


@pytest.fixture
def counter(qapp, monkeypatch):
    import fnt.usv.usv_detector.fnt_mask_store as MS
    store = _CountingStore()
    monkeypatch.setattr(MS, 'td_count', store)
    monkeypatch.setattr(MS, 'masks_sibling_path', lambda p: str(p) + ".mad")

    class W:
        _training_label_count = MADMainWindow._training_label_count
        _invalidate_label_count = MADMainWindow._invalidate_label_count

        def __init__(self):
            self.files = [f"/rec/f{i}.wav" for i in range(230)]

        def _training_source_paths(self):
            return self.files

    return W(), store


def test_the_first_count_reads_every_store(counter):
    win, store = counter
    assert win._training_label_count() == 230
    assert len(store.opened) == 230


def test_a_second_count_reads_nothing(counter):
    """The keystroke path: 230 network opens down to zero."""
    win, store = counter
    win._training_label_count()
    store.opened.clear()
    assert win._training_label_count() == 230
    assert store.opened == []


def test_a_decision_re_reads_only_its_own_recording(counter):
    """One store, not 230 — and still the true count, not a guess."""
    win, store = counter
    win._training_label_count()
    store.opened.clear()
    win._invalidate_label_count("/rec/f7.wav")
    win._training_label_count()
    assert len(store.opened) == 1
    assert store.opened[0].endswith("f7.wav.mad")


def test_the_count_reflects_the_re_read(counter):
    """Caching must not freeze the number the button shows."""
    import fnt.usv.usv_detector.fnt_mask_store as MS
    win, store = counter
    win._training_label_count()
    MS.td_count = lambda h5: 5              # that file now holds 5 examples
    win._invalidate_label_count("/rec/f7.wav")
    assert win._training_label_count() == 229 + 5


def test_invalidating_everything_is_possible_but_explicit(counter):
    win, store = counter
    win._training_label_count()
    store.opened.clear()
    win._invalidate_label_count()           # no argument = all
    win._training_label_count()
    assert len(store.opened) == 230


def test_a_newly_added_recording_is_picked_up(counter):
    """A path not in the cache is read; no explicit clear needed on add."""
    win, store = counter
    win._training_label_count()
    store.opened.clear()
    win.files.append("/rec/new.wav")
    assert win._training_label_count() == 231
    assert len(store.opened) == 1


def test_an_unreadable_store_counts_zero_and_is_not_retried(counter):
    """A broken sidecar must not put the 230-open loop back per keystroke."""
    import fnt.usv.usv_detector.fnt_mask_store as MS
    win, _ = counter

    def boom(h5):
        raise OSError("gone")

    MS.td_count = boom
    assert win._training_label_count() == 0
    calls = []
    MS.td_count = lambda h5: calls.append(h5) or 1
    win._training_label_count()
    assert calls == []                      # zeros were cached, not re-read


def test_a_decision_invalidates_the_current_recording():
    src = inspect.getsource(MADMainWindow._after_review_decision)
    assert "_invalidate_label_count" in src


def test_saving_an_example_invalidates_it():
    src = inspect.getsource(MADMainWindow._save_component_example)
    assert "_invalidate_label_count" in src


def test_a_full_rebuild_invalidates_it_as_the_delete_backstop():
    src = inspect.getsource(MADMainWindow._refresh_annotation_list)
    assert "_invalidate_label_count" in src


def test_the_tail_steps_are_marked():
    """So a regression shows up in the log instead of being argued about."""
    src = inspect.getsource(MADMainWindow._touch_annotation_rows)
    for m in ("t:labelcount", "t:filelist", "t:marks", "t:gallery"):
        assert m in src


def test_the_threshold_sits_above_a_normal_decision():
    """After the fixes a decision runs ~200-400 ms, nearly all of it the store
    write to a network share. A threshold at the edge of perceptibility would
    log every keystroke and the line would stop carrying information; this one
    still catches the class of bug that was found this way (0.8-3.2 s)."""
    assert MADMainWindow.SLOW_DECISION_MS >= 500
    assert MADMainWindow.SLOW_DECISION_MS < 800


def test_the_diagnostic_toggle_still_forces_every_action(win):
    """The baseline case: comparing a fast action against a slow one needs the
    fast one logged, which the threshold otherwise hides."""
    win.SLOW_DECISION_MS = 0
    with win._decision_timer("Skip") as mark:
        mark("snapshot")
    assert win.logged and win.logged[0].startswith("Skip took")
