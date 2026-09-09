"""Agent tracks: one animal, one fading line behind it.

Both views used to draw the last handful of frames as a *scatter*, pooled
across the whole cohort — around each animal that reads as a smear of identical
dots with no direction, no usable history, and nothing marking where the animal
actually is. A polyline per animal has a head and a tail; these tests pin that
shape rather than any particular styling.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fnt.abma.core.compose import design           # noqa: E402
from fnt.abma.gui.abma_canvas import ArenaCanvas   # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _walk(view, steps=40, n=6, seed=0):
    """Drive a view with a random walk and return the final positions."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(2.0, 20.0, (n, 2))
    sex = np.array([1.0] * (n // 2) + [0.0] * (n - n // 2))
    for _ in range(steps):
        xy = xy + rng.normal(0, 0.05, (n, 2))
        view.update_agents(xy[:, 0], xy[:, 1], sex, heading=np.zeros(n),
                           alive=np.ones(n, bool))
    return xy


@pytest.fixture
def canvas_2d(qapp):
    cfg = design(preset="voleterra", males=3, females=3, days=0.01)
    c = ArenaCanvas()
    c.set_arena(cfg.arena)
    return c


@pytest.fixture
def view_3d(qapp):
    pytest.importorskip("pyqtgraph.opengl")
    from fnt.abma.gui.pg_canvas import Arena3DView

    cfg = design(preset="voleterra", males=3, females=3, days=0.01)
    v = Arena3DView()
    v.set_arena(cfg.arena)
    return v


# --------------------------------------------------------------------------- #
# Shape of the track
# --------------------------------------------------------------------------- #
def test_2d_keeps_one_track_per_animal(canvas_2d):
    _walk(canvas_2d, steps=40)
    assert len(canvas_2d._tracks) == 6
    assert all(len(t) == 40 for t in canvas_2d._tracks)
    # one artist per animal, not one per remembered frame
    assert len(canvas_2d._trail_artists) == 6


def test_3d_keeps_one_line_per_animal(view_3d):
    _walk(view_3d, steps=40)
    assert len(view_3d._tracks) == 6
    assert len(view_3d._trail_lines) == 6
    assert view_3d._trail_lines[0].pos.shape == (40, 3)


def test_3d_track_fades_from_tail_to_head(view_3d):
    _walk(view_3d, steps=30)
    alpha = view_3d._trail_lines[0].color[:, 3]
    assert alpha[0] < alpha[-1], "the track does not fade toward its tail"
    assert np.all(np.diff(alpha) >= -1e-9), "fade is not monotonic"


def test_3d_marks_the_current_position(view_3d):
    """The head of the track must be unambiguous without inspecting anything."""
    xy = _walk(view_3d, steps=25)
    head = np.asarray(view_3d._now.pos)
    assert len(head) == 6
    assert np.allclose(head[:, 0], xy[:, 0])
    assert np.allclose(head[:, 1], xy[:, 1])


def test_the_track_head_is_the_latest_position(view_3d, canvas_2d):
    xy3 = _walk(view_3d, steps=20)
    xy2 = _walk(canvas_2d, steps=20)
    assert np.allclose(view_3d._trail_lines[0].pos[-1, :2], xy3[0])
    assert np.allclose(np.asarray(canvas_2d._tracks[0])[-1], xy2[0])


# --------------------------------------------------------------------------- #
# Length
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("view_name", ["canvas_2d", "view_3d"])
def test_a_track_is_capped_at_the_requested_length(view_name, request):
    view = request.getfixturevalue(view_name)
    view.set_trail_length(15)
    _walk(view, steps=60)
    assert all(len(t) == 15 for t in view._tracks)


@pytest.mark.parametrize("view_name", ["canvas_2d", "view_3d"])
def test_shortening_a_track_keeps_the_most_recent_steps(view_name, request):
    view = request.getfixturevalue(view_name)
    _walk(view, steps=40)
    newest = list(view._tracks[0])[-5:]
    view.set_trail_length(5)
    assert list(view._tracks[0]) == newest


@pytest.mark.parametrize("view_name", ["canvas_2d", "view_3d"])
def test_tracks_can_be_hidden_without_being_forgotten(view_name, request):
    view = request.getfixturevalue(view_name)
    _walk(view, steps=20)
    view.set_trails_visible(False)
    _walk(view, steps=5)
    assert len(view._tracks[0]) == 25, "history was dropped when hidden"
    view.set_trails_visible(True)
    _walk(view, steps=1)
    assert len(view._tracks[0]) == 26


def test_2d_draws_nothing_while_tracks_are_hidden(canvas_2d):
    _walk(canvas_2d, steps=10)
    canvas_2d.set_trails_visible(False)
    _walk(canvas_2d, steps=2)
    assert canvas_2d._trail_artists == []


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def test_3d_clearing_playback_forgets_the_tracks(view_3d):
    _walk(view_3d, steps=10)
    view_3d.clear_playback()
    assert view_3d._tracks == []
    assert view_3d._trail_lines == []


@pytest.mark.parametrize("view_name", ["canvas_2d", "view_3d"])
def test_a_growing_roster_gets_its_own_tracks(view_name, request):
    """A protocol event can add animals mid-run; they need tracks too."""
    view = request.getfixturevalue(view_name)
    _walk(view, steps=5, n=4)
    _walk(view, steps=5, n=7, seed=1)
    assert len(view._tracks) == 7


def test_a_single_frame_draws_no_line_yet(view_3d):
    """One point is not a path; it must not raise or draw a degenerate line."""
    view_3d.update_agents(np.array([1.0]), np.array([1.0]), np.array([1.0]),
                          heading=np.zeros(1), alive=np.ones(1, bool))
    assert view_3d._trail_lines[0].pos.shape[0] == 0
    assert len(np.asarray(view_3d._now.pos)) == 1


# --------------------------------------------------------------------------- #
# Window wiring
# --------------------------------------------------------------------------- #
def test_the_toolbar_cycles_track_length_across_both_views(qapp):
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow

    win = ABMAWindow()
    try:
        win._load_config(design(preset="voleterra", males=3, females=3,
                                days=0.02))
        win._rebuild_preview()
        seen = []
        for _ in range(len(win._TRAIL_LENGTHS)):
            win._cycle_trail()
            n, _label = win._TRAIL_LENGTHS[win._trail_state]
            seen.append(n)
            assert win.view_2d._trails_visible == (n > 0)
            if win.view_3d is not None:
                assert win.view_3d._trails_visible == (n > 0)
                assert win.view_3d._trail_len == max(2, n)
        assert 0 in seen and 1000 in seen
    finally:
        win.close()
