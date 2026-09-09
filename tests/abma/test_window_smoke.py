"""The ABMA window builds, previews, and drives the new inspection column.

Qt offscreen has no fonts, so this asserts on structure and state rather than
pixels. It exists because the science panel is a third splitter pane wired into
selection, preview and run paths — the kind of change that breaks window
construction long before it breaks a unit test.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication          # noqa: E402

from fnt.abma.core.presets import get_preset      # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def window(app):
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow
    w = ABMAWindow()
    yield w
    w.close()


def test_window_builds_with_three_panes(window):
    assert window._split.count() == 3
    assert window.science is not None


def test_loading_the_vole_preset_populates_the_roster(window):
    window._load_config(get_preset("Prairie vole — anosmia"))
    window._rebuild_preview()
    assert len(window.science.roster.cards) == 8
    assert window.science.coupling.rows, "dynamics never reached the diagram"


def test_preview_ticks_feed_the_panel(window):
    window._load_config(get_preset("Prairie vole — anosmia"))
    window._rebuild_preview()
    window.science.select(0)
    for _ in range(6):
        window._preview_tick()
    plot = window.science.pages[0].plot
    assert len(plot.history["drive_wander"]) >= 5
    assert window.science.drives.values


def test_selecting_in_the_arena_selects_in_the_panel(window):
    window._load_config(get_preset("Prairie vole — anosmia"))
    window._rebuild_preview()
    window._select_agent(3)
    assert window.science.selected_index() == 3
    assert window.inspector.selected_index() == 3
    window._select_agent(-1)
    assert window.science.selected_index() is None


def test_roster_selection_does_not_pop_the_hover_card(window):
    window._load_config(get_preset("Prairie vole — anosmia"))
    window._rebuild_preview()
    window.inspector.hide()
    window._select_agent(2, from_roster=True)
    assert window.science.selected_index() == 2
    assert not window.inspector.isVisible()


def test_territory_map_toggle_switches_rasterising_on_and_off(window):
    window._load_config(get_preset("Prairie vole — anosmia"))
    window._rebuild_preview()
    window.btn_scent.setChecked(True)
    assert window._preview_sim.emit_scent_map is True
    for _ in range(3):
        window._preview_tick()
    assert window.view_2d._scent_visible is True
    window.btn_scent.setChecked(False)
    assert window._preview_sim.emit_scent_map is False


def test_blank_experiment_has_no_animals_and_does_not_crash(window):
    from fnt.abma.core.config import blank_experiment
    window._load_config(blank_experiment())
    window._rebuild_preview()
    assert window.science.roster.cards == {}
    window._preview_tick()


def test_the_3d_view_still_takes_a_frame_carrying_the_new_fields(window):
    """The territory map is 2D-only; the 3D view must not be handed it.

    `_push_frame` gates the extra arguments on the view supporting them, so
    this guards against a frame gaining a field and breaking the GL path.
    """
    from fnt.abma.gui.abma_main_pyqt import _HAVE_GL
    if not _HAVE_GL:
        pytest.skip("PyOpenGL not available; the 3D view is not built")
    window._load_config(get_preset("Prairie vole — anosmia"))
    window._rebuild_preview()
    window.btn_scent.setChecked(True)
    window._preview_tick()
    assert "scent_rgba" in window._last_frame
    window._set_view(1)                      # switch to the GL scene
    try:
        window._push_frame(window.view_3d, window._last_frame)
    finally:
        window._set_view(0)
