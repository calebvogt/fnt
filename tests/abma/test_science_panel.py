"""The inspection column, driven headlessly.

Qt offscreen has no fonts, so these exercise state and signals rather than
pixels: does a roster appear for a population, does clicking a card select that
animal, do the drive bars pick up the decomposition the engine emits, and does
a frame that predates a field simply read as zero instead of crashing.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication            # noqa: E402
from PyQt5.QtTest import QTest                      # noqa: E402
from PyQt5.QtCore import Qt                         # noqa: E402

from fnt.abma.core.config import default_dynamics   # noqa: E402
from fnt.abma.core.presets import vole_anosmia      # noqa: E402
from fnt.abma.core.simulation import Simulation     # noqa: E402
from fnt.abma.gui.science_panel import (            # noqa: E402
    SciencePanel, DRIVES, CONDITION,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def running_sim():
    """A few steps of the real vole preset, so frames carry real drives."""
    cfg = vole_anosmia()
    cfg.days = 0.01
    sim = Simulation(cfg, trial_index=0)
    elapsed = 0.0
    for _ in range(30):
        sim.step(elapsed, cfg.dt)
        elapsed += cfg.dt
    return sim


@pytest.fixture
def panel(app, running_sim):
    p = SciencePanel()
    p.set_population(running_sim.agent_static())
    p.set_dynamics(default_dynamics())
    return p


def test_roster_has_one_card_per_animal(panel, running_sim):
    assert len(panel.roster.cards) == running_sim.n
    first = panel.roster.cards[0]
    assert first.meta["sexid"] == running_sim.agents[0].sexid


def test_clicking_a_card_selects_that_animal(panel, qtbot=None):
    picked = []
    panel.selected.connect(picked.append)
    QTest.mouseClick(panel.roster.cards[2], Qt.LeftButton)
    assert picked == [2]
    assert panel.selected_index() == 2
    assert panel.roster.cards[2].isChecked()
    assert not panel.roster.cards[0].isChecked()


def test_selection_shows_the_animals_identity(panel, running_sim):
    panel.select(1)
    meta = running_sim.agent_static()[1]
    assert meta["sexid"] in panel.title.text()
    assert "aggression" in panel.footer.text()


def test_drive_bars_read_the_decomposition(panel, running_sim):
    panel.select(0)
    for k in range(4):
        panel.update_frame(running_sim._frame(60.0 * k))
    labels = {label for label, _, _ in panel.drives.values}
    assert labels, "no drives reached the panel"
    # the vole preset runs with scent marking, so these are in play
    assert {"own scent", "exploration", "other animals"} <= labels
    assert any(value > 0 for _, value, _ in panel.drives.values)
    # the geometric home spring belongs to the non-scent path and must not
    # appear at all under a marking config
    assert "home range" not in labels


def test_a_drive_that_is_in_play_keeps_its_row_when_it_hits_zero(panel,
                                                                 running_sim):
    """Rows that appear and vanish as an animal moves are unreadable."""
    panel.select(0)
    frame = running_sim._frame(0.0)
    frame["drive_territory"] = np.full(running_sim.n, 0.4)
    panel.update_frame(frame)
    assert "rival scent" in {label for label, _, _ in panel.drives.values}
    frame["drive_territory"] = np.zeros(running_sim.n)
    panel.update_frame(frame)
    assert "rival scent" in {label for label, _, _ in panel.drives.values}


def test_traces_accumulate_across_frames(panel, running_sim):
    panel.select(0)
    for k in range(5):
        panel.update_frame(running_sim._frame(60.0 * k))
    plot = panel.pages[0].plot
    assert len(plot.history["drive_wander"]) == 5
    condition = panel.pages[1].plot
    assert len(condition.history["energy"]) == 5


def test_selecting_a_different_animal_clears_the_traces(panel, running_sim):
    panel.select(0)
    panel.update_frame(running_sim._frame(0.0))
    panel.select(1)
    assert len(panel.pages[0].plot.history["drive_wander"]) == 0


def test_roster_tracks_live_state(panel, running_sim):
    frame = running_sim._frame(60.0)
    frame["health"] = np.full(running_sim.n, 42.0)
    frame["activity"] = np.zeros(running_sim.n, int)
    panel.update_frame(frame)
    card = panel.roster.cards[0]
    assert card.health == pytest.approx(42.0)
    assert card.activity == 0


def test_a_frame_without_the_new_fields_is_tolerated(panel):
    """An old buffered frame (or a bare preview one) must not crash the panel."""
    panel.select(0)
    panel.update_frame({"x": np.zeros(8), "health": np.full(8, 90.0)})
    assert all(value == 0 for _, value, _ in panel.drives.values
               if value)  # nothing invented from a frame that lacks drives


def test_coupling_diagram_takes_the_dynamics_table(panel, running_sim):
    assert len(panel.coupling.rows) == len(default_dynamics())
    panel.select(0)
    panel.update_frame(running_sim._frame(0.0))
    assert set(panel.coupling.values) >= {key for key, _, _ in CONDITION}


def test_pages_switch_without_error(panel, running_sim):
    panel.select(0)
    panel.update_frame(running_sim._frame(0.0))
    shown = []
    for i in range(panel.picker.count()):
        panel.picker.setCurrentIndex(i)
        pages = panel.pages + [panel.coupling]
        shown.append([k for k, w in enumerate(pages) if not w.isHidden()])
    # exactly one page visible at a time, and each selection shows its own
    assert shown == [[i] for i in range(panel.picker.count())]


def test_every_named_drive_exists_in_the_record_schema():
    """The panel and the archive must agree on what a drive is called."""
    from fnt.abma.core.record import VALUE_FIELDS
    for key, _, _ in DRIVES:
        assert key in VALUE_FIELDS
    for key, _, _ in CONDITION:
        assert key in VALUE_FIELDS


def test_scrubbing_rebuilds_the_trace_instead_of_appending(panel, running_sim):
    """Dragging the timeline backwards must not draw a trace that doubles back.

    Live frames append; a scrubbed frame refills the history from the buffer up
    to that index, so the plot always reads left-to-right in run order.
    """
    buffer = [running_sim._frame(60.0 * k) for k in range(12)]
    panel.select(0)
    for fr in buffer:                       # watch it live to the end
        panel.update_frame(fr)
    plot = panel.pages[1].plot              # condition page, real values
    assert len(plot.history["energy"]) == 12

    panel.update_frame(buffer[3], buffer=buffer, index=3)   # scrub back
    assert len(plot.history["energy"]) == 4
    expected = [float(fr["energy"][0]) for fr in buffer[:4]]
    assert list(plot.history["energy"]) == pytest.approx(expected)

    panel.update_frame(buffer[9], buffer=buffer, index=9)   # scrub forward
    assert len(plot.history["energy"]) == 10
