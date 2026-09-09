"""Open the ABMA window on a given config and start the run — no clicking.

The point is a loop that a person and an assistant can share: describe a run,
have it built and started on screen, watch it, and say what looks wrong. Doing
that through the normal GUI means loading a preset, editing the population
table, setting the duration, and clicking through a confirmation — six manual
steps between a sentence and a moving arena, none of which the person watching
wanted to perform.

``watch(cfg)`` collapses them. It builds the real :class:`ABMAWindow` — the same
window, the same views, the same inspection column — loads the config, turns on
the things worth looking at (the territory map, the animal roster), and starts
the run once the window is actually on screen.

It is deliberately *not* a headless mode. Nothing here bypasses the engine or
draws a simplified picture: what you watch is the run that is being written to
disk, and closing the window leaves a normal project folder behind.
"""
from __future__ import annotations

import os

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from ..core.compose import summary
from ..core.config import ExperimentConfig


def build_window(cfg: ExperimentConfig, out_dir: str | None = None,
                 territory_map: bool = True, select: int | None = 0):
    """An ABMAWindow loaded with ``cfg`` and set up for watching.

    Returns the window without starting anything, so a caller can inspect or
    adjust it first. ``select`` pre-selects an animal in the roster so the
    drive panel has something to show from the first frame.
    """
    from .abma_main_pyqt import ABMAWindow

    win = ABMAWindow()
    win._load_config(cfg)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        win.in_outdir.setText(out_dir)
    if territory_map:
        # the emergent territory mosaic is the thing worth watching in a
        # scent-marking run, and it is off by default
        win.btn_scent.setChecked(bool(cfg.scent.enabled))
    win._rebuild_preview()
    if select is not None and win.science.roster.cards:
        win._select_agent(select, from_roster=True)
    return win


def watch(cfg: ExperimentConfig, out_dir: str | None = None,
          autostart: bool = True, territory_map: bool = True,
          exec_: bool = True) -> int:
    """Show the window on ``cfg`` and (by default) start the run.

    ``autostart=False`` opens it loaded but idle, for reviewing a design before
    committing the compute. ``exec_=False`` returns without entering the Qt
    event loop, which is what a test wants.
    """
    app = QApplication.instance() or QApplication([])
    win = build_window(cfg, out_dir, territory_map=territory_map)
    win.show()
    win._append_log(summary(cfg))
    if autostart:
        project_dir = win._resolve_run_dir(cfg, ask=False)
        # start after the window is actually mapped, so the first frames have
        # somewhere to draw and the user sees the arena before it moves
        QTimer.singleShot(150, lambda: win._start_run(cfg, project_dir))
    if not exec_:
        return win
    return app.exec_()
