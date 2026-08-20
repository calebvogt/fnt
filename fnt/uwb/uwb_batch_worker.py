"""Headless worker that runs ONE queued UWB export in its own process.

Invoked by the Export Queue as::

    python -m fnt.uwb.uwb_batch_worker <job.json>

Running each trial in a separate process is what makes an unattended batch
survivable. A UWB export is a long, native-heavy pipeline (pandas/numpy
smoothing, multi-GB CSV writes, matplotlib + OpenCV rendering for hours); a
memory-safety fault anywhere in that native stack takes down the whole
interpreter, and when every trial shared one process a single fault killed the
entire remaining queue. Here the damage is bounded to one trial: the worker dies
with a non-zero exit code, the parent marks that job Failed and moves on. Each
trial also starts with a fresh heap, so nothing accumulates across a long run.

The job file is written by the parent and carries everything needed to reproduce
the queued job without a GUI:

    {"path": <db path>, "table": <table>, "config": <get_config_dict() snapshot>,
     "conflict_choice": <ExportConflictDialog result>, "temp_frames_dir": <str|null>}

Exit codes: 0 = export finished, 1 = export failed or the database not loadable,
2 = bad invocation. A native crash surfaces as whatever code Windows assigns.
"""

import json
import os
import sys


def _run(job_path):
    # Offscreen: the worker builds the real window (the export pipeline reads its
    # widgets) but never shows it, so no window appears and no display is needed.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QEventLoop

    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    app = QApplication(sys.argv)

    from fnt.uwb.uwb_preprocessing_pyqt import (
        UWBQuickVisualizationWindow, ExportConflictDialog, _install_faulthandler)
    _install_faulthandler()

    win = UWBQuickVisualizationWindow()

    # Batch mode: suppresses every interactive dialog, skips the live preview and
    # the multi-GB fast-scrub index build, and makes the tag/day scan synchronous.
    win._batch_active = True
    win._suppress_dialogs()

    # Mirror the window's log to stdout so the parent can stream progress.
    _orig_log = win.log_message

    def _log(msg):
        try:
            _orig_log(msg)
        except Exception:
            pass
        print(msg, flush=True)

    win.log_message = _log

    # Replay the settings captured when the job was queued rather than reading
    # the database's own fnt_config.json.
    win._pending_config_override = job.get("config")
    if not win._load_database_path(job["path"]):
        print(f"WORKER: could not load {job['path']}", flush=True)
        return 1

    # The table comes from the captured config, but pin it explicitly in case the
    # config's table is missing from this database.
    table = job.get("table")
    if table and win.table_name != table:
        idx = win.combo_table.findText(table)
        if idx >= 0:
            win.combo_table.setCurrentIndex(idx)

    # Tag/day scan runs synchronously in batch mode, but wait defensively in case
    # any background query is still in flight.
    deadline = 120
    waited = 0.0
    while getattr(win, "_db_workers", None) and waited < deadline:
        app.processEvents(QEventLoop.AllEvents, 50)
        import time as _t
        _t.sleep(0.05)
        waited += 0.05

    if not any(cb.isChecked() for cb in win.tag_checkboxes.values()):
        for cb in win.tag_checkboxes.values():
            cb.setChecked(True)

    win._batch_conflict_choice = job.get("conflict_choice", ExportConflictDialog.OVERWRITE)
    win._batch_temp_frames_dir = job.get("temp_frames_dir")
    # Animation pass of a two-phase batch: render from the smoothed CSV the data
    # pass already wrote instead of re-smoothing every tag from the database.
    win._batch_reuse_smoothed = bool(job.get("reuse_smoothed", False))

    win.export_data()

    # export_data hands off to a background plot thread and then the animation
    # render; both clear `exporting` when finished. Pump until it settles.
    import time as _t
    while win.exporting:
        app.processEvents(QEventLoop.AllEvents, 50)
        _t.sleep(0.02)
    app.processEvents(QEventLoop.AllEvents, 50)

    failed = bool(getattr(win, "_last_export_failed", False))
    corrupt = bool(getattr(win, "_export_output_corrupt", False))
    print(f"WORKER: export {'FAILED' if failed else 'OK'}"
          + (" (output failed verification)" if corrupt else ""), flush=True)
    if corrupt:
        # Distinct from 1: the queue retries this one, because the same
        # job run again usually writes a clean file.
        return UWBQuickVisualizationWindow.EXIT_OUTPUT_CORRUPT
    return 1 if failed else 0


def main():
    if len(sys.argv) < 2:
        print("usage: python -m fnt.uwb.uwb_batch_worker <job.json>", file=sys.stderr)
        return 2
    try:
        return _run(sys.argv[1])
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WORKER: exception {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
