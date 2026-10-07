"""Background thread for an RFID run, so the window stays responsive.

A run reads a million-plus reads and takes tens of seconds; on the GUI thread
that is a frozen window the OS offers to kill. The worker calls the same
:func:`fnt.rfid.pipeline.run` a script would, relays its progress as signals,
and hands the finished :class:`~fnt.rfid.pipeline.RunResult` back whole.
"""

from __future__ import annotations

import copy
import traceback

from PyQt5.QtCore import QThread, pyqtSignal

from .config.defaults import RFIDConfig
from .pipeline import RunResult, run


class RFIDWorker(QThread):
    """Run the pipeline off the GUI thread.

    Signals:
        progress(int, str): percent complete and the latest log line
        finished_run(object): the RunResult (also on error or cancel)
    """

    progress = pyqtSignal(int, str)
    finished_run = pyqtSignal(object)

    def __init__(self, config: RFIDConfig, write: bool = True, parent=None):
        super().__init__(parent)
        # A private copy: the user can keep editing the window mid-run without
        # changing the settings this run is recorded as having used.
        self.config = copy.deepcopy(config)
        self.write = write
        self.result: RunResult | None = None
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            self.result = run(self.config, progress=self._relay, write=self.write,
                              cancelled=lambda: self._cancel)
        except Exception as exc:                          # noqa: BLE001
            self.result = RunResult(config=self.config,
                                    error=f"{type(exc).__name__}: {exc}")
            self.result.log.append(traceback.format_exc())
        self.finished_run.emit(self.result)

    def _relay(self, percent: int, message: str):
        self.progress.emit(int(percent), str(message))
