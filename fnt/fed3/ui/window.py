"""The FED3 window: Setup, then Running, then Stopped, over one run engine.

Opened from FNT's FED tab in its own process (fnt.tool_launcher), or directly:

    python -m fnt.fed3.ui.window
    python -m fnt.fed3.ui.window --demo 4 --speed 60     # simulated FED3s

The devices run the experiment; this window watches and records. Closing it
mid-run doesn't stop anything on the devices, and reopening it picks the run
back up.
"""
import argparse
import os
import sys

from PyQt5.QtCore import QSettings, Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from ..run import Engine, EngineThread
from . import theme
from .log_panel import LogPanel
from .run_view import RunView
from .editor import format_delta
from .setup_view import SetupView
from .stage_bar import StageBar

POLL_MS = 250
LOG_HEIGHT = 170


class FedWindow(QMainWindow):
    def __init__(self, demo=0, speed=1.0, runs_root=None, settings=None):
        super().__init__()
        self.setWindowTitle("FED3" + (" (simulated devices)" if demo else ""))
        self.resize(1250, 820)
        self.setStyleSheet(theme.STYLESHEET)
        self.settings = settings or QSettings("FieldNeuroToolbox",
                                              "FED3" + ("-demo" if demo else ""))

        default_root = os.path.join(os.path.expanduser("~"), "Documents", "FED3_Runs")
        root = runs_root or self.settings.value("runs_root", "") or default_root
        self.world = None
        if demo:
            from ..demo import DemoWorld
            self.world = DemoWorld(count=demo, speed=speed)
            root = runs_root or os.path.join(default_root, "demo")
            self.engine = Engine(root, ports=self.world.ports, open_port=self.world.open,
                                 now=self.world.now, monotonic=self.world.monotonic)
        else:
            self.engine = Engine(root)
        resumed = self.engine.resume_latest()

        self.stack = QStackedWidget()
        self.setup_view = SetupView(self.engine, self.settings)
        self.run_view = RunView(self.engine)
        self.stack.addWidget(self.setup_view)
        self.stack.addWidget(self.run_view)
        # The log runs along the whole bottom, under Setup and Running alike.
        self.log = LogPanel()
        self.split = QSplitter(Qt.Vertical)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self.stack)
        self.split.addWidget(self.log)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        self.stage = StageBar()
        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(self.stage)
        column.addWidget(self.split, 1)
        self.setCentralWidget(central)
        self.log_height = int(self.settings.value("log_height", LOG_HEIGHT) or LOG_HEIGHT)
        self.split.setSizes([self.height() - self.log_height, self.log_height])
        self.split.splitterMoved.connect(self._log_resized)
        self.log.toggled.connect(self._log_toggled)
        if str(self.settings.value("log_open", "true")).lower() == "false":
            self.log.set_open(False)

        self.thread = EngineThread(self.engine, before_step=self.world.tick if self.world else None)
        self.thread.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(POLL_MS)
        if resumed:
            self.statusBar().showMessage(f"Resumed the unfinished run in {resumed}", 15000)

    def _log_resized(self, *_):
        if self.log.is_open():
            self.log_height = self.split.sizes()[1]
            self.settings.setValue("log_height", self.log_height)

    def _log_toggled(self, opened):
        self.settings.setValue("log_open", opened)
        total = sum(self.split.sizes())
        log = self.log_height if opened else self.log.header.sizeHint().height()
        self.split.setSizes([total - log, log])

    def _refresh(self):
        snapshot = self.engine.snapshot()
        if not snapshot:
            return
        self.log.update_log(snapshot)
        self.stage.set_stage(snapshot["state"], self._stage_note(snapshot))
        if snapshot["state"] == "setup":
            if self.stack.currentWidget() is not self.setup_view:
                self.stack.setCurrentWidget(self.setup_view)
            self.setup_view.update_view(snapshot)
        else:
            if self.stack.currentWidget() is not self.run_view:
                self.stack.setCurrentWidget(self.run_view)
            self.run_view.update_view(snapshot)

    def _stage_note(self, snapshot):
        run = snapshot.get("run")
        if snapshot["state"] == "setup":
            return self.setup_view.start_note.text() or "Ready to start"
        elapsed = format_delta((snapshot["now"] - run["t0"]).total_seconds())
        if snapshot["state"] == "running":
            return f"{run['name']} · {elapsed} since t0"
        return f"{run['name']} · still recording · New run goes back to Setup"

    def closeEvent(self, event):
        if self.engine.state == "running" and not self.world:
            answer = QMessageBox.question(
                self, "Close FED3",
                "The run keeps going on the devices, and their cards keep every row.\n\n"
                "While this window is closed FNT isn't copying data, checking clocks or "
                "moving phases that need it. Reopen it to pick the run back up.\n\n"
                "Close anyway?")
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        self.timer.stop()
        self.thread.stop()
        self.thread.join(timeout=3)
        for device in self.engine.devices.values():
            if device.link:
                device.link.close()
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnt-fed-window")
    parser.add_argument("--demo", type=int, default=0, help="number of simulated FED3s")
    parser.add_argument("--speed", type=float, default=1.0, help="demo time multiplier")
    parser.add_argument("--runs", help="folder for run records")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    window = FedWindow(demo=args.demo, speed=args.speed, runs_root=args.runs)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
