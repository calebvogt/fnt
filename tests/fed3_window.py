"""The FED3 window, driven like a person would, against simulated devices.

Runs offscreen. It checks the path through the window, not the engine's
behaviour (tests/fed3_run.py does that): a protocol is chosen, devices are
assigned, Start becomes possible only when it should, a problem shows on its
card until acknowledged, and Stop and New run lead where they say.
"""
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QSettings                       # noqa: E402
from PyQt5.QtWidgets import QApplication, QPushButton    # noqa: E402

from fnt.fed3 import protocol as proto                   # noqa: E402
from fnt.fed3.ui.editor import TEMPLATES, AddPhaseDialog  # noqa: E402
from fnt.fed3.ui.window import FedWindow                 # noqa: E402

RESULTS = []
PROTOCOL = """\
format: fnt-fed3-protocol/1
name: Window test
devices: [A, B]
phases:
  - name: Free
    until: {after: 1h}
    all: {active: none, free_feed: 1}
  - name: FR1
    all: {active: both}
"""


def check(label, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    tmp = tempfile.mkdtemp(prefix="fed3window-")
    try:
        run_checks(app, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    return 0 if all(RESULTS) else 1


def wait(app, condition, seconds=20):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


def run_checks(app, tmp):
    print("--- the editor's starting points ---")
    for name, text in TEMPLATES.items():
        compiled, problems = proto.check(text)
        check(f"template '{name}' passes its checks",
              compiled is not None and not [p for p in problems if p.level == "error"])
    dialog = AddPhaseDialog()
    dialog.name.setText("Added")
    dialog.ends.setCurrentIndex(2)
    dialog.tone.setChecked(True)
    dialog.flash.setChecked(True)
    added = TEMPLATES["Timed feeding"].replace(
        "phases:\n", "phases:\n" + dialog.yaml().replace("  - name: Added", "  - name: Added"))
    compiled, problems = proto.check(added)
    check("a phase from the Add phase form is valid YAML and a valid phase",
          compiled is not None and compiled.phases[0].name == "Added",
          "; ".join(p.message for p in problems))

    print("\n--- setup ---")
    settings = QSettings(os.path.join(tmp, "settings.ini"), QSettings.IniFormat)
    window = FedWindow(demo=2, speed=600, runs_root=os.path.join(tmp, "runs"), settings=settings)
    window.show()
    setup = window.setup_view
    check("found both simulated FED3s",
          wait(app, lambda: sum(d["connected"] for d in window.engine.snapshot().get("devices", [])) == 2))
    check("Start is off without a protocol", not setup.start.isEnabled())
    setup._set_protocol(PROTOCOL, "")
    check("one row per protocol device", setup.table.rowCount() == 2)
    check("each row offers both FED3s",
          wait(app, lambda: all(setup.table.cellWidget(r, 1).count() == 3 for r in range(2))))
    check("Start is off until every device has a FED3", not setup.start.isEnabled())
    setup.table.cellWidget(0, 1).setCurrentIndex(1)
    setup.table.cellWidget(1, 1).setCurrentIndex(1)
    wait(app, lambda: False, 0.5)
    check("the same FED3 twice is refused",
          not setup.start.isEnabled() and "own FED3" in setup.start_note.text())
    setup.table.cellWidget(1, 1).setCurrentIndex(2)
    check("with each device assigned, Start is on", wait(app, setup.start.isEnabled, 5))
    check("each row shows the device is ready",
          wait(app, lambda: "SD ok" in setup.table.item(0, 2).text(), 5))

    print("\n--- running ---")
    setup.start.click()
    check("Start leads to the running view",
          wait(app, lambda: window.stack.currentWidget() is window.run_view))
    view = window.run_view
    check("one card per device", sorted(view.cards) == ["A", "B"])
    check("the header names the run", view.title.text() == "Window test")
    check("the timeline says what comes next",
          wait(app, lambda: "next change" in view.timeline.text(), 10))

    window.world.transports["SIM2"].act(lambda d: setattr(d, "jams_pending", 99))
    window.world.transports["SIM2"].act(lambda d: d.retrieve())
    check("a jam shows on its card",
          wait(app, lambda: any("jammed" in m for _, _, m in view.cards["B"].problem_keys), 30))
    card = view.cards["B"]
    acks = [b for b in card.findChildren(QPushButton) if b.text() == "Acknowledge"]
    check("with an Acknowledge button", len(acks) == 1)
    if acks:
        acks[0].click()
        check("which takes the flag off the card",
              wait(app, lambda: not card.property("problem"), 10))

    print("\n--- stopping ---")
    window.engine.submit("stop")
    check("Stop leads to the stopped view, with totals",
          wait(app, lambda: "Stopped" in view.timeline.text() and "pellets" in view.timeline.text()))
    check("Stop and Edit are gone; New run is offered",
          not view.stop.isVisible() and view.close_run.isVisible())
    view.close_run.click()
    check("New run returns to setup",
          wait(app, lambda: window.stack.currentWidget() is window.setup_view))
    window.close()
    check("the window closes its engine", wait(app, lambda: not window.thread.is_alive(), 5))


if __name__ == "__main__":
    sys.exit(main())
