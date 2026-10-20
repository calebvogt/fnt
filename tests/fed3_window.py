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
from PyQt5.QtCore import Qt                              # noqa: E402
from PyQt5.QtWidgets import QApplication, QPushButton    # noqa: E402

from fnt.fed3 import protocol as proto                   # noqa: E402
from fnt.fed3.ui import editor as ed                     # noqa: E402
from fnt.fed3.ui import theme                            # noqa: E402
from fnt.fed3.ui.editor import TEMPLATES, PhaseDialog    # noqa: E402
from fnt.fed3.ui.log_panel import event_style            # noqa: E402
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
    for name in ed.CLASSIC:
        if name == "Timed feeding":      # its daily window is written in the text
            continue
        blocks = ed.phase_blocks(TEMPLATES[name])
        settings, missing = ed.settings_from_phase(blocks[0].mapping)
        check(f"the form can edit '{name}' without losing anything", not missing,
              "; ".join(missing))
    blocks = ed.phase_blocks(TEMPLATES["Timed feeding"])
    dialog = PhaseDialog(places=[b.name for b in blocks])
    check("a new phase placed last ends never", dialog.ends.currentData() == "never")
    dialog.place.setCurrentIndex(1)
    check("placed before another, it needs an end", dialog.ends.currentData() == "after")
    dialog.name.setText("Added")
    dialog.ends.setCurrentIndex(2)
    dialog.tone.setChecked(True)
    dialog.flash.setChecked(True)
    dialog.bnc.setChecked(True)
    dialog.clicks["counted"].setChecked(True)
    added = ed.insert_phase(TEMPLATES["Timed feeding"], dialog.yaml, dialog.before())
    compiled, problems = proto.check(added)
    check("a phase from the Add phase form is valid YAML and a valid phase",
          compiled is not None and compiled.phases[0].name == "Added",
          "; ".join(p.message for p in problems))
    blocks = ed.phase_blocks(added)
    settings, missing = ed.settings_from_phase(blocks[0].mapping)
    edit = PhaseDialog(settings=settings)
    check("Edit phase reads it back as it was written",
          not missing and edit.yaml(blocks[0].indent) == "".join(
              added.splitlines(keepends=True)[blocks[0].start:blocks[0].end]))
    edit.ratio.setValue(4)
    edited = ed.replace_phase(added, 0, edit.yaml(blocks[0].indent))
    compiled, _ = proto.check(edited)
    check("and an edit from the form changes that phase only",
          compiled and compiled.schedules["FED"].phases[0].rule["ratio"] == "4"
          and [p.name for p in compiled.phases] == ["Added", "Restricted"])
    print("\n--- the form can't contradict itself ---")
    form = PhaseDialog(settings=dict(ed.NEW_PHASE, ends="rewards", amount="50",
                                     ratio=3, tone=True, click_counted=True))
    form.free.setChecked(True)
    check("free feeding sets no active poke and locks it",
          form.active.currentText() == "none" and not form.active.isEnabled())
    check("and switches off the ratio, reward and cues",
          not any(w.isEnabled() for w in (form.kind, form.ratio, form.timeout, form.quota,
                                           form.reward, form.tone, form.flash, form.bnc,
                                           *form.clicks.values())))
    check("and a phase can't end on rewards that can't be earned",
          form.ends.currentData() == "after"
          and not form.ends.model().item(2).isEnabled())
    written = form.settings()
    check("so what it writes is plain free feeding",
          written["active"] == "none" and written["ratio"] == 1 and not written["tone"]
          and not written["click_counted"])
    form.free.setChecked(False)
    check("unticking it gives the poke back", form.active.currentText() == "both"
          and form.ratio.isEnabled() and form.ends.model().item(2).isEnabled())
    form.kind.setCurrentText("fixed")
    form.ratio.setValue(1)
    check("FR1 has no 'not yet a reward' poke to click on",
          not form.clicks["counted"].isEnabled())
    check("no timeout, no timeout click; no quota, no depleted click",
          not form.clicks["timeout"].isEnabled() and not form.clicks["depleted"].isEnabled())
    form.timeout.setValue(30)
    check("a timeout makes its click available", form.clicks["timeout"].isEnabled())
    compiled, problems = proto.check(
        "format: fnt-fed3-protocol/1\nname: x\ndevices: [A]\nphases:\n"
        "  - name: a\n    until: {rewards: 5}\n    all: {active: none}\n  - name: b\n")
    check("the checker refuses the same thing written in the text",
          compiled is None and any("only ends on rewards" in p.message for p in problems))
    removed = ed.remove_phase(edited, 0)
    check("Remove phase takes it out again", removed.strip() == TEMPLATES["Timed feeding"].strip())

    print("\n--- the log's colours ---")
    check("problems are red", event_style("problem: jammed")[0] == theme.BAD)
    check("phase changes stand out", event_style("phase 2: Hold;goto")[0] == theme.LOG_PHASE)
    check("a stop is a run event, not a phase change",
          event_style("run stopped: the devices keep their last phase")[0] == theme.GOOD)
    check("anything that failed is red", event_style("edit failed: no run")[0] == theme.BAD)
    check("things people did have their own colour",
          event_style("feed by hand: FEED,result=started")[0] == theme.LOG_MANUAL)

    print("\n--- setup ---")
    settings = QSettings(os.path.join(tmp, "settings.ini"), QSettings.IniFormat)
    window = FedWindow(demo=2, speed=600, runs_root=os.path.join(tmp, "runs"), settings=settings)
    window.show()
    setup = window.setup_view
    check("found both simulated FED3s",
          wait(app, lambda: sum(d["connected"] for d in window.engine.snapshot().get("devices", [])) == 2))
    check("Start is off without a protocol", not setup.start.isEnabled())
    check("the step bar shows Setup", wait(app, lambda: window.stage.state == "setup", 5))
    check("one row per FED3 plugged in, by port, before any protocol",
          wait(app, lambda: setup.table.rowCount() == 2, 5)
          and [setup.table.item(r, 0).text() for r in range(2)] == ["SIM1", "SIM2"])
    check("the columns are Port, ID, Status and Tests",
          [setup.table.horizontalHeaderItem(c).text() for c in range(4)]
          == ["Port", "ID", "Status", "Tests"])
    ident = lambda r: setup.table.cellWidget(r, 1)       # noqa: E731
    check("the ID is a list showing each FED3's number",
          [ident(r).currentData() for r in range(2)] == [1, 2])
    tests = setup.table.cellWidget(0, 3)
    check("Tests has Lights, Tone and Feed together",
          [b.text() for b in tests.findChildren(QPushButton)] == ["Lights", "Tone", "Feed"])
    setup._set_protocol(PROTOCOL, "")
    check("a FED3's number is its place in the protocol: FED001 plays A, FED002 plays B",
          wait(app, lambda: setup.table.item(0, 2).text().startswith("plays A")
               and setup.table.item(1, 2).text().startswith("plays B"), 5)
          and ident(0).itemText(0) == "FED001  (A)")
    check("so with both plugged in, Start is on", wait(app, setup.start.isEnabled, 5))
    ident(0).setCurrentIndex(ident(0).findData(2))
    numbers = lambda: {d["port"]: d["id"] for d in window.engine.snapshot()["devices"]}  # noqa: E731
    check("choosing a number another FED3 has swaps the two",
          wait(app, lambda: numbers() == {"SIM1": 2, "SIM2": 1}, 10))
    check("and the rows follow, in number order",
          wait(app, lambda: [setup.table.item(r, 0).text() for r in range(2)]
               == ["SIM2", "SIM1"], 5))
    ident(0).setCurrentIndex(ident(0).findData(2))      # SIM2 back to 2, SIM1 to 1
    check("and back", wait(app, lambda: numbers() == {"SIM1": 1, "SIM2": 2}, 10))
    single = TEMPLATES["FR1"]
    setup._set_protocol(single, "")
    check("a one-device protocol uses FED001; the other isn't in it",
          wait(app, lambda: "not in this protocol" in setup.table.item(1, 2).text(), 5)
          and setup.start.isEnabled())
    setup._set_protocol(PROTOCOL, "")
    check("each row shows the device is ready",
          wait(app, lambda: "SD ok" in setup.table.item(0, 2).text(), 5))
    tests = setup.table.cellWidget(0, 3)
    wait(app, lambda: False, 1)          # let the renumbering above settle
    tests = setup.table.cellWidget(0, 3)
    cues = len(window.world.transports["SIM1"].device.cues)
    [b for b in tests.findChildren(QPushButton) if b.text() == "Tone"][0].click()
    check("Tone plays a tone on that FED3",
          wait(app, lambda: len(window.world.transports["SIM1"].device.cues) > cues, 5)
          and window.world.transports["SIM1"].device.cues[-1][2] == "tone:4000:200")
    lights = setup.table.cellWidget(0, 3).findChild(QPushButton, "lights")
    lights.click()
    check("Lights turns a FED3's lights on",
          wait(app, lambda: window.world.transports["SIM1"].device.lights, 5))
    check("and the button shows it, from the device's own state",
          wait(app, lambda: lights.isChecked(), 5))

    print("\n--- running ---")
    setup.start.click()
    check("starting the run turns the lights off",
          wait(app, lambda: not window.world.transports["SIM1"].device.lights, 10))
    check("Start leads to the running view",
          wait(app, lambda: window.stack.currentWidget() is window.run_view))
    view = window.run_view
    check("the step bar moves to Running, with Setup ticked",
          wait(app, lambda: window.stage.state == "running", 5)
          and window.stage.steps["setup"][2].text().startswith("✓"))
    check("one card per device", sorted(view.cards) == ["A", "B"])
    check("the plot's title chooses its time zone, standard time first",
          wait(app, lambda: view.zone_box.count() == 3, 5)
          and view.zone_box.currentData() == "standard")
    view.zone_box.setCurrentIndex(2)
    check("and switching it redraws in that zone",
          wait(app, lambda: view.plot.axes.xaxis.get_major_formatter().tz is not None
               and str(view.plot.axes.xaxis.get_major_formatter().tz) == "UTC", 5))
    check("the header names the run", view.title.text() == "Window test")
    check("the timeline says what comes next",
          wait(app, lambda: "next phase" in view.timeline.text(), 10))
    card = view.cards["A"]
    check("each card draws the FED3 with its counts",
          wait(app, lambda: card.graphic.counts is not None and card.graphic.connected, 10))
    check("and has its actions as labelled buttons",
          {"Feed", "Test cue…", "Set clock"} <= {b.text() for b in card.findChildren(QPushButton)})
    [b for b in card.findChildren(QPushButton) if b.text() == "Feed"][0].click()
    check("Feed feeds, and the log says so",
          wait(app, lambda: any("A  feed by hand" in line for line in window.log.lines()), 10))

    print("\n--- the log ---")
    log = window.log
    check("the log runs along the bottom of the window",
          window.split.widget(1) is log and window.split.orientation() == Qt.Vertical)
    check("and shows what happened",
          wait(app, lambda: any("run started" in line for line in log.lines()), 5))
    log.toggle.click()
    check("the log collapses to its header", not log.text.isVisible() and not log.is_open())
    check("and remembers", str(settings.value("log_open")).lower() == "false")
    window.engine.submit("feed", device="A")
    check("while collapsed it counts what's new", wait(app, lambda: "new" in log.badge.text(), 10))
    log.toggle.click()
    check("and opens again", log.text.isVisible() and log.badge.text() == "")

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
    check("the step bar shows Stopped", window.stage.state == "stopped"
          and "recording" in window.stage.note.text())
    check("Stop and Edit are gone; New run is offered",
          not view.stop.isVisible() and view.close_run.isVisible())
    folder = window.engine.snapshot()["run"]["folder"]
    path = os.path.join(folder, [n for n in os.listdir(folder) if n.startswith("A_FED")][0])
    size = os.path.getsize(path)
    check("the devices are still recorded after Stop",
          wait(app, lambda: os.path.getsize(path) > size, 30))
    view.close_run.click()
    check("New run returns to setup",
          wait(app, lambda: window.stack.currentWidget() is window.setup_view))
    window.close()
    check("the window closes its engine", wait(app, lambda: not window.thread.is_alive(), 5))


if __name__ == "__main__":
    sys.exit(main())
