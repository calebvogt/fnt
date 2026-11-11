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
from PyQt5.QtTest import QTest                           # noqa: E402
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget    # noqa: E402

from fnt.fed3 import protocol as proto                   # noqa: E402
from fnt.fed3.ui import editor as ed                     # noqa: E402
from fnt.fed3.ui import theme                            # noqa: E402
from fnt.fed3.ui.editor import TEMPLATES, PhaseDialog    # noqa: E402
from fnt.fed3.ui.log_panel import event_style            # noqa: E402
from fnt.fed3.ui.window import FedWindow                 # noqa: E402

RESULTS = []
PROTOCOL = """\
format: fnt-fed3-protocol/3
name: Window test
devices: [FED001, FED002]
phases:
  - name: Free
    until: {after: 1h}
    rule: {active: none, free_feed: 1}
  - name: FR1
    rule: {active: both}
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
          compiled and compiled.schedules["FED001"].phases[0].rule["ratio"] == "4"
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
        "format: fnt-fed3-protocol/3\nname: x\ndevices: [FED001]\nphases:\n"
        "  - name: a\n    until: {rewards: 5}\n    rule: {active: none}\n  - name: b\n")
    check("the checker refuses the same thing written in the text",
          compiled is None and any("only ends on rewards" in p.message for p in problems))
    removed = ed.remove_phase(edited, 0)
    check("Remove phase takes it out again", removed.strip() == TEMPLATES["Timed feeding"].strip())

    print("\n--- the form with several devices, and in format 1 files ---")
    keys, several = ed.syntax_of(PROTOCOL)
    check("a two-device protocol is several devices, in the current format",
          keys == proto.KEY_NAMES[proto.FORMAT] and several)
    form = PhaseDialog(places=["Free", "FR1"], keys=keys, several=several)
    check("ends on counts say each device moves on alone",
          "each device on its own" in form.ends.itemText(2))
    form.name.setText("Each")
    form.place.setCurrentIndex(2)
    form.ends.setCurrentIndex(2)
    form.amount.setText("20")
    added = ed.insert_phase(PROTOCOL, form.yaml, form.before())
    compiled, problems = proto.check(added)
    check("it writes of: each, which the checker accepts without a warning",
          "{rewards: 20, of: each}" in added and compiled is not None and not problems,
          "; ".join(p.message for p in problems))
    block = ed.phase_blocks(added)[1]
    settings, missing = ed.settings_from_phase(block.mapping, keys)
    check("and reads it back as an end on each device's rewards",
          not missing and settings["ends"] == "rewards" and settings["amount"] == "20")
    every = ed.phase_yaml(dict(ed.NEW_PHASE, ends="all_rewards", amount="9", fallback="2d"),
                          keys=keys, several=True)
    check("every device's rewards is written with of: all",
          "[{rewards: 9, of: all}, {after: 2d}]" in every)

    print("\n--- choosing the FED3s, and a rule for each ---")
    check("IDs can be typed as numbers, names or ranges",
          ed.parse_devices("1-3, fed007 9")
          == (["FED001", "FED002", "FED003", "FED007", "FED009"], None)
          and ed.parse_devices("1 - 2")[0] == ["FED001", "FED002"])
    check("and anything else says what's wrong",
          "isn't a FED3 ID" in ed.parse_devices("1, x")[1]
          and ed.parse_devices("0-2")[1] and ed.parse_devices(" ")[1])
    check("runs of IDs are shown short",
          ed.device_span(["FED001", "FED002", "FED003", "FED007", "FED008"])
          == "FED001–FED003, FED007, FED008")
    block_list = "name: x\ndevices:\n  - FED001\n  - FED002\n\nphases: []\n"
    check("Devices rewrites the list, one written line by line too",
          ed.set_devices(block_list, ["FED003"]) == "name: x\ndevices: [FED003]\n\nphases: []\n")
    check("and keeps a comment after it",
          ed.set_devices("devices: [FED001]  # cage A\n", ["FED002"])
          == "devices: [FED002]   # cage A\n")
    four = ["FED001", "FED002", "FED003", "FED004"]
    for name, text in TEMPLATES.items():
        compiled, problems = proto.check(ed.with_devices(text, four))
        check(f"template '{name}' for four FED3s checks without a warning",
              compiled is not None and not problems, "; ".join(p.message for p in problems))
    check("a template that names its own FED3s keeps them",
          ed.with_devices(TEMPLATES["Four-patch competition"], ["FED009"])
          == TEMPLATES["Four-patch competition"])
    cued = ed.with_devices(TEMPLATES["FR1, both pokes, with a cue"], four)
    form = PhaseDialog(places=["FR1"], devices=four)
    check("with several FED3s the form has a row for each",
          list(form.per_device) == four)
    form.name.setText("Mixed")
    form.place.setCurrentIndex(1)          # before FR1, so it ends after 24h
    form.tone.setChecked(True)
    for name, ratio in (("FED002", 2), ("FED003", 3), ("FED004", 3)):
        widgets, _ = form.per_device[name]
        widgets["ratio_kind"].setCurrentIndex(widgets["ratio_kind"].findData("random"))
        widgets["ratio"].setValue(ratio)
    widgets, _ = form.per_device["FED001"]
    widgets["active"].setCurrentIndex(widgets["active"].findData("both"))   # same as above
    check("each row says what that FED3 runs",
          [summary.text() for _, summary in form.per_device.values()]
          == ["FR1 · both", "RR2 · both", "RR3 · both", "RR3 · both"])
    check("a random ratio on one FED3 makes the counted click available",
          form.clicks["counted"].isEnabled())
    mixed = ed.insert_phase(cued, form.yaml, form.before())
    compiled, problems = proto.check(mixed)
    check("it writes per_device as FR/RR/PR, FED3s with the same change sharing a key",
          "    per_device:\n      FED002: {ratio: RR2}\n"
          "      FED003, FED004: {ratio: RR3}\n" in mixed
          and "FED001" not in mixed.split("per_device:")[1].split("- name: FR1")[0])
    check("which the checker accepts and compiles per FED3",
          compiled is not None
          and [compiled.schedules[d].phases[0].rule["ratio"] for d in four] == ["1", "2", "3", "3"]
          and compiled.schedules["FED003"].phases[0].rule["ratio_kind"] == "random",
          "; ".join(p.message for p in problems))
    block = ed.phase_blocks(mixed)[0]
    settings, missing = ed.settings_from_phase(block.mapping, devices=four)
    again = PhaseDialog(settings=settings, devices=four)
    check("Edit phase reads the per-device rules back as they were written",
          not missing and again.yaml(block.indent)
          == "".join(mixed.splitlines(keepends=True)[block.start:block.end]))
    again.free.setChecked(True)
    check("free feeding switches the table off and drops it",
          not again.per_device["FED002"][0]["ratio"].isEnabled()
          and again.settings()["only"] == {} and "per_device:" not in again.yaml())
    for phase, why in (
            ("  - name: a\n    per_device: {FED002: {timeout: 30s}}\n", "timeout"),
            ("  - name: a\n    per_device: {FED009: {ratio: 2}}\n", "isn't in devices"),
            ("  - name: a\n    rule: {active: none}\n    per_device: {FED002: {active: left}}\n",
             "no poke is active")):
        text = "format: fnt-fed3-protocol/3\nname: x\ndevices: [FED001, FED002]\nphases:\n" + phase
        settings, missing = ed.settings_from_phase(ed.phase_blocks(text)[0].mapping,
                                                   devices=["FED001", "FED002"])
        check(f"the form leaves to the text a per-device rule it can't show ({why})",
              any(why in m for m in missing), "; ".join(missing))
    print("\n--- the form writes FR/RR/PR, classic and every_phase ---")
    check("the classic templates use the classic cue and the short ratio",
          "        reward: classic\n" in TEMPLATES["FR1"]
          and "      ratio: RR3\n" in TEMPLATES["Random ratio"]
          and "      ratio: PR\n" in TEMPLATES["Progressive ratio"])
    settings, missing = ed.settings_from_phase(ed.phase_blocks(TEMPLATES["FR1"])[0].mapping)
    check("and the form reads classic back as the tone and light bar",
          not missing and settings["tone"] and settings["flash_target"] == "bar"
          and settings["flash_ms"] == 1000)
    stepped = TEMPLATES["Step up the ratio by rewards"]
    base, missing = ed.every_phase_of(stepped)
    check("every_phase is read as where each phase starts",
          not missing and base["active"] == "both" and base["tone"]
          and base["flash_target"] == "poke")
    blocks = ed.phase_blocks(stepped)
    settings, missing = ed.settings_from_phase(blocks[1].mapping, base=base)
    check("so a phase that only says ratio: FR3 shows the cue it inherits",
          not missing and settings["ratio"] == 3 and settings["tone"]
          and settings["active"] == "both")
    form = PhaseDialog(settings=settings, base=base)
    check("and is written back as only what differs",
          form.yaml(blocks[1].indent) == "  - name: FR3\n    until: {rewards: 100}\n"   # one FED3
                                         "    rule:\n      active: both\n      ratio: FR3\n")
    form.tone.setChecked(False)
    form.flash.setChecked(False)
    form.ratio.setValue(1)
    quiet = form.yaml(blocks[1].indent)
    check("turning off what it inherits writes none, and FR1 is written when it isn't the base",
          "      cue:\n        reward: none\n" in quiet and "ratio" not in quiet)
    compiled, problems = proto.check(ed.replace_phase(stepped, 1, quiet))
    check("which the checker compiles to no cue on that phase only",
          compiled and compiled.schedules["FED001"].phases[1].rule["cue.reward"] == "none"
          and compiled.schedules["FED001"].phases[0].rule["cue.reward"] != "none",
          "; ".join(p.message for p in problems))
    new = PhaseDialog(places=[b.name for b in blocks], base=base)
    check("a new phase starts from every_phase",
          new.active.currentText() == "both" and new.tone.isChecked())
    odd = stepped.replace("  active: both\n", "  active: both\n  min_poke_ms: 50\n", 1)
    check("an every_phase the form can't show is named, so the form stays out",
          any("min_poke_ms" in m for m in ed.every_phase_of(odd)[1]))
    v2 = PROTOCOL.replace(proto.FORMAT, proto.FORMAT_2)
    keys, _ = ed.syntax_of(v2)
    form = PhaseDialog(places=["Free", "FR1"], keys=keys, devices=["FED001", "FED002"])
    widgets, _ = form.per_device["FED002"]
    widgets["ratio_kind"].setCurrentIndex(widgets["ratio_kind"].findData("random"))
    widgets["ratio"].setValue(2)
    check("a format 2 file gets only rather than per_device",
          "    only:\n      FED002: {ratio: RR2}\n" in form.yaml())

    editor = ed.ProtocolEditor(TEMPLATES["FR1"], connected=four)
    check("the editor shows the protocol's FED3s", editor.devices_button.text()
          == "Devices: FED001…")
    editor._set_text(ed.set_devices(editor.text(), four))
    editor._use_template("FR3")
    check("and a template takes them",
          ed.devices_of(editor.text()) == four
          and editor.devices_button.text() == "Devices: FED001–FED004…")
    editor.close()

    v1 = (PROTOCOL.replace(proto.FORMAT, proto.FORMAT_1).replace("rule: ", "all: "))
    keys, several = ed.syntax_of(v1)
    check("a format 1 file is seen as one", keys == proto.KEY_NAMES[proto.FORMAT_1])
    written = ed.phase_yaml(dict(ed.NEW_PHASE, ends="all_rewards", amount="9", fallback=""),
                            keys=keys, several=several)
    check("the form writes format 1 keys into it",
          "    all:\n" in written and "{rewards: 9, devices: all}" in written
          and "rule:" not in written and " of:" not in written)
    edited = ed.insert_phase(v1, lambda indent: written, 1)
    compiled, problems = proto.check(edited)
    check("so the file still checks", compiled is not None,
          "; ".join(p.message for p in problems))
    settings, missing = ed.settings_from_phase(ed.phase_blocks(edited)[1].mapping, keys)
    check("and the form reads format 1 back", not missing and settings["ends"] == "all_rewards")

    print("\n--- the log's colours ---")
    check("problems are red", event_style("problem: jammed")[0] == theme.BAD)
    check("phase changes stand out", event_style("phase 2: Hold;goto")[0] == theme.LOG_PHASE)
    check("a stop is a run event, not a phase change",
          event_style("run stopped: the devices keep their last phase")[0] == theme.GOOD)
    check("anything that failed is red", event_style("edit failed: no run")[0] == theme.BAD)
    check("things people did have their own colour",
          event_style("feed by hand: dispensing a pellet")[0] == theme.LOG_MANUAL)
    check("a FED3 powered up or flashed restarts quietly",
          event_style("boot: reset=power")[0] == theme.MUTED)
    check("a watchdog restart doesn't", event_style("boot: reset=watchdog")[0] == theme.WARN)
    check("a feed that jams is red", event_style("feed by hand: it's jammed (failed)")[0]
          == theme.BAD)

    print("\n--- the plot and the cards' layout ---")
    from datetime import datetime, timedelta, timezone
    from matplotlib.dates import num2date
    from fnt.fed3.ui.plot import PelletPlot
    from fnt.fed3.ui.run_view import CardArea
    t0 = datetime(2026, 11, 2, 20, 55, tzinfo=timezone.utc)
    plot = PelletPlot()
    plot.update_plot({"run_id": "R", "copies": {}, "binding": {"FED001": 1}, "t0": t0},
                     timezone.utc, t0 + timedelta(hours=3))
    left, right = (num2date(x) for x in plot.axes.get_xlim())
    check("a plot with no pellets yet spans the run, not some other day",
          abs(left - t0) < timedelta(seconds=1)
          and abs(right - t0 - timedelta(hours=3)) < timedelta(seconds=1), f"{left} to {right}")
    from fnt.fed3.ui.run_view import pellet_waiting
    check("a card knows a pellet is waiting from the live rows, before STATUS says",
          pellet_waiting([{"event": "dispense"}, {"event": "clock"}], {"well": "0"})
          and not pellet_waiting([{"event": "dispense"}, {"event": "retrieve"}], {"well": "1"})
          and pellet_waiting([], {"well": "1"}))
    area = CardArea()
    area.resize(1400, 400)
    area.set_cards([QWidget() for _ in range(4)])
    check("four cards that fit three across go two by two, not three and one",
          area.columns == 2)

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
    status = lambda r: setup.table.item(r, 2).text()     # noqa: E731
    check("with no protocol, Edit is off and Start says what's needed",
          not setup.edit_button.isEnabled() and "Choose a protocol" in setup.start_note.text())
    box = lambda r: setup.table.cellWidget(r, 1)         # noqa: E731
    shown = lambda: [box(r).text() for r in range(2)]   # noqa: E731

    def type_id(row, keys, enter=True):
        """Select a row's ID number and type over it as a person would, then Enter."""
        box(row).setFocus()
        box(row).selectAll()
        QTest.keyClicks(box(row), keys)
        if enter:
            QTest.keyClick(box(row), Qt.Key_Return)

    check("the ID shows each FED3's ID",
          wait(app, lambda: shown() == ["FED001", "FED002"], 5))
    tests = setup.table.cellWidget(0, 3)
    check("Tests has Lights, Tone and Feed together",
          [b.text() for b in tests.findChildren(QPushButton)] == ["Lights", "Tone", "Feed"])
    setup._set_protocol(PROTOCOL, "")
    check("the protocol names its FED3s by ID, and Setup says which",
          wait(app, lambda: "uses FED001, FED002" in setup.hint.text(), 5)
)
    check("so with FED001 and FED002 plugged in, Start is on",
          wait(app, setup.start.isEnabled, 5) and setup.binding() == {"FED001": 1, "FED002": 2})
    numbers = lambda: {d["port"]: d["id"] for d in window.engine.snapshot()["devices"]}  # noqa: E731
    type_id(0, "7")
    check("typing an ID and pressing Enter changes the FED3's own ID",
          wait(app, lambda: numbers() == {"SIM1": 7, "SIM2": 2}, 10)
          and box(0).text() == "FED007")
    check("FED007 isn't in this protocol, so Start says FED001 is missing",
          wait(app, lambda: status(0).startswith("not in this protocol")
               and "FED001 isn't plugged in" in setup.start_note.text()
               and not setup.start.isEnabled(), 5))
    check("and FED001 gets a greyed row of its own, so the table is the protocol's list",
          setup.table.rowCount() == 3 and setup.table.cellWidget(2, 1).text() == "FED001"
          and status(2).startswith("not plugged in"))
    type_id(0, "2")
    check("typing an ID another FED3 has swaps the two, and says so",
          wait(app, lambda: numbers() == {"SIM1": 2, "SIM2": 7}, 10)
          and "swapped" in setup.id_note.text())
    check("the rows stay put, each with its new ID",
          [setup.table.item(r, 0).text() for r in range(2)] == ["SIM1", "SIM2"]
          and wait(app, lambda: shown() == ["FED002", "FED007"], 2))
    wait(app, lambda: False, 1)          # a few refreshes: nothing jumps back
    check("the choice holds once the FED3s have taken it",
          shown() == ["FED002", "FED007"])
    type_id(1, "042")
    check("with or without leading zeros",
          wait(app, lambda: numbers() == {"SIM1": 2, "SIM2": 42}, 10) and box(1).text() == "FED042")
    type_id(1, "FEDx7", enter=False)
    check("FED is fixed and can't be typed over: only the digits go in",
          box(1).text() == "FED7", box(1).text())
    QTest.keyClicks(box(1), "1234")
    check("three digits at most", box(1).text() == "FED712", box(1).text())
    type_id(1, "042")
    for keys in ("", "000"):
        type_id(1, keys)
    QTest.keyClick(box(1), Qt.Key_Up)
    QTest.keyClick(box(1), Qt.Key_Return)
    check("a blank, 0 or an arrow key leave the ID as it was",
          wait(app, lambda: False, 1) is False and numbers() == {"SIM1": 2, "SIM2": 42}
          and box(1).text() == "FED042", box(1).text())
    type_id(0, "1")
    type_id(1, "2")
    check("and back", wait(app, lambda: numbers() == {"SIM1": 1, "SIM2": 2}, 10)
          and wait(app, setup.start.isEnabled, 5))
    single = TEMPLATES["FR1"]
    setup._set_protocol(single, "")
    check("a one-FED3 protocol uses FED001; FED002 isn't in it",
          wait(app, lambda: status(1).startswith("not in this protocol"), 5)
          and setup.start.isEnabled())
    setup._set_protocol(PROTOCOL, "")
    check("each row shows the device is ready",
          wait(app, lambda: "SD ok" in status(0), 5))
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

    print("\n--- a FED3 still in another run ---")
    from fnt.fed3 import device_sim as dev
    from PyQt5.QtWidgets import QMessageBox
    sim1 = window.world.transports["SIM1"]
    sim1.act(lambda d: d.command(f"RUN:BEGIN,ELSEWHRE,{dev.iso(d.rtc())},"
                                 f"{d._crc_text(d.sched_crc)}"))
    check("its row says it's still in that run",
          wait(app, lambda: "still in run ELSEWHRE" in status(0), 10), status(0))
    check("and the note under Start says starting ends it",
          wait(app, lambda: "starting ends run ELSEWHRE on FED001" in setup.start_note.text(), 5),
          setup.start_note.text())
    shown = []
    QMessageBox.exec_ = lambda box: shown.append(box.text())          # Cancel
    setup.start.click()
    wait(app, lambda: False, 1)
    check("Start asks before ending it, naming the FED3 and the run",
          shown and "FED001" in shown[0] and "ELSEWHRE" in shown[0])
    check("and cancelling starts nothing",
          window.engine.snapshot()["state"] == "setup" and sim1.device.run_id == "ELSEWHRE")
    QMessageBox.exec_ = lambda box: [b for b in box.buttons()
                                     if b.text() == "End it and start"][0].click()

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
    check("one card per device", sorted(view.cards) == ["FED001", "FED002"])
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
    card = view.cards["FED001"]
    check("each card draws the FED3 with its counts",
          wait(app, lambda: card.graphic.counts is not None and card.graphic.connected, 10))
    check("and has its actions as labelled buttons",
          {"Feed", "More ▾"} <= {b.text() for b in card.findChildren(QPushButton)})
    more = [b for b in card.findChildren(QPushButton) if b.text() == "More ▾"][0]
    check("the rarer ones behind More",
          [a.text() for a in more.menu().actions()] == ["Play a cue…", "Set its clock now…"])
    [b for b in card.findChildren(QPushButton) if b.text() == "Feed"][0].click()
    check("Feed feeds, and the log says so",
          wait(app, lambda: any("FED001  feed by hand" in line for line in window.log.lines()), 10))

    run = window.engine.snapshot()["run"]
    editor = ed.ProtocolEditor(run["protocol_text"], running=True, t0=run["t0"],
                               edit_check=lambda text: "it can't" if "Renamed" in text else None)
    editor._set_text(run["protocol_text"].replace("name: Free", "name: Renamed"))
    check("the editor keeps what each FED3 is sent closed until asked",
          not editor.sent.isVisibleTo(editor) and editor.sent_toggle.text().startswith("▶"))
    editor.sent_toggle.click()
    check("and opens it", editor.sent.isVisibleTo(editor) and "FED001" in editor.sent.toPlainText())
    check("the running editor says at once when the run can't take an edit, and won't apply it",
          not editor.ok.isEnabled() and "can't take this" in editor.verdict.text())
    editor._set_text(run["protocol_text"])
    check("and allows it again once it can", editor.ok.isEnabled())

    print("\n--- the log ---")
    log = window.log
    check("the log runs along the bottom of the window",
          window.split.widget(1) is log and window.split.orientation() == Qt.Vertical)
    check("and shows what happened",
          wait(app, lambda: any("run started" in line for line in log.lines()), 5))
    log.toggle.click()
    check("the log collapses to its header", not log.text.isVisible() and not log.is_open())
    check("and remembers", str(settings.value("log_open")).lower() == "false")
    window.engine.submit("feed", device="FED001")
    check("while collapsed it counts what's new", wait(app, lambda: "new" in log.badge.text(), 10))
    log.toggle.click()
    check("and opens again", log.text.isVisible() and log.badge.text() == "")

    window.world.transports["SIM2"].act(lambda d: setattr(d, "jams_pending", 99))
    window.world.transports["SIM2"].act(lambda d: d.retrieve())
    check("a jam shows on its card",
          wait(app, lambda: any("jammed" in m for _, _, m in view.cards["FED002"].problem_keys), 30))
    card = view.cards["FED002"]
    acks = [b for b in card.findChildren(QPushButton) if b.text() == "Acknowledge"]
    check("with an Acknowledge button", len(acks) == 1)
    if acks:
        acks[0].click()
        check("which takes the flag off the card",
              wait(app, lambda: not card.property("problem"), 10))

    print("\n--- stopping ---")
    window.engine.submit("stop")
    check("Stop leads to the stopped view",
          wait(app, lambda: "Stopped" in view.timeline.text()))
    check("and the header says how long it ran, not time since t0",
          wait(app, lambda: view.elapsed.text().startswith("ran "), 5))
    check("the step bar shows Stopped", window.stage.state == "stopped"
          and "recording" in window.stage.note.text())
    check("Stop and Edit are gone; New run is offered",
          not view.stop.isVisible() and view.close_run.isVisible())
    folder = window.engine.snapshot()["run"]["folder"]
    check("each FED3's data file is named by its ID",
          {"FED001.csv", "FED002.csv"} <= set(os.listdir(folder)))
    path = os.path.join(folder, "FED001.csv")
    size = os.path.getsize(path)
    check("the devices are still recorded after Stop",
          wait(app, lambda: os.path.getsize(path) > size, 30))
    view.close_run.click()
    check("New run returns to setup",
          wait(app, lambda: window.stack.currentWidget() is window.setup_view))
    window.close()
    check("the window closes its engine", wait(app, lambda: not window.thread.is_alive(), 5))

    print("\n--- FED3s straight from flashing, all FED001 ---")
    from fnt.fed3 import demo

    class Fresh(demo.DemoWorld):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for transport in self.transports.values():
                transport.device.device_id = 1
    demo.DemoWorld = Fresh
    settings = QSettings(os.path.join(tmp, "fresh.ini"), QSettings.IniFormat)
    window = FedWindow(demo=2, speed=600, runs_root=os.path.join(tmp, "fresh"),
                       settings=settings)
    window.show()
    setup = window.setup_view
    setup._set_protocol(PROTOCOL, "")
    check("both FED3s show FED001, FED002 has a greyed row, and Start says to renumber one",
          wait(app, lambda: setup.table.rowCount() == 3
               and [setup.table.cellWidget(r, 1).text() for r in range(3)]
               == ["FED001", "FED001", "FED002"], 10)
          and wait(app, lambda: "SIM1 and SIM2 are both FED001" in setup.id_problems.text(), 5)
          and not setup.start.isEnabled())
    type_id(1, "2")
    check("giving one of them FED002 is all it takes",
          wait(app, setup.start.isEnabled, 10))
    setup.start.click()
    check("and the mix-up from Setup doesn't follow the run onto its cards",
          wait(app, lambda: window.stack.currentWidget() is window.run_view, 10)
          and wait(app, lambda: bool(window.run_view.cards), 5)
          and not any(p["kind"] == "duplicate_id" for p in window.engine.snapshot()["problems"]))
    window.engine.submit("stop")
    wait(app, lambda: window.engine.snapshot()["state"] == "stopped", 10)
    window.close()
    wait(app, lambda: not window.thread.is_alive(), 5)


if __name__ == "__main__":
    sys.exit(main())
