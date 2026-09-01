"""Drive the scheduler through the real form against four real FED3 devices.

Unlike the other suites this one needs hardware: four devices connected and
running the supported firmware. It fills the scheduling form the way a person
would, waits for the events to fire, and checks what actually reached each
device — the part no offline test can prove. Run it before an experiment, not in
a routine check; with no devices attached it reports that it was skipped.
"""
import os, sys, tempfile, time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer, QEventLoop, QTime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fnt.fed3.fed_widgets import FEDTabWidget
from fnt.fed3 import fed_scheduler as sched
from fnt.fed3 import fed_protocol as proto

for n in ("question", "warning", "critical", "information"):
    setattr(QMessageBox, n, staticmethod(lambda *a, **k: QMessageBox.Yes))

app = QApplication([])
tab = FEDTabWidget()
tab.sessions_dir = tempfile.mkdtemp(prefix="fnt-sched-")
T0 = time.time()
RESULTS = []

def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
def wait(s):
    l = QEventLoop(); QTimer.singleShot(int(s*1000), l.quit); l.exec_()
def check(label, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}",
          flush=True)

end = time.time() + 60
while time.time() < end:
    wait(1.0)
    if len([d for d in tab.devices
            if d.is_connected and d.firmware and d.current_file]) >= 4:
        break
conn = sorted([d for d in tab.devices if d.is_connected and d.firmware],
              key=lambda d: str(d.device_id))
log(f"connected: {[d.name for d in conn]}")
if len(conn) < 4:
    print(f"SKIPPED — needs four connected FED3 devices, found {len(conn)}",
          flush=True)
    tab.cleanup()
    sys.exit(0)

sent = {d.name: [] for d in conn}
for d in conn:
    d.link.command_sent.connect(
        lambda cmd, ok, detail, n=d.name: sent[n].append(cmd))

def add(target, action, delay_s, *, lights=None, mode=None, ratio=None,
        repeat=sched.REPEAT_NONE):
    """Fill the real form and press Add Event."""
    tab.sched_target_combo.setCurrentText(target)
    tab.sched_kind_combo.setCurrentIndex(1)          # "After a delay"
    tab.sched_delay_days.setValue(0)
    tab.sched_delay_time.setTime(QTime(0, 0, int(delay_s)))
    tab.sched_action_combo.setCurrentText(action)
    tab.sched_repeat_combo.setCurrentText(repeat)
    if lights is not None:
        tab.sched_lights_combo.setCurrentText("Lights ON" if lights else "Lights OFF")
    if mode is not None:
        tab.sched_mode_combo.setCurrentText(mode)
        if ratio is not None:
            tab.sched_params.ratio_spin.setValue(ratio)
    before = {e.id for e in tab.scheduler.events}
    tab._add_scheduled_event()
    new = [e for e in tab.scheduler.events if e.id not in before]
    return new[0] if new else None

print("\n--- the target list offers every cage plus All Devices ---", flush=True)
targets = [tab.sched_target_combo.itemText(i)
           for i in range(tab.sched_target_combo.count())]
check("All Devices is offered", sched.ALL_DEVICES in targets, str(targets))
check("every connected cage is offered",
      all(d.name in targets for d in conn), str(targets))

print("\n--- events created through the form ---", flush=True)
lights_ev = add(sched.ALL_DEVICES, sched.ACTION_LIGHTS, 20, lights=True)
mode_ev = add(conn[1].name, sched.ACTION_SET_MODE, 30,
              mode="Fixed Ratio (FR)", ratio=3)
trial_ev = add(conn[2].name, sched.ACTION_NEW_TRIAL, 40)
feed_ev = add(conn[0].name, sched.ACTION_DISPENSE, 50)
hourly_ev = add(sched.ALL_DEVICES, sched.ACTION_LIGHTS, 60, lights=False,
                repeat=sched.REPEAT_HOURLY)
disabled_ev = add(sched.ALL_DEVICES, sched.ACTION_DISPENSE, 25)
tab._set_event_enabled(disabled_ev.id, False)
doomed = add(sched.ALL_DEVICES, sched.ACTION_DISPENSE, 15)
tab._delete_event(doomed.id)

check("six events are scheduled", len(tab.scheduler.events) == 6,
      str(len(tab.scheduler.events)))
check("the deleted event is gone", tab.scheduler.get(doomed.id) is None)
# isVisible() is false while the collapsed section hides it; isHidden() is the
# widget's own flag, which is what _rebuild_scheduler_table sets.
check("the table is showing now that it has rows",
      not tab.sched_table.isHidden())
check("the table has a row per event",
      tab.sched_table.rowCount() == 6, str(tab.sched_table.rowCount()))
tab._refresh_readouts()
summary = tab.scheduler_section.toggle_button.text()
check("the collapsed header names the next event",
      "next:" in summary and "Lights ON" in summary, summary)

print("\n--- waiting for them to fire ---", flush=True)
from datetime import datetime, timedelta
deadline = time.time() + 110
while time.time() < deadline:
    wait(2.0)
    fired = [e for e in tab.scheduler.events if e.status != sched.STATUS_PENDING]
    # The repeating event has fired once its next occurrence is an hour out
    # rather than the sixty seconds it was created with.
    rearmed = hourly_ev.target_time - datetime.now() > timedelta(minutes=30)
    if len(fired) >= 4 and rearmed:
        break
# Commands are written on the serial thread and reported back through a queued
# signal; without spinning the loop the last one has not been delivered yet.
wait(4)
log("statuses: " + ", ".join(
    f"{e.describe_action()[:14]}->{e.status}" for e in tab.scheduler.events))

print("\n--- what actually reached the devices ---", flush=True)
check("lights went to every device",
      all(any(c.startswith("LIGHTS") for c in sent[d.name]) for d in conn),
      str({k: v for k, v in sent.items()}))
check("the lights event is marked executed",
      lights_ev.status == sched.STATUS_DONE, lights_ev.status)
check("the mode change went only to its target",
      any(c.startswith("MODE") for c in sent[conn[1].name])
      and not any(c.startswith("MODE") for c in sent[conn[0].name]),
      str(sent[conn[1].name]))
check("the new trial went only to its target",
      any(c == proto.CMD_NEW_TRIAL for c in sent[conn[2].name])
      and not any(c == proto.CMD_NEW_TRIAL for c in sent[conn[3].name]))
check("the dispense went only to its target",
      sent[conn[0].name].count(proto.CMD_FEED) == 1
      and proto.CMD_FEED not in sent[conn[3].name],
      str(sent[conn[0].name]))
check("a disabled event never fired",
      disabled_ev.status == sched.STATUS_PENDING, disabled_ev.status)
check("the repeating event re-armed instead of finishing",
      hourly_ev.status == sched.STATUS_PENDING
      and hourly_ev.target_time - datetime.now() > timedelta(minutes=30),
      f"{hourly_ev.status} @ {hourly_ev.target_time:%H:%M:%S}")

print("\n--- renaming a cage keeps its events pointed at it ---", flush=True)
old_name = conn[1].name
conn[1].name_edit.setText("Cage B")
tab._on_label_edited(conn[1])
check("the event follows the rename", mode_ev.target == "Cage B", mode_ev.target)
check("the target list followed too",
      "Cage B" in [tab.sched_target_combo.itemText(i)
                   for i in range(tab.sched_target_combo.count())])
conn[1].name_edit.setText("")
tab._on_label_edited(conn[1])
check("and back again", mode_ev.target == old_name, mode_ev.target)

print("\n--- clearing finished events ---", flush=True)
before_rows = tab.sched_table.rowCount()
tab._clear_finished_events()
remaining = {e.id for e in tab.scheduler.events}
check("executed one-shots are cleared",
      lights_ev.id not in remaining and feed_ev.id not in remaining)
check("the repeating event stays", hourly_ev.id in remaining)
check("the disabled pending event stays", disabled_ev.id in remaining)
check("the table shrank", tab.sched_table.rowCount() < before_rows,
      f"{before_rows} -> {tab.sched_table.rowCount()}")

print("\n--- the schedule survives a restart ---", flush=True)
state = tab.scheduler.to_list()
restored = sched.Scheduler()
restored.load(state)
check("every remaining event round-trips",
      {e.id for e in restored.events} == remaining, str(len(restored.events)))
check("the re-armed time round-trips",
      restored.get(hourly_ev.id).target_time == hourly_ev.target_time)

print("\n--- an event whose target is gone fails rather than fires blind ---",
      flush=True)
orphan = add(sched.ALL_DEVICES, sched.ACTION_DISPENSE, 1)
orphan.target = "FED 99"
wait(4)
check("it is marked failed", orphan.status == sched.STATUS_FAILED, orphan.status)
check("and says why", "not connected" in orphan.last_result, orphan.last_result)

# The run turned every cage's light on; leave the hardware as it was found.
for d in conn:
    tab._execute_action(d, sched.ACTION_LIGHTS, {"lights": False})
wait(3)

passed = sum(RESULTS)
print(f"\n{passed}/{len(RESULTS)} checks passed", flush=True)
tab.cleanup()
sys.exit(0 if passed == len(RESULTS) else 1)
