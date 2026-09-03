"""Device identity and naming in the Connected Devices panel.

Runs the real widget headlessly (no hardware, no serial) and walks a slot
through the states a device actually goes through: created empty, identified by
its PING, labelled by the user, cleared back to the default, unplugged, and
plugged in again. The naming rules used to be spread across four call sites that
each set ``device_id`` and patched the title by hand, and they disagreed — an
unidentified slot was called "Device 1" while an identified one was called
"FED 4", and the auto-filled name was written into the user's own label field so
it could never fall back.

Run with: python tests/fed3_naming.py
"""

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import QApplication            # noqa: E402

from fnt.fed3 import fed_session as session_mod     # noqa: E402
from fnt.fed3.fed_widgets import FEDTabWidget       # noqa: E402


class Checks:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def __call__(self, label, got, want):
        if got == want:
            self.passed += 1
            print(f"  PASS  {label}: {got!r}")
        else:
            self.failed += 1
            print(f"  FAIL  {label}: {got!r} != {want!r}")


def main():
    tmp = tempfile.mkdtemp(prefix="fnt-naming-")
    app = QApplication([])
    tab = FEDTabWidget()
    # A scratch registry, so running the tests cannot clobber real device names.
    tab.device_names = session_mod.DeviceNames(os.path.join(tmp, "device_names.json"))
    check = Checks()

    print("--- an empty panel says it is empty ---")
    check("no slots at startup", len(tab.devices), 0)
    check("empty message shown", tab.devices_empty.isHidden(), False)

    print("--- a slot with no device does not claim to be one ---")
    device = tab.add_device_slot(refresh=False)
    check("title", device.box.title(), "Slot 1")
    check("name", device.name, "Slot 1")
    check("placeholder shows the default", device.name_edit.placeholderText(), "Slot 1")
    check("empty message hidden", tab.devices_empty.isHidden(), True)

    print("--- identifying itself renames the slot ---")
    tab._adopt_identity(device, "4", "2.0")
    check("title", device.box.title(), "FED 4")
    check("name", device.name, "FED 4")
    check("the label field stays the user's", device.label, "")
    check("placeholder shows the new default",
          device.name_edit.placeholderText(), "FED 4")

    print("--- a user label wins, and is remembered ---")
    device.name_edit.setText("Cage A")
    tab._on_label_edited(device)
    check("title", device.box.title(), "Cage A")
    check("stored against the device ID", tab.device_names.get("4"), "Cage A")
    check("scheduler targets follow the rename",
          [tab.sched_target_combo.itemText(i)
           for i in range(tab.sched_target_combo.count())],
          ["All Devices", "Cage A"])

    print("--- clearing the label falls back and forgets ---")
    device.name_edit.setText("")
    tab._on_label_edited(device)
    check("name", device.name, "FED 4")
    check("forgotten", tab.device_names.get("4"), "")

    print("--- the label returns when the device is plugged in again ---")
    tab.device_names.set("4", "Cage A")
    tab._remove_device(device)
    check("no slots", len(tab.devices), 0)
    check("empty message back", tab.devices_empty.isHidden(), False)
    again = tab.add_device_slot(refresh=False)
    tab._adopt_identity(again, "4", "2.0")
    check("name", again.name, "Cage A")
    check("label field filled", again.label, "Cage A")

    print("--- a repeated STATUS does not churn the name ---")
    before = again.last_known_name
    tab._apply_status(again, {"id": "4", "fw": "2.0"})
    tab._apply_status(again, {"id": 4, "fw": "2.0"})   # int, as JSON resume gives
    check("name unchanged", again.name, before)

    print("--- a label typed just now outranks the remembered one ---")
    fresh = tab.add_device_slot(refresh=False)
    fresh.name_edit.setText("Bench rig")
    tab._adopt_identity(fresh, "4", "2.0")
    check("typed label kept", fresh.name, "Bench rig")

    print("--- pointing a slot at an unidentified port withdraws the identity ---")
    moved = tab.add_device_slot(refresh=False)
    tab._adopt_identity(moved, "9", "2.0")
    moved.name_edit.setText("Cage B")
    tab._on_label_edited(moved)
    check("named", moved.name, "Cage B")
    tab._adopt_identity(moved, None, None, replace=True)
    check("falls back to the slot", moved.name, f"Slot {moved.slot_num}")
    check("id cleared", moved.device_id, None)
    check("but the name is still remembered", tab.device_names.get("9"), "Cage B")
    tab._adopt_identity(moved, "9", "2.0", replace=True)
    check("and comes back with the device", moved.name, "Cage B")

    print("--- folder names survive Windows' reserved and trailing-dot rules ---")
    safe = session_mod._safe_name
    check("a plain name is left alone", safe("Cage A"), "Cage_A")
    check("path separators cannot escape", safe("../etc"), ".._etc")
    check("a reserved device name is escaped", safe("COM3"), "COM3_")
    check("reserved matching is case-blind", safe("nul"), "nul_")
    check("reserved applies before an extension", safe("con.log"), "con.log_")
    check("trailing dots are dropped", safe("Cage B."), "Cage_B")
    check("separators become underscores, not nothing", safe("///"), "___")
    check("an empty name still yields a folder", safe(""), "unnamed")
    check("a name of only dots yields a folder", safe("..."), "unnamed")
    check("a name that merely starts with one is fine", safe("COM3B"), "COM3B")

    print("--- session state carries the label, not the resolved name ---")
    state = again.to_state()
    check("label", state["label"], "Cage A")
    check("name", state["name"], "Cage A")
    unlabelled = tab.add_device_slot(refresh=False)
    tab._adopt_identity(unlabelled, "7", "2.0")
    check("no label for an unlabelled device", unlabelled.to_state()["label"], "")

    total = check.passed + check.failed
    print(f"\n{check.passed}/{total} checks passed")
    del app
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
