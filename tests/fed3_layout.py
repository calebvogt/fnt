"""Layout and safety invariants for the FED3 tab.

These are the properties the console was reorganised to have, and every one of
them is invisible to the other suites: ordering, disclosure defaults, the guard
in front of a bulk dispense, and a flow layout that shares a top edge.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMessageBox, QPushButton, QWidget,
)

from fnt.fed3 import fed_protocol as proto
from fnt.fed3.fed_ui import CollapsibleSection, FlowLayout
from fnt.fed3.fed_widgets import FEDTabWidget

RESULTS = []


def check(label, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")


class StubLink:
    def __init__(self):
        self.sent = []

    def is_live(self):
        return True

    def send(self, command):
        self.sent.append(command)

    def stop(self):
        pass


def connected_slot(tab):
    device = tab.add_device_slot(refresh=False)
    device.link = StubLink()
    device.handshake_done = True
    return device


def main(tab):
    print("--- the page is ordered by what a running experiment is checked for ---")
    order = []
    for index in range(tab.content_layout.count()):
        widget = tab.content_layout.itemAt(index).widget()
        if widget is not None:
            order.append(widget)
    check("devices come before the plot",
          order.index(tab.devices_group) < order.index(tab.plot_group))
    check("the plot comes before setup",
          order.index(tab.plot_group) < order.index(tab.setup_section))
    check("setup comes before the scheduler",
          order.index(tab.setup_section) < order.index(tab.scheduler_section))

    print("\n--- set-once panels start put away ---")
    check("bulk actions start collapsed",
          not tab.setup_section.toggle_button.isChecked())
    check("the scheduler starts collapsed",
          not tab.scheduler_section.toggle_button.isChecked())
    check("the scheduler header states the state it hides",
          "no events" in tab.scheduler_section.toggle_button.text(),
          tab.scheduler_section.toggle_button.text())
    check("an empty scheduler table takes no room",
          not tab.sched_table.isVisible())

    print("\n--- an unconnected slot is nothing but setup ---")
    device = tab.add_device_slot(refresh=False)
    check("a fresh slot opens on its setup panel", device.setup_btn.isChecked())
    # isVisible() is false while an ancestor is hidden, and nothing here is
    # shown on screen; isHidden() reports the widget's own flag.
    device.setup_btn.setChecked(False)
    check("collapsing hides the setup panel", device.setup_panel.isHidden())
    device.setup_btn.setChecked(True)
    check("expanding shows it again", not device.setup_panel.isHidden())

    print("\n--- Dispense All asks before moving four motors ---")
    targets = [connected_slot(tab) for _ in range(3)]
    asked = []

    def fake_question(parent, title, text, *args, **kwargs):
        asked.append(text)
        return QMessageBox.Cancel

    real_question = QMessageBox.question
    QMessageBox.question = staticmethod(fake_question)
    try:
        tab._dispense_all()
        check("it asks first", len(asked) == 1)
        check("the question names every device",
              asked and all(d.name in asked[0] for d in targets),
              asked[0].replace("\n", " ") if asked else "")
        check("cancelling sends nothing",
              all(d.link.sent == [] for d in targets),
              str([d.link.sent for d in targets]))

        QMessageBox.question = staticmethod(
            lambda *a, **k: QMessageBox.Yes)
        tab._dispense_all()
        check("confirming dispenses to every connected device",
              all(proto.CMD_FEED in d.link.sent for d in targets),
              str([d.link.sent for d in targets]))
    finally:
        QMessageBox.question = real_question

    print("\n--- a header is a label, not a mnemonic ---")
    section = CollapsibleSection("Setup & bulk actions", "a & b")
    check("ampersands survive in the title",
          "Setup && bulk actions" in section.toggle_button.text(),
          section.toggle_button.text())

    print("\n--- the flow layout shares a top edge and fills the row ---")
    host = QWidget()
    flow = FlowLayout(margin=0, spacing=8)
    host.setLayout(flow)
    boxes = []
    for height in (120, 60, 90):
        box = QLabel("x")
        box.setFixedSize(180, height)
        flow.addWidget(box)
        boxes.append(box)
    flow.setGeometry(QRect(0, 0, 600, 400))
    tops = {b.y() for b in boxes}
    check("every item on a row starts at the same y", len(tops) == 1, str(tops))
    check("each item keeps its own height",
          [b.height() for b in boxes] == [120, 60, 90],
          str([b.height() for b in boxes]))
    # Unjustified, the three 180px items plus two 8px gaps would stop at 556.
    bare = 180 * 3 + 8 * 2
    right = max(b.x() + b.width() for b in boxes)
    check("leftover width is shared out rather than left ragged",
          bare < right <= 600, f"{right} (bare {bare})")

    print("\n--- buttons are sized for the labels they will hold ---")
    lights = [b for b in tab.findChildren(QPushButton)
              if b.text().startswith("Lights")]
    check("a toggle reserves room for its other caption",
          lights and all(
              b.minimumWidth() >= b.fontMetrics().boundingRect("Lights: ON").width()
              for b in lights),
          str([b.minimumWidth() for b in lights[:3]]))
    check("Start Recording reserves room for Stop Recording",
          tab.record_btn.minimumWidth()
          >= tab.record_btn.fontMetrics().boundingRect("Start Recording").width(),
          str(tab.record_btn.minimumWidth()))

    passed = sum(RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    _app = QApplication([])
    _tab = FEDTabWidget()
    try:
        _code = main(_tab)
    finally:
        _tab.cleanup()
    sys.exit(_code)
