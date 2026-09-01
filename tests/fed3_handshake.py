"""The handshake runs once per connection, and drift is reported honestly.

Both invariants here regressed silently once already. Every heartbeat PING is
answered with a PONG identical to the one that opens a connection, so treating a
PONG as a handshake re-set the device RTC every 30s for the length of an
experiment — and the drift figure printed alongside it compared a whole-second
device clock against a sub-second host clock, so it read "-1s off" whether the
device was perfect or hours out.

Driven through the real FEDTabWidget methods against a stub link; no serial port
and no hardware.
"""
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from fnt.fed3 import fed_protocol as proto
from fnt.fed3.fed_widgets import FEDTabWidget

RESULTS = []


def check(label, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")


class StubLink:
    """Just enough of Fed3Link for _send: it is live and it records."""

    def __init__(self):
        self.sent = []
        self.stopped = False

    def is_live(self):
        return True

    def send(self, command):
        self.sent.append(command)

    def stop(self):
        """Refusing a device disconnects it, which stops its link."""
        self.stopped = True


def fresh_device(tab):
    """A new slot wired to a stub link, as if it had just connected."""
    device = tab.add_device_slot(refresh=False)
    if device is None:
        device = tab.devices[-1]
    device.link = StubLink()
    device.handshake_done = False
    device.refused = False
    device.firmware = None
    return device


def pong(fw=None):
    return f"PONG_FED3,ID:7,FW:{fw or proto.firmware_requirement()}"


def main(tab):
    print("--- the first PONG of a connection is a handshake ---")
    device = fresh_device(tab)
    tab._apply_handshake(device, pong())
    first = list(device.link.sent)
    check("the device is accepted", device.handshake_done)
    clock_cmds = [c for c in first
                  if c == proto.CMD_STATUS or c.startswith("SYNC:")]
    check("its clock is both read and set",
          {proto.CMD_STATUS} <= set(clock_cmds)
          and any(c.startswith("SYNC:") for c in clock_cmds), str(first))
    check("STATUS comes first, so the reading predates the correction",
          clock_cmds[:1] == [proto.CMD_STATUS], str(first))

    print("\n--- a heartbeat PONG is not a handshake ---")
    device.link.sent.clear()
    for _ in range(5):
        tab._apply_handshake(device, pong())
    check("no command is sent in reply", device.link.sent == [],
          str(device.link.sent))
    check("the clock is not re-set",
          not any(c.startswith("SYNC:") for c in device.link.sent))
    check("the device stays accepted", device.handshake_done)

    print("\n--- reconnecting re-arms the handshake ---")
    # _on_link_connected clears the flag; a recycled link must re-identify the
    # device, which may not be the one that was there before.
    device.handshake_done = False
    device.link.sent.clear()
    tab._apply_handshake(device, pong())
    check("the handshake runs again after a reconnect",
          any(c.startswith("SYNC:") for c in device.link.sent),
          str(device.link.sent))

    print("\n--- firmware is still refused on the first PONG ---")
    device = fresh_device(tab)
    tab._apply_handshake(device, pong(fw="0.1"))
    check("an unsupported device is refused", device.refused)
    check("a refused device is not marked handshaken", not device.handshake_done)

    print("\n--- drift is measured against a whole-second host clock ---")
    device = fresh_device(tab)
    device.handshake_done = True

    def drift_detail(device_time):
        tab._note_clock_drift(device, device_time.strftime("%Y-%m-%dT%H:%M:%S"))
        return device.last_device_time

    now = datetime.now().replace(microsecond=0)
    check("a device on the current second records that second",
          drift_detail(now) == now, str(device.last_device_time))
    check("a device 30s ahead records 30s ahead",
          drift_detail(now + timedelta(seconds=30)) == now + timedelta(seconds=30))
    check("a device a day behind records a day behind",
          drift_detail(now - timedelta(days=1)) == now - timedelta(days=1))

    device.last_device_time = None
    tab._note_clock_drift(device, "")
    check("an empty reading is ignored rather than logged as drift",
          device.last_device_time is None)
    tab._note_clock_drift(device, "not-a-time")
    check("an unparseable reading is ignored too",
          device.last_device_time is None)

    print("\n--- the two clock log lines say different things ---")
    device = fresh_device(tab)
    logged = []
    real_log_action = tab._log_action
    tab._log_action = lambda desc, **kw: (logged.append((desc, kw.get("detail", ""))),
                                          real_log_action(desc, **kw))[1]
    iso = now.strftime("%Y-%m-%dT%H:%M:%S")
    tab._record_clock_sync(device, iso)
    tab._note_clock_drift(device, iso)
    tab._log_action = real_log_action
    synced = [d for name, d in logged if name == "Clock synced"]
    checked = [d for name, d in logged if name == "Clock checked"]
    check("the sync line carries no offset figure",
          synced and "off)" not in synced[-1] and "vs host" not in synced[-1],
          synced[-1] if synced else "(no line)")
    check("the reading line does carry one",
          checked and "vs host" in checked[-1],
          checked[-1] if checked else "(no line)")
    passed = sum(RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    _app = QApplication([])
    _tab = FEDTabWidget()
    try:
        _code = main(_tab)
    finally:
        # The port scanner runs in a QThread; leaving it alive past interpreter
        # shutdown turns any failure into a second, unrelated traceback.
        _tab.cleanup()
    sys.exit(_code)
