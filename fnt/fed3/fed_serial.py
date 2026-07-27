"""Serial transport for FED3 devices.

Exactly one thread owns a port for the whole time it is open. Previously three
different code paths could open the same device: the tracker thread, the
one-shot ``send_custom_command()`` helper, and the port scanner (which opened
every candidate port on every refresh). Overlapping opens are what produced
"port busy" lockouts and left device screens wedged.

The rules enforced here:

* :class:`Fed3Link` opens the port once and holds it. All writes go through its
  command queue and are issued from the owning thread.
* A port claimed by a live link is registered in :data:`PORT_REGISTRY`. The
  scanner refuses to probe a claimed port, so refreshing ports can never disturb
  a running experiment.
* Reads are bulk, not line-at-a-time, so a multi-kilobyte SD transfer drains the
  device's USB buffer fast enough that it never blocks in ``Serial.write()``.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

from PyQt5.QtCore import QThread, pyqtSignal

from . import fed_protocol as proto

BAUD = 115200

# Seconds to let a freshly opened board settle before talking to it. The SAMD21
# enumerates its USB CDC endpoint slightly after the port becomes openable.
SETTLE_SECONDS = 2.0

# A link that has heard nothing at all for this long is treated as wedged and
# recycled. Devices are otherwise silent between pokes, so this is only a
# backstop against a half-open port that never errors.
SILENCE_TIMEOUT = 0.0   # 0 disables; see Fed3Link(silence_timeout=...)


class _PortRegistry:
    """Tracks which serial ports are currently owned by a live link."""

    def __init__(self):
        self._lock = threading.Lock()
        self._claimed = {}          # port -> owner label

    def claim(self, port, owner):
        with self._lock:
            if port in self._claimed:
                return False
            self._claimed[port] = owner
            return True

    def release(self, port):
        with self._lock:
            self._claimed.pop(port, None)

    def owner(self, port):
        with self._lock:
            return self._claimed.get(port)

    def claimed_ports(self):
        with self._lock:
            return set(self._claimed)


PORT_REGISTRY = _PortRegistry()


class Fed3Link(QThread):
    """Owns one serial port: bulk reads out, queued commands in.

    Signals carry no device identity; the caller binds a link to a device.
    """

    line_received = pyqtSignal(str)
    connected = pyqtSignal()
    disconnected = pyqtSignal(str)      # human-readable reason
    command_sent = pyqtSignal(str, bool, str)   # command, ok, detail

    def __init__(self, port, owner="", silence_timeout=SILENCE_TIMEOUT, parent=None):
        super().__init__(parent)
        self.port = port
        self.owner = owner or port
        self.silence_timeout = silence_timeout
        self._commands = Queue()
        self._running = False
        self._claimed = False
        self._buf = bytearray()
        self._last_rx = 0.0

    # --- public API (GUI thread) -----------------------------------------

    def send(self, command):
        """Queue a command. Delivery happens on the owning thread.

        Returns False only when the link is not running; a queued command is
        reported through :attr:`command_sent` once it is actually written.
        """
        if not self._running:
            return False
        self._commands.put(command)
        return True

    def stop(self, wait_ms=3000):
        self._running = False
        self._commands.put(None)        # wake the loop immediately
        if not self.wait(wait_ms):
            self.terminate()
            self.wait(1000)
        # terminate() skips the finally block, so release defensively.
        if self._claimed:
            PORT_REGISTRY.release(self.port)
            self._claimed = False

    def is_live(self):
        return self._running and self.isRunning()

    # --- owning thread ----------------------------------------------------

    def run(self):
        try:
            import serial
        except ImportError:
            self.disconnected.emit(
                "pyserial is not installed. Install it with: pip install pyserial")
            return

        if not PORT_REGISTRY.claim(self.port, self.owner):
            self.disconnected.emit(
                f"{self.port} is already in use by {PORT_REGISTRY.owner(self.port)}")
            return
        self._claimed = True

        ser = None
        reason = "closed"
        try:
            ser = serial.Serial(self.port, BAUD, timeout=0.1,
                                dsrdtr=False, rtscts=False)
            # DTR must be asserted for the firmware's `if (Serial)` guard to see
            # a live host; without it the board sits in its boot menu.
            try:
                ser.dtr = True
                ser.rts = True
            except Exception:
                pass

            self._running = True
            if not self._sleep_interruptible(SETTLE_SECONDS):
                return

            ser.reset_input_buffer()
            self._last_rx = time.monotonic()
            self.connected.emit()

            while self._running:
                if not self._port_present():
                    reason = "device disconnected (port disappeared)"
                    break
                self._drain_commands(ser)
                if not self._running:
                    break
                if not self._pump_reads(ser):
                    reason = "serial read failed"
                    break
                if self._is_silent():
                    reason = "no data from device"
                    break
        except Exception as exc:  # noqa: BLE001 - surfaced to the GUI log
            reason = f"{type(exc).__name__}: {exc}"
        finally:
            self._running = False
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
            if self._claimed:
                PORT_REGISTRY.release(self.port)
                self._claimed = False
            self.disconnected.emit(reason)

    def _sleep_interruptible(self, seconds):
        """Sleep in slices so stop() during startup is honoured promptly."""
        deadline = time.monotonic() + seconds
        while self._running and time.monotonic() < deadline:
            time.sleep(0.05)
        return self._running

    def _port_present(self):
        # Only meaningful on POSIX; Windows surfaces removal as a read error.
        return not self.port.startswith("/dev/") or os.path.exists(self.port)

    def _is_silent(self):
        return (self.silence_timeout > 0
                and time.monotonic() - self._last_rx > self.silence_timeout)

    def _drain_commands(self, ser):
        while True:
            try:
                command = self._commands.get_nowait()
            except Empty:
                return
            if command is None:          # stop() sentinel
                self._running = False
                return
            if not command.endswith("\n"):
                command += "\n"
            try:
                ser.write(command.encode("utf-8"))
                ser.flush()
                self.command_sent.emit(command.strip(), True, "sent")
            except Exception as exc:  # noqa: BLE001
                self.command_sent.emit(command.strip(), False, str(exc))
                self._running = False
                return

    def _pump_reads(self, ser):
        """Read everything waiting and emit complete lines. False on failure."""
        try:
            waiting = ser.in_waiting
            # read() honours the 0.1 s timeout when nothing is waiting, which is
            # what paces this loop without a sleep.
            chunk = ser.read(waiting if waiting else 1)
        except Exception:
            return False

        if not chunk:
            return True

        self._last_rx = time.monotonic()
        self._buf.extend(chunk)
        while True:
            idx = self._buf.find(b"\n")
            if idx < 0:
                break
            raw = bytes(self._buf[:idx + 1])
            del self._buf[:idx + 1]
            self.line_received.emit(raw.decode("utf-8", errors="replace"))
        return True


class PortScannerWorker(QThread):
    """Probes unclaimed serial ports for FED3 devices.

    Ports owned by a live :class:`Fed3Link` are never opened. Opening a port
    asserts DTR and restarts the device's USB session, so probing a running
    experiment is exactly the thing that used to knock devices offline.
    """

    finished_scan = pyqtSignal(list, list)   # [(port, status, id, fw)], [all ports]

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            from serial.tools import list_ports
        except ImportError:
            # Reported as an empty scan rather than an exception in a QThread,
            # which Qt would print and swallow.
            self.finished_scan.emit([], [])
            return

        ports = list(list_ports.comports())
        claimed = PORT_REGISTRY.claimed_ports()

        results = [(p, "In use (FNT)", None, None) for p in sorted(claimed)]
        probe = [p for p in ports
                 if is_candidate_port(p) and p.device not in claimed]

        if probe:
            with ThreadPoolExecutor(max_workers=min(10, len(probe))) as pool:
                results.extend(r for r in pool.map(_probe_port, probe) if r)

        self.finished_scan.emit(results, [p.device for p in ports])


def _probe_port(port_info):
    """Open one port, PING it, and report what answered."""
    import serial

    device = port_info.device
    ser = None
    try:
        ser = serial.Serial(device, BAUD, timeout=1.0, dsrdtr=False, rtscts=False)
        try:
            ser.dtr = True
            ser.rts = True
        except Exception:
            pass
        time.sleep(SETTLE_SECONDS)
        ser.reset_input_buffer()
        ser.write(f"{proto.CMD_PING}\n".encode())
        ser.flush()

        deadline = time.monotonic() + 1.5
        response = ""
        while time.monotonic() < deadline:
            response += ser.read_all().decode("utf-8", errors="ignore")
            if "PONG_FED3" in response:
                break
            time.sleep(0.05)

        if "PONG_FED3" in response:
            device_id, firmware = proto.parse_pong(response)
            return device, "FED3 Active", device_id, firmware
    except Exception as exc:  # noqa: BLE001
        text = str(exc).lower()
        if any(k in text for k in ("busy", "already open", "permission denied", "access is denied")):
            return device, "Busy/In Use", None, None
    finally:
        if ser is not None and ser.is_open:
            try:
                ser.close()
            except Exception:
                pass

    return device, "Unresponsive", None, None


def is_candidate_port(p):
    """Whether a port is worth probing for a FED3.

    Filters Bluetooth, legacy motherboard COM ports and Intel AMT SOL, which
    either hang on open or take seconds to time out.
    """
    dev = (getattr(p, "device", "") or "").lower()
    desc = (getattr(p, "description", "") or "").lower()
    hwid = (getattr(p, "hwid", "") or "").lower()

    if any(k in desc or k in dev or k in hwid
           for k in ("bluetooth", "bth", "rfcomm")):
        return False
    if "communications port" in desc or "standard serial port" in desc:
        return False
    if hwid.startswith("acpi") or "/dev/ttys" in dev:
        return False
    if "intel" in desc and ("active management" in desc or "sol" in desc):
        return False

    raw = f"{getattr(p, 'device', '')} {getattr(p, 'description', '')} {getattr(p, 'hwid', '')}"
    if any(k in raw for k in ("ACM", "ttyACM", "USB", "Arduino", "Feather", "Adafruit", "CDC")):
        return True
    return getattr(p, "vid", None) is not None


def list_serial_ports():
    """Available FED3-candidate port names; empty if pyserial is missing."""
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports() if is_candidate_port(p)]
    except Exception:
        return []
