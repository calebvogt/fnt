"""One-shot FED3 commands for scripts and the CLI.

The GUI does **not** use these: it holds a persistent :class:`~fnt.fed3.fed_serial.Fed3Link`
per device and sends through that. These helpers exist for command-line use and
for talking to a device the GUI is not tracking. They refuse to open a port that
a live link already owns, which is what stops a stray one-shot command from
stealing the port out from under a running experiment.
"""

import time
from datetime import datetime

from . import fed_protocol as proto
from .fed_serial import (
    BAUD,
    PORT_REGISTRY,
    SETTLE_SECONDS,
    is_candidate_port,
    list_serial_ports,
)

__all__ = [
    "is_candidate_port",
    "list_serial_ports",
    "sync_time",
    "send_custom_command",
]


def sync_time(port=None, baud=BAUD, timeout=1, wait=0.5):
    """Set a FED3's RTC to the host clock. Returns ``(ok, message)``."""
    return send_custom_command(proto.cmd_sync(datetime.now()),
                               port=port, baud=baud, timeout=timeout, wait=wait)


def send_custom_command(command, port=None, baud=BAUD, timeout=1, wait=0.5):
    """Open a port, send one command, report what came back.

    Returns ``(ok, message)``.
    """
    try:
        import serial
        from serial.tools import list_ports
    except Exception:
        return False, "pyserial not installed. Install with: pip install pyserial"

    if port is None:
        port = _autodetect_port(list_ports)
        if port is None:
            return False, "No serial ports detected"

    owner = PORT_REGISTRY.owner(port)
    if owner is not None:
        return False, (f"{port} is held by {owner}. Send the command through the "
                       f"device's live connection instead of opening the port again.")

    if not command.endswith("\n"):
        command += "\n"

    out = [f"Opening {port} @ {baud}", f"Sending: {command.strip()}"]
    ser = None
    try:
        ser = serial.Serial(port, baud, timeout=timeout, dsrdtr=False, rtscts=False)
        try:
            ser.dtr = True
            ser.rts = True
        except Exception:
            pass
        time.sleep(SETTLE_SECONDS)
        ser.reset_input_buffer()

        ser.write(command.encode("utf-8"))
        ser.flush()
        time.sleep(wait)

        try:
            while ser.in_waiting > 0:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    out.append(f"FED3 says: {line}")
        except Exception:
            pass    # keep whatever we got and close cleanly

        if len(out) == 2:
            out.append("No response received from device.")
        return True, "\n".join(out)
    except Exception as exc:  # noqa: BLE001
        return False, f"Error opening {port}: {exc}"
    finally:
        if ser is not None and ser.is_open:
            ser.close()


def _autodetect_port(list_ports):
    candidates = [p for p in list_ports.comports() if is_candidate_port(p)]
    if not candidates:
        candidates = list(list_ports.comports())
    if not candidates:
        return None
    for p in candidates:
        blob = f"{p.device} {p.description}"
        if any(k in blob for k in ("ACM", "ttyACM", "USB", "Arduino", "CDC")):
            return p.device
    return candidates[0].device


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send time sync to a FED3 device.")
    parser.add_argument("--port", "-p", default=None,
                        help="Serial port (e.g. /dev/ttyACM0 or COM3)")
    parser.add_argument("--baud", "-b", type=int, default=BAUD)
    args = parser.parse_args()

    ok, message = sync_time(port=args.port, baud=args.baud)
    print(message)
    if not ok:
        raise SystemExit(1)
