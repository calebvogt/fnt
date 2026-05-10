"""FED3 time-sync helper (moved into fnt.fed3 package).

Provides `list_serial_ports()` and `sync_time()` used by the GUI.
"""

import time
from datetime import datetime


def list_serial_ports():
    """Return a list of available serial port device names.

    Returns an empty list if pyserial is not installed or no ports found.
    """
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports()]
    except Exception:
        return []


def sync_time(port=None, baud=115200, timeout=1, wait=0.5):
    """Send a SYNC command to a FED3 device.

    Args:
        port: Serial port string (e.g. '/dev/ttyACM0' or 'COM3'). If None,
              the function will attempt to auto-detect a sensible port.
        baud: Baud rate (default 115200).
        timeout: Serial timeout in seconds.
        wait: Seconds to wait after sending before reading responses.

    Returns:
        (success: bool, message: str)
    """
    try:
        import serial
        from serial.tools import list_ports
    except Exception:
        return False, "pyserial not installed. Install with: pip install pyserial"

    # Auto-detect port if not provided
    if port is None:
        ports = list(list_ports.comports())
        if not ports:
            return False, "No serial ports detected"
        candidate = None
        for p in ports:
            dev = getattr(p, "device", "")
            desc = getattr(p, "description", "")
            if any(k in dev or k in desc for k in ("ACM", "ttyACM", "USB", "Arduino", "CDC")):
                candidate = dev
                break
        if candidate is None:
            candidate = ports[0].device
        port = candidate

    out_lines = []
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        # Allow device to reset/settle after opening
        time.sleep(2.0)

        now = datetime.now()
        sync_string = now.strftime("SYNC:%Y,%m,%d,%H,%M,%S\n")
        out_lines.append(f"Opening {port} @ {baud}")
        out_lines.append(f"Sending: {sync_string.strip()}")
        ser.write(sync_string.encode("utf-8"))
        time.sleep(wait)

        try:
            while ser.in_waiting > 0:
                response = ser.readline().decode("utf-8", errors="ignore").strip()
                out_lines.append(f"FED3 says: {response}")
        except Exception:
            # If reading fails, continue to close port
            pass

        ser.close()
        if not out_lines:
            out_lines.append("No response received from device.")
        return True, "\n".join(out_lines)

    except Exception as e:
        return False, f"Error opening {port}: {e}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Send time sync to FED3 device.")
    parser.add_argument("--port", "-p", help="Serial port (e.g., /dev/ttyACM0 or COM3)", default=None)
    parser.add_argument("--baud", "-b", type=int, default=115200)
    args = parser.parse_args()
    ok, msg = sync_time(port=args.port, baud=args.baud)
    print(msg)
    if not ok:
        raise SystemExit(1)
