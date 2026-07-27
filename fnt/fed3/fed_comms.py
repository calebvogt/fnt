"""FED3 device communication helper.

Provides `list_serial_ports()`, `sync_time()`, and `Fed3Tracker` for GUI.
"""

import time
import threading
from datetime import datetime


def is_candidate_port(p):
    """Check if a serial port is a potential candidate for a FED3 device.

    Filters out known unresponsive or unrelated ports (e.g. Bluetooth,
    motherboard serial ports, Intel AMT SOL) while keeping USB/ACM/CDC devices.
    """
    dev = getattr(p, "device", "") or ""
    desc = getattr(p, "description", "") or ""
    hwid = getattr(p, "hwid", "") or ""
    
    dev_lower = dev.lower()
    desc_lower = desc.lower()
    hwid_lower = hwid.lower()
    
    # 1. Exclude Bluetooth ports
    if "bluetooth" in desc_lower or "bth" in hwid_lower or "bluetooth" in dev_lower or "rfcomm" in dev_lower:
        return False
        
    # 2. Exclude standard motherboard / legacy / physical COM ports
    if "communications port" in desc_lower or "standard serial port" in desc_lower:
        return False
    if hwid_lower.startswith("acpi"):
        return False
    if "/dev/ttys" in dev_lower:
        return False
        
    # 3. Exclude Intel Active Management Technology / SOL
    if "intel" in desc_lower and ("active management" in desc_lower or "sol" in desc_lower):
        return False
        
    # 4. Include ports that have explicit USB / ACM / CDC / Arduino / Feather / Adafruit keywords
    if any(k in dev or k in desc or k in hwid for k in ("ACM", "ttyACM", "USB", "Arduino", "Feather", "Adafruit", "CDC")):
        return True
        
    # 5. Include any port with a valid USB Vendor ID
    if getattr(p, "vid", None) is not None:
        return True
        
    return False


class Fed3Tracker:
    def __init__(self, port, baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self._running = False
        self.lock = threading.Lock()

    def start(self, callback, connected_callback=None):
        try:
            import serial
            import os
            
            with self.lock:
                self.ser = serial.Serial(self.port, self.baud, timeout=1, dsrdtr=False, rtscts=False)
                try:
                    # Set DTR/RTS to True to reset the SAMD21 board and bypass boot animation
                    self.ser.dtr = True
                    self.ser.rts = True
                except Exception:
                    pass
                self._running = True

            time.sleep(2.0) # Allow device to reset/settle after opening
            
            # Check self._running again after sleep
            if not self._running:
                return

            if connected_callback:
                try:
                    connected_callback()
                except Exception:
                    pass
            
            while self._running:
                # Check if port still exists on the filesystem (works on Linux/macOS)
                if self.port.startswith("/dev/") and not os.path.exists(self.port):
                    if self._running:
                        callback("ERROR: Device disconnected (port disappeared)")
                    self._running = False
                    break

                ser_ref = None
                with self.lock:
                    if self._running:
                        ser_ref = self.ser

                if ser_ref is None or not ser_ref.is_open:
                    break

                try:
                    in_wait = ser_ref.in_waiting
                except Exception as e:
                    if self._running:
                        callback(f"ERROR: Serial port error: {e}")
                    self._running = False
                    break

                if in_wait > 0:
                    try:
                        line = ser_ref.readline()
                        if not line and in_wait > 0:
                            # We expected bytes but got nothing, check if port still exists
                            if self.port.startswith("/dev/") and not os.path.exists(self.port):
                                if self._running:
                                    callback("ERROR: Device disconnected")
                                self._running = False
                                break
                        if not self._running:
                            break
                        line_str = line.decode("utf-8", errors="ignore")
                        if line_str:
                            callback(line_str)
                    except Exception as e:
                        if not self._running:
                            break
                        if isinstance(e, (serial.SerialException, OSError)):
                            callback(f"ERROR: Serial read error: {e}")
                            self._running = False
                            break
                        pass
                else:
                    time.sleep(0.05)
        except Exception as e:
            if self._running:
                callback(f"ERROR: {e}")
            self._running = False
        finally:
            self.stop()

    def stop(self):
        self._running = False
        with self.lock:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

    def send_command(self, command):
        with self.lock:
            if self.ser and self.ser.is_open:
                if not command.endswith('\n'):
                    command += '\n'
                try:
                    self.ser.write(command.encode("utf-8"))
                    return True, f"Sent: {command.strip()}"
                except Exception as e:
                    return False, f"Send error: {e}"
            return False, "Port not open."


def list_serial_ports():
    """Return a list of available serial port device names.

    Returns an empty list if pyserial is not installed or no ports found.
    """
    try:
        from serial.tools import list_ports
        return [p.device for p in list_ports.comports() if is_candidate_port(p)]
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
    now = datetime.now()
    sync_string = now.strftime("SYNC:%Y,%m,%d,%H,%M,%S\n")
    return send_custom_command(sync_string, port=port, baud=baud, timeout=timeout, wait=wait)


def send_custom_command(command, port=None, baud=115200, timeout=1, wait=0.5):
    """Send a custom command to a FED3 device.

    Args:
        command: String to send to the device.
        port: Serial port string.
        baud: Baud rate.
        timeout: Serial timeout.
        wait: Seconds to wait for response.

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
        ports = [p for p in list_ports.comports() if is_candidate_port(p)]
        if not ports:
            # Fallback to all ports if no candidate matches
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
    ser = None
    try:
        ser = serial.Serial(port, baud, timeout=timeout, dsrdtr=False, rtscts=False)
        try:
            ser.dtr = True
            ser.rts = True
        except Exception:
            pass
        # Allow device to reset/settle after opening
        time.sleep(2.0)

        if not command.endswith('\n'):
            command += '\n'
            
        out_lines.append(f"Opening {port} @ {baud}")
        out_lines.append(f"Sending: {command.strip()}")
        ser.write(command.encode("utf-8"))
        time.sleep(wait)

        try:
            while ser.in_waiting > 0:
                response = ser.readline().decode("utf-8", errors="ignore").strip()
                if response:
                    out_lines.append(f"FED3 says: {response}")
        except Exception:
            # If reading fails, continue to close port
            pass

        if not out_lines:
            out_lines.append("No response received from device.")
        return True, "\n".join(out_lines)

    except Exception as e:
        return False, f"Error opening {port}: {e}"
    finally:
        if ser and ser.is_open:
            ser.close()


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
