"""QThread wrappers for FED3 serial tracking and scanning."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from PyQt5.QtCore import QThread, pyqtSignal
from . import fed_comms

class Fed3TrackerWorker(QThread):
    line_received = pyqtSignal(str, str)     # dev_name, line
    error_received = pyqtSignal(str, str)    # dev_name, error
    connected = pyqtSignal()

    def __init__(self, port, dev_name, parent=None):
        super().__init__(parent)
        self.port = port
        self.dev_name = dev_name
        self.tracker = fed_comms.Fed3Tracker(self.port)

    def run(self):
        def callback(line):
            if line.strip().startswith("ERROR:"):
                self.error_received.emit(self.dev_name, line.strip())
            else:
                self.line_received.emit(self.dev_name, line)
        self.tracker.start(callback, lambda: self.connected.emit())

    def stop(self):
        self.tracker.stop()
        self.quit()
        self.wait()


class PortScannerWorker(QThread):
    finished_scan = pyqtSignal(list, list)

    def __init__(self, active_ports=None, parent=None):
        super().__init__(parent)
        self.active_ports = active_ports or []

    def run(self):
        import serial
        from serial.tools import list_ports

        ports = list(list_ports.comports())
        # We only check candidate ports for active FED3 to prevent hangs on legacy/virtual ports
        ping_ports = [p for p in ports if fed_comms.is_candidate_port(p)]

        def check_port(p):
            dev = p.device
            if dev in self.active_ports:
                return (dev, "Active", None)

            ser = None
            try:
                ser = serial.Serial()
                ser.port = dev
                ser.baudrate = 115200
                ser.timeout = 1.0
                ser.dsrdtr = False
                ser.rtscts = False
                ser.open()
                try:
                    # Set DTR/RTS to True to reset the SAMD21 board and bypass boot animation for fast PING response
                    ser.dtr = True
                    ser.rts = True
                except Exception:
                    pass
                # Wait for board to settle
                time.sleep(2.0)
                ser.write(b"PING\n")
                time.sleep(0.2)
                response = ser.read_all().decode('utf-8', errors='ignore')
                if "PONG_FED3" in response:
                    device_id = None
                    if "ID:" in response:
                        try:
                            parts = response.split("ID:")
                            if len(parts) > 1:
                                raw_id = parts[1].strip().split()[0]
                                device_id = "".join([c for c in raw_id if c.isdigit()])
                        except Exception:
                            pass
                    return (dev, "FED3 Active", device_id)
            except Exception as e:
                err_str = str(e)
                if "busy" in err_str.lower() or "already open" in err_str.lower() or "permission denied" in err_str.lower():
                    return (dev, "Busy/In Use", None)
            finally:
                if ser and ser.is_open:
                    try:
                        ser.close()
                    except Exception:
                        pass

            return (dev, "Unresponsive", None)

        # Check ports in parallel to avoid long UI hangs
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(check_port, ping_ports)

        valid_ports = [r for r in results if r is not None]
        # Return all system ports to allow selection of non-FED ports
        all_devs = [p.device for p in ports]
        self.finished_scan.emit(valid_ports, all_devs)
