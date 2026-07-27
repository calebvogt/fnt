"""FED3 export and file download protocol.

Implements the file list and download state machines with liveness timeouts
to prevent GUI hangs if serial transfer gets interrupted.
"""

import os
import zlib
from PyQt5.QtCore import QTimer, Qt

class FedDownloadManager:
    def __init__(self, parent_widget=None):
        self.parent_widget = parent_widget

    def _setup_timeout(self, device, timeout_ms=30000):
        """Start or restart a liveness timer for the device download/list session."""
        if device.download_timer is not None:
            device.download_timer.stop()
            device.download_timer.deleteLater()
            device.download_timer = None

        timer = QTimer(self.parent_widget)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._handle_timeout(device))
        device.download_timer = timer
        timer.start(timeout_ms)

    def _clear_timeout(self, device):
        """Stop and clear the liveness timer."""
        if device.download_timer is not None:
            device.download_timer.stop()
            device.download_timer.deleteLater()
            device.download_timer = None

    def _handle_timeout(self, device):
        """Called when no data is received within the timeout window."""
        self._clear_timeout(device)
        dev_name = device.name_edit.text().strip() or device.box.title()

        if device.file_list_pending:
            device.file_list_pending = False
            callback = device.file_list_callback
            if callback:
                device.file_list_callback = None
                callback(False, f"Timeout: No response from {dev_name} for 30 seconds.")

        elif device.download_pending:
            device.download_pending = False
            device.download_started = False
            device.download_waiting_crc = False
            callback = device.download_callback
            if callback:
                device.download_callback = None
                callback(False, f"Timeout: No response from {dev_name} during file transfer for 30 seconds.", "")

    def start_file_list(self, device, callback):
        """Initiate the file listing state on the device."""
        self._clear_timeout(device)
        device.file_list_buffer = []
        device.file_list_callback = callback
        device.file_list_pending = True
        self._setup_timeout(device)

    def start_download(self, device, filename, callback):
        """Initiate the file download state on the device."""
        self._clear_timeout(device)
        device.download_filename = filename
        device.download_lines = []
        device.download_started = False
        device.download_waiting_crc = False
        device.download_callback = callback
        device.download_pending = True
        self._setup_timeout(device)

    def cancel_operations(self, device):
        """Safely cancel any active file transfer operations."""
        self._clear_timeout(device)
        device.file_list_pending = False
        device.file_list_callback = None
        device.download_pending = False
        device.download_started = False
        device.download_waiting_crc = False
        device.download_callback = None

    def handle_line(self, device, line):
        """Process an incoming serial line, checking if it belongs to list or download.
        
        Returns:
            bool: True if the line was handled/intercepted, False otherwise.
        """
        stripped_line = line.strip()

        # 1. Intercept CRC line in file download mode
        if device.download_pending and device.download_waiting_crc:
            self._clear_timeout(device)
            if stripped_line.startswith("CRC32:"):
                crc_str = stripped_line[6:].strip()
                device.download_pending = False
                device.download_waiting_crc = False
                callback = device.download_callback
                if callback:
                    device.download_callback = None
                    callback(True, device.download_lines, crc_str)
            elif stripped_line.startswith("ERROR:"):
                device.download_pending = False
                device.download_waiting_crc = False
                callback = device.download_callback
                if callback:
                    device.download_callback = None
                    callback(False, stripped_line, "")
            else:
                # If we're waiting for CRC but got something else, keep timeout active
                self._setup_timeout(device)
            return True

        # 2. Intercept file list mode
        if device.file_list_pending:
            self._setup_timeout(device) # Reset liveness timeout on activity
            if stripped_line.startswith("FILE:"):
                parts = stripped_line[5:].split(',')
                if len(parts) == 2:
                    filename = parts[0].strip()
                    try:
                        size = int(parts[1].strip())
                    except ValueError:
                        size = 0
                    device.file_list_buffer.append((filename, size))
            elif stripped_line == "END_LIST":
                self._clear_timeout(device)
                device.file_list_pending = False
                callback = device.file_list_callback
                if callback:
                    device.file_list_callback = None
                    callback(True, device.file_list_buffer)
            elif stripped_line.startswith("ERROR:"):
                self._clear_timeout(device)
                device.file_list_pending = False
                callback = device.file_list_callback
                if callback:
                    device.file_list_callback = None
                    callback(False, stripped_line)
            return True

        # 3. Intercept file download mode
        if device.download_pending:
            self._setup_timeout(device) # Reset liveness timeout on activity
            filename = device.download_filename
            
            if not device.download_started:
                if stripped_line == f"FILE_DATA_START:{filename}":
                    device.download_started = True
                    device.download_waiting_crc = False
                    device.download_lines = []
                elif stripped_line.startswith("ERROR:"):
                    self._clear_timeout(device)
                    device.download_pending = False
                    callback = device.download_callback
                    if callback:
                        device.download_callback = None
                        callback(False, stripped_line, "")
                return True
                
            # If download started, check if it's the end / EOT (\x04)
            if "\x04" in line:
                parts = line.split("\x04")
                if parts[0]:
                    device.download_lines.append(parts[0])
                # Data ended, wait for next line to get CRC
                device.download_started = False
                device.download_waiting_crc = True
                self._setup_timeout(device)
                return True
                
            device.download_lines.append(line)
            return True

        return False
