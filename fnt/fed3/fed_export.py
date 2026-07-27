"""SD-card transfer state machine for one FED3 device.

Handles ``LIST_FILES``, ``FSIZE`` and ``GET_FILE`` against a single device, one
operation at a time — the firmware answers ``ERROR:STREAM_BUSY`` if a second
transfer is requested while one is in flight, so overlapping requests are
prevented here rather than recovered from.

Two things changed from the original implementation:

* Downloads are **range-based**. ``GET_FILE:<name>,<offset>`` returns only the
  bytes after ``offset``, which is what makes an interrupted transfer resumable
  and lets a growing log be tailed instead of re-sent in full.
* A timeout now sends ``ABORT`` to the device before giving up. Previously the
  host dropped the operation while the device carried on streaming into a buffer
  nobody was draining, which left the next command talking to the tail of a
  half-finished file.
"""

import zlib

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from . import fed_protocol as proto

# Generous: the device yields between chunks, so a large file legitimately takes
# a while. This only has to catch a device that has stopped responding entirely.
TIMEOUT_MS = 20000

_IDLE, _LISTING, _SIZING, _DOWNLOADING, _AWAITING_CRC = range(5)


class Fed3Transfer(QObject):
    """Serialises SD-card operations for one device.

    ``send`` is a callable taking a command string and returning True if it was
    queued to the device.
    """

    log = pyqtSignal(str, bool)      # message, ok

    def __init__(self, send, parent=None):
        super().__init__(parent)
        self._send = send
        self._state = _IDLE
        self._callback = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

        self._filename = None
        self._offset = 0
        self._expected_size = None
        self._lines = []

    # --- public API -------------------------------------------------------

    @property
    def busy(self):
        return self._state != _IDLE

    def list_files(self, callback):
        """``callback(ok, [(name, size), ...] | error_message)``."""
        return self._begin(_LISTING, callback, proto.CMD_LIST_FILES)

    def file_size(self, filename, callback):
        """``callback(ok, size | error_message)``."""
        self._filename = filename
        return self._begin(_SIZING, callback, proto.cmd_file_size(filename))

    def download(self, filename, offset, callback):
        """Fetch ``filename`` from ``offset``.

        ``callback(ok, payload_bytes | error_message, start_offset)`` — the
        offset is echoed back because the device clamps it when the host's
        mirror is somehow ahead of the file (e.g. the log was rotated).
        """
        self._filename = filename
        self._offset = offset
        return self._begin(_DOWNLOADING, callback,
                           proto.cmd_get_file(filename, offset))

    def cancel(self, reason="cancelled"):
        """Abandon the current operation and tell the device to stop sending."""
        if self._state == _IDLE:
            return
        was_streaming = self._state in (_DOWNLOADING, _AWAITING_CRC)
        callback = self._finish()
        if was_streaming:
            self._send(proto.CMD_ABORT)
        if callback:
            self._invoke(callback, False, reason)

    # --- line handling ----------------------------------------------------

    def handle_line(self, line):
        """Consume a line if it belongs to the active operation.

        Returns True when the line was part of a transfer and should not be
        treated as a normal device message.
        """
        if self._state == _IDLE:
            return False

        self._timer.start(TIMEOUT_MS)     # any activity keeps the operation alive
        stripped = line.strip()

        if self._state == _LISTING:
            return self._handle_listing(stripped)
        if self._state == _SIZING:
            return self._handle_sizing(stripped)
        if self._state == _AWAITING_CRC:
            return self._handle_crc(stripped)
        return self._handle_download(line, stripped)

    # --- per-state handlers ----------------------------------------------

    def _handle_listing(self, stripped):
        entry = proto.parse_file_entry(stripped)
        if entry is not None:
            self._lines.append(entry)
            return True
        if stripped == "END_LIST":
            files = self._lines
            callback = self._finish()
            self._invoke(callback, True, files)
            return True
        if proto.is_error(stripped):
            self._fail(stripped)
            return True
        return True     # ignore chatter that arrives mid-listing

    def _handle_sizing(self, stripped):
        parsed = proto.parse_file_size(stripped)
        if parsed is not None:
            name, size = parsed
            if name == self._filename:
                callback = self._finish()
                self._invoke(callback, True, size)
            return True
        if proto.is_error(stripped):
            self._fail(stripped)
            return True
        return True

    def _handle_download(self, line, stripped):
        if self._expected_size is None:
            # Still waiting for the header.
            header = proto.parse_data_start(stripped)
            if header is not None:
                name, offset, size = header
                self._offset = offset          # device may have clamped it
                self._expected_size = size if size is not None else -1
                return True
            if proto.is_error(stripped):
                self._fail(stripped)
                return True
            return True

        if proto.EOT in line:
            head, _, _tail = line.partition(proto.EOT)
            if head:
                self._lines.append(head)
            self._state = _AWAITING_CRC
            return True

        # The device aborts a stream (host stalled, USB lost, NEW_TRIAL) by
        # emitting an error in the middle of the payload. Without this the error
        # line would be appended as file content and the transfer would hang
        # until the timeout, leaving the mirror unable to retry.
        if proto.is_error(stripped):
            self._fail(stripped)
            return True

        self._lines.append(line)
        return True

    def _handle_crc(self, stripped):
        crc = proto.parse_crc(stripped)
        if crc is not None:
            payload = "".join(self._lines).encode("utf-8", errors="replace")
            offset = self._offset
            callback = self._finish()
            calculated = zlib.crc32(payload) & 0xFFFFFFFF
            if calculated != crc:
                self._invoke(
                    callback, False,
                    f"CRC32 mismatch (device {crc:08X}, received {calculated:08X}); "
                    f"{len(payload)} bytes discarded")
            else:
                self._invoke(callback, True, payload, offset)
            return True
        if proto.is_error(stripped):
            self._fail(stripped)
            return True
        return True     # a stray line before the CRC; keep waiting

    # --- plumbing ---------------------------------------------------------

    def _begin(self, state, callback, command):
        if self.busy:
            self._invoke(callback, False, "device is busy with another transfer")
            return False
        self._state = state
        self._callback = callback
        # Reset the accumulator before sending: a device can answer before the
        # send call returns, and leftovers from the previous operation would
        # otherwise be delivered as part of this one.
        self._lines = []
        self._expected_size = None
        self._timer.start(TIMEOUT_MS)
        if not self._send(command):
            callback = self._finish()
            self._invoke(callback, False, f"could not send {command}")
            return False
        return True

    def _finish(self):
        """Return to idle and hand back the pending callback."""
        self._timer.stop()
        callback = self._callback
        self._callback = None
        self._state = _IDLE
        self._expected_size = None
        return callback

    def _fail(self, message):
        callback = self._finish()
        self._invoke(callback, False, message)

    def _on_timeout(self):
        was_streaming = self._state in (_DOWNLOADING, _AWAITING_CRC)
        callback = self._finish()
        if was_streaming:
            # Stop the device streaming into a buffer nobody is reading.
            self._send(proto.CMD_ABORT)
        self._invoke(callback, False,
                     f"device stopped responding after {TIMEOUT_MS // 1000}s")

    @staticmethod
    def _invoke(callback, ok, payload, *extra):
        """Call a pending callback. Download callbacks must default their
        ``offset`` parameter, since failure paths report only (ok, message)."""
        if callback is not None:
            callback(ok, payload, *extra)
