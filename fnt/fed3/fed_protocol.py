"""FED3 serial wire protocol.

Single source of truth for what the host sends and what the device sends back,
so parsing is not re-derived in the serial thread, the export manager and the
GUI. Mirrors ``ClassicFed3withTimeSync.ino`` in the fnt-fed3 repository.

Host -> device
--------------
``PING``                     discovery handshake
``STATUS``                   full device state snapshot
``SYNC:YYYY,MM,DD,HH,MM,SS`` set the RTC
``LIST_FILES``               enumerate FED*.CSV on the SD card
``FSIZE:<name>``             current size of one file (cheap "anything new?")
``GET_FILE:<name>[,<off>]``  stream a file from a byte offset
``ABORT``                    cancel an in-flight transfer
``NEW_TRIAL``                zero counters and roll a new SD file
``FEED`` / ``LIGHTS:ON`` / ``LIGHTS:OFF``
``MODE:<spec>``              see MODE_COMMANDS

Device -> host
--------------
``PONG_FED3,ID:<n>,FW:<v>``
``STATUS,FW:..,ID:..,TIME:..,MODE:..,SESSION:..,FR:..,L:..,R:..,P:..,FILE:..``
``EVT,<iso>,<LEFT|RIGHT|PELLET>,<left>,<right>,<pellet>,<millis>``
``SYNCED,<iso>``
``FILE:<name>,<bytes>`` ... ``END_LIST``
``FSIZE:<name>,<bytes>``
``FILE_DATA_START:<name>,<offset>,<size>`` then raw bytes, ``0x04``, ``CRC32:<hex>``
``ERROR:<code>[:<detail>]``

Firmware older than 2.0 emits prose event lines ("Left Poke, Total: 12") and a
``FILE_DATA_START:<name>`` header with no range. Both are still parsed so a
device that has not been reflashed keeps working, with reduced fidelity.
"""

from datetime import datetime

FW_VERSION_REQUIRED = "2.0"

EOT = "\x04"

# --- commands -------------------------------------------------------------

CMD_PING = "PING"
CMD_STATUS = "STATUS"
CMD_ABORT = "ABORT"
CMD_LIST_FILES = "LIST_FILES"
CMD_NEW_TRIAL = "NEW_TRIAL"
CMD_FEED = "FEED"
CMD_LIGHTS_ON = "LIGHTS:ON"
CMD_LIGHTS_OFF = "LIGHTS:OFF"


def cmd_sync(when=None):
    return (when or datetime.now()).strftime("SYNC:%Y,%m,%d,%H,%M,%S")


def cmd_get_file(filename, offset=0):
    return f"GET_FILE:{filename},{int(offset)}"


def cmd_file_size(filename):
    return f"FSIZE:{filename}"


# Mode label (as shown in the GUI) -> builder returning (command, description).
# Keeping this as data rather than a 60-line if/elif chain means the device
# panel, the global panel and the scheduler all emit provably identical commands.
def _fr(ratio, _timeout):
    if ratio in (1, 3, 5):
        return f"MODE:FR{ratio}", f"FR{ratio}"
    return f"MODE:FR,{ratio}", f"FR (ratio {ratio})"


MODE_COMMANDS = {
    "Fixed Ratio (FR)":      (_fr,                                                 ("ratio",)),
    "Progressive Ratio (PR)": (lambda r, t: ("MODE:PR", "PR"),                     ()),
    "Random Ratio (RR)":     (lambda r, t: (f"MODE:RR,{r}", f"RR (avg {r})"),      ("ratio",)),
    "FR with Timeout":       (lambda r, t: (f"MODE:FRTO,{r},{t}",
                                            f"FR{r} timeout {t}s"),                ("ratio", "timeout")),
    "Free Feeding":          (lambda r, t: ("MODE:FREE", "Free feeding"),          ()),
    "Extinction":            (lambda r, t: ("MODE:EXTINCT", "Extinction"),         ()),
    "Light Tracking":        (lambda r, t: ("MODE:LIGHTTRK", "Light tracking"),    ()),
    "FR Reversed":           (lambda r, t: (("MODE:FR1_R", "FR1 reversed") if r == 1
                                            else (f"MODE:FR_R,{r}", f"FR{r} reversed")), ("ratio",)),
    "PR Reversed":           (lambda r, t: ("MODE:PR_R", "PR reversed"),           ()),
    "Opto Stimulation":      (lambda r, t: ("MODE:OPTO", "Opto stim"),             ()),
    "Opto Reversed":         (lambda r, t: ("MODE:OPTO_R", "Opto stim reversed"),  ()),
    "Timed Feeding":         (lambda r, t: ("MODE:TIMED", "Timed feeding"),        ()),
}

MODE_LABELS = list(MODE_COMMANDS)


def mode_command(label, ratio=1, timeout=30):
    """Return ``(command, human_description)`` for a GUI mode label."""
    entry = MODE_COMMANDS.get(label)
    if entry is None:
        return None, None
    builder, _fields = entry
    return builder(ratio, timeout)


def mode_fields(label):
    """Which parameter widgets a mode needs: subset of ``{"ratio", "timeout"}``."""
    entry = MODE_COMMANDS.get(label)
    return set(entry[1]) if entry else set()


# --- device -> host parsing ----------------------------------------------

EVENT_LEFT = "LEFT"
EVENT_RIGHT = "RIGHT"
EVENT_PELLET = "PELLET"


class DeviceEvent:
    """One behavioural event reported by the device.

    ``counts`` are absolute running totals, not deltas, so a host that dropped a
    line (or reconnected mid-session) resynchronizes on the next event.
    ``device_time`` is the device RTC; None on legacy firmware.
    """

    __slots__ = ("kind", "device_time", "counts", "device_millis")

    def __init__(self, kind, device_time=None, counts=None, device_millis=None):
        self.kind = kind
        self.device_time = device_time
        self.counts = counts or {}
        self.device_millis = device_millis


def parse_event(line):
    """Parse a behavioural event line, or return None if it is not one."""
    stripped = line.strip()

    if stripped.startswith("EVT,"):
        parts = stripped.split(",")
        if len(parts) < 6:
            return None
        kind = parts[2].strip().upper()
        if kind not in (EVENT_LEFT, EVENT_RIGHT, EVENT_PELLET):
            return None
        try:
            counts = {
                "left": int(parts[3]),
                "right": int(parts[4]),
                "pellet": int(parts[5]),
            }
        except ValueError:
            return None
        millis = None
        if len(parts) > 6:
            try:
                millis = int(parts[6])
            except ValueError:
                pass
        return DeviceEvent(kind, _parse_iso(parts[1]), counts, millis)

    return _parse_legacy_event(stripped)


def _parse_legacy_event(stripped):
    """Pre-2.0 prose form: ``Left Poke, Total: 12``."""
    upper = stripped.upper()
    if "TOTAL:" not in upper:
        return None
    if upper.startswith("LEFT POKE"):
        kind, key = EVENT_LEFT, "left"
    elif upper.startswith("RIGHT POKE"):
        kind, key = EVENT_RIGHT, "right"
    elif upper.startswith("PELLET"):
        kind, key = EVENT_PELLET, "pellet"
    else:
        return None
    try:
        total = int(upper.split("TOTAL:")[1].strip().split()[0])
    except (IndexError, ValueError):
        return None
    return DeviceEvent(kind, None, {key: total})


def parse_pong(text):
    """Extract ``(device_id, firmware)`` from a PING response blob."""
    if "PONG_FED3" not in text:
        return None, None
    device_id = firmware = None
    if "ID:" in text:
        raw = text.split("ID:", 1)[1].strip().split(",")[0].split()[0]
        digits = "".join(c for c in raw if c.isdigit())
        device_id = digits or None
    if "FW:" in text:
        firmware = text.split("FW:", 1)[1].strip().split(",")[0].split()[0] or None
    return device_id, firmware


def parse_status(line):
    """Parse a ``STATUS,K:V,...`` line into a dict, or None."""
    stripped = line.strip()
    if not stripped.startswith("STATUS,"):
        return None
    status = {}
    for token in stripped[len("STATUS,"):].split(","):
        if ":" in token:
            key, _, value = token.partition(":")
            status[key.strip().lower()] = value.strip()
    return status


def parse_file_entry(line):
    """``FILE:<name>,<bytes>`` -> ``(name, size)``."""
    stripped = line.strip()
    if not stripped.startswith("FILE:"):
        return None
    name, _, size = stripped[len("FILE:"):].partition(",")
    try:
        return name.strip(), int(size.strip())
    except ValueError:
        return name.strip(), 0


def parse_file_size(line):
    """``FSIZE:<name>,<bytes>`` -> ``(name, size)``."""
    stripped = line.strip()
    if not stripped.startswith("FSIZE:"):
        return None
    name, _, size = stripped[len("FSIZE:"):].partition(",")
    try:
        return name.strip(), int(size.strip())
    except ValueError:
        return None


def parse_data_start(line):
    """``FILE_DATA_START:<name>[,<offset>,<size>]`` -> ``(name, offset, size)``.

    Legacy firmware omits the range; offset 0 and an unknown size are assumed.
    """
    stripped = line.strip()
    if not stripped.startswith("FILE_DATA_START:"):
        return None
    parts = stripped[len("FILE_DATA_START:"):].split(",")
    name = parts[0].strip()
    offset, size = 0, None
    try:
        if len(parts) > 1:
            offset = int(parts[1])
        if len(parts) > 2:
            size = int(parts[2])
    except ValueError:
        pass
    return name, offset, size


def parse_crc(line):
    """``CRC32:<hex>`` -> int, or None."""
    stripped = line.strip()
    if not stripped.startswith("CRC32:"):
        return None
    try:
        return int(stripped[len("CRC32:"):].strip(), 16)
    except ValueError:
        return None


def is_error(line):
    return line.strip().startswith("ERROR:")


def _parse_iso(text):
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%dT%H:%M:%S")
    except (ValueError, AttributeError):
        return None
