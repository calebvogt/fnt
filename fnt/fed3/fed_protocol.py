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
``PONG_FED3,ID:<n>[,FW:<v>]``
``STATUS,ID:..,FW:..,TIME:..,MODE:..,SESSION:..,FR:..,L:..,R:..,P:..,FILE:..``
``EVT,<iso>,<LEFT|RIGHT|PELLET>,<left>,<right>,<pellet>,<millis>``
``SYNCED,<iso>``
``FILE:<name>,<bytes>`` ... ``END_LIST``
``FSIZE:<name>,<bytes>``
``FILE_DATA_START:<name>,<offset>,<size>`` then raw bytes, ``0x04``, ``CRC32:<hex>``
``ERROR:<code>[:<detail>]``

Firmware requirement
--------------------
Firmware at :data:`MIN_FIRMWARE` or newer answers ``PING`` with a ``FW:`` field.
A board that does not is refused outright rather than driven in a reduced mode.

That is a deliberate choice. Older firmware parses ``GET_FILE:`` by taking
everything after the colon as the filename, so the documented
``GET_FILE:<name>,<offset>`` asks it for a file literally named ``"<name>,0"``
and gets ``ERROR:FILE_NOT_FOUND`` — every download and every mirror pull fails.
Supporting that alongside the current protocol would mean two parsers, two
transfer modes and two event formats, and it is precisely those extra paths
where silent data loss has hidden before. :func:`is_supported` gates a clear
error instead.
"""

from datetime import datetime

EOT = "\x04"

# Firmware from this version on supports ranged GET_FILE, STATUS, FSIZE and
# structured EVT lines, and announces itself in the PING response. Anything
# older is refused; see the module docstring.
MIN_FIRMWARE = (2, 0)


def parse_version(firmware):
    """``"2.1"`` -> ``(2, 1)``. None for anything unparseable.

    Short versions are zero-padded to the length of :data:`MIN_FIRMWARE`.
    Without that, ``"2"`` parsed to ``(2,)``, and a shorter tuple sorts *below*
    a longer one with the same prefix — so a device announcing ``FW:2`` was
    refused as too old while ``FW:3`` was accepted.
    """
    if not firmware:
        return None
    parts = []
    for chunk in str(firmware).strip().split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        return None
    parts += [0] * (len(MIN_FIRMWARE) - len(parts))
    return tuple(parts)


def is_supported(firmware):
    """Whether FNT can drive a device running this firmware at all."""
    version = parse_version(firmware)
    return version is not None and version >= MIN_FIRMWARE


def firmware_requirement():
    """Human-readable statement of what a device must be running."""
    return ".".join(str(part) for part in MIN_FIRMWARE)

# --- commands -------------------------------------------------------------

CMD_PING = "PING"
CMD_STATUS = "STATUS"
CMD_ABORT = "ABORT"
CMD_LIST_FILES = "LIST_FILES"
CMD_NEW_TRIAL = "NEW_TRIAL"
CMD_FEED = "FEED"
CMD_LIGHTS_ON = "LIGHTS:ON"
CMD_LIGHTS_OFF = "LIGHTS:OFF"

# Reply prefix naming the SD log a device rolled onto after NEW_TRIAL.
REPLY_NEW_TRIAL = "NEW_TRIAL_STARTED:"

# Progress during a dispense, one line per attempt: ``FEEDING:<turn>/<max>``.
# A jammed hopper can take minutes of motor work before the device gives up, and
# without this the host cannot tell that from a device that has stopped talking.
REPLY_FEEDING = "FEEDING:"


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

# Not a behaviour: the device could not deliver a pellet and gave up. It travels
# as an event so that it is timestamped by the device, queued rather than spliced
# during a transfer, and lands in events.csv beside the pokes it interrupts.
EVENT_JAM = "JAM"

EVENT_KINDS = (EVENT_LEFT, EVENT_RIGHT, EVENT_PELLET, EVENT_JAM)


class DeviceEvent:
    """One behavioural event reported by the device.

    ``counts`` are absolute running totals, not deltas, so a host that dropped a
    line (or reconnected mid-session) resynchronizes on the next event.
    ``device_time`` is the device RTC.
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
        if kind not in EVENT_KINDS:
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

    return None


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


def parse_feeding(line):
    """``FEEDING:<turn>/<max>`` -> ``(turn, max)``, or None."""
    stripped = line.strip()
    if not stripped.startswith(REPLY_FEEDING):
        return None
    turn, _, total = stripped[len(REPLY_FEEDING):].partition("/")
    try:
        return int(turn), int(total)
    except ValueError:
        return None


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
    """``FILE_DATA_START:<name>,<offset>,<size>`` -> ``(name, offset, size)``."""
    stripped = line.strip()
    if not stripped.startswith("FILE_DATA_START:"):
        return None
    parts = stripped[len("FILE_DATA_START:"):].split(",")
    if len(parts) < 3:
        return None
    try:
        return parts[0].strip(), int(parts[1]), int(parts[2])
    except ValueError:
        return None


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


# Asynchronous device chatter: lines the device can emit at any moment, which are
# never part of a transfer's own protocol. Firmware 2.0's file streamer runs in
# loop() alongside event reporting, so a nosepoke during a download lands in the
# middle of the payload; without this list the line would be written into the
# mirrored CSV (failing the CRC) *and* lost as an event.
#
# None of these prefixes can begin a FED3 CSV row — those start with the header
# word ``MM:DD:YYYY`` or a ``MM:DD:YYYY hh:mm:ss`` stamp — so matching them
# cannot swallow file content.
_OUT_OF_BAND_PREFIXES = (
    "EVT,",
    "STATUS,",
    "SYNCED,",
    "PONG_FED3",
    REPLY_NEW_TRIAL,
    REPLY_FEEDING,
    "ABORT_OK",
    "Mode set to",
    "Pellet dispensed manually.",
    "Lights turned",
)


def is_out_of_band(line):
    """Whether a line is asynchronous chatter rather than transfer payload."""
    stripped = line.strip()
    return any(stripped.startswith(prefix) for prefix in _OUT_OF_BAND_PREFIXES)


def _parse_iso(text):
    try:
        return datetime.strptime(text.strip(), "%Y-%m-%dT%H:%M:%S")
    except (ValueError, AttributeError):
        return None
