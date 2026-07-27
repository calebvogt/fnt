"""State for one FED3 device slot.

Groups the three things a slot owns — its Qt widgets, its live connection, and
its recorded data — so the tab widget can hand a device around instead of
threading a dozen parallel dictionaries through every call.

The previous version accepted arbitrary ``**kwargs`` and supported dict-style
subscripting, which meant a typo in a key silently created a new attribute
instead of failing. Fields are explicit here and accessed as attributes.
"""

from datetime import datetime


class FedDevice:
    """One device slot: UI, connection, and recorded state."""

    def __init__(self, slot_num, widgets):
        self.slot_num = slot_num

        # --- UI (populated by the tab when the slot is built) -------------
        for name, widget in widgets.items():
            setattr(self, name, widget)

        # --- identity ----------------------------------------------------
        # Tracks the last committed display name so a rename can be propagated
        # to anything that references the device by name (scheduled events).
        self.last_known_name = f"Device {slot_num}"
        self.saved_port = ""
        self.device_id = None           # on-board ID reported by PING
        self.firmware = None            # firmware version string, None if legacy

        # --- connection --------------------------------------------------
        self.link = None                # Fed3Link while connected
        self.transfer = None            # Fed3Transfer, created with the link
        self.mirror = None              # DeviceMirror while a session is recording
        self.has_connected = False      # ever connected, so a drop is unexpected
        self.connect_attempts = 0
        self.last_sync_time = None
        self.last_device_time = None    # device RTC at the last sync

        # --- recorded state ----------------------------------------------
        self.is_tracking = False
        self.events = []                # datetimes of pellet events, for the plot
        self.stats = {"left": 0, "right": 0, "pellet": 0}
        self.tracking_start_time = None
        self.event_log = None           # DeviceEventLog while recording

    # --- naming -----------------------------------------------------------

    @property
    def name(self):
        """Display name: the user's label, else the slot title."""
        return self.name_edit.text().strip() or f"Device {self.slot_num}"

    @property
    def port(self):
        """Currently selected port, or "" when the slot is unassigned."""
        text = (self.port_combo.currentData()
                or self.port_combo.currentText() or "").strip()
        return "" if text in ("Scanning...", "No FED3 found") else text

    @property
    def is_connected(self):
        return self.link is not None and self.link.is_live()

    # --- state ------------------------------------------------------------

    def reset_counters(self):
        self.stats = {"left": 0, "right": 0, "pellet": 0}
        self.events = []
        self.tracking_start_time = datetime.now()

    def apply_counts(self, counts):
        """Adopt the device's absolute totals.

        The device is authoritative: taking its counts rather than incrementing
        locally means a reconnection resynchronizes instead of under-counting
        everything that happened while the link was down.
        """
        for key in ("left", "right", "pellet"):
            if key in counts:
                self.stats[key] = counts[key]

    def to_state(self):
        """Serializable slot state for session resume."""
        return {
            "slot_num": self.slot_num,
            "name": self.name,
            "port": self.port,
            "device_id": self.device_id,
            "firmware": self.firmware,
            "mode": self.mode_combo.currentText(),
            "ratio": self.ratio_spin.value(),
            "timeout": self.timeout_spin.value(),
            "stats": dict(self.stats),
            "tracking_start_time": (self.tracking_start_time.isoformat()
                                    if self.tracking_start_time else None),
        }
