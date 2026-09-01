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
        # It has to be updated whenever *anything* feeding ``name`` changes, not
        # just when the label is edited: identifying a device changes its
        # default name too.
        self.last_known_name = f"Slot {slot_num}"
        self.saved_port = ""
        self.device_id = None           # on-board ID reported by PING
        self.firmware = None            # FW reported by PING; None until handshake
        self.refused = False            # firmware too old; not to be reconnected
        self.current_file = None        # SD log the device is writing to now

        # --- connection --------------------------------------------------
        self.link = None                # Fed3Link while connected
        self.transfer = None            # Fed3Transfer, created with the link
        self.mirror = None              # DeviceMirror while a session is recording
        self.has_connected = False      # ever connected, so a drop is unexpected
        # The handshake runs once per connection. Heartbeat PINGs are answered
        # with the same PONG, and re-running the handshake on each one would
        # re-set the device clock every 30s for the length of an experiment.
        self.handshake_done = False
        self.connect_attempts = 0
        self.awaiting_pong_since = None  # host_now() when a heartbeat PING went out
        self.reconnect_gave_up = False   # backoff exhausted; already reported
        self.last_sync_time = None
        self.last_device_time = None    # device RTC at the last sync
        # When this device's data last reached disk. Shown on the card, because
        # in an unattended run "still connected" and "still being recorded" are
        # different questions and only the second one matters to the data.
        self.last_mirror_update = None

        # --- recorded state ----------------------------------------------
        self.is_tracking = False
        self.events = []                # datetimes of pellet events, for the plot
        self.stats = {"left": 0, "right": 0, "pellet": 0}
        self.tracking_start_time = None
        self.event_log = None           # DeviceEventLog while recording

    # --- naming -----------------------------------------------------------

    @property
    def label(self):
        """The user's own name for this device, or "" if they have not set one."""
        return self.name_edit.text().strip()

    @property
    def default_name(self):
        """The name to use when the user has not supplied a label.

        A device that has identified itself is called by the number printed on
        it, matching how the devices are referred to at the bench. A slot that
        has no device yet is deliberately *not* called "FED n": it would be
        claiming an identity it has not been given, and an empty card reading
        "FED 1" is indistinguishable from a real FED 1 that has stopped talking.
        """
        if self.device_id:
            return f"FED {self.device_id}"
        return f"Slot {self.slot_num}"

    @property
    def name(self):
        """Display name: the user's label, else :attr:`default_name`."""
        return self.label or self.default_name

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
            "label": self.label,
            "port": self.port,
            "device_id": self.device_id,
            "firmware": self.firmware,
            "current_file": self.current_file,
            "mode": self.mode_combo.currentText(),
            "ratio": self.ratio_spin.value(),
            "timeout": self.timeout_spin.value(),
            "stats": dict(self.stats),
            "tracking_start_time": (self.tracking_start_time.isoformat()
                                    if self.tracking_start_time else None),
        }
