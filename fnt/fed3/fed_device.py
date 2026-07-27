"""FED3 Device data structure.

Defines the FedDevice class, wrapping device state, UI components,
and tracking/download variables. Supports both attribute access
and dictionary-like subscript access for backward compatibility.
"""

class FedDevice:
    def __init__(self, **kwargs):
        # UI Elements
        self.box = kwargs.get('box')
        self.slot_num = kwargs.get('slot_num')
        self.name_edit = kwargs.get('name_edit')
        self.port_combo = kwargs.get('port_combo')
        self.remove_btn = kwargs.get('remove_btn')
        self.mode_combo = kwargs.get('mode_combo')
        self.apply_btn = kwargs.get('apply_btn')
        self.fr_label = kwargs.get('fr_label')
        self.ratio_spin = kwargs.get('ratio_spin')
        self.timeout_label = kwargs.get('timeout_label')
        self.timeout_spin = kwargs.get('timeout_spin')
        self.timeout_unit_label = kwargs.get('timeout_unit_label')
        self.last_sync_label = kwargs.get('last_sync_label')
        self.feed_btn = kwargs.get('feed_btn')
        self.lights_toggle_btn = kwargs.get('lights_toggle_btn')
        self.reset_btn = kwargs.get('reset_btn')
        self.export_btn = kwargs.get('export_btn')
        self.timer = kwargs.get('timer')
        self.svg_container = kwargs.get('svg_container')
        self.svg_title = kwargs.get('svg_title')
        self.svg_view = kwargs.get('svg_view')

        # State Variables
        self.is_syncing = kwargs.get('is_syncing', False)
        self.is_tracking = kwargs.get('is_tracking', False)
        self.tracker_worker = kwargs.get('tracker_worker', None)
        self.log_file = kwargs.get('log_file', None)
        self.events = kwargs.get('events', [])
        self.tracking_start_time = kwargs.get('tracking_start_time', None)

        # File List state
        self.file_list_pending = kwargs.get('file_list_pending', False)
        self.file_list_buffer = kwargs.get('file_list_buffer', [])
        self.file_list_callback = kwargs.get('file_list_callback', None)

        # File Download state
        self.download_pending = kwargs.get('download_pending', False)
        self.download_filename = kwargs.get('download_filename', None)
        self.download_lines = kwargs.get('download_lines', [])
        self.download_started = kwargs.get('download_started', False)
        self.download_waiting_crc = kwargs.get('download_waiting_crc', False)
        self.download_callback = kwargs.get('download_callback', None)

        # Download safety timer to prevent indefinite hangs
        self.download_timer = kwargs.get('download_timer', None)

        # Statistics / counters
        self.stats = kwargs.get('stats', {'left': 0, 'right': 0, 'pellet': 0})

    def __getitem__(self, key):
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def set(self, key, value):
        setattr(self, key, value)
