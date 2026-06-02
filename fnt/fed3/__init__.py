"""fnt.fed3 package: FED3 device helpers.

Expose list_serial_ports and sync_time from fed_comms.py for a cleaner package API.
"""

from .fed_comms import list_serial_ports, sync_time

__all__ = ["list_serial_ports", "sync_time"]
