"""fnt.fed3: FED3 device control, live monitoring and session recording."""

from .fed_comms import list_serial_ports, sync_time
from .fed_widgets import FEDTabWidget

__all__ = ["list_serial_ports", "sync_time", "FEDTabWidget"]
