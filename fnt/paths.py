"""Small path helpers shared across FNT tools.

`is_network_path` exists in several tools already (video trim, video
preprocessing, SLEAP inference, UWB preprocessing). New code should import it
from here rather than adding a fifth copy.
"""

import ctypes
import os

#: GetDriveTypeW's code for a network drive.
_DRIVE_REMOTE = 4


def is_network_path(path):
    """True if `path` lives on a network or mapped drive.

    Matters because FNT routinely reads from and writes to lab shares, and a
    share that stalls mid-operation is a real failure mode: the input side of
    the trim and preprocessing tools already stages files locally for exactly
    this reason.
    """
    if not path:
        return False
    text = str(path)
    if text.startswith("\\\\") or text.startswith("//"):
        return True                      # UNC path
    if os.name == "nt":
        drive = os.path.splitdrive(os.path.abspath(text))[0]
        if drive and drive.endswith(":"):
            try:
                return ctypes.windll.kernel32.GetDriveTypeW(
                    drive + "\\") == _DRIVE_REMOTE
            except Exception:
                return False
    return False


def describe_location(path):
    """'network drive (Z:)' or 'local disk', for user-facing messages."""
    if not path:
        return "unset"
    if is_network_path(path):
        drive = os.path.splitdrive(os.path.abspath(str(path)))[0]
        return f"network drive ({drive})" if drive else "network location"
    return "local disk"
