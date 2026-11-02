"""Keep long MAD runs off Windows' background power throttling.

Windows 11 throttles processes it judges to be in the background -- anything
without a foreground window, which is every headless ``mad`` run and a
minimized GUI -- and on a hybrid CPU that means efficiency cores and low
clocks. Measured on the lab PC (i9-14900K): the same 10-minute recording took
47 s per file throttled and 27 s with the opt-out, and a 378-file project
re-detection took 6.0 h instead of 3.3 h. The GPU pass slows too, because
feeding it is CPU work.

The opt-out is per process and is not inherited by child processes.
"""
from __future__ import annotations

import sys

_PROCESS_POWER_THROTTLING = 4          # PROCESS_INFORMATION_CLASS
_THROTTLING_EXECUTION_SPEED = 0x1


def keep_full_speed() -> bool:
    """Ask Windows never to throttle this process's execution speed.

    Best-effort and silent: returns True when the opt-out took, False on any
    other platform or on failure. Never raises -- a run must not fail because
    a scheduling hint could not be set.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        import ctypes.wintypes as w

        class _State(ctypes.Structure):
            _fields_ = [("Version", w.ULONG), ("ControlMask", w.ULONG),
                        ("StateMask", w.ULONG)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Declared, not defaulted: ctypes would pass the 64-bit pseudo-handle
        # as a 32-bit int and the call would fail.
        k32.GetCurrentProcess.restype = w.HANDLE
        k32.SetProcessInformation.argtypes = [w.HANDLE, ctypes.c_int,
                                              ctypes.c_void_p, w.DWORD]
        k32.SetProcessInformation.restype = w.BOOL
        # Control execution speed, state off: "never throttle".
        state = _State(1, _THROTTLING_EXECUTION_SPEED, 0)
        return bool(k32.SetProcessInformation(
            k32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
            ctypes.byref(state), ctypes.sizeof(state)))
    except Exception:
        return False
