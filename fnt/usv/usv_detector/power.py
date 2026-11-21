"""Keep long MAD runs off Windows' background power throttling.

Windows 11 throttles processes it judges to be in the background -- anything
without a foreground window, which is every headless ``mad`` run and a
minimized GUI -- and on a hybrid CPU that means efficiency cores and low
clocks. Measured on the lab PC (i9-14900K): the same 10-minute recording took
47 s per file throttled and 27 s with the opt-out, and a 378-file project
re-detection took 6.0 h instead of 3.3 h. The GPU pass slows too, because
feeding it is CPU work.

The opt-out is per process and is not inherited by child processes, so a
tool that runs ffmpeg opts each child out by its pid. Video PreProcessing does:
on 2026-11-01 the lab PC's display went off and its x265 encodes were held on
the 16 E-cores (CPU 96% -> 50%, ~40% fewer videos per hour) until opted out.
"""
from __future__ import annotations

import sys

_PROCESS_POWER_THROTTLING = 4          # PROCESS_INFORMATION_CLASS
_THROTTLING_EXECUTION_SPEED = 0x1
_PROCESS_SET_INFORMATION = 0x0200


def keep_full_speed(pid: int | None = None) -> bool:
    """Ask Windows never to throttle this process's (or ``pid``'s) execution speed.

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
        k32.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        k32.OpenProcess.restype = w.HANDLE
        k32.CloseHandle.argtypes = [w.HANDLE]
        if pid is None:
            handle = k32.GetCurrentProcess()
        else:
            handle = k32.OpenProcess(_PROCESS_SET_INFORMATION, False, int(pid))
            if not handle:
                return False
        # Control execution speed, state off: "never throttle".
        state = _State(1, _THROTTLING_EXECUTION_SPEED, 0)
        try:
            return bool(k32.SetProcessInformation(
                handle, _PROCESS_POWER_THROTTLING,
                ctypes.byref(state), ctypes.sizeof(state)))
        finally:
            if pid is not None:
                k32.CloseHandle(handle)
    except Exception:
        return False
