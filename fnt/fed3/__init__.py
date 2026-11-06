"""fnt.fed3: FED3 device control, live monitoring and session recording.

Nothing is imported eagerly: the protocol compiler and the ``fnt-fed`` command
must load without Qt. The names below resolve on first use.
"""
import importlib

_LAZY = {
    "FedWindow": "ui.window",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
