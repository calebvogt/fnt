"""Keep the test suite out of the user's real application settings.

The GUI tests build real windows, and the windows save what they remember --
splitter layout, detection settings, recent projects -- to QSettings, which on
Windows is the user's registry. An offscreen window has no real fonts, so its
"remembered" layout is nonsense: a test run left MAD's saved splitter at
924 + 1472 px, wider than the screen, and the next real launch opened with a
300 px dead band beside the spectrogram.

Every ``QSettings(organization, application)`` made during a test session is
redirected to an INI file in a throwaway folder instead. Patched on the class
in ``PyQt5.QtCore`` before any FNT module is imported (pytest loads this file
first), so modules that bind ``QSettings`` at import pick up the redirect too.
"""
import atexit
import os
import shutil
import tempfile

try:
    from PyQt5 import QtCore
except Exception:      # PyQt5-free test runs need nothing here
    QtCore = None

if QtCore is not None and not getattr(QtCore.QSettings, "_fnt_isolated", False):
    _DIR = tempfile.mkdtemp(prefix="fnt_test_settings_")
    atexit.register(shutil.rmtree, _DIR, True)
    _Real = QtCore.QSettings

    class _IsolatedSettings(_Real):
        _fnt_isolated = True

        def __init__(self, *args, **kwargs):
            org_app = (len(args) >= 2 and isinstance(args[0], str)
                       and isinstance(args[1], str)
                       and not os.path.splitext(args[0])[1])
            if org_app:
                path = os.path.join(_DIR, f"{args[0]}_{args[1]}.ini")
                super().__init__(path, _Real.IniFormat, *args[2:], **kwargs)
            elif not args and not kwargs:
                super().__init__(os.path.join(_DIR, "default.ini"),
                                 _Real.IniFormat)
            else:
                super().__init__(*args, **kwargs)

    QtCore.QSettings = _IsolatedSettings
