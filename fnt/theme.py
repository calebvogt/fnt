"""The shared FNT dark look.

Every FNT tool now runs in its own process (see :mod:`fnt.tool_host`), so a
stylesheet set on the launcher window no longer reaches any of them -- each tool
has to establish its own appearance. Without that a tool inherits the native
platform theme, which is dark on macOS and light on Windows, and ends up looking
nothing like the rest of the toolbox.

Pinning the Fusion style plus an explicit palette themes every standard widget
uniformly -- inputs, lists, combo boxes, menus, scrollbars, tooltips -- rather
than only the containers a per-widget stylesheet happens to reach.

Originally written for the Mask Audio Detector; lifted here so tools share one
definition instead of each carrying a copy that drifts.
"""

from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

# Matches the launcher's own stylesheet in fnt/gui_pyqt.py.
WINDOW_BG = QColor(43, 43, 43)
INPUT_BG = QColor(30, 30, 30)
BUTTON_BG = QColor(53, 53, 53)
TEXT = QColor(220, 220, 220)
DISABLED_TEXT = QColor(120, 120, 120)
ACCENT = QColor(0, 120, 212)

#: Blue call-to-action button, matching the other FNT tools.
BLUE_BUTTON_STYLE = """
    QPushButton { background-color: #0078d4; color: #ffffff; padding: 6px 12px;
                  border: none; font-weight: bold; }
    QPushButton:hover { background-color: #1a88e0; }
    QPushButton:pressed { background-color: #006cbe; }
    QPushButton:disabled { background-color: #444444; color: #888888; }
"""


def dark_palette():
    """The FNT dark palette."""
    p = QPalette()
    p.setColor(QPalette.Window, WINDOW_BG)
    p.setColor(QPalette.WindowText, TEXT)
    p.setColor(QPalette.Base, INPUT_BG)
    p.setColor(QPalette.AlternateBase, WINDOW_BG)
    p.setColor(QPalette.ToolTipBase, INPUT_BG)
    p.setColor(QPalette.ToolTipText, TEXT)
    p.setColor(QPalette.Text, TEXT)
    p.setColor(QPalette.Button, BUTTON_BG)
    p.setColor(QPalette.ButtonText, TEXT)
    p.setColor(QPalette.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.Link, ACCENT)
    p.setColor(QPalette.Highlight, ACCENT)
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, DISABLED_TEXT)
    return p


def apply_dark_theme(app=None):
    """Give `app` (default: the running QApplication) the FNT dark look.

    Safe to call more than once and safe to call before any window exists.
    """
    app = app or QApplication.instance()
    if app is None:
        return
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    app.setPalette(dark_palette())
