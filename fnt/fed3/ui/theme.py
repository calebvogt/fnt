"""Colours and the stylesheet for the FED3 window, matching FNT's dark theme."""

BACKGROUND = "#2b2b2b"
PANEL = "#1e1e1e"
BORDER = "#3f3f3f"
TEXT = "#cccccc"
MUTED = "#8a8a8a"
ACCENT = "#0078d4"
GOOD = "#4caf50"
WARN = "#e0a030"
BAD = "#e05050"

# One colour per device in plots and cards, in binding order.
SERIES = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#fff176",
          "#4db6ac", "#f06292"]

STYLESHEET = f"""
QWidget {{ background-color: {BACKGROUND}; color: {TEXT}; font-size: 10pt; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 4px; margin-top: 14px;
             padding: 8px; font-weight: bold; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
QPushButton {{ background-color: {ACCENT}; color: white; border: none;
               padding: 6px 14px; border-radius: 4px; font-weight: bold; }}
QPushButton:hover {{ background-color: #106ebe; }}
QPushButton:pressed {{ background-color: #005a9e; }}
QPushButton:disabled {{ background-color: {BORDER}; color: {MUTED}; padding: 6px 14px; }}
QPushButton[quiet="true"] {{ background-color: {PANEL}; color: {TEXT};
                             border: 1px solid {BORDER}; font-weight: normal; }}
QPushButton[quiet="true"]:hover {{ background-color: {BORDER}; }}
QPushButton[danger="true"] {{ background-color: #a33; }}
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QListWidget, QTableWidget {{
    background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 3px; padding: 3px; }}
QPlainTextEdit {{ font-family: Consolas, "DejaVu Sans Mono", monospace; }}
QHeaderView::section {{ background-color: {PANEL}; border: none;
                        border-bottom: 1px solid {BORDER}; padding: 4px; }}
QFrame#card {{ background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QFrame#card[problem="true"] {{ border: 2px solid {BAD}; }}
QFrame#card QLabel {{ background: transparent; }}
QLabel#title {{ font-size: 14pt; font-weight: bold; }}
QLabel#muted {{ color: {MUTED}; }}
QToolTip {{ background-color: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; }}
"""


def quiet(button):
    """A secondary button: outlined instead of filled."""
    button.setProperty("quiet", True)
    return button


def danger(button):
    button.setProperty("danger", True)
    return button
