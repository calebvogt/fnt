"""Single source of truth for MuseStudio's visual language.

Colours are chosen so meaning is consistent everywhere:

* **Frequency bands** run cool -> warm with frequency (delta violet ... gamma
  coral), and the same colour is used for a band in the bar chart, the history
  plot, the band selector and the spectrogram legend.
* **Electrodes** are hemisphere-coded — left sensors are blues, right sensors
  are ambers — so any left/right asymmetry is visible at a glance without
  reading labels.
"""

# --- surfaces & text -------------------------------------------------------
BG = "#0E1116"          # window background
SURFACE = "#161B22"     # panels / group boxes
SURFACE_HI = "#1C2229"  # inputs, raised rows
BORDER = "#2A323C"
BORDER_HI = "#3A4552"

TEXT = "#E6EDF3"
TEXT_DIM = "#9AA7B4"
TEXT_FAINT = "#6E7B8A"

ACCENT = "#4CC2FF"
ACCENT_HI = "#6FD0FF"
ACCENT_DIM = "#1F6F94"
WARN = "#FFB020"
DANGER = "#FF6B6B"
GOOD = "#16C79A"

PLOT_BG = "#11151B"
GRID = "#232B35"

# --- frequency bands (cool -> warm with frequency) -------------------------
BAND_COLORS = {
    "delta": "#7C5CFF",
    "theta": "#2E9BFF",
    "alpha": "#16C79A",
    "beta": "#FFB020",
    "gamma": "#FF6B6B",
}

# --- electrodes: left = blues, right = ambers ------------------------------
ELECTRODE_COLORS = {
    "TP9": "#4CC2FF",
    "AF7": "#2E9BFF",
    "AF8": "#FFB020",
    "TP10": "#FF8F3C",
    "AUX": "#8B949E",
}
LEFT_COLOR = "#2E9BFF"
RIGHT_COLOR = "#FFB020"

# Fallback cycle for streams we don't have semantic colours for (optics etc.).
SERIES_COLORS = [
    "#4CC2FF", "#16C79A", "#FFB020", "#FF6B6B",
    "#7C5CFF", "#FF8FD0", "#59D4E8", "#A8B4C0",
]


def electrode_color(name, index=0):
    """Colour for a channel: hemisphere-coded when we recognise the label."""
    upper = str(name).upper()
    for key, colour in ELECTRODE_COLORS.items():
        if key in upper:
            return colour
    return SERIES_COLORS[index % len(SERIES_COLORS)]


def band_color(band):
    return BAND_COLORS.get(str(band).lower(), ACCENT)


def apply_pyqtgraph_defaults():
    """Global pyqtgraph look — call once before building plot widgets."""
    import pyqtgraph as pg

    pg.setConfigOptions(
        antialias=True,
        background=PLOT_BG,
        foreground=TEXT_DIM,
        imageAxisOrder="row-major",
    )


def style_plot(plot_item, *, x_label=None, y_label=None, title=None):
    """Apply the house style to a pyqtgraph PlotItem."""
    plot_item.showGrid(x=True, y=True, alpha=0.12)
    plot_item.setMenuEnabled(False)
    plot_item.hideButtons()
    if title:
        plot_item.setTitle(title, color=TEXT_DIM, size="9pt")
    for side in ("left", "bottom"):
        axis = plot_item.getAxis(side)
        axis.setPen(BORDER)
        axis.setTextPen(TEXT_FAINT)
    if x_label:
        plot_item.setLabel("bottom", x_label, color=TEXT_FAINT, size="8pt")
    if y_label:
        plot_item.setLabel("left", y_label, color=TEXT_FAINT, size="8pt")


# --- application stylesheet ------------------------------------------------
STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-size: 12px;
}}
QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
    color: {TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {TEXT_DIM};
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QPushButton {{
    background-color: {SURFACE_HI};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}}
QPushButton:hover {{ background-color: #232B34; border-color: {ACCENT_DIM}; }}
QPushButton:pressed {{ background-color: #10151A; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER}; background-color: #14181E; }}
QPushButton[accent="true"] {{
    background-color: {ACCENT_DIM};
    border-color: {ACCENT};
    color: #EAF7FF;
}}
QPushButton[accent="true"]:hover {{ background-color: #2A87AF; }}
QPushButton[danger="true"] {{
    background-color: #6E2530;
    border-color: {DANGER};
    color: #FFE9E9;
}}
QComboBox, QSpinBox, QLineEdit {{
    background-color: {SURFACE_HI};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {ACCENT_DIM};
}}
QComboBox:hover, QSpinBox:hover {{ border-color: {ACCENT_DIM}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE_HI};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    selection-background-color: {ACCENT_DIM};
    outline: none;
}}
QLabel {{ background: transparent; }}
QCheckBox, QRadioButton {{ color: {TEXT}; spacing: 6px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {BORDER_HI};
    background-color: {SURFACE_HI};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 7px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QSlider::groove:horizontal {{
    height: 4px; background: {BORDER}; border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT_DIM}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 10px; height: 10px;
    margin: -4px 0;
    border-radius: 5px;
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background: {SURFACE};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 7px 22px;
    margin-right: 2px;
    min-width: 82px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
QTableWidget {{
    background-color: {PLOT_BG};
    alternate-background-color: {SURFACE};
    color: {TEXT};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QHeaderView::section {{
    background-color: {SURFACE_HI};
    color: {TEXT_DIM};
    padding: 5px;
    border: none;
    border-bottom: 1px solid {BORDER};
    font-weight: 600;
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER_HI}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {BORDER_HI}; border-radius: 5px; min-width: 30px; }}
QProgressBar {{
    background: {SURFACE_HI};
    border: 1px solid {BORDER};
    border-radius: 4px;
    height: 6px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}
QToolTip {{
    background-color: {SURFACE_HI};
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    padding: 5px;
}}
"""
