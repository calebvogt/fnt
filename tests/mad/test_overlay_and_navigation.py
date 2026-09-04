"""Overlay colours mean one thing, and the wheel gestures stay distinct.

Two things worth pinning down, because both have moved before:

* **Colour is semantic.** Confirmed is green, pending yellow, rejected red,
  selected white -- and it stays that way whichever Color Map is loaded. An
  earlier revision derived the palette from the colormap to maximise contrast;
  the cost was that a screenshot could not be read without knowing which map
  produced it, which is worse than the contrast was good. Legibility is the
  halo's job now, not the hue's.
* **The overview strip has to survive a thousand calls.** One full-height line
  per detection puts a line in every pixel column of a ~1000 px strip and
  overdraws into a solid block. Ticks collapse per column into a short lane.

Runs under pytest, or directly, since the lab environments do not all carry
pytest.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QColor, QWheelEvent
from PyQt5.QtWidgets import QApplication

import fnt.usv.mad_pyqt as M
from fnt.usv.audio_widgets import WaveformOverviewWidget

DARK_MAPS = ('viridis', 'magma', 'inferno', 'grayscale')
GREEN, YELLOW, RED, WHITE = ((55, 230, 95), (255, 215, 40),
                             (255, 60, 60), (255, 255, 255))
_APP = None


def app():
    """A QApplication that outlives the test -- if it is collected mid-test the
    process dies with no Python traceback to explain it."""
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


# ----------------------------------------------------------------------
# Colour means one thing
# ----------------------------------------------------------------------
def test_every_dark_colormap_uses_the_same_semantic_colours():
    for cmap in DARK_MAPS:
        pal = M._OVERLAY_PALETTES[cmap]
        assert pal['confirmed'] == GREEN, cmap
        assert pal['pending'] == YELLOW, cmap
        assert pal['rejected'] == RED, cmap
        assert pal['selected'] == WHITE, cmap


def test_the_dark_maps_share_one_palette_object():
    """Not just equal -- the same dict, so they cannot drift apart."""
    assert len({id(M._OVERLAY_PALETTES[c]) for c in DARK_MAPS}) == 1


def test_the_light_background_keeps_the_hues_but_flips_the_contrast():
    """Inverted grayscale is the one light background: white and bright yellow
    would vanish on it, so the meanings darken and the halo goes white."""
    inv = M._OVERLAY_PALETTES['grayscale_inv']
    r, g, b = inv['confirmed']
    assert g > r and g > b, "confirmed must still read as green"
    assert inv['rejected'][0] > max(inv['rejected'][1:]), "rejected still red"
    assert inv['halo'] == WHITE
    assert inv['selected'] != WHITE, "a white selection is invisible here"


def test_an_unknown_colormap_still_gets_a_palette():
    assert M.overlay_palette('some-map-we-never-shipped') is M._DEFAULT_PALETTE
    assert M.overlay_palette(None) is M._DEFAULT_PALETTE


# ----------------------------------------------------------------------
# The overview strip under a realistic call count
# ----------------------------------------------------------------------
def _overview(n_marks=3000, width=400):
    app()
    ov = WaveformOverviewWidget()
    ov.resize(width, 40)
    ov.set_audio_data(
        np.random.default_rng(0).normal(0, .1, 40_000).astype(np.float32), 4000)
    pend, conf = (QColor(*YELLOW), 2), (QColor(*GREEN), 0)
    ov.set_status_marks([(i / 300.0, *(pend if i % 97 == 0 else conf))
                         for i in range(n_marks)])
    return ov


def test_ticks_never_paint_over_the_waveform():
    """The strip's job is showing the waveform; ticks live in a lane above it."""
    ov = _overview()
    img = ov.grab().toImage()
    tick_colours = {GREEN, YELLOW, RED}
    for x in range(0, ov.width(), 3):
        c = QColor(img.pixel(x, int(ov.height() * 0.6)))
        assert (c.red(), c.green(), c.blue()) not in tick_colours, f"x={x}"


def test_the_lane_still_shows_the_ticks():
    ov = _overview()
    img = ov.grab().toImage()
    assert len({QColor(img.pixel(x, 2)).name()
                for x in range(ov.width())}) > 1


def test_legacy_two_tuples_are_still_accepted():
    """Priority is optional -- older callers pass (seconds, colour)."""
    app()
    ov = WaveformOverviewWidget()
    ov.resize(400, 40)
    ov.set_audio_data(np.zeros(4000, dtype=np.float32), 4000)
    ov.set_status_marks([(0.5, QColor(*RED)), (0.6, QColor(*GREEN))])
    assert ov.grab().toImage().width() > 0


# ----------------------------------------------------------------------
# Wheel gestures
# ----------------------------------------------------------------------
def _spectrogram():
    app()
    sg = M.MADSpectrogramWidget()
    sg.resize(600, 400)
    sg.total_duration = 10.0
    sg.view_start, sg.view_end = 2.0, 4.0
    pans, zooms = [], []
    sg.pan_requested.connect(pans.append)
    sg.zoom_requested.connect(lambda f, c: zooms.append(f))
    return sg, pans, zooms


def _wheel(sg, dy, mods=Qt.NoModifier):
    rect = sg._get_spec_rect()
    pos = QPointF(rect.center().x(), rect.center().y())
    sg.wheelEvent(QWheelEvent(pos, pos, QPoint(0, 0), QPoint(0, dy),
                              Qt.NoButton, mods, Qt.NoScrollPhase, False))


def test_plain_wheel_zooms():
    sg, pans, zooms = _spectrogram()
    _wheel(sg, 120)
    assert len(zooms) == 1 and pans == []


def test_shift_wheel_pans_and_does_not_zoom():
    sg, pans, zooms = _spectrogram()
    _wheel(sg, 120, Qt.ShiftModifier)
    assert zooms == [], "shift+wheel must not zoom"
    assert pans == [-0.25], "wheel up scrolls left"
    _wheel(sg, -120, Qt.ShiftModifier)
    assert pans == [-0.25, 0.25]


def test_paint_mode_still_owns_the_wheel():
    """Brush radius beats both gestures -- it is tuned mid-stroke."""
    sg, pans, zooms = _spectrogram()
    sg.paint_mode = 'brush'
    before = sg.brush_radius_px
    _wheel(sg, 120, Qt.ShiftModifier)
    assert sg.brush_radius_px != before
    assert pans == [] and zooms == []


if __name__ == "__main__":
    import sys
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("  OK   " + name, flush=True)
        except Exception:
            fails += 1
            print("  FAIL " + name, flush=True)
            traceback.print_exc()
    print("")
    print("ALL OK" if not fails else str(fails) + " FAILURE(S)", flush=True)
    sys.stdout.flush()
    os._exit(1 if fails else 0)
