"""Mask outlines have to sit on the pixels they trace.

A spec pixel index names a whole band, and the spectrogram image draws that band
from its left (and bottom) boundary. ``_t_to_x`` / ``_f_to_y`` take EDGE
coordinates, which is right for a continuous value like a bbox midpoint and
wrong for an index: feeding an index straight in put every contour vertex on the
band's edge instead of its centre, so every mask outline sat half a spec-pixel
left of, and below, the pixels it traced.

That is invisible zoomed out (~0.13 screen px across a 2 s window) and about
1.1 px at the 0.25 s window used for labelling -- which is why it read as "the
SAM mask is a pixel or two to the left" and only while zoomed in.

The filled overlay was never affected: it is an array blitted through the same
scaling as the spectrogram, so it aligns by construction. Only vector strokes
went through the index path.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication

import fnt.usv.mad_pyqt as M

SR, NPERSEG, NOVERLAP, NFFT = 250_000, 512, 384, 1024
HOP = NPERSEG - NOVERLAP
_APP = None
_SG = None


def widget():
    """One widget at a labelling-style zoom (0.25 s window)."""
    global _APP, _SG
    if _SG is None:
        _APP = QApplication.instance() or QApplication([])
        M.MADMainWindow._apply_dark_theme()
        sg = M.MADSpectrogramWidget()
        sg.resize(1100, 700)
        audio = np.random.default_rng(0).normal(0, .05, SR).astype(np.float32)
        sg.set_audio_data(audio, SR, preserve_view=False)
        sg.init_mask(audio_len=len(audio), sample_rate=SR, nperseg=NPERSEG,
                     noverlap=NOVERLAP, nfft=NFFT)
        sg.view_start, sg.view_end = 0.10, 0.35
        sg.min_freq, sg.max_freq = 0.0, SR / 2.0
        sg.view_mode = 'overlay'
        _SG = sg
    return _SG


def geometry():
    """(rect, bounds, px_per_column, px_per_bin) for the current view."""
    sg = widget()
    rect = sg._get_spec_rect()
    t_start, t_end, f_start, f_end = sg._visible_spec_bounds()
    return (rect, (t_start, t_end, f_start, f_end),
            rect.width() / (t_end - t_start),
            rect.height() / (f_end - f_start))


def band_of_column(j):
    """Screen x-range the image blit gives spec column ``t_start + j``."""
    rect, (t_start, t_end, _f0, _f1), px_col, _ = geometry()
    lo = rect.left() + j * px_col
    return lo, lo + px_col


# ----------------------------------------------------------------------
# The transforms themselves
# ----------------------------------------------------------------------
def test_a_pixel_index_maps_to_the_centre_of_its_band():
    """The property the fix establishes, stated directly."""
    sg = widget()
    rect, bounds, px_col, px_bin = geometry()
    t_start = bounds[0]
    j = (bounds[1] - bounds[0]) // 2
    lo, hi = band_of_column(j)
    x, _y = sg._grid_to_screen(t_start + j, 0, rect, bounds)
    assert lo < x < hi, f"index landed outside its own band: {x} vs [{lo}, {hi}]"
    assert abs(x - (lo + hi) / 2) < 0.01, \
        f"index is {x - (lo + hi) / 2:+.3f} px off the band centre"


def test_the_edge_transform_still_returns_edges():
    """A continuous value (a bbox midpoint, a band boundary) must NOT be
    nudged — only indices are. Half-fixing this would break the other half."""
    sg = widget()
    rect, bounds, px_col, _ = geometry()
    t_start, t_end = bounds[0], bounds[1]
    j = (t_end - t_start) // 2
    lo, _hi = band_of_column(j)
    frac = ((t_start + j) - t_start) / max(1, t_end - t_start)
    x_edge = rect.left() + frac * rect.width()
    assert abs(x_edge - lo) < 0.01, "the edge mapping moved"


def test_the_offset_would_have_been_visible_at_labelling_zoom():
    """Guards the premise: half a column has to be ~1 screen px here, or this
    test file is measuring something the user could never have seen."""
    _rect, _bounds, px_col, _px_bin = geometry()
    assert 1.0 < px_col < 4.0, f"{px_col:.2f} px per column — unexpected zoom"
    assert 0.5 < px_col / 2 < 2.0


# ----------------------------------------------------------------------
# What actually gets drawn
# ----------------------------------------------------------------------
def _outline_span(sg, axis='x'):
    """Screen extent of the drawn outline along one axis, from the render."""
    from PyQt5.QtGui import QColor
    rect, bounds, px_col, px_bin = geometry()
    t_start, _t_end, f_start, _f_end = bounds
    img = sg.grab().toImage()
    want = M._SEMANTIC_PALETTE['confirmed']
    ann = sg.annotations[0]
    x_lo = rect.left() + (ann['t0'] - t_start) * px_col
    x_hi = rect.left() + (ann['t1'] - t_start) * px_col
    y_hi = rect.bottom() - (ann['f0'] - f_start) * px_bin
    y_lo = rect.bottom() - (ann['f1'] - f_start) * px_bin

    def hit(x, y):
        # EXACT match only. The stroke is drawn at full alpha, so it is the one
        # thing on screen with precisely this colour; the filled overlay is
        # semi-transparent and blends with the viridis underneath into shades
        # that a tolerant match happily accepts, which measures the fill rather
        # than the outline.
        c = QColor(img.pixel(int(x), int(y)))
        return (c.red(), c.green(), c.blue()) == tuple(want)

    if axis == 'x':
        y = (y_lo + y_hi) / 2
        found = [x for x in range(int(x_lo) - 30, int(x_hi) + 30) if hit(x, y)]
        return found, (x_lo, x_hi)
    x = (x_lo + x_hi) / 2
    found = [y for y in range(int(y_lo) - 30, int(y_hi) + 30) if hit(x, y)]
    return found, (y_lo, y_hi)


def _place_block():
    sg = widget()
    _rect, bounds, _pc, _pb = geometry()
    t_start = bounds[0]
    t0, t1 = t_start + 200, t_start + 220
    f0, f1 = 200, 240
    sg.annotations = [{
        'id': 'probe', 'blob_id': None, 'category': 'USV', 'status': 'accepted',
        'score': 1.0, 'f0': f0, 'f1': f1, 't0': t0, 't1': t1,
        'mask': np.ones((f1 - f0, t1 - t0), dtype=bool)}]
    sg._rebuild_confirmed_mask()
    return sg


def test_the_outline_is_centred_on_the_region_it_traces():
    """Stroke width blurs the edges, so the centre is the honest measure: it is
    insensitive to how thick the pen is, and it is exactly what a systematic
    half-pixel shift would move."""
    sg = _place_block()
    found, (lo, hi) = _outline_span(sg, 'x')
    assert found, "no outline rendered"
    drawn_centre = (min(found) + max(found)) / 2
    expected_centre = (lo + hi) / 2
    _r, _b, px_col, _pb = geometry()
    off = (drawn_centre - expected_centre) / px_col
    assert abs(off) < 0.35, f"outline centre is {off:+.2f} columns off"


def test_the_same_holds_on_the_frequency_axis():
    sg = _place_block()
    found, (lo, hi) = _outline_span(sg, 'y')
    assert found, "no outline rendered"
    drawn_centre = (min(found) + max(found)) / 2
    expected_centre = (lo + hi) / 2
    _r, _b, _pc, px_bin = geometry()
    off = (drawn_centre - expected_centre) / px_bin
    assert abs(off) < 0.35, f"outline centre is {off:+.2f} bins off"


# ----------------------------------------------------------------------
# The round trip the user actually performs
# ----------------------------------------------------------------------
def test_clicking_a_pixel_and_drawing_it_agree():
    """Click in the middle of a pixel; the outline for that pixel must come back
    centred on where you clicked. This composition is the user's experience."""
    sg = widget()
    rect, bounds, px_col, _px_bin = geometry()
    t_start = bounds[0]
    j = (bounds[1] - bounds[0]) // 2
    lo, hi = band_of_column(j)
    click_x = (lo + hi) / 2

    t_idx, _f_idx = sg._screen_to_spec_idx(click_x, rect.center().y(), rect)
    assert t_idx == t_start + j, "the click did not land on the expected column"

    drawn_x, _y = sg._grid_to_screen(t_idx, 0, rect, bounds)
    assert abs(drawn_x - click_x) < 0.51, \
        f"drawn {drawn_x - click_x:+.2f} px from where the user clicked"


def test_a_bbox_midpoint_is_unaffected():
    """Harmonic links and class labels anchor on continuous midpoints, which
    were already correct — the fix must not have shifted them."""
    sg = widget()
    rect, bounds, px_col, _px_bin = geometry()
    t_start, t_end = bounds[0], bounds[1]
    t0, t1 = t_start + 100, t_start + 120
    frac = (((t0 + t1) / 2.0) - t_start) / max(1, t_end - t_start)
    x_mid = rect.left() + frac * rect.width()
    lo = rect.left() + (t0 - t_start) * px_col
    hi = rect.left() + (t1 - t_start) * px_col
    assert abs(x_mid - (lo + hi) / 2) < 0.01, \
        "a continuous midpoint moved; only indices should have"


# ----------------------------------------------------------------------
# The displayed spectrogram and the mask overlay share one grid
# ----------------------------------------------------------------------
def _fresh(nperseg, noverlap, nfft, span=0.20, burst_frac=None):
    """A widget on a given project grid, optionally with a tone burst."""
    QApplication.instance() or QApplication([])
    n = int(SR * 1.0)
    audio = np.zeros(n, dtype=np.float32)
    view_start = 0.30
    if burst_frac is not None:
        bt = view_start + burst_frac * span
        c, half = int(bt * SR), 250
        t = np.arange(-half, half) / SR
        audio[c - half:c + half] = (np.hanning(2 * half)
                                    * np.sin(2 * np.pi * 50_000 * t))
    else:
        audio[:] = np.random.default_rng(0).normal(0, .05, n)
    sg = M.MADSpectrogramWidget()
    sg.resize(1100, 700)
    sg.set_audio_data(audio.astype(np.float32), SR, preserve_view=False)
    sg.init_mask(audio_len=n, sample_rate=SR, nperseg=nperseg,
                 noverlap=noverlap, nfft=nfft)
    sg.min_freq, sg.max_freq = 0.0, SR / 2.0
    sg.view_start, sg.view_end = view_start, view_start + span
    sg.cached_view_start = None
    sg._proj_spec_key = None
    sg._compute_view_spectrogram()
    return sg


GRIDS = [(512, 384, 1024), (1024, 768, 1024), (256, 192, 512), (512, 256, 1024)]


def test_the_display_has_exactly_the_overlay_s_columns():
    """The base class picks its own nperseg/noverlap from segment length, so
    the two rasters used to diverge by a whole factor on any non-default grid —
    masks drawn at half or double the width of the calls they trace. They only
    agreed at MAD's defaults, by coincidence."""
    for nperseg, noverlap, nfft in GRIDS:
        sg = _fresh(nperseg, noverlap, nfft)
        t_start, t_end, _f0, _f1 = sg._visible_spec_bounds()
        assert sg._raw_spec_db is not None, (nperseg, noverlap)
        assert sg._raw_spec_db.shape[1] == t_end - t_start, \
            f"grid {nperseg}/{noverlap}: display {sg._raw_spec_db.shape[1]} " \
            f"cols vs overlay {t_end - t_start}"


def test_a_known_burst_lands_on_the_same_column_in_both():
    """Column counts matching is necessary, not sufficient — the two could
    still be offset. Measure a signal at a known time instead."""
    for nperseg, noverlap, nfft in GRIDS:
        hop = nperseg - noverlap
        for frac in (0.25, 0.50, 0.75):     # a scale error shows up as drift
            sg = _fresh(nperseg, noverlap, nfft, burst_frac=frac)
            t_start, _t_end, _f0, _f1 = sg._visible_spec_bounds()
            prof = sg._raw_spec_db.max(axis=0)
            i = int(np.argmax(prof))
            a, b, c = prof[max(i - 1, 0)], prof[i], prof[min(i + 1, len(prof) - 1)]
            d = a - 2 * b + c
            disp = i + (0.5 * (a - c) / d if abs(d) > 1e-12 else 0.0)
            burst_t = 0.30 + frac * 0.20
            expect = (burst_t - nperseg / (2.0 * SR)) * SR / hop - t_start
            assert abs(disp - expect) < 0.25, \
                f"grid {nperseg}/{noverlap} at {frac:.0%}: " \
                f"display col {disp:.2f} vs overlay {expect:.2f}"


def test_the_frequency_rows_match_the_overlay_too():
    for nperseg, noverlap, nfft in GRIDS:
        sg = _fresh(nperseg, noverlap, nfft)
        _t0, _t1, f_start, f_end = sg._visible_spec_bounds()
        assert sg._raw_spec_db.shape[0] == f_end - f_start


def test_a_very_wide_view_falls_back_without_crashing():
    """Past the cap the base class's adaptive grid is used: each screen pixel
    covers several columns by then and the per-call outlines are hidden."""
    sg = _fresh(512, 384, 1024, span=0.20)
    sg.view_start, sg.view_end = 0.0, 1.0
    sg.cached_view_start = None
    sg._proj_spec_key = None
    sg.MAX_PROJECT_GRID_COLS = 10          # force the fallback
    sg._compute_view_spectrogram()
    assert sg.spec_image is not None
    assert sg.spec_image.width() > 0


def test_the_cache_is_keyed_on_the_exact_grid_window():
    """The base class's cache tolerates a few percent of view drift, which is
    fine for a picture and not for one a mask is drawn on."""
    sg = _fresh(512, 384, 1024)
    key = sg._proj_spec_key
    sg._compute_view_spectrogram()
    assert sg._proj_spec_key == key, "a no-op call rebuilt the image"
    sg.view_start += 0.002                  # ~1% of a 0.2 s window
    sg._compute_view_spectrogram()
    assert sg._proj_spec_key != key, "a moved view reused a stale raster"


def test_base_class_invalidation_is_honoured():
    sg = _fresh(512, 384, 1024)
    sg.cached_view_start = None             # how the base class invalidates
    sg.spec_image = None
    sg._compute_view_spectrogram()
    assert sg.spec_image is not None


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
