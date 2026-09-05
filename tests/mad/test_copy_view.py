"""Copying the spectrogram view out at a chosen resolution.

Mirrors the UWB preprocessing tool's Copy View: a dpi dropdown and a button,
rendering the panel at its own geometry so raising dpi multiplies pixels rather
than changing the layout.

The trap this suite exists to guard is that the extra pixels can be empty.
paintEvent scaled spec_image to the LOGICAL widget rect and then let the
painter transform magnify that intermediate — nearest-neighbour — so a naive
export carried no more spectrogram information than the screen did, however
many pixels it had. Measurably: the old path yields exactly the same number of
distinct spectrogram columns at every zoom level, quantized to the layout.
Resampling once, straight to the output resolution, recovers what the data
actually has: nothing extra below a ~0.7 s window (the view holds fewer STFT
columns than the panel has pixels), and measured 1.4x at 1 s, 1.8x at 2 s and
2.3x at 5 s on a 1378 px panel.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget
from scipy.io import wavfile

import fnt.usv.audio_widgets as AW
import fnt.usv.mad_pyqt as M
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR = 250_000
_BARE = None
_GUI = None


def bare():
    """A SpectrogramWidget in a real layout, holding a real chirp."""
    global _BARE
    if _BARE is None:
        app = QApplication.instance() or QApplication([])
        host = QWidget()
        lay = QVBoxLayout(host)
        w = AW.SpectrogramWidget()
        lay.addWidget(w)
        host.resize(1400, 860)
        host.show()
        app.processEvents()
        t = np.arange(int(SR * 8.0)) / SR
        rng = np.random.default_rng(0)
        sig = (0.3 * np.sin(2 * np.pi * (50000 + 20000 *
                                         np.sin(2 * np.pi * 11 * t)) * t)
               + 0.05 * rng.normal(size=t.size))
        w.audio_data = sig.astype(np.float32)
        w.sample_rate = SR
        w.total_duration = 8.0
        w.min_freq, w.max_freq = 0, 125000
        _BARE = (w, host, app)
    return _BARE[0]


def at_view(seconds):
    w = bare()
    w.view_start, w.view_end = 1.0, 1.0 + seconds
    w._compute_view_spectrogram()
    return w


def gui():
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_copy_")
        p = os.path.join(root, "r0.wav")
        wavfile.write(p, SR, (np.random.default_rng(3).normal(0, .05, SR * 2)
                              * 32767).astype(np.int16))
        proj = os.path.join(root, "p")
        create_mad_project(proj)
        app = QApplication.instance() or QApplication([])
        M.MADMainWindow._apply_dark_theme()
        w = M.MADMainWindow()
        w._activate_project(MADProjectConfig.load(proj))
        w._register_audio_files([p])
        w._append_audio_paths([p])
        w._wait_for_audio_load()
        w.resize(1400, 900)
        w.show()
        app.processEvents()
        _GUI = (w, p, app, root)
    return _GUI[0], _GUI[1]


def gray(img):
    img = img.convertToFormat(QImage.Format_RGB888)
    b = img.constBits()
    b.setsize(img.byteCount())
    a = np.frombuffer(b, np.uint8).reshape(img.height(), img.bytesPerLine())
    return a[:, :img.width() * 3].reshape(
        img.height(), img.width(), 3).astype(np.float64).mean(2)


def spec_band(img, scale):
    """The pixels inside the plot area, clear of the axis margins."""
    a = gray(img)
    return a[int(20 * scale):img.height() - int(50 * scale),
             int(60 * scale):img.width() - int(20 * scale)]


def columns(img, scale):
    band = spec_band(img, scale)
    return len(set(map(tuple, band.T.round(1).tolist())))


# ----------------------------------------------------------------------
# The screen path is untouched
# ----------------------------------------------------------------------
def test_export_scale_is_one_by_default():
    """paintEvent branches on it, and this widget repaints on every brush
    stroke and box drag. It must be exactly 1.0 unless a copy is in flight."""
    assert AW.SpectrogramWidget.export_scale == 1.0
    assert at_view(1.0).export_scale == 1.0


def test_a_normal_paint_is_byte_identical_to_before_the_change():
    """At scale 1 paintEvent must take the old code path exactly."""
    w = at_view(1.0)
    a = QImage(w.width(), w.height(), QImage.Format_RGB32)
    a.fill(0)
    w.render(a)
    b = QImage(w.width(), w.height(), QImage.Format_RGB32)
    b.fill(0)
    w.render(b)
    assert a == b
    assert np.abs(gray(a) - gray(b)).max() == 0


def test_export_scale_is_reset_even_when_the_render_explodes():
    w = at_view(1.0)
    w.render = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        w.render_view_image(dpi=300)
    except RuntimeError:
        pass
    finally:
        del w.render
    assert w.export_scale == 1.0


def test_the_live_widget_is_never_resized():
    """A copy must not touch the layout you are looking at. render() draws the
    widget at its own geometry through a scaled painter, so there is no window
    in which a repaint could catch the panel at the wrong size."""
    w = at_view(1.0)
    before = (w.size(), w.minimumSize(), w.maximumSize())
    seen = []
    w.resizeEvent = lambda e: seen.append(e)
    try:
        for dpi in (150, 300, 600):
            w.render_view_image(dpi=dpi)
            QApplication.processEvents()
    finally:
        del w.resizeEvent
    assert (w.size(), w.minimumSize(), w.maximumSize()) == before
    assert seen == [], f"{len(seen)} resize(s) fired during a copy"


# ----------------------------------------------------------------------
# The raster fix: one resample, at output resolution
# ----------------------------------------------------------------------
def _old_way(w, scale):
    """What a naive implementation gives: paintEvent scales spec_image to the
    logical rect, and the painter transform magnifies that intermediate."""
    img = QImage(int(w.width() * scale), int(w.height() * scale),
                 QImage.Format_RGB32)
    img.setDevicePixelRatio(scale)
    img.fill(QColor(30, 30, 30))
    w.render(img)                                # export_scale stays 1.0
    return img


def test_the_naive_path_is_quantized_to_the_layout():
    """Proof the bug was real: however many pixels it has, the old path can
    only ever carry as many distinct columns as the panel had."""
    counts = {columns(_old_way(at_view(v), 3.125), 3.125)
              for v in (0.3, 1.0, 2.0, 5.0)}
    assert len(counts) == 1, f"expected one quantized count, got {counts}"
    assert counts.pop() <= at_view(1.0).width()


def test_the_copy_carries_full_output_resolution():
    w = at_view(2.0)
    img, _ = w.render_view_image(dpi=300)
    assert columns(img, 3.125) > 2 * w.width()


def test_real_detail_is_recovered_only_where_it_exists():
    """Below the ~0.7 s crossover the view holds fewer STFT columns than the
    panel has pixels, so there is nothing to recover and the copy says so by
    being no sharper. Above it, there is."""
    def gain(view):
        w = at_view(view)
        new, _ = w.render_view_image(dpi=300)
        old = _old_way(w, 3.125)
        g = lambda im: np.abs(np.diff(spec_band(im, 3.125), axis=1)).mean()
        return g(new) / max(g(old), 1e-9)

    assert gain(0.3) < 1.2, "claimed detail where the data has none"
    assert gain(2.0) > 1.6, "failed to recover detail the screen throws away"
    assert gain(5.0) > 2.0
    assert gain(5.0) > gain(2.0) > gain(1.0) > gain(0.3)


# ----------------------------------------------------------------------
# Geometry and tagging
# ----------------------------------------------------------------------
def test_dpi_multiplies_pixels_and_leaves_the_physical_size_alone():
    """The UWB contract: the picture stays the size of the panel, and dpi
    makes it finer rather than bigger."""
    w = at_view(1.0)
    sizes = {}
    for dpi in (96, 150, 300, 600):
        img, eff = w.render_view_image(dpi=dpi, max_megapixels=1e9)
        assert abs(eff - dpi) < 1, (dpi, eff)
        assert (img.width(), img.height()) == (
            round(w.width() * dpi / 96), round(w.height() * dpi / 96))
        sizes[dpi] = img.width() / eff
    assert max(sizes.values()) - min(sizes.values()) < 0.01, sizes
    assert abs(sizes[300] - w.width() / 96.0) < 0.02


def test_the_image_carries_its_dpi():
    """Without this the clipboard DIB says 96 and PowerPoint places a 4200 px
    copy edge-to-edge across the slide."""
    for dpi in (150, 300, 600):
        img, eff = bare().render_view_image(dpi=dpi, max_megapixels=1e9)
        assert abs(img.dotsPerMeterX() - eff / 0.0254) < 2
        assert abs(eff - dpi) < 1
        assert img.dotsPerMeterY() == img.dotsPerMeterX()


def test_the_image_is_opaque():
    """An alpha channel makes Qt write a CF_DIBV5 whose dpi field is zeroed."""
    img, _ = bare().render_view_image(dpi=300)
    assert not img.hasAlphaChannel()


def test_aspect_ratio_is_preserved():
    w = at_view(1.0)
    img, _ = w.render_view_image(dpi=300)
    assert abs(img.height() / img.width() - w.height() / w.width()) < 0.01


def test_no_spectrogram_means_no_picture():
    """Better than inventing one."""
    app = QApplication.instance() or QApplication([])
    assert AW.SpectrogramWidget().render_view_image(dpi=300) == (None, 0.0)
    assert app is not None


def test_600_dpi_is_capped_on_a_full_size_panel():
    """Not a bug — a 1378x838 panel at 600 dpi is 45 Mpx, and Qt materialises
    every clipboard format on shutdown. The dpi that comes back is the honest
    one, and the caller reports it."""
    w = at_view(1.0)
    img, eff = w.render_view_image(dpi=600)
    assert eff < 600
    assert img.width() * img.height() / 1e6 <= AW.MAX_EXPORT_MP * 1.01


def test_the_megapixel_cap_reduces_dpi_rather_than_failing():
    w = at_view(1.0)
    img, eff = w.render_view_image(dpi=1200, max_megapixels=4.0)
    assert img.width() * img.height() / 1e6 <= 4.05
    assert eff < 1200, "cap should have lowered the effective dpi"
    # still the panel's physical size, just coarser
    assert abs(img.width() / eff - w.width() / 96.0) < 0.05


# ----------------------------------------------------------------------
# The controls, in MAD
# ----------------------------------------------------------------------
def test_the_button_and_dpi_box_live_in_the_spectrogram_controls_bar():
    w, _ = gui()
    assert w._controls_bar.isAncestorOf(w.btn_copy_view)
    assert w._controls_bar.isAncestorOf(w.combo_copy_dpi)


def test_neither_control_can_steal_the_arrow_keys():
    """The blanket no-focus loop walks only self.left_column, so anything
    added to this bar has to opt out by hand or it eats pan/zoom."""
    from PyQt5.QtCore import Qt
    w, _ = gui()
    assert w.btn_copy_view.focusPolicy() == Qt.NoFocus
    assert w.combo_copy_dpi.focusPolicy() == Qt.NoFocus


def test_the_dpi_choices_match_the_uwb_tool():
    w, _ = gui()
    items = [w.combo_copy_dpi.itemText(i)
             for i in range(w.combo_copy_dpi.count())]
    assert items == ["150 dpi", "300 dpi", "600 dpi"], items
    assert w.combo_copy_dpi.currentData() == 300, "300 should be the default"


def test_copying_puts_an_image_on_the_clipboard_at_the_chosen_dpi():
    w, _ = gui()
    QApplication.clipboard().clear()
    w.combo_copy_dpi.setCurrentIndex(2)                 # 600 dpi
    w.copy_spectrogram_view()
    got = QApplication.clipboard().image()
    assert not got.isNull()
    assert got.width() == round(w.spectrogram.width() * 600 / 96), got.width()
    assert "600 dpi" in w.status_bar.currentMessage(), \
        w.status_bar.currentMessage()
    w.combo_copy_dpi.setCurrentIndex(1)


def test_a_capped_copy_says_so_instead_of_under_delivering_quietly():
    w, _ = gui()
    real = AW.MAX_EXPORT_MP
    try:
        AW.MAX_EXPORT_MP = 1.0
        M.MAX_EXPORT_MP = 1.0
        w.combo_copy_dpi.setCurrentIndex(2)             # 600 dpi
        w.copy_spectrogram_view()
        msg = w.status_bar.currentMessage()
        assert "capped" in msg and "600" in msg, msg
    finally:
        AW.MAX_EXPORT_MP = real
        M.MAX_EXPORT_MP = real
        w.combo_copy_dpi.setCurrentIndex(1)


def test_saving_writes_a_png_carrying_its_dpi():
    w, wav = gui()
    out = os.path.join(os.path.dirname(wav), "view.png")
    w.combo_copy_dpi.setCurrentIndex(1)                 # 300 dpi
    img, dpi = w._render_spectrogram_view()
    assert img.save(out, "PNG")
    back = QImage(out)
    assert not back.isNull()
    assert abs(back.dotsPerMeterX() - 300 / 0.0254) < 2, back.dotsPerMeterX()
    os.remove(out)


def test_the_save_action_is_in_the_file_menu():
    w, _ = gui()
    assert "View" in w.act_save_view.text()
    assert w.act_save_view.shortcut().toString() == "Ctrl+Shift+S"


def test_copying_does_not_leave_playback_running_or_a_brush_dot():
    w, _ = gui()
    w.spectrogram._cursor_pos = object()
    before = w.spectrogram._cursor_pos
    w.copy_spectrogram_view()
    assert w.spectrogram._cursor_pos is before, "cursor state not restored"
    assert not getattr(w, 'is_playing', False)


def test_the_classic_detector_got_the_same_controls():
    """Both USV tools paint from the same widget; the copy lives there, so
    wiring the second host is a button and a combo."""
    from PyQt5.QtCore import Qt
    import fnt.usv.classic_audio_detector as C
    assert C.ClassicAudioDetectorWindow.COPY_VIEW_DPIS == (150, 300, 600)
    assert hasattr(C.ClassicAudioDetectorWindow, 'copy_spectrogram_view')
    app = QApplication.instance() or QApplication([])
    cad = C.ClassicAudioDetectorWindow()
    cad.resize(1400, 900)
    cad.show()
    app.processEvents()
    assert cad.btn_copy_view.focusPolicy() == Qt.NoFocus
    assert cad.combo_copy_dpi.focusPolicy() == Qt.NoFocus
    cad.copy_spectrogram_view()                  # nothing loaded: must not crash
    assert "Nothing to copy" in cad.status_bar.currentMessage()
    cad.close()


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
