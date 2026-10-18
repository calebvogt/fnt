"""MAD's review view draws each detection's brightness ridge through its mask.

The ridge (mad_ridge, stored in /pred_ridge) is read with the prediction crops
when a recording opens, converted to spectrogram bins, and drawn as a line in
every view mode. The "Ridges" toggle above the spectrogram hides it, and a mask
replaced by an edit hides its ridge, which no longer fits it.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from scipy.io import wavfile

import fnt.usv.mad_pyqt as M
import fnt.usv.usv_detector.fnt_mask_store as MS
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR = 250_000
T_OFF, F_OFF = 1000, 300
_APP = None
_WINDOWS = []          # keep every window alive for the run (see test_csv_independence)


def _trill(n=120, f_mid=40, amp=24, period=24):
    """(level, mask, rows): a bright sinusoidal line inside a blob mask."""
    h = f_mid + amp + 12
    level = np.full((h, n), 40, np.uint8)
    rows = np.round(f_mid + amp * np.sin(2 * np.pi * np.arange(n) / period)).astype(int)
    for t, r in enumerate(rows):
        level[r - 1:r + 2, t] = (180, 250, 180)
    mask = np.zeros((h, n), np.uint8)
    mask[f_mid - amp - 3:f_mid + amp + 4, :] = 1
    return level, mask, rows


def _window():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    root = tempfile.mkdtemp(prefix="mad_ridge_view_")
    wav = os.path.join(root, "rec.wav")
    wavfile.write(wav, SR, (np.random.default_rng(0).normal(0, .05, SR * 2)
                            * 32767).astype(np.int16))
    proj = os.path.join(root, "p")
    create_mad_project(proj)
    M.MADMainWindow._apply_dark_theme()
    w = M.MADMainWindow()
    w._activate_project(MADProjectConfig.load(proj))
    w._register_audio_files([wav])
    w._append_audio_paths([wav])
    w._wait_for_audio_load()
    sp = w._spec_params()
    h5 = MS.masks_sibling_path(wav)
    MS.set_grid_attrs(h5, sample_rate=SR, nperseg=sp['nperseg'],
                      noverlap=sp['noverlap'], nfft=sp['nfft'],
                      n_freq_bins=w.spectrogram.n_freq_bins,
                      n_time_frames=w.spectrogram.n_time_frames)
    level, mask, rows = _trill()
    MS.write_pred_masks(h5, [{'blob_id': 1, 'mask': mask, 'spec': level,
                              'f_off': F_OFF, 't_off': T_OFF, 'score': 0.9,
                              'class': 'USV'}])
    w._loaded_wav_path = None
    w.audio_data = None
    w.current_file_idx = 0
    w._load_current_file()
    w._wait_for_audio_load()
    _WINDOWS.append(w)
    return w, rows


def _show_call(w):
    """Zoom to the call (per-call overlays only draw zoomed in) and paint."""
    sg = w.spectrogram
    w.resize(1400, 900)
    t = T_OFF * sg.hop / sg.sample_rate
    sg.view_start, sg.view_end = t - 0.01, t + 0.07
    return sg.grab().toImage()


def _ridge_pixels(img, color):
    n = 0
    for y in range(0, img.height()):
        for x in range(0, img.width()):
            c = img.pixelColor(x, y)
            if (c.red(), c.green(), c.blue()) == color:
                n += 1
    return n


def test_the_ridge_loads_with_the_detection_in_bins():
    w, rows = _window()
    preds = [a for a in w.spectrogram.annotations if a.get('status') == 'prediction']
    assert len(preds) == 1
    a = preds[0]
    assert len(a['ridge_bins']) == a['mask'].shape[1]
    # bins on the full grid: the crop's own row of the bright line, plus f_off
    assert np.max(np.abs(a['ridge_bins'] - (F_OFF + rows))) <= 1.01


def test_the_ridge_is_drawn_and_the_toggle_hides_it():
    w, _ = _window()
    col = M.overlay_palette(getattr(w.spectrogram, 'colormap_name', None))['ridge']
    for mode in ('overlay', 'spec'):
        w.spectrogram.view_mode = mode
        w.chk_ridges.setChecked(True)
        assert _ridge_pixels(_show_call(w), col) > 50, mode
        w.chk_ridges.setChecked(False)
        assert _ridge_pixels(_show_call(w), col) == 0, mode
    w.chk_ridges.setChecked(True)


def test_an_edited_mask_hides_its_ridge():
    w, _ = _window()
    sg = w.spectrogram
    a = next(a for a in sg.annotations if a.get('status') == 'prediction')
    col = M.overlay_palette(getattr(sg, 'colormap_name', None))['ridge']
    w.chk_ridges.setChecked(True)
    assert _ridge_pixels(_show_call(w), col) > 50
    a['mask'] = a['mask'].copy()            # what an edit does: a new mask
    assert _ridge_pixels(_show_call(w), col) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
