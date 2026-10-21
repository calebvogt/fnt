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


def _strip(h5):
    """Remove detection 1's ridge and the file's ridge stamp."""
    import h5py
    with h5py.File(h5, 'a') as f:
        del f[MS.PRED_RIDGE_GROUP]
        for k in ('ridge_turns', 'ridge_rate_hz', 'ridge_depth_khz', 'ridge_trill'):
            del f[MS.PRED_GROUP]['1'].attrs[k]
        if MS.RIDGE_VERSION_ATTR in f.attrs:
            del f.attrs[MS.RIDGE_VERSION_ATTR]


def _window(crop=None, strip=False):
    """A main window with one recording and one detection: the trill, or
    ``crop`` = (level, mask). ``strip`` removes its ridge and the file's ridge
    stamp before the recording opens, as in a file written before ridges."""
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
    if crop is not None:
        level, mask = crop
    MS.write_pred_masks(h5, [{'blob_id': 1, 'mask': mask, 'spec': level,
                              'f_off': F_OFF, 't_off': T_OFF, 'score': 0.9,
                              'class': 'USV'}])
    if strip:
        _strip(h5)
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


def test_pieces_sounding_together_draw_a_line_each():
    level = np.full((60, 80), 30, np.uint8)
    mask = np.zeros((60, 80), np.uint8)
    for row in (10, 45):                      # a harmonic over its fundamental
        level[row, :] = 250
        mask[row - 2:row + 3, :] = 1
    w, _ = _window(crop=(level, mask))
    a = next(a for a in w.spectrogram.annotations if a.get('status') == 'prediction')
    assert np.shape(a['ridge_bins']) == (2, 80)
    col = M.overlay_palette(getattr(w.spectrogram, 'colormap_name', None))['ridge']
    w.chk_ridges.setChecked(True)
    assert _ridge_pixels(_show_call(w), col) > 50


def test_opening_a_recording_fills_its_missing_ridges():
    w, rows = _window(strip=True)
    a = next(a for a in w.spectrogram.annotations if a.get('status') == 'prediction')
    assert np.max(np.abs(a['ridge_bins'] - (F_OFF + rows))) <= 1.01
    h5 = MS.masks_sibling_path(w._active_wav_path())
    assert not MS.ridge_status(h5)['needed']                 # and stamped


def test_the_ridge_checkbox_is_the_projects_setting():
    w, _ = _window(strip=True)
    try:
        assert w.chk_infer_ridge.isChecked() and w._project.ridge
        w.chk_infer_ridge.setChecked(False)
        assert not MS.ridge_enabled()
        assert MADProjectConfig.load(w._project.project_dir).ridge is False
        assert w._infer_perf_kwargs()['ridge'] is False
    finally:
        w.chk_infer_ridge.setChecked(True)
    assert MS.ridge_enabled() and w._project.ridge


def test_a_run_with_nothing_to_infer_still_fills_the_ridges(monkeypatch):
    import time
    import fnt.usv.usv_detector.mad_inference as MI
    w, _ = _window()
    wav = w._active_wav_path()
    _strip(MS.masks_sibling_path(wav))            # as written before ridges
    monkeypatch.setattr(MI, "load_model", lambda *a, **k: (None, {}, "cpu"))
    lines, shown = [], []
    monkeypatch.setattr(w, "_log", lambda msg, *a, **k: lines.append(str(msg)))
    monkeypatch.setattr(w, "_show_inference_summary_dialog",
                        lambda results, cfg=None, ridge_line=None: shown.append((results, ridge_line)))
    cfg = MI.MADInferenceConfig(model_path="m.pt", **w._infer_perf_kwargs())
    w._ridge_checked = set()
    w._start_inference(cfg, [], reporter=w.infer_panel, skip_scope_prompt=True,
                       ridge_fill=[wav])
    t0 = time.time()
    while getattr(w, '_infer_worker', None) is not None and time.time() - t0 < 60:
        _APP.processEvents()
        time.sleep(0.01)
    assert not MS.ridge_status(MS.masks_sibling_path(wav))['needed']
    assert any("ridges rec.wav: 1 detection(s)" in s for s in lines)
    assert any(s.startswith("Ridges of the 1 already-analysed file(s): 1 filled")
               for s in lines)
    assert not w._is_infer_locked(wav)
    assert shown == [([], lines[-1])]              # the summary names the ridge pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
