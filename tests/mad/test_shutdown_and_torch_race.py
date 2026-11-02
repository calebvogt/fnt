"""Two process-level faults MAD hit, each only visible from outside the process.

1. **A window open at exit aborted the process.** Every window starts a
   device probe thread that imports torch (seconds). Exiting while it ran
   destroyed a running QThread -- Qt's fatal, 0xC0000409, no message -- so every
   GUI test module passed and then exited 127. MAD now joins its workers at
   exit (bounded, never terminate()).

2. **scipy failed while torch was importing on another thread.** scipy >= 1.15
   checks each argument with ``is_torch_array``, which reads ``sys.modules``
   and finds a half-built torch: "partially initialized module 'torch' has no
   attribute 'Tensor'". Play, Het, or the first file's spectrogram in MAD's
   first seconds could fail. MAD primes scipy's per-type cache before the
   probe starts.

Both are exercised in a fresh interpreter, because both depend on what is and
is not imported yet and on what happens at exit.
"""
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("PyQt5")

ENV = dict(os.environ, QT_QPA_PLATFORM="offscreen")


def run(code, timeout=180):
    p = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                       env=ENV, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def test_exiting_with_a_window_open_is_clean():
    rc, out, err = run("""
        from PyQt5.QtWidgets import QApplication
        app = QApplication([])
        import fnt.usv.mad_pyqt as M
        w = M.MADMainWindow()          # probe thread starts, never closed
        print("built", flush=True)
    """)
    assert "built" in out
    assert rc == 0, f"exit {rc} (0xC0000409 = a running QThread destroyed)\n{err}"


def test_scipy_works_while_the_probe_imports_torch():
    """The exact interleaving: the probe thread starts importing torch, and
    scipy.signal runs on the main thread for the first time meanwhile."""
    pytest.importorskip("torch")
    rc, out, err = run("""
        import threading, numpy as np
        import fnt.usv.mad_pyqt as M
        assert "torch" not in __import__("sys").modules, "test needs no torch"
        primed = M._guard_scipy_against_partial_torch()
        import scipy.signal as s
        th = threading.Thread(target=lambda: __import__("torch"))
        th.start()
        x = np.random.default_rng(0).normal(size=200_000)
        bad = None
        while th.is_alive():
            try:
                s.resample_poly(x, 24, 125)
                s.sosfiltfilt(s.butter(8, 0.1, output="sos"), x)
                s.spectrogram(x, fs=250000, nperseg=512)
            except Exception as e:
                bad = repr(e)
                break
        th.join()
        print("primed" if primed else "fell back", "|", bad or "OK")
    """)
    assert rc == 0, err
    assert out.strip().endswith("OK"), out


def test_the_guard_is_free_when_torch_is_already_loaded():
    rc, out, _ = run("""
        import sys
        sys.modules.setdefault("torch", type(sys)("torch"))
        import fnt.usv.mad_pyqt as M
        print(M._guard_scipy_against_partial_torch())
    """)
    assert rc == 0 and out.strip() == "True"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
