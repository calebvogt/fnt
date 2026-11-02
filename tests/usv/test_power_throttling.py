"""MAD opts out of Windows' background power throttling.

Windows 11 throttles windowless processes onto efficiency cores. A headless
`mad analyze` took 47 s per recording instead of 27, and a 378-file project
re-detection 6.0 h instead of 3.3 h; at that rate the full anosmia data set
(37,028 recordings) is ~20 days instead of ~12.

Runs under pytest, or directly.
"""
import sys

import pytest

from fnt.usv.usv_detector import power


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only API")
def test_the_opt_out_takes_on_windows():
    assert power.keep_full_speed() is True


def test_elsewhere_it_is_a_harmless_no_op(monkeypatch):
    monkeypatch.setattr(power.sys, "platform", "linux")
    assert power.keep_full_speed() is False


def test_a_failure_never_raises(monkeypatch):
    import ctypes
    monkeypatch.setattr(power.sys, "platform", "win32")

    def boom(*a, **k):
        raise OSError("no kernel32 here")
    monkeypatch.setattr(ctypes, "WinDLL", boom, raising=False)
    assert power.keep_full_speed() is False


def test_every_cli_command_opts_out_before_it_runs(monkeypatch):
    import argparse
    from fnt.usv.usv_detector import cli
    order = []
    monkeypatch.setattr(power, "keep_full_speed",
                        lambda: order.append("opt-out") or True)
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("work").set_defaults(
        func=lambda a: order.append("work") or 0)
    monkeypatch.setattr(cli, "build_parser", lambda: p)
    assert cli.main(["work"]) == 0
    assert order == ["opt-out", "work"]


def test_opening_mad_opts_out(monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    import fnt.usv.mad_pyqt as M
    app = QApplication.instance() or QApplication([])  # noqa: F841
    calls = []
    monkeypatch.setattr(power, "keep_full_speed", lambda: calls.append(1))
    w = M.MADMainWindow()
    try:
        assert calls == [1]
    finally:
        w.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
