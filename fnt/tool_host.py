"""Standalone host process for a single FNT tool window.

FNT launches every tool window as its own OS process. This module is the
entry point those child processes run:

    python -m fnt.tool_host <module> <attr> [--title "Window title"]

``<attr>`` names either a QWidget/QMainWindow subclass or a zero-argument
factory returning one. Hosting the window here instead of calling each tool's
own ``main()`` means tools need no standalone entry point of their own, and
every tool gets identical crash logging and startup behaviour for free.

WHY SEPARATE PROCESSES
----------------------
FNT's tools lean on native C extensions — SQLite (frequently over SMB network
shares), numpy/MKL, matplotlib, OpenCV, FFmpeg. A fault in any of them raises
no Python exception: it terminates the interpreter outright. When every tool
shared one process, a single native access violation destroyed all open tools
and any long-running job with them. A recorded instance:

    Windows fatal exception: access violation
      thread A: uwb_preprocessing_pyqt.py  _query    (SQLite read)
      thread B: videoProcessing.py         _reader   (FFmpeg output)

The UWB database read faulted and took an unrelated, in-progress video
preprocessing job down with it. No ``try``/``except`` can prevent that — only
an address-space boundary can. One process per tool makes a crash cost exactly
one window, and as a bonus gives each tool its own GIL and Qt event loop, so a
heavy export no longer stalls another tool's UI.
"""

import argparse
import faulthandler
import os
import sys
from datetime import datetime

# Kept alive for the life of the process so faulthandler's file target is never
# garbage collected out from under the signal handler.
_FAULT_LOG = None


def crash_log_path():
    """Path of the shared native-crash log (always on local disk).

    Deliberately not on a network share: the share hiccuping is one of the
    things that causes these crashes, so the log must not depend on it.
    """
    return os.path.join(os.path.expanduser("~"), ".fnt", "faulthandler_crash.log")


def install_faulthandler(label=""):
    """Write a native-crash traceback for every thread to the crash log.

    A hard crash inside a C extension produces no Python traceback. faulthandler
    installs a fatal-signal handler that dumps each thread's Python stack on the
    way down, which is what makes these crashes diagnosable at all.
    """
    global _FAULT_LOG
    try:
        path = crash_log_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _FAULT_LOG = open(path, "a", buffering=1, encoding="utf-8")
        tag = f" [{label}]" if label else ""
        _FAULT_LOG.write(
            f"\n===== FNT tool{tag} pid={os.getpid()} "
            f"{datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        _FAULT_LOG.flush()
        faulthandler.enable(file=_FAULT_LOG, all_threads=True)
    except Exception:
        # Never let logging setup block a tool from starting.
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            pass


def resource_path(relative_path):
    """Locate a bundled resource, working from source and from PyInstaller."""
    try:
        base_path = sys._MEIPASS          # PyInstaller extraction dir
    except Exception:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def fnt_icon_path():
    """Path to the FNT window icon, or None if it isn't present."""
    path = resource_path(os.path.join("icons", "fnt_icon.ico"))
    return path if os.path.exists(path) else None


def _prepare_windows_taskbar():
    """Give tool windows the FNT identity in the Windows taskbar.

    Without an explicit AppUserModelID, Windows attributes the window to the
    interpreter that created it and shows pythonw.exe's generic icon. Setting
    the SAME id the launcher uses also makes Windows group every FNT window --
    launcher and tools alike -- under one taskbar button.

    Must run before QApplication is constructed.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "fnt.fieldneurotoolbox.gui.01")
    except Exception:
        pass


def _enable_hidpi():
    """Match the launcher's HiDPI setup so tools render identically.

    Tool windows used to be created inside the launcher process and inherited
    these; now that each runs standalone it has to opt in itself, or layouts
    clip at 125%/150% Windows display scaling.

    Must run before QApplication is constructed.
    """
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication

        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
            try:
                QApplication.setHighDpiScaleFactorRoundingPolicy(
                    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
            except Exception:
                pass
    except Exception:
        pass


def _suppress_child_consoles():
    """Stop console programs launched by a tool from popping up cmd windows.

    Tool hosts run under pythonw.exe, which has no console of its own. On
    Windows a console program (ffmpeg, ffprobe, sleap-track...) started from a
    process with no console gets a brand-new console window allocated for it --
    so a batch of 1300 videos would flash up 1300 empty black windows.

    The per-call fix is creationflags=CREATE_NO_WINDOW, but the tools make
    subprocess calls from dozens of places across many modules and any missed
    call site reintroduces the flicker. Since a hosted GUI tool should never
    want a console window, default the flag here instead: subprocess.run,
    call and check_output all construct a Popen, so patching Popen covers
    every entry point, including ones added later.

    An explicit creationflags= or startupinfo= from the caller is left alone.
    """
    if os.name != "nt":
        return

    import subprocess

    CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    original_init = subprocess.Popen.__init__

    # Popen's positional signature: (args, bufsize, executable, stdin, stdout,
    # stderr, preexec_fn, close_fds, shell, cwd, env, universal_newlines,
    # startupinfo, creationflags, ...) -- startupinfo/creationflags are #12/#13
    # after self, so only override when the caller passed neither.
    def patched_init(self, *args, **kwargs):
        caller_set_it = (
            len(args) >= 13                     # startupinfo/creationflags positional
            or kwargs.get("startupinfo")
            or kwargs.get("creationflags")
        )
        if not caller_set_it:
            kwargs["creationflags"] = CREATE_NO_WINDOW
        return original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = patched_init


def load_window(module_name, attr_name):
    """Import ``module_name`` and instantiate ``attr_name`` from it."""
    import importlib

    module = importlib.import_module(module_name)
    try:
        target = getattr(module, attr_name)
    except AttributeError:
        raise SystemExit(
            f"{module_name} has no attribute {attr_name!r}")
    return target()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fnt.tool_host")
    parser.add_argument("module", help="module holding the window class")
    parser.add_argument("attr", help="window class or zero-arg factory")
    parser.add_argument("--title", default=None, help="window title override")
    parser.add_argument("--post-show", default=None, dest="post_show",
                        help="zero-arg window method to call after show()")
    args = parser.parse_args(argv)

    install_faulthandler(args.title or f"{args.module}.{args.attr}")

    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QApplication, QMessageBox

    # Both must happen before QApplication is constructed.
    _prepare_windows_taskbar()
    _enable_hidpi()
    # Must be in place before the tool imports/runs anything that shells out.
    _suppress_child_consoles()

    app = QApplication(sys.argv)
    # Closing the tool's last window should end the process.
    app.setQuitOnLastWindowClosed(True)

    # Show the FNT icon rather than the bare interpreter's.
    app.setApplicationName("FieldNeuroToolbox")
    icon = fnt_icon_path()
    if icon:
        app.setWindowIcon(QIcon(icon))

    try:
        window = load_window(args.module, args.attr)
    except BaseException as e:
        # Surface import/construction failures to the user rather than dying
        # silently — from the parent's side a silent exit is indistinguishable
        # from a crash.
        QMessageBox.critical(
            None, "Tool failed to start",
            f"{args.title or args.attr} could not be opened.\n\n{type(e).__name__}: {e}")
        return 1

    if args.title:
        try:
            window.setWindowTitle(args.title)
        except Exception:
            pass

    window.show()
    try:
        window.raise_()
        window.activateWindow()
    except Exception:
        pass

    # Some tools open a modal chooser right after showing (e.g. ABMA's
    # New/Open project dialog).
    if args.post_show:
        try:
            getattr(window, args.post_show)()
        except Exception as e:
            QMessageBox.warning(
                None, "Tool startup",
                f"{args.title or args.attr} opened, but "
                f"{args.post_show}() failed:\n\n{type(e).__name__}: {e}")

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
