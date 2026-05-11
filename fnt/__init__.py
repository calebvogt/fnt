"""fnt package

Keep package init minimal to avoid importing submodules at import-time. Some
submodules create threads or Qt objects during import which can cause
warnings or crashes (e.g., QSocketNotifier errors) when the application
imports the package before a QApplication/QCoreApplication is created.

Import submodules explicitly where needed (e.g., `from fnt import gui_pyqt`) or
use importlib.import_module.
"""

__all__ = []
