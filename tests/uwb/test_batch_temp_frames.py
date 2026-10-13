"""A batch render's temp-frames folder is never animation_Tracking itself.

The render deletes its temp-frames folder when it finishes. A headless worker
job with no temp folder used to fall back to animation_Tracking, so the cleanup
would have deleted the finished videos along with the frames.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fnt.uwb.uwb_preprocessing_pyqt import UWBQuickVisualizationWindow  # noqa: E402

_APP = []


@pytest.fixture
def window():
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    w = UWBQuickVisualizationWindow()
    yield w
    w.close()


def test_fallback_is_a_subfolder_of_animation_tracking(window, tmp_path):
    window._batch_temp_frames_dir = None
    got = window._batch_anim_temp_dir(str(tmp_path), "T1")
    anim = os.path.join(str(tmp_path), "T1_FNT_analysis", "animation_Tracking")
    assert os.path.normpath(got) != os.path.normpath(anim)
    assert os.path.dirname(os.path.normpath(got)) == os.path.normpath(anim)


def test_the_runs_chosen_folder_wins(window, tmp_path):
    window._batch_temp_frames_dir = str(tmp_path / "frames" / "temp_animation_frames")
    assert window._batch_anim_temp_dir(str(tmp_path), "T1") == \
        window._batch_temp_frames_dir


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
