"""A project's labelling guide (LABELING_GUIDE.md) shows on the Guide button of
MAD and Mask Tracker, opens in a window that edits it, and travels with a
portable session.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication

from fnt import labeling_guide as LG

_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def test_the_button_shows_the_guide_on_hover_and_opens_an_editor():
    _app()
    d = tempfile.mkdtemp(prefix="guide_")
    b = LG.GuideButton()
    b.set_project(d)
    assert b.text() == "Guide (none)" and b.isEnabled()
    LG.write_guide(d, "# Rules\n\n- Do not label mice inside the resource zones")
    b.refresh()
    assert b.text() == "Guide" and "inside the resource zones" in b.toolTip()
    b.open_guide()
    dlg = b._dialog
    assert "inside the resource zones" in dlg.view.toPlainText()
    dlg._toggle_edit()
    dlg.edit.setPlainText("# Rules\n\n- Label every mouse in the arena")
    dlg._save()
    assert "every mouse in the arena" in LG.read_guide(d) and "every mouse" in b.toolTip()
    dlg.close()


def test_mask_tracker_and_mad_show_their_projects_guide():
    _app()
    import fnt.videoTracking.mask_tracker_gui as G
    d = tempfile.mkdtemp(prefix="guide_mtt_")
    os.makedirs(os.path.join(d, "training_frames"))
    with open(os.path.join(d, "project_config.json"), "w") as fh:
        json.dump({"project_dir": d, "video_paths": [], "categories": ["mouse"], "source_videos": []}, fh)
    LG.write_guide(d, "- overhead: arena only")
    w = G.MaskTrackerWindow()
    w._launched = True
    w._load_project(os.path.join(d, "project_config.json"))
    assert "arena only" in w.btn_guide.toolTip()
    w.close()

    import fnt.usv.mad_pyqt as M
    from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project
    p = os.path.join(tempfile.mkdtemp(prefix="guide_mad_"), "p")
    create_mad_project(p)
    LG.write_guide(p, "- accept every piece of a split call")
    m = M.MADMainWindow()
    m._activate_project(MADProjectConfig.load(p))
    assert "split call" in m.btn_guide.toolTip()
    m.close()


def test_a_portable_session_carries_the_guide():
    from fnt.videoTracking import mtt_portable as P
    import numpy as np
    import cv2
    d = tempfile.mkdtemp(prefix="guide_pack_")
    os.makedirs(os.path.join(d, "training_frames"))
    cv2.imwrite(os.path.join(d, "training_frames", "a.png"), np.zeros((4, 4, 3), np.uint8))
    with open(os.path.join(d, "project_config.json"), "w") as fh:
        json.dump({"project_dir": d, "categories": ["mouse"]}, fh)
    LG.write_guide(d, "- the rules")
    s = P.pack_session(d, ["a.png"], tempfile.mkdtemp(), "s")
    assert LG.read_guide(s).strip() == "- the rules"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
