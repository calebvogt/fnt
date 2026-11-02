"""YOLO training data paths and the Windows 260-character limit.

Ultralytics opens the training copy by its resolved path, and a mapped drive
resolves to its network path (Z: -> \\\\server\\share). A run folder on Z: put
165 of 177 resource-zone label files past 260 characters; Windows could not
open them and Ultralytics trained those frames as empty. The copy now goes in
a short temp folder, and a path still too long stops the run.

Runs under pytest, or directly.
"""
import os

import pytest

from fnt.videoTracking.yolo_training import _paths_past_max_path


def test_reports_paths_at_or_past_the_limit(tmp_path):
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5")
    if os.name != "nt":
        assert _paths_past_max_path(str(tmp_path), limit=1) == []
        return
    assert _paths_past_max_path(str(tmp_path)) == []
    long = _paths_past_max_path(str(tmp_path), limit=len(str(tmp_path.resolve())) + 5)
    assert [os.path.basename(p) for p in long] == ["a.txt"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
