"""The mask store (track_store) opens at any path length: VoleCosm NVR videos have
long names, and "<video>_MaskTracker/<video>_tracks.h5" in a deep folder passes
Windows' 260-character limit, where HDF5 alone cannot create the file."""
import os

import numpy as np
import pytest

from fnt.videoTracking.track_store import (TrackMaskReader, TrackMaskWriter,
                                           apply_identity_edits, long_path,
                                           store_exists)


def _deep_store(tmp_path):
    d = str(tmp_path)
    while len(d) < 200:                                    # a folder Windows can still make
        d = os.path.join(d, "VoleCosm_Camera3_{68E56750-0315}"[: 230 - len(d)])
    os.makedirs(d, exist_ok=True)
    name = "VoleCosm_Camera3_{68E56750-0315-0003-9BE9-799F88475F0B}_20260209000000_processed_tracks.h5"
    return os.path.join(d, name)


@pytest.mark.skipif(os.name != "nt", reason="Windows path limit")
def test_a_store_past_the_windows_path_limit_is_written_read_and_edited(tmp_path):
    path = _deep_store(tmp_path)
    assert len(os.path.abspath(path)) > 260
    mask = np.zeros((20, 30), bool)
    mask[5:10, 8:16] = True
    with TrackMaskWriter(path, fps=10, width=30, height=20) as w:
        for f in range(3):
            w.add_frame(f, {1: {"mask": mask, "bbox": (8, 5, 16, 10), "centroid": (12, 7.5),
                                "score": 0.9, "area": int(mask.sum())}})
    assert store_exists(path) and not os.path.exists(path)   # plain os.path can't see it
    apply_identity_edits(path, np.array([2, 2, 2]))
    with TrackMaskReader(path) as r:
        assert list(r.object_id) == [2, 2, 2]
        assert list(r.frame) == [0, 1, 2]
    os.remove(long_path(path))


def test_short_paths_are_left_alone(tmp_path):
    p = str(tmp_path / "a_tracks.h5")
    assert long_path(p) == os.path.abspath(p)
