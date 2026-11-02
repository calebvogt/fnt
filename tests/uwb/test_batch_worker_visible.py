"""A UWB batch worker can run on screen, titled with its trial and pass.

Runs launched from a script ("headless") still have to show their progress, so
a job may ask for the worker's window to be shown. Without the flag the worker
stays offscreen, as the GUI's own queue needs.

Runs under pytest, or directly.
"""
import json
import os
import subprocess
import sys

import pytest

from fnt.uwb.uwb_batch_worker import window_title


def test_title_names_the_trial_and_pass():
    base = {"path": "Z:/x/T007_VoleCosmData.sql", "config": {"save_animation": False}}
    assert window_title(base) == "UWB export: T007_VoleCosmData (data pass)"
    assert window_title(dict(base, reuse_smoothed=True)) == \
        "UWB export: T007_VoleCosmData (plots pass)"
    anim = dict(base, reuse_smoothed=True, config={"save_animation": True})
    assert window_title(anim) == "UWB export: T007_VoleCosmData (animation pass)"


@pytest.mark.parametrize("show", [False])
def test_a_missing_database_still_exits_1(tmp_path, show):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"path": str(tmp_path / "missing.sqlite"),
                               "table": "T", "config": {}, "show_window": show}))
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run([sys.executable, "-m", "fnt.uwb.uwb_batch_worker", str(job)],
                       capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 1, r.stdout[-2000:] + r.stderr[-2000:]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
