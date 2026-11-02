"""A project lives where its file is, not where it was created.

``mad_project_info.json`` records the ``project_dir`` it was created in.
``load`` used to keep that value, so a project opened from anywhere else -- a
copy on another drive, the same share mounted at /Volumes on the Mac -- sent
every save, its training_data/ and its models/ back to the original path.
"""
import json
import os
import shutil

from fnt.usv.usv_detector.mad_project import (
    PROJECT_INFO_FILENAME, MADProjectConfig, create_mad_project,
)


def _make(tmp_path, name="orig"):
    d = tmp_path / name
    create_mad_project(str(d))
    return d


def test_a_copied_project_saves_into_the_copy_not_the_original(tmp_path):
    orig = _make(tmp_path)
    copy = tmp_path / "copy"
    shutil.copytree(orig, copy)
    before = (orig / PROJECT_INFO_FILENAME).read_bytes()

    cfg = MADProjectConfig.load(str(copy))
    cfg.tags = ["edited in the copy"]
    cfg.save()

    assert (orig / PROJECT_INFO_FILENAME).read_bytes() == before, \
        "saving the copy rewrote the original"
    saved = json.loads((copy / PROJECT_INFO_FILENAME).read_text())
    assert saved["tags"] == ["edited in the copy"]


def test_derived_folders_follow_the_file(tmp_path):
    orig = _make(tmp_path)
    moved = tmp_path / "moved"
    shutil.move(str(orig), str(moved))

    cfg = MADProjectConfig.load(str(moved / PROJECT_INFO_FILENAME))

    assert os.path.normcase(cfg.project_dir) == os.path.normcase(str(moved))
    assert os.path.normcase(cfg.training_data_dir).startswith(
        os.path.normcase(str(moved)))


def test_a_stale_stored_path_that_does_not_exist_is_ignored(tmp_path):
    """The Mac case: the stored path is a Windows drive that is not there."""
    d = _make(tmp_path)
    info = d / PROJECT_INFO_FILENAME
    data = json.loads(info.read_text())
    data["project_dir"] = "Z:/Work/Projects/FNT_MAD_models\\somewhere_else"
    info.write_text(json.dumps(data))

    cfg = MADProjectConfig.load(str(d))
    cfg.save()

    assert os.path.normcase(cfg.project_dir) == os.path.normcase(str(d))
    assert os.path.normcase(json.loads(info.read_text())["project_dir"]) \
        == os.path.normcase(str(d))


def test_loading_by_folder_or_by_file_agrees(tmp_path):
    d = _make(tmp_path)
    a = MADProjectConfig.load(str(d))
    b = MADProjectConfig.load(str(d / PROJECT_INFO_FILENAME))
    assert a.project_dir == b.project_dir
