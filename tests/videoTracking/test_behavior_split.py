"""Behaviour classifier validation in the Mask Tracker.

Validation holds out whole source videos, never individual samples: the same
huddle seen from each of its two mice, or the next clip of the same session,
would otherwise sit on both sides of the split. With one video, whole clips are
held out instead. A class found in only one video stays in training. Scores are
reported per behaviour (recall, precision, F1, confusion matrix) as well as
balanced accuracy and macro F1, because plain accuracy mostly measures the
commonest behaviour.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fnt.videoTracking import behavior_classifier as bc  # noqa: E402


def dataset(n_videos=8, clips_per_video=6, mice_per_clip=2, n_classes=3):
    """Labels, videos and clips; both mice of a clip share its label."""
    labels, videos, clips = [], [], []
    for v in range(n_videos):
        for c in range(clips_per_video):
            lbl = (v + c) % n_classes
            for _ in range(mice_per_clip):
                labels.append(lbl)
                videos.append(rf"Z:\trial{v}\CAM13_video{v}.mp4")
                clips.append(f"clip_v{v}_c{c}")
    return labels, videos, clips


def test_whole_videos_are_held_out():
    labels, videos, clips = dataset()
    tr, va, info = bc.grouped_split(labels, videos, clips, 0.25, seed=1)
    assert info["unit"] == "video"
    assert va and tr
    assert sorted(tr + va) == list(range(len(labels)))
    tv = {bc.video_key(videos[i]) for i in tr}
    vv = {bc.video_key(videos[i]) for i in va}
    assert not tv & vv
    assert vv == set(info["val_units"])
    # Every class is validated and still trained.
    for c in range(3):
        assert info["per_class"][c]["val"] > 0
        assert info["per_class"][c]["train"] > 0
    assert 0.1 <= len(va) / len(labels) <= 0.45


def test_split_is_reproducible():
    labels, videos, clips = dataset()
    a = bc.grouped_split(labels, videos, clips, 0.25, seed=3)
    b = bc.grouped_split(labels, videos, clips, 0.25, seed=3)
    assert a[0] == b[0] and a[1] == b[1]


def test_one_video_holds_out_whole_clips():
    labels, videos, clips = dataset(n_videos=1, clips_per_video=12)
    tr, va, info = bc.grouped_split(labels, videos, clips, 0.25)
    assert info["unit"] == "clip"
    assert va
    assert not {clips[i] for i in tr} & {clips[i] for i in va}


def test_class_from_one_video_stays_in_training():
    labels, videos, clips = dataset()
    # 'attack' (class 3) appears only in video 5.
    labels += [3, 3]
    videos += [r"Z:\trial5\CAM13_video5.mp4"] * 2
    clips += ["clip_v5_attack"] * 2
    tr, va, info = bc.grouped_split(labels, videos, clips, 0.25)
    assert info["train_only_classes"] == [3]
    assert "cam13_video5.mp4" not in info["val_units"]
    assert info["per_class"][3] == {"train": 2, "val": 0}


def test_same_video_through_two_mappings_is_one_group():
    assert bc.video_key(r"Z:\Work\a\CAM13_x.mp4") == bc.video_key(
        r"\\nas\Vogt_28TB\Work\a\cam13_X.MP4")


def test_no_validation_when_fraction_is_zero():
    labels, videos, clips = dataset()
    tr, va, info = bc.grouped_split(labels, videos, clips, 0.0)
    assert va == [] and len(tr) == len(labels)


def test_classification_report():
    # 3 classes; class 2 never appears in the labels.
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 2]
    r = bc.classification_report(y_true, y_pred, 3)
    assert r["confusion"].tolist() == [[3, 1, 0], [0, 1, 1], [0, 0, 0]]
    assert r["accuracy"] == pytest.approx(4 / 6)
    assert r["recall"][0] == pytest.approx(0.75)
    assert r["recall"][1] == pytest.approx(0.5)
    assert r["precision"][1] == pytest.approx(0.5)
    assert np.isnan(r["recall"][2])
    # Absent classes are left out of the averages.
    assert r["balanced_accuracy"] == pytest.approx((0.75 + 0.5) / 2)
    f0 = 2 * 1.0 * 0.75 / 1.75
    assert r["macro_f1"] == pytest.approx((f0 + 0.5) / 2)


def _write_clip(clip_dir, rng, n_frames=8, h=96, w=96, two=False):
    os.makedirs(clip_dir, exist_ok=True)
    arrays = {}
    for obj in range(2 if two else 1):
        m = np.zeros((n_frames, h, w), bool)
        x0 = int(rng.integers(5, 40)) + obj * 30
        for t in range(n_frames):
            m[t, 30:45, x0 + t:x0 + t + 20] = True
        arrays[f"obj_{obj}"] = m
    np.savez_compressed(os.path.join(clip_dir, "masks.npz"), **arrays)


def test_trainer_writes_split_and_per_class_scores(monkeypatch):
    """The training worker end to end on tiny synthetic clips.

    A small random-weight backbone stands in for the ImageNet one so the test
    neither downloads weights nor needs a GPU.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torch import nn

    from fnt.videoTracking.mask_tracker_gui import _ClassifierTrainWorker

    def tiny_backbone(name, in_channels, freeze):
        net = nn.Sequential(nn.Conv2d(in_channels, 4, 3, stride=4),
                            nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        return net, 4

    monkeypatch.setattr(bc, "_build_backbone", tiny_backbone)

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        samples, labels = [], []
        for v in range(4):
            for c in range(4):
                clip_dir = os.path.join(tmp, "clips", f"v{v}_c{c}")
                lbl = c % 2
                _write_clip(clip_dir, rng, two=bool(lbl))
                for obj in range(2 if lbl else 1):
                    samples.append({"clip_dir": clip_dir, "obj_id": obj,
                                    "fps": 15.0, "video": f"D:/vids/video{v}.mp4"})
                    labels.append(lbl)
        out = os.path.join(tmp, "model")
        cfg = {
            "samples": samples, "labels": labels,
            "class_names": ["alone", "huddle"],
            "epochs": 2, "lr": 1e-3, "batch_size": 4, "val_split": 0.25,
            "augment": False, "aug_rotation": 0, "freeze_backbone": False,
            "early_stopping": False, "patience": 2,
            "backbone": "ResNet-18", "output_dir": out,
        }
        w = _ClassifierTrainWorker(cfg)
        logs, errors = [], []
        w.log_message.connect(logs.append)
        w.error.connect(errors.append)
        w.run()
        assert not errors, errors

        split = json.load(open(os.path.join(out, "split.json")))
        assert split["unit"] == "video"
        assert split["val_units"]
        assert set(split["per_class"]) == {"alone", "huddle"}

        summary = json.load(open(os.path.join(out, "training_summary.json")))
        assert summary["val_held_out_by"] == "video"
        assert set(summary["val_per_class"]) == {"alone", "huddle"}
        assert summary["best_val_macro_f1"] is not None

        rows = open(os.path.join(out, "val_confusion.csv")).read().splitlines()
        assert rows[0].split(",")[1:] == ["alone", "huddle"]
        log = open(os.path.join(out, "training_log.csv")).read().splitlines()
        assert log[0].endswith("val_balanced_acc,val_macro_f1")
        assert log[1].split(",")[3] != ""  # val_acc is filled in now
        assert any("held out whole" in m for m in logs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
