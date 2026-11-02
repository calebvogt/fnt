"""Scene-aware behavior classifier for the Mask Tracker Tool.

The original classifier read one animal's silhouette on a black background.
That representation cannot express huddling or attack, because the second
animal is cropped away before the network ever sees it, so a "Rodent Social"
preset trained on it was learning from segmentation accidents rather than
from interaction.

This model takes both halves of the scene-aware representation built in
:mod:`silhouette_extractor`:

* a six-channel image, the focal animal's time-coded contours in channels
  0-2 and its neighbours' in channels 3-5, sharing one canvas; and
* a short vector of pairwise quantities -- separation, closing speed,
  relative heading, contact -- that a 128-pixel image encodes badly.

An ImageNet backbone still does the visual work. Its first convolution is
widened from three input channels to six by tiling the pretrained weights and
halving them, which keeps the pretrained filters meaningful while letting the
neighbour channels contribute from the first step. The pairwise vector is
concatenated onto the pooled image features and a small head reads both.

A clip containing a single animal produces zero neighbour channels and
"nobody nearby" pairwise values, so solo recordings stay perfectly usable and
existing solo labels keep their meaning.

Heavy deps (torch, torchvision) are imported lazily so the module is safe to
import from the GUI on a machine without them.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .silhouette_extractor import PAIRWISE_FIELDS

# Written into classifier_config.json so inference knows which representation
# a checkpoint expects. Absent means a legacy three-channel silhouette model.
INPUT_KIND_SCENE = "scene6"

INPUT_SIZE = 128


def _inflate_first_conv(conv, in_channels: int):
    """Widen a pretrained 3-channel convolution to ``in_channels``.

    The pretrained kernels are tiled across the new channel groups and scaled
    down by the same factor, so the layer's output magnitude is unchanged on
    an image whose extra channels repeat the original. Starting from tiled
    pretrained filters rather than random ones matters here because the
    labelled clip sets are small.
    """
    import torch
    from torch import nn

    old_w = conv.weight.data
    out_ch, old_in, kh, kw = old_w.shape
    new = nn.Conv2d(
        in_channels, out_ch, kernel_size=(kh, kw), stride=conv.stride,
        padding=conv.padding, dilation=conv.dilation, bias=conv.bias is not None,
    )
    with torch.no_grad():
        reps = int(np.ceil(in_channels / old_in))
        tiled = old_w.repeat(1, reps, 1, 1)[:, :in_channels]
        new.weight.copy_(tiled * (old_in / float(in_channels)))
        if conv.bias is not None:
            new.bias.copy_(conv.bias.data)
    return new


def _build_backbone(name: str, in_channels: int, freeze: bool):
    """Pretrained backbone returning a pooled feature vector."""
    import torch
    from torch import nn
    from torchvision import models

    if name == "ResNet-34":
        net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        net.conv1 = _inflate_first_conv(net.conv1, in_channels)
        head_attr = "fc"
    elif name == "MobileNetV3":
        net = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        net.features[0][0] = _inflate_first_conv(net.features[0][0], in_channels)
        head_attr = "classifier"
    else:
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.conv1 = _inflate_first_conv(net.conv1, in_channels)
        head_attr = "fc"

    if freeze:
        for p in net.parameters():
            p.requires_grad = False
        # The widened first convolution is new work no matter what, so it
        # trains even when the rest of the backbone is held still.
        first = net.conv1 if head_attr == "fc" else net.features[0][0]
        for p in first.parameters():
            p.requires_grad = True

    setattr(net, head_attr, nn.Identity())
    with torch.no_grad():
        dim = int(net(torch.zeros(1, in_channels, INPUT_SIZE, INPUT_SIZE)).shape[1])
    return net, dim


def build_scene_model(
    backbone: str,
    n_classes: int,
    n_features: int = len(PAIRWISE_FIELDS),
    freeze_backbone: bool = False,
    in_channels: int = 6,
):
    """Assemble the image-plus-features classifier."""
    import torch
    from torch import nn

    net, dim = _build_backbone(backbone, in_channels, freeze_backbone)

    class SceneBehaviorNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = net
            self.head = nn.Sequential(
                nn.Linear(dim + n_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes),
            )

        def forward(self, image, features):
            pooled = self.backbone(image)
            return self.head(torch.cat([pooled, features], dim=1))

    return SceneBehaviorNet()


def feature_stats(vectors: np.ndarray) -> Tuple[List[float], List[float]]:
    """Mean and standard deviation for whitening the pairwise vector.

    Computed on the training split only and stored with the checkpoint. The
    entries span wildly different units -- body lengths, degrees, fractions --
    so feeding them raw lets whichever happens to be largest dominate the
    first layer.
    """
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        n = len(PAIRWISE_FIELDS)
        return [0.0] * n, [1.0] * n
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.tolist(), std.tolist()


def normalize_features(vec: np.ndarray, mean, std) -> np.ndarray:
    return ((np.asarray(vec, dtype=np.float32) - np.asarray(mean, dtype=np.float32))
            / np.asarray(std, dtype=np.float32)).astype(np.float32)


def video_key(path: Optional[str]) -> str:
    """Grouping key for a clip's source video.

    The file name rather than the full path, so one video reached through two
    drive mappings (Z:\\ and a UNC share) is still one group. Two different
    videos that happen to share a name merge into one group, which only makes
    the split coarser, never leakier.
    """
    return os.path.basename(str(path or "")).lower()


def grouped_split(labels, videos, clips, val_fraction: float, seed: int = 0):
    """Hold out whole videos for validation, keeping every class represented.

    Samples are one labelled animal in one clip. Splitting them at random puts
    near-copies on both sides: the same huddle seen from each of its two mice,
    and neighbouring clips of the same animals in the same light. Validation
    then measures memory of a session rather than recognition of a behaviour.
    Holding out whole videos asks the question that matters: does the model
    score video it has never seen?

    With a single source video there is nothing to hold out, so whole clips
    are held out instead (one clip's animals always stay together) and
    ``info["unit"]`` says so; that score is still optimistic.

    Videos are chosen greedily to bring each class's validation share towards
    ``val_fraction`` without overshooting it. A class found in only one video
    keeps that video in training, because holding it out would leave nothing
    to learn the class from; it is reported in ``info["train_only_classes"]``.

    Returns ``(train_idx, val_idx, info)``.
    """
    import random
    from collections import Counter

    labels = list(labels)
    n = len(labels)
    video_keys = [video_key(v) if v else f"clip:{c}" for v, c in zip(videos, clips)]
    if len(set(video_keys)) >= 2:
        unit_kind, units = "video", video_keys
    else:
        unit_kind, units = "clip", [str(c) for c in clips]

    n_cls = Counter(labels)
    unit_counts: Dict[str, Counter] = {}
    for u, lbl in zip(units, labels):
        unit_counts.setdefault(u, Counter())[lbl] += 1
    units_with: Dict[object, set] = {}
    for u, cnt in unit_counts.items():
        for lbl in cnt:
            units_with.setdefault(lbl, set()).add(u)

    info = {
        "unit": unit_kind,
        "seed": seed,
        "val_units": [],
        "train_only_classes": sorted(
            (lbl for lbl, us in units_with.items() if len(us) < 2), key=str),
    }

    val: set = set()
    if val_fraction > 0 and len(unit_counts) >= 2:
        order = sorted(unit_counts)
        random.Random(seed).shuffle(order)
        target = {lbl: val_fraction * k for lbl, k in n_cls.items()}
        got: Counter = Counter()

        def keeps_training(u):
            # Every class must keep at least one unit on the training side.
            return all(len(units_with[lbl] - val - {u}) >= 1 for lbl in unit_counts[u])

        def gain(u):
            g = 0.0
            for lbl, k in unit_counts[u].items():
                need = max(0.0, target[lbl] - got[lbl])
                g += (min(k, need) - 0.5 * max(0.0, k - need)) / n_cls[lbl]
            return g

        while sum(got.values()) < val_fraction * n:
            cands = [u for u in order if u not in val and keeps_training(u)]
            if not cands:
                break
            best = max(cands, key=gain)  # ties go to the earliest in shuffled order
            if gain(best) <= 0:
                break
            val.add(best)
            got.update(unit_counts[best])

        # A class that could be validated but drew no validation examples gets
        # its smallest eligible unit, even if that overshoots the fraction.
        for lbl in sorted(n_cls, key=str):
            if got[lbl] or len(units_with[lbl]) < 2:
                continue
            cands = [u for u in order
                     if lbl in unit_counts[u] and u not in val and keeps_training(u)]
            if cands:
                pick = min(cands, key=lambda u: sum(unit_counts[u].values()))
                val.add(pick)
                got.update(unit_counts[pick])

    train_idx = [i for i in range(n) if units[i] not in val]
    val_idx = [i for i in range(n) if units[i] in val]
    info["val_units"] = sorted(val)
    info["n_units"] = len(unit_counts)
    info["per_class"] = {
        lbl: {"train": sum(1 for i in train_idx if labels[i] == lbl),
              "val": sum(1 for i in val_idx if labels[i] == lbl)}
        for lbl in sorted(n_cls, key=str)
    }
    info["classes_without_val"] = [
        lbl for lbl, c in info["per_class"].items() if c["val"] == 0]
    return train_idx, val_idx, info


def classification_report(y_true, y_pred, n_classes: int) -> dict:
    """Confusion matrix and per-class precision, recall and F1.

    Behaviour classes are badly unbalanced -- hours of resting for seconds of
    fighting -- so plain accuracy mostly measures the common class. Balanced
    accuracy (mean per-class recall) and macro F1 weight every behaviour
    equally; classes absent from ``y_true`` are left out of both averages.
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / np.maximum(support, 1), np.nan)
        precision = np.where(predicted > 0, tp / np.maximum(predicted, 1),
                             np.where(support > 0, 0.0, np.nan))
        f1 = np.where((precision + recall) > 0,
                      2 * precision * recall / (precision + recall), 0.0)
    f1 = np.where(support > 0, f1, np.nan)
    present = support > 0
    total = int(cm.sum())
    return {
        "confusion": cm,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": float(tp.sum() / total) if total else 0.0,
        "balanced_accuracy": float(np.nanmean(recall[present])) if present.any() else 0.0,
        "macro_f1": float(np.nanmean(f1[present])) if present.any() else 0.0,
    }


def image_to_tensor(scene: np.ndarray):
    """(H, W, C) uint8 composite to a normalised CHW float tensor."""
    import torch

    arr = np.asarray(scene, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_scene_classifier(model_dir: str, device: Optional[str] = None):
    """Load a trained scene classifier for inference.

    Returns ``(model, class_names, device, config)``, or None when the
    directory holds a legacy three-channel silhouette checkpoint, so callers
    can fall back rather than crash on an old model.
    """
    import torch

    cfg_path = os.path.join(model_dir, "classifier_config.json")
    weights = os.path.join(model_dir, "best_classifier.pth")
    if not os.path.isfile(cfg_path) or not os.path.isfile(weights):
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("input_kind") != INPUT_KIND_SCENE:
        return None

    if device is None:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    class_names = cfg.get("class_names", [])
    model = build_scene_model(
        cfg.get("backbone", "ResNet-18"),
        n_classes=cfg.get("n_classes", len(class_names)),
        n_features=len(cfg.get("feature_fields", PAIRWISE_FIELDS)),
        freeze_backbone=False,
        in_channels=cfg.get("in_channels", 6),
    )
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()
    return model, class_names, device, cfg


def predict_scene(model, cfg, scene: np.ndarray, features: np.ndarray, device: str):
    """Class probabilities for one window. Returns a 1-D numpy array."""
    import torch

    img = image_to_tensor(scene).unsqueeze(0).to(device)
    vec = normalize_features(features, cfg["feature_mean"], cfg["feature_std"])
    feats = torch.from_numpy(vec).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(model(img, feats), dim=1)[0].cpu().numpy()
