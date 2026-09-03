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
