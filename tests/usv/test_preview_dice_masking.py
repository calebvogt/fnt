"""The live preview must score a tile the way validation does.

Watching a run, tiles showed a red predicted blob with no green outline over
calls the user knew were labelled, and reported dice far below the run's
val_dice. Both symptoms had one cause: the preview ignored the supervision
weight.

Supervision is column-wise (``supervision_weight``): a time column holding any
labelled pixel is supervised across all frequencies; a column with no label is
not scored at all. So a model finding a real call in an unlabelled column is
neither right nor wrong — validation excludes it. The preview counted it as a
false positive, and drew nothing to say the region was unscored, which made a
correctly-ignored area look like a forgotten label.
"""
import numpy as np
import pytest

from fnt.usv.usv_detector.mad_labels import positive_target, supervision_weight


def _dice(pred_bin, gt, weight=None):
    """The validation loop's formula, optionally masked."""
    if weight is not None:
        pred_bin = pred_bin & weight
        gt = gt & weight
    inter = float(np.logical_and(pred_bin, gt).sum())
    den = float(pred_bin.sum() + gt.sum())
    return (2.0 * inter / den) if den else 1.0


@pytest.fixture
def tile():
    """One labelled call, plus an unlabelled call elsewhere in the tile."""
    H, W = 64, 128
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[30:34, 10:40] = 1
    prob = np.zeros((H, W), dtype=np.float32)
    prob[30:34, 10:40] = 0.9        # the labelled call, found
    prob[45:49, 70:100] = 0.9       # an unlabelled call, also found
    return mask, prob


def test_supervision_is_column_wise(tile):
    mask, _ = tile
    w = supervision_weight(mask)
    assert w[:, 10:40].all()            # every frequency in labelled columns
    assert not w[:, 70:100].any()       # nothing in unlabelled ones


def test_the_unmasked_dice_understates_a_perfect_tile(tile):
    """What the preview used to report."""
    mask, prob = tile
    gt = positive_target(mask) > 0.5
    assert _dice(prob > 0.5, gt) < 0.7


def test_the_masked_dice_matches_validation(tile):
    """The model got the labelled call exactly right, so dice is 1.0."""
    mask, prob = tile
    gt = positive_target(mask) > 0.5
    w = supervision_weight(mask)
    assert _dice(prob > 0.5, gt, w) == pytest.approx(1.0)


def test_a_genuine_miss_still_scores_badly(tile):
    """Masking must not launder real errors away."""
    mask, _ = tile
    gt = positive_target(mask) > 0.5
    w = supervision_weight(mask)
    prob = np.zeros(mask.shape, dtype=np.float32)   # found nothing at all
    assert _dice(prob > 0.5, gt, w) == pytest.approx(0.0)


def test_a_false_positive_inside_a_labelled_column_still_counts(tile):
    """Supervised means supervised: a wrong blob in a labelled column is wrong."""
    mask, _ = tile
    gt = positive_target(mask) > 0.5
    w = supervision_weight(mask)
    prob = np.zeros(mask.shape, dtype=np.float32)
    prob[30:34, 10:40] = 0.9        # the real call
    prob[50:60, 12:38] = 0.9        # spurious, but in a SUPERVISED column
    assert _dice(prob > 0.5, gt, w) < 0.75


def test_the_preview_builder_masks_and_reports_the_unscored_region(tile):
    """End to end through the real builder, with a stub model."""
    torch = pytest.importorskip("torch")
    from fnt.usv.usv_detector.mad_training import _build_epoch_previews

    mask, prob = tile
    spec = np.zeros(mask.shape, dtype=np.float32)
    target = positive_target(mask)
    weight = supervision_weight(mask).astype(np.float32)

    class Stub(torch.nn.Module):
        def forward(self, x):
            # Return logits that sigmoid to `prob`, ignoring the input.
            p = torch.from_numpy(prob).to(x.device)
            logits = torch.log(p.clamp(1e-6, 1 - 1e-6)
                               / (1 - p.clamp(1e-6, 1 - 1e-6)))
            return logits[None, None].repeat(x.shape[0], 1, 1, 1)

    tiles = _build_epoch_previews(
        Stub(), 'cpu', np.stack([spec]), np.stack([target]),
        np.stack([weight]), [0], [], np.random.default_rng(0), 1)
    assert len(tiles) == 1
    t = tiles[0]
    assert t['dice'] == pytest.approx(1.0, abs=1e-3)
    assert t['unsup'] is not None
    assert t['unsup'].any(), "the unlabelled columns must be marked unscored"


def test_a_tile_with_no_weights_still_builds():
    """weights=None keeps the old behaviour rather than crashing."""
    torch = pytest.importorskip("torch")
    from fnt.usv.usv_detector.mad_training import _build_epoch_previews

    spec = np.zeros((16, 16), dtype=np.float32)
    target = np.zeros((16, 16), dtype=np.float32)
    target[4:8, 4:8] = 1.0

    class Stub(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    tiles = _build_epoch_previews(
        Stub(), 'cpu', np.stack([spec]), np.stack([target]), None,
        [0], [], np.random.default_rng(0), 1)
    assert len(tiles) == 1
    assert tiles[0]['unsup'] is None
    assert tiles[0]['dice'] is not None
