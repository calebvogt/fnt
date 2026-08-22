"""Derive a ground-cover map for an arena from an overhead (drone) photo.

The user marks the four inner corners of the enclosure in the image; we fit a
homography from arena metres onto image pixels, sample a grid of cells over the
interior, and classify each sampled pixel as live vegetation using the Excess
Green index (ExG = 2g - r - b on chromatic coordinates). The result is a
relative-cover grid suitable for :class:`~fnt.abma.core.config.GrassSpec`.

Only *live green* cover is measurable this way — dry standing thatch is close
in colour to bare soil — so the map is used as the spatial pattern and a floor
is added to represent the roughly uniform dead layer.
"""
from __future__ import annotations

import numpy as np


def _homography(src, dst):
    """3x3 matrix mapping src points (N>=4, 2) onto dst points."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, V = np.linalg.svd(np.array(A, float))
    return V[-1].reshape(3, 3)


def analyse_photo(path, corners, arena_w, arena_h, grid=8, samples=26,
                  exg_threshold=0.045):
    """Measure live-green cover over an arena from an overhead photo.

    ``corners`` are the four inner corners of the enclosure in image pixels,
    given in arena order: SW, SE, NE, NW. Returns a dict with ``cover``
    (grid x grid array of green fraction, **rows south->north, cols west->east**),
    plus summary statistics.
    """
    from PIL import Image
    img = np.asarray(Image.open(path).convert("RGB"), float)
    ih, iw, _ = img.shape
    src = np.array([[0, 0], [arena_w, 0], [arena_w, arena_h], [0, arena_h]],
                   float)
    Hm = _homography(src, np.array(corners, float))

    def to_img(X, Y):
        q = Hm @ np.stack([X, Y, np.ones_like(X)])
        return q[0] / q[2], q[1] / q[2]

    cover = np.zeros((grid, grid))
    for i in range(grid):                      # i: south -> north
        for j in range(grid):                  # j: west -> east
            ys = (i + (np.arange(samples) + .5) / samples) * arena_h / grid
            xs = (j + (np.arange(samples) + .5) / samples) * arena_w / grid
            X, Y = np.meshgrid(xs, ys)
            px, py = to_img(X.ravel(), Y.ravel())
            px = np.clip(px.astype(int), 0, iw - 1)
            py = np.clip(py.astype(int), 0, ih - 1)
            c = img[py, px]
            s = c.sum(1) + 1e-6
            r, g, b = c[:, 0] / s, c[:, 1] / s, c[:, 2] / s
            exg = 2 * g - r - b
            # ignore equipment: blue boxes / bright white markers
            obj = (c[:, 2] > c[:, 0] + 18) & (c[:, 2] > c[:, 1] + 18)
            obj |= c.min(1) > 190
            ok = ~obj
            cover[i, j] = float((exg[ok] > exg_threshold).mean()) if ok.any() else 0.0
    return {
        "cover": cover,
        "mean_green": float(cover.mean()),
        "min_green": float(cover.min()),
        "max_green": float(cover.max()),
        "north_half": float(cover[grid // 2:].mean()),
        "south_half": float(cover[:grid // 2].mean()),
        "west_half": float(cover[:, :grid // 2].mean()),
        "east_half": float(cover[:, grid // 2:].mean()),
    }


def to_cover_map(cover, floor=0.40, decimals=2):
    """Relative-cover grid for GrassSpec: dead layer ``floor`` + green on top."""
    c = np.asarray(cover, float)
    peak = c.max()
    rel = c / peak if peak > 0 else np.zeros_like(c)
    return np.round(floor + (1.0 - floor) * rel, decimals).tolist()


def suggested_dry_fraction(mean_green, total_cover=0.55):
    """Share of drawn blades that should be straw-coloured.

    ``mean_green`` is the measured live fraction of ground; ``total_cover``
    is the assumed total (live + dead) vegetative cover, which a colour photo
    cannot separate from bare soil.
    """
    if total_cover <= 0:
        return 1.0
    return float(np.clip(1.0 - mean_green / total_cover, 0.0, 1.0))
