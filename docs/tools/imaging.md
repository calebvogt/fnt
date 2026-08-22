# Imaging

Microscopy image viewing, processing, and quantification tools for **Zeiss CZI** files. Designed for fluorescence microscopy workflows in neuroscience (e.g., immunohistochemistry, reporter imaging).

Launch from the FNT main GUI **Imaging** tab.

## CZI Viewer

Multi-channel microscopy image viewer and processor:

- **Load CZI files** — reads multi-channel, multi-scene Zeiss CZI images
- **False coloring** — assign colors to channels (GFP green, Cy3 magenta, DAPI blue, custom)
- **Per-channel adjustments** — brightness, contrast, gamma, sharpness, and min/max display range
- **Background subtraction** — Rolling Ball or Gaussian methods with configurable radius
- **Channel merging** — composite overlay of selected channels
- **Annotations** — add text labels and shape annotations directly on the image
- **Export** — save processed images to PNG or TIFF

## Image Quantification

Cell counting and analysis tool for CZI images:

- **Multi-channel detection** — detect cells/particles per channel with configurable thresholds
- **Watershed separation** — split touching cells using watershed segmentation
- **Colocalization analysis** — measure co-expression across channels via centroid overlap and Dice coefficient
- **ROI-based density** — draw regions of interest and calculate cell density (cells/mm²)
- **Export** — results to CSV for downstream statistical analysis

## Requirements

The Imaging tools need these optional packages:

```bash
pip install aicspylibczi fsspec Pillow scikit-image
```

- **aicspylibczi** — reads Zeiss CZI file format
- **fsspec** — required by aicspylibczi
- **Pillow** — image export and annotation rendering
- **scikit-image** — background subtraction (rolling ball) and watershed segmentation
