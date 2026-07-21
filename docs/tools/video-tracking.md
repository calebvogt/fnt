# Video Tracking

Interactive animal tracking using Meta's **Segment Anything Model (SAM)** combined with classical computer vision. Designed for common behavioral paradigms such as the Open Field Test.

Launch from the FNT main GUI **Video** tab → **Video Tracking**.

## Why SAM?

Traditional background subtraction requires extensive per-environment parameter tuning (lighting, reflections, shadows, enrichment objects). SAM provides:

- **Zero-shot segmentation** — works on any object without training
- **Interactive setup** — click on the animal once, then tracking runs automatically
- **Robustness to clutter** — handles reflections, shadows, and enrichment objects
- **Minimal per-video setup** — suited to batch processing

## Tracking pipeline

1. **User clicks on the animal** → SAM segments it
2. **Optical flow tracking** → fast frame-to-frame tracking (every frame)
3. **Periodic SAM updates** → refine segmentation every N frames (default 30)
4. **Kalman filter** → predict position during occlusions
5. **CSV export** → trajectory plus behavioral metrics

## Workflow

1. **Select videos** — one or more (supports batch processing)
2. **Select the SAM checkpoint** — a downloaded `.pth` file; choose model type (vit_h/vit_l/vit_b) and device (cuda/cpu)
3. **Set up tracking** — load the first frame, click on the animal, and draw the arena ROI
4. **Run tracking** — watch the real-time trajectory visualization; cancel any time
5. **Outputs** — `{video}_trajectory.csv` (frame-by-frame position) and `{video}_oft_metrics.txt` (behavioral summary)

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| SAM update interval | 30 frames | Lower = more accurate but slower |
| Confidence threshold | 0.5 | Below this, switches to Kalman prediction |
| Model | vit_h / vit_l / vit_b | Accuracy vs. speed tradeoff |

## SAM model checkpoints

| Model | Size | Download |
|-------|------|----------|
| ViT-H (Huge) | 2.6 GB | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| ViT-L (Large) | 1.3 GB | [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) |
| ViT-B (Base) | 375 MB | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |

Use **ViT-H** for best accuracy, **ViT-B** for faster processing on limited hardware. GPU acceleration is strongly recommended — see [Installation](../installation.md#gpu-support-optional).

!!! note
    See the [module README](https://github.com/calebvogt/fnt/blob/main/fnt/videoTracking/README.md) for the full architecture reference, troubleshooting guide, and custom-analysis examples.
