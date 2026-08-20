# Video Tracking Module

Interactive tracking tools for behavioral neuroscience experiments using Meta's **Segment Anything Model (SAM)** combined with classical computer vision techniques.

## Overview

This module provides test-specific tracking GUIs optimized for common behavioral paradigms:
- **Open Field Test (OFT)** - Track single or multiple animals in open arena
- **Light-Dark Box (LDB)** - Track with occlusion handling for dark compartment *(coming soon)*

## Why SAM?

Traditional background subtraction requires extensive parameter tweaking for each environment (lighting, reflections, shadows, enrichment objects). SAM provides:
- ✅ **Zero-shot segmentation** - Works on any object without training
- ✅ **Interactive setup** - Click on animal once, tracking runs automatically
- ✅ **Robust to clutter** - Handles reflections, shadows, enrichment objects
- ✅ **Minimal per-video setup** - Perfect for batch processing

## Installation

### 1. Core Dependencies
```bash
pip install opencv-python torch segment-anything pandas numpy
```

### 2. Download SAM Model Checkpoint

Download one of the SAM model checkpoints (required for tracking):

| Model | Size | Download Link |
|-------|------|---------------|
| **ViT-H (Huge)** | 2.6 GB | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| **ViT-L (Large)** | 1.3 GB | [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) |
| **ViT-B (Base)** | 375 MB | [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |

**Recommendation:** Use **ViT-H** for best accuracy, **ViT-B** for faster processing on limited hardware.

Save the checkpoint to a known location (e.g., `~/models/sam_vit_h_4b8939.pth`).

### 3. GPU Support (Optional but Recommended)

For faster processing, ensure PyTorch can use your GPU:
```bash
# CUDA (NVIDIA GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Check if GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Open Field Test Tracker

Launch from FNT main GUI: **Video Tracking** tab → **Open Field Test** button

#### Workflow

1. **Select Videos**
   - Click "Select Video Files"
   - Choose one or more video files (supports batch processing)

2. **Select SAM Checkpoint**
   - Click "Select SAM Checkpoint"
   - Navigate to your downloaded `.pth` file
   - Choose model type (vit_h, vit_l, or vit_b)
   - Select device (cuda for GPU, cpu for CPU)

3. **Setup Tracking**
   - Click "Load First Frame" to display the first video frame
   - Click "Click on Animal" and then click on the animal in the video
   - SAM will segment the animal automatically
   - Click "Draw Arena Circle":
     - First click: arena center
     - Second click: arena edge (sets radius)
   - The tool will automatically draw a center zone (inner 50% radius)

4. **Run Tracking**
   - Click "Start Tracking"
   - Watch real-time visualization of trajectory
   - Progress bar shows completion status
   - Cancel anytime if needed

5. **Output Files**
   - `{video_name}_trajectory.csv` - Frame-by-frame position data
   - `{video_name}_oft_metrics.txt` - Behavioral summary metrics

#### Output Format

**Trajectory CSV:**
| Column | Description |
|--------|-------------|
| `frame` | Frame number (0-indexed) |
| `x` | X coordinate (pixels) |
| `y` | Y coordinate (pixels) |
| `confidence` | Tracking confidence (0-1) |
| `time_s` | Time in seconds |
| `vx` | X velocity (pixels/second) |
| `vy` | Y velocity (pixels/second) |
| `speed` | Total speed (pixels/second) |

**OFT Metrics:**
- Total distance traveled (pixels)
- Time in center zone (seconds and %)
- Average speed (pixels/second)
- Duration and frame rate

### Light-Dark Box Tracker

**Status:** Coming Soon!

The LDB tracker will include:
- Rectangular ROI definition (light zone, dark zone, entrance)
- Zone-based tracking with occlusion handling
- Entry/exit detection with timestamps
- Anxiety metrics (time in light/dark, transitions, latency)

## Architecture

### Tracking Pipeline

```
┌─────────────────┐
│  User clicks    │ → SAM segments animal
│  on animal      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Optical Flow   │ → Fast frame-to-frame tracking
│  Tracking       │   (runs every frame)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Periodic SAM   │ → Refine segmentation every N frames
│  Updates        │   (default: every 30 frames)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kalman Filter  │ → Predict position during occlusions
│  Prediction     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CSV Export     │ → Trajectory + behavioral metrics
└─────────────────┘
```

### Key Components

**`sam_tracker_base.py`**
- `SAMTrackerBase` - Core tracking class
- Interactive point selection
- SAM segmentation integration
- Optical flow tracking (Lucas-Kanade)
- Kalman filter for motion prediction
- Trajectory export with metrics

**`oft_tracker_gui.py`**
- PyQt5 GUI for Open Field Test
- Interactive video display widget
- Circle drawing for arena ROI
- Real-time tracking visualization
- Batch processing support

**`ldb_tracker_gui.py`**
- Light-Dark Box tracker (placeholder)
- Will include rectangular ROI drawing
- Zone-based occlusion handling
- Entry/exit detection

## Parameters

### SAM Update Interval
- **Default:** 30 frames
- **Lower values** (10-20): More accurate but slower
- **Higher values** (50-100): Faster but may lose accuracy during rapid movements

### Confidence Threshold
- **Default:** 0.5
- Minimum confidence for optical flow tracking
- If confidence drops below threshold, switches to Kalman prediction

### Model Selection
- **vit_h (huge)**: Best accuracy, slowest, ~2.6 GB
- **vit_l (large)**: Good balance, ~1.3 GB
- **vit_b (base)**: Fastest, lowest accuracy, ~375 MB

## Troubleshooting

### "SAM checkpoint path required"
Download a SAM checkpoint file (see Installation section) and select it in the GUI.

### "CUDA not available" (running on CPU)
PyTorch is using CPU instead of GPU. This will work but be slower. To enable GPU:
1. Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
2. Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`

### Tracking loses animal
Try:
- **Lower SAM update interval** (e.g., 20 frames instead of 30)
- **Click more precisely** on the animal during initialization
- **Reduce video frame rate** if animal moves very quickly
- **Check lighting** - extreme shadows or reflections can confuse tracking

### Slow processing
- Use **vit_b** model instead of vit_h
- Increase **SAM update interval** to 50-100 frames
- Enable **GPU acceleration** (see CUDA installation above)
- Reduce video resolution or frame rate during preprocessing

### "Import error" or missing dependencies
Install all required packages:
```bash
pip install opencv-python torch segment-anything pandas numpy PyQt5
```

## Advanced Usage

### Batch Processing Multiple Videos

The OFT tracker supports batch processing:
1. Select multiple videos using Ctrl+Click in file dialog
2. Complete setup for first video (click animal, draw arena)
3. After first video completes, click "Yes" to process next video
4. Repeat for each video in batch

**Pro Tip:** If videos have similar framing, you can use the same ROI by noting the coordinates from the first video.

### Custom Analysis

The trajectory CSV files can be imported into Python for custom analysis:

```python
import pandas as pd
import numpy as np

# Load trajectory
df = pd.read_csv('video_trajectory.csv')

# Calculate custom metrics
total_distance = df['speed'].sum() / fps  # Total distance
max_speed = df['speed'].max()  # Peak speed
center_time = df[df['in_center']]['time_s'].count() / fps  # Time in center

# Plot trajectory
import matplotlib.pyplot as plt
plt.plot(df['x'], df['y'], alpha=0.5)
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.title('Animal Trajectory')
plt.gca().invert_yaxis()  # Match video coordinates
plt.show()
```

### ROI Coordinate Scaling

If you need to analyze videos at different resolutions, scale the ROI coordinates:

```python
# Original video: 1920x1080, arena center (960, 540), radius 400
# Downsampled to: 960x540
scale_factor = 960 / 1920
new_center_x = 960 * scale_factor  # 480
new_center_y = 540 * scale_factor  # 270
new_radius = 400 * scale_factor    # 200
```

## Roadmap

- [x] SAM tracker base class
- [x] Open Field Test GUI
- [x] Integration with main FNT GUI
- [ ] Light-Dark Box tracker with occlusion handling
- [ ] Multi-animal tracking support
- [ ] Elevated Plus Maze tracker
- [ ] Social interaction tracking (dual animals)
- [ ] Export to BORIS format
- [ ] Real-time tracking preview during recording

## References

- **Segment Anything Model (SAM):** [Kirillov et al., ICCV 2023](https://github.com/facebookresearch/segment-anything)
- **Lucas-Kanade Optical Flow:** Bouguet, J.-Y. (2001). Pyramidal implementation of the affine lucas kanade feature tracker.
- **Kalman Filter:** Kalman, R. E. (1960). A new approach to linear filtering and prediction problems.

## Citation

If you use this tracking module in your research, please cite:

```bibtex
@software{fnt_video_tracking,
  title = {FieldNeuroToolbox Video Tracking Module},
  author = {Vogt, Caleb and Contributors},
  year = {2024},
  url = {https://github.com/calebvogt/fnt}
}
```

## License

Part of FieldNeuroToolbox. See main repository for license information.

## Support

For issues, questions, or feature requests:
- **GitHub Issues:** [github.com/calebvogt/fnt/issues](https://github.com/calebvogt/fnt/issues)
- **Discussions:** [github.com/calebvogt/fnt/discussions](https://github.com/calebvogt/fnt/discussions)
