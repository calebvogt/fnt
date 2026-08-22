# Video Tracking Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install opencv-python torch segment-anything pandas numpy
```

### Step 2: Download SAM Checkpoint
Choose one model checkpoint:
- **Best accuracy:** [sam_vit_h_4b8939.pth (2.6 GB)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- **Balanced:** [sam_vit_l_0b3195.pth (1.3 GB)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- **Fastest:** [sam_vit_b_01ec64.pth (375 MB)](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

### Step 3: Launch FNT
```bash
python -m fnt.gui_pyqt
```

### Step 4: Run OFT Tracker
1. Go to **Video Tracking** tab
2. Click **Open Field Test** button
3. Follow the GUI workflow:
   - Select video(s)
   - Select SAM checkpoint
   - Load first frame
   - Click on animal
   - Draw arena circle
   - Start tracking!

## 📊 Output Files

After tracking completes, you'll find two files:

### `{video_name}_trajectory.csv`
Frame-by-frame position data:
```csv
frame,x,y,confidence,time_s,vx,vy,speed
0,512.3,384.2,1.0,0.000,0.0,0.0,0.0
1,513.1,385.0,0.98,0.033,24.2,24.2,34.3
...
```

### `{video_name}_oft_metrics.txt`
Summary metrics:
```
Distance traveled: 45623.2 pixels
Time in center zone: 42.5 s (14.2%)
Average speed: 152.1 pixels/s
```

## 🎯 GUI Workflow

```
┌──────────────────────────────────────┐
│ 1. Select Videos                     │
│    └─ Single or multiple files       │
├──────────────────────────────────────┤
│ 2. Select SAM Checkpoint             │
│    └─ Choose .pth file               │
├──────────────────────────────────────┤
│ 3. Setup Tracking                    │
│    ├─ Load first frame               │
│    ├─ Click on animal (SAM segments) │
│    └─ Draw arena circle              │
├──────────────────────────────────────┤
│ 4. Run Tracking                      │
│    └─ Watch real-time visualization  │
├──────────────────────────────────────┤
│ 5. Review Results                    │
│    └─ CSV trajectory + metrics       │
└──────────────────────────────────────┘
```

## ⚙️ Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Model Type** | vit_h | SAM model size (vit_h/vit_l/vit_b) |
| **Device** | cuda | Use GPU (cuda) or CPU (cpu) |
| **SAM Update Interval** | 30 frames | How often to refine with SAM |

### Tuning Tips
- **Slow tracking?** → Use vit_b model or increase interval to 50
- **Losing animal?** → Decrease interval to 20 or click more precisely
- **GPU not detected?** → Install CUDA PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## 🔍 Troubleshooting

### "SAM checkpoint path required"
**Fix:** Download a .pth file from links above and select it in GUI

### Tracking drifts away from animal
**Fix:** 
- Decrease SAM update interval (try 20 instead of 30)
- Click more precisely on animal center during setup
- Ensure good contrast between animal and background

### Very slow processing
**Fix:**
- Use vit_b model instead of vit_h
- Enable GPU: verify with `python -c "import torch; print(torch.cuda.is_available())"`
- Increase SAM update interval to 50-100

## 🎬 Example Analysis

Load trajectory in Python for custom analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('mouse_trajectory.csv')

# Plot trajectory
plt.figure(figsize=(10, 8))
plt.plot(df['x'], df['y'], alpha=0.6, linewidth=0.5)
plt.scatter(df['x'].iloc[0], df['y'].iloc[0], c='green', s=100, label='Start')
plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='red', s=100, label='End')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.title('Mouse Trajectory - Open Field Test')
plt.legend()
plt.gca().invert_yaxis()  # Match video coordinates
plt.axis('equal')
plt.savefig('trajectory_plot.png', dpi=300)
plt.show()

# Speed analysis
plt.figure(figsize=(12, 4))
plt.plot(df['time_s'], df['speed'])
plt.xlabel('Time (seconds)')
plt.ylabel('Speed (pixels/second)')
plt.title('Movement Speed Over Time')
plt.savefig('speed_analysis.png', dpi=300)
plt.show()

# Summary statistics
print(f"Total distance: {df['speed'].sum() / 30:.1f} pixels")  # Assuming 30 fps
print(f"Mean speed: {df['speed'].mean():.1f} px/s")
print(f"Max speed: {df['speed'].max():.1f} px/s")
print(f"Duration: {df['time_s'].iloc[-1]:.1f} seconds")
```

## 🆘 Getting Help

- **Documentation:** See full README.md in `fnt/videoTracking/`
- **Issues:** [github.com/calebvogt/fnt/issues](https://github.com/calebvogt/fnt/issues)
- **Discussions:** [github.com/calebvogt/fnt/discussions](https://github.com/calebvogt/fnt/discussions)

## 📋 Requirements Checklist

- [ ] Python 3.8+
- [ ] OpenCV: `pip install opencv-python`
- [ ] PyTorch: `pip install torch`
- [ ] SAM: `pip install segment-anything`
- [ ] Pandas: `pip install pandas`
- [ ] NumPy: `pip install numpy`
- [ ] PyQt5: `pip install PyQt5`
- [ ] SAM checkpoint downloaded (.pth file)
- [ ] GPU with CUDA (optional but recommended)

## 🎓 Best Practices

1. **Test with one video first** - Get parameters right before batch processing
2. **Use consistent lighting** - Reduces need for per-video adjustments
3. **Click precisely on animal center** - Better initial segmentation = better tracking
4. **Monitor confidence values** - Values < 0.5 indicate potential tracking issues
5. **Save your SAM checkpoint location** - You'll use it for every session
6. **Check first 100 frames manually** - Verify tracking quality before full run

## 📈 Coming Soon

- **Light-Dark Box tracker** with occlusion handling
- **Multi-animal tracking** for social interaction studies
- **Elevated Plus Maze** specific metrics
- **Real-time tracking** during recording
- **Export to BORIS** for behavioral annotation

---

**Version:** 0.1.0  
**Last Updated:** 2024  
**Part of:** FieldNeuroToolbox
