# ToDo

 # UWB
- animation fix; the trailing track persists and fades until the last frames. something is up with the timing of the display, point movement ends prematurely until track ends, losing time at the end of the day. 
- add ROI analysis with JSON memory integration for ROI labels
- social behavior refinement? Deactivate this module, too costly to run at the moment
- Spatial behavior analysis - deactivate, too costly to run. 
- behavioral classification module
- For animation output, add ability to check boxes related to whether aligned IR cameras should be shown, different types of plots, etc, broken up into panels. 
- In the long term, it may also be really cool to have a command center at voleterra that shows plots that are update every 10 minutes or so with what is going on out in the field. 
- uwb_animate
	- add behavior classification function
	- incorporate behavior classifier visualization to uwb_animate
	- behavior color across time; across days; [[Kathleen Murphy, CUB]]; 
    - Kathleen sent me her features python script
	- rest; movement; chase; follow; huddle; 
	- distance between the centroids; nose distance; centroid velocity; velocity towards other animal; delta distance between animals; orientation between animals; variance over a time window stdev of variance over x time window; medians over rolling windows;
	- simba, supervised, didnt want to label; alpha tracker is unsupervised; simba features are overly complicated; spectral clustering or hierarchical clustering, she tried both; 
- Show USV rate for each resource zone to the right of the animation. 
- Show the UWB tracking alongside the actual footage; 

# Doric WiFP
- RIP my old matlab code from changwoo and get it to work with the WiFP data. 
- basically we want to replicate many of the features outlined here: https://neuro.doriclenses.com/pages/data-analysis-solutions



# FED devices
- make a basic pipeline and visualization tool; can likely rip fromm their own python based tool and just incorporate it here. 

# USV
- make my own USV detector to replace DASl; ML or other. 
- USV analysis; implement unsupervised clustering, and quick plots feature
- add ability to overlay the usv calls with the video, temporally aligned based on shared timestamp (show DAS detection overlay)
- create UMAP of prairie vole vocalization types
- 


## MAD (Mask Audio Detector)
- self-supervised pretraining on our own unlabeled audio. ImageNet-pretrained resnet50 is a poor prior for spectrograms; we have ~3.7 TB of unlabeled VoleCosm USV audio sitting there. Masked-autoencoder or contrastive pretraining of the encoder on our own spectrograms, then fine-tune the U-Net. Highest ceiling of the generalization ideas, also the most work — do it once labeling breadth stops being the bottleneck.
- harmonic linker. MAD has no way to relate a harmonic to its fundamental (CAD's harmonic columns are deliberately omitted). Current labeling policy is fundamentals only, so this only matters if harmonic structure itself becomes a variable of interest.
- cross-recording call-type clustering / UMAP off the stored per-call embeddings (`mad embeddings`) — overlaps with the UMAP item above.
- active learning is a *workflow* change not a code one: label the calls whose score sits near the decision threshold rather than labeling in file order. Sort the Detections list by Score and work the middle.
- watch the train/val split when adding labeled recordings. With very unbalanced per-file label counts, `grouped_split` can hold out most of the data; the run summary reports `split_level` and val tile count — check it rather than trusting the number.

# video tracking
- improve the video tracking/optical flow SAM module tab; still underperforming

# video processing
 - explore CLAHE algo implementation for video processing; may not be necessary

# Camera Grid
Cross-camera sync is currently anchored on the day-bucket stamp plus accumulated
segment duration. Measured on T005: each camera carries a small, STABLE offset
from that anchor (cam1 +0.36s, cam4 +0.85s, cam2 +1.11s, cam3 +1.47s; sd 0.08s
for the well-read cameras over a full day), giving about a 1.0s spread between
cells. Fine for scoring a transition, but the residual is structured rather than
random, so it is correctable.

- **Manual clock calibration.** Per-camera frame-by-frame nudge: the user steps
  each cell until its burnt-in clock ticks over, and every camera is aligned to
  the first frame of the SAME named second. One frame at 30fps is 33ms, roughly
  30x tighter than today. Offsets are stable across a trial, so this is a
  one-time-per-trial step that amortises over a multi-day encode. Sidesteps the
  OCR problem below entirely, since a human reads a cluttered clock easily.
  Must display each camera's current clock VALUE, not just "a tick" - aligning
  cam1's :30 to cam3's :32 would lock in a whole-second error.
- **Audio cross-correlation calibration.** Cameras sharing a room hear the same
  sounds, so cross-correlating their audio would align them to the millisecond -
  far beyond what a 1-second burnt-in clock can resolve, and with no OCR at all.
  Only possible from RAW footage: the preprocessing step strips the mono PCM
  track the ViewTron writes. Highest achievable precision of the three options.
- **Automatic clock calibration** (partially prototyped). Detect the frame where
  the burnt-in clock ticks and derive a sub-second phase. Works well on cameras
  whose clock sits over dark background (cam1/cam4 measured to sd ~0.09s), but
  the template matcher fails on cam5, where the clock overlays bright clutter,
  and is spotty on cam2/cam3. Temporal differencing would likely isolate the
  digits - they change every second while the background does not.
- Recorder profiles beyond ViewTron, for NVRs whose offload windows are not
  midnight-to-midnight and whose naming we do not yet model.
- Consider caching per-camera normalised streams so re-running a trial with a
  different layout does not re-decode everything.

 # sleap tools
- for roi tool; auto load the keypoint tracking and allow user to scroll through with the track labels. 
 - do NOT use sleap-render commmand from CLI; it is wicked slow. 
- allow scroll wheel zoom and click to pan feature in the video preview. 


# Sleap EthoScope / Behavioral Catagorizer
- create behavioral classifiers for ethogram like behaviors; clustering here? or just use keypoint-moseq?
-


# Imaging Tool (CZI Viewer)

## Completed
- [x] CZI file loading with channel detection
- [x] Per-channel brightness, contrast, gamma, sharpness adjustments
- [x] Brightness thresholding with dual-handle slider
- [x] Rolling ball and Gaussian background subtraction with downsampling
- [x] False coloring with customizable palette
- [x] Scale bar from metadata (draggable position)
- [x] Text and shape annotations (arrow, line, circle, rectangle, freehand)
- [x] Export to PNG/TIFF with annotations and scale bar
- [x] Export settings JSON for traceability
- [x] Per-file settings preservation during navigation

## Cell Counting / Quantification Module (Standalone Tool)

### Overview
Standalone "Image Quantification" tool in the Imaging tab for automated cell/fiber counting on fluorescence microscopy images.

### Phase 1: Basic Cell Counting ✅
- [x] Intensity thresholding with adjustable threshold levels (Otsu/Triangle/Li/Manual)
- [x] Binary mask preview overlay (filled + contour outline modes)
- [x] Particle analysis with size filtering (min/max area in µm²)
- [x] Watershed separation for touching cells
- [x] Results table: count, mean area, total area, mean intensity, circularity

### Phase 2: Multi-Channel Analysis ✅
- [x] Count cells per channel independently
- [x] Colocalization analysis (% overlap between channels, Dice coefficient)
- [x] Cell-by-cell intensity measurements per channel
- [x] False-color composite preview with per-channel colors

### Phase 3: ROI-Based Counting ✅
- [x] Draw ROI regions for localized counting
- [x] Compare counts across ROIs
- [x] Density calculations (cells per mm²)

### Phase 4: Fiber Quantification
- [ ] Skeletonization for fiber/neurite tracing
- [ ] Total fiber length measurement
- [ ] Branch point detection and counting
- [ ] Fiber density per area

### Phase 5: Export & Reporting ✅
- [x] CSV export with all measurements (per-channel + colocalization + ROI density)
- [x] Overlay image export (PNG/TIFF) with masks, centroids, ROI rectangles
- [x] Batch processing across all loaded files (combined CSV output)
- [x] Analysis settings persistence (Save/Load JSON for reproducibility)

### UX Features ✅
- [x] Zoom-to-particle on table row selection (auto-pan/zoom to centroid)
- [x] Contour outline mode for mask preview (vs filled overlay)
- [x] Keyboard shortcuts (Ctrl+R, Ctrl+E, Ctrl+O, F, M, Left/Right, Ctrl+±)
- [x] Per-channel color composite preview with additive blending

### Technical Considerations
- [x] Uses scikit-image for segmentation (threshold_otsu, watershed, label)
- [ ] Consider deep learning option (Cellpose, StarDist) for advanced segmentation
- [x] Results linked to current display settings for reproducibility (auto-saved config JSON)
- [x] All measurements use calibrated units (µm, µm²) when pixel size is available

