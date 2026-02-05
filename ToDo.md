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
- 

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


# video tracking
- improve the video tracking/optical flow SAM module tab; still underperforming

# video processing
 - explore CLAHE algo implementation for video processing; may not be necessary

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

## Cell Counting / Quantification Module (Roadmap)

### Overview
Add a "Quantification" section below Export in the left panel for automated cell/fiber counting on fluorescence microscopy images.

### Phase 1: Basic Cell Counting
- [ ] Intensity thresholding with adjustable threshold levels
- [ ] Binary mask preview overlay on the image
- [ ] Particle analysis with size filtering (min/max area in µm²)
- [ ] Watershed separation for touching cells
- [ ] Results table: count, mean area, total area, mean intensity

### Phase 2: Multi-Channel Analysis
- [ ] Count cells per channel independently
- [ ] Colocalization analysis (% overlap between channels)
- [ ] Cell-by-cell intensity measurements per channel

### Phase 3: ROI-Based Counting
- [ ] Draw ROI regions for localized counting
- [ ] Compare counts across ROIs
- [ ] Density calculations (cells per mm²)

### Phase 4: Fiber Quantification
- [ ] Skeletonization for fiber/neurite tracing
- [ ] Total fiber length measurement
- [ ] Branch point detection and counting
- [ ] Fiber density per area

### Phase 5: Export & Reporting
- [ ] "Export Analysis" button
- [ ] CSV export with all measurements
- [ ] Overlay export showing detected objects
- [ ] Batch processing across all loaded files

### Technical Considerations
- Use scikit-image for segmentation (threshold_otsu, watershed, label)
- Consider deep learning option (Cellpose, StarDist) for advanced segmentation
- Results should be linked to the current display settings for reproducibility
- All measurements should use calibrated units (µm, µm²) when pixel size is available

