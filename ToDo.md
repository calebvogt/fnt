# ToDo

# video tracking
- improve the video tracking/optical flow SAM module tab; still underperforming

# video processing
 - explore CLAHE algo implementation for video processing; may not be necessary

 # sleap ROI Tool
- allow scroll wheel zoom and click to pan feature in the video preview. 

# Sleap EthoScope / Behavioral Catagorizer
- create behavioral classifiers for ethogram like behaviors; clustering here? or just use keypoint-moseq?
- 

 # UWB
- add ability to "play" the preview window
- add animate video functionality to quick viz tool. export options
- add battery level figure to export options
- create smoothed plots
- UWB Plots: velocity threshold input not taking, always defaulting to 0.1 (default) even if user inputs values. 
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

# USV
- USV analysis; implement unsupervised clustering, and quick plots feature
- add ability to overlay the usv calls with the video, temporally aligned based on shared timestamp (show DAS detection overlay)
- create UMAP of prairie vole vocalization types
- 
