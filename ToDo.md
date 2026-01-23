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

