# http://bioconductor.org/packages/release/bioc/html/rhdf5.html
# install.packages("BiocManager")
# BiocManager::install("rhdf5")
library(rhdf5)
setwd("Y:/Data/FieldProject/video_behavior_sleap/2_ephys_rig_low_quality")

h5ls("T001_CAJ_1_PreField_Social_1_OFT_Basler acA640-90uc (24393403)_20221002_150413135.h5") ## view groups
instance_scores <- h5read("T001_CAJ_2_PostField_Social_1_OFT_Basler_acA800-510uc__22592284__20221009_125954721.h5", "instance_scores")
tracking_scores <- h5read("T001_CAJ_2_PostField_Social_1_OFT_Basler_acA800-510uc__22592284__20221009_125954721.h5", "tracking_scores")
point_scores <- h5read("T001_CAJ_2_PostField_Social_1_OFT_Basler_acA800-510uc__22592284__20221009_125954721.h5", "point_scores")
point_scores[,,1]
point_scores[,,2]
