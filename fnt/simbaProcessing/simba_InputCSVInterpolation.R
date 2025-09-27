## Created by Caleb C. Vogt, PhD Candidate @ Cornell University
library(tidyverse)
library(data.table)
library(zoo)
library(beepr)

wd <- setwd("Y:/Data/FieldProject/video_behavior_sleap/3_liddell_bucket/simba/project_folder/csv/input_csv") ## confirm that it switched.
dir.create("proc")
filenames <- list.files(wd,pattern="*.csv") ## these should be csvs resulting from recent import of .h5 files into simba. Check if there are added header rows, adjust line 13 if needed. 
aa=1
for(aa in 1:length(filenames)){
  df <- fread(filenames[aa],header=T,skip=2) ## read in the file, drop the any header rows added by simba
  df[df==0] <- NA
  df[df=="0.0"] <- NA # replace frame 0 NA with 0 again
  df[1,1] <- 0 ## replace frame 0 NA with 0 
  x_columns <- grep("_x$",colnames(df)) ## Get the indices of columns ending with '_x'
  first_row <- NA
  for (i in 1:nrow(df)) { ## find the first row within the '_x' cols that is not NA, then break the loop. 
    if (any(!is.na(df[i,..x_columns]))) {
      first_row <- i
      break
    }
  }
  sub_df <- df[first_row:nrow(df),] ## get rows where mouse is detected 
  filled_df <- as.data.table(zoo::na.fill(sub_df, "extend")) ## smooth linear interpolation of mouse rows
  df[first_row:nrow(df),] <- filled_df ## rejoin interpolated data with the initial empty frames. 
  df[is.na(df)] <- 0 ## required by simba v1.71.7 for non-visible keypoints
  
  
  colnames(df)[1] <- "" ## make empty cell for first column name to help with simba reading. 
  write.table(df,paste0("proc/",filenames[aa]),row.names=F,col.names=T,sep =",")
  print(paste("Finished converting file",filenames[[aa]]))
}
beepr::beep(4)
## copy files into the outlier_corrected_movement_location folder