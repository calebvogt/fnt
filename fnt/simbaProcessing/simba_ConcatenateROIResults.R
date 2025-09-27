## Created by Caleb C. Vogt, PhD Candidate @ Cornell University
library(tidyverse)
library(dplyr)
library(readr)
library(stringr)

setwd('Y:/Data/FieldProject/video_behavior_sleap/1_ephys_rig_high_quality/simba/project_folder/logs')
movement <- read.csv('Movement_log_20230822174032.csv')
entries <- read.csv('ROI_entry_data_20230822172036.csv')
time <- read.csv('ROI_time_data_20230822172036.csv')

## create ROI_analysis_concatenated.csv for each simba project
df1 <- movement %>%
  pivot_wider(names_from=MEASURE,values_from=VALUE) %>%
  select(VIDEO,`Distance (cm)`,`Velocity (cm/s)`) %>% 
  rename(video=VIDEO,distance_cm=`Distance (cm)`,velocity_cm_s=`Velocity (cm/s)`)

df2 <- entries %>%
  group_by(VIDEO) %>%
  summarise(
    shape_arena_entries=sum(ENTRIES[SHAPE=="arena"],na.rm=T),
    shape_1_entries=sum(ENTRIES[SHAPE==1],na.rm=T),
    shape_2_entries=sum(ENTRIES[SHAPE==2],na.rm=T)
  ) %>% 
  rename(video=VIDEO)

df3 <- time %>%
  group_by(VIDEO) %>%
  summarise(
    shape_arena_time=sum(TIME[SHAPE==1],na.rm=T),
    shape_1_time=sum(TIME[SHAPE==1],na.rm=T),
    shape_2_time=sum(TIME[SHAPE==2],na.rm=T)
  ) %>% 
  rename(video=VIDEO)

df4 <- df1 %>% 
  inner_join(df2,by="video") %>%
  inner_join(df3,by="video")

# Write the reshaped data to a CSV file. 
write.csv(df4,'ROI_AnalysisConcatenated.csv',row.names=F)

## optional: concatenate ROI_AnalysisConcatenated.csv files across project batches. 
## move copies of all files to be merged to single folder, rename with some kind of numeric index appended "_1"
setwd('Y:/Data/FieldProject/video_behavior_sleap/test')
files <- list.files(pattern="*ROI*",full.names=T)
list_of_dfs <- lapply(files,read.csv)
df <- bind_rows(list_of_dfs)

df1 <- df %>% 
  arrange(video) %>% 
  mutate(trial=substr(video,1,4),
         code=stringr::str_extract(video,"(?<=_)[A-Z]{3}"),
         prepost=tolower(str_extract(video,"Pre|Post")), ## Extracts "Pre" or "Post" and converts to lowercase) %>%  # Extracts the 3-letter code after the underscore
         task=tolower(str_extract(video,"Social|ObJDisp2|ObjDisp|ObjNov"))) %>%  
  select(video,trial,code,prepost,task,everything()) %>% 
  unique()




