## batch_sleap_rename_cleaning.py
## you will need to delete or move the original uncleaned .slp files before running this. 
import os

wd = os.chdir(r"C:\Users\sheehan-lab\Desktop\16_AG_MUS_temp\2_MESH") ## CHANGE THIS

for filename in os.listdir(wd):
   os.rename(filename, filename.replace('.cleaned', ''))
