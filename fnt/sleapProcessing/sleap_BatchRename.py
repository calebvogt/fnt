## batch_sleap_rename.py
import os

wd = os.chdir(r"C:\Users\Caleb Vogt\Desktop\12_Matthew_Mouse_TrackingSLEAP_Transfer\Batch_1") ## CHANGE THIS

for filename in os.listdir(wd):
   os.rename(filename, filename.replace('.mp4.predictions', ''))
