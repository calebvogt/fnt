import h5py
with h5py.File('Y:/Data/FieldProject/video_behavior_sleap/2_ephys_rig_low_quality/T001_CAJ_1_PreField_Social_1_OFT_Basler acA640-90uc (24393403)_20221002_150413135.h5', 'r') as f:
    occupancy_matrix = f['track_occupancy'][:]
    tracks_matrix = f['tracks'][:]

print(occupancy_matrix.shape)
print(tracks_matrix.shape)