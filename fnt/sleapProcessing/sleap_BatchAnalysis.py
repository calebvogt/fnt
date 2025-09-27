### MODIFIED FROM: https://colab.research.google.com/github/murthylab/sleap-notebooks/blob/master/Analysis_examples.ipynb#scrollTo=lZ9wU0jzM8Pb

import os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import pandas as pd


os.chdir(r'C:\Users\Caleb Vogt\Desktop\SLEAP_CM_Female_Preference_v2\new and old')

# varify the path using getcwd() 
cwd = os.getcwd() 
  
# print the current directory 
print("Current working directory is:", cwd)

# load h5 file as names
file1 ="C57-494_Preference_D1_Pregnant_070219_Mup20-H2O_MB2_CAM6_12_32_19_830_trim.mp4.predictions.slp"
file2 = "C57-494_Preference_D2_Pregnant_070319_Mup20-H2O_MB2_CAM6_12_23_41_810_trim.mp4.predictions.slp"


#read in as pandas
pd_1 = pd.read_hdf(file1)
pd_1 = pd.read_hdf(file1,'instances')

pd_2 = pd.read_hdf(file2,'instances')






##file1
with h5py.File(file1, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    frames = f['frames']
    instances = f['instances']
    metadata =  f['metadata']
    points =  f['points']
    pred_points = f['pred_points']
    suggestions =  f['suggestions_json']
    tracks_json = f['tracks_json']
    videos_json = f['videos_json']
    
    

for key in points.keys():
    print(key)







## New prediction, does not import due to metadata issue
list(f.keys())
meta1 = f['metadata']
meta1[:]


## old precition, imports but without tracking
list(ff.keys())
meta2 = ff['metadata']
meta2[:]




f = h5py.File('C57-494_Preference_D1_Pregnant_070219_Mup20-H2O_MB2_CAM6_12_32_19_830_trim.mp4.predictions.slp', 'r')
ff = h5py.File('C57-494_Preference_D2_Pregnant_070319_Mup20-H2O_MB2_CAM6_12_23_41_810_trim.mp4.predictions.slp', 'r')






with h5py.File('C57-494_Preference_D2_Pregnant_070319_Mup20-H2O_MB2_CAM6_12_23_41_810_trim.mp4.predictions.analysis.h5', 'r') as f:
    occupancy_matrix = f['track_occupancy'][:]
    tracks_matrix = f['tracks'][:]

print(occupancy_matrix.shape)
print(tracks_matrix.shape)


with h5py.File('C57-494_Preference_D2_Pregnant_070319_Mup20-H2O_MB2_CAM6_12_23_41_810_trim.mp4.predictions.analysis.h5', "r") as f:  ##change file name here. 
    dset_names = list(f.keys())
    locations = f["tracks"][:].T       ### T for transpose
    node_names = [n.decode() for n in f["node_names"][:]]

print("===filename===")
print(filename)
print()

print("===HDF5 datasets===")
print(dset_names)
print()

print("===locations data shape===")
print(locations.shape)
print()

print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")
print()




frame_count, node_count, _, instance_count = locations.shape

print("frame count:", frame_count)
print("node count:", node_count)
print("instance count:", instance_count)


from scipy.interpolate import interp1d

def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


locations = fill_missing(locations)



### Visualize Thorax movement across video

HEAD_INDEX = 0
THORAX_INDEX = 1
ABDO_INDEX = 2

head_loc = locations[:, HEAD_INDEX, :, :]
thorax_loc = locations[:, THORAX_INDEX, :, :]
abdo_loc = locations[:, ABDO_INDEX, :, :]

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set('notebook', 'ticks', font_scale=1.2)
mpl.rcParams['figure.figsize'] = [15,6]




##PLOT FIGURE

plt.figure()
plt.plot(thorax_loc[:,0,0], 'y',label='fly-0')
plt.plot(thorax_loc[:,0,1], 'g',label='fly-1')

plt.plot(-1*thorax_loc[:,1,0], 'y')
plt.plot(-1*thorax_loc[:,1,1], 'g')

plt.legend(loc="center right")
plt.title('Thorax locations')


plt.figure(figsize=(7,7))
plt.plot(thorax_loc[:,0,0],thorax_loc[:,1,0], 'y',label='fly-0')
plt.plot(thorax_loc[:,0,1],thorax_loc[:,1,1], 'g',label='fly-1')
plt.legend()

plt.xlim(0,1024)
plt.xticks([])

plt.ylim(0,1024)
plt.yticks([])
plt.title('Thorax tracks')
