import pickle
import numpy as np
import shutil

# Get the the list of frame number for the first 500
with open('/home/hpaat/my_exp/MTrans-evidential/small_experiments/experiment_100.01/train_first_500.pkl', 'rb') as f:
    frames_500_list = pickle.load(f)

source_path = '/home/hpaat/KITTI/data_object_label_2/training/label_2/'
dest_path = '/home/hpaat/my_exp/MTrans-evidential/pseudo_label_v15.5.2.v5_replace_500/'

# Replace the first 500
for frame in frames_500_list:
    # Source file path
    source_file = source_path + str(frame) + '.txt'

    # Destination file path
    destination_file = dest_path + str(frame) + '.txt'

    # Copy the file and replace the destination if it exists
    shutil.copy2(source_file, destination_file)