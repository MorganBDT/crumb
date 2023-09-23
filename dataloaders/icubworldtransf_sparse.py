import os
import pandas as pd
from shutil import copyfile
from tqdm import tqdm

# Replace with the actual paths
csv_file_path = 'icubworldtransf_dirmap.csv'
dataset_root_dir = '/media/KLAB37/datasets/icubworldtransf'

# Read the dataframe
df = pd.read_csv(csv_file_path)

# Create the new directory
new_root_dir = os.path.join(os.path.dirname(dataset_root_dir), 'icubworldtransf_sparse')
os.makedirs(new_root_dir, exist_ok=True)

# Iterate over each row in the dataframe with tqdm progress bar
for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    im_path = row['im_path']
    source_path = os.path.join(dataset_root_dir, im_path)

    # Create the same im_path inside icubworldtransf_sparse
    destination_path = os.path.join(new_root_dir, im_path)
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    # Copy the file
    copyfile(source_path, destination_path)
