import os
from pathlib import Path

import requests
from datasets import Dataset

from embdata.describe import describe

# Set the download directory
download_dir = "bridge_v2_tfds"

# Download the BRIDGE dataset v2
base_url = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/"


for i in range(100):
    file_name = f"bridge_dataset-train.tfrecord-{i:05d}-of-00100.gz"
    url = base_url + file_name
    file_path = Path(download_dir) / file_name
    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content)


# Download the validation data files
for i in range(128):
    file_name = f"bridge_dataset-val.tfrecord-{i:05d}-of-00128.gz"
    url = base_url + file_name
    file_path = Path(download_dir) / file_name
    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content)

# Download the additional files
for file_name in ["dataset_info.json", "features.json"]:
    url = base_url + file_name
    file_path = os.path.join(download_dir, file_name)
    response = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(response.content)


# Load the dataset
dataset = Dataset.load_from_disk(download_dir)

# Describe the dataset

describe(dataset)
