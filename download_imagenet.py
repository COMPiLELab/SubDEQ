import os
import requests
import zipfile


########################
##Download Tiny ImageNet


# Define the URL and the local filename
url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
local_filename = 'tiny-imagenet-200.zip'

# Download the file
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

# Unzip the file
with zipfile.ZipFile(local_filename, 'r') as zip_ref:
    zip_ref.extractall('.')

# Define the data directory
DATA_DIR = 'tiny-imagenet-200'

# Clean up by removing the zip file if desired
os.remove(local_filename)

print(f'Data downloaded and extracted to {DATA_DIR}')
