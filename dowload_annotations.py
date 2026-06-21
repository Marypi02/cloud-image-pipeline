# Local script to download and split sample images

import os 
import urllib.request
import zipfile

# the direct and official URL to the COCO annotations
url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
zip_path = "annotations_trainval2017.zip"
extract_dir = "./coco_annotations"

print("Starting download of COCO annotations...")

try:
    # Download the zip file
    urllib.request.urlretrieve(url, zip_path)
    print("Download completed successfully.")
    
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Extraction completed successfully.")
    
    # Clean up the zip file
    os.remove(zip_path)
except Exception as e:
    print(f"An error occurred: {e}")
