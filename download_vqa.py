'''
Helper script to download the VQAv2 dataset using the vqa_dataset_wget.py loading script.
'''
import datasets
import os

# Define the target directory for raw downloads and extractions
# The vqa_dataset_wget.py script will place zips and extracted folders (like questions_train, images_train) inside this directory.
target_raw_data_dir = "/work/com-304/vqav2_data_raw"

print(f"Attempting to download and process VQAv2 dataset using vqa_dataset_wget.py")
print(f"Raw data (zips, extracted contents) will be stored in: {target_raw_data_dir}")

# Ensure the target directory for raw data exists
os.makedirs(target_raw_data_dir, exist_ok=True)

try:
    # Load the dataset. This will trigger the download and extraction logic
    # in _split_generators of vqa_dataset_wget.py using target_raw_data_dir.
    # The processed Hugging Face dataset will be cached separately (usually ~/.cache/huggingface/datasets)
    # unless a cache_dir is specified in load_dataset.
    print("Starting dataset loading process. This will take a while as it downloads and extracts large files...")
    vqa_dataset = datasets.load_dataset(
        path="vqav2_data/vqa_dataset_wget.py", # Path to your local loading script
        data_dir=target_raw_data_dir,         # This directory is passed to the script for downloads
        # trust_remote_code=True # Not needed as it's a local script path
    )
    print("\nSuccessfully downloaded, processed, and loaded the VQAv2 dataset.")
    print("Dataset structure:")
    print(vqa_dataset)
    print(f"\nRaw data (zip files, extracted image folders, question/annotation JSONs) should now be in {target_raw_data_dir}")

except Exception as e:
    print(f"\nAn error occurred during dataset loading: {e}")
    print("Please check the following:")
    print("1. You have 'wget' and 'unzip' installed and in your system PATH.")
    print(f"2. You have write permissions to {target_raw_data_dir} and sufficient disk space.")
    print("3. The URLs in vqa_dataset_wget.py are accessible.")
    print("4. If downloads were interrupted, you might need to manually clear partially downloaded files from " 
          f"{target_raw_data_dir} and try again.") 