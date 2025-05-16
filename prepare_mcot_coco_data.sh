#!/bin/bash
# Script to download and prepare MS-COCO dataset for MCoT training
set -e  # Exit immediately if a command fails

# Create data directory
mkdir -p data/coco
cd data/coco

# Download COCO images and annotations
echo "Downloading COCO 2017 train images..."
wget -c http://images.cocodataset.org/zips/train2017.zip
echo "Downloading COCO 2017 validation images..."
wget -c http://images.cocodataset.org/zips/val2017.zip
echo "Downloading COCO 2017 annotations..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract the downloaded files with -n option (never overwrite existing files)
echo "Extracting train images..."
unzip -n train2017.zip
echo "Extracting validation images..."
unzip -n val2017.zip
echo "Extracting annotations..."
unzip -n annotations_trainval2017.zip

# Go back to the project root directory
cd ../..

# Check if shards already exist to avoid reprocessing
if [ -d "data/coco_mcot_shards" ]; then
    echo "WebDataset shards already exist at data/coco_mcot_shards"
    echo "Remove this directory if you want to recreate the shards."
else
    # Create WebDataset shards for MCoT training
    echo "Processing COCO data into WebDataset shards for MCoT..."
    python create_mcot_coco_shards.py \
        --data-dir data/coco \
        --output-dir data/coco_mcot_shards \
        --samples-per-shard 1000 \
        --max-train-samples 100000 \
        --max-val-samples 5000 \
        --seed 42
fi

echo "COCO data preparation for MCoT completed."
echo "The data is ready for use with the MCoT training pipeline." 