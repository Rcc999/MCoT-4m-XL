#!/bin/bash
# Script to test MCoT inference with a HuggingFace model
set -e  # Exit immediately if a command fails

echo "=== MCoT HuggingFace Inference Test ==="
echo "Checking for required dependencies..."

# Check for Python
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check for required Python packages
echo "Checking for required Python packages..."
python -c "import torch; import tokenizers; import huggingface_hub; import PIL; print('All required packages are installed.')" || {
    echo "Error: Some required Python packages are missing."
    echo "Please install the required packages with:"
    echo "pip install torch tokenizers huggingface_hub pillow matplotlib"
    exit 1
}

# Ensure the tokenizer is prepared with MCoT tokens
if [ ! -f "tokenizers/mcot_tokenizer.json" ]; then
    echo "Extending tokenizer with MCoT tokens..."
    mkdir -p tokenizers
    python extend_tokenizer_for_mcot.py
else
    echo "Using existing MCoT tokenizer at tokenizers/mcot_tokenizer.json"
fi

# Make sure the results directory exists
mkdir -p results

# Path to a sample image
# Replace with your own image path
IMAGE_PATH="data/sample_image.jpg"

# If no sample image is available, try to download one from the web
if [ ! -f "$IMAGE_PATH" ]; then
    echo "No sample image found, downloading one..."
    mkdir -p data
    # Download a sample image from COCO
    wget -O $IMAGE_PATH "http://images.cocodataset.org/val2017/000000000139.jpg"
else
    echo "Using existing sample image at $IMAGE_PATH"
fi

# Run inference directly with the HuggingFace model
echo ""
echo "=== Running MCoT inference with the 4M-XL-21 model from HuggingFace ==="
echo "This may take a few minutes for the first run as the model needs to be downloaded..."
echo ""

python run_mcot_inference.py \
    --model-path "EPFL-VILAB/4M-21_XL" \
    --image-path "$IMAGE_PATH" \
    --output-dir results \
    --verbose

echo ""
echo "=== Inference complete ==="
echo "Check the results directory for outputs:"
echo "  - PNG visualization: results/mcot_result_*.png"
echo "  - JSON data: results/mcot_result_*.json" 