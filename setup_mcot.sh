#!/bin/bash
# Setup script for MCOT environment

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed or not in your PATH. Please install Python first."
    exit 1
fi

# Create necessary directories
mkdir -p ckpt datasets

# Install required packages
echo "Installing required packages..."
pip install PyYAML requests tqdm torch torchvision safetensors

# Download datasets
echo "Downloading datasets (this may take a while)..."
python download_datasets.py

# Update configuration paths
echo "Updating configuration paths..."
python update_paths.py

echo ""
echo "Setup complete!"
echo "----------------------------------------------------------"
echo "Please place your model checkpoint at: ckpt/mcotmodel.safetensors"
echo ""
echo "Next steps:"
echo "1. Run: python extend_vocabulary.py --checkpoint-path ckpt/mcotmodel.safetensors --output-checkpoint-path ckpt/mcotmodel.safetensors"
echo "2. Run: python run_training_4m.py --config cfgs/mcot_vqa_finetune.yaml --finetune ckpt/mcotmodel.safetensors --output_dir ./output/mcot_vqa_finetune"
echo "3. Run: python run_training_4m.py --config cfgs/mcot_post_training.yaml --finetune ./output/mcot_vqa_finetune/checkpoint-best.safetensors --output_dir ./output/mcot_post_training"
echo "4. Run: python run_mcot_evaluation.py --checkpoint ./output/mcot_post_training/checkpoint-best.safetensors --vqa-dataset datasets/vqav2/val --planning-dataset datasets/mscoco/val --reflection-dataset datasets/richhf18k/val --correction-dataset datasets/cocostuff/val --output-dir ./evaluation_results"
echo "----------------------------------------------------------" 