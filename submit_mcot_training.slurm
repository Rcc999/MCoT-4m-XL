#!/bin/bash
#SBATCH --job-name=mcot_training
#SBATCH --time=48:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=mcot_training_%j.out
#SBATCH --error=mcot_training_%j.err

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load modules and activate environment (adjust as needed for your cluster)
echo "Loading modules and activating environment..."
source ~/.bashrc

# Confirm the loaded environment
which python
python --version
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

# Prepare tokenizer and dataset if needed
echo "Preparing tokenizer with MCoT tokens..."
if [ ! -f "tokenizers/mcot_tokenizer.json" ]; then
    mkdir -p tokenizers
    python extend_tokenizer_for_mcot.py
fi

# Check if dataset already exists
if [ ! -d "data/coco_mcot_shards" ]; then
    echo "Preparing COCO dataset for MCoT..."
    bash prepare_mcot_coco_data.sh
else
    echo "COCO dataset already prepared at data/coco_mcot_shards"
fi

# Run the training script
echo "Starting MCoT training..."
bash train_mcot_model.sh

# Print job end time
echo "Job end time: $(date)" 