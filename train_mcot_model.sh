#!/bin/bash
# Training script for MCoT model using the 4M framework
# This script runs the MCoT training for Planning and Acting stages
set -e  # Exit immediately if a command fails

# Set environment variables
export CONFIG_FILE="cfgs/mcot_config.yaml"
export OUTPUT_DIR="outputs/mcot_training"
export MODEL_PATH="EPFL-VILAB/4M-21_XL"  # Using HuggingFace model ID for 4M-XL-21

# Set WandB logging if needed (optional)
export WANDB_API_KEY="c80687eb51acc4024f6907e16bcf29fd0f9862c1"  # Add your WandB API key if using WandB logging
export WANDB_ENTITY="rayane-charifchefchaouni-epfl"   # Add your WandB entity name if using WandB logging
export WANDB_PROJECT="mcot-4m-xl"

# Create output directory
mkdir -p $OUTPUT_DIR

# Step 1: Extend the tokenizer with MCoT special tokens (if needed)
if [ ! -f "tokenizers/mcot_tokenizer.json" ]; then
    echo "Extending tokenizer with MCoT special tokens..."
    mkdir -p tokenizers
    python extend_tokenizer_for_mcot.py \
        --input-tokenizer fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json \
        --output-path tokenizers/mcot_tokenizer.json
else
    echo "Using existing MCoT tokenizer at tokenizers/mcot_tokenizer.json"
fi

# Step 2: Prepare COCO dataset (if needed)
if [ ! -d "data/coco_mcot_shards" ]; then
    echo "Preparing COCO dataset for MCoT training..."
    bash prepare_mcot_coco_data.sh
else
    echo "Using existing COCO dataset at data/coco_mcot_shards"
fi

# Step 3: Run MCoT training
echo "Starting MCoT training..."
torchrun \
    --nproc_per_node=1 \
    run_training_4m.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size 32 \
    --accum_iter 2 \
    --epochs 20 \
    --blr 3e-5 \
    --warmup_epochs 2 \
    --model_path $MODEL_PATH \
    --use_lora true \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target_modules "query,value" \
    --num_workers 4 \
    --log_wandb

echo "MCoT training completed!"
echo "Model checkpoints are saved in: $OUTPUT_DIR" 