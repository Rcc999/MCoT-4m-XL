#!/bin/bash
# Simplified MCoT training script for testing
set -e

export CONFIG_FILE="cfgs/mcot_config.yaml"
export OUTPUT_DIR="outputs/mcot_test"
mkdir -p $OUTPUT_DIR

echo "Starting MCoT test training..."
torchrun \
    --nproc_per_node=1 \
    run_training_4m.py \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size 2 \
    --accum_iter 2 \
    --epochs 1 \
    --blr 3e-5 \
    --num_workers 4 \
    --log_wandb \
    --wandb_entity "rayane-charifchefchaouni-epfl"

echo "MCoT test training completed!"
