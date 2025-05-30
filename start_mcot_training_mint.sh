#!/bin/bash

# Enhanced 4M MCoT Training with MINT Paper Features
# Based on proven FSDP training pipeline

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your available GPUs
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training configuration
MODEL="fm_base_12e_12d_swiglu_nobias"
BATCH_SIZE=8  # Adjust based on GPU memory
EPOCHS=50
ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4

# Data configuration
DATA_CONFIG="configs/mcot_data_config.json"
DATASET_DIRS="data/"  # Directory containing your MCoT training data

# Output configuration
OUTPUT_DIR="output/mcot_training_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/mcot_training_$(date +%Y%m%d_%H%M%S)"

# MCoT specific configuration
MCOT_STEPS="planning,acting,reflection,correction"
PLANNING_WEIGHT=1.0
ACTING_WEIGHT=1.0
REFLECTION_WEIGHT=1.0
CORRECTION_WEIGHT=1.0

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p configs

echo "Starting 4M MCoT Training with MINT Features..."
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE per GPU"
echo "Epochs: $EPOCHS"
echo "Output directory: $OUTPUT_DIR"
echo "Data config: $DATA_CONFIG"
echo "MCoT steps: $MCOT_STEPS"

# Check if data config exists
if [ ! -f "$DATA_CONFIG" ]; then
    echo "Error: Data config file not found at $DATA_CONFIG"
    echo "Please ensure the data configuration file exists."
    exit 1
fi

# Check if dataset directory exists
if [ ! -d "data/" ]; then
    echo "Warning: Data directory 'data/' not found."
    echo "Please ensure your MCoT training data is available."
fi

# Launch training with FSDP (supports multi-GPU)
python run_training_4m_mcot_fsdp.py \
    --model "$MODEL" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --accum_iter "$ACCUMULATION_STEPS" \
    --lr "$LEARNING_RATE" \
    --data_config "$DATA_CONFIG" \
    --dataset_dirs "$DATASET_DIRS" \
    --output_dir "$OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --mcot_steps "$MCOT_STEPS" \
    --mcot_planning_weight "$PLANNING_WEIGHT" \
    --mcot_acting_weight "$ACTING_WEIGHT" \
    --mcot_reflection_weight "$REFLECTION_WEIGHT" \
    --mcot_correction_weight "$CORRECTION_WEIGHT" \
    --enable_mint_features \
    --dtype bfloat16 \
    --use_act_checkpoint \
    --clip_grad 1.0 \
    --weight_decay 0.05 \
    --warmup_epochs 5 \
    --save_ckpt_freq 10 \
    --wandb_project "4m-mcot-mint" \
    --run_name "mcot_training_$(date +%Y%m%d_%H%M%S)" \
    --num_workers 8 \
    --pin_mem

echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"
