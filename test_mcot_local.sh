#!/bin/bash
# Local testing script without GPU

# Set config file path
export CONFIG_FILE="cfgs/mcot_data_config.yaml"

# Run a minimal local test
python run_training_4m.py \
  --local_debug \
  --data_config $CONFIG_FILE \
  --output_dir tmp_local_output \
  --local_batch_size 1 \
  --accum_iter 1 \
  --epochs 1 \
  --num_workers 0 \
  --device cpu
