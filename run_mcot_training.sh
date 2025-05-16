#!/bin/bash
#SBATCH --job-name=mcot_training
#SBATCH --time=9:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=mcot_training_%j.out
#SBATCH --error=mcot_training_%j.err

# === Accept arguments ===
# Argument 1: WandB API Key (optional)
WANDB_KEY=$1
# Argument 2: WandB Entity (Defaults to as8148-epfl if not provided)
WANDB_ENTITY=$2

# === Initialization ===
set -x # Print commands before executing them
cat $0 # Print the script itself to the output log
export MASTER_PORT=25678 # Use a specific port
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) # Get master node hostname

# Setup WandB configuration
USE_WANDB="false"
if [ ! -z "$WANDB_KEY" ]; then
  export WANDB_API_KEY=$WANDB_KEY
  export WANDB_ENTITY=${WANDB_ENTITY:-"rayane-charifchefchaouni-epfl"}
  USE_WANDB="true"
else
  echo "No WandB API key provided, WandB logging will be disabled"
fi

export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1 # Recommended for multi-process torch

# Set the config file to our MCoT config
export CONFIG_FILE="cfgs/mcot_data_config.yaml"

echo "Config file: $CONFIG_FILE"
if [ "$USE_WANDB" = "true" ]; then
  echo "WandB logging enabled with entity: $WANDB_ENTITY"
else
  echo "WandB logging disabled"
fi
echo "Master Addr: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Node list: $SLURM_JOB_NODELIST"

# === Run main script ===
srun --label bash -c "
  export TORCHRUN_ARGS=\"--node-rank=\${SLURM_PROCID} \
     --master-addr=\${MASTER_ADDR} \
     --master-port=\${MASTER_PORT} \
     --nnodes=\${SLURM_NNODES} \
     --nproc-per-node=2\" # Set to number of GPUs per node requested

  echo \"SLURM PROCID: \${SLURM_PROCID}\"
  echo \"Torchrun args: \${TORCHRUN_ARGS}\"
  echo \"Hostname: \$(hostname)\"
  echo \"Node ID: \${SLURM_NODEID}\"

  # Activate conda environment
  source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
  conda activate fourm

  # Change to the project directory
  cd \"/home/rcharif/MCoT-4m-XL\"

  # Build command based on whether WandB is enabled
  CMD=\"torchrun \${TORCHRUN_ARGS} run_training_4m.py \\
    --data_config \${CONFIG_FILE} \\
    --output_dir outputs/mcot_training \\
    --batch_size 2 \\
    --accum_iter 2 \\
    --epochs 1 \\
    --blr 3e-5 \\
    --num_workers 5\"
  
  # Add WandB parameters if enabled
  if [ \"$USE_WANDB\" = \"true\" ]; then
    CMD=\"\${CMD} --log_wandb --wandb_entity \${WANDB_ENTITY}\"
  else
    CMD=\"\${CMD} --no_log_wandb\"
  fi
  
  # Execute the command
  eval \$CMD
" 