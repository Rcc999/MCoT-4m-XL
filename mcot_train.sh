#!/bin/bash
#SBATCH --job-name=mcot_fsdp_multinode_train
#SBATCH --time=12:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/mcot_fsdp_train_%j.out
#SBATCH --error=logs/mcot_fsdp_train_%j.err

# === Accept arguments ===
# Argument 1: Path to the main MCOT YAML config file
CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
  echo "Error: CONFIG_FILE (argument 1) is required."
  exit 1
fi

# Hardcoded WandB settings
WANDB_KEY="c80687eb51acc4024f6907e16bcf29fd0f9862c1"
WANDB_ENTITY="rayane-charifchefchaouni-epfl"
WANDB_PROJECT="MCoT-4M"

# === Initialization ===
mkdir -p logs
set -x
cat $0

export MASTER_PORT=${MASTER_PORT:-25679}
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WANDB_API_KEY=$WANDB_KEY
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export OMP_NUM_THREADS=1

echo "MCOT Training Job"
echo "Config File: $CONFIG_FILE"
echo "WandB Entity: $WANDB_ENTITY"
echo "WandB Project: $WANDB_PROJECT"
echo "Master Addr: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Node list: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Pass necessary variables into the environment of the srun command
export CONFIG_FILE_SRUN="$CONFIG_FILE"
export WANDB_ENTITY_SRUN="$WANDB_ENTITY"
export WANDB_PROJECT_SRUN="$WANDB_PROJECT"
export DATA_ROOT_SRUN="/work/com-304/my_mscoco_for_4m"
export OUTPUT_DIR_SRUN="./outputs/mcot_fsdp_run_${SLURM_JOB_ID}"

# === Run main script ===
srun --label bash -c '
  # These variables are inherited or set from the outer script
  # MASTER_ADDR, MASTER_PORT, SLURM_PROCID, SLURM_NNODES are set by SLURM/srun or outer script

  # Construct torchrun arguments
  TORCHRUN_ARGS="--node-rank=${SLURM_PROCID} \
     --master-addr=${MASTER_ADDR} \
     --master-port=${MASTER_PORT} \
     --nnodes=${SLURM_NNODES} \
     --nproc-per-node=${SLURM_GPUS_ON_NODE:-2}"

  echo "--- Inside srun on $(hostname) ---"
  echo "SLURM_PROCID (Rank): ${SLURM_PROCID}"
  echo "Torchrun args: ${TORCHRUN_ARGS}"
  echo "Master Addr: ${MASTER_ADDR}"
  echo "Master Port: ${MASTER_PORT}"
  echo "NNodes: ${SLURM_NNODES}"
  echo "Config File (from env): ${CONFIG_FILE_SRUN}"
  echo "WandB Entity (from env): ${WANDB_ENTITY_SRUN}"
  echo "WandB Project (from env): ${WANDB_PROJECT_SRUN}"
  echo "Data Root (from env): ${DATA_ROOT_SRUN}"
  echo "Output Dir (from env): ${OUTPUT_DIR_SRUN}"
  echo "------------------------------------"

  # Test WandB login before running (only on rank 0)
  if [ "${SLURM_PROCID}" = "0" ]; then
    echo "Testing WandB login with API key..."
    python3 -c "import wandb; wandb.login()" || {
      echo "WandB login failed! Check your API key and permissions."
      echo "Continuing without WandB logging..."
      export USE_WANDB="0"
    }
  fi

  # Activate conda environment
  source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
  conda activate fourm

  # Change to the project directory
  cd "/home/rcharif/MCoT-4m-XL/"

  # Create output directory for this specific job
  mkdir -p ${OUTPUT_DIR_SRUN}
  
  # Add project root and scripts directories to PYTHONPATH to fix import issues
  export PYTHONPATH="/home/rcharif/MCoT-4m-XL:${PYTHONPATH}"

  # Create an empty __init__.py file in the scripts directory if it doesn't exist
  touch /home/rcharif/MCoT-4m-XL/scripts/__init__.py
  
  echo "PYTHONPATH set to: ${PYTHONPATH}"
  echo "Checking if mcot_data_utils.py exists:"
  ls -la /home/rcharif/MCoT-4m-XL/scripts/

  # Launch training with proper WandB parameters 
  WANDB_ARGS=""
  if [ "${USE_WANDB}" != "0" ]; then
    WANDB_ARGS="--log_wandb --wandb_project ${WANDB_PROJECT_SRUN} --wandb_entity ${WANDB_ENTITY_SRUN}"
    echo "Using WandB args: ${WANDB_ARGS}"
  else
    WANDB_ARGS="--no_log_wandb"
    echo "Disabling WandB logging due to previous login failure"
  fi

  # Launch training - IMPORTANT: Keep all arguments on a single line
  torchrun ${TORCHRUN_ARGS} run_training_mcot.py --config "${CONFIG_FILE_SRUN}" --data_root "${DATA_ROOT_SRUN}" --output_dir "${OUTPUT_DIR_SRUN}" --batch_size 2 --epochs 3 --num_workers 6 ${WANDB_ARGS} --dist_url env://
' 