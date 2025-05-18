#!/bin/bash
#SBATCH --job-name=mcot_fsdp_multinode_train # Descriptive job name
#SBATCH --time=12:00:00 # Adjusted time, can be further tuned
#SBATCH --account=com-304 # User's account
#SBATCH --qos=com-304 # User's qos
#SBATCH --gres=gpu:2 # GPUs per node
#SBATCH --nodes=2 # Number of nodes
#SBATCH --ntasks-per-node=1 # One task (srun bash -c ...) per node
#SBATCH --cpus-per-task=16  # Increased CPUs: (2 main torchrun procs * ~2 CPUs) + (2 GPU procs * 6 workers_per_gpu_proc) = 4 + 12 = 16
#SBATCH --mem=128G # Increased memory
#SBATCH --output=logs/mcot_fsdp_train_%j.out # Log to a 'logs' subdirectory
#SBATCH --error=logs/mcot_fsdp_train_%j.err  # Log to a 'logs' subdirectory

# === Accept arguments ===
# Argument 1: Path to the main MCOT YAML config file
CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
  echo "Error: CONFIG_FILE (argument 1) is required."
  exit 1
fi

# Argument 2: WandB API Key
WANDB_KEY=$2
if [ -z "$WANDB_KEY" ]; then
  echo "Error: WANDB_KEY (argument 2) is required."
  exit 1
fi

# Argument 3 (Optional): WandB Entity
WANDB_ENTITY=${3:-rayane-charifchefchaouni} # User's WandB entity or default

# === Initialization ===
mkdir -p logs # Create logs directory if it doesn't exist
set -x # Print commands before executing them
cat $0 # Print the script itself to the output log

export MASTER_PORT=${MASTER_PORT:-25679} # Use environment MASTER_PORT or default
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WANDB_API_KEY=$WANDB_KEY
export NCCL_DEBUG=${NCCL_DEBUG:-INFO} # Use environment NCCL_DEBUG or default to INFO
export OMP_NUM_THREADS=1 # Recommended for PyTorch multiprocessing

echo "MCOT Training Job"
echo "Config File: $CONFIG_FILE"
echo "WandB Entity: $WANDB_ENTITY"
echo "Master Addr: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Node list: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Pass necessary variables into the environment of the srun command
export CONFIG_FILE_SRUN="$CONFIG_FILE"
export WANDB_ENTITY_SRUN="$WANDB_ENTITY"
export DATA_ROOT_SRUN="/work/com-304/my_mscoco_for_4m" # Define data_root here
export OUTPUT_DIR_SRUN="./outputs/mcot_fsdp_run_${SLURM_JOB_ID}" # Unique output dir per job

# === Run main script ===
srun --label bash -c '
  # These variables are inherited or set from the outer script
  # MASTER_ADDR, MASTER_PORT, SLURM_PROCID, SLURM_NNODES are set by SLURM/srun or outer script

  # Construct torchrun arguments
  # Using SLURM_PROCID for node_rank is correct as SLURM_PROCID is unique across all tasks in the job.
  # Since ntasks-per-node=1, SLURM_PROCID will effectively be the node ID (0 for first node, 1 for second, etc.)
  TORCHRUN_ARGS="--node-rank=${SLURM_PROCID} \
     --master-addr=${MASTER_ADDR} \
     --master-port=${MASTER_PORT} \
     --nnodes=${SLURM_NNODES} \
     --nproc-per-node=${SLURM_GPUS_ON_NODE:-2}" # Use SLURM_GPUS_ON_NODE if available, else default to gres

  echo "--- Inside srun on $(hostname) ---"
  echo "SLURM_PROCID (Rank): ${SLURM_PROCID}"
  echo "Torchrun args: ${TORCHRUN_ARGS}"
  echo "Master Addr: ${MASTER_ADDR}"
  echo "Master Port: ${MASTER_PORT}"
  echo "NNodes: ${SLURM_NNODES}"
  echo "Config File (from env): ${CONFIG_FILE_SRUN}"
  echo "WandB Entity (from env): ${WANDB_ENTITY_SRUN}"
  echo "Data Root (from env): ${DATA_ROOT_SRUN}"
  echo "Output Dir (from env): ${OUTPUT_DIR_SRUN}"
  echo "------------------------------------"

  # Activate conda environment
  source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
  conda activate fourm

  # Change to the project directory
  cd "/home/rcharif/MCoT-4m-XL/" # User's project directory

  # Create output directory for this specific job
  mkdir -p ${OUTPUT_DIR_SRUN}

  # Launch training
  torchrun ${TORCHRUN_ARGS} run_training_mcot.py \
    --config "${CONFIG_FILE_SRUN}" \
    --data_root "${DATA_ROOT_SRUN}" \
    --output_dir "${OUTPUT_DIR_SRUN}" \
    --batch_size 2   # Per-GPU batch size \
    --epochs 3       # Changed to 3 epochs \
    --num_workers 6 # Changed to 6 DataLoader workers: (16 cpus - 2*2 main_procs_cpus) / 2 gpu_procs = 6 \
    --log_wandb \
    --wandb_entity "${WANDB_ENTITY_SRUN}" \
    --dist_url env://  # Important for PyTorch to use SLURM variables
    # Add other arguments for run_training_mcot.py as needed, e.g., --save_ckpt_freq
' 