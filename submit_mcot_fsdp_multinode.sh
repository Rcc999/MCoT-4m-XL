#!/bin/bash
#SBATCH --job-name=mcot_fsdp_train
#SBATCH --time=9:00:00 # Adjust as needed
#SBATCH --account=com-304 # Keep user's account
#SBATCH --qos=com-304 # Keep user's qos
#SBATCH --gres=gpu:2 # Keep as per original
#SBATCH --nodes=2 # Keep as per original
#SBATCH --ntasks-per-node=1 # Keep as per original
#SBATCH --cpus-per-task=4 # Keep as per original
#SBATCH --mem=128G # Keep as per original
#SBATCH --output=mcot_fsdp_train_%j.out
#SBATCH --error=mcot_fsdp_train_%j.err

# === Accept arguments ===
# Argument 1: Path to the main MCOT YAML config file
CONFIG_FILE=$1
# Argument 2: WandB API Key
WANDB_KEY=$2
# Argument 3 (Optional): WandB Entity (Defaults to user's EPFL ID if common, or a placeholder)
WANDB_ENTITY=${3:-rayane-charifchefchouni-epfl} # Updated with your WandB entity

# === Initialization ===
set -x # Print commands before executing them
cat $0 # Print the script itself to the output log
export MASTER_PORT=25679 # Changed port slightly to avoid conflict if old jobs run
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WANDB_API_KEY=$WANDB_KEY
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1

echo "MCOT Training Job"
echo "Arguments provided: CONFIG_FILE=$CONFIG_FILE, WANDB_ENTITY=$WANDB_ENTITY"
echo "Master Addr: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Node list: $SLURM_JOB_NODELIST"

# Pass the CONFIG_FILE and WANDB_ENTITY variables into the environment of the srun command
export CONFIG_FILE="$CONFIG_FILE"
export WANDB_ENTITY="$WANDB_ENTITY"

# === Run main script ===
srun --label bash -c "
  export TORCHRUN_ARGS=\\"--node-rank=\\${SLURM_PROCID} \\
     --master-addr=\\${MASTER_ADDR} \\
     --master-port=\\${MASTER_PORT} \\
     --nnodes=\\${SLURM_NNODES} \\
     --nproc-per-node=2\\" # Matches --gres=gpu:2

  echo \\"SLURM PROCID: \\${SLURM_PROCID}\\"
  echo \\"Torchrun args: \\${TORCHRUN_ARGS}\\"
  echo \\"Hostname: \\\$(hostname)\\"
  echo \\"Node ID: \\${SLURM_NODEID}\\"

  # Activate conda environment
  source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
  conda activate fourm

  # Change to the project directory
  cd \\"/home/rcharif/MCoT-4m-XL/\\" # User's project directory

  # Ensure data_root is available, e.g., by setting it as an env var if not in YAML
  # Or by passing it as an argument to the training script.
  # The training script run_training_mcot.py now accepts a --data_root argument.
  DATA_ROOT_ARG=\\"/work/com-304/my_mscoco_for_4m\\" # Define data_root here

  torchrun \\${TORCHRUN_ARGS} run_training_mcot.py \\
    --config \\${CONFIG_FILE} \\
    --data_root \\${DATA_ROOT_ARG} \\
    --log_wandb \\
    --wandb_entity \\${WANDB_ENTITY} \\
    --num_workers 5 # DataLoader workers


" 