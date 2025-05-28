#!/bin/bash
#SBATCH --job-name=vqav2_train_multi
#SBATCH --time=12:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:2          # Requesting 2 GPUs per node
#SBATCH --nodes=2             # Requesting 2 nodes
#SBATCH --ntasks-per-node=1   # Typically 1 task per node when using torchrun for multi-GPU
#SBATCH --cpus-per-task=8     # Increased CPUs per task for 2 GPUs (e.g., 4 per GPU)
#SBATCH --mem=64G             # Memory per node
#SBATCH --output=vqav2_multi_node_%j.out
#SBATCH --error=vqav2_multi_node_%j.err

# === Accept arguments ===
WANDB_KEY=$1        # First argument: WandB API Key

# === Initialization ===
set -e # Exit immediately if a command exits with a non-zero status.
set -x # Print commands and their arguments as they are executed.
cat $0 # Print the script itself to the output for records

# Environment variables for torchrun
export MASTER_PORT=29500 # Same port as used in single-node torchrun
# Get the hostname of the first node in the allocation to act as master
# MASTER_ADDR will be automatically set by Slurm on the first node if using srun correctly,
# but we can also explicitly set it by getting the first node from SLURM_NODELIST.
# However, a more robust way for srun is often to let torchrun discover it via 'env://' if using --rdzv_backend=c10d
# For this example, we'll get it from SLURM_NODELIST and assume the first node is the master.
# This requires SLURM_NODELIST to be correctly populated.
FIRST_NODE_IN_ALLOCATION=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$FIRST_NODE_IN_ALLOCATION

export WANDB_API_KEY=$WANDB_KEY
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONFAULTHANDLER=1 # For better tracebacks on segfaults

echo "--- SLURM INFO ---"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_PROCID: $SLURM_PROCID (should be set by srun within the bash -c)"
echo "SLURM_LOCALID: $SLURM_LOCALID (should be set by srun within the bash -c)"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "--- END SLURM INFO ---"

CONFIG_FILE="cfgs/default/4m/finetune/4m-xl_mod21_vqav2_finetune.yaml"
NPROC_PER_NODE=2 # Should match --gres=gpu:N

# === Run main script ===
# The srun command will execute the script on each allocated node.
# SLURM_PROCID within the srun context will give the rank among all tasks (0 to N_NODES*NTASKS_PER_NODE - 1).
# For torchrun, node_rank is 0 to N_NODES-1.
# If ntasks-per-node=1, then SLURM_PROCID is effectively the node_rank.

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
  CURRENT_NODE_RANK=\\$SLURM_PROCID 
  # PROCID is the global rank. If ntasks-per-node is 1, then PROCID is the node_rank.

  echo \"Running on node: \$(hostname), Slurm PROCID for this srun task: \\$SLURM_PROCID, calculated NODE_RANK: \\$CURRENT_NODE_RANK\"

  # Ensure the MCoT-4m-XL directory is the current working directory
  # cd /home/rcharif/MCoT-4m-XL || { echo \'Failed to cd to MCoT-4m-XL\'; exit 1; }

  # Activate conda environment if needed - Assuming it's activated before submitting sbatch
  # source /path/to/your/conda/bin/activate fourm

  TORCHRUN_ARGS=\"--nnodes=\\$SLURM_NNODES \\
                 --nproc_per_node=$NPROC_PER_NODE \\
                 --node_rank=\\$CURRENT_NODE_RANK \\
                 --master_addr=\\$MASTER_ADDR \\
                 --master_port=\\$MASTER_PORT \\
                 --rdzv_backend=c10d\"
  
  echo \"Torchrun args: \\$TORCHRUN_ARGS\"

  # Ensure PYTORCH_CUDA_ALLOC_CONF is seen by torchrun processes
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  torchrun \\$TORCHRUN_ARGS run_training_4m_fsdp.py \\
    --config $CONFIG_FILE \\
    --log_wandb
"

echo "Slurm job finished." 