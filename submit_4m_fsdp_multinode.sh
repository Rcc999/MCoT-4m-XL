#!/bin/bash
#SBATCH --job-name=4m_mcot_fsdp_train
#SBATCH --time=5:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G             
#SBATCH --output=4m_mcot_fsdp_train_%j.out 
#SBATCH --error=4m_mcot_fsdp_train_%j.err 

CONFIG_FILE=$1
WANDB_KEY=$2
WANDB_ENTITY=${3:-as8148-epfl}

set -x 
cat $0 
export MASTER_PORT=25678 
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) 
export WANDB_API_KEY=$WANDB_KEY
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1 

echo "Arguments provided: CONFIG_FILE=$CONFIG_FILE, WANDB_ENTITY=$WANDB_ENTITY"
echo "Master Addr: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Node list: $SLURM_JOB_NODELIST"

export CONFIG_FILE="$CONFIG_FILE"
export WANDB_ENTITY="$WANDB_ENTITY"

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
  cd \"/work/com-304/MCoT-4m-XL-1\"

  torchrun \${TORCHRUN_ARGS} run_training_4m_mcot_fsdp.py \
    --config \${CONFIG_FILE} \
    --log_wandb \
    --wandb_entity \${WANDB_ENTITY} \
    --num_workers 5


" 
