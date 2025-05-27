#!/usr/bin/env python3
"""
Script to start MCoT training with proper configuration.
This demonstrates how to run the complete MCoT fine-tuning pipeline.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    # Set up paths
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Training arguments for MCoT
    cmd = [
        sys.executable, "run_training_4m_mcot_clean.py",
        "--model", "fm_base_12e_12d_swiglu_nobias",
        "--batch_size", "4",  # Small batch size for testing
        "--epochs", "2",      # Short training for testing
        "--accum_iter", "2",
        "--blr", "1e-5",
        "--warmup_epochs", "1",
        "--weight_decay", "0.05",
        "--data_config", "cfgs/default/4m/data/mcot/mcot.yaml",
        "--output_dir", "./output/mcot_test",
        "--log_dir", "./logs/mcot_test",
        "--device", "cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu",
        "--dtype", "bfloat16",
        "--use_act_checkpoint",
        
        # MCoT specific parameters
        "--mcot_steps", "planning,acting,reflection,correction",
        "--mcot_planning_weight", "1.0",
        "--mcot_acting_weight", "1.0", 
        "--mcot_reflection_weight", "1.0",
        "--mcot_correction_weight", "1.0",
        
        # Token budgets
        "--num_input_tokens", "256",
        "--num_target_tokens", "256",
        "--min_input_tokens", "128",
        "--min_target_tokens", "128",
        
        # Other settings
        "--loss_type", "mod",
        "--eval_freq", "1",
        "--save_ckpt_freq", "1",
        "--seed", "42",
        "--num_workers", "2"
    ]
    
    print("Starting MCoT training with command:")
    print(" ".join(cmd))
    print()
    
    # Create output directories
    Path("./output/mcot_test").mkdir(parents=True, exist_ok=True)
    Path("./logs/mcot_test").mkdir(parents=True, exist_ok=True)
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True)
        print("MCoT training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"MCoT training failed with error: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
