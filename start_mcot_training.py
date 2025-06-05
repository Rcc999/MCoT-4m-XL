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
    
    # Training arguments for MCoT - Production Configuration
    cmd = [
        sys.executable, "run_training_4m_mcot_fsdp.py",
        "--model", "fm_base_12e_12d_swiglu_nobias",
        "--batch_size", "32",
        "--epochs", "50",
        "--accum_iter", "4",
        "--blr", "5e-5",  # Base learning rate optimized for MCoT fine-tuning
        "--min_blr", "1e-6",
        "--warmup_epochs", "3",
        "--cooldown_epochs", "5",
        "--weight_decay", "0.1",
        "--clip_grad", "1.0",
        "--data_config", "cfgs/default/4m/data/mcot/mcot.yaml",
        "--output_dir", "./output/mcot_training",
        "--device", "cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu",
        "--dtype", "bfloat16",
        "--use_act_checkpoint",
        
        # MCoT specific parameters with optimized weights
        "--mcot_steps", "planning,acting,reflection,correction",
        "--mcot_planning_weight", "1.0",  # Base weight for planning
        "--mcot_acting_weight", "1.2",   # Slightly higher for generation quality
        "--mcot_reflection_weight", "1.5", # Higher weight for reflection learning
        "--mcot_correction_weight", "1.3", # Higher weight for correction accuracy
        "--enable_mint_features",  # Enable MINT paper features
        
        # Optimized token budgets for MCoT
        "--num_input_tokens", "512",   # Larger context for complex reasoning
        "--num_target_tokens", "512",  # Sufficient for detailed outputs
        "--min_input_tokens", "256",
        "--min_target_tokens", "256",
        
        # Training optimization
        "--loss_type", "mod",
        "--eval_freq", "5",
        "--save_ckpt_freq", "10",
        "--seed", "42",
        "--num_workers", "8",
        "--pin_mem",
        "--dist_eval",
        "--fixed_eval",
        
        # Learning rate scheduling
        "--scheduler", "cosine",
        "--opt", "adamw",
        "--opt_betas", "0.9", "0.95",
        "--opt_eps", "1e-8",
        
        # Mixed precision and memory optimization
        "--skip_nan_grad",
        
        # Logging
        "--print_all"
    ]
    
    print("Starting MCoT production training with command:")
    print(" ".join(cmd))
    print()
    print("🎯 Training Configuration:")
    print(f"   • Model: fm_base_12e_12d_swiglu_nobias")
    print(f"   • Batch size: 32 (effective: 128 with accumulation)")
    print(f"   • Epochs: 50")
    print(f"   • Learning rate: 5e-5 → 1e-6 (cosine schedule)")
    print(f"   • Token budget: 512 input/target tokens")
    print(f"   • MCoT steps: planning, acting, reflection, correction")
    print(f"   • MINT features: enabled")
    print(f"   • Mixed precision: bfloat16")
    print()
    
    # Create output directories
    Path("./output/mcot_training").mkdir(parents=True, exist_ok=True)
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True)
        print("MCoT production training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"MCoT production training failed with error: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("MCoT training interrupted by user")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
