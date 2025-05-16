#!/usr/bin/env python
"""
Script to monitor the training progress of an MCoT model.

This script checks:
1. Existence and timestamps of model checkpoints
2. WandB logs for loss trends
3. GPU utilization and memory usage

Usage:
python monitor_training.py --checkpoint-dir /path/to/checkpoints --wandb-name name_of_run
"""

import os
import time
import glob
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' to enable WandB monitoring.")

def parse_args():
    parser = argparse.ArgumentParser(description="Monitor MCoT model training progress")
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs",
                        help="Directory containing model checkpoints")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="WandB run name to monitor (if using WandB)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="WandB entity name (if using WandB)")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="WandB project name (if using WandB)")
    parser.add_argument("--interval", type=int, default=300,
                        help="Monitoring interval in seconds")
    parser.add_argument("--output-dir", type=str, default="./monitoring",
                        help="Directory to save monitoring outputs")
    return parser.parse_args()

def check_checkpoints(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """Check for model checkpoints and their timestamps."""
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint*"))
    
    checkpoints = []
    for path in checkpoint_paths:
        stats = os.stat(path)
        checkpoints.append({
            "path": path,
            "size_mb": stats.st_size / (1024 * 1024),
            "timestamp": stats.st_mtime,
            "age": datetime.now().timestamp() - stats.st_mtime
        })
    
    # Sort by timestamp (newest first)
    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return checkpoints

def get_gpu_stats() -> List[Dict[str, Any]]:
    """Get GPU utilization and memory usage stats."""
    try:
        # Run nvidia-smi to get GPU stats
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", 
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True
        )
        
        gpu_stats = []
        for line in result.stdout.strip().split("\n"):
            values = line.split(", ")
            if len(values) >= 5:
                gpu_stats.append({
                    "index": int(values[0]),
                    "name": values[1],
                    "utilization_pct": float(values[2]),
                    "memory_used_mb": float(values[3]),
                    "memory_total_mb": float(values[4]),
                    "memory_pct": float(values[3]) / float(values[4]) * 100 if float(values[4]) > 0 else 0
                })
        
        return gpu_stats
    except subprocess.CalledProcessError:
        print("Error: Failed to get GPU stats. Is nvidia-smi available?")
        return []
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return []

def check_wandb_logs(wandb_name: str, wandb_entity: str, wandb_project: str) -> Optional[Dict[str, Any]]:
    """Check WandB logs for loss trends."""
    if not WANDB_AVAILABLE or not wandb_name:
        return None
    
    try:
        api = wandb.Api()
        
        # Construct the run path
        if wandb_entity and wandb_project:
            run_path = f"{wandb_entity}/{wandb_project}/{wandb_name}"
        else:
            # Try to guess based on the WandB API's available runs
            matching_runs = []
            for run in api.runs():
                if run.name == wandb_name:
                    matching_runs.append(run)
            
            if not matching_runs:
                print(f"Error: No WandB run found with name {wandb_name}")
                return None
            
            if len(matching_runs) > 1:
                print(f"Warning: Multiple WandB runs found with name {wandb_name}. Using the first one.")
            
            run = matching_runs[0]
        
        # Get the run
        run = api.run(run_path) if 'run_path' in locals() else matching_runs[0]
        
        # Get the run's history
        history = run.history()
        
        # Extract relevant metrics
        metrics = {}
        for col in history.columns:
            if 'loss' in col.lower():
                metrics[col] = history[col].tolist()
        
        return {
            "summary": run.summary._json_dict,
            "metrics": metrics
        }
    except Exception as e:
        print(f"Error accessing WandB logs: {e}")
        return None

def plot_training_progress(
    checkpoints: List[Dict[str, Any]], 
    wandb_data: Optional[Dict[str, Any]],
    output_path: str
):
    """Plot training progress including checkpoint creation and loss trends."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot checkpoint creation times
    if checkpoints:
        timestamps = [cp["timestamp"] for cp in checkpoints]
        checkpoint_dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        checkpoint_sizes = [cp["size_mb"] for cp in checkpoints]
        
        # Reverse to show chronological order
        checkpoint_dates.reverse()
        checkpoint_sizes.reverse()
        
        ax1.plot(checkpoint_dates, list(range(1, len(checkpoint_dates) + 1)), 'b-o')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Checkpoint Count')
        ax1.set_title('Checkpoint Creation Progress')
        ax1.grid(True)
        
        # Annotate with checkpoint sizes
        for i, (date, size) in enumerate(zip(checkpoint_dates, checkpoint_sizes)):
            ax1.annotate(f"{size:.1f}MB", (date, i + 1), textcoords="offset points", 
                        xytext=(0, 10), ha='center')
    else:
        ax1.text(0.5, 0.5, 'No checkpoints found', horizontalalignment='center',
                verticalalignment='center', transform=ax1.transAxes)
    
    # Plot loss trends from WandB if available
    if wandb_data and 'metrics' in wandb_data and wandb_data['metrics']:
        # Find the first loss metric
        loss_key = next((k for k in wandb_data['metrics'].keys() if 'loss' in k.lower()), None)
        
        if loss_key and wandb_data['metrics'][loss_key]:
            loss_values = wandb_data['metrics'][loss_key]
            ax2.plot(loss_values, 'r-')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Loss')
            ax2.set_title(f'Training Loss: {loss_key}')
            ax2.grid(True)
            
            # Add moving average
            window_size = min(10, len(loss_values))
            if window_size > 1:
                moving_avg = np.convolve(loss_values, np.ones(window_size)/window_size, mode='valid')
                ax2.plot(range(window_size-1, len(loss_values)), moving_avg, 'g-', label=f'{window_size}-step Moving Avg')
                ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No loss metrics found in WandB data', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'WandB data not available', 
                horizontalalignment='center', verticalalignment='center', 
                transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting monitoring with {args.interval}s interval. Press Ctrl+C to stop.")
    print(f"Monitoring outputs will be saved to {args.output_dir}")
    
    try:
        while True:
            # Get current timestamp
            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            
            print(f"\n=== Monitoring update at {now.strftime('%Y-%m-%d %H:%M:%S')} ===")
            
            # Check for checkpoints
            checkpoints = check_checkpoints(args.checkpoint_dir)
            if checkpoints:
                print(f"Found {len(checkpoints)} checkpoints.")
                
                if checkpoints:
                    latest = checkpoints[0]
                    latest_time = datetime.fromtimestamp(latest["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Latest checkpoint: {latest['path']} ({latest_time}, {latest['size_mb']:.2f}MB, {timedelta(seconds=int(latest['age']))} ago)")
            else:
                print("No checkpoints found.")
            
            # Get GPU stats
            gpu_stats = get_gpu_stats()
            if gpu_stats:
                print("\nGPU Stats:")
                for gpu in gpu_stats:
                    print(f"GPU {gpu['index']} ({gpu['name']}): {gpu['utilization_pct']:.1f}% util, "
                         f"{gpu['memory_used_mb']:.0f}/{gpu['memory_total_mb']:.0f}MB ({gpu['memory_pct']:.1f}%)")
            
            # Check WandB logs
            wandb_data = None
            if WANDB_AVAILABLE and args.wandb_name:
                wandb_data = check_wandb_logs(args.wandb_name, args.wandb_entity, args.wandb_project)
                if wandb_data:
                    print("\nWandB Run Summary:")
                    # Print some key metrics if available
                    summary = wandb_data['summary']
                    if 'latest_loss' in summary:
                        print(f"Latest loss: {summary['latest_loss']:.4f}")
                    elif any('loss' in k.lower() for k in summary.keys()):
                        loss_keys = [k for k in summary.keys() if 'loss' in k.lower()]
                        for key in loss_keys:
                            print(f"{key}: {summary[key]:.4f}")
            
            # Plot training progress
            plot_path = os.path.join(args.output_dir, f"training_progress_{timestamp_str}.png")
            plot_training_progress(checkpoints, wandb_data, plot_path)
            print(f"Training progress plot saved to {plot_path}")
            
            # Save monitoring data
            monitoring_data = {
                "timestamp": now.timestamp(),
                "checkpoints": checkpoints,
                "gpu_stats": gpu_stats
            }
            
            json_path = os.path.join(args.output_dir, f"monitoring_data_{timestamp_str}.json")
            with open(json_path, "w") as f:
                json.dump(monitoring_data, f, indent=2)
            
            # Wait for next interval
            print(f"\nNext update in {args.interval} seconds...")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    main() 