#!/usr/bin/env python3
# Update configuration files to use the new dataset and model paths

import os
import glob
import yaml
from pathlib import Path
import re
import json

def update_yaml_files():
    """Update YAML configuration files to use the new paths"""
    print("Updating YAML configuration files...")
    
    # Get list of all YAML files in cfgs directory
    cfg_files = glob.glob("cfgs/*.yaml")
    
    for file_path in cfg_files:
        print(f"Processing {file_path}")
        
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update dataset paths
        if 'train' in config and 'datasets' in config['train']:
            for dataset_name, dataset_cfg in config['train']['datasets'].items():
                if 'data_path' in dataset_cfg:
                    # Update based on dataset type
                    if 'vqa' in dataset_name.lower():
                        dataset_cfg['data_path'] = 'datasets/vqav2'
                    elif 'coco' in dataset_name.lower() and 'stuff' not in dataset_name.lower():
                        dataset_cfg['data_path'] = 'datasets/mscoco'
                    elif 'rich' in dataset_name.lower() or 'artifact' in dataset_name.lower():
                        dataset_cfg['data_path'] = 'datasets/richhf18k'
                    elif 'stuff' in dataset_name.lower():
                        dataset_cfg['data_path'] = 'datasets/cocostuff'
        
        # Update validation dataset paths
        if 'val' in config and 'datasets' in config['val']:
            for dataset_name, dataset_cfg in config['val']['datasets'].items():
                if 'data_path' in dataset_cfg:
                    # Update based on dataset type
                    if 'vqa' in dataset_name.lower():
                        dataset_cfg['data_path'] = 'datasets/vqav2'
                    elif 'coco' in dataset_name.lower() and 'stuff' not in dataset_name.lower():
                        dataset_cfg['data_path'] = 'datasets/mscoco'
                    elif 'rich' in dataset_name.lower() or 'artifact' in dataset_name.lower():
                        dataset_cfg['data_path'] = 'datasets/richhf18k'
                    elif 'stuff' in dataset_name.lower():
                        dataset_cfg['data_path'] = 'datasets/cocostuff'
        
        # Write updated config back to file
        with open(file_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

def update_evaluation_script():
    """Update evaluation script to use the new paths"""
    print("Updating evaluation script...")
    
    eval_script = "run_mcot_evaluation.py"
    if not os.path.exists(eval_script):
        print(f"Warning: {eval_script} not found, skipping")
        return
    
    with open(eval_script, 'r') as f:
        content = f.read()
    
    # Update dataset paths
    content = re.sub(r'--vqa-dataset\s+[\'"]?([^\s\'"]+)[\'"]?', 
                    '--vqa-dataset "datasets/vqav2/val"', content)
    content = re.sub(r'--planning-dataset\s+[\'"]?([^\s\'"]+)[\'"]?', 
                    '--planning-dataset "datasets/mscoco/val"', content)
    content = re.sub(r'--reflection-dataset\s+[\'"]?([^\s\'"]+)[\'"]?', 
                    '--reflection-dataset "datasets/richhf18k/val"', content)
    content = re.sub(r'--correction-dataset\s+[\'"]?([^\s\'"]+)[\'"]?', 
                    '--correction-dataset "datasets/cocostuff/val"', content)
    
    # Update model checkpoint path
    content = re.sub(r'--checkpoint\s+[\'"]?([^\s\'"]+)[\'"]?', 
                    '--checkpoint "ckpt/mcotmodel.safetensors"', content)
    
    with open(eval_script, 'w') as f:
        f.write(content)

def update_readme():
    """Update README_MCOT.md to use the new paths"""
    print("Updating README_MCOT.md...")
    
    readme_path = "README_MCOT.md"
    if not os.path.exists(readme_path):
        print(f"Warning: {readme_path} not found, skipping")
        return
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    # Update paths in the code blocks
    content = re.sub(r'/path/to/4m-21-xl\.safetensors', 'ckpt/mcotmodel.safetensors', content)
    content = re.sub(r'/path/to/4m-21-xl-mcot\.safetensors', 'ckpt/mcotmodel.safetensors', content)
    content = re.sub(r'/path/to/vqav2/val', 'datasets/vqav2/val', content)
    content = re.sub(r'/path/to/mscoco/val', 'datasets/mscoco/val', content)
    content = re.sub(r'/path/to/richhf18k/val', 'datasets/richhf18k/val', content)
    content = re.sub(r'/path/to/cocostuff/val', 'datasets/cocostuff/val', content)
    
    with open(readme_path, 'w') as f:
        f.write(content)

def create_sample_config():
    """Create sample config files if they don't exist"""
    cfgs_dir = Path("cfgs")
    cfgs_dir.mkdir(exist_ok=True)
    
    # Create sample VQA finetune config
    vqa_config_path = cfgs_dir / "mcot_vqa_finetune.yaml"
    if not vqa_config_path.exists():
        vqa_config = {
            'train': {
                'datasets': {
                    'vqa': {
                        'data_path': 'datasets/vqav2',
                        'in_domains': 'img-text',
                        'out_domains': 'text'
                    }
                }
            },
            'val': {
                'datasets': {
                    'vqa_val': {
                        'data_path': 'datasets/vqav2/val',
                        'in_domains': 'img-text',
                        'out_domains': 'text'
                    }
                }
            },
            'model': 'fm_base_12e_12d_swiglu_nobias',
            'batch_size': 64,
            'epochs': 10,
            'lr': 5e-5
        }
        with open(vqa_config_path, 'w') as f:
            yaml.dump(vqa_config, f, default_flow_style=False)
    
    # Create sample MCOT post-training config
    mcot_config_path = cfgs_dir / "mcot_post_training.yaml"
    if not mcot_config_path.exists():
        mcot_config = {
            'train': {
                'datasets': {
                    'planning': {
                        'data_path': 'datasets/mscoco',
                        'in_domains': 'text',
                        'out_domains': 'img'
                    },
                    'reflection': {
                        'data_path': 'datasets/richhf18k',
                        'in_domains': 'img',
                        'out_domains': 'text-img'
                    },
                    'correction': {
                        'data_path': 'datasets/cocostuff',
                        'in_domains': 'img-text',
                        'out_domains': 'img'
                    }
                },
                'weights': [0.3, 0.3, 0.4]
            },
            'val': {
                'datasets': {
                    'planning_val': {
                        'data_path': 'datasets/mscoco/val',
                        'in_domains': 'text',
                        'out_domains': 'img'
                    },
                    'reflection_val': {
                        'data_path': 'datasets/richhf18k/val',
                        'in_domains': 'img',
                        'out_domains': 'text-img'
                    },
                    'correction_val': {
                        'data_path': 'datasets/cocostuff/val',
                        'in_domains': 'img-text',
                        'out_domains': 'img'
                    }
                }
            },
            'model': 'fm_base_12e_12d_swiglu_nobias',
            'batch_size': 32,
            'epochs': 5,
            'lr': 1e-5
        }
        with open(mcot_config_path, 'w') as f:
            yaml.dump(mcot_config, f, default_flow_style=False)

def main():
    print("Updating configuration files to use the new dataset and model paths")
    
    # Create ckpt directory if it doesn't exist
    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(exist_ok=True)
    
    # Create necessary config files if they don't exist
    create_sample_config()
    
    # Update configuration files
    update_yaml_files()
    update_evaluation_script()
    update_readme()
    
    print("\nAll configuration files updated!")
    print("Please ensure your model is located at: ckpt/mcotmodel.safetensors")
    print("Datasets will be downloaded to: datasets/")
    print("\nRun the download_datasets.py script to download all required datasets.")

if __name__ == "__main__":
    main() 