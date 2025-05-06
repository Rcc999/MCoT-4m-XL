#!/usr/bin/env python3
# Setup directory structure for MCOT datasets and model

import os
from pathlib import Path

# Create main directories
DATASETS_DIR = Path("datasets")
CKPT_DIR = Path("ckpt")

def setup_directories():
    """Create directory structure for datasets and model checkpoint"""
    print("Setting up directory structure for MCOT datasets and model...")
    
    # Create main directories
    DATASETS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    
    # Create dataset-specific directories
    vqa_dir = DATASETS_DIR / "vqav2"
    coco_dir = DATASETS_DIR / "mscoco" 
    richhf_dir = DATASETS_DIR / "richhf18k"
    cocostuff_dir = DATASETS_DIR / "cocostuff"
    
    vqa_dir.mkdir(exist_ok=True)
    coco_dir.mkdir(exist_ok=True)
    richhf_dir.mkdir(exist_ok=True)
    cocostuff_dir.mkdir(exist_ok=True)
    
    # Create val directories for evaluation
    (vqa_dir / "val").mkdir(exist_ok=True)
    (coco_dir / "val").mkdir(exist_ok=True)
    (richhf_dir / "val").mkdir(exist_ok=True)
    (cocostuff_dir / "val").mkdir(exist_ok=True)
    
    print("\nDirectory structure created. Please manually download and place the following datasets:")
    print(f"1. VQAv2 dataset: {vqa_dir.absolute()}")
    print(f"   Download from: https://visualqa.org/download.html")
    print(f"2. MS-COCO dataset: {coco_dir.absolute()}")
    print(f"   Download from: https://cocodataset.org/")
    print(f"3. RichHF-18K artifacts: {richhf_dir.absolute()}")
    print(f"   Download from: https://github.com/richzhang/PerceptualSimilarity")
    print(f"4. COCO-Stuff dataset: {cocostuff_dir.absolute()}")
    print(f"   Download from: https://github.com/nightrome/cocostuff")
    print(f"\nPlace your model checkpoint at: {CKPT_DIR.absolute()}/mcotmodel.safetensors")

if __name__ == "__main__":
    setup_directories()
