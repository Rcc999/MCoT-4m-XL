#!/usr/bin/env python
"""
Script to create WebDataset shards for MCoT Planning and Acting stages from MS-COCO data.

Requirements:
- webdataset
- pycocotools
- numpy
- PIL

Usage:
python create_mcot_coco_shards.py --data-dir data/coco
"""

import os
import json
import random
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image
import webdataset as wds
from pycocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser(description="Create WebDataset shards for MCoT")
    parser.add_argument("--data-dir", type=str, default="data/coco", 
                        help="Directory containing COCO data")
    parser.add_argument("--output-dir", type=str, default="data/coco_mcot_shards",
                        help="Output directory for WebDataset shards")
    parser.add_argument("--samples-per-shard", type=int, default=1000,
                        help="Number of samples per WebDataset shard")
    parser.add_argument("--max-train-samples", type=int, default=100000,
                        help="Maximum number of training samples to process")
    parser.add_argument("--max-val-samples", type=int, default=5000,
                        help="Maximum number of validation samples to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def load_coco_data(data_dir: str, split: str = "train"):
    """Load COCO dataset for the specified split."""
    anno_file = os.path.join(data_dir, f"annotations/instances_{split}2017.json")
    captions_file = os.path.join(data_dir, f"annotations/captions_{split}2017.json")
    image_dir = os.path.join(data_dir, f"{split}2017")
    
    # Load instances and captions
    coco_instances = COCO(anno_file)
    coco_captions = COCO(captions_file)
    
    return coco_instances, coco_captions, image_dir

def generate_planning_samples(
    coco_instances: COCO, 
    coco_captions: COCO, 
    image_dir: str,
    max_samples: int = 100000
) -> List[Dict[str, Any]]:
    """Generate samples for Planning stage from COCO data."""
    
    # Get image ids that have both captions and instances (bounding boxes)
    caption_img_ids = set(coco_captions.getImgIds())
    instance_img_ids = set(coco_instances.getImgIds())
    valid_img_ids = list(caption_img_ids.intersection(instance_img_ids))
    
    # Shuffle and limit samples
    random.shuffle(valid_img_ids)
    valid_img_ids = valid_img_ids[:max_samples]
    
    samples = []
    for img_id in valid_img_ids:
        # Get image info
        img_info = coco_instances.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(image_dir, file_name)
        
        # Skip if image file doesn't exist
        if not os.path.exists(img_path):
            continue
        
        # Get captions for this image
        caption_annos = coco_captions.loadAnns(coco_captions.getAnnIds(imgIds=img_id))
        if not caption_annos:
            continue
        
        # Randomly select a caption to use as input prompt
        caption = random.choice(caption_annos)["caption"]
        
        # Get bounding boxes and categories for this image
        instance_annos = coco_instances.loadAnns(coco_instances.getAnnIds(imgIds=img_id))
        if not instance_annos:
            continue
        
        # Extract bounding box data: [x, y, width, height, category_name]
        bbox_data = []
        for ann in instance_annos:
            if "bbox" in ann and "category_id" in ann:
                category_id = ann["category_id"]
                category_name = coco_instances.loadCats(category_id)[0]["name"]
                x, y, w, h = ann["bbox"]
                bbox_data.append([x, y, w, h, category_name])
        
        if not bbox_data:
            continue
        
        # For planning stage:
        # - Input: image + caption prompt
        # - Target: plan text (simplified version of caption) + bbox layout
        
        # For simplicity, use the original caption as target plan text
        # In a more sophisticated implementation, you could generate a plan text
        target_plan_text = caption
        
        # Create bbox annotation in the expected format
        bbox_json = {
            "image_width": img_info["width"],
            "image_height": img_info["height"],
            "annotations": bbox_data
        }
        
        samples.append({
            "image_path": img_path,
            "caption_prompt": f"Describe what you see in this image.",
            "target_plan_text": target_plan_text,
            "bboxes": bbox_json,
            "img_id": img_id
        })
    
    return samples

def generate_acting_samples(
    planning_samples: List[Dict[str, Any]],
    max_samples: int = 100000
) -> List[Dict[str, Any]]:
    """Generate samples for Acting stage from Planning outputs."""
    
    # For simplicity, we'll reuse the planning samples
    # In a real implementation, you might want to use actual planning outputs
    samples = []
    for i, plan_sample in enumerate(planning_samples[:max_samples]):
        samples.append({
            "image_path": plan_sample["image_path"],
            "plan_text": plan_sample["target_plan_text"],
            "plan_bboxes": plan_sample["bboxes"],
            "target_final_caption": plan_sample["target_plan_text"],
            "img_id": plan_sample["img_id"]
        })
    
    return samples

def create_planning_shards(samples: List[Dict[str, Any]], output_dir: str, split: str, samples_per_shard: int):
    """Create WebDataset shards for Planning stage."""
    os.makedirs(os.path.join(output_dir, f"planning_{split}"), exist_ok=True)
    
    # Split samples into shards
    num_shards = (len(samples) + samples_per_shard - 1) // samples_per_shard
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, len(samples))
        shard_samples = samples[start_idx:end_idx]
        
        shard_path = os.path.join(output_dir, f"planning_{split}/{shard_idx:05d}.tar")
        with wds.TarWriter(shard_path) as sink:
            for i, sample in enumerate(shard_samples):
                # Create a key for this sample (must be unique within the shard)
                key = f"{sample['img_id']:012d}"
                
                # Load image
                img_pil = Image.open(sample["image_path"]).convert("RGB")
                
                # Serialize bbox data
                bbox_json_str = json.dumps(sample["bboxes"])
                
                # Add sample to the shard
                sink.write({
                    "__key__": key,
                    "image.jpg": img_pil,
                    "caption_prompt.txt": sample["caption_prompt"],
                    "target_plan_text.txt": sample["target_plan_text"],
                    "bboxes.json": bbox_json_str
                })
        
        print(f"Created Planning shard {shard_idx+1}/{num_shards} for {split} split: {shard_path}")

def create_acting_shards(samples: List[Dict[str, Any]], output_dir: str, split: str, samples_per_shard: int):
    """Create WebDataset shards for Acting stage."""
    os.makedirs(os.path.join(output_dir, f"acting_{split}"), exist_ok=True)
    
    # Split samples into shards
    num_shards = (len(samples) + samples_per_shard - 1) // samples_per_shard
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, len(samples))
        shard_samples = samples[start_idx:end_idx]
        
        shard_path = os.path.join(output_dir, f"acting_{split}/{shard_idx:05d}.tar")
        with wds.TarWriter(shard_path) as sink:
            for i, sample in enumerate(shard_samples):
                # Create a key for this sample (must be unique within the shard)
                key = f"{sample['img_id']:012d}"
                
                # Load image
                img_pil = Image.open(sample["image_path"]).convert("RGB")
                
                # Serialize bbox data
                plan_bboxes_json_str = json.dumps(sample["plan_bboxes"])
                
                # Add sample to the shard
                sink.write({
                    "__key__": key,
                    "image.jpg": img_pil,
                    "plan_text.txt": sample["plan_text"],
                    "plan_bboxes.json": plan_bboxes_json_str,
                    "target_final_caption.txt": sample["target_final_caption"]
                })
        
        print(f"Created Acting shard {shard_idx+1}/{num_shards} for {split} split: {shard_path}")

def update_config(output_dir: str):
    """Update the MCoT config with the actual paths to shards."""
    config_path = "cfgs/mcot_data_config.yaml"
    
    # Read the current config
    with open(config_path, "r") as f:
        config = f.read()
    
    # Replace placeholder paths with actual paths
    config = config.replace(
        '/path/to/coco/webdataset/planning/{00000..01999}.tar',
        f"{output_dir}/planning_train/{{00000..00099}}.tar"
    )
    config = config.replace(
        '/path/to/coco/webdataset/acting/{00000..01999}.tar',
        f"{output_dir}/acting_train/{{00000..00099}}.tar"
    )
    config = config.replace(
        '/path/to/coco/webdataset/planning_val/{00000..00099}.tar',
        f"{output_dir}/planning_val/{{00000..00005}}.tar"
    )
    config = config.replace(
        '/path/to/coco/webdataset/acting_val/{00000..00099}.tar',
        f"{output_dir}/acting_val/{{00000..00005}}.tar"
    )
    
    # Write the updated config
    with open(config_path, "w") as f:
        f.write(config)
    
    print(f"Updated {config_path} with actual shard paths")

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process training data
    print(f"Processing COCO training data from {args.data_dir}...")
    coco_train_instances, coco_train_captions, train_img_dir = load_coco_data(args.data_dir, "train")
    train_planning_samples = generate_planning_samples(
        coco_train_instances, coco_train_captions, train_img_dir, args.max_train_samples
    )
    print(f"Generated {len(train_planning_samples)} training samples for Planning")
    
    train_acting_samples = generate_acting_samples(train_planning_samples, args.max_train_samples)
    print(f"Generated {len(train_acting_samples)} training samples for Acting")
    
    # Process validation data
    print(f"Processing COCO validation data from {args.data_dir}...")
    coco_val_instances, coco_val_captions, val_img_dir = load_coco_data(args.data_dir, "val")
    val_planning_samples = generate_planning_samples(
        coco_val_instances, coco_val_captions, val_img_dir, args.max_val_samples
    )
    print(f"Generated {len(val_planning_samples)} validation samples for Planning")
    
    val_acting_samples = generate_acting_samples(val_planning_samples, args.max_val_samples)
    print(f"Generated {len(val_acting_samples)} validation samples for Acting")
    
    # Create WebDataset shards
    print("Creating Planning shards...")
    create_planning_shards(train_planning_samples, args.output_dir, "train", args.samples_per_shard)
    create_planning_shards(val_planning_samples, args.output_dir, "val", args.samples_per_shard)
    
    print("Creating Acting shards...")
    create_acting_shards(train_acting_samples, args.output_dir, "train", args.samples_per_shard)
    create_acting_shards(val_acting_samples, args.output_dir, "val", args.samples_per_shard)
    
    # Update the config file with actual paths
    update_config(os.path.abspath(args.output_dir))
    
    print("Done!")

if __name__ == "__main__":
    main() 