#!/usr/bin/env python
"""
Script to run inference with a trained MCoT model.

This script takes an image as input, runs it through the Planning and Acting stages of MCoT,
and outputs the results (bounding boxes and final caption).

Usage:
    python run_mcot_inference.py --model-path /path/to/mcot_model.pt --image-path /path/to/image.jpg --output-dir results
    
    # Or use a HuggingFace model directly:
    python run_mcot_inference.py --model-path EPFL-VILAB/4M-21_XL --image-path /path/to/image.jpg --output-dir results
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tokenizers import Tokenizer

# Import model and processing functions
from fourm.models.fm import FM  # Fixed import - UnifiedModel doesn't exist, FM is the correct class
from fourm.models.generate import GenerationSampler
from fourm.data.unified_datasets import load_and_preprocess_planning_sample

# For HuggingFace model loading (will be imported conditionally in load_model_and_tokenizer)
try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Run MCoT inference")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Path to the trained MCoT model checkpoint")
    parser.add_argument("--image-path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizers/mcot_tokenizer.json",
                        help="Path to the tokenizer with MCoT tokens")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save inference results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed outputs")
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, tokenizer_path: str, device: str):
    """
    Load the model and tokenizer.
    
    Args:
        model_path: Either a local path to a model checkpoint or a HuggingFace model ID (e.g., 'EPFL-VILAB/4M-21_XL')
        tokenizer_path: Path to the tokenizer JSON file with MCoT tokens
        device: Device to load the model on ('cuda' or 'cpu')
    """
    # Load tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Load model - check if model_path is a HuggingFace model ID
    if '/' in model_path and not os.path.exists(model_path):
        # This looks like a HuggingFace model ID, try to load it with from_pretrained
        try:
            model = FM.from_pretrained(model_path)
            print(f"Successfully loaded model from HuggingFace Hub: {model_path}")
        except ImportError:
            print("Could not import FM from fourm.models.fm. Make sure the 4M package is installed.")
            raise
        except Exception as e:
            print(f"Failed to load model from HuggingFace Hub: {e}")
            raise
    else:
        # Load from local checkpoint
        model = FM.from_pretrained(model_path)
        
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def prepare_image_for_planning(image_path: str, tokenizer, device: str, modality_info: Dict[str, Any]):
    """Prepare image and text inputs for the Planning stage."""
    image = Image.open(image_path).convert("RGB")
    
    # For planning stage, provide a generic prompt
    planning_prompt = "Describe what you see in this image."
    
    # Create a sample dictionary similar to what WebDataset would provide
    sample = {
        "image.jpg": image,
        "caption_prompt.txt": planning_prompt
    }
    
    # Process the sample with planning preprocessing
    planning_start_token_id = tokenizer.token_to_id("[PLANNING_START]")
    prompt_tokens = tokenizer.encode(planning_prompt).ids
    input_prompt_with_prefix = [planning_start_token_id] + prompt_tokens
    
    # Create processed sample with image and text input
    processed = {}
    
    # Add image
    img_key = next((k for k, v in modality_info.items() if v.get('type') == 'img' and 'rgb' in k), 'rgb@224')
    processed[img_key] = image
    
    # Add caption with [PLANNING_START] prefix
    processed['caption'] = torch.tensor(input_prompt_with_prefix, dtype=torch.long).to(device)
    
    return processed, image


def run_planning_stage(model: FM, inputs: Dict[str, torch.Tensor], tokenizer, device: str, verbose: bool = False):
    """Run the Planning stage to generate bounding boxes and plan text."""
    # Create a generation sampler
    sampler = GenerationSampler(model, tokenizer)
    
    # Define generation parameters
    generation_config = {
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.0,
        'max_new_tokens': 512,
    }
    
    # Generate outputs for Planning stage (caption and bounding boxes)
    with torch.no_grad():
        # Generate plan text first (uses 'caption' modality)
        plan_text_output = sampler.generate(
            inputs, 
            'caption', 
            **generation_config
        )
        
        # Generate bounding boxes (uses 'det' modality)
        bbox_output = sampler.generate(
            inputs,
            'det',
            **generation_config
        )
    
    # Decode outputs
    plan_text = tokenizer.decode(plan_text_output[0])
    bbox_text = tokenizer.decode(bbox_output[0])
    
    if verbose:
        print(f"Planning stage plan text: {plan_text}")
        print(f"Planning stage bbox output: {bbox_text}")
    
    # Parse bounding box data from generated text
    bboxes = parse_bbox_output(bbox_text)
    
    return plan_text, bboxes


def parse_bbox_output(bbox_text: str) -> List[List[float]]:
    """Parse bounding box data from the model's output text."""
    # Extract bounding box data
    bboxes = []
    
    # Look for [OBJECT] category [X] x [Y] y [W] w [H] h patterns
    parts = bbox_text.split("[OBJECT]")[1:] if "[OBJECT]" in bbox_text else []
    
    for part in parts:
        try:
            # Extract category name (before [X])
            category = part.split("[X]")[0].strip()
            
            # Extract coordinates
            x_part = part.split("[X]")[1].split("[Y]")[0].strip()
            y_part = part.split("[Y]")[1].split("[W]")[0].strip()
            w_part = part.split("[W]")[1].split("[H]")[0].strip()
            h_part = part.split("[H]")[1].split("[OBJECT]")[0].strip() if "[OBJECT]" in part else part.split("[H]")[1].strip()
            
            # Convert to float
            x = float(x_part) / 1000.0  # Assuming coordinates are normalized to 1000 bins
            y = float(y_part) / 1000.0
            w = float(w_part) / 1000.0
            h = float(h_part) / 1000.0
            
            bboxes.append([x, y, w, h, category])
        except Exception as e:
            print(f"Error parsing bbox data: {e}, part: {part}")
            continue
    
    return bboxes


def prepare_for_acting(image: Image.Image, plan_text: str, tokenizer, device: str, modality_info: Dict[str, Any]):
    """Prepare inputs for the Acting stage."""
    # Create processed sample with image and plan text
    processed = {}
    
    # Add image
    img_key = next((k for k, v in modality_info.items() if v.get('type') == 'img' and 'rgb' in k), 'rgb@224')
    processed[img_key] = image
    
    # Encode plan text with [ACTING_START] prefix
    acting_start_token_id = tokenizer.token_to_id("[ACTING_START]")
    plan_tokens = tokenizer.encode(plan_text).ids
    input_plan_with_prefix = [acting_start_token_id] + plan_tokens
    
    # Add plan text as caption input
    processed['caption'] = torch.tensor(input_plan_with_prefix, dtype=torch.long).to(device)
    
    return processed


def run_acting_stage(model: FM, inputs: Dict[str, torch.Tensor], tokenizer, device: str, verbose: bool = False):
    """Run the Acting stage to generate the final caption."""
    # Create a generation sampler
    sampler = GenerationSampler(model, tokenizer)
    
    # Define generation parameters
    generation_config = {
        'temperature': 0.7,
        'top_p': 0.9,
        'repetition_penalty': 1.0,
        'max_new_tokens': 256,
    }
    
    # Generate outputs for Acting stage (final caption)
    with torch.no_grad():
        caption_output = sampler.generate(
            inputs,
            'caption',
            **generation_config
        )
    
    # Decode output
    final_caption = tokenizer.decode(caption_output[0])
    
    if verbose:
        print(f"Acting stage raw output: {final_caption}")
    
    # Clean up the caption (remove special tokens, etc.)
    final_caption = clean_caption(final_caption)
    
    return final_caption


def clean_caption(caption: str) -> str:
    """Clean up generated caption by removing special tokens and formatting."""
    # Remove any special tokens and formatting
    caption = caption.replace("[ACTING_START]", "").strip()
    caption = caption.replace("[EOS]", "").strip()
    
    return caption


def visualize_results(image: Image.Image, bboxes: List[List[float]], plan_text: str, 
                     final_caption: str, output_path: str):
    """Visualize the Planning and Acting results."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)
    
    # Planning stage (top subplot)
    ax1.imshow(img_array)
    ax1.set_title(f"Planning Stage\nPlan: {plan_text}", fontsize=12)
    
    # Draw each bounding box
    img_h, img_w = image.height, image.width
    for bbox in bboxes:
        x, y, w, h, label = bbox
        
        # Convert normalized coordinates to pixel coordinates if needed
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
            x, w = x * img_w, w * img_w
            y, h = y * img_h, h * img_h
        
        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the patch to the Axes
        ax1.add_patch(rect)
        
        # Add label
        ax1.text(x, y, label, fontsize=10, 
                bbox=dict(facecolor='yellow', alpha=0.5))
    
    # Acting stage (bottom subplot)
    ax2.imshow(img_array)
    ax2.set_title(f"Acting Stage\nFinal Caption: {final_caption}", fontsize=12)
    
    # Hide axes
    ax1.axis('off')
    ax2.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path, args.device)
    
    # Define modality info for preprocessing
    modality_info = {
        'rgb@224': {
            'type': 'img',
            'num_channels': 3,
            'input_size': 224,
        },
        'caption': {
            'type': 'seq',
            'token_type': 'text',
        },
        'det': {
            'type': 'seq',
            'token_type': 'text',
            'coord_bins': 1000,
        }
    }
    
    # Run Planning stage
    print("Running Planning stage...")
    inputs, original_image = prepare_image_for_planning(args.image_path, tokenizer, args.device, modality_info)
    plan_text, bboxes = run_planning_stage(model, inputs, tokenizer, args.device, args.verbose)
    print(f"Planning complete. Plan text: {plan_text}")
    print(f"Detected objects: {len(bboxes)}")
    
    # Run Acting stage
    print("Running Acting stage...")
    acting_inputs = prepare_for_acting(original_image, plan_text, tokenizer, args.device, modality_info)
    final_caption = run_acting_stage(model, acting_inputs, tokenizer, args.device, args.verbose)
    print(f"Acting complete. Final caption: {final_caption}")
    
    # Visualize results
    image_name = Path(args.image_path).stem
    output_path = os.path.join(args.output_dir, f"mcot_result_{image_name}.png")
    visualize_results(original_image, bboxes, plan_text, final_caption, output_path)
    print(f"Results visualized and saved to {output_path}")
    
    # Save results as JSON
    results = {
        "image_path": args.image_path,
        "planning": {
            "plan_text": plan_text,
            "bboxes": bboxes
        },
        "acting": {
            "final_caption": final_caption
        }
    }
    
    json_path = os.path.join(args.output_dir, f"mcot_result_{image_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
    main() 