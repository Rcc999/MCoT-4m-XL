#!/usr/bin/env python
"""
Test script for MCoT Planning and Acting pipeline.

This script takes a single image and runs it through both MCoT stages:
1. Planning: Generate bounding boxes and plan text
2. Acting: Use the plan to generate the final caption

Usage:
python test_mcot_pipeline.py --model-path /path/to/model_checkpoint --image-path /path/to/image.jpg
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
from transformers import AutoTokenizer

# Import your model and preprocessing functions
from fourm.model.unified_model import UnifiedModel
from fourm.data.unified_datasets import load_and_preprocess_planning_sample

def parse_args():
    parser = argparse.ArgumentParser(description="Test MCoT Planning and Acting pipeline")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Path to the model checkpoint")
    parser.add_argument("--image-path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output-dir", type=str, default="test_results",
                        help="Directory to save test results")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizers/custom_tokenizer",
                        help="Path to the tokenizer")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    return parser.parse_args()

def load_model(model_path: str, device: str) -> UnifiedModel:
    """Load the model from checkpoint."""
    model = UnifiedModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model

def prepare_image_for_planning(image_path: str, tokenizer: AutoTokenizer, device: str):
    """Prepare image and text inputs for the Planning stage."""
    image = Image.open(image_path).convert("RGB")
    
    # For planning stage, provide a generic prompt
    planning_prompt = "Describe what you see in this image."
    
    # Create a sample dictionary similar to what WebDataset would provide
    sample = {
        "image.jpg": image,
        "caption_prompt.txt": planning_prompt
    }
    
    # Process the sample using the same function used during training
    processed = load_and_preprocess_planning_sample(sample, tokenizer)
    
    # Move to the specified device
    for k, v in processed.items():
        if isinstance(v, torch.Tensor):
            processed[k] = v.to(device)
    
    return processed, image

def run_planning_stage(model: UnifiedModel, inputs: Dict[str, torch.Tensor], tokenizer: AutoTokenizer, verbose: bool = False):
    """Run the Planning stage to generate bounding boxes and plan text."""
    with torch.no_grad():
        outputs = model.generate(**{k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() == 1 else v 
                               for k, v in inputs.items()})
    
    # Decode outputs
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    if verbose:
        print(f"Planning stage raw output: {decoded_output}")
    
    # Extract bounding boxes and plan text from the decoded output
    bboxes, plan_text = parse_planning_output(decoded_output)
    
    return bboxes, plan_text

def prepare_for_acting(image: Image.Image, plan_text: str, tokenizer: AutoTokenizer, device: str):
    """Prepare inputs for the Acting stage."""
    # Create a sample dictionary similar to what WebDataset would provide
    sample = {
        "image.jpg": image,
        "plan_text.txt": plan_text
    }
    
    # For simplicity, reuse similar processing as planning but with plan text instead of caption prompt
    # In a real implementation, you'd use the specific acting preprocessor
    processed = {
        "input_ids": tokenizer.encode(f"Generate a caption based on this plan: {plan_text}", return_tensors="pt")[0].to(device),
        "attention_mask": torch.ones(len(tokenizer.encode(f"Generate a caption based on this plan: {plan_text}"))).to(device),
        "pixel_values": sample["image.jpg"] # Directly use the image
    }
    
    return processed

def run_acting_stage(model: UnifiedModel, inputs: Dict[str, torch.Tensor], tokenizer: AutoTokenizer, verbose: bool = False):
    """Run the Acting stage to generate the final caption."""
    with torch.no_grad():
        outputs = model.generate(**{k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() == 1 else v 
                               for k, v in inputs.items()})
    
    # Decode outputs
    final_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if verbose:
        print(f"Acting stage raw output: {tokenizer.decode(outputs[0], skip_special_tokens=False)}")
    
    return final_caption

def parse_planning_output(decoded_output: str) -> Tuple[List[List[float]], str]:
    """
    Parse the planning stage output to extract bounding boxes and plan text.
    
    Expected format: 
    [PLANNING_START] This is a plan text. [OBJECT] cat [X] 0.2 [Y] 0.3 [W] 0.1 [H] 0.1 [OBJECT] dog...
    
    Note: This is a placeholder implementation. You'll need to adapt it based on your model's 
    actual output format and tokenization scheme.
    """
    # Placeholder implementation
    # In your actual implementation, you'd parse the model's output based on your tokenization scheme
    
    # Example parsing logic (simplified):
    plan_text = "A cat and a dog are in the image."
    bboxes = [[0.2, 0.3, 0.1, 0.1, "cat"], [0.5, 0.6, 0.2, 0.15, "dog"]]
    
    # You would actually extract these from decoded_output using regex or string parsing
    # For example:
    # - Extract text between [PLANNING_START] and first [OBJECT] for plan_text
    # - Find all occurrences of [OBJECT] ... [X] ... [Y] ... [W] ... [H] ... patterns for bboxes
    
    return bboxes, plan_text

def visualize_results(image: Image.Image, bboxes: List[List[float]], plan_text: str, 
                     final_caption: str, output_path: str):
    """Visualize the Planning and Acting results."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 18))
    
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)
    
    # Planning stage (top subplot)
    ax1.imshow(img_array)
    ax1.set_title(f"Planning Stage\nPlan: {plan_text}", fontsize=12)
    
    # Draw each bounding box
    for bbox in bboxes:
        x, y, w, h, label = bbox
        
        # Convert normalized coordinates to pixel coordinates if needed
        img_h, img_w = image.height, image.width
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
            x, w = x * img_w, w * img_w
            y, h = y * img_h, h * img_h
        
        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the patch to the Axes
        ax1.add_patch(rect)
        
        # Add label
        ax1.text(x, y, label, fontsize=12, 
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Run Planning stage
    print("Running Planning stage...")
    inputs, original_image = prepare_image_for_planning(args.image_path, tokenizer, device)
    bboxes, plan_text = run_planning_stage(model, inputs, tokenizer, args.verbose)
    print(f"Planning complete. Plan text: {plan_text}")
    print(f"Detected objects: {len(bboxes)}")
    
    # Run Acting stage
    print("Running Acting stage...")
    acting_inputs = prepare_for_acting(original_image, plan_text, tokenizer, device)
    final_caption = run_acting_stage(model, acting_inputs, tokenizer, args.verbose)
    print(f"Acting complete. Final caption: {final_caption}")
    
    # Visualize results
    output_path = os.path.join(args.output_dir, f"mcot_test_results_{Path(args.image_path).stem}.png")
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
    
    json_path = os.path.join(args.output_dir, f"mcot_test_results_{Path(args.image_path).stem}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {json_path}")

if __name__ == "__main__":
    main() 