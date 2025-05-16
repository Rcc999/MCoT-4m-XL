#!/usr/bin/env python
"""
Visualization script for MCoT model's Planning and Acting stages.

This script visualizes:
1. Planning stage: Input image with predicted bounding boxes
2. Acting stage: Input image with final caption

Usage:
python visualization/visualize_mcot.py --model-path /path/to/model_checkpoint --image-path /path/to/image.jpg
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
import cv2

# Import your model
from fourm.model.unified_model import UnifiedModel
from fourm.data.unified_datasets import load_and_preprocess_planning_sample

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MCoT model predictions")
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Path to the model checkpoint")
    parser.add_argument("--image-path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output-dir", type=str, default="visualization/outputs",
                        help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizers/custom_tokenizer",
                        help="Path to the tokenizer")
    return parser.parse_args()

def load_model(model_path: str, device: str) -> UnifiedModel:
    """Load the model from checkpoint."""
    model = UnifiedModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model

def prepare_image_for_model(image_path: str, tokenizer: AutoTokenizer, device: str):
    """Prepare image and text inputs for the model."""
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

def run_planning_inference(model: UnifiedModel, inputs: Dict[str, torch.Tensor], tokenizer: AutoTokenizer):
    """Run the planning stage inference."""
    with torch.no_grad():
        outputs = model.generate(**{k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v 
                                  for k, v in inputs.items()})
    
    # Decode outputs
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract bounding boxes from the decoded output
    # This depends on your model's output format
    bboxes, plan_text = parse_planning_output(decoded_output)
    
    return bboxes, plan_text

def run_acting_inference(model: UnifiedModel, inputs: Dict[str, torch.Tensor], 
                         plan_text: str, tokenizer: AutoTokenizer):
    """Run the acting stage inference."""
    # Replace the caption prompt with the plan text
    inputs["input_ids"] = tokenizer.encode(
        f"Generate a caption based on this plan: {plan_text}",
        return_tensors="pt"
    ).to(inputs["pixel_values"].device)
    
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    
    with torch.no_grad():
        outputs = model.generate(**{k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() == 1 else v 
                                  for k, v in inputs.items()})
    
    # Decode outputs
    final_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return final_caption

def parse_planning_output(decoded_output: str) -> Tuple[List[List[float]], str]:
    """
    Parse the planning stage output to extract bounding boxes and plan text.
    This is a placeholder implementation - you'll need to adapt it to your model's output format.
    """
    # Placeholder implementation
    # In your real implementation, parse the model's output based on your tokenization scheme
    
    # Example of expected format: 
    # [PLANNING_START] This is a plan text. [OBJECT] cat [X] 0.2 [Y] 0.3 [W] 0.1 [H] 0.1 [OBJECT] dog...
    
    # This is just a placeholder - replace with actual parsing logic
    bboxes = [[0.2, 0.3, 0.1, 0.1, "cat"], [0.5, 0.6, 0.2, 0.15, "dog"]]
    plan_text = "A cat and a dog are in the image."
    
    return bboxes, plan_text

def visualize_planning(image: Image.Image, bboxes: List[List[float]], plan_text: str, 
                      output_path: str):
    """Visualize the planning stage: image with bounding boxes."""
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    
    # Display the image
    ax.imshow(img_array)
    
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
        ax.add_patch(rect)
        
        # Add label
        plt.text(x, y, label, fontsize=12, 
                 bbox=dict(facecolor='yellow', alpha=0.5))
    
    # Add plan text as title
    ax.set_title(f"Plan: {plan_text}", fontsize=12)
    
    # Hide axes
    plt.axis('off')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def visualize_acting(image: Image.Image, final_caption: str, output_path: str):
    """Visualize the acting stage: image with final caption."""
    # Convert PIL image to numpy array for matplotlib
    img_array = np.array(image)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    
    # Display the image
    ax.imshow(img_array)
    
    # Add final caption as title
    ax.set_title(f"Final Caption: {final_caption}", fontsize=14, wrap=True)
    
    # Hide axes
    plt.axis('off')
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight')
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
    
    # Prepare image for model
    inputs, original_image = prepare_image_for_model(args.image_path, tokenizer, device)
    
    # Run planning inference
    bboxes, plan_text = run_planning_inference(model, inputs, tokenizer)
    
    # Run acting inference
    final_caption = run_acting_inference(model, inputs, plan_text, tokenizer)
    
    # Save planning visualization
    planning_output_path = os.path.join(args.output_dir, "planning_output.png")
    visualize_planning(original_image, bboxes, plan_text, planning_output_path)
    
    # Save acting visualization
    acting_output_path = os.path.join(args.output_dir, "acting_output.png")
    visualize_acting(original_image, final_caption, acting_output_path)
    
    # Print results
    print(f"Planning Stage Result:")
    print(f"Plan Text: {plan_text}")
    print(f"Bounding Boxes: {bboxes}")
    print()
    print(f"Acting Stage Result:")
    print(f"Final Caption: {final_caption}")
    print()
    print(f"Visualizations saved to:")
    print(f"Planning: {planning_output_path}")
    print(f"Acting: {acting_output_path}")

if __name__ == "__main__":
    main() 