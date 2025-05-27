# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generation script for Multimodal Chain of Thought (MCoT) using 4M model.
Implements step-by-step MCoT image generation:
1. Planning: Caption and layout planning
2. Acting: Initial image generation
3. Reflection: Artifact detection
4. Correction: Inpainting/correction
"""

import argparse
import os
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tokenizers import Tokenizer

from fourm.models.generate import generate_text, generate_image, modify_image, generate_batch
from fourm.utils import create_model, load_safetensors
from fourm.models.mcot_fixed import add_mcot_to_model
from mcot_data import mcot_utils

# Set up argument parser
def get_args_parser():
    parser = argparse.ArgumentParser('4M MCoT Generation', add_help=False)
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the MCoT-finetuned model checkpoint')
    parser.add_argument('--model', default='fm_xlarge_24e_24d_swiglu_nobias', type=str,
                        help='Model architecture')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for image generation')
    parser.add_argument('--output_dir', default='./mcot_generation_output',
                        help='Directory to save generated images and intermediates')
    parser.add_argument('--show_intermediate_steps', action='store_true',
                        help='Save images for all intermediate MCoT steps')
    parser.add_argument('--tokenizer_path', default='fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json',
                        help='Path to tokenizer')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for inference')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for generation')
    parser.add_argument('--num_images', default=1, type=int,
                        help='Number of images to generate')
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', default=0.9, type=float,
                        help='Top-p sampling threshold')
    parser.add_argument('--top_k', default=50, type=int,
                        help='Top-k sampling threshold')
    parser.add_argument('--cfg_scale', default=3.0, type=float,
                        help='Classifier-free guidance scale')
    return parser


def main(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    
    # Create model
    print(f"Creating model: {args.model}")
    model = create_model(args.model)
    
    # Add MCoT capabilities
    print("Adding MCoT capabilities to model")
    model = add_mcot_to_model(model)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.model_path}")
    if args.model_path.endswith('.safetensors'):
        checkpoint = load_safetensors(args.model_path)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    else:
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'model' in checkpoint:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Move model to device
    model = model.to(args.device)
    model.eval()
    
    # Set up generation parameters
    generation_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'cfg_scale': args.cfg_scale
    }
    
    # Process for multiple images if requested
    for img_idx in range(args.num_images):
        print(f"Generating image {img_idx+1}/{args.num_images}")
        
        # Create a subfolder for each image
        img_dir = os.path.join(args.output_dir, f"image_{img_idx}")
        os.makedirs(img_dir, exist_ok=True)
        
        # Save prompt
        with open(os.path.join(img_dir, "prompt.txt"), "w") as f:
            f.write(args.prompt)
        
        # Run MCoT generation process
        result = generate_with_mcot(model, args.prompt, tokenizer, img_dir, args.show_intermediate_steps, 
                                    device=args.device, **generation_params)
        
        # Save final result
        result_img = result["final_image"]
        result_img.save(os.path.join(img_dir, "final.png"))
        
        # Save reasoning steps as text
        with open(os.path.join(img_dir, "mcot_reasoning.txt"), "w") as f:
            f.write(f"Prompt: {args.prompt}\n\n")
            f.write(f"Planning: {result['planning']}\n\n")
            f.write(f"Acting: {result['acting']}\n\n")
            f.write(f"Reflection: {result['reflection']}\n\n")
            f.write(f"Correction: {result['correction']}\n\n")
        
        print(f"Generated image saved to {os.path.join(img_dir, 'final.png')}")


def generate_with_mcot(model, prompt, tokenizer, output_dir, save_intermediates=False, device="cuda", **generation_params):
    """
    Generate an image using the MCoT methodology.
    
    Args:
        model: MCoT-enhanced 4M model
        prompt: Text prompt for image generation
        tokenizer: Tokenizer for text processing
        output_dir: Directory to save output files
        save_intermediates: Whether to save intermediate steps
        device: Device to use for inference
        **generation_params: Additional generation parameters
    
    Returns:
        Dictionary containing the final image and reasoning steps
    """
    result = {}
    
    # Step 1: Planning
    print("MCoT Step 1: Planning")
    planning_prompt = mcot_utils.format_mcot_input(prompt, "planning")
    planning_output = generate_text(model, planning_prompt, tokenizer, **generation_params)
    result["planning"] = planning_output
    
    # Save planning output
    with open(os.path.join(output_dir, "1_planning.txt"), "w") as f:
        f.write(planning_output)
    print(f"Planning: {planning_output[:100]}...")
    
    # Step 2: Acting (Initial image generation)
    print("MCoT Step 2: Acting")
    acting_prompt = mcot_utils.format_mcot_input(prompt, "acting")
    # Use the planning output to enhance generation
    acting_prompt += f"\nPlanning: {planning_output}"
    
    # Generate image using the acting prompt
    acting_image = generate_image(model, acting_prompt, tokenizer, **generation_params)
    result["acting"] = "Image generated based on the planning"
    result["acting_image"] = acting_image
    
    # Save acting image
    if save_intermediates:
        acting_image.save(os.path.join(output_dir, "2_acting.png"))
    
    # Step 3: Reflection
    print("MCoT Step 3: Reflection")
    reflection_prompt = mcot_utils.format_mcot_input(prompt, "reflection")
    # Use the image for reflection
    reflection_output = generate_text(model, reflection_prompt, tokenizer, 
                                     image=acting_image, **generation_params)
    result["reflection"] = reflection_output
    
    # Save reflection output
    with open(os.path.join(output_dir, "3_reflection.txt"), "w") as f:
        f.write(reflection_output)
    print(f"Reflection: {reflection_output[:100]}...")
    
    # Step 4: Correction
    print("MCoT Step 4: Correction")
    correction_prompt = mcot_utils.format_mcot_input(prompt, "correction")
    # Use the reflection output to guide correction
    correction_prompt += f"\nReflection: {reflection_output}"
    
    # Modify the image based on the correction prompt
    correction_image = modify_image(model, correction_prompt, acting_image, tokenizer, **generation_params)
    result["correction"] = "Image corrected based on the reflection"
    result["correction_image"] = correction_image
    
    # Save correction image
    if save_intermediates:
        correction_image.save(os.path.join(output_dir, "4_correction.png"))
    
    # Set final image
    result["final_image"] = correction_image
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser('4M MCoT Generation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
