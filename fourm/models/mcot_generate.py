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
MCOT generation module for the 4M model
This implements the sequential four-stage MCOT process:
1. Planning: Generate captions and bounding boxes
2. Acting: Generate images based on Planning outputs
3. Reflection: Generate heatmaps identifying artifacts
4. Correction: Fix artifacts identified in Reflection
"""

import copy
import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

from fourm.models.generate import GenerationSampler, empty_img_modality, empty_seq_modality
from fourm.utils.generation import cosine_schedule, linear_schedule

# MCOT special token constants
PLANNING_START_TOKEN = "[PLANNING_START]"
ACTING_START_TOKEN = "[ACTING_START]"
REFLECTION_START_TOKEN = "[REFLECTION_START]"
CORRECTION_START_TOKEN = "[CORRECTION_START]"


class MCOTGenerationSampler(nn.Module):
    """
    MCOT Generation Sampler that implements the four-stage generation process.
    This builds on top of the existing GenerationSampler to implement MCOT.
    """
    
    def __init__(self, model, marker_tokens=None):
        """
        Initialize the MCOT Generation Sampler.
        
        Args:
            model: The 4M model for generation.
            marker_tokens: Dictionary of stage marker tokens.
        """
        super().__init__()
        self.model = model
        
        # Initialize underlying generation sampler
        self.sampler = GenerationSampler(model)
        
        # Set up marker tokens
        self.marker_tokens = marker_tokens or {
            "planning": PLANNING_START_TOKEN,
            "acting": ACTING_START_TOKEN,
            "reflection": REFLECTION_START_TOKEN,
            "correction": CORRECTION_START_TOKEN,
        }
    
    def _prepare_planning_input(self, mod_dict, text_tokenizer):
        """
        Prepare input for the Planning stage.
        
        Args:
            mod_dict: Dictionary containing the input modalities.
            text_tokenizer: Text tokenizer.
            
        Returns:
            Modified mod_dict with Planning token prepended to text.
        """
        # Create a copy to avoid modifying the original
        planning_dict = copy.deepcopy(mod_dict)
        
        # Prepend the Planning token to the text input if it exists
        if "text" in planning_dict:
            planning_marker = self.marker_tokens["planning"]
            # Get the token ID for the Planning marker
            planning_token_id = text_tokenizer.token_to_id(planning_marker)
            
            if planning_token_id is not None:
                # Get the tensor from the text modality
                text_tensor = planning_dict["text"]["tensor"]
                
                # Create a new tensor with the Planning token at the beginning
                new_tensor = torch.zeros_like(text_tensor)
                new_tensor[:, 0] = planning_token_id
                
                # Find the first non-zero token in the text (skip padding)
                non_zero_idx = torch.nonzero(text_tensor, as_tuple=True)[1]
                if len(non_zero_idx) > 0:
                    first_idx = non_zero_idx[0].item()
                    
                    # Copy the original text after the Planning token
                    new_tensor[:, 1:] = text_tensor[:, first_idx:]
                
                # Update the text tensor
                planning_dict["text"]["tensor"] = new_tensor
                
                # Update the input and target masks
                planning_dict["text"]["input_mask"][:, 0] = False  # The Planning token is input
                planning_dict["text"]["target_mask"][:, 0] = True  # The Planning token is not a target
        
        return planning_dict
    
    def _prepare_acting_input(self, mod_dict, planning_outputs, text_tokenizer):
        """
        Prepare input for the Acting stage based on Planning outputs.
        
        Args:
            mod_dict: Dictionary containing the original input modalities.
            planning_outputs: Dictionary containing the outputs from the Planning stage.
            text_tokenizer: Text tokenizer.
            
        Returns:
            Modified mod_dict for the Acting stage.
        """
        # Create a copy to avoid modifying the original
        acting_dict = copy.deepcopy(mod_dict)
        
        # Add the Planning outputs to the input for Acting
        for key in planning_outputs:
            if key in acting_dict:
                # Overwrite with planning outputs
                acting_dict[key] = planning_outputs[key]
            else:
                # Add new planning outputs
                acting_dict[key] = planning_outputs[key]
        
        # Prepend the Acting token to the text input if it exists
        if "text" in acting_dict:
            acting_marker = self.marker_tokens["acting"]
            # Get the token ID for the Acting marker
            acting_token_id = text_tokenizer.token_to_id(acting_marker)
            
            if acting_token_id is not None:
                # Get the tensor from the text modality
                text_tensor = acting_dict["text"]["tensor"]
                
                # Create a new tensor with the Acting token at the beginning
                new_tensor = torch.zeros_like(text_tensor)
                new_tensor[:, 0] = acting_token_id
                
                # Find the first non-zero token in the text (skip padding)
                non_zero_idx = torch.nonzero(text_tensor, as_tuple=True)[1]
                if len(non_zero_idx) > 0:
                    first_idx = non_zero_idx[0].item()
                    
                    # Copy the original text after the Acting token
                    new_tensor[:, 1:] = text_tensor[:, first_idx:]
                
                # Update the text tensor
                acting_dict["text"]["tensor"] = new_tensor
                
                # Update the input and target masks
                acting_dict["text"]["input_mask"][:, 0] = False  # The Acting token is input
                acting_dict["text"]["target_mask"][:, 0] = True  # The Acting token is not a target
        
        # Prepare target RGB modality for image generation
        if "rgb" in acting_dict:
            batch_size, num_tokens = acting_dict["rgb"]["tensor"].shape
            device = acting_dict["rgb"]["tensor"].device
            
            # Initialize an empty target modality
            acting_dict["rgb_target"] = {
                "tensor": torch.zeros((batch_size, num_tokens), dtype=torch.int64, device=device),
                "input_mask": torch.ones((batch_size, num_tokens), dtype=torch.bool, device=device),
                "target_mask": torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device),
            }
            
            # Set up for image generation
            acting_dict = empty_img_modality(acting_dict, "rgb_target")
        
        return acting_dict
    
    def _prepare_reflection_input(self, mod_dict, acting_outputs, text_tokenizer):
        """
        Prepare input for the Reflection stage based on Acting outputs.
        
        Args:
            mod_dict: Dictionary containing the original input modalities.
            acting_outputs: Dictionary containing the outputs from the Acting stage.
            text_tokenizer: Text tokenizer.
            
        Returns:
            Modified mod_dict for the Reflection stage.
        """
        # Create a copy to avoid modifying the original
        reflection_dict = copy.deepcopy(mod_dict)
        
        # Use the generated image from Acting as input for Reflection
        if "rgb_target" in acting_outputs:
            reflection_dict["rgb"] = acting_outputs["rgb_target"]
        
        # Prepend the Reflection token to the text input if it exists
        if "text" in reflection_dict:
            reflection_marker = self.marker_tokens["reflection"]
            # Get the token ID for the Reflection marker
            reflection_token_id = text_tokenizer.token_to_id(reflection_marker)
            
            if reflection_token_id is not None:
                # Get the tensor from the text modality
                text_tensor = reflection_dict["text"]["tensor"]
                
                # Create a new tensor with the Reflection token at the beginning
                new_tensor = torch.zeros_like(text_tensor)
                new_tensor[:, 0] = reflection_token_id
                
                # Find the first non-zero token in the text (skip padding)
                non_zero_idx = torch.nonzero(text_tensor, as_tuple=True)[1]
                if len(non_zero_idx) > 0:
                    first_idx = non_zero_idx[0].item()
                    
                    # Copy the original text after the Reflection token
                    new_tensor[:, 1:] = text_tensor[:, first_idx:]
                
                # Update the text tensor
                reflection_dict["text"]["tensor"] = new_tensor
                
                # Update the input and target masks
                reflection_dict["text"]["input_mask"][:, 0] = False  # The Reflection token is input
                reflection_dict["text"]["target_mask"][:, 0] = True  # The Reflection token is not a target
        
        # Prepare target heatmap modality for artifact identification
        batch_size = reflection_dict["rgb"]["tensor"].shape[0]
        device = reflection_dict["rgb"]["tensor"].device
        
        # Number of tokens for the heatmap (assume square heatmap with same size as image)
        num_tokens = 64*64  # Default size for heatmap (can be adjusted)
        
        # Initialize an empty target modality for heatmap
        reflection_dict["heatmap"] = {
            "tensor": torch.zeros((batch_size, num_tokens), dtype=torch.int64, device=device),
            "input_mask": torch.ones((batch_size, num_tokens), dtype=torch.bool, device=device),
            "target_mask": torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device),
        }
        
        # Set up for heatmap generation
        reflection_dict = empty_img_modality(reflection_dict, "heatmap")
        
        return reflection_dict
    
    def _prepare_correction_input(self, mod_dict, reflection_outputs, text_tokenizer):
        """
        Prepare input for the Correction stage based on Reflection outputs.
        
        Args:
            mod_dict: Dictionary containing the original input modalities.
            reflection_outputs: Dictionary containing the outputs from the Reflection stage.
            text_tokenizer: Text tokenizer.
            
        Returns:
            Modified mod_dict for the Correction stage.
        """
        # Create a copy to avoid modifying the original
        correction_dict = copy.deepcopy(mod_dict)
        
        # Use the generated heatmap from Reflection as input for Correction
        if "heatmap" in reflection_outputs:
            correction_dict["segmentation"] = reflection_outputs["heatmap"]
        
        # Prepend the Correction token to the text input if it exists
        if "text" in correction_dict:
            correction_marker = self.marker_tokens["correction"]
            # Get the token ID for the Correction marker
            correction_token_id = text_tokenizer.token_to_id(correction_marker)
            
            if correction_token_id is not None:
                # Get the tensor from the text modality
                text_tensor = correction_dict["text"]["tensor"]
                
                # Create a new tensor with the Correction token at the beginning
                new_tensor = torch.zeros_like(text_tensor)
                new_tensor[:, 0] = correction_token_id
                
                # Find the first non-zero token in the text (skip padding)
                non_zero_idx = torch.nonzero(text_tensor, as_tuple=True)[1]
                if len(non_zero_idx) > 0:
                    first_idx = non_zero_idx[0].item()
                    
                    # Copy the original text after the Correction token
                    new_tensor[:, 1:] = text_tensor[:, first_idx:]
                
                # Update the text tensor
                correction_dict["text"]["tensor"] = new_tensor
                
                # Update the input and target masks
                correction_dict["text"]["input_mask"][:, 0] = False  # The Correction token is input
                correction_dict["text"]["target_mask"][:, 0] = True  # The Correction token is not a target
        
        # Prepare target RGB modality for corrected image generation
        if "rgb" in correction_dict:
            batch_size, num_tokens = correction_dict["rgb"]["tensor"].shape
            device = correction_dict["rgb"]["tensor"].device
            
            # Initialize an empty target modality for corrected image
            correction_dict["rgb_corrected"] = {
                "tensor": torch.zeros((batch_size, num_tokens), dtype=torch.int64, device=device),
                "input_mask": torch.ones((batch_size, num_tokens), dtype=torch.bool, device=device),
                "target_mask": torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device),
            }
            
            # Set up for corrected image generation
            correction_dict = empty_img_modality(correction_dict, "rgb_corrected")
        
        return correction_dict
    
    @torch.no_grad()
    def generate_mcot(self, mod_dict, schedules, top_k=0.0, top_p=0.0, text_tokenizer=None, verbose=False, seed=None):
        """
        Generate using the MCOT four-stage process.
        
        Args:
            mod_dict: Dictionary containing the input modalities.
            schedules: Dictionary of generation schedules for each stage.
            top_k: Top-k value for sampling.
            top_p: Top-p value for sampling.
            text_tokenizer: Text tokenizer.
            verbose: Whether to print verbose output.
            seed: Random seed for reproducibility.
            
        Returns:
            Dictionary with outputs from all four stages.
        """
        if verbose:
            print("Starting MCOT generation")
        
        # Set random seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        # Check if text tokenizer is provided
        if text_tokenizer is None:
            raise ValueError("Text tokenizer is required for MCOT generation")
        
        results = {
            "planning": {},
            "acting": {},
            "reflection": {},
            "correction": {}
        }
        
        # 1. Planning Stage
        if verbose:
            print("Stage 1: Planning")
        
        planning_input = self._prepare_planning_input(mod_dict, text_tokenizer)
        planning_outputs = self.sampler.generate(
            planning_input, 
            schedules["planning"],
            top_k=top_k,
            top_p=top_p,
            text_tokenizer=text_tokenizer,
            verbose=verbose,
            seed=seed
        )
        results["planning"] = planning_outputs
        
        # 2. Acting Stage
        if verbose:
            print("Stage 2: Acting")
        
        acting_input = self._prepare_acting_input(mod_dict, planning_outputs, text_tokenizer)
        acting_outputs = self.sampler.generate(
            acting_input, 
            schedules["acting"],
            top_k=top_k,
            top_p=top_p,
            text_tokenizer=text_tokenizer,
            verbose=verbose,
            seed=seed
        )
        results["acting"] = acting_outputs
        
        # 3. Reflection Stage
        if verbose:
            print("Stage 3: Reflection")
        
        reflection_input = self._prepare_reflection_input(mod_dict, acting_outputs, text_tokenizer)
        reflection_outputs = self.sampler.generate(
            reflection_input, 
            schedules["reflection"],
            top_k=top_k,
            top_p=top_p,
            text_tokenizer=text_tokenizer,
            verbose=verbose,
            seed=seed
        )
        results["reflection"] = reflection_outputs
        
        # 4. Correction Stage
        if verbose:
            print("Stage 4: Correction")
        
        correction_input = self._prepare_correction_input(mod_dict, reflection_outputs, text_tokenizer)
        correction_outputs = self.sampler.generate(
            correction_input, 
            schedules["correction"],
            top_k=top_k,
            top_p=top_p,
            text_tokenizer=text_tokenizer,
            verbose=verbose,
            seed=seed
        )
        results["correction"] = correction_outputs
        
        if verbose:
            print("MCOT generation completed")
        
        return results


def build_mcot_generation_schedules(modality_info):
    """
    Build the generation schedules for the four MCOT stages.
    
    Args:
        modality_info: Dictionary with information about the modalities.
        
    Returns:
        Dictionary of schedules for the four stages.
    """
    schedules = {}
    
    # 1. Planning Stage (captions and bounding boxes)
    schedules["planning"] = {
        "target_modality": ["caption", "bbox"],
        "decoding_type": "autoregressive",
        "decoding_steps": 100,  # Adjust as needed
        "token_decoding_schedule": "linear",
        "temp": 0.7,
        "temp_schedule": "onex",
        "cfg_scale": 1.0,
        "cfg_schedule": "linear",
        "cfg_grow_conditioning": False
    }
    
    # 2. Acting Stage (image generation)
    schedules["acting"] = {
        "target_modality": "rgb_target",
        "decoding_type": "maskgit",
        "decoding_steps": 12,  # Typically 12 steps for image generation
        "token_decoding_schedule": "cosine",
        "temp": 4.0,
        "temp_schedule": "linear",
        "cfg_scale": 7.5,
        "cfg_schedule": "linear",
        "cfg_grow_conditioning": True
    }
    
    # 3. Reflection Stage (heatmap generation)
    schedules["reflection"] = {
        "target_modality": "heatmap",
        "decoding_type": "maskgit",
        "decoding_steps": 8,  # Fewer steps for heatmap
        "token_decoding_schedule": "cosine",
        "temp": 3.0,
        "temp_schedule": "linear",
        "cfg_scale": 5.0,
        "cfg_schedule": "linear",
        "cfg_grow_conditioning": True
    }
    
    # 4. Correction Stage (corrected image generation)
    schedules["correction"] = {
        "target_modality": "rgb_corrected",
        "decoding_type": "maskgit",
        "decoding_steps": 12,  # Same as image generation
        "token_decoding_schedule": "cosine",
        "temp": 4.0,
        "temp_schedule": "linear",
        "cfg_scale": 7.5,
        "cfg_schedule": "linear",
        "cfg_grow_conditioning": True
    }
    
    return schedules 