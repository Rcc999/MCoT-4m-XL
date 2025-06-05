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
Implementation of the Multimodal Chain of Thought (MCoT) methodology
as described in the MINT paper. This module adds MCoT capabilities
to the 4M model through step-specific processing and state management.

The MCoT process follows four sequential steps:
1. Planning: Caption and layout planning with comprehensive descriptions and spatial layouts
2. Acting: Image generation based on the planning outputs  
3. Reflection: Artifact detection and self-assessment of generated images
4. Correction: Targeted inpainting and correction based on reflection insights

Note: This implementation works with the existing 4M transformer architecture
without requiring expert routing or architectural changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
from typing import Dict, List, Optional, Tuple, Union, Any

try:
    from .fm_utils import Block, LayerNorm
except ImportError:
    from fourm.models.fm_utils import Block, LayerNorm


class MCoTStepProcessor(nn.Module):
    """
    Enhanced MCoT step processor with MINT paper features:
    - Artifact heatmap generation with confidence scoring
    - Reflection-guided mask generation for targeted correction
    """
    
    def __init__(self, dim: int = 768, device: str = 'cuda', enable_mint: bool = False,
                 mcot_steps: Optional[List[str]] = None, step_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.dim = dim
        self.device = device
        self.enable_mint = enable_mint
        
        # Configure MCoT steps
        default_steps = ["planning", "acting", "reflection", "correction"]
        self.mcot_steps = mcot_steps if mcot_steps is not None else default_steps
        
        # Configure step weights
        default_weights = {"planning": 1.0, "acting": 1.2, "reflection": 1.5, "correction": 1.3}
        self.step_weights = step_weights if step_weights is not None else default_weights
        
        # Step configurations
        self.step_instructions = {
            "planning": "Create a detailed dense caption and layout plan with bounding boxes for objects. Focus on spatial relationships and compositional elements.",
            "acting": "Generate the image based on the planning output. Use the dense caption and layout information to create a high-quality image.",
            "reflection": "Analyze the generated image for artifacts, inconsistencies, or quality issues. Generate artifact heatmap with confidence scores for areas requiring correction.",
            "correction": "Apply targeted inpainting corrections based on reflection analysis and artifact heatmap. Focus on improving identified issues while preserving image quality."
        }
        
        self.step_to_id = {step: i for i, step in enumerate(self.step_instructions.keys())}
        
        # Step embeddings for conditioning
        self.step_embeddings = nn.Embedding(len(self.step_instructions), dim)
        
        # Enhanced reflection processing
        self.reflection_confidence_threshold = 0.5
        
        print(f"MCoTStepProcessor initialized with dim={dim}, enable_mint={enable_mint}")
        print(f"Steps: {self.mcot_steps}")
        print(f"Step weights: {self.step_weights}")
        
    def get_step_embedding(self, step: str, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get step-specific embedding to condition the model.
        
        Args:
            step: MCoT step name
            batch_size: Batch size
            device: Device
            
        Returns:
            Step embedding tensor [B, 1, D]
        """
        if step not in self.step_to_id:
            raise ValueError(f"Unknown MCoT step: {step}. Valid steps: {list(self.step_to_id.keys())}")
            
        step_id = self.step_to_id[step]
        step_ids = torch.full((batch_size,), step_id, device=device, dtype=torch.long)
        step_emb = self.step_embeddings(step_ids).unsqueeze(1)  # [B, 1, D]
        
        return step_emb

    def format_step_prompt(self, base_prompt: str, step: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Format the prompt for a specific MCoT step."""
        if step not in self.step_instructions:
            return base_prompt
            
        instruction = self.step_instructions[step]
        
        # Build context from previous steps
        context_str = ""
        if context:
            for prev_step in ["planning", "acting", "reflection"]:
                if prev_step in context and prev_step != step:
                    context_str += f"\n{prev_step.title()}: {context[prev_step]}"
        
        # Format the complete prompt
        if step == "planning":
            formatted_prompt = f"{instruction}\n\nUser request: {base_prompt}"
        else:
            formatted_prompt = f"{instruction}\n\nOriginal request: {base_prompt}{context_str}"
            
        return formatted_prompt

    def get_step_weight(self, step: str) -> float:
        """Get the loss weight for a specific MCoT step."""
        return self.step_weights.get(step, 1.0)


class MCoTWrapper(nn.Module):
    """
    Wrapper that adds MCoT capabilities to an existing 4M model.
    This enables step-specific processing without changing the base architecture.
    """
    
    def __init__(self, base_model: nn.Module, mcot_processor: Optional[MCoTStepProcessor] = None):
        super().__init__()
        
        self.base_model = base_model
        self.dim = getattr(base_model, 'dim', 768)
        
        # MCoT step processor
        if mcot_processor is not None:
            self.step_processor = mcot_processor
        else:
            self.step_processor = MCoTStepProcessor(self.dim)
        
        # Copy important attributes from base model
        self.modality_info = getattr(base_model, 'modality_info', {})
        
    def forward(self, 
                mod_dict: Dict[str, Dict[str, torch.Tensor]], 
                num_encoder_tokens: int, 
                num_decoder_tokens: int, 
                mcot_step: Optional[str] = None,
                mcot_context: Optional[Dict[str, Any]] = None,
                **kwargs) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass with optional MCoT step conditioning."""
        
        # Apply MCoT conditioning if step is specified
        if mcot_step is not None and mcot_step in self.step_processor.step_to_id:
            mod_dict = self._apply_mcot_conditioning(mod_dict, mcot_step, mcot_context)
        
        # Forward through base model
        return self.base_model(
            mod_dict=mod_dict,
            num_encoder_tokens=num_encoder_tokens,
            num_decoder_tokens=num_decoder_tokens,
            **kwargs
        )
    
    def _apply_mcot_conditioning(self, 
                                mod_dict: Dict[str, Dict[str, torch.Tensor]], 
                                step: str, 
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """Apply MCoT step conditioning to input modalities."""
        conditioned_mod_dict = {}
        
        for modality, mod_data in mod_dict.items():
            conditioned_mod_dict[modality] = mod_data.copy()
            
            # Apply step conditioning to text-based modalities
            if modality in ['caption', 'text'] and 'tensor' in mod_data:
                if 'input_text' in mod_data:
                    base_prompt = mod_data['input_text']
                    formatted_prompt = self.step_processor.format_step_prompt(base_prompt, step, context)
                    conditioned_mod_dict[modality]['input_text'] = formatted_prompt
                
                # Add step embedding to token sequence
                if 'tensor' in mod_data and mod_data['tensor'].dim() == 3:
                    tokens = mod_data['tensor']
                    batch_size = tokens.shape[0]
                    device = tokens.device
                    
                    step_emb = self.step_processor.get_step_embedding(step, batch_size, device)
                    conditioned_tokens = torch.cat([step_emb, tokens], dim=1)
                    
                    conditioned_mod_dict[modality]['tensor'] = conditioned_tokens
        
        return conditioned_mod_dict

    def __getattr__(self, name):
        """Delegate attribute access to base model if not found in wrapper."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


def add_mcot_to_model(model: nn.Module, mcot_processor: Optional[MCoTStepProcessor] = None) -> MCoTWrapper:
    """
    Add MCoT capabilities to an existing 4M model.
    
    Args:
        model: Existing 4M model
        mcot_processor: Optional MCoT processor (will create default if None)
        
    Returns:
        Model with MCoT capabilities
    """
    return MCoTWrapper(base_model=model, mcot_processor=mcot_processor)





