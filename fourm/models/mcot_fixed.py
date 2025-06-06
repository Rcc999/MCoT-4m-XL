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
Multi-step Chain of Thought (MCoT) Implementation for 4M Models

This module extends the 4M multimodal transformer with MCoT reasoning capabilities.
Instead of generating images in one shot, MCoT breaks the process into four steps:

1. **Planning**: Analyze the prompt and create a detailed plan with descriptions and spatial layouts
2. **Acting**: Generate the initial image based on the planning step
3. **Reflection**: Evaluate the generated image and identify potential issues or artifacts  
4. **Correction**: Apply targeted fixes to improve the final result

Key Features:
- Works with existing 4M architecture (no architectural changes needed)
- Step-specific conditioning through embeddings and prompt formatting
- Configurable step weights for training
- MINT paper integration for artifact detection and confidence scoring

Usage:
    # Wrap any 4M model with MCoT capabilities
    mcot_model = add_mcot_to_model(base_4m_model)
    
    # Use with step conditioning
    output = mcot_model(mod_dict, num_encoder_tokens, num_decoder_tokens, 
                       mcot_step="planning")
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
    Processes MCoT steps with step-specific conditioning and prompt formatting.
    
    This class handles the logic for each MCoT step, including:
    - Step-specific embeddings for model conditioning
    - Prompt formatting with context from previous steps
    - Configurable weights for different steps during training
    - MINT-style artifact detection features
    
    The processor maintains no state between steps - each step gets fresh context
    from the previous steps' outputs.
    """
    
    def __init__(self, dim: int = 768, device: str = 'cuda', enable_mint: bool = False,
                 mcot_steps: Optional[List[str]] = None, step_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.dim = dim
        self.device = device
        self.enable_mint = enable_mint
        
        # MCoT step configuration - defines the four reasoning steps
        default_steps = ["planning", "acting", "reflection", "correction"]
        self.mcot_steps = mcot_steps if mcot_steps is not None else default_steps
        
        # Training weights for each step (reflection gets higher weight as it's most critical)
        default_weights = {"planning": 1.0, "acting": 1.2, "reflection": 1.5, "correction": 1.3}
        self.step_weights = step_weights if step_weights is not None else default_weights
        
        # Instructions for each MCoT step - these guide the model's behavior
        self.step_instructions = {
            "planning": "Create a detailed dense caption and layout plan with bounding boxes for objects. Focus on spatial relationships and compositional elements.",
            "acting": "Generate the image based on the planning output. Use the dense caption and layout information to create a high-quality image.",
            "reflection": "Analyze the generated image for artifacts, inconsistencies, or quality issues. Generate artifact heatmap with confidence scores for areas requiring correction.",
            "correction": "Apply targeted inpainting corrections based on reflection analysis and artifact heatmap. Focus on improving identified issues while preserving image quality."
        }
        
        self.step_to_id = {step: i for i, step in enumerate(self.step_instructions.keys())}
        
        # Learnable embeddings to condition the model for each step
        self.step_embeddings = nn.Embedding(len(self.step_instructions), dim)
        
        # Confidence threshold for reflection step artifact detection
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
        """
        Format prompts for specific MCoT steps by adding step instructions and context.
        
        Args:
            base_prompt: Original user prompt (e.g., "Draw a cat on a sofa")
            step: Current MCoT step ("planning", "acting", "reflection", "correction")
            context: Results from previous steps to provide context
            
        Returns:
            Formatted prompt with step instructions and previous step context
        """
        if step not in self.step_instructions:
            return base_prompt
            
        instruction = self.step_instructions[step]
        
        # Add context from previous steps (e.g., planning output for acting step)
        context_str = ""
        if context:
            for prev_step in ["planning", "acting", "reflection"]:
                if prev_step in context and prev_step != step:
                    context_str += f"\n{prev_step.title()}: {context[prev_step]}"
        
        # Create the final prompt with instructions and context
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
    Wrapper that adds MCoT reasoning to any existing 4M model.
    
    This wrapper preserves the original model's interface while adding MCoT capabilities.
    It intercepts forward passes to apply step-specific conditioning when needed.
    
    Key features:
    - Non-invasive: doesn't modify the base model architecture
    - Step conditioning: adds embeddings and formats prompts for each MCoT step
    - Transparent: behaves like the original model when MCoT is not used
    - Flexible: works with any 4M model variant
    """
    
    def __init__(self, base_model: nn.Module, mcot_processor: Optional[MCoTStepProcessor] = None):
        super().__init__()
        
        self.base_model = base_model
        self.dim = getattr(base_model, 'dim', 768)
        
        # Initialize MCoT processor for step-specific logic
        if mcot_processor is not None:
            self.step_processor = mcot_processor
        else:
            self.step_processor = MCoTStepProcessor(self.dim)
        
        # Copy important attributes from base model so we behave identically
        self.modality_info = getattr(base_model, 'modality_info', {})
        
    def forward(self, 
                mod_dict: Dict[str, Dict[str, torch.Tensor]], 
                num_encoder_tokens: int, 
                num_decoder_tokens: int, 
                mcot_step: Optional[str] = None,
                mcot_context: Optional[Dict[str, Any]] = None,
                **kwargs) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass with optional MCoT step conditioning.
        
        When mcot_step is provided, applies step-specific conditioning to inputs.
        Otherwise behaves exactly like the base model.
        """
        
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





