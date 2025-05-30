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


def create_artifact_heatmap_generator(model_config: Dict[str, Any]) -> Optional[nn.Module]:
    """Factory function to create artifact heatmap generator."""
    try:
        import sys
        import os
        mcot_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'mcot_data')
        if mcot_data_path not in sys.path:
            sys.path.insert(0, mcot_data_path)
        
        from artifact_heatmap import ArtifactHeatmapGenerator
        return ArtifactHeatmapGenerator(
            image_size=model_config.get("image_size", 512),
            patch_size=model_config.get("patch_size", 16),
            feature_dim=model_config.get("feature_dim", 768),
            num_heads=model_config.get("num_heads", 8)
        )
    except ImportError as e:
        print(f"Warning: Could not import ArtifactHeatmapGenerator: {e}")
        return None


def create_reflection_guided_mask_generator(model_config: Dict[str, Any]) -> Optional[nn.Module]:
    """Factory function to create reflection-guided mask generator."""
    try:
        import sys
        import os
        mcot_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'mcot_data')
        if mcot_data_path not in sys.path:
            sys.path.insert(0, mcot_data_path)
        
        from reflection_guided_masks import ReflectionGuidedMaskGenerator
        return ReflectionGuidedMaskGenerator(
            image_size=model_config.get("image_size", 512),
            min_mask_size=model_config.get("min_mask_size", 16),
            max_mask_size=model_config.get("max_mask_size", 128),
            mask_expansion_ratio=model_config.get("mask_expansion_ratio", 1.2)
        )
    except ImportError as e:
        print(f"Warning: Could not import ReflectionGuidedMaskGenerator: {e}")
        return None


class MCoTStepProcessor(nn.Module):
    """
    Enhanced MCoT step processor with MINT paper features:
    - Artifact heatmap generation with confidence scoring
    - Reflection-guided mask generation for targeted correction
    """
    
    def __init__(self, dim: int, device: str = 'cuda'):
        super().__init__()
        
        self.dim = dim
        self.device = device
        
        # Original step configurations
        self.step_instructions = {
            "planning": "Create a detailed dense caption and layout plan with bounding boxes for objects. Focus on spatial relationships and compositional elements.",
            "acting": "Generate the image based on the planning output. Use the dense caption and layout information to create a high-quality image.",
            "reflection": "Analyze the generated image for artifacts, inconsistencies, or quality issues. Generate artifact heatmap with confidence scores for areas requiring correction.",
            "correction": "Apply targeted inpainting corrections based on reflection analysis and artifact heatmap. Focus on improving identified issues while preserving image quality."
        }
        
        self.step_to_id = {step: i for i, step in enumerate(self.step_instructions.keys())}
        
        # Step embeddings for conditioning
        self.step_embeddings = nn.Embedding(len(self.step_instructions), dim)
        
        # Enhanced MINT paper features
        model_config = {
            "image_size": 512,
            "patch_size": 16,
            "feature_dim": dim,
            "num_heads": 8,
            "min_mask_size": 16,
            "max_mask_size": 128,
            "mask_expansion_ratio": 1.2
        }
        
        # Initialize MINT paper components (optional, will be None if import fails)
        self.artifact_heatmap_generator = create_artifact_heatmap_generator(model_config)
        self.reflection_guided_mask_generator = create_reflection_guided_mask_generator(model_config)
        
        # Enhanced reflection processing
        self.reflection_confidence_threshold = 0.5
        
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

    def process_reflection_with_mint(self, image: torch.Tensor, reflection_text: str) -> Dict[str, Any]:
        """
        Enhanced reflection processing using MINT paper features.
        
        Args:
            image: Input image tensor [B, C, H, W]
            reflection_text: Base reflection text
            
        Returns:
            Enhanced reflection results with artifact analysis
        """
        results = {
            "base_reflection": reflection_text,
            "mint_features_available": False
        }
        
        if self.artifact_heatmap_generator is not None:
            try:
                # Generate artifact heatmaps
                heatmap_results = self.artifact_heatmap_generator(image)
                
                # Generate enhanced reflection text
                enhanced_reflections = self.artifact_heatmap_generator.generate_reflection_text(
                    heatmap_results, threshold=self.reflection_confidence_threshold
                )
                
                results.update({
                    "mint_features_available": True,
                    "artifact_heatmaps": heatmap_results["artifact_heatmaps"],
                    "confidence_map": heatmap_results["confidence_map"],
                    "enhanced_reflections": enhanced_reflections,
                    "attention_weights": heatmap_results["attention_weights"]
                })
                
            except Exception as e:
                print(f"Warning: MINT artifact detection failed: {e}")
                
        return results

    def generate_correction_masks(self, artifact_analysis: Dict[str, Any], 
                                image: Optional[torch.Tensor] = None,
                                strategy: str = 'adaptive') -> Dict[str, Any]:
        """
        Generate targeted correction masks using reflection-guided approach.
        
        Args:
            artifact_analysis: Results from reflection processing
            image: Optional input image for context
            strategy: Masking strategy
            
        Returns:
            Generated masks and metadata
        """
        results = {
            "masks_generated": False,
            "strategy": strategy
        }
        
        if (self.reflection_guided_mask_generator is not None and 
            artifact_analysis.get("mint_features_available", False)):
            try:
                mask_results = self.reflection_guided_mask_generator(
                    artifact_heatmaps=artifact_analysis["artifact_heatmaps"],
                    confidence_map=artifact_analysis["confidence_map"],
                    image=image,
                    strategy=strategy
                )
                
                results.update({
                    "masks_generated": True,
                    "correction_masks": mask_results["final_masks"],
                    "brushstroke_masks": mask_results["brushstroke_masks"],
                    "mask_metadata": {
                        "initial_masks": mask_results["initial_masks"],
                        "refined_masks": mask_results["refined_masks"]
                    }
                })
                
            except Exception as e:
                print(f"Warning: MINT mask generation failed: {e}")
                
        return results


class MCoTWrapper(nn.Module):
    """
    Wrapper that adds MCoT capabilities to an existing 4M model.
    This enables step-specific processing without changing the base architecture.
    """
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        
        self.base_model = base_model
        self.dim = getattr(base_model, 'dim', 768)
        
        # MCoT step processor with MINT features
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


def add_mcot_to_model(model: nn.Module) -> MCoTWrapper:
    """
    Add MCoT capabilities to an existing 4M model.
    
    Args:
        model: Existing 4M model
        
    Returns:
        Model with MCoT capabilities
    """
    return MCoTWrapper(base_model=model)



