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
MINT Paper Reflection-Guided Mask Generation
Implements targeted mask generation for correction step based on artifact analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math
from PIL import Image, ImageDraw


class ReflectionGuidedMaskGenerator(nn.Module):
    """
    Generates targeted masks for correction based on reflection analysis.
    Implements MINT paper's reflection-guided correction methodology.
    """
    
    def __init__(self, image_size: int = 512, min_mask_size: int = 16, 
                 max_mask_size: int = 128, mask_expansion_ratio: float = 1.2):
        super().__init__()
        
        self.image_size = image_size
        self.min_mask_size = min_mask_size
        self.max_mask_size = max_mask_size
        self.mask_expansion_ratio = mask_expansion_ratio
        
        # Mask refinement network
        self.mask_refiner = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Context-aware adjustment network
        self.context_network = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=5, padding=2),  # Image + initial mask
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, artifact_heatmaps: Dict[str, torch.Tensor], 
                confidence_map: torch.Tensor,
                image: Optional[torch.Tensor] = None,
                strategy: str = 'adaptive') -> Dict[str, torch.Tensor]:
        """
        Generate reflection-guided masks for targeted correction.
        
        Args:
            artifact_heatmaps: Dictionary of artifact-specific heatmaps
            confidence_map: Confidence scores for each spatial location
            image: Optional input image for context-aware masking
            strategy: Masking strategy ('adaptive', 'conservative', 'aggressive')
            
        Returns:
            Dictionary containing generated masks and metadata
        """
        batch_size = confidence_map.shape[0]
        
        # Combine artifact heatmaps with confidence weighting
        combined_heatmap = torch.zeros_like(confidence_map)
        for artifact_type, heatmap in artifact_heatmaps.items():
            # Weight by artifact severity and confidence
            weighted_heatmap = heatmap * confidence_map
            combined_heatmap = torch.maximum(combined_heatmap, weighted_heatmap)
        
        # Apply strategy-specific thresholding
        if strategy == 'conservative':
            threshold = 0.7
            expansion_factor = 1.0
        elif strategy == 'aggressive':
            threshold = 0.3
            expansion_factor = 1.5
        else:  # adaptive
            threshold = 0.5
            expansion_factor = self.mask_expansion_ratio
        
        # Generate initial masks
        initial_masks = (combined_heatmap > threshold).float()
        
        # Apply mask refinement
        refined_masks = self.mask_refiner(initial_masks.unsqueeze(1)).squeeze(1)
        
        # Context-aware adjustment if image is provided
        if image is not None:
            context_input = torch.cat([
                F.interpolate(image, size=(self.image_size, self.image_size), mode='bilinear'),
                refined_masks.unsqueeze(1)
            ], dim=1)
            context_adjustment = self.context_network(context_input).squeeze(1)
            refined_masks = refined_masks * context_adjustment
        
        # Generate brushstroke masks (compatible with BrushNet)
        brushstroke_masks = self._generate_brushstroke_masks(
            refined_masks, expansion_factor=expansion_factor
        )
        
        # Apply size constraints
        final_masks = self._apply_size_constraints(brushstroke_masks)
        
        return {
            'initial_masks': initial_masks,
            'refined_masks': refined_masks,
            'brushstroke_masks': brushstroke_masks,
            'final_masks': final_masks,
            'combined_heatmap': combined_heatmap
        }
    
    def _generate_brushstroke_masks(self, base_masks: torch.Tensor, 
                                  expansion_factor: float = 1.2) -> torch.Tensor:
        """Generate brushstroke-style masks from base masks."""
        batch_size = base_masks.shape[0]
        brushstroke_masks = torch.zeros_like(base_masks)
        
        for b in range(batch_size):
            mask = base_masks[b].cpu().numpy()
            brushstroke_mask = self._create_brushstroke_from_mask(mask, expansion_factor)
            brushstroke_masks[b] = torch.from_numpy(brushstroke_mask).to(base_masks.device)
        
        return brushstroke_masks
    
    def _create_brushstroke_from_mask(self, mask: np.ndarray, 
                                    expansion_factor: float) -> np.ndarray:
        """Create brushstroke pattern from binary mask."""
        h, w = mask.shape
        brushstroke_mask = np.zeros_like(mask)
        
        # Find connected components
        from scipy import ndimage
        labeled_mask, num_features = ndimage.label(mask > 0.5)
        
        for feature_id in range(1, num_features + 1):
            component_mask = (labeled_mask == feature_id)
            
            # Get component properties
            coords = np.where(component_mask)
            if len(coords[0]) == 0:
                continue
                
            center_y, center_x = np.mean(coords[0]), np.mean(coords[1])
            
            # Determine brush shape based on component shape
            extent_y = np.max(coords[0]) - np.min(coords[0])
            extent_x = np.max(coords[1]) - np.min(coords[1])
            
            if extent_y > extent_x * 1.5:
                brush_shape = 'vertical'
            elif extent_x > extent_y * 1.5:
                brush_shape = 'horizontal'
            else:
                brush_shape = 'circular'
            
            # Generate brush strokes
            brush_mask = self._generate_brush_strokes(
                h, w, center_x, center_y, extent_x, extent_y, 
                brush_shape, expansion_factor
            )
            
            brushstroke_mask = np.maximum(brushstroke_mask, brush_mask)
        
        return brushstroke_mask
    
    def _generate_brush_strokes(self, h: int, w: int, center_x: float, center_y: float,
                              extent_x: float, extent_y: float, shape: str,
                              expansion_factor: float) -> np.ndarray:
        """Generate brush stroke patterns."""
        brush_mask = np.zeros((h, w))
        
        # Create PIL image for drawing
        pil_img = Image.fromarray((brush_mask * 255).astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)
        
        if shape == 'circular':
            radius = max(extent_x, extent_y) * expansion_factor / 2
            radius = max(self.min_mask_size // 2, min(radius, self.max_mask_size // 2))
            
            # Draw circular brush
            draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], fill=255)
            
        elif shape == 'horizontal':
            width = extent_x * expansion_factor
            height = max(self.min_mask_size, extent_y * expansion_factor)
            width = max(self.min_mask_size, min(width, self.max_mask_size))
            height = min(height, self.max_mask_size)
            
            # Draw elongated horizontal brush
            draw.ellipse([
                center_x - width/2, center_y - height/2,
                center_x + width/2, center_y + height/2
            ], fill=255)
            
        else:  # vertical
            width = max(self.min_mask_size, extent_x * expansion_factor)
            height = extent_y * expansion_factor
            width = min(width, self.max_mask_size)
            height = max(self.min_mask_size, min(height, self.max_mask_size))
            
            # Draw elongated vertical brush
            draw.ellipse([
                center_x - width/2, center_y - height/2,
                center_x + width/2, center_y + height/2
            ], fill=255)
        
        brush_mask = np.array(pil_img) / 255.0
        return brush_mask
    
    def _apply_size_constraints(self, masks: torch.Tensor) -> torch.Tensor:
        """Apply minimum and maximum size constraints to masks."""
        batch_size = masks.shape[0]
        constrained_masks = torch.zeros_like(masks)
        
        for b in range(batch_size):
            mask = masks[b].cpu().numpy()
            
            # Apply morphological operations to enforce size constraints
            from scipy import ndimage
            from skimage import morphology
            
            # Remove small components
            min_area = (self.min_mask_size ** 2) // 4
            cleaned_mask = morphology.remove_small_objects(
                mask > 0.5, min_size=min_area
            ).astype(float)
            
            # Limit maximum component size
            labeled_mask, num_features = ndimage.label(cleaned_mask > 0.5)
            for feature_id in range(1, num_features + 1):
                component = (labeled_mask == feature_id)
                if component.sum() > (self.max_mask_size ** 2):
                    # Erode large components
                    eroded = ndimage.binary_erosion(component, iterations=2)
                    cleaned_mask[component] = 0
                    cleaned_mask[eroded] = 1
            
            constrained_masks[b] = torch.from_numpy(cleaned_mask).to(masks.device)
        
        return constrained_masks


def create_reflection_guided_mask_generator(config: Dict[str, Any]) -> ReflectionGuidedMaskGenerator:
    """Factory function to create reflection-guided mask generator."""
    return ReflectionGuidedMaskGenerator(
        image_size=config.get('image_size', 512),
        min_mask_size=config.get('min_mask_size', 16),
        max_mask_size=config.get('max_mask_size', 128),
        mask_expansion_ratio=config.get('mask_expansion_ratio', 1.2)
    )
