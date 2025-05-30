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
MINT Paper Artifact Heatmap Generation
Implements artifact detection and confidence scoring as described in the MINT paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class ArtifactHeatmapGenerator(nn.Module):
    """
    Generates artifact heatmaps with confidence scores for reflection step.
    Based on MINT paper methodology for spatial artifact localization.
    """
    
    def __init__(self, image_size: int = 512, patch_size: int = 16, 
                 feature_dim: int = 768, num_heads: int = 8):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_patches = (image_size // patch_size) ** 2
        
        # Multi-head attention for spatial artifact localization
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Artifact classification heads
        self.artifact_types = [
            'blurry_regions', 'color_inconsistency', 'texture_artifacts',
            'object_distortion', 'lighting_issues', 'composition_problems'
        ]
        
        self.artifact_classifiers = nn.ModuleDict({
            artifact_type: nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid()
            ) for artifact_type in self.artifact_types
        })
        
        # Confidence scoring network
        self.confidence_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Patch feature encoder
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(3, feature_dim // 4, kernel_size=patch_size//4, stride=patch_size//4),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 4, feature_dim // 2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=2, stride=2),
        )
        
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate artifact heatmap with confidence scores.
        
        Args:
            image: Input image [B, C, H, W]
            
        Returns:
            Dictionary containing heatmaps and confidence scores
        """
        batch_size = image.shape[0]
        
        # Extract patch features
        patch_features = self.patch_encoder(image)  # [B, D, H_p, W_p]
        patch_features = patch_features.flatten(2).transpose(1, 2)  # [B, N_patches, D]
        
        # Apply spatial attention for context
        attended_features, attention_weights = self.spatial_attention(
            patch_features, patch_features, patch_features
        )
        
        # Generate artifact-specific heatmaps
        artifact_scores = {}
        for artifact_type in self.artifact_types:
            scores = self.artifact_classifiers[artifact_type](attended_features)  # [B, N_patches, 1]
            artifact_scores[artifact_type] = scores.squeeze(-1)  # [B, N_patches]
        
        # Generate confidence scores
        confidence_scores = self.confidence_network(attended_features).squeeze(-1)  # [B, N_patches]
        
        # Reshape to spatial format
        h_patches = w_patches = int(np.sqrt(self.num_patches))
        
        heatmaps = {}
        for artifact_type, scores in artifact_scores.items():
            heatmap = scores.view(batch_size, h_patches, w_patches)
            # Upsample to full image resolution
            heatmap = F.interpolate(
                heatmap.unsqueeze(1), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(1)
            heatmaps[artifact_type] = heatmap
        
        # Reshape confidence scores
        confidence_map = confidence_scores.view(batch_size, h_patches, w_patches)
        confidence_map = F.interpolate(
            confidence_map.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        return {
            'artifact_heatmaps': heatmaps,
            'confidence_map': confidence_map,
            'attention_weights': attention_weights,
            'patch_features': attended_features
        }
    
    def generate_reflection_text(self, heatmap_results: Dict[str, torch.Tensor], 
                               threshold: float = 0.5) -> List[str]:
        """
        Generate reflection text based on artifact heatmaps.
        
        Args:
            heatmap_results: Results from forward pass
            threshold: Confidence threshold for artifact detection
            
        Returns:
            List of reflection texts for each image in batch
        """
        batch_size = heatmap_results['confidence_map'].shape[0]
        reflection_texts = []
        
        for b in range(batch_size):
            detected_artifacts = []
            
            for artifact_type, heatmap in heatmap_results['artifact_heatmaps'].items():
                artifact_map = heatmap[b]
                confidence = heatmap_results['confidence_map'][b]
                
                # Find regions with high artifact scores and confidence
                mask = (artifact_map > threshold) & (confidence > threshold)
                if mask.sum() > 0:
                    coverage = mask.float().mean().item()
                    detected_artifacts.append({
                        'type': artifact_type,
                        'coverage': coverage,
                        'max_score': artifact_map[mask].max().item()
                    })
            
            # Generate reflection text
            if detected_artifacts:
                text = "Upon reflection, I notice several issues in the generated image: "
                for artifact in detected_artifacts:
                    artifact_name = artifact['type'].replace('_', ' ')
                    if artifact['coverage'] > 0.1:
                        text += f"significant {artifact_name} (affecting {artifact['coverage']:.1%} of image), "
                    else:
                        text += f"localized {artifact_name}, "
                text = text.rstrip(", ") + ". These areas require targeted correction."
            else:
                text = "The generated image appears to have good overall quality with no major artifacts detected."
            
            reflection_texts.append(text)
        
        return reflection_texts


def create_artifact_heatmap_generator(config: Dict[str, Any]) -> ArtifactHeatmapGenerator:
    """Factory function to create artifact heatmap generator."""
    return ArtifactHeatmapGenerator(
        image_size=config.get('image_size', 512),
        patch_size=config.get('patch_size', 16),
        feature_dim=config.get('feature_dim', 768),
        num_heads=config.get('num_heads', 8)
    )
