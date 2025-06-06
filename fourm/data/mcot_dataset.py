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
MCoT Dataset Implementation for 4M Model Training

This dataset handles the directory structure used by MCoT training data:
- Each example is a folder containing an image and MCoT annotations
- Supports both train/val splits
- Integrates with 4M's modality system for seamless training

Expected directory structure:
    dataset_dir/
        train/
            example_001/
                image.jpg
                mcot_annotations.json
            example_002/
                ...
        val/
            example_003/
                ...

The annotations JSON contains the MCoT step outputs (planning, acting, reflection, correction).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from fourm.data.modality_info import MODALITY_INFO
from fourm.data.modality_transforms import get_transform_key


class MCoTDatasetFromDirectory(Dataset):
    """
    PyTorch Dataset that loads MCoT training data from directory structure.
    
    This dataset is designed for the standard MCoT data format where each example
    is stored as a folder containing:
    - image.jpg: The reference image
    - mcot_annotations.json: MCoT step outputs and metadata
    
    The dataset automatically discovers all valid examples and handles missing files
    gracefully (creates placeholder images if needed).
    
    Args:
        data_path: Root directory containing train/val folders
        all_domains: List of modalities to include (e.g., ['rgb', 'caption', 'planning'])
        split: 'train' or 'val'
        Other args: Standard 4M dataset parameters for transforms and tokenization
    """
    
    def __init__(self, 
                 data_path: str,
                 all_domains: List[str],
                 modality_info: Dict[str, Any],
                 modality_transforms: Dict[str, Any],
                 image_augmenter: Any,
                 text_tokenizer: Any,
                 input_tokens_range: Tuple[int, int],
                 target_tokens_range: Tuple[int, int],
                 split: str = 'train'):
        
        self.data_path = Path(data_path)
        self.all_domains = all_domains
        self.modality_info = modality_info
        self.modality_transforms = modality_transforms
        self.image_augmenter = image_augmenter
        self.text_tokenizer = text_tokenizer
        self.input_tokens_range = input_tokens_range
        self.target_tokens_range = target_tokens_range
        self.split = split
        
        # Discover all valid example directories (must contain mcot_annotations.json)
        split_dir = self.data_path / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        self.example_dirs = []
        for example_dir in split_dir.iterdir():
            if example_dir.is_dir() and (example_dir / "mcot_annotations.json").exists():
                self.example_dirs.append(example_dir)
        
        if not self.example_dirs:
            raise ValueError(f"No valid examples found in {split_dir}")
        
        print(f"Found {len(self.example_dirs)} MCoT examples in {split} split")
    
    def __len__(self) -> int:
        return len(self.example_dirs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load a single MCoT example from the directory structure.
        
        Returns a dictionary containing only the modalities requested in all_domains.
        This allows flexible training - you can train on just images + captions,
        or include specific MCoT steps, etc.
        """
        example_dir = self.example_dirs[idx]
        
        # Load MCoT annotations (contains step outputs and metadata)
        annotations_file = example_dir / "mcot_annotations.json"
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        # Load image (create placeholder if missing to avoid crashes)
        image_file = example_dir / "image.jpg"
        if image_file.exists():
            image = Image.open(image_file).convert('RGB')
        else:
            # Create placeholder image if file missing
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Build modality dict with only requested modalities (all_domains)
        mod_dict = {}
        
        # Include image if any RGB modality is requested 
        if any(domain.startswith('rgb') for domain in self.all_domains):
            mod_dict['rgb'] = image  # Raw PIL image, will be processed by image_augmenter in UnifiedDataTransform
        
        # Include text caption if requested
        if 'caption' in self.all_domains:
            mod_dict['caption'] = annotations.get('prompt', annotations.get('caption', 'Default caption'))
        
        # Include MCoT step outputs only if specifically requested
        # This allows training on subsets of the MCoT process
        mcot_mappings = {
            'planning': annotations.get('planning', 'Planning step data'),
            'acting': annotations.get('acting', 'Acting step data'), 
            'reflection': annotations.get('reflection', 'Reflection step data'),
            'correction': annotations.get('correction', 'Correction step data'),
            'final_response': annotations.get('final_response', 'Final response data')
        }
        
        for mcot_key, mcot_value in mcot_mappings.items():
            if mcot_key in self.all_domains:
                mod_dict[mcot_key] = mcot_value
        
        return mod_dict