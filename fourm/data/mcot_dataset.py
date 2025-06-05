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
Custom dataset loader for MCoT training that handles the directory structure
created by the wget script: train/example_N/image.jpg and mcot_annotations.json
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

import fourm.utils as utils
from fourm.data.masking import UnifiedMasking
from fourm.data.modality_transforms import UnifiedDataTransform


class MCoTDatasetFromDirectory(Dataset):
    """
    PyTorch Dataset that loads MCoT data from the directory structure
    created by mcot_dataset_wget.py:
    
    data_path/
    ├── train/
    │   ├── example_0/
    │   │   ├── image.jpg
    │   │   └── mcot_annotations.json
    │   ├── example_1/
    │   │   ├── image.jpg
    │   │   └── mcot_annotations.json
    │   └── ...
    └── val/
        ├── example_0/
        │   ├── image.jpg
        │   └── mcot_annotations.json
        └── ...
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 modality_info: Dict = None,
                 all_domains: List[str] = None,
                 transform=None):
        
        self.data_path = Path(data_path)
        self.split = split
        self.modality_info = modality_info or {}
        self.all_domains = all_domains or ['rgb', 'caption']
        self.transform = transform
        
        # Expect proper directory structure with train/ and val/ subdirectories
        split_dir = self.data_path / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Find all example directories
        self.example_dirs = []
        for example_dir in split_dir.iterdir():
            if example_dir.is_dir() and (example_dir / "mcot_annotations.json").exists():
                self.example_dirs.append(example_dir)
        
        if not self.example_dirs:
            raise ValueError(f"No valid examples found in {split_dir}")
        
        # Sort for reproducible ordering
        self.example_dirs.sort(key=lambda x: x.name)
        
        print(f"Found {len(self.example_dirs)} examples in {split} split")
    
    def __len__(self) -> int:
        return len(self.example_dirs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load example from MCoT directory structure."""
        example_dir = self.example_dirs[idx]
        
        # Load annotations
        annotations_file = example_dir / "mcot_annotations.json"
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        # Load image
        image_file = example_dir / "image.jpg"
        if image_file.exists():
            image = Image.open(image_file).convert('RGB')
        else:
            # Create placeholder image if missing
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Create sample dictionary in the format expected by 4M
        sample_dict = {}
        
        # Add RGB modality
        if 'rgb' in self.all_domains:
            sample_dict['rgb'] = image
        
        # Add caption modality - use final response or planning step
        if 'caption' in self.all_domains:
            caption_text = annotations.get('final_response', 
                          annotations.get('correction',
                          annotations.get('planning', 'No caption available')))
            sample_dict['caption'] = caption_text
        
        # Apply transforms if provided
        if self.transform:
            sample_dict = self.transform(sample_dict)
        
        return sample_dict


def build_mcot_pretraining_dataset(
        data_path: str, 
        all_domains: List[str], 
        modality_info: Dict, 
        modality_transforms: Dict, 
        image_augmenter, 
        text_tokenizer, 
        input_tokens_range: Tuple[int, int], 
        target_tokens_range: Tuple[int, int],
        sampling_weights=None,
        split: str = 'train'):
    """
    Build MCoT dataset for 4M pretraining.
    
    Args:
        data_path: Path to MCoT dataset directory
        all_domains: List of modalities to use  
        modality_info: Modality information dictionary
        modality_transforms: Transform dictionary for each modality
        image_augmenter: Image augmentation pipeline
        text_tokenizer: Text tokenizer
        input_tokens_range: Range for input token budget
        target_tokens_range: Range for target token budget
        sampling_weights: Sampling weights for Dirichlet distributions
        split: Dataset split ('train' or 'val')
        
    Returns:
        MCoT dataset for 4M training
    """
    
    # Create unified transform pipeline - use transforms.Compose, not torch.nn.Sequential
    transform = transforms.Compose([
        UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
        UnifiedMasking(modality_info=modality_info, text_tokenizer=text_tokenizer,
                       input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
                       sampling_weights=sampling_weights)
    ])
    
    return MCoTDatasetFromDirectory(
        data_path=data_path,
        split=split,
        modality_info=modality_info,
        all_domains=all_domains,
        transform=transform
    )


def build_mcot_pretraining_dataloader(
        data_path: str,
        all_domains: List[str], 
        modality_info: Dict, 
        modality_transforms: Dict, 
        image_augmenter, 
        text_tokenizer, 
        input_tokens_range: Tuple[int, int], 
        target_tokens_range: Tuple[int, int],
        num_gpus: int, 
        num_workers: int, 
        batch_size: int, 
        epoch_size: Optional[int] = None,
        sampling_weights=None,
        split: str = 'train'):
    """
    Build MCoT dataloader for 4M pretraining.
    
    Args:
        data_path: Path to MCoT dataset directory
        all_domains: List of modalities to use
        modality_info: Modality information dictionary
        modality_transforms: Transform dictionary for each modality
        image_augmenter: Image augmentation pipeline
        text_tokenizer: Text tokenizer
        input_tokens_range: Range for input token budget
        target_tokens_range: Range for target token budget
        num_gpus: Number of GPUs
        num_workers: Number of data loader workers
        batch_size: Batch size
        epoch_size: Optional epoch size
        sampling_weights: Sampling weights for Dirichlet distributions
        split: Dataset split ('train' or 'val')
        
    Returns:
        DataLoader for MCoT training
    """
    
    dataset = build_mcot_pretraining_dataset(
        data_path=data_path,
        all_domains=all_domains,
        modality_info=modality_info,
        modality_transforms=modality_transforms,
        image_augmenter=image_augmenter,
        text_tokenizer=text_tokenizer,
        input_tokens_range=input_tokens_range,
        target_tokens_range=target_tokens_range,
        sampling_weights=sampling_weights,
        split=split
    )
    
    # Create distributed sampler
    if split == 'train':
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_gpus, rank=utils.get_rank(), 
            shuffle=True, drop_last=True
        )
    else:
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_gpus, rank=utils.get_rank(), 
            shuffle=False, drop_last=False
        )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader
