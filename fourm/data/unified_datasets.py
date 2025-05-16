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
import copy
import io
import itertools
import os
import re
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import braceexpand
import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import IterableDataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from webdataset.filters import pipelinefilter, reraise_exception
from webdataset.handlers import warn_and_continue
import json

# Standard COCO 2017 Categories (91 classes, though typically only 80 are used in detections)
# This mapping is commonly used. IDs are as per COCO standard.
COCO_CATEGORIES_2017_ID_TO_NAME = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie',
    33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
    # IDs 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91 are typically not in common detection sets.
}

try:
    # Optionally load huggingface datasets
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node
except ImportError:
    print("Huggingface datasets not installed. Please install with `pip install datasets`.")

from fourm.data.masking import TransferMasking, UnifiedMasking
from fourm.data.modality_transforms import (CropSettingsTransform, IdentityTransform,
                                      MaskTransform, UnifiedDataTransform,
                                      get_transform_key,
                                      RGBTransform, DepthTransform, CaptionTransform, DetectionTransform, TokTransform)
from fourm.data.multimodal_dataset_folder import MultiModalDatasetFolder
from fourm.utils.dist import get_rank, get_world_size

# REMOVE THE CIRCULAR IMPORT
# from .unified_datasets import get_mcot_planning_data_pipeline, get_mcot_acting_data_pipeline

# The get_mcot_planning_data_pipeline and get_mcot_acting_data_pipeline functions are defined later in this file

# Assume access to the MCoT special token IDs, perhaps passed via text_tokenizer or a config
# For now, defining them here as placeholders. These should be obtained from the tokenizer.
# Example: PLANNING_START_TOKEN_ID = text_tokenizer.token_to_id("[PLANNING_START]")
# These would ideally be part of a shared config or loaded with the tokenizer.

def _tokenize_bboxes_for_planning(bbox_data, image_width, image_height, text_tokenizer, coord_bins=1000):
    """
    Tokenizes bounding box data for the planning stage.
    Assumes bbox_data is a list of lists: [[x_abs, y_abs, w_abs, h_abs, category_id], ...]
    Assumes text_tokenizer has COCO category names and coord bin tokens ("<coord_0>"..."<coord_N-1>")
    """
    bbox_tokens = []
    coco_categories = {} # In a real scenario, load this from COCO metadata or config
    # Example mapping - this should be comprehensive for COCO
    # For now, assume category_id is directly a string name or can be mapped.
    # If category_id is an int, it needs to be mapped to a name first.
    # This is a placeholder: You'll need a proper mapping from COCO category_id to category_name
    # that exists in your text_tokenizer.
    # E.g., if text_tokenizer was trained with "person", "car", etc.

    # Placeholder: coco_category_id_to_name = {1: "person", 2: "bicycle", ...}
    # This mapping should be available, e.g. passed in or loaded from a file.
    # For this example, let's assume category_id is already a name string in bbox_data
    # or that text_tokenizer can handle integer IDs if they are special tokens.

    for bbox in bbox_data:
        if len(bbox) < 5: continue # Basic validation

        x_abs, y_abs, w_abs, h_abs = bbox[0], bbox[1], bbox[2], bbox[3]
        category_name_or_id = bbox[4] # This needs to be a string like "person" or "cat"

        # Convert to x_min, y_min, x_max, y_max and normalize
        x_min_norm = x_abs / image_width
        y_min_norm = y_abs / image_height
        x_max_norm = (x_abs + w_abs) / image_width
        y_max_norm = (y_abs + h_abs) / image_height

        # Digitize to bins
        x_min_bin = int(x_min_norm * (coord_bins -1))
        y_min_bin = int(y_min_norm * (coord_bins -1))
        x_max_bin = int(x_max_norm * (coord_bins -1))
        y_max_bin = int(y_max_norm * (coord_bins -1))
        
        try:
            # Attempt to get token ID for category name
            # This assumes category_name_or_id is a string like "person"
            # If it's an int ID that's already a token, this would need adjustment
            # or a direct mapping to a token_id if text_tokenizer.token_to_id expects strings
            cat_token_id = text_tokenizer.token_to_id(str(category_name_or_id))
            if cat_token_id is None:
                # Fallback or error if category not in tokenizer
                # For now, skip if not found, but ideally handle this (e.g., map to <UNK> or error)
                print(f"Warning: Category '{category_name_or_id}' not found in tokenizer. Skipping box.")
                continue

            bbox_tokens.extend([
                cat_token_id,
                text_tokenizer.token_to_id(f"<coord_{max(0, min(x_min_bin, coord_bins-1))}>"),
                text_tokenizer.token_to_id(f"<coord_{max(0, min(y_min_bin, coord_bins-1))}>"),
                text_tokenizer.token_to_id(f"<coord_{max(0, min(x_max_bin, coord_bins-1))}>"),
                text_tokenizer.token_to_id(f"<coord_{max(0, min(y_max_bin, coord_bins-1))}>"),
            ])
        except Exception as e:
            print(f"Error tokenizing bbox for category {category_name_or_id}: {e}")
            continue # Skip this bounding box
            
    # Filter out None tokens that might result from missing coord tokens
    # (though this shouldn't happen if coord_bins tokens are guaranteed)
    valid_bbox_tokens = [token for token in bbox_tokens if token is not None]
    return valid_bbox_tokens


def load_and_preprocess_planning_sample(raw_sample, text_tokenizer, modality_info):
    """
    Processes a raw sample from MS-COCO webdataset for the Planning MCoT stage.
    Output: A dictionary of modalities for UnifiedMasking.
            The [PLANNING_START] token is prefixed to the input caption/prompt.
    Expects raw_sample to contain:
    - 'image.jpg' or 'image.png': PIL Image (after wds.decode)
    - 'caption_prompt.txt': Raw text for the input prompt (string)
    - 'target_plan_text.txt': Raw text for the target textual part of the plan (string)
    - 'bboxes.json': JSON string containing list of bboxes [x,y,w,h, category_name_or_id] and image_width, image_height
                     e.g. {"image_width": 640, "image_height": 480, "annotations": [[x,y,w,h,"cat"], ...]}
    - '__key__': Sample key (string)
    """
    # 1. Extract image, caption prompt, target plan text, and bounding box data
    pil_image = raw_sample.get("image.jpg") or raw_sample.get("image.png") # PIL Image
    input_prompt_text = raw_sample.get("caption_prompt.txt", "")
    target_plan_text = raw_sample.get("target_plan_text.txt", "")
    
    bbox_json_str = raw_sample.get("bboxes.json", "{}")
    try:
        bbox_content = json.loads(bbox_json_str)
        # It's common for COCO annotations to be under an "annotations" key,
        # and image dimensions might be separate or within the image file itself.
        # Assuming bboxes.json provides dimensions.
        image_width = bbox_content.get("image_width", pil_image.width if pil_image else 1) # Default if not in json
        image_height = bbox_content.get("image_height", pil_image.height if pil_image else 1) # Default if not in json
        # Check if annotations are nested (common in COCO original format)
        # or if bbox_content is directly the list of bboxes.
        # For this example, assume "annotations" key holds the list of bbox data.
        raw_bbox_data = bbox_content.get("annotations", []) 
        if not isinstance(raw_bbox_data, list): # If "annotations" is not there, maybe bbox_content is the list
             if isinstance(bbox_content, list): raw_bbox_data = bbox_content
             else: raw_bbox_data = []

    except json.JSONDecodeError:
        print(f"Warning: Could not decode bboxes.json for sample {raw_sample.get('__key__')}")
        raw_bbox_data = []
        image_width, image_height = (pil_image.width if pil_image else 1), (pil_image.height if pil_image else 1)


    # 2. Get [PLANNING_START] token ID
    planning_start_token_id = text_tokenizer.token_to_id("[PLANNING_START]")
    if planning_start_token_id is None:
        raise ValueError("[PLANNING_START] token not found in tokenizer.")

    # 3. Tokenize the input prompt and prefix with [PLANNING_START]
    prompt_tokens = text_tokenizer.encode(input_prompt_text).ids
    # Key for this should match modality_info for caption, e.g. 'caption'
    # And what UnifiedMasking.determine_mcot_stage expects.
    # Assuming 'caption' is the modality name for text sequences.
    input_prompt_with_prefix = [planning_start_token_id] + prompt_tokens
    
    # 4. Tokenize target plan text (this will be one of the target modalities)
    # Assuming 'caption' is also used for this target text modality name.
    target_plan_text_tokens = text_tokenizer.encode(target_plan_text).ids

    # 5. Tokenize target bounding boxes (this will be another target modality)
    # Assuming 'det' is the modality name for bbox sequences.
    # The coord_bins should ideally come from modality_info['det'] or config.
    coord_bins = modality_info.get('det', {}).get('coord_bins', 1000) # Example: get from modality_info if defined
    target_bbox_layout_tokens = _tokenize_bboxes_for_planning(raw_bbox_data, image_width, image_height, text_tokenizer, coord_bins)
    
    processed_sample = {
        # Input for planning stage
        # The key 'caption' should match a modality name in modality_info
        'caption': torch.tensor(input_prompt_with_prefix, dtype=torch.long),
        
        # Target modalities for planning stage
        # Using 'target_caption' and 'target_det' to distinguish from input 'caption' if UnifiedMasking needs it,
        # or could reuse 'caption' and 'det' if UnifiedMasking handles input/target distinction.
        # For now, let's assume UnifiedMasking uses the dict structure and can differentiate.
        # The actual target keys will depend on how UnifiedMasking is configured for 'planning' stage.
        # Let's assume the modality names themselves are used, and UnifiedMasking sorts out targets.
        # The `out_domains` in the config will specify 'caption-det' for planning.
        # `UnifiedMasking` will select these as targets.

        # These are the actual outputs of the planning stage
        # Key name 'caption' for textual plan, 'det' for layout plan
        # This is a bit confusing. Let's clarify:
        # Input to Planning: 'caption_prompt' (text) + 'image' (implicitly via 'rgb@224')
        # Output of Planning: 'plan_text' (text) + 'plan_bboxes' (det tokens)

        # Revised structure based on common patterns:
        # Input modalities are named directly by their modality type (e.g. 'rgb@224', 'caption')
        # Target modalities are also named by their type. UnifiedMasking uses in_domains/out_domains.
        
        # Input for the model (passed to UnifiedDataTransform then UnifiedMasking)
        # 'rgb@224' or 'rgb': PIL Image, to be processed by UnifiedDataTransform
        # 'caption': Tokenized input prompt with [PLANNING_START]
    }
    if pil_image:
        # Use the actual image modality key from modality_info, e.g., 'rgb@224'
        image_mod_key = next((k for k, v in modality_info.items() if v.get('type') == 'img' and 'rgb' in k), 'rgb@224')
        processed_sample[image_mod_key] = pil_image
    
    processed_sample[modality_info.get('caption', {}).get('name', 'caption')] = torch.tensor(input_prompt_with_prefix, dtype=torch.long)

    # These are the targets for the planning stage, as specified by 'out_domains' = 'caption-det' (example)
    # We need to ensure these keys match the modality names for caption and det.
    # Assuming target text uses 'caption' modality and target bboxes use 'det' modality.
    # To avoid key collision if input prompt is also 'caption', we might need a convention.
    # However, UnifiedMasking.forward receives a single sample dict.
    # It splits it into input_dict and target_dict based on in_domains and out_domains.
    # So, if 'caption' is in out_domains, it will be treated as a target.
    # If 'caption' is in in_domains, it's an input. If in both, it's input & target.
    
    # Let's ensure distinct keys for clarity if necessary, or rely on UnifiedMasking.
    # For MCoT, the initial input 'caption' (prompt with [PLANNING_START]) is for stage determination
    # and also part of the input to the first stage.
    # Targets for planning:
    processed_sample['mcot_target_for_caption'] = torch.tensor(target_plan_text_tokens, dtype=torch.long) # This should map to 'caption' for target.
    processed_sample['mcot_target_for_det'] = torch.tensor(target_bbox_layout_tokens, dtype=torch.long) # This should map to 'det' for target.

    # The keys in processed_sample should be actual modality names for UnifiedDataTransform.
    # Then UnifiedMasking will use in_domains/out_domains to pick inputs/targets.
    # Let's structure `processed_sample` with keys that are modality names.
    # `mcot_target_for_caption` should be placed under the key for the 'caption' modality IF 'caption' is an out_domain.
    # `mcot_target_for_det` should be placed under the key for the 'det' modality IF 'det' is an out_domain.

    # Final refined structure for what `load_and_preprocess_planning_sample` returns:
    # This dict is then passed to UnifiedDataTransform and then to UnifiedMasking collate_fn.
    final_sample = {}
    if pil_image:
        # Find the primary RGB image key from modality_info
        img_key = next((k for k, v in modality_info.items() if v.get('type') == 'img' and 'rgb' in k), 'rgb@224')
        final_sample[img_key] = pil_image
    
    # Input text prompt (with MCoT prefix)
    # This will be used as input if 'caption' is in in_domains.
    # If 'caption' is also in out_domains, then `mcot_target_for_caption` are its target.
    final_sample['caption'] = torch.tensor(input_prompt_with_prefix, dtype=torch.long)

    # If 'caption' is an output domain, these are its target tokens.
    # This assumes UnifiedMasking collate_fn can handle providing separate targets
    # for a modality that is also an input. Often, for seq2seq, the input sequence
    # itself becomes the target sequence for teacher forcing after masking.
    # For MCoT planning, the output plan text is DIFFERENT from input prompt.
    # So we need a way to provide these mcot_target_for_caption to 'caption' modality for loss calculation.
    #
    # A robust way: UnifiedMasking receives the full dict.
    # `in_domains` (e.g. ['caption_prompt', 'rgb@224']) are model inputs.
    # `out_domains` (e.g. ['plan_text', 'plan_bboxes']) are model outputs.
    # This requires modality_info to have entries like 'caption_prompt', 'plan_text', 'plan_bboxes'.
    # 'caption_prompt' and 'plan_text' can share vocab with 'caption'.
    # 'plan_bboxes' can share vocab with 'det'.

    # Let's assume a simpler model for now where out_domains are 'caption' and 'det',
    # and UnifiedMasking knows that for 'planning' stage:
    # - input 'caption' is the prompt.
    # - target for 'caption' modality is `mcot_target_for_caption`.
    # - target for 'det' modality is `mcot_target_for_det`.
    # This might be achieved by `load_and_preprocess_planning_sample` adding specific keys
    # that `UnifiedMasking` then uses for this stage:
    final_sample['mcot_target_for_caption'] = torch.tensor(target_plan_text_tokens, dtype=torch.long)
    final_sample['mcot_target_for_det'] = torch.tensor(target_bbox_layout_tokens, dtype=torch.long)
    # And `UnifiedMasking` for 'planning' stage would know to use these for loss calculation
    # against outputs of 'caption' and 'det' modalities respectively.

    return final_sample


def load_and_preprocess_acting_sample(raw_sample, text_tokenizer, modality_info):
    """
    Processes a raw sample from MS-COCO webdataset for the Acting MCoT stage.
    Assumes acting stage takes image, a textual plan, and a bbox plan (from planning stage output)
    and generates a final detailed caption.
    
    Expects raw_sample to contain:
    - 'image.jpg' or 'image.png': PIL Image
    - 'plan_text.txt': Textual part of the plan (string)
    - 'plan_bboxes.json': JSON string for bbox part of the plan {"image_width": w, "image_height": h, "annotations": [[x,y,w,h,cat],...]}
    - 'target_final_caption.txt': Ground truth final caption for the acting stage (string)
    - '__key__': Sample key (string)
    """
    pil_image = raw_sample.get("image.jpg") or raw_sample.get("image.png")
    plan_text = raw_sample.get("plan_text.txt", "")
    plan_bboxes_json_str = raw_sample.get("plan_bboxes.json", "{}")
    target_final_caption_text = raw_sample.get("target_final_caption.txt", "")

    try:
        bbox_content = json.loads(plan_bboxes_json_str)
        image_width = bbox_content.get("image_width", pil_image.width if pil_image else 1)
        image_height = bbox_content.get("image_height", pil_image.height if pil_image else 1)
        raw_plan_bbox_data = bbox_content.get("annotations", [])
        if not isinstance(raw_plan_bbox_data, list):
            if isinstance(bbox_content, list): raw_plan_bbox_data = bbox_content
            else: raw_plan_bbox_data = []
    except json.JSONDecodeError:
        print(f"Warning: Could not decode plan_bboxes.json for acting sample {raw_sample.get('__key__')}")
        raw_plan_bbox_data = []
        image_width, image_height = (pil_image.width if pil_image else 1), (pil_image.height if pil_image else 1)

    acting_start_token_id = text_tokenizer.token_to_id("[ACTING_START]")
    if acting_start_token_id is None: 
        raise ValueError("[ACTING_START] token not found in tokenizer.")

    coord_bins = modality_info.get('det', {}).get('coord_bins', 1000) # Reuse from det modality
    # Tokenize the input plan (text + bboxes)
    # This combined plan is the primary sequence input to the acting stage.
    # The key for this input sequence should be 'caption' or a similar sequence modality name.
    tokenized_plan_input = _tokenize_plan_for_acting(plan_text, raw_plan_bbox_data, image_width, image_height, text_tokenizer, coord_bins)
    acting_input_sequence_with_prefix = [acting_start_token_id] + tokenized_plan_input

    # Tokenize the target final caption
    target_final_caption_tokens = text_tokenizer.encode(target_final_caption_text).ids

    output_dict = {}
    if pil_image:
        img_key = next((k for k, v in modality_info.items() if v.get('type') == 'img' and 'rgb' in k and '@' in k), 'rgb@224')
        output_dict[img_key] = pil_image

    # Input sequence for acting (plan with prefix)
    # This uses the 'caption' modality as it's a sequence of text and bbox tokens.
    output_dict['caption'] = torch.tensor(acting_input_sequence_with_prefix, dtype=torch.long)

    # Target for the acting stage (final caption)
    # This also uses the 'caption' modality for its output.
    # UnifiedMasking needs to handle this based on 'acting' stage context.
    output_dict['mcot_target_for_caption'] = torch.tensor(target_final_caption_tokens, dtype=torch.long)
    
    return output_dict


def load_and_preprocess_reflection_sample(raw_sample, text_tokenizer, semantic_seg_vqvae_tokenizer, modality_info):
    """
    Processes a raw sample for the Reflection MCoT stage. RichHF-18K.
    [REFLECTION_START] token is prefixed.
    """
    reflection_start_token_id = text_tokenizer.token_to_id("[REFLECTION_START]")
    if reflection_start_token_id is None: raise ValueError("[REFLECTION_START] token not found.")

    # Inputs for Reflection: generated image (from acting), original prompt
    # raw_generated_image_from_acting = raw_sample.get('gen_image_from_acting.png')
    # generated_image_tokens_from_acting = some_image_vqvae.encode(raw_generated_image_from_acting)
    generated_image_tokens_from_acting = [] # Placeholder
    
    # raw_original_prompt = raw_sample.get('original_prompt.txt')
    # original_prompt_tokens = text_tokenizer.encode(raw_original_prompt).ids
    original_prompt_tokens = [] # Placeholder

    # Prefix for determine_mcot_stage
    reflection_prefix_tokens = [reflection_start_token_id]

    # Target for Reflection: heatmap tokens from artifact annotation
    # raw_artifact_annotation = raw_sample.get('artifact_annotation_richhf18k') # Data for heatmap
    # binary_mask_64x64 = preprocess_artifact_to_binary_mask(raw_artifact_annotation, size=(64,64))
    # heatmap_tokens = semantic_seg_vqvae_tokenizer.encode(binary_mask_64x64) # Placeholder
    heatmap_tokens = [] # Placeholder

    return {
        'reflection_prefix_tokens': torch.tensor(reflection_prefix_tokens, dtype=torch.long),
        'generated_image_tokens_from_acting': torch.tensor(generated_image_tokens_from_acting, dtype=torch.long),
        'original_prompt_tokens': torch.tensor(original_prompt_tokens, dtype=torch.long),
        'heatmap_tokens': torch.tensor(heatmap_tokens, dtype=torch.long),
    }


def load_and_preprocess_correction_sample(raw_sample, text_tokenizer, image_vqvae_tokenizer, modality_info):
    """
    Processes a raw sample for the Correction MCoT stage. COCO-Stuff.
    [CORRECTION_START] token is prefixed.
    """
    correction_start_token_id = text_tokenizer.token_to_id("[CORRECTION_START]")
    if correction_start_token_id is None: raise ValueError("[CORRECTION_START] token not found.")

    # Inputs for Correction: generated image, heatmap, inpaint_mask
    # raw_generated_image_from_acting = raw_sample.get('gen_image_from_acting.png')
    # generated_image_tokens_from_acting = image_vqvae_tokenizer.encode(raw_generated_image_from_acting)
    generated_image_tokens_from_acting = [] # Placeholder

    # heatmap_tokens_from_reflection = raw_sample.get('heatmap_tokens') # Already tokenized
    heatmap_tokens_from_reflection = [] # Placeholder

    # raw_inpaint_mask = raw_sample.get('inpaint_mask_cocostuff.png') # Binary mask
    # inpaint_mask_tokens = some_mask_tokenizer_or_representation(raw_inpaint_mask) # Placeholder
    inpaint_mask_tokens = [] # Placeholder
    
    # Prefix for determine_mcot_stage
    correction_prefix_tokens = [correction_start_token_id]

    # Target for Correction: corrected image region tokens
    # raw_target_corrected_image_region = raw_sample.get('corrected_region.png')
    # corrected_image_region_tokens = image_vqvae_tokenizer.encode(raw_target_corrected_image_region) # Placeholder
    corrected_image_region_tokens = [] # Placeholder

    return {
        'correction_prefix_tokens': torch.tensor(correction_prefix_tokens, dtype=torch.long),
        'generated_image_tokens_from_acting': torch.tensor(generated_image_tokens_from_acting, dtype=torch.long),
        'heatmap_tokens_from_reflection': torch.tensor(heatmap_tokens_from_reflection, dtype=torch.long),
        'inpaint_mask_tokens': torch.tensor(inpaint_mask_tokens, dtype=torch.long),
        'corrected_image_region_tokens': torch.tensor(corrected_image_region_tokens, dtype=torch.long),
    }


def build_fm_pretraining_dataset(
        data_path, all_domains, modality_info, modality_transforms, 
        image_augmenter, text_tokenizer, 
        input_tokens_range, target_tokens_range,
        sampling_weights=None):
    """Builds the FourM pre-training dataset based on the given arguments.
    This function should mainly used for smaller datasets (e.g. validation sets), 
    while large training sets should be loaded with build_wds_fm_pretraining_dataloader in webdataset format.
    
    Args:
        data_path: Path to the dataset.
        all_domains: List of all modalities to be used.
        modality_info: Dictionary containing information about the modalities.
        modality_transforms: Dictionary containing the transforms for each modality.
        image_augmenter: Image augmenter.
        text_tokenizer: Text tokenizer (for sequence modalities).
        input_tokens_range: Range of the input token budget.
        target_tokens_range: Range of the target token budget.
        sampling_weights: Sampling weights for the mixture of Dirichlet distributions.

    Returns:
        FourM pre-training dataset as a PyTorch Dataset.
    """

    transform = transforms.Compose([
        UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
        UnifiedMasking(modality_info=modality_info, text_tokenizer=text_tokenizer,
                       input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
                       sampling_weights=sampling_weights),
         ])

    # Remove vq domains that require a tokenizer
    modalities_without_vq = [mod for mod in all_domains if not modality_info[mod].get("requires_tokenizer", False)]
    # If we are using a pre-tokenized modality, we default to pre-computed crop settings
    if any([modality_info[domain].get("pretokenized", False) for domain in all_domains]):
        modalities_without_vq.append("crop_settings")
        modality_transforms = copy.deepcopy(modality_transforms)
        modality_transforms["crop_settings"] = CropSettingsTransform()

    modality_paths = {mod: modality_info[mod]['path'] for mod in modality_info if modality_info[mod].get('path', None) is not None}
    
    return MultiModalDatasetFolder(root=data_path, modalities=modalities_without_vq, modality_paths=modality_paths,
                                   modality_transforms=modality_transforms, transform=transform)


def build_fm_transfer_dataset(
    data_path, modality_info, transform, modality_transforms, all_domains, 
    load_mask_valid: bool = False, max_samples: Optional[int] = None, 
    pre_shuffle: bool = False, cache: bool = False):
    """Builds the FourM transfer dataset based on the given arguments.
    
    Args:
        data_path: Path to the dataset.
        modality_info: Dictionary containing information about the modalities.
        transform: Transform to be applied to the dataset.
        modality_transforms: Dictionary containing the transforms for each modality.
        all_domains: List of all modalities to be used.
        load_mask_valid: Whether to load the mask_valid "modality".
        max_samples: Maximum number of samples to load.
        pre_shuffle: Whether to shuffle the dataset before loading.
        cache: Whether to cache the dataset in memory.

    Returns:
        FourM transfer dataset as a PyTorch Dataset.
    """

    # Remove vq domains that require a tokenizer
    modalities_without_vq = [mod for mod in all_domains if not modality_info[mod].get("requires_tokenizer", False)]
    # If we are using a pre-tokenized modality, we default to pre-computed crop settings
    if any([modality_info[domain].get("pretokenized", False) for domain in all_domains]):
        modalities_without_vq.append("crop_settings")
        modality_transforms = copy.deepcopy(modality_transforms)
        modality_transforms["crop_settings"] = CropSettingsTransform()

    if load_mask_valid:
        modalities_without_vq.append("mask_valid")
        modality_transforms = copy.deepcopy(modality_transforms)
        modality_transforms["mask_valid"] = MaskTransform()

    modality_paths = {mod: modality_info[mod]['path'] for mod in modality_info if modality_info[mod].get('path', None) is not None}

    return MultiModalDatasetFolder(root=data_path, modalities=modalities_without_vq, modality_paths=modality_paths,
                                   modality_transforms=modality_transforms, transform=transform, max_samples=max_samples, 
                                   pre_shuffle=pre_shuffle, cache=cache)


### Webdatasets (wds) functions

def _keyless_map(data, f, handler=reraise_exception):
    """Map samples without adding __key__."""
    for sample in data:
        try:
            result = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        if result is None:
            continue
        yield result

map = pipelinefilter(_keyless_map)

def check_dots(s):
    if '.gz' in s:
        return s.count('.') == 2
    return s.count('.') == 1

def remove_ext_with_gz(s):
    if s.endswith('.gz'):
        s = s.replace(".gz", "")
    return os.path.splitext(s)[0]

def wds_decoder(key, value):
    if key == "png" or key.endswith(".png"):
        img = Image.open(io.BytesIO(value))
        return img
    elif key == "jpg" or key.endswith(".jpg"):
        img = Image.open(io.BytesIO(value))
        return img
    elif key == "jpeg" or key.endswith(".jpeg"):
        img = Image.open(io.BytesIO(value))
        return img
    elif key == 'npy' or key.endswith("npy"):
        content = np.load(io.BytesIO(value), allow_pickle=True)
        # try:
        #     content = np.load(io.BytesIO(value))
        # except:
        #     content = np.load(io.BytesIO(value), allow_pickle=True)
        return content
    elif key == "jpx" or key.endswith('.jpx'):
        img = Image.open(io.BytesIO(value))
        return img
    elif 'output' in key:
        return int(value)
    else:
        # If not an image, use the basic handlers (.txt, .json, .pickle, .npz, ...)
        # See https://github.com/webdataset/webdataset/blob/main/webdataset/autodecode.py
        return None

def repeat_fn(src, n_repeats=5):
    """
    Repeat each sample n_repeats times.
    E.g. A B C ... repeated 3 times becomes A A A B B B C C C ...
    Depending on the downstream application, a shuffle should be added after this.
    """
    for sample in src:
        for _ in range(n_repeats):
            yield sample
            
def remove_extensions(sample):
    """
    In webdatasets, we identify the type of a given modality by adding an extension
    in the form f"{modality_name}.{modality_extension}", e.g. "rgb.jpg" or "caption.json".
    This function removes them and returns a dictionary of {f"{modality_name}": modality}.
    """
    return {remove_ext_with_gz(k): v for k, v in sample.items()}

def filter_metadata(sample, metadata=['__key__', '__url__', 'file_name', 'class_name', 'class_idx']):
    """ Filters out non-modality entries specified in metadata when loading tar files with webdatasets. """
    return {k: v for k, v in sample.items() if k not in metadata}

def apply_modality_transforms(sample, modality_transforms):
    """ Applies a dictionary of modality-specific transforms to a dictionary of modalities. """
    return {k: (modality_transforms[get_transform_key(k)](v) if k in modality_transforms else v) for k, v in sample.items() }

def tok_to_int64(sample):
    """
    Pre-computed tokens are saved as int16, but we need them as int64 instead.
    """
    return {k: (v.astype('int64') if 'tok_' in k else v) for k, v in sample.items()}

def rename_modalities(sample, modality_paths):
    """
    Renames modalities to their corresponding names in modality_paths.
    """
    return {out_path: sample[loaded_path] for out_path, loaded_path in modality_paths.items()}

def extract_modality_names(s):
    # Regular expression pattern to match anything enclosed in '{' and '}', and comma separated
    pattern = r'\{([^}]*)\}'
    match = re.search(pattern, s)
    return match.group(1).split(',') if match else []

def identity(sample):
    """ Identity function that does nothing. """
    return sample

def multi_tarfile_samples(src_iter: Iterable[Dict[str, Any]], 
                          modality_name_map: Dict[str, str] = None, 
                          handler: Callable[[Exception], bool] = warn_and_continue):
    """Webdataset does not support splitting up shards by modality, so we need to do this manually.
    Usually, we would need to save all modalities in the same tar file, e.g. shard_root_train/{00000..12345}.tar, 
    where each shard contains 1000 samples and each sample contains all modalities.
    This is not flexible when adding new modalities, so we instead save each modality in a separate tar file,
    e.g. shard_root_train_rgb/{00000..12345}.tar, shard_root_train_caption/{00000..12345}.tar, etc., where each shard contains
    again 1000 samples, but each sample contains only one modality. All samples in all shards have to be aligned.

    This function takes an iterator over shard URLs, where we use brace expansion to specify multiple tar files per modality.
    E.g. shard_root_train_[rgb,caption]/00123.tar will be expanded to shard_root_train_rgb/00123.tar and shard_root_train_caption/00123.tar,
    and the samples from these two tar files will be combined into a single sample.

    Args:
        src_iter: Iterator over shards that *already brace expanded the shard numbers*, 
            e.g. {'url': 'shard_root_train_[rgb,caption]/00000.tar'}, {'url': 'shard_root_train_[rgb,caption]/00001.tar'}, ...
            This function will also work when no square braces for multiple modalities are used, e.g. {'url': 'shard_root_train/00000.tar'}, ...
            It can be a drop-in replacement for wds.tarfile_samples.
        modality_name_map: Optional dictionary specifying a mapping from modality folder names to arbitrary other names.
        handler: Function that handles exceptions. If it returns True, the shard is skipped. If it returns False, the function exits.

    Yields:
        Dictionary of aligned samples from all modalities.
    """
    for src in src_iter:
        
        # Multi tar file URLs use brace expansion with square braces
        multi_tar_urls = src['url'].translate(str.maketrans('[]', '{}'))
        modality_names = extract_modality_names(multi_tar_urls)
        if len(modality_names) == 0:
            # Case where multi-modal braceexpand is not used, e.g. shard_dir/shard00000.tar
            modality_names = [None]
            multi_tar_urls = [multi_tar_urls]
        elif len(modality_names) == 1:
            # Brace expand doesn't work with a single entry, e.g. shard_dir/[foo]/shard00000.tar
            multi_tar_urls = [multi_tar_urls.replace("{", "").replace("}", "")]
        else:
            # Remaining cases where multiple modalities are specified, e.g. shard_dir/[foo,bar]/shard00000.tar
            multi_tar_urls = list(braceexpand.braceexpand(multi_tar_urls))

        # Create tar iterators for shards of all modalities
        tar_iters = [wds.tarfile_samples([{'url': tar_url}]) for tar_url in multi_tar_urls]
        
        try:
            # Loop over these iterators in parallel and combine the tar files from different modalities
            for multi_tar_files in zip(*tar_iters):
                
                merged_dict = {}
                merged_dict['__key__'] = multi_tar_files[0]['__key__']
                merged_dict['__url__'] = src['url']
                
                for modality_name, modality_dict in zip(modality_names, multi_tar_files):
                    _key = modality_dict.pop('__key__')
                    _url = modality_dict.pop('__url__')

                    if _key != merged_dict['__key__']:
                        raise ValueError(f"Divergence detected! Trying to merge keys {_key} of {modality_name} and {merged_dict['__key__']} of merged_dict with modalities {merged_dict.keys()}.")
                        
                    tar_is_multimodal = len(modality_dict) > 1
                    for k, v in modality_dict.items():
                        if tar_is_multimodal or check_dots(k) or modality_name is None:
                            # We don't change the keys in the following cases:
                            # 1. The shard contains multiple modalities. Then they *have* to follow the idx.modality_id.ext convention
                            # 2. If any key contains a dot, this means it already has the idx.modality_id.ext format (idx. is already removed at this stage)
                            # 3. If the modality name is None, no modality folder was specified (see beginning of function)
                            merged_dict[k] = v
                        else:
                            mapped_name = modality_name if modality_name_map is None else modality_name_map.get(modality_name, modality_name)
                            merged_dict[f'{mapped_name}.{k}'] = v

                yield merged_dict

        except Exception as e:
            print(e)
            print(f"Exception occurred while processing {src['url']}.")
            if handler(e):
                print('Skipping shard...')
                continue
            else:
                break

def build_wds_fm_pretraining_dataloader(
        data_path, all_domains, modality_info, modality_transforms, image_augmenter, 
        text_tokenizer, input_tokens_range, target_tokens_range,
        num_gpus, num_workers, batch_size, epoch_size, sampling_weights=None, modality_name_map=None,
        shuffle_buffer_load=1000, shuffle_buffer_repeat=5000, n_repeats=5):
    """Builds the WebDataset FourM pre-training dataloader based on the given arguments.
    
    Args:
        data_path: Path to the dataset.
        all_domains: List of all modalities to be used.
        modality_info: Dictionary containing information about the modalities.
        modality_transforms: Dictionary containing the transforms for each modality.
        image_augmenter: Image augmenter.
        text_tokenizer: Text tokenizer (for sequence modalities).
        input_tokens_range: Range of the input token budget.
        target_tokens_range: Range of the target token budget.
        num_gpus: Number of GPUs.
        num_workers: Number of workers.
        batch_size: Batch size.
        epoch_size: Number of samples per "epoch". (Here, epoch refers to an interrupted training loop without evaluation or checkpointing).
        sampling_weights: Sampling weights for the mixture of Dirichlet distributions.
        modality_name_map: Optional dictionary specifying a mapping from modality folder names to arbitrary other names.
        shuffle_buffer_load: Shuffle buffer size when loading samples from tar files (first shuffle).
        shuffle_buffer_repeat: Shuffle buffer size after repeating samples (second shuffle).
        n_repeats: Number of times to repeat each sample.

    Returns:
        FourM pre-training dataloader as a WebDataset DataLoader.
    """

    modality_paths = {mod: modality_info[mod].get('path', None) or mod for mod in modality_info}

    # Remove vq domains that require a tokenizer
    modalities_without_vq = [mod for mod in all_domains if not modality_info[mod].get("requires_tokenizer", False)]
    # If we are using a pre-tokenized modality, we default to pre-computed crop settings
    if any([modality_info[domain].get("pretokenized", False) for domain in all_domains]):
        modalities_without_vq.append("crop_settings")
        modality_transforms = copy.deepcopy(modality_transforms)
        modality_transforms["crop_settings"] = CropSettingsTransform()
        modality_paths["crop_settings"] = "crop_settings"

    # Webdatasets always adds __key__ to the dictionary, so we add a transform that does nothing with it
    modality_transforms["__key__"] = IdentityTransform()

    transform = transforms.Compose([
        UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
        UnifiedMasking(modality_info=modality_info, text_tokenizer=text_tokenizer,
                       input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
                       sampling_weights=sampling_weights)
    ])
    
    datapipe = wds.DataPipeline(
        # Infinitely sample shards from the shard list with replacement. Each worker is seeded independently.
        wds.ResampledShards(data_path),
        partial(multi_tarfile_samples, modality_name_map=modality_name_map), # Extract individual samples from single or multi-modal tar files
        wds.shuffle(shuffle_buffer_load), # Shuffle with a buffer of given size
        wds.decode(wds_decoder), # Decode from bytes to PIL images, numpy arrays, etc.
        wds.filters.compose(partial(repeat_fn, n_repeats=n_repeats)), # Repeats each sample n times -> A A A B B B C C C ...
        wds.shuffle(shuffle_buffer_repeat), # Shuffle again with a buffer of given size
        wds.map(remove_extensions), # Remove "file extensions" from dictionary keys
        map(filter_metadata), # Remove non-task keys
        map(tok_to_int64), # Convert pre-computed tokens to int64
        map(partial(rename_modalities, modality_paths=modality_paths)), # Rename modalities to their corresponding names in modality_paths
        map(transform), # Apply data augmentation and masking
        wds.batched(batch_size, collation_fn=default_collate, partial=False)
            if batch_size is not None else map(identity), # Batching
    )

    if epoch_size is not None:
        batch_size_iter = batch_size if batch_size is not None else 1
        datapipe = datapipe.with_epoch(epoch_size // (num_gpus * num_workers * batch_size_iter)) # Pre-define iterator length
    
    if batch_size is not None:
        # Perform multi-threaded dataloading
        return wds.WebLoader(datapipe, num_workers=num_workers, batch_size=None)
    else:
        return datapipe


def build_wds_divae_dataloader(
    data_path, modality_info, modality_transforms, image_augmenter,  
    num_gpus, num_workers, batch_size, epoch_size, shuffle_buffer_load=1000, 
    shuffle_buffer_repeat=5000, n_repeats=1):

    modality_paths = {mod: modality_info[mod].get('path', None) or mod for mod in modality_info}

    # Webdatasets always adds __key__ to the dictionary, so we add a transform that does nothing with it
    modality_transforms["__key__"] = IdentityTransform()

    transform = UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter)

    datapipe = wds.DataPipeline(
        # Infinitely sample shards from the shard list with replacement. Each worker is seeded independently.
        wds.ResampledShards(data_path),
        multi_tarfile_samples, # Extract individual samples from single or multi-modal tar files
        wds.shuffle(shuffle_buffer_load), # Shuffle with a buffer of given size
        wds.decode(wds_decoder), # Decode from bytes to PIL images, numpy arrays, etc.
        wds.filters.compose(partial(repeat_fn, n_repeats=n_repeats)), # Repeats each sample n times -> A A A B B B C C C ...
        wds.shuffle(shuffle_buffer_repeat), # Shuffle again with a buffer of given size
        map(remove_extensions), # Remove "file extensions" from dictionary keys
        map(filter_metadata), # Remove non-task keys
        map(tok_to_int64), # Convert pre-computed tokens to int64
        map(partial(rename_modalities, modality_paths=modality_paths)), # Rename modalities to their corresponding names in modality_paths
        map(transform), # Apply data augmentation and masking
        wds.batched(batch_size, collation_fn=default_collate, partial=False)
            if batch_size is not None else map(identity), # Batching
    )

    if epoch_size is not None:
        batch_size_iter = batch_size if batch_size is not None else 1
        datapipe = datapipe.with_epoch(epoch_size // (num_gpus * num_workers * batch_size_iter)) # Pre-define iterator length
    
    if batch_size is not None:
        # Perform multi-threaded dataloading
        return wds.WebLoader(datapipe, num_workers=num_workers, batch_size=None)
    else:
        return datapipe


### Huggingface datasets functions

def text_to_caption(sample):
    """ Rename "text" to "caption". """
    return {'caption': sample['text']}


def prepare_hf_sample_for_domains(sample, expected_domains):
    """
    Prepares a HuggingFace dataset sample by mapping it to the expected domain names
    and removing all other metadata fields to prevent KeyError in transformations.
    
    Args:
        sample: Sample dictionary from HuggingFace dataset
        expected_domains: List of expected domain names, which may include specifiers (e.g., ['rgb@224', 'caption'])
        
    Returns:
        Processed sample dictionary with only the expected domains, or None if a required domain is missing
    """
    processed_sample = {}
    
    # Create a mapping from base domain (e.g., 'rgb') to full domain name (e.g., 'rgb@224')
    base_to_full_domain_map = {}
    for full_domain_name in expected_domains:
        base_name = full_domain_name.split('@')[0]
        base_to_full_domain_map[base_name] = full_domain_name

    # Handle 'rgb' based domains
    if 'rgb' in base_to_full_domain_map:
        full_rgb_domain_key = base_to_full_domain_map['rgb']
        if 'image' in sample:
            processed_sample[full_rgb_domain_key] = sample['image']
        else:
            # If 'rgb' (or variant like 'rgb@224') is expected but 'image' field is missing
            return None 
        
    # Handle 'caption' based domains
    if 'caption' in base_to_full_domain_map:
        full_caption_domain_key = base_to_full_domain_map['caption']
        if 'question' in sample:  # For VQAv2 dataset
            processed_sample[full_caption_domain_key] = sample['question']
        elif 'text' in sample:  # For other datasets
            processed_sample[full_caption_domain_key] = sample['text']
        else:
            # If 'caption' is expected but neither 'question' nor 'text' is present
            return None
    
    # Final check: ensure all expected domains are now keys in processed_sample
    if all(domain_key in processed_sample for domain_key in expected_domains):
        return processed_sample
    else:
        # Skip this sample if any expected domain is missing
        return None


def build_huggingface_pretraining_dataloader(
        data_path, all_domains, modality_info, modality_transforms, image_augmenter, 
        text_tokenizer, input_tokens_range, target_tokens_range,
        num_gpus, num_workers, batch_size, epoch_size, split,
        streaming=True, rename_text_to_caption=True
        , shuffle_buffer_load=10_000, shuffle_seed=0):

    # Load huggingface dataset and split samples across workers. Shuffle samples in each worker
    dataset = load_dataset(data_path, split=split, streaming=streaming,trust_remote_code=True)
    dataset = split_dataset_by_node(dataset, rank=get_rank(), world_size=get_world_size())
    dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_load)

    modality_info = {mod: modality_info[mod] for mod in modality_info if mod in all_domains}
    
    transform = transforms.Compose([
        UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
        UnifiedMasking(modality_info=modality_info, text_tokenizer=text_tokenizer,
                       input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range)
    ])
    
    # Create a partial function to prepare the sample with the expected domains
    prepare_fn = partial(prepare_hf_sample_for_domains, expected_domains=all_domains)
    
    # Define a filter function that removes None samples (which indicate invalid samples)
    def filter_none(sample):
        return sample is not None
    
    datapipe = wds.DataPipeline(
        dataset,
        map(prepare_fn),  # First apply our custom preparation function to filter and map fields
        wds.select(filter_none),  # Skip any samples that were marked as None/invalid
        map(transform),  # Apply data augmentation and masking
        wds.batched(batch_size, collation_fn=default_collate, partial=False)
            if batch_size is not None else map(identity),  # Batching
    )

    datapipe.n_shards = dataset.n_shards
    num_workers = min(num_workers, dataset.n_shards)

    if epoch_size is not None:
        batch_size_iter = batch_size if batch_size is not None else 1
        datapipe = datapipe.with_epoch(epoch_size // (num_gpus * num_workers * batch_size_iter))  # Pre-define iterator length

    if batch_size is not None:
        # Perform multi-threaded dataloading
        return wds.WebLoader(datapipe, num_workers=num_workers, batch_size=None)
    else:
        return datapipe


### Multi-dataset loading utils
def make_empty_mod_dict(modality_info):
    empty_mod_dicts = {}

    for mod_name, mod_info in modality_info.items():
        empty_mod = {}

        # Tensor
        if 'num_channels' in mod_info and 'input_size' in mod_info:
            # Handle image-like modalities
            max_tokens = mod_info['max_tokens']
            empty_mod['tensor'] = torch.zeros((mod_info['num_channels'], mod_info['input_size'], mod_info['input_size']), dtype=torch.float32)
        elif mod_name == 't5_caption':
            # Handle T5 embedding
            max_tokens = mod_info['max_tokens']
            orig_emb_dim = mod_info['encoder_embedding']().orig_emb_dim
            empty_mod['tensor'] = torch.zeros((max_tokens, orig_emb_dim), dtype=torch.float32)
        elif mod_info['type'] in ['seq', 'seq_emb', 'seq_token']:
            # Handle all other discrete sequence modalities
            max_tokens = (mod_info['max_tokens'] + 1) * 2
            empty_mod['tensor'] = torch.zeros((max_tokens), dtype=torch.int32)
        else:
            max_tokens = mod_info['max_tokens']
            empty_mod['tensor'] = torch.zeros((max_tokens), dtype=torch.int32)
            
        # Input and target masks
        empty_mod['input_mask'] = torch.ones((max_tokens), dtype=torch.bool)
        empty_mod['target_mask'] = torch.ones((max_tokens), dtype=torch.bool)

        # Decoder attention mask
        empty_mod['decoder_attention_mask'] = torch.zeros((max_tokens), dtype=torch.int32)
        
        empty_mod_dicts[mod_name] = empty_mod
        
    return empty_mod_dicts


class MixtureDataset(IterableDataset):
    def __init__(self, data_iters, weights, modality_info):
        self.orig_data_iters = data_iters
        self.data_iters = [iter(data_iter) for data_iter in data_iters]  # Create initial iterators
        self.sampling_probs = np.array(weights) / sum(weights)
        self.modality_info = modality_info
        # Ensure that data_iters are not empty if weights are provided
        if not self.data_iters and any(w > 0 for w in self.sampling_probs):
            raise ValueError("MixtureDataset received no data_iters, but sampling_probs indicate data was expected.")
        elif not self.data_iters:
            print("Warning: MixtureDataset initialized with no data_iters.")

    def reset_iterator(self, idx):
        """ Reset the iterator when exhausted. """
        self.data_iters[idx] = iter(self.orig_data_iters[idx])

    def __iter__(self): # This is __iter__, not __next__ for an iterable dataset
        if not self.data_iters:
            # If there are no iterators, this dataset is empty. Stop iteration.
            return iter([]) # Yield an empty iterator
            
        while True:
            dataset_idx = np.random.choice(len(self.sampling_probs), p=self.sampling_probs)
            try:
                data = next(self.data_iters[dataset_idx])
            except StopIteration:  # If the iterator is exhausted
                self.reset_iterator(dataset_idx)  # Reset it
                try:
                    data = next(self.data_iters[dataset_idx])
                except StopIteration: # If it's still exhausted (e.g. empty source after reset)
                    # This can happen if a source dataset is truly empty or becomes empty.
                    # Depending on desired behavior, either skip or raise an error.
                    # For robustness, let's try to pick another dataset or skip if all are exhausted (which `with_epoch` should prevent for long runs).
                    # However, if all iterators provided to MixtureDataset are empty stubs, this loop could be problematic.
                    # The check in __init__ should catch the case of all-empty-stubs if weights are non-zero.
                    # If weights are zero for those, np.random.choice will fail earlier or this path won't be hit often for them.
                    print(f"Warning: Iterator {dataset_idx} exhausted even after reset. This might indicate an empty underlying dataset.")
                    # To prevent infinite loops on consistently empty datasets, consider a max retry or re-evaluating sampling_probs.
                    # For now, we continue, relying on other iterators or eventual epoch end by WebLoader.
                    continue 

            # The `load_and_preprocess_*_sample` functions are expected to have been called
            # by the individual data pipelines feeding into `data_iters`.
            # `MixtureDataset` now just merges these preprocessed sample dicts with a full template.
            mod_dict_template = make_empty_mod_dict(self.modality_info)
            # `data` here is the dict from one of the `load_and_preprocess_*_sample` calls (via the stage-specific pipeline)
            # It should already contain the MCoT prefix in the relevant token field.
            mod_dict_template.update(data) 
            yield mod_dict_template


def build_mixture_dataloader(data_iters, weights, modality_info, batch_size, num_workers, epoch_size, num_gpus, collate_fn=None):
    if not data_iters and any(w > 0 for w in weights):
        # This case should ideally be caught by MixtureDataset.__init__ if weights are passed there
        # or handled in the calling code (run_training_4m.py) by not calling this if no valid_train_iters.
        raise ValueError("build_mixture_dataloader received no data_iters, but weights indicate data was expected.")
    elif not data_iters:
        print("Warning: build_mixture_dataloader received no data_iters. Returning an empty list (no dataloader).")
        return [] # Return an empty list or handle as appropriate

    mixture_pipe = wds.DataPipeline(
        MixtureDataset(data_iters, weights, modality_info),
        # If collate_fn is UnifiedMasking, it expects a list of sample dicts (the batch)
        # So batching must happen *before* UnifiedMasking.__call__
        # wds.batched applies collation_fn to the list of items in the batch.
        wds.batched(batch_size, collation_fn=collate_fn if collate_fn is not None else default_collate, partial=False),
    )
    
    # Calculate samples_per_epoch_per_worker
    # Ensure num_gpus and num_workers are at least 1 to avoid division by zero if epoch_size is set
    effective_gpus = max(1, num_gpus)
    effective_workers_for_epoch_calc = max(1, num_workers) # For WebLoader, epoch is per worker.

    if epoch_size is not None and batch_size > 0:
        # samples_per_epoch_total = epoch_size
        # batches_per_epoch_total = samples_per_epoch_total // (batch_size * effective_gpus) # This is total batches across all GPUs
        # The .with_epoch for WebLoader is num_batches_per_worker
        # num_samples_per_worker_per_epoch = epoch_size // (effective_gpus * effective_workers_for_epoch_calc)
        # num_batches_per_worker_per_epoch = num_samples_per_worker_per_epoch // batch_size
        
        # Simpler: epoch_size is total samples. build_mixture_dataloader takes total batch_size implicitly via args.batch_size.
        # WebLoader.with_epoch takes number of batches for THAT worker.
        # Total batches per epoch = total_samples_epoch / (global_batch_size_per_step)
        # global_batch_size_per_step = batch_size_per_gpu * num_gpus
        # num_batches_for_this_worker = total_batches_per_epoch / num_workers (if using WDS sharding and workers properly)
        # More directly: epoch_size is total samples for the epoch.
        # Each worker gets epoch_size / (num_gpus * num_workers) samples if data is perfectly distributed.
        # Then number of batches per worker = (samples per worker) / batch_size_per_gpu.
        if effective_gpus > 0 and effective_workers_for_epoch_calc > 0 and batch_size > 0:
             batches_per_worker = epoch_size // (effective_gpus * effective_workers_for_epoch_calc * batch_size)
             if batches_per_worker > 0:
                mixture_pipe = mixture_pipe.with_epoch(batches_per_worker)
             else:
                print(f"Warning: Calculated batches_per_worker is {batches_per_worker} for epoch_size {epoch_size}. Iterator may be very short or empty.")
        else:
            print("Warning: Cannot set epoch size due to zero gpus, workers or batch_size.")
            
    # If num_workers is 0, WebLoader might not be appropriate or will run in main thread.
    # wds.WebLoader expects batch_size=None because batching is done by wds.batched above.
    mixture_loader = wds.WebLoader(mixture_pipe, num_workers=max(0, num_workers), batch_size=None) 
    
    return mixture_loader


class UnifiedMasking(TransferMasking):
    def __init__(self, modality_info, text_tokenizer, 
                 input_tokens_range, target_tokens_range, 
                 sampling_weights=None, type_probs=None, 
                 force_end_of_generation_token_for_text_target=True,
                 force_target_mask_for_padding=False):
        self.modality_info = modality_info
        self.text_tokenizer = text_tokenizer
        self.pad_id = text_tokenizer.token_to_id("[PAD]")
        self.eos_id = text_tokenizer.token_to_id("[EOS]")
        if self.pad_id is None or self.eos_id is None:
            raise ValueError("PAD or EOS token not found in tokenizer.")

        self.mcot_planning_start_token_id = text_tokenizer.token_to_id("[PLANNING_START]")
        self.mcot_acting_start_token_id = text_tokenizer.token_to_id("[ACTING_START]")
        self.mcot_reflection_start_token_id = text_tokenizer.token_to_id("[REFLECTION_START]")
        self.mcot_correction_start_token_id = text_tokenizer.token_to_id("[CORRECTION_START]")

        if any(id is None for id in [self.mcot_planning_start_token_id, 
                                     self.mcot_acting_start_token_id, 
                                     self.mcot_reflection_start_token_id, 
                                     self.mcot_correction_start_token_id]):
            raise ValueError("One or more MCoT special tokens not found in tokenizer.")
        
        self.max_input_tokens = input_tokens_range[1] if isinstance(input_tokens_range, tuple) else input_tokens_range
        self.max_target_tokens = target_tokens_range[1] if isinstance(target_tokens_range, tuple) else target_tokens_range
        self.mod_keys = list(self.modality_info.keys())
        self.force_end_of_generation_token_for_text_target = force_end_of_generation_token_for_text_target

    def determine_mcot_stage(self, sample: Dict[str, Any]):
        first_token_id = None
        if 'caption_tokens' in sample and isinstance(sample['caption_tokens'], torch.Tensor) and sample['caption_tokens'].numel() > 0:
            first_token_id = sample['caption_tokens'][0].item()
        elif 'acting_prefix_tokens' in sample and isinstance(sample['acting_prefix_tokens'], torch.Tensor) and sample['acting_prefix_tokens'].numel() > 0:
            first_token_id = sample['acting_prefix_tokens'][0].item()
        elif 'reflection_prefix_tokens' in sample and isinstance(sample['reflection_prefix_tokens'], torch.Tensor) and sample['reflection_prefix_tokens'].numel() > 0:
            first_token_id = sample['reflection_prefix_tokens'][0].item()
        elif 'correction_prefix_tokens' in sample and isinstance(sample['correction_prefix_tokens'], torch.Tensor) and sample['correction_prefix_tokens'].numel() > 0:
            first_token_id = sample['correction_prefix_tokens'][0].item()
        # Add more robust checks if needed, e.g., a dedicated 'mcot_stage_marker_modality' key

        if first_token_id == self.mcot_planning_start_token_id:
            return "planning"
        elif first_token_id == self.mcot_acting_start_token_id:
            return "acting"
        elif first_token_id == self.mcot_reflection_start_token_id:
            return "reflection"
        elif first_token_id == self.mcot_correction_start_token_id:
            return "correction"
        return None

    def _collate_modalities(self, sample: Dict[str, Any], selected_modalities: List[str], max_len: int, 
                            is_target: bool, current_mcot_stage: Optional[str] = None):
        collated_tokens = []
        for mod_key in selected_modalities:
            if mod_key in sample and sample[mod_key] is not None:
                tokens = sample[mod_key]
                if not isinstance(tokens, torch.Tensor):
                    tokens = torch.tensor(tokens, dtype=torch.long)
                if tokens.ndim == 0: tokens = tokens.unsqueeze(0)
                collated_tokens.append(tokens)
        
        if not collated_tokens:
            final_tokens = torch.full((max_len,), self.pad_id, dtype=torch.long)
            attention_mask = torch.zeros(max_len, dtype=torch.long)
            return final_tokens, attention_mask

        current_sequence = torch.cat(collated_tokens)
        seq_len = current_sequence.size(0)

        # EOS handling for targets
        has_eos_appended = False
        if is_target and self.force_end_of_generation_token_for_text_target:
            # Add EOS if it's a text-like generation target (e.g., planning caption, VQA answer)
            # This needs to be stage-specific or modality_info driven
            is_text_generation_target = False
            if current_mcot_stage == "planning" and 'target_dense_caption' in selected_modalities:
                 is_text_generation_target = True 
            elif current_mcot_stage is None and 'answer_tokens' in selected_modalities: # VQA
                 is_text_generation_target = True
            # Add other MCoT stages if they produce text sequences needing EOS

            if is_text_generation_target:
                if seq_len < max_len:
                    current_sequence = torch.cat((current_sequence[:seq_len], torch.tensor([self.eos_id], dtype=torch.long)))
                    seq_len += 1
                    has_eos_appended = True
                elif seq_len == max_len: # Replace last token with EOS if full
                    current_sequence[seq_len - 1] = self.eos_id
                    has_eos_appended = True # Though it overwrote, an EOS is present
        
        if seq_len > max_len:
            current_sequence = current_sequence[:max_len]
            seq_len = max_len
            # If EOS was appended and then truncated, it's gone. This logic might need refinement
            # to ensure EOS is preserved if possible during truncation.
        
        padding_needed = max_len - seq_len
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        
        if padding_needed > 0:
            padding = torch.full((padding_needed,), self.pad_id, dtype=torch.long)
            current_sequence = torch.cat((current_sequence, padding))
            padding_mask = torch.zeros(padding_needed, dtype=torch.long)
            attention_mask = torch.cat((attention_mask, padding_mask))
        
        return current_sequence, attention_mask

    def forward(self, sample: Dict[str, Any]):
        mcot_stage = self.determine_mcot_stage(sample)

        input_modalities_for_stage = []
        target_modalities_for_stage = []
        is_vqa_sample = 'is_vqa_sample' in sample and sample['is_vqa_sample']

        if mcot_stage == "planning":
            input_modalities_for_stage = ['caption_tokens'] # Contains [PLANNING_START] and prompt
            # Optionally add 'image_patches' if planning takes image context
            # if 'image_patches' in sample: input_modalities_for_stage.append('image_patches')
            target_modalities_for_stage = ['target_dense_caption', 'target_layout_tokens']
        elif mcot_stage == "acting":
            input_modalities_for_stage = ['acting_prefix_tokens', 'original_image_tokens', 'plan_tokens']
            target_modalities_for_stage = ['generated_image_tokens']
        elif mcot_stage == "reflection":
            input_modalities_for_stage = ['reflection_prefix_tokens', 'generated_image_tokens_from_acting', 'original_prompt_tokens']
            target_modalities_for_stage = ['heatmap_tokens']
        elif mcot_stage == "correction":
            input_modalities_for_stage = ['correction_prefix_tokens', 'generated_image_tokens_from_acting', 'heatmap_tokens_from_reflection', 'inpaint_mask_tokens']
            target_modalities_for_stage = ['corrected_image_region_tokens']
        elif is_vqa_sample:
            mcot_stage = "vqa" # Set stage for collate helper
            # Standard VQA: input text (question), input image. Target text (answer).
            # Keys must match what VQA data pipeline provides in `sample`.
            if 'text_tokens' in sample: input_modalities_for_stage.append('text_tokens')
            if 'image_patches' in sample: input_modalities_for_stage.append('image_patches') 
            # Or use modality_info to determine VQA inputs more generically if VQA samples also conform to it
            # else:
            #     for mod_key in self.mod_keys:
            #         if self.modality_info[mod_key].get('is_vqa_input', False): # Requires new flags in modality_info
            #              input_modalities_for_stage.append(mod_key)
            if 'answer_tokens' in sample: target_modalities_for_stage.append('answer_tokens')
            # else:
            #      for mod_key in self.mod_keys:
            #         if self.modality_info[mod_key].get('is_vqa_target', False):
            #             target_modalities_for_stage.append(mod_key)
            
            if not input_modalities_for_stage or not target_modalities_for_stage:
                print(f"Warning: VQA sample missing expected text/image inputs or answer tokens. Sample keys: {list(sample.keys())}")
                # Fallback to empty/padded to avoid crashing, but signals a data issue
                input_ids, input_mask = self._collate_modalities(sample, [], self.max_input_tokens, False, mcot_stage)
                target_ids, target_mask = self._collate_modalities(sample, [], self.max_target_tokens, True, mcot_stage)
                return {'input_ids': input_ids, 'input_mask': input_mask, 'target_ids': target_ids, 'target_mask': target_mask, 'mcot_stage': 'vqa_error'}
        else:
            print(f"Warning: Unknown sample type. MCoT stage: {mcot_stage}, VQA: {is_vqa_sample}. Sample keys: {list(sample.keys())}")
            input_ids, input_mask = self._collate_modalities(sample, [], self.max_input_tokens, False, mcot_stage)
            target_ids, target_mask = self._collate_modalities(sample, [], self.max_target_tokens, True, mcot_stage)
            return {'input_ids': input_ids, 'input_mask': input_mask, 'target_ids': target_ids, 'target_mask': target_mask, 'mcot_stage': 'unknown'}

        input_ids, input_mask = self._collate_modalities(sample, input_modalities_for_stage, self.max_input_tokens, is_target=False, current_mcot_stage=mcot_stage)
        target_ids, target_mask = self._collate_modalities(sample, target_modalities_for_stage, self.max_target_tokens, is_target=True, current_mcot_stage=mcot_stage)
        
        return {
            'input_ids': input_ids, 
            'input_mask': input_mask, 
            'target_ids': target_ids, 
            'target_mask': target_mask,
            'mcot_stage': mcot_stage
        }

    def __call__(self, mod_dict_batch: List[Dict[str, Any]]):
        batch_input_ids = []
        batch_input_masks = []
        batch_target_ids = []
        batch_target_masks = []
        # batch_mcot_stages = [] # If needed for metrics/logging

        for i, sample in enumerate(mod_dict_batch):
            try:
                processed_sample = self.forward(sample)
                batch_input_ids.append(processed_sample['input_ids'])
                batch_input_masks.append(processed_sample['input_mask'])
                batch_target_ids.append(processed_sample['target_ids'])
                batch_target_masks.append(processed_sample['target_mask'])
                # batch_mcot_stages.append(processed_sample['mcot_stage'])
            except Exception as e:
                print(f"Error processing sample {i} in batch: {e}. Sample keys: {list(sample.keys())}")
                # Optionally, skip this sample or raise, or provide default padded tensors
                # For now, let's re-raise to make issues visible during development
                raise
        
        try:
            final_input_ids = torch.stack(batch_input_ids)
            final_input_mask = torch.stack(batch_input_masks)
            final_target_ids = torch.stack(batch_target_ids)
            final_target_mask = torch.stack(batch_target_masks)
        except RuntimeError as e:
            print(f"Error stacking batch tensors: {e}")
            for i, (iid, imsk, tid, tmsk) in enumerate(zip(batch_input_ids, batch_input_masks, batch_target_ids, batch_target_masks)):
                print(f"Sample {i} shapes: iid={iid.shape}, imsk={imsk.shape}, tid={tid.shape}, tmsk={tmsk.shape}")
            raise
        
        return {
            'input_ids': final_input_ids, 
            'input_mask': final_input_mask, 
            'target_ids': final_target_ids, 
            'target_mask': final_target_mask,
            # 'mcot_stages': batch_mcot_stages
        }


def get_mcot_planning_data_pipeline(
    data_path: str, # Path to webdataset shards for COCO planning
    text_tokenizer, # Pre-configured text tokenizer with MCoT, COCO class, and coord tokens
    modality_info: Dict[str, Any],
    # image_augmenter is usually part of UnifiedDataTransform, applied later
    # input_tokens_range, target_tokens_range are for UnifiedMasking, applied later
    # Common WebDataset args:
    num_gpus: int,
    num_workers: int,
    batch_size: int, # Per GPU batch size
    epoch_size: Optional[int] = None, # Number of samples per epoch
    shuffle_buffer_load: int = 1000,
    shuffle_buffer_repeat: int = 5000,
    # Other MCoT specific args might be needed by load_and_preprocess_planning_sample if not via modality_info
    # e.g. keys for image, caption, bbox in the webdataset sample
    image_file_key: str = "image.jpg", # Or "image.png"
    caption_prompt_key: str = "caption_prompt.txt",
    target_plan_text_key: str = "target_plan_text.txt",
    bbox_json_key: str = "bboxes.json", # Expects {"image_width": w, "image_height": h, "annotations": [[x,y,w,h,cat_name],...]}
    handler: Callable = wds.warn_and_continue,
    **kwargs # To catch other dataset_config args
    ):
    """
    Constructs a WebDataset pipeline for MCoT Planning data from MS-COCO.
    Output from this pipeline's mapped functions feeds into load_and_preprocess_planning_sample.
    The result of this function (a WebDataset IterableDataset) will then typically be wrapped by:
    1. UnifiedDataTransform (applies image_augmenter, modality-specific transforms)
    2. UnifiedMasking (as a collate_fn for the DataLoader, handles MCoT logic, masking)
    """
    print(f"Setting up MCoT Planning pipeline from: {data_path}")

    if not data_path:
        raise ValueError("data_path for MCoT Planning must be provided.")
    
    # Expand brace notation for data_path if used (e.g., 'path/to/shards-{0000..0099}.tar')
    if isinstance(data_path, str):
        shard_urls = list(braceexpand.braceexpand(data_path))
    elif isinstance(data_path, list):
        shard_urls = data_path
    else:
        raise TypeError(f"data_path must be a string or list of strings, got {type(data_path)}")

    if not shard_urls:
        raise ValueError(f"No shards found for data_path: {data_path}")

    dataset_size = epoch_size # If None, WebDataset will iterate indefinitely or until shards end.
    
    # Create a partial function for load_and_preprocess_planning_sample
    # to pass fixed arguments like text_tokenizer and modality_info,
    # and to ensure it uses the correct keys from the webdataset sample.
    # The webdataset sample `x` will be passed as the first argument.
    def _custom_planning_processor(sample):
        # Adapt raw_sample keys here before passing to the main processor
        # This allows flexibility if webdataset keys differ from what load_and_preprocess_planning_sample expects
        adapted_sample = {
            'image.jpg': sample.get(image_file_key.replace('.png', '.jpg')), # Prefer jpg, then png
            'image.png': sample.get(image_file_key.replace('.jpg', '.png')),
            'caption_prompt.txt': sample.get(caption_prompt_key),
            'target_plan_text.txt': sample.get(target_plan_text_key),
            'bboxes.json': sample.get(bbox_json_key),
            '__key__': sample.get('__key__'),
        }
        # Ensure text fields are decoded if they are bytes
        for key in ['caption_prompt.txt', 'target_plan_text.txt', 'bboxes.json']:
            if isinstance(adapted_sample[key], bytes):
                adapted_sample[key] = adapted_sample[key].decode('utf-8')
        
        return load_and_preprocess_planning_sample(adapted_sample, text_tokenizer, modality_info)

    pipeline = [
        wds.ResampledShards(shard_urls), # Resamples shards for multiple epochs
        wds.tarfile_to_samples(handler=handler),
        wds.shuffle(shuffle_buffer_load, initial=shuffle_buffer_load, handler=handler),
        # Decode necessary files: images to PIL, text to string, json to string (for later json.loads)
        wds.decode(
            wds.handle_extension(image_file_key.split('.')[-1], "pilrgb"), # "pilrgb" for PIL RGB
            wds.handle_extension("txt", "txt"), # .txt to string
            wds.handle_extension("json", "txt"), # .json to string, to be parsed by json.loads
            handler=handler
        ),
        wds.map(_custom_planning_processor, handler=handler),
        # The output of _custom_planning_processor is a dict of PIL images and torch.LongTensors.
        # This is ready for UnifiedDataTransform and then UnifiedMasking (collate_fn).
    ]
    
    # Create the IterableDataset
    dataset = wds.DataPipeline(*pipeline)

    if dataset_size is not None:
        dataset = dataset.with_epoch(dataset_size // (num_gpus * num_workers) if num_gpus * num_workers > 0 else dataset_size)


    # Note: DataLoader will be created by the caller (e.g., in run_training_4m.py)
    # The caller will also provide UnifiedDataTransform and UnifiedMasking (as collate_fn).
    # Example:
    # dataloader = wds.WebLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     collate_fn=unified_masking_instance, # UnifiedMasking instance
    #     persistent_workers=num_workers > 0,
    # )
    # dataloader = dataloader.ddp_equalize(dataset_size // batch_size if dataset_size else None)
    
    return dataset # Returns the wds.DataPipeline IterableDataset


def get_mcot_acting_data_pipeline(
    data_path: str, # Path to webdataset shards for COCO acting
    text_tokenizer, 
    modality_info: Dict[str, Any],
    # image_vqvae_tokenizer, # Not used if acting output is text
    # image_augmenter, input_tokens_range, target_tokens_range are handled by caller
    num_gpus: int,
    num_workers: int,
    batch_size: int,
    epoch_size: Optional[int] = None,
    shuffle_buffer_load: int = 1000,
    shuffle_buffer_repeat: int = 5000,
    # Keys for data in webdataset samples for acting stage
    image_file_key: str = "image.jpg",
    plan_text_key: str = "plan_text.txt",
    plan_bboxes_json_key: str = "plan_bboxes.json",
    target_final_caption_key: str = "target_final_caption.txt",
    handler: Callable = wds.warn_and_continue,
    **kwargs # To catch other dataset_config args
    ):
    """
    Constructs a WebDataset pipeline for MCoT Acting data from MS-COCO.
    Acting stage takes image, plan (text+bboxes), and generates a final caption.
    """
    print(f"Setting up MCoT Acting pipeline from: {data_path}")

    if not data_path:
        raise ValueError("data_path for MCoT Acting must be provided.")

    if isinstance(data_path, str):
        shard_urls = list(braceexpand.braceexpand(data_path))
    elif isinstance(data_path, list):
        shard_urls = data_path
    else:
        raise TypeError(f"data_path must be a string or list of strings, got {type(data_path)}")

    if not shard_urls:
        raise ValueError(f"No shards found for data_path: {data_path}")

    dataset_size = epoch_size

    def _custom_acting_processor(sample):
        adapted_sample = {
            'image.jpg': sample.get(image_file_key.replace('.png', '.jpg')),
            'image.png': sample.get(image_file_key.replace('.jpg', '.png')),
            'plan_text.txt': sample.get(plan_text_key),
            'plan_bboxes.json': sample.get(plan_bboxes_json_key),
            'target_final_caption.txt': sample.get(target_final_caption_key),
            '__key__': sample.get('__key__'),
        }
        for key in ['plan_text.txt', 'plan_bboxes.json', 'target_final_caption.txt']:
            if isinstance(adapted_sample[key], bytes):
                adapted_sample[key] = adapted_sample[key].decode('utf-8')
        
        return load_and_preprocess_acting_sample(adapted_sample, text_tokenizer, modality_info)

    pipeline = [
        wds.ResampledShards(shard_urls),
        wds.tarfile_to_samples(handler=handler),
        wds.shuffle(shuffle_buffer_load, initial=shuffle_buffer_load, handler=handler),
        wds.decode(
            wds.handle_extension(image_file_key.split('.')[-1], "pilrgb"),
            wds.handle_extension("txt", "txt"),
            wds.handle_extension("json", "txt"), 
            handler=handler
        ),
        wds.map(_custom_acting_processor, handler=handler),
    ]
    
    dataset = wds.DataPipeline(*pipeline)

    if dataset_size is not None:
        dataset = dataset.with_epoch(dataset_size // (num_gpus * num_workers) if num_gpus * num_workers > 0 else dataset_size)
    
    return dataset


def get_mcot_reflection_data_pipeline(
    data_path, text_tokenizer, semantic_seg_vqvae_tokenizer, modality_info, image_augmenter,
    input_tokens_range, target_tokens_range,
    num_gpus, num_workers, batch_size, epoch_size,
    # ... other args
    ):
    """
    Constructs a WebDataset pipeline for MCoT Reflection data from RichHF-18K.
    """
    print(f"Setting up MCoT Reflection pipeline from: {data_path}")
    # Similar structure
    # 1. WDS setup for RichHF-18K.
    # 2. Map to a function that calls load_and_preprocess_reflection_sample
    #    - Needs raw generated image (from acting), raw original prompt, raw artifact annotation.
    #    - semantic_seg_vqvae_tokenizer used inside.
    print("WARNING: MCoT Reflection pipeline is a STUB and will not load data.")
    return [] # Placeholder


def get_mcot_correction_data_pipeline(
    data_path, text_tokenizer, image_vqvae_tokenizer, modality_info, image_augmenter,
    input_tokens_range, target_tokens_range,
    num_gpus, num_workers, batch_size, epoch_size,
    # ... other args
    ):
    """
    Constructs a WebDataset pipeline for MCoT Correction data from COCO-Stuff.
    """
    print(f"Setting up MCoT Correction pipeline from: {data_path}")
    # Similar structure
    # 1. WDS setup for COCO-Stuff.
    # 2. Map to a function that calls load_and_preprocess_correction_sample
    #    - Needs raw generated image, raw heatmap tokens (or data to make them),
    #      raw inpaint mask (or data to make it), raw target corrected image region.
    #    - image_vqvae_tokenizer used inside.
    print("WARNING: MCoT Correction pipeline is a STUB and will not load data.")
    return [] # Placeholder


# The build_mixture_dataloader in run_training_4m.py will then be responsible for
# creating these individual MCoT dataloaders (and the VQA one) and passing their
# iterators and weights to the MixtureDataset.
# ... (rest of the file, e.g., main dataloader functions like get_train_dataloader)

def modality_info_to_transforms_dict(modality_info, phase='train'):
    """Convert modality_info to transforms_dict for UnifiedDataTransform"""
    transforms_dict = {}
    for mod_name, info in modality_info.items():
        if info['type'] == 'img' and not info.get('pretokenized', False):
            transforms_dict[mod_name] = RGBTransform()
        elif info['type'] == 'depth' and not info.get('pretokenized', False):
            transforms_dict[mod_name] = DepthTransform()
        elif info['type'] == 'seq' or info['type'] == 'seq_token':
            transforms_dict[mod_name] = CaptionTransform()
        elif info['type'] == 'det':
            transforms_dict[mod_name] = DetectionTransform()
        elif info['type'] == 'tok':
            transforms_dict[mod_name] = TokTransform()
    return transforms_dict

def get_train_dataloader(dataset_config, modality_info, sampling_weights, text_tokenizer, input_size, 
                         num_input_tokens, num_target_tokens, min_input_tokens, min_target_tokens,
                         num_tasks, num_workers, dataset_batch_size, epoch_size, is_val=False):
    
    in_domains = sorted(list(dataset_config['in_domains'].split('-')))
    out_domains = sorted(list(dataset_config['out_domains'].split('-')))
    all_domains = sorted(list(set(in_domains) | set(out_domains)))

    modality_transforms = modality_info_to_transforms_dict(modality_info, phase='train' if not is_val else 'val')

    # Determine input and target token ranges for UnifiedMasking
    current_num_input_tokens = dataset_config.get('num_input_tokens', num_input_tokens)
    current_num_target_tokens = dataset_config.get('num_target_tokens', num_target_tokens)
    current_min_input_tokens = dataset_config.get('min_input_tokens', min_input_tokens) 
    current_min_target_tokens = dataset_config.get('min_target_tokens', min_target_tokens)
    # Ensure defaults if None
    current_min_input_tokens = current_num_input_tokens if current_min_input_tokens is None else current_min_input_tokens
    current_min_target_tokens = current_num_target_tokens if current_min_target_tokens is None else current_min_target_tokens
    input_tokens_range = (current_min_input_tokens, current_num_input_tokens)
    target_tokens_range = (current_min_target_tokens, current_num_target_tokens)

    # Get common arguments for WebDataset based loaders
    world_size = get_world_size()
    global_rank = get_rank()

    data_loader = None
    dataset = None

    if dataset_config['type'] == 'mcot_planning':
        print(f"Creating MCoT Planning DataLoader for {dataset_config.get('data_path')}")
        
        # Configure image augmenter
        if any([modality_info[mod].get('pretokenized', False) for mod in all_domains if mod in modality_info]):
            image_augmenter = PreTokenizedImageAugmenter(
                target_size=input_size, 
                no_aug=(not dataset_config.get('tok_train_aug', True)), 
                main_domain=dataset_config.get('main_augment_domain')
            )
        else:
            image_augmenter = RandomCropImageAugmenter(
                target_size=input_size, 
                hflip=dataset_config.get('hflip', True if not is_val else False), 
                crop_scale=tuple(dataset_config.get('crop_scale', (0.5, 1.0))),
                crop_ratio=tuple(dataset_config.get('crop_ratio', (0.75, 1.33))),
            )
            
        # Set up the masking
        unified_masking_instance = UnifiedMasking(
            modality_info=modality_info,
            text_tokenizer=text_tokenizer,
            input_tokens_range=input_tokens_range,
            target_tokens_range=target_tokens_range,
        )
        
        # Set up the data transforms
        unified_data_transform_instance = UnifiedDataTransform(
            transforms_dict=modality_transforms,
            image_augmenter=image_augmenter
        )
        
        dataset = get_mcot_planning_data_pipeline(
            data_path=dataset_config['data_path'],
            text_tokenizer=text_tokenizer,
            modality_info=modality_info,
            num_gpus=world_size,
            num_workers=num_workers,
            batch_size=dataset_batch_size,
            epoch_size=dataset_config.get('epoch_size', epoch_size),
            shuffle_buffer_load=dataset_config.get('wds_shuffle_buffer_tar', 1000),
            shuffle_buffer_repeat=dataset_config.get('wds_shuffle_buffer_repeat', 5000),
            image_file_key=dataset_config.get('image_file_key', 'image.jpg'),
            caption_prompt_key=dataset_config.get('caption_prompt_key', 'caption_prompt.txt'),
            target_plan_text_key=dataset_config.get('target_plan_text_key', 'target_plan_text.txt'),
            bbox_json_key=dataset_config.get('bbox_json_key', 'bboxes.json'),
        )
        
        # Apply transforms to the dataset items
        dataset = dataset.map(unified_data_transform_instance)
        
        data_loader = wds.WebLoader(
            dataset,
            batch_size=dataset_batch_size,
            num_workers=num_workers,
            collate_fn=unified_masking_instance,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    elif dataset_config['type'] == 'mcot_acting':
        print(f"Creating MCoT Acting DataLoader for {dataset_config.get('data_path')}")
        
        # Configure image augmenter
        if any([modality_info[mod].get('pretokenized', False) for mod in all_domains if mod in modality_info]):
            image_augmenter = PreTokenizedImageAugmenter(
                target_size=input_size, 
                no_aug=(not dataset_config.get('tok_train_aug', True)), 
                main_domain=dataset_config.get('main_augment_domain')
            )
        else:
            image_augmenter = RandomCropImageAugmenter(
                target_size=input_size, 
                hflip=dataset_config.get('hflip', True if not is_val else False), 
                crop_scale=tuple(dataset_config.get('crop_scale', (0.5, 1.0))),
                crop_ratio=tuple(dataset_config.get('crop_ratio', (0.75, 1.33))),
            )
            
        # Set up the masking
        unified_masking_instance = UnifiedMasking(
            modality_info=modality_info,
            text_tokenizer=text_tokenizer,
            input_tokens_range=input_tokens_range,
            target_tokens_range=target_tokens_range,
        )
        
        # Set up the data transforms
        unified_data_transform_instance = UnifiedDataTransform(
            transforms_dict=modality_transforms,
            image_augmenter=image_augmenter
        )
        
        dataset = get_mcot_acting_data_pipeline(
            data_path=dataset_config['data_path'],
            text_tokenizer=text_tokenizer,
            modality_info=modality_info,
            num_gpus=world_size,
            num_workers=num_workers,
            batch_size=dataset_batch_size,
            epoch_size=dataset_config.get('epoch_size', epoch_size),
            shuffle_buffer_load=dataset_config.get('wds_shuffle_buffer_tar', 1000),
            shuffle_buffer_repeat=dataset_config.get('wds_shuffle_buffer_repeat', 5000),
            image_file_key=dataset_config.get('image_file_key', 'image.jpg'),
            plan_text_key=dataset_config.get('plan_text_key', 'plan_text.txt'),
            plan_bboxes_json_key=dataset_config.get('plan_bboxes_json_key', 'plan_bboxes.json'),
            target_final_caption_key=dataset_config.get('target_final_caption_key', 'target_final_caption.txt'),
        )
        
        dataset = dataset.map(unified_data_transform_instance)
        
        data_loader = wds.WebLoader(
            dataset,
            batch_size=dataset_batch_size,
            num_workers=num_workers,
            collate_fn=unified_masking_instance,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    elif dataset_config['type'] == 'multimodal':
        # Original multimodal handling code
        data_path = dataset_config['data_path']
        
        # Setup image augmenter based on dataset type
        is_pretokenized = any([modality_info[mod].get('pretokenized', False) for mod in all_domains])
        if is_pretokenized:
            image_augmenter = PreTokenizedImageAugmenter(
                target_size=input_size, 
                no_aug=(not dataset_config.get('tok_train_aug', True)), 
                main_domain=dataset_config.get('main_augment_domain')
            )
        else:
            image_augmenter = RandomCropImageAugmenter(
                target_size=input_size, 
                hflip=dataset_config.get('hflip', True if not is_val else False), 
                crop_scale=tuple(dataset_config.get('crop_scale', (0.5, 1.0))),
                crop_ratio=tuple(dataset_config.get('crop_ratio', (0.75, 1.33))),
            )

        # Use webdataset if data_path is provided and has .tar extension
        if data_path is not None and ('tar' in data_path or '{' in data_path):
            print(f"Loading multimodal data from {data_path}")
            if dataset_config.get('modality_name_map', None) is not None:
                modality_name_map = {y: x for x, y in dataset_config['modality_name_map'].items()}
            else:
                modality_name_map = None
                
            if dataset_config.get('from_huggingface_hub', False):
                return build_huggingface_pretraining_dataloader(
                    data_path=data_path, all_domains=all_domains, modality_info=modality_info, 
                    modality_transforms=modality_transforms, image_augmenter=image_augmenter, 
                    text_tokenizer=text_tokenizer, input_tokens_range=input_tokens_range, 
                    target_tokens_range=target_tokens_range, num_gpus=world_size, num_workers=num_workers, 
                    batch_size=dataset_batch_size, epoch_size=dataset_config.get('epoch_size', epoch_size),
                    split=dataset_config.get('split', 'train'),
                    streaming=dataset_config.get('streaming', True),
                    rename_text_to_caption=dataset_config.get('rename_text_to_caption', True),
                    shuffle_buffer_load=dataset_config.get('wds_shuffle_buffer_tar', 10_000),
                    shuffle_seed=dataset_config.get('shuffle_seed', 0)
                )
            else:
                return build_wds_fm_pretraining_dataloader(
                    data_path=data_path, all_domains=all_domains, modality_info=modality_info, 
                    modality_transforms=modality_transforms, image_augmenter=image_augmenter,
                    text_tokenizer=text_tokenizer, input_tokens_range=input_tokens_range,
                    target_tokens_range=target_tokens_range, num_gpus=world_size, num_workers=num_workers, 
                    batch_size=dataset_batch_size, epoch_size=dataset_config.get('epoch_size', epoch_size),
                    sampling_weights=sampling_weights, modality_name_map=modality_name_map,
                    shuffle_buffer_load=dataset_config.get('wds_shuffle_buffer_tar', 1000),
                    shuffle_buffer_repeat=dataset_config.get('wds_shuffle_buffer_repeat', 5000),
                    n_repeats=dataset_config.get('n_repeats', 5)
                )
        # Use MultiModalDatasetFolder for direct folder loading
        else:
            val_transform = UnifiedMasking(modality_info=modality_info, text_tokenizer=text_tokenizer,
                           input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
                           sampling_weights=sampling_weights)
            transform = transforms.Compose([
                UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
                val_transform])
            dataset = build_fm_pretraining_dataset(
                data_path=dataset_config['data_path'], all_domains=all_domains,
                modality_info=modality_info, modality_transforms=modality_transforms,
                image_augmenter=image_augmenter, text_tokenizer=text_tokenizer,
                input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
                sampling_weights=sampling_weights
            )
            sampler = None
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset_batch_size,
                shuffle=(sampler is None and not is_val),
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
                drop_last=not is_val
            )

    elif dataset_config['type'] == 'mcot_planning_huggingface':
        # Your new HF-based pipeline code here
        dataset = get_mcot_planning_huggingface_pipeline(
            data_path=dataset_config['data_path'],
            text_tokenizer=text_tokenizer,
            modality_info=modality_info,
            num_gpus=world_size,
            num_workers=num_workers,
            batch_size=dataset_batch_size,
            year=dataset_config.get('year', 2017),
            # Other HF-specific params
        )
        
        # Then similar handling as with WebDataset

    elif dataset_config['type'] == 'huggingface':
        print(f"Creating Hugging Face DataLoader for {dataset_config['data_path']}")
        
        # Configure image augmenter (reuse your existing code)
        image_augmenter = RandomCropImageAugmenter(
            target_size=input_size, 
            hflip=dataset_config.get('hflip', True if not is_val else False), 
            crop_scale=tuple(dataset_config.get('crop_scale', (0.5, 1.0))),
            crop_ratio=tuple(dataset_config.get('crop_ratio', (0.75, 1.33))),
        )
        
        # Set up masking 
        unified_masking_instance = UnifiedMasking(
            modality_info=modality_info,
            text_tokenizer=text_tokenizer,
            input_tokens_range=input_tokens_range,
            target_tokens_range=target_tokens_range,
        )
        
        # Set up transforms
        unified_data_transform_instance = UnifiedDataTransform(
            transforms_dict=modality_transforms,
            image_augmenter=image_augmenter
        )
        
        # Load and process HuggingFace dataset
        data_loader = build_huggingface_pretraining_dataloader(
            data_path=dataset_config['data_path'],
            all_domains=all_domains,
            modality_info=modality_info,
            modality_transforms=modality_transforms,
            image_augmenter=image_augmenter,
            text_tokenizer=text_tokenizer,
            input_tokens_range=input_tokens_range,
            target_tokens_range=target_tokens_range,
            num_gpus=world_size,
            num_workers=num_workers,
            batch_size=dataset_batch_size,
            epoch_size=dataset_config.get('epoch_size', epoch_size),
            split=dataset_config.get('split', 'train'),
            year=dataset_config.get('year', 2017),
            coco_task=dataset_config.get('coco_task', ['captions', 'instances']),
            streaming=dataset_config.get('streaming', True),
        )

    return data_loader


def _tokenize_plan_for_acting(plan_text, plan_bbox_data, image_width, image_height, text_tokenizer, coord_bins=1000):
    """
    Tokenizes a plan (text + bboxes) to serve as input for the acting stage.
    """
    plan_text_tokens = text_tokenizer.encode(plan_text).ids
    # For bboxes, reuse the planning tokenizer logic for consistency
    plan_bbox_tokens = _tokenize_bboxes_for_planning(plan_bbox_data, image_width, image_height, text_tokenizer, coord_bins)
    
    # Combine them, perhaps with a separator if needed, or just concatenate.
    # For now, concatenate. This combined sequence becomes an input to acting.
    # The tokenizer should have a general <SEP> token if explicit separation is desired.
    # sep_token_id = text_tokenizer.token_to_id("[SEP_PLAN]") # Example separator
    # combined_tokens = plan_text_tokens + [sep_token_id] + plan_bbox_tokens
    combined_tokens = plan_text_tokens + plan_bbox_tokens
    return combined_tokens


def get_mcot_planning_huggingface_pipeline(
    data_path: str,  # HF dataset path
    all_domains: List[str], 
    modality_info: Dict[str, Any],
    modality_transforms: Dict, 
    image_augmenter, 
    text_tokenizer,
    input_tokens_range, 
    target_tokens_range, 
    num_gpus: int,
    num_workers: int,
    batch_size: int, # Note: batch_size here is for context, not applied directly by this func
    epoch_size: Optional[int] = None,
    split: Optional[str] = None,
    year: int = 2017,
    coco_task: Union[str, List[str]] = "instances",
    streaming: bool = True,
    shuffle_buffer_load: int = 10_000,
    shuffle_seed: int = 0,
    **kwargs
):
    actual_coco_task = coco_task
    if isinstance(coco_task, list):
        if "instances" in coco_task:
            actual_coco_task = "instances"
        elif coco_task:
            actual_coco_task = coco_task[0]
        else:
            actual_coco_task = "instances" 
            print(f"Warning: coco_task list was empty or problematic: {coco_task}. Defaulting to 'instances'.")

    hf_config_name = f"{year}-{actual_coco_task}"
    print(f"INFO: Loading Hugging Face dataset {data_path} with config_name='{hf_config_name}', split='{split}'")

    try:
        dataset = load_dataset(data_path, name=hf_config_name, split=split, streaming=streaming, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: Failed to load Hugging Face dataset {data_path} with config {hf_config_name}, split {split}. Error: {e}")
        # Return an empty datapipe or raise error to prevent downstream issues
        return wds.DataPipeline(iter([])) # Empty datapipe

    dataset = split_dataset_by_node(dataset, rank=get_rank(), world_size=get_world_size())
    if streaming:
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer_load)

    def _mcot_hf_planning_processor(raw_hf_sample):
        # Uses prepare_hf_sample_for_domains to get base image/caption domains
        # Then adds MCoT-specific 'det' domain.
        
        # `all_domains` passed to this pipeline should be specific to this dataset's config (in_domains + out_domains)
        processed_sample = prepare_hf_sample_for_domains(raw_hf_sample, all_domains)

        if processed_sample is None:
            # print(f"DEBUG: prepare_hf_sample_for_domains returned None for sample {raw_hf_sample.get('image_id', 'unknown')}")
            return None

        det_domain_key = next((d for d in all_domains if modality_info[d]['type'] == 'seq' and modality_info[d].get('is_output_modality') and d == 'det'), None)

        if det_domain_key: # If 'det' is an expected domain for this dataset part
            if 'objects' in raw_hf_sample and 'bbox' in raw_hf_sample['objects'] and \
               'category' in raw_hf_sample['objects'] and \
               raw_hf_sample.get('width') and raw_hf_sample.get('height'):
                
                img_width = raw_hf_sample['width']
                img_height = raw_hf_sample['height']
                annotations_for_tokenizer = []
                
                for i in range(len(raw_hf_sample['objects']['bbox'])):
                    bbox_xywh = raw_hf_sample['objects']['bbox'][i] 
                    category_id = raw_hf_sample['objects']['category'][i] # This is an int ID from COCO
                    
                    category_name = COCO_CATEGORIES_2017_ID_TO_NAME.get(category_id)
                    
                    if category_name is None:
                        # print(f"WARN: Unknown COCO category ID {category_id} for sample (ID: {raw_hf_sample.get('image_id', 'unknown')}). Skipping this bbox.")
                        continue # Skip this bounding box if its category is not in our map
                        
                    annotations_for_tokenizer.append([bbox_xywh[0], bbox_xywh[1], bbox_xywh[2], bbox_xywh[3], category_name])

                if annotations_for_tokenizer:
                    bbox_data_for_tok = {
                        "image_width": img_width, 
                        "image_height": img_height, 
                        "annotations": annotations_for_tokenizer
                    }
                    try:
                        tokenized_det_string = _tokenize_bboxes_for_planning(
                            bbox_data_for_tok, img_width, img_height, text_tokenizer, 
                            coord_bins=modality_info[det_domain_key].get('coord_bins', 1000)
                        )
                        processed_sample[det_domain_key] = tokenized_det_string
                    except Exception as e:
                        print(f"WARN: Error tokenizing bboxes for sample (ID: {raw_hf_sample.get('image_id', 'unknown')}), task {actual_coco_task}: {e}. 'det' domain might be missing. Check category name mapping and tokenizer.")
                        pass 
            # else:
                # print(f"DEBUG: Missing 'objects' or image dimensions for 'det' processing in sample {raw_hf_sample.get('image_id', 'unknown')}")


        # Ensure all keys declared in `all_domains` (for this dataset part) exist in processed_sample,
        # filling with a default or special value if not produced. This is important for UnifiedMasking.
        # However, UnifiedMasking itself should handle missing keys by creating empty/padded tensors.
        # So, just returning the sample as is, after processing.
        return processed_sample

    # This is for transforms like ToTensor, Normalize for images, etc.
    # It does NOT include MCoT specific tokenization or UnifiedMasking's main logic.
    unified_data_transform = transforms.Compose([
        UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
    ])

    def filter_none(sample):
        return sample is not None

    datapipe = wds.DataPipeline(
        dataset,
        map(_mcot_hf_planning_processor),
        wds.select(filter_none), # Filter out samples that failed critical processing
        map(unified_data_transform), # Apply image transforms, etc.
    )
    # This datapipe yields individual, processed (but unbatched) samples.
    # Epoch setting and batching will be handled by build_mixture_dataloader.
    return datapipe


### Multi-dataset loading utils
def get_train_dataloader(dataset_config, modality_info, sampling_weights, text_tokenizer, input_size, 
                         num_input_tokens, num_target_tokens, min_input_tokens, min_target_tokens,
                         num_tasks, num_workers, dataset_batch_size, epoch_size, is_val=False):
    
    modality_transforms = modality_info_to_transforms_dict(modality_info, phase='train' if not is_val else 'val')
    image_augmenter = None # Initialize if/as needed by your UnifiedDataTransform config
    
    all_dataset_pipes = [] # Stores unbatched datapipes or WebLoader instances
    
    input_tokens_range = (min_input_tokens, num_input_tokens)
    target_tokens_range = (min_target_tokens, num_target_tokens)

    for i, ds_cfg_i in enumerate(dataset_config.datasets):
        dataset_name = list(ds_cfg_i.keys())[0]
        cfg = ds_cfg_i[dataset_name]
        
        current_epoch_size = cfg.get("epoch_size", epoch_size) 
        current_batch_size = cfg.get("batch_size", dataset_batch_size) 
        current_split = cfg.get("split", "train" if not is_val else "validation")
        
        # Determine domains for this specific dataset config from its in_domains/out_domains
        current_in_domains = [d.strip() for d in cfg.get("in_domains", "").split(',') if d.strip()]
        current_out_domains = [d.strip() for d in cfg.get("out_domains", "").split(',') if d.strip()]
        current_all_domains = list(set(current_in_domains + current_out_domains))
        
        if not current_all_domains: # Fallback if not specified in config (should be specified)
            print(f"Warning: in_domains/out_domains not specified for dataset {dataset_name}. Using all known modalities.")
            current_all_domains = list(modality_info.keys())

        if cfg.type == "webdataset":
            loader_fn = None
            # MCoT specific webdataset pipelines
            if dataset_name.startswith("coco_planning"): # coco_planning_val
                loader_fn = get_mcot_planning_data_pipeline
            elif dataset_name.startswith("coco_acting"): # coco_acting_val
                loader_fn = get_mcot_acting_data_pipeline
            # Add other MCoT WDS pipelines here (e.g., reflection, correction)
            # elif dataset_name.startswith("coco_reflection"):
            #     loader_fn = get_mcot_reflection_data_pipeline
            # elif dataset_name.startswith("coco_correction"):
            #     loader_fn = get_mcot_correction_data_pipeline
            else:
                # Generic WebDataset pretraining dataloader (if you have one)
                # loader_fn = build_wds_fm_pretraining_dataloader # Example
                print(f"Warning: No specific MCoT WebDataset loader for {dataset_name} of type {cfg.type}. Skipping.")
                continue 
            
            # These WDS pipelines typically return a WebLoader instance, which is iterable and handles its own workers/batching.
            # MixtureDataset should be able to handle iterating these WebLoaders.
            wds_loader_kwargs = {
                k: v for k, v in cfg.items() if k not in [
                    'type', 'data_path', 'epoch_size', 'batch_size', 'split', 
                    'in_domains', 'out_domains', 'year', 'coco_task', # Common / HF specific
                    'wds_shuffle_buffer_load', 'wds_shuffle_buffer_repeat', # WDS specific handled by loader_fn
                    'image_file_key', 'caption_prompt_key', 'target_plan_text_key', 'bbox_json_key' # MCoT WDS specific
                ]
            }

            data_iterable = loader_fn(
                data_path=cfg.data_path,
                text_tokenizer=text_tokenizer,
                modality_info=modality_info, # Global modality_info
                num_gpus=get_world_size(), 
                num_workers=num_workers, # Workers for this WebLoader instance
                batch_size=current_batch_size, 
                epoch_size=current_epoch_size, 
                # WDS specific args from cfg, ensure they are named as expected by the pipeline
                shuffle_buffer_load=cfg.get("wds_shuffle_buffer_load", 1000),
                shuffle_buffer_repeat=cfg.get("wds_shuffle_buffer_repeat", 5000),
                # MCoT specific args from cfg for WDS pipelines
                image_file_key=cfg.get("image_file_key", "image.jpg"), # Example default
                caption_prompt_key=cfg.get("caption_prompt_key", "caption_prompt.txt"), # Example
                target_plan_text_key=cfg.get("target_plan_text_key", "target_plan_text.txt"), # Example
                bbox_json_key=cfg.get("bbox_json_key", "bboxes.json"), # Example
                **wds_loader_kwargs # Pass remaining cfg items
            )
            all_dataset_pipes.append(data_iterable)

        elif cfg.type == "huggingface":
            print(f"INFO: Setting up HuggingFace dataset {dataset_name} (split: {current_split}, tasks: {cfg.get('coco_task', 'N/A')})")
            loader_fn = None
            # MCoT specific HuggingFace pipelines
            if "planning" in dataset_name.lower(): # e.g., coco_planning, coco_planning_val
                loader_fn = get_mcot_planning_huggingface_pipeline
            # elif "acting" in dataset_name.lower():
            #     loader_fn = get_mcot_acting_huggingface_pipeline # Define if needed
            else:
                # Generic HuggingFace pretraining dataloader (if you have one)
                # loader_fn = build_huggingface_pretraining_dataloader # Example, ensure it returns unbatched pipe
                print(f"Warning: No specific MCoT HuggingFace loader for {dataset_name} of type {cfg.type}. Skipping.")
                continue

            hf_pipeline_kwargs = {
                k: v for k, v in cfg.items() if k not in [
                    'type', 'data_path', 'epoch_size', 'batch_size', 'split', 
                    'in_domains', 'out_domains', 'year', 'coco_task', 'streaming',
                    'hf_shuffle_buffer_load', 'hf_shuffle_seed' # Handled by loader_fn
                ]
            }

            # This datapipe from HF pipeline yields *unbatched* samples.
            # It will be consumed by MixtureDataset, then batched by build_mixture_dataloader.
            datapipe = loader_fn(
                data_path=cfg.data_path,
                all_domains=current_all_domains, 
                modality_info=modality_info, 
                modality_transforms=modality_transforms, 
                image_augmenter=image_augmenter, 
                text_tokenizer=text_tokenizer,
                input_tokens_range=input_tokens_range,
                target_tokens_range=target_tokens_range,
                num_gpus=get_world_size(), 
                num_workers=num_workers, # These workers are for the WebLoader created by build_mixture_dataloader
                batch_size=current_batch_size, # Passed for context, not applied by HF pipe itself
                epoch_size=current_epoch_size, # Total samples for this dataset part for the epoch
                split=current_split,
                year=cfg.get("year", 2017), 
                coco_task=cfg.get("coco_task", "instances"), 
                streaming=cfg.get("streaming", True),
                shuffle_buffer_load=cfg.get("hf_shuffle_buffer_load", 10_000), # For HF streaming shuffle
                shuffle_seed=cfg.get("hf_shuffle_seed", 0), # For HF streaming shuffle
                **hf_pipeline_kwargs # Pass remaining cfg items
            )
            all_dataset_pipes.append(datapipe)
            
        else:
            print(f"Warning: Unknown dataset type '{cfg.type}' for {dataset_name}. Skipping.")
            continue

    if not all_dataset_pipes:
        if not is_val: # For training, this is usually an error.
            print("ERROR: No valid training data iterators/pipes were created. Check dataset configurations and types.")
        else: # For validation, it might be acceptable to have no val sets.
            print("INFO: No validation datasets configured or loaded.")
        # build_mixture_dataloader can handle empty all_dataset_pipes and return an empty loader.
        # So, we let it proceed. The training loop should handle an empty dataloader.
        
    unified_masking_instance = UnifiedMasking(
        modality_info=modality_info, text_tokenizer=text_tokenizer,
        input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
        # Add other UnifiedMasking params from config if they exist e.g. force_end_of_generation_token_for_text_target
    )

    final_dataloader = build_mixture_dataloader(
        data_iters=all_dataset_pipes, 
        weights=sampling_weights, 
        modality_info=modality_info, 
        batch_size=dataset_batch_size, # Global batch size per GPU for the mixed data
        num_workers=num_workers, # Workers for the final WebLoader created by build_mixture_dataloader
        epoch_size=epoch_size, # Overall epoch_size for the mixed dataset
        num_gpus=get_world_size(),
        collate_fn=unified_masking_instance 
    )
    
    # Calculate num_training_steps_per_epoch based on the final dataloader
    num_training_steps_per_epoch = 0
    if hasattr(final_dataloader, '__len__') and len(final_dataloader) > 0:
        num_training_steps_per_epoch = len(final_dataloader)
    elif epoch_size is not None and dataset_batch_size > 0 and get_world_size() > 0 and sum(sampling_weights) > 0 : # Ensure there's data to be loaded
        # Fallback calculation if __len__ is not set (e.g. if epoch_size was None for build_mixture_dataloader but set globally)
        num_training_steps_per_epoch = epoch_size // (dataset_batch_size * get_world_size())
        if num_training_steps_per_epoch == 0 and epoch_size > 0 : # If total epoch size is very small
             print(f"Warning: num_training_steps_per_epoch calculated as 0. epoch_size={epoch_size}, global_batch_size={dataset_batch_size * get_world_size()}. Ensure epoch_size is sufficient.")


    if not is_val:
        print(f"INFO: Created training dataloader. Estimated steps per epoch: {num_training_steps_per_epoch}")
    else:
        print(f"INFO: Created validation dataloader. Estimated steps per epoch: {num_training_steps_per_epoch}")

    return final_dataloader, num_training_steps_per_epoch
