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
MCOT dataset handling for the 4M model
This extends unified_datasets.py to support the Multimodal Chain of Thought paradigm
with stage markers and multi-task learning.
"""

import copy
import os
from typing import Dict, List, Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from fourm.data.masking import UnifiedMasking
from fourm.data.modality_transforms import UnifiedDataTransform
from fourm.data.unified_datasets import (
    build_fm_pretraining_dataset,
    build_wds_fm_pretraining_dataloader,
    MixtureDataset
)
from fourm.utils import get_sentinel_to_id_mapping
from fourm.utils.dist import get_rank, get_world_size


# MCOT special token constants
PLANNING_START_TOKEN = "[PLANNING_START]"
ACTING_START_TOKEN = "[ACTING_START]"
REFLECTION_START_TOKEN = "[REFLECTION_START]"
CORRECTION_START_TOKEN = "[CORRECTION_START]"


class MCOTUnifiedMasking(UnifiedMasking):
    """
    Extends UnifiedMasking to handle MCOT special tokens and task-specific masking.
    This class applies different masks based on the stage markers (Planning, Reflection, Correction).
    """

    def __init__(self, modality_info, text_tokenizer, input_tokens_range, target_tokens_range,
                 sampling_weights=None, mcot_task_markers=None):
        """
        Args:
            modality_info: Dictionary containing information about the modalities.
            text_tokenizer: Text tokenizer.
            input_tokens_range: Range for the input token budget.
            target_tokens_range: Range for the target token budget.
            sampling_weights: Sampling weights for the mixture of Dirichlet distributions.
            mcot_task_markers: Dictionary mapping task names to their marker tokens.
        """
        super().__init__(modality_info, text_tokenizer, input_tokens_range, target_tokens_range, sampling_weights)

        # Set up task markers with defaults if not provided
        self.mcot_task_markers = mcot_task_markers or {
            "planning": PLANNING_START_TOKEN,
            "acting": ACTING_START_TOKEN,
            "reflection": REFLECTION_START_TOKEN,
            "correction": CORRECTION_START_TOKEN,
        }

        # Get token IDs for the special markers
        self.marker_token_ids = {}
        vocab = text_tokenizer.get_vocab()
        for task, marker in self.mcot_task_markers.items():
            if marker in vocab:
                self.marker_token_ids[task] = vocab[marker]
            else:
                print(f"Warning: {marker} token not found in vocabulary")

        # Map tasks to specific input/target modalities
        # Based on update.pdf description [cite: 40, 41]
        self.task_modality_config = {
            "planning": {
                "input": ["rgb", "text"], # "prompt input" implies text [cite: 17]
                "target": ["caption", "bbox"] # dense caption/layout tokens [cite: 17]
            },
            "reflection": {
                "input": ["rgb", "text"], # "generated image and the original prompt" [cite: 18]
                "target": ["heatmap"] # heatmap tokens [cite: 18]
            },
            "correction": {
                 # generated image, heatmap, and an appropriate mask [cite: 19]
                 # Mapping 'heatmap' and 'mask' to 'segmentation' based on context
                "input": ["rgb", "text", "segmentation"],
                "target": ["rgb"] # corrected image tokens [cite: 19]
            },
            "vqa": { # Standard VQA
                "input": ["rgb", "text"],
                "target": ["caption"] # Assuming answer maps to caption
            }
            # Note: Acting stage is handled at inference [cite: 44]
        }

    def _detect_mcot_task(self, sample):
        """
        Detect the MCOT task based on sample contents (target modalities).
        This assumes target modalities are unique identifiers for the tasks
        as defined in task_modality_config.
        """
        if "bbox" in sample and "caption" in sample:
            return "planning"
        elif "heatmap" in sample:
            return "reflection"
        elif "segmentation" in sample: # Input for correction
             # If 'rgb' is the only target, it's likely correction
            if set(self.task_modality_config["correction"]["target"]) == {k for k, v in sample.items() if isinstance(v, dict) and 'target_mask' in v and v['target_mask'].any()}:
                 return "correction"
        # Default to VQA if no specific MCOT target modalities are found
        return "vqa"

    def _prepend_stage_marker(self, sample, task):
        """
        Prepend the appropriate stage marker token to the text input.
        Assumes 'text' is the key for the text modality.
        """
        if task not in self.marker_token_ids:
            print(f"Warning: No marker defined for task '{task}'")
            return sample

        marker_id = self.marker_token_ids[task]
        text_modality_key = "text" # Assuming 'text' is the key

        if text_modality_key in sample and isinstance(sample[text_modality_key], torch.Tensor):
            text_tensor = sample[text_modality_key]

            # Simple prepend: create a new tensor with marker + original text
            # Assumes text_tensor is 1D
            new_tensor_list = [torch.tensor([marker_id], dtype=text_tensor.dtype, device=text_tensor.device)]
            
            # Find first non-padding token (assuming 0 is padding)
            non_padding_indices = torch.nonzero(text_tensor).squeeze()
            if non_padding_indices.numel() > 0:
                 first_token_idx = non_padding_indices[0] if non_padding_indices.dim() > 0 else non_padding_indices
                 new_tensor_list.append(text_tensor[first_token_idx:])
            
            new_tensor = torch.cat(new_tensor_list)

            # Pad back to original length if necessary (simple right padding)
            original_length = text_tensor.shape[0]
            if new_tensor.shape[0] < original_length:
                padding = torch.zeros(original_length - new_tensor.shape[0], dtype=new_tensor.dtype, device=new_tensor.device)
                new_tensor = torch.cat([new_tensor, padding])
            elif new_tensor.shape[0] > original_length:
                 new_tensor = new_tensor[:original_length] # Truncate if exceeds somehow

            sample[text_modality_key] = new_tensor
        elif text_modality_key in sample:
             print(f"Warning: '{text_modality_key}' in sample but is not a Tensor.")
        # else: No text modality to prepend to

        return sample

    def _apply_task_specific_masks(self, sample, task, input_mask_dict, target_mask_dict):
        """
        Apply task-specific masks based on the detected task.
        Masks are applied to the dictionaries generated by the parent class.
        A mask value of 1 means masked (ignore), 0 means visible.
        """
        if task not in self.task_modality_config:
            print(f"Warning: Task '{task}' not found in task_modality_config. Using default masks.")
            return input_mask_dict, target_mask_dict

        task_config = self.task_modality_config[task]

        # INPUT MASKING: Mask modalities NOT needed for the task's input [cite: 40]
        required_inputs = set(task_config.get("input", []))
        for mod_key in input_mask_dict:
             mod_base_name = mod_key.split('_')[0] # Handle potential suffixes like _seqlen
             if mod_base_name in required_inputs:
                 input_mask_dict[mod_key].fill_(0) # Visible
             else:
                 input_mask_dict[mod_key].fill_(1) # Masked

        # TARGET MASKING: Mask modalities NOT needed for the task's target [cite: 41]
        required_targets = set(task_config.get("target", []))
        for mod_key in target_mask_dict:
            mod_base_name = mod_key.split('_')[0]
            if mod_base_name in required_targets:
                target_mask_dict[mod_key].fill_(0) # Visible (predict this)
            else:
                target_mask_dict[mod_key].fill_(1) # Masked (don't predict)

        return input_mask_dict, target_mask_dict

    def __call__(self, sample):
        """
        Apply MCOT-specific masking to the sample.
        """
        # Create a deep copy to avoid modifying the original sample dict in-place
        # especially important if used within datasets/dataloaders
        processed_sample = copy.deepcopy(sample)

        # Detect MCOT task based on contents of the *original* sample
        task = self._detect_mcot_task(sample)

        # Prepend the stage marker token to the text modality *before* masking
        processed_sample = self._prepend_stage_marker(processed_sample, task)

        # Now apply standard unified masking from the parent class
        # This will generate initial input/target masks based on sampling strategy
        processed_sample = super().__call__(processed_sample)

        # Extract the masks generated by the parent call
        # These masks are based on the general unified masking logic
        input_mask_dict = {}
        target_mask_dict = {}
        modalities_in_sample = list(processed_sample.keys()) # Keys might have changed

        for mod in modalities_in_sample:
             # Check if the item is a dictionary containing the mask structure
             if isinstance(processed_sample[mod], dict):
                  if 'input_mask' in processed_sample[mod]:
                       input_mask_dict[mod] = processed_sample[mod]['input_mask']
                  if 'target_mask' in processed_sample[mod]:
                       target_mask_dict[mod] = processed_sample[mod]['target_mask']

        # Apply MCOT task-specific masks, potentially overriding parent masks
        # This ensures only relevant modalities are visible for input/target per task
        input_mask_dict, target_mask_dict = self._apply_task_specific_masks(
             processed_sample, task, input_mask_dict, target_mask_dict
        )

        # Update the processed sample with the final task-specific masks
        for mod in input_mask_dict:
             if mod in processed_sample and isinstance(processed_sample[mod], dict) and 'input_mask' in processed_sample[mod]:
                  processed_sample[mod]['input_mask'] = input_mask_dict[mod]

        for mod in target_mask_dict:
             if mod in processed_sample and isinstance(processed_sample[mod], dict) and 'target_mask' in processed_sample[mod]:
                  processed_sample[mod]['target_mask'] = target_mask_dict[mod]

        return processed_sample


def build_mcot_dataset(
        data_path, all_domains, modality_info, modality_transforms,
        image_augmenter, text_tokenizer,
        input_tokens_range, target_tokens_range,
        sampling_weights=None, mcot_task_markers=None):
    """
    Builds the MCOT dataset based on the given arguments.
    This extends the regular FourM dataset with MCOT-specific processing.

    Args:
        data_path: Path to the dataset.
        all_domains: List of all modalities to be used.
        modality_info: Dictionary containing information about the modalities.
        modality_transforms: Dictionary containing the transforms for each modality.
        image_augmenter: Image augmenter.
        text_tokenizer: Text tokenizer.
        input_tokens_range: Range for the input token budget.
        target_tokens_range: Range for the target token budget.
        sampling_weights: Sampling weights for the mixture of Dirichlet distributions.
        mcot_task_markers: Dictionary mapping task names to their marker tokens.

    Returns:
        MCOT dataset as a PyTorch Dataset.
    """
    # Define the sequence of transformations
    # 1. UnifiedDataTransform: Applies basic modality transforms and augmentation.
    # 2. MCOTUnifiedMasking: Adds MCOT stage markers and applies task-specific masking.
    transform = torch.nn.Sequential(
        UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
        MCOTUnifiedMasking(modality_info=modality_info, text_tokenizer=text_tokenizer,
                          input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
                          sampling_weights=sampling_weights, mcot_task_markers=mcot_task_markers),
    )

    # Use the standard dataset building function but pass our combined transform
    # Note: The standard build_fm_pretraining_dataset might internally create its own
    # UnifiedMasking. We might need to modify build_fm_pretraining_dataset or
    # create a parallel version that accepts a pre-defined transform sequence.
    # Assuming build_fm_pretraining_dataset can accept a custom transform/masking strategy:
    # (This is a hypothetical adaptation, the actual 4M codebase might need changes)

    # --- Hypothetical Adaptation ---
    # Assuming build_fm_pretraining_dataset has an argument like `custom_transform`
    # return build_fm_pretraining_dataset(
    #     data_path=data_path,
    #     all_domains=all_domains,
    #     modality_info=modality_info,
    #     # Pass modality transforms separately if UnifiedDataTransform isn't applied inside build_fm...
    #     # modality_transforms=modality_transforms,
    #     # image_augmenter=image_augmenter,
    #     text_tokenizer=text_tokenizer,
    #     # input_tokens_range=input_tokens_range, # Masking handles this
    #     # target_tokens_range=target_tokens_range, # Masking handles this
    #     # sampling_weights=sampling_weights, # Masking handles this
    #     custom_transform=transform # Pass our combined transform sequence
    # )
    # --- End Hypothetical Adaptation ---

    # --- Alternative: Assume build_fm_pretraining_dataset returns a base dataset ---
    # ---           and we wrap it with our MCOT transform.                 ---
    # This is more likely if build_fm_pretraining_dataset is less flexible.
    base_dataset = build_fm_pretraining_dataset(
         data_path=data_path,
         all_domains=all_domains,
         modality_info=modality_info,
         modality_transforms=modality_transforms, # Applied by UnifiedDataTransform later
         image_augmenter=image_augmenter, # Applied by UnifiedDataTransform later
         text_tokenizer=text_tokenizer,
         input_tokens_range=input_tokens_range, # Needed by MCOTUnifiedMasking
         target_tokens_range=target_tokens_range, # Needed by MCOTUnifiedMasking
         sampling_weights=sampling_weights # Needed by MCOTUnifiedMasking
         # Note: The internal masking of build_fm_pretraining_dataset might conflict.
         # Ideally, disable internal masking if possible, or ensure MCOTUnifiedMasking runs last.
    )

    # Wrap the base dataset with our MCOT transform sequence
    # Need a simple Dataset wrapper that applies the transform
    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, transform):
            self.base_dataset = base_dataset
            self.transform = transform

        def __len__(self):
            # Delegate length check to the base dataset
            if hasattr(self.base_dataset, '__len__'):
                 return len(self.base_dataset)
            else:
                 # Handle iterable-style datasets if needed (might require epoch size)
                 raise NotImplementedError("Base dataset must have __len__")

        def __getitem__(self, idx):
            sample = self.base_dataset[idx]
            return self.transform(sample)

    return TransformedDataset(base_dataset, transform)


def build_wds_mcot_dataloader(
        data_path, all_domains, modality_info, modality_transforms, image_augmenter,
        text_tokenizer, input_tokens_range, target_tokens_range,
        num_gpus, num_workers, batch_size, epoch_size, sampling_weights=None, modality_name_map=None,
        shuffle_buffer_load=1000, shuffle_buffer_repeat=5000, n_repeats=5, mcot_task_markers=None):
    """
    Builds a WebDataset dataloader for MCOT training.
    This extends the regular WDS dataloader with MCOT-specific processing.

    Args:
        (Same as build_wds_fm_pretraining_dataloader with additional mcot_task_markers parameter)

    Returns:
        MCOT dataloader as a PyTorch DataLoader.
    """

    # Define the MCOT processing function to be applied to each sample from WebDataset
    # This function encapsulates both the standard transforms and the MCOT masking
    def mcot_process_fn(sample):
        # 1. Apply standard unified transforms (including augmentation)
        transform = UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter)
        processed_sample = transform(sample)

        # 2. Apply MCOT-specific masking (includes marker prepending and task masking)
        masking = MCOTUnifiedMasking(modality_info=modality_info, text_tokenizer=text_tokenizer,
                                    input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
                                    sampling_weights=sampling_weights, mcot_task_markers=mcot_task_markers)
        processed_sample = masking(processed_sample)

        return processed_sample

    # Use the standard WDS dataloader builder, but override the default processing
    # by passing our custom `mcot_process_fn` via the `map_fns` or similar argument.
    # The exact mechanism depends on how `build_wds_fm_pretraining_dataloader` is implemented.

    # --- Hypothetical Adaptation ---
    # Assuming build_wds_fm_pretraining_dataloader accepts a `custom_processor` argument:
    # return build_wds_fm_pretraining_dataloader(
    #     data_path=data_path,
    #     all_domains=all_domains,
    #     modality_info=modality_info,
    #     # These might not be needed if custom_processor handles everything
    #     # modality_transforms=modality_transforms,
    #     # image_augmenter=image_augmenter,
    #     text_tokenizer=text_tokenizer,
    #     # input_tokens_range=input_tokens_range, # Handled by processor
    #     # target_tokens_range=target_tokens_range, # Handled by processor
    #     num_gpus=num_gpus,
    #     num_workers=num_workers,
    #     batch_size=batch_size,
    #     epoch_size=epoch_size,
    #     sampling_weights=sampling_weights, # Handled by processor
    #     modality_name_map=modality_name_map,
    #     shuffle_buffer_load=shuffle_buffer_load,
    #     shuffle_buffer_repeat=shuffle_buffer_repeat,
    #     n_repeats=n_repeats,
    #     custom_processor=mcot_process_fn # Pass our function
    # )
    # --- End Hypothetical Adaptation ---

    # --- Alternative: Modify the internal processing pipeline ---
    # If the standard builder doesn't allow easy overriding, modification might be needed.
    # The goal is to replace the default map/transform step with `mcot_process_fn`.
    # This requires inspecting the implementation of `build_wds_fm_pretraining_dataloader`.

    # Placeholder: Return None or raise error if adaptation assumption is wrong.
    raise NotImplementedError("Integration with build_wds_fm_pretraining_dataloader requires checking its implementation details for custom processing hooks.")
    # return None


def build_mcot_mixture_dataloader(data_iters, weights, modality_info, batch_size, num_workers, epoch_size, num_gpus):
    """
    Builds a mixture dataloader for MCOT multi-task training.
    This combines multiple task-specific dataloaders (as iterators) for Planning, Reflection, Correction, and VQA tasks.

    Args:
        data_iters: Dictionary mapping task names (e.g., 'planning', 'vqa') to their corresponding data iterators.
                    Each iterator should yield processed samples ready for MCOT.
        weights: Dictionary mapping task names to their sampling weights (probabilities).
        modality_info: Dictionary containing information about the modalities (needed by MixtureDataset).
        batch_size: Batch size per GPU.
        num_workers: Number of workers for the dataloader.
        epoch_size: Total number of samples constituting one epoch across all GPUs.
        num_gpus: Number of GPUs for distributed training.

    Returns:
        MCOT mixture dataloader as a PyTorch DataLoader.
    """
    # Ensure iterators and weights match
    if set(data_iters.keys()) != set(weights.keys()):
        raise ValueError("Keys in data_iters and weights must match.")

    # Create the MixtureDataset which handles sampling from different task iterators
    dataset = MixtureDataset(data_iters=data_iters, weights=weights, modality_info=modality_info)


    # Custom collate function to handle potentially heterogeneous batches
    # Although MixtureDataset yields one sample at a time, the DataLoader groups them.
    # If samples from different tasks have different structures (e.g., different modalities present),
    # default_collate might fail. We need a more robust collate function.
    def mcot_mixture_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collates a batch of samples potentially coming from different MCOT tasks.
        It pads or handles missing modalities appropriately.
        """
        # Get all unique modality keys present across the batch
        all_keys = set()
        for sample in batch:
            all_keys.update(sample.keys())

        collated_batch = {}
        # Use the first sample to infer types and device for padding
        elem = batch[0]

        for key in all_keys:
            # Check if the key represents a modality dictionary (e.g., {'rgb': {'data': tensor, 'input_mask': tensor}})
            # This structure is typical in 4M after transforms/masking.
            is_modality_dict = isinstance(elem.get(key), dict) and 'data' in elem.get(key, {})

            if is_modality_dict:
                 # Handle modality dictionaries (data, masks, etc.)
                 collated_batch[key] = {}
                 sub_keys = elem[key].keys()
                 for sub_key in sub_keys:
                      tensors_to_stack = []
                      requires_padding = False
                      max_len = 0
                      ref_tensor = None # Reference for dtype/device

                      # Collect tensors and check if padding is needed
                      for sample in batch:
                           if key in sample and sub_key in sample[key] and isinstance(sample[key][sub_key], torch.Tensor):
                                tensor = sample[key][sub_key]
                                tensors_to_stack.append(tensor)
                                if ref_tensor is None: ref_tensor = tensor
                                if tensor.ndim > 0: # Only pad sequences (dim > 0)
                                     max_len = max(max_len, tensor.shape[0])
                                     if tensor.shape[0] != elem[key][sub_key].shape[0]:
                                          requires_padding = True
                           else:
                                # Handle missing sub_key: need to create appropriate padding later
                                tensors_to_stack.append(None) # Placeholder for missing tensor
                                requires_padding = True # Need padding if any sample is missing the tensor

                      if ref_tensor is None: # If no sample had this sub_key
                            continue

                      # Pad and stack if necessary
                      if requires_padding and max_len > 0:
                            padded_tensors = []
                            for tensor in tensors_to_stack:
                                if tensor is None:
                                    # Create padding tensor (e.g., zeros) matching the shape
                                    pad_shape = list(ref_tensor.shape)
                                    pad_shape[0] = max_len
                                    padding = torch.zeros(pad_shape, dtype=ref_tensor.dtype, device=ref_tensor.device)
                                    # Specific handling for masks (often filled with 1s for padding)
                                    if 'mask' in sub_key:
                                        padding.fill_(1)
                                    padded_tensors.append(padding)
                                elif tensor.shape[0] < max_len:
                                    pad_width = max_len - tensor.shape[0]
                                    # Assuming padding value is 0 for data, 1 for masks
                                    pad_value = 1 if 'mask' in sub_key else 0
                                    padding = torch.full((pad_width,) + tensor.shape[1:], pad_value, dtype=tensor.dtype, device=tensor.device)
                                    padded_tensors.append(torch.cat([tensor, padding], dim=0))
                                else:
                                    padded_tensors.append(tensor)
                            collated_batch[key][sub_key] = torch.stack(padded_tensors, dim=0)
                      else:
                            # If no padding needed or tensors are 0-dim (e.g., scalars)
                             valid_tensors = [t for t in tensors_to_stack if t is not None]
                             if valid_tensors:
                                 try:
                                     collated_batch[key][sub_key] = torch.stack(valid_tensors, dim=0)
                                 except RuntimeError as e:
                                     print(f"Error stacking {key}/{sub_key}: {e}. Tensors: {[t.shape for t in valid_tensors]}")
                                     # Fallback or error handling
                                     collated_batch[key][sub_key] = valid_tensors # Store as list if stacking fails
                             # else: leave sub_key out if no sample had it
            else:
                 # Handle non-modality dictionary items (e.g., metadata, IDs) using default collate logic
                 items_to_collate = [sample[key] for sample in batch if key in sample]
                 if items_to_collate:
                      try:
                           collated_batch[key] = default_collate(items_to_collate)
                      except Exception as e:
                            print(f"Warning: Could not collate key '{key}' with default_collate: {e}. Storing as list.")
                            collated_batch[key] = items_to_collate # Store as list if default collate fails

        return collated_batch


    # Calculate total batch size across all GPUs
    total_batch_size = batch_size * num_gpus

    # Create the DataLoader
    # Note: MixtureDataset is an iterable-style dataset, so __len__ might not be defined.
    # DataLoader handles this, but epoch length is determined by `epoch_size`.
    # Sampler needs to be handled carefully for distributed training with iterable datasets.
    # Often, epoch_size determines how many samples are drawn per epoch.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, # Batch size per worker/GPU
        # sampler=None, # Use default sampler logic for iterable datasets
        num_workers=num_workers,
        collate_fn=mcot_mixture_collate, # Use our custom collate
        pin_memory=True, # Recommended for GPU training
        drop_last=True # Ensure all batches have the same size, important for some distributed modes
    )

    # Wrap with an epoch-limiting iterator if epoch_size is provided
    # This is common practice for iterable datasets like WebDataset or MixtureDataset
    if epoch_size is not None:
        from torch.utils.data import IterDataPipe
        from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

        class EpochWrapper(IterDataPipe):
            def __init__(self, datapipe, epoch_samples):
                self.datapipe = datapipe
                self.epoch_samples = epoch_samples // get_world_size() # Samples per GPU

            def __iter__(self):
                count = 0
                for data in self.datapipe:
                    if count >= self.epoch_samples:
                        return # Stop iteration for this epoch
                    yield data
                    # Assuming batch size is handled by DataLoader, count batches
                    count += 1 # This assumes dataloader yields batches
                    # If dataloader yields samples, need count += batch_size

            def __len__(self):
                 # Provide length for progress bars etc.
                 # Assumes dataloader yields batches
                 return self.epoch_samples

        # Need to handle the internal iterator of the DataLoader
        # This might require custom logic or relying on libraries like `webdataset` patterns
        # For simplicity, let's assume we wrap the dataloader itself,
        # although wrapping the underlying datapipe before DataLoader is often better.
        # This simple wrapper might not work perfectly with multiprocessing workers.

        # A more robust way involves controlling the sampler or using dataloader iter directly
        # For now, returning the raw dataloader and letting the training loop handle epoch size.
        print(f"Warning: Mixture dataloader created. Epoch size ({epoch_size}) needs to be handled by the training loop.")

    return dataloader


def build_mcot_huggingface_dataset(
        dataset_name, split, all_domains, modality_info, modality_transforms,
        image_augmenter, text_tokenizer,
        input_tokens_range, target_tokens_range,
        sampling_weights=None, mcot_task_markers=None):
    """
    Builds the MCOT dataset from a Hugging Face dataset.

    Args:
        dataset_name: Name of the Hugging Face dataset.
        split: Dataset split (train, validation, test).
        all_domains: List of all modalities expected by the MCOT pipeline.
        modality_info: Dictionary containing information about the modalities.
        modality_transforms: Dictionary containing the transforms for each modality.
        image_augmenter: Image augmenter.
        text_tokenizer: Text tokenizer.
        input_tokens_range: Range for the input token budget.
        target_tokens_range: Range for the target token budget.
        sampling_weights: Sampling weights for the mixture of Dirichlet distributions.
        mcot_task_markers: Dictionary mapping task names to their marker tokens.

    Returns:
        MCOT dataset as a PyTorch Dataset.
    """
    from datasets import load_dataset
    
    # Load the dataset from Hugging Face
    hf_dataset = load_dataset(dataset_name, split=split)
    
    # Create a PyTorch dataset adapter
    class HuggingFaceAdapter(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, all_domains, transform=None):
            self.hf_dataset = hf_dataset
            self.all_domains = all_domains
            self.transform = transform
            
        def __len__(self):
            return len(self.hf_dataset)
            
        def __getitem__(self, idx):
            example = self.hf_dataset[idx]
            
            # Convert HF example to MCOT format
            # This conversion depends on the specific dataset structure
            sample = {}
            
            # Map domain-specific data
            for domain in self.all_domains:
                if domain == "rgb" and "image" in example:
                    # Convert image
                    sample[domain] = self._process_image(example["image"])
                elif domain == "text" and "question" in example:
                    # Convert text
                    sample[domain] = example["question"]
                elif domain == "caption" and "answer" in example:
                    # Convert caption
                    sample[domain] = example["answer"]
                # Add more domain-specific conversions as needed
            
            # Apply transforms if provided
            if self.transform is not None:
                sample = self.transform(sample)
                
            return sample
        
        def _process_image(self, image):
            # Process image based on the expected format
            # This may involve resizing, converting to tensor, etc.
            return image
    
    # Create transform
    transform = torch.nn.Sequential(
        UnifiedDataTransform(transforms_dict=modality_transforms, image_augmenter=image_augmenter),
        MCOTUnifiedMasking(modality_info=modality_info, text_tokenizer=text_tokenizer,
                          input_tokens_range=input_tokens_range, target_tokens_range=target_tokens_range,
                          sampling_weights=sampling_weights, mcot_task_markers=mcot_task_markers),
    )
    
    # Create dataset adapter
    return HuggingFaceAdapter(hf_dataset, all_domains, transform=transform)