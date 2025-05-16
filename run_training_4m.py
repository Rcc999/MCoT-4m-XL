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
import argparse
import datetime
import json
import math
import os
import resource
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from tokenizers import Tokenizer

import fourm.utils as utils
from fourm.data import (
    build_mixture_dataloader,
    get_val_dataloader, 
    setup_sampling_mod_info,
    EmptyAugmenter,
)
from fourm.data.pretrain_utils import get_train_dataloader as get_pretrain_default_train_dataloader
from fourm.data.unified_datasets import (
    get_mcot_planning_data_pipeline,
    get_mcot_acting_data_pipeline,
    get_mcot_reflection_data_pipeline,
    get_mcot_correction_data_pipeline,
    get_train_dataloader as get_mcot_unified_train_dataloader,
    UnifiedMasking,
)
from fourm.models import fm
from fourm.data.modality_info import MODALITY_INFO
from fourm.utils import NativeScalerWithGradNormCount as NativeScaler
from fourm.utils import create_model
from fourm.utils.optim_factory import create_optimizer


def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('4M pre-training script (using DDP)', add_help=True)
    parser.add_argument('--run_name', type=str, default='auto')

    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (default: %(default)s). '
                             'Effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs (default: %(default)s)')
    parser.add_argument('--total_tokens', default=-1, type=int,
                        help='Number of total input tokens (in billions), only applicable if epochs is negative. '
                             'Sets the number of epochs to approximate this amount of tokens.')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--save_ckpt_freq', default=20, type=int,
                        help='Checkpoint saving frequency in epochs (default: %(default)s)')

    # Model parameters
    parser.add_argument('--model', default='fm_base_12e_12d_swiglu_nobias', type=str,
                        help='Name of model to train (default: %(default)s)')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Base patch size for image-like modalities (default: %(default)s)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Images input size for backbone (default: %(default)s)')
    parser.add_argument('--num_register_tokens', default=0, type=int,
                        help='Number of register tokens to add to encoder (default: %(default)s)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32', 'bf16', 'fp16', 'fp32'],
                        help='Data type (default: %(default)s')

    parser.add_argument('--num_input_tokens', type=int, default=128, 
                        help="Token budget for the input")
    parser.add_argument('--num_target_tokens', type=int, default=128, 
                        help="Token budget for the target")
    parser.add_argument('--min_input_tokens', type=int, default=None,
                        help="Minimum token budget for the input (None to set it to num_input_tokens)")
    parser.add_argument('--min_target_tokens', type=int, default=None,
                        help="Minimum token budget for the target (None to set it to num_target_tokens)")
    
    parser.add_argument('--loss_type', type=str, choices=['mod', 'token'], default='mod',
                        help="If mod, loss is the mean of the per-modality loss. If token, loss is the mean of the per-token loss (default: %(default)s)")


    # Weight init / fine-tune parameters
    parser.add_argument('--finetune', default='', help='finetune from checkpoint (for two-stage training)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str,
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--opt_eps', default=1e-8, type=float,
                        help='Optimizer epsilon (default: %(default)s)')
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+',
                        help='Optimizer betas (default: %(default)s)')
    parser.add_argument('--compute_grad_norm', action='store_true')
    parser.add_argument('--no_compute_grad_norm', action='store_false', dest='compute_grad_norm')
    parser.set_defaults(compute_grad_norm=True)
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Clip gradient norm (default: %(default)s)')
    parser.add_argument('--skip_grad', type=float, default=None,
                        help='Skip update if gradient norm larger than threshold (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: %(default)s)')
    parser.add_argument('--weight_decay_end', type=float, default=None, 
                        help="Final value of the weight decay. (Set the same value as args.weight_decay to keep weight decay value constant)")

    parser.add_argument('--blr', type=float, default=1e-4, 
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256 (default: %(default)s)')
    parser.add_argument('--min_blr', type=float, default=0., 
                        help='Lower base lr bound for cyclic schedulers that hit 0 (default: %(default)s)')
    parser.add_argument('--frozen_model_blr', type=float, default=-1, 
                        help='base lr bound for frozen model (default: %(default)s)')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'inverse_sqrt-10000'],
                        help='Learning rate scheduler type (default: %(default)s')
    parser.add_argument('--warmup_epochs', type=int, default=10, 
                        help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--warmup_steps', type=int, default=-1, 
                        help='Steps to warmup LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--warmup_tokens', type=int, default=-1,
                        help='Total tokens to warmup LR, if scheduler supports (default: %(default)s)')
     # Cooldown for inverse sqrt and other "infinite" LR schedules
    parser.add_argument('--cooldown_epochs', type=int, default=10,
                        help='Epochs to cool down LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--cooldown_steps', type=int, default=-1,
                        help='Steps to cool down LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--cooldown_tokens', type=int, default=-1,
                        help='Total tokens to cool down LR, if scheduler supports (default: %(default)s)')
    # For warm-starting from a trained model
    parser.add_argument('--frozen_model_epochs', default=0, type=int,
                        help='Number of epochs where only input/output embeddings are trained (default: %(default)s)')
    parser.add_argument('--frozen_model_tokens', default=0, type=int,
                        help='Number of tokens where only input/output embeddings are trained (default: %(default)s)')
    parser.add_argument('--frozen_embedding_domain', default=None, type=str,
                        help='Embeddings of domains that are frozen during training (default: %(default)s)')    

    # Dataset parameters
    parser.add_argument('--data_config', type=str, default="",
                        help="Path to data config to specify dataset and modality mixture parameters.")
    parser.add_argument('--epoch_size', type=int, help="Number of samples per epoch")
    parser.add_argument('--s3_endpoint', default='', type=str, help='S3 endpoint URL')
    parser.add_argument('--s3_data_endpoint', default=None, type=str, 
                        help='S3 endpoint URL for the data (if different). If set to None, will be set to s3_endpoint')
    parser.add_argument('--s3_multipart_chunksize_mb', default=512, type=int)
    parser.add_argument('--s3_multipart_threshold_mb', default=512, type=int)
    parser.add_argument('--s3_max_io_queue', default=100, type=int)

    # Text tokenizer
    parser.add_argument('--text_tokenizer_path', default='fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json',
                        help="Path to trained text tokenizer")

    # Eval
    parser.add_argument('--eval_freq', default=10, type=int, help="frequency of evaluation")
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--no_dist_eval', action='store_false', dest='dist_eval',
                    help='Disabling distributed evaluation')
    parser.set_defaults(dist_eval=True)

    parser.add_argument('--fixed_eval', action='store_true')
    parser.add_argument('--no_fixed_eval', action='store_false', dest='fixed_eval')
    parser.set_defaults(fixed_eval=True)
    parser.add_argument('--fixed_eval_input_tokens', default=128, type=int,
                        help="Number of input tokens for the fixed evaluation")
    parser.add_argument('--fixed_eval_target_tokens', default=128, type=int,
                        help="Number of target tokens for the fixed evaluation")
    parser.add_argument('--fixed_eval_batch_size', default=32, type=int,
                        help="Batch size for the fixed evaluation")

    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')

    # Misc.
    parser.add_argument('--output_dir', default='',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')

    parser.add_argument('--seed', default=0, type=int, help='Random seed ')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--no_find_unused_params', action='store_false', dest='find_unused_params')
    parser.set_defaults(find_unused_params=False)

    parser.add_argument('--rlimit', default=4096, type=int, 
                        help='Increase rlimit to avoid "RuntimeError: received 0 items of ancdata".')
    parser.add_argument('--print_all', action='store_true', default=False)
    parser.add_argument('--show_user_warnings', default=False, action='store_true')
    parser.add_argument('--s3_save_dir', type=str, default="")

    # Distributed training parameters
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

     # Wandb logging
    parser.add_argument('--log_wandb', default=False, action='store_true',
                        help='Log training and validation metrics to wandb')
    parser.add_argument('--no_log_wandb', action='store_false', dest='log_wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='Project name on wandb')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='User or team name on wandb')
    parser.add_argument('--wandb_run_name', default='auto', type=str,
                        help='Run name on wandb')

    # Add new arguments for local debugging
    parser.add_argument('--local_debug', action='store_true',
                        help='Run in local debug mode: disables DDP, overrides some args for quick testing.')
    parser.add_argument('--local_batch_size', type=int, default=2, 
                        help="Batch size to use when --local_debug is active (overrides --batch_size).")
    parser.add_argument('--local_output_dir', type=str, default="tmp_local_output",
                        help="Output directory for local debug mode.")

    # Parse config file if there is one
    args_config, remaining = config_parser.parse_known_args()

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)            

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file is specified.
    args = parser.parse_args(remaining)

    # Add the config path as a final args if given
    args.config_path = args_config.config

    return args


def setup_modality_info(args):
    # Global modality info
    modality_info = {mod: MODALITY_INFO[mod] for mod in args.all_domains}
    
    # Max tokens
    for mod in modality_info:
        image_size, patch_size = modality_info[mod].get('input_size', args.input_size), modality_info[mod].get('patch_size', args.patch_size)
        num_patches = (image_size // patch_size) ** 2
        if modality_info[mod]['type'] == 'img':
            modality_info[mod]['max_tokens'] = num_patches

    return modality_info


def setup_data(args):
    # Set number of tokens for the sampling
    if args.min_input_tokens is None:
        args.min_input_tokens = args.num_input_tokens
    if args.min_target_tokens is None:
        args.min_target_tokens = args.num_target_tokens

    # Load text tokenizer
    print(f"Loading text tokenizer from: {args.text_tokenizer_path}")
    # args.text_tokenizer_path = utils.s3_download_file(args.s3_endpoint, args.text_tokenizer_path, args.output_dir)
    text_tokenizer = Tokenizer.from_file(args.text_tokenizer_path)
    mcot_special_tokens = ["[PLANNING_START]", "[ACTING_START]", "[REFLECTION_START]", "[CORRECTION_START]"]
    num_added_toks = text_tokenizer.add_special_tokens(mcot_special_tokens)
    if num_added_toks > 0:
        print(f"Added {num_added_toks} MCoT special tokens to the tokenizer.")
        args.tokenizer_object_for_resize = text_tokenizer
    else:
        print("MCoT special tokens already exist in the tokenizer.")
        args.tokenizer_object_for_resize = text_tokenizer

    # Placeholder: Load other VQ-VAE tokenizers needed for MCoT stages
    # These paths should come from args or a config
    # Example:
    # image_vqvae_tokenizer_path = getattr(args, 'image_vqvae_tokenizer_path', None)
    # semantic_seg_vqvae_tokenizer_path = getattr(args, 'semantic_seg_vqvae_tokenizer_path', None)
    image_vqvae_tokenizer = None # Replace with actual loading if specific MCoT WDS paths are used
    semantic_seg_vqvae_tokenizer = None # Replace with actual loading if specific MCoT WDS paths are used
    # if image_vqvae_tokenizer_path:
    #     print(f"Loading image VQ-VAE tokenizer from: {image_vqvae_tokenizer_path}")
    #     # image_vqvae_tokenizer = YourImageVQVAELoader(image_vqvae_tokenizer_path)
    # if semantic_seg_vqvae_tokenizer_path:
    #     print(f"Loading semantic segmentation VQ-VAE tokenizer from: {semantic_seg_vqvae_tokenizer_path}")
    #     # semantic_seg_vqvae_tokenizer = YourSemanticVQVAELoader(semantic_seg_vqvae_tokenizer_path)

    # Define a default image augmenter for WDS MCoT pipelines if not specified otherwise
    image_augmenter_for_wds_mcot = EmptyAugmenter() # Or more sophisticated based on config


    print(f"Loading data config from: {args.data_config}")
    with open(args.data_config, "r") as f:
        data_config = yaml.safe_load(f)
    
    train_config = data_config['train']['datasets']

    args.in_domains = sorted(set.union(*[set(cfg['in_domains'].split('-')) for cfg in train_config.values()]))
    args.out_domains = sorted(set.union(*[set(cfg['out_domains'].split('-')) for cfg in train_config.values()]))
    args.all_domains = sorted(list(set(args.in_domains) | set(args.out_domains)))

    modality_info = setup_modality_info(args)

    if any([cfg['data_path'].startswith('s3') for cfg in train_config.values()]):
        utils.s3_utils.override_wds_s3_tar_loading(args.s3_data_endpoint, args.s3_multipart_threshold_mb, args.s3_multipart_chunksize_mb, args.s3_max_io_queue)
    
    train_iters = []
    shards_per_dataset = [] 

    # Define a map for MCoT pipeline functions or use if/elif
    DATA_LOADER_FN_MAP_MCOT_SPECIFIC = {
        "mcot_planning": get_mcot_planning_data_pipeline,
        "mcot_acting": get_mcot_acting_data_pipeline,
        "mcot_reflection": get_mcot_reflection_data_pipeline,
        "mcot_correction": get_mcot_correction_data_pipeline,
    }

    for dataset_name, dataset_cfg in train_config.items():
        print(f'Setting up dataset {dataset_name} / train')
        
        # Calculate all_domains for this specific dataset_cfg
        current_in_domains = sorted(dataset_cfg['in_domains'].split('-'))
        current_out_domains = sorted(dataset_cfg['out_domains'].split('-'))
        current_all_domains = sorted(list(set(current_in_domains) | set(current_out_domains)))

        dataset_mod_info, sampling_weights = setup_sampling_mod_info(dataset_cfg, modality_info)
        
        dataset_type = dataset_cfg.get("type")
        dataiter = None

        if dataset_type == "huggingface": # For MCoT datasets specified via Hugging Face
            print(f"Using MCoT unified dataloader for Hugging Face dataset: {dataset_name}")
            # Wrap the single dataset_cfg to match the structure expected by get_train_dataloader
            # which iterates over dataset_config.datasets, where each item is a dict like {name: config}
            dataset_config_wrapper = {'datasets': [{dataset_name: dataset_cfg}]}
            dataiter = get_mcot_unified_train_dataloader(
                dataset_config=dataset_config_wrapper, # Pass the wrapped config
                modality_info=dataset_mod_info,
                text_tokenizer=text_tokenizer,
                input_size=args.input_size,
                num_input_tokens=args.num_input_tokens,
                num_target_tokens=args.num_target_tokens,
                min_input_tokens=args.min_input_tokens,
                min_target_tokens=args.min_target_tokens,
                num_tasks=args.num_tasks,
                num_workers=args.num_workers,
                dataset_batch_size=None, 
                epoch_size=None, 
                sampling_weights=sampling_weights 
            )
        elif dataset_type in DATA_LOADER_FN_MAP_MCOT_SPECIFIC:
            loader_fn = DATA_LOADER_FN_MAP_MCOT_SPECIFIC[dataset_type]
            print(f"Using MCoT specific pipeline '{dataset_type}' for {dataset_name} (likely WDS-based)")
            
            mcot_pipeline_args = {
                "data_path": dataset_cfg['data_path'],
                "all_domains": current_all_domains,
                "modality_info": dataset_mod_info,
                "text_tokenizer": text_tokenizer,
                "image_augmenter": image_augmenter_for_wds_mcot, # Use defined augmenter
                "input_tokens_range": (args.min_input_tokens, args.num_input_tokens),
                "target_tokens_range": (args.min_target_tokens, args.num_target_tokens),
                "num_gpus": args.num_tasks, 
                "num_workers": args.num_workers,
                "batch_size": None, # For mixture dataloader (unbatched)
                "epoch_size": None, # For mixture dataloader (unbatched)
                "shuffle_buffer_load": dataset_cfg.get('wds_shuffle_buffer_tar', 1_000),
                "n_repeats": dataset_cfg.get('wds_n_repeats', 1),
            }
            
            # Add specific VQVAE tokenizers if required by the MCoT stage
            if dataset_type == "mcot_acting":
                mcot_pipeline_args["image_vqvae_tokenizer"] = image_vqvae_tokenizer
            elif dataset_type == "mcot_reflection":
                mcot_pipeline_args["semantic_seg_vqvae_tokenizer"] = semantic_seg_vqvae_tokenizer
            elif dataset_type == "mcot_correction": # Assuming correction might also use image_vqvae_tokenizer
                 mcot_pipeline_args["image_vqvae_tokenizer"] = image_vqvae_tokenizer
            
            dataiter = loader_fn(**mcot_pipeline_args)
        else: # Fallback to general pretraining loader (from pretrain_utils)
            print(f"Warning: Dataset type '{dataset_type}' (or not specified) for '{dataset_name}' not handled by MCoT specific loaders. Falling back to pretrain_utils default loader.")
            dataiter = get_pretrain_default_train_dataloader(
                dataset_config=dataset_cfg, 
                modality_info=dataset_mod_info, 
                sampling_weights=sampling_weights, 
                text_tokenizer=text_tokenizer, 
                input_size=args.input_size, 
                num_input_tokens=args.num_input_tokens, 
                num_target_tokens=args.num_target_tokens,
                min_input_tokens=args.min_input_tokens, 
                min_target_tokens=args.min_target_tokens,
                num_tasks=args.num_tasks, 
                num_workers=args.num_workers,
                dataset_batch_size=None, # For mixture dataloader (unbatched)
                epoch_size=None # For mixture dataloader (unbatched)
            )
        
        if dataiter is not None:
            train_iters.append(dataiter)
            if hasattr(dataiter, 'n_shards'): # WebDataset typically has this
                shards_per_dataset.append(dataiter.n_shards)
            elif isinstance(dataiter, list) and not dataiter: # Empty list from STUB
                print(f"Warning: Data iterator for {dataset_name} is an empty list. Training may fail if weights are non-zero.")
        else:
            # This case should ideally not be reached if fallbacks are robust.
             print(f"ERROR: Failed to create a data iterator for dataset {dataset_name} with type '{dataset_type}'. Check configuration and loader logic.")
             # Consider raising an error here if a data iterator is crucial and not created.
             # For now, it will result in an empty train_iters list if all fail, handled later.

    num_workers = min(min(shards_per_dataset), args.num_workers) if shards_per_dataset else args.num_workers
    
    # Filter out empty iterators from STUBS before passing to MixtureDataset
    # This is a temporary fix. Real iterators should be returned by pipeline functions.
    valid_train_iters = [it for it in train_iters if not (isinstance(it, list) and not it)] # Keep existing filter for stubs
    if not valid_train_iters and any(w > 0 for w in data_config['train'].get('weights', [])):
        # If all iters failed or were stubs, but weights were specified, this is an issue.
        raise ValueError("No valid data iterators found (or all were STUBs), but weights are specified. Check MCoT pipeline implementations and dataset configurations.")
    elif not valid_train_iters:
        print("Warning: No valid data iterators to train on. Data loader train will be empty.")
        # build_mixture_dataloader might handle an empty list of iterators if weights are also empty or not provided.
    
    weights = data_config['train'].get('weights', [])
    if len(weights) != len(valid_train_iters) and valid_train_iters: # Adjust weights if iterators were skipped or if mismatch
        print(f"Warning: Number of weights ({len(weights)}) does not match number of valid data iterators ({len(valid_train_iters)}). Using equal weights for valid iterators.")
        weights = [1.0] * len(valid_train_iters)
    elif not valid_train_iters and not weights: # No iterators, no weights, this is fine.
        print("No data iterators and no weights specified. Training will proceed with no data if epoch_size is not 0.")
        weights = [] # Ensure weights is an empty list for build_mixture_dataloader
    elif not valid_train_iters and any(w > 0 for w in weights): # Should be caught by the earlier ValueError
         raise ValueError("No valid data iterators found, but weights are specified. Check MCoT pipeline stubs and configurations.")


    epoch_size = args.epoch_size
    # Ensure UnifiedMasking is used as the collate_fn in build_mixture_dataloader or that
    # individual MCoT pipelines are already structured to output what the model expects
    # directly after their own batching (if any).
    # The current MCoT pipeline stubs return [], not dataloaders, so build_mixture_dataloader will need actual iterables.

    # One crucial point: UnifiedMasking is currently a transform applied per-sample *before* batching
    # in some examples, or as a collate_fn *after* batching in others.
    # If your MCoT pipeline functions (get_mcot_*) are intended to return batch_size=None WebLoaders
    # (i.e. unbatched iterators of single processed samples), then UnifiedMasking
    # should ideally be the collate_fn for the *final* build_mixture_dataloader.
    # Let's assume for now the MCoT pipeline stubs will eventually return iterators of single samples
    # and build_mixture_dataloader will use UnifiedMasking as collate_fn.

    # Instantiate UnifiedMasking to be used as the collate function
    unified_masker_for_collate = UnifiedMasking(
        modality_info=modality_info, # Global modality_info for all possible modalities
        text_tokenizer=text_tokenizer,
        input_tokens_range=(args.min_input_tokens, args.num_input_tokens),
        target_tokens_range=(args.min_target_tokens, args.num_target_tokens),
        # sampling_weights and type_probs are not needed here as stage determination is by token
        force_end_of_generation_token_for_text_target=getattr(args, 'force_eos_for_text_target', True), # Add as arg if configurable
        force_target_mask_for_padding=getattr(args, 'force_target_mask_for_padding', False) # Add as arg if configurable
    )

    data_loader_train = build_mixture_dataloader(
        data_iters=valid_train_iters, 
        weights=weights, 
        modality_info=modality_info, 
        batch_size=args.batch_size, 
        num_workers=num_workers, 
        epoch_size=epoch_size, 
        num_gpus=args.num_tasks,
        collate_fn=unified_masker_for_collate # Pass the UnifiedMasking instance here
    )

    num_training_steps_per_epoch = epoch_size // (args.batch_size * args.num_tasks) if epoch_size and args.batch_size and args.num_tasks else 0

    # Val
    if 'val' in data_config:
        val_config = data_config['val']['datasets']

        data_loaders_val, data_loaders_fixed_eval = {}, {}
        for dataset_name, dataset_cfg in val_config.items():

            dataset_mod_info, sampling_weights = setup_sampling_mod_info(train_config[dataset_name], modality_info)

            data_loaders_val[dataset_name] = get_val_dataloader(
                dataset_config=dataset_cfg, dataset_name=dataset_name, train_configs=train_config,
                modality_info=dataset_mod_info, sampling_weights=sampling_weights, text_tokenizer=text_tokenizer,
                input_size=args.input_size, num_input_tokens=args.num_input_tokens, num_target_tokens=args.num_target_tokens,
                min_input_tokens=args.min_input_tokens, min_target_tokens=args.min_target_tokens, fixed_eval=False, 
                fixed_eval_input_tokens=args.fixed_eval_input_tokens, fixed_eval_target_tokens=args.fixed_eval_target_tokens,
                dist_eval=args.dist_eval, num_tasks=args.num_tasks, num_workers=args.num_workers,
                batch_size=int(1.5*args.batch_size), pin_mem=args.pin_mem,
            )
            if args.fixed_eval:
                data_loaders_fixed_eval[dataset_name] = get_val_dataloader(
                    dataset_config=dataset_cfg, dataset_name=dataset_name, train_configs=train_config,
                    modality_info=dataset_mod_info, sampling_weights=sampling_weights, text_tokenizer=text_tokenizer,
                    input_size=args.input_size, num_input_tokens=args.num_input_tokens, num_target_tokens=args.num_target_tokens,
                    min_input_tokens=args.min_input_tokens, min_target_tokens=args.min_target_tokens, fixed_eval=True, 
                    fixed_eval_input_tokens=args.fixed_eval_input_tokens, fixed_eval_target_tokens=args.fixed_eval_target_tokens,
                    dist_eval=args.dist_eval, num_tasks=args.num_tasks, num_workers=args.num_workers,
                    batch_size=int(1.5*args.batch_size), pin_mem=args.pin_mem,
                )
  
        data_loaders_fixed_eval = data_loaders_fixed_eval if data_loaders_fixed_eval else None

    else:
        data_loaders_val, data_loaders_fixed_eval = None, None

    return modality_info, data_loader_train, num_training_steps_per_epoch, data_loaders_val, data_loaders_fixed_eval


def get_model(args, modality_info):
    """Creates and returns model from arguments
    """
    print(f"Creating model: {args.model} for modalities {list(modality_info.keys())}")

    encoder_embeddings = {}
    for mod in args.in_domains:
        info = modality_info[mod]
        if info.get("encoder_embedding", None) is not None:
            if info["type"] == "img":
                image_size, patch_size = info.get('input_size', args.input_size), info.get('patch_size', args.patch_size)
                encoder_embeddings[mod] = info["encoder_embedding"](patch_size=patch_size, image_size=image_size)
            else:
                encoder_embeddings[mod] = info["encoder_embedding"]()

    decoder_embeddings = {}
    for mod in args.out_domains:
        info = modality_info[mod]
        if info.get("decoder_embedding", None) is not None:
            if info["type"] == "img":
                image_size, patch_size = info.get('input_size', args.input_size), info.get('patch_size', args.patch_size)
                decoder_embeddings[mod] = info["decoder_embedding"](patch_size=patch_size, image_size=image_size)
            else:
                decoder_embeddings[mod] = info["decoder_embedding"]()

    model = create_model(
        args.model,
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings=decoder_embeddings,
        modality_info=modality_info,
        num_register_tokens=args.num_register_tokens,
    )

    # Resize token embeddings if MCoT tokens were added
    if hasattr(args, 'tokenizer_object_for_resize') and args.tokenizer_object_for_resize is not None:
        print(f"Resizing token embeddings for model. Old vocab size: {model.vocab_size if hasattr(model, 'vocab_size') else 'N/A'}")
        # Ensure model has vocab_size attribute or similar before attempting to access it.
        # This check might need to be more specific depending on the model architecture.
        # For now, we assume resize_token_embeddings handles cases where vocab_size isn't directly on model.
        model.resize_token_embeddings(len(args.tokenizer_object_for_resize))
        print(f"New vocab size: {model.vocab_size if hasattr(model, 'vocab_size') else 'N/A'}. New tokenizer length: {len(args.tokenizer_object_for_resize)}")
        # Clean up to prevent it from being passed around unintentionally
        delattr(args, 'tokenizer_object_for_resize')


    return model


def main(args):
    ## Distributed init / Local debug mode setup
    if args.local_debug:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.num_tasks = 1 # Crucial for data loaders and other logic expecting world_size
        
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("Local Debug: CUDA not available, switching to CPU.")
            args.device = 'cpu'
        # No DDP-specific args.gpu needed for local mode if DDP is disabled
        
        print(f"Running in LOCAL DEBUG mode on device: {args.device}. Distributed training disabled.")
        
        # Override critical args for a quick local test
        args.epochs = 1
        args.num_workers = 0  # Often better for debugging
        # Use the dedicated local_batch_size, falling back to the general batch_size if not specified by user in local_debug
        args.batch_size = args.local_batch_size if hasattr(args, 'local_batch_size') else args.batch_size 
        args.output_dir = args.local_output_dir if hasattr(args, 'local_output_dir') else "tmp_local_output"
        args.log_wandb = False

        # Adjust epoch_size for local run. If epoch_size is in config, scale it.
        # Otherwise, set a small default for a few steps.
        if hasattr(args, 'epoch_size') and args.epoch_size is not None:
            original_epoch_size_from_config = args.epoch_size
            # For local debug, we assume epoch_size from config is total samples.
            # No need to divide by num_tasks here as it's already a single task.
            # We might want to make it smaller just for testing, e.g. ensure it's at least a few batches.
            args.epoch_size = max(args.batch_size * 2, original_epoch_size_from_config // 10 if original_epoch_size_from_config > args.batch_size * 20 else original_epoch_size_from_config) 
            args.epoch_size = min(args.epoch_size, original_epoch_size_from_config) # Don't exceed original
            print(f"Local Debug: Original epoch_size from config={original_epoch_size_from_config}, using adjusted epoch_size={args.epoch_size} for local run.")
        else:
            args.epoch_size = args.batch_size * 5 # Default to 5 steps if not set in config
            print(f"Local Debug: epoch_size not in config, setting to {args.epoch_size} for a few steps.")


        print(f"Local Debug Overrides: epochs={args.epochs}, num_workers={args.num_workers}, batch_size={args.batch_size}, output_dir='{args.output_dir}', log_wandb={args.log_wandb}, effective_epoch_size={args.epoch_size}")
        # Ensure output directory exists for local run
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            
    else:
        utils.init_distributed_mode(args)
        # args.num_tasks is set to world_size by init_distributed_mode or should be.
        # If not, ensure it here:
        if hasattr(utils, 'get_world_size'):
             args.num_tasks = utils.get_world_size()
        else: # Fallback if the util is not as expected
             args.num_tasks = 1 
             if torch.distributed.is_initialized():
                 args.num_tasks = torch.distributed.get_world_size()
    
    device = torch.device(args.device)
    
    # utils.get_rank() will be 0 if not DDP initialized via utils.init_distributed_mode
    seed = args.seed + utils.get_rank() 
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if not args.show_user_warnings:
        warnings.filterwarnings("ignore", category=UserWarning)

    if args.dtype in ['float16', 'fp16']:
        dtype = torch.float16
    elif args.dtype in ['bfloat16', 'bf16']:
        dtype = torch.bfloat16
    elif args.dtype in ['float32', 'fp32']:
        dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {args.dtype}")
    
    # global_rank is used for logging, ensure it's 0 for local debug
    global_rank = args.rank if hasattr(args, 'rank') else 0 # Should be set by now either way
    
    ## Data
    modality_info, data_loader_train, num_training_steps_per_epoch, data_loaders_val, data_loaders_fixed_eval = setup_data(args)

    ## Model
    model = get_model(args, modality_info)

    # Logger
    # Use global_rank for conditional logging
    if global_rank == 0 and args.log_wandb: 
        log_writer = utils.WandbLogger(args)
    else:
        log_writer = None

    ## Training phases / epochs
    if not args.local_debug: # Only do these complex calculations if not in local_debug
        if args.epochs < 0:
            if args.total_tokens < 0:
                print("Epochs and total tokens are both set to negative values, stopping training.")
                sys.exit(1) # Changed from exit(1) to sys.exit(1) for clarity
            else:
                train_dataset_size = args.epoch_size # or len(dataset_train)
                args.epochs = math.ceil(args.total_tokens * 1e9 / ((args.num_input_tokens + args.num_target_tokens) * train_dataset_size))
                print(f"Total tokens: {args.total_tokens}B")
                print(f"Setting the number of epochs accordingly to {args.epochs}")
        elif args.total_tokens > 0:
            print("Epochs and total tokens are both non-negative, stopping training.")
            sys.exit(1) # Changed

        # Warmup
        if args.warmup_epochs < 0 and args.warmup_steps < 0:
            if args.warmup_tokens < 0:
                print("Warmup epochs, steps and total tokens all set to negative values, stopping training.")
                sys.exit(1) # Changed
            else:
                args.warmup_steps = math.ceil(args.warmup_tokens * 1e9 / ((args.num_input_tokens + args.num_target_tokens) * args.batch_size * args.num_tasks)) # utils.get_world_size() became args.num_tasks

        # Cooldown
        if args.cooldown_epochs < 0 and args.cooldown_steps < 0:
            if args.cooldown_tokens < 0 and args.scheduler in ['inverse_sqrt']: # Fixed variable name lr_schedule to scheduler
                print("Cooldown epochs, steps and total tokens all set to negative values, stopping training.")
                sys.exit(1) # Changed
            else:
                args.cooldown_steps = math.ceil(args.cooldown_tokens * 1e9 / ((args.num_input_tokens + args.num_target_tokens) * args.batch_size * args.num_tasks)) # utils.get_world_size() became args.num_tasks
        
        # Frozen
        if args.frozen_model_epochs <= 0:
            if args.frozen_model_tokens > 0:
                train_dataset_size = args.epoch_size # or len(dataset_train)
                args.frozen_model_epochs = math.ceil(args.frozen_model_tokens * 1e9 / ((args.num_input_tokens + args.num_target_tokens) * train_dataset_size))
            # else: # Removed print("No frozen models during training.") to match original structure if epochs is 0
        else:
            if args.frozen_model_tokens > 0:
                print("Frozen_model_epochs and frozen_model_tokens are both non-negative, stopping training.")
                sys.exit(1) # Changed
    # else: for local_debug, epochs is already 1, skip these complex calculations

    print(args)

    ## Starting from pre-trained model
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu')
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint.get('model_ema', checkpoint.get('model', checkpoint)) # Handle EMA models
        if 'model' not in checkpoint and 'model_ema' not in checkpoint and 'state_dict' in checkpoint: # More robust checkpoint key search
            checkpoint_model = checkpoint['state_dict']
        
        # Remove pos_emb if they cause size mismatches
        # It's better to handle this inside model.load_state_dict if possible, or make it configurable
        # For now, retain original logic but be aware of its potential issues.
        # checkpoint_model = {k: v for k, v in checkpoint_model.items() if ".pos_emb" not in k}


        # Create a new state_dict that only contains keys present in the current model
        model_state_dict = model.state_dict()
        filtered_checkpoint_model = {}
        for k, v in checkpoint_model.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                filtered_checkpoint_model[k] = v
            elif k.replace("module.", "") in model_state_dict and model_state_dict[k.replace("module.","")].shape == v.shape: # Handle DDP prefix
                filtered_checkpoint_model[k.replace("module.","")] = v
            else:
                print(f"Skipping layer {k} from checkpoint due to mismatch or absence in current model.")


        msg = model.load_state_dict(filtered_checkpoint_model, strict=False)
        print(f"Loaded finetuning checkpoint from {args.finetune}. Load message: {msg}")


    model.to(device)
    model_without_ddp = model # Start with model_without_ddp as the base model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model = %s" % str(model_without_ddp)) # Log the model structure
    print(f"Number of params: {n_parameters / 1e6} M")

    # batch_size_no_accum is batch_size per GPU * num_gpus
    # For local_debug, num_tasks (world_size) is 1.
    batch_size_no_accum = args.batch_size * args.num_tasks 
    total_batch_size = args.batch_size * args.accum_iter * args.num_tasks
    
    # LR scaling should be based on the effective total batch size relative to a base (e.g., 256)
    # args.lr is calculated based on args.blr and this total_batch_size
    if not args.local_debug: # Keep original LR calculation for DDP
        args.lr = args.blr * total_batch_size / 256
        args.min_lr = args.min_blr * total_batch_size / 256
        if args.frozen_model_blr > 0:
            args.frozen_model_lr = args.frozen_model_blr * total_batch_size / 256
        else:
            args.frozen_model_lr = args.blr * total_batch_size / 256 # Default to main blr for frozen part
    else: # For local debug, batch_size is small, avoid overly tiny LR. Use blr directly or a small scale.
        args.lr = args.blr 
        args.min_lr = args.min_blr
        args.frozen_model_lr = args.frozen_model_blr if args.frozen_model_blr > 0 else args.blr


    print("LR = %.8f" % args.lr)
    print("Min LR = %.8f" % args.min_lr)
    print("Total (effective) batch size = %d" % total_batch_size)
    print("Accumulate grad iterations = %d" % args.accum_iter)
    # num_training_steps_per_epoch might be 0 if epoch_size is small, handle this
    if num_training_steps_per_epoch == 0 and args.epoch_size > 0 and total_batch_size > 0 :
        num_training_steps_per_epoch = math.ceil(args.epoch_size / total_batch_size) * args.accum_iter
        print(f"Recalculated num_training_steps_per_epoch for local debug: {num_training_steps_per_epoch} (epoch_size {args.epoch_size}, total_batch_size {total_batch_size})")
    elif num_training_steps_per_epoch == 0 :
        num_training_steps_per_epoch = 1 # Ensure at least one step for very small local tests
        print(f"Warning: num_training_steps_per_epoch is 0. Setting to 1 for local debug.")


    print("Number of training steps = %d" % num_training_steps_per_epoch)
    # Calculate examples per epoch based on effective items processed by data_loader_train
    # For iterable datasets, len(data_loader_train) is not directly available.
    # num_training_steps_per_epoch * batch_size_no_accum / accum_iter
    # The number of examples is epoch_size for WebDatasets.
    # For local debug, epoch_size is already adjusted.
    print("Number of training examples per epoch (approx) = %d" % (args.epoch_size if args.epoch_size is not None else "Unknown (IterableDataset)"))


    # Conditional DDP wrapping
    if args.distributed: # This will be False if args.local_debug is True
        # device_ids should be [args.gpu] as set by utils.init_distributed_mode
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    # else: model is already model_without_ddp and on the correct device (set by model.to(device))

    optimizer = create_optimizer(args, model_without_ddp) # Always pass model_without_ddp
    loss_scaler = NativeScaler(enabled=(dtype == torch.float16 and not args.local_debug)) # Disable scaler for local CPU debug if using fp16/bf16 conceptually

    ## LR and WD schedules
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay

    # Schedule calculations should adapt to args.epochs which is 1 for local_debug
    if args.frozen_model_epochs > 0 and not args.local_debug : # No frozen epochs in local_debug override
        frozen_lr_schedule_values = utils.constant_scheduler(args.frozen_model_lr, args.frozen_model_epochs, num_training_steps_per_epoch)
        frozen_wd_schedule_values = utils.constant_scheduler(args.weight_decay, args.frozen_model_epochs, num_training_steps_per_epoch)
        main_schedule_epochs = args.epochs - args.frozen_model_epochs
    else:
        frozen_lr_schedule_values = np.array([]) 
        frozen_wd_schedule_values = np.array([])
        main_schedule_epochs = args.epochs # Will be 1 for local_debug

    # Ensure main_schedule_epochs is at least 1 for scheduler calculations
    main_schedule_epochs = max(1, main_schedule_epochs)

    if args.scheduler == 'cosine':
        main_lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, main_schedule_epochs, num_training_steps_per_epoch, 
            warmup_epochs=args.warmup_epochs if not args.local_debug else 0, # No warmup for 1 epoch local
            warmup_steps=args.warmup_steps if not args.local_debug else 0   # No warmup for 1 epoch local
        )
        wd_schedule_values = utils.cosine_scheduler( # WD schedule for the main part
            args.weight_decay, args.weight_decay_end, main_schedule_epochs, num_training_steps_per_epoch
        )
    elif 'inverse_sqrt' in args.scheduler:
        try:
            timescale = int(args.scheduler.split('-')[-1])
        except:
            timescale = 10_000
        main_lr_schedule_values = utils.inverse_sqrt_scheduler(
            args.lr, args.min_lr, main_schedule_epochs, num_training_steps_per_epoch, 
            warmup_epochs=args.warmup_epochs if not args.local_debug else 0, 
            warmup_steps=args.warmup_steps if not args.local_debug else 0,
            cooldown_epochs=args.cooldown_epochs if not args.local_debug else 0, # No cooldown for 1 epoch local
            cooldown_steps=args.cooldown_steps if not args.local_debug else 0,
            timescale=timescale
        )
        wd_schedule_values = utils.inverse_sqrt_scheduler( # WD schedule for the main part
            args.weight_decay, args.weight_decay_end, main_schedule_epochs, num_training_steps_per_epoch,
            cooldown_epochs=args.cooldown_epochs if not args.local_debug else 0,
            cooldown_steps=args.cooldown_steps if not args.local_debug else 0,
            timescale=timescale
        )
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler} not implemented.")
    
    lr_schedule_values = np.concatenate((frozen_lr_schedule_values, main_lr_schedule_values))
    wd_schedule_values = np.concatenate((frozen_wd_schedule_values, wd_schedule_values))
    
    # Ensure schedules are not empty if num_training_steps_per_epoch is very small
    if len(lr_schedule_values) == 0 and num_training_steps_per_epoch > 0:
        lr_schedule_values = np.full(num_training_steps_per_epoch * main_schedule_epochs, args.lr)
        print("Warning: LR schedule was empty, filled with constant LR.")
    if len(wd_schedule_values) == 0 and num_training_steps_per_epoch > 0:
        wd_schedule_values = np.full(num_training_steps_per_epoch * main_schedule_epochs, args.weight_decay)
        print("Warning: WD schedule was empty, filled with constant WD.")

    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values, default=args.weight_decay), min(wd_schedule_values,default=args.weight_decay_end if args.weight_decay_end is not None else args.weight_decay)))


    # Auto-load from checkpoint
    # Skip auto-resume for local debug to ensure a fresh quick test, unless specifically requested
    if not args.local_debug or (args.local_debug and args.resume): 
        utils.auto_load_model(
            args=args, model=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler) # Pass model_without_ddp

    ## Eval (on trained model) - Skip for local_debug by default unless --eval is also passed
    if args.eval and not args.local_debug: # Or add a specific flag like --local_eval
        if data_loaders_val is not None:
            for dataset_name, data_loader_val in data_loaders_val.items():
                prefix = '[Eval] ' if not dataset_name else f'[Eval ({dataset_name})] '
                eval_stats = evaluate(model_without_ddp, data_loader_val, device, # Pass model_without_ddp
                                    num_input_tokens=args.num_input_tokens,
                                    num_target_tokens=args.num_target_tokens,
                                    all_domains=args.all_domains, dtype=dtype,
                                    prefix=prefix, loss_type=args.loss_type)

                print("Eval Stats:" if not dataset_name else f"Eval Stats ({dataset_name}):")
                print(eval_stats)
                print()

        if data_loaders_fixed_eval is not None:
            for dataset_name, data_loader_fixed_eval in data_loaders_fixed_eval.items():
                prefix = '[Fixed Eval] ' if not dataset_name else f'[Fixed Eval ({dataset_name})] '
                fixed_eval_stats = evaluate(model_without_ddp, data_loader_fixed_eval, device, # Pass model_without_ddp
                                            num_input_tokens=args.fixed_eval_input_tokens,
                                            num_target_tokens=args.fixed_eval_target_tokens,
                                            all_domains=args.all_domains, dtype=dtype,
                                            prefix=prefix, loss_type=args.loss_type)
                print("Fixed Eval Stats:" if not dataset_name else f"Fixed Eval Stats ({dataset_name}):")
                print(fixed_eval_stats)
                print()
        sys.exit(0) # Exit after eval if --eval is true


    ## Training
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs): # args.epochs will be 1 for local_debug
        if log_writer is not None and global_rank == 0: # Check global_rank for logging
            # For local debug, step might be small, ensure it increments if num_training_steps_per_epoch is small
            log_writer.set_step(epoch * num_training_steps_per_epoch if num_training_steps_per_epoch > 0 else epoch)
        
        # For DDP, set epoch for sampler if data_loader_train has a sampler
        if hasattr(data_loader_train, 'sampler') and hasattr(data_loader_train.sampler, 'set_epoch') and args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, # Pass DDP model if distributed, else base model
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            frozen_model_epochs=args.frozen_model_epochs if not args.local_debug else 0, # No frozen for local debug
            loss_scaler=loss_scaler,
            accum_iter=args.accum_iter,
            max_norm=args.clip_grad,
            max_skip_norm=args.skip_grad,
            log_writer=log_writer if global_rank == 0 else None, # Only log for rank 0
            start_steps=epoch * num_training_steps_per_epoch if num_training_steps_per_epoch > 0 else epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_input_tokens=args.num_input_tokens,
            num_target_tokens=args.num_target_tokens,
            all_domains=args.all_domains,
            dtype=dtype,
            loader_len=num_training_steps_per_epoch, # Pass adjusted loader_len
            output_dir=args.output_dir,
            compute_grad_norm=args.compute_grad_norm,
            loss_type=args.loss_type,
            total_batch_size=total_batch_size,
            is_local_debug=args.local_debug # Pass local_debug flag
        )
        if args.output_dir and not args.local_debug : # Only save checkpoints if not local_debug or if explicitly enabled
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                if epoch + 1 == args.epochs: # Final checkpoint
                    use_s3 = len(args.s3_save_dir) > 0 and not args.local_debug
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, ckpt_name='final', use_s3=use_s3)
        elif args.output_dir and args.local_debug:
             print(f"Local Debug: Skipping checkpoint saving for epoch {epoch}.")
                    

        log_stats = {**{k: v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if not args.local_debug : # Only log token counts if not local_debug to avoid tiny numbers
            log_stats.update({
                     'input_tokens_seen_b': (epoch + 1) * num_training_steps_per_epoch * (total_batch_size / args.accum_iter) * args.num_input_tokens / 1e9,
                     'target_tokens_seen_b': (epoch + 1) * num_training_steps_per_epoch * (total_batch_size / args.accum_iter) * args.num_target_tokens / 1e9,
                     'total_tokens_seen_b': (epoch + 1) * num_training_steps_per_epoch * (total_batch_size / args.accum_iter) * (args.num_input_tokens + args.num_target_tokens) / 1e9,
            })

        # Evaluation during training - skip for local_debug unless explicitly requested
        if not args.local_debug and data_loaders_val is not None and ((epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs):
            for dataset_name, data_loader_val in data_loaders_val.items():
                prefix = '[Eval] ' if not dataset_name else f'[Eval ({dataset_name})] '
                eval_stats = evaluate(model_without_ddp, data_loader_val, device, num_input_tokens=args.num_input_tokens, num_target_tokens=args.num_target_tokens, # Pass model_without_ddp
                                    all_domains=args.all_domains, dtype=dtype, prefix=prefix, loss_type=args.loss_type)
                extra_stats = {**{k: v for k, v in eval_stats.items()}}
                log_stats.update(extra_stats)

        if not args.local_debug and data_loaders_fixed_eval is not None and ((epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs):
            for dataset_name, data_loader_fixed_eval in data_loaders_fixed_eval.items():
                prefix = '[Fixed Eval] ' if not dataset_name else f'[Fixed Eval ({dataset_name})] '
                fixed_eval_stats = evaluate(model_without_ddp, data_loader_fixed_eval, device, num_input_tokens=args.fixed_eval_input_tokens, num_target_tokens=args.fixed_eval_target_tokens, # Pass model_without_ddp
                                            all_domains=args.all_domains, dtype=dtype, prefix=prefix, loss_type=args.loss_type)
                extra_stats = {**{k: v for k, v in fixed_eval_stats.items()}}
                log_stats.update(extra_stats)

        if log_writer is not None and global_rank == 0: # Check global_rank
            log_writer.update(log_stats)

        if args.output_dir and utils.is_main_process(): # utils.is_main_process() is DDP-aware
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    num_input_tokens: int, num_target_tokens: int, loss_type: str, device: torch.device, epoch: int, frozen_model_epochs: int, 
                    loss_scaler, accum_iter, max_norm: float = None, max_skip_norm: float = None, log_writer=None,
                    lr_scheduler=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    all_domains: List[str] = [], dtype: torch.dtype = torch.float16, loader_len: Optional[int] = None,
                    output_dir=None, compute_grad_norm=True, total_batch_size=None, is_local_debug=False):
    
    model.train()
    if frozen_model_epochs > 0 and epoch < frozen_model_epochs:
        if args.frozen_embedding_domain is None:
            model.module.freeze_shared_params()
        
        else:
            model.module.freeze_params_except_specific_embeddings(args.frozen_embedding_domain)
    else:
        model.module.unfreeze_all()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for step, x in enumerate(metric_logger.log_every(data_loader, print_freq, iter_len=loader_len, header=header)):
        # Assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        update_grad = (step + 1) % accum_iter == 0

        if step % accum_iter == 0:
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

        mod_dict = {
            modality: {k: v.to(device, non_blocking=True) for k, v in d.items()}
            for modality, d in x.items()
            if modality in all_domains
        }

        # Only sync if we update grad (for accum_iter)
        # See https://muellerzr.github.io/blog/gradient_accumulation.html
        with nullcontext() if update_grad else model.no_sync():

            with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
                loss, mod_loss = model(mod_dict, num_encoder_tokens=num_input_tokens, num_decoder_tokens=num_target_tokens, loss_type=loss_type)

                loss_value = loss.item()
                mod_loss_values = {f'{mod}_loss': l.item() for mod, l in mod_loss.items()}

            if not math.isfinite(loss_value):
                torch.save(mod_dict, os.path.join(output_dir, "debug_mod_dict.pt"))
                print(f"Loss is {loss_value}, stopping training", file=sys.stderr)
                sys.exit(1)

            loss = loss / accum_iter
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, skip_grad=max_skip_norm,
                                    parameters=model.parameters(), compute_grad_norm=compute_grad_norm, 
                                    update_grad=update_grad)
            if update_grad:
                optimizer.zero_grad()

            if dtype == torch.float16:
                loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(**mod_loss_values)
        if dtype == torch.float16:
            metric_logger.update(loss_scale=loss_scale_value) 
        min_lr = 1.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                    'grad_norm': grad_norm,
                }
            )
            log_writer.update(mod_loss_values)

            if total_batch_size is not None:
                log_writer.update(
                    {
                        'input_tokens_seen_b': it * (total_batch_size / accum_iter) * num_input_tokens / 1e9,
                        'target_tokens_seen_b': it * (total_batch_size / accum_iter) * num_target_tokens / 1e9,
                        'total_tokens_seen_b': it * (total_batch_size / accum_iter) * (num_input_tokens + num_target_tokens) / 1e9,
                    }
                )

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    torch.cuda.empty_cache()

    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, num_input_tokens, num_target_tokens, loss_type,
             all_domains: List[str], dtype: torch.dtype = torch.float16, prefix="[Eval] "):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = prefix

    # switch to evaluation mode
    model.eval()

    print_freq = 10
    iter_len = len(data_loader) if hasattr(data_loader, '__len__') else -1 # Dealing with iterable datasets

    for x in metric_logger.log_every(data_loader, print_freq, iter_len=iter_len, header=header):

        mod_dict = {
            modality: {k: v.to(device, non_blocking=True) for k, v in d.items()}
            for modality, d in x.items()
            if modality in all_domains
        }

        with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
            loss, mod_loss = model(mod_dict, num_encoder_tokens=num_input_tokens, num_decoder_tokens=num_target_tokens, loss_type=loss_type)

            loss_value = loss.item()
            mod_loss_values = {f'{mod}_loss': l.item() for mod, l in mod_loss.items()}

        metric_logger.update(loss=loss_value)
        metric_logger.update(**mod_loss_values)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Eval averaged stats:", metric_logger)
    torch.cuda.empty_cache()
    
    return {prefix + k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args()

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (args.rlimit, rlimit[1]))
    
    utils.setup_run_name(args)
    utils.setup_s3_args(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)

