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
4M MCoT Training Script with MINT Paper Features
Based on the proven run_training_4m_fsdp.py but adapted for MCoT training.

This script adds MCoT capabilities to 4M training, incorporating:
- Step-specific loss computation for Planning, Acting, Reflection, Correction
- Artifact heatmap generation during reflection
- Reflection-guided mask generation for correction
- MINT paper methodology integration
- SeeTRUE-Feedback dataset integration for enhanced reflection training
"""

import argparse
import datetime
import functools
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
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import fourm.utils as utils
import fourm.utils.fsdp_utils as fsdp_utils
from fourm.data import (build_mixture_dataloader, get_train_dataloader, get_val_dataloader, setup_sampling_mod_info)
from fourm.models import fm
from fourm.models.fm_utils import Block, DecoderBlock
from fourm.models.mcot_fixed import MCoTStepProcessor, add_mcot_to_model
from fourm.data.modality_info import MODALITY_INFO
from fourm.utils import create_model, load_safetensors
from fourm.utils.optim_factory import create_optimizer

# Optional Wandb logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    print("Wandb not available - logging will be limited")
    HAS_WANDB = False


def get_args():
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('4M MCoT training script (using FSDP)', add_help=False)
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
    parser.add_argument('--use_act_checkpoint', action='store_true')
    parser.add_argument('--no_use_act_checkpoint', action='store_false', dest='use_act_checkpoint')
    parser.set_defaults(use_act_checkpoint=False)

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

    parser.add_argument('--num_input_tokens', type=int, default=128, help="Token budget for the input")
    parser.add_argument('--num_target_tokens', type=int, default=128, help="Token budget for the target")
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
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Clip gradient norm (default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: %(default)s)')
    parser.add_argument('--weight_decay_end', type=float, default=None, 
                        help="Final value of the weight decay. (Set the same value as args.weight_decay to keep weight decay value constant)")
    parser.add_argument('--skip_nan_grad', action='store_true', 
                        help="Skips the batch if the grad norm is NaN, requires having grad clipping activated")

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
    parser.add_argument('--cooldown_epochs', type=int, default=10,
                        help='Epochs to cool down LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--cooldown_steps', type=int, default=-1, 
                        help='Steps to cool down LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--cooldown_tokens', type=int, default=-1, 
                        help='Total tokens to cool down LR, if scheduler supports (default: %(default)s)')
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

    parser.add_argument('--rlimit', default=4096, type=int, 
                        help='Increase rlimit to avoid "RuntimeError: received 0 items of ancdata".')
    parser.add_argument('--print_all', action='store_true', default=False)
    parser.add_argument('--s3_save_dir', type=str, default="")
    parser.add_argument('--show_user_warnings', default=False, action='store_true')

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

    # MCoT-specific parameters
    parser.add_argument('--mcot_steps', type=str, default='planning,acting,reflection,correction',
                        help='Comma-separated list of MCoT steps to train')
    parser.add_argument('--mcot_planning_weight', type=float, default=1.0, 
                        help="Weight for planning step loss")
    parser.add_argument('--mcot_acting_weight', type=float, default=1.0, 
                        help="Weight for acting step loss")
    parser.add_argument('--mcot_reflection_weight', type=float, default=1.0, 
                        help="Weight for reflection step loss")
    parser.add_argument('--mcot_correction_weight', type=float, default=1.0, 
                        help="Weight for correction step loss")
    parser.add_argument('--enable_mint_features', action='store_true',
                        help="Enable MINT paper features (artifact heatmaps, reflection-guided masks)")

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
    """Setup modality information based on args - matches base script structure."""
    # Global modality info
    modality_info = {mod: MODALITY_INFO[mod] for mod in MODALITY_INFO.keys()}
    
    # Max tokens
    for mod in modality_info:
        image_size, patch_size = modality_info[mod].get('input_size', args.input_size), modality_info[mod].get('patch_size', args.patch_size)
        num_patches = (image_size // patch_size) ** 2
        if modality_info[mod]['type'] == 'img':
            modality_info[mod]['max_tokens'] = num_patches

    return modality_info


def setup_data(args):
    """Setup data loaders following base 4M script structure with MCoT data integration."""
    text_tokenizer = Tokenizer.from_file(args.text_tokenizer_path)
    
    # Load data config
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Parse config and set domains
    train_config = data_config['train']['datasets']
    args.in_domains = sorted(set.union(*[set(cfg['in_domains'].split('-')) for cfg in train_config.values()]))
    args.out_domains = sorted(set.union(*[set(cfg['out_domains'].split('-')) for cfg in train_config.values()]))
    args.all_domains = sorted(list(set(args.in_domains) | set(args.out_domains)))
    
    # Set up shared modality info
    modality_info = setup_modality_info(args)
    
    # Initialize train loaders
    if any([cfg['data_path'].startswith('s3') for cfg in train_config.values()]):
        utils.s3_utils.override_wds_s3_tar_loading(args.s3_data_endpoint, args.s3_multipart_threshold_mb, args.s3_multipart_chunksize_mb, args.s3_max_io_queue)
    
    num_trainsets = len(train_config)
    num_workers = args.num_workers
    
    if num_trainsets == 1:
        # Single dataset - directly configure the loader
        dataset_name, dataset_cfg = list(train_config.items())[0]
        print(f'Setting up single dataset {dataset_name} / train')
        dataset_mod_info, sampling_weights = setup_sampling_mod_info(dataset_cfg, modality_info)
        
        # Check for MCoT data path override
        if hasattr(args, 'mcot_data_path') and args.mcot_data_path:
            # Override data path for MCoT training
            dataset_cfg = dict(dataset_cfg)
            dataset_cfg['data_path'] = args.mcot_data_path
            print(f"Using MCoT data path: {args.mcot_data_path}")
        
        data_loader_train = get_train_dataloader(
            dataset_config=dataset_cfg, modality_info=dataset_mod_info,
            sampling_weights=sampling_weights, text_tokenizer=text_tokenizer, input_size=args.input_size,
            num_input_tokens=args.num_input_tokens, num_target_tokens=args.num_target_tokens,
            min_input_tokens=args.min_input_tokens, min_target_tokens=args.min_target_tokens,
            num_tasks=args.num_tasks, num_workers=num_workers,
            dataset_batch_size=args.batch_size,
            epoch_size=args.epoch_size,
            use_text_to_caption_transform=True
        )
        
        if hasattr(data_loader_train, 'n_shards') and data_loader_train.n_shards > 0:
            num_workers = min(data_loader_train.n_shards, args.num_workers)
    else:
        # Multiple datasets - use mixture approach
        train_iters = []
        shards_per_dataset = []
        for dataset_name, dataset_cfg in train_config.items():
            print(f'Setting up dataset {dataset_name} / train for mixture')
            dataset_mod_info, sampling_weights = setup_sampling_mod_info(dataset_cfg, modality_info)
            
            # Check for MCoT data path override
            if hasattr(args, 'mcot_data_path') and args.mcot_data_path:
                dataset_cfg = dict(dataset_cfg)
                dataset_cfg['data_path'] = args.mcot_data_path
                print(f"Using MCoT data path for {dataset_name}: {args.mcot_data_path}")
            
            dataiter = get_train_dataloader(
                dataset_config=dataset_cfg, modality_info=dataset_mod_info,
                sampling_weights=sampling_weights, text_tokenizer=text_tokenizer, input_size=args.input_size,
                num_input_tokens=args.num_input_tokens, num_target_tokens=args.num_target_tokens,
                min_input_tokens=args.min_input_tokens, min_target_tokens=args.min_target_tokens,
                num_tasks=args.num_tasks, num_workers=num_workers,
                dataset_batch_size=None,  # For mixture, individual loaders yield single samples
                epoch_size=None,  # MixtureDataset handles epoch_size
                use_text_to_caption_transform=True
            )
            train_iters.append(dataiter)
            if hasattr(dataiter, 'n_shards') and dataiter.n_shards > 0:
                shards_per_dataset.append(dataiter.n_shards)
        
        # Adjust num_workers for the MixtureLoader
        if shards_per_dataset:
            num_workers = min(min(shards_per_dataset), args.num_workers)
        
        weights = data_config['train'].get('weights', [1.0] * num_trainsets)
        data_loader_train = build_mixture_dataloader(
            data_iters=train_iters, weights=weights, modality_info=modality_info,
            batch_size=args.batch_size, num_workers=num_workers,
            epoch_size=args.epoch_size, num_gpus=args.num_tasks
        )
    
    # Calculate training steps
    num_training_steps_per_epoch = args.epoch_size // (args.batch_size * args.num_tasks)
    
    # Setup validation loaders
    data_loaders_val, data_loaders_fixed_eval = None, None
    if 'val' in data_config:
        val_config = data_config['val']['datasets']
        data_loaders_val, data_loaders_fixed_eval = {}, {}
        
        for dataset_name, dataset_cfg in val_config.items():
            dataset_mod_info, sampling_weights = setup_sampling_mod_info(train_config[dataset_name], modality_info)
            
            # MCoT validation can use same data path override
            if hasattr(args, 'mcot_data_path') and args.mcot_data_path:
                dataset_cfg = dict(dataset_cfg)
                # For validation, might want to use a different split or path
                val_path = args.mcot_data_path.replace('train', 'val') if 'train' in args.mcot_data_path else args.mcot_data_path
                dataset_cfg['data_path'] = val_path
            
            data_loaders_val[dataset_name] = get_val_dataloader(
                dataset_config=dataset_cfg, dataset_name=dataset_name, train_configs=train_config,
                modality_info=dataset_mod_info, sampling_weights=sampling_weights, text_tokenizer=text_tokenizer,
                input_size=args.input_size, num_input_tokens=args.num_input_tokens, num_target_tokens=args.num_target_tokens,
                min_input_tokens=args.min_input_tokens, min_target_tokens=args.min_target_tokens, fixed_eval=False,
                fixed_eval_input_tokens=args.fixed_eval_input_tokens, fixed_eval_target_tokens=args.fixed_eval_target_tokens,
                dist_eval=args.dist_eval, num_tasks=args.num_tasks, num_workers=args.num_workers,
                batch_size=int(1.5*args.batch_size), pin_mem=args.pin_mem,
                use_text_to_caption_transform=True
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
                    use_text_to_caption_transform=True
                )
        
        data_loaders_val = data_loaders_val if data_loaders_val else None
        data_loaders_fixed_eval = data_loaders_fixed_eval if data_loaders_fixed_eval else None
    
    return modality_info, data_loader_train, num_training_steps_per_epoch, data_loaders_val, data_loaders_fixed_eval


def get_model(args, modality_info):
    """Create and enhance 4M model with MCoT capabilities - simplified version."""
    print(f"Creating model: {args.model}")
    
    # Create base 4M model using standard factory
    model = create_model(
        args.model,
        input_size=args.input_size,
        patch_size=args.patch_size,
        drop_path_rate=getattr(args, 'drop_path', 0.0),
        drop_rate=getattr(args, 'drop', 0.0),
        attn_drop_rate=getattr(args, 'attn_drop_rate', 0.0),
        head_drop_rate=getattr(args, 'head_drop_rate', 0.0),
        modality_info=modality_info,
        num_input_tokens=args.num_input_tokens,
        num_target_tokens=args.num_target_tokens,
        num_register_tokens=args.num_register_tokens,
    )
    
    # Enhance with MCoT capabilities if specified
    if hasattr(args, 'mcot_steps') and args.mcot_steps:
        print("Adding MCoT capabilities to base model...")
        mcot_processor = MCoTStepProcessor(
            dim=getattr(model, 'dim', 768),
            device=args.device,
            enable_mint=getattr(args, 'enable_mint_features', False),
            mcot_steps=args.mcot_steps.split(','),
            step_weights={
                'planning': args.mcot_planning_weight,
                'acting': args.mcot_acting_weight,
                'reflection': args.mcot_reflection_weight,
                'correction': args.mcot_correction_weight
            }
        )
        
        # Wrap model with MCoT capabilities
        model = add_mcot_to_model(model, mcot_processor)
    
    # Load pretrained weights if specified
    if args.finetune:
        print(f"Loading pretrained model from: {args.finetune}")
        if args.finetune.endswith('.safetensors'):
            state_dict = load_safetensors(args.finetune)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        
        # Load with strict=False to allow for new MCoT parameters
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint with message: {msg}")
    
    return model


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = None,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, num_training_steps_per_epoch=None, update_freq=None,
                    use_amp=False, args=None):
    """Train one epoch following base script structure with MCoT enhancements."""
    
    model.train(True)
    
    # Handle frozen model epochs if specified
    frozen_model_epochs = getattr(args, 'frozen_model_epochs', 0)
    if frozen_model_epochs > 0 and epoch < frozen_model_epochs:
        frozen_embedding_domain = getattr(args, 'frozen_embedding_domain', None)
        if frozen_embedding_domain is None:
            model.freeze_shared_params()
        else:
            model.freeze_params_except_specific_embeddings(frozen_embedding_domain)
    else:
        model.unfreeze_all()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # Add MCoT-specific metrics if applicable
    if hasattr(args, 'mcot_steps') and args.mcot_steps:
        for step in args.mcot_steps.split(','):
            step = step.strip()
            metric_logger.add_meter(f'{step}_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = f'Epoch: [{epoch}]'
    print_freq = 10
    
    optimizer.zero_grad()
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        
        it = start_steps + step  # global training iteration
        
        # Update learning rate and weight decay
        if data_iter_step % update_freq == 0:
            if lr_schedule_values is not None:
                for param_group in optimizer.param_groups:
                    if lr_schedule_values[it] is not None:
                        param_group['lr'] = lr_schedule_values[it] * param_group.get('lr_scale', 1.0)
            if wd_schedule_values is not None and len(wd_schedule_values) > it:
                for param_group in optimizer.param_groups:
                    if param_group['weight_decay'] > 0:
                        param_group['weight_decay'] = wd_schedule_values[it]
        
        # Prepare batch data - handle both standard 4M format and MCoT format
        if isinstance(batch, dict) and 'mod_dict' in batch:
            # MCoT format batch
            mod_dict = batch['mod_dict']
            mcot_step = batch.get('mcot_step', 'planning')
            mcot_context = batch.get('mcot_context', {})
        else:
            # Standard 4M format batch
            mod_dict = {
                modality: {k: v.to(device, non_blocking=True) for k, v in d.items()}
                for modality, d in batch.items()
                if modality in args.all_domains
            }
            mcot_step = None
            mcot_context = {}
        
        # Move data to device
        if not isinstance(batch, dict) or 'mod_dict' not in batch:
            mod_dict = {
                modality: {k: v.to(device, non_blocking=True) for k, v in d.items()}
                for modality, d in mod_dict.items()
                if modality in args.all_domains
            }
        
        update_grad = (data_iter_step + 1) % update_freq == 0
        
        # Use gradient sync context for FSDP
        with nullcontext() if update_grad else model.no_sync():
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Forward pass with MCoT support
                if hasattr(model, 'mcot_processor') and mcot_step:
                    # MCoT-enhanced forward pass
                    outputs = model(
                        mod_dict, 
                        num_encoder_tokens=args.num_input_tokens,
                        num_decoder_tokens=args.num_target_tokens,
                        mcot_step=mcot_step,
                        mcot_context=mcot_context,
                        loss_type=args.loss_type
                    )
                else:
                    # Standard 4M forward pass
                    outputs = model(
                        mod_dict,
                        num_encoder_tokens=args.num_input_tokens,
                        num_decoder_tokens=args.num_target_tokens,
                        loss_type=args.loss_type
                    )
                
                # Extract loss and individual losses
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    loss, mod_loss = outputs[0], outputs[1]
                elif hasattr(outputs, 'loss'):
                    loss = outputs.loss
                    mod_loss = getattr(outputs, 'mod_loss', {})
                else:
                    loss = outputs
                    mod_loss = {}
                
                loss_value = loss.item()
                mod_loss_values = {f'{mod}_loss': l.item() for mod, l in mod_loss.items()} if isinstance(mod_loss, dict) else {}
            
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                if args.output_dir:
                    torch.save(mod_dict, os.path.join(args.output_dir, "debug_mod_dict.pt"))
                sys.exit(1)
            
            loss = loss / update_freq
            loss.backward()
            
            if update_grad:
                nan_gradients = False
                
                if max_norm is not None:
                    if hasattr(model, 'clip_grad_norm_'):
                        grad_norm = model.clip_grad_norm_(max_norm)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    
                    # Check for NaN gradients if specified
                    if getattr(args, 'skip_nan_grad', False):
                        if hasattr(grad_norm, 'isnan') and grad_norm.isnan():
                            nan_gradients = True
                    
                    grad_norm = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
                else:
                    grad_norm = None
                
                if not (getattr(args, 'skip_nan_grad', False) and nan_gradients):
                    optimizer.step()
                elif nan_gradients:
                    print(f"Skipping step {data_iter_step} in epoch {epoch} due to NaN gradients")
                
                optimizer.zero_grad()
        
        torch.cuda.synchronize()
        
        # Update metrics
        metric_logger.update(loss=loss_value)
        metric_logger.update(**mod_loss_values)
        
        min_lr = 10.
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
        if weight_decay_value is not None:
            metric_logger.update(weight_decay=weight_decay_value)
        
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm)
        
        # Log to wandb
        if log_writer is not None and update_grad:
            log_writer.update({
                'loss': loss_value,
                'lr': max_lr,
                'min_lr': min_lr,
                'weight_decay': weight_decay_value or 0.0,
            })
            log_writer.update(mod_loss_values)
            if grad_norm is not None:
                log_writer.update({'grad_norm': grad_norm})
            
            # Add token tracking
            total_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()
            log_writer.update({
                'input_tokens_seen_b': it * (total_batch_size / args.accum_iter) * args.num_input_tokens / 1e9,
                'target_tokens_seen_b': it * (total_batch_size / args.accum_iter) * args.num_target_tokens / 1e9,
                'total_tokens_seen_b': it * (total_batch_size / args.accum_iter) * (args.num_input_tokens + args.num_target_tokens) / 1e9,
            })
            log_writer.set_step()
        
        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + data_iter_step)
    
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    torch.cuda.empty_cache()
    
    return {'[Epoch] ' + k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, args):
    """Evaluate model following base script structure."""
    model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        # Prepare batch data - handle both formats
        if isinstance(batch, dict) and 'mod_dict' in batch:
            # MCoT format batch
            mod_dict = batch['mod_dict']
            mcot_step = batch.get('mcot_step', 'planning')
            mcot_context = batch.get('mcot_context', {})
        else:
            # Standard 4M format batch
            mod_dict = {
                modality: {k: v.to(device, non_blocking=True) for k, v in d.items()}
                for modality, d in batch.items()
                if modality in args.all_domains
            }
            mcot_step = None
            mcot_context = {}
        
        with torch.no_grad():
            # Forward pass with MCoT support
            if hasattr(model, 'mcot_processor') and mcot_step:
                outputs = model(
                    mod_dict,
                    num_encoder_tokens=args.num_input_tokens,
                    num_decoder_tokens=args.num_target_tokens,
                    mcot_step=mcot_step,
                    mcot_context=mcot_context,
                    loss_type=args.loss_type
                )
            else:
                outputs = model(
                    mod_dict,
                    num_encoder_tokens=args.num_input_tokens,
                    num_decoder_tokens=args.num_target_tokens,
                    loss_type=args.loss_type
                )
            
            # Extract loss
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                loss, mod_loss = outputs[0], outputs[1]
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
                mod_loss = getattr(outputs, 'mod_loss', {})
            else:
                loss = outputs
                mod_loss = {}
            
            metric_logger.update(loss=loss.item())
            if isinstance(mod_loss, dict):
                mod_loss_values = {f'{mod}_loss': l.item() for mod, l in mod_loss.items()}
                metric_logger.update(**mod_loss_values)
    
    # Gather stats
    metric_logger.synchronize_between_processes()
    print('* Loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_mcot_step_loss(outputs, targets, step, criterion):
    """Compute loss for a specific MCoT step."""
    # Extract step-specific outputs and targets
    if isinstance(outputs, tuple):
        # Handle case where outputs is (loss, logits)
        step_outputs = outputs[1] if len(outputs) > 1 else outputs[0]
    else:
        step_outputs = outputs
    
    if isinstance(targets, dict) and step in targets:
        step_targets = targets[step]
    else:
        step_targets = targets
    
    # Compute step loss
    if hasattr(criterion, step):
        # Use step-specific criterion if available
        step_criterion = getattr(criterion, step)
        loss = step_criterion(step_outputs, step_targets)
    else:
        # Use default criterion
        loss = criterion(step_outputs, step_targets)
    
    return loss


def compute_mcot_total_loss(step_losses, step_weights=None):
    """Compute total weighted MCoT loss."""
    if step_weights is None:
        return sum(step_losses.values()) / len(step_losses)
    
    total_loss = 0.0
    total_weight = 0.0
    
    for step, loss in step_losses.items():
        weight = step_weights.get(step, 1.0)
        total_loss += loss * weight
        total_weight += weight
    
    return total_loss / total_weight if total_weight > 0 else total_loss


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = None,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, num_training_steps_per_epoch=None, update_freq=None,
                    use_amp=False, args=None):
    """Train one epoch with MCoT-specific processing."""
    
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', "{:.8f}")
    metric_logger.add_meter('min_lr', "{:.8f}")
    # Add MCoT-specific metrics
    metric_logger.add_meter('seetrue_usage', "{:.3f}")
    metric_logger.add_meter('reflection_enhanced', "{:.3f}")
    header = f'Epoch: [{epoch}]'
    print_freq = 10
    
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # Setup MCoT step weights
    mcot_step_weights = {
        'planning': args.mcot_planning_weight,
        'acting': args.mcot_acting_weight,
        'reflection': args.mcot_reflection_weight,
        'correction': args.mcot_correction_weight
    }
    
    mcot_steps = args.mcot_steps.split(',')
    mcot_steps = [step.strip() for step in mcot_steps]
    
    # Track SeeTRUE-Feedback usage for monitoring
    seetrue_usage_count = 0
    reflection_enhanced_count = 0
    total_reflection_samples = 0
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        
        it = start_steps + step  # global training iteration
        
        # Update learning rate and weight decay
        if lr_schedule_values is not None:
            for param_group in optimizer.param_groups:
                if lr_schedule_values[it] is not None:
                    param_group['lr'] = lr_schedule_values[it]
        if wd_schedule_values is not None and len(wd_schedule_values) > it:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = wd_schedule_values[it]

        # Parse batch data from collate function output
        # The collate function returns a dictionary with keys: 'mod_dict', 'image_ids', 'mcot_steps', 'target_texts', 'step_data', etc.
        if not isinstance(batch, dict):
            print(f"Warning: Expected batch to be dict, got {type(batch)}. Skipping batch.")
            continue
            
        mod_dict = batch.get('mod_dict', {})
        mcot_steps_batch = batch.get('mcot_steps', [])
        target_texts_batch = batch.get('target_texts', [])
        step_data = batch.get('step_data', {})
        batch_size = batch.get('batch_size', len(mcot_steps_batch))
        
        if not mod_dict or not mcot_steps_batch:
            print("Warning: Empty batch data, skipping.")
            continue
        
        # Process each sample in the batch
        batch_loss = 0.0
        batch_samples = 0
        
        for i in range(batch_size):
            if i >= len(mcot_steps_batch) or i >= len(target_texts_batch):
                continue
                
            sample_mcot_step = mcot_steps_batch[i]
            sample_target_text = target_texts_batch[i]
            
            # Create sample mod_dict
            sample_mod_dict = {}
            for modality, batch_data in mod_dict.items():
                if isinstance(batch_data, torch.Tensor):
                    sample_mod_dict[modality] = batch_data[i:i+1]  # Keep batch dimension
                elif isinstance(batch_data, list) and i < len(batch_data):
                    sample_mod_dict[modality] = [batch_data[i]]
                else:
                    sample_mod_dict[modality] = batch_data
            
            # Process MCoT step for this sample
            step_losses = {}
            
            # Use the current step as specified in batch or default to planning
            current_step = sample_mcot_step if sample_mcot_step in mcot_steps else 'planning'
            
            # Setup context for MCoT processing
            mcot_context = {}
            
            # Extract SeeTRUE-Feedback data if available for reflection step
            seetrue_data = None
            if current_step == 'reflection' and step_data:
                total_reflection_samples += 1
                
                # step_data is a dict with step names as keys
                reflection_step_data = step_data.get('reflection', {})
                if reflection_step_data and 'seetrue_data' in reflection_step_data:
                    seetrue_batch = reflection_step_data['seetrue_data']
                    if isinstance(seetrue_batch, list) and i < len(seetrue_batch):
                        seetrue_data = seetrue_batch[i]
                    elif not isinstance(seetrue_batch, list):
                        seetrue_data = seetrue_batch
                        
                    if seetrue_data:
                        mcot_context['seetrue_data'] = seetrue_data
                        seetrue_usage_count += 1
                        reflection_enhanced_count += 1
                        # Log SeeTRUE usage for monitoring
                        if args.enable_mint_features:
                            print(f"✅ Using SeeTRUE-Feedback data for reflection training on sample {i}")
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Forward pass for this step with enhanced context
                outputs = model(
                    mod_dict=sample_mod_dict,
                    num_encoder_tokens=args.num_input_tokens,
                    num_decoder_tokens=args.num_target_tokens,
                    mcot_step=current_step,
                    mcot_context=mcot_context
                )
                
                # Extract loss from outputs
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    step_loss = outputs[0]  # loss is first element
                elif hasattr(outputs, 'loss'):
                    step_loss = outputs.loss
                else:
                    # Skip if no loss can be computed
                    continue
                
                # Weight the step loss
                weighted_step_loss = step_loss * mcot_step_weights.get(current_step, 1.0)
                step_losses[current_step] = weighted_step_loss
            
            # Add sample loss to batch
            if step_losses:
                sample_loss = sum(step_losses.values()) / len(step_losses)
                batch_loss += sample_loss
                batch_samples += 1
        
        # Compute final batch loss
        if batch_samples > 0:
            loss = batch_loss / batch_samples
        else:
            # Skip batch if no valid samples
            continue
        
        loss_value = loss.item()
        
        if not np.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # Backward pass
        if loss_scaler is not None:
            # Use mixed precision training
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                       parameters=model.parameters(), create_graph=False,
                       update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
        else:
            # Standard training
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()

        torch.cuda.synchronize()

        # Update metrics
        metric_logger.update(loss=loss_value)
        for step_name, step_loss in step_losses.items():
            metric_logger.update(**{f'loss_{step_name}': step_loss.item()})
        
        # Update SeeTRUE usage metrics
        if total_reflection_samples > 0:
            seetrue_usage_rate = seetrue_usage_count / total_reflection_samples
            reflection_enhanced_rate = reflection_enhanced_count / total_reflection_samples
            metric_logger.update(seetrue_usage=seetrue_usage_rate)
            metric_logger.update(reflection_enhanced=reflection_enhanced_rate)
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        
        # Log to wandb
        if log_writer is not None and (data_iter_step + 1) % update_freq == 0:
            log_writer.update(loss=loss_value, head="loss")
            for step_name, step_loss in step_losses.items():
                log_writer.update(**{f'loss_{step_name}': step_loss.item()}, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.set_step()

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, args):
    """Evaluate model on validation set."""
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()
    mcot_steps = args.mcot_steps.split(',')
    mcot_steps = [step.strip() for step in mcot_steps]

    for batch in metric_logger.log_every(data_loader, 10, header):
        # Parse batch data from collate function output
        if not isinstance(batch, dict):
            continue
            
        mod_dict = batch.get('mod_dict', {})
        mcot_steps_batch = batch.get('mcot_steps', [])
        target_texts_batch = batch.get('target_texts', [])
        batch_size = batch.get('batch_size', len(mcot_steps_batch))
        
        if not mod_dict or not mcot_steps_batch:
            continue
        
        # Process each sample in the batch
        batch_loss = 0.0
        batch_samples = 0
        
        for i in range(batch_size):
            if i >= len(mcot_steps_batch) or i >= len(target_texts_batch):
                continue
                
            sample_mcot_step = mcot_steps_batch[i]
            sample_target_text = target_texts_batch[i]
            
            # Create sample mod_dict
            sample_mod_dict = {}
            for modality, batch_data in mod_dict.items():
                if isinstance(batch_data, torch.Tensor):
                    sample_mod_dict[modality] = batch_data[i:i+1]
                elif isinstance(batch_data, list) and i < len(batch_data):
                    sample_mod_dict[modality] = [batch_data[i]]
                else:
                    sample_mod_dict[modality] = batch_data
            
            # Process MCoT step for this sample
            current_step = sample_mcot_step if sample_mcot_step in mcot_steps else 'planning'
            mcot_context = {}
            
            # Extract SeeTRUE-Feedback data if available for reflection step
            step_data = batch.get('step_data', {})
            if current_step == 'reflection' and i < len(step_data) if isinstance(step_data, list) else step_data:
                sample_step_data = step_data[i] if isinstance(step_data, list) else step_data
                if isinstance(sample_step_data, dict):
                    seetrue_data = sample_step_data.get('seetrue_data')
                    if seetrue_data:
                        mcot_context['seetrue_data'] = seetrue_data
            
            with torch.no_grad():
                outputs = model(
                    mod_dict=sample_mod_dict,
                    num_encoder_tokens=args.num_input_tokens,
                    num_decoder_tokens=args.num_target_tokens,
                    mcot_step=current_step,
                    mcot_context=mcot_context
                )
                
                if isinstance(outputs, tuple) and len(outputs) >= 2:
                    step_loss = outputs[0]
                elif hasattr(outputs, 'loss'):
                    step_loss = outputs.loss
                else:
                    continue
                
                batch_loss += step_loss
                batch_samples += 1
        
        # Update metrics for this batch
        if batch_samples > 0:
            avg_batch_loss = batch_loss / batch_samples
            metric_logger.update(loss=avg_batch_loss.item())

    # Gather stats
    metric_logger.synchronize_between_processes()
    print('* Loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    """Main training function."""
    utils.init_distributed_mode(args)

    print(f"Job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    
    # Set num_tasks for distributed training - essential for data loading
    num_tasks = utils.get_world_size()
    args.num_tasks = num_tasks
    
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Setup data
    modality_info, data_loader_train, num_training_steps_per_epoch, data_loaders_val, data_loaders_fixed_eval = setup_data(args)
    
    # Setup model
    model = get_model(args, modality_info)
    model.to(device)

    # Setup FSDP
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model = {args.model}")
    print(f"Number of params: {n_parameters / 1e6:.1f}M")

    # Setup mixed precision
    mp_policy = None
    if args.dtype in ['fp16', 'float16']:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif args.dtype in ['bf16', 'bfloat16']:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    # Wrap with FSDP
    if utils.get_world_size() > 1:
        model = FSDP(
            model,
            auto_wrap_policy=transformer_auto_wrap_policy,
            mixed_precision=mp_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=None,
            limit_all_gathers=True,
            device_id=torch.cuda.current_device(),
        )

    # Setup optimizer
    optimizer = create_optimizer(
        args, model_without_ddp if hasattr(model, 'module') else model
    )

    # Setup learning rate scheduler
    lr_schedule_values = utils.cosine_scheduler(
        args.blr, args.min_blr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay, args.epochs, num_training_steps_per_epoch
    )

    # Setup loss scaler for mixed precision
    loss_scaler = utils.NativeScalerWithGradNormCount() if args.dtype in ['fp16', 'bf16'] else None

    # Setup logging
    log_writer = None
    if HAS_WANDB and args.log_wandb and utils.is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.run_name,
            config=vars(args)
        )
        log_writer = wandb

    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        # Train one epoch
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            max_norm=args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.accum_iter,
            use_amp=(args.dtype in ['fp16', 'bf16']),
            args=args
        )
        
        # Evaluate
        if data_loaders_val is not None:
            val_stats = evaluate(model, data_loaders_val, device, args)
            print(f"Validation loss: {val_stats['loss']:.4f}")
            
            if log_writer:
                for k, v in val_stats.items():
                    log_writer.log({f"val/{k}": v}, step=epoch)

        # Save checkpoint
        if args.output_dir and (epoch % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs):
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{epoch}.pth')
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Log training stats
        if log_writer:
            for k, v in train_stats.items():
                log_writer.log({f"train/{k}": v}, step=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
