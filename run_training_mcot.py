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
import functools
import json
import math
import os
import logging
import resource
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any

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
from torch.utils.data import DataLoader, DistributedSampler

# MCOT specific imports
from scripts.mcot_dataset import MCoTDataset 
from scripts.mcot_loader import mcot_collate_fn, PAD_TOKEN_ID

# 4M imports
import fourm.utils as utils
import fourm.utils.fsdp_utils as fsdp_utils
from fourm.utils.dist import init_distributed_mode, is_main_process, get_rank, get_world_size
from fourm.utils.optim_factory import create_optimizer
from fourm.utils.tokenizer.text_tokenizer import get_tokenizer
from fourm.utils.checkpoint import save_model, auto_load_model
from fourm.models.fm import FM
from fourm.models.fm_utils import Block, DecoderBlock
from fourm.data.modality_info import MODALITY_INFO

# Define a simple TensorboardLogger
class TensorboardLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            print("Warning: TensorBoard not available. Install with 'pip install tensorboard'")
            self.writer = None
            
    def log(self, metrics, step=None):
        if self.writer is None:
            return
            
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.writer.add_scalar(k, v, step)
            
    def flush(self):
        if self.writer is not None:
            self.writer.flush()
            
    def close(self):
        if self.writer is not None:
            self.writer.close()

def get_args():
    config_parser = parser = argparse.ArgumentParser(description='MCOT Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('MCOT Training Script (using FSDP)', add_help=False)
    parser.add_argument('--run_name', type=str, default='auto')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (default: %(default)s). '
                             'Effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Number of epochs (default: %(default)s)')
    parser.add_argument('--total_tokens', default=-1, type=int,
                        help='Number of total input tokens (in billions), only applicable if epochs is negative. '
                             'Sets the number of epochs to approximate this amount of tokens.')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--save_ckpt_freq', default=5, type=int,
                        help='Checkpoint saving frequency in epochs (default: %(default)s)')
    parser.add_argument('--use_act_checkpoint', action='store_true',
                        help='Use activation checkpointing to save memory during training')
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

    # MCOT specific parameters
    parser.add_argument('--plan_max_seq_length', default=512, type=int,
                        help='Maximum sequence length for planning target sequence')
    parser.add_argument('--acting_max_seq_length', default=768, type=int,
                        help='Maximum sequence length for acting input sequence')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory for MSCOCO data')
    parser.add_argument('--coord_bins', default=1000, type=int,
                        help='Number of bins for coordinate quantization')
                        
    # Weight init / fine-tune parameters
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str,
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--opt_eps', default=1e-8, type=float,
                        help='Optimizer epsilon (default: %(default)s)')
    parser.add_argument('--opt_betas', default=[0.9, 0.95], type=float, nargs='+',
                        help='Optimizer betas (default: %(default)s)')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='Clip gradient norm (default: %(default)s)')
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
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'inverse_sqrt-10000'],
                        help='Learning rate scheduler type (default: %(default)s')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='Epochs to warmup LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--warmup_steps', type=int, default=-1,
                        help='Steps to warmup LR, if scheduler supports (default: %(default)s)')
    parser.add_argument('--warmup_tokens', type=int, default=-1,
                        help='Total tokens to warmup LR, if scheduler supports (default: %(default)s)')

    # Misc.
    parser.add_argument('--output_dir', default='./outputs/mcot_training',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int, 
                        help='Number of data loading workers')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--rlimit', default=4096, type=int, 
                        help='Increase rlimit to avoid "RuntimeError: received 0 items of ancdata".')
    
    # Evaluation
    parser.add_argument('--eval_freq', default=1, type=int, help="frequency of evaluation")
    parser.add_argument('--eval_split', default='val', type=str, 
                        help="Dataset split to use for evaluation")
    
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

    # Parse config file if there is one
    args_config, remaining = config_parser.parse_known_args()

    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)            

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file is specified.
    args = parser.parse_args(remaining)

    # Add the config path as a final arg if given
    args.config_path = args_config.config

    return args

def setup_modality_info(config):
    """
    Set up modality information for the MCOT tasks:
    - tok_rgb@224: image tokens (input for planning, target for acting)
    - plan_sequence: tokenized plan sequence (output of planning, input to acting)
    """
    # Global modality info
    all_domains = ['tok_rgb@224', 'plan_sequence']
    
    # Add any domains from the config
    if 'domains_in' in config:
        all_domains.extend([d for d in config['domains_in'] if d not in all_domains])
    if 'domains_out' in config:
        all_domains.extend([d for d in config['domains_out'] if d not in all_domains])
    
    # Remove duplicates and sort
    all_domains = sorted(list(set(all_domains)))
    
    # Get modality info for all domains
    modality_info = {}
    for mod in all_domains:
        if mod in MODALITY_INFO:
            modality_info[mod] = MODALITY_INFO[mod]
        else:
            # For plan_sequence, we need to add a fallback if it's not in MODALITY_INFO
            if mod == 'plan_sequence':
                # Copy from 'caption' if available, or create a basic text modality
                if 'caption' in MODALITY_INFO:
                    modality_info[mod] = MODALITY_INFO['caption'].copy()
                    modality_info[mod]['max_tokens'] = 512  # Default for plan sequence
                else:
                    modality_info[mod] = {
                        'type': 'text',
                        'max_tokens': 512,
                        'vocab_size': 30000,  # Estimate based on tokenizer
                    }
                print(f"Created fallback modality info for {mod}")
            else:
                print(f"Warning: Missing modality information for {mod}")
    
    # Max tokens for image modalities
    for mod in modality_info:
        if modality_info[mod]['type'] == 'img':
            image_size = modality_info[mod].get('input_size', config.get('input_size', 224))
            patch_size = modality_info[mod].get('patch_size', config.get('patch_size', 16))
            num_patches = (image_size // patch_size) ** 2
            modality_info[mod]['max_tokens'] = num_patches

    return modality_info, all_domains

def get_model(config, modality_info, input_domains, output_domains):
    """Creates and returns FM model from arguments with appropriate modality embeddings"""
    
    # Save the model type string before removing it from config
    model_type = config.get('model', 'fm_base_12e_12d_swiglu_nobias')
    print(f"Creating model: {model_type} for input modalities {input_domains} and output modalities {output_domains}")
    
    # Save alphas_config separately - it's not used by FourM.__init__
    alphas_config = config.get('alphas_config', {
        'plan_sequence': {'alpha': 1.0},
        'tok_rgb@224': {'alpha': 1.0}
    })
    
    # Important: We don't need to create modality_info here since the FM class does that
    # internally based on domains_in and domains_out
    
    # Create encoder and decoder embeddings just so we can inspect them
    # (FM will create its own internally)
    encoder_embeddings = {}
    for mod in input_domains:
        if mod in modality_info:
            info = modality_info[mod]
            if info.get("encoder_embedding", None) is not None:
                if info["type"] == "img":
                    # Use .get() with defaults for input_size and patch_size
                    image_size = info.get('input_size', config.get('image_size', 224))
                    patch_size = info.get('patch_size', config.get('patch_size', 16))
                    print(f"Creating encoder embedding for {mod} with image_size={image_size}, patch_size={patch_size}")
                    # Don't actually create the embedding - FM will do this internally
                    # encoder_embeddings[mod] = info["encoder_embedding"](patch_size=patch_size, image_size=image_size)
        else:
            print(f"Warning: Modality {mod} not found in modality_info. FM will handle this internally.")

    # Create full model configuration with only the parameters expected by FourM.__init__
    # Based on FM.__init__, these parameters are used to create embeddings but are removed before being passed to FourM.__init__:
    # 'image_size', 'patch_size', 'norm_bias', 'domains_in', 'domains_out', 'input_size'
    model_config = {
        # These are used by FM but not passed to FourM
        'image_size': config.get('image_size', 224),
        'patch_size': config.get('patch_size', 16),
        'domains_in': input_domains,
        'domains_out': output_domains,
        'norm_bias': config.get('norm_bias', False),
        
        # Parameters that will be passed to FourM.__init__
        'dim': config.get('dim', 768),
        'mlp_ratio': config.get('mlp_ratio', 4),
        'encoder_depth': config.get('encoder_depth', 12),
        'decoder_depth': config.get('decoder_depth', 12),
        'num_heads': config.get('num_heads', 12),
        'act_layer': config.get('act_layer', 'SiLU'),
        'gated_mlp': config.get('gated_mlp', True),
        'qkv_bias': config.get('qkv_bias', False),
        'proj_bias': config.get('proj_bias', False),
        'mlp_bias': config.get('mlp_bias', False),
        'num_register_tokens': config.get('num_register_tokens', 0),
        'share_modality_embeddings': config.get('share_modality_embeddings', False),
        
        # REMOVED: Additional parameters not accepted by FourM.__init__
        # 'alphas_config': alphas_config,
    }
    
    # Create the model
    print("Creating FM model with config keys:", list(model_config.keys()))
    model = FM(config=model_config)
    
    # After model creation, we can attach the alphas_config if needed for loss calculation later
    model.alphas_config = alphas_config
    
    return model

def load_checkpoint(model, finetune_path):
    """Load checkpoint for fine-tuning"""
    if not finetune_path:
        return
        
    print(f"Loading fine-tuning checkpoint from: {finetune_path}")
    
    # Handle different checkpoint formats (local file, URL, safetensors)
    if finetune_path.endswith('.safetensors'):
        try:
            # Try to use safetensors to load the file
            from safetensors import safe_open
            from safetensors.torch import load_file
            
            # If it's a URL, download it first
            if finetune_path.startswith('http'):
                local_path = torch.hub.get_dir()
                local_path = os.path.join(local_path, "checkpoints", "model.safetensors")
                if not os.path.exists(local_path):
                    print(f"Downloading safetensors file to {local_path}")
                    torch.hub.download_url_to_file(finetune_path, local_path, progress=True)
                finetune_path = local_path
            
            # Load the safetensors file
            print(f"Loading with safetensors from {finetune_path}")
            checkpoint = load_file(finetune_path)
        except ImportError:
            print("safetensors not available. Please install with: pip install safetensors")
            print("Falling back to PyTorch loading (may not work for safetensors files)")
            checkpoint = torch.load(finetune_path, map_location='cpu')
    elif finetune_path.startswith('http'):
        checkpoint = torch.hub.load_state_dict_from_url(finetune_path, map_location='cpu')
    else:
        checkpoint = torch.load(finetune_path, map_location='cpu')
    
    # Extract model state dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    
    # Handle key mismatches for fine-tuning (e.g., new output heads)
    msg = model.load_state_dict(checkpoint, strict=False)
    print(f"Checkpoint loaded with message: {msg}")
    
    # Log missing and unexpected keys
    if len(msg.missing_keys) > 0:
        print(f"Missing keys: {msg.missing_keys}")
    if len(msg.unexpected_keys) > 0:
        print(f"Unexpected keys: {msg.unexpected_keys}")

def train_one_epoch(model, data_loader, optimizer, device, criterion, epoch, 
                   accum_iter, max_norm=None, log_writer=None, lr_scheduler=None, 
                   args=None, world_size=1):
    """Train the model for one epoch with MCOT-specific loss computation"""
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    header = f'Epoch: [{epoch}]'
    batch_size = args.batch_size
    
    total_batch_size = batch_size * accum_iter * world_size
    num_training_steps_per_epoch = len(data_loader) // accum_iter
    print(f"LR = {optimizer.param_groups[0]['lr']:.6f}")
    
    step = 0
    optimizer.zero_grad()
    
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # Determine if we're on the last step or should accumulate gradients
        is_last_step_in_batch = (data_iter_step + 1) % accum_iter == 0
        do_optimizer_step = is_last_step_in_batch
        
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        
        # Create a mod_dict structure for the FourM model
        mod_dict = {}
        
        # For encoder (input) modality: tok_rgb@224
        if 'planning_input_image_tokens' in batch:
            mod_dict['tok_rgb@224'] = {
                'tensor': batch['planning_input_image_tokens'],
                'input_mask': torch.zeros_like(batch['planning_input_image_tokens'], dtype=torch.bool),
                'target_mask': torch.ones_like(batch['planning_input_image_tokens'], dtype=torch.bool),
                'decoder_attention_mask': torch.zeros_like(batch['planning_input_image_tokens'], dtype=torch.int),
            }
        
        # For decoder (output) modalities:
        # 1. plan_sequence (planning stage output)
        if 'planning_target_sequence' in batch:
            plan_target = batch['planning_target_sequence']
            mod_dict['plan_sequence'] = {
                'tensor': plan_target,
                'input_mask': torch.ones_like(plan_target, dtype=torch.bool),
                'target_mask': torch.zeros_like(plan_target, dtype=torch.bool),
                'decoder_attention_mask': torch.zeros_like(plan_target, dtype=torch.int),
            }
            # Set first position in attention mask to sequence length (for causal masking)
            batch_size = plan_target.size(0)
            seq_len = plan_target.size(1)
            mod_dict['plan_sequence']['decoder_attention_mask'][:, 0] = seq_len
        
        # 2. tok_rgb@224 (acting stage output - image reconstruction)
        if 'acting_target_image_tokens' in batch:
            act_target = batch['acting_target_image_tokens']
            if 'tok_rgb@224' not in mod_dict:
                # If not already created for input
                mod_dict['tok_rgb@224'] = {
                    'tensor': act_target,
                    'input_mask': torch.ones_like(act_target, dtype=torch.bool),
                    'target_mask': torch.zeros_like(act_target, dtype=torch.bool),
                    'decoder_attention_mask': torch.zeros_like(act_target, dtype=torch.int),
                }
                # Set first position in attention mask to sequence length
                mod_dict['tok_rgb@224']['decoder_attention_mask'][:, 0] = act_target.size(1)
        
        # Calculate the number of encoder and decoder tokens
        # Use all available tokens for the encoder
        num_encoder_tokens = 0
        if 'planning_input_image_tokens' in batch:
            num_encoder_tokens = batch['planning_input_image_tokens'].size(1)
        else:
            # Default to a reasonable value if input not available
            num_encoder_tokens = 256
        
        # For decoder, count tokens from all output modalities
        num_decoder_tokens = 0
        if 'planning_target_sequence' in batch:
            num_decoder_tokens += batch['planning_target_sequence'].size(1)
        if 'acting_target_image_tokens' in batch:
            num_decoder_tokens += batch['acting_target_image_tokens'].size(1)
        
        if num_decoder_tokens == 0:
            # Default to a reasonable value
            num_decoder_tokens = 512
        
        # Forward pass - model should return a dict of logits for each output modality
        outputs = model(mod_dict, num_encoder_tokens=num_encoder_tokens, num_decoder_tokens=num_decoder_tokens, return_logits=True)
        
        # Print output keys and shapes for debugging
        print(f"Model output keys: {list(outputs.keys())}")
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                print(f"Output '{k}' shape: {v.shape}")
        
        # Calculate losses
        task_losses = {}
        total_loss = 0.0
        num_tasks = 0

        # Planning loss (plan_sequence modality)
        if 'plan_sequence' in outputs and 'planning_target_sequence' in batch:
            plan_logits = outputs['plan_sequence']
            plan_targets = batch['planning_target_sequence']
            
            # Handle autoregressive sequence prediction
            if plan_logits.dim() == 3 and plan_targets.dim() == 2:
                # Standard case - reshape for loss calculation
                # Slice logits to match target sequence length
                seq_len_targets = plan_targets.size(1)
                plan_logits_sliced = plan_logits[:, :seq_len_targets, :]

                # Explicitly reshape before passing to criterion
                reshaped_plan_logits = plan_logits_sliced.reshape(-1, plan_logits_sliced.size(-1))
                reshaped_plan_targets = plan_targets.reshape(-1)

                loss_planning = criterion(reshaped_plan_logits, reshaped_plan_targets)
            else:
                # Skip this batch if shapes are incompatible
                print(f"Warning: Skipping planning loss due to incompatible shapes: {plan_logits.shape} vs {plan_targets.shape}")
                raise ValueError(f"Incompatible shapes for planning: plan_logits {plan_logits.shape}, plan_targets {plan_targets.shape}")
            
            # Get alpha weight for this modality
            plan_alpha = 1.0  # Default if not specified
            if args.config_path and 'alphas_config' in args.__dict__:
                plan_alpha = args.alphas_config.get('plan_sequence', {}).get('alpha', 1.0)
            
            # Apply weight and add to total
            task_losses['planning'] = loss_planning.item()
            total_loss += plan_alpha * loss_planning
            num_tasks += 1
        
        # Acting loss (tok_rgb@224 modality)
        if 'tok_rgb@224' in outputs and 'acting_target_image_tokens' in batch:
            acting_logits = outputs['tok_rgb@224']
            acting_targets = batch['acting_target_image_tokens']
            
            # Handle image token prediction
            if acting_logits.dim() == 3 and acting_targets.dim() == 2:
                # Standard case - reshape for loss calculation
                # Slice logits to match target sequence length
                seq_len_targets_acting = acting_targets.size(1)
                acting_logits_sliced = acting_logits[:, :seq_len_targets_acting, :]
                
                # Explicitly reshape before passing to criterion
                reshaped_acting_logits = acting_logits_sliced.reshape(-1, acting_logits_sliced.size(-1))
                reshaped_acting_targets = acting_targets.reshape(-1)

                loss_acting = criterion(reshaped_acting_logits, reshaped_acting_targets)
            else:
                # Skip this batch if shapes are incompatible
                print(f"Warning: Skipping acting loss due to incompatible shapes: {acting_logits.shape} vs {acting_targets.shape}")
                raise ValueError(f"Incompatible shapes for acting: acting_logits {acting_logits.shape}, acting_targets {acting_targets.shape}")
            
            # Get alpha weight for this modality
            acting_alpha = 1.0  # Default if not specified
            if args.config_path and 'alphas_config' in args.__dict__:
                acting_alpha = args.alphas_config.get('tok_rgb@224', {}).get('alpha', 1.0)
            
            # Apply weight and add to total
            task_losses['acting'] = loss_acting.item()
            total_loss += acting_alpha * loss_acting
            num_tasks += 1
        
        # Average loss across tasks if more than one
        if num_tasks > 0:
            total_loss = total_loss / num_tasks
            
            # Scale loss for gradient accumulation
            loss = total_loss / accum_iter
            loss.backward()
            
            # Update weights if this is the last accumulation step
            if do_optimizer_step:
                # Gradient clipping if specified
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Update LR scheduler if step-based
                if lr_scheduler is not None:
                    lr_scheduler.step_update(epoch * num_training_steps_per_epoch + step)
                step += 1
                
            # Log metrics
            metric_logger.update(loss=total_loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
            # Log individual task losses
            for task, loss_val in task_losses.items():
                metric_logger.update(**{f"{task}_loss": loss_val})
    
    # Gather all metrics from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # Log to wandb if enabled
    if log_writer is not None:
        log_writer.log({
            "train_loss": metric_logger.loss.global_avg,
            "learning_rate": metric_logger.lr.global_avg,
            "epoch": epoch,
            **{f"train_{k}": v.global_avg for k, v in metric_logger.meters.items() 
               if k not in ['loss', 'lr']},
        })
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, device, criterion, prefix=""):
    """Evaluate the model on the validation set with MCOT-specific metrics"""
    model.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'{prefix}Eval:'
    
    # Track task-specific and overall metrics
    total_samples = 0
    total_loss = 0
    task_metrics = {
        'planning': {'loss': 0, 'samples': 0},
        'acting': {'loss': 0, 'samples': 0}
    }
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
                
        # Create a mod_dict structure for the FourM model
        mod_dict = {}
        
        # For encoder (input) modality: tok_rgb@224
        if 'planning_input_image_tokens' in batch:
            mod_dict['tok_rgb@224'] = {
                'tensor': batch['planning_input_image_tokens'],
                'input_mask': torch.zeros_like(batch['planning_input_image_tokens'], dtype=torch.bool),
                'target_mask': torch.ones_like(batch['planning_input_image_tokens'], dtype=torch.bool),
                'decoder_attention_mask': torch.zeros_like(batch['planning_input_image_tokens'], dtype=torch.int),
            }
        
        # For decoder (output) modalities:
        # 1. plan_sequence (planning stage output)
        if 'planning_target_sequence' in batch:
            plan_target = batch['planning_target_sequence']
            mod_dict['plan_sequence'] = {
                'tensor': plan_target,
                'input_mask': torch.ones_like(plan_target, dtype=torch.bool),
                'target_mask': torch.zeros_like(plan_target, dtype=torch.bool),
                'decoder_attention_mask': torch.zeros_like(plan_target, dtype=torch.int),
            }
            # Set first position in attention mask to sequence length (for causal masking)
            batch_size = plan_target.size(0)
            seq_len = plan_target.size(1)
            mod_dict['plan_sequence']['decoder_attention_mask'][:, 0] = seq_len
        
        # 2. tok_rgb@224 (acting stage output - image reconstruction)
        if 'acting_target_image_tokens' in batch:
            act_target = batch['acting_target_image_tokens']
            if 'tok_rgb@224' not in mod_dict:
                # If not already created for input
                mod_dict['tok_rgb@224'] = {
                    'tensor': act_target,
                    'input_mask': torch.ones_like(act_target, dtype=torch.bool),
                    'target_mask': torch.zeros_like(act_target, dtype=torch.bool),
                    'decoder_attention_mask': torch.zeros_like(act_target, dtype=torch.int),
                }
                # Set first position in attention mask to sequence length
                mod_dict['tok_rgb@224']['decoder_attention_mask'][:, 0] = act_target.size(1)
        
        # Calculate the number of encoder and decoder tokens
        # Use all available tokens for the encoder
        num_encoder_tokens = 0
        if 'planning_input_image_tokens' in batch:
            num_encoder_tokens = batch['planning_input_image_tokens'].size(1)
        else:
            # Default to a reasonable value if input not available
            num_encoder_tokens = 256
        
        # For decoder, count tokens from all output modalities
        num_decoder_tokens = 0
        if 'planning_target_sequence' in batch:
            num_decoder_tokens += batch['planning_target_sequence'].size(1)
        if 'acting_target_image_tokens' in batch:
            num_decoder_tokens += batch['acting_target_image_tokens'].size(1)
        
        if num_decoder_tokens == 0:
            # Default to a reasonable value
            num_decoder_tokens = 512
        
        # Forward pass
        outputs = model(mod_dict, num_encoder_tokens=num_encoder_tokens, num_decoder_tokens=num_decoder_tokens, return_logits=True)
        
        # Calculate metrics
        batch_size = next(iter(batch.values())).size(0)
        total_samples += batch_size
        
        # Planning metrics
        if 'plan_sequence' in outputs and 'planning_target_sequence' in batch:
            plan_logits = outputs['plan_sequence']
            plan_targets = batch['planning_target_sequence']
            
            # Handle autoregressive sequence prediction
            if plan_logits.dim() == 3 and plan_targets.dim() == 2:
                # Standard case - reshape for loss calculation
                # Slice logits to match target sequence length
                seq_len_targets = plan_targets.size(1)
                plan_logits_sliced = plan_logits[:, :seq_len_targets, :]

                # Explicitly reshape before passing to criterion
                reshaped_plan_logits = plan_logits_sliced.reshape(-1, plan_logits_sliced.size(-1))
                reshaped_plan_targets = plan_targets.reshape(-1)

                loss_planning = criterion(reshaped_plan_logits, reshaped_plan_targets)
            else:
                # Skip this batch if shapes are incompatible
                print(f"Warning: Skipping planning loss due to incompatible shapes: {plan_logits.shape} vs {plan_targets.shape}")
                continue
            
            # Update planning metrics
            task_metrics['planning']['loss'] += loss_planning.item() * batch_size
            task_metrics['planning']['samples'] += batch_size
            
            # Log per-sample metrics
            metric_logger.update(planning_loss=loss_planning.item())
        
        # Acting metrics  
        if 'tok_rgb@224' in outputs and 'acting_target_image_tokens' in batch:
            acting_logits = outputs['tok_rgb@224']
            acting_targets = batch['acting_target_image_tokens']
            
            # Handle image token prediction
            if acting_logits.dim() == 3 and acting_targets.dim() == 2:
                # Standard case - reshape for loss calculation
                # Slice logits to match target sequence length
                seq_len_targets_acting = acting_targets.size(1)
                acting_logits_sliced = acting_logits[:, :seq_len_targets_acting, :]
                
                # Explicitly reshape before passing to criterion
                reshaped_acting_logits = acting_logits_sliced.reshape(-1, acting_logits_sliced.size(-1))
                reshaped_acting_targets = acting_targets.reshape(-1)

                loss_acting = criterion(reshaped_acting_logits, reshaped_acting_targets)
            else:
                # Skip this batch if shapes are incompatible
                print(f"Warning: Skipping acting loss due to incompatible shapes: {acting_logits.shape} vs {acting_targets.shape}")
                continue
            
            # Update acting metrics
            task_metrics['acting']['loss'] += loss_acting.item() * batch_size
            task_metrics['acting']['samples'] += batch_size
            
            # Log per-sample metrics
            metric_logger.update(acting_loss=loss_acting.item())
            
            # Calculate reconstruction accuracy (percentage of correctly predicted tokens)
            pred_tokens = torch.argmax(acting_logits, dim=-1)
            
            # Ensure shapes match for accuracy calculation
            if pred_tokens.shape != acting_targets.shape:
                pred_tokens = pred_tokens[:acting_targets.size(0), :acting_targets.size(1)]
                
            # Calculate accuracy only on non-padding tokens
            mask = acting_targets != PAD_TOKEN_ID
            correct = (pred_tokens == acting_targets) & mask
            total = mask.sum()
            
            if total > 0:
                accuracy = correct.sum().float() / total
                metric_logger.update(acting_accuracy=accuracy.item())
                task_metrics['acting']['accuracy'] = accuracy.item()
    
    # Compute overall and per-task average metrics
    aggregated_metrics = {}
    
    # Calculate overall loss (weighted average of task losses)
    overall_loss = 0
    overall_weight = 0
    
    for task, metrics in task_metrics.items():
        if metrics['samples'] > 0:
            # Average loss for this task
            task_avg_loss = metrics['loss'] / metrics['samples']
            aggregated_metrics[f'{task}_loss'] = task_avg_loss
            
            # Add to overall weighted average
            overall_loss += metrics['loss']
            overall_weight += metrics['samples']
            
            # Include any other metrics
            for k, v in metrics.items():
                if k != 'loss' and k != 'samples':
                    aggregated_metrics[f'{task}_{k}'] = v
    
    # Calculate overall average loss
    if overall_weight > 0:
        aggregated_metrics['loss'] = overall_loss / overall_weight
    
    # Update metric logger
    for k, v in aggregated_metrics.items():
        metric_logger.meters[k].update(v, n=1)
    
    # Gather metrics from all processes
    metric_logger.synchronize_between_processes()
    print(f"{prefix} Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def create_embeddings_for_missing_modality(mod_name):
    """Create fallback embeddings for missing modalities"""
    
    print(f"Creating fallback embeddings for missing modality: {mod_name}")
    
    # Handle plan_sequence modality which might not be in MODALITY_INFO
    if mod_name == 'plan_sequence':
        from fourm.models.modality_encoders import TextEmbedding
        
        # Create a basic text embedding
        # If errors occur here, modify based on actual TextEmbedding implementation
        try:
            return TextEmbedding(
                vocab_size=30000,  # Large enough for MCOT vocab
                max_seq_len=512,   # Default for plan sequences
                embed_dim=768      # Match model dim
            )
        except Exception as e:
            print(f"Failed to create fallback embedding for {mod_name}: {e}")
            print("You may need to add this modality to MODALITY_INFO in the fourm code")
            return None
    
    print(f"No fallback embedding creation method for modality: {mod_name}")
    return None

def main(args):
    # Debug print
    if is_main_process():
        print("Available args attributes:", dir(args))
        print("Config path:", getattr(args, 'config_path', None))
    
    # Initialize distributed environment
    init_distributed_mode(args)
    device = torch.device(args.device)
    
    # Set random seed
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Enable cudnn benchmarking
    cudnn.benchmark = True
    
    # Set data type
    if args.dtype in ['float16', 'fp16']:
        dtype = torch.float16
    elif args.dtype in ['bfloat16', 'bf16']:
        dtype = torch.bfloat16
    elif args.dtype in ['float32', 'fp32']:
        dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {args.dtype}")
    
    # Load model configuration
    if hasattr(args, 'config_path') and args.config_path:
        with open(args.config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            if is_main_process():
                print("Loaded model config:", model_config.keys())
    else:
        model_config = {
            'model': args.model,
            'patch_size': args.patch_size,
            'input_size': args.input_size,
            'alphas_config': {
                'plan_sequence': {'alpha': 1.0},
                'tok_rgb@224': {'alpha': 1.0}
            }
        }
    
    # Set up modality information
    modality_info, all_domains = setup_modality_info(model_config)
    
    # Get input and output modalities, either from config or use defaults
    input_domains = model_config.get('input_modalities', ['tok_rgb@224'])
    output_domains = model_config.get('output_modalities', ['plan_sequence', 'tok_rgb@224'])
    
    if is_main_process():
        print(f"Using input modalities: {input_domains}")
        print(f"Using output modalities: {output_domains}")
        print(f"All available domains: {all_domains}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger for main process
    if is_main_process():
        log_writer = TensorboardLogger(log_dir=args.output_dir)
        if args.log_wandb:
            log_writer = utils.WandbLogger(args)
    else:
        log_writer = None
    
    # Create MCOT datasets and dataloaders
    train_dataset = MCoTDataset(
        data_root=args.data_root,
        split='train',
        tokenizer_path=model_config.get('tokenizer_path', ''),
        plan_max_seq_length=args.plan_max_seq_length,
        acting_max_seq_length=args.acting_max_seq_length,
        coord_bins=args.coord_bins
    )
    
    if is_main_process():
        print(f"Training dataset size: {len(train_dataset)} samples")
    
    # Set up distributed sampler for training
    world_size = get_world_size()
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    
    # Create dataloader with appropriate collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=mcot_collate_fn,
        drop_last=True
    )
    
    # Create validation dataset if eval_freq > 0
    if args.eval_freq > 0:
        val_dataset = MCoTDataset(
            data_root=args.data_root,
            split=args.eval_split,
            tokenizer_path=model_config.get('tokenizer_path', ''),
            plan_max_seq_length=args.plan_max_seq_length,
            acting_max_seq_length=args.acting_max_seq_length,
            coord_bins=args.coord_bins
        )
        
        if is_main_process():
            print(f"Validation dataset size: {len(val_dataset)} samples")
        
        # Create dataloader with appropriate collate function
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            collate_fn=mcot_collate_fn,
            drop_last=False
        )
    else:
        val_loader = None
    
    # Create model
    model = get_model(model_config, modality_info, input_domains, output_domains)

    # Load fine-tuning checkpoint if specified
    if args.finetune:
        load_checkpoint(model, args.finetune)
    
    # Check if we're running in distributed mode
    is_distributed = world_size > 1
    
    if is_distributed:
        # Set mixed precision for FSDP
        mixed_precision_policy = None
        if args.dtype != 'float32':
            mixed_precision_policy = MixedPrecision(
                param_dtype=getattr(torch, args.dtype),
                # Keep buffers as float32 (for better stability)
                buffer_dtype=torch.float32,
                reduce_dtype=getattr(torch, args.dtype)
            )
        
        # Create FSDP wrapping policy
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Block,
                DecoderBlock
            }
        )
        
        # Wrap model with FSDP
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            limit_all_gathers=True
        )
        
        # Apply activation checkpointing if enabled
        if args.use_act_checkpoint:
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=lambda submodule: isinstance(submodule, (Block, DecoderBlock)),
                activation_checkpointing_kwargs={"use_reentrant": False}
            )
    else:
        # When not using distributed training, just move model to the device
        model = model.to(device)
        
        # Apply activation checkpointing if enabled
        if args.use_act_checkpoint:
            for module in model.modules():
                if isinstance(module, (Block, DecoderBlock)):
                    module = checkpoint_wrapper(module, activation_checkpointing_kwargs={"use_reentrant": False})
    
    # Calculate learning rate scale (adjust for batch size) BEFORE creating the optimizer
    total_batch_size = args.batch_size * args.accum_iter * world_size
    args.lr = args.blr * total_batch_size / 256
    if is_main_process():
        print(f"Scaled learning rate (base): {args.lr}")
    
    # Create criterion (loss function)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    # Create optimizer
    optimizer = create_optimizer(args, model)
    
    # Create scheduler
    if args.scheduler == 'cosine':
        # Calculate steps per epoch
        num_steps_per_epoch = len(train_loader) // args.accum_iter
        
        # Convert epoch-based warmup to step-based
        if args.warmup_epochs > 0:
            args.warmup_steps = args.warmup_epochs * num_steps_per_epoch
        
        # Calculate the total number of training steps
        total_steps = args.epochs * num_steps_per_epoch
        
        # Create scheduler using cosine_scheduler function
        try:
            print(f"Creating cosine scheduler with: base_lr={args.lr}, min_lr={args.min_blr}, epochs={args.epochs}, steps_per_epoch={num_steps_per_epoch}, warmup_steps={args.warmup_steps}")
            lr_schedule = utils.scheduler.cosine_scheduler(
                args.lr,
                args.min_blr,
                args.epochs,
                num_steps_per_epoch,
                warmup_steps=args.warmup_steps
            )
        except AssertionError:
            # If we get an assertion error, it's likely because of a mismatch in the number of steps
            print("WARNING: Error creating cosine scheduler with the provided parameters.")
            print(f"Recalculating scheduling parameters to ensure consistency...")
            
            # Adjust the warmup steps if they exceed the total steps
            if args.warmup_steps >= total_steps:
                args.warmup_steps = total_steps // 10  # Default to 10% warmup
                print(f"Adjusted warmup_steps to {args.warmup_steps} (was exceeding total steps)")
            
            # Create the schedule with the adjusted parameters
            lr_schedule = utils.scheduler.cosine_scheduler(
                args.lr,
                args.min_blr,
                args.epochs,
                num_steps_per_epoch,
                warmup_steps=args.warmup_steps
            )
            print(f"Successfully created scheduler with {len(lr_schedule)} steps")
        
        # Create a simple step update function that uses the schedule
        def lr_step_update(optimizer, step):
            # Ensure step is within bounds of lr_schedule
            if step < len(lr_schedule):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule[step]
            else:
                print(f"WARNING: Step {step} exceeds scheduler length {len(lr_schedule)}, using min_lr")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.min_blr
        
        # Assign the step update function to lr_scheduler
        lr_scheduler = type('lr_scheduler', (object,), {
            'step_update': lambda current_step_value: lr_step_update(optimizer, current_step_value)
        })
    else:
        lr_scheduler = None
    
    # Resume from checkpoint if available
    if args.auto_resume and args.resume == '':
        import glob
        all_checkpoints = glob.glob(os.path.join(args.output_dir, 'checkpoint-*.pth'))
        if len(all_checkpoints) > 0:
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(args.output_dir, f'checkpoint-{latest_ckpt}.pth')
        
        if is_main_process() and args.resume:
            print(f"Auto resuming from {args.resume}")
    
    if args.resume:
        if is_distributed:
            # Use FSDP utilities to load checkpoint properly
            fsdp_utils.auto_load_model_fsdp(args, model, optimizer)
        else:
            # Regular checkpoint loading for non-distributed case
            checkpoint = torch.load(args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    if is_main_process():
        print(f"Start training for {args.epochs} epochs")
    
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        # Train one epoch
        train_stats = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            epoch=epoch,
            accum_iter=args.accum_iter,
            max_norm=args.clip_grad,
            log_writer=log_writer,
            lr_scheduler=lr_scheduler,
            args=args,
            world_size=world_size
        )
        
        # Update epoch-based scheduler
        if lr_scheduler is None:
            # If no step-based scheduler was created, update once per epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        
        # Save checkpoint
        if (epoch % args.save_ckpt_freq == 0 or epoch == args.epochs - 1) and args.output_dir:
            if is_distributed:
                fsdp_utils.save_model_fsdp(args, epoch, model, optimizer)
            else:
                # Save a regular checkpoint for non-distributed case
                if is_main_process():
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint-{epoch}.pth'))
        
        # Evaluate
        if val_loader is not None and (epoch % args.eval_freq == 0 or epoch == args.epochs - 1):
            val_stats = evaluate(
                model=model,
                data_loader=val_loader,
                device=device,
                criterion=criterion,
                prefix="[Validation]"
            )
            
            # Log validation metrics to wandb or tensorboard
            if log_writer is not None:
                log_writer.log({
                    **{f"val_{k}": v for k, v in val_stats.items()},
                    "epoch": epoch
                })
        
        if log_writer is not None:
            log_writer.flush()
        
        if is_main_process():
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **({"val_" + k: v for k, v in val_stats.items()} if val_loader is not None and epoch % args.eval_freq == 0 else {}),
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"]
            }
            
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    # End of training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    if is_main_process():
        print(f"Training time {total_time_str}")
    
    # Clean up
    if log_writer is not None:
        log_writer.close()

if __name__ == '__main__':
    args = get_args()
    
    # Use HuggingFace model URL for finetuning if not specified
    if not args.finetune:
        print("No finetune path specified, using default 4M model from HuggingFace")
        args.finetune = 'https://huggingface.co/EPFL-VILAB/4M-7_B_CC12M/resolve/main/model.safetensors'
        print(f"Using finetune path: {args.finetune}")
    
    # In case distributed stuff needs it
    if args.rlimit > 0:
        resource.setrlimit(resource.RLIMIT_NOFILE, (args.rlimit, args.rlimit))
        
    main(args) 