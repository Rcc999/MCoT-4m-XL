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
Enhanced 4M MCoT Training Script with MINT Paper Features
Based on the proven run_training_4m_fsdp.py but adapted for MCoT training.

This script adds MCoT capabilities to 4M finetuning, incorporating:
- Step-specific loss computation for Planning, Acting, Reflection, Correction
- Artifact heatmap generation during reflection
- Reflection-guided mask generation for correction
- MINT paper methodology integration
"""

import argparse
import datetime
import json
import os
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from tokenizers import Tokenizer

# PyTorch FSDP imports
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper
)
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# 4M imports
from fourm.data import (
    build_mixture_dataloader, get_train_dataloader, get_val_dataloader, setup_sampling_mod_info
)
from fourm.models import fm
from fourm.models.fm_utils import Block, DecoderBlock
from fourm.models.mcot_fixed import MCoTWrapper, add_mcot_to_model
from fourm.data.modality_info import MODALITY_INFO
from fourm.utils import create_model, load_safetensors
from fourm.utils.optim_factory import create_optimizer
from fourm.utils.dist import init_distributed_mode, is_main_process, get_rank, get_world_size
from fourm.utils.misc import NativeScalerWithGradNormCount
from fourm.utils.scheduler import cosine_scheduler
from fourm.utils.logger import MetricLogger

# MCoT data utilities
try:
    from mcot_data import mcot_utils
    MCOT_UTILS_AVAILABLE = True
except ImportError:
    print("Warning: mcot_utils not available. Some MCoT features may be limited.")
    MCOT_UTILS_AVAILABLE = False

# Optional wandb logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def get_args():
    """Parse command line arguments for MCoT training."""
    config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser('4M MCoT training script (using FSDP)', add_help=False)
    parser.add_argument('--run_name', type=str, default='mcot_training')

    # Basic training parameters
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (default: %(default)s). '
                             'Effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of epochs (default: %(default)s)')
    parser.add_argument('--total_tokens', default=-1, type=int,
                        help='Number of total input tokens (in billions), only applicable if epochs is negative.')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')
    parser.add_argument('--save_ckpt_freq', default=10, type=int,
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

    # Token budget parameters
    parser.add_argument('--num_input_tokens', type=int, default=256, help="Token budget for the input")
    parser.add_argument('--num_target_tokens', type=int, default=256, help="Token budget for the target")
    parser.add_argument('--min_input_tokens', type=int, default=None,
                        help="Minimum token budget for the input")
    parser.add_argument('--min_target_tokens', type=int, default=None,
                        help="Minimum token budget for the target")

    # MCoT specific parameters
    parser.add_argument('--mcot_steps', type=str, default="planning,acting,reflection,correction", 
                        help="MCoT steps to include in training, comma-separated")
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

    # Dataset parameters
    parser.add_argument('--data_config', type=str, required=True,
                        help='Path to data config file for MCoT dataset')
    parser.add_argument('--dataset_dirs', type=str, nargs='+', required=True,
                        help='Directories containing MCoT training data')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to MCoT data (JSON file or directory from wget script)')
    parser.add_argument('--tokenizer_path', type=str, default='fourm/utils/tokenizer/trained/text_tokenizer_4m.json',
                        help='Path to text tokenizer')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: %(default)s)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer epsilon (default: %(default)s)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: %(default)s)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--layer_decay', type=float, default=None,
                        help='Layer-wise lr decay (default: %(default)s)')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Lower lr bound for cyclic schedulers that hit 0 (default: %(default)s)')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='Steps to warmup LR, if scheduler supports')

    # Checkpointing and logging
    parser.add_argument('--finetune', default='',
                        help='Path to finetune from checkpoint')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch')
    parser.add_argument('--output_dir', default='./output/mcot_training',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logs/mcot_training',
                        help='Path to save logs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    # Logging
    parser.add_argument('--wandb_project', type=str, default='4m-mcot',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--disable_wandb', action='store_true',
                        help='Disable wandb logging')

    return parser


def setup_modality_info(args):
    """Setup modality information for MCoT training."""
    # Use standard 4M modality info but add MCoT-specific modalities if needed
    modality_info = MODALITY_INFO.copy()
    
    # Add step-specific modalities if needed
    mcot_steps = args.mcot_steps.split(',')
    for step in mcot_steps:
        step = step.strip()
        if f'{step}_text' not in modality_info:
            # Add step-specific text modality
            modality_info[f'{step}_text'] = modality_info['caption'].copy()
            modality_info[f'{step}_text']['name'] = f'{step}_text'
    
    return modality_info


def setup_mcot_data(args):
    """Setup MCoT-specific data loaders for the wget dataset format."""
    print("Setting up MCoT data loaders...")
    
    # Determine data path - can be JSON file or directory from wget script
    data_path = args.data_path
    if not data_path:
        # Try common locations
        json_path = os.path.join(args.dataset_dirs[0], "mcot_training_dataset.json")
        dir_path = args.dataset_dirs[0]
        
        if os.path.exists(json_path):
            data_path = json_path
        elif os.path.exists(dir_path):
            data_path = dir_path
        else:
            raise FileNotFoundError(f"No MCoT data found at {args.dataset_dirs[0]}")
    
    print(f"Using MCoT data from: {data_path}")
    
    # Setup modality info
    modality_info = setup_modality_info(args)
    
    # Create MCoT dataset based on data format
    if data_path.endswith('.json'):
        # JSON format
        import json
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        from mcot_data.mcot_torch_dataset import MCoTDataset
        mcot_dataset = MCoTDataset(
            data=data,
            modality_info=modality_info,
            input_size=args.input_size,
            num_input_tokens=args.num_input_tokens,
            num_target_tokens=args.num_target_tokens
        )
    else:
        # Directory format from wget script
        from mcot_data.mcot_torch_dataset import MCoTDatasetFromWgetOutput
        mcot_dataset = MCoTDatasetFromWgetOutput(
            dataset_dir=data_path,
            modality_info=modality_info,
            input_size=args.input_size,
            num_input_tokens=args.num_input_tokens,
            num_target_tokens=args.num_target_tokens,
            split='train'
        )
        
        # Create validation dataset
        val_dataset = MCoTDatasetFromWgetOutput(
            dataset_dir=data_path,
            modality_info=modality_info,
            input_size=args.input_size,
            num_input_tokens=args.num_input_tokens,
            num_target_tokens=args.num_target_tokens,
            split='val'
        )
    
    # Split dataset if using JSON format
    if data_path.endswith('.json'):
        train_size = int(0.9 * len(mcot_dataset))
        val_size = len(mcot_dataset) - train_size
        
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(mcot_dataset, [train_size, val_size])
    else:
        train_dataset = mcot_dataset
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=train_dataset.collate_fn if hasattr(train_dataset, 'collate_fn') else train_dataset.dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=val_dataset.collate_fn if hasattr(val_dataset, 'collate_fn') else val_dataset.dataset.collate_fn
    )
    
    return train_loader, val_loader, modality_info


def get_mcot_model(args, modality_info):
    """Create 4M model with MCoT capabilities."""
    print(f"Creating model: {args.model}")
    
    # Create base 4M model
    base_model = create_model(
        args.model,
        modality_info=modality_info,
        input_size=args.input_size,
        patch_size=args.patch_size,
        num_register_tokens=args.num_register_tokens
    )
    
    # Add MCoT wrapper
    model = add_mcot_to_model(base_model)
    
    # Load from checkpoint if specified
    if args.finetune:
        print(f"Loading checkpoint from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'base_model.' prefix if present for loading into MCoT wrapper
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('base_model.'):
                new_key = key[11:]  # Remove 'base_model.' prefix
                new_state_dict[f'base_model.{new_key}'] = value
            else:
                new_state_dict[f'base_model.{key}'] = value
        
        # Load state dict with strict=False to allow for new MCoT parameters
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint with message: {msg}")
    
    return model


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
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', "{:.8f}")
    metric_logger.add_meter('min_lr', "{:.8f}")
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
            
            # Setup context (empty for first step)
            mcot_context = {}
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                # Forward pass for this step
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

    metric_logger = MetricLogger(delimiter="  ")
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
    init_distributed_mode(args)

    print(f"Job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Setup data
    train_loader, val_loader, modality_info = setup_mcot_data(args)
    
    # Setup model
    model = get_mcot_model(args, modality_info)
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
    if get_world_size() > 1:
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
    num_training_steps_per_epoch = len(train_loader) // args.accum_iter
    lr_schedule_values = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    wd_schedule_values = cosine_scheduler(
        args.weight_decay, args.weight_decay, args.epochs, num_training_steps_per_epoch
    )

    # Setup loss scaler for mixed precision
    loss_scaler = NativeScalerWithGradNormCount() if args.dtype in ['fp16', 'bf16'] else None

    # Setup logging
    log_writer = None
    if HAS_WANDB and not args.disable_wandb and is_main_process():
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
            model, train_loader, optimizer, device, epoch, loss_scaler,
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
        if val_loader is not None:
            val_stats = evaluate(model, val_loader, device, args)
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
    args = get_args().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
