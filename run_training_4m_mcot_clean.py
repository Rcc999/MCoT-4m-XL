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
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Optional, Dict

# Core dependencies - these are required for 4M MCoT training
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml
from tokenizers import Tokenizer

# Import remaining dependencies
import fourm.utils as utils
from fourm.data import (build_mixture_dataloader, setup_sampling_mod_info)
from fourm.data.pretrain_utils import get_train_dataloader, get_val_dataloader
from fourm.data.modality_info import MODALITY_INFO
from fourm.models import fm
from fourm.models.mcot_fixed import MCoTWrapper, add_mcot_to_model
from fourm.data.modality_info import MODALITY_INFO
from fourm.utils import create_model, load_safetensors
from fourm.utils.optim_factory import create_optimizer

# Import mcot_utils for MCoT training functionality  
from mcot_data import mcot_utils

# Import wandb for experiment tracking (optional)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def get_args_parser():
    parser = argparse.ArgumentParser('4M MCoT training', add_help=False)
    parser.add_argument('--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Per-device batch size')
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--model', default='fm_base_12e_12d_swiglu_nobias', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='Image input size for vision backbone')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch size')
    parser.add_argument('--finetune', default='',
                        help='Path to finetune from checkpoint (for two-stage training)')
    parser.add_argument('--num_register_tokens', default=0, type=int,
                        help='Number of learnable register tokens to add to encoder')
    parser.add_argument('--device', default='cuda',
                        help='Device to use')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32', 'bf16', 'fp16', 'fp32'],
                        help='Mixed precision training data type')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Start epoch')
    parser.add_argument('--output_dir', default='./output/mcot',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logs/mcot',
                        help='Path to save logs')

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

    # Dataset settings
    parser.add_argument('--data_config', default='', type=str,
                        help='Path to data config yaml')
    parser.add_argument('--tokenizer_path', default='fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json',
                        help='Path to tokenizer json file')
    parser.add_argument('--num_input_tokens', type=int, default=512,
                        help="Token budget for the input")
    parser.add_argument('--num_target_tokens', type=int, default=256,
                        help="Token budget for the target")
    parser.add_argument('--min_input_tokens', type=int, default=None,
                        help="Minimum token budget for the input (None to set it to num_input_tokens)")
    parser.add_argument('--min_target_tokens', type=int, default=None,
                        help="Minimum token budget for the target (None to set it to num_target_tokens)")
    parser.add_argument('--loss_type', type=str, choices=['mod', 'token'], default='mod',
                        help="If mod, loss is the mean of the per-modality loss. If token, loss is the mean of the per-token loss")
    parser.add_argument('--epoch_size', default=100_000, type=int,
                        help='Number of iters per epoch. -1 for the size of the data loader')
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='Evaluation frequency in epochs')
    parser.add_argument('--save_ckpt_freq', default=2, type=int,
                        help='Checkpoint saving frequency in epochs')

    # Optimizer settings
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon')
    parser.add_argument('--opt_betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas')
    parser.add_argument('--clip_grad', default=None, type=float,
                        help='Clip gradient norm')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay')

    # Learning rate schedule settings
    parser.add_argument('--blr', type=float, default=2e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * batch_size / 256')
    parser.add_argument('--min_blr', type=float, default=0., metavar='MIN_LR',
                        help='Minimum base learning rate during cosine decay')
    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='Epochs to warmup LR, if scheduler supports')

    # Augmentation settings
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor')
    parser.add_argument('--use_act_checkpoint', action='store_true',
                        help='Use activation checkpointing')
    parser.set_defaults(use_act_checkpoint=True)  # Default to True for MCoT due to increased complexity

    # Model freezing
    parser.add_argument('--freeze_epochs', type=int, default=0,
                        help='Number of epochs to freeze shared parameters')

    # Wandb logging
    parser.add_argument('--log_wandb', action='store_true',
                        help='log training and validation metrics to wandb')
    parser.set_defaults(log_wandb=False)
    parser.add_argument('--wandb_project', default=None, type=str,
                        help='wandb project name')
    parser.add_argument('--wandb_entity', default=None, type=str,
                        help='wandb entity name')
    parser.add_argument('--wandb_run_name', default=None, type=str,
                        help='wandb run name')
    parser.add_argument('--eval_freq', default=1, type=int,
                        help='Frequency of evaluation (in epochs)')
    parser.add_argument('--save_ckpt_freq', default=1, type=int,
                        help='Frequency of checkpoint saving (in epochs)')
    parser.add_argument('--clip_grad', default=None, type=float,
                        help='Gradient clipping norm')
    
    # Additional training arguments
    parser.add_argument('--num_tasks', default=1, type=int,
                        help='Number of training tasks/processes')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--epoch_size', default=None, type=int,
                        help='Number of samples per epoch')
    
    return parser


def compute_mcot_step_loss(outputs, targets, step, criterion):
    """
    Compute step-specific loss for MCoT training.
    
    Args:
        outputs: Model outputs for the step
        targets: Target values for the step  
        step: MCoT step name (planning, acting, reflection, correction)
        criterion: Loss function
        
    Returns:
        Step-specific loss value
    """
    if step == "planning":
        # Planning step focuses on text generation (captions and layout descriptions)
        if isinstance(outputs, dict) and "text" in outputs and isinstance(targets, dict) and "text" in targets:
            text_loss = criterion(outputs["text"], targets["text"])
            return text_loss
    
    elif step == "acting":
        # Acting step focuses on image generation
        if isinstance(outputs, dict) and "image" in outputs and isinstance(targets, dict) and "image" in targets:
            image_loss = criterion(outputs["image"], targets["image"])
            return image_loss
    
    elif step == "reflection":
        # Reflection step focuses on quality assessment and artifact detection
        reflection_loss = 0.0
        if isinstance(outputs, dict) and isinstance(targets, dict):
            if "quality_score" in outputs and "quality_score" in targets:
                quality_loss = criterion(outputs["quality_score"], targets["quality_score"])
                reflection_loss += quality_loss
            
            if "artifacts" in outputs and "artifacts" in targets:
                artifact_loss = criterion(outputs["artifacts"], targets["artifacts"])
                reflection_loss += artifact_loss
                
        return reflection_loss
    
    elif step == "correction":
        # Correction step focuses on improved image generation
        if isinstance(outputs, dict) and "corrected_image" in outputs and isinstance(targets, dict) and "target_image" in targets:
            correction_loss = criterion(outputs["corrected_image"], targets["target_image"])
            return correction_loss
    
    # Default: use standard loss computation
    return criterion(outputs, targets)


def compute_mcot_total_loss(step_losses, step_weights=None):
    """
    Compute total MCoT loss by combining step-specific losses.
    
    Args:
        step_losses: Dictionary of losses for each step
        step_weights: Optional weights for each step
        
    Returns:
        Total weighted loss
    """
    if step_weights is None:
        # Default equal weighting
        step_weights = {step: 1.0 for step in ["planning", "acting", "reflection", "correction"]}
    
    total_loss = 0.0
    for step, loss in step_losses.items():
        if step in step_weights and loss is not None:
            total_loss += step_weights[step] * loss
    
    return total_loss


def train_one_epoch(model, data_loader, optimizer,
                    num_input_tokens, num_target_tokens, loss_type, device, epoch, 
                    frozen_model_epochs, accum_iter, max_norm=None, log_writer=None,
                    lr_scheduler=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    all_domains=None, dtype=None, loader_len=None,
                    output_dir=None, total_batch_size=None, mcot_steps=None, 
                    mcot_step_weights=None):
    """
    Train for one epoch with MCoT methodology following MINT paper and 4M FSDP integration.
    
    Implements the four-step MCoT process:
    1. Planning: Dense caption and layout planning
    2. Acting: Image generation based on planning  
    3. Reflection: Artifact detection and quality assessment
    4. Correction: Targeted inpainting and correction
    """
    model.train()
    if frozen_model_epochs > 0 and epoch < frozen_model_epochs:
        if hasattr(model, 'freeze_shared_params'):
            model.freeze_shared_params()
    else:
        if hasattr(model, 'unfreeze_all'):
            model.unfreeze_all()
        
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # Default values for parameters
    if all_domains is None:
        all_domains = []
    if dtype is None:
        dtype = torch.float16
    if mcot_steps is None:
        mcot_steps = ["planning", "acting", "reflection", "correction"]
    if mcot_step_weights is None:
        mcot_step_weights = {step: 1.0 for step in mcot_steps}
    
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

        # Create modality dictionary following 4M patterns
        mod_dict = {
            modality: {k: v.to(device, non_blocking=True) for k, v in d.items()}
            for modality, d in x.items()
            if modality in all_domains
        }

        # Use FSDP-compatible training loop with MCoT step processing
        step_losses = {}
        mcot_context = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # Only sync if we update grad (for accum_iter) - FSDP pattern
        with model.no_sync() if not update_grad and hasattr(model, 'no_sync') else nullcontext():
            
            # Process each MCoT step sequentially following MINT methodology
            for mcot_step in mcot_steps:
                with torch.cuda.amp.autocast(dtype=dtype, enabled=dtype != torch.float32):
                    # Forward pass through MCoT model
                    outputs = model(
                        mod_dict=mod_dict,
                        num_encoder_tokens=num_input_tokens,
                        num_decoder_tokens=num_target_tokens,
                        mcot_step=mcot_step,
                        mcot_context=mcot_context
                    )
                    
                    # Extract loss from outputs
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        # Base model returns (loss, mod_loss) during training
                        step_loss, step_mod_loss = outputs
                    else:
                        # Fallback: compute loss using step-specific targets
                        step_targets = _extract_step_targets(x, mcot_step)
                        step_loss = _compute_mcot_step_loss(outputs, step_targets, mcot_step)
                    
                    # Update MCoT context with step outputs for next step
                    if hasattr(model, 'step_processor'):
                        parsed_output = model.step_processor.parse_step_output(
                            _decode_model_output(outputs), mcot_step
                        )
                        mcot_context.update(parsed_output)
                    mcot_context[f"{mcot_step}_completed"] = True
                    
                    # Weight the step loss
                    step_weight = mcot_step_weights.get(mcot_step, 1.0)
                    weighted_loss = step_loss * step_weight
                    step_losses[mcot_step] = step_loss
                    total_loss = total_loss + weighted_loss
            
            # Average the total loss across steps and accumulation
            total_loss = total_loss / len(mcot_steps) / accum_iter
            
            # Check for finite loss
            loss_value = total_loss.item()
            step_loss_values = {f'{step}_loss': l.item() for step, l in step_losses.items()}

            if not math.isfinite(loss_value):
                if output_dir:
                    torch.save(mod_dict, os.path.join(output_dir, "debug_mod_dict.pt"))
                print(f"Loss is {loss_value}, stopping training", file=sys.stderr)
                sys.exit(1)

            # Backward pass
            total_loss.backward()

        if update_grad:
            if max_norm is not None and hasattr(model, 'clip_grad_norm_'):
                grad_norm = model.clip_grad_norm_(max_norm)
            elif max_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_norm = None
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # Update metrics
        metric_logger.update(loss=loss_value)
        metric_logger.update(**step_loss_values)
        
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
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)

        if log_writer is not None:
            log_writer.update(
                {
                    'loss': loss_value,
                    'lr': max_lr,
                    'weight_decay': weight_decay_value,
                }
            )
            log_writer.update(step_loss_values)
            if grad_norm is not None:
                log_writer.update({'grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm})

            if total_batch_size is not None:
                log_writer.update(
                    {
                        'input_tokens_seen_b': it * (total_batch_size / accum_iter) * num_input_tokens / 1e9,
                        'target_tokens_seen_b': it * (total_batch_size /accum_iter) * num_target_tokens / 1e9,
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


def evaluate(model, data_loader, device, args, mcot_steps, mcot_step_weights):
    """
    Evaluate the MCoT model following MINT paper specifications.
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    
    with torch.no_grad():
        for step, x in enumerate(metric_logger.log_every(data_loader, 10, "Validation: ")):
            # Process MCoT batch according to MINT paper methodology
            mcot_batch = mcot_utils.parse_mcot_batch(x) if mcot_utils else x
            
            # Create modality dictionary following 4M patterns  
            mod_dict = {
                modality: {k: v.to(device, non_blocking=True) for k, v in d.items()}
                for modality, d in x.items()
                if modality in MODALITY_INFO
            }
            
            # MCoT evaluation loop with step-wise processing
            mcot_context = {}
            step_losses = {}
            total_loss = 0.0
            
            # Sequential MCoT processing as per MINT paper
            for mcot_step in mcot_steps:
                with torch.cuda.amp.autocast(enabled=args.dtype in ['float16', 'bfloat16']):
                    # Execute MCoT step using the MCoTWrapper implementation
                    outputs = model(
                        mod_dict=mod_dict,
                        num_encoder_tokens=args.num_input_tokens,
                        num_decoder_tokens=args.num_target_tokens,
                        mcot_step=mcot_step,
                        mcot_context=mcot_context
                    )
                    
                    # Extract loss from outputs
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        step_loss, _ = outputs
                    else:
                        step_targets = _extract_step_targets(x, mcot_step)
                        step_loss = _compute_mcot_step_loss(outputs, step_targets, mcot_step)
                    
                    # Update MCoT context for next step
                    if hasattr(model, 'step_processor'):
                        parsed_output = model.step_processor.parse_step_output(
                            _decode_model_output(outputs), mcot_step
                        )
                        mcot_context.update(parsed_output)
                    mcot_context[f"{mcot_step}_completed"] = True
                    
                    # Weight the step loss
                    step_weight = mcot_step_weights.get(mcot_step, 1.0)
                    step_losses[mcot_step] = step_loss
                    total_loss += step_loss * step_weight
            
            # Average loss across steps
            total_loss = total_loss / len(mcot_steps)
            
            # Update metrics
            loss_value = total_loss.item()
            if math.isfinite(loss_value):
                metric_logger.update(loss=loss_value)
                
                # Log individual step losses
                for step_name, step_loss in step_losses.items():
                    metric_logger.update(**{f"{step_name}_loss": step_loss.item()})
    
    # Return metrics
    metric_logger.synchronize_between_processes()
    print("Validation:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def setup_mcot_data_loaders(args, tokenizer):
    """
    Setup MCoT-specific data loaders for training and validation.
    
    Args:
        args: Training arguments
        tokenizer: Text tokenizer
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load dataset config
    with open(args.data_config, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Setup modality info for MCoT training
    all_domains = list(MODALITY_INFO.keys())
    
    # Create train data loader
    train_loader = get_train_dataloader(
        cfg=data_cfg,
        modality_info=MODALITY_INFO,
        tokenizer=tokenizer,
        epoch_size=args.epoch_size,
        batch_size=args.batch_size,
        num_workers=getattr(args, 'num_workers', 4),
        min_input_tokens=args.min_input_tokens,
        min_target_tokens=args.min_target_tokens,
        num_input_tokens=args.num_input_tokens,
        num_target_tokens=args.num_target_tokens,
        loss_type=args.loss_type
    )
    
    # Create validation data loader
    val_loader = get_val_dataloader(
        cfg=data_cfg,
        modality_info=MODALITY_INFO,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=getattr(args, 'num_workers', 4),
        min_input_tokens=args.min_input_tokens,
        min_target_tokens=args.min_target_tokens,
        num_input_tokens=args.num_input_tokens,
        num_target_tokens=args.num_target_tokens,
        loss_type=args.loss_type
    )
    
    return train_loader, val_loader


def main(args):
    # Set up CUDA device
    device = torch.device(args.device)
    cudnn.benchmark = True
    
    # Fix random seeds for reproducibility
    if args.seed != -1:
        utils.fix_random_seeds(args.seed)

    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Log run information
    print(f"Job directory: {os.path.dirname(os.path.realpath(__file__))}")
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # Parse MCoT steps and weights from args
    mcot_steps = args.mcot_steps.split(",") if args.mcot_steps else ["planning", "acting", "reflection", "correction"]
    mcot_step_weights = {
        "planning": args.mcot_planning_weight,
        "acting": args.mcot_acting_weight,
        "reflection": args.mcot_reflection_weight,
        "correction": args.mcot_correction_weight
    }

    print(f"Using MCoT steps: {mcot_steps}")
    print(f"MCoT step weights: {mcot_step_weights}")

    # Load and process dataset config
    with open(args.data_config, 'r') as f:
        data_cfg = yaml.safe_load(f)
        
    # Override MCoT steps and weights from config if provided
    if 'extras' in data_cfg and 'mcot_steps' in data_cfg['extras']:
        mcot_steps = data_cfg['extras']['mcot_steps']
    if 'extras' in data_cfg and 'mcot_planning_weight' in data_cfg['extras']:
        mcot_step_weights["planning"] = data_cfg['extras']['mcot_planning_weight']
    if 'extras' in data_cfg and 'mcot_acting_weight' in data_cfg['extras']:
        mcot_step_weights["acting"] = data_cfg['extras']['mcot_acting_weight']
    if 'extras' in data_cfg and 'mcot_reflection_weight' in data_cfg['extras']:
        mcot_step_weights["reflection"] = data_cfg['extras']['mcot_reflection_weight']
    if 'extras' in data_cfg and 'mcot_correction_weight' in data_cfg['extras']:
        mcot_step_weights["correction"] = data_cfg['extras']['mcot_correction_weight']

    # Load tokenizer
    tokenizer = None
    if os.path.exists(args.tokenizer_path):
        tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # Create model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model, 
        encoder_embeddings=None, 
        decoder_embeddings=None, 
        use_act_checkpoint=args.use_act_checkpoint, 
        num_register_tokens=args.num_register_tokens
    )
    
    # Add MCoT capabilities to the model
    print("Adding MCoT capabilities to model")
    model = add_mcot_to_model(model)
    
    # Move model to device
    model.to(device)

    # Load finetune checkpoint if specified
    if args.finetune:
        print(f"Loading checkpoint from: {args.finetune}")
        if args.finetune.endswith('.safetensors'):
            checkpoint = load_safetensors(args.finetune)
            utils.load_checkpoint_to_model(model, checkpoint)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')
            utils.load_checkpoint_to_model(model, checkpoint['model'])
            print("Model checkpoint loaded.")

    # Create optimizer
    param_groups = utils.get_parameter_groups(model, weight_decay=args.weight_decay)
    optimizer = create_optimizer(args, param_groups)
    
    # Setup loss scaler for mixed precision
    loss_scaler = utils.NativeScalerWithGradNormCount()

    # Attempt to resume from checkpoint if requested
    args.resume_from_checkpoint = None
    if args.auto_resume and os.path.exists(args.output_dir):
        args.resume_from_checkpoint = utils.get_last_checkpoint_for_resume(args.output_dir)
    
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
        utils.load_checkpoint_to_model(model, checkpoint['model'])
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print(f"Resumed from epoch {args.start_epoch}")

    # Create data loaders
    train_loader, val_loader = setup_mcot_data_loaders(args, tokenizer)
    
    print(f"Effective batch size: {args.batch_size * args.accum_iter}")
    
    # Set up learning rate scheduler
    if args.min_input_tokens is None:
        args.min_input_tokens = args.num_input_tokens
    if args.min_target_tokens is None:
        args.min_target_tokens = args.num_target_tokens
    
    # Learning rate calculation
    base_lr = args.blr * args.batch_size * args.accum_iter / 256
    print(f"Base learning rate: {base_lr:.8f}")
    
    # Setup wandb logging
    if args.log_wandb:
        if HAS_WANDB:
            wandb_run_name = args.wandb_run_name if args.wandb_run_name else f"mcot-train-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=wandb_run_name, config=args)
        else:
            print("Warning: wandb not available, skipping wandb logging")

    # Start the training loop
    print(f"Starting MCoT training for {args.epochs} epochs")
    start_time = time.time()
    
    # Main training loop
    for epoch in range(args.start_epoch, args.epochs):
        # Training for one epoch
        train_stats = train_one_epoch(
            model, train_loader, optimizer, 
            args.num_input_tokens, args.num_target_tokens, args.loss_type,
            device, epoch, args.freeze_epochs, args.accum_iter, 
            max_norm=args.clip_grad, log_writer=None, lr_scheduler=None,
            start_steps=epoch * len(train_loader), all_domains=list(MODALITY_INFO.keys()),
            dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16,
            loader_len=len(train_loader), output_dir=args.output_dir,
            total_batch_size=args.batch_size * args.accum_iter,
            mcot_steps=mcot_steps, mcot_step_weights=mcot_step_weights
        )
        
        # Save checkpoint
        if args.output_dir and (epoch + 1) % args.save_ckpt_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{epoch+1}.pth')
            print(f"Saving checkpoint to {checkpoint_path}")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'scaler': loss_scaler.state_dict(),
            }, checkpoint_path)
        
        # Evaluation
        if (epoch + 1) % args.eval_freq == 0:
            val_stats = evaluate(model, val_loader, device, args, mcot_steps, mcot_step_weights)
            print(f"Validation loss: {val_stats['loss']:.4f}")
            
            # Log to wandb
            if args.log_wandb and HAS_WANDB:
                log_stats = {**{f"train_{k}": v for k, v in train_stats.items()},
                             **{f"val_{k}": v for k, v in val_stats.items()},
                             "epoch": epoch}
                wandb.log(log_stats)
    
    # Save final model
    if args.output_dir:
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint-final.pth')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': args.epochs,
            'args': args,
            'scaler': loss_scaler.state_dict(),
        }, checkpoint_path)
        
        # Also save as safetensors
        print("Saving final model as safetensors")
        safetensors_path = os.path.join(args.output_dir, 'model-final.safetensors')
        model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        utils.save_safetensors(model_state_dict, safetensors_path)

    # Training completed
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'MCoT training time: {total_time_str}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser('4M MCoT training script', parents=[get_args_parser()])
    args = args_parser.parse_args()
    
    # Load from config file if specified
    if args.config:
        with open(args.config, 'r') as f:
            config_args = yaml.safe_load(f)
            # Update args with config values
            for k, v in config_args.items():
                if v is not None:  # Skip None values
                    setattr(args, k, v)
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    # Start training
    main(args)

def _extract_step_targets(sample_data, mcot_step):
    """
    Extract step-specific targets from the training sample.
    
    Args:
        sample_data: Training sample dictionary
        mcot_step: Current MCoT step
        
    Returns:
        Targets appropriate for the MCoT step
    """
    targets = {}
    
    if mcot_step == "planning":
        # Planning targets: dense caption and layout
        if 'planning' in sample_data:
            targets['text'] = sample_data['planning']
        elif 'caption' in sample_data and 'tensor' in sample_data['caption']:
            targets['text'] = sample_data['caption']['tensor']
            
    elif mcot_step == "acting":
        # Acting targets: generated image
        if 'rgb' in sample_data and 'tensor' in sample_data['rgb']:
            targets['image'] = sample_data['rgb']['tensor']
        elif 'acting' in sample_data:
            targets['image'] = sample_data['acting']
            
    elif mcot_step == "reflection":
        # Reflection targets: quality assessment
        if 'reflection' in sample_data:
            targets['quality_assessment'] = sample_data['reflection']
        elif 'caption' in sample_data and 'tensor' in sample_data['caption']:
            targets['quality_assessment'] = sample_data['caption']['tensor']
            
    elif mcot_step == "correction":
        # Correction targets: corrected image
        if 'correction' in sample_data:
            targets['corrected_image'] = sample_data['correction']
        elif 'rgb' in sample_data and 'tensor' in sample_data['rgb']:
            targets['corrected_image'] = sample_data['rgb']['tensor']
    
    return targets


def _compute_mcot_step_loss(outputs, targets, mcot_step):
    """
    Compute MCoT step-specific loss.
    
    Args:
        outputs: Model outputs
        targets: Target values  
        mcot_step: MCoT step name
        
    Returns:
        Computed loss tensor
    """
    if isinstance(outputs, dict) and 'loss' in outputs:
        return outputs['loss']
    elif isinstance(outputs, torch.Tensor):
        # For text generation steps, use cross-entropy loss
        if mcot_step in ["planning", "reflection"]:
            if 'text' in targets:
                return F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets['text'].view(-1))
        # For image generation steps, use MSE loss  
        elif mcot_step in ["acting", "correction"]:
            if 'image' in targets:
                return F.mse_loss(outputs, targets['image'])
                
    # Default: return zero loss if no suitable computation found
    return torch.tensor(0.0, device=outputs.device if isinstance(outputs, torch.Tensor) else 'cpu')


def _decode_model_output(outputs):
    """
    Decode model outputs to text for MCoT step processing.
    
    Args:
        outputs: Model outputs
        
    Returns:
        Decoded text string
    """
    if isinstance(outputs, dict):
        # Look for text output in various keys
        for key in ['text', 'caption', 'logits', 'prediction']:
            if key in outputs:
                return str(outputs[key])
        return "Generated output"
    elif isinstance(outputs, torch.Tensor):
        # Convert tensor to meaningful text using 4M tokenizer integration
        if outputs.dim() == 2 and outputs.shape[-1] > 1000:  # Likely token predictions
            try:
                # Get the most likely tokens
                predicted_tokens = torch.argmax(outputs, dim=-1)
                
                # Try to decode using available tokenizer
                if hasattr(model, 'decoder') and hasattr(model.decoder, 'tokenizer'):
                    decoded_text = model.decoder.tokenizer.decode(predicted_tokens[0].cpu().numpy())
                    return decoded_text
                elif hasattr(model, 'text_tokenizer'):
                    decoded_text = model.text_tokenizer.decode(predicted_tokens[0].cpu().numpy())
                    return decoded_text
                else:
                    # Fallback to token representation
                    return f"Predicted tokens: {predicted_tokens[0][:10].tolist()}..." if predicted_tokens.shape[-1] > 10 else f"Tokens: {predicted_tokens[0].tolist()}"
            except Exception as e:
                return f"Tensor output (decode error: {str(e)}) shape: {outputs.shape}"
        else:
            return f"Tensor output shape: {outputs.shape}, sample values: {outputs.flatten()[:5].tolist()}"
    else:
        return str(outputs)
