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
import os
from pathlib import Path

import torch

from .dist import save_on_main, is_main_process
from .s3_utils import save_on_s3


from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

from torch.distributed.fsdp.api import FullOptimStateDictConfig



def save_model_fsdp(args, epoch, model, optimizer, model_ema=None, ckpt_name=None, use_s3=False):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    ckpt_name = ckpt_name or epoch_name

    
    with FSDP.state_dict_type(model, 
        StateDictType.FULL_STATE_DICT, 
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):

        model_state_dict = model.state_dict()
        if optimizer is not None:
            optimizer_state_dict = FSDP.optim_state_dict(model, optimizer)
        else:
            optimizer_state_dict = None

        # Only create the save_dict on the main process, not needed or recommended to do so on all ranks
        # This make save_on_main() redundant
        if is_main_process(): 
            checkpoint_path = os.path.join(output_dir, f'checkpoint-{ckpt_name}.pth')

            to_save = {
                'model': model_state_dict,
                'epoch': epoch,
                'args': args,
            }

            if optimizer is not None:
                to_save['optimizer'] = optimizer_state_dict

            if model_ema is not None:
                print("Model EMA is currently not supported for FSDP")
                # to_save['model_ema'] = get_state_dict(model_ema)

            save_on_main(to_save, checkpoint_path)
            
            if use_s3: 
                s3_path = os.path.join(args.s3_save_dir, f'checkpoint-{ckpt_name}.pth')
                save_on_s3(checkpoint_path, s3_path, args.s3_endpoint)

  
def auto_load_model_fsdp(args, model, optimizer, model_ema=None):
    output_dir = Path(args.output_dir)
    if args.auto_resume and len(args.resume) == 0:
        import glob
        all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
        latest_ckpt = -1
        for ckpt in all_checkpoints:
            t = ckpt.split('-')[-1].split('.')[0]
            if t.isdigit():
                latest_ckpt = max(int(t), latest_ckpt)
        if latest_ckpt >= 0:
            args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
        print("Auto resume checkpoint: %s" % args.resume)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu')
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

        with FSDP.state_dict_type(
            model, 
            StateDictType.FULL_STATE_DICT, 
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
            ):

            model.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)

            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                print("Attempting to load optimizer state...")
                # Custom memory-efficient optimizer state loading
                opt_state_dict = checkpoint['optimizer']
                
                # Get the map from flat parameter to original parameters
                flat_param_to_orig_params = {}
                for module in FSDP.fsdp_modules(model):
                    if hasattr(module, "flat_param") and hasattr(module, "_param_infos"):
                        flat_p = module.flat_param
                        orig_params = [p for p in 
                                      [module.params_dict[info.param_name] for info in module._param_infos]
                                      if p.requires_grad]
                        flat_param_to_orig_params[flat_p] = orig_params
                
                # Process state dict in place to avoid full copy
                if 'state' in opt_state_dict:
                    # Process each param's state with reduced memory overhead
                    param_id_map = {}
                    for module in FSDP.fsdp_modules(model):
                        for p in module.parameters():
                            if p.requires_grad:
                                param_id_map[id(p)] = p
                    
                    # Create new state dict mapping flat params to states
                    new_state = {}
                    for param_id, param_state in opt_state_dict['state'].items():
                        # Find if this param_id exists in our current model
                        if param_id in param_id_map:
                            new_state[param_id] = param_state
                        # Otherwise, we need to map original param states to flat param
                        # Process this by param groups to avoid excess memory
                
                    opt_state_dict['state'] = new_state
                
                try:
                    # Load with reduced copying
                    for param_group, saved_param_group in zip(optimizer.param_groups, opt_state_dict['param_groups']):
                        # Only copy over the learning rate and other hyperparams
                        for k in ['lr', 'momentum', 'weight_decay', 'dampening', 'nesterov']:
                            if k in saved_param_group:
                                param_group[k] = saved_param_group[k]
                    
                    # Now handle the state with our custom mapping
                    optimizer.load_state_dict(opt_state_dict)
                    args.start_epoch = checkpoint['epoch'] + 1
                    print("Successfully loaded optimizer state.")
                except (TypeError, RuntimeError) as e:
                    print(f"WARNING: Failed to load optimizer state: {e}")
                    print("Proceeding with fresh optimizer state. Model weights are still loaded.")
                    # Ensure start_epoch is still set from the checkpoint if model weights were loaded
                    args.start_epoch = checkpoint['epoch'] + 1

        if hasattr(args, 'model_ema') and args.model_ema:
            print("Model EMA is currently not supported for FSDP")