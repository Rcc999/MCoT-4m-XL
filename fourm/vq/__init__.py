import sys, os
import torch
import argparse

from .vqvae import VQ, VQVAE, DiVAE, VQControlNet
from .scheduling import *
from ..utils import load_safetensors


def get_image_tokenizer(tokenizer_id: str, 
                        tokenizers_root: str = './tokenizer_ckpts', 
                        encoder_only: bool = False, 
                        device: str = 'cuda', 
                        verbose: bool = True,
                        return_None_on_fail: bool = False,):
    """
    Load a pretrained image tokenizer from a checkpoint.
    Now attempts to load .safetensors first, then falls back to .pth for compatibility.

    Args:
        tokenizer_id (str): ID of the tokenizer to load (name of the checkpoint file without extension).
        tokenizers_root (str): Path to the directory containing the tokenizer checkpoints.
        encoder_only (bool): Set to True to load only the encoder part of the tokenizer.
        device (str): Device to load the tokenizer on.
        verbose (bool): Set to True to print load_state_dict warning/success messages
        return_None_on_fail (bool): Set to True to return None if the tokenizer fails to load

    Returns:
        model (nn.Module): The loaded tokenizer.
        config (argparse.Namespace): The tokenizer configuration.
    """
    
    safetensors_file_name = f'{tokenizer_id}.safetensors'
    safetensors_path = os.path.join(tokenizers_root, safetensors_file_name)
    
    pth_file_name = f'{tokenizer_id}.pth'
    pth_path = os.path.join(tokenizers_root, pth_file_name)

    loaded_from_safetensors = False
    config_obj = None
    model_state_dict = None

    if os.path.exists(safetensors_path):
        if verbose:
            print(f'Loading tokenizer {tokenizer_id} from {safetensors_path} ... ', end='')
        try:
            model_state_dict, config_dict = load_safetensors(safetensors_path)
            # Convert config_dict to a Namespace or similar object for attribute access
            # The original script uses ckpt['args'] which is often an argparse.Namespace
            config_obj = argparse.Namespace(**config_dict)
            loaded_from_safetensors = True
            if verbose:
                print('success (safetensors).')
        except Exception as e:
            if verbose:
                print(f'failed to load .safetensors: {e}. Attempting .pth.')
            if return_None_on_fail and not os.path.exists(pth_path):
                return None, None
    
    if not loaded_from_safetensors:
        if os.path.exists(pth_path):
            if verbose:
                print(f'Loading tokenizer {tokenizer_id} from {pth_path} ... ', end='')
            try:
                ckpt = torch.load(pth_path, map_location='cpu')
                model_state_dict = ckpt['model']
                config_obj = ckpt['args'] # This is already an argparse.Namespace or similar
                if verbose:
                    print('success (pth).')
            except Exception as e:
                if verbose:
                    print(f'failed to load .pth: {e}.')
                if return_None_on_fail:
                    return None, None
                else:
                    raise
        elif return_None_on_fail: # safetensors failed or didn't exist, and pth doesn't exist
            return None, None
        else: # Neither file exists and not returning None on fail
            raise FileNotFoundError(f"Neither {safetensors_path} nor {pth_path} found.")

    if config_obj is None or model_state_dict is None: # Should not happen if logic above is correct
        if return_None_on_fail:
            return None, None
        else:
            raise ValueError("Failed to load checkpoint or config.")

    # --- Model Type Determination and Instantiation ---
    # For safetensors, config_obj comes from the file's metadata.
    # For .pth, it's ckpt['args'].
    
    # Handle renamed arguments (original logic, adapt as needed for safetensors config)
    # This assumes config_obj has attributes like 'domain', 'quantizer_type' etc.
    if hasattr(config_obj, 'domain'):
        if 'CLIP' in config_obj.domain or 'DINO' in config_obj.domain or 'ImageBind' in config_obj.domain:
            config_obj.patch_proj = False
        elif 'sam' in config_obj.domain and hasattr(config_obj, 'mask_size'):
            config_obj.input_size_min = config_obj.mask_size
            config_obj.input_size_max = config_obj.mask_size
            config_obj.input_size = config_obj.mask_size
    
    config_obj.quant_type = getattr(config_obj, 'quantizer_type', getattr(config_obj, 'quant_type', None))
    config_obj.enc_type = getattr(config_obj, 'encoder_type', getattr(config_obj, 'enc_type', None))
    config_obj.dec_type = getattr(config_obj, 'decoder_type', getattr(config_obj, 'dec_type', None))
    
    # Determine image_size based on available attributes
    if hasattr(config_obj, 'input_size'):
        config_obj.image_size = config_obj.input_size
    elif hasattr(config_obj, 'input_size_max'):
        config_obj.image_size = config_obj.input_size_max
    elif not hasattr(config_obj, 'image_size'): # if still not set
        config_obj.image_size = 224 # Default or raise error

    config_obj.image_size_enc = getattr(config_obj, 'image_size_enc', None)
    config_obj.image_size_dec = getattr(config_obj, 'image_size_dec', None)
    config_obj.image_size_sd = getattr(config_obj, 'image_size_sd', None) # For DiVAE
    config_obj.ema_decay = getattr(config_obj, 'quantizer_ema_decay', getattr(config_obj, 'ema_decay', None))
    config_obj.enable_xformer = getattr(config_obj, 'use_xformer', getattr(config_obj, 'enable_xformer', False)) # Default to False if not present
    
    # n_channels might be in config or need inference (original logic)
    if not hasattr(config_obj, 'n_channels'):
        if 'cls_emb.weight' in model_state_dict and hasattr(config_obj, 'n_labels'): # n_labels might also come from config
            config_obj.n_labels, config_obj.n_channels = model_state_dict['cls_emb.weight'].shape
        elif 'encoder.linear_in.weight' in model_state_dict:
            config_obj.n_channels = model_state_dict['encoder.linear_in.weight'].shape[1]
        elif 'encoder.proj.weight' in model_state_dict: # Fallback
             config_obj.n_channels = model_state_dict['encoder.proj.weight'].shape[1]
        else:
            # If still not found, set a default or raise error.
            # This might be critical, so ensure it's covered by safetensor config or pth.
            config_obj.n_channels = getattr(config_obj, 'n_channels', 3) # Default to 3 for RGB

    config_obj.sync_codebook = getattr(config_obj, 'sync_codebook', False) # Default if not present

    if encoder_only:
        model_type_str = 'VQ' # VQ class handles encoder only parts
        config_obj.model_type = model_type_str # Explicitly set model_type on config_obj
        # Filter model_state_dict for encoder parts
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'decoder' not in k and 'post_quant_proj' not in k}
    else:
        # Determine model_type from config_obj if possible, otherwise infer
        model_type_str = getattr(config_obj, 'model_type', None)
        if model_type_str is None: # Fallback to inference for .pth files or if not in safetensors config
            if any(['controlnet' in k for k in model_state_dict.keys()]):
                model_type_str = 'VQControlNet'
            elif hasattr(config_obj, 'beta_schedule'): # DiVAE specific
                model_type_str = 'DiVAE'
            else:
                model_type_str = 'VQVAE'
        config_obj.model_type = model_type_str # Ensure it's set on config_obj

    model_type = getattr(sys.modules[__name__], config_obj.model_type)
    
    # Convert Namespace to dict for model instantiation if it's not already (e.g. from argparse)
    if isinstance(config_obj, argparse.Namespace):
        config_for_model = vars(config_obj)
    else: # Assuming it's already a dict if not Namespace (e.g. from safetensors if not converted)
        config_for_model = config_obj

    model = model_type(**config_for_model)
    
    msg = model.load_state_dict(model_state_dict, strict=False)
    if verbose:
        if msg.missing_keys or msg.unexpected_keys:
            print(f"State dict loading messages: {msg}")
        else:
            print(f"Model state_dict loaded successfully.")

    return model.to(device).eval(), config_obj
