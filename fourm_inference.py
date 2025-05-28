#!/usr/bin/env python3
"""
VQA Inference Script for 4M-21 XL model
Implements VQA inference matching the EXACT TransferMasking training format.
"""
import torch
import numpy as np
from PIL import Image
from tokenizers import Tokenizer
from safetensors.torch import load_file
import torch.nn.functional as F
import requests
import os
import copy
from torchvision import transforms
import yaml
import json

from fourm.models.fm import FM
from fourm.models.generate import GenerationSampler, init_full_input_modality, init_empty_target_modality
from fourm.data.modality_info import MODALITY_INFO
from fourm.utils.checkpoint import load_safetensors
from fourm.utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def load_image_from_url(url):
    """Load image from URL with fallback to dummy image"""
    try:
        img = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        try:
            print("Falling back to a local dummy image.")
            if not os.path.exists("dummy_image.jpg"):
                from PIL import ImageDraw
                img_fallback = Image.new('RGB', (224, 224), color='red')
                draw = ImageDraw.Draw(img_fallback)
                draw.text((10, 10), "Dummy Image", fill=(255, 255, 0))
                img_fallback.save("dummy_image.jpg")
            img = Image.open("dummy_image.jpg").convert('RGB')
        except Exception as fallback_e:
            print(f"Error loading fallback image: {fallback_e}")
            return None
    return img

def create_vqa_modality_dict_transfer_format(question, image_tensor, text_tokenizer, device, max_answer_len=30):
    """Create VQA modality dictionary for autoregressive generation"""
    
    # Get configuration values 
    max_tokens = 256  # From MODALITY_INFO
    batch_size = image_tensor.shape[0]
    
    # For autoregressive generation, we need to set up masks correctly:
    # - input_mask: False = visible to model, True = masked  
    # - target_mask: False = target (to be generated), True = not target (input)
    
    # Tokenize question 
    question_tokens_list = text_tokenizer.encode(question).ids
    
    # Get EOS token ID and append it to the question
    eos_id = text_tokenizer.token_to_id("[EOS]")
    s2_id = text_tokenizer.token_to_id("[S_2]") # Explicitly use [S_2]

    if eos_id is None:
        print("Warning: [EOS] token not found in tokenizer.")
    if s2_id is None:
        print("ERROR: [S_2] token not found in tokenizer. This is critical for the current test.")
        # Potentially return or raise an error if s2_id is crucial and not found

    if eos_id is not None:
        question_tokens_list.append(eos_id)
    
    seq_len = len(question_tokens_list) # seq_len now includes Question + EOS

    # Ensure prompt (question + EOS) fits
    if seq_len >= max_tokens:
        print(f"Warning: Question + EOS ({seq_len} tokens) is too long for max_tokens ({max_tokens}). Truncating.")
        original_question_part = question_tokens_list
        num_special_to_append = 0 # Only EOS needs to be re-appended if truncated
        
        # Check if [EOS] was the last token and needs re-appending
        if eos_id is not None and original_question_part[-1] == eos_id:
            original_question_part = original_question_part[:-1]
            num_special_to_append = 1 # [EOS] to re-append
        
        max_question_part_len = max_tokens - num_special_to_append 
        truncated_question_part = original_question_part[:max_question_part_len]
        
        question_tokens_list = truncated_question_part
        # Re-append EOS if it was part of num_special_to_append
        if eos_id is not None and num_special_to_append == 1:
             question_tokens_list.append(eos_id)
        
        seq_len = len(question_tokens_list)

    # Initialize tensors and masks - key insight: for autoregressive generation,
    # the model starts with input tokens and generates sequentially after them
    caption_tensor = torch.zeros((batch_size, max_tokens), dtype=torch.long, device=device)
    
    # CRITICAL FIX: For autoregressive generation, we only make the INPUT TOKENS visible,
    # and ALL other positions should be marked as targets to be generated
    input_mask = torch.ones((batch_size, max_tokens), dtype=torch.bool, device=device)  # All masked initially
    target_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=device)  # All targets initially
    decoder_attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.long, device=device) # Use long for cumsum values
    
    # Place question tokens (now including EOS) at the beginning
    if seq_len > 0: # Make sure there's something to place
        caption_tensor[0, :seq_len] = torch.tensor(question_tokens_list, dtype=torch.long, device=device)
    
    # Question tokens (including EOS) are visible inputs (not targets)
    input_mask[0, :seq_len] = False  # Question tokens are visible (not masked)
    target_mask[0, :seq_len] = False # Question tokens are NOT masked as keys in self-attention.
    
    # Construct decoder_attention_mask for hybrid attention:
    # - Question part: bidirectional (first token value = seq_len, rest 0)
    # - Answer part: causal (each token value = 1)
    if seq_len > 0:
        decoder_attention_mask[0, 0] = seq_len # First question token
        # Remaining question tokens (from index 1 to seq_len-1) are already 0.
        # Update: Set remaining question tokens to 1 to allow attention
        decoder_attention_mask[0, 1:seq_len] = 1
    # Answer part (from index seq_len onwards)
    decoder_attention_mask[0, seq_len:] = 1 # For causal attention in answer part
    
    # All positions after the question are targets to be generated
    # target_mask[0, seq_len:] = False  (already set above)
    # input_mask[0, seq_len:] = True   (already set above)
    
    print(f"Question (processed for model): '{text_tokenizer.decode(question_tokens_list if seq_len > 0 else [])}'") # Decode only if list is not empty
    print(f"Question tokens (processed, incl. EOS, length {seq_len}): {question_tokens_list if seq_len > 0 else '[]'}")
    print(f"Caption tensor shape: {caption_tensor.shape}")
    if seq_len > 0:
        print(f"Question input tokens (from tensor): {caption_tensor[0, :seq_len].tolist()}")
        print(f"Input mask (False=visible): first {min(seq_len + 5, max_tokens)} positions: {input_mask[0, :min(seq_len + 5, max_tokens)].tolist()}")
        print(f"Target mask (False=target): first {min(seq_len + 5, max_tokens)} positions: {target_mask[0, :min(seq_len + 5, max_tokens)].tolist()}")
    else:
        print("No question tokens to display for tensor/masks.")
    print(f"Target mask (False=target): positions {seq_len}-{min(seq_len+10, max_tokens)}: {target_mask[0, seq_len:min(seq_len+10, max_tokens)].tolist()}")
    
    # Setup modality dictionary for autoregressive generation
    mod_dict = {
        'rgb@224': {
            'tensor': image_tensor,
        },
        'caption': {
            'tensor': caption_tensor,
            'input_mask': input_mask,
            'target_mask': target_mask,
            'decoder_attention_mask': decoder_attention_mask,
        }
    }
    
    # Initialize the image modality as full input (completely visible)
    mod_dict = init_full_input_modality(mod_dict, MODALITY_INFO, 'rgb@224', device)
    
    return mod_dict, seq_len

def load_finetune_config(yaml_path="cfgs/default/4m/finetune/4m-xl_mod21_vqav2_finetune.yaml"):
    """Load relevant parts of the finetuning YAML config."""
    try:
        with open(yaml_path, 'r') as f:
            finetune_cfg = yaml.safe_load(f)
        
        # Extract only the necessary model and tokenizer related parts
        # to avoid overriding everything from the training args.
        model_config_from_yaml = {
            "model_name": finetune_cfg.get("model"), # 'fm_xlarge_24e_24d_swiglu_nobias'
            "patch_size": finetune_cfg.get("patch_size"),
            "input_size": finetune_cfg.get("input_size"),
            "use_act_checkpoint": finetune_cfg.get("use_act_checkpoint") 
            # We'll get vocab size from MODALITY_INFO directly for caption
        }
        # Check if critical keys were found
        if not model_config_from_yaml["model_name"]:
            print(f"Warning: 'model' key not found in {yaml_path}")
        return model_config_from_yaml
    except Exception as e:
        print(f"Error loading or parsing finetuning YAML {yaml_path}: {e}")
        return {}

def main_inference():
    """Main VQA inference function"""
    model_name = 'fm_xlarge_24e_24d_swiglu_nobias'
    finetuned_safetensor_path = 'vqa.safetensors' 
    tokenizer_path = 'fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check files exist
    # For inference, we now expect the checkpoint to be a directory containing the model files
    # and potentially a config.json, or just the safetensors file with config loaded separately.
    if not os.path.exists(finetuned_safetensor_path):
        # If it's a file path, check its directory for config.json
        # If it's a directory path, check inside it for model files and config.json
        if os.path.isfile(finetuned_safetensor_path):
            checkpoint_dir = os.path.dirname(finetuned_safetensor_path)
            model_file_to_load = finetuned_safetensor_path
        else: # Assumed to be a directory
            checkpoint_dir = finetuned_safetensor_path
            # Attempt to find a common model file name if finetuned_safetensor_path is a dir
            # This logic might need to be more robust based on actual saved checkpoint structure by FSDP
            potential_model_files = ["consolidated.safetensors", "model.safetensors", "pytorch_model.bin"]
            model_file_to_load = None
            for f_name in potential_model_files:
                if os.path.exists(os.path.join(checkpoint_dir, f_name)):
                    model_file_to_load = os.path.join(checkpoint_dir, f_name)
                    break
            if model_file_to_load is None:
                print(f"ERROR: Finetuned model file (e.g., consolidated.safetensors) not found in directory {checkpoint_dir}")
                return
        print(f"Using checkpoint directory: {checkpoint_dir}")
        print(f"Using model file: {model_file_to_load}")
    else: # finetuned_safetensor_path directly exists
        if os.path.isfile(finetuned_safetensor_path):
            checkpoint_dir = os.path.dirname(finetuned_safetensor_path)
            model_file_to_load = finetuned_safetensor_path
        else: # It's a directory that exists, treat as checkpoint_dir
            checkpoint_dir = finetuned_safetensor_path
            potential_model_files = ["consolidated.safetensors", "model.safetensors", "pytorch_model.bin"]
            model_file_to_load = None
            for f_name in potential_model_files:
                if os.path.exists(os.path.join(checkpoint_dir, f_name)):
                    model_file_to_load = os.path.join(checkpoint_dir, f_name)
                    break
            if model_file_to_load is None:
                print(f"ERROR: Finetuned model file (e.g., consolidated.safetensors) not found in directory {checkpoint_dir}")
                return
        print(f"Using checkpoint directory: {checkpoint_dir}")
        print(f"Using model file: {model_file_to_load}")


    if not os.path.exists(model_file_to_load):
        print(f"ERROR: Resolved model file not found at {model_file_to_load}")
        return

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        return

    # Load tokenizer
    text_tokenizer = Tokenizer.from_file(tokenizer_path)

    # Attempt to load config.json from the checkpoint directory
    loaded_model_config = None
    config_json_path = os.path.join(checkpoint_dir, 'config.json')
    if os.path.exists(config_json_path):
        print(f"Found config.json at {config_json_path}. Loading it.")
        try:
            with open(config_json_path, 'r') as f:
                loaded_model_config = json.load(f)
            print("Successfully loaded model config from training.")
        except Exception as e:
            print(f"Warning: Could not load config.json from {config_json_path}: {e}. Will proceed with default/manual config.")
    else:
        print(f"Warning: config.json not found in {checkpoint_dir}. Will proceed with default/manual config from YAML or script.")


    # VQA domains - exactly as defined in training config
    # These could also come from loaded_model_config if available and preferred
    vqa_in_domains = loaded_model_config.get('domains_in', ['rgb@224', 'caption']) if loaded_model_config else ['rgb@224', 'caption']
    vqa_out_domains = loaded_model_config.get('domains_out', ['caption']) if loaded_model_config else ['caption']
    
    print(f"Loading finetuned model state_dict from {model_file_to_load}...")
    # load_safetensors should ideally just return the state_dict if config is handled separately
    # For now, assume it might return a config_from_ckpt that we might merge or ignore if loaded_model_config exists
    state_dict, config_from_safetensor_meta = load_safetensors(model_file_to_load)

    if loaded_model_config:
        config = loaded_model_config
        # If config_from_safetensor_meta exists and has info not in loaded_model_config, could merge here if necessary
        # For now, prioritize config.json from training.
        if config_from_safetensor_meta:
            print("Ignoring metadata config from safetensor as config.json was loaded.")
    elif config_from_safetensor_meta:
        print("Using config from safetensor metadata as config.json was not found.")
        config = config_from_safetensor_meta
    else:
        config = {}
        print("Warning: No model configuration found from config.json or safetensor metadata.")

    # <<< START DEBUG PRINTS >>>
    print("\n=== Effective Config (from config.json or safetensor metadata if available) ===")
    if config:
        for key, value in config.items():
            if 'vocab' in key.lower() or 'caption' in key.lower() or 'token' in key.lower():
                print(f"  {key}: {value}")
        # Specifically look for vocab size if nested
        if 'modality_info' in config and 'caption' in config['modality_info']:
            print(f"  config['modality_info']['caption'].get('vocab_size'): {config['modality_info']['caption'].get('vocab_size')}")
        if 'decoder_embeddings_config' in config and 'caption' in config['decoder_embeddings_config']:
             print(f"  config['decoder_embeddings_config']['caption'].get('vocab_size'): {config['decoder_embeddings_config']['caption'].get('vocab_size')}")
    else:
        print("  config is empty or None.")

    print("\n=== State Dict Shapes for Caption Decoder Head ===")
    caption_head_weight_key = "decoder_embeddings.caption.head.weight"
    # caption_head_bias_key = "decoder_embeddings.caption.head.bias" # Bias key not directly checked for shape here, weight is primary

    # Handle potential FSDP prefix by checking common variations
    # The FSDP cleaning happens *after* this print block in the original script,
    # so we need to anticipate the prefixed key if it exists in the raw state_dict.
    
    # Check if state_dict is None or empty before proceeding
    if not state_dict:
        print("ERROR: State dictionary is empty after loading. Cannot proceed.")
        return

    resolved_caption_head_weight_key = None
    fsdp_prefix = "_fsdp_wrapped_module."
    
    # Check for prefixed key first
    if fsdp_prefix + caption_head_weight_key in state_dict:
        resolved_caption_head_weight_key = fsdp_prefix + caption_head_weight_key
    # Else check for non-prefixed key
    elif caption_head_weight_key in state_dict:
        resolved_caption_head_weight_key = caption_head_weight_key
    
    if resolved_caption_head_weight_key:
        print(f"  Shape of '{resolved_caption_head_weight_key}': {state_dict[resolved_caption_head_weight_key].shape}")
    else:
        print(f"  Key for caption decoder head weight ('{caption_head_weight_key}' with/without FSDP prefix) not found in state_dict.")
        print(f"  First 10 available keys in state_dict: {list(state_dict.keys())[:10]}")

    # It's also useful to see the vocab_size of the tokenizer you are using
    print(f"\n=== Tokenizer Vocab Size ===")
    print(f"  text_tokenizer.get_vocab_size(): {text_tokenizer.get_vocab_size()}")
    # <<< END DEBUG PRINTS >>>

    # Setup config for XL model if missing or incomplete from loaded config
    # IMPORTANT: Use 'config' (from config.json/metadata) as the base, then update if necessary

    if not config or not config.get('model_name'): # If config is empty or key fields missing, load defaults
        print("Checkpoint config is missing or incomplete. Loading VQA finetuning config from YAML as fallback/supplement...")
        finetune_yaml_config = load_finetune_config() # This YAML is a fallback, might be mostly overridden

        # Base XL config (compatible with fm_xlarge_24e_24d_swiglu_nobias)
        # Updated to align with EPFL-VILAB/4M-21_XL config.json
        config.update({
            'dim': 2048,
            'encoder_depth': 24,
            'decoder_depth': 24,
            'num_heads': 32,
            'mlp_ratio': 4.0,
            'qkv_bias': False,
            'proj_bias': False,
            'mlp_bias': False,
            'gated_mlp': True,
            'act_layer': 'SiLU',
            'norm_bias': False,  # From HF config.json
            'share_modality_embeddings': False,  # From HF config.json
            'use_act_checkpoint': finetune_yaml_config.get("use_act_checkpoint", False) if 'finetune_yaml_config' in locals() and finetune_yaml_config else False
        })
        
        # Override with specifics from our finetuning YAML if they exist
        if finetune_yaml_config.get("patch_size") is not None:
            config['patch_size'] = finetune_yaml_config["patch_size"]
        if finetune_yaml_config.get("input_size") is not None:
            config['image_size'] = finetune_yaml_config["input_size"] # FM uses image_size
        
        # The model name itself for FM init is passed directly, not as part of config dict usually
        # but the FM class might use config['model_name'] if we add it.
        # For now, the model name is hardcoded in FM() call later.
        print(f"Using manually constructed config: {config}")

    # Ensure these are always set based on VQA requirements
    config['domains_in'] = vqa_in_domains
    config['domains_out'] = vqa_out_domains
    config.setdefault('image_size', finetune_yaml_config.get("input_size", 224) if 'finetune_yaml_config' in locals() else 224)
    config.setdefault('patch_size', finetune_yaml_config.get("patch_size", 16) if 'finetune_yaml_config' in locals() else 16)
    
    # Create and load model
    # The 'model_name' argument to FM usually determines the base architecture.
    # If config_from_ckpt was missing, we ensure we are using the one from YAML.
    # The FM(config=config) will then use this config.
    
    # Determine model_name for FM constructor and add it to the config dictionary
    effective_model_name = config.get('model_name_or_type', config.get('model_name', None))
    if not effective_model_name:
        # Try to get from finetune_yaml_config if that was loaded as a fallback
        if 'finetune_yaml_config' in locals() and finetune_yaml_config.get("model_name"):
            effective_model_name = finetune_yaml_config["model_name"]
        else: # Fallback if still not found
            effective_model_name = model_name # model_name='fm_xlarge_24e_24d_swiglu_nobias' from script beginning
            print(f"Warning: model_name for FM not found in any loaded config, using default from script: {effective_model_name}")
    
    config['model_name_or_type'] = effective_model_name # Add/overwrite in config

    # Ensure the model's configuration for 'caption' modality uses the vocab size
    # from the loaded tokenizer. This is critical for matching the dimensions of
    # embedding layers and the final 'to_logits' projection layer in the decoder.
    actual_tokenizer_vocab_size = text_tokenizer.get_vocab_size()
    print(f"Updating model configuration: 'caption' vocab_size set to {actual_tokenizer_vocab_size} from tokenizer.")

    # Ensure 'modality_info' and 'caption' sub-dictionary exist in the config
    # This part is crucial if the loaded config from training doesn't have modality_info structured perfectly
    if 'modality_info' not in config or not isinstance(config['modality_info'], dict):
        config['modality_info'] = {}
    if not isinstance(config.get('modality_info', {}).get('caption'), dict):
        # If modality_info.caption exists but is not a dict (e.g. from a simplified saved config),
        # or if modality_info.caption doesn't exist, initialize it.
        config['modality_info']['caption'] = {}

    config['modality_info']['caption']['vocab_size'] = actual_tokenizer_vocab_size
    
    # If other crucial keys from MODALITY_INFO were not in the saved config, merge them carefully.
    # For instance, if saved config.modality_info.caption only has vocab_size.
    # This is a safeguard. Ideally, the saved config.json is comprehensive.
    if 'caption' in MODALITY_INFO:
        for key, value in MODALITY_INFO['caption'].items():
            if key not in config['modality_info']['caption']:
                config['modality_info']['caption'][key] = value
                print(f"Added missing MODALITY_INFO key to config: modality_info.caption.{key} = {value}")

    print(f"Final config being passed to FM: {{key: type(value) for key, value in config.items()}}") # Print types to check for complex objects
    # For full detail: import pprint; print(f"Final config being passed to FM: {pprint.pformat(config)}")

    # model_fourm_instance = FM(model_name=effective_model_name, config=config) # OLD CALL
    # The FM class should ideally primarily use the passed 'config' dictionary.
    model_fourm_instance = FM(config=config) # NEW CALL
    model_fourm_instance.to(device)
    model_fourm_instance.eval()
    
    # Clean up state dict keys (handle FSDP wrapping)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_fsdp_wrapped_module."):
            new_state_dict[k[len("_fsdp_wrapped_module."):]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict # This is the state_dict that will be used by load_state_dict

    # <<< REVISED DEBUG PRINT for the cleaned state_dict, focusing on to_logits >>>
    print("\n=== State Dict Shapes for Caption Decoder (POST FSDP CLEANING) ===")
    caption_to_logits_weight_key = "decoder_embeddings.caption.to_logits.weight"
    if caption_to_logits_weight_key in state_dict:
        print(f"  Shape of CLEANED '{caption_to_logits_weight_key}': {state_dict[caption_to_logits_weight_key].shape}")
        caption_to_logits_bias_key = "decoder_embeddings.caption.to_logits.bias"
        if caption_to_logits_bias_key in state_dict:
            print(f"  Shape of CLEANED '{caption_to_logits_bias_key}': {state_dict[caption_to_logits_bias_key].shape}")
        else:
            print(f"  Key for CLEANED caption decoder '{caption_to_logits_bias_key}' NOT FOUND.")
    else:
        print(f"  Key for CLEANED caption decoder '{caption_to_logits_weight_key}' NOT FOUND.")
        print(f"  First 10 available keys in CLEANED state_dict: {list(state_dict.keys())[:10]}")
    
    # Also check the token embedding layer itself, as it's also vocab-dependent
    caption_token_emb_key = "decoder_embeddings.caption.token_emb.weight"
    if caption_token_emb_key in state_dict:
        print(f"  Shape of CLEANED '{caption_token_emb_key}': {state_dict[caption_token_emb_key].shape}")
    else:
        print(f"  Key for CLEANED caption decoder '{caption_token_emb_key}' NOT FOUND.")

    # <<< END REVISED DEBUG PRINT >>>

    try:
        load_result = model_fourm_instance.load_state_dict(state_dict, strict=False)
        print(f"Model load result: Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
        if load_result.missing_keys:
            print(f"WARNING: The following VITAL keys were MISSING from the checkpoint and are RANDOMLY INITIALIZED: {load_result.missing_keys}")
            if any("decoder_embeddings.caption.head" in k for k in load_result.missing_keys):
                print("CRITICAL WARNING: Caption decoder head is missing. Model will output nonsense for text.")
        if not load_result.missing_keys and not load_result.unexpected_keys:
            print("Successfully loaded finetuned model: <All keys matched successfully>")
        elif not load_result.unexpected_keys:
            print("Model loaded, but some keys were missing (see warnings). Check if these are critical.")
        # else: there were unexpected keys, which strict=False allows but might indicate other issues.

    except RuntimeError as e:
        print(f"ERROR loading state_dict: {e}")
        return

    generation_sampler = GenerationSampler(model_fourm_instance)

    # Test VQA inference
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg" 
    question_text = "How many cats are in the photo?"

    pil_image = load_image_from_url(image_url)
    if pil_image is None:
        return

    # Image preprocessing - exact same as training
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    image_tensor = image_transform(pil_image).unsqueeze(0).to(device)

    # --- Hyperparameter Sweep Setup ---
    temperatures_to_test = [0.2, 0.5, 0.7, 0.8, 1.0]
    top_ps_to_test = [0.0, 0.8, 0.85, 0.9, 1.0] # 0.0 might disable or default, 1.0 considers all
    top_ks_to_test = [0, 50, 100]   # 0 might disable
    
    sweep_summary_results = [] # Initialize list to store results for summary

    for temp in temperatures_to_test:
        for top_p_val in top_ps_to_test:
            for top_k_val in top_ks_to_test:
                print(f"\n--- SWEEP RUN ---")
                print(f"Temperature: {temp}, top_p: {top_p_val}, top_k: {top_k_val}")
                print(f"Question: {question_text}")
                print("Generating VQA answer...")

                # Create VQA modality dictionary with transfer format optimized for VQA
                # This needs to be recreated or reset if generation modifies it in-place,
                # but for this script, it's okay to create it fresh each time.
                mod_dict, question_length = create_vqa_modality_dict_transfer_format(
                    question=question_text,
                    image_tensor=image_tensor,
                    text_tokenizer=text_tokenizer,
                    device=device,
                    max_answer_len=30 # This was not used by the function anyway
                )

                # Debug info for VQA path (can be made less verbose if needed for sweep)
                # print(f"Question length (VQA): {question_length}")
                # if 'caption' in mod_dict:
                #     print(f"Caption modality keys after processing (VQA): {list(mod_dict['caption'].keys())}")
                # if 'rgb@224' in mod_dict:
                #     print(f"rgb@224 modality keys after processing (VQA): {list(mod_dict['rgb@224'].keys())}")

                # Define generation schedule for VQA autoregressive generation
                max_new_tokens_vqa = 64
                generation_schedule = [
                    {
                        'target_domain': 'caption', # VQA answer is treated as a caption
                        'scheme': 'autoregressive',
                        'num_tokens': max_new_tokens_vqa,
                        'temperature': temp, # Use sweep variable
                        'cfg_scale': 4.0, # Restore CFG
                        'cfg_cond_domains': ['rgb@224'], # Condition on image
                    }
                ]
                # <<< END VQA Inference Path (Restored) >>>

                # print(f"Generation schedule: {generation_schedule}")

                with torch.no_grad():
                    # print(f"Input tensor shape: {mod_dict['caption']['tensor'].shape}")
                    # print(f"Input mask shape: {mod_dict['caption']['input_mask'].shape}")
                    # print(f"Target mask shape: {mod_dict['caption']['target_mask'].shape}")

                    # Debug: print key information (can be made less verbose)
                    # print(f"Input mask (False=visible, first 20): {mod_dict['caption']['input_mask'][0][:20].tolist()}")
                    # print(f"Target mask (False=target, positions 250-270): {mod_dict['caption']['target_mask'][0][250:270].tolist()}")

                    # Call generation
                    generated_result = generation_sampler.generate(
                        mod_dict=copy.deepcopy(mod_dict), # Use deepcopy if mod_dict is modified by generate
                        schedule=generation_schedule,
                        top_k=top_k_val, # Use sweep variable
                        top_p=top_p_val, # Use sweep variable
                        text_tokenizer=text_tokenizer,
                        verbose=False # Set to False for cleaner sweep output, True for debugging a single run
                    )

                    # Extract generated tokens
                    generated_tokens = generated_result['caption']['tensor'][0].cpu().tolist()
                    # print(f"Generated {len(generated_tokens)} tokens")

                    # Debug: show special token mappings
                    eos_id = text_tokenizer.token_to_id("[EOS]")
                    pad_id = text_tokenizer.token_to_id("[PAD]")
                    # print(f"Special tokens - EOS: {eos_id}, PAD: {pad_id}")

                    # Extract answer tokens from the generated sequence (autoregressive format)
                    # For autoregressive, the answer follows directly after the question
                    question_end_pos = question_length
                    raw_generated_part = generated_tokens[question_end_pos:] # Contains all generated tokens after prompt

                    num_expected_generated = generation_schedule[0].get('num_tokens', 5)
                    # print(f"Raw generated part (all tokens after prompt, {len(raw_generated_part)} tokens, expecting up to {num_expected_generated} new): {raw_generated_part[:num_expected_generated + 5]}..." )

                    # Get special token IDs for filtering
                    eos_id = text_tokenizer.token_to_id("[EOS]")
                    pad_id = text_tokenizer.token_to_id("[PAD]")

                    # Process answer section to extract meaningful answer tokens
                    final_tokens = []

                    actual_generated_sequence = raw_generated_part[:num_expected_generated]
                    # print(f"Actual generated sequence by sampler ({len(actual_generated_sequence)} tokens): {actual_generated_sequence}")

                    for token_pos_in_answer, token_id_val in enumerate(actual_generated_sequence):
                        if token_id_val == eos_id:
                            # print(f"Found EOS at position {token_pos_in_answer} in answer, stopping collection.")
                            break

                        if token_id_val == pad_id or token_id_val == 0: # Treat 0 as PAD
                            # print(f"Found PAD/0 at position {token_pos_in_answer} in answer, stopping collection.")
                            break

                        final_tokens.append(token_id_val)

                    # print(f"Final answer tokens: {final_tokens}")

                    # Now try to decode the answer
                    if final_tokens:
                        try:
                            answer_text = text_tokenizer.decode(final_tokens)
                            # print(f"Raw decoded answer: '{answer_text}'")

                            answer_text = answer_text.strip()
                            import re
                            answer_text = re.sub(r'\\[S_\\d+\\]', '', answer_text)
                            answer_text = re.sub(r'\\s+', ' ', answer_text)
                            answer_text = answer_text.strip()

                        except Exception as e:
                            # print(f"Error decoding answer tokens: {e}")
                            answer_text = f"[Decode error: {str(e)}]"
                    else:
                        answer_text = "[No valid answer tokens found]"

                    print(f"=== VQA RESULT (Temp: {temp}, P: {top_p_val}, K: {top_k_val}) ===")
                    # print(f"Question: {question_text}") # Repetitive, already printed above
                    print(f"Generated Answer: {answer_text}")

                    # Store results for final summary
                    sweep_summary_results.append({
                        'temperature': temp,
                        'top_p': top_p_val,
                        'top_k': top_k_val,
                        'question': question_text,
                        'answer': answer_text
                    })

                    # Debug information
                    # print(f"\n=== DEBUG INFO (VQA) ===")
                    # try:
                    #     question_tokens_from_output = generated_tokens[:question_length]
                    #     print(f"Question (from output tensor): {text_tokenizer.decode(question_tokens_from_output)}")
                    #     print(f"Generated answer tokens (final_tokens): {final_tokens}")
                    #     print(f"Raw generated VQA answer text: {answer_text}")
                    # except Exception as e:
                    #     print(f"Debug failed: {e}")
                print(f"--- END SWEEP RUN (Temp: {temp}, P: {top_p_val}, K: {top_k_val}) ---")

    # --- Print Consolidated Summary of Sweep --- 
    print("\n\n===========================================")
    print("=== HYPERPARAMETER SWEEP SUMMARY ===")
    print("===========================================")
    if sweep_summary_results:
        for result in sweep_summary_results:
            print(f"\n--- Configuration ---")
            print(f"  Temperature: {result['temperature']}")
            print(f"  Top_p:       {result['top_p']}")
            print(f"  Top_k:       {result['top_k']}")
            print(f"  Question:    {result['question']}")
            print(f"  Answer:      {result['answer']}")
    else:
        print("No results were collected from the sweep.")
    print("===========================================")

if __name__ == '__main__':
    main_inference()
