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
    question_tokens = text_tokenizer.encode(question).ids
    seq_len = len(question_tokens)
    
    # Initialize tensors and masks - key insight: for autoregressive generation,
    # the model starts with input tokens and generates sequentially after them
    caption_tensor = torch.zeros((batch_size, max_tokens), dtype=torch.long, device=device)
    
    # CRITICAL FIX: For autoregressive generation, we only make the INPUT TOKENS visible,
    # and ALL other positions should be marked as targets to be generated
    input_mask = torch.ones((batch_size, max_tokens), dtype=torch.bool, device=device)  # All masked initially
    target_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=device)  # All targets initially
    decoder_attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=device)
    
    # Place question tokens at the beginning 
    caption_tensor[0, :seq_len] = torch.tensor(question_tokens, dtype=torch.long, device=device)
    
    # Question tokens are visible inputs (not targets)
    input_mask[0, :seq_len] = False  # Question tokens are visible (not masked)
    target_mask[0, :seq_len] = True  # Question tokens are NOT targets (they are inputs) 
    decoder_attention_mask[0, :seq_len] = 1  # Attend to question tokens
    
    # All positions after the question are targets to be generated
    # target_mask[0, seq_len:] = False  (already set above)
    # input_mask[0, seq_len:] = True   (already set above)
    
    print(f"Question: '{question}'")
    print(f"Question tokens ({len(question_tokens)}): {question_tokens}")
    print(f"Caption tensor shape: {caption_tensor.shape}")
    print(f"Question input tokens: {caption_tensor[0, :seq_len].tolist()}")
    print(f"Input mask (False=visible): first 20 positions: {input_mask[0, :20].tolist()}")
    print(f"Target mask (False=target): first 20 positions: {target_mask[0, :20].tolist()}")
    print(f"Target mask (False=target): positions {seq_len}-{seq_len+10}: {target_mask[0, seq_len:seq_len+10].tolist()}")
    
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

def main_inference():
    """Main VQA inference function"""
    model_name = 'fm_xlarge_24e_24d_swiglu_nobias'
    finetuned_safetensor_path = 'vqa.safetensors' 
    tokenizer_path = 'fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check files exist
    if not os.path.exists(finetuned_safetensor_path):
        print(f"ERROR: Finetuned safetensor file not found at {finetuned_safetensor_path}")
        return

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        return

    # Load tokenizer
    text_tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # VQA domains - exactly as defined in training config
    vqa_in_domains = ['rgb@224', 'caption']
    vqa_out_domains = ['caption']
    
    print(f"Loading finetuned model from {finetuned_safetensor_path}...")
    state_dict, config = load_safetensors(finetuned_safetensor_path)
    
    # Setup config for XL model if missing
    if 'dim' not in config:
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
            'norm_bias': True,
        })
    
    config['domains_in'] = vqa_in_domains
    config['domains_out'] = vqa_out_domains
    config.setdefault('image_size', 224)
    config.setdefault('patch_size', 16)
    
    # Create and load model
    model_fourm_instance = FM(config=config)
    model_fourm_instance.to(device)
    model_fourm_instance.eval()
    
    # Clean up state dict keys (handle FSDP wrapping)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_fsdp_wrapped_module."):
            new_state_dict[k[len("_fsdp_wrapped_module."):]] = v
        else:
            new_state_dict[k] = v
    state_dict = new_state_dict

    try:
        msg = model_fourm_instance.load_state_dict(state_dict, strict=True)
        print(f"Successfully loaded finetuned model: {msg}")
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
    
    print(f"Question: {question_text}")
    print("Generating VQA answer...")

    # Create VQA modality dictionary with transfer format optimized for VQA
    mod_dict, question_length = create_vqa_modality_dict_transfer_format(
        question=question_text,
        image_tensor=image_tensor,
        text_tokenizer=text_tokenizer,
        device=device,
        max_answer_len=30
    )
    
    # Apply decoder embedding forward_embed to add required keys ('ids', 'emb', 'x')
    # This is what the FM model normally does in its forward method
    print("Processing modality dictionary through decoder embeddings...")
    processed_mod_dict = {}
    for mod, d in mod_dict.items():
        if mod in model_fourm_instance.decoder_embeddings:
            print(f"Processing {mod} through decoder embedding...")
            processed_mod_dict[mod] = model_fourm_instance.decoder_embeddings[mod].forward_embed(d)
        else:
            processed_mod_dict[mod] = d
    mod_dict = processed_mod_dict
    
    # Debug info
    print(f"Question length: {question_length}")
    print(f"Caption modality keys after processing: {list(mod_dict['caption'].keys())}")
    
    # Define generation schedule for VQA autoregressive generation
    # For autoregressive, num_tokens should be the number of NEW tokens to generate
    max_new_tokens = 30  # Generate up to 30 new answer tokens
    generation_schedule = [
        {
            'target_domain': 'caption',
            'scheme': 'autoregressive',
            'num_tokens': max_new_tokens,  # Number of NEW tokens to generate, not total length
            'temperature': 0.7,  # Temperature for sampling
            'cfg_scale': 1.0,
            'cfg_cond_domains': [],
        }
    ]
    
    print(f"Generation schedule: {generation_schedule}")

    with torch.no_grad():
        print(f"Input tensor shape: {mod_dict['caption']['tensor'].shape}")
        print(f"Input mask shape: {mod_dict['caption']['input_mask'].shape}")
        print(f"Target mask shape: {mod_dict['caption']['target_mask'].shape}")
        
        # Debug: print key information
        print(f"Input mask (False=visible, first 20): {mod_dict['caption']['input_mask'][0][:20].tolist()}")
        print(f"Target mask (False=target, positions 250-270): {mod_dict['caption']['target_mask'][0][250:270].tolist()}")
        
        # Call generation
        generated_result = generation_sampler.generate(
            mod_dict=mod_dict,
            schedule=generation_schedule,
            top_k=50,
            top_p=0.9,
            text_tokenizer=text_tokenizer,
            verbose=True
        )
        
        # Extract generated tokens
        generated_tokens = generated_result['caption']['tensor'][0].cpu().tolist()
        print(f"Generated {len(generated_tokens)} tokens")
        
        # Debug: show special token mappings
        eos_id = text_tokenizer.token_to_id("[EOS]")
        pad_id = text_tokenizer.token_to_id("[PAD]")
        print(f"Special tokens - EOS: {eos_id}, PAD: {pad_id}")
        
        # Extract answer tokens from the generated sequence (autoregressive format)
        # For autoregressive, the answer follows directly after the question
        question_end_pos = question_length
        answer_section = generated_tokens[question_end_pos:]
        print(f"Answer section ({len(answer_section)} tokens): {answer_section[:20]}...")
        
        # Get special token IDs for filtering
        eos_id = text_tokenizer.token_to_id("[EOS]")
        pad_id = text_tokenizer.token_to_id("[PAD]")
        
        # Process answer section to extract meaningful answer tokens
        final_tokens = []
        
        for i, token in enumerate(answer_section):
            # Stop at EOS, PAD, or null tokens
            if token in [eos_id, pad_id] or token == 0:
                print(f"Stopping at position {i}, token {token} (EOS/PAD/0)")
                break
                
            # Collect all non-special tokens as answer
            final_tokens.append(token)
        
        print(f"Final answer tokens: {final_tokens}")
        
        # Now try to decode the answer
        if final_tokens:
            try:
                # Try to decode the full answer
                answer_text = text_tokenizer.decode(final_tokens)
                print(f"Raw decoded answer: '{answer_text}'")
                
                # Clean up the answer text
                answer_text = answer_text.strip()
                
                # Additional cleanup: remove any remaining special tokens if they somehow got through
                import re
                answer_text = re.sub(r'\[S_\d+\]', '', answer_text)  # Remove any sentinel tokens
                answer_text = re.sub(r'\s+', ' ', answer_text)  # Normalize whitespace
                answer_text = answer_text.strip()
                    
            except Exception as e:
                print(f"Error decoding answer tokens: {e}")
                answer_text = f"[Decode error: {str(e)}]"
        else:
            answer_text = "[No valid answer tokens found]"
            
        print(f"\n=== VQA RESULT ===")
        print(f"Question: {question_text}")
        print(f"Generated Answer: {answer_text}")
        
        # Debug information
        print(f"\n=== DEBUG INFO ===")
        try:
            question_tokens = generated_tokens[:question_length]
            print(f"Question: {text_tokenizer.decode(question_tokens)}")
            print(f"Answer tokens: {final_tokens}")
            print(f"Raw answer: {answer_text}")
        except Exception as e:
            print(f"Debug failed: {e}")

if __name__ == '__main__':
    main_inference()
