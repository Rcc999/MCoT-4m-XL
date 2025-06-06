#!/usr/bin/env python3
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
import datetime
import re

from fourm.models.fm import FM
from fourm.models.generate import GenerationSampler, init_full_input_modality, init_empty_target_modality
from fourm.data.modality_info import MODALITY_INFO
from fourm.utils.checkpoint import load_safetensors
from fourm.utils.data_constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def load_image_from_path(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
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
    max_tokens = 256
    batch_size = image_tensor.shape[0]
    
    question_tokens_list = text_tokenizer.encode(question).ids
    
    eos_id = text_tokenizer.token_to_id("[EOS]")
    s2_id = text_tokenizer.token_to_id("[S_2]")

    if eos_id is None:
        print("Warning: [EOS] token not found in tokenizer.")
    if s2_id is None:
        print("ERROR: [S_2] token not found in tokenizer. This is critical for the current test.")

    if eos_id is not None:
        question_tokens_list.append(eos_id)
    
    seq_len = len(question_tokens_list)

    if seq_len >= max_tokens:
        print(f"Warning: Question + EOS ({seq_len} tokens) is too long for max_tokens ({max_tokens}). Truncating.")
        original_question_part = question_tokens_list
        num_special_to_append = 0
        
        if eos_id is not None and original_question_part[-1] == eos_id:
            original_question_part = original_question_part[:-1]
            num_special_to_append = 1
        
        max_question_part_len = max_tokens - num_special_to_append 
        truncated_question_part = original_question_part[:max_question_part_len]
        
        question_tokens_list = truncated_question_part
        if eos_id is not None and num_special_to_append == 1:
             question_tokens_list.append(eos_id)
        
        seq_len = len(question_tokens_list)

    caption_tensor = torch.zeros((batch_size, max_tokens), dtype=torch.long, device=device)
    
    input_mask = torch.ones((batch_size, max_tokens), dtype=torch.bool, device=device)
    target_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool, device=device)
    decoder_attention_mask = torch.zeros((batch_size, max_tokens), dtype=torch.long, device=device)
    
    if seq_len > 0:
        caption_tensor[0, :seq_len] = torch.tensor(question_tokens_list, dtype=torch.long, device=device)
    
    input_mask[0, :seq_len] = False
    target_mask[0, :seq_len] = False
    
    if seq_len > 0:
        decoder_attention_mask[0, 0] = seq_len
        decoder_attention_mask[0, 1:seq_len] = 1
    decoder_attention_mask[0, seq_len:] = 1
    
    print(f"Question (processed for model): '{text_tokenizer.decode(question_tokens_list if seq_len > 0 else [])}'")
    print(f"Question tokens (processed, incl. EOS, length {seq_len}): {question_tokens_list if seq_len > 0 else '[]'}")
    print(f"Caption tensor shape: {caption_tensor.shape}")
    if seq_len > 0:
        print(f"Question input tokens (from tensor): {caption_tensor[0, :seq_len].tolist()}")
        print(f"Input mask (False=visible): first {min(seq_len + 5, max_tokens)} positions: {input_mask[0, :min(seq_len + 5, max_tokens)].tolist()}")
        print(f"Target mask (False=target): first {min(seq_len + 5, max_tokens)} positions: {target_mask[0, :min(seq_len + 5, max_tokens)].tolist()}")
    else:
        print("No question tokens to display for tensor/masks.")
    print(f"Target mask (False=target): positions {seq_len}-{min(seq_len+10, max_tokens)}: {target_mask[0, seq_len:min(seq_len+10, max_tokens)].tolist()}")
    
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
    
    mod_dict = init_full_input_modality(mod_dict, MODALITY_INFO, 'rgb@224', device)
    
    return mod_dict, seq_len

def load_finetune_config(yaml_path="cfgs/default/4m/finetune/4m-xl_mod21_vqav2_finetune.yaml"):
    try:
        with open(yaml_path, 'r') as f:
            finetune_cfg = yaml.safe_load(f)
        
        model_config_from_yaml = {
            "model_name": finetune_cfg.get("model"),
            "patch_size": finetune_cfg.get("patch_size"),
            "input_size": finetune_cfg.get("input_size"),
            "use_act_checkpoint": finetune_cfg.get("use_act_checkpoint")
        }
        if not model_config_from_yaml["model_name"]:
            print(f"Warning: 'model' key not found in {yaml_path}")
        return model_config_from_yaml
    except Exception as e:
        print(f"Error loading or parsing finetuning YAML {yaml_path}: {e}")
        return {}

def get_next_output_filename(base_name):
    if not os.path.exists(base_name):
        return base_name
    
    name, ext = os.path.splitext(base_name)
    counter = 1
    
    while True:
        new_name = f"{name}_{counter}{ext}"
        if not os.path.exists(new_name):
            return new_name
        counter += 1

def main_inference():
    model_name = 'EPFL-VILAB/4M-7-T2I_XL_CC12M'
    tokenizer_path = 'fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json'
    finetuned_weights_path = 'checkpoint-2.safetensors'
    
    output_dir = 'vqa_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    test_cases_file = 'test_cases.json'
    output_file = os.path.join(output_dir, 'vqa_results.json')
    output_file = get_next_output_filename(output_file)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Results will be saved to: {output_file}")

    if not os.path.exists(test_cases_file):
        print(f"Error: Test cases file not found at {test_cases_file}")
        return

    try:
        with open(test_cases_file, 'r') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"Error loading test cases from {test_cases_file}: {e}")
        return

    results = {
        "model_name": model_name,
        "device": device,
        "timestamp": datetime.datetime.now().isoformat(),
        "results": []
    }

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        return

    text_tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Loading pre-trained model from HuggingFace Hub: {model_name}")
    try:
        model_fourm_instance = FM.from_pretrained(model_name)
        model_fourm_instance.to(device)
        model_fourm_instance.eval()
        
        if os.path.exists(finetuned_weights_path):
            print(f"Loading finetuned weights from {finetuned_weights_path}")
            try:
                state_dict, _ = load_safetensors(finetuned_weights_path)
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("_fsdp_wrapped_module."):
                        new_state_dict[k[len("_fsdp_wrapped_module."):]] = v
                    else:
                        new_state_dict[k] = v
                
                load_result = model_fourm_instance.load_state_dict(new_state_dict, strict=False)
                print(f"Model load result: Missing keys: {load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}")
                if load_result.missing_keys:
                    print(f"WARNING: The following keys were MISSING from the checkpoint: {load_result.missing_keys}")
                if not load_result.missing_keys and not load_result.unexpected_keys:
                    print("Successfully loaded finetuned weights: <All keys matched successfully>")
                elif not load_result.unexpected_keys:
                    print("Finetuned weights loaded, but some keys were missing (see warnings).")
            except Exception as e:
                print(f"Error loading finetuned weights: {e}")
                print("Continuing with base model weights...")
        else:
            print(f"Warning: Finetuned weights not found at {finetuned_weights_path}")
            print("Using base model weights...")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with local configuration...")
        
        config = {
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
            'norm_bias': False,
            'share_modality_embeddings': False,
            'use_act_checkpoint': False,
            'image_size': 224,
            'patch_size': 16,
            'domains_in': ['rgb@224', 'caption'],
            'domains_out': ['caption'],
            'model_name_or_type': model_name
        }
        
        config['modality_info'] = {}
        config['modality_info']['caption'] = {}
        config['modality_info']['caption']['vocab_size'] = text_tokenizer.get_vocab_size()
        
        if 'caption' in MODALITY_INFO:
            for key, value in MODALITY_INFO['caption'].items():
                if key not in config['modality_info']['caption']:
                    config['modality_info']['caption'][key] = value
        
        model_fourm_instance = FM(config=config)
        model_fourm_instance.to(device)
        model_fourm_instance.eval()

    generation_sampler = GenerationSampler(model_fourm_instance)

    for test_case in test_cases["test_cases"]:
        image_path = test_case["image_path"]
        questions = test_case["questions"]
        
        print(f"\nProcessing test case:")
        print(f"Image path: {image_path}")
        
        pil_image = load_image_from_path(image_path)
        if pil_image is None:
            print("Skipping this test case due to image loading error")
            continue

        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
        image_tensor = image_transform(pil_image).unsqueeze(0).to(device)

        test_case_results = []

        for question_text in questions:
            print(f"\nQuestion: {question_text}")

            temperatures_to_test = [0.3, 0.5, 0.7]
            top_ps_to_test = [0.9, 0.95, 0.98]
            top_ks_to_test = [40, 50, 60]
            
            question_results = []
            
            for temp in temperatures_to_test:
                for top_p_val in top_ps_to_test:
                    for top_k_val in top_ks_to_test:
                        print(f"\n--- SWEEP RUN ---")
                        print(f"Temperature: {temp}, top_p: {top_p_val}, top_k: {top_k_val}")
                        print(f"Question: {question_text}")
                        print("Generating VQA answer...")

                        mod_dict, question_length = create_vqa_modality_dict_transfer_format(
                            question=question_text,
                            image_tensor=image_tensor,
                            text_tokenizer=text_tokenizer,
                            device=device,
                            max_answer_len=16
                        )

                        max_new_tokens_vqa = 100
                        generation_schedule = [
                            {
                                'target_domain': 'caption',
                                'scheme': 'autoregressive',
                                'num_tokens': max_new_tokens_vqa,
                                'temperature': temp,
                                'cfg_scale': 5.0,
                                'cfg_cond_domains': ['rgb@224'],
                            }
                        ]

                        with torch.no_grad():
                            try:
                                generated_result = generation_sampler.generate(
                                    mod_dict=copy.deepcopy(mod_dict),
                                    schedule=generation_schedule,
                                    top_k=top_k_val,
                                    top_p=top_p_val,
                                    text_tokenizer=text_tokenizer,
                                    verbose=False
                                )

                                generated_tokens = generated_result['caption']['tensor'][0].cpu().tolist()
                                question_end_pos = question_length
                                raw_generated_part = generated_tokens[question_end_pos:]
                                num_expected_generated = generation_schedule[0].get('num_tokens', 5)
                                
                                eos_id = text_tokenizer.token_to_id("[EOS]")
                                pad_id = text_tokenizer.token_to_id("[PAD]")
                                
                                final_tokens = []
                                actual_generated_sequence = raw_generated_part[:num_expected_generated]
                                
                                for token_id_val in actual_generated_sequence:
                                    if token_id_val in [eos_id, pad_id, 0]:
                                        break
                                    final_tokens.append(token_id_val)

                                try:
                                    answer_text = text_tokenizer.decode(final_tokens)
                                    answer_text = re.sub(r'\[.*?\]', '', answer_text)
                                    answer_text = re.sub(r'\s+', ' ', answer_text)
                                    answer_text = answer_text.strip()
                                    
                                    answer_text = answer_text.replace(' .', '.').replace(' ,', ',')
                                    answer_text = re.sub(r'([.,!?])\1+', r'\1', answer_text)
                                    
                                    if answer_text and not answer_text[0].isupper():
                                        answer_text = answer_text[0].upper() + answer_text[1:]
                                    
                                except Exception as e:
                                    answer_text = f"[Decode error: {str(e)}]"
                                    print(f"Warning: Error processing answer: {e}")

                                question_results.append({
                                    'temperature': temp,
                                    'top_p': top_p_val,
                                    'top_k': top_k_val,
                                    'answer': answer_text,
                                    'raw_tokens': final_tokens
                                })

                                print(f"Generated Answer: {answer_text}")
                                
                            except Exception as e:
                                print(f"Error during generation: {e}")
                                question_results.append({
                                    'temperature': temp,
                                    'top_p': top_p_val,
                                    'top_k': top_k_val,
                                    'answer': f"[Generation error: {str(e)}]",
                                    'error': str(e)
                                })

            test_case_results.append({
                "question": question_text,
                "sweep_results": question_results
            })

        results["results"].append({
            "image_path": image_path,
            "questions": test_case_results
        })

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nAll results have been saved to {output_file}")

if __name__ == '__main__':
    main_inference()
