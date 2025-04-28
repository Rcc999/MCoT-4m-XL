#!/usr/bin/env python3
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
Script to extend the 4M model's vocabulary with MCOT special tokens
This adds four new special stage-marker tokens to the existing tokenizer.
"""

import argparse
import os
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece

def parse_args():
    parser = argparse.ArgumentParser(description="Extend vocabulary with MCOT special tokens")
    parser.add_argument("--tokenizer-path", type=str, 
                        default="fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json", 
                        help="Path to the original tokenizer")
    parser.add_argument("--output-path", type=str, 
                        default="fourm/utils/tokenizer/trained/text_tokenizer_4m_mcot.json", 
                        help="Path to save the extended tokenizer")
    parser.add_argument("--checkpoint-path", type=str, 
                        required=True, 
                        help="Path to the 4M model checkpoint to update")
    parser.add_argument("--output-checkpoint-path", type=str, 
                        required=True, 
                        help="Path to save the updated model checkpoint")
    return parser.parse_args()

def extend_tokenizer(tokenizer_path, output_path):
    """
    Extend the tokenizer with MCOT special tokens.
    """
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Get the current vocabulary size
    original_vocab_size = tokenizer.get_vocab_size()
    print(f"Original vocabulary size: {original_vocab_size}")
    
    # Add the new special tokens
    special_tokens = [
        "[PLANNING_START]",
        "[ACTING_START]",
        "[REFLECTION_START]",
        "[CORRECTION_START]"
    ]
    
    # Add special tokens to the tokenizer
    for token in special_tokens:
        tokenizer.add_tokens([token])
    
    # Save the extended tokenizer
    tokenizer.save(output_path)
    print(f"Extended tokenizer saved to {output_path}")
    print(f"New vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Return the size difference for model embedding adjustment
    return tokenizer.get_vocab_size() - original_vocab_size, original_vocab_size

def update_model_embeddings(checkpoint_path, output_checkpoint_path, vocab_size_diff, original_vocab_size):
    """
    Update the model's text embeddings to accommodate the extended vocabulary.
    """
    print(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Update the input embedding for text
    text_emb_key = "encoder_embeddings.text_embeddings.embedding.weight"
    if text_emb_key in checkpoint["model"]:
        original_embedding = checkpoint["model"][text_emb_key]
        print(f"Original embedding shape: {original_embedding.shape}")
        
        # Create a new embedding matrix with extended vocabulary
        new_vocab_size = original_vocab_size + vocab_size_diff
        new_embedding = torch.zeros(
            (new_vocab_size, original_embedding.shape[1]),
            dtype=original_embedding.dtype
        )
        
        # Copy the original embeddings
        new_embedding[:original_vocab_size] = original_embedding
        
        # Initialize the new token embeddings with random values similar to the original distribution
        mean_val = original_embedding.mean().item()
        std_val = original_embedding.std().item()
        
        # Initialize the new tokens with similar statistical properties
        with torch.no_grad():
            torch.nn.init.normal_(new_embedding[original_vocab_size:], mean=mean_val, std=std_val)
        
        # Update the checkpoint
        checkpoint["model"][text_emb_key] = new_embedding
        print(f"Updated embedding shape: {new_embedding.shape}")
    else:
        print(f"Warning: '{text_emb_key}' not found in the model checkpoint")
    
    # Save the updated checkpoint
    print(f"Saving updated model checkpoint to {output_checkpoint_path}")
    torch.save(checkpoint, output_checkpoint_path)
    print("Checkpoint updated successfully")

def main():
    args = parse_args()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_checkpoint_path), exist_ok=True)
    
    # Extend the tokenizer and get the vocabulary size difference
    vocab_size_diff, original_vocab_size = extend_tokenizer(args.tokenizer_path, args.output_path)
    
    # Update the model's embeddings
    update_model_embeddings(args.checkpoint_path, args.output_checkpoint_path, vocab_size_diff, original_vocab_size)
    
    print("Vocabulary extension completed successfully!")

if __name__ == "__main__":
    main() 