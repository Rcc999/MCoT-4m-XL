#!/usr/bin/env python
"""
Script to extend the 4M text tokenizer with MCoT-specific special tokens.

This script loads the existing text tokenizer, adds the new tokens needed for MCoT stages,
and saves the updated tokenizer to disk.

Usage:
    python extend_tokenizer_for_mcot.py --input-tokenizer fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json --output-path tokenizers/mcot_tokenizer.json
"""

import os
import json
import argparse
from pathlib import Path
from tokenizers import AddedToken, Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Extend 4M tokenizer with MCoT tokens")
    parser.add_argument("--input-tokenizer", type=str, 
                        default="fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json",
                        help="Path to the input tokenizer JSON file")
    parser.add_argument("--output-path", type=str,
                        default="tokenizers/mcot_tokenizer.json",
                        help="Path to save the extended tokenizer")
    return parser.parse_args()


def extend_tokenizer_with_mcot_tokens(tokenizer):
    """Add MCoT-specific special tokens to the tokenizer."""
    
    # Define MCoT stage markers
    mcot_stage_tokens = [
        AddedToken("[PLANNING_START]", single_word=True, normalized=False),
        AddedToken("[ACTING_START]", single_word=True, normalized=False),
        AddedToken("[REFLECTION_START]", single_word=True, normalized=False),
        AddedToken("[CORRECTION_START]", single_word=True, normalized=False),
    ]
    
    # Define object and bbox coordinate tokens for planning stage
    object_tokens = [
        AddedToken("[OBJECT]", single_word=True, normalized=False),
        AddedToken("[X]", single_word=True, normalized=False),
        AddedToken("[Y]", single_word=True, normalized=False),
        AddedToken("[W]", single_word=True, normalized=False),
        AddedToken("[H]", single_word=True, normalized=False),
    ]
    
    # Add all tokens to the tokenizer
    for token in mcot_stage_tokens + object_tokens:
        tokenizer.add_tokens([token])
    
    print(f"Added {len(mcot_stage_tokens)} MCoT stage tokens and {len(object_tokens)} object/bbox tokens")
    
    return tokenizer


def main():
    args = parse_args()
    
    print(f"Loading tokenizer from {args.input_tokenizer}")
    tokenizer = Tokenizer.from_file(args.input_tokenizer)
    
    # Extend with MCoT tokens
    tokenizer = extend_tokenizer_with_mcot_tokens(tokenizer)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save extended tokenizer
    tokenizer.save(args.output_path)
    print(f"Extended tokenizer saved to {args.output_path}")
    
    # Print vocabulary size for verification
    print(f"New vocabulary size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main() 