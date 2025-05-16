#!/bin/bash
# Script to prepare the tokenizer with MCoT special tokens
set -e  # Exit immediately if a command fails

echo "=== Preparing MCoT Tokenizer ==="

# Check if tokenizer already exists
if [ -f "tokenizers/mcot_tokenizer.json" ]; then
    echo "MCoT tokenizer already exists at 'tokenizers/mcot_tokenizer.json'"
    echo "Remove this file if you want to recreate it."
    exit 0
fi

# Ensure the tokenizers directory exists
mkdir -p tokenizers

# Run the tokenizer extension script
echo "Extending the 4M tokenizer with MCoT special tokens..."
python extend_tokenizer_for_mcot.py \
    --input-tokenizer fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json \
    --output-path tokenizers/mcot_tokenizer.json

# Verify the tokenizer was created
if [ -f "tokenizers/mcot_tokenizer.json" ]; then
    echo "✅ MCoT tokenizer successfully created!"
    echo "Path: tokenizers/mcot_tokenizer.json"
else
    echo "❌ Failed to create MCoT tokenizer"
    exit 1
fi 