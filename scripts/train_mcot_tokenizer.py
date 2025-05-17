import os
import argparse
from pathlib import Path
import glob
import json

# Assuming the script is in ml-4m/scripts/, so we adjust the path to import from fourm
# If your ml-4m directory is not in the python path, you might need to add it:
import sys
FOURM_ROOT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(FOURM_ROOT_PATH))

from fourm.utils.tokenizer import (
    train_unified_wordpiece_tokenizer,
    generate_sentinel_tokens,
    generate_coord_tokens,
    AddedToken
)

# --- Configuration ---
# Adjust these paths as necessary
FOURM_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MSCOCO_PROCESSED_CAPTION_DIR = FOURM_ROOT / "my_mscoco_for_4m" / "caption"
DEFAULT_SAVE_PATH = FOURM_ROOT / "fourm/utils/tokenizer/trained/text_tokenizer_4m_mcot.json"
# Path to the local object_classes.json within the MCoT-4m-XL repository
LOCAL_OBJECT_CLASSES_PATH = FOURM_ROOT / "fourm/utils/tokenizer/object_classes.json"

# Define your MCOT stage marker tokens
MCOT_STAGE_MARKERS = [
    AddedToken(content="[PLANNING_START]", single_word=True, normalized=False),
    AddedToken(content="[ACTING_START]", single_word=True, normalized=False),
    AddedToken(content="[REFLECTION_START]", single_word=True, normalized=False), # Add if needed later
    AddedToken(content="[CORRECTION_START]", single_word=True, normalized=False), # Add if needed later
]

def get_args():
    parser = argparse.ArgumentParser('Train MCOT-Extended Unified WordPiece Tokenizer', add_help=False)
    parser.add_argument('--text_files_dir', type=str,
                        default=str(DEFAULT_MSCOCO_PROCESSED_CAPTION_DIR),
                        help="Directory containing MS-COCO caption text files (e.g., my_mscoco_for_4m/caption/). "
                             "The script will glob for all .txt files in train/ and validation/ subdirs.")
    parser.add_argument('--save_file', type=str,
                        default=str(DEFAULT_SAVE_PATH),
                        help="Path to save the trained MCOT tokenizer JSON file.")
    parser.add_argument('--base_vocab_size', type=int, default=30000,
                        help="Target vocabulary size for WordPiece, *before* adding special MCOT tokens.")
    parser.add_argument('--num_sentinels', type=int, default=200,
                        help="Number of standard sentinel tokens (e.g., [S_0], [S_1]...).")
    parser.add_argument('--coord_bins', type=int, default=1000,
                        help="Number of coordinate bins for detection tokens (e.g., v0=0...v3=999).")
    parser.add_argument('--object_classes', type=str, default='coco', choices=['none', 'coco'],
                        help="Include special tokens for object class names (e.g., from COCO dataset).")
    parser.add_argument('--lowercase', action='store_true', default=True,
                        help="Convert text to lowercase before tokenization.")
    parser.add_argument('--no_lowercase', action='store_false', dest='lowercase')
    parser.add_argument('--min_frequency', type=int, default=2, # Default from HuggingFace tokenizers
                        help="The minimum frequency a token must have to be included in the vocabulary.")
    return parser.parse_args()

def collect_caption_files(text_files_dir_str: str) -> list[str]:
    """
    Collects all .txt files from train and validation subdirectories.
    """
    text_files_dir = Path(text_files_dir_str)
    caption_files = []
    for split in ["train", "validation"]: # Or whatever your split names are
        split_dir = text_files_dir / split
        if split_dir.exists() and split_dir.is_dir():
            files_in_split = list(split_dir.glob("**/*.txt")) # Recursive glob
            caption_files.extend([str(f) for f in files_in_split])
            print(f"Found {len(files_in_split)} caption files in {split_dir}")
        else:
            print(f"Warning: Directory not found or not a directory: {split_dir}")
    
    if not caption_files:
        print(f"Error: No .txt caption files found in {text_files_dir}/train/ or {text_files_dir}/validation/. Please check the path and structure.")
        exit(1)
    return caption_files

def train_mcot_tokenizer(args):
    # 1. Collect all caption files from your processed MS-COCO directory
    #    (e.g., from ./my_mscoco_for_4m/caption/train/*.txt and ./my_mscoco_for_4m/caption/validation/*.txt)
    caption_files = collect_caption_files(args.text_files_dir)
    print(f"Total caption files found for tokenizer training: {len(caption_files)}")

    # 2. Generate standard 4M special tokens
    sentinel_tokens = generate_sentinel_tokens(num=args.num_sentinels)
    coord_tokens = generate_coord_tokens(bins=args.coord_bins)
    
    object_class_tokens = []
    if args.object_classes == 'coco':
        if not LOCAL_OBJECT_CLASSES_PATH.exists():
            print(f"Error: Local object_classes.json not found at {LOCAL_OBJECT_CLASSES_PATH}")
            print("Please ensure this file exists in your repository structure.")
            exit(1)
        try:
            with open(LOCAL_OBJECT_CLASSES_PATH, 'r') as f:
                all_classes_data = json.load(f)
            coco_class_names = all_classes_data.get("coco")
            if coco_class_names and isinstance(coco_class_names, list):
                for name in coco_class_names:
                    # Match behavior of original func: single_word=False, normalized=True (default for AddedToken)
                    object_class_tokens.append(AddedToken(content=name, single_word=False, normalized=True))
                print(f"Successfully loaded and generated {len(object_class_tokens)} COCO object class tokens locally.")
            else:
                print(f"Warning: 'coco' key not found or not a list in {LOCAL_OBJECT_CLASSES_PATH}. No object class tokens will be added.")
        except Exception as e:
            print(f"Error reading or processing {LOCAL_OBJECT_CLASSES_PATH}: {e}. No object class tokens will be added.")
    
    # Combine all special tokens: standard 4M ones + your MCOT stage markers
    all_special_tokens = [
        AddedToken(content="[UNK]", single_word=True, normalized=False), # unk_token
        AddedToken(content="[PAD]", single_word=True, normalized=False), # pad_token
        AddedToken(content="[SOS]", single_word=True, normalized=False), # sos_token
        AddedToken(content="[EOS]", single_word=True, normalized=False)  # eos_token
    ]
    all_special_tokens.extend(sentinel_tokens)
    all_special_tokens.extend(coord_tokens)
    if object_class_tokens: # Check if list is not empty
        all_special_tokens.extend(object_class_tokens)
    all_special_tokens.extend(MCOT_STAGE_MARKERS)

    print(f"Training MCOT-extended tokenizer on {len(caption_files)} files.")
    print(f"Target base vocabulary size: {args.base_vocab_size}")
    print(f"Number of special tokens to add: {len(all_special_tokens)}")

    # 3. Train the tokenizer
    # This function is from `fourm.utils.tokenizer.text_tokenizer`
    tokenizer = train_unified_wordpiece_tokenizer(
        files=caption_files,
        vocab_size=args.base_vocab_size, # Target size for WordPiece model tokens
        # The special tokens below are passed to the trainer and guaranteed to be in the vocab
        unk_token="[UNK]", # Already in all_special_tokens if using AddedToken
        pad_token="[PAD]",
        sos_token="[SOS]",
        eos_token="[EOS]",
        sentinel_tokens=sentinel_tokens, # Pass the generated AddedToken objects
        coord_tokens=coord_tokens,       # Pass the generated AddedToken objects
        object_class_tokens=object_class_tokens if args.object_classes != 'none' and object_class_tokens else None,
        additional_special_tokens=MCOT_STAGE_MARKERS, # Your new MCOT tokens
        min_frequency=args.min_frequency,
        lowercase=args.lowercase,
    )

    # 4. Save the tokenizer
    save_path = Path(args.save_file)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))

    final_vocab_size = tokenizer.get_vocab_size()
    print(f"MCOT-extended tokenizer saved to: {save_path}")
    print(f"Final vocabulary size: {final_vocab_size}")
    print("Ensure `[PLANNING_START]` and `[ACTING_START]` are present with their IDs:")
    # Check if tokens exist before trying to get their ID to prevent errors if they weren't added
    if tokenizer.token_to_id('[PLANNING_START]') is not None:
        print(f"Token for [PLANNING_START]: {tokenizer.token_to_id('[PLANNING_START]')}")
    else:
        print("Warning: [PLANNING_START] not found in tokenizer vocab.")
    if tokenizer.token_to_id('[ACTING_START]') is not None:
        print(f"Token for [ACTING_START]: {tokenizer.token_to_id('[ACTING_START]')}")
    else:
        print("Warning: [ACTING_START] not found in tokenizer vocab.")

if __name__ == "__main__":
    args = get_args()
    train_mcot_tokenizer(args) 