import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os

from tokenizers import Tokenizer

# --- Constants for MCOT ---
# These should match what's in your scripts/train_mcot_tokenizer.py and the trained tokenizer
PLANNING_START_TOKEN = "[PLANNING_START]"
ACTING_START_TOKEN = "[ACTING_START]" # We'll use this later
MCOT_TOKENIZER_PATH = str(Path(__file__).resolve().parent.parent / "fourm/utils/tokenizer/trained/text_tokenizer_4m_mcot.json")
DEFAULT_COORD_BINS = 1000 # Should match the --coord_bins used for training the MCOT tokenizer

# --- Tokenizer Loading ---

def load_mcot_tokenizer(tokenizer_path: str = MCOT_TOKENIZER_PATH) -> Tokenizer:
    """
    Loads the MCOT tokenizer from a given file path.
    
    Args:
        tokenizer_path (str): Path to the tokenizer JSON file.
        
    Returns:
        Tokenizer: The loaded tokenizer.
    """
    print(f"Attempting to load tokenizer from: {tokenizer_path}")
    print(f"Absolute path: {os.path.abspath(tokenizer_path)}")
    print(f"File exists: {os.path.exists(tokenizer_path)}")
    
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"Successfully loaded tokenizer with vocab size: {len(tokenizer.get_vocab())}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        # Try with absolute path from pwd
        alt_path = os.path.join(os.getcwd(), tokenizer_path)
        print(f"Trying alternative path: {alt_path}")
        if os.path.exists(alt_path):
            try:
                tokenizer = Tokenizer.from_file(alt_path)
                print(f"Successfully loaded tokenizer from alternative path with vocab size: {len(tokenizer.get_vocab())}")
                return tokenizer
            except Exception as e2:
                print(f"Failed to load from alternative path: {str(e2)}")
        
        # Try looking in a few common locations
        common_locations = [
            "./tokenizer_ckpts/text_tokenizer_4m_mcot.json",
            "../tokenizer_ckpts/text_tokenizer_4m_mcot.json",
            "fourm/utils/tokenizer/trained/text_tokenizer_4m_mcot.json"
        ]
        
        for loc in common_locations:
            print(f"Trying common location: {loc}")
            if os.path.exists(loc):
                try:
                    tokenizer = Tokenizer.from_file(loc)
                    print(f"Successfully loaded tokenizer from common location with vocab size: {len(tokenizer.get_vocab())}")
                    return tokenizer
                except Exception as e3:
                    print(f"Failed to load from common location: {str(e3)}")
        
        raise ValueError(f"Could not load tokenizer from path: {tokenizer_path} or any common locations.")

# --- Planning Stage Data Transformation ---

def format_detection_string_for_plan(
    detection_data: Dict[str, List[Dict[str, Any]]], 
    coord_bins: int = DEFAULT_COORD_BINS
) -> str:
    """
    Converts detection JSON data into a single string compatible with 4M's detection format
    and ready for tokenization for the planning stage.

    Args:
        detection_data: Parsed JSON content from a .json detection file.
                        Expected structure: {'instances': [{'boxes': [xmin, ymin, xmax, ymax], 'class_name': str}, ...]}
        coord_bins: The number of bins used for coordinate quantization.

    Returns:
        A single string representing all formatted bounding boxes.
    """
    instance_strings = []
    if 'instances' in detection_data:
        for instance in detection_data['instances']:
            try:
                boxes = instance['boxes'] # [xmin_norm, ymin_norm, xmax_norm, ymax_norm]
                class_name = instance['class_name']

                # Normalize and bin coordinates
                # Ensure coordinates are within [0, 1] before binning
                xmin_norm = max(0.0, min(float(boxes[0]), 1.0))
                ymin_norm = max(0.0, min(float(boxes[1]), 1.0))
                xmax_norm = max(0.0, min(float(boxes[2]), 1.0))
                ymax_norm = max(0.0, min(float(boxes[3]), 1.0))

                # Bin coordinates
                xmin_binned = int(xmin_norm * (coord_bins -1)) # Max bin is coord_bins - 1
                ymin_binned = int(ymin_norm * (coord_bins -1))
                xmax_binned = int(xmax_norm * (coord_bins -1))
                ymax_binned = int(ymax_norm * (coord_bins -1))
                
                # Ensure binned coordinates are within valid range [0, coord_bins-1]
                xmin_binned = max(0, min(xmin_binned, coord_bins - 1))
                ymin_binned = max(0, min(ymin_binned, coord_bins - 1))
                xmax_binned = max(0, min(xmax_binned, coord_bins - 1))
                ymax_binned = max(0, min(ymax_binned, coord_bins - 1))


                # Format according to 4M conventions (v0=... v1=... etc.)
                # Class names with spaces should be handled (e.g. "hot dog" -> "hot_dog" or keep as is if tokenizer handles it)
                # The `train_unified_wordpiece_tokenizer` likely handles multi-word classes by tokenizing them.
                # Standard 4M object class tokens are usually single words or underscore separated.
                # Let's assume class_name is used as is, and the tokenizer will handle it.
                instance_str = f"v0={xmin_binned} v1={ymin_binned} v2={xmax_binned} v3={ymax_binned} {class_name}"
                instance_strings.append(instance_str)
            except (KeyError, IndexError, ValueError) as e:
                print(f"Warning: Skipping an instance due to malformed data: {instance}. Error: {e}")
                continue
    
    return " ".join(instance_strings) # Concatenate all instance strings with a space

def create_plan_sequence(
    caption_text: str,
    detection_data: Dict[str, List[Dict[str, Any]]],
    tokenizer: Tokenizer,
    max_seq_length: int = 512, # Should match modality_info.py for plan_sequence
    coord_bins: int = DEFAULT_COORD_BINS,
    add_eos_to_caption: bool = True,
    add_eos_to_bboxes: bool = True,
    add_eos_to_plan: bool = True
) -> List[int]:
    """
    Creates the target token ID sequence for the MCOT Planning task.
    Sequence: [PLANNING_START_TOKEN_ID] <tokenized_caption_ids> [EOS_ID_IF_ENABLED] <tokenized_bbox_string_ids> [EOS_ID_IF_ENABLED] [EOS_ID_IF_ENABLED_FOR_OVERALL_PLAN]

    Args:
        caption_text: The raw caption string.
        detection_data: Parsed JSON content from a .json detection file.
        tokenizer: The MCOT tokenizer instance.
        max_seq_length: The maximum allowed length for the tokenized sequence (including special tokens).
        coord_bins: Number of bins for coordinate quantization.
        add_eos_to_caption: Whether to add an EOS token after the caption part.
        add_eos_to_bboxes: Whether to add an EOS token after the bounding box part.
        add_eos_to_plan: Whether to add an EOS token at the very end of the plan sequence.

    Returns:
        A list of token IDs representing the plan sequence, truncated if necessary.
    """
    planning_start_token_id = tokenizer.token_to_id(PLANNING_START_TOKEN)
    eos_token_id = tokenizer.token_to_id("[EOS]")

    if planning_start_token_id is None:
        raise ValueError(f"'{PLANNING_START_TOKEN}' not found in the tokenizer vocabulary.")
    if eos_token_id is None:
        raise ValueError("'[EOS]' not found in the tokenizer vocabulary.")

    # Tokenize caption
    # The tokenizer.encode() method automatically adds special tokens like [SOS] and [EOS] if configured to do so
    # during its training or if its a property of the specific tokenizer class.
    # For `tokenizers` library WordPiece model, `add_special_tokens=True` is default for encode.
    # We might want more control. Let's assume we don't want the default SOS from tokenizer.encode().
    caption_encoding = tokenizer.encode(caption_text, add_special_tokens=False)
    caption_token_ids = caption_encoding.ids
    if add_eos_to_caption:
        caption_token_ids += [eos_token_id]

    # Format and tokenize bounding boxes
    bbox_string = format_detection_string_for_plan(detection_data, coord_bins)
    bbox_encoding = tokenizer.encode(bbox_string, add_special_tokens=False)
    bbox_token_ids = bbox_encoding.ids
    if add_eos_to_bboxes and bbox_token_ids: # Only add EOS if there were bboxes
        bbox_token_ids += [eos_token_id]
    
    # Combine into plan sequence
    plan_token_ids = [planning_start_token_id] + caption_token_ids + bbox_token_ids
    
    if add_eos_to_plan:
        plan_token_ids += [eos_token_id]

    # Truncate if exceeds max_seq_length
    if len(plan_token_ids) > max_seq_length:
        # Truncate from the end, but always keep PLANNING_START_TOKEN
        plan_token_ids = plan_token_ids[:max_seq_length-1] + [eos_token_id] if add_eos_to_plan else plan_token_ids[:max_seq_length]
        if plan_token_ids[0] != planning_start_token_id and max_seq_length > 0 : # Ensure START token is not cut
             plan_token_ids = [planning_start_token_id] + plan_token_ids[1:max_seq_length-1] + ([eos_token_id] if add_eos_to_plan and max_seq_length > 1 else [])


    # Ensure first token is PLANNING_START_TOKEN if max_length allows
    if max_seq_length > 0 and plan_token_ids[0] != planning_start_token_id:
        # This case should ideally not happen if truncation is done carefully
        # or if max_seq_length is always > 0.
        # If it does, it indicates an issue with truncation or very small max_seq_length.
        # For safety, one could force it if space: plan_token_ids = [planning_start_token_id] + plan_token_ids[-(max_seq_length-1):]
        pass


    return plan_token_ids[:max_seq_length] # Final truncation

# --- Acting Stage Data Transformation ---

def create_acting_sequence(
    detection_data: Dict[str, List[Dict[str, Any]]],
    tokenizer: Tokenizer,
    max_seq_length: int = 512, # Should match modality_info.py for a detection-like sequence
    coord_bins: int = DEFAULT_COORD_BINS,
    add_eos_to_sequence: bool = True
) -> List[int]:
    """
    Creates the target token ID sequence for the MCOT Acting task.
    Sequence: [ACTING_START_TOKEN_ID] <tokenized_bbox_string_ids> [EOS_ID_IF_ENABLED]

    Args:
        detection_data: Parsed JSON content from a .json detection file.
        tokenizer: The MCOT tokenizer instance.
        max_seq_length: The maximum allowed length for the tokenized sequence.
        coord_bins: Number of bins for coordinate quantization.
        add_eos_to_sequence: Whether to add an EOS token at the very end of the sequence.

    Returns:
        A list of token IDs representing the acting sequence, truncated if necessary.
    """
    acting_start_token_id = tokenizer.token_to_id(ACTING_START_TOKEN)
    eos_token_id = tokenizer.token_to_id("[EOS]")

    if acting_start_token_id is None:
        raise ValueError(f"'{ACTING_START_TOKEN}' not found in the tokenizer vocabulary.")
    if eos_token_id is None:
        raise ValueError("'[EOS]' not found in the tokenizer vocabulary.")

    # Format and tokenize bounding boxes
    # We can reuse the same formatting function as for the planning stage's bbox part
    bbox_string = format_detection_string_for_plan(detection_data, coord_bins)
    bbox_encoding = tokenizer.encode(bbox_string, add_special_tokens=False)
    bbox_token_ids = bbox_encoding.ids
    
    # Combine into acting sequence
    acting_token_ids = [acting_start_token_id] + bbox_token_ids
    
    if add_eos_to_sequence and bbox_token_ids: # Only add EOS if there were bboxes and requested
        acting_token_ids += [eos_token_id]
    elif add_eos_to_sequence and not bbox_token_ids: # If no bboxes, but EOS requested, add it after start token
        acting_token_ids += [eos_token_id]


    # Truncate if exceeds max_seq_length
    if len(acting_token_ids) > max_seq_length:
        if add_eos_to_sequence:
            acting_token_ids = acting_token_ids[:max_seq_length-1] + [eos_token_id]
        else:
            acting_token_ids = acting_token_ids[:max_seq_length]
        
        # Ensure ACTING_START_TOKEN is present if max_length allows
        if max_seq_length > 0 and acting_token_ids[0] != acting_start_token_id:
            # This case implies max_seq_length is very small or an issue.
            # Force it if possible.
            if add_eos_to_sequence and max_seq_length > 1:
                 acting_token_ids = [acting_start_token_id] + acting_token_ids[1:max_seq_length-1] + [eos_token_id]
            elif not add_eos_to_sequence and max_seq_length > 0:
                 acting_token_ids = [acting_start_token_id] + acting_token_ids[1:max_seq_length]
            # If max_seq_length is 0 or 1 (and EOS needed), it might be impossible to keep both.
            # The outer truncation handles this.

    return acting_token_ids[:max_seq_length] # Final truncation


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # This is a placeholder for where your actual data root would be
    # In a real scenario, these paths would come from your dataset loading logic
    EXAMPLE_DATA_ROOT = Path("/work/com-304/my_mscoco_for_4m") # Corrected path
    # If running this script directly, ensure my_mscoco_for_4m exists relative to MCoT-4m-XL root,
    # or change EXAMPLE_DATA_ROOT to an absolute path like "/work/com-304/my_mscoco_for_4m"

    # Find an example caption and detection file from your processed dataset
    example_caption_file = None
    example_det_file = None
    
    train_caption_dir = EXAMPLE_DATA_ROOT / "caption" / "train"
    train_det_dir = EXAMPLE_DATA_ROOT / "det" / "train"

    if train_caption_dir.exists() and train_det_dir.exists():
        try:
            example_caption_file = next(train_caption_dir.glob("*.txt"))
            # Find a corresponding .json file (assuming image IDs match)
            example_det_file = train_det_dir / f"{example_caption_file.stem}.json"
            if not example_det_file.exists():
                example_det_file = None # Reset if corresponding det not found
        except StopIteration:
            pass # No files found

    if example_caption_file and example_det_file:
        print(f"Using example caption: {example_caption_file}")
        print(f"Using example detection: {example_det_file}")

        with open(example_caption_file, 'r', encoding='utf-8') as f:
            caption_text_content = f.read().strip()
        
        with open(example_det_file, 'r', encoding='utf-8') as f:
            detection_json_content = json.load(f)

        print(f"\\nOriginal Caption:\\n{caption_text_content}")
        print(f"\\nOriginal Detection JSON:\\n{json.dumps(detection_json_content, indent=2)}")

        mcot_tokenizer = load_mcot_tokenizer()
        
        formatted_bbox_str = format_detection_string_for_plan(detection_json_content, coord_bins=DEFAULT_COORD_BINS)
        print(f"\\nFormatted BBox String:\\n{formatted_bbox_str}")

        plan_sequence_ids = create_plan_sequence(
            caption_text_content,
            detection_json_content,
            mcot_tokenizer
        )
        print(f"\\nPlan Sequence Token IDs (length {len(plan_sequence_ids)}):\\n{plan_sequence_ids}")

        decoded_plan_sequence = mcot_tokenizer.decode(plan_sequence_ids, skip_special_tokens=False)
        print(f"\\nDecoded Plan Sequence (with special tokens):\\n{decoded_plan_sequence}")
        
        decoded_plan_sequence_skip_special = mcot_tokenizer.decode(plan_sequence_ids, skip_special_tokens=True)
        print(f"\\nDecoded Plan Sequence (skipping special tokens):\\n{decoded_plan_sequence_skip_special}")
        
        # Check if PLANNING_START_TOKEN is at the beginning
        planning_start_id = mcot_tokenizer.token_to_id(PLANNING_START_TOKEN)
        if plan_sequence_ids and plan_sequence_ids[0] == planning_start_id:
            print(f"\\nConfirmed: Plan sequence starts with {PLANNING_START_TOKEN} (ID: {planning_start_id})")
        else:
            print(f"\\nWarning: Plan sequence does NOT start with {PLANNING_START_TOKEN}. First token ID: {plan_sequence_ids[0] if plan_sequence_ids else 'N/A'}")

        print("\\n--- Testing Acting Sequence ---")
        acting_sequence_ids = create_acting_sequence(
            detection_json_content,
            mcot_tokenizer
        )
        print(f"\\nActing Sequence Token IDs (length {len(acting_sequence_ids)}):\\n{acting_sequence_ids}")

        decoded_acting_sequence = mcot_tokenizer.decode(acting_sequence_ids, skip_special_tokens=False)
        print(f"\\nDecoded Acting Sequence (with special tokens):\\n{decoded_acting_sequence}")
        
        decoded_acting_sequence_skip_special = mcot_tokenizer.decode(acting_sequence_ids, skip_special_tokens=True)
        print(f"\\nDecoded Acting Sequence (skipping special tokens):\\n{decoded_acting_sequence_skip_special}")

        # Check if ACTING_START_TOKEN is at the beginning
        acting_start_id = mcot_tokenizer.token_to_id(ACTING_START_TOKEN)
        if acting_sequence_ids and acting_sequence_ids[0] == acting_start_id:
            print(f"\\nConfirmed: Acting sequence starts with {ACTING_START_TOKEN} (ID: {acting_start_id})")
        else:
            print(f"\\nWarning: Acting sequence does NOT start with {ACTING_START_TOKEN}. First token ID: {acting_sequence_ids[0] if acting_sequence_ids else 'N/A'}")

    else:
        print("Could not find example caption/detection files for testing in:")
        print(f"  {train_caption_dir}")
        print(f"  {train_det_dir}")
        print("Please ensure your MSCOCO data is processed and paths are correct for __main__ example.") 