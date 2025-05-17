import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

try:
    # Assuming mcot_dataset.py is in the same directory or PYTHONPATH is set up
    from mcot_dataset import MCoTDataset, MCOT_TOKENIZER_PATH
    from mcot_data_utils import load_mcot_tokenizer
except ImportError:
    # Fallback for local execution if scripts/ is not directly in PYTHONPATH
    from .mcot_dataset import MCoTDataset, MCOT_TOKENIZER_PATH
    from .mcot_data_utils import load_mcot_tokenizer

logger = logging.getLogger(__name__)

# Determine [PAD] token ID
# This should ideally come from a config or be passed, but for now, we infer it.
# Based on train_mcot_tokenizer.py, [PAD] is usually the second special token (ID 1 after [UNK]=0).
# We load the tokenizer here to get the pad_token_id dynamically.
try:
    tokenizer_for_pad_id = load_mcot_tokenizer(MCOT_TOKENIZER_PATH)
    PAD_TOKEN_ID = tokenizer_for_pad_id.token_to_id("[PAD]")
    if PAD_TOKEN_ID is None:
        logger.warning("[PAD] token not found in tokenizer, defaulting to 1. This might be incorrect.")
        PAD_TOKEN_ID = 1 
except FileNotFoundError:
    logger.warning(f"Tokenizer file not found at {MCOT_TOKENIZER_PATH} for determining PAD_ID. Defaulting to 1. This might be incorrect.")
    PAD_TOKEN_ID = 1


def mcot_collate_fn(batch: list) -> dict:
    """
    Custom collate function for MCoTDataset.
    Pads 'planning_target_sequence' and 'acting_input_sequence'.
    Handles potentially missing image tokens for 'planning_input_image_tokens' 
    and 'acting_target_image_tokens'.
    """
    image_ids = [item['image_id'] for item in batch]
    raw_captions = [item.get('raw_caption', '') for item in batch]

    # Pad planning_target_sequence
    planning_target_sequence_list = [item['planning_target_sequence'] for item in batch]
    padded_planning_target_sequence = pad_sequence(planning_target_sequence_list, batch_first=True, padding_value=PAD_TOKEN_ID)

    # Pad acting_input_sequence (if present)
    acting_input_sequence_list = [item['acting_input_sequence'] for item in batch if 'acting_input_sequence' in item]
    padded_acting_input_sequence = None
    if acting_input_sequence_list:
        padded_acting_input_sequence = pad_sequence(acting_input_sequence_list, batch_first=True, padding_value=PAD_TOKEN_ID)

    collated_batch = {
        'image_id': image_ids,
        'raw_caption': raw_captions,
        'planning_target_sequence': padded_planning_target_sequence,
    }
    if padded_acting_input_sequence is not None:
        collated_batch['acting_input_sequence'] = padded_acting_input_sequence

    # Handle image tokens (planning_input_image_tokens and acting_target_image_tokens)
    # These should be present or absent together if load_image_tokens was true/false during dataset init
    # or if a specific file was missing.
    if 'planning_input_image_tokens' in batch[0]: # Check if the first item has the key
        planning_img_tokens_list = []
        acting_target_img_tokens_list = [] # Should be identical to planning_img_tokens_list
        valid_image_token_indices = [] 

        for i, item in enumerate(batch):
            if 'planning_input_image_tokens' in item and item['planning_input_image_tokens'] is not None:
                planning_img_tokens_list.append(item['planning_input_image_tokens'])
                # acting_target_image_tokens should also be present if planning_input_image_tokens is
                if 'acting_target_image_tokens' in item and item['acting_target_image_tokens'] is not None:
                    acting_target_img_tokens_list.append(item['acting_target_image_tokens'])
                else: # Should not happen if dataset logic is correct
                    logger.warning(f"acting_target_image_tokens missing for item {item['image_id']} when planning_input_image_tokens present.")
                valid_image_token_indices.append(i)

        if planning_img_tokens_list:
            try:
                collated_batch['planning_input_image_tokens'] = torch.stack(planning_img_tokens_list)
                if acting_target_img_tokens_list and len(acting_target_img_tokens_list) == len(planning_img_tokens_list):
                    collated_batch['acting_target_image_tokens'] = torch.stack(acting_target_img_tokens_list)
                elif acting_target_img_tokens_list: # Mismatch, log warning
                     logger.warning("Mismatch in presence/count of planning_input vs acting_target image tokens during collation.")
                
                collated_batch['image_tokens_present_mask'] = torch.tensor(valid_image_token_indices, dtype=torch.long)
            except RuntimeError as e:
                logger.error(f"Error stacking image_tokens: {e}. Shapes: {[it.shape for it in planning_img_tokens_list]}")
        # else: No image tokens in any sample of this batch with the key, or all were None
            
    return collated_batch


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting MCoT DataLoader test...")

    # Adjust this path to your actual processed MSCOCO data root
    TEST_DATA_ROOT = "/work/com-304/my_mscoco_for_4m"
    
    if not Path(TEST_DATA_ROOT).exists():
        logger.error(f"Test data root ({TEST_DATA_ROOT}) does not exist. Skipping DataLoader test.")
    else:
        logger.info(f"Attempting to load 'val' split from: {TEST_DATA_ROOT} for DataLoader test.")
        try:
            mcot_val_dataset = MCoTDataset(
                data_root=TEST_DATA_ROOT,
                split='val',
                tokenizer_path=MCOT_TOKENIZER_PATH, # Use default from mcot_dataset
                plan_max_seq_length=128, 
                acting_max_seq_length=128 
            )
            logger.info(f"Successfully instantiated MCoTDataset for 'val' split. Number of samples: {len(mcot_val_dataset)}")

            if len(mcot_val_dataset) > 0:
                val_dataloader = DataLoader(
                    mcot_val_dataset,
                    batch_size=4, # Small batch size for testing
                    shuffle=False, # No need to shuffle for this test
                    collate_fn=mcot_collate_fn,
                    num_workers=0 # Using 0 workers for simplicity in testing; can increase later
                )
                logger.info("Successfully created DataLoader for 'val' split.")
                logger.info(f"PAD_TOKEN_ID used for collation: {PAD_TOKEN_ID}")

                num_batches_to_check = 2
                for i, batch_data in enumerate(val_dataloader):
                    if i >= num_batches_to_check:
                        break
                    logger.info(f"--- Batch {i+1} ---")
                    logger.info(f"Image IDs: {batch_data['image_id']}")
                    # logger.info(f"Raw Captions: {batch_data['raw_caption']}") # Can be long
                    logger.info(f"Planning Target Sequence shape: {batch_data['planning_target_sequence'].shape}")
                    logger.info(f"Planning Target Sequence dtype: {batch_data['planning_target_sequence'].dtype}")
                    
                    if 'acting_input_sequence' in batch_data:
                        logger.info(f"Acting Input Sequence shape: {batch_data['acting_input_sequence'].shape}")
                        logger.info(f"Acting Input Sequence dtype: {batch_data['acting_input_sequence'].dtype}")
                    else:
                        logger.info("Acting Input Sequence not present in this batch (or not generated for any sample).")
                    
                    if 'planning_input_image_tokens' in batch_data:
                        logger.info(f"Planning Input Image Tokens shape: {batch_data['planning_input_image_tokens'].shape}")
                        logger.info(f"Planning Input Image Tokens dtype: {batch_data['planning_input_image_tokens'].dtype}")
                        if 'acting_target_image_tokens' in batch_data:
                            logger.info(f"Acting Target Image Tokens shape: {batch_data['acting_target_image_tokens'].shape}")
                        logger.info(f"Image Tokens present mask: {batch_data.get('image_tokens_present_mask')}")
                    else:
                        logger.info("Image Tokens (planning_input/acting_target) not present in this batch.")
                    
                    # Example: check padding on the first sample of the batch for planning_target_sequence
                    first_plan_seq = batch_data['planning_target_sequence'][0]
                    num_pad_tokens = (first_plan_seq == PAD_TOKEN_ID).sum()
                    logger.info(f"First planning_target_sequence in batch (last 10 tokens): ...{first_plan_seq[-10:].tolist()}")
                    logger.info(f"Number of PAD tokens in first planning_target_sequence: {num_pad_tokens} (total length: {len(first_plan_seq)})")

        except FileNotFoundError as e:
            logger.error(f"DataLoader test failed due to FileNotFoundError: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during DataLoader testing: {e}", exc_info=True)

    logger.info("MCoT DataLoader test finished.") 