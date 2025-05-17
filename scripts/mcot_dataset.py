import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

# Assuming mcot_data_utils.py is in the same directory or PYTHONPATH is set up
try:
    from mcot_data_utils import (
        load_mcot_tokenizer,
        create_plan_sequence,
        # create_acting_sequence, # No longer directly used for main acting input
        MCOT_TOKENIZER_PATH, # Default tokenizer path
        DEFAULT_COORD_BINS   # Default coordinate bins
    )
except ImportError:
    # Fallback for cases where the script might be run from a different context
    # This assumes MCoT-4m-XL is the root and scripts/ is in PYTHONPATH
    # Or that mcot_data_utils is discoverable
    print("Attempting to import MCOT data utils from parent dir if running as script")
    from .mcot_data_utils import (
        load_mcot_tokenizer,
        create_plan_sequence,
        # create_acting_sequence,
        MCOT_TOKENIZER_PATH,
        DEFAULT_COORD_BINS
    )


logger = logging.getLogger(__name__)


class MCoTDataset(Dataset):
    """
    PyTorch Dataset for MCOT tasks using preprocessed MSCOCO data.
    Loads image VQ tokens, captions, and detection data, then generates
    plan and acting sequences using the MCOT tokenizer and utility functions.
    """
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        tokenizer_path: str = MCOT_TOKENIZER_PATH,
        plan_max_seq_length: int = 512,
        acting_max_seq_length: int = 768, # Max length for combined acting input: [ACT_START]<img_toks><plan_toks>
        coord_bins: int = DEFAULT_COORD_BINS,
        load_image_tokens: bool = True # Flag to control loading of image tokens
    ):
        """
        Args:
            data_root (str): Path to the root directory of the processed MSCOCO data 
                             (e.g., /work/com-304/my_mscoco_for_4m).
            split (str): Dataset split, 'train' or 'val'.
            tokenizer_path (str): Path to the trained MCOT text tokenizer JSON file.
            plan_max_seq_length (int): Maximum sequence length for the plan target sequence.
            acting_max_seq_length (int): Maximum sequence length for the combined acting input sequence.
            coord_bins (int): Number of bins for coordinate quantization in detection strings.
            load_image_tokens (bool): Whether to load image tokens. Useful for tasks
                                      that might only need text data.
        """
        self.data_root = Path(data_root)
        self.split = split
        self.tokenizer_path = tokenizer_path
        self.plan_max_seq_length = plan_max_seq_length
        self.acting_max_seq_length = acting_max_seq_length
        self.coord_bins = coord_bins
        self.load_image_tokens = load_image_tokens

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")

        # Load tokenizer with robust fallbacks
        try:
            self.tokenizer = load_mcot_tokenizer(self.tokenizer_path)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {self.tokenizer_path}: {e}")
            # Try alternate paths if the first one fails
            alternate_paths = [
                "./tokenizer_ckpts/text_tokenizer_4m_mcot.json",
                "./fourm/utils/tokenizer/trained/text_tokenizer_4m_mcot.json",
                "/home/rcharif/MCoT-4m-XL/tokenizer_ckpts/text_tokenizer_4m_mcot.json"
            ]
            
            for alt_path in alternate_paths:
                try:
                    logger.info(f"Trying alternate tokenizer path: {alt_path}")
                    self.tokenizer = load_mcot_tokenizer(alt_path)
                    logger.info(f"Successfully loaded tokenizer from {alt_path}")
                    break
                except Exception as e2:
                    logger.warning(f"Failed to load from {alt_path}: {e2}")
            else:
                # If all attempts fail, raise the original error
                raise ValueError(f"Could not load tokenizer from any path. Original error: {e}")

        actual_split_name = 'validation' if self.split == 'val' else self.split
        logger.info(f"Using actual split directory name: {actual_split_name} for input split '{self.split}'")

        self.caption_dir = self.data_root / "caption" / actual_split_name
        self.det_dir = self.data_root / "det" / actual_split_name
        self.img_tok_dir = self.data_root / "tok_rgb@224" / actual_split_name

        if not self.caption_dir.exists():
            raise FileNotFoundError(f"Caption directory not found: {self.caption_dir}")
        if not self.det_dir.exists():
            raise FileNotFoundError(f"Detection directory not found: {self.det_dir}")
        if self.load_image_tokens and not self.img_tok_dir.exists():
            raise FileNotFoundError(f"Image token directory not found: {self.img_tok_dir}")

        # Use caption files as the source of truth for sample IDs
        self.sample_ids = sorted([f.stem for f in self.caption_dir.glob("*.txt")])

        if not self.sample_ids:
            raise RuntimeError(f"No samples found in {self.caption_dir}. Check data and paths.")
        
        logger.info(f"Initialized MCoTDataset for split '{self.split}' with {len(self.sample_ids)} samples.")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample_id = self.sample_ids[idx]

        caption_file = self.caption_dir / f"{sample_id}.txt"
        det_file = self.det_dir / f"{sample_id}.json"
        
        try:
            with open(caption_file, 'r', encoding='utf-8') as f:
                caption_text = f.read().strip()
            
            with open(det_file, 'r', encoding='utf-8') as f:
                detection_data = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Error loading data for sample ID {sample_id}: {e}")
            raise e 
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for sample ID {sample_id} ({det_file}): {e}")
            raise e

        # Generate plan sequence (target for planning stage)
        # This is [PLAN_START_ID] <tokenized_caption_ids> <tokenized_bbox_string_ids>
        planning_target_sequence_ids = create_plan_sequence(
            caption_text=caption_text,
            detection_data=detection_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.plan_max_seq_length,
            coord_bins=self.coord_bins
        )
        
        item = {
            'image_id': sample_id,
            'raw_caption': caption_text, # For inspection
            'planning_target_sequence': torch.tensor(planning_target_sequence_ids, dtype=torch.long),
        }

        # Load image tokens (input for planning, part of input for acting, target for acting)
        planning_input_image_tokens_tensor = None
        if self.load_image_tokens:
            img_tok_file = self.img_tok_dir / f"{sample_id}.npy"
            try:
                image_tokens_np = np.load(img_tok_file) 
                planning_input_image_tokens_tensor = torch.tensor(image_tokens_np, dtype=torch.long).squeeze()
                item['planning_input_image_tokens'] = planning_input_image_tokens_tensor
                item['acting_target_image_tokens'] = planning_input_image_tokens_tensor # Target for acting is image reconstruction
            except FileNotFoundError:
                logger.warning(f"Image token file not found for {sample_id} at {img_tok_file}, skipping image-dependent items.")
            except Exception as e:
                logger.error(f"Error loading image tokens for {sample_id} from {img_tok_file}: {e}")

        # Construct acting input sequence: [ACTING_START_ID] <image_tokens> <plan_sequence_tokens>
        # This is only possible if image tokens were loaded.
        if planning_input_image_tokens_tensor is not None:
            act_start_token_id = self.tokenizer.token_to_id("[ACTING_START]")
            if act_start_token_id is None:
                raise ValueError("[ACTING_START] token not found in tokenizer. Please check your MCOT tokenizer.")

            # Convert planning_input_image_tokens_tensor to a list of IDs if it's a tensor
            img_tok_list = planning_input_image_tokens_tensor.tolist()
            
            # planning_target_sequence_ids is already a list of IDs
            acting_input_ids = [act_start_token_id] + img_tok_list + planning_target_sequence_ids
            
            # Truncate if necessary
            if len(acting_input_ids) > self.acting_max_seq_length:
                acting_input_ids = acting_input_ids[:self.acting_max_seq_length]
                # Ensure EOS is present if it was truncated and should have one (logic depends on how EOS is handled)
                # For now, simple truncation. EOS handling for concatenated sequences can be complex.
                # The original plan/acting sequences from mcot_data_utils handle their own EOS.

            item['acting_input_sequence'] = torch.tensor(acting_input_ids, dtype=torch.long)
        
        return item

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting MCoTDataset test...")

    # Adjust this path to your actual processed MSCOCO data root
    # TEST_DATA_ROOT = "/path/to/your/my_mscoco_for_4m" 
    TEST_DATA_ROOT = "/work/com-304/my_mscoco_for_4m" # Example path
    
    # Ensure the tokenizer path is correct if not using default from mcot_data_utils
    # TEST_TOKENIZER_PATH = str(Path(__file__).resolve().parent.parent / "fourm/utils/tokenizer/trained/text_tokenizer_4m_mcot.json")
    TEST_TOKENIZER_PATH = MCOT_TOKENIZER_PATH


    if not Path(TEST_DATA_ROOT).exists() or not Path(TEST_TOKENIZER_PATH).exists():
        logger.error(f"Test data root ({TEST_DATA_ROOT}) or tokenizer path ({TEST_TOKENIZER_PATH}) does not exist. Skipping test.")
    else:
        logger.info(f"Attempting to load 'val' split from: {TEST_DATA_ROOT}")
        try:
            # Test with a small split like 'val' first if available
            mcot_val_dataset = MCoTDataset(
                data_root=TEST_DATA_ROOT,
                split='val', # Using 'val' for potentially smaller size and faster test
                tokenizer_path=TEST_TOKENIZER_PATH,
                plan_max_seq_length=128, # Smaller max length for testing
                acting_max_seq_length=128 # Smaller max length for testing
            )
            logger.info(f"Successfully instantiated MCoTDataset for 'val' split. Number of samples: {len(mcot_val_dataset)}")

            if len(mcot_val_dataset) > 0:
                logger.info("Fetching the first sample from 'val' split...")
                first_sample = mcot_val_dataset[0]
                logger.info(f"First sample ID: {first_sample['image_id']}")
                logger.info(f"Raw caption: {first_sample['raw_caption']}")
                if 'planning_input_image_tokens' in first_sample:
                    logger.info(f"Planning Input Image tokens shape: {first_sample['planning_input_image_tokens'].shape}")
                    logger.info(f"Planning Input Image tokens dtype: {first_sample['planning_input_image_tokens'].dtype}")
                else:
                    logger.info("Planning Input Image tokens not loaded or not found for the first sample.")
                
                logger.info(f"Planning Target Sequence (first 10): {first_sample['planning_target_sequence'][:10]}...")
                logger.info(f"Planning Target Sequence shape: {first_sample['planning_target_sequence'].shape}")
                logger.info(f"Planning Target Sequence dtype: {first_sample['planning_target_sequence'].dtype}")
                
                if 'acting_input_sequence' in first_sample:
                    logger.info(f"Acting Input Sequence (first 10): {first_sample['acting_input_sequence'][:10]}...")
                    logger.info(f"Acting Input Sequence shape: {first_sample['acting_input_sequence'].shape}")
                    logger.info(f"Acting Input Sequence dtype: {first_sample['acting_input_sequence'].dtype}")
                else:
                    logger.info("Acting Input Sequence not generated (likely missing image tokens).")

                if 'acting_target_image_tokens' in first_sample:
                    logger.info(f"Acting Target Image tokens shape: {first_sample['acting_target_image_tokens'].shape}")
                else:
                    logger.info("Acting Target Image tokens not available.")

                # You can add more checks here, e.g., decoding the sequences
                plan_decoded = mcot_val_dataset.tokenizer.decode(first_sample['planning_target_sequence'].tolist(), skip_special_tokens=False)
                logger.info(f"Decoded Plan Target (first sample, with special tokens): {plan_decoded}")

                if 'acting_input_sequence' in first_sample:
                    act_input_decoded = mcot_val_dataset.tokenizer.decode(first_sample['acting_input_sequence'].tolist(), skip_special_tokens=False)
                    logger.info(f"Decoded Acting Input (first sample, with special tokens): {act_input_decoded}")
                
                # Remove old acting_target_ids logging
                # logger.info(f\"Acting target IDs ( первые 10 ): {first_sample[\'acting_target_ids\'][:10]}...\")
                # logger.info(f\"Acting target IDs shape: {first_sample[\'acting_target_ids\'].shape}\")
                # logger.info(f\"Acting target IDs dtype: {first_sample[\'acting_target_ids\'].dtype}\")
                # act_decoded = mcot_val_dataset.tokenizer.decode(first_sample[\'acting_target_ids\'].tolist(), skip_special_tokens=False)
                # logger.info(f\"Decoded Acting (first sample, with special tokens): {act_decoded}\")

            # Test with 'train' split if 'val' worked
            logger.info(f"Attempting to load 'train' split from: {TEST_DATA_ROOT}")
            mcot_train_dataset = MCoTDataset(
                data_root=TEST_DATA_ROOT,
                split='train',
                tokenizer_path=TEST_TOKENIZER_PATH
            )
            logger.info(f"Successfully instantiated MCoTDataset for 'train' split. Number of samples: {len(mcot_train_dataset)}")
            if len(mcot_train_dataset) > 0:
                 logger.info("Fetching the first sample from 'train' split...")
                 train_first_sample = mcot_train_dataset[0]
                 logger.info(f"Train First sample ID: {train_first_sample['image_id']}")


        except FileNotFoundError as e:
            logger.error(f"Test failed due to FileNotFoundError: {e}")
            logger.error("Please ensure your data_root, split, and tokenizer_path are correct and data is prepared.")
        except Exception as e:
            logger.error(f"An unexpected error occurred during dataset testing: {e}", exc_info=True)

    logger.info("MCoTDataset test finished.") 