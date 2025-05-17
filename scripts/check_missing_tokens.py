import logging
from pathlib import Path

try:
    from mcot_dataset import MCoTDataset, MCOT_TOKENIZER_PATH
except ImportError:
    from .mcot_dataset import MCoTDataset, MCOT_TOKENIZER_PATH # Fallback for local execution

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def count_missing_image_tokens(data_root: str, split: str, tokenizer_path: str) -> tuple[int, int]:
    """
    Counts the number of samples missing their .npy image token files for a given split.

    Args:
        data_root: Path to the root of the processed dataset.
        split: Dataset split ('train' or 'val').
        tokenizer_path: Path to the MCOT tokenizer.

    Returns:
        A tuple (total_samples, missing_files_count).
    """
    logger.info(f"Checking split: {split}...")
    try:
        dataset = MCoTDataset(
            data_root=data_root,
            split=split,
            tokenizer_path=tokenizer_path,
            load_image_tokens=False # We don't need to load them, just check paths
        )
    except Exception as e:
        logger.error(f"Could not initialize MCoTDataset for split '{split}'. Error: {e}")
        return 0, 0

    total_samples = len(dataset)
    missing_files_count = 0

    if total_samples == 0:
        logger.warning(f"No samples found for split '{split}'.")
        return 0, 0

    for i in range(total_samples):
        sample_id = dataset.sample_ids[i]
        # Construct the expected path to the .npy file directly
        # dataset.img_tok_dir already incorporates the correct mapping for 'val' to 'validation'
        img_tok_file_path = dataset.img_tok_dir / f"{sample_id}.npy"
        
        if not img_tok_file_path.exists():
            missing_files_count += 1
            # logger.debug(f"Missing token file for split '{split}', ID '{sample_id}': {img_tok_file_path}") # Optional: log every missing file
    
    logger.info(f"Split '{split}': Total samples = {total_samples}, Missing image token files = {missing_files_count}")
    return total_samples, missing_files_count

if __name__ == '__main__':
    logger.info("Starting check for missing image VQ tokens...")

    DATA_ROOT = "/work/com-304/my_mscoco_for_4m"
    TOKENIZER_PATH = MCOT_TOKENIZER_PATH # Use the one defined in mcot_dataset, ultimately from mcot_data_utils

    if not Path(DATA_ROOT).exists():
        logger.error(f"Data root directory {DATA_ROOT} not found. Aborting check.")
    else:
        overall_total_samples = 0
        overall_missing_files = 0

        for split_name in ['train', 'val']:
            total, missing = count_missing_image_tokens(DATA_ROOT, split_name, TOKENIZER_PATH)
            overall_total_samples += total
            overall_missing_files += missing
            logger.info(f"Summary for '{split_name}': {missing}/{total} image token files missing ({((missing/total)*100 if total > 0 else 0):.2f}% missing).")
            logger.info("---")

        if overall_total_samples > 0:
            percentage_missing_overall = (overall_missing_files / overall_total_samples) * 100
            logger.info(f"Overall Summary: {overall_missing_files}/{overall_total_samples} total image token files missing ({percentage_missing_overall:.2f}% missing).")
        else:
            logger.info("No samples found in any split to check.")

    logger.info("Finished check for missing image VQ tokens.") 