import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
import numpy as np
import subprocess # For calling save_vq_tokens.py

# --- Configuration ---
# Path to the 4M repository (adjust if your script is not in ml-4m/scripts/)
FOURM_ROOT = Path(__file__).resolve().parent.parent
MSCOCO_HF_DATASET_NAME = "shunk031/MSCOCO"
MSCOCO_INSTANCES_CONFIG = "2017-instances"
MSCOCO_CAPTIONS_CONFIG = "2017-captions"
OUTPUT_ROOT = FOURM_ROOT / "my_mscoco_for_4m" # Or your desired output path

# COCO category ID to class name mapping
# This can be loaded from COCO's annotation files or a predefined mapping.
# For simplicity, we'll use a placeholder. You should populate this correctly.
# You can find a relevant mapping in `fourm/utils/tokenizer/object_classes.json` for "coco"
COCO_CATEGORY_MAP_PATH = FOURM_ROOT / "fourm/utils/tokenizer/object_classes.json"
with open(COCO_CATEGORY_MAP_PATH, 'r') as f:
    ALL_OBJECT_CLASSES = json.load(f)
    COCO_CATEGORIES_LIST = ALL_OBJECT_CLASSES.get("coco", []) # List of class names

# Create a reverse map from class name to its index in the COCO_CATEGORIES_LIST,
# if your tokenizer expects class indices. However, DetectionTransform uses class names directly.
# The shunk031/MSCOCO dataset might provide category names directly or category_ids.
# If it provides category_ids, you'll need a map from those specific IDs to names.
# For now, we assume shunk031/MSCOCO provides 'category_name' or we can map 'category_id'.

# Placeholder for actual COCO ID -> Name mapping if needed from shunk031/MSCOCO annotations
# Example: coco_id_to_name = {1: 'person', 2: 'bicycle', ...}
# You'll need to inspect shunk031/MSCOCO 'segments_info' for how categories are provided.
# Let's assume 'segments_info' contains 'category_name' directly for now for simplicity.
# If it's 'category_id', you'll need to load the official COCO annotations to build this map.
# For `shunk031/MSCOCO`, the category names seem to be directly available in `segments_info[*]['category_name']`.

SPLITS = ["train", "validation"] # Adjust if shunk031/MSCOCO uses different split names like 'val'

# --- Helper Functions ---
def get_image_id(sample, split_name):
    """
    Extracts a unique image ID from the Hugging Face dataset sample.
    The 'id' field is usually a good candidate.
    For shunk031/MSCOCO, `sample['image_id']` seems to be the direct unique ID.
    """
    img_id = sample.get('image_id')
    if img_id is None:
        # Fallback if 'image_id' is not present, try 'id' or generate one
        img_id = sample.get('id', None)
        if img_id is None:
            # If no standard ID field, construct one from file_name or a hash.
            # This is a placeholder, ensure shunk031/MSCOCO has a consistent ID.
            file_name = sample.get('file_name', f"unknown_image_{random.randint(0, 1e9)}")
            img_id = Path(file_name).stem
    return str(img_id)


def process_dataset_split(hf_dataset_split, split_name, output_root):
    """
    Processes a single split of the dataset (e.g., 'train' or 'validation').
    """
    # Create output directories
    rgb_dir = output_root / "rgb" / split_name
    caption_dir = output_root / "caption" / split_name
    det_dir = output_root / "det" / split_name

    rgb_dir.mkdir(parents=True, exist_ok=True)
    caption_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {split_name} split...")
    for sample in tqdm(hf_dataset_split):
        image_id = get_image_id(sample, split_name)
        pil_image = sample['image'] # This is a PIL.Image object

        # 1. Save RGB Image
        image_path = rgb_dir / f"{image_id}.jpg"
        if not image_path.exists(): # Avoid re-saving if already processed
            pil_image.save(image_path)

        # 2. Save Caption
        # shunk031/MSCOCO provides annotations in 'annotations', which is a list (usually one item)
        # containing 'caption' (a string, not a list of 5 for this dataset structure)
        # and 'segments_info' (list of objects).
        # If it had multiple captions, you'd select one:
        # captions = sample.get('annotations', [{}])[0].get('caption', ["No caption available."])
        # caption_to_save = captions[0] if isinstance(captions, list) and captions else str(captions)
        
        # For shunk031/MSCOCO, 'annotations' is a list of dicts, each dict is an annotation for THAT image.
        # Each dict has 'caption' and 'segments_info'.
        # We'll take the caption from the first annotation object.
        annotations = sample.get('annotations')
        if not annotations or not isinstance(annotations, list) or not annotations[0]:
            print(f"Warning: No annotations found for image_id {image_id}. Skipping caption and det.")
            caption_to_save = "No caption available."
            object_instances_for_det = []
        else:
            # Assuming the first annotation object is representative for captions and all segments.
            # In COCO, typically one annotation object (for captions task) might contain multiple segments.
            # Or, for object detection, multiple annotation objects might exist per image, one per instance.
            # The structure of `shunk031/MSCOCO` implies `annotations` is a list, take first.
            first_annotation = annotations[0]
            caption_to_save = first_annotation.get('caption', "No caption available.")
            
            # Prepare Bounding Boxes for 'det'
            object_instances_for_det = []
            if 'segments_info' in first_annotation and first_annotation['segments_info'] is not None:
                img_width, img_height = pil_image.size
                for segment_info in first_annotation['segments_info']:
                    if 'bbox' not in segment_info or 'category_name' not in segment_info: # or 'category_id'
                        # print(f"Warning: Bbox or category_name missing for an object in image {image_id}")
                        continue

                    bbox_abs = segment_info['bbox'] # [x_min, y_min, width, height]
                    class_name = segment_info['category_name'] # Already the name

                    x_abs, y_abs, w_abs, h_abs = bbox_abs
                    xmin_norm = x_abs / img_width
                    ymin_norm = y_abs / img_height
                    xmax_norm = (x_abs + w_abs) / img_width
                    ymax_norm = (y_abs + h_abs) / img_height

                    # Ensure coordinates are within [0, 1]
                    xmin_norm = max(0.0, min(xmin_norm, 1.0))
                    ymin_norm = max(0.0, min(ymin_norm, 1.0))
                    xmax_norm = max(0.0, min(xmax_norm, 1.0))
                    ymax_norm = max(0.0, min(ymax_norm, 1.0))
                    
                    # Ensure xmax > xmin and ymax > ymin
                    if xmax_norm <= xmin_norm: xmax_norm = xmin_norm + 1e-5 # Add a tiny epsilon
                    if ymax_norm <= ymin_norm: ymax_norm = ymin_norm + 1e-5 
                    xmax_norm = min(xmax_norm, 1.0) # Re-clip if epsilon pushed it over
                    ymax_norm = min(ymax_norm, 1.0)


                    object_instances_for_det.append({
                        'boxes': [xmin_norm, ymin_norm, xmax_norm, ymax_norm],
                        'class_name': class_name,
                        'score': 1.0  # Dummy score, as DetectionTransform might expect it
                    })
            else: # No segments_info
                pass # object_instances_for_det remains empty

        caption_path = caption_dir / f"{image_id}.txt"
        if not caption_path.exists():
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption_to_save)

        # 3. Save Bounding Box JSON for 'det'
        # The DetectionTransform expects a structure like {'instances': [{'boxes': [...], 'class_name': ..., 'score': ...}]}
        det_data_for_json = {'instances': object_instances_for_det}
        det_path = det_dir / f"{image_id}.json"
        if not det_path.exists() or object_instances_for_det: # Save if not exists or if there's data
            with open(det_path, 'w', encoding='utf-8') as f:
                json.dump(det_data_for_json, f)

    print(f"Finished processing {split_name} split.")

def tokenize_rgb_images(output_root, split_name, fourm_root_path):
    """
    Calls the save_vq_tokens.py script to pre-tokenize RGB images.
    """
    print(f"Starting RGB tokenization for {split_name} split...")
    rgb_images_path = output_root / "rgb" # save_vq_tokens expects the parent of the split folder
    
    # Ensure the script path is correct
    # Assuming save_vq_tokens.py is in the root of the ml-4m directory
    save_tokens_script_path = fourm_root_path / "save_vq_tokens.py"

    if not save_tokens_script_path.exists():
        print(f"Error: save_vq_tokens.py not found at {save_tokens_script_path}")
        print("Please ensure the script is in the root of the ml-4m repository.")
        return

    cmd = [
        "python", str(save_tokens_script_path),
        "--tokenizer_id", "EPFL-VILAB/4M_tokenizers_rgb_16k_224-448",
        "--tokenizers_root", str(fourm_root_path / "tokenizer_ckpts"), # Create this dir if it doesn't exist
        "--data_root", str(rgb_images_path.parent), # Pass the parent of 'rgb' so it finds 'rgb/train' or 'rgb/val'
        "--split", split_name, # This will be appended to data_root/rgb/
        "--task", "rgb", # The modality folder name inside data_root/split
        "--folder_suffix", "tok_rgb@224", # This will be the output folder name for tokens
        "--n_crops", "1", # Usually 1 for pre-tokenization, crop_settings will be center crop
        "--input_size", "224",
        "--num_workers", "8", # Adjust as needed
        "--batch_size_dataloader", "64", # Adjust as needed
        "--batch_size", "256", # GPU batch size for tokenization
        "--verbose"
    ]
    # Create tokenizer_ckpts directory if it doesn't exist, as save_vq_tokens might expect it
    (fourm_root_path / "tokenizer_ckpts").mkdir(exist_ok=True)
    
    # The save_vq_tokens.py script will save tokens to:
    # data_root / task_tok_rgb@224 / split / image_id.npy
    # e.g. ./my_mscoco_for_4m/rgb_tok_rgb@224/train/image_id.npy
    # We want it in ./my_mscoco_for_4m/tok_rgb@224/{split}/
    # So, we need to adjust either the script or move files after.
    # For now, let's assume we'll move/rename.
    # A simpler way: save_vq_tokens will create `rgb_tok_rgb@224`. We want `tok_rgb@224`.
    # Let's set `data_root` to `output_root` and task to `rgb`.
    # Then output will be `output_root/rgb_tok_rgb@224/...`
    # It's easier to set `tokens_dir` inside save_vq_tokens.py to just use `args.folder_suffix`
    # Assuming current save_vq_tokens.py saves to data_root / f"{args.task}_{args.folder_suffix}"
    # We need the output to be output_root / args.folder_suffix / split_name
    # The current save_vq_tokens.py script saves tokens to: os.path.join(root, tokens_dir) where tokens_dir is task + folder_suffix
    # and root is data_root / split. This is a bit convoluted.
    # Let's assume save_vq_tokens.py can be made to save to a direct output path or we post-process.

    # Simpler approach for save_vq_tokens.py:
    # It saves into `os.path.join(args.data_root, args.split, f'{args.task}_{args.folder_suffix}')`
    # We want tokens in `OUTPUT_ROOT / "tok_rgb@224" / split_name`
    # And crop_settings in `OUTPUT_ROOT / "crop_settings" / split_name`
    # Current `save_vq_tokens.py` saves tokens to `os.path.join(root, tokens_dir)` where `root` is `data_root/split` and `tokens_dir` is `task_folder_suffix`.
    # So, `data_root/split/task_folder_suffix`.
    # Let data_root be `OUTPUT_ROOT`. task is `rgb`. split is `split_name`. folder_suffix is `tok_rgb@224`.
    # Output tokens path from script: `OUTPUT_ROOT / split_name / rgb_tok_rgb@224`
    # Output crop settings path from script: `OUTPUT_ROOT / split_name / crop_settings`
    # This is not ideal.

    # The `save_vq_tokens.py` script is structured to take a `data_root` which contains modality folders like `rgb`.
    # It then creates an output folder like `rgb_tok_rgb@224` inside `data_root/split/`.
    # We want the output to be `my_mscoco_for_4m/tok_rgb@224/{split}`.
    # Let's adjust the call to save_vq_tokens.py:
    # data_root will be OUTPUT_ROOT
    # task will be 'rgb'
    # split will be split_name
    # folder_suffix will be 'tok_rgb@224' (this is the target output dir name, not a suffix to task)
    # The script should be modified to save into `data_root / folder_suffix / split`
    # For now, I'll write the command assuming `save_vq_tokens.py` is flexible or we rename after.
    # The most straightforward way with current save_vq_tokens.py:
    # It saves tokens into `os.path.join(root, tokens_dir)` which is `os.path.join(data_root, split, task + "_" + folder_suffix)`
    # And crop_settings into `os.path.join(root, crop_settings_dir)` which is `os.path.join(data_root, split, "crop_settings")`
    # Let `data_root` be `OUTPUT_ROOT`.
    # It will create `OUTPUT_ROOT / split_name / rgb_tok_rgb@224`
    # and `OUTPUT_ROOT / split_name / crop_settings`

    cmd_tokenize = [
        "python", str(save_tokens_script_path),
        "--tokenizer_id", "EPFL-VILAB/4M_tokenizers_rgb_16k_224-448",
        "--tokenizers_root", str(fourm_root_path / "tokenizer_ckpts"),
        "--data_root", str(output_root), # e.g., ./my_mscoco_for_4m
        "--split", split_name,           # e.g., train
        "--task", "rgb",                 # Modality folder to process under data_root/split
        "--folder_suffix", "tok_rgb@224_temp", # Temporary suffix
        "--n_crops", "1",
        "--input_size", "224",
        "--num_workers", "4", # Adjust based on your machine
        "--batch_size_dataloader", "32",
        "--batch_size", "128", # GPU batch size
        "--verbose"
    ]
    print(f"Running tokenization command: {' '.join(cmd_tokenize)}")
    try:
        subprocess.run(cmd_tokenize, check=True)
        
        # Post-tokenization: Move files to the desired structure
        # Tokens were saved to: OUTPUT_ROOT / split_name / rgb_tok_rgb@224_temp
        # Crop settings were saved to: OUTPUT_ROOT / split_name / crop_settings
        
        temp_token_dir = output_root / split_name / "rgb_tok_rgb@224_temp"
        target_token_dir = output_root / "tok_rgb@224" / split_name
        target_token_dir.mkdir(parents=True, exist_ok=True)

        if temp_token_dir.exists():
            for class_folder in temp_token_dir.iterdir():
                if class_folder.is_dir(): # MSCOCO does not have class folders for images
                    target_class_folder = target_token_dir # Save directly into split folder
                    # target_class_folder.mkdir(parents=True, exist_ok=True) # Not needed
                    for file_path in class_folder.glob("*.npy"):
                        # rename to target_token_dir / file_path.name (not target_class_folder)
                        file_path.rename(target_token_dir / file_path.name)
            # Clean up temp directory structure
            # Assuming save_vq_tokens directly puts .npy files into temp_token_dir if no classes
            if not any(temp_token_dir.iterdir()): # if it's empty after moving files
                 temp_token_dir.rmdir()
            else: # If there were class folders, remove them
                for class_folder in temp_token_dir.iterdir():
                    if class_folder.is_dir():
                         for file_path in class_folder.glob("*.npy"): # move any remaining
                            file_path.rename(target_token_dir / file_path.name)
                         class_folder.rmdir() # Remove now empty class folder
                if not any(temp_token_dir.iterdir()): # Check again
                    temp_token_dir.rmdir()

            print(f"Tokens moved to {target_token_dir}")
        else:
            print(f"Warning: Expected temporary token directory {temp_token_dir} not found after tokenization.")

        # Crop settings were saved to OUTPUT_ROOT / split_name / crop_settings
        # Move them to OUTPUT_ROOT / crop_settings / split_name
        temp_crop_dir = output_root / split_name / "crop_settings"
        target_crop_dir = output_root / "crop_settings" / split_name
        target_crop_dir.mkdir(parents=True, exist_ok=True)

        if temp_crop_dir.exists():
            # Similar logic for crop_settings, assuming they might also be in class folders by save_vq_tokens.py
            # Or directly in temp_crop_dir
            if any(p.is_dir() for p in temp_crop_dir.iterdir()): # Check for class subfolders
                for class_folder in temp_crop_dir.iterdir():
                    if class_folder.is_dir():
                        # target_class_folder_crop = target_crop_dir / class_folder.name # if we want to preserve class structure
                        # target_class_folder_crop.mkdir(parents=True, exist_ok=True)
                        for file_path in class_folder.glob("*.npy"):
                            file_path.rename(target_crop_dir / file_path.name) # Flatten into target_crop_dir/split_name
                # Clean up temp crop directory structure (class folders)
                for class_folder in temp_crop_dir.iterdir():
                    if class_folder.is_dir():
                        class_folder.rmdir()
            else: # Files are directly in temp_crop_dir
                 for file_path in temp_crop_dir.glob("*.npy"):
                    file_path.rename(target_crop_dir / file_path.name)

            if not any(temp_crop_dir.iterdir()): # if it's empty after moving files
                temp_crop_dir.rmdir()
            print(f"Crop settings moved to {target_crop_dir}")
        else:
            print(f"Warning: Expected temporary crop settings directory {temp_crop_dir} not found.")


    except subprocess.CalledProcessError as e:
        print(f"Error during tokenization for {split_name}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during tokenization for {split_name}: {e}")

    print(f"Finished RGB tokenization for {split_name} split.")


# --- Main Script ---
if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load the Hugging Face datasets
    print(f"Loading Hugging Face dataset: {MSCOCO_HF_DATASET_NAME} with config '{MSCOCO_INSTANCES_CONFIG}' for instances...")
    try:
        coco_dataset_instances_hf = load_dataset(MSCOCO_HF_DATASET_NAME, MSCOCO_INSTANCES_CONFIG)
    except Exception as e:
        print(f"Failed to load INSTANCES dataset {MSCOCO_HF_DATASET_NAME} ({MSCOCO_INSTANCES_CONFIG}). Error: {e}")
        exit(1)
    print("Instances dataset loaded successfully.")

    print(f"Loading Hugging Face dataset: {MSCOCO_HF_DATASET_NAME} with config '{MSCOCO_CAPTIONS_CONFIG}' for captions...")
    try:
        coco_dataset_captions_hf = load_dataset(MSCOCO_HF_DATASET_NAME, MSCOCO_CAPTIONS_CONFIG)
    except Exception as e:
        print(f"Failed to load CAPTIONS dataset {MSCOCO_HF_DATASET_NAME} ({MSCOCO_CAPTIONS_CONFIG}). Error: {e}")
        exit(1)
    print("Captions dataset loaded successfully.")

    # --- BEGIN INSPECTION CODE ---
    print("\n----------------------------------------")
    print(f"INSPECTING FIRST SAMPLE FROM '{MSCOCO_INSTANCES_CONFIG}' (train split):")
    if 'train' in coco_dataset_instances_hf:
        try:
            first_sample_instances = next(iter(coco_dataset_instances_hf['train']))
            print("  Instance Sample keys:", list(first_sample_instances.keys()))
            if 'image_id' in first_sample_instances:
                print(f"    Image ID: {first_sample_instances['image_id']}")
            if 'annotations' in first_sample_instances:
                print("    'annotations' key found in INSTANCE sample.")
                print("      Type of 'annotations':", type(first_sample_instances['annotations']))
                if isinstance(first_sample_instances['annotations'], list) and len(first_sample_instances['annotations']) > 0:
                    print("      First item in INSTANCE annotations:", first_sample_instances['annotations'][0])
                    if isinstance(first_sample_instances['annotations'][0], dict):
                        print("        Keys in first INSTANCE annotation item:", list(first_sample_instances['annotations'][0].keys()))
                elif isinstance(first_sample_instances['annotations'], dict):
                    print("      INSTANCE 'annotations' is a DICT. Keys:", list(first_sample_instances['annotations'].keys()))
            # Add more specific checks for bbox/category if needed based on mscoco.py (InstanceExample)
            if 'file_name' in first_sample_instances: print(f"    File Name: {first_sample_instances['file_name']}")

        except StopIteration:
            print("  Could not get a sample from INSTANCES 'train' split (it might be empty).")
        except Exception as e:
            print(f"  Error during INSTANCES sample inspection: {e}")
    else:
        print("  Warning: 'train' split not found in coco_dataset_instances_hf.")
    print("----------------------------------------")

    print(f"INSPECTING FIRST SAMPLE FROM '{MSCOCO_CAPTIONS_CONFIG}' (train split):")
    if 'train' in coco_dataset_captions_hf:
        try:
            first_sample_captions = next(iter(coco_dataset_captions_hf['train']))
            print("  Caption Sample keys:", list(first_sample_captions.keys()))
            if 'image_id' in first_sample_captions:
                print(f"    Image ID: {first_sample_captions['image_id']}")
            if 'annotations' in first_sample_captions:
                print("    'annotations' key found in CAPTION sample.")
                print("      Type of 'annotations':", type(first_sample_captions['annotations']))
                if isinstance(first_sample_captions['annotations'], list) and len(first_sample_captions['annotations']) > 0:
                    print("      First item in CAPTION annotations:", first_sample_captions['annotations'][0])
                    if isinstance(first_sample_captions['annotations'][0], dict):
                         print("        Keys in first CAPTION annotation item:", list(first_sample_captions['annotations'][0].keys()))
                         if 'caption' in first_sample_captions['annotations'][0]:
                            print(f"          Caption text: {first_sample_captions['annotations'][0]['caption'][:100]}...")
            # Check for a simpler 'captions' list if present
            if 'captions' in first_sample_captions:
                 print(f"    Found 'captions' (plural) key in CAPTION sample. Content: {first_sample_captions['captions']}")
            if 'file_name' in first_sample_captions: print(f"    File Name: {first_sample_captions['file_name']}")

        except StopIteration:
            print("  Could not get a sample from CAPTIONS 'train' split (it might be empty).")
        except Exception as e:
            print(f"  Error during CAPTIONS sample inspection: {e}")
    else:
        print("  Warning: 'train' split not found in coco_dataset_captions_hf.")
    print("----------------------------------------")
    print("Exiting after inspection. Please review the output above to adapt parsing logic.")
    exit()
    # --- END INSPECTION CODE ---

    # Process each split
    # This part will need significant rework to use both datasets
    # For now, we are exiting above. The logic below is placeholder.
    for split_key_hf in coco_dataset_instances_hf.keys(): # e.g., 'train', 'validation', 'test'
        if split_key_hf not in SPLITS: # Allow users to define which HF splits map to their 'train'/'val'
            print(f"Skipping Hugging Face split: {split_key_hf} as it's not in defined SPLITS: {SPLITS}")
            continue
        
        # Determine the 4M split name (e.g. 'validation' -> 'val')
        # For this script, we assume direct mapping if names are in SPLITS
        current_4m_split_name = split_key_hf # e.g. "train" or "validation"
        
        process_dataset_split(coco_dataset_instances_hf[split_key_hf], current_4m_split_name, OUTPUT_ROOT)

    # After processing images, captions, and det, run tokenization for RGB images
    # This needs to be done *after* images are saved by process_dataset_split
    for split_name_for_tokenization in SPLITS: # Iterate through the desired 4M splits
         # Check if the corresponding rgb data exists for this split
        if (OUTPUT_ROOT / "rgb" / split_name_for_tokenization).exists():
            tokenize_rgb_images(OUTPUT_ROOT, split_name_for_tokenization, FOURM_ROOT)
        else:
            print(f"Skipping tokenization for {split_name_for_tokenization} as {OUTPUT_ROOT / 'rgb' / split_name_for_tokenization} does not exist.")

    print("--------------------------------------------------------------------")
    print(f"Data preparation finished. Output at: {OUTPUT_ROOT}")
    print("Final directory structure should be:")
    print(f"{OUTPUT_ROOT}/")
    print("├── rgb/")
    print("│   ├── train/ (.jpg files)")
    print("│   └── validation/ (.jpg files)")
    print("├── caption/")
    print("│   ├── train/ (.txt files)")
    print("│   └── validation/ (.txt files)")
    print("├── det/")
    print("│   ├── train/ (.json files)")
    print("│   └── validation/ (.json files)")
    print("├── tok_rgb@224/")
    print("│   ├── train/ (.npy files)")
    print("│   └── validation/ (.npy files)")
    print("└── crop_settings/ (created by save_vq_tokens.py, moved here)")
    print("    ├── train/ (.npy files)")
    print("    └── validation/ (.npy files)")
    print("--------------------------------------------------------------------") 