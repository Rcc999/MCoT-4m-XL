import os
import json
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import subprocess

# --- Configuration ---
FOURM_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = Path("/work/com-304/my_mscoco_for_4m")
DOWNLOAD_DIR = OUTPUT_ROOT / "downloads"
EXTRACT_DIR = OUTPUT_ROOT / "extracted_coco_data"

# URLs for COCO 2017 dataset
URL_TRAIN_IMAGES = "http://images.cocodataset.org/zips/train2017.zip"
URL_VAL_IMAGES = "http://images.cocodataset.org/zips/val2017.zip"
URL_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# We'll use 'train' and 'validation' as split names internally, matching 4M structure.
# COCO uses 'train2017' and 'val2017' in its file names and folder structures.
SPLITS_CONFIG = {
    "train": {
        "images_zip_url": URL_TRAIN_IMAGES,
        "images_zip_name": "train2017.zip",
        "images_folder_in_zip": "train2017", # Extracted image folder name
        "annotations_file": "annotations/instances_train2017.json",
        "captions_file": "annotations/captions_train2017.json"
    },
    "validation": {
        "images_zip_url": URL_VAL_IMAGES,
        "images_zip_name": "val2017.zip",
        "images_folder_in_zip": "val2017", # Extracted image folder name
        "annotations_file": "annotations/instances_val2017.json",
        "captions_file": "annotations/captions_val2017.json"
    }
}
# Common annotations zip (contains files for both train and val)
ANNOTATIONS_ZIP_URL = URL_ANNOTATIONS
ANNOTATIONS_ZIP_NAME = "annotations_trainval2017.zip"

# COCO category map from fourm utils (though we'll also get it from annotations file)
COCO_CATEGORY_MAP_PATH = FOURM_ROOT / "fourm/utils/tokenizer/object_classes.json"
# Fallback if issues reading from file, but primary source should be COCO annotation file.
try:
    with open(COCO_CATEGORY_MAP_PATH, 'r') as f:
        ALL_OBJECT_CLASSES = json.load(f)
        FALLBACK_COCO_CATEGORIES_LIST = ALL_OBJECT_CLASSES.get("coco", [])
except Exception:
    FALLBACK_COCO_CATEGORIES_LIST = []


# --- Helper Functions ---

def run_command(command_list, cwd=None):
    """Runs a shell command and checks for errors."""
    print(f"Running command: {' '.join(command_list)}")
    try:
        subprocess.run(command_list, check=True, cwd=cwd)
        print(f"Command successful: {' '.join(command_list)}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command_list)}")
        print(f"Error: {e}")
        raise # Re-raise the exception to stop the script

def download_file(url, target_dir):
    """Downloads a file using wget."""
    target_dir.mkdir(parents=True, exist_ok=True)
    file_name = Path(url).name
    target_file_path = target_dir / file_name
    if not target_file_path.exists():
        print(f"Downloading {url} to {target_file_path}...")
        run_command(["wget", url, "-P", str(target_dir)])
    else:
        print(f"File {target_file_path} already exists. Skipping download.")
    return target_file_path

def extract_zip(zip_path, target_dir):
    """Extracts a zip file."""
    target_dir.mkdir(parents=True, exist_ok=True)
    # Check if extraction target (e.g., specific folder from zip) already exists to avoid re-extraction
    # This is a simple check; more robust would be to check for a sentinel file or specific content
    # For COCO, the image zips extract into folders like 'train2017', 'val2017'
    # The annotations zip extracts into an 'annotations' folder
    expected_extraction_folder_name = ""
    if "train2017" in zip_path.name:
        expected_extraction_folder_name = "train2017"
    elif "val2017" in zip_path.name:
        expected_extraction_folder_name = "val2017"
    elif "annotations_trainval2017" in zip_path.name:
        expected_extraction_folder_name = "annotations"

    if expected_extraction_folder_name and (target_dir / expected_extraction_folder_name).exists():
         print(f"Content from {zip_path.name} likely already extracted to {target_dir / expected_extraction_folder_name}. Skipping extraction.")
         return

    print(f"Extracting {zip_path} to {target_dir}...")
    run_command(["unzip", "-o", str(zip_path), "-d", str(target_dir)]) # -o for overwrite

def process_coco_split(split_name_4m, config, output_root_4m, coco_extract_dir):
    """
    Processes a single split (train/validation) using manually parsed COCO JSONs.
    split_name_4m: 'train' or 'validation' (for 4M structure)
    config: The SPLITS_CONFIG entry for this split
    output_root_4m: The base output directory (e.g., my_mscoco_for_4m)
    coco_extract_dir: Directory where COCO data was extracted (e.g., EXTRACT_DIR)
    """
    print(f"Processing {split_name_4m} split...")

    # Paths for 4M output structure
    rgb_dir_4m = output_root_4m / "rgb" / split_name_4m
    caption_dir_4m = output_root_4m / "caption" / split_name_4m
    det_dir_4m = output_root_4m / "det" / split_name_4m
    rgb_dir_4m.mkdir(parents=True, exist_ok=True)
    caption_dir_4m.mkdir(parents=True, exist_ok=True)
    det_dir_4m.mkdir(parents=True, exist_ok=True)

    # Paths to extracted COCO annotation files
    instances_ann_path = coco_extract_dir / config["annotations_file"]
    captions_ann_path = coco_extract_dir / config["captions_file"]
    
    # Path to extracted COCO images for this split
    coco_images_dir_for_split = coco_extract_dir / config["images_folder_in_zip"]

    if not instances_ann_path.exists() or not captions_ann_path.exists():
        print(f"Error: Annotation files not found for {split_name_4m}:")
        print(f"  Instances: {instances_ann_path} (exists: {instances_ann_path.exists()})")
        print(f"  Captions: {captions_ann_path} (exists: {captions_ann_path.exists()})")
        return

    if not coco_images_dir_for_split.exists():
        print(f"Error: Extracted COCO images directory not found for {split_name_4m}: {coco_images_dir_for_split}")
        return

    print(f"Loading instances annotations from: {instances_ann_path}")
    with open(instances_ann_path, 'r') as f:
        instances_data = json.load(f)
    
    print(f"Loading captions annotations from: {captions_ann_path}")
    with open(captions_ann_path, 'r') as f:
        captions_data = json.load(f)

    # Build mappings
    image_id_to_image_info = {img['id']: img for img in instances_data['images']}
    
    category_id_to_name = {cat['id']: cat['name'] for cat in instances_data.get('categories', [])}
    if not category_id_to_name:
        print("Warning: Could not load categories from instances annotation file. Attempting fallback (less ideal).")
        # This fallback is less reliable as IDs might not match what COCO JSON provides.
        # It's better to ensure 'categories' is in your instances_*.json file.
        category_id_to_name = {i: name for i, name in enumerate(FALLBACK_COCO_CATEGORIES_LIST, 1)}


    image_id_to_captions_list = {}
    for ann in captions_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_captions_list:
            image_id_to_captions_list[img_id] = []
        image_id_to_captions_list[img_id].append(ann['caption'])

    image_id_to_instances_list = {}
    for ann in instances_data['annotations']:
        img_id = ann['image_id']
        if img_id not in image_id_to_instances_list:
            image_id_to_instances_list[img_id] = []
        image_id_to_instances_list[img_id].append(ann)

    # Process each image
    for image_id, image_info in tqdm(image_id_to_image_info.items(), desc=f"Processing {split_name_4m} images"):
        img_file_name = image_info['file_name']
        img_width = image_info['width']
        img_height = image_info['height']
        
        # Use string image_id for filenames, consistent with previous script versions
        str_image_id = str(image_id)

        # 1. RGB Image
        src_image_path = coco_images_dir_for_split / img_file_name
        target_image_path_4m = rgb_dir_4m / f"{str_image_id}.jpg"

        if not src_image_path.exists():
            print(f"Warning: Source image {src_image_path} not found. Skipping image {str_image_id}.")
            continue
        
        if not target_image_path_4m.exists():
            try:
                pil_image = Image.open(src_image_path)
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                pil_image.save(target_image_path_4m)
            except Exception as e:
                print(f"Warning: Could not process or save image {src_image_path}. Error: {e}. Skipping image {str_image_id}.")
                continue
        
        # 2. Caption
        captions_for_image = image_id_to_captions_list.get(image_id, ["No caption available."])
        caption_to_save = random.choice(captions_for_image) if captions_for_image else "No caption available."
        target_caption_path_4m = caption_dir_4m / f"{str_image_id}.txt"
        if not target_caption_path_4m.exists():
            with open(target_caption_path_4m, 'w', encoding='utf-8') as f:
                f.write(caption_to_save)

        # 3. Detections
        object_instances_for_det = []
        raw_instances_for_image = image_id_to_instances_list.get(image_id, [])
        for inst_ann in raw_instances_for_image:
            coco_bbox = inst_ann['bbox'] # [x_min, y_min, width, height]
            category_id = inst_ann['category_id']
            class_name = category_id_to_name.get(category_id, "unknown")

            x_abs, y_abs, w_abs, h_abs = coco_bbox
            xmin_norm = x_abs / img_width
            ymin_norm = y_abs / img_height
            xmax_norm = (x_abs + w_abs) / img_width
            ymax_norm = (y_abs + h_abs) / img_height

            # Ensure coordinates are within [0, 1] and valid
            xmin_norm = max(0.0, min(xmin_norm, 1.0))
            ymin_norm = max(0.0, min(ymin_norm, 1.0))
            xmax_norm = max(0.0, min(xmax_norm, 1.0))
            ymax_norm = max(0.0, min(ymax_norm, 1.0))
            
            if xmax_norm <= xmin_norm: xmax_norm = xmin_norm + 1e-5
            if ymax_norm <= ymin_norm: ymax_norm = ymin_norm + 1e-5
            xmax_norm = min(xmax_norm, 1.0)
            ymax_norm = min(ymax_norm, 1.0)

            object_instances_for_det.append({
                'boxes': [xmin_norm, ymin_norm, xmax_norm, ymax_norm],
                'class_name': class_name,
                'score': 1.0 # Dummy score
            })
        
        det_data_for_json = {'instances': object_instances_for_det}
        target_det_path_4m = det_dir_4m / f"{str_image_id}.json"
        # Save even if empty, as 4M might expect a file for every image.
        # Or only save if object_instances_for_det is not empty. Let's save if not exists, or if data.
        if not target_det_path_4m.exists() or object_instances_for_det:
            with open(target_det_path_4m, 'w', encoding='utf-8') as f:
                json.dump(det_data_for_json, f)

    print(f"Finished processing {split_name_4m} split.")


def tokenize_rgb_images(output_root_4m, split_name_4m, fourm_root_path):
    """Calls the save_vq_tokens.py script to pre-tokenize RGB images."""
    print(f"Starting RGB tokenization for {split_name_4m} split...")
    
    save_tokens_script_path = fourm_root_path / "save_vq_tokens.py"
    if not save_tokens_script_path.exists():
        print(f"Error: save_vq_tokens.py not found at {save_tokens_script_path}")
        return

    # Tokenizer checkpoints directory (save_vq_tokens might need it)
    tokenizer_ckpts_dir = fourm_root_path / "tokenizer_ckpts"
    tokenizer_ckpts_dir.mkdir(exist_ok=True)

    cmd_tokenize = [
        "python", str(save_tokens_script_path),
        "--tokenizer_id", "EPFL-VILAB/4M_tokenizers_rgb_16k_224-448",
        "--tokenizers_root", str(tokenizer_ckpts_dir),
        "--data_root", str(output_root_4m), # e.g., ./my_mscoco_for_4m
        "--split", split_name_4m,        # e.g., train
        "--task", "rgb",                  # Modality folder to process under data_root/split
        "--folder_suffix", "tok_rgb@224_temp", # Temporary suffix for output folder
        "--n_crops", "1",
        "--input_size", "224",
        "--num_workers", "1",
        "--batch_size_dataloader", "32",
        "--batch_size", "128", 
        "--verbose"
    ]
    print(f"Running tokenization command: {' '.join(cmd_tokenize)}")
    try:
        run_command(cmd_tokenize)
        
        # Post-tokenization: Move files to the desired structure
        # Tokens were saved by save_vq_tokens.py to a path like:
        # OUTPUT_ROOT / <split_name> / <task_name>_<folder_suffix_temp> / <class_name> / <file_id>.npy
        # For us: OUTPUT_ROOT / split_name / "rgb_tok_rgb@224_temp" / "rgb" / <file_id>.npy (since class_name is also rgb for flat structure)
        temp_token_base_dir = output_root_4m / split_name_4m / f"rgb_tok_rgb@224_temp" # e.g. /work/.../my_mscoco_for_4m/train/rgb_tok_rgb@224_temp
        target_token_base_dir = output_root_4m / "tok_rgb@224" / split_name_4m # e.g. /work/.../my_mscoco_for_4m/tok_rgb@224/train

        if temp_token_base_dir.exists():
            target_token_base_dir.mkdir(parents=True, exist_ok=True)
            # Iterate through class_name subdirectories (e.g., the 'rgb' folder if coco is flat)
            for class_folder_in_temp in temp_token_base_dir.iterdir():
                if class_folder_in_temp.is_dir():
                    target_class_folder_final = target_token_base_dir / class_folder_in_temp.name
                    target_class_folder_final.mkdir(parents=True, exist_ok=True)
                    for file_path in class_folder_in_temp.glob("*.npy"):
                        file_path.rename(target_class_folder_final / file_path.name)
                    
                    # Attempt to remove the now-empty class folder from temp
                    try:
                        if not any(class_folder_in_temp.iterdir()): # Check if directory is empty
                            class_folder_in_temp.rmdir()
                    except OSError: pass # Ignore if not empty or other permission issues
            
            # Attempt to remove the temp_token_base_dir if it's now empty
            try:
                if not any(temp_token_base_dir.iterdir()): # Check if directory is empty
                    temp_token_base_dir.rmdir()
            except OSError: pass
            print(f"Tokens moved from {temp_token_base_dir} to {target_token_base_dir}")
        else:
            print(f"Warning: Expected temporary token directory {temp_token_base_dir} not found after tokenization.")

        # Move crop settings
        # Crop settings were saved by save_vq_tokens.py to:
        # OUTPUT_ROOT / <split_name> / "crop_settings" / <task_name> / <class_name> / <file_id>.npy
        # For us: OUTPUT_ROOT / split_name / "crop_settings" / "rgb" / "rgb" / <file_id>.npy
        
        actual_temp_crop_settings_location = output_root_4m / split_name_4m / "crop_settings" / "rgb" / "rgb" # This is .../crop_settings/rgb
        target_crop_base_dir = output_root_4m / "crop_settings" / split_name_4m # This is .../crop_settings/train

        if actual_temp_crop_settings_location.exists() and actual_temp_crop_settings_location.is_dir():
            target_crop_base_dir.mkdir(parents=True, exist_ok=True)
            # Now, actual_temp_crop_settings_location points to the <task_name> folder (e.g., 'rgb')
            # Inside this, there is the <class_name> folder (also 'rgb' for us)
            for class_folder_in_temp_crop in actual_temp_crop_settings_location.iterdir(): # Iterates over <class_name> folders, e.g. 'rgb'
                if class_folder_in_temp_crop.is_dir():
                    # class_folder_in_temp_crop.name will be 'rgb' (the class name)
                    final_target_for_class_crops = target_crop_base_dir / class_folder_in_temp_crop.name
                    final_target_for_class_crops.mkdir(parents=True, exist_ok=True)
                    
                    for file_path in class_folder_in_temp_crop.glob("*.npy"):
                        file_path.rename(final_target_for_class_crops / file_path.name)
                    
                    # Attempt to remove the now-empty class folder from temp
                    try:
                        if not any(class_folder_in_temp_crop.iterdir()): 
                            class_folder_in_temp_crop.rmdir()
                    except OSError: pass
            
            # Attempt to remove the <task_name> folder (actual_temp_crop_settings_location) if it's empty
            try:
                if not any(actual_temp_crop_settings_location.iterdir()):
                    actual_temp_crop_settings_location.rmdir()
            except OSError: pass

            # Attempt to remove the parent "crop_settings" folder in the temp location if it's empty
            parent_of_actual_temp_crop_settings = actual_temp_crop_settings_location.parent
            try:
                if parent_of_actual_temp_crop_settings.exists() and not any(parent_of_actual_temp_crop_settings.iterdir()):
                    parent_of_actual_temp_crop_settings.rmdir()
            except OSError: pass

            print(f"Crop settings moved from {actual_temp_crop_settings_location.parent} to {target_crop_base_dir.parent}")
        else:
            print(f"Warning: Expected temporary crop task directory {actual_temp_crop_settings_location} not found.")

    except Exception as e: # Catch subprocess.CalledProcessError or others
        print(f"An error occurred during tokenization for {split_name_4m}: {e}")

    print(f"Finished RGB tokenization for {split_name_4m} split.")


# --- Main Script ---
if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print("--- Starting MSCOCO Data Preparation (Manual Download & Parse) ---")

    # Check if initial processing seems complete to allow skipping
    initial_processing_assumed_complete = True
    for split_4m in SPLITS_CONFIG.keys():
        if not (OUTPUT_ROOT / "rgb" / split_4m).exists() or \
           not (OUTPUT_ROOT / "caption" / split_4m).exists() or \
           not (OUTPUT_ROOT / "det" / split_4m).exists():
            initial_processing_assumed_complete = False
            break

    if initial_processing_assumed_complete:
        print("Initial data (rgb, caption, det) directories already exist for all splits. Skipping download, extraction, and initial processing.")
    else:
        print("Initial data directories not found for all splits. Proceeding with full download, extraction, and processing.")
        # 1. Download and Extract Annotations (common for train/val)
        print("\n--- Downloading and Extracting Annotations ---")
        annotations_zip_path = download_file(ANNOTATIONS_ZIP_URL, DOWNLOAD_DIR)
        extract_zip(annotations_zip_path, EXTRACT_DIR) # Extracts to EXTRACT_DIR/annotations/

        # 2. Download, Extract, and Process each split
        for split_4m, config in SPLITS_CONFIG.items():
            print(f"\n--- Downloading and Extracting Images for {split_4m} ---")
            images_zip_path = download_file(config["images_zip_url"], DOWNLOAD_DIR)
            extract_zip(images_zip_path, EXTRACT_DIR) # Extracts to EXTRACT_DIR/train2017 or EXTRACT_DIR/val2017
            
            print(f"\n--- Processing data for {split_4m} split ---")
            process_coco_split(split_4m, config, OUTPUT_ROOT, EXTRACT_DIR)

    # 3. Tokenize RGB images for each split (this will always run if initial processing is done or skipped)
    for split_4m in SPLITS_CONFIG.keys():
        print(f"\n--- Tokenizing RGB images for {split_4m} split ---")
        # Ensure the rgb data exists for this split before tokenizing
        if (OUTPUT_ROOT / "rgb" / split_4m).exists():
            tokenize_rgb_images(OUTPUT_ROOT, split_4m, FOURM_ROOT)
        else:
            print(f"Skipping tokenization for {split_4m} as {OUTPUT_ROOT / 'rgb' / split_4m} does not exist.")
    
    print("\n--------------------------------------------------------------------")
    print(f"Data preparation finished. Output at: {OUTPUT_ROOT}")
    print("Final directory structure should be:")
    print(f"{OUTPUT_ROOT}/")
    print("├── rgb/\n│   ├── train/ (.jpg files)\n│   └── validation/ (.jpg files)")
    print("├── caption/\n│   ├── train/ (.txt files)\n│   └── validation/ (.txt files)")
    print("├── det/\n│   ├── train/ (.json files)\n│   └── validation/ (.json files)")
    print("├── tok_rgb@224/\n│   ├── train/ (.npy files)\n│   └── validation/ (.npy files)")
    print("└── crop_settings/\n    ├── train/ (.npy files)\n    └── validation/ (.npy files)")
    print("Also, check for temporary download/extraction folders:")
    print(f"  Downloads: {DOWNLOAD_DIR}")
    print(f"  Extractions: {EXTRACT_DIR}")
    print("--------------------------------------------------------------------") 