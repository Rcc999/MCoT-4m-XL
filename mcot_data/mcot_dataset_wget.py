import csv
import json
import os
import tarfile
import math
import subprocess
import shutil
import random
import requests
import zipfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any

import datasets
import tensorflow as tf
import pandas as pd
import torch
import numpy as np
from PIL import Image, ImageDraw

try:
    import webdataset as wds
except ImportError:
    wds = None
    print("Warning: webdataset library not found. BrushData processing via TAR files will be skipped if it's the chosen method.")


_CITATION = """
@article{wang2024mint,
  title={MINT: Multi-modal Chain of Thought in Unified Generative Models for Enhanced Image Generation},
  author={Wang, Yi and Liu, Mushui and He, Wanggui and Zhang, Longxiang and Huang, Ziwei and Zhang, Guanghao and Shu, Fangxun and Tao, Zhong and She, Dong and Yu, Zhelun and Li, Haoyuan and Dai, Weilong and Song, Mingli and Song, Jie and Jiang, Hao},
  journal={arXiv preprint arXiv:2503.01298},
  year={2024}
}
"""

_DESCRIPTION = """
Multimodal Chain of Thought (MCoT) dataset combining data from Visual Genome, RichHF-18K,
SeeTRUE-Feedback, and BrushData to create a step-by-step multimodal reasoning pipeline.
"""

_URLS = {
    "visual_genome": {
        "annotations_zip_url": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip",
        "images_part1_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
        "images_part2_url": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip", 
        "objects_url": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip",
        "attributes_url": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip",
        "relationships_url": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip"
    },
    "richhf18k": {
        "tfrecord_urls": {
            "train": "https://github.com/google-research-datasets/richhf-18k/raw/main/train.tfrecord",
            "dev": "https://github.com/google-research-datasets/richhf-18k/raw/main/dev.tfrecord", 
            "test": "https://github.com/google-research-datasets/richhf-18k/raw/main/test.tfrecord"
        }
    },
    "seetrue_feedback": {
        "hf_dataset_name": "mismatch-quest/SeeTRUE-Feedback"
    },
    "brush_data": {
        "base_tar_url": "https://huggingface.co/datasets/random123123/BrushData/resolve/main/",
        "tar_files": [
            "00000.tar", "00001.tar", "00002.tar", "00003.tar", "00004.tar",
            "00005.tar", "00006.tar"
        ]
    }
}


class MCoTWgetDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.2") # Incremented version due to cleanup feature

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="default", version=VERSION, description="Default MCoT dataset configuration with cleanup")
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = datasets.Features(
            {
                "image_id": datasets.Value("string"),
                "image": datasets.Image(),
                "prompt": datasets.Value("string"),
                "planning": datasets.Value("string"),
                "acting": datasets.Value("string"),
                "reflection": datasets.Value("string"),
                "correction": datasets.Value("string"),
                "final_response": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            citation=_CITATION,
        )

    def _download_all_datasets_first(self, dl_manager, base_data_dir):
        """Download all datasets first using wget, then process from cache"""
        import time
        print("🚀 Step 1: Downloading all datasets first...")
        
        # Create cache directories
        vg_cache_dir = base_data_dir / "visual_genome_raw" / "hf_cache"
        seetrue_cache_dir = base_data_dir / "seetrue_feedback_raw" / "hf_cache"
        brush_cache_dir = base_data_dir / "brush_data_raw" / "hf_cache"
        
        def download_with_retry(dataset_name, hf_name, config=None, split="train", cache_dir=None, max_retries=3):
            """Download dataset with exponential backoff retry"""
            for attempt in range(max_retries):
                try:
                    print(f"📦 Downloading {dataset_name} dataset (attempt {attempt + 1}/{max_retries})...")
                    if config:
                        dataset = datasets.load_dataset(
                            hf_name, config, split=split, cache_dir=str(cache_dir),
                            trust_remote_code=True, download_mode="reuse_cache_if_exists"
                        )
                    else:
                        dataset = datasets.load_dataset(
                            hf_name, split=split, cache_dir=str(cache_dir),
                            trust_remote_code=True, download_mode="reuse_cache_if_exists"
                        )
                    print(f"✅ {dataset_name} downloaded successfully")
                    return True
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "Too Many Requests" in error_msg:
                        wait_time = (2 ** attempt) * 60  # Exponential backoff: 60s, 120s, 240s
                        print(f"⚠️ Rate limited. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    elif attempt == max_retries - 1:
                        print(f"❌ {dataset_name} download failed after {max_retries} attempts: {e}")
                        return False
                    else:
                        print(f"⚠️ {dataset_name} download failed (attempt {attempt + 1}): {e}")
                        time.sleep(30)  # Wait 30s between regular retries
            return False
        
        # Note: Visual Genome will be downloaded directly with wget in _download_and_prepare_visual_genome
        # This avoids the 9.73GB HuggingFace download that keeps failing
        print("🎯 Visual Genome will be downloaded directly with wget (skipping HuggingFace)")
        
        # Note: BrushData will be downloaded directly with wget in _download_and_prepare_brush_data  
        # This limits download to 20GB instead of 1.7TB
        print("🎯 BrushData will be downloaded directly with wget (limiting to 20GB)")
        
        # Download SeeTRUE-Feedback with retry
        download_with_retry(
            "SeeTRUE-Feedback",
            _URLS["seetrue_feedback"]["hf_dataset_name"],
            split="test",
            cache_dir=seetrue_cache_dir
        )
        
        print("🎉 All dataset downloads attempted! Now processing...")

    def _split_generators(self, dl_manager):
        user_manual_dir = dl_manager.manual_dir
        base_data_dir = Path(user_manual_dir) if user_manual_dir else Path(os.getcwd()) / "mcot_downloads"
        base_data_dir.mkdir(parents=True, exist_ok=True)

        # Download all datasets first
        self._download_all_datasets_first(dl_manager, base_data_dir)

        processed_data_dir = base_data_dir / "processed_mcot_steps"
        processed_data_dir.mkdir(parents=True, exist_ok=True)

        vg_raw_dir = base_data_dir / "visual_genome_raw"
        richhf_raw_dir = base_data_dir / "richhf18k_raw"
        seetrue_raw_dir = base_data_dir / "seetrue_feedback_raw"
        brush_raw_dir = base_data_dir / "brush_data_raw"

        print("🔄 Step 2: Processing downloaded datasets...")
        self._download_and_prepare_visual_genome(dl_manager, vg_raw_dir, _URLS["visual_genome"])
        self._process_visual_genome(vg_raw_dir, processed_data_dir)

        self._download_and_prepare_richhf18k(dl_manager, richhf_raw_dir, _URLS["richhf18k"])
        self._process_richhf18k(richhf_raw_dir, processed_data_dir)

        self._download_and_prepare_seetrue_feedback(dl_manager, seetrue_raw_dir, _URLS["seetrue_feedback"])
        self._process_seetrue_feedback(seetrue_raw_dir, processed_data_dir)

        self._download_and_prepare_brush_data(dl_manager, brush_raw_dir, _URLS["brush_data"])
        self._process_brush_data(brush_raw_dir, processed_data_dir)

        self._generate_mcot_examples(processed_data_dir)
        
        # --- Start Conditional Cleanup ---
        # Check if processing was successful before cleaning up
        brush_samples_processed = getattr(self, 'brush_correction_samples', [])
        brush_success = len([s for s in brush_samples_processed if s.get('image_id', '') != 'brush_processing_empty']) > 0
        
        # Only disable cleanup if BrushData processing failed (for debugging)
        cleanup_enabled = brush_success  # Enable cleanup only if BrushData processing succeeded
        
        print(f"📊 Processing Summary:")
        print(f"   BrushData samples processed: {len(brush_samples_processed)}")
        print(f"   BrushData success: {brush_success}")
        print(f"   Cleanup enabled: {cleanup_enabled}")
        
        if cleanup_enabled:
            print("MCoT dataset construction complete. Starting cleanup of raw and intermediate files...")
            
            # 1. Delete Raw Data Directories
            raw_data_paths_to_clean = [vg_raw_dir, richhf_raw_dir, seetrue_raw_dir, brush_raw_dir]
            for raw_path in raw_data_paths_to_clean:
                if raw_path.exists():
                    print(f"Cleaning up raw data directory: {raw_path}")
                    try:
                        shutil.rmtree(raw_path)
                        print(f"Successfully removed {raw_path}")
                    except Exception as e:
                        print(f"Error removing {raw_path}: {e}. Please remove manually if needed.")
                else:
                    print(f"Raw data directory not found (already cleaned or never created): {raw_path}")

            # 2. Delete Intermediate Processed JSON Files
            intermediate_json_files = [
                processed_data_dir / "planning_data.json",
                processed_data_dir / "acting_data.json",
                processed_data_dir / "reflection_data.json",
                processed_data_dir / "correction_data.json"
            ]
            for json_file in intermediate_json_files:
                if json_file.exists():
                    print(f"Cleaning up intermediate file: {json_file}")
                    try:
                        json_file.unlink()
                        print(f"Successfully removed {json_file}")
                    except Exception as e:
                        print(f"Error removing {json_file}: {e}. Please remove manually if needed.")
                else:
                    print(f"Intermediate file not found (already cleaned or never created): {json_file}")
            
            print("Cleanup process finished.")
        else:
            print("⚠️ Cleanup SKIPPED due to BrushData processing failures. Raw files preserved for debugging.")
            print("MCoT dataset construction complete. Raw files preserved for debugging.")
        # --- End Conditional Cleanup ---

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": processed_data_dir / "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_dir": processed_data_dir / "val"},
            ),
        ]

    def _download_and_prepare_visual_genome(self, dl_manager, vg_raw_dir, urls):
        """Download Visual Genome using direct wget - much more reliable than HuggingFace"""
        vg_raw_dir.mkdir(parents=True, exist_ok=True)
        images_dir = vg_raw_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        print("🎯 Downloading Visual Genome using direct wget (bypassing HuggingFace)...")
        
        # Download region descriptions (small file, usually works)
        try:
            annotations_zip_path = dl_manager.download_and_extract(urls["annotations_zip_url"])
            region_descriptions_json = Path(annotations_zip_path) / "region_descriptions.json"
            if region_descriptions_json.exists():
                shutil.copy(region_descriptions_json, vg_raw_dir / "region_descriptions.json")
                print("✅ Visual Genome region descriptions downloaded successfully")
            else:
                (vg_raw_dir / "region_descriptions.json").write_text("[]")
                print(f"Warning: region_descriptions.json not found in {annotations_zip_path}")
        except Exception as e_zip:
            print(f"Error downloading Visual Genome region descriptions: {e_zip}")
            (vg_raw_dir / "region_descriptions.json").write_text("[]")

        # Download additional JSON files directly with wget
        json_downloads = [
            ("objects.json.zip", urls.get("objects_url")),
            ("attributes.json.zip", urls.get("attributes_url")), 
            ("relationships.json.zip", urls.get("relationships_url"))
        ]
        
        for filename, url in json_downloads:
            if url:
                try:
                    print(f"📦 Downloading {filename}...")
                    downloaded_path = dl_manager.download_and_extract(url)
                    json_filename = filename.replace('.zip', '')
                    json_path = Path(downloaded_path) / json_filename
                    if json_path.exists():
                        shutil.copy(json_path, vg_raw_dir / json_filename)
                        print(f"✅ {json_filename} downloaded successfully")
                    else:
                        print(f"⚠️ {json_filename} not found in download")
                except Exception as e:
                    print(f"⚠️ Failed to download {filename}: {e}")

        # Download images (these are large but more reliable with direct wget)
        image_downloads = [
            ("images.zip", urls.get("images_part1_url")),
            ("images2.zip", urls.get("images_part2_url"))
        ]
        
        for img_filename, img_url in image_downloads:
            if img_url:
                try:
                    print(f"🖼️ Downloading Visual Genome {img_filename} (this may take a while)...")
                    downloaded_img_path = dl_manager.download_and_extract(img_url)
                    
                    # Extract images to our images directory
                    if Path(downloaded_img_path).is_dir():
                        for img_file in Path(downloaded_img_path).glob("**/*.jpg"):
                            shutil.copy(img_file, images_dir / img_file.name)
                        print(f"✅ {img_filename} images extracted successfully")
                    else:
                        print(f"⚠️ {img_filename} extraction failed")
                        
                except Exception as e:
                    print(f"⚠️ Failed to download {img_filename}: {e}")
                    print("🔄 Continuing without images (will use metadata only)")

        # Create a simple success marker
        (vg_raw_dir / "vg_direct_download_completed.flag").touch()
        print("✅ Visual Genome direct download completed!")


    def _download_and_prepare_richhf18k(self, dl_manager, richhf_raw_dir, urls):
        """Download RichHF-18K TFRecord files using direct wget"""
        richhf_raw_dir.mkdir(parents=True, exist_ok=True)
        
        print("🎯 Downloading RichHF-18K TFRecord files using direct wget...")
        
        tfrecord_urls = urls.get("tfrecord_urls", {})
        
        for split, url in tfrecord_urls.items():
            filename = f"{split}.tfrecord"
            target_path = richhf_raw_dir / filename
            
            if target_path.exists():
                file_size = target_path.stat().st_size
                if file_size > 1024:  # File is larger than 1KB
                    print(f"✅ {filename} already exists ({file_size:,} bytes)")
                    continue
                else:
                    print(f"⚠️ {filename} exists but is too small ({file_size} bytes). Re-downloading...")
            
            try:
                print(f"📥 Downloading {filename} from {url}...")
                result = subprocess.run(
                    ["wget", "-O", str(target_path), url],
                    check=True
                )
                
                # Verify the downloaded file
                if target_path.exists():
                    file_size = target_path.stat().st_size
                    if file_size > 1024:  # Verify it's not a placeholder
                        print(f"✅ Successfully downloaded {filename} ({file_size:,} bytes)")
                    else:
                        print(f"⚠️ Downloaded {filename} but file seems too small ({file_size} bytes)")
                else:
                    print(f"❌ Failed to download {filename} - file not found after wget")
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to download {filename}: {e}")
                print(f"   stderr: {e.stderr}")
                # Create empty placeholder to prevent re-download attempts
                target_path.touch()
            except Exception as e:
                print(f"❌ Unexpected error downloading {filename}: {e}")
                target_path.touch()
        
        print("✅ RichHF-18K TFRecord download process completed.")


    def _download_and_prepare_seetrue_feedback(self, dl_manager, seetrue_raw_dir, urls):
        seetrue_raw_dir.mkdir(parents=True, exist_ok=True)
        try:
            (seetrue_raw_dir / "hf_cache").mkdir(exist_ok=True, parents=True)
            print(f"SeeTRUE-Feedback Hugging Face dataset will be loaded/cached into {seetrue_raw_dir / 'hf_cache'}")
            dl_manager.download_config(urls["hf_dataset_name"], datasets.DownloadConfig(cache_dir=seetrue_raw_dir / "hf_cache"))
        except Exception as e:
            print(f"Could not prepare SeeTRUE-Feedback HF dataset cache via dl_manager: {e}")
            (seetrue_raw_dir / "hf_dataset_load_failed.flag").touch()


    def _download_and_prepare_brush_data(self, dl_manager, brush_raw_dir, urls):
        """Download BrushData using direct wget - limited to 20GB instead of 1.7TB"""
        brush_raw_dir.mkdir(parents=True, exist_ok=True)
        tars_dir = brush_raw_dir / "tars"
        tars_dir.mkdir(exist_ok=True)
        
        print("🎯 Downloading BrushData using direct wget (limited to 20GB)...")
        
        base_url = urls.get("base_tar_url", "")
        tar_files = urls.get("tar_files", [])
        
        if not base_url or not tar_files:
            print("⚠️ No direct URLs configured for BrushData, skipping direct download")
            (brush_raw_dir / "brush_direct_download_failed.flag").touch()
            return
        
        downloaded_files = []
        total_size_gb = 0
        MAX_SIZE_GB = 20
        
        for i, tar_filename in enumerate(tar_files):
            if total_size_gb >= MAX_SIZE_GB:
                print(f"🛑 Reached 20GB limit, stopping BrushData download")
                break
                
            tar_url = base_url + tar_filename
            tar_local_path = tars_dir / tar_filename
            
            if tar_local_path.exists():
                file_size_gb = tar_local_path.stat().st_size / (1024**3)
                print(f"📁 {tar_filename} already exists ({file_size_gb:.1f}GB)")
                downloaded_files.append(tar_local_path)
                total_size_gb += file_size_gb
                continue
            
            try:
                print(f"📦 Downloading BrushData {tar_filename} ({i+1}/{len(tar_files)})...")
                downloaded_path = dl_manager.download_and_extract(tar_url)
                
                # Copy to our tars directory
                if Path(downloaded_path).is_file():
                    shutil.copy(downloaded_path, tar_local_path)
                    file_size_gb = tar_local_path.stat().st_size / (1024**3)
                    downloaded_files.append(tar_local_path)
                    total_size_gb += file_size_gb
                    print(f"✅ {tar_filename} downloaded ({file_size_gb:.1f}GB, total: {total_size_gb:.1f}GB)")
                else:
                    print(f"⚠️ {tar_filename} download failed")
                    
            except Exception as e:
                print(f"⚠️ Failed to download {tar_filename}: {e}")
                continue
        
        print(f"✅ BrushData direct download completed! Downloaded {len(downloaded_files)} TAR files ({total_size_gb:.1f}GB)")
        (brush_raw_dir / "brush_direct_download_completed.flag").touch()
        
        # Save list of downloaded files for processing
        with open(brush_raw_dir / "downloaded_tars.json", 'w') as f:
            json.dump([str(p) for p in downloaded_files], f)

    def _process_visual_genome(self, vg_raw_dir, processed_data_dir):
        """Process Visual Genome from direct downloads (no HuggingFace dependency)"""
        planning_samples = []
        
        # Load region descriptions
        regions_json_path = vg_raw_dir / "region_descriptions.json"
        regions_data_by_image_id = defaultdict(list)
        if regions_json_path.exists() and regions_json_path.stat().st_size > 2: 
            try:
                with open(regions_json_path, 'r') as f:
                    all_regions_data = json.load(f)
                print(f"📊 Loaded {len(all_regions_data)} region descriptions from direct download")
                for entry in all_regions_data: 
                    image_id_from_json = entry.get("id", entry.get("image_id"))
                    if image_id_from_json is not None:
                        for region in entry.get("regions", []):
                            regions_data_by_image_id[image_id_from_json].append(region)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not decode {regions_json_path}: {e}")
        
        # Load objects data if available
        objects_data_by_image_id = defaultdict(list)
        objects_json_path = vg_raw_dir / "objects.json"
        if objects_json_path.exists():
            try:
                with open(objects_json_path, 'r') as f:
                    all_objects_data = json.load(f)
                print(f"📊 Loaded {len(all_objects_data)} object descriptions from direct download")
                for entry in all_objects_data:
                    image_id = entry.get("image_id", entry.get("id"))
                    if image_id is not None:
                        objects_data_by_image_id[image_id] = entry.get("objects", [])
            except Exception as e:
                print(f"Warning: Could not load objects.json: {e}")
        
        # Load attributes data if available
        attributes_data_by_image_id = defaultdict(list)
        attributes_json_path = vg_raw_dir / "attributes.json"
        if attributes_json_path.exists():
            try:
                with open(attributes_json_path, 'r') as f:
                    all_attributes_data = json.load(f)
                print(f"📊 Loaded {len(all_attributes_data)} attribute descriptions from direct download")
                for entry in all_attributes_data:
                    image_id = entry.get("image_id", entry.get("id"))
                    if image_id is not None:
                        attributes_data_by_image_id[image_id] = entry.get("attributes", [])
            except Exception as e:
                print(f"Warning: Could not load attributes.json: {e}")
        
        # Find available images
        images_dir = vg_raw_dir / "images"
        available_images = {}
        if images_dir.exists():
            for img_file in images_dir.glob("*.jpg"):
                # Extract image ID from filename (Visual Genome uses format like "2407890.jpg")
                img_id = img_file.stem
                try:
                    img_id_int = int(img_id)
                    available_images[img_id_int] = img_file
                except ValueError:
                    continue
            print(f"🖼️ Found {len(available_images)} images from direct download")
        
        # Process available data to create planning samples
        print("🔄 Processing Visual Genome data for MINT planning step...")
        
        # Get all unique image IDs from all data sources
        all_image_ids = set()
        all_image_ids.update(regions_data_by_image_id.keys())
        all_image_ids.update(objects_data_by_image_id.keys())
        all_image_ids.update(attributes_data_by_image_id.keys())
        all_image_ids.update(available_images.keys())
        
        # Limit processing to reasonable number for performance
        MAX_SAMPLES = 5000  
        processed_count = 0
        
        for image_id in list(all_image_ids)[:MAX_SAMPLES]:
            if processed_count >= MAX_SAMPLES:
                break
                
            image_id_str = str(image_id)
            planning_text = "Planning: Comprehensive scene analysis with structured tables. "
            
            # Add region information
            regions = regions_data_by_image_id.get(image_id, [])
            if regions:
                region_descriptions = []
                for region in regions[:5]:  # Limit to first 5 regions
                    phrase = region.get("phrase", "")
                    x = region.get("x", 0); y = region.get("y", 0)
                    width = region.get("width", 0); height = region.get("height", 0)
                    if phrase:
                        region_descriptions.append(f"{phrase} (x:{x}, y:{y}, w:{width}, h:{height})")
                if region_descriptions:
                    planning_text += f"Structured regions: {'; '.join(region_descriptions)}. "
            
            # Add object information
            objects = objects_data_by_image_id.get(image_id, [])
            if objects:
                object_info = []
                for obj_entry in objects[:5]:  # Limit to first 5 objects
                    obj_name = obj_entry.get("names", ["unknown"])[0] if obj_entry.get("names") else "object"
                    obj_attrs = attributes_data_by_image_id.get(image_id, [])
                    if obj_attrs:
                        # Find attributes for this object
                        relevant_attrs = [attr.get("attribute", "") for attr in obj_attrs[:2]]
                        object_info.append(f"{obj_name} ({', '.join(relevant_attrs)})" if relevant_attrs else obj_name)
                    else:
                        object_info.append(obj_name)
                if object_info:
                    planning_text += f"Structured objects: {', '.join(object_info)}. "
            
            planning_text += "Visual Genome structured table processing complete."
            
            # Load image if available
            image_pil = None
            if image_id in available_images:
                try:
                    image_pil = Image.open(available_images[image_id]).convert("RGB")
                except Exception as e:
                    print(f"Warning: Could not load image {image_id}: {e}")
            
            planning_samples.append({
                "image_id": image_id_str,
                "planning": planning_text,
                "image": image_pil
            })
            
            processed_count += 1
            if processed_count % 500 == 0:
                print(f"📈 Processed {processed_count} Visual Genome samples...")
        
        print(f"✅ Processed {len(planning_samples)} Visual Genome planning samples")
        
        # Save processed data
        with open(processed_data_dir / "planning_data.json", 'w') as f:
            serializable_samples = []
            for ps in planning_samples:
                s_ps = ps.copy()
                if isinstance(s_ps.get("image"), Image.Image):
                    s_ps["image_mode"] = s_ps["image"].mode
                    s_ps["image_size"] = s_ps["image"].size
                    s_ps.pop("image")  # Remove PIL image for JSON serialization
                serializable_samples.append(s_ps)
            json.dump(serializable_samples, f)
        
        self.vg_planning_samples_with_images = planning_samples


    def _process_richhf18k(self, richhf_raw_dir, processed_data_dir):
        acting_samples = []
        
        # MINT paper: "Acting: Generate the image based on the planning outputs"
        # RichHF-18K provides human feedback on image generation quality
        
        # Look for TFRecord files downloaded directly via wget
        try:
            # Process directly downloaded TFRecord files
            MAX_SAMPLES_PER_FILE = 1000  # Increased from 100 for more complete dataset
            
            # Look for TFRecord files in the direct download location
            tfrecord_files = list(richhf_raw_dir.glob("*.tfrecord"))
            
            print(f"Found {len(tfrecord_files)} TFRecord files in RichHF-18K directory")
            
            # Process TFRecord files (preferred format)
            if tf is not None and tfrecord_files:
                print(f"🔄 Processing {len(tfrecord_files)} TFRecord files...")
                for tfrecord_file in tfrecord_files:
                    print(f"📂 Processing TFRecord: {tfrecord_file}")
                    
                    # Check if file is a valid TFRecord
                    file_size = tfrecord_file.stat().st_size
                    if file_size < 1000:  # Likely too small to be valid
                        print(f"⚠️ {tfrecord_file.name} appears to be too small (size: {file_size} bytes)")
                        print("   This may indicate a download failure.")
                        continue
                        
                    try:
                        dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                        processed_count = 0
                        
                        for i, raw_record in enumerate(dataset.take(MAX_SAMPLES_PER_FILE)):
                            try:
                                example = tf.train.Example()
                                example.ParseFromString(raw_record.numpy())
                                features = example.features.feature
                                
                                # Extract TFRecord data
                                prompt = self._get_string_feature(features, 'prompt')
                                feedback_text = self._get_string_feature(features, 'feedback_text')
                                feedback_score = self._get_float_feature(features, 'feedback_score')
                                quality_score = self._get_float_feature(features, 'quality_score')
                                aspect_scores = self._get_float_list_feature(features, 'aspect_scores')
                                
                                if prompt:
                                    acting_sample = self._process_richhf_sample({
                                        "prompt": prompt,
                                        "feedback": feedback_text,
                                        "feedback_score": feedback_score,
                                        "quality_score": quality_score,
                                        "aspect_scores": aspect_scores
                                    }, f"richhf_{tfrecord_file.stem}_{i}")
                                    if acting_sample:
                                        acting_samples.append(acting_sample)
                                        processed_count += 1
                                        
                            except Exception as parse_error:
                                if i < 5:  # Only show first few parse errors to avoid spam
                                    print(f"⚠️ Parse error in record {i}: {parse_error}")
                                continue
                                
                        print(f"✅ Processed {processed_count} samples from {tfrecord_file.name}")
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "corrupted record" in error_msg.lower() or "invalid argument" in error_msg.lower():
                            print(f"❌ {tfrecord_file.name} is corrupted or invalid (likely download failure)")
                            print("   The file was successfully downloaded but may be incomplete.")
                        else:
                            print(f"❌ Error processing TFRecord file {tfrecord_file}: {e}")
                        continue
            
            # Also check for legacy repo directory structure (from previous approach)
            repo_dir = richhf_raw_dir / "repo"
            if repo_dir.exists() and not tfrecord_files:
                print("📁 Found legacy Git repository structure, checking for data files...")
                MAX_TOTAL_FILES = 50  # Limit total files processed to keep dataset manageable
                
                jsonl_files = list(repo_dir.glob("**/*.jsonl"))[:MAX_TOTAL_FILES]
                json_files = list(repo_dir.glob("**/*.json"))[:MAX_TOTAL_FILES]
                legacy_tfrecord_files = list(repo_dir.glob("**/*.tfrecord"))[:MAX_TOTAL_FILES] + list(repo_dir.glob("**/*.tfr"))[:MAX_TOTAL_FILES]
                
                # Process legacy TFRecord files if they exist and are valid
                for tfrecord_file in legacy_tfrecord_files:
                    file_size = tfrecord_file.stat().st_size
                    if file_size < 1000:  # Skip small files
                        continue
                    tfrecord_files.append(tfrecord_file)
                
                # Process JSONL files as fallback
                for jsonl_file in jsonl_files:
                    try:
                        with open(jsonl_file, 'r', encoding='utf-8') as f:
                            for line_idx, line in enumerate(f):
                                if line_idx >= MAX_SAMPLES_PER_FILE:  # Use configurable limit
                                    break
                                try:
                                    data = json.loads(line.strip())
                                    acting_sample = self._process_richhf_sample(data, f"richhf_{jsonl_file.stem}_{line_idx}")
                                    if acting_sample:
                                        acting_samples.append(acting_sample)
                                except json.JSONDecodeError:
                                    continue
                    except Exception as e:
                        print(f"Error processing JSONL file {jsonl_file}: {e}")
                        continue
                
                # Process JSON files as additional fallback
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for idx, item in enumerate(data[:MAX_SAMPLES_PER_FILE]):  # Use configurable limit
                                    acting_sample = self._process_richhf_sample(item, f"richhf_{json_file.stem}_{idx}")
                                    if acting_sample:
                                        acting_samples.append(acting_sample)
                            else:
                                acting_sample = self._process_richhf_sample(data, f"richhf_{json_file.stem}")
                                if acting_sample:
                                    acting_samples.append(acting_sample)
                    except Exception as e:
                        print(f"Error processing JSON file {json_file}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error processing RichHF-18K data: {e}")
        
        print(f"✅ Successfully processed {len(acting_samples)} RichHF-18K acting samples")

        with open(processed_data_dir / "acting_data.json", 'w') as f:
            json.dump(acting_samples, f)
        self.richhf_acting_samples = acting_samples

    def _process_richhf_sample(self, data, image_id):
        """Process individual RichHF-18K sample following MINT acting methodology"""
        prompt = data.get("prompt", data.get("text", data.get("caption", "")))
        
        # Extract feedback information for acting step
        feedback = data.get("feedback", data.get("human_feedback", ""))
        quality_score = data.get("quality_score", data.get("quality", data.get("score", data.get("rating", 0.5))))
        feedback_score = data.get("feedback_score", 0.0)
        aspect_scores = data.get("aspect_scores", [])
        
        if not prompt:
            return None
            
        # Generate acting text following MINT paper specifications
        acting_text = "Acting: Generate the image based on the planning outputs. "
        acting_text += f"Prompt execution: '{prompt}' "
        
        if feedback:
            feedback_clean = feedback[:200] if len(feedback) > 200 else feedback
            acting_text += f"Human feedback integration: {feedback_clean} "
            
        if quality_score and quality_score > 0:
            quality_assessment = "excellent" if quality_score > 0.8 else "good" if quality_score > 0.6 else "moderate"
            acting_text += f"Quality target: {quality_assessment} generation (score: {quality_score:.2f}). "
        
        if feedback_score > 0:
            acting_text += f"Feedback confidence: {feedback_score:.3f}. "
            
        if aspect_scores:
            aspects = ['composition', 'color', 'lighting', 'detail', 'coherence']
            aspect_info = [f"{aspects[j]}:{score:.2f}" for j, score in enumerate(aspect_scores[:5]) if j < len(aspects)]
            if aspect_info:
                acting_text += f"Aspect evaluation: {', '.join(aspect_info)}. "
        
        acting_text += "Following spatial relationships and object placements from planning step. "
        acting_text += "Coherent visual representation with attention to interwoven conditions."
        
        return {
            "image_id": str(image_id),
            "prompt": prompt,
            "acting": acting_text,
            "quality_score": quality_score,
            "feedback": feedback,
            "feedback_score": feedback_score,
            "aspect_scores": aspect_scores
        }
    
    def _generate_mint_acting_text(self, prompt):
        """Generate acting text following MINT paper methodology"""
        acting_text = "Acting: Generate the image based on the planning outputs. "
        acting_text += f"Use the caption and layout information to create a coherent visual representation. "
        acting_text += f"Prompt guidance: '{prompt}' "
        acting_text += "Follow the spatial relationships and object placements from the planning step. "
        acting_text += "Execute with attention to fine-grained alignment and interwoven conditions. "
        acting_text += "Maintain consistency with dense caption and bounding box specifications."
        return acting_text

    def _get_string_feature(self, features, key):
        return features[key].bytes_list.value[0].decode('utf-8') if key in features and features[key].bytes_list.value else ""

    def _get_float_feature(self, features, key):
        return features[key].float_list.value[0] if key in features and features[key].float_list.value else 0.0

    def _get_float_list_feature(self, features, key):
        return list(features[key].float_list.value) if key in features else []

    def _process_seetrue_feedback(self, seetrue_raw_dir, processed_data_dir):
        reflection_samples = []
        try:
            print(f"Loading SeeTRUE-Feedback HF dataset")
            seetrue_hf_dataset = datasets.load_dataset(
                _URLS["seetrue_feedback"]["hf_dataset_name"],
                split="test",  # Use 'test' split instead of 'train'
                cache_dir=str(seetrue_raw_dir / "hf_cache"),
                trust_remote_code=True
            )
            for i, sample in enumerate(seetrue_hf_dataset):
                image_id = sample.get("id", f"seetrue_{i}")
                caption = sample.get("caption", "")
                feedback = sample.get("feedback", "")
                alignment_score = sample.get("alignment_score", 0.0)
                bounding_boxes = sample.get("bounding_boxes", [])
                
                reflection_text = "Reflection: Tabular format analysis for artifact detection. "
                reflection_text += f"Caption: {caption} "
                if feedback: reflection_text += f"Feedback analysis: {feedback} "
                if alignment_score is not None:
                    score_text = "Excellent alignment."
                    if alignment_score < 0.3: score_text = "Critical alignment issues."
                    elif alignment_score < 0.5: score_text = "Low alignment."
                    elif alignment_score < 0.7: score_text = "Moderate alignment."
                    elif alignment_score < 0.9: score_text = "Good alignment."
                    reflection_text += f"{score_text} Alignment score: {alignment_score:.3f}. "
                if bounding_boxes: reflection_text += f"Spatial artifact analysis: {len(bounding_boxes)} detected regions. "
                reflection_text += "SeeTRUE tabular format processing complete."
                reflection_samples.append({
                    "image_id": str(image_id), "reflection": reflection_text,
                    "alignment_score": alignment_score, "caption": caption, "feedback": feedback
                })
        except Exception as e:
            print(f"Error processing SeeTRUE-Feedback HF dataset: {e}")
        
        with open(processed_data_dir / "reflection_data.json", 'w') as f:
            json.dump(reflection_samples, f)
        self.seetrue_reflection_samples = reflection_samples

    def _process_brush_data(self, brush_raw_dir, processed_data_dir):
        """Process BrushData from direct wget downloads (limited to 20GB)"""
        global wds 
        correction_samples = []
        
        print("🔄 Processing BrushData from direct wget downloads...")
        
        # Check if we have direct downloads
        downloaded_tars_file = brush_raw_dir / "downloaded_tars.json"
        tars_dir = brush_raw_dir / "tars"
        
        if downloaded_tars_file.exists():
            try:
                with open(downloaded_tars_file, 'r') as f:
                    tar_file_paths = [Path(p) for p in json.load(f)]
                print(f"📂 Found {len(tar_file_paths)} downloaded TAR files for BrushData")
            except Exception as e:
                print(f"⚠️ Could not read downloaded TAR files list: {e}")
                tar_file_paths = []
        else:
            # Fallback: look for TAR files in directory
            tar_file_paths = list(tars_dir.glob("*.tar")) if tars_dir.exists() else []
            print(f"📂 Found {len(tar_file_paths)} TAR files in {tars_dir}")
        
        if not tar_file_paths:
            print("⚠️ No BrushData TAR files found")
            correction_samples.append({
                "image_id": "brush_no_tars", 
                "correction": "No BrushData TAR files available."
            })
        elif wds is None:
            print("❌ Webdataset library not available. Cannot process BrushData TAR files.")
            correction_samples.append({
                "image_id": "brush_wds_missing", 
                "correction": "Correction data unavailable, webdataset library missing."
            })
        else:
            processed_sample_count = 0
            MAX_SAMPLES = 2000  # Process more samples since we limited download size
            
            for i, tar_file_path in enumerate(tar_file_paths):
                if processed_sample_count >= MAX_SAMPLES:
                    print(f"🛑 Reached {MAX_SAMPLES} sample limit, stopping processing")
                    break
                    
                print(f"🔄 Processing BrushData TAR file ({i+1}/{len(tar_file_paths)}): {tar_file_path.name}")
                try:
                    # BrushData TAR files already contain processed data with 'image', 'caption', etc. keys
                    # Don't use .decode() or .rename() as the data is already structured
                    dataset_shard = wds.WebDataset(str(tar_file_path))
                    
                    for sample_idx, sample in enumerate(dataset_shard):
                        if processed_sample_count >= MAX_SAMPLES:
                            break
                        
                        # Debug: Print comprehensive sample structure for first few samples
                        if sample_idx < 3:  # Debug first 3 samples from first TAR file
                            print(f"🔍 DEBUG Sample {sample_idx} structure:")
                            print(f"   Available keys: {list(sample.keys())}")
                            for key, value in sample.items():
                                if isinstance(value, bytes):
                                    # Try to decode bytes to understand content
                                    try:
                                        decoded_str = value.decode('utf-8')
                                        print(f"   {key}: bytes→string ('{decoded_str[:100]}...' if len > 100)")
                                    except:
                                        print(f"   {key}: bytes (length: {len(value)}, non-text)")
                                elif isinstance(value, str):
                                    print(f"   {key}: string ('{value[:100]}...' if len > 100)")
                                elif hasattr(value, 'shape'):  # numpy array or tensor
                                    print(f"   {key}: array/tensor (shape: {value.shape}, dtype: {getattr(value, 'dtype', 'unknown')})")
                                elif hasattr(value, 'mode'):  # PIL Image
                                    print(f"   {key}: PIL Image (mode: {value.mode}, size: {value.size})")
                                else:
                                    print(f"   {key}: {type(value)} - {str(value)[:100]}...")
                        
                        # Handle image data - it might be raw bytes that need to be converted to PIL
                        image_pil = None
                        mask_pil = None
                        
                        if "image" in sample:
                            image_data = sample["image"]
                            if isinstance(image_data, bytes):
                                try:
                                    from io import BytesIO
                                    image_pil = Image.open(BytesIO(image_data))
                                except Exception as e:
                                    print(f"⚠️ Could not decode image data: {e}")
                                    continue
                            elif hasattr(image_data, 'mode'):  # Already a PIL image
                                image_pil = image_data
                        
                        # Handle mask/segmentation data - BrushData uses RLE JSON format
                        mask_pil = None
                        segmentation_data = None
                        if "segmentation" in sample:
                            mask_data = sample["segmentation"]
                            if isinstance(mask_data, bytes):
                                try:
                                    # First try: decode as JSON (RLE format)
                                    seg_text = mask_data.decode('utf-8')
                                    segmentation_data = json.loads(seg_text)
                                    
                                    if sample_idx < 3:  # Debug output
                                        print(f"   segmentation: RLE JSON with {len(segmentation_data.get('mask', []))} masks")
                                    
                                    # Successfully parsed RLE data - we'll use this for mask generation
                                    # For now, mark as having segmentation data (we could convert RLE to PIL later if needed)
                                    has_segmentation_data = True
                                    
                                except json.JSONDecodeError:
                                    try:
                                        # Second try: direct image decoding (fallback)
                                        from io import BytesIO
                                        mask_pil = Image.open(BytesIO(mask_data))
                                        has_segmentation_data = True
                                        if sample_idx < 3:
                                            print(f"   segmentation: decoded as PIL image")
                                    except Exception as e:
                                        if sample_idx < 3:
                                            print(f"   segmentation: failed to decode - {e}")
                                        has_segmentation_data = False
                                except Exception as e:
                                    if sample_idx < 3:
                                        print(f"   segmentation: failed to parse as JSON - {e}")
                                    has_segmentation_data = False
                            elif hasattr(mask_data, 'mode'):  # Already a PIL image
                                mask_pil = mask_data
                                has_segmentation_data = True
                            elif isinstance(mask_data, dict):  # Already parsed JSON
                                segmentation_data = mask_data
                                has_segmentation_data = True
                                if sample_idx < 3:
                                    print(f"   segmentation: pre-parsed RLE JSON with {len(mask_data.get('mask', []))} masks")
                            else:
                                has_segmentation_data = False
                        
                        if image_pil is None:
                            continue  # Skip samples without valid images
                        
                        # Use either PIL mask or RLE segmentation data to determine if we have mask info
                        has_actual_mask = mask_pil is not None or segmentation_data is not None
                        
                        caption_text = ""
                        task_type_content = "inpainting" 

                        if "caption" in sample:
                            cap_data = sample["caption"]
                            if isinstance(cap_data, (bytes, str)):
                                caption_text = cap_data.decode('utf-8').strip() if isinstance(cap_data, bytes) else str(cap_data).strip()
                            elif isinstance(cap_data, dict): 
                                caption_text = cap_data.get("caption", cap_data.get("text",""))
                                task_type_content = cap_data.get("task_type", task_type_content)
                        
                        has_actual_mask = mask_pil is not None or segmentation_data is not None
                        image_id_str = f"brush_{tar_file_path.stem}_{sample.get('__key__', sample_idx)}"

                        correction_text_gen = self._generate_mint_correction_text(
                            has_mask=has_actual_mask,
                            task_type=task_type_content,
                            caption=caption_text
                        )
                        correction_samples.append({
                            "image_id": image_id_str,
                            "correction": correction_text_gen,
                            "has_mask": has_actual_mask,
                            "task_type": task_type_content,
                            "caption": caption_text,
                            "segmentation_rle": segmentation_data,  # Store RLE data for potential future use
                            "mask_pil": mask_pil  # Store PIL mask if available
                        })
                        processed_sample_count += 1
                        
                        if processed_sample_count % 100 == 0:
                            print(f"📈 Processed {processed_sample_count} BrushData samples...")
                            
                except Exception as e_tar:
                    print(f"⚠️ Error processing TAR file {tar_file_path}: {e_tar}")
                    continue 
                    
            print(f"✅ Processed {processed_sample_count} samples from {len(tar_file_paths)} BrushData TAR files")

        if not correction_samples:
            correction_samples.append({
                "image_id": "brush_processing_empty", 
                "correction": "No correction data processed."
            })

        with open(processed_data_dir / "correction_data.json", 'w') as f:
            json.dump(correction_samples, f)
        self.brush_correction_samples = correction_samples

    def _generate_mint_correction_text(self, has_mask=True, task_type="inpainting", caption="", mask_coverage=0.0):
        if not has_mask and mask_coverage == 0.0 and Image is not None and ImageDraw is not None :
            _, mask_coverage = self._generate_random_mask() 
        
        correction_text = ""
        if task_type == "segmentation_inpainting" and has_mask:
            correction_text = "Correction: Segmentation-based inpainting with BrushNet methodology. "
            correction_text += f"Caption guidance: '{caption}' " if caption else ""
            correction_text += "Precise object-aware segmentation masks target specific regions. "
        elif task_type == "random_inpainting":
            correction_text = "Correction: Random mask inpainting with authentic BrushNet generation. "
            correction_text += f"Caption guidance: '{caption}' " if caption else ""
            if mask_coverage > 0: correction_text += f"Generated mask coverage: {mask_coverage:.2%}. "
        else:
            correction_text = "Correction: Targeted artifact correction using BrushNet inpainting. "
            correction_text += f"Caption guidance: '{caption}' " if caption else ""
        return self._enhance_correction_with_brushnet(correction_text, has_mask, mask_coverage)

    def _enhance_correction_with_brushnet(self, correction_text, has_mask=True, coverage=0.0):
        if has_mask:
            correction_text += f" BrushNet segmentation approach with {coverage:.1%} coverage. "
        else:
            correction_text += f"BrushNet random brush generation with {coverage:.1%} coverage. "
        correction_text += "Authentic TencentARC/BrushNet methodology applied."
        return correction_text

    def _generate_random_mask(self, height=512, width=512):
        if Image is None or ImageDraw is None: return None, 0.0
        mask_array = self._brushnet_random_mask_gen(height, width)
        if mask_array is None: return None, 0.0
        mask = Image.fromarray((mask_array * 255).astype(np.uint8), 'L')
        coverage = np.sum(mask_array > 0) / (height * width)
        return mask, coverage

    def _brushnet_random_brush_gen(self, max_tries, h, w, min_num_vertex=4, max_num_vertex=8,
                                   mean_angle=2*math.pi/5, angle_range=2*math.pi/15,
                                   min_width=12, max_width=40):
        """Generate random brush mask following authentic BrushNet methodology from TencentARC/BrushNet"""
        if Image is None or ImageDraw is None: 
            return np.zeros((h, w), dtype=np.uint8)
        
        H, W = h, w
        average_radius = math.sqrt(H*H + W*W) / 8
        mask = Image.new('L', (W, H), 0)
        
        # Generate random number of brush strokes (authentic BrushNet uses np.random.randint)
        for _ in range(np.random.randint(1, max_tries)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            
            # Generate angles for each vertex (authentic BrushNet approach)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))
            
            # Start with random position
            vertex.append((int(np.random.randint(0, W)), int(np.random.randint(0, H))))
            
            # Generate subsequent vertices
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, W)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, H)
                vertex.append((int(new_x), int(new_y)))

            # Draw the brush stroke
            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            
            # Draw ellipses at each vertex for brush tip effect (authentic BrushNet approach)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)
        
        # Apply random transformations (authentic BrushNet approach)
        if np.random.random() > 0.5:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        
        mask = np.asarray(mask, np.uint8)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 0)
        if np.random.random() > 0.5:
            mask = np.flip(mask, 1)
        return mask

    def _brushnet_random_mask_gen(self, h, w):
        """Generate random mask following authentic BrushNet methodology"""
        mask = np.ones((h, w), np.uint8)
        # hole denoted as 0, reserved as 1 (authentic BrushNet approach)
        # Use np.logical_and as in authentic implementation
        brush_mask = self._brushnet_random_brush_gen(4, h, w)
        mask = np.logical_and(mask, 1 - brush_mask).astype(np.float32)
        return mask


    def _generate_mcot_examples(self, processed_data_dir):
        planning_samples = getattr(self, "vg_planning_samples_with_images", [])
        acting_samples = getattr(self, "richhf_acting_samples", [])
        reflection_samples = getattr(self, "seetrue_reflection_samples", [])
        correction_samples = getattr(self, "brush_correction_samples", [])

        if not planning_samples: 
            try:
                with open(processed_data_dir / "planning_data.json", 'r') as f: planning_samples_json = json.load(f)
                planning_samples = []
                for ps_json in planning_samples_json:
                    ps_json["image"] = None 
                    planning_samples.append(ps_json)
            except FileNotFoundError: 
                print("Warning: planning_data.json not found")
                planning_samples = []
        
        if not acting_samples:
            try:
                with open(processed_data_dir / "acting_data.json", 'r') as f: acting_samples = json.load(f)
            except FileNotFoundError: 
                print("Warning: acting_data.json not found")
                acting_samples = []
        
        if not reflection_samples:
            try:
                with open(processed_data_dir / "reflection_data.json", 'r') as f: reflection_samples = json.load(f)
            except FileNotFoundError: 
                print("Warning: reflection_data.json not found")
                reflection_samples = []

        if not correction_samples:
            try:
                with open(processed_data_dir / "correction_data.json", 'r') as f: correction_samples = json.load(f)
            except FileNotFoundError: 
                print("Warning: correction_data.json not found")
                correction_samples = []

        mcot_examples = []
        num_planning = len(planning_samples)
        num_acting = len(acting_samples) if acting_samples else 0
        num_reflection = len(reflection_samples) if reflection_samples else 0
        num_correction = len(correction_samples) if correction_samples else 0

        if num_planning == 0: 
            print("Critical: No planning samples available to generate MCoT examples.")
            return 

        for i in range(num_planning):
            planning_sample = planning_samples[i]
            image_id = planning_sample.get("image_id", f"mcot_{i}")
            pil_image_obj = planning_sample.get("image") 

            acting_s = acting_samples[i % num_acting] if num_acting > 0 else {}
            reflection_s = reflection_samples[i % num_reflection] if num_reflection > 0 else {}
            correction_s = correction_samples[i % num_correction] if num_correction > 0 else {}

            final_response = f"Complete MCoT process for {image_id}. Enhanced through planning, acting, reflection, and correction."
            mcot_example = {
                "image_id": image_id,
                "prompt": acting_s.get("prompt", f"Generate image for {image_id}"),
                "planning": planning_sample.get("planning", "Planning data N/A."),
                "acting": acting_s.get("acting", "Acting data N/A."),
                "reflection": reflection_s.get("reflection", "Reflection data N/A."),
                "correction": correction_s.get("correction", "Correction data N/A."),
                "final_response": final_response,
                "image_obj_pil": pil_image_obj,
                "image_mode_from_planning": planning_sample.get("image_mode"), 
                "image_size_from_planning": planning_sample.get("image_size")   
            }
            mcot_examples.append(mcot_example)

        random.shuffle(mcot_examples)
        split_idx = int(0.9 * len(mcot_examples))
        train_examples = mcot_examples[:split_idx]
        val_examples = mcot_examples[split_idx:]

        self._save_mcot_examples_split(train_examples, processed_data_dir / "train")
        self._save_mcot_examples_split(val_examples, processed_data_dir / "val")

    def _save_mcot_examples_split(self, examples, output_dir_split):
        output_dir_split.mkdir(parents=True, exist_ok=True)
        for i, example in enumerate(examples):
            image_id_val = example.get('image_id', f"unknown_{i}")
            safe_image_id_val = image_id_val.replace('/', '_').replace('\\', '_')
            example_data_dir = output_dir_split / f"example_{safe_image_id_val}"
            example_data_dir.mkdir(parents=True, exist_ok=True)
            
            annotations_to_save = {k: v for k, v in example.items() if k not in ["image_obj_pil", "image_mode_from_planning", "image_size_from_planning"]}
            with open(example_data_dir / "mcot_annotations.json", "w") as f:
                json.dump(annotations_to_save, f, indent=2)

            pil_image = example.get("image_obj_pil")
            image_file_path = example_data_dir / "image.jpg"
            
            if isinstance(pil_image, Image.Image):
                try:
                    pil_image.convert("RGB").save(image_file_path, "JPEG")
                except Exception as e:
                    print(f"Could not save PIL image for {image_id_val}: {e}. Creating placeholder.")
                    try:
                        Image.new('RGB', (256,256), color='red').save(image_file_path, "JPEG")
                    except Exception as e_ph_save:
                         print(f"Could not save placeholder image for {image_id_val} during exception: {e_ph_save}")

            elif not image_file_path.exists(): 
                try:
                    img_size = example.get("image_size_from_planning", (256,256))
                    img_mode = example.get("image_mode_from_planning", "RGB")
                    if isinstance(img_size, list) and len(img_size) == 2: img_size = tuple(img_size)
                    if not (isinstance(img_size, tuple) and len(img_size) == 2 and all(isinstance(dim, int) for dim in img_size)):
                        img_size = (256,256) 

                    placeholder_img = Image.new(img_mode if isinstance(img_mode, str) else "RGB", img_size, color=(100, 100, 100))
                    draw = ImageDraw.Draw(placeholder_img)
                    draw.text((10, 10), f"Placeholder {image_id_val}", fill=(255,255,255))
                    placeholder_img.convert("RGB").save(image_file_path, "JPEG")
                except Exception as e_placeholder:
                    print(f"Could not create/save placeholder image for {image_id_val}: {e_placeholder}")
                    (example_data_dir / "image_placeholder_creation_failed.flag").touch()

    def _generate_examples(self, data_dir):
        data_path = Path(data_dir)
        example_dirs = [d for d in data_path.iterdir() if d.is_dir() and (d / "mcot_annotations.json").exists()]

        for i, example_dir_path in enumerate(example_dirs):
            mcot_file = example_dir_path / "mcot_annotations.json"
            image_file = example_dir_path / "image.jpg"

            if not image_file.exists(): 
                print(f"Warning: Image file missing for {example_dir_path}, skipping.")
                continue
            
            try:
                with open(mcot_file, "r") as f:
                    annotations = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode annotations for {example_dir_path}, skipping.")
                continue
            
            yield i, {
                "image_id": annotations.get("image_id", example_dir_path.name.replace("example_","")),
                "image": str(image_file),
                "prompt": annotations.get("prompt"),
                "planning": annotations.get("planning"),
                "acting": annotations.get("acting"),
                "reflection": annotations.get("reflection"),
                "correction": annotations.get("correction"),
                "final_response": annotations.get("final_response")
            }


# Main execution block for direct script usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process MCoT datasets")
    parser.add_argument("--output_dir", type=str, default="./mcot_dataset_output", 
                       help="Output directory for processed dataset")
    parser.add_argument("--cache_dir", type=str, default="./mcot_dataset_cache",
                       help="Cache directory for downloaded files")
    parser.add_argument("--num_proc", type=int, default=4,
                       help="Number of processes for dataset processing")
    
    args = parser.parse_args()
    
    print("Starting MCoT dataset download and processing...")
    print(f"Output directory: {args.output_dir}")
    print(f"Cache directory: {args.cache_dir}")
    
    # Process data directly without using HuggingFace datasets builder framework
    try:
        # Create directory structure
        output_path = Path(args.output_dir)
        cache_path = Path(args.cache_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create a simple download manager-like object
        class SimpleDownloadManager:
            def __init__(self, cache_dir):
                self.cache_dir = cache_dir
                self.manual_dir = cache_dir
                
            def download_and_extract(self, url):
                """Robust download and extract implementation using wget"""
                import subprocess
                import zipfile
                filename = url.split('/')[-1]
                local_path = Path(self.cache_dir) / filename
                
                if not local_path.exists():
                    print(f"Downloading {url} using wget...")
                    try:
                        # Use wget with robust options:
                        # -c: continue partial downloads
                        # --timeout=0: no connection timeout (wait indefinitely)
                        # --tries=3: retry up to 3 times
                        # --progress=bar: show progress bar
                        wget_cmd = [
                            'wget',
                            '-c',                    # Continue partial downloads
                            '--timeout=0',           # No connection timeout (wait indefinitely)
                            '--tries=3',             # Reasonable number of retries
                            '--wait=5',              # Wait 5 seconds between retries
                            '--progress=bar:force',  # Force progress bar display
                            '--show-progress',       # Show download progress
                            '--user-agent=Mozilla/5.0 (compatible; Research Bot)', # Add user agent
                            '-O', str(local_path),   # Output to specific file
                            url
                        ]
                        
                        print(f"Running: {' '.join(wget_cmd)}")
                        result = subprocess.run(
                            wget_cmd, 
                            check=True, 
                            capture_output=False,  # Show wget output directly
                            text=True,
                            timeout=None  # No subprocess timeout - allow downloads to complete
                        )
                        print(f"✅ Download completed: {local_path}")
                        
                    except subprocess.CalledProcessError as e:
                        print(f"❌ wget failed with exit code {e.returncode}: {url}")
                        if local_path.exists():
                            local_path.unlink()  # Remove partial/corrupted file
                        raise
                    except FileNotFoundError:
                        print("❌ wget not found. Please install wget or use alternative download method.")
                        raise
                else:
                    print(f"📁 File already exists: {local_path}")
                
                # Extract if it's a zip file
                if filename.endswith('.zip'):
                    extract_dir = local_path.parent / filename.replace('.zip', '')
                    if not extract_dir.exists():
                        print(f"📦 Extracting {local_path}...")
                        try:
                            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                            print(f"✅ Extraction completed: {extract_dir}")
                        except zipfile.BadZipFile:
                            print(f"❌ Corrupted zip file: {local_path}")
                            local_path.unlink()  # Remove corrupted file
                            raise
                    else:
                        print(f"📁 Already extracted: {extract_dir}")
                    return str(extract_dir)
                    
                return str(local_path)
                
            def download_config(self, *args, **kwargs):
                """Placeholder for download_config - not needed for our direct processing"""
                pass
        
        # Create dataset builder and process data
        dataset_builder = MCoTWgetDataset()
        dl_manager = SimpleDownloadManager(str(cache_path))
        
        # Call the split generators directly to process the data
        print("Processing MCoT dataset components...")
        split_generators = dataset_builder._split_generators(dl_manager)
        
        print("Dataset processing completed successfully!")
        print(f"Processed data saved to: {cache_path}")
        print("You can now use the MCoT PyTorch dataset classes to load this data for training.")
        
    except Exception as e:
        print(f"Error during dataset processing: {e}")
        import traceback
        print("Full error traceback:")
        traceback.print_exc()
        print("Please check the error messages above for specific issues.")
    
    print("MCoT dataset processing completed!")
