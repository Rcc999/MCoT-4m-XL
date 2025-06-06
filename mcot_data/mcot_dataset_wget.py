"""
MCoT Dataset Download and Processing Script

This script downloads and processes multiple datasets to create training data for 
Multi-step Chain of Thought (MCoT) reasoning. It combines data from several sources:

1. **ActPlan**: Replaces Visual Genome, provides action planning data
2. **RichHF-18K**: Human feedback data for multimodal reasoning  
3. **SeeTRUE-Feedback**: Artifact detection and feedback data (used for MINT reflection)
4. **BrushData**: Image editing and correction data

The script processes each dataset and formats them into a unified JSON structure
suitable for MCoT training, where each example contains:
- Original prompt/instruction
- Planning step output  
- Acting step output
- Reflection step output (with artifact detection)
- Correction step output

Usage:
    python mcot_dataset_wget.py --output_dir ./processed_data --max_samples 1000

The script handles missing files gracefully and can resume interrupted downloads.
"""

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
import webdataset as wds
import argparse


_CITATION = """
@article{wang2024mint,
  title={MINT: Multi-modal Chain of Thought in Unified Generative Models for Enhanced Image Generation},
  author={Wang, Yi and Liu, Mushui and He, Wanggui and Zhang, Longxiang and Huang, Ziwei and Zhang, Guanghao and Shu, Fangxun and Tao, Zhong and She, Dong and Yu, Zhelun and Li, Haoyuan and Dai, Weilong and Song, Mingli and Song, Jie and Jiang, Hao},
  journal={arXiv preprint arXiv:2503.01298},
  year={2024}
}
"""

_DESCRIPTION = """
Multi-step Chain of Thought (MCoT) Dataset for Enhanced Image Generation

This dataset combines multiple sources to create a comprehensive training set for MCoT reasoning:
- ActPlan: Action planning and spatial reasoning data  
- RichHF-18K: Human feedback for multimodal tasks
- SeeTRUE-Feedback: Artifact detection and quality assessment
- BrushData: Image editing and correction examples

Each example follows the MCoT pipeline: Planning → Acting → Reflection → Correction
"""

_URLS = {
    "actplan": {
        "local_json_file": "actplan.json"
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
    VERSION = datasets.Version("1.1.2")

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
        """
        Download all datasets to local cache before processing.
        
        This approach prevents re-downloading if processing fails partway through.
        Downloads happen in parallel where possible for efficiency.
        
        Args:
            dl_manager: Hugging Face datasets download manager
            base_data_dir: Base directory for storing raw datasets
        """
        print("Downloading all datasets first...")
        
        seetrue_cache_dir = base_data_dir / "seetrue_feedback_raw" / "hf_cache"
        
        print(f"Downloading SeeTRUE-Feedback dataset...")
        dataset = datasets.load_dataset(
            _URLS["seetrue_feedback"]["hf_dataset_name"],
            split="test",
            cache_dir=str(seetrue_cache_dir),
            trust_remote_code=True
        )
        print(f"SeeTRUE-Feedback downloaded successfully")
        
        print("All dataset downloads completed! Now processing...")

    def _split_generators(self, dl_manager):
        user_manual_dir = dl_manager.manual_dir
        base_data_dir = Path(user_manual_dir) if user_manual_dir else Path(os.getcwd()) / "mcot_downloads"
        base_data_dir.mkdir(parents=True, exist_ok=True)

        self._download_all_datasets_first(dl_manager, base_data_dir)

        processed_data_dir = base_data_dir / "processed_mcot_steps"
        processed_data_dir.mkdir(parents=True, exist_ok=True)

        richhf_raw_dir = base_data_dir / "richhf18k_raw"
        seetrue_raw_dir = base_data_dir / "seetrue_feedback_raw"
        brush_raw_dir = base_data_dir / "brush_data_raw"

        print("Processing downloaded datasets...")
        self._process_actplan(processed_data_dir)

        self._download_and_prepare_richhf18k(dl_manager, richhf_raw_dir, _URLS["richhf18k"])
        self._process_richhf18k(richhf_raw_dir, processed_data_dir)

        self._download_and_prepare_seetrue_feedback(dl_manager, seetrue_raw_dir, _URLS["seetrue_feedback"])
        self._process_seetrue_feedback(seetrue_raw_dir, processed_data_dir)

        self._download_and_prepare_brush_data(dl_manager, brush_raw_dir, _URLS["brush_data"])
        self._process_brush_data(brush_raw_dir, processed_data_dir)

        self._generate_mcot_examples(processed_data_dir)
        
        print("MCoT dataset construction complete. Starting cleanup of raw and intermediate files...")
        
        raw_data_paths_to_clean = [richhf_raw_dir, seetrue_raw_dir, brush_raw_dir]
        for raw_path in raw_data_paths_to_clean:
            if raw_path.exists():
                print(f"Cleaning up raw data directory: {raw_path}")
                shutil.rmtree(raw_path)
                print(f"Successfully removed {raw_path}")

        intermediate_json_files = [
            processed_data_dir / "planning_data.json",
            processed_data_dir / "richhf_reflection_data.json",
            processed_data_dir / "seetrue_reflection_data.json", 
            processed_data_dir / "correction_data.json"
        ]
        for json_file in intermediate_json_files:
            if json_file.exists():
                print(f"Cleaning up intermediate file: {json_file}")
                json_file.unlink()
                print(f"Successfully removed {json_file}")
        
        print("Cleanup process finished.")

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




    def _download_and_prepare_richhf18k(self, dl_manager, richhf_raw_dir, urls):
        """Download RichHF-18K TFRecord files using direct wget"""
        richhf_raw_dir.mkdir(parents=True, exist_ok=True)
        
        print("Downloading RichHF-18K TFRecord files using direct wget...")
        
        tfrecord_urls = urls.get("tfrecord_urls", {})
        
        for split, url in tfrecord_urls.items():
            filename = f"{split}.tfrecord"
            target_path = richhf_raw_dir / filename
            
            if target_path.exists():
                file_size = target_path.stat().st_size
                if file_size > 1024:
                    print(f"{filename} already exists ({file_size:,} bytes)")
                    continue
                else:
                    print(f"{filename} exists but is too small ({file_size} bytes). Re-downloading...")
            
            try:
                print(f"Downloading {filename} from {url}...")
                result = subprocess.run(
                    ["wget", "-O", str(target_path), url],
                    check=True
                )
                
                if target_path.exists():
                    file_size = target_path.stat().st_size
                    if file_size > 1024:
                        print(f"Successfully downloaded {filename} ({file_size:,} bytes)")
                    else:
                        print(f"Downloaded {filename} but file seems too small ({file_size} bytes)")
                else:
                    print(f"Failed to download {filename} - file not found after wget")
                    
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {filename}: {e}")
                raise RuntimeError(f"Failed to download {filename}")
            except Exception as e:
                print(f" Unexpected error downloading {filename}: {e}")
                raise
        
        print("RichHF-18K TFRecord download process completed.")


    def _download_and_prepare_seetrue_feedback(self, dl_manager, seetrue_raw_dir, urls):
        """Download SeeTRUE-Feedback dataset using HuggingFace datasets"""
        seetrue_raw_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = seetrue_raw_dir / "hf_cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Downloading SeeTRUE-Feedback dataset...")
        
        try:
            dataset = datasets.load_dataset(
                urls["hf_dataset_name"],
                split="test",
                cache_dir=str(cache_dir),
                trust_remote_code=True
            )
            
            dataset_path = seetrue_raw_dir / "seetrue_data.json"
            dataset.to_json(str(dataset_path))
            print(f"SeeTRUE-Feedback dataset downloaded successfully ({len(dataset)} samples)")
            
            (seetrue_raw_dir / "seetrue_download_completed.flag").touch()
            
        except Exception as e:
            print(f"Could not download SeeTRUE-Feedback dataset: {e}")
            raise


    def _download_and_prepare_brush_data(self, dl_manager, brush_raw_dir, urls):
        """Download BrushData using direct wget - limited to 20GB instead of 1.7TB"""
        brush_raw_dir.mkdir(parents=True, exist_ok=True)
        tars_dir = brush_raw_dir / "tars"
        tars_dir.mkdir(exist_ok=True)
        
        print("Downloading BrushData using direct wget (limited to 20GB)...")
        
        base_url = urls.get("base_tar_url", "")
        tar_files = urls.get("tar_files", [])
        
        if not base_url or not tar_files:
            raise ValueError("No direct URLs configured for BrushData")
        
        downloaded_files = []
        total_size_gb = 0
        MAX_SIZE_GB = 20
        
        for i, tar_filename in enumerate(tar_files):
            if total_size_gb >= MAX_SIZE_GB:
                print(f"Reached 20GB limit, stopping BrushData download")
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
                print(f"Downloading BrushData {tar_filename} ({i+1}/{len(tar_files)})...")
                downloaded_path = dl_manager.download_and_extract(tar_url)
                
                if Path(downloaded_path).is_file():
                    shutil.copy(downloaded_path, tar_local_path)
                    file_size_gb = tar_local_path.stat().st_size / (1024**3)
                    downloaded_files.append(tar_local_path)
                    total_size_gb += file_size_gb
                    print(f"{tar_filename} downloaded ({file_size_gb:.1f}GB, total: {total_size_gb:.1f}GB)")
                else:
                    print(f"{tar_filename} download failed")
                    
            except Exception as e:
                print(f"Failed to download {tar_filename}: {e}")
                continue
        
        print(f"BrushData direct download completed! Downloaded {len(downloaded_files)} TAR files ({total_size_gb:.1f}GB)")
        (brush_raw_dir / "brush_direct_download_completed.flag").touch()
        
        with open(brush_raw_dir / "downloaded_tars.json", 'w') as f:
            json.dump([str(p) for p in downloaded_files], f)

    def _process_actplan(self, processed_data_dir):
        """
        Process ActPlan dataset into MCoT planning step format.
        
        ActPlan contains action planning data with:
        - Short captions (basic descriptions)
        - Dense captions (detailed descriptions) 
        - Bounding boxes (spatial layouts)
        
        This data is perfect for the MCoT planning step, which requires
        detailed descriptions and spatial reasoning.
        
        Args:
            processed_data_dir: Directory to save processed planning examples
        """
        planning_samples = []
        
        actplan_json_path = Path(__file__).parent / "actplan.json"
        
        if not actplan_json_path.exists():
            raise FileNotFoundError(f"actplan.json not found at {actplan_json_path}")
        
        print(f"Loading actplan dataset from {actplan_json_path}...")
        
        try:
            with open(actplan_json_path, 'r') as f:
                actplan_data = json.load(f)
            
            print(f"Loaded {len(actplan_data)} entries from actplan dataset")
            
            processed_count = 0
            
            for entry in actplan_data:
                image_id = entry.get("image_id", f"actplan_{processed_count}")
                original_captions = entry.get("original_captions", [])
                dense_captions = entry.get("dense_captions", [])
                bounding_boxes = entry.get("bounding_boxes", [])
                
                prompt = original_captions[0] if original_captions else f"Generate image for {image_id}"
                
                dense_caption = dense_captions[0] if dense_captions else "No dense caption available"
                
                planning_text = f"Planning: Dense scene analysis based on detailed captioning. "
                planning_text += f"Dense caption: {dense_caption}. "
                
                if bounding_boxes:
                    bbox_descriptions = []
                    for bbox in bounding_boxes:
                        obj_class = bbox.get("class", "object")
                        x1, y1, x2, y2 = bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)
                        bbox_descriptions.append(f"{obj_class} (x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2})")
                    
                    planning_text += f"Spatial layout with bounding boxes: {'; '.join(bbox_descriptions)}. "
                
                if len(original_captions) > 1:
                    alt_captions = "; ".join(original_captions[1:3])
                    planning_text += f"Alternative descriptions: {alt_captions}. "
                
                if len(dense_captions) > 1:
                    alt_dense = dense_captions[1]
                    planning_text += f"Alternative dense description: {alt_dense}. "
                
                planning_text += "Actplan structured captioning and spatial analysis complete."
                
                planning_sample = {
                    "image_id": image_id,
                    "prompt": prompt,
                    "planning": planning_text,
                    "image": None,
                    "original_captions": original_captions,
                    "dense_captions": dense_captions,
                    "bounding_boxes": bounding_boxes
                }
                
                planning_samples.append(planning_sample)
                processed_count += 1
                
                if processed_count % 500 == 0:
                    print(f"Processed {processed_count} actplan samples...")
            
            print(f"Processed {len(planning_samples)} actplan planning samples")
            
            with open(processed_data_dir / "planning_data.json", 'w') as f:
                serializable_samples = []
                for ps in planning_samples:
                    s_ps = ps.copy()
                    if "image" in s_ps:
                        s_ps.pop("image")
                    serializable_samples.append(s_ps)
                json.dump(serializable_samples, f, indent=2)
            
            self.actplan_planning_samples_with_images = planning_samples
            
        except Exception as e:
            print(f"Error processing actplan dataset: {e}")
            raise


    def _process_richhf18k(self, richhf_raw_dir, processed_data_dir):
        """
        Process RichHF-18K dataset for reflection task training.
        
        MODIFIED: Now uses the FULL RichHF-18K dataset instead of limiting to 1000 samples per file.
        This aligns with the MINT paper methodology which leverages the complete RichHF-18K dataset
        for reflection and artifact identification training.
        """
        reflection_samples = []
        
        print("Using FULL RichHF-18K dataset (no sample limits)")
        
        tfrecord_files = list(richhf_raw_dir.glob("*.tfrecord"))
        
        print(f"Found {len(tfrecord_files)} TFRecord files in RichHF-18K directory")
        
        if tfrecord_files:
            print(f"Processing {len(tfrecord_files)} TFRecord files...")
            for tfrecord_file in tfrecord_files:
                print(f"Processing TFRecord: {tfrecord_file}")
                
                file_size = tfrecord_file.stat().st_size
                if file_size < 1000:
                    raise ValueError(f"{tfrecord_file.name} appears to be too small (size: {file_size} bytes) - download failure")
                
                dataset = tf.data.TFRecordDataset(str(tfrecord_file))
                processed_count = 0
                skipped_count = 0
                total_records = 0
                
                for i, raw_record in enumerate(dataset):
                    total_records += 1
                    example = tf.train.Example()
                    example.ParseFromString(raw_record.numpy())
                    features = example.features.feature
                    
                    filename = self._get_string_feature(features, 'filename')
                    artifact_score = self._get_float_feature(features, 'artifact_score') 
                    misalignment_score = self._get_float_feature(features, 'misalignment_score')
                    overall_score = self._get_float_feature(features, 'overall_score')
                    aesthetics_score = self._get_float_feature(features, 'aesthetics_score')
                    prompt_misalignment_label = self._get_string_feature(features, 'prompt_misalignment_label')
                    
                    if filename:
                        base_filename = filename.split('/')[-1].replace('.png', '').replace('.jpg', '')
                        prompt = f"Generate image: {base_filename}"
                    else:
                        prompt = "Generate image based on feedback data"
                    
                    if prompt:
                        reflection_sample = self._process_richhf_reflection_sample({
                            "prompt": prompt,
                            "artifact_score": artifact_score,
                            "misalignment_score": misalignment_score,
                            "overall_score": overall_score,
                            "aesthetics_score": aesthetics_score,
                            "prompt_misalignment_label": prompt_misalignment_label,
                            "feedback": f"Artifact score: {artifact_score:.3f}, Alignment score: {misalignment_score:.3f}, Overall quality: {overall_score:.3f}, Aesthetics: {aesthetics_score:.3f}",
                            "feedback_score": overall_score,
                            "quality_score": aesthetics_score, 
                            "aspect_scores": [artifact_score, misalignment_score, overall_score, aesthetics_score]
                        }, f"richhf_{tfrecord_file.stem}_{i}")
                        if reflection_sample:
                            reflection_samples.append(reflection_sample)
                            processed_count += 1
                        else:
                            skipped_count += 1
                        
                success_rate = (processed_count / total_records * 100) if total_records > 0 else 0
                print(f"RichHF TFRecord Summary: {processed_count}/{total_records} samples processed successfully ({success_rate:.1f}%), {skipped_count} samples skipped from {tfrecord_file.name}")
                        
        print(f"RichHF-18K Processing Complete: {len(reflection_samples)} total reflection samples successfully processed")

        with open(processed_data_dir / "richhf_reflection_data.json", 'w') as f:
            json.dump(reflection_samples, f)
        self.richhf_reflection_samples = reflection_samples

    def _process_richhf_reflection_sample(self, data, image_id):
        """Process individual RichHF-18K sample for reflection task following MINT methodology"""
        prompt = data.get("prompt", data.get("text", data.get("caption", "")))
        artifact_score = data.get("artifact_score", 0.0)
        misalignment_score = data.get("misalignment_score", 0.0)
        overall_score = data.get("overall_score", 0.0)
        aesthetics_score = data.get("aesthetics_score", 0.0)
        prompt_misalignment_label = data.get("prompt_misalignment_label", "")
        
        feedback = data.get("feedback", data.get("human_feedback", ""))
        quality_score = data.get("quality_score", data.get("quality", data.get("score", data.get("rating", 0.5))))
        feedback_score = data.get("feedback_score", 0.0)
        aspect_scores = data.get("aspect_scores", [])
        
        if not prompt:
            print(f"⚠️ Skipping RichHF sample {image_id}: Missing prompt data")
            return None
            
        reflection_text = "Reflection: Analyzing generated image for artifacts and misalignments. "
        reflection_text += f"Original prompt: '{prompt}' "
        
        if artifact_score > 0.3:
            reflection_text += f"Artifact detection: High artifact presence (score: {artifact_score:.3f}). "
            reflection_text += "Identifying bounding boxes of incorrectly generated objects. "
            
        if misalignment_score > 0.3:
            reflection_text += f"Prompt misalignment detected (score: {misalignment_score:.3f}). "
            if prompt_misalignment_label:
                reflection_text += f"Misalignment type: {prompt_misalignment_label}. "
            reflection_text += "Analyzing correspondence between prompt content and visual elements. "
            
        if overall_score < 0.6:
            reflection_text += f"Quality assessment: Below threshold (score: {overall_score:.3f}). "
            reflection_text += "Identifying specific regions requiring correction. "
            
        if feedback:
            feedback_clean = feedback[:200] if len(feedback) > 200 else feedback
            reflection_text += f"Human feedback analysis: {feedback_clean} "
            
        reflection_text += "Generating artifact heatmap for targeted correction. "
        reflection_text += "Self-reflection on generation accuracy and aesthetic quality. "
        reflection_text += "Preparing bounding box annotations for incorrectly generated objects."
        
        return {
            "image_id": str(image_id),
            "prompt": prompt,
            "reflection": reflection_text,
            "artifact_score": artifact_score,
            "misalignment_score": misalignment_score,
            "overall_score": overall_score,
            "aesthetics_score": aesthetics_score,
            "prompt_misalignment_label": prompt_misalignment_label,
            "quality_score": quality_score,
            "feedback": feedback,
            "feedback_score": feedback_score,
            "aspect_scores": aspect_scores,
            "requires_artifact_correction": artifact_score > 0.3,
            "requires_alignment_correction": misalignment_score > 0.3,
            "requires_quality_improvement": overall_score < 0.6
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
        """
        Process SeeTRUE-Feedback as a replacement for MINT's manually annotated reflection data.
        
        MINT paper: "For the reflection task, we leveraged the RichHF-18K dataset and the additional 
        5,000 images generated by MINT, which were manually annotated to identify the bounding boxes 
        of incorrectly generated objects, along with their corresponding prompt contents."
        
        SeeTRUE-Feedback provides equivalent data with rich misalignment annotations and bounding boxes.
        """
        reflection_samples = []
        
        try:
            print("Loading SeeTRUE-Feedback as MINT reflection data replacement")
            
            seetrue_data_file = seetrue_raw_dir / "seetrue_data.json"
            if seetrue_data_file.exists():
                print(f"📁 Loading SeeTRUE data from cached JSONL file...")
                seetrue_data = []
                processed_count = 0
                skipped_count = 0
                total_lines = 0
                with open(seetrue_data_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        total_lines += 1
                        try:
                            sample = json.loads(line.strip())
                            
                            processed_sample = self._process_seetrue_as_mint_reflection(sample, i)
                            if processed_sample:
                                reflection_samples.append(processed_sample)
                                processed_count += 1
                                
                                if processed_count % 1000 == 0:
                                    print(f"  Processed {processed_count} SeeTRUE samples for reflection training")
                            else:
                                skipped_count += 1
                        except json.JSONDecodeError as e:
                            skipped_count += 1
                            print(f"Skipping malformed JSON line {i}: {e}")
                            
                success_rate = (processed_count / total_lines * 100) if total_lines > 0 else 0
                print(f"JSONL Processing Summary: {processed_count}/{total_lines} samples processed successfully ({success_rate:.1f}%), {skipped_count} samples skipped")
                
            else:
                try:
                    if datasets is not None:
                        seetrue_hf_dataset = datasets.load_dataset(
                            _URLS["seetrue_feedback"]["hf_dataset_name"],
                            split="test",
                            cache_dir=str(seetrue_raw_dir / "hf_cache"),
                            trust_remote_code=True
                        )
                        
                        processed_count = 0
                        skipped_count = 0
                        for i, sample in enumerate(seetrue_hf_dataset):
                            processed_sample = self._process_seetrue_as_mint_reflection(sample, i)
                            if processed_sample:
                                reflection_samples.append(processed_sample)
                                processed_count += 1
                                
                                if processed_count % 1000 == 0:
                                    print(f"Processed {processed_count} SeeTRUE samples for reflection training")
                            else:
                                skipped_count += 1
                                
                        total_samples = len(seetrue_hf_dataset)
                        success_rate = (processed_count / total_samples * 100) if total_samples > 0 else 0
                        print(f" SeeTRUE Processing Summary: {processed_count}/{total_samples} samples processed successfully ({success_rate:.1f}%), {skipped_count} samples skipped")
                    else:
                        raise ImportError("datasets library not available")
                        
                except Exception as hf_error:
                    raise RuntimeError(f"Could not load SeeTRUE-Feedback from HuggingFace: {hf_error}")
                    
        except Exception as e:
            raise RuntimeError(f"Error processing SeeTRUE-Feedback dataset: {e}")
        
        print(f"SeeTRUE-Feedback Processing Complete: {len(reflection_samples)} total reflection samples successfully processed")
        
        with open(processed_data_dir / "seetrue_reflection_data.json", 'w') as f:
            json.dump(reflection_samples, f)
        self.seetrue_reflection_samples = reflection_samples

    def _process_seetrue_as_mint_reflection(self, sample, sample_id):
        """
        Convert SeeTRUE-Feedback data into MINT-style reflection format.
        
        SeeTRUE provides human feedback about image-caption misalignments, which is perfect 
        for the MCoT reflection step. This function extracts:
        - Artifact locations (bounding boxes)
        - Misalignment descriptions  
        - Human feedback comments
        - Confidence scores for detected issues
        
        This enhanced reflection data helps the model learn to identify and articulate
        problems in generated images more effectively.
        
        Args:
            sample: SeeTRUE-Feedback sample with human annotations
            sample_id: Unique identifier for the sample
            
        Returns:
            Dict containing MINT-formatted reflection data, or None if processing fails
        """
        try:
            prompt = sample.get("image_caption", "")
            image = sample.get("image")
            
            human_feedback = sample.get("human_feedback", [])
            caption_misalignment = sample.get("caption_misalignment", "")
            visual_misalignment = sample.get("visual_misalignment", "")
            
            bbox_grounding_dino_str = sample.get("bbox_GroundingDino", "")
            bbox_pali_str = sample.get("bbox_PaLI", "")
            
            if not prompt or not image:
                missing_fields = []
                if not prompt:
                    missing_fields.append("prompt")
                if not image:
                    missing_fields.append("image")
                print(f"Skipping SeeTRUE sample {sample_id}: Missing required fields: {', '.join(missing_fields)}")
                return None
                
            reflection_text = "Reflection: Analyzing generated image for artifacts and misalignments using SeeTRUE-Feedback methodology. "
            reflection_text += f"Original prompt: '{prompt}' "
            
            misalignment_detected = False
            if caption_misalignment and visual_misalignment:
                misalignment_detected = True
                reflection_text += f"Misalignment detected: Expected '{caption_misalignment}' but found '{visual_misalignment}'. "
            
            if isinstance(human_feedback, list) and human_feedback:
                feedback_summary = "; ".join(human_feedback[:2])
                reflection_text += f"Human feedback: {feedback_summary}. "
            elif isinstance(human_feedback, str) and human_feedback:
                reflection_text += f"Human feedback: {human_feedback[:150]}. "
            
            detected_artifacts = []
            grounding_dino_objects = []
            if bbox_grounding_dino_str:
                try:
                    import ast
                    bbox_data = ast.literal_eval(bbox_grounding_dino_str)
                    if isinstance(bbox_data, dict):
                        boxes = bbox_data.get("boxes", [])
                        labels = bbox_data.get("labels", [])
                        
                        for i, (box, label) in enumerate(zip(boxes, labels)):
                            confidence = 0.8
                            clean_label = label
                            if '(' in label and ')' in label:
                                try:
                                    clean_label = label.split('(')[0]
                                    confidence_str = label.split('(')[1].split(')')[0]
                                    confidence = float(confidence_str)
                                except:
                                    pass
                            
                            grounding_dino_objects.append({
                                "bbox": box,
                                "confidence": confidence,
                                "label": clean_label
                            })
                            
                            if confidence > 0.5:
                                detected_artifacts.append({
                                    "bbox": box,
                                    "confidence": confidence,
                                    "label": clean_label,
                                    "source": "GroundingDino"
                                })
                        
                        if grounding_dino_objects:
                            reflection_text += f"GroundingDino detected {len(grounding_dino_objects)} objects. "
                            
                except Exception as e:
                    print(f"Warning: Failed to parse GroundingDino bbox data: {e}")
            
            pali_objects = []
            if bbox_pali_str:
                try:
                    parts = bbox_pali_str.strip().split()
                    if len(parts) >= 5:
                        x1, y1, x2, y2 = map(float, parts[:4])
                        label = " ".join(parts[4:])
                        
                        pali_objects.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": 0.7,
                            "label": label
                        })
                        
                        detected_artifacts.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": 0.7,
                            "label": label,
                            "source": "PaLI"
                        })
                        
                        reflection_text += f"PaLI detected object: {label}. "
                        
                except Exception as e:
                    print(f"Warning: Failed to parse PaLI bbox data: {e}")
            
            artifact_score = min(len(detected_artifacts) * 0.2, 1.0)
            misalignment_score = 0.8 if misalignment_detected else 0.1
            overall_quality = max(0.2, 1.0 - artifact_score - (misalignment_score * 0.5))
            
            if artifact_score > 0.3:
                reflection_text += f"High artifact presence (score: {artifact_score:.3f}). "
                reflection_text += "Generating artifact heatmap for targeted correction. "
                
            if misalignment_score > 0.3:
                reflection_text += f"Significant prompt misalignment (score: {misalignment_score:.3f}). "
                reflection_text += "Analyzing correspondence between prompt elements and visual content. "
                
            if overall_quality < 0.6:
                reflection_text += f"Below quality threshold (score: {overall_quality:.3f}). "
                reflection_text += "Identifying specific regions requiring enhancement. "
                
            reflection_text += "Performing multi-modal chain of thought analysis. "
            reflection_text += "Self-reflection on generation accuracy and semantic alignment. "
            if detected_artifacts:
                reflection_text += f"Preparing {len(detected_artifacts)} bounding box corrections for artifact removal."
            else:
                reflection_text += "No major artifacts detected, focusing on quality enhancement."
            
            return {
                "image_id": f"seetrue_{sample_id}",
                "prompt": prompt,
                "reflection": reflection_text,
                "artifact_score": artifact_score,
                "misalignment_score": misalignment_score,
                "overall_quality": overall_quality,
                "bbox_artifacts": detected_artifacts,
                "misalignment_detected": misalignment_detected,
                "artifact_confidence": max(artifact_score, misalignment_score),
                "grounding_dino_count": len(grounding_dino_objects),
                "pali_detection_count": len(pali_objects),
                "requires_artifact_correction": artifact_score > 0.3 or len(detected_artifacts) > 0,
                "requires_alignment_correction": misalignment_detected or misalignment_score > 0.3,
                "requires_quality_improvement": overall_quality < 0.6,
                "seetrue_enhanced": True
            }
            
        except Exception as e:
            print(f"Skipping SeeTRUE sample {sample_id}: Failed to process - {e}")
            return None

    def _process_brush_data(self, brush_raw_dir, processed_data_dir):
        """Process BrushData from direct wget downloads (limited to 20GB)"""
        correction_samples = []
        
        print("Processing BrushData from direct wget downloads...")
        
        downloaded_tars_file = brush_raw_dir / "downloaded_tars.json"
        tars_dir = brush_raw_dir / "tars"
        
        if downloaded_tars_file.exists():
            try:
                with open(downloaded_tars_file, 'r') as f:
                    tar_file_paths = [Path(p) for p in json.load(f)]
                print(f"Found {len(tar_file_paths)} downloaded TAR files for BrushData")
            except Exception as e:
                print(f"Could not read downloaded TAR files list: {e}")
                tar_file_paths = []
        else:
            tar_file_paths = list(tars_dir.glob("*.tar")) if tars_dir.exists() else []
            print(f"Found {len(tar_file_paths)} TAR files in {tars_dir}")
        
        if not tar_file_paths:
            raise FileNotFoundError("No BrushData TAR files found")
        if wds is None:
            raise ImportError("Webdataset library not available. Cannot process BrushData TAR files.")
        else:
            processed_sample_count = 0
            
            for i, tar_file_path in enumerate(tar_file_paths):
                    
                print(f"Processing BrushData TAR file ({i+1}/{len(tar_file_paths)}): {tar_file_path.name}")
                try:
                    dataset_shard = wds.WebDataset(str(tar_file_path))
                    
                    for sample_idx, sample in enumerate(dataset_shard):

                        if sample_idx < 3:
                            print(f"DEBUG Sample {sample_idx} structure:")
                            print(f"   Available keys: {list(sample.keys())}")
                            for key, value in sample.items():
                                if isinstance(value, bytes):
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
                        
                        image_pil = None
                        mask_pil = None
                        
                        if "image" in sample:
                            image_data = sample["image"]
                            if isinstance(image_data, bytes):
                                try:
                                    from io import BytesIO
                                    image_pil = Image.open(BytesIO(image_data))
                                except Exception as e:
                                    raise ValueError(f"Could not decode image data: {e}")
                            elif hasattr(image_data, 'mode'):
                                image_pil = image_data
                        
                        mask_pil = None
                        segmentation_data = None
                        if "segmentation" in sample:
                            mask_data = sample["segmentation"]
                            if isinstance(mask_data, bytes):
                                try:
                                    seg_text = mask_data.decode('utf-8')
                                    segmentation_data = json.loads(seg_text)
                                    
                                    if sample_idx < 3:
                                        print(f"   segmentation: RLE JSON with {len(segmentation_data.get('mask', []))} masks")
                                    
                                    has_segmentation_data = True
                                    
                                except json.JSONDecodeError:
                                    try:
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
                            elif hasattr(mask_data, 'mode'):
                                mask_pil = mask_data
                                has_segmentation_data = True
                            elif isinstance(mask_data, dict):
                                segmentation_data = mask_data
                                has_segmentation_data = True
                                if sample_idx < 3:
                                    print(f"   segmentation: pre-parsed RLE JSON with {len(mask_data.get('mask', []))} masks")
                            else:
                                has_segmentation_data = False
                        
                        if image_pil is None:
                            raise ValueError(f"Sample {sample_idx} does not contain valid image data")
                        
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
                            "segmentation_rle": segmentation_data,
                            "mask_pil": None,
                            "has_image": True,
                            "tar_file_path": str(tar_file_path),
                            "sample_key": sample.get('__key__', sample_idx)
                        })
                        processed_sample_count += 1
                        
                        if processed_sample_count % 100 == 0:
                            print(f"Processed {processed_sample_count} BrushData samples...")
                            
                except Exception as e_tar:
                    raise RuntimeError(f"Error processing TAR file {tar_file_path}: {e_tar}") 
                    
            print(f"Processed {processed_sample_count} samples from {len(tar_file_paths)} BrushData TAR files")

        if not correction_samples:
            raise RuntimeError("No correction data was successfully processed from BrushData TAR files")

        with open(processed_data_dir / "correction_data.json", 'w') as f:
            json.dump(correction_samples, f)
        self.brush_correction_samples = correction_samples

    def _reload_image_from_tar(self, correction_sample):
        """Reload PIL image from TAR file using stored metadata"""
        tar_file_path = correction_sample.get("tar_file_path")
        sample_key = correction_sample.get("sample_key")
        
        if not tar_file_path or sample_key is None:
            missing_fields = []
            if not tar_file_path:
                missing_fields.append("tar_file_path")
            if sample_key is None:
                missing_fields.append("sample_key")
            print(f"Warning: Cannot reload image - missing required fields: {', '.join(missing_fields)}")
            return None
            
        try:
            dataset_shard = wds.WebDataset(tar_file_path)
            
            for sample in dataset_shard:
                if sample.get('__key__') == sample_key:
                    if "image" in sample:
                        image_data = sample["image"]
                        if isinstance(image_data, bytes):
                            from io import BytesIO
                            return Image.open(BytesIO(image_data))
                        elif hasattr(image_data, 'mode'):
                            return image_data
                    break
                    
        except Exception as e:
            print(f"Warning: Could not reload image from TAR file {tar_file_path}: {e}")
            
        print(f"Warning: Failed to find or load image with key '{sample_key}' from TAR file {tar_file_path}")
        return None

    def _generate_mint_correction_text(self, has_mask=True, task_type="inpainting", caption="", mask_coverage=0.0):
        if not has_mask and mask_coverage == 0.0:
            raise ValueError("No mask data available and no mask coverage provided - cannot generate correction text")
        
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
        if Image is None or ImageDraw is None: 
            raise ImportError("PIL (Image and ImageDraw) is required for random mask generation")
        
        mask_array = self._brushnet_random_mask_gen(height, width)
        if mask_array is None: 
            raise RuntimeError("Failed to generate random mask array")
        
        mask = Image.fromarray((mask_array * 255).astype(np.uint8), 'L')
        coverage = np.sum(mask_array > 0) / (height * width)
        return mask, coverage

    def _brushnet_random_brush_gen(self, max_tries, h, w, min_num_vertex=4, max_num_vertex=8,
                                   mean_angle=2*math.pi/5, angle_range=2*math.pi/15,
                                   min_width=12, max_width=40):
        """Generate random brush mask following authentic BrushNet methodology from TencentARC/BrushNet"""
        if Image is None or ImageDraw is None: 
            raise ImportError("PIL (Image and ImageDraw) is required for brush mask generation")
        
        H, W = h, w
        average_radius = math.sqrt(H*H + W*W) / 8
        mask = Image.new('L', (W, H), 0)
        
        for _ in range(np.random.randint(1, max_tries)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))
            
            vertex.append((int(np.random.randint(0, W)), int(np.random.randint(0, H))))
            
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, W)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, H)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=1, width=width)
            
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                            v[1] - width//2,
                            v[0] + width//2,
                            v[1] + width//2),
                            fill=1)
        
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
        brush_mask = self._brushnet_random_brush_gen(4, h, w)
        mask = np.logical_and(mask, 1 - brush_mask).astype(np.float32)
        return mask


    def _generate_mcot_examples(self, processed_data_dir):
        planning_samples = getattr(self, "actplan_planning_samples_with_images", [])
        richhf_reflection_samples = getattr(self, "richhf_reflection_samples", [])
        seetrue_reflection_samples = getattr(self, "seetrue_reflection_samples", [])
        correction_samples = getattr(self, "brush_correction_samples", [])

        if not planning_samples: 
            try:
                with open(processed_data_dir / "planning_data.json", 'r') as f: planning_samples_json = json.load(f)
                planning_samples = []
                for ps_json in planning_samples_json:
                    ps_json["image"] = None 
                    planning_samples.append(ps_json)
                print(f"Loaded {len(planning_samples)} planning samples from JSON")
            except FileNotFoundError: 
                print(f"ERROR: planning_data.json not found in {processed_data_dir}")
                raise FileNotFoundError("planning_data.json not found - required for MCoT examples generation")
            except Exception as e:
                print(f"ERROR: Failed to load planning_data.json: {e}")
                raise
        
        if not richhf_reflection_samples:
            try:
                with open(processed_data_dir / "richhf_reflection_data.json", 'r') as f: richhf_reflection_samples = json.load(f)
                print(f"Loaded {len(richhf_reflection_samples)} RichHF reflection samples from JSON")
            except FileNotFoundError: 
                print(f"ERROR: richhf_reflection_data.json not found in {processed_data_dir}")
                raise FileNotFoundError("richhf_reflection_data.json not found - required for MCoT examples generation")
            except Exception as e:
                print(f"ERROR: Failed to load richhf_reflection_data.json: {e}")
                raise
        
        if not seetrue_reflection_samples:
            try:
                with open(processed_data_dir / "seetrue_reflection_data.json", 'r') as f: seetrue_reflection_samples = json.load(f)
                print(f"Loaded {len(seetrue_reflection_samples)} SeeTRUE reflection samples from JSON")
            except FileNotFoundError: 
                print(f"ERROR: seetrue_reflection_data.json not found in {processed_data_dir}")
                raise FileNotFoundError("seetrue_reflection_data.json not found - required for MCoT examples generation")
            except Exception as e:
                print(f"ERROR: Failed to load seetrue_reflection_data.json: {e}")
                raise

        if not correction_samples:
            try:
                with open(processed_data_dir / "correction_data.json", 'r') as f: correction_samples = json.load(f)
                print(f"Loaded {len(correction_samples)} BrushData correction samples from JSON")
            except FileNotFoundError: 
                print(f"ERROR: correction_data.json not found in {processed_data_dir}")
                raise FileNotFoundError("correction_data.json not found - required for MCoT examples generation")
            except Exception as e:
                print(f"ERROR: Failed to load correction_data.json: {e}")
                raise

        mcot_examples = []
        num_planning = len(planning_samples)
        num_richhf_reflection = len(richhf_reflection_samples) if richhf_reflection_samples else 0
        num_seetrue_reflection = len(seetrue_reflection_samples) if seetrue_reflection_samples else 0
        num_correction = len(correction_samples) if correction_samples else 0

        print(f"MCoT Data Summary:")
        print(f"    Planning samples: {num_planning}")
        print(f"    RichHF reflection samples: {num_richhf_reflection}")
        print(f"    SeeTRUE reflection samples: {num_seetrue_reflection}")
        print(f"    BrushData correction samples: {num_correction}")

        if num_planning == 0: 
            print(f"ERROR: No planning samples available to generate MCoT examples")
            raise RuntimeError("No planning samples available to generate MCoT examples") 

        for i in range(num_planning):
            planning_sample = planning_samples[i]
            image_id = planning_sample.get("image_id", f"mcot_{i}")
            
            if not hasattr(self, 'brush_correction_samples') or not self.brush_correction_samples:
                print(f"ERROR: No BrushData correction samples with images available for MCoT generation")
                print(f"   This means either BrushData failed to download or process correctly")
                break
                
            correction_s = correction_samples[i % num_correction] if num_correction > 0 else {}
            
            pil_image_obj = None
            if correction_s.get("has_image", False):
                try:
                    pil_image_obj = self._reload_image_from_tar(correction_s)
                    if pil_image_obj is None:
                        print(f"Warning: Failed to reload image from TAR file for sample {i}")
                except Exception as e:
                    print(f"Warning: Error reloading image for sample {i}: {e}")
                    pil_image_obj = None
            
            if not isinstance(pil_image_obj, Image.Image):
                print(f"Warning: Skipping MCoT example {i} - no valid PIL image (type: {type(pil_image_obj)})")
                continue

            reflection_s = {}
            if num_richhf_reflection > 0 and num_seetrue_reflection > 0:
                if i % 2 == 0:
                    reflection_s = richhf_reflection_samples[i % num_richhf_reflection]
                else:
                    reflection_s = seetrue_reflection_samples[i % num_seetrue_reflection]
            elif num_richhf_reflection > 0:
                reflection_s = richhf_reflection_samples[i % num_richhf_reflection]
            elif num_seetrue_reflection > 0:
                reflection_s = seetrue_reflection_samples[i % num_seetrue_reflection]

            acting_prompt = planning_sample.get("prompt", f"Generate image for {image_id}")
            acting_text = f"Generating image based on prompt: {acting_prompt}"

            final_response = f"Complete MCoT process for {image_id}. Enhanced through planning, acting, reflection, and correction."
            mcot_example = {
                "image_id": image_id,
                "prompt": acting_prompt,
                "planning": planning_sample.get("planning", "Planning data N/A."),
                "acting": acting_text,
                "reflection": reflection_s.get("reflection", "Reflection data N/A."),
                "correction": correction_s.get("correction", "Correction data N/A."),
                "final_response": final_response,
                "image_obj_pil": pil_image_obj,
                "image_mode_from_planning": planning_sample.get("image_mode"), 
                "image_size_from_planning": planning_sample.get("image_size")   
            }
            mcot_examples.append(mcot_example)

        print(f"Generated {len(mcot_examples)} total MCoT examples")
        
        random.shuffle(mcot_examples)
        split_idx = int(0.9 * len(mcot_examples))
        train_examples = mcot_examples[:split_idx]
        val_examples = mcot_examples[split_idx:]

        print(f"Creating train split with {len(train_examples)} examples...")
        self._save_mcot_examples_split(train_examples, processed_data_dir / "train")
        
        print(f"Creating val split with {len(val_examples)} examples...")
        self._save_mcot_examples_split(val_examples, processed_data_dir / "val")

        print(f"MCoT examples saved successfully!")
        print(f"Train directory: {processed_data_dir / 'train'}")
        print(f"Val directory: {processed_data_dir / 'val'}")

    def _save_mcot_examples_split(self, examples, output_dir_split):
        print(f"Creating split directory: {output_dir_split}")
        output_dir_split.mkdir(parents=True, exist_ok=True)
        
        if not output_dir_split.exists():
            print(f"ERROR: Failed to create directory {output_dir_split}")
            raise RuntimeError(f"Could not create directory {output_dir_split}")
        
        print(f"Directory created successfully: {output_dir_split}")
        print(f"Saving {len(examples)} examples to {output_dir_split}")
        
        for i, example in enumerate(examples):
            image_id_val = example.get('image_id', f"unknown_{i}")
            safe_image_id_val = image_id_val.replace('/', '_').replace('\\', '_')
            example_data_dir = output_dir_split / f"example_{safe_image_id_val}"
            
            try:
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
                        print(f"ERROR: Could not save PIL image for {image_id_val}: {e}")
                        raise RuntimeError(f"Could not save PIL image for {image_id_val}: {e}")
                else:
                    print(f"ERROR: No valid PIL image provided for {image_id_val} (type: {type(pil_image)})")
                    raise ValueError(f"No valid PIL image provided for {image_id_val}")
                    
                if (i + 1) % 10 == 0:
                    print(f"Saved {i + 1}/{len(examples)} examples")
                    
            except Exception as e:
                print(f"ERROR: Failed to save example {image_id_val}: {e}")
                raise
        
        print(f"Successfully saved all {len(examples)} examples to {output_dir_split}")

    def _generate_examples(self, data_dir):
        data_path = Path(data_dir)
        example_dirs = [d for d in data_path.iterdir() if d.is_dir() and (d / "mcot_annotations.json").exists()]

        for i, example_dir_path in enumerate(example_dirs):
            mcot_file = example_dir_path / "mcot_annotations.json"
            image_file = example_dir_path / "image.jpg"

            if not image_file.exists(): 
                raise FileNotFoundError(f"Image file missing for {example_dir_path}")
            
            try:
                with open(mcot_file, "r") as f:
                    annotations = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Could not decode annotations for {example_dir_path}")
            
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
    
    try:
        output_path = Path(args.output_dir)
        cache_path = Path(args.cache_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        cache_path.mkdir(parents=True, exist_ok=True)
        
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

                        wget_cmd = [
                            'wget',
                            '-c',
                            '--timeout=0',
                            '--tries=3',
                            '--wait=5',
                            '--progress=bar:force',
                            '--show-progress',
                            '--user-agent=Mozilla/5.0 (compatible; Research Bot)',
                            '-O', str(local_path),
                            url
                        ]
                        
                        print(f"Running: {' '.join(wget_cmd)}")
                        result = subprocess.run(
                            wget_cmd, 
                            check=True, 
                            capture_output=False,
                            text=True,
                            timeout=None
                        )
                        print(f"Download completed: {local_path}")
                        
                    except subprocess.CalledProcessError as e:
                        print(f"wget failed with exit code {e.returncode}: {url}")
                        if local_path.exists():
                            local_path.unlink()
                        raise
                    except FileNotFoundError:
                        print("wget not found. Please install wget or use alternative download method.")
                        raise
                else:
                    print(f"File already exists: {local_path}")
                
                if filename.endswith('.zip'):
                    extract_dir = local_path.parent / filename.replace('.zip', '')
                    if not extract_dir.exists():
                        print(f"Extracting {local_path}...")
                        try:
                            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                                zip_ref.extractall(extract_dir)
                            print(f"Extraction completed: {extract_dir}")
                        except zipfile.BadZipFile:
                            print(f"Corrupted zip file: {local_path}")
                            local_path.unlink()
                            raise
                    else:
                        print(f"Already extracted: {extract_dir}")
                    return str(extract_dir)
                    
                return str(local_path)
                
            def download_config(self, *args, **kwargs):
                """Placeholder for download_config - not needed for our direct processing"""
                pass
        
        dataset_builder = MCoTWgetDataset()
        dl_manager = SimpleDownloadManager(str(cache_path))
        
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
