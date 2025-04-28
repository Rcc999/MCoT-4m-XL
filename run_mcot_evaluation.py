#!/usr/bin/env python3
# Copyright 2024 EPFL and Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation script for the MCOT model.
Evaluates the model on various metrics:
1. VQAv2 accuracy (for VQA baseline)
2. COCO layout box IoU (for Planning stage)
3. RichHF-18K mask F1 (for Reflection stage)
4. COCO-Stuff PSNR/SSIM (for Correction stage)
"""

import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from fourm.data.mcot_dataset import build_mcot_dataset, PLANNING_START_TOKEN, REFLECTION_START_TOKEN, CORRECTION_START_TOKEN, build_mcot_huggingface_dataset
from fourm.models.mcot_generate import MCOTGenerationSampler, build_mcot_generation_schedules
from fourm.utils import create_model
from fourm.data.modality_info import MODALITY_INFO

# Evaluation metrics
from sklearn.metrics import f1_score, precision_score, recall_score
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure


def parse_args():
    parser = argparse.ArgumentParser(description="MCOT Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--tokenizer-path", type=str, 
                        default="fourm/utils/tokenizer/trained/text_tokenizer_4m_mcot.json", 
                        help="Path to the text tokenizer")
    parser.add_argument("--vqa-dataset", type=str, required=True, help="Path to the VQA dataset")
    parser.add_argument("--planning-dataset", type=str, required=True, help="Path to the COCO dataset for planning evaluation")
    parser.add_argument("--reflection-dataset", type=str, required=True, help="Path to the RichHF dataset for reflection evaluation")
    parser.add_argument("--correction-dataset", type=str, required=True, help="Path to the COCO-Stuff dataset for correction evaluation")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-outputs", action="store_true", help="Save model outputs for visualization")
    parser.add_argument("--use-huggingface", action="store_true", help="Use datasets from Hugging Face")
    parser.add_argument("--vqa-dataset-hf", type=str, default="HuggingFaceM4/VQAv2", help="HuggingFace dataset name for VQA")
    parser.add_argument("--planning-dataset-hf", type=str, default="facebook/coco", help="HuggingFace dataset name for Planning")
    parser.add_argument("--reflection-dataset-hf", type=str, default="", help="HuggingFace dataset name for Reflection")
    parser.add_argument("--correction-dataset-hf", type=str, default="shunk031/cocostuff", help="HuggingFace dataset name for Correction")
    return parser.parse_args()


def calculate_box_iou(pred_boxes, target_boxes):
    """
    Calculate IoU between predicted and target bounding boxes.
    
    Args:
        pred_boxes: Predicted bounding boxes in format [x1, y1, x2, y2]
        target_boxes: Target bounding boxes in format [x1, y1, x2, y2]
        
    Returns:
        IoU scores
    """
    # Calculate intersection coordinates
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    # Calculate areas
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou


def calculate_mask_f1(pred_masks, target_masks):
    """
    Calculate F1 score between predicted and target binary masks.
    
    Args:
        pred_masks: Predicted binary masks
        target_masks: Target binary masks
        
    Returns:
        F1 scores
    """
    # Flatten the masks
    pred_flat = pred_masks.reshape(-1).cpu().numpy()
    target_flat = target_masks.reshape(-1).cpu().numpy()
    
    # Calculate metrics
    f1 = f1_score(target_flat, pred_flat, average='binary', zero_division=0)
    precision = precision_score(target_flat, pred_flat, average='binary', zero_division=0)
    recall = recall_score(target_flat, pred_flat, average='binary', zero_division=0)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def calculate_image_metrics(pred_images, target_images):
    """
    Calculate PSNR and SSIM between predicted and target images.
    
    Args:
        pred_images: Predicted RGB images
        target_images: Target RGB images
        
    Returns:
        PSNR and SSIM scores
    """
    # Calculate PSNR
    psnr = peak_signal_noise_ratio(pred_images, target_images)
    
    # Calculate SSIM
    ssim = structural_similarity_index_measure(pred_images, target_images)
    
    return {
        'psnr': psnr.item(),
        'ssim': ssim.item()
    }


def evaluate_vqa(model, dataloader, tokenizer, device, num_samples):
    """
    Evaluate the model on VQA task.
    
    Args:
        model: The MCOT model
        dataloader: DataLoader for VQA dataset
        tokenizer: Text tokenizer
        device: Device to run the evaluation on
        num_samples: Number of samples to evaluate
        
    Returns:
        VQA accuracy
    """
    print("Evaluating VQA performance...")
    
    sampler = MCOTGenerationSampler(model)
    schedules = build_mcot_generation_schedules(MODALITY_INFO)
    
    # List to store the accuracies
    accuracies = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= num_samples:
                break
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        if torch.is_tensor(batch[key][sub_key]):
                            batch[key][sub_key] = batch[key][sub_key].to(device)
                elif torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # Generate VQA response (Stage 1 only - Planning)
            results = sampler.sampler.generate(
                batch,
                schedules["planning"],
                top_k=0.0,
                top_p=0.9,
                text_tokenizer=tokenizer,
                verbose=False
            )
            
            # Extract predicted answer from caption
            pred_answers = []
            for caption_tensor in results["caption"]["tensor"]:
                # Decode the tensor to get the answer string
                tokens = caption_tensor.cpu().numpy()
                answer = tokenizer.decode(tokens.tolist())
                
                # Clean up the answer (remove special tokens, padding, etc.)
                clean_answer = answer.replace("[PAD]", "").replace("[S_1]", "").replace("[S_2]", "").strip()
                pred_answers.append(clean_answer)
            
            # Extract ground truth answer
            gt_answers = []
            for caption_tensor in batch["caption"]["tensor"]:
                tokens = caption_tensor.cpu().numpy()
                answer = tokenizer.decode(tokens.tolist())
                
                # Clean up the answer
                clean_answer = answer.replace("[PAD]", "").replace("[S_1]", "").replace("[S_2]", "").strip()
                gt_answers.append(clean_answer)
            
            # Calculate accuracy (exact match)
            batch_accuracy = sum(1 for pred, gt in zip(pred_answers, gt_answers) if pred.lower() == gt.lower()) / len(pred_answers)
            accuracies.append(batch_accuracy)
    
    # Calculate overall accuracy
    vqa_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    print(f"VQA Accuracy: {vqa_accuracy:.4f}")
    return vqa_accuracy


def evaluate_planning(model, dataloader, tokenizer, device, num_samples):
    """
    Evaluate the model on Planning task (COCO layout box IoU).
    
    Args:
        model: The MCOT model
        dataloader: DataLoader for COCO dataset
        tokenizer: Text tokenizer
        device: Device to run the evaluation on
        num_samples: Number of samples to evaluate
        
    Returns:
        Box IoU scores
    """
    print("Evaluating Planning performance (bbox IoU)...")
    
    sampler = MCOTGenerationSampler(model)
    schedules = build_mcot_generation_schedules(MODALITY_INFO)
    
    # List to store the IoU scores
    iou_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= num_samples:
                break
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        if torch.is_tensor(batch[key][sub_key]):
                            batch[key][sub_key] = batch[key][sub_key].to(device)
                elif torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # Prepare input with Planning token
            planning_input = batch.copy()
            if "text" in planning_input:
                # Add planning token to text input
                planning_token_id = tokenizer.token_to_id(PLANNING_START_TOKEN)
                if planning_token_id is not None:
                    text_tensor = planning_input["text"]["tensor"]
                    new_tensor = torch.zeros_like(text_tensor)
                    new_tensor[:, 0] = planning_token_id
                    
                    # Find first non-zero token
                    non_zero_idx = torch.nonzero(text_tensor, as_tuple=True)[1]
                    if len(non_zero_idx) > 0:
                        first_idx = non_zero_idx[0].item()
                        new_tensor[:, 1:] = text_tensor[:, first_idx:]
                    
                    planning_input["text"]["tensor"] = new_tensor
            
            # Generate Planning outputs
            results = sampler.sampler.generate(
                planning_input,
                schedules["planning"],
                top_k=0.0,
                top_p=0.9,
                text_tokenizer=tokenizer,
                verbose=False
            )
            
            # Extract predicted boxes
            if "bbox" in results:
                # Convert bbox tensor to [x1, y1, x2, y2] format
                pred_boxes = []
                for bbox_tensor in results["bbox"]["tensor"]:
                    # Decode the tensor to get bbox coordinates
                    bbox_str = tokenizer.decode(bbox_tensor.cpu().numpy().tolist())
                    
                    # Parse bbox string to get coordinates
                    try:
                        # This would depend on the format of your bbox strings
                        # Example format: "x1 y1 x2 y2"
                        coords = [float(c) for c in bbox_str.split() if c.replace('.', '', 1).isdigit()]
                        if len(coords) >= 4:
                            # Keep only the first bbox if multiple are present
                            pred_boxes.append(coords[:4])
                        else:
                            pred_boxes.append([0.0, 0.0, 1.0, 1.0])  # Default box if parsing fails
                    except:
                        pred_boxes.append([0.0, 0.0, 1.0, 1.0])  # Default box if parsing fails
                
                pred_boxes = torch.tensor(pred_boxes, device=device)
                
                # Extract ground truth boxes
                gt_boxes = []
                for bbox_tensor in batch["bbox"]["tensor"]:
                    # Decode the tensor to get bbox coordinates
                    bbox_str = tokenizer.decode(bbox_tensor.cpu().numpy().tolist())
                    
                    # Parse bbox string to get coordinates
                    try:
                        coords = [float(c) for c in bbox_str.split() if c.replace('.', '', 1).isdigit()]
                        if len(coords) >= 4:
                            gt_boxes.append(coords[:4])
                        else:
                            gt_boxes.append([0.0, 0.0, 1.0, 1.0])
                    except:
                        gt_boxes.append([0.0, 0.0, 1.0, 1.0])
                
                gt_boxes = torch.tensor(gt_boxes, device=device)
                
                # Calculate IoU between predicted and ground truth boxes
                batch_ious = calculate_box_iou(pred_boxes, gt_boxes)
                iou_scores.extend(batch_ious.cpu().numpy().tolist())
    
    # Calculate average IoU
    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    
    print(f"Average Box IoU: {avg_iou:.4f}")
    return avg_iou


def evaluate_reflection(model, dataloader, tokenizer, device, num_samples):
    """
    Evaluate the model on Reflection task (RichHF-18K mask F1).
    
    Args:
        model: The MCOT model
        dataloader: DataLoader for RichHF dataset
        tokenizer: Text tokenizer
        device: Device to run the evaluation on
        num_samples: Number of samples to evaluate
        
    Returns:
        Mask F1 scores
    """
    print("Evaluating Reflection performance (mask F1)...")
    
    sampler = MCOTGenerationSampler(model)
    schedules = build_mcot_generation_schedules(MODALITY_INFO)
    
    # Lists to store the metrics
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= num_samples:
                break
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        if torch.is_tensor(batch[key][sub_key]):
                            batch[key][sub_key] = batch[key][sub_key].to(device)
                elif torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # Prepare input with Reflection token
            reflection_input = batch.copy()
            if "text" in reflection_input:
                # Add reflection token to text input
                reflection_token_id = tokenizer.token_to_id(REFLECTION_START_TOKEN)
                if reflection_token_id is not None:
                    text_tensor = reflection_input["text"]["tensor"]
                    new_tensor = torch.zeros_like(text_tensor)
                    new_tensor[:, 0] = reflection_token_id
                    
                    # Find first non-zero token
                    non_zero_idx = torch.nonzero(text_tensor, as_tuple=True)[1]
                    if len(non_zero_idx) > 0:
                        first_idx = non_zero_idx[0].item()
                        new_tensor[:, 1:] = text_tensor[:, first_idx:]
                    
                    reflection_input["text"]["tensor"] = new_tensor
            
            # Generate Reflection outputs (heatmaps)
            results = sampler.sampler.generate(
                reflection_input,
                schedules["reflection"],
                top_k=0.0,
                top_p=0.9,
                text_tokenizer=tokenizer,
                verbose=False
            )
            
            # Extract predicted heatmaps and ground truth masks
            if "heatmap" in results and "heatmap" in batch:
                # Convert heatmaps to binary masks (threshold at 0.5)
                pred_heatmaps = results["heatmap"]["tensor"].float() / 255.0  # Normalize if needed
                pred_masks = (pred_heatmaps > 0.5).float()
                
                # Get ground truth masks
                gt_masks = batch["heatmap"]["tensor"].float() / 255.0  # Normalize if needed
                gt_masks = (gt_masks > 0.5).float()
                
                # Calculate F1 scores
                metrics = calculate_mask_f1(pred_masks, gt_masks)
                
                f1_scores.append(metrics['f1'])
                precision_scores.append(metrics['precision'])
                recall_scores.append(metrics['recall'])
    
    # Calculate average metrics
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    
    print(f"Mask F1: {avg_f1:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")
    
    return {
        'f1': avg_f1,
        'precision': avg_precision,
        'recall': avg_recall
    }


def evaluate_correction(model, dataloader, tokenizer, device, num_samples):
    """
    Evaluate the model on Correction task (COCO-Stuff PSNR/SSIM).
    
    Args:
        model: The MCOT model
        dataloader: DataLoader for COCO-Stuff dataset
        tokenizer: Text tokenizer
        device: Device to run the evaluation on
        num_samples: Number of samples to evaluate
        
    Returns:
        PSNR and SSIM scores
    """
    print("Evaluating Correction performance (PSNR/SSIM)...")
    
    sampler = MCOTGenerationSampler(model)
    schedules = build_mcot_generation_schedules(MODALITY_INFO)
    
    # Lists to store the metrics
    psnr_scores = []
    ssim_scores = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= num_samples:
                break
            
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], dict):
                    for sub_key in batch[key]:
                        if torch.is_tensor(batch[key][sub_key]):
                            batch[key][sub_key] = batch[key][sub_key].to(device)
                elif torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
            
            # Prepare input with Correction token
            correction_input = batch.copy()
            if "text" in correction_input:
                # Add correction token to text input
                correction_token_id = tokenizer.token_to_id(CORRECTION_START_TOKEN)
                if correction_token_id is not None:
                    text_tensor = correction_input["text"]["tensor"]
                    new_tensor = torch.zeros_like(text_tensor)
                    new_tensor[:, 0] = correction_token_id
                    
                    # Find first non-zero token
                    non_zero_idx = torch.nonzero(text_tensor, as_tuple=True)[1]
                    if len(non_zero_idx) > 0:
                        first_idx = non_zero_idx[0].item()
                        new_tensor[:, 1:] = text_tensor[:, first_idx:]
                    
                    correction_input["text"]["tensor"] = new_tensor
            
            # Generate Correction outputs (corrected images)
            results = sampler.sampler.generate(
                correction_input,
                schedules["correction"],
                top_k=0.0,
                top_p=0.9,
                text_tokenizer=tokenizer,
                verbose=False
            )
            
            # Extract predicted corrected images and ground truth images
            if "rgb_corrected" in results and "rgb" in batch:
                # Get predicted images
                pred_images = results["rgb_corrected"]["tensor"].float() / 255.0  # Normalize if needed
                
                # Get ground truth images
                gt_images = batch["rgb"]["tensor"].float() / 255.0  # Normalize if needed
                
                # Calculate PSNR and SSIM
                metrics = calculate_image_metrics(pred_images, gt_images)
                
                psnr_scores.append(metrics['psnr'])
                ssim_scores.append(metrics['ssim'])
    
    # Calculate average metrics
    avg_psnr = sum(psnr_scores) / len(psnr_scores) if psnr_scores else 0.0
    avg_ssim = sum(ssim_scores) / len(ssim_scores) if ssim_scores else 0.0
    
    print(f"PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim
    }


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = create_model(args.checkpoint, device=args.device)
    model.eval()
    
    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    
    # Define evaluation dataloaders
    print("Preparing evaluation datasets...")
    
    if args.use_huggingface:
        # VQA dataset from Hugging Face
        vqa_dataset = build_mcot_huggingface_dataset(
            dataset_name=args.vqa_dataset_hf,
            split="validation",
            all_domains=["rgb", "text", "caption"],
            modality_info=MODALITY_INFO,
            modality_transforms=None,
            image_augmenter=None,
            text_tokenizer=tokenizer,
            input_tokens_range=(128, 128),
            target_tokens_range=(128, 128)
        )
        vqa_dataloader = DataLoader(vqa_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Planning dataset (COCO) from Hugging Face
        planning_dataset = build_mcot_huggingface_dataset(
            dataset_name=args.planning_dataset_hf,
            split="validation",
            all_domains=["rgb", "caption", "bbox"],
            modality_info=MODALITY_INFO,
            modality_transforms=None,
            image_augmenter=None,
            text_tokenizer=tokenizer,
            input_tokens_range=(128, 128),
            target_tokens_range=(128, 128)
        )
        planning_dataloader = DataLoader(planning_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Reflection dataset (RichHF) from Hugging Face
        reflection_dataset = build_mcot_huggingface_dataset(
            dataset_name=args.reflection_dataset_hf,
            split="validation",
            all_domains=["rgb", "text", "heatmap"],
            modality_info=MODALITY_INFO,
            modality_transforms=None,
            image_augmenter=None,
            text_tokenizer=tokenizer,
            input_tokens_range=(128, 128),
            target_tokens_range=(128, 128)
        )
        reflection_dataloader = DataLoader(reflection_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Correction dataset (COCO-Stuff) from Hugging Face
        correction_dataset = build_mcot_huggingface_dataset(
            dataset_name=args.correction_dataset_hf,
            split="validation",
            all_domains=["rgb", "text", "segmentation"],
            modality_info=MODALITY_INFO,
            modality_transforms=None,
            image_augmenter=None,
            text_tokenizer=tokenizer,
            input_tokens_range=(128, 128),
            target_tokens_range=(128, 128)
        )
        correction_dataloader = DataLoader(correction_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        # VQA dataset
        vqa_dataset = build_mcot_dataset(
            data_path=args.vqa_dataset,
            all_domains=["rgb", "text", "caption"],
            modality_info=MODALITY_INFO,
            modality_transforms=None,  # Will be filled in by the function
            image_augmenter=None,
            text_tokenizer=tokenizer,
            input_tokens_range=(128, 128),
            target_tokens_range=(128, 128)
        )
        vqa_dataloader = DataLoader(vqa_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Planning dataset (COCO)
        planning_dataset = build_mcot_dataset(
            data_path=args.planning_dataset,
            all_domains=["rgb", "caption", "bbox"],
            modality_info=MODALITY_INFO,
            modality_transforms=None,
            image_augmenter=None,
            text_tokenizer=tokenizer,
            input_tokens_range=(128, 128),
            target_tokens_range=(128, 128)
        )
        planning_dataloader = DataLoader(planning_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Reflection dataset (RichHF)
        reflection_dataset = build_mcot_dataset(
            data_path=args.reflection_dataset,
            all_domains=["rgb", "text", "heatmap"],
            modality_info=MODALITY_INFO,
            modality_transforms=None,
            image_augmenter=None,
            text_tokenizer=tokenizer,
            input_tokens_range=(128, 128),
            target_tokens_range=(128, 128)
        )
        reflection_dataloader = DataLoader(reflection_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # Correction dataset (COCO-Stuff)
        correction_dataset = build_mcot_dataset(
            data_path=args.correction_dataset,
            all_domains=["rgb", "text", "segmentation"],
            modality_info=MODALITY_INFO,
            modality_transforms=None,
            image_augmenter=None,
            text_tokenizer=tokenizer,
            input_tokens_range=(128, 128),
            target_tokens_range=(128, 128)
        )
        correction_dataloader = DataLoader(correction_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Run evaluations
    print("\nRunning evaluations...")
    
    # VQA evaluation
    vqa_accuracy = evaluate_vqa(model, vqa_dataloader, tokenizer, args.device, args.num_samples)
    
    # Planning evaluation (bbox IoU)
    planning_iou = evaluate_planning(model, planning_dataloader, tokenizer, args.device, args.num_samples)
    
    # Reflection evaluation (mask F1)
    reflection_metrics = evaluate_reflection(model, reflection_dataloader, tokenizer, args.device, args.num_samples)
    
    # Correction evaluation (PSNR/SSIM)
    correction_metrics = evaluate_correction(model, correction_dataloader, tokenizer, args.device, args.num_samples)
    
    # Aggregate results
    results = {
        "vqa": {
            "accuracy": vqa_accuracy
        },
        "planning": {
            "box_iou": planning_iou
        },
        "reflection": reflection_metrics,
        "correction": correction_metrics
    }
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEvaluation results saved to: {results_path}")
    print("\nSummary of results:")
    print(f"VQA Accuracy: {vqa_accuracy:.4f}")
    print(f"Planning Box IoU: {planning_iou:.4f}")
    print(f"Reflection Mask F1: {reflection_metrics['f1']:.4f}")
    print(f"Correction PSNR: {correction_metrics['psnr']:.4f}, SSIM: {correction_metrics['ssim']:.4f}")


if __name__ == "__main__":
    main() 