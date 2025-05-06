# Multimodal Chain of Thought (MCOT) for 4M Models

This repository implements Multimodal Chain of Thought (MCOT) on top of the 4M model codebase. The implementation adapts the 4M-21_XL model to integrate the MCOT paradigm, inspired by the MINT paper. This approach enables explicit multi-stage reasoning through a sequence of Planning, Acting, Reflection, and Correction steps.

## Overview

MCOT extends the 4M model with four explicit stages of visual reasoning:

1. **Planning**: Generates captions and bounding boxes to plan what to create
2. **Acting**: Generates images based on the Planning outputs
3. **Reflection**: Identifies artifacts or issues in the generated images
4. **Correction**: Fixes the identified artifacts to create improved images

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/ml-4m.git
cd ml-4m
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the pre-trained 4M-21_XL model and place it in the `ckpt` directory as `mcotmodel.safetensors`.

4. Set up datasets and update configuration paths:

```bash
# Download all required datasets (VQAv2, MS-COCO, RichHF-18K, COCO-Stuff)
python download_datasets.py

# Update configuration files to use the new paths
python update_paths.py
```

## Project Structure

- `extend_vocabulary.py`: Script to extend 4M tokenizer with MCOT tokens
- `fourm/data/mcot_dataset.py`: MCOT dataset implementation
- `fourm/models/mcot_generate.py`: MCOT generation implementation
- `run_mcot_evaluation.py`: Evaluation script for MCOT model
- `cfgs/`: Configuration files for MCOT training and post-training
- `download_datasets.py`: Script to download and set up all required datasets
- `update_paths.py`: Script to update configuration files with correct paths

## Usage

### 1. Extend Vocabulary

First, extend the 4M model's vocabulary to include the MCOT stage markers:

```bash
python extend_vocabulary.py \
  --checkpoint-path ckpt/mcotmodel.safetensors \
  --output-checkpoint-path ckpt/mcotmodel.safetensors
```

### 2. VQA Fine-tuning (Stage 1)

Fine-tune the model for Visual Question Answering using the VQAv2 dataset:

```bash
python run_training_4m.py \
  --config cfgs/mcot_vqa_finetune.yaml \
  --finetune ckpt/mcotmodel.safetensors \
  --output_dir ./output/mcot_vqa_finetune
```

### 3. MCOT Post-training (Stage 2)

Train the model for all four MCOT stages using a mixture of datasets:

```bash
python run_training_4m.py \
  --config cfgs/mcot_post_training.yaml \
  --finetune ./output/mcot_vqa_finetune/checkpoint-best.safetensors \
  --output_dir ./output/mcot_post_training
```

### 4. Evaluation

Evaluate the model's performance on all four stages:

```bash
python run_mcot_evaluation.py \
  --checkpoint ./output/mcot_post_training/checkpoint-best.safetensors \
  --vqa-dataset datasets/vqav2/val \
  --planning-dataset datasets/mscoco/val \
  --reflection-dataset datasets/richhf18k/val \
  --correction-dataset datasets/cocostuff/val \
  --output-dir ./evaluation_results
```

## Datasets

The `download_datasets.py` script will download and set up the following datasets:

- **VQAv2 dataset**: For VQA fine-tuning (stored in `datasets/vqav2`)
- **MS-COCO captions and bounding boxes**: For Planning (stored in `datasets/mscoco`)
- **RichHF-18K artifact annotations**: For Reflection (stored in `datasets/richhf18k`)
- **COCO-Stuff segmentation masks**: For Correction (stored in `datasets/cocostuff`)

## Configuration

The configuration files in the `cfgs/` directory control the model architecture, training parameters, and dataset paths:

- `cfgs/mcot_vqa_finetune.yaml`: Configuration for VQA fine-tuning
- `cfgs/data_vqa.yaml`: Dataset configuration for VQA
- `cfgs/mcot_post_training.yaml`: Configuration for MCOT post-training
- `cfgs/data_mcot.yaml`: Multi-task dataset configuration for MCOT

The `update_paths.py` script ensures all configuration files use the correct paths for the model and datasets.

## Model Architecture

The MCOT model uses the 4M-21_XL model as its base and introduces four new special tokens for the different stages. The model follows a multi-task training approach with mixed batches containing samples from all four tasks.

## Evaluation Metrics

The model is evaluated on the following metrics:

1. **VQA**: VQAv2 accuracy
2. **Planning**: COCO layout box IoU
3. **Reflection**: RichHF-18K mask F1 score
4. **Correction**: COCO-Stuff PSNR/SSIM for inpainted regions

## References

- 4M: Massively Multimodal Masked Modeling
- MINT: Evaluating LLMs in Multimodal In-context Learning
- VQAv2: Visual Question Answering v2
- MS-COCO: Microsoft Common Objects in Context
- RichHF-18K: Perceptual Similarity Annotations
- COCO-Stuff: COCO Semantic Segmentation
