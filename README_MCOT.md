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

3. Download the pre-trained 4M-21_XL model (follow instructions in the main README.md).

## Project Structure

- `extend_vocabulary.py`: Script to extend 4M tokenizer with MCOT tokens
- `fourm/data/mcot_dataset.py`: MCOT dataset implementation
- `fourm/models/mcot_generate.py`: MCOT generation implementation
- `run_mcot_evaluation.py`: Evaluation script for MCOT model
- `cfgs/`: Configuration files for MCOT training and post-training

## Usage

### 1. Extend Vocabulary

First, extend the 4M model's vocabulary to include the MCOT stage markers:

```bash
python extend_vocabulary.py \
  --checkpoint-path /path/to/4m-21-xl.pt \
  --output-checkpoint-path /path/to/4m-21-xl-mcot.pt
```

### 2. VQA Fine-tuning (Stage 1)

Fine-tune the model for Visual Question Answering using the VQAv2 dataset:

```bash
python run_training_4m.py \
  --config cfgs/mcot_vqa_finetune.yaml \
  --finetune /path/to/4m-21-xl-mcot.pt \
  --output_dir ./output/mcot_vqa_finetune
```

### 3. MCOT Post-training (Stage 2)

Train the model for all four MCOT stages using a mixture of datasets:

```bash
python run_training_4m.py \
  --config cfgs/mcot_post_training.yaml \
  --finetune ./output/mcot_vqa_finetune/checkpoint-best.pth \
  --output_dir ./output/mcot_post_training
```

### 4. Evaluation

Evaluate the model's performance on all four stages:

```bash
python run_mcot_evaluation.py \
  --checkpoint ./output/mcot_post_training/checkpoint-best.pth \
  --vqa-dataset /path/to/vqav2/val \
  --planning-dataset /path/to/mscoco/val \
  --reflection-dataset /path/to/richhf18k/val \
  --correction-dataset /path/to/cocostuff/val \
  --output-dir ./evaluation_results
```

## Datasets

- **VQA Fine-tuning**: [VQAv2 dataset](https://visualqa.org/download.html)
- **Planning**: [MS-COCO captions and bounding boxes](https://cocodataset.org/)
- **Reflection**: [RichHF-18K artifact annotations](https://github.com/richzhang/PerceptualSimilarity)
- **Correction**: [COCO-Stuff segmentation masks](https://github.com/nightrome/cocostuff)

## Configuration

The configuration files in the `cfgs/` directory control the model architecture, training parameters, and dataset paths:

- `cfgs/mcot_vqa_finetune.yaml`: Configuration for VQA fine-tuning
- `cfgs/data_vqa.yaml`: Dataset configuration for VQA
- `cfgs/mcot_post_training.yaml`: Configuration for MCOT post-training
- `cfgs/data_mcot.yaml`: Multi-task dataset configuration for MCOT

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
