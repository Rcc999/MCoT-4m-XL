# Multimodal Chain of Thought (MCoT) Implementation for 4M-XL

This repository contains the implementation of Multimodal Chain of Thought (MCoT) using the 4M-XL model. The implementation focuses on the first two stages of MCoT:

1. **Planning**: Generate a plan text and bounding box layout from an image
2. **Acting**: Generate a detailed final caption based on the image and plan

## Setup

### Requirements

The implementation relies on the 4M framework and its dependencies. Make sure you have the following installed:

- PyTorch >= 1.12
- torchvision
- tokenizers
- matplotlib
- webdataset
- pycocotools
- PIL
- huggingface_hub (for loading models from HuggingFace)

### Directory Structure

```
MCoT-4m-XL/
├── fourm/                    # 4M framework code
├── cfgs/                     # Configuration files
│   ├── mcot_config.yaml      # MCoT training configuration
├── tokenizers/               # Extended tokenizers
├── data/                     # Dataset storage
│   ├── coco/                 # MS-COCO dataset
│   ├── coco_mcot_shards/     # Processed WebDataset shards for MCoT
├── outputs/                  # Training outputs and checkpoints
├── results/                  # Inference results
├── extend_tokenizer_for_mcot.py  # Script to extend tokenizer with MCoT tokens
├── prepare_mcot_coco_data.sh     # Script to download and prepare COCO dataset
├── train_mcot_model.sh           # MCoT training script
├── run_mcot_inference.py         # Inference script for MCoT
└── README_MCOT_IMPLEMENTATION.md # This file
```

## Implementation Steps

### 1. Extend Tokenizer with MCoT Tokens

First, extend the 4M tokenizer with special tokens needed for MCoT:

```bash
mkdir -p tokenizers
python extend_tokenizer_for_mcot.py \
    --input-tokenizer fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json \
    --output-path tokenizers/mcot_tokenizer.json
```

This script adds the following special tokens:
- `[PLANNING_START]`, `[ACTING_START]`, `[REFLECTION_START]`, `[CORRECTION_START]` for stage markers
- `[OBJECT]`, `[X]`, `[Y]`, `[W]`, `[H]` for bounding box encoding

### 2. Prepare MS-COCO Dataset

Download and prepare the MS-COCO dataset for MCoT training:

```bash
bash prepare_mcot_coco_data.sh
```

This script will:
1. Download COCO train and validation images and annotations
2. Process them into WebDataset shards suitable for MCoT training
3. Create separate shards for Planning and Acting stages

### 3. Train the MCoT Model

Train the MCoT model using the 4M-XL-21 base model from HuggingFace:

```bash
# The script uses the 4M-XL-21 model from HuggingFace: 'EPFL-VILAB/4M-21_XL'
bash train_mcot_model.sh
```

This script will:
1. Extend the tokenizer with MCoT tokens
2. Prepare the dataset if needed
3. Run the training process using the 4M training framework

Training uses LoRA to efficiently fine-tune the large 4M-XL model.

### 4. Run Inference

After training, you can run inference on new images:

```bash
python run_mcot_inference.py \
    --model-path outputs/mcot_training/checkpoint-best.pt \
    --image-path path/to/your/image.jpg \
    --output-dir results \
    --verbose
```

This will:
1. Run the Planning stage to generate a plan and bounding boxes
2. Run the Acting stage to generate a detailed caption
3. Visualize the results and save them to the specified output directory

## Configuration

The MCoT training is configured in `cfgs/mcot_config.yaml`. You can adjust parameters such as:

- Model path: Currently uses `EPFL-VILAB/4M-21_XL` from HuggingFace
- Batch size, learning rate, and other training parameters
- LoRA parameters for fine-tuning
- Dataset paths and sizes

## Customization

To customize the implementation:

- **Add more MCoT stages**: Extend the implementation to include Reflection and Correction stages as described in the proposal.
- **Use different datasets**: Modify `prepare_mcot_coco_data.sh` and `cfgs/mcot_config.yaml` to use different datasets.
- **Adjust LoRA parameters**: Modify the LoRA fine-tuning settings in `cfgs/mcot_config.yaml` for different trade-offs between performance and efficiency.
- **Use a different 4M model**: Change `model_path` in config to use a different model variant (e.g., `EPFL-VILAB/4M-21_L` for the Large model)

## References

1. 4M: Massively Multimodal Masked Modeling - [github.com/facebookresearch/four-m](https://github.com/facebookresearch/four-m)
2. MINT: Multimodal-Instructed-Thinker - [arXiv:2403.11056](https://arxiv.org/abs/2403.11056) 