# Implementing Multimodal Chain of Thoughts Extension for 4M

A COM-304 project implementing the Multimodal Chain of Thought (MCoT) extension for the 4M vision model, inspired by the MINT paper. This project extends 4M's capabilities with a four-stage reasoning pipeline for enhanced complex image generation.

## Project Overview

Our implementation replaces single-step text-to-image generation with a structured four-stage reasoning process:

1. **Planning**: Dense captioning with spatial layouts and bounding boxes
2. **Acting**: Image generation based on detailed planning output
3. **Reflection**: Artifact detection with confidence scoring
4. **Correction**: Targeted inpainting for identified issues

### Key Contributions

- Complete MCoT Architecture: Non-invasive wrapper extending existing 4M models
- ActPlan Dataset: 28 000+ enhanced COCO entries with AI-generated dense captions
- Multi-Source Integration: Unified training pipeline with RichHF-18K, SeeTRUE-Feedback, and BrushData
- Production Infrastructure: FSDP-optimized distributed training with mixed precision
- VQA Baseline: Comprehensive experiments validating our approach

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: A100 with 80GB VRAM)
- 50GB+ free disk space for datasets and models

### Installation

1. **Clone and Setup Environment**

   ```bash
   git clone <repository-url>
   cd MCoT-4M-XL

   # Activate the fourm environment (required)
   conda activate fourm
   # OR if using venv:
   source fourm_env/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Package Versions

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
datasets>=2.12.0
Pillow>=9.5.0
numpy>=1.24.0
opencv-python>=4.7.0
safetensors>=0.3.1
wandb>=0.15.0
rclone>=1.60.0
fourm (can be installed using the 4M repository)
```

## Model Downloads

### Method 1: Manual Download (Google Drive)

1. **4M-XL-21 Fine-tuned Model**

   - Download from: https://drive.google.com/drive/folders/1xTY4k4zEJIOC6vo6o9qEA3Xh_dfulttf?usp=sharing
   - Extract to: `models/4m-xl-21-finetuned/`

2. **4M-XL-T2I Model**
   - Download from: https://drive.google.com/file/d/1CVHUH2kLJHLEYjeYJ_k2XcuYPmWXR3Sp/view?usp=sharing
   - Extract to: `models/4m-xl-t2i/`

### Method 2: Automated Download (Recommended)

```bash
# Install rclone if not already installed
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive access
rclone config

# Download models automatically
python scripts/download_models.py
```

### Model Structure

```
models/
├── 4m-xl-21-finetuned/
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
└── 4m-xl-t2i/
    ├── model.safetensors
    ├── config.json
    └── tokenizer.json
```

## Project Structure

```
MCoT-4M-XL/
├── fourm/                          # Core 4M framework
│   ├── data/
│   │   ├── mcot_dataset.py         # MCoT dataset integration
│   │   ├── unified_datasets.py     # Multi-source dataset handling
│   │   └── pretrain_utils.py       # Data preprocessing utilities
│   ├── models/
│   │   ├── fm.py                   # Base 4M transformer architecture
│   │   ├── mcot_fixed.py          # MCoT wrapper and step processor
│   │   └── fm_utils.py             # Model utilities and components
│   ├── utils/
│   │   ├── logger.py               # Training logging and monitoring
│   │   └── tokenizer/              # Text tokenization utilities
│   └── vq/                         # Vector quantization modules
├── cfgs/                           # Configuration files
│   ├── default/4m/data/mcot/
│   │   └── mcot.yaml              # MCoT data pipeline config
│   └── mcot_training_config.yaml  # Main training configuration
├── configs/
│   └── mcot_data_config.json      # Dataset integration settings
├── mcot_data/                      # MCoT-specific datasets
│   ├── actplan.json               # ActPlan dataset (385k+ entries)
│   ├── mcot_dataset_wget.py       # Dataset download automation
│   └── mcot_torch_dataset.py      # PyTorch dataset implementation
├── models/                         # Downloaded model weights
├── scripts/                        # Utility scripts
├── run_training_4m_mcot_fsdp.py  # Main MCoT training script
├── fourm_inference.py             # Inference and evaluation
├── fourm_inference_vqa.py         # VQA-specific inference script
└── submit_4m_fsdp_multinode.sh   # SLURM submission script
```

## Usage Instructions

### 1. VQA Baseline Experiments

Run VQA fine-tuning to establish baseline:

```bash
# Fine-tune 4M on VQAv2 dataset using our VQAv2 training script
python vqav2_training_script.py \
    --model_path models/4m-xl-21/ \
    --dataset_path data/vqa2017/ \
    --output_dir models/4m-xl-21-finetuned/ \
    --epochs 4

# Run inference on fine-tuned VQA models
python fourm_inference_vqa.py \
    --model_path models/4m-xl-21-finetuned/ \
    --dataset_path data/vqa2017/ \
    --output_dir results/vqa_baseline/ \
    --split validation
```

#### VQA Inference Script Usage

The `fourm_inference_vqa.py` script performs comprehensive VQA inference with parameter sweeping:

**Basic Usage:**

```bash
python fourm_inference_vqa.py
```

**Configuration:**
Edit the script parameters at the top of `main_inference()`:

- `model_name`: HuggingFace model identifier (default: 'EPFL-VILAB/4M-7-T2I_XL_CC12M')
- `finetuned_weights_path`: Path to fine-tuned checkpoint (default: 'checkpoint-2.safetensors')
- `test_cases_file`: JSON file with test cases (default: 'test_cases.json')
- `output_dir`: Directory for results (default: 'vqa_outputs')

**Test Cases Format:**
Create a `test_cases.json` file:

```json
{
  "test_cases": [
    {
      "image_path": "path/to/image1.jpg",
      "questions": [
        "What color is the car?",
        "How many people are in the image?"
      ]
    },
    {
      "image_path": "path/to/image2.jpg",
      "questions": ["What is the person doing?", "What objects are visible?"]
    }
  ]
}
```

**Parameter Sweeping:**
The script automatically tests different generation parameters:

- **Temperature**: [0.3, 0.5, 0.7] - Controls randomness
- **Top-p**: [0.9, 0.95, 0.98] - Nucleus sampling threshold
- **Top-k**: [40, 50, 60] - Number of top tokens to consider

**Output:**
Results are saved as JSON in `vqa_outputs/vqa_results.json` with:

- Generated answers for each parameter combination
- Raw token sequences
- Generation metadata and timestamps

### 2. MCoT Training

**Single GPU Training:**

```bash
python run_training_4m_mcot_fsdp.py \
    --config cfgs/mcot_training_config.yaml \
    --data_config cfgs/default/4m/data/mcot/mcot.yaml \
    --output_dir checkpoints/mcot_training/
```

**Multi-GPU Training (Recommended):**

```bash
torchrun --nproc_per_node=8 run_training_4m_mcot_fsdp.py \
    --config cfgs/mcot_training_config.yaml \
    --data_config cfgs/default/4m/data/mcot/mcot.yaml \
    --output_dir checkpoints/mcot_training/ \
    --use_fsdp
```

**SLURM Cluster Training:**

```bash
sbatch submit_4m_fsdp_multinode.sh
```

### 3. MCoT Inference

```bash
python fourm_inference.py \
    --model_path checkpoints/mcot_training/latest/ \
    --task mcot_generation \
    --prompt "A cat sitting on a red sofa in a living room" \
    --steps planning,acting,reflection,correction \
    --output_dir results/mcot_inference/
```

### 4. Dataset Creation

Generate ActPlan dataset from MS COCO:

```bash
cd mcot_data/
python mcot_dataset_wget.py --create_actplan --num_workers 8
```

## Evaluation and Results

### VQA Results

Our VQA baseline experiments (4 epochs, VQAv2 2017):

- **Training Loss**: 0.88 (epoch 3)
- **Validation Loss**: 1.3 (epoch 3)
- **Key Finding**: 4M excels at generative tasks but struggles with discriminative QA

### MCoT Training Status

**Current Status**: MCoT training is in progress with the complete implementation framework ready.

**Results Availability**: Training results and performance metrics are available at:

- **Project Website**: [Merciercharles.github.io](https://Merciercharles.github.io)
- **Model Hub**: [Coming Soon]
- **Evaluation Benchmarks**: Planning (IoU), Reflection (F1), Correction (PSNR/SSIM)

## Configuration Options

### Training Configuration (`cfgs/mcot_training_config.yaml`)

Key parameters you can modify:

```yaml
model: "fm_base_12e_12d_swiglu_nobias"
batch_size: 32
epochs: 2
blr: 5e-5

mcot_steps: "planning,acting,reflection,correction"
mcot_planning_weight: 1.0
mcot_acting_weight: 1.2
mcot_reflection_weight: 1.5
mcot_correction_weight: 1.3
enable_mint_features: true

use_act_checkpoint: true
dtype: "bfloat16"
clip_grad: 1.0
```

### Data Configuration (`cfgs/default/4m/data/mcot/mcot.yaml`)

Configure dataset sources and preprocessing:

```yaml
datasets:
  - actplan_dataset
  - richhf_18k
  - seetrue_feedback
  - brush_data

preprocessing:
  image_size: 224
  text_max_length: 512
  enable_augmentation: true
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   batch_size: 16
   use_act_checkpoint: true
   ```

2. **Missing Model Weights**

   ```bash
   ls -la models/
   python scripts/download_models.py
   ```

3. **Dataset Loading Errors**
   ```bash
   rm -rf data/cache/
   python mcot_data/mcot_dataset_wget.py --force_download
   ```

## References

- [4M Paper](https://arxiv.org/abs/2312.06647): Massively Multimodal Masked Modeling
- [MINT Paper](https://arxiv.org/abs/2503.01298): Multimodal Interleaved Chain-of-Thought
- [VQAv2 Dataset](https://visualqa.org/)
- [MS COCO Dataset](https://cocodataset.org/)

## Team

- **Rayane Charif**
- **Andrew Siminszky**
- **Charles Mercier**

**Note**: This is a research project developed for COM-304. The implementation builds upon the original 4M framework and extends it with MCoT capabilities for enhanced multimodal reasoning.
