# Machine Chain of Thought (MCoT) Implementation

This repository contains an implementation of Machine Chain of Thought (MCoT), a multi-stage approach for visual reasoning and caption generation using the 4M architecture. The implementation currently supports two stages:

1. **Planning Stage**: Takes an image and optional caption prompt, generates a planning text and bounding box layout.
2. **Acting Stage**: Takes an image and the plan from the previous stage, generates a final detailed caption.

## Data Preparation

The implementation uses MS-COCO dataset for training and evaluation. WebDataset format is used for efficient loading and distributed training.

### Download COCO Dataset

```bash
mkdir -p data/coco
cd data/coco

# Download images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

### Create WebDataset Shards

```bash
python create_mcot_coco_shards.py --data-dir data/coco --samples-per-shard 500 --max-train-samples 5000 --max-val-samples 500
```

This script creates WebDataset shards for both Planning and Acting stages in `data/coco_mcot_shards/` directory and updates paths in `cfgs/mcot_data_config.yaml`.

## Training

### Configuration

The `cfgs/mcot_data_config.yaml` file contains configuration for MCoT training, including dataset paths, modality information, and training parameters.

### Running Training

```bash
# Submit a SLURM job
sbatch run_mcot_training.sh YOUR_WANDB_API_KEY
```

This will start training using FSDP (Fully Sharded Data Parallel) across multiple GPUs.

### Monitoring Training

Use the `monitor_training.py` script to monitor training progress:

```bash
python monitor_training.py --checkpoint-dir ./outputs --wandb-name mcot_training
```

This will periodically check for model checkpoints, GPU utilization, and WandB logs.

## Evaluation

### Evaluation Script

The `eval_mcot_model.py` script evaluates a trained MCoT model on validation data:

```bash
python eval_mcot_model.py --model-path /path/to/model_checkpoint --val-data-dir data/coco_mcot_shards
```

This script computes:
- **Planning metrics**: mAP, precision, recall for object detection
- **Acting metrics**: BLEU-4, ROUGE-L, METEOR for caption generation

### Visualization

The `visualization/visualize_mcot.py` script visualizes MCoT Planning and Acting results:

```bash
python visualization/visualize_mcot.py --model-path /path/to/model_checkpoint --image-path /path/to/image.jpg
```

This creates visualizations of:
- Planning stage: Image with predicted bounding boxes
- Acting stage: Image with final caption

## Testing Individual Images

The `test_mcot_pipeline.py` script can be used to test the full MCoT pipeline on a single image:

```bash
python test_mcot_pipeline.py --model-path /path/to/model_checkpoint --image-path /path/to/image.jpg --verbose
```

This runs the image through both Planning and Acting stages and saves the results in the `test_results` directory.

## Notes on Implementation

- **Special Tokens**: The tokenizer is extended with special tokens for MCoT tasks, including:
  - `[PLANNING_START]`, `[OBJECT]`, `[X]`, `[Y]`, `[W]`, `[H]`
  - `[ACTING_START]`

- **Modality Information**: The implementation uses a unified approach with different input and output domains:
  - Planning: Input = caption+image, Output = plan text+bbox layout
  - Acting: Input = plan+image, Output = final caption
  
- **Data Processing**: The `unified_datasets.py` file contains functions for loading and preprocessing data for MCoT stages.

## Citation

If you use this implementation in your research, please cite:

```
@inproceedings{mcot,
  title={Machine Chain of Thought: Multi-Stage Visual Reasoning with Neural Models},
  author={Your Name},
  booktitle={Your Conference},
  year={2023}
}
``` 