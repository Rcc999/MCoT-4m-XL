'''
Script to create a subset of VQAv2 training data.
This script loads the training questions and annotations, selects a random fraction,
and saves new JSON files for the subset.
'''
import json
import os
import random
from pathlib import Path

# Configuration
raw_data_root = Path("/work/com-304/vqav2_data_raw")
output_subset_dir = Path("/work/com-304/vqav2_subset_25pct") # Directory to save subset files
subset_fraction = 0.25  # e.g., 0.25 for 25%
random_seed = 42 # For reproducibility

# Original VQAv2 file names (as extracted by vqa_dataset_wget.py)
original_questions_file = raw_data_root / "questions_train" / "v2_OpenEnded_mscoco_train2014_questions.json"
original_annotations_file = raw_data_root / "annotations_train" / "v2_mscoco_train2014_annotations.json"

# Output subset file names
subset_questions_file = output_subset_dir / f"v2_OpenEnded_mscoco_train2014_questions_subset{int(subset_fraction*100)}pct.json"
subset_annotations_file = output_subset_dir / f"v2_mscoco_train2014_annotations_subset{int(subset_fraction*100)}pct.json"

def create_subset():
    print(f"Raw data root: {raw_data_root}")
    print(f"Output subset directory: {output_subset_dir}")
    print(f"Subset fraction: {subset_fraction}")

    if not original_questions_file.exists():
        print(f"ERROR: Original questions file not found: {original_questions_file}")
        return
    if not original_annotations_file.exists():
        print(f"ERROR: Original annotations file not found: {original_annotations_file}")
        return

    os.makedirs(output_subset_dir, exist_ok=True)
    random.seed(random_seed)

    print(f"Loading original questions from: {original_questions_file}")
    with open(original_questions_file, 'r') as f:
        questions_data = json.load(f)
    
    print(f"Loading original annotations from: {original_annotations_file}")
    with open(original_annotations_file, 'r') as f:
        annotations_data = json.load(f)

    # VQAv2 questions are a list under the 'questions' key
    # VQAv2 annotations are a list under the 'annotations' key
    all_question_entries = questions_data.get("questions", [])
    all_annotation_entries = annotations_data.get("annotations", [])

    if not all_question_entries:
        print("ERROR: No questions found in the questions JSON file.")
        return
    if not all_annotation_entries:
        print("ERROR: No annotations found in the annotations JSON file.")
        return

    # Create a mapping from question_id to annotation for easier lookup
    annotations_map = {ann['question_id']: ann for ann in all_annotation_entries}

    num_original_questions = len(all_question_entries)
    num_subset_questions = int(num_original_questions * subset_fraction)

    print(f"Original number of questions: {num_original_questions}")
    print(f"Target number of subset questions: {num_subset_questions}")

    # Shuffle and select a subset of questions
    # We must ensure that for every selected question, its annotation also exists.
    # A simple way is to select from question_ids that are present in both.
    
    valid_question_ids = list(annotations_map.keys())
    # Filter all_question_entries to only include those whose IDs are in annotations_map
    # This is important because not all questions in the question file might have annotations,
    # though for VQAv2 train/val this should be a 1:1 mapping.
    # OPTIMIZATION: Use direct dictionary lookup in annotations_map for efficiency
    print("Filtering questions to find those with matching annotations (this may take a moment)...")
    questions_with_annotations = [q for q in all_question_entries if q['question_id'] in annotations_map]
    
    num_questions_with_annotations = len(questions_with_annotations)
    if num_questions_with_annotations < num_original_questions:
        print(f"Warning: {num_original_questions - num_questions_with_annotations} questions did not have matching annotations and were excluded before sampling.")

    if num_subset_questions > num_questions_with_annotations:
        print(f"Warning: Requested subset size ({num_subset_questions}) is larger than available questions with annotations ({num_questions_with_annotations}). Using all available.")
        num_subset_questions = num_questions_with_annotations
        subset_questions_selected = questions_with_annotations
    else:
        random.shuffle(questions_with_annotations) # Shuffle the list of question dicts
        subset_questions_selected = questions_with_annotations[:num_subset_questions]

    subset_question_ids = {q['question_id'] for q in subset_questions_selected}
    # OPTIMIZATION: Ensure subset_annotations_selected is also efficient
    # The current way is fine as subset_question_ids is a set, making lookups efficient.
    subset_annotations_selected = [annotations_map[qid] for qid in subset_question_ids if qid in annotations_map]

    print(f"Actual number of questions selected for subset: {len(subset_questions_selected)}")
    print(f"Actual number of annotations selected for subset: {len(subset_annotations_selected)}")

    # Create new JSON structures
    # The structure should mirror the original VQAv2 JSONs
    subset_questions_data = {
        'info': questions_data.get('info', {}),
        'task_type': questions_data.get('task_type', ''),
        'data_type': questions_data.get('data_type', ''),
        'data_subtype': questions_data.get('data_subtype', ''), # Preserve original subtype if present
        'license': questions_data.get('license', {}),
        'questions': subset_questions_selected
    }
    # Update data_subtype for clarity
    subset_questions_data['data_subtype'] = f"{questions_data.get('data_subtype', 'train2014')}_subset{int(subset_fraction*100)}pct"


    subset_annotations_data = {
        'info': annotations_data.get('info', {}),
        'data_type': annotations_data.get('data_type', ''),
        'data_subtype': annotations_data.get('data_subtype', ''), # Preserve original subtype
        'license': annotations_data.get('license', {}),
        'annotations': subset_annotations_selected
    }
    # Update data_subtype for clarity
    subset_annotations_data['data_subtype'] = f"{annotations_data.get('data_subtype', 'train2014')}_subset{int(subset_fraction*100)}pct"

    print(f"Saving subset questions to: {subset_questions_file}")
    with open(subset_questions_file, 'w') as f:
        json.dump(subset_questions_data, f)

    print(f"Saving subset annotations to: {subset_annotations_file}")
    with open(subset_annotations_file, 'w') as f:
        json.dump(subset_annotations_data, f)

    print("Subset creation complete.")
    print(f"Image directory remains the same: {raw_data_root / 'images_train' / 'train2014'}")
    print("You will need to update your data configuration (e.g., vqav2.yaml) to point to these new subset JSON files.")

if __name__ == '__main__':
    create_subset() 