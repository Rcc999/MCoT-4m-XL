import json
import glob
import os
from collections import defaultdict

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    # Read dense captions
    dense_captions = {}
    for file_path in glob.glob('dense_output1/dense_captions_part_*.json'):
        part_captions = read_json_file(file_path)
        dense_captions.update(part_captions)

    # Read original captions
    original_captions = read_json_file('/work/com-304/coco_17/annotations/captions_train2017.json')
    captions_by_image = defaultdict(list)
    for caption in original_captions['annotations']:
        image_id = str(caption['image_id'])
        captions_by_image[image_id].append(caption['caption'])

    # Read bounding boxes
    instances = read_json_file('/work/com-304/coco_17/annotations/instances_train2017.json')
    boxes_by_image = defaultdict(list)
    for ann in instances['annotations']:
        image_id = str(ann['image_id'])
        category_id = ann['category_id']
        category_name = next(cat['name'] for cat in instances['categories'] if cat['id'] == category_id)
        bbox = ann['bbox']
        boxes_by_image[image_id].append({
            'x1': int(bbox[0]),
            'y1': int(bbox[1]),
            'x2': int(bbox[0] + bbox[2]),
            'y2': int(bbox[1] + bbox[3]),
            'class': category_name
        })

    # Combine all data
    combined_data = []
    for image_id in dense_captions.keys():
        if image_id in captions_by_image and image_id in boxes_by_image:
            combined_data.append({
                'image_id': image_id,
                'original_captions': captions_by_image[image_id],
                'dense_captions': dense_captions[image_id],
                'bounding_boxes': boxes_by_image[image_id]
            })

    # Write combined data to file in the user's home directory
    output_path = os.path.expanduser('~/combined_output.json')
    with open(output_path, 'w') as f:
        json.dump(combined_data, f, indent=2)
    print(f"Output written to: {output_path}")

if __name__ == '__main__':
    main() 