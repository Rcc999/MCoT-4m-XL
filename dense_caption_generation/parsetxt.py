#!/usr/bin/env python3
"""
parsetxt.py
=============================
Minimal helper ― *no network, no SSL issues*
-------------------------------------------
Convert a **local** COCO annotations ZIP (`annotations_trainval2017.zip`) into a
TXT file where each line contains:

```
IMAGE_ID | ['CAPTION1', 'CAPTION2', 'CAPTION3', 'CAPTION4', 'CAPTION5'] | v0=x0 v1=y0 v2=x1 v3=y1 label …
```

Only two steps:

```bash
python parsetxt.py /work/com-304/coco_17/annotations_trainval2017.zip \
       outputs_txt/coco_lines.txt --num 100 --split val2017
```

* Requires **pycocotools** and **tqdm** (no `requests`, no internet).
* Optional `--download-images DIR` saves the corresponding JPEGs in DIR (offline
  use with other scripts).
"""
from __future__ import annotations

import argparse, json, random, zipfile, shutil, tempfile
from pathlib import Path
from typing import List
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser("COCO ZIP → caption‑bbox lines")
parser.add_argument("zip_path", help="Path to annotations_trainval2017.zip")
parser.add_argument("out_txt", help="Output TXT file (lines as specified)")
parser.add_argument("--num", type=int, default=10, help="Number of images to sample")
parser.add_argument("--split", default="val2017", choices=["train2017", "val2017"], help="Dataset split")
parser.add_argument("--download-images", metavar="DIR", help="Optionally copy JPEGs into DIR (must exist)")
args = parser.parse_args()

zip_path = Path(args.zip_path).expanduser()
if not zip_path.is_file():
    raise FileNotFoundError(f"ZIP not found: {zip_path}")

# Create outputs_txt directory if it doesn't exist
output_dir = Path("outputs_txt")
output_dir.mkdir(exist_ok=True)

# Generate unique output filename
base_name = Path(args.out_txt).stem
extension = Path(args.out_txt).suffix
counter = 1
while (output_dir / f"{base_name}_{counter}{extension}").exists():
    counter += 1
output_path = output_dir / f"{base_name}_{counter}{extension}"

# -----------------------------------------------------------------------------
# Extract captions + instances JSON into temp dir (no full unzip to disk)
# -----------------------------------------------------------------------------

tmp_dir = Path(tempfile.mkdtemp(prefix="coco_zip_"))
with zipfile.ZipFile(zip_path, "r") as z:
    for name in [f"annotations/captions_{args.split}.json", f"annotations/instances_{args.split}.json"]:
        z.extract(name, path=tmp_dir)

ann_dir = tmp_dir / "annotations"

# -----------------------------------------------------------------------------
# Load COCO data
# -----------------------------------------------------------------------------

from pycocotools.coco import COCO  # heavy import only here
print("Loading COCO JSON…")
cap = COCO(str(ann_dir / f"captions_{args.split}.json"))
ins = COCO(str(ann_dir / f"instances_{args.split}.json"))

img_ids = cap.getImgIds()
random.shuffle(img_ids)
img_ids = img_ids[: args.num]

# -----------------------------------------------------------------------------
# Build lines
# -----------------------------------------------------------------------------

out_lines: List[str] = []
for img_id in tqdm(img_ids, desc="Building TXT"):
    caps = cap.imgToAnns[img_id][:5]
    dets = ins.imgToAnns.get(img_id, [])
    box_txt = " ".join(
        f"v0={d['bbox'][0]:.0f} v1={d['bbox'][1]:.0f} "
        f"v2={d['bbox'][0]+d['bbox'][2]:.0f} v3={d['bbox'][1]+d['bbox'][3]:.0f} "
        f"{ins.cats[d['category_id']]['name']}" for d in dets
    )
    # Group all captions for this image into a list
    captions = [c['caption'] for c in caps]
    out_lines.append(f"{img_id} | {captions} | {box_txt}\n")

output_path.write_text("".join(out_lines), encoding="utf-8")
print(f"✅ Wrote {len(out_lines)} lines → {output_path}")

# -----------------------------------------------------------------------------
# Copy JPEGs (optional)
# -----------------------------------------------------------------------------

if args.download_images:
    dest_dir = Path(args.download_images)
    if not dest_dir.is_dir():
        raise NotADirectoryError(dest_dir)
    print("Copying JPEGs…")
    with zipfile.ZipFile(zip_path, "r") as z:
        for img_id in tqdm(img_ids, desc="JPEGs"):
            jpg_name = f"{args.split}/{int(img_id):012}.jpg"
            try:
                z.extract(jpg_name, path=dest_dir)
                # Move extracted file up one level (remove split folder)
                (dest_dir / jpg_name).rename(dest_dir / Path(jpg_name).name)
                shutil.rmtree(dest_dir / args.split, ignore_errors=True)
            except KeyError:
                print(f"[WARN] JPEG {jpg_name} missing in ZIP")
print("Done.")
