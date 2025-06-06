#!/usr/bin/env python3
"""Dense-caption generator – Vertex AI (Gemini 1.5 Flash)

Input  : text file where each line is
         IMAGE_ID | ['caption1', …, 'caption5'] | bbboxes
Output : JSON list → {"image_id": "...", "captions": {"densecaption0": "...", …}}

The script sends all captions to Vertex AI and asks the model to expand each short
caption into a dense one that starts with "The image …".
"""

import argparse
import json
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def get_next_output_path(base_name: str) -> str:
    """Get the next available output path in dense_output directory."""
    # Create dense_output directory if it doesn't exist
    os.makedirs("dense_output", exist_ok=True)
    
    # Remove .json extension if present
    base_name = os.path.splitext(base_name)[0]
    
    # Start with base name
    counter = 0
    while True:
        if counter == 0:
            output_path = os.path.join("dense_output", f"{base_name}.json")
        else:
            output_path = os.path.join("dense_output", f"{base_name}_{counter}.json")
        
        if not os.path.exists(output_path):
            return output_path
        counter += 1


def parse_file(file_path: str) -> list[tuple[str, list[str]]]:
    results = []
    with open(file_path, "r") as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.strip().split("|")
            image_id = parts[0].strip()
            captions = eval(parts[1].strip())
            results.append((image_id, captions))
    return results


def make_prompt(all_captions: list[tuple[str, list[str]]]) -> str:
    # Format all captions with their image IDs
    captions_text = ""
    for image_id, captions in all_captions:
        captions_text += f"\nImage ID: {image_id}\n"
        captions_text += "\n".join(f"{i+1}. {cap}" for i, cap in enumerate(captions))
        captions_text += "\n"
    
    return (
        "okay now according to this file that is structured as follow:\n\n"
        "Each line follows this format:\n\n"
        "IMAGE_ID | ['CAPTION1', 'CAPTION2', 'CAPTION3', 'CAPTION4', 'CAPTION5'] | BB_BOXES\n\n"
        "For example:\n\n"
        "241364 | ['A mirror that is sitting behind a sink.', 'A bathroom with a sink and a toilet in it.', 'A white bathroom with chrome fixtures and blue tile.', 'A hotel bathroom with a large sink sticking out of the counter.', 'A decently sized bathroom with a nice sink'] | v0=48 v1=217 v2=283 v3=405 sink v0=271 v1=548 v2=441 v3=640 toilet\n\n"
        "This preserves: Image ID (3359636318) - so you can map back to the original data\n\n"
        "Captions the 5 captions this is for that image\n\n"
        "I want you for each group of caption so index 0 to 4 from the same image_id to generate a dense caption that start with \"The image ...\"\n\n"
        "here is examples that i want:\n\n"
        "caption: a photo of a person and an apple\n\n"
        "dense caption: \"The image shows a person sitting at a table with a shiny red apple in front of him. The background is softly blurred, suggesting a cozy indoor setting, possibly a cafe or restaurant, with warm lighting and a welcoming ambiance. The person is dressed casually, wearing a comfortable blue sweater and glasses, and his short hair frames his face. The scene captures a moment of casual enjoyment, as the apple adds a pop of color to the minimalist arrangement on the table.\"\n\n"
        "caption: a photo of three birds\n\n"
        "dense caption: \"The image shows three small birds perched on a branch. The birds have distinctive plumage patterns, including orange and gray feathers, black beaks, and bright blue wings. They are sitting in a row, facing forward, and appear to be observing their surroundings. The background is blurred, focusing attention on the birds.\"\n\n"
        "caption: a photo of a purple dog above a black dining table\n\n"
        "dense caption: \"The image depicts a modern, minimalist setting with a sleek, black rectangular table as the central focus. The overall atmosphere is clean, highlighted by a cute, fluffy purple dog sitting playfully on the table. The background features a wooden wall with vertical paneling. In front of the table, there are two chairs with a light-colored fabric, suggesting a contemporary design aesthetic.\"\n\n"
        "give many details for a pretty long dense caption and be certain (no possibly typically ...)\n\n"
        "your output should be like this:\n\n"
        "Image ID: [image_id]\n"
        "The image [detailed description 1]\n"
        "The image [detailed description 2]\n"
        "The image [detailed description 3]\n"
        "The image [detailed description 4]\n"
        "The image [detailed description 5]\n\n"
        f"Now process these images:\n{captions_text}"
    )


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    print("Starting dense caption generation...")
    
    ap = argparse.ArgumentParser()
    ap.add_argument("input_txt", help="Path to the input text file")
    ap.add_argument("--output", default="dense_captions", help="Base name for output file (without extension)")
    ap.add_argument("--project",  default="mproject-453620" , help="GCP project ID (falls back to ADC)")
    ap.add_argument("--region",   default="us-central1", help="Vertex AI region")
    ap.add_argument("--temperature", type=float, default=0.5)
    args = ap.parse_args()

    # Get the next available output path
    output_path = get_next_output_path(args.output)
    print(f"Output will be written to: {output_path}")

    print(f"Initializing Vertex AI with project: {args.project}, region: {args.region}")
    vertexai.init(project=args.project, location=args.region)
    model = GenerativeModel("gemini-2.5-flash-preview-05-20")  # fastest multimodal model
    print("Vertex AI initialized successfully")

    print(f"Reading input file: {args.input_txt}")
    all_captions = parse_file(args.input_txt)
    print(f"Found {len(all_captions)} images to process")
    
    print("Generating dense captions for all images...")
    resp = model.generate_content(
        make_prompt(all_captions),
        generation_config={"temperature": args.temperature}
    )
    
    print("Raw response from model:")
    print(resp.text)
    print("\nProcessing response...")
    
    # Process the response to extract dense captions for each image
    results = {}
    current_image_id = None
    current_captions = []
    
    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Image ID:"):
            # Save previous image's captions if they exist
            if current_image_id is not None and current_captions:
                results[current_image_id] = current_captions
            # Start new image
            current_image_id = line.split("Image ID:")[1].strip()
            current_captions = []
        elif line.startswith("The image"):
            current_captions.append(line)
    
    # Don't forget to add the last image
    if current_image_id is not None and current_captions:
        results[current_image_id] = current_captions

    print(f"\nWriting results to: {output_path}")
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print("Results written successfully")
    print(f"Total images processed: {len(results)}")


if __name__ == "__main__":
    main()
