#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import argparse
import re

# Command-line arguments
# 
parser = argparse.ArgumentParser(description='Square Image Cropping using CLIPSeg')
parser.add_argument('-i', '--input_path', required=True, help='Path to input directory containing nested image folders')
parser.add_argument('-o', '--output_path', required=True, help='Path to output directory (will contain crop/ and mask/ folders)')
args = parser.parse_args()

input_path = Path(args.input_path).resolve()
output_path = Path(args.output_path).resolve()
crop_root = output_path / "crop"
mask_root = output_path / "mask"


# Load model and processor
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Prompt cleaner
def clean_prompt(name):
    cleaned = re.sub(r"[\(\)_\-\d]+", " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return f"a {cleaned}"

# Run segmentation and crop

def crop_square_image(processor, model, image_dir, rel_path):
    crop_dir = crop_root / rel_path
    mask_dir = mask_root / rel_path
    crop_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(image_dir.glob("*")):
        if not img_path.is_file() or not img_path.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        name = img_path.stem
        prompt = clean_prompt(name)
        print(f"Processing: {img_path.name} | Prompt: '{prompt}'")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image: {e}")
            continue

        # Process input
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            mask_prob = torch.sigmoid(logits)

        # Generate binary mask
        mask_np = np.squeeze(mask_prob.cpu().numpy())
        mask = (mask_np > 0.5).astype("uint8") * 255
        mask_img = Image.fromarray(mask).resize(image.size, resample=Image.NEAREST)
        mask_np_resized = np.array(mask_img)

        # Save mask
        mask_file = mask_dir / f"{name}_clipseg_mask.png"
        mask_img.save(mask_file)
        print(f"Saved mask to {mask_file}")

        # Get bounding box
        coords = np.argwhere(mask_np_resized > 0)
        if coords.shape[0] == 0:
            print(f"No object found in mask for {name}. Skipping crop.")
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        width = x_max - x_min + 1
        height = y_max - y_min + 1
        side_length = int(max(width, height) * 0.95)

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        new_x_min = max(0, x_center - side_length // 2)
        new_y_min = max(0, y_center - side_length // 2)
        new_x_max = min(image.width, new_x_min + side_length)
        new_y_max = min(image.height, new_y_min + side_length)

        new_x_min = max(0, new_x_max - side_length)
        new_y_min = max(0, new_y_max - side_length)

        if new_x_max <= new_x_min or new_y_max <= new_y_min:
            print(f"Invalid crop box. Skipping.")
            continue

        # Crop and save
        cropped_image = image.crop((new_x_min, new_y_min, new_x_max, new_y_max))
        crop_file = crop_dir / f"{name}_square.png"
        cropped_image.save(crop_file)
        print(f"Saved square crop to {crop_file}")

# Recursively find image folders
for folder in input_path.rglob("*"):
    if folder.is_dir():
        try:
            image_files = [f for f in folder.glob("*") if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png'))]
        except Exception as e:
            continue

        if image_files:
            rel_path = folder.relative_to(input_path)
            crop_square_image(processor, model, folder, rel_path)

