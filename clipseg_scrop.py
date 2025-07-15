#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import argparse
import os
import re

# Command-line arguments
parser = argparse.ArgumentParser(description='Square Image Cropping using CLIPSeg')
parser.add_argument('-i', '--input_path', required=True, help='Path to input directory containing subfolders with images')
parser.add_argument('-o', '--output_path', required=True, help='Path to output directory for saving masks and crops')
args = parser.parse_args()

input_path = Path(args.input_path)
output_path = Path(args.output_path)

# Find all subfolders
all_folders = [p for p in input_path.glob("*/") if p.is_dir()]

# Load model and processor
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.eval().cuda()

def clean_prompt(name):
    # Remove digits, parentheses, underscores, dashes, etc.
    cleaned = re.sub(r"[\(\)_\-\d]+", " ", name)
    # Collapse multiple spaces, trim, lowercase
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return f"a {cleaned}"

def crop_square_image(processor, model, input_dir, output_dir):
    folder_name = input_dir.stem
    mask_dir = output_dir / f"{folder_name}_masks"
    crop_dir = output_dir / f"{folder_name}_cropped"
    mask_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(Path(input_dir).glob("*")):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        name = img_path.stem
        prompt = clean_prompt(name)
        print(f" Processing {img_path.name} with prompt: '{prompt}'")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f" Failed to open image {img_path}: {e}")
            continue

        # Prepare input
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")

        # Run inference
        with torch.no_grad():
            logits = model(**inputs).logits
            mask_prob = torch.sigmoid(logits)

        # Convert to binary mask
        mask_np = np.squeeze(mask_prob.cpu().numpy())
        mask = (mask_np > 0.5).astype("uint8") * 255

        # Resize mask
        mask_img = Image.fromarray(mask).resize(image.size, resample=Image.NEAREST)
        mask_np_resized = np.array(mask_img)

        # Save mask
        mask_file = mask_dir / f"{name}_clipseg_mask.png"
        mask_img.save(mask_file)
        print(f" Saved mask to {mask_file}")

        # Get object bounding box
        coords = np.argwhere(mask_np_resized > 0)
        if coords.shape[0] == 0:
            print(f" No object found in mask for {name}. Skipping crop.")
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Compute square crop
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        side_length = max(width, height)

        # Shrink by 5%
        shrink_factor = 0.05
        side_length = int(side_length * (1 - shrink_factor))

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        new_x_min = max(0, x_center - side_length // 2)
        new_y_min = max(0, y_center - side_length // 2)
        new_x_max = min(image.width, new_x_min + side_length)
        new_y_max = min(image.height, new_y_min + side_length)

        new_x_min = max(0, new_x_max - side_length)
        new_y_min = max(0, new_y_max - side_length)

        if new_x_max <= new_x_min or new_y_max <= new_y_min:
            print(f" Final crop box invalid for {name}. Skipping.")
            continue

        # Crop and save
        cropped_image = image.crop((new_x_min, new_y_min, new_x_max, new_y_max))
        crop_file = crop_dir / f"{name}_square.png"
        cropped_image.save(crop_file)
        print(f" Saved square crop to {crop_file}")

# main loop over folders
for folder in all_folders:
    curr_folder = folder.stem
    out_folder = output_path / curr_folder
    out_folder.mkdir(parents=True, exist_ok=True)
    crop_square_image(processor, model, folder, out_folder)
