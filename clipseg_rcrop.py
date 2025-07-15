#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import argparse
from glob2 import glob as glob
import pdb 
import os

#Coomandline Arguments
parser = argparse.ArgumentParser(description = 'Image Cropping')
parser.add_argument('-i', '--input_path', default=None)
parser.add_argument('-o', '--output_path', default=None)
args = parser.parse_args()
input_path= args.input_path
output_path= args.output_path

#Loading all possible subfolders
all_folders = glob(f"{input_path}/*/")
pdb.set_trace()
# Load processor & model
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
model.eval().cuda()

## Lets pass directories as arguments
def crop_rimage(processor, model, input_path, output_path):

    im_files = glob(f"{input_path}/*")
    # Loop through image files --> including subfolders
    for img_path in im_files:
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        name = img_path.stem  
        prompt = f"a {name}" 
        print(f" Processing {img_path.name} with prompt: '{prompt}'")

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Process inputs
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")

        # Run inference
        with torch.no_grad():
            logits = model(**inputs).logits
            mask_prob = torch.sigmoid(logits)

        # Convert to binary mask
        mask_np = np.squeeze(mask_prob.cpu().numpy())
        mask = (mask_np > 0.5).astype("uint8") * 255

        # Resize mask to match image size
        mask_img = Image.fromarray(mask).resize(image.size, resample=Image.NEAREST)
        mask_np_resized = np.array(mask_img)

        # Save mask
        mask_file = f"{output_path}/{name}_clipseg_mask.png"
        mask_img.save(mask_file)
        print(f" Saved mask to {mask_file}")

        # Get bounding box of mask
        coords = np.argwhere(mask_np_resized > 0)
        if coords.shape[0] == 0:
            print(f" No object found in mask for {name}. Skipping crop.") # add image name 
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Compute dimensions
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        # Shrink each side by 5%, keep center
        shrink_factor = 0.05
        shrink_w = int(width * shrink_factor)
        shrink_h = int(height * shrink_factor)

        x_min_new = max(0, x_min + shrink_w // 2)
        x_max_new = min(image.width, x_max - shrink_w // 2)
        y_min_new = max(0, y_min + shrink_h // 2)
        y_max_new = min(image.height, y_max - shrink_h // 2)

        if x_max_new <= x_min_new or y_max_new <= y_min_new:
            print(f" Cropped area is too small for {name}. Skipping.")
            continue

        # Crop and save
        cropped_image = image.crop((x_min_new, y_min_new, x_max_new, y_max_new))
        crop_file =  f"{output_path}/{name}_rectangle.png"
        cropped_image.save(crop_file)
        print(f" Saved cropped image to {crop_file}")

for folder in all_folders:
    pdb.set_trace()
    curr_folder = folder.split("/")[-2]
    final_output = f"{output_path}/{curr_folder}"
    os.makedirs(final_output,exist_ok=True)
    crop_rimage(processor,model,curr_folder, final_output)
