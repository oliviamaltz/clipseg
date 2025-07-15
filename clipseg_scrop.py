#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def main():
    # Load processor & model
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.eval().cuda()

    # Set directories
    input_dir = Path("test_stimuli")
    output_dir = Path("output_masks")
    crop_dir = Path("sq_crops")
    output_dir.mkdir(exist_ok=True)
    crop_dir.mkdir(exist_ok=True)

    # Loop through image files (supports .jpg, .jpeg, .png)
    for img_path in sorted(input_dir.glob("*")):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        name = img_path.stem
        prompt = f"a {name}"
        print(f" Processing {img_path.name} with prompt: '{prompt}'")

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Prepare input
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")

        # Run inference
        with torch.no_grad():
            logits = model(**inputs).logits
            mask_prob = torch.sigmoid(logits)

        # Convert to binary mask
        mask_np = np.squeeze(mask_prob.cpu().numpy())
        mask = (mask_np > 0.5).astype("uint8") * 255

        # Resize mask to original image size
        mask_img = Image.fromarray(mask).resize(image.size, resample=Image.NEAREST)
        mask_np_resized = np.array(mask_img)

        # Save resized mask
        mask_file = output_dir / f"{name}_clipseg_mask.png"
        mask_img.save(mask_file)
        print(f" Saved mask to {mask_file}")

        # Get object bounding box from mask
        coords = np.argwhere(mask_np_resized > 0)
        if coords.shape[0] == 0:
            print(f" No object found in mask for {name}. Skipping crop.")
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Compute square crop centered on object
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        side_length = max(width, height)

        # Shrink square side by 5%
        shrink_factor = 0.05
        side_length = int(side_length * (1 - shrink_factor))

        # Center square
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        # Compute square bounds
        new_x_min = max(0, x_center - side_length // 2)
        new_y_min = max(0, y_center - side_length // 2)
        new_x_max = min(image.width, new_x_min + side_length)
        new_y_max = min(image.height, new_y_min + side_length)

        # Re-adjust min if max was clipped
        new_x_min = max(0, new_x_max - side_length)
        new_y_min = max(0, new_y_max - side_length)

        # Final check
        if new_x_max <= new_x_min or new_y_max <= new_y_min:
            print(f" Final crop box invalid for {name}. Skipping.")
            continue

        
        cropped_image = image.crop((new_x_min, new_y_min, new_x_max, new_y_max))
        crop_file = crop_dir / f"{name}_square.png"
        cropped_image.save(crop_file)
        print(f" Saved square crop to {crop_file}")

if __name__ == "__main__":
    main()
