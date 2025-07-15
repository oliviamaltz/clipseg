#!/usr/bin/env python3
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

def main():
    # Load processor & model
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model     = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.eval().cuda()

    #  Load your image
    img_path = Path("test_stimuli/cow.jpg")
    image    = Image.open(img_path).convert("RGB")

    # 3) Build inputs for the text prompt
    prompt = "a cow"  # change to "an airplane" or "a cat" as needed
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda")

    #  Run inference
    with torch.no_grad():
        logits    = model(**inputs).logits        # shape: (1,1,H,W)
        mask_prob = torch.sigmoid(logits)

    # Convert to numpy and remove any singleton dims in one go
    mask_np = np.squeeze(mask_prob.cpu().numpy())   # now shape is (H, W)

    # Binarize and scale
    mask = (mask_np > 0.5).astype("uint8") * 255

    #  Ensure output folder exists
    output_dir = Path("output_masks")
    output_dir.mkdir(exist_ok=True)

    #  Save the mask into output_masks/
    out_file = output_dir / f"{prompt.replace(' ', '_')}_clipseg_mask.png"
    Image.fromarray(mask).save(out_file)
    print(f"Saved mask to {out_file}")

if __name__ == "__main__":
    main()
