import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import json
from scipy.spatial.transform import Rotation as R
import os
import argparse
import numpy as np
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process images for camera pose estimation')
parser.add_argument('--frame_dir', type=str, required=True, help='Directory containing input images')
parser.add_argument('--batch_size', type=int, default=8, help='Number of images per batch')
args = parser.parse_args()

# Get list of image files from directory
image_dir = args.frame_dir
image_extensions = ('.jpg', '.jpeg', '.png')
image_names = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if os.path.isfile(os.path.join(image_dir, f)) 
               and f.lower().endswith(image_extensions)]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize model
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

intrinsic = []

# Process images in batches with tqdm
for i in tqdm(range(0, len(image_names), args.batch_size), desc="Processing batches", unit="batch"):
    current_image_names = image_names[i:i + args.batch_size]
    
    images = load_and_preprocess_images(current_image_names).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extri, intri = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        extri, intri = extri.squeeze(0), intri.squeeze(0)

    for i in intri:
        i_result = {
            "fx": float(i[0, 0]),
            "fy": float(i[1, 1]),
            "cx": float(i[0, 2]),
            "cy": float(i[1, 2])
        }
        intrinsic.append(i_result)

    # Clear cache after each batch
    torch.cuda.empty_cache()

# Calculate mean of intrinsic parameters
fx_mean = np.mean([i["fx"] for i in intrinsic])
fy_mean = np.mean([i["fy"] for i in intrinsic])
cx_mean = np.mean([i["cx"] for i in intrinsic])
cy_mean = np.mean([i["cy"] for i in intrinsic])

mean_intrinsic = {
    "fx": float(fx_mean),
    "fy": float(fy_mean),
    "cx": float(cx_mean),
    "cy": float(cy_mean)
}

# Save results
with open(os.path.join(image_dir, 'intrinsics.json'), 'w') as f:
    json.dump(mean_intrinsic, f)