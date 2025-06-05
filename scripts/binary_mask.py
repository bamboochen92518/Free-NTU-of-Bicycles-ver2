import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
import json

def generate_binary_mask(frame_path, mask_path, output_path, intrinsics):
    # Load frame and mask
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Failed to load frame: {frame_path}")
    mask_data = np.load(mask_path)["mask"]
    mask = mask_data.squeeze() if mask_data.ndim > 2 else mask_data
    # Normalize mask to [0, 255]
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Use SegFormer mask if non-empty
    if np.sum(mask == 255) == 0:
        print(f"Warning: Mask is all black for {frame_path}")
    
    # Dilate mask
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Save binary mask frame (white for mask, black for non-mask)
    binary_mask = np.where(final_mask > 0, 255, 0).astype(np.uint8)
    binary_mask_path = os.path.join(os.path.dirname(output_path), f"{os.path.basename(output_path).split('.')[0]}.jpg")
    cv2.imwrite(binary_mask_path, binary_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary mask frames")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory of input frames")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory of masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for binary mask frames")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load intrinsics from JSON file
    intrinsics_path = os.path.join(args.frame_dir, 'intrinsics.json')
    if not os.path.exists(intrinsics_path):
        raise FileNotFoundError(f"Intrinsics file not found at {intrinsics_path}")
    with open(intrinsics_path, 'r') as f:
        intrinsics_data = json.load(f)
    intrinsics = [
        intrinsics_data['fx'],
        intrinsics_data['fy'],
        intrinsics_data['cx'],
        intrinsics_data['cy']
    ]
    
    # Process all jpg frames in frame_dir
    for frame_file in tqdm(os.listdir(args.frame_dir), desc="Generating binary masks", unit="frame"):
        if frame_file.endswith(".jpg"):
            frame_path = os.path.join(args.frame_dir, frame_file)
            frame_name = frame_file.split('.')[0]
            mask_path = os.path.join(args.mask_dir, f"mask_frame_{frame_name.split('_')[1]}.npz")
            output_path = os.path.join(args.output_dir, frame_file)
            if os.path.exists(mask_path):
                generate_binary_mask(frame_path, mask_path, output_path, intrinsics)
            else:
                print(f"Mask not found for {frame_file}, skipping...")