import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import json
from scripts.utils import render_point_cloud

def remove_object(frame_path, mask_path, depth_map, points_path, render_output_path, mask_output_path, intrinsics, debug):
    # Load frame and mask
    height, width = 294, 518
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"Failed to load frame: {frame_path}")
    mask_data = np.load(mask_path)["mask"]
    mask = mask_data.squeeze() if mask_data.ndim > 2 else mask_data
    # Convert to binary mask: white (255) for mask, black (0) for background
    mask = (mask > 0).astype(np.uint8) * 255  # Threshold to binary
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Use SegFormer mask if non-empty
    if np.sum(mask == 255) == 0:
        print(f"Warning: Mask is all black for {frame_path}")
    
    # Load point cloud and colors
    points = np.load(points_path)
    orig_img = Image.open(frame_path).convert('RGB')
    orig_width, orig_height = orig_img.size  # Get original image size
    orig_resized = orig_img.resize((518, 294), Image.LANCZOS)
    colors = np.array(orig_resized)
    points_flat = points.reshape(-1, 3)
    
    combined_mask = mask
    if debug:
        print(f"Combined mask stats for {frame_path}: white_pixels={np.sum(combined_mask == 255)}")
    
    # Project points to 2D with debug
    fx, fy, cx, cy = intrinsics
    valid_mask = points_flat[:, 2] > 0
    if debug:
        print(f"Points with positive depth: {np.sum(valid_mask)}")
    points_flat = points_flat[valid_mask]
    
    # Project points to get u, v coordinates
    u = (fx * points_flat[:, 0] / points_flat[:, 2] + cx).astype(np.int32)
    v = (fy * points_flat[:, 1] / points_flat[:, 2] + cy).astype(np.int32)
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v = u[valid], v[valid]
    points_flat = points_flat[valid]
    
    # Sample colors at projected coordinates
    colors_valid = np.zeros((len(u), 3), dtype=np.uint8)
    for i in range(len(u)):
        colors_valid[i] = colors[v[i], u[i]]
    
    # Filter points based on binary mask
    keep_mask = np.ones(len(u), dtype=bool)
    for i in range(len(u)):
        if combined_mask[v[i], u[i]] == 255:  # Only check for white pixels
            keep_mask[i] = False
    points_flat = points_flat[keep_mask]
    colors_valid = colors_valid[keep_mask]
    u = u[keep_mask]
    v = v[keep_mask]
    if debug:
        print(f"Points after filtering: {len(points_flat)}")
    
    # Render filtered point cloud on a blank image
    rendered_img = render_point_cloud(points_flat, colors_valid, frame.shape[:2], intrinsics)
    
    # Save pre-inpainting image
    rendered_path = os.path.join(os.path.dirname(render_output_path), os.path.basename(render_output_path))
    rendered_img.save(rendered_path)
    
    # Dilate binary mask
    kernel = np.ones((5, 5), np.uint8)
    final_mask = cv2.dilate(mask, kernel, iterations=1)
    # Ensure final mask remains binary
    final_mask = (final_mask > 0).astype(np.uint8) * 255
    # Resize mask to original image size
    final_mask = cv2.resize(final_mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
    # Save binary mask
    mask_path = os.path.join(os.path.dirname(mask_output_path), os.path.basename(mask_output_path))
    cv2.imwrite(mask_path, final_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pre-inpainting images with filtered point clouds")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory of input frames")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory of masks")
    parser.add_argument("--render_output_dir", type=str, required=True, help="Directory for pre-inpainting images")
    parser.add_argument("--mask_output_dir", type=str, required=True, help="Directory for binary masks")
    parser.add_argument("--vggt_dir", type=str, default="output/vggt_results", help="Directory of VGGT results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save intermediate outputs")
    args = parser.parse_args()
    
    os.makedirs(args.render_output_dir, exist_ok=True)
    os.makedirs(args.mask_output_dir, exist_ok=True)
    
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
    
    # Load processed frames
    processed_frames_file = os.path.join(args.vggt_dir, "processed_frames.txt")
    with open(processed_frames_file, "r") as f:
        processed_frames = [line.strip() for line in f.readlines()]
    
    for frame_file in tqdm(processed_frames, desc="Generating pre-inpainting images and masks", unit="frame"):
        if frame_file.endswith(".jpg"):
            frame_path = os.path.join(args.frame_dir, frame_file)
            frame_name = frame_file.split('.')[0]
            mask_path = os.path.join(args.mask_dir, f"mask_frame_{frame_name.split('_')[1]}.npz")
            depth_path = os.path.join(args.vggt_dir, f"depth_{frame_name}.npy")
            points_path = os.path.join(args.vggt_dir, f"points_{frame_name}.npy")
            render_output_path = os.path.join(args.render_output_dir, frame_file)
            mask_output_path = os.path.join(args.mask_output_dir, f"mask_{frame_file}")
            depth_map = np.load(depth_path)
            remove_object(frame_path, mask_path, depth_map, points_path, render_output_path, mask_output_path, intrinsics, args.debug)