import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import make_image_grid
import json

def remove_object(frame_path, mask_path, depth_map, points_path, output_path, intrinsics, debug, model):
    try:
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
        
        # Load point cloud and colors
        points = np.load(points_path)
        orig_img = Image.open(frame_path).convert('RGB')
        orig_resized = orig_img.resize((518, 294), Image.LANCZOS)
        colors = np.array(orig_resized)
        
        # Depth-based mask
        if np.sum(mask == 255) > 0:
            mask_binary = mask == 255
            object_depths = depth_map[mask_binary]
            depth_threshold = np.percentile(object_depths, 50) if len(object_depths) > 0 else np.percentile(depth_map, 70)
        else:
            depth_threshold = np.percentile(depth_map, 70)
        depth_mask = (depth_map > depth_threshold).astype(np.uint8) * 255
        if debug:
            print(f"Depth mask stats for {frame_path}: white_pixels={np.sum(depth_mask == 255)}")
            depth_mask_path = os.path.join(os.path.dirname(output_path), f"depth_mask_{os.path.basename(output_path)}")
            cv2.imwrite(depth_mask_path, depth_mask)
            print(f"Depth mask saved as {depth_mask_path}")
        
        combined_mask = cv2.bitwise_or(mask, depth_mask)
        if debug:
            print(f"Combined mask stats for {frame_path}: white_pixels={np.sum(combined_mask == 255)}")
        
        # Project points to 2D
        height, width = frame.shape[:2]
        fx, fy, cx, cy = intrinsics
        points_flat = points.reshape(-1, 3)
        colors_flat = colors.reshape(-1, 3)
        valid_mask = points_flat[:, 2] > 0
        points_flat = points_flat[valid_mask]
        colors_flat = colors_flat[valid_mask]
        u = (fx * points_flat[:, 0] / points_flat[:, 2] + cx).astype(np.int32)
        v = (fy * points_flat[:, 1] / points_flat[:, 2] + cy).astype(np.int32)
        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v = u[valid], v[valid]
        points_flat = points_flat[valid]
        colors_flat = colors_flat[valid]
        
        # Filter points based on mask
        keep_mask = np.ones(len(u), dtype=bool)
        for i in range(len(u)):
            if combined_mask[v[i], u[i]] > 0:
                keep_mask[i] = False
        points_flat = points_flat[keep_mask]
        colors_flat = colors_flat[keep_mask]
        if debug:
            print(f"Points after filtering: {len(points_flat)}")
        
        # Render filtered point cloud
        rendered_img = frame.copy()
        depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
        if len(points_flat) > 0:
            u = (fx * points_flat[:, 0] / points_flat[:, 2] + cx).astype(np.int32)
            v = (fy * points_flat[:, 1] / points_flat[:, 2] + cy).astype(np.int32)
            valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            u, v = u[valid], v[valid]
            z = points_flat[valid, 2]
            colors_valid = colors_flat[valid]
            for i in range(len(u)):
                if z[i] < depth_buffer[v[i], u[i]]:
                    depth_buffer[v[i], u[i]] = z[i]
                    rendered_img[v[i], u[i]] = colors_valid[i]
        
        if debug:
            # Save pre-inpainting image
            rendered_path = os.path.join(os.path.dirname(output_path), f"rendered_{os.path.basename(output_path)}")
            cv2.imwrite(rendered_path, rendered_img)
            print(f"Pre-inpainting image saved as {rendered_path}")
        
        # Prepare Stable Diffusion inpainting inputs
        rendered_pil = Image.fromarray(cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB))
        sparse_mask = (rendered_img.sum(axis=2) == 0).astype(np.uint8) * 255
        if debug:
            print(f"Sparse mask stats for {frame_path}: white_pixels={np.sum(sparse_mask == 255)}")
            sparse_mask_path = os.path.join(os.path.dirname(output_path), f"sparse_mask_{os.path.basename(output_path)}")
            cv2.imwrite(sparse_mask_path, sparse_mask)
            print(f"Sparse mask saved as {sparse_mask_path}")
        
        # Use SegFormer mask if non-empty
        if np.sum(mask == 255) > 0:
            final_mask = mask
        else:
            final_mask = cv2.bitwise_or(combined_mask, sparse_mask)
            if debug:
                print(f"Warning: Using fallback mask for {frame_path}")
        if debug:
            print(f"Final mask stats for {frame_path}: min={final_mask.min()}, max={final_mask.max()}, white_pixels={np.sum(final_mask == 255)}")
            if final_mask.max() == 0:
                print(f"Warning: Final mask is all black for {frame_path}")
        
        # Dilate mask
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.dilate(final_mask, kernel, iterations=1)
        
        mask_pil = Image.fromarray(final_mask).resize((512, 512), Image.NEAREST)
        init_image = rendered_pil.resize((512, 512), Image.LANCZOS)
        
        if debug:
            # Save final mask
            mask_path = os.path.join(os.path.dirname(output_path), f"mask_{os.path.basename(output_path)}")
            cv2.imwrite(mask_path, final_mask)
            print(f"Final mask saved as {mask_path}")
        
        # Run Stable Diffusion inpainting
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = AutoPipelineForInpainting.from_pretrained(
            model,
            torch_dtype=torch.float16,
            variant="fp16"
        ).to(device)
        pipe.enable_model_cpu_offload()
        
        prompt = "Fill in with content similar to the surroundings"
        generator = torch.Generator(device=device).manual_seed(92)
        inpainted_image = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_pil,
            generator=generator,
            strength=1.0,
            guidance_scale=7.5,
            num_inference_steps=100
        ).images[0]
        
        if debug:
            # Save post-inpainting image
            inpainted_path = os.path.join(os.path.dirname(output_path), f"inpainted_{os.path.basename(output_path)}")
            inpainted_image.save(inpainted_path)
            print(f"Post-inpainting image saved as {inpainted_path}")
        
        # Resize and blend
        inpainted_image = inpainted_image.resize((width, height), Image.LANCZOS)
        inpainted_np = np.array(inpainted_image)
        final_mask = final_mask.astype(bool)
        rendered_img[final_mask] = cv2.cvtColor(inpainted_np, cv2.COLOR_RGB2BGR)[final_mask]
        
        # Save final output
        cv2.imwrite(output_path, rendered_img)
        print(f"Output saved as {output_path}")
        
        if debug:
            # Save debug grid
            grid = make_image_grid([orig_img.resize((512, 512)), mask_pil, inpainted_image], rows=1, cols=3)
            grid_path = os.path.join(os.path.dirname(output_path), f"grid_{os.path.basename(output_path)}")
            grid.save(grid_path)
            print(f"Debug grid saved as {grid_path}")

    except Exception as e:
        print(f"An error occurred for {frame_path}: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove objects and inpaint frames")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory of input frames")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory of masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for processed frames")
    parser.add_argument("--vggt_dir", type=str, default="output/vggt_results", help="Directory of VGGT results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save intermediate outputs")
    parser.add_argument("--model", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                        choices=["diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "runwayml/stable-diffusion-inpainting"],
                        help="Inpainting model to use")
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
    
    # Load processed frames
    processed_frames_file = os.path.join(args.vggt_dir, "processed_frames.txt")
    with open(processed_frames_file, "r") as f:
        processed_frames = [line.strip() for line in f.readlines()]
    
    for frame_file in tqdm(processed_frames, desc="Removing objects", unit="frame"):
        if frame_file.endswith(".jpg"):
            frame_path = os.path.join(args.frame_dir, frame_file)
            frame_name = frame_file.split('.')[0]
            mask_path = os.path.join(args.mask_dir, f"mask_frame_{frame_name.split('_')[1]}.npz")
            depth_path = os.path.join(args.vggt_dir, f"depth_{frame_name}.npy")
            points_path = os.path.join(args.vggt_dir, f"points_{frame_name}.npy")
            output_path = os.path.join(args.output_dir, frame_file)
            depth_map = np.load(depth_path)
            remove_object(frame_path, mask_path, depth_map, points_path, output_path, intrinsics, args.debug, args.model)