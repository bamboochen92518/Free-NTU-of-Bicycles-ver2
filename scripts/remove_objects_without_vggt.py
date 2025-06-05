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

def remove_object(frame_path, mask_path, output_path, intrinsics, debug, model):
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
        
        # Convert frame to PIL for inpainting
        orig_img = Image.open(frame_path).convert('RGB')
        
        # Use SegFormer mask if non-empty
        if np.sum(mask == 255) == 0:
            print(f"Warning: Mask is all black for {frame_path}")
        
        # Dilate mask
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Resize images for inpainting
        mask_pil = Image.fromarray(final_mask).resize((512, 512), Image.NEAREST)
        init_image = orig_img.resize((512, 512), Image.LANCZOS)
        
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
        height, width = frame.shape[:2]
        inpainted_image = inpainted_image.resize((width, height), Image.LANCZOS)
        inpainted_np = np.array(inpainted_image)
        final_mask = final_mask.astype(bool)
        frame[final_mask] = cv2.cvtColor(inpainted_np, cv2.COLOR_RGB2BGR)[final_mask]
        
        # Save final output
        cv2.imwrite(output_path, frame)
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
    
    # Process all jpg frames in frame_dir
    for frame_file in tqdm(os.listdir(args.frame_dir), desc="Removing objects", unit="frame"):
        if frame_file.endswith(".jpg"):
            frame_path = os.path.join(args.frame_dir, frame_file)
            frame_name = frame_file.split('.')[0]
            mask_path = os.path.join(args.mask_dir, f"mask_frame_{frame_name.split('_')[1]}.npz")
            output_path = os.path.join(args.output_dir, frame_file)
            if os.path.exists(mask_path):
                remove_object(frame_path, mask_path, output_path, intrinsics, args.debug, args.model)
            else:
                print(f"Mask not found for {frame_file}, skipping...")