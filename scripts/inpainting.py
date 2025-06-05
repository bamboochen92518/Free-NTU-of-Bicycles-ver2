import cv2
import os
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from diffusers import AutoPipelineForInpainting

def inpaint_image(frame_path, mask_path, output_path, debug, model):
    # Load frame and mask
    frame = cv2.imread(frame_path)
    mask = cv2.imread(mask_path)
    
    # Prepare Stable Diffusion inpainting inputs
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    
    init_image = image.resize((512, 512), Image.LANCZOS)
    init_mask = mask.resize((512, 512), Image.LANCZOS)
    
    if debug:
        # Save final mask
        mask_path = os.path.join(os.path.dirname(output_path), f"mask_{os.path.basename(output_path)}")
        init_mask.save(mask_path)
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
        mask_image=init_mask,
        generator=generator,
        strength=1.0,
        guidance_scale=7.5,
        num_inference_steps=100
    ).images[0]
    
    # Resize and save final output
    inpainted_image = inpainted_image.resize((frame.shape[1], frame.shape[0]), Image.LANCZOS)
    inpainted_np = np.array(inpainted_image)
    final_output = cv2.cvtColor(inpainted_np, cv2.COLOR_RGB2BGR)
    
    # Save final output
    cv2.imwrite(output_path, final_output)
    print(f"Output saved as {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpaint frames using Stable Diffusion")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory of input frames")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory of mask images (JPG)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for inpainted frames")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to save intermediate outputs")
    parser.add_argument("--model", type=str, default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                        choices=["diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "runwayml/stable-diffusion-inpainting"],
                        help="Inpainting model to use")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all JPG files in frame_dir
    for frame_file in tqdm(os.listdir(args.frame_dir), desc="Inpainting frames", unit="frame"):
        if frame_file.endswith(".jpg"):
            frame_path = os.path.join(args.frame_dir, frame_file)
            frame_name = frame_file.split('.')[0]
            mask_path = os.path.join(args.mask_dir, f"mask_frame_{frame_name.split('_')[1]}.jpg")
            output_path = os.path.join(args.output_dir, frame_file)
            if os.path.exists(mask_path):
                inpaint_image(frame_path, mask_path, output_path, args.debug, args.model)
            else:
                print(f"Mask file not found for {frame_path}: {mask_path}")