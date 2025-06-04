import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch

def segment_and_mask_frames(frame_dir, mask_dir, masked_frames_dir, label):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_frames_dir, exist_ok=True)
    
    # Load SegFormer model and processor
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(device)
    model.eval()
    
    # Cityscapes label mapping
    label_map = {
        "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4, "pole": 5,
        "traffic light": 6, "traffic sign": 7, "vegetation": 8, "terrain": 9, "sky": 10,
        "person": 11, "rider": 12, "car": 13, "truck": 14, "bus": 15, "train": 16,
        "motorcycle": 17, "bicycle": 18
    }
    
    if label not in label_map:
        raise ValueError(f"Label '{label}' not supported. Supported labels: {list(label_map.keys())}")
    
    target_label_id = label_map[label]
    print(f"[INFO] Segmenting and masking objects with label: {label} (ID: {target_label_id})")
    
    frame_files = [f for f in sorted(os.listdir(frame_dir)) if f.endswith(".jpg")]
    if not frame_files:
        raise ValueError(f"[ERROR] No .jpg files found in {frame_dir}")
    
    for frame_file in tqdm(frame_files, desc="Segmenting and masking frames", unit="frame"):
        frame_path = os.path.join(frame_dir, frame_file)
        frame_name = frame_file.split('.')[0]
        
        # Load frame
        image = Image.open(frame_path).convert("RGB")
        frame_np = np.array(image)
        
        # Segment frame
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # Shape: (1, num_labels, H/4, W/4)
            
            # Upsample logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=image.size[::-1], mode="bilinear", align_corners=False
            )
            
            # Get predicted segmentation mask
            pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()  # Shape: (H, W)
            
            # Create binary mask for target label
            mask = (pred_seg == target_label_id).astype(np.uint8) * 255
            
            # Save binary mask as NPZ
            np.savez_compressed(os.path.join(mask_dir, f"mask_{frame_name}.npz"), mask=mask)
            
            # Apply mask to frame (set masked areas to black)
            if mask.shape != frame_np.shape[:2]:
                print(f"[WARNING] Mask shape {mask.shape} does not match frame shape {frame_np.shape[:2]} for {frame_file}, resizing")
                mask = np.array(Image.fromarray(mask).resize(frame_np.shape[:2][::-1], Image.NEAREST))
            
            masked_frame = frame_np.copy()
            mask_binary = (mask == 255)  # Convert to boolean mask
            masked_frame[mask_binary] = [0, 0, 0]  # Black out masked areas
            
            # Save masked frame
            masked_image = Image.fromarray(masked_frame)
            output_path = os.path.join(masked_frames_dir, frame_file)
            masked_image.save(output_path)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment objects in video frames and apply masks")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory of input frames")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory to save binary mask .npz files")
    parser.add_argument("--masked_frames_dir", type=str, required=True, help="Directory to save masked frames")
    parser.add_argument("--label", type=str, required=True, help="Object label to segment (e.g., bicycle)")
    args = parser.parse_args()
    
    segment_and_mask_frames(args.frame_dir, args.mask_dir, args.masked_frames_dir, args.label)