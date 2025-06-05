import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import open3d as o3d
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from torchvision import transforms
import json
from scripts.utils import render_point_cloud

def visualize_depth_map(depth_map, output_path, frame_shape):
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6) * 255
    depth_normalized = depth_normalized.astype(np.uint8)
    depth_image = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    if depth_image.shape[:2] != frame_shape:
        depth_image = cv2.resize(depth_image, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(output_path, depth_image)

def save_point_cloud(points, colors, output_path):
    if points.ndim == 3:
        points = points.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = colors / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, pcd)

def load_point_cloud(input_path):
    # Read the point cloud
    pcd = o3d.io.read_point_cloud(input_path)
    
    # Convert points and colors to NumPy arrays
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    
    # Convert colors from [0, 1] to [0, 255] to match save_point_cloud format
    colors = (colors * 255.0).astype(np.uint8)
    
    return points, colors

def compute_psnr(img1, img2):
    img1_np = np.array(img1).astype(np.float32)
    img2_np = np.array(img2).astype(np.float32)
    return psnr(img1_np, img2_np, data_range=255)

def compute_lpips(img1, img2, lpips_model):
    transform = transforms.ToTensor()
    img1_tensor = transform(img1).unsqueeze(0).to(device='cuda' if torch.cuda.is_available() else 'cpu')
    img2_tensor = transform(img2).unsqueeze(0).to(device='cuda' if torch.cuda.is_available() else 'cpu')
    return lpips_model(img1_tensor, img2_tensor).item()

def run_vggt(frame_dir, batch_size=2, sample_rate=1, output_dir="output/vggt_results"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'depth_vis'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'point_clouds'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'rendered_frames'), exist_ok=True)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    frame_files = [f for f in sorted(os.listdir(frame_dir)) if f.endswith(".jpg")]
    frame_files = frame_files[::sample_rate]
    if not frame_files:
        raise ValueError(f"No valid frames in {frame_dir}")
    first_frame_path = os.path.join(frame_dir, frame_files[0])
    with Image.open(first_frame_path) as img:
        orig_shape = img.size[::-1]
    if not all(orig_shape):
        raise ValueError(f"Invalid frame dimensions: {orig_shape}")
    # Load intrinsics from JSON file
    intrinsics_path = os.path.join(frame_dir, 'intrinsics.json')
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
    processed_frames = []
    psnr_values = []
    lpips_values = []
    for i in tqdm(range(0, len(frame_files), batch_size), desc="Processing batches", unit="batch"):
        batch_files = frame_files[i:i + batch_size]
        batch_paths = [os.path.join(frame_dir, f) for f in batch_files]
        images = load_and_preprocess_images(batch_paths).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
                batch_depth = predictions["depth"].cpu().numpy()
                batch_points = predictions["world_points"].cpu().numpy()
        for j in range(batch_depth.shape[1]):
            frame_file = batch_files[j]
            frame_name = batch_files[j].split('.')[0]
            depth_map = batch_depth[0][j].squeeze()
            depth_map_resized = cv2.resize(depth_map, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
            points = batch_points[0][j]
            if points.ndim > 3:
                points = points.squeeze(0)
            orig_img = Image.open(os.path.join(frame_dir, frame_file)).convert('RGB')
            orig_resized = orig_img.resize((518, 294), Image.LANCZOS)
            orig_colors = np.array(orig_resized)
            depth_path = os.path.join(output_dir, f"depth_{frame_name}.npy")
            np.save(depth_path, depth_map_resized)
            points_path = os.path.join(output_dir, f"points_{frame_name}.npy")
            np.save(points_path, points)
            points_ply_path = os.path.join(output_dir, 'point_clouds', f"points_{frame_name}.ply")
            save_point_cloud(points, orig_colors, points_ply_path)
            depth_vis_path = os.path.join(output_dir, 'depth_vis', f"depth_vis_{frame_name}.jpg")
            visualize_depth_map(depth_map_resized, depth_vis_path, orig_shape)
            rendered_img = render_point_cloud(points, orig_colors, orig_shape, intrinsics)
            rendered_img_path = os.path.join(output_dir, 'rendered_frames', f"rendered_{frame_name}.jpg")
            rendered_img.save(rendered_img_path)
            psnr_val = compute_psnr(orig_img, rendered_img)
            lpips_val = compute_lpips(orig_img, rendered_img, lpips_model)
            psnr_values.append(psnr_val)
            lpips_values.append(lpips_val)
            processed_frames.append(frame_file)
        torch.cuda.empty_cache()
    with open(os.path.join(output_dir, "processed_frames.txt"), "w") as f:
        for frame in processed_frames:
            f.write(f"{frame}\n")
    metrics_report = os.path.join(output_dir, "metrics_report.txt")
    with open(metrics_report, "w") as f:
        f.write(f"VGGT Metrics\n")
        f.write(f"Frames processed: {len(frame_files)}\n")
        f.write(f"PSNR (mean ± std): {np.mean(psnr_values):.4f} ± {np.std(psnr_values):.4f}\n")
        f.write(f"LPIPS (mean ± std): {np.mean(lpips_values):.4f} ± {np.std(lpips_values):.4f}\n")
        f.write(f"Outputs saved to: {output_dir}\n")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VGGT for 3D reconstruction")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory of input frames")
    parser.add_argument("--batch_size", type=int, default=2, help="Number of frames per batch")
    parser.add_argument("--sample_rate", type=int, default=1, help="Sample every nth frame")
    parser.add_argument("--output_dir", type=str, default="output/vggt_results", help="Directory to save results")
    args = parser.parse_args()
    output_dir = run_vggt(
        args.frame_dir, args.batch_size, args.sample_rate, args.output_dir
    )