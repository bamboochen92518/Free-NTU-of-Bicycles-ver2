import os
import numpy as np
from PIL import Image
import cv2
import argparse
from tqdm import tqdm
import open3d as o3d
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
from torchvision import transforms
from pytorch_fid import fid_score

def visualize_depth_map(depth_map, output_path, frame_shape):
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6) * 255
    depth_normalized = depth_normalized.astype(np.uint8)
    depth_image = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    if depth_image.shape[:2] != frame_shape:
        depth_image = cv2.resize(depth_image, (frame_shape[1], frame_shape[0]))
    cv2.imwrite(output_path, depth_image)
    print(f"[DEBUG] Saved depth map visualization to: {output_path}")

def save_point_cloud(points, output_path):
    try:
        if points.ndim == 3:
            points = points.reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"[DEBUG] Saved point cloud to: {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save point cloud to {output_path}: {e}")

def compute_psnr(img1, img2):
    img1_np = np.array(img1).astype(np.float32)
    img2_np = np.array(img2).astype(np.float32)
    return psnr(img1_np, img2_np, data_range=255)

def compute_lpips(img1, img2, lpips_model):
    transform = transforms.ToTensor()
    img1_tensor = transform(img1).unsqueeze(0).to(device='cuda' if torch.cuda.is_available() else 'cpu')
    img2_tensor = transform(img2).unsqueeze(0).to(device='cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        return lpips_model(img1_tensor, img2_tensor).item()

def visualize_and_compute_metrics(vggt_dir, frame_dir, processed_frames_dir, output_dir="output/vggt_metrics"):
    lpips_model = lpips.LPIPS(net='vgg').to('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.join(output_dir, 'depth_vis'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'point_clouds'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed_frames'), exist_ok=True)
    processed_frames_file = os.path.join(vggt_dir, "processed_frames.txt")
    if not os.path.exists(processed_frames_file):
        raise FileNotFoundError(f"[ERROR] Processed frames list not found at {processed_frames_file}")
    with open(processed_frames_file, "r") as f:
        processed_frames = [line.strip() for line in f.readlines()]
    if not processed_frames:
        raise ValueError(f"[ERROR] No frames listed in {processed_frames_file}")
    print(f"[INFO] Found {len(processed_frames)} processed frames in {processed_frames_file}")
    frame_files = [f for f in sorted(os.listdir(frame_dir)) if f.endswith(".jpg")]
    if not frame_files:
        raise FileNotFoundError(f"[ERROR] No frames found in {frame_dir}")
    first_frame_path = os.path.join(frame_dir, frame_files[0])
    with Image.open(first_frame_path) as img:
        frame_shape = img.size[::-1]
    print(f"[INFO] Expected frame dimensions (H, W): {frame_shape}")
    psnr_values = []
    lpips_values = []
    valid_frames = 0
    missing_files = 0
    invalid_shapes = 0
    for frame_file in tqdm(processed_frames, desc="Processing VGGT results", unit="frame"):
        if not frame_file.endswith(".jpg"):
            print(f"[WARNING] Skipping non-JPG file: {frame_file}")
            continue
        frame_name = frame_file.split('.')[0]
        depth_path = os.path.join(vggt_dir, f"depth_{frame_name}.npy")
        points_path = os.path.join(vggt_dir, f"points_{frame_name}.npy")
        orig_frame_path = os.path.join(frame_dir, frame_file)
        proc_frame_path = os.path.join(processed_frames_dir, frame_file)
        if not os.path.exists(depth_path):
            print(f"[ERROR] Depth map missing for {frame_file}: {depth_path}")
            missing_files += 1
            continue
        if not os.path.exists(points_path):
            print(f"[ERROR] Point cloud missing for {frame_file}: {points_path}")
            missing_files += 1
            continue
        if not os.path.exists(orig_frame_path):
            print(f"[ERROR] Original frame missing: {orig_frame_path}")
            missing_files += 1
            continue
        if not os.path.exists(proc_frame_path):
            print(f"[ERROR] Processed frame missing: {proc_frame_path}")
            missing_files += 1
            continue
        try:
            depth_map = np.load(depth_path)
            if depth_map.size == 0:
                print(f"[ERROR] Depth map is empty for {frame_file}: {depth_path}")
                missing_files += 1
                continue
            if depth_map.ndim > 2:
                depth_map = depth_map.squeeze()
            if depth_map.ndim != 2:
                print(f"[ERROR] Depth map shape {depth_map.shape} is not 2D for {frame_file}")
                invalid_shapes += 1
                continue
            if depth_map.shape != frame_shape:
                print(f"[WARNING] Depth map shape {depth_map.shape} does not match frame shape {frame_shape} for {frame_file}, resizing")
                depth_map = cv2.resize(depth_map, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_LINEAR)
            depth_vis_path = os.path.join(output_dir, 'depth_vis', f"depth_vis_{frame_name}.jpg")
            visualize_depth_map(depth_map, depth_vis_path, frame_shape)
            points = np.load(points_path)
            points_ply_path = os.path.join(output_dir, 'point_clouds', f"points_{frame_name}.ply")
            save_point_cloud(points, points_ply_path)
            orig_img = Image.open(orig_frame_path).convert('RGB')
            proc_img = Image.open(proc_frame_path).convert('RGB')
            psnr_val = compute_psnr(orig_img, proc_img)
            lpips_val = compute_lpips(orig_img, proc_img, lpips_model)
            psnr_values.append(psnr_val)
            lpips_values.append(lpips_val)
            proc_img.save(os.path.join(output_dir, 'processed_frames', frame_file))
            print(f"[DEBUG] Saved processed frame to: {os.path.join(output_dir, 'processed_frames', frame_file)}")
            valid_frames += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {frame_file}: {e}")
            missing_files += 1
            continue
    try:
        fid_value = fid_score.calculate_fid_given_paths(
            [frame_dir, processed_frames_dir],
            batch_size=50,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dims=2048
        )
        print(f"[INFO] FID score: {fid_value:.4f}")
    except Exception as e:
        print(f"[ERROR] Failed to compute FID: {e}")
        fid_value = None
    metrics_report = os.path.join(output_dir, "metrics_report.txt")
    with open(metrics_report, "w") as f:
        f.write(f"VGGT and Metrics Summary\n")
        f.write(f"Total frames checked: {len(processed_frames)}\n")
        f.write(f"Valid frames: {valid_frames}\n")
        f.write(f"Missing or empty files: {missing_files}\n")
        f.write(f"Invalid shapes: {invalid_shapes}\n")
        f.write(f"PSNR (mean ± std): {np.mean(psnr_values):.4f} ± {np.std(psnr_values):.4f}\n")
        f.write(f"LPIPS (mean ± std): {np.mean(lpips_values):.4f} ± {np.std(lpips_values):.4f}\n")
        f.write(f"FID: {fid_value if fid_value is not None else 'N/A'}\n")
        f.write(f"Visualizations saved to: {output_dir}/depth_vis\n")
        f.write(f"Point clouds saved to: {output_dir}/point_clouds\n")
        f.write(f"Processed frames saved to: {output_dir}/processed_frames\n")
    print(f"[INFO] Metrics report saved to: {metrics_report}")
    if missing_files > 0 or invalid_shapes > 0:
        raise RuntimeError(f"[ERROR] Validation failed: {missing_files} missing/empty files, {invalid_shapes} invalid shapes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize VGGT results and compute metrics (PSNR, LPIPS, FID)")
    parser.add_argument("--vggt_dir", type=str, required=True, help="Directory of VGGT results")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory of original frames")
    parser.add_argument("--processed_frames_dir", type=str, required=True, help="Directory of inpainted frames")
    parser.add_argument("--output_dir", type=str, default="output/vggt_metrics", help="Directory to save visualizations and metrics")
    args = parser.parse_args()
    visualize_and_compute_metrics(args.vggt_dir, args.frame_dir, args.processed_frames_dir, args.output_dir)