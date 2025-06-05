import cv2
import os
import argparse

def frames_to_video(frame_dir, output_video_path, fps=30, scale=1.0, mask_mode=False):
    # Select frames based on mask_mode
    frame_prefix = "mask_frame_" if mask_mode else "frame_"
    frames = [
        os.path.join(frame_dir, f)
        for f in sorted(os.listdir(frame_dir))
        if f.startswith(frame_prefix) and f.endswith(".jpg")
    ]
    if not frames:
        raise ValueError(f"No valid frames found in the directory with prefix '{frame_prefix}'.")
    
    # Load first frame to get dimensions
    frame = cv2.imread(frames[0])
    if frame is None:
        raise ValueError(f"Could not read first frame: {frames[0]}")
    height, width, _ = frame.shape
    
    # Apply scaling
    new_width = int(width * scale)
    new_height = int(height * scale)
    if new_width <= 0 or new_height <= 0:
        raise ValueError("Scale factor results in invalid dimensions (width or height <= 0).")
    
    # Initialize video writer with scaled dimensions
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_width, new_height))

    for frame_path in frames:
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}, skipping...")
            continue
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        video_writer.write(resized_frame)
    
    video_writer.release()
    print(f"Video reconstruction complete: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct video from frames with optional scaling")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory of processed frames")
    parser.add_argument("--output_video", type=str, required=True, help="Path for output video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for frames (e.g., 0.6667 for 2/3)")
    parser.add_argument("--mask_mode", action="store_true", help="Use mask frames (mask_frame_*.jpg) instead of regular frames (frame_*.jpg)")
    args = parser.parse_args()
    
    frames_to_video(args.frame_dir, args.output_video, args.fps, args.scale, args.mask_mode)