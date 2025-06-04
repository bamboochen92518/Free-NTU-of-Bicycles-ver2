import cv2
import os
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    with tqdm(total=frame_count, desc="Extracting frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id:04d}.jpg"), frame)
            frame_id += 1
            pbar.update(1)
    cap.release()
    return frame_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video with progress bar")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save frames")
    args = parser.parse_args()
    num_frames = extract_frames(args.video_path, args.output_dir)
    print(f"Extracted {num_frames} frames")
