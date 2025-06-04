import cv2
import os
import argparse

def frames_to_video(frame_dir, output_video_path, fps=30):
    frames = [
        os.path.join(frame_dir, f)
        for f in sorted(os.listdir(frame_dir))
        if f.startswith("frame_") and f.endswith(".jpg")
    ]
    if not frames:
        raise ValueError("No valid frames found in the directory.")
    
    frame = cv2.imread(frames[0])
    height, width, _ = frame.shape
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame_path in frames:
        video_writer.write(cv2.imread(frame_path))
    
    video_writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct video from frames")
    parser.add_argument("--frame_dir", type=str, required=True, help="Directory of processed frames")
    parser.add_argument("--output_video", type=str, required=True, help="Path for output video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()
    
    frames_to_video(args.frame_dir, args.output_video, args.fps)
    print("Video reconstruction complete")