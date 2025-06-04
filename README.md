# Free NTU of Bicycle ver2

## Overview

This project integrates **SegFormer** for semantic segmentation and **VGGT** for 3D scene reconstruction to automatically remove specified objects (e.g., bicycles) from a video. The complete pipeline includes:

1. Extracting frames from the video
2. Segmenting and masking target objects
3. Performing 3D scene reconstruction
4. Inpainting masked regions using depth information
5. Computing quality metrics
6. Reconstructing the final output video

## Project Structure

```
Free-NTU-of-Bicycle-ver2/
├── third_party/
│   └── vggt/                        # VGGT 3D reconstruction module
├── input/
│   └── video.mp4                    # Input video
├── output/
│   ├── frames/                      # Extracted frames and camera intrinsics
│   ├── masks_npz/                   # Segmentation masks (NPZ format)
│   ├── masked_frames/              # Masked frames (objects blacked out)
│   ├── vggt_results/                # VGGT depth maps and point clouds
│   ├── vggt_metrics/                # Visualizations and evaluation metrics
│   ├── processed_frames/            # Inpainted frames
│   └── processed_frames_continue/   # Continuously inpainted frames
├── scripts/
│   ├── extract_frames.py            # Frame extraction with progress bar
│   ├── segment.py                   # SegFormer-based object segmentation
│   ├── mask_frames.py               # Object masking utility
│   ├── run_vggt.py                  # VGGT 3D reconstruction
│   ├── visualize_vggt_metrics.py    # Metric visualization and evaluation
│   ├── remove_objects.py            # Inpainting using segmentation and depth
│   ├── remove_objects_continuous.py # Continuous inpainting version
│   ├── reconstruct_video.py         # Frame-to-video reconstruction
│   └── intrinsic.py                 # Camera intrinsics estimation
├── requirements.txt                 # Python dependencies
├── output_video.mp4                 # Final output video
└── README.md                        # Project documentation
```

## Setup Instructions

### Environment Setup

Clone the repository:

```bash
git clone git@github.com:bamboochen92518/Free-NTU-of-Bicycles-ver2.git
cd Free-NTU-of-Bicycles-ver2
```

Set up a virtual environment and install dependencies:

```bash
python -m venv VGGTenv
source VGGTenv/bin/activate
pip install -r requirements.txt
```

Place your input video (e.g., `video.mp4`) into the `input/` directory.

## Step-by-Step Guide

### 1. Extract Video Frames

**Script**: `scripts/extract_frames.py`
**Function**: Extracts frames from the input video.

```bash
python scripts/extract_frames.py \
  --video_path input/video.mp4 \
  --output_dir output/frames
```

**Output**:

* Extracted frames: `output/frames/frame_0000.jpg`, `frame_0001.jpg`, ...
* Camera intrinsics saved in the same directory

### 2. Get Camera Intrinsics

**Script**: `scripts/intrinsic.py`
**Function**: Estimates camera intrinsics for the extracted frames.

```bash
cd third_party/vggt
python ../../scripts/intrinsic.py --frame_dir ../../output/frames
```

### 3. Segment Objects Using SegFormer

**Script**: `scripts/segment.py`
**Function**: Segments frames and generates masks for the specified object class.

```bash
python scripts/segment.py \
  --frame_dir output/frames \
  --mask_dir output/masks_npz \
  --masked_frames_dir output/masked_frames \
  --label bicycle
```

**Supported labels**:
`road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle`

**Outputs**:

* Segmentation masks: `output/masks_npz/mask_frame_0000.npz`
* Masked images: `output/masked_frames/frame_0000.png`

### 4. Run VGGT for 3D Reconstruction

**Script**: `scripts/run_vggt.py`
**Function**: Generates depth maps and point clouds.

```bash
cd third_party/vggt
python ../../scripts/run_vggt.py \
  --frame_dir ../../output/frames \
  --output_dir ../../output/vggt_results \
  --batch_size 2 \
  --sample_rate 1
```

**Outputs**:

* Depth maps: `depth_frame_0000.npy`
* Point clouds: `points_frame_0000.npy`

### 5. Remove Objects and Inpaint

**Script**: `scripts/remove_objects.py`
**Function**: Uses VGGT depth and masks to inpaint masked regions.

```bash
python scripts/remove_objects.py \
  --frame_dir output/frames \
  --mask_dir output/masks_npz \
  --output_dir output/processed_frames \
  --vggt_dir output/vggt_results
```

**Alternative (continuous inpainting)**
**Script**: `scripts/remove_objects_continuous.py`
Feeds the previous output frame into the next step for temporal consistency.

```bash
python scripts/remove_objects_continuous.py \
  --frame_dir output/frames \
  --mask_dir output/masks_npz \
  --output_dir output/processed_frames_continue \
  --vggt_dir output/vggt_results
```

**Output**: Inpainted frames stored in `output/processed_frames` or `output/processed_frames_continue`

*Note: This uses OpenCV’s `INPAINT_TELEA`. For better results, consider advanced methods like [LaMa](https://github.com/advimman/lama).*

### 6. Reconstruct the Video

**Script**: `scripts/reconstruct_video.py`
**Function**: Converts processed frames back into a video.

```bash
python scripts/reconstruct_video.py \
  --frame_dir output/processed_frames \
  --output_video output_video.mp4 \
  --fps 30
```

**Output**: `output_video.mp4` (final inpainted video)

## Additional Notes

* **Paths**: Adjust file paths based on your environment and directory structure.
* **Virtual Environment**: Ensure your virtual environment is activated before running scripts.
* **VGGT**: Requires a GPU and automatically downloads pretrained weights.
* **Evaluation Metrics**: The project evaluates inpainting quality using PSNR, LPIPS, and FID, and visualizes 3D results.