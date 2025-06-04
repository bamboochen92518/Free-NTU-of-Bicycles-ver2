# Video Object Removal Project

## Overview

This project integrates **SegFormer** for semantic segmentation and **VGGT** for 3D scene reconstruction to remove specified objects (e.g., bicycles) from a video. The full pipeline includes:

1. Extracting frames from the video
2. Segmenting and masking specified objects
3. Performing 3D scene reconstruction
4. Inpainting masked regions
5. Computing quality metrics
6. Reconstructing the final video

## Project Structure

```
video-object-removal/
├── third_party/
│   └── vggt/                        # VGGT 3D reconstruction module
├── input/
│   └── video.mp4                    # Input video
├── output/
│   ├── frames/                      # Extracted frames
│   ├── masks_npz/                   # Segmentation masks (NPZ format)
│   ├── masked_frames/              # Masked frames (objects blacked out)
│   ├── vggt_results/                # VGGT depth maps and point clouds
│   ├── vggt_metrics/                # Visualizations and evaluation metrics
│   └── processed_frames/            # Inpainted frames
├── scripts/
│   ├── extract_frames.py            # Frame extraction with progress bar
│   ├── segment.py                   # SegFormer-based object segmentation
│   ├── mask_frames.py               # Object masking utility
│   ├── run_vggt.py                  # VGGT 3D reconstruction
│   ├── visualize_vggt_metrics.py    # Metric visualization and evaluation
│   ├── remove_objects.py            # Inpainting based on segmentation and depth
│   ├── reconstruct_video.py         # Frame-to-video reconstruction
│   └── run_pipeline.sh              # Full pipeline execution script
├── requirements.txt                 # Python dependencies
├── output_video.mp4                 # Final output video
└── README.md                        # Project documentation
```

## Setup Instructions

### 1. Environment Setup

Clone the repository:

```bash
git clone <your-repo-url>
cd Free-NTU-of-Bicycles-ver2
```

Install dependencies in a virtual environment:

```bash
python -m venv VGGTenv
source VGGTenv/bin/activate
pip install -r requirements.txt
```

Set up two environments:

* **`streetunveiler`**: For most scripts
* **`VGGTenv`**: For running VGGT-related scripts

Place your input video (e.g., `video.mp4`) into the `input/` directory.

## Step-by-Step Guide

### 2. Extract Video Frames

**Script**: `scripts/extract_frames.py`
**Function**: Extracts frames from video with progress tracking.

```bash
python scripts/extract_frames.py --video_path input/video.mp4 --output_dir output/frames
```

**Output**: `output/frames/frame_0000.jpg`, `frame_0001.jpg`, ...

### 3. Segment Objects Using SegFormer

**Script**: `scripts/segment.py`
**Function**: Segments frames to identify target objects.

```bash
python scripts/segment.py \
  --frame_dir output/frames \
  --mask_dir output/masks_npz \
  --masked_frames_dir output/masked_frames \
  --label bicycle
```

**Supported labels** include:
`road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle`

**Output**:

* Masks in `.npz` format: `output/masks_npz/mask_frame_0000.npz`
* Masked images: `output/masked_frames/frame_0000.png`

### 4. Run VGGT for 3D Reconstruction

**Script**: `scripts/run_vggt.py`
**Function**: Generates depth maps and 3D point clouds.

```bash
cd third_party/vggt/
python ../../scripts/run_vggt.py \
  --frame_dir ../../output/frames \
  --output_dir ../../output/vggt_results \
  --batch_size 2 \
  --sample_rate 1
```

**Output**:

* Depth: `depth_frame_0000.npy`
* Point cloud: `points_frame_0000.npy`

### 5. Remove Objects and Inpaint

**Script**: `scripts/remove_objects.py`
**Function**: Uses masks and VGGT depth to inpaint masked areas.

```bash
conda activate streetunveiler
python scripts/remove_objects.py \
  --frame_dir output/frames \
  --mask_dir output/masks_npz \
  --output_dir output/processed_frames \
  --vggt_dir output/vggt_results \
  --label bicycle
```

**Output**:
Inpainted frames in `output/processed_frames/frame_0000.jpg`, etc.

> *Note*: Uses OpenCV's `INPAINT_TELEA`. For improved quality, consider advanced inpainting methods like [LaMa](https://github.com/advimman/lama).

### 6. Reconstruct the Video

**Script**: `scripts/reconstruct_video.py`
**Function**: Combines inpainted frames into a video.

```bash
conda activate streetunveiler
python scripts/reconstruct_video.py \
  --frame_dir output/processed_frames \
  --output_video output_video.mp4 \
  --fps 30
```

**Output**: Final video saved as `output_video.mp4`

### 7. Run the Full Pipeline (Optional)

**Script**: `scripts/run_pipeline.sh`
**Function**: Automates the full workflow.

```bash
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh --video_path input/video.mp4 --label bicycle
```

**Options**:

* `--fps`: Frame rate (default 30)
* `--cuda_device`: GPU device index
* `--batch_size`: VGGT batch size
* `--sample_rate`: Sampling frequency for VGGT

## Additional Notes

* **Paths**: Update paths according to your project structure.
* **Environments**: Ensure correct Conda/venv activation during each step.
* **VGGT**: Requires GPU acceleration. Automatically downloads the pretrained model.
* **Metrics**: Evaluated using PSNR, LPIPS, and FID for inpainting quality. Includes depth and point cloud visualizations.
* **SegFormer Masks**: Stored as `.npz` for compatibility with inpainting modules.
