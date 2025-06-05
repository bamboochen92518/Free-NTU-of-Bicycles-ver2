# Free NTU of Bicycle ver2

## Overview

This project integrates **SegFormer** for semantic segmentation and **VGGT** for 3D scene reconstruction to automatically remove specified objects (e.g., bicycles) from videos. The pipeline includes:

1. Extracting frames from the input video
2. Segmenting and masking target objects
3. Performing 3D scene reconstruction
4. Inpainting masked regions using depth information
5. Computing quality metrics
6. Reconstructing the final output video

An alternative pipeline without VGGT is also provided for continuous inpainting.

## Project Structure

```
Free-NTU-of-Bicycle-ver2/
├── input/
│   └── video.mp4                         # Input video
├── output/
│   ├── binary_mask/                      # Binary mask (jpg)
│   ├── binary_mask_with_vggt/            # Binary mask after VGGT removes point cloud and renders
│   ├── frames/                           # Extracted frames and camera intrinsics
│   ├── mask_frames/                      # Masked frames (objects blacked out)
│   ├── mask_frames_with_vggt/            # Masked frames with VGGT
│   ├── masks_npz/                        # Segmentation masks (NPZ format)
│   ├── processed_frames/                 # Inpainted frames with VGGT
│   ├── vggt_results/                     # VGGT depth maps and point clouds
├── scripts/
│   ├── extract_frames.py                 # Frame extraction with progress bar
│   ├── inpainting.py                     # Inpainting script
│   ├── intrinsic.py                      # Camera intrinsics estimation
│   ├── mask_frames.py                    # Object masking utility
│   ├── reconstruct_video.py              # Frame-to-video reconstruction
│   ├── remove_objects.py                 # Apply SegFormer mask on point cloud and render
│   ├── run_vggt.py                       # VGGT 3D reconstruction
│   ├── segment.py                        # SegFormer-based object segmentation
│   ├── utils.py                          # Utility functions
│   └── visualize_vggt_metrics.py         # Metric visualization and evaluation
├── third_party/
│   ├── DiffuEraser/                      # DiffuEraser
│   └── vggt/                             # VGGT 3D reconstruction module
├── output_video.mp4                      # Final output video
├── README.md                             # Project documentation
└── requirements.txt                      # Python dependencies
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
source VGGTenv/bin/activate  # On Windows: VGGTenv\Scripts\activate
pip install -r requirements.txt
```

Place your input video (e.g., `video.mp4`) in the `input/` directory.

## Step-by-Step Guide (SegFormer + VGGT + Stable Diffusion)

### 1. Extract Video Frames

**Script**: `scripts/extract_frames.py`
**Description**: Extracts frames from the input video.

```bash
python scripts/extract_frames.py \
  --video_path input/video.mp4 \
  --output_dir output/frames
```

**Output**:

* Frames: `output/frames/frame_0000.jpg`, `frame_0001.jpg`, ...
* Camera intrinsics: `output/frames/intrinsics.json`

### 2. Estimate Camera Intrinsics

**Script**: `scripts/intrinsic.py`
**Description**: Estimates camera intrinsics for the extracted frames.

```bash
cd third_party/vggt
python ../../scripts/intrinsic.py --frame_dir ../../output/frames
```

**Output**: Updates `output/frames/intrinsics.json`

### 3. Segment Objects Using SegFormer

**Script**: `scripts/segment.py`
**Description**: Segments frames and generates masks for the specified object class.

```bash
python scripts/segment.py \
  --frame_dir output/frames \
  --mask_dir output/masks_npz \
  --masked_frames_dir output/mask_frames \
  --mask_output_dir output/binary_mask \
  --label bicycle
```

**Supported labels**:
`road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle`

**Outputs**:

* Segmentation masks: `output/masks_npz/mask_frame_0000.npz`, ...
* Masked images: `output/mask_frames/frame_0000.png`, ...
* Binary masks: `output/binary_mask/mask_frame_0000.png`

### 4. Run VGGT for 3D Reconstruction (Optional)

**Script**: `scripts/run_vggt.py`
**Description**: Generates depth maps and point clouds for 3D reconstruction.

```bash
cd third_party/vggt
python ../../scripts/run_vggt.py \
  --frame_dir ../../output/frames \
  --output_dir ../../output/vggt_results \
  --batch_size 2 \
  --sample_rate 1
```

**Outputs**:

* Depth maps: `output/vggt_results/depth_frame_0000.npy`, ...
* Point clouds: `output/vggt_results/points_frame_0000.npy`, ...
* Processed frame list: `output/vggt_results/processed_frames.txt`

### 5. Remove Objects with VGGT (Optional)

**Script**: `scripts/remove_objects.py`
**Description**: Applies VGGT depth maps and point clouds along with segmentation masks to refine object removal.

```bash
python scripts/remove_objects.py \
  --frame_dir output/frames \
  --mask_dir output/masks_npz \
  --mask_output_dir output/binary_mask_with_vggt \
  --render_output_dir output/mask_frames_with_vggt \
  --vggt_dir output/vggt_results
```

**Outputs**:

* Masked frames: `output/mask_frames_with_vggt/frame_0000.png`, ...
* Binary masks: `output/binary_mask_with_vggt/mask_frame_0000.png`

### 6. Inpaint Masked Regions

**Script**: `scripts/inpainting.py`
**Description**: Inpaints masked regions using a Stable Diffusion model.

```bash
python scripts/inpainting.py \
  --frame_dir output/frames \
  --mask_dir output/binary_mask_with_vggt/ \
  --output_dir output/processed_frames \
  --model runwayml/stable-diffusion-inpainting
```

**Arguments**:

* `--model`: Choose the inpainting model:

  * `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (default)
  * `runwayml/stable-diffusion-inpainting`

**Output**:

* Inpainted frames: `output/processed_frames/frame_0000.jpg`, ...

### 7. Reconstruct the Final Video

**Script**: `scripts/reconstruct_video.py`
**Description**: Combines inpainted frames into a video.

```bash
python scripts/reconstruct_video.py \
  --frame_dir output/processed_frames \
  --output_video output_video.mp4 \
  --fps 30
```

To generate a binary mask video (e.g., for DiffuEraser), add the `--mask_mode` flag.

**Output**:

* Final video: `output_video.mp4`

## Alternative Pipeline: SegFormer + VGGT + DiffuEraser

This variant of the pipeline uses **DiffuEraser** instead of Stable Diffusion for video inpainting. Please refer to the `third_party/DiffuEraser` directory for implementation details and additional instructions.

### Setup Environment

First, create and activate a dedicated Anaconda environment for DiffuEraser:

```bash
cd third_party/DiffuEraser

# Create new conda environment
conda create -n diffueraser python=3.9.19  
conda activate diffueraser

# Install Python dependencies
pip install -r requirements.txt
```

### Install Pretrained Weights

Ensure Git LFS is installed:

```bash
conda install -c conda-forge git-lfs
git lfs install
```

Download the required weights:

```bash
cd weights

git clone https://huggingface.co/lixiaowen/diffuEraser
git clone https://huggingface.co/wangfuyun/PCM_Weights

mkdir ProPainter
cd ProPainter
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth
```

### Run DiffuEraser

1. Reconstruct the input and binary mask videos:

```bash
cd ../../..
python scripts/reconstruct_video.py --frame_dir output/frames --output_video input.mp4 --fps 30
python scripts/reconstruct_video.py --frame_dir output/binary_mask/ --output_video mask.mp4 --fps 30 --mask_mode
```

2. Run the inpainting process:

```bash
cd third_party/DiffuEraser
python run_diffueraser.py \
  --input_video ../../input.mp4 \
  --input_mask ../../mask.mp4 \
  --video_length 21 \
  --base_model_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --vae_path stabilityai/sd-vae-ft-mse \
  --max_img_size 640
```

### Output

The inpainted video will be saved at:

```bash
third_party/DiffuEraser/results/diffueraser_result.mp4
```

## Additional Notes

* **Paths**: Modify file paths as needed if your directory structure differs.
* **Virtual Environment**: Always activate the virtual environment (`source VGGTenv/bin/activate`) before running scripts.
* **VGGT**: Requires a GPU and will automatically download pretrained weights.
* **Inpainting Models**: All models use `torch.float16` for performance. Ensure your GPU supports it.
* **Evaluation Metrics**: Inpainting quality is evaluated using PSNR, LPIPS, and FID. Visualizations are saved in `output/vggt_metrics/`.
* **Debug Mode**: Add `--debug` to any script to save intermediate outputs for troubleshooting. Omit it for faster processing and reduced disk usage.
