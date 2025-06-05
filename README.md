# Free NTU of Bicycle ver2

## Overview

This project integrates **SegFormer** for semantic segmentation and **VGGT** for 3D scene reconstruction to automatically remove specified objects (e.g., bicycles) from a video. The pipeline includes:

1. Extracting frames from the video
2. Segmenting and masking target objects
3. Performing 3D scene reconstruction
4. Inpainting masked regions using depth information
5. Computing quality metrics
6. Reconstructing the final output video

An alternative pipeline without VGGT is also provided for continuous inpainting.

## Project Structure

```
Free-NTU-of-Bicycle-ver2/
├── third_party/
│   ├── vggt/                        # VGGT 3D reconstruction module
│   └── DiffuEraser/                 # DiffuEraser
├── input/
│   └── video.mp4                    # Input video
├── output/
│   ├── frames/                      # Extracted frames and camera intrinsics
│   ├── masks_npz/                   # Segmentation masks (NPZ format)
│   ├── masked_frames/               # Masked frames (objects blacked out)
│   ├── vggt_results/                # VGGT depth maps and point clouds
│   ├── processed_frames/            # Inpainted frames with VGGT
│   └── processed_frames_without_vggt/ # Inpainted frames without VGGT
├── scripts/
│   ├── extract_frames.py            # Frame extraction with progress bar
│   ├── segment.py                   # SegFormer-based object segmentation
│   ├── mask_frames.py               # Object masking utility
│   ├── run_vggt.py                  # VGGT 3D reconstruction
│   ├── visualize_vggt_metrics.py    # Metric visualization and evaluation
│   ├── remove_objects.py            # Inpainting using segmentation and depth
│   ├── remove_objects_without_vggt.py # Continuous inpainting without VGGT
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
source VGGTenv/bin/activate  # On Windows: VGGTenv\Scripts\activate
pip install -r requirements.txt
```

Place your input video (e.g., `video.mp4`) in the `input/` directory.

## Step-by-Step Guide (segformer + vggt + Stable Diffusion)

### 1. Extract Video Frames

**Script**: `scripts/extract_frames.py`  
**Function**: Extracts frames from the input video.

```bash
python scripts/extract_frames.py \
  --video_path input/video.mp4 \
  --output_dir output/frames
```

**Output**:  

- Extracted frames: `output/frames/frame_0000.jpg`, `frame_0001.jpg`, ...  
- Camera intrinsics: `output/frames/intrinsics.json`

### 2. Get Camera Intrinsics

**Script**: `scripts/intrinsic.py`  
**Function**: Estimates camera intrinsics for the extracted frames.

```bash
cd third_party/vggt
python ../../scripts/intrinsic.py --frame_dir ../../output/frames
```

**Output**: Updates `output/frames/intrinsics.json`

### 3. Segment Objects Using SegFormer

**Script**: `scripts/segment.py`  
**Function**: Segments frames and generates masks for the specified object class.

```bash
python scripts/segment.py   --frame_dir output/frames   --mask_dir output/masks_npz   --masked_frames_dir output/mask_frames   --mask_output_dir output/binary_mask   --label bicycle
```

**Supported labels**:  
`road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle`

**Outputs**:  

- Segmentation masks: `output/masks_npz/mask_frame_0000.npz`, ...  
- Masked images: `output/masked_frames/frame_0000.png`, ...
- binary mask: `output/binary_mask/mask_frame_0000.png`

### 4. Run VGGT for 3D Reconstruction (optional)

**Script**: `scripts/run_vggt.py`  
**Function**: Generates depth maps and point clouds for 3D reconstruction.

```bash
cd third_party/vggt
python ../../scripts/run_vggt.py \
  --frame_dir ../../output/frames \
  --output_dir ../../output/vggt_results \
  --batch_size 2 \
  --sample_rate 1
```

**Outputs**:  

- Depth maps: `output/vggt_results/depth_frame_0000.npy`, ...  
- Point clouds: `output/vggt_results/points_frame_0000.npy`, ...  
- Processed frames list: `output/vggt_results/processed_frames.txt`

### 5. Remove Objects with vggt (optional)

**Script**: `scripts/remove_objects.py`  
**Function**: Uses VGGT depth maps and point clouds along with segmentation masks to inpaint masked regions.

```bash
python scripts/remove_objects.py   --frame_dir output/frames   --mask_dir output/masks_npz   --mask_output_dir output/binary_mask_with_vggt --render_output_dir output/mask_frames_with_vggt   --vggt_dir output/vggt_results
```

**Outputs**:  

- Masked images: `output/mask_frames_with_vggt/frame_0000.png`, ...
- binary mask: `output/binary_mask_with_vggt/mask_frame_0000.png`

### 5. Inpaint

```bash
python scripts/inpainting.py   --frame_dir output/frames   --mask_dir output/binary_mask_with_vggt/   --output_dir output/processed_frame --model runwayml/stable-diffusion-inpainting

```

**Arguments**:  
 
- `--model`: Selects the inpainting model. Options:  
  - `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` (default)  
  - `runwayml/stable-diffusion-inpainting`

**Output**:  

- Inpainted frames: `output/processed_frames/frame_0000.jpg`, ...  

### 6. Reconstruct the Video

**Script**: `scripts/reconstruct_video.py`  
**Function**: Combines inpainted frames into a final video.

```bash
python scripts/reconstruct_video.py \
  --frame_dir output/processed_frames \
  --output_video output_video.mp4 \
  --fps 30
```

If you want to generate binary mask video (for DiffuEraser or other video inpainting), add an argument `--mask_mode`

**Output**:  

- Final video: `output_video.mp4`

**Note**: Use `--frame_dir output/processed_frames_without_vggt` for the continuous inpainting output.

## Method 2. segformer + vggt + DiffuEraser

## Additional Notes

- **Paths**: Adjust file paths if your directory structure differs.  
- **Virtual Environment**: Activate the virtual environment (`source VGGTenv/bin/activate`) before running scripts.  
- **VGGT**: Requires a GPU and automatically downloads pretrained weights.  
- **Inpainting Models**: Both models use `torch.float16` for efficiency. Ensure a compatible GPU is available.  
- **Evaluation Metrics**: Inpainting quality is evaluated using PSNR, LPIPS, and FID, with visualizations saved in `output/vggt_metrics/`.  
- **Debug Mode**: Use `--debug` to save intermediate outputs for troubleshooting, but omit it for faster processing and minimal storage use.