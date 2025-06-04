#!/bin/zsh
set -e
VIDEO_PATH="input/video.mp4"
LABEL="bicycle"
FPS=30
CUDA_DEVICE=0
BATCH_SIZE=2
SAMPLE_RATE=1
FRAME_DIR="output/frames"
MASK_DIR="output/masks_npz"
MASKED_FRAMES_DIR="output/masked_frames"
VGGT_DIR="output/vggt_results"
VGGT_METRICS_DIR="output/vggt_metrics"
PROCESSED_FRAMES_DIR="output/processed_frames"
MODEL_DIR="models/vggt"
OUTPUT_VIDEO="output_video.mp4"
usage() {
    echo "Usage: $0 [--video_path <path>] [--label <object>] [--fps <fps>] [--cuda_device <device>] [--batch_size <size>] [--sample_rate <rate>]"
    echo "  --video_path: Path to input video (default: $VIDEO_PATH)"
    echo "  --label: Object to remove (default: $LABEL). Supported: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle"
    echo "  --fps: Frames per second for output video (default: $FPS)"
    echo "  --cuda_device: CUDA device ID (default: $CUDA_DEVICE)"
    echo "  --batch_size: Batch size for VGGT processing (default: $BATCH_SIZE)"
    echo "  --sample_rate: Sample every nth frame for VGGT (default: $SAMPLE_RATE)"
    exit 1
}
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video_path) VIDEO_PATH="$2"; shift ;;
        --label) LABEL="$2"; shift ;;
        --fps) FPS="$2"; shift ;;
        --cuda_device) CUDA_DEVICE="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --sample_rate) SAMPLE_RATE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done
if [ ! -f "$VIDEO_PATH" ]; then
    echo "[ERROR] Input video not found: $VIDEO_PATH"
    exit 1
fi
echo "[INFO] Starting video object removal pipeline for label: $LABEL"
echo "[INFO] Activating streetunveiler environment..."
conda activate streetunveiler
echo "[INFO] Extracting frames from $VIDEO_PATH..."
python scripts/extract_frames.py --video_path "$VIDEO_PATH" --output_dir "$FRAME_DIR"
echo "[INFO] Segmenting objects ($LABEL)..."
python scripts/segment.py --frame_dir "$FRAME_DIR" --mask_dir "$MASK_DIR" --label "$LABEL"
echo "[INFO] Masking frames for label $LABEL..."
python scripts/mask_frames.py --frame_dir "$FRAME_DIR" --mask_dir "$MASK_DIR" --label "$LABEL" --output_dir "$MASKED_FRAMES_DIR"
echo "[INFO] Activating VGGT environment..."
source ../VGGTenv/bin/activate
echo "[INFO] Running VGGT for 3D reconstruction..."
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python scripts/run_vggt.py --frame_dir "$FRAME_DIR" --model_dir "$MODEL_DIR" --batch_size "$BATCH_SIZE" --sample_rate "$SAMPLE_RATE" --output_dir "$VGGT_DIR"
echo "[INFO] Visualizing VGGT results and computing metrics..."
python scripts/visualize_vggt_metrics.py --vggt_dir "$VGGT_DIR" --frame_dir "$FRAME_DIR" --processed_frames_dir "$PROCESSED_FRAMES_DIR" --output_dir "$VGGT_METRICS_DIR"
echo "[INFO] Activating streetunveiler environment..."
conda activate streetunveiler
echo "[INFO] Removing objects and inpainting..."
python scripts/remove_objects.py --frame_dir "$FRAME_DIR" --mask_dir "$MASK_DIR" --output_dir "$PROCESSED_FRAMES_DIR" --vggt_dir "$VGGT_DIR" --label "$LABEL"
echo "[INFO] Reconstructing video..."
python scripts/reconstruct_video.py --frame_dir "$PROCESSED_FRAMES_DIR" --output_video "$OUTPUT_VIDEO" --fps "$FPS"
echo "[INFO] Pipeline completed successfully. Output video saved to: $OUTPUT_VIDEO"