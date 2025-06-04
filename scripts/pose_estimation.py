import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import json
from scipy.spatial.transform import Rotation as R
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process images for camera pose estimation')
parser.add_argument('--input-dir', type=str, required=True, help='Directory containing input images')
args = parser.parse_args()

# Get list of image files from directory
image_dir = args.input_dir
image_extensions = ('.jpg', '.jpeg', '.png')
image_names = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if os.path.isfile(os.path.join(image_dir, f)) 
               and f.lower().endswith(image_extensions)]
image_names = sorted(image_names)  # Ensure consistent order

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize model
model = torch.compile(VGGT.from_pretrained("facebook/VGGT-1B")).to(device)
model.eval()

# Define dataset for parallel loading
class ImageDataset(Dataset):
    def __init__(self, frame_paths):
        self.frame_paths = frame_paths
        self.transform = transforms.Compose([
            transforms.Resize((294, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        img = Image.open(self.frame_paths[idx]).convert('RGB')
        return self.transform(img)

intrinsic = []
extrinsic = []
current_image_names = []

for i, image_path in enumerate(image_names):
    current_image_names.append(image_path)
    
    if (i + 1) % 20 == 0 or (i + 1) == len(image_names):
        dataset = ImageDataset(current_image_names)
        images = next(iter(DataLoader(dataset, batch_size=len(current_image_names), num_workers=4, pin_memory=True))).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = model.aggregator(images)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extri, intri = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                extri, intri = extri.squeeze(0), intri.squeeze(0)

        for idx, e in enumerate(extri):
            rotation_matrix = e[:, :3]
            translation = e[:, 3]
            rot = R.from_matrix(rotation_matrix.cpu().numpy())
            quat = rot.as_quat()
            e_result = {
                "image_name": os.path.basename(current_image_names[idx]),  # Store image name for mapping
                "position": {
                    "x": float(translation[0]),
                    "y": float(translation[1]),
                    "z": float(translation[2]),
                },
                "heading": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3]),
                }
            }
            extrinsic.append(e_result)

        for idx, i_val in enumerate(intri):
            i_result = {
                "image_name": os.path.basename(current_image_names[idx]),
                "fx": float(i_val[0, 0]),
                "fy": float(i_val[1, 1]),
                "cx": float(i_val[0, 2]),
                "cy": float(i_val[1, 2])
            }
            intrinsic.append(i_result)

        current_image_names = []
        torch.cuda.empty_cache()

# Ensure output directory exists
os.makedirs(image_dir, exist_ok=True)

# Save results asynchronously
with ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(lambda: json.dump(extrinsic, open(os.path.join(image_dir, 'poses.json'), 'w'), indent=2))
    executor.submit(lambda: json.dump(intrinsic, open(os.path.join(image_dir, 'intrinsics.json'), 'w'), indent=2))