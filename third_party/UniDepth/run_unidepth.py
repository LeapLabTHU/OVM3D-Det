import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
torch.backends.cudnn.enabled = False

from PIL import Image
from tqdm import tqdm
import json
import argparse
from unidepth.models import UniDepthV1


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Configuration")
    parser.add_argument('--dataset', type=str, default='SUNRGBD', help='Name of the dataset')
    return parser.parse_args()

version="v1"
backbone="ViTL14"

model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def process(dataset):
    for mode in ['train', 'val']:
        with open(f'datasets/Omni3D/{dataset}_{mode}.json', 'r') as file:
            data = json.load(file)

        for i in tqdm(range(len(data['images']))):
            filename = data['images'][i]['file_path']
            rgb = torch.from_numpy(np.array(Image.open(f'datasets/{filename}'))).permute(2, 0, 1)
            intrinsics = np.array(data['images'][i]['K']).reshape(3,3)
            intrinsics = torch.from_numpy(intrinsics).float()
            file_name = data['images'][i]['id']

            predictions = model.infer(rgb, intrinsics)
            depth = predictions["depth"]
            intrinsics = predictions["intrinsics"]

            outdir = f'pseudo_label/{dataset}/{mode}/depth'
            os.makedirs(outdir, exist_ok=True)
            np.save(os.path.join(outdir, f"{file_name}"), depth.cpu().numpy().squeeze(0).squeeze(0))


if __name__ == "__main__":
    args = parse_args()
    print(f"Dataset name: {args.dataset}")
    process(args.dataset)
