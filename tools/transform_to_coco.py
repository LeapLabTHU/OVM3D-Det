import json
import torch
import numpy as np
from tqdm import tqdm
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Configuration")
    parser.add_argument('--dataset_name', type=str, default='SUNRGBD', help='Name of the dataset')
    return parser.parse_args()


def main(dataset_name):
    mode_list = ['train', 'val']

    # Class mapping for each dataset
    thing_classes_dict = {"KITTI": {'pedestrian': 0, 'car': 1, 'dontcare': 2, 'cyclist': 3, 'van': 4, 'truck': 5, 'tram': 6, 'person': 7},
                        "nuScenes": {'truck': 5, 'pedestrian': 0, 'traffic cone': 8, 'barrier': 9, 'car': 1, 'motorcycle': 10, 'bicycle': 11, 'bus': 12, 'trailer': 13},
                        "ARKitScenes": {'bed': 39, 'table': 37, 'chair': 18, 'fireplace': 97, 'machine': 61, 'cabinet': 29, 'oven': 57, 'shelves': 26, 'sink': 28, 'stove': 53, 'bathtub': 30, 'toilet': 32, 'sofa': 43, 'television': 44, 'refrigerator': 49},
                        "SUNRGBD": {'chair': 18, 'door': 31, 'table': 37, 'shelves': 26, 'kitchen pan': 51, 'bin': 52, 'counter': 38, 'cabinet': 29, 'stove': 53, 'sink': 28, 'books': 14, 'refrigerator': 49, 'microwave': 54, 'bottle': 15, 'plates': 55, 'bowl': 56, 'oven': 57, 'vase': 58, 'faucet': 59, 'towel': 22, 'tissues': 60, 'machine': 61, 'printer': 62, 'desk': 33, 'monitor': 63, 'podium': 64, 'bookcase': 35, 'dresser': 41, 'cart': 65, 'projector': 66, 'electronics': 67, 'computer': 68, 'box': 34, 'picture': 36, 'laptop': 20, 'pillow': 42, 'bed': 39, 'air conditioner': 69, 'lamp': 25, 'night stand': 40, 'board': 50, 'sofa': 43, 'coffee maker': 71, 'toaster': 72, 'potted plant': 73, 'stationery': 48, 'painting': 74, 'bag': 75, 'tray': 76, 'cup': 19, 'drawers': 70, 'keyboard': 77, 'shoes': 21, 'bicycle': 11, 'blanket': 78, 'television': 44, 'rack': 79, 'mirror': 27, 'clothes': 47, 'phone': 80, 'mouse': 81, 'person': 7, 'fire extinguisher': 82, 'toys': 83, 'ladder': 84, 'fan': 85, 'toilet': 32, 'bathtub': 30, 'glass': 86, 'clock': 87, 'toilet paper': 88, 'closet': 89, 'curtain': 46, 'window': 24, 'fume hood': 90, 'utensils': 91, 'floor mat': 45, 'soundsystem': 92, 'fire place': 93, 'shower curtain': 94, 'blinds': 23, 'remote': 95, 'pen': 96}}
    thing_classes = thing_classes_dict[dataset_name]

    for mode in mode_list:
        # Load dataset and info from the JSON file
        file_path = f'datasets/Omni3D/{dataset_name}_{mode}.json'
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Load 3D information for the images
        input_folder = f'pseudo_label/{dataset_name}/{mode}'
        info = torch.load(f'{input_folder}/info_3d.pth')

        dataset_id = data['info']['id']
        annotations = []
        num = 1

        for i in tqdm(range(len(data['images']))):
            im_id = data['images'][i]['id']
            if bool(info[im_id]):
                cat = info[im_id]["phrases"]
                score = info[im_id]["conf"]
                bbox = info[im_id]["boxes"]

                # If 3D information is available, create the annotation for each object
                if "boxes3d" in info[im_id] and len(cat) == len(info[im_id]["boxes3d"]):
                    bbox3D = info[im_id]["boxes3d"]
                    center_cam = info[im_id]["center_cam"]
                    dimension = info[im_id]["dimensions"]
                    R_cam = info[im_id]["R_cam"]
                    
                    for j in range(len(cat)):
                        obj = {}
                        obj['id'] = dataset_id * 10000000 + num
                        obj['image_id'] = im_id
                        obj['dataset_id'] = dataset_id
                        obj['category_name'] = cat[j]
                        obj['category_id'] = thing_classes[cat[j]]

                        obj['valid3D'] = True
                        obj['bbox2D_tight'] = [-1,-1,-1,-1]
                        obj['bbox2D_trunc'] = [-1,-1,-1,-1]
                        obj['bbox2D_proj'] = bbox[j].tolist()
                        obj['bbox3D_cam'] = bbox3D[j].tolist()
                        obj['center_cam'] = center_cam[j].tolist()
                        obj['dimensions'] = [float(x) for x in dimension[j]]
                        obj['R_cam'] = R_cam[j].tolist()

                        obj['behind_camera'] = False
                        obj['visibility'] = 1.0
                        obj['truncation'] = 0.0
                        obj['segmentation_pts'] = -1
                        obj['lidar_pts'] = -1
                        obj['depth_error'] = -1

                        obj['score'] = float(score[j])

                        annotations.append(obj)
                        num += 1

                else:
                    # If no 3D information is available, create a dummy object (e.g., a car)
                    obj = {}
                    obj['id'] = dataset_id * 10000000 + num
                    obj['image_id'] = im_id
                    obj['dataset_id'] = dataset_id
                    obj['category_name'] = 'car'
                    obj['category_id'] = 0

                    obj['valid3D'] = False
                    obj['bbox2D_tight'] = [-1,-1,-1,-1]
                    obj['bbox2D_trunc'] = [-1,-1,-1,-1]
                    obj['bbox2D_proj'] = [-1,-1,-1,-1]
                    obj['bbox3D_cam'] = [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.5, 1.5, 0.5], [0.5, 1.5, 0.5], [0.5, 0.5, 1.5], [1.5, 0.5, 1.5], [1.5, 1.5, 1.5], [0.5, 1.5, 1.5]]
                    obj['center_cam'] = [-1,-1,-1]
                    obj['dimensions'] = [-1,-1,-1]
                    obj['R_cam'] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

                    obj['behind_camera'] = False
                    obj['visibility'] = 1.0
                    obj['truncation'] = 0.0
                    obj['segmentation_pts'] = -1
                    obj['lidar_pts'] = -1
                    obj['depth_error'] = -1

                    annotations.append(obj)
                    num += 1

        # Update the JSON file with the pseudo-labels
        new_data = {'info': data['info'], 'images': data['images'], 'categories': data['categories'], 'annotations': annotations}

        os.makedirs("datasets/Omni3D_pl", exist_ok=True)
        new_file_path = f'datasets/Omni3D_pl/{dataset_name}_{mode}.json'
        with open(new_file_path, 'w') as new_file:
            json.dump(new_data, new_file)

if __name__ == "__main__":
    args = parse_args()
    print(f"Dataset name: {args.dataset_name}")
    main(args.dataset_name)
