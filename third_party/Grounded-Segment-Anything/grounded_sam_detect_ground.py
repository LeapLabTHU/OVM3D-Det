import sys
import argparse
import os
import copy
import io

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO


from huggingface_hub import hf_hub_download

from tqdm import tqdm
import json


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   


def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        im.save(image_file_path)

    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))


def show_mask(mask, image, random_color=True):
    mask_image = np.zeros((mask.shape[2],mask.shape[3],4))
    for i in range(mask.shape[0]):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_i = mask[i][0]
        mask_image[mask_i] = color.reshape(1, 1, -1)
        
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def generate_substrings(input_string):
    input_list = input_string.split()
    result = []
    n = len(input_list)
    for i in range(n):
        for j in range(i, n):
            result.append(' '.join(input_list[i:j+1]))
    return result

# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename).cuda()

device = 'cuda'
sam_checkpoint = 'sam_vit_h_4b8939.pth'
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, help='dataset')
args = parser.parse_args()
dataset = args.dataset

# Determine the prompt based on whether the dataset is for outdoor or indoor environments.
# - For outdoor datasets, the prompt is set to "ground".
# - For indoor datasets, the prompt is set to "floor".
TEXT_PROMPT_DICT = {'KITTI': "ground",
                    'nuScenes': "ground",
                    'ARKitScenes': "floor",
                    'SUNRGBD': "floor"}

torch.backends.cudnn.enabled = False


for mode in ['train', 'val']:
    with open(f'datasets/Omni3D/{dataset}_{mode}.json', 'r') as file:
        data = json.load(file)
    info = {}
    no_box_list = []
    for k in tqdm(range(len(data['images']))):
        filename = data['images'][k]['file_path']
        file_name = data['images'][k]['id']

        TEXT_PROMPT = TEXT_PROMPT_DICT[dataset]
        BOX_TRESHOLD = 0.25
        TEXT_TRESHOLD = 0.15

        image_source, image = load_image(f'datasets/{filename}')

        boxes, logits, phrases, features, encoded_text, non_zero_idx_list, logits_all = predict(
            model=groundingdino_model, 
            image=image,
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD
        )
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB

        # If no box is detected, use an empty dict as a dummy
        if len(boxes) == 0:
            no_box_list.append(file_name)
            record = {}
            info[file_name] = record
            continue

        # set image
        sam_predictor.set_image(image_source)
        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        masks, _, _ = sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    multimask_output = False,
                )

        boxes_list = []
        conf_list = []
        phrases_list = []
        mask_list = []

        for i in range(len(phrases)):
            phrase = phrases[i]
            if phrase != ' ':
                conf = logits[i]
                substrings = generate_substrings(phrase)
                substrings = [x for x in substrings if x in TEXT_PROMPT.split('. ')]
                for substring in substrings:
                    conf_list.append(conf.cpu().numpy())
                    phrases_list.append(substring)
                    boxes_list.append(boxes_xyxy[i].cpu().numpy())
                    mask_list.append(masks[i].cpu().numpy().astype(bool))

        if len(boxes_list) == 0:
            no_box_list.append(file_name)
            record = {}
            info[file_name] = record
            continue

        record = {}
        record['boxes'] = np.array(boxes_list)
        record['conf'] = np.array(conf_list)
        record['phrases'] = phrases_list
        info[file_name] = record

        outdir = f'pseudo_label/{dataset}/{mode}/ground_mask'
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, f"{file_name}"), np.array(mask_list))
        # annotated_frame_with_mask = show_mask(masks.cpu(), annotated_frame)
        # ground_sam_image = Image.fromarray(annotated_frame_with_mask)
        # ground_sam_image.save(f"{outdir}/{file_name}_ground.png") 

    torch.save(info, f'pseudo_label/{dataset}/{mode}/info_ground.pth')