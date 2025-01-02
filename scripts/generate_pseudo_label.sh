#!/bin/bash

DATASET=$1

# Step 1: Predict depth using UniDepth
CUDA_VISIBLE_DEVICES=0 python third_party/UniDepth/run_unidepth.py --dataset $DATASET

# Step 2: Segment novel objects using Grounded-SAM
CUDA_VISIBLE_DEVICES=0 python third_party/Grounded-Segment-Anything/grounded_sam_detect.py --dataset $DATASET
CUDA_VISIBLE_DEVICES=0 python third_party/Grounded-Segment-Anything/grounded_sam_detect_ground.py --dataset $DATASET

# Step 3: Generate pseudo 3D bounding boxes
python tools/generate_pseudo_bbox.py \
  --config-file configs/Base_Omni3D_${DATASET}.yaml \
  OUTPUT_DIR output/generate_pseudo_label/$DATASET \

# Step 4: Convert to COCO dataset format
python tools/transform_to_coco.py --dataset_name $DATASET