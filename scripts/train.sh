DATASET=$1

CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py \
  --config-file configs/Base_Omni3D_$DATASET.yaml --dist-url tcp://0.0.0.0:12345 --num-gpus 2 \
    DATASETS.FOLDER_NAME "Omni3D_pl" \
    OUTPUT_DIR output/training/$DATASET

