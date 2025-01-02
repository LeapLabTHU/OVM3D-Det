CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
  --config-file configs/Base_Omni3D_KITTI.yaml --dist-url tcp://0.0.0.0:12345 --num-gpus 1 \
    DATASETS.FOLDER_NAME "Omni3D_pl" \
    OUTPUT_DIR output/training/KITTI

