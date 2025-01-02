DATASET=$1

CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    --eval-only --config-file checkpoints/$DATASET/config.yaml --dist-url tcp://0.0.0.0:12345 --num-gpus 1 \
    MODEL.WEIGHTS checkpoints/$DATASET/model_recent.pth \
    OUTPUT_DIR output/test/$DATASET
