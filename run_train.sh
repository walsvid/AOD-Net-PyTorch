#!/usr/bin/env bash

python train.py --epochs 10 \
                --net_name aod \
                --lr 1e-4 \
                --use_gpu true \
                --gpu 3 \
                --ori_data_path /home/wc/data/Haze/train/ori/ \
                --haze_data_path /home/wc/data/Haze/train/haze/ \
                --threads 2 \
                --batch_size 8 \
                --print_gap 500 \
                --model_dir /home/wc/workspace/AOD-Net.pytorch/models \
                --log_dir /home/wc/workspace/AOD-Net.pytorch/logs