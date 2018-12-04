#!/usr/bin/env bash

python train.py --epochs 10 \
                --net_name aod-xavier \
                --lr 1e-4 \
                --use_gpu true \
                --gpu 3 \
                --ori_data_path /home/wc/data/Haze/train/ori/ \
                --haze_data_path /home/wc/data/Haze/train/haze/ \
                --val_ori_data_path /home/wc/data/Haze/val/ori/ \
                --val_haze_data_path /home/wc/data/Haze/val/haze/ \
                --num_workers 2 \
                --batch_size 8 \
                --val_batch_size 16 \
                --print_gap 500 \
                --model_dir /home/wc/workspace/AOD-Net.pytorch/models \
                --log_dir /home/wc/workspace/AOD-Net.pytorch/logs \
                --sample_output_folder /home/wc/workspace/AOD-Net.pytorch/samples