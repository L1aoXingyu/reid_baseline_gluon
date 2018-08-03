#!/usr/bin/env bash

python3 main_reid.py train --save_dir='/home/test2/mxnet-ckpt/market1501_triplet' --max_epoch=400 \
--eval_step=50 --model_name='triplet'