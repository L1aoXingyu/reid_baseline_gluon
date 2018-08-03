#!/usr/bin/env bash

python3 main_reid.py train --save_dir='/home/test2/mxnet-ckpt/market1501_softmax_triplet' \
--max_epoch=400 --eval_step=10 --model_name='softmax_triplet'
