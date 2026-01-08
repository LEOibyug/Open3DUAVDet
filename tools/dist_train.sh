#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4 nohup torchrun --nproc_per_node=5 train.py --launcher=pytorch > log.txt&

# CUDA_VISIBLE_DEVICES=3 python test.py