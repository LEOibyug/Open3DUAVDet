#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup torchrun --nproc_per_node=6 test.py --launcher=pytorch > log_t.txt&

# CUDA_VISIBLE_DEVICES=3 python test.py