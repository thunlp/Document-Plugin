#!/bin/bash
DATASET=$1
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=20089 \
    train.py -c config/OpenQA/FineTune/${DATASET}FT.config \
    -g 0,1 \
    2>&1 | tee log/OpenQA/FineTune/${DATASET}-FT-large.log
