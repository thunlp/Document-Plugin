#!/bin/bash
DATASET=$1
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=20086 \
    train.py -c config/OpenQA/Adapter/${DATASET}Adapter.config \
    -g 0,1 \
    2>&1 | tee log/OpenQA/Adapter/${DATASET}-adapter-large.log
