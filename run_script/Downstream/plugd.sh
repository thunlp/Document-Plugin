#!/bin/bash
DATASET=$1
PlugD=$2
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=20089 \
    train.py -c config/Downstream/PlugD/${DATASET}PlugD.config \
    -g 0,1 \
    --checkpoint ${PlugD} \
    2>&1 | tee log/Downstream/${DATASET}-plugd-large.log
