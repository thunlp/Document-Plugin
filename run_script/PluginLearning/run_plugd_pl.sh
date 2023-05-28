#!/bin/sh

python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=20086 \
    train.py \
    -c config/PL/PlugDL-c4.config \
    -g 0,1,2,3,4,5,6,7 \
    2>&1 | tee log/PL/PlugD.log
