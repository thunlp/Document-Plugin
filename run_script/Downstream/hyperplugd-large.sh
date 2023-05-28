#!/bin/bash
DATASET=$1
CKP=$2
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=20089 \
    train.py -c config/Downstream/HyperPlugD/${DATASET}HyperPlugD.config \
    -g 0,1 \
    --checkpoint ${CKP} #\
    # 2>&1 | tee log/Downstream/plugd/${DATASET}-hyperplugd-large-${SFX}.log
