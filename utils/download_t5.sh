#!/bin/bash
mkdir -p checkpoint/PLMs/t5-large
mkdir -p checkpoint/PLMs/t5-large/tokenizer
wget https://openbmb.oss-cn-hongkong.aliyuncs.com/model_center/t5-large/config.json -cP checkpoint/PLMs/t5-large
wget https://openbmb.oss-cn-hongkong.aliyuncs.com/model_center/t5-large/pytorch_model.pt -cP checkpoint/PLMs/t5-large
wget https://huggingface.co/t5-large/resolve/main/config.json -cP checkpoint/PLMs/t5-large/tokenizer
wget https://huggingface.co/t5-large/resolve/main/generation_config.json -cP checkpoint/PLMs/t5-large/tokenizer
wget https://huggingface.co/t5-large/resolve/main/tokenizer.json -cP checkpoint/PLMs/t5-large/tokenizer
wget https://huggingface.co/t5-large/resolve/main/spiece.model -cP checkpoint/PLMs/t5-large/tokenizer
