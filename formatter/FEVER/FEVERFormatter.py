import json
import torch
import os
import numpy as np

import random
from transformers import T5Tokenizer,T5Config

class FEVERFormatter:
    def __init__(self, config, mode, *args, **params):
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"), "tokenizer"))
        self.max_len = config.getint("train", "max_len")

        self.label2id = {
            "SUPPORTS": 0,
            "REFUTES": 1,
            # "NOT ENOUGH INFO": 2
        }

    def process(self, data):

        # claims = [d["claim"] for d in data]
        claims = [d["input"] for d in data]
        
        ret = self.tokenizer(claims, max_length=self.max_len, padding="max_length", truncation=True)

        labels = [self.label2id[d["output"][1]["answer"]] for d in data]
        ret["labels"] = labels
        for key in ret:
            ret[key] = torch.LongTensor(ret[key])
        return ret
