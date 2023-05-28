import json
import torch
import os
import numpy as np

import random
from transformers import T5Tokenizer, T5Config
from transformers.file_utils import is_torch_fx_proxy

class TextClassificationPlugDFormatter:
    def __init__(self, config, mode, *args, **params):
        self.ctx_len = config.getint("train", "ctx_len")
        self.mode = mode
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"), "tokenizer"))
        self.label2id = json.load(open(config.get("data", "label2id"), "r"))
        self.query = (" or ".join(list(self.label2id.keys())) + "?").lower()
    
    def process(self, data):
        ques = [self.query] * len(data)
        ctxs = [d["text"] for d in data]
        labels = [self.label2id[d["label"]] for d in data]

        question = self.tokenizer(ques)
        context = self.tokenizer(ctxs, max_length=self.ctx_len, padding="max_length", truncation=True)

        model_inputs = {
            "que_input_ids": question["input_ids"],
            "que_attention_mask": question["attention_mask"],
            "ctx_input_ids": context["input_ids"],
            "ctx_attention_mask": context["attention_mask"],
            "labels": labels,
        }

        model_inputs["decoder_input_ids"] = [[0]] * len(data)
        model_inputs["decoder_length"] = [1] * len(data)

        for key in model_inputs:
            model_inputs[key] = torch.LongTensor(model_inputs[key])

        return model_inputs
