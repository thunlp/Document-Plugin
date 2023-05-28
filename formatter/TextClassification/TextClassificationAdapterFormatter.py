import json
import torch
import os
import numpy as np

import random
from transformers import T5Tokenizer

class TextClassificationAdapterFormatter:
    def __init__(self, config, mode, *args, **params):
        self.ctx_len = config.getint("train", "ctx_len")
        self.mode = mode
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"), "tokenizer"))
        self.label2id = json.load(open(config.get("data", "label2id"), "r"))
        self.query = self.tokenizer.encode((" or ".join(list(self.label2id.keys())) + "?").lower(), add_special_tokens=False)

    def tokenize(self, doc):
        doctoken = self.tokenizer.encode(doc, add_special_tokens=False)
        alltokens = doctoken[:self.ctx_len - len(self.query) - 1] + self.query + [self.tokenizer.eos_token_id]
        mask = [1] * len(alltokens) + [0] * (self.ctx_len - len(alltokens))
        if len(alltokens) < self.ctx_len:
            alltokens += [self.tokenizer.pad_token_id] * (self.ctx_len - len(alltokens))
        return alltokens, mask

    def process(self, data):
        inputids, attmask = [], []
        for doc in data:
            inp, ma = self.tokenize(doc["text"])
            inputids.append(inp), attmask.append(ma)

        labels = [self.label2id[d["label"]] for d in data]

        model_inputs = {
            "input_ids": inputids,
            "attention_mask": attmask,
            "labels": labels,
        }

        model_inputs["decoder_input_ids"] = [[0]] * len(data)
        model_inputs["decoder_length"] = [1] * len(data)

        for key in model_inputs:
            model_inputs[key] = torch.LongTensor(model_inputs[key])

        return model_inputs
