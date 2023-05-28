import json
import torch
import os
import numpy as np

import random
from transformers import T5Tokenizer,T5Config

class FEVERCtxFormatter:
    def __init__(self, config, mode, *args, **params):
        self.max_len = config.getint("train", "max_len")
        self.mode = mode
        self.config = config
        self.model_type = config.get("model", "model_type")
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"), "tokenizer"))
        self.max_len = config.getint("train", "max_len")

        if self.model_type == "PostT5" and self.mode != "test":
            self.max_len = 128
        self.label2id = {
            "SUPPORTS": 0,
            "REFUTES": 1,
            # "NOT ENOUGH INFO": 2
        }
        self.top_ctx = 3

    def process(self, data):
        # claims = [d["claim"] for d in data]
        claims = [d["input"] for d in data]
        ctxs = ["\n".join([text["text"] for text in d["output"][0]["provenance"][:self.top_ctx]]) for d in data]

        if self.model_type == "PostT5" and self.mode != "test":
            text = ["claim: %s" % c for c in claims]
        else:
            text = ["claim: %s \n Context: %s" % (c, ctx) for c, ctx in zip(claims, ctxs)]

        ret = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True)

        labels = [self.label2id[d["output"][1]["answer"]] for d in data]
        ret["labels"] = labels
        for key in ret:
            ret[key] = torch.LongTensor(ret[key])
        return ret



class FEVERCtxPlugDFormatter:
    def __init__(self, config, mode, *args, **params):
        self.max_len = config.getint("train", "max_len")
        self.ctx_len = config.getint("train", "ctx_len")
        self.mode = mode
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"), "tokenizer"))

        self.label2id = {
            "SUPPORTS": 0,
            "REFUTES": 1,
            # "NOT ENOUGH INFO": 2
        }
        self.top_ctx = 3

    def process(self, data):
        # claims = [d["claim"] for d in data]
        claims = [d["input"] + "yes or no? <extra_id_0>" for d in data]
        ctxs = ["\n".join([text["text"] for text in d["output"][0]["provenance"][:self.top_ctx]]) for d in data]

        ctx_info = self.tokenizer(ctxs, max_length=self.ctx_len, padding="max_length", truncation=True)
        query_info = self.tokenizer(claims, max_length=self.max_len, padding="max_length", truncation=True)

        ret = {
            "que_input_ids": query_info["input_ids"],
            "que_attention_mask": query_info["attention_mask"],
            "ctx_input_ids": ctx_info["input_ids"],
            "ctx_attention_mask": ctx_info["attention_mask"],
            "labels": [self.label2id[d["output"][1]["answer"]] for d in data],
            "decoder_input_ids": [[0]] * len(data),
            "decoder_length": [1] * len(data)
        }

        for key in ret:
            ret[key] = torch.LongTensor(ret[key])
        return ret



class FEVERCtxED2LMFormatter:
    def __init__(self, config, mode, *args, **params):
        self.max_len = config.getint("train", "max_len")
        self.ctx_len = config.getint("train", "ctx_len")
        self.mode = mode
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"), "tokenizer"))

        self.label2id = {
            "SUPPORTS": 0,
            "REFUTES": 1,
            # "NOT ENOUGH INFO": 2
        }
        self.top_ctx = 3

    def process(self, data):
        # claims = [d["claim"] for d in data]
        claims = [d["input"] for d in data]
        ctxs = ["\n".join([text["text"] for text in d["output"][0]["provenance"][:self.top_ctx]]) for d in data]

        ctx_info = self.tokenizer(ctxs, max_length=self.ctx_len, padding="max_length", truncation=True)
        decoder_inp, position = [], []
        for query in claims:
            qtoken = self.tokenizer.encode(query + "Answer:", add_special_tokens=False)
            p = len(qtoken)
            qtoken = qtoken + [0] * (self.max_len - len(qtoken))
            decoder_inp.append(qtoken[:self.max_len])
            position.append(min(p, self.max_len - 1))

        ret = {
            "decoder_input_ids": decoder_inp,
            "decoder_attention_mask": [[1] * self.max_len] * len(data),
            "ctx_input_ids": ctx_info["input_ids"],
            "ctx_attention_mask": ctx_info["attention_mask"],
            "labels": [self.label2id[d["output"][1]["answer"]] for d in data],
            "position": position,
        }

        for key in ret:
            ret[key] = torch.LongTensor(ret[key])
        return ret
