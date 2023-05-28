from transformers import T5Tokenizer
import torch
from torch import nn
from model.metric import squad_metric, squad_train_metric
import bmtrain as bmt
import os
from ..T5Adapter.T5Adapter import T5Adapter
from ..PlugD.PlugD import PlugD,HyperPlugD
from model_center.model import T5Config

class Seq2Seq(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Seq2Seq, self).__init__()
        self.plmpath = os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"))
        self.t5config = T5Config.from_pretrained(self.plmpath)

        self.model_type = config.get("model", "model_type")
        if self.model_type == "t5" or self.model_type == "PostT5":
            self.model = T5Adapter(config)
        elif self.model_type == "PlugD" or self.model_type == "PostPlugD":
            self.model = PlugD(config)
        elif self.model_type == "HyperPlugD" or self.model_type == "PostHyperPlugD":
            self.model = HyperPlugD(config)
        else:
            raise ValueError("model_type has not been defined")

        self.ans_len = config.getint("train", "ans_max_len")
        self.loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(self.plmpath, "tokenizer"))

        self.RL = ("ELI5" in config.get("output", "model_name"))

    def forward(self, data, config, gpu_list, acc_result, mode):
        device = data["input_ids"].device if "input_ids" in data else data["ctx_input_ids"].device
        if mode == "train":
            if self.model_type in ["PostPlugD", "PostHyperPlugD"]:
                logits = self.model(data, no_ctx=True) * (100*self.t5config.dim_model**-0.5)
            else:
                logits = self.model(data) * (100*self.t5config.dim_model**-0.5)
            vocab_size = logits.shape[-1]

            loss = self.loss_func(logits.view(-1, vocab_size), data["labels"].view(-1))
            predict = torch.argmax(logits, dim = 2)
            acc_result = squad_train_metric(predict, data["labels"], acc_result)
        else:
            if self.model_type in ["PostPlugD", "PostHyperPlugD"] and mode == "valid":
                answer = self.model.generate_greedy(data, gen_length=self.ans_len, no_ctx=True)
            else:
                answer = self.model.generate_greedy(data, gen_length=self.ans_len)
            loss = torch.tensor(0.0).to(device)

            acc_result = squad_metric(answer, data["answers"], acc_result, self.tokenizer, RL=self.RL)
        return {"loss": loss, "acc_result": acc_result}

