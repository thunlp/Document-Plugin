from model.PlugD.PlugD import PlugD,HyperPlugD

from model.T5Adapter.T5Adapter import T5Adapter
from torch import nn
from model.metric import softmax_acc,microf1
import bmtrain as bmt
import json
from model_center.model import T5Config
import os
import torch

class TextClassifier(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(TextClassifier, self).__init__()
        self.plmpath = os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"))
        self.t5config = T5Config.from_pretrained(self.plmpath)
        self.model_type = config.get("model", "model_type")
        print("model type:", self.model_type)
        if self.model_type == "t5" or self.model_type ==  "PostT5":
            self.model = T5Adapter(config)
        elif self.model_type == "PlugD" or self.model_type == "PostPlugD":
            self.model = PlugD(config)
        elif self.model_type == "HyperPlugD":
            self.model = HyperPlugD(config)
        else:
            raise ValueError("model_type has not been defined")
        
        self.labelids = json.loads(config.get("data", "labelids"))

        self.loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    def forward(self, data, config, gpu_list, acc_result, mode):
        batch = data["input_ids"].size(0) if "input_ids" in data else data["ctx_input_ids"].size(0)
        device = data["input_ids"].device if "input_ids" in data else data["ctx_input_ids"].device

        data["decoder_input_ids"] = torch.zeros(batch, 2, dtype=torch.long, device=device)
        data["decoder_length"] = torch.ones(batch, dtype=torch.long, device=device) + 1
        data["decoder_input_ids"][:,1] = 32099
        if self.model_type == "PostPlugD" and mode != "test":
            logits = self.model(data, no_ctx=True)
        else:
            logits = self.model(data)
        scores = logits[:,1,self.labelids] *(100*self.t5config.dim_model**-0.5)
        
        
        loss = self.loss_func(scores, data["labels"])

        acc_result = softmax_acc(scores, data["labels"], acc_result)

        return {"loss": loss, "acc_result": acc_result}
