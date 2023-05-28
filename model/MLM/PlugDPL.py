from random import random
from transformers import T5Tokenizer
from ..PlugD.PlugD import PlugD,HyperPlugD
from ..Basic.DeltaT5 import DeltaT5

import torch
from torch import nn
from model.metric import mlm_acc_loss
import bmtrain as bmt
from bmtrain import print_rank

class PlugDPlugLearning(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(PlugDPlugLearning, self).__init__()

        self.model = PlugD(config, pretrain=True)
        self.loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100, reduction="none")

        self.layerth = config.getint("train", "layerth")

    def cal_dec(self, hiddens, mask, dec_inp, dec_mask, labels):
        logits = self.model.cal_dec(hiddens, mask, dec_inp, dec_mask)
        
        return self.cal_loss(logits * (100*self.model.plm_config.dim_model**-0.5), labels)
    
    def cal_loss(self, logits, labels):
        batch, seq_len, vocab_size = logits.size()

        loss_shape = self.loss_func(logits.view(-1, vocab_size), labels.view(-1))
        loss_mask = (labels != -100).sum(dim=1).float()
        loss_mask[loss_mask == 0] = torch.inf
        loss = (loss_shape.view(batch, seq_len).sum(dim=1) / loss_mask).mean()
        return loss

    def forward(self, data, config, gpu_list, acc_result, mode):

        parameters, ctx_last_hidden = self.model.generate_doc_plug_train(
                input_ids = data["ctx_input_ids"],
                attention_mask = data["ctx_attention_mask"],
            )


        deltas = {"type": "prefix", "prefix_num": ctx_last_hidden.size(1), "layerth": self.layerth}
        ctx_attention_mask = data["ctx_attention_mask"]

        output, logits = self.model.backbone(
                input_ids=data["que_input_ids"],
                attention_mask=data["que_attention_mask"],
                decoder_input_ids=data["decoder_input_ids"],
                decoder_attention_mask=data["decoder_attention_mask"],
                deltas = deltas,
                pfxatt_mask=ctx_attention_mask,
                parameters = parameters
            )
        loss = self.cal_loss(logits * (100*self.model.plm_config.dim_model**-0.5), data["labels"])

        predict = torch.argmax(logits, dim = -1)

        acc_result = mlm_acc_loss(predict, data["labels"], acc_result, loss)
        return {"loss": loss, "acc_result": acc_result}
