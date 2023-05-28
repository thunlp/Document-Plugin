from transformers import T5Tokenizer
from ..Basic.DeltaT5 import DeltaT5
import torch
from torch import nn
from model.metric import softmax_acc
import bmtrain as bmt
import os
from model_center.model import T5Config
from ..Basic.layers import Linear
from tools import reduce
import time

class T5Adapter(nn.Module):
    def __init__(self, config):
        super(T5Adapter, self).__init__()
        self.plmpath = os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"))
        print("load pre-trained model from", self.plmpath)
        self.t5config = T5Config.from_pretrained(self.plmpath)

        self.backbone = DeltaT5.from_pretrained(self.plmpath)
        self.tokenizer = T5Tokenizer.from_pretrained(os.path.join(self.plmpath, "tokenizer"))

        self.finetune = config.getboolean("train", "finetune")
        # self.enc_adapter = Linear(1, 4 * self.t5config.num_encoder_layers * self.t5config.dim_model * 32, init_std=0.01)
        # self.dec_adapter = Linear(1, 4 * self.t5config.num_encoder_layers * self.t5config.dim_model * 32, init_std=0.01)
        if self.finetune:
            self.enc_adapter = None
            self.dec_adapter = None
        else:
            self.bottleneck_dim = config.getint("train", "bottleneck_dim")
            self.enc_adapter = Linear(1, 2 * self.t5config.num_encoder_layers * self.t5config.dim_model * self.bottleneck_dim, init_std=0.01)
            self.dec_adapter = Linear(1, 2 * self.t5config.num_encoder_layers * self.t5config.dim_model * self.bottleneck_dim, init_std=0.01)

            bmt.init_parameters(self.enc_adapter)
            bmt.init_parameters(self.dec_adapter)

            self.freeze_module(self.backbone)

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def get_adapter(self, device):
        if self.finetune:
            return None, None
        enc_adapters = self.enc_adapter(torch.ones(1, 1, dtype=torch.half, device=device)).view(self.t5config.num_encoder_layers, 2, 1, self.t5config.dim_model, self.bottleneck_dim)
        dec_adapters = self.dec_adapter(torch.ones(1, 1, dtype=torch.half, device=device)).view(self.t5config.num_decoder_layers, 2, 1, self.t5config.dim_model, self.bottleneck_dim)
        return enc_adapters, dec_adapters

    def forward(self, data):
        enc_adapters, dec_adapters = self.get_adapter(data["input_ids"].device)

        logits = self.backbone(
                input_ids = data["input_ids"],
                attention_mask = data["attention_mask"],
                decoder_input_ids = data["decoder_input_ids"],
                decoder_length = data["decoder_length"] if "decoder_length" in data else None,
                decoder_attention_mask=data["decoder_attention_mask"] if "decoder_attention_mask" in data else None,
                return_logits = True,
                enc_adapters = enc_adapters,
                dec_adapters = dec_adapters,
            )
        return logits

    def generate_greedy(self, data, gen_length=20):
        enc_adapters, dec_adapters = self.get_adapter(data["input_ids"].device)

        batch, device = data["input_ids"].size(0), data["input_ids"].device

        dec_input_ids = torch.zeros(batch, gen_length + 4, dtype=torch.long).to(device)
        dec_input_ids[:,1] = 32099
        length = torch.LongTensor([gen_length + 4] * batch).to(device)
        position =  1 # batch

        # print(self.tokenizer.decode(data["input_ids"][0]))
        # print("==" * 15)
        predict, logits = self.backbone(
            input_ids = data["input_ids"],
            attention_mask = data["attention_mask"],
            decoder_input_ids = dec_input_ids,
            decoder_length=length,
            enc_adapters = enc_adapters,
            dec_adapters = dec_adapters,
        )

        encoder_outputs = predict.encoder_last_hidden_state
        answer = [torch.argmax(logits[:, position], dim=-1)]
        end = (answer[-1] == 1)
        for i in range(gen_length):
            if not end.all():
                position += 1
                dec_input_ids[:, position] = answer[-1]
            predict, logits = self.backbone(
                encoder_outputs=encoder_outputs,
                attention_mask = data["attention_mask"],
                decoder_input_ids=dec_input_ids,
                decoder_length=length,
                enc_adapters = enc_adapters,
                dec_adapters = dec_adapters,
            )
            if not end.all():
                answer.append(torch.argmax(logits[:, position], dim=-1))
                answer[-1][end] = 1
                end = end | (answer[-1] == 1)
            all_end = reduce(end.all().int(), "sum")
            if all_end == bmt.world_size():
                break
        answer = torch.cat([a.unsqueeze(1) for a in answer], dim=1).contiguous()
        return answer

