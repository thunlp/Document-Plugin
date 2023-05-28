import torch

from model_center.layer import LayerNorm
from model_center.layer.blocks import CrossAttentionBlock,FFNBlock
import bmtrain as bmt
from typing import *

from .DeltaBlocks import SelfAttentionDeltaBlock,FFNDeltaBlock

import torch.nn.functional as F

import torch
from model_center.layer import Decoder, Embedding, Linear, RelativePositionEmbedding
from model_center.model.basemodel import BaseModel
from model_center.model.config import T5Config
from transformers.modeling_outputs import Seq2SeqModelOutput
from model_center.utils import check_web_and_convert_path
import os
from bmtrain import print_rank
import math

def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class TransformerDeltaBlock(torch.nn.Module):
    def __init__(self, 
                 dim_model : int,  dim_ff : int, num_heads : int, dim_head : int,
                 is_decoder : bool = False, dtype = torch.half, int8 = False,
                 norm_init_var : float = 1.0, norm_bias : bool = False, norm_eps : float = 1e-5, 
                 att_init_mean : float = 0.0, att_init_std : float = 0.02, att_bias : bool = False, att_mask_value : float = float("-inf"),
                 ffn_init_mean : float = 0.0, ffn_init_std : float = 0.02, ffn_bias : bool = False, ffn_activate_fn : str = "gated_gelu",
                 pos_bias_type : str = "none", post_layer_norm : bool = False,
                 parallel_ffn : bool = False, length_scale : bool = False, attn_scale : bool = False, dropout_p : float = 0,
                 layer_no : int = -1,
                ):
        super().__init__()

        self.is_decoder = is_decoder
        self.layer_no = layer_no

        self.self_att = SelfAttentionDeltaBlock(
            dim_model = dim_model, num_heads = num_heads, dim_head = dim_head, 
            dtype = dtype, int8 = int8, 
            norm_eps = norm_eps, norm_init_var = norm_init_var, norm_bias = norm_bias,
            att_init_mean = att_init_mean, att_init_std = att_init_std, att_bias = att_bias, att_mask_value = att_mask_value,
            pos_bias_type = pos_bias_type,
            post_layer_norm = post_layer_norm,
            length_scale = length_scale, attn_scale = attn_scale, dropout_p = dropout_p,
            layer_no = layer_no
        )

        if is_decoder:
            self.cross_att = CrossAttentionBlock(
                dim_model = dim_model, num_heads = num_heads, dim_head = dim_head, 
                dtype = dtype, int8 = int8, 
                norm_eps = norm_eps, norm_init_var = norm_init_var, norm_bias = norm_bias,
                att_init_mean = att_init_mean, att_init_std = att_init_std, att_bias = att_bias, att_mask_value = att_mask_value,
                pos_bias_type = pos_bias_type, length_scale = length_scale,
                attn_scale = attn_scale, dropout_p = dropout_p,
            )
        else:
            self.cross_att = None

        self.ffn = FFNDeltaBlock(
            dim_model = dim_model, dim_ff = dim_ff,
            dtype = dtype, int8 = int8,
            norm_eps = norm_eps, norm_init_var = norm_init_var, norm_bias = norm_bias,
            ffn_init_mean = ffn_init_mean, ffn_init_std = ffn_init_std, ffn_bias = ffn_bias, ffn_activate_fn = ffn_activate_fn,
            length_scale = length_scale, dropout_p = dropout_p, post_layer_norm = post_layer_norm,
            layer_no = layer_no
        )

        self.parallel_ffn = parallel_ffn

    # def cal_adapter(self, map1: torch.Tensor, map2: torch.Tensor, hiddens: torch.Tensor):
    #     '''
    #     map1: batch, hidden_size, adapter_rank
    #     map2: batch, adapter_rank, hidden_size
    #     hiddens: batch, seq_len, hidden_size
    #     '''
    #     x = torch.matmul(hiddens, map1)
    #     x = F.relu(x)
    #     x = torch.matmul(x, map2) 
    #     return hiddens + x

    def forward(self,
                self_hidden_states : torch.Tensor,
                self_attention_mask : torch.Tensor,
                self_position_bias : Optional[torch.Tensor] = None,
                deltas : Optional[Dict] = None,
                parameters : torch.Tensor = None,
                cross_hidden_states = None,
                cross_attention_mask = None,
                cross_position_bias = None,
                adapters = None,
                lora = None,
            ):
        # (batch, dim_model, seq_self)
        if deltas is not None and deltas["type"] == "prefix" and self.layer_no >= deltas["layerth"]:# and not self.is_decoder:
            if "diffl" in deltas and deltas["diffl"]:
                pfx = parameters[self.layer_no]
            else:
                pfx = parameters#[self.layer_no]#deltas["parameters"][self.layer_no]
                # print_rank("with plug %s" % self.layer_no)
        else:
            pfx = None
        if pfx is None and self_attention_mask.size(2) != self_hidden_states.size(1):
            self_attention_mask = self_attention_mask[:,:,:self_hidden_states.size(1)]
            self_position_bias = self_position_bias[:,:,:self_hidden_states.size(1)]
        
        if deltas is not None and "qlen" in deltas:
            qlen = deltas["qlen"]
            # sattention_mask = self_attention_mask.clone() # batch, seq_len, seq_len
            # sattention_mask[:,qlen:,:qlen] = 0
            # sattention_mask[:,:qlen,qlen:] = 0

            # print(self_attention_mask.shape, self_position_bias.shape)
            if self.layer_no < deltas["layerth"]:
                # print(adapters[0].shape)
                qhidden_states = self.self_att(self_hidden_states[:,:qlen],
                                        attention_mask = self_attention_mask[:,:qlen,:qlen],
                                        position_bias = self_position_bias[:,:qlen,:qlen],
                                        adapters=adapters[0] if not deltas["fix_encoder"] else adapters)

                dhidden_states = self.self_att(self_hidden_states[:,qlen:],
                                        attention_mask = self_attention_mask[:,qlen:,qlen:],
                                        position_bias = self_position_bias[:,qlen:,qlen:],
                                        adapters=adapters[1] if not deltas["fix_encoder"] else None
                                        )
                hidden_states = torch.cat([qhidden_states, dhidden_states], dim=1)
            else:
                # print(adapters[0].shape)
                hidden_states = self.self_att(self_hidden_states,
                                      attention_mask = self_attention_mask,
                                      position_bias = self_position_bias,
                                      prefix=pfx,
                                      adapters=adapters[0] if not deltas["fix_encoder"] else adapters)
        else:
            hidden_states = self.self_att(self_hidden_states,
                                      attention_mask = self_attention_mask,
                                      position_bias = self_position_bias,
                                      prefix=pfx,
                                      adapters=adapters,
                                      lora=lora)

        # (batch, dim_model, seq_self)
        if self.is_decoder and self.cross_att is not None:
            hidden_states = self.cross_att(hidden_states = hidden_states,
                                           key_value_states = cross_hidden_states,
                                           attention_mask = cross_attention_mask,
                                           position_bias = cross_position_bias)
        # if adapters is not None:
        #     layer_adapter = adapters[self.layer_no] #4, batch, hidden, delta_rank
        #     map1, map2 = layer_adapter[2], layer_adapter[3].transpose(1, 2)
        #     hidden_states = self.cal_adapter(map1, map2, hidden_states)

        # (batch, dim_model, seq_self)
        if self.parallel_ffn:
            hidden_states_2 = self.ffn(self_hidden_states, adapters=adapters)
            hidden_states = hidden_states - self_hidden_states + hidden_states_2
        else:
            if deltas is not None and "qlen" in deltas:
                qlen = deltas["qlen"]
                if self.layer_no < deltas["layerth"]:
                    # print(adapters[0].shape)
                    qhidden_states = self.ffn(hidden_states[:,:qlen], 
                                adapters=adapters[0] if not deltas["fix_encoder"] else adapters)
                    dhidden_states = self.ffn(hidden_states[:,qlen:],
                                adapters=adapters[1] if not deltas["fix_encoder"] else adapters)
                    hidden_states = torch.cat([qhidden_states, dhidden_states], dim=1)
                else:
                    hidden_states = self.ffn(hidden_states, adapters=adapters[0] if not deltas["fix_encoder"] else adapters)
            else:
                hidden_states = self.ffn(hidden_states, adapters=adapters)

        # if adapters is not None:
        #     layer_adapter = adapters[self.layer_no] #2, batch, hidden, delta_rank
        #     map1, map2 = layer_adapter[0], layer_adapter[1].transpose(1, 2)
        #     hidden_states = self.cal_adapter(map1, map2, hidden_states)

        return hidden_states


class DeltaEncoder(torch.nn.Module):
    def __init__(self, 
            num_layers : int, dim_model : int, dim_ff : int, num_heads : int, dim_head : int,
            dtype : torch.dtype = torch.half, int8 : bool = False, 
            norm_init_var : float = 1.0, norm_bias : bool = False, norm_eps : float = 1e-5, 
            att_init_mean : float = 0.0, att_init_std : float = 0.02, att_bias : bool = False, att_mask_value : float = float("-inf"),
            ffn_init_mean : float = 0.0, ffn_init_std : float = 0.02, ffn_bias : bool = False, ffn_activate_fn : str = "gated_gelu",
            pos_bias_type : str = "none", post_layer_norm : bool = False,
            length_scale : bool = False, attn_scale : bool = False,
            dropout_p : float = 0, parallel_ffn : bool = False,
        ):

        super().__init__()
        
        self.num_layers = num_layers

        self.layers = bmt.TransformerBlockList([
        # self.layers = torch.nn.ModuleList([
            bmt.CheckpointBlock(
                TransformerDeltaBlock(
                    dim_model = dim_model, dim_ff = dim_ff, num_heads = num_heads, dim_head = dim_head,
                    is_decoder = False, dtype = dtype, int8 = int8,
                    norm_eps = norm_eps, norm_init_var = norm_init_var, norm_bias = norm_bias,
                    att_init_mean = att_init_mean, att_init_std = att_init_std, att_bias = att_bias, att_mask_value = att_mask_value,
                    ffn_init_mean = ffn_init_mean, ffn_init_std = ffn_init_std, ffn_bias = ffn_bias, ffn_activate_fn = ffn_activate_fn,
                    pos_bias_type = pos_bias_type, post_layer_norm = post_layer_norm,
                    length_scale = length_scale, attn_scale = attn_scale, dropout_p = dropout_p,
                    parallel_ffn = parallel_ffn,
                    layer_no = ln,
                )
            )
            for ln in range(num_layers)
        ])

        self.output_layernorm = LayerNorm(
                    dim_norm = dim_model, 
                    bias = norm_bias, 
                    dtype = dtype,
                    eps = norm_eps,
                    init_var = norm_init_var)

        self.parallel_ffn = parallel_ffn

    def forward(self, hidden_states : torch.Tensor,
                      attention_mask : torch.Tensor,
                      position_bias : torch.Tensor = None,
                      deltas : Optional[Dict] = None,
                      parameters : torch.Tensor = None,
                      adapters : torch.Tensor = None,
                      lora : torch.Tensor = None,
                      ):

        # (batch, seq_enc, dim_model)
        hidden_states = self.layers(hidden_states, attention_mask, position_bias, deltas, parameters, None, None, None, adapters, lora)
        # all_hidden = [hidden_states]
        # for layer in self.layers:
        #     hidden_states = layer(hidden_states, attention_mask, position_bias, deltas, parameters, None, None, None, adapters=adapters, lora=lora)
        #     all_hidden.append(hidden_states)
        # (batch, seq_enc, dim_model)
        
        if deltas is not None and deltas["type"] == "prefix":
            if "diffl" in deltas and deltas["diffl"]:
                pfx = parameters[-1]
            else:
                pfx = parameters
            hidden_states = torch.cat([hidden_states, pfx], dim=1)
        # print("before layer norm", hidden_states)

        hidden_states = self.output_layernorm(hidden_states)

        return hidden_states#, all_hidden


class DeltaDecoder(torch.nn.Module):
    def __init__(self, 
            num_layers : int, dim_model : int,  dim_ff : int, num_heads : int, dim_head : int,
            dtype : torch.dtype = torch.half, int8 : bool = False, 
            norm_init_var : float = 1.0, norm_bias : bool = False, norm_eps : float = 1e-5, 
            att_init_mean : float = 0.0, att_init_std : float = 0.02, att_bias : bool = False, att_mask_value : float = float("-inf"),
            ffn_init_mean : float = 0.0, ffn_init_std : float = 0.02, ffn_bias : bool = False, ffn_activate_fn : str = "gated_gelu",
            pos_bias_type : str = "none", length_scale : bool = False, attn_scale : bool = False,
            dropout_p : float = 0,
            parallel_ffn : bool = False,
        ):

        super().__init__()
        
        self.num_layers = num_layers

        self.layers = bmt.TransformerBlockList([
            bmt.CheckpointBlock(
                TransformerDeltaBlock(
                    dim_model = dim_model, 
                    dim_ff = dim_ff,
                    num_heads = num_heads,
                    dim_head = dim_head,
                    is_decoder = True,
                    dtype = dtype, 
                    int8 = int8,
                    norm_init_var = norm_init_var,
                    norm_bias = norm_bias,
                    norm_eps = norm_eps, 
                    att_init_mean = att_init_mean, 
                    att_init_std = att_init_std,
                    att_bias = att_bias,
                    att_mask_value = att_mask_value,
                    ffn_init_mean = ffn_init_mean, 
                    ffn_init_std = ffn_init_std,
                    ffn_bias = ffn_bias,
                    ffn_activate_fn = ffn_activate_fn,
                    pos_bias_type = pos_bias_type,
                    length_scale = length_scale,
                    attn_scale = attn_scale,
                    dropout_p = dropout_p,
                    parallel_ffn = parallel_ffn,
                    layer_no=ln
                )
            )
            for ln in range(num_layers)
        ])

        self.output_layernorm = LayerNorm(
                    dim_norm = dim_model, 
                    bias = norm_bias, 
                    dtype = dtype,
                    eps = norm_eps, 
                    init_var = norm_init_var)

    def forward(self, hidden_states : torch.Tensor,
                      attention_mask : torch.Tensor,
                      position_bias : torch.Tensor,
                      cross_hidden_states = None,
                      cross_attention_mask = None,
                      cross_position_bias = None,
                      deltas = None,
                      parameters : torch.Tensor = None,
                      adapters : torch.Tensor = None,
                      lora : torch.Tensor = None,
                      ):

        # (batch, dim_model, seq_dec)
        # print_rank(hidden_states.size(), attention_mask.size(), cross_hidden_states.size(), cross_attention_mask.size())
        hidden_states = self.layers(hidden_states, attention_mask, position_bias, deltas, parameters,
                            cross_hidden_states, cross_attention_mask, cross_position_bias, adapters, lora)
        # (batch, dim_model, seq_dec)
        hidden_states = self.output_layernorm(hidden_states)
        return hidden_states


class DeltaT5(BaseModel): 
    _CONFIG_TYPE = T5Config

    def __init__(self,
        config: T5Config, output_hidden=False, doc_pos=True,
        ):
        
        super().__init__()

        self.config = config
        self.output_hidden = output_hidden

        self.encoder = DeltaEncoder(
            num_layers = config.num_encoder_layers, dim_model = config.dim_model, dim_ff = config.dim_ff, num_heads = config.num_heads, dim_head = config.dim_head,
            dtype = config.dtype, int8 = config.int8,
            norm_eps = config.norm_eps, norm_init_var = config.norm_init_var, norm_bias = config.norm_bias,
            att_init_mean = config.att_init_mean, att_init_std = config.att_init_std, att_bias = config.att_bias, att_mask_value = float(config.att_mask_value),
            pos_bias_type = config.pos_bias_type,
            ffn_init_mean = config.ffn_init_mean, ffn_init_std = config.ffn_init_std, ffn_bias = config.ffn_bias, ffn_activate_fn = config.ffn_activate_fn,
            length_scale = config.length_scale, attn_scale = config.attn_scale,
            dropout_p = config.dropout_p,
        )

        self.decoder = DeltaDecoder(
            num_layers = config.num_decoder_layers, dim_model = config.dim_model, dim_ff = config.dim_ff, num_heads = config.num_heads, dim_head = config.dim_head,
            dtype = config.dtype, int8 = config.int8,
            norm_eps = config.norm_eps, norm_init_var = config.norm_init_var, norm_bias = config.norm_bias,
            att_init_mean = config.att_init_mean, att_init_std = config.att_init_std, att_bias = config.att_bias, att_mask_value = float(config.att_mask_value),
            pos_bias_type = config.pos_bias_type,
            ffn_init_mean = config.ffn_init_mean, ffn_init_std = config.ffn_init_std, ffn_bias = config.ffn_bias, ffn_activate_fn = config.ffn_activate_fn,
            length_scale = config.length_scale, attn_scale = config.attn_scale,
            dropout_p = config.dropout_p,
        )

        self.input_embedding = Embedding(
            vocab_size = config.vocab_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.position_bias_enc = RelativePositionEmbedding(
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            bidirectional = True, 
            dtype = config.dtype,
            init_mean = config.pos_init_mean,
            init_std = config.pos_init_std,
        )

        self.position_bias_dec = RelativePositionEmbedding(
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            bidirectional = False, 
            dtype = config.dtype,
            init_mean = config.pos_init_mean,
            init_std = config.pos_init_std,
        )

        if doc_pos:
            self.docplug_position_bias = Embedding(vocab_size=1, embedding_size=config.num_heads, dtype = config.dtype, int8 = config.int8, init_mean = config.emb_init_mean, init_std = config.emb_init_std,)
        else:
            self.docplug_position_bias = None

        self.tied = config.tied
        self.cls_head = config.cls_head
        if self.cls_head:
            self.cls_projection = Linear(
                dim_out = self.cls_head,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )
        if not self.tied:
            self.output_projection = Linear(
                dim_out = config.vocab_size,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], config=None, **kwargs):
        if config is None:
            config = cls._CONFIG_TYPE.from_pretrained(pretrained_model_name_or_path)
        path = check_web_and_convert_path(pretrained_model_name_or_path, 'model')
        model = cls(config, **kwargs)
        bmt.init_parameters(model)
        bmt.load(model, os.path.join(path, 'pytorch_model.pt'), strict=False)
        return model

    def forward(self, 
                input_ids = None, # (batch, seq_enc)
                length = None, # (batch)
                decoder_input_ids = None, # (batch, seq_dec)
                decoder_length = None, # (batch)
                attention_mask = None, # (batch, seq_enc)
                decoder_attention_mask = None, # (batch, seq_dec)
                deltas : Optional[Dict] = None,
                parameters : Optional[torch.Tensor] = None,
                pfxatt_mask : Optional[torch.Tensor] = None,
                enc_adapters : Optional[torch.Tensor] = None,
                dec_adapters : Optional[torch.Tensor] = None,
                enc_lora : Optional[torch.Tensor] = None,
                dec_lora : Optional[torch.Tensor] = None,
                head_mask = None, # unused
                decoder_head_mask = None, # unused
                cross_attn_head_mask = None, # unused
                encoder_outputs = None,
                inputs_embeds = None, 
                decoder_inputs_embeds = None,
                output_attentions = None, # unused
                output_hidden_states = None, # unused
                return_dict = True,
                return_logits = False,
                only_encoder : bool = False,
    ):
        # print_rank("step")
        # encoder
        if encoder_outputs is None:
            assert input_ids is not None or inputs_embeds is not None

            if input_ids is not None:
                batch = input_ids.size(0)
                seq_enc = input_ids.size(1)
                device = input_ids.device
            else:
                batch = inputs_embeds.size(0)
                seq_enc = inputs_embeds.size(1)
                device = inputs_embeds.device
            
            has_prefix = False
            if deltas is not None and deltas["type"] == "prefix":
                prefix_num = deltas["prefix_num"]
                # seq_enc = seq_enc + prefix_num
                has_prefix = True

            with torch.no_grad():
                if attention_mask is not None:
                    # if has_prefix:
                    #     attention_mask = torch.cat([torch.ones(batch, prefix_num, dtype=torch.half).to(device), attention_mask], dim = 1)
                    attention_mask = attention_mask.to(torch.bool)
                else:
                    attention_mask = torch.arange(seq_enc, device=device)[None, :].repeat(batch, 1) < length[:, None]
                # (batch, seq_enc, seq_enc)
                if has_prefix:
                    if pfxatt_mask is None:
                        # print("pfxatt_mask is None")
                        pfxatt_mask = torch.ones(batch, 1, prefix_num, dtype=torch.bool, device=attention_mask.device)
                    enc_attention_mask = attention_mask.view(batch, seq_enc, 1) & torch.cat([attention_mask.view(batch, 1, seq_enc), pfxatt_mask.unsqueeze(1)], dim=2)
                    # print(enc_attention_mask.shape)
                    # print(enc_attention_mask)
                else:
                    enc_attention_mask = attention_mask.view(batch, seq_enc, 1) & attention_mask.view(batch, 1, seq_enc)

            # (num_heads, seq_enc, seq_enc)
            if has_prefix:
                enc_position_bias = self.position_bias_enc(seq_enc, seq_enc + prefix_num)
                # print_rank(enc_position_bias.shape)
                if not self.docplug_position_bias is None:
                    docpos = self.docplug_position_bias(torch.zeros(1, 1, dtype=torch.long, device=device)).squeeze() # num_head
                    enc_position_bias[:, :, seq_enc:] = docpos.unsqueeze(1).unsqueeze(1)
            else:
                enc_position_bias = self.position_bias_enc(seq_enc, seq_enc)

            # (batch, dim_model, seq_enc)
            if inputs_embeds is None:
                hidden_states_enc = self.input_embedding(input_ids)
            else:
                hidden_states_enc = inputs_embeds

            # (batch, dim_model, seq_enc)
            # print(enc_attention_mask.shape)
            # encoder_outputs, all_enc_hidden = self.encoder(hidden_states_enc, enc_attention_mask, enc_position_bias, deltas, parameters, adapters=enc_adapters)
            encoder_outputs = self.encoder(hidden_states_enc, enc_attention_mask, enc_position_bias, deltas, parameters, adapters=enc_adapters, lora=enc_lora)
            if only_encoder:
                return encoder_outputs

        
        # decoder
        assert decoder_input_ids is not None or decoder_inputs_embeds is not None

        if decoder_input_ids is not None:
            batch = decoder_input_ids.size(0)
            seq_dec = decoder_input_ids.size(1)
            device = decoder_input_ids.device
        else:
            batch = decoder_inputs_embeds.size(0)
            seq_dec = decoder_inputs_embeds.size(1)
            device = decoder_inputs_embeds.device

        seq_enc = encoder_outputs.size(1)
        # has_prefix=False
        if seq_enc != attention_mask.size(1): # and deltas is not None and deltas["type"] == "prefix":
            # prefix_num = self.delta_config.delta_rank
            prefix_num = seq_enc - attention_mask.size(1)
            # has_prefix=True
            if pfxatt_mask is None:
                # print("pfxatt_mask is None")
                pfxatt_mask = torch.ones(batch, prefix_num, dtype=torch.bool, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, pfxatt_mask], dim = 1)
            attention_mask = attention_mask.to(torch.bool)

        with torch.no_grad():
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(torch.bool)
            else:
                decoder_attention_mask = torch.arange(seq_dec, device=device)[None, :].repeat(batch, 1) < decoder_length[:, None]
            directional_mask_2d = torch.arange(seq_dec, device=device) <= torch.arange(seq_dec, device=device).view(-1, 1)
            # (batch, seq_dec, seq_dec)
            dec_attention_mask = decoder_attention_mask.view(batch, seq_dec, 1) & decoder_attention_mask.view(batch, 1, seq_dec) & directional_mask_2d.view(1, seq_dec, seq_dec)
            # (batch, seq_dec, seq_enc)
            if attention_mask.dim() == 3:
                cross_attention_mask = attention_mask.max(dim=1, keepdim=True)[0] & decoder_attention_mask.view(batch, seq_dec, 1)
            else:
                cross_attention_mask = attention_mask.view(batch, 1, seq_enc) & decoder_attention_mask.view(batch, seq_dec, 1)

        # (num_heads, seq_dec, seq_dec)
        dec_position_bias = self.position_bias_dec(seq_dec, seq_dec)

        # (batch, seq_dec, dim_model)
        if decoder_inputs_embeds is None:
            hidden_states_dec = self.input_embedding(decoder_input_ids)
        else:
            hidden_states_dec = decoder_inputs_embeds
        # (batch, seq_dec, dim_model)
        decoder_outputs = self.decoder(hidden_states_dec, dec_attention_mask, dec_position_bias,
                                       encoder_outputs, cross_attention_mask, None, deltas=None, parameters=None, adapters=dec_adapters, lora=dec_lora)

        # (batch, seq_dec, vocab_output_size)
        # print_rank("output:", decoder_outputs.mean().item(), decoder_outputs.std().item())
        if self.cls_head:
            logits = self.cls_projection(decoder_outputs)
        elif self.tied:
            logits = self.input_embedding.projection(decoder_outputs)
        elif not self.tied:
            logits = self.output_projection(decoder_outputs)

        if return_logits:
            return logits#*(100*self.config.dim_model**-0.5)

        if not return_dict:
            return tuple(decoder_outputs, None, None, None, None)
        else:
            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs,
                encoder_last_hidden_state=encoder_outputs,
                past_key_values=None,
                encoder_hidden_states=None,
                decoder_hidden_states=None,
                decoder_attentions=None,
                cross_attentions=None,
                encoder_attentions=None,
            ), logits



class DeltaT5OnlyEnc(BaseModel): 
    _CONFIG_TYPE = T5Config

    def __init__(self,
        config: T5Config, output_hidden=False
        ):
        
        super().__init__()

        self.config = config
        self.output_hidden = output_hidden

        self.encoder = DeltaEncoder(
            num_layers = config.num_encoder_layers, dim_model = config.dim_model, dim_ff = config.dim_ff, num_heads = config.num_heads, dim_head = config.dim_head,
            dtype = config.dtype, int8 = config.int8,
            norm_eps = config.norm_eps, norm_init_var = config.norm_init_var, norm_bias = config.norm_bias,
            att_init_mean = config.att_init_mean, att_init_std = config.att_init_std, att_bias = config.att_bias, att_mask_value = float(config.att_mask_value),
            pos_bias_type = config.pos_bias_type,
            ffn_init_mean = config.ffn_init_mean, ffn_init_std = config.ffn_init_std, ffn_bias = config.ffn_bias, ffn_activate_fn = config.ffn_activate_fn,
            length_scale = config.length_scale, attn_scale = config.attn_scale,
            dropout_p = config.dropout_p,
        )

        self.input_embedding = Embedding(
            vocab_size = config.vocab_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.position_bias_enc = RelativePositionEmbedding(
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            bidirectional = True, 
            dtype = config.dtype,
            init_mean = config.pos_init_mean,
            init_std = config.pos_init_std,
        )


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], config=None, **kwargs):
        if config is None:
            config = cls._CONFIG_TYPE.from_pretrained(pretrained_model_name_or_path)
        path = check_web_and_convert_path(pretrained_model_name_or_path, 'model')
        model = cls(config, **kwargs)
        bmt.init_parameters(model)
        bmt.load(model, os.path.join(path, 'pytorch_model.pt'), strict=False)
        return model

    def forward(self, 
                input_ids = None, # (batch, seq_enc)
                length = None, # (batch)
                attention_mask = None, # (batch, seq_enc)

                enc_adapters : Optional[torch.Tensor] = None,

                enc_lora : Optional[torch.Tensor] = None,
                inputs_embeds = None, 
    ):
        # print_rank("step")
        # encoder
        assert input_ids is not None or inputs_embeds is not None

        if input_ids is not None:
            batch = input_ids.size(0)
            seq_enc = input_ids.size(1)
            device = input_ids.device
        else:
            batch = inputs_embeds.size(0)
            seq_enc = inputs_embeds.size(1)
            device = inputs_embeds.device
        

        with torch.no_grad():
            if attention_mask is not None:

                attention_mask = attention_mask.to(torch.bool)
            else:
                attention_mask = torch.arange(seq_enc, device=device)[None, :].repeat(batch, 1) < length[:, None]
            # (batch, seq_enc, seq_enc)
            enc_attention_mask = attention_mask.view(batch, seq_enc, 1) & attention_mask.view(batch, 1, seq_enc)

        # (num_heads, seq_enc, seq_enc)
        enc_position_bias = self.position_bias_enc(seq_enc, seq_enc)

        # (batch, dim_model, seq_enc)
        if inputs_embeds is None:
            hidden_states_enc = self.input_embedding(input_ids)
        else:
            hidden_states_enc = inputs_embeds

        # (batch, dim_model, seq_enc)
        # print(enc_attention_mask.shape)
        # encoder_outputs, all_enc_hidden = self.encoder(hidden_states_enc, enc_attention_mask, enc_position_bias, deltas, parameters, adapters=enc_adapters)
        encoder_outputs = self.encoder(hidden_states_enc, enc_attention_mask, enc_position_bias, None, None, adapters=enc_adapters, lora=enc_lora)


        return encoder_outputs
