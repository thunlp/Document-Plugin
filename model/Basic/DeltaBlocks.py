
import torch

from model_center.layer import Attention,LayerNorm,FeedForward,Linear
import bmtrain as bmt
from typing import *
import torch.nn.functional as F
from tools import print_rank
import math

class LoRAAttention(bmt.DistributedModule):
    def __init__(self, dim_in : int, 
                       dim_head : int,
                       num_heads : int, 
                       dim_out : int = None,
                       dtype = torch.half,
                       int8 = False, 
                       init_mean = 0.0, 
                       init_std = 0.02,
                       bias = False,
                       mask_value : float = float("-inf"),
                       pos_bias_type : str = "none",
                       length_scale : bool = False,
                       attn_scale : bool = False,
                       dropout_p : float= 0,
                       shared_key_and_value = False,
                       layer_no : int = -1,
        ):

        super().__init__()

        if dim_out is None:
            dim_out = dim_in

        num_heads_kv = 1 if shared_key_and_value else num_heads 

        self.project_q = Linear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.project_k = Linear(
            dim_in = dim_in,
            dim_out = num_heads_kv * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.project_v = Linear(
            dim_in = dim_in,
            dim_out = num_heads_kv * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.attention_out = Linear(
            dim_in = num_heads * dim_head,
            dim_out = dim_out,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )
        self.init_mean = init_mean
        self.init_std = init_std
        self.dim_in = dim_in
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.dim_out = dim_out
        self.int8 = int8
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.mask_value = mask_value
        self.dtype = dtype
        self.dropout_p = dropout_p
        self.shared_key_and_value = shared_key_and_value

        if dropout_p:
            self.attention_dropout = torch.nn.Dropout(dropout_p)
        else:
            self.attention_dropout = None
        
        self.bias = bias
        self.pos_bias_type = pos_bias_type
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layer_no = layer_no

    def cal_lora(self, map1: torch.Tensor, map2: torch.Tensor, hiddens: torch.Tensor):
        '''
        map1: batch, hidden_size, lora_rank
        map2: batch, lora_rank, hidden_size
        hiddens: batch, seq_len, hidden_size
        '''
        x = torch.matmul(hiddens, map1)
        x = torch.matmul(x, map2) 

        return hiddens + x * 2

    def forward(self, query : torch.Tensor,
                      key_value : torch.Tensor,
                      attention_mask : torch.Tensor,
                      position_bias : Optional[torch.Tensor] = None,
                      lora: torch.Tensor = None,
                      use_cache: bool = False,
                      past_key_value = None,
        ):

        batch_size = query.size(0)
        len_q = query.size(1)
        len_k = key_value.size(1)

        h_q = self.project_q(query)             # (batch, len_q, num_heads * dim_head)
        h_k = self.project_k(key_value)         # (batch, len_k, num_heads * dim_head)
        h_v = self.project_v(key_value)         # (batch, len_k, num_heads * dim_head)

        if lora is not None:
            layer_lora = lora[self.layer_no]
            query_A, query_B = layer_lora[0], layer_lora[2].transpose(1, 2)
            key_A, key_B = layer_lora[1], layer_lora[3].transpose(1, 2)
            h_q = self.cal_lora(query_A, query_B, h_q)
            h_k = self.cal_lora(key_A, key_B, h_k)

        h_q = h_q.view(batch_size, len_q, self.num_heads, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads, len_q, dim_head)
        h_k = h_k.view(batch_size, len_k, self.num_heads_kv, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads_kv, len_k, dim_head)
        h_v = h_v.view(batch_size, len_k, self.num_heads_kv, self.dim_head).permute(0, 2, 1, 3)   # (batch, num_heads_kv, len_k, dim_head)

        # if self.shared_key_and_value:
        #     h_k = h_k.repeat(1, self.num_heads, 1, 1)
        #     h_v = h_v.repeat(1, self.num_heads, 1, 1)

        h_q = h_q.contiguous()      # (batch * num_heads, len_q, dim_head)
        h_k = h_k.contiguous()      # (batch * num_heads, len_k, dim_head)
        h_v = h_v.contiguous()      # (batch * num_heads, len_k, dim_head)

        if past_key_value is not None:
            h_k = torch.cat([past_key_value[0], h_k], dim=-2)
            h_v = torch.cat([past_key_value[1], h_v], dim=-2)
            len_k = h_k.size(-2)

        current_key_value = (h_k, h_v) if use_cache else None

        if self.pos_bias_type == "rotary":
            h_q, h_k = position_bias(h_q, h_k)

        # (batch, num_heads, len_q, dim_head) @ (batch, num_heads_kv, len_k, dim_head)T 
        # => (batch, num_heads, len_q, len_k)
        
        score = torch.matmul(h_q, h_k.transpose(2, 3))
        if self.attn_scale:
            score = score / math.sqrt(self.dim_head)

        # (batch, num_heads, len_q, len_k) 
        # score = score.view(batch_size, self.num_heads, len_q, len_k)

        if self.pos_bias_type == "relative":
            if position_bias is not None:
                # (batch, num_heads, len_q, len_k) + (1, num_heads, len_q, len_k) 
                score = score + position_bias
                # if position_bias.shape[1] != position_bias[2]:
                #     print_rank(position_bias.shape, position_bias[0,0].tolist())
                #     print_rank(position_bias[0,12].tolist())
                #     print_rank("==========")
                
        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(self.mask_value, device=score.device, dtype=score.dtype)
        )   # (batch, num_heads, len_q, len_k)

        score = self.softmax(score)

        # avoid nan in softmax
        score = torch.masked_fill(
            score,
            attention_mask.view(batch_size, 1, len_q, len_k)==False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype)
        )
        #.view(batch_size * self.num_heads, len_q, len_k) # (batch * num_heads, len_q, len_k)

        if self.attention_dropout is not None:
            score = self.attention_dropout(score)

         # (batch * num_heads, len_q, len_k) @ (batch * num_heads, len_k, dim_head) = (batch * num_heads, len_q, dim_head)
        score = torch.matmul(score, h_v)

        score = score.view(batch_size, self.num_heads, len_q, self.dim_head).permute(0, 2, 1, 3) # (batch, len_q, num_heads, dim_head)
        score = score.reshape(batch_size, len_q, self.num_heads * self.dim_head) # (batch, len_q, num_heads * dim_head)

        # (1#batch, dim_model, num_heads * dim_head) @ (batch, num_heads * dim_head, len_q) = (batch, dim_model, len_q)
        score = self.attention_out(score)

        if use_cache:
            return score, current_key_value
        else:
            return score



class SelfAttentionDeltaBlock(torch.nn.Module):
    """  The whole cross-attention block. A sequence of operation. Consists of layernorm, self-attention and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        num_heads (int): num_heads used in :py:class:`model_center.layer.Attention`.
        dim_head (int): dim_head used in :py:class:`model_center.layer.Attention`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        att_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.Attention`. Defaults to 0.0.
        att_init_std (float, optional): init_std used in :py:class:`model_center.layer.Attention`. Defaults to 0.02.
        att_bias (bool, optional): bias used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        att_mask_value (float, optional): mask_value used in in :py:class:`model_center.layer.Attention`. Defaults to float("-inf").
        pos_bias_type (str, optional): pos_bias_type used in :py:class:`model_center.layer.Attention`. Defaults to "none".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        attn_scale (bool, optional): attn_scale used in in :py:class:`model_center.layer.Attention`. Defaults to False.
        dropout_p (float, optional): Defaults to 0.
    """
    def __init__(self, 
                 dim_model : int, num_heads : int, dim_head : int, 
                 dtype = torch.half, int8 = False, 
                 norm_init_var : float = 1.0, norm_bias : bool = False, norm_eps : float = 1e-5, 
                 att_init_mean : float = 0.0, att_init_std : float = 0.02, att_bias : bool = False, att_mask_value : float = float("-inf"),
                 pos_bias_type : str = "none",
                 post_layer_norm : bool = False,
                 length_scale : bool = False, attn_scale : bool = False, dropout_p : float = 0,
                 layer_no : int = -1,
                ):

        super().__init__()

        self.layernorm_before_attention = LayerNorm(
            dim_norm = dim_model, bias = norm_bias, dtype = dtype, eps = norm_eps, init_var = norm_init_var,
        )

        self.self_attention = LoRAAttention(
            dim_in = dim_model, num_heads = num_heads, dim_head = dim_head, dim_out = dim_model, 
            dtype = dtype, int8 = int8, 
            init_mean = att_init_mean, init_std = att_init_std, bias = att_bias,
            mask_value = att_mask_value,
            pos_bias_type = pos_bias_type,
            length_scale = length_scale, attn_scale = attn_scale, dropout_p = dropout_p,
            layer_no=layer_no
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.post_layer_norm = post_layer_norm
        self.layer_no = layer_no
    
    def cal_adapter(self, map1: torch.Tensor, map2: torch.Tensor, hiddens: torch.Tensor):
        '''
        map1: batch, hidden_size, adapter_rank
        map2: batch, adapter_rank, hidden_size
        hiddens: batch, seq_len, hidden_size
        '''
        x = torch.matmul(hiddens, map1)
        x = F.relu(x)
        x = torch.matmul(x, map2) 

        return hiddens + x

    def forward(self,
                hidden_states : torch.Tensor,
                attention_mask : torch.Tensor,
                position_bias : Optional[torch.Tensor] = None,
                prefix : Optional[torch.Tensor] = None,
                adapters : Optional[torch.Tensor] = None,
                lora : Optional[torch.Tensor] = None,
            ):
        """
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Input of self-attention block. It can be the embedding of a batch of sequences.
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_self, seq_self)``): Avoid invalid areas to participate in the calculation.  
            position_bias (:obj:`torch.Tensor` of shape ``(num_heads, seq_self, seq_self)``): Provide positional information to self-attention block.
        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of attention block.
        """    
        # print("hidden_states", hidden_states)
        x = self.layernorm_before_attention(hidden_states)

        if adapters is not None and adapters.size(1) == 4:
            layer_adapter = adapters[self.layer_no] #2, batch, hidden, delta_rank
            map1, map2 = layer_adapter[1], layer_adapter[3].transpose(1, 2)
            x = self.cal_adapter(map1, map2, x)

        if self.post_layer_norm:
            hidden_states = x
        if prefix is not None:
            batch, prefix_num, _ = prefix.size()
            seq_len = hidden_states.size(1)
            seq = torch.cat([x, self.layernorm_before_attention(prefix)], dim=1).contiguous()
            # print_rank("=" * 15 + str(self.layer_no) + " with doc_plug" + "=" * 15)
            x = self.self_attention(x, seq, attention_mask, position_bias, lora=lora)
            # print_rank("=" * 35)
        else:
            x = self.self_attention(x, x, attention_mask, position_bias, lora=lora)
            pfx_out = None
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        return hidden_states

class FFNDeltaBlock(torch.nn.Module):
    """ The whole feed-forward block. A sequence of operation. Consists of layernorm, feed-forward and residual connection.

    Args:
        dim_model (int): main dimension of modules in transformer blocks.
        dim_ff (int): dim_ff used in :py:class:`model_center.layer.FeedForward`.
        dtype (optional): Defaults to torch.half.
        norm_init_var (float, optional): init_var used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1.0.
        norm_bias (bool, optional): bias used in :py:class:`model_center.layer.LayerNorm`. Defaults to False.
        norm_eps (float, optional): eps used in :py:class:`model_center.layer.LayerNorm`. Defaults to 1e-5.
        ffn_init_mean (float, optional): init_mean used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.0.
        ffn_init_std (float, optional): init_std used in :py:class:`model_center.layer.FeedForward`. Defaults to 0.02.
        ffn_bias (bool, optional): bias used in :py:class:`model_center.layer.FeedForward`. Defaults to False.
        ffn_activate_fn (str, optional): activate_fn used in :py:class:`model_center.layer.FeedForward`. Defaults to "gated_gelu".
        post_layer_norm (bool, optional): whether to use post-layernorm. Defaults to False, which means pre-layernorm.
        dropout_p (float, optional): Defaults to 0.
    """

    def __init__(self, 
                 dim_model : int, 
                 dim_ff : int,
                 dtype = torch.half, 
                 int8 = False,
                 norm_init_var : float = 1.0,
                 norm_bias : bool = False,
                 norm_eps : float = 1e-5, 
                 ffn_init_mean : float = 0.0, 
                 ffn_init_std : float = 0.02,
                 ffn_bias : bool = False,
                 ffn_activate_fn : str = "gated_gelu",
                 post_layer_norm : bool = False,
                 length_scale : bool = False,
                 dropout_p : float = 0,
                 layer_no : int = -1,
                ):
        super().__init__()

        self.layernorm_before_ffn = LayerNorm(
            dim_norm = dim_model, 
            bias = norm_bias, 
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )

        self.ffn = FeedForward(
            dim_in = dim_model, 
            dim_ff = dim_ff, 
            dim_out = dim_model, 
            dtype = dtype, 
            int8 = int8,
            init_mean = ffn_init_mean, 
            init_std = ffn_init_std,
            bias = ffn_bias,
            activate_fn = ffn_activate_fn,
            length_scale = length_scale,
        )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.post_layer_norm = post_layer_norm
        self.layer_no = layer_no

    def cal_adapter(self, map1: torch.Tensor, map2: torch.Tensor, hiddens: torch.Tensor):
        '''
        map1: batch, hidden_size, adapter_rank
        map2: batch, adapter_rank, hidden_size
        hiddens: batch, seq_len, hidden_size
        '''
        x = torch.matmul(hiddens, map1)
        x = F.relu(x)
        x = torch.matmul(x, map2) 

        return hiddens + x

    def forward(self,
                hidden_states : torch.Tensor,
                adapters = None,
               ):
        """ 
        Args:
            hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``): Hidden states before feed forward layer.

        Return:
            :obj:`torch.Tensor` of shape ``(batch, seq_self, dim_model)``: The output of feed-forward block

        """ 
        x = self.layernorm_before_ffn(hidden_states)

        if adapters is not None:
            layer_adapter = adapters[self.layer_no] #2, batch, hidden, delta_rank
            if layer_adapter.size(0) == 2:
                map1, map2 = layer_adapter[0], layer_adapter[1].transpose(1, 2)
            else:
                map1, map2 = layer_adapter[0], layer_adapter[2].transpose(1, 2)
            x = self.cal_adapter(map1, map2, x)

        if self.post_layer_norm:
            hidden_states = x
        x = self.ffn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = hidden_states + x
        return hidden_states
