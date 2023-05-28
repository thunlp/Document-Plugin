from typing import Optional

import torch
import bmtrain as bmt
import math
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_out, length_scale=True, layer_norm=False, init_std=0.02, dim_mid=None, bias=False):
        super().__init__()
        if layer_norm:
            self.layer_norm = LayerNorm(dim_in, bias=False, eps=1e-6, group="mapper")
        else:
            self.layer_norm = None
        self.dim_mid = dim_in if dim_mid is None else dim_mid
        self.layer1 = Linear(dim_in=dim_in, dim_out=self.dim_mid, init_std=init_std, length_scale=length_scale, bias=bias, group="mapper")
        self.layer2 = Linear(dim_in=self.dim_mid, dim_out=dim_out, init_std=init_std, length_scale=length_scale, bias=bias, group="mapper")
        self.act = torch.nn.ReLU()

    def forward(self, rep: torch.Tensor):
        if self.layer_norm is not None:
            return self.layer2(self.act(self.layer1(self.layer_norm(rep))))
        else:
            return self.layer2(self.act(self.layer1(rep)))


class Linear(bmt.DistributedModule):
    r"""A fully connected layer, which performs :math:`\pmb{y} = \mathbf{W} \pmb{x} + \pmb{b}`

    Args:
        dim_in (int): input dimension of :math:`\pmb{x}`
        dim_out (int): output dimension of :math:`\pmb{y}`
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 1.
        bias (bool, optional): whether to add bias term :math:`\pmb{b}`. Defaults to False.
    """
    def __init__(self,
                 dim_in : int,
                 dim_out : int,
                 length_scale : bool = False,
                 length_scale_before : bool = False,
                 dtype = torch.half,
                #  dtype = torch.float32,
                 int8 : bool = False,
                 init_mean : float = 0.0,
                 init_std : float = 1,
                 bias : bool = False,
                 group : Optional[str] = None,
                ):
        super().__init__()
        self.dim_in = dim_in
        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            # init_method=bmt.ParameterInitializer(torch.nn.init.kaiming_uniform_, a=init_std),
            group = group,
        )
        self.bias = bmt.DistributedParameter(
            torch.empty((dim_out,), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.zeros_),
            group = group,
        ) if bias else None
        self.length_scale = length_scale
        self.length_scale_before = length_scale_before
        self.int8 = int8

    def forward(self, x : torch.Tensor):
        """ 
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer

        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.

        """
        if self.length_scale and self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        x = F.linear(x, self.weight)
        if self.length_scale and not self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        if self.bias is not None:
            x = x + self.bias
        return x



class KaimingLinear(bmt.DistributedModule):
    r"""A fully connected layer, which performs :math:`\pmb{y} = \mathbf{W} \pmb{x} + \pmb{b}`

    Args:
        dim_in (int): input dimension of :math:`\pmb{x}`
        dim_out (int): output dimension of :math:`\pmb{y}`
        dtype (optional): Defaults to torch.half.
        init_mean (float, optional): mean of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 0.
        init_std (float, optional): std of :math:`\mathbf{W}\sim\mathcal{N}(\text{mean}, \text{std}^2)`. Defaults to 1.
        bias (bool, optional): whether to add bias term :math:`\pmb{b}`. Defaults to False.
    """
    def __init__(self,
                 dim_in : int,
                 dim_out : int,
                 length_scale : bool = False,
                 length_scale_before : bool = False,
                 dtype = torch.half,
                 int8 : bool = False,
                 init_a : float = 1,
                 bias : bool = False,
                 group : Optional[str] = None,
                 zeros : bool = False
                ):
        super().__init__()
        self.dim_in = dim_in
        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            # init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std),
            init_method=bmt.ParameterInitializer(torch.nn.init.kaiming_uniform_, a=init_a) if not zeros else bmt.ParameterInitializer(torch.nn.init.zeros_),
            group = group,
        )
        self.bias = bmt.DistributedParameter(
            torch.empty((dim_out,), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.zeros_),
            group = group,
        ) if bias else None
        self.length_scale = length_scale
        self.length_scale_before = length_scale_before
        self.int8 = int8

    def forward(self, x : torch.Tensor):
        """ 
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer

        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.

        """
        if self.length_scale and self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        x = F.linear(x, self.weight)
        if self.length_scale and not self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        if self.bias is not None:
            x = x + self.bias
        return x



class LayerNorm(bmt.DistributedModule):
    r"""
    `LayerNorm <https://arxiv.org/abs/1607.06450>`_ if bias = True: :math:`y = {x-\text{E}[x]\over \text{Var}[x]+\text{eps}} * w + \text{bias}`

    `RMS LayerNorm <https://arxiv.org/abs/1910.07467>`_ if bias = False: :math:`y = {x\over \text{Var}[x]+\text{eps}} * w`

    Args:
        dim_norm (int): norm dimesion
        dtype (optional): Defaults to torch.half.
        bias (bool, optional): whether to add the :math:`\text{bias}` term. Defaults to True.
        eps (float, optional): :math:`\text{eps}` term. Defaults to 1e-5.
        init_var (float, optional): weight will be all initialized to init_var. Defaults to 1.0.
    """
    def __init__(self, dim_norm : int, 
                       dtype=torch.half, 
                       bias=True, 
                       eps : float = 1e-5,
                       init_var = 1.0,
                       group : Optional[str] = None,
                       ):

        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = bmt.DistributedParameter(
            torch.ones(dim_norm, dtype=dtype) * init_var, group=group)
        self.bias = bmt.DistributedParameter(
            torch.zeros(dim_norm, dtype=dtype), group=group) if bias else None
    
    def forward(self, x : torch.Tensor):
        """ 
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``): Input tensor that need to be normalized.

        Return:
            :obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_norm)``: The layernorm output. 

        """
        assert x.size(-1) == self.dim_norm
        
        if self.bias is not None:
            return F.layer_norm(x, (self.dim_norm,), self.weight, self.bias, self.eps)
        else:
            return rms_layernorm(x, self.weight, self.eps)

@torch.jit.script
def rms_layernorm(hidden : torch.Tensor, weight : torch.Tensor, eps :float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight