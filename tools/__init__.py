import torch.distributed as dist
import logging
from bmtrain.global_var import config as bmtconfig
from bmtrain import nccl
import torch
from transformers.file_utils import is_torch_fx_proxy

def output_log(logger: logging.Logger, info: str, level: int = logging.INFO, *args):
    if not (dist.is_initialized() and dist.get_rank() != 0):
        logger._log(level, info, args)

def print_rank(*arg):
    if not (dist.is_initialized() and dist.get_rank() != 0):
        print(*arg)

def reduce(var : torch.Tensor, op: str = "avg"):
    ret = torch.empty_like(var)
    nccl.allReduce(
        var.storage(),
        ret.storage(),
        op,
        bmtconfig['comm']
    )
    return ret

def shift_tokens_right(input_ids, pad_token_id: int, decoder_start_token_id: int):

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
