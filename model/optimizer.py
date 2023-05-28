import torch.optim as optim
import torch
from torch.optim import AdamW
import bmtrain as bmt

def get_params_for_prompt_optimization(module: torch.nn.Module):
    # params = [{"params": [], "lr": 5e-4}, {"params": [], "lr": 1e-5}]
    params = []
    names = []
    for t in module.named_modules():
        if "nograd" in t[0]: # or "doc2para2B" in t[0] or "doc2para1B" in t[0]:
            continue
        # if "mapper" in t[0]:
        #     params[0]["params"].extend([p for p in list(t[1]._parameters.values()) if p is not None])
        # else:
        #     params[1]["params"].extend([p for p in list(t[1]._parameters.values()) if p is not None])
        # params.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
        params.extend([p for p in list(t[1]._parameters.values()) if p is not None])
        names.append(t[0])

    return params, names

def init_optimizer(model, config, *args, **params):
    optimizer_type = config.get("train", "optimizer")
    learning_rate = config.getfloat("train", "learning_rate")

    if config.getboolean("train", "ignore_no_grad"):
        param_group, param_names = get_params_for_prompt_optimization(model)
        print("ignore parameters with nograd in name, and only %s parameters are turned" % len(param_group))
        # print(param_names)
    else:
        param_group = model.parameters()
        # param_group = [{"params": model.ctx_encoder.parameters(), "lr": 1e-3}, {"params": model.que_model.parameters(), "lr": 1e-4}]
        print("all parameters are turned")
    
    optimizer = bmt.optim.AdamOffloadOptimizer(param_group, lr=learning_rate,
                                weight_decay=config.getfloat("train", "weight_decay"))
    # if optimizer_type == "adam":
    #     optimizer = optim.Adam(param_group, lr=learning_rate,
    #                            weight_decay=config.getfloat("train", "weight_decay"))
    # elif optimizer_type == "sgd":
    #     optimizer = optim.SGD(param_group, lr=learning_rate,
    #                           weight_decay=config.getfloat("train", "weight_decay"))
    # elif optimizer_type == "AdamW":
    #     optimizer = AdamW(param_group, lr=learning_rate,
    #                          weight_decay=config.getfloat("train", "weight_decay"))
    # else:
    #     raise NotImplementedError

    return optimizer
