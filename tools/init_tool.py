import logging
import torch

from reader.reader import init_dataset, init_formatter, init_test_dataset
from model import get_model
from model.optimizer import init_optimizer
from .output_init import init_output_function
from torch import nn
from tools import output_log
import bmtrain as bmt
from bmtrain import print_rank
from transformers import get_linear_schedule_with_warmup
from bmtrain.store import DistributedStateDictWrapper
from model.scheduler import T5Scheduler
logger = logging.getLogger(__name__)


def init_all(config, gpu_list, checkpoint, mode, *args, **params):
    result = {}

    output_log(logger, "Begin to initialize dataset and formatter...")
    if mode == "train":
        # init_formatter(config, ["train", "valid"], *args, **params)
        result["train_dataset"], result["valid_dataset"] = init_dataset(config, *args, **params)
    else:
        # init_formatter(config, ["test"], *args, **params)
        result["test_dataset"] = init_test_dataset(config, *args, **params)

    output_log(logger, "Begin to initialize models...")

    model = get_model(config.get("model", "model_name"))(config, gpu_list, *args, **params)
    optimizer = init_optimizer(model, config, *args, **params)

    lrsche_type = config.get("train", "scheduler")
    if lrsche_type == "linear":
        lr_scheduler = bmt.lr_scheduler.Linear(optimizer, start_lr=config.getfloat('train', 'learning_rate'), warmup_iter=config.getint('train', 'warmup_steps'), end_iter=config.getint('train', 'training_steps'))
    elif lrsche_type == "t5":
        lr_scheduler = T5Scheduler(optimizer, start_lr=config.getfloat('train', 'learning_rate'), warmup_iter=config.getint('train', 'warmup_steps'), end_iter=config.getint('train', 'training_steps'), num_iter=params["skip_step"])
    # lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.getint('train', 'warmup_steps'), num_training_steps=config.getint('train', 'training_steps'))
    trained_epoch = 0
    global_step = 0


    # model = model.to(gpu_list[bmt.rank()])
    try:
        # print("read checkpoint", bmt.rank())
        # for i in range(bmt.world_size()):
        #     if bmt.rank() == i:
        #         parameters = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        #     bmt.synchronize()
        # print("load checkpoint", parameters.keys(), bmt.rank())
        # if bmt.rank() == 0:
        #     model.load_state_dict(DistributedStateDictWrapper(parameters["model"]))
        # else:
        #     print_rank(bmt.rank(), "load_state_dict")
        #     model.load_state_dict({})
        output_log(logger, "try load checkpoint from %s" % checkpoint, logging.INFO)
        if checkpoint is not None and checkpoint != "None":
            parameters = {"trained_epoch": 0}
            print_rank(bmt.load(model, checkpoint, strict=False))
            if mode == "train":
                trained_epoch = parameters["trained_epoch"]
                if "optimizer" in parameters and config.get("train", "optimizer") == parameters["optimizer_name"]:
                    optimizer.load_state_dict(parameters["optimizer"])
                else:
                    output_log(logger, "Optimizer changed, do not load parameters of optimizer.", logging.WARNING)
                if "global_step" in parameters:
                    global_step = parameters["global_step"]
                if "lr_scheduler" in parameters:
                    lr_scheduler.load_state_dict(parameters["lr_scheduler"])
        if config.get("model", "adapter_path") != "None":
            print_rank(bmt.load(model, config.get("model", "adapter_path"), strict=False))
    except Exception as e:
        information = "Cannot load checkpoint file with error %s" % str(e)
        if mode == "test":
            output_log(logger, information, logging.ERROR)
            raise e
        else:
            output_log(logger, information, logging.WARNING)

    # model = bmt.BMTrainModelWrapper(model)
    result["model"] = model
    if mode == "train":
        result["optimizer"] = optimizer
        result["lr_scheduler"] = lr_scheduler
        result["trained_epoch"] = trained_epoch
        result["output_function"] = init_output_function(config)
        result["global_step"] = global_step

    output_log(logger, "Initialize done.", logging.INFO)

    return result
