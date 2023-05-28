import logging
import os
from threading import local
from typing import List
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from timeit import default_timer as timer
from bmtrain import print_rank
import bmtrain as bmt
from tools import reduce
from kara_storage.pytorch.base import KaraPytorchDatasetBase
logger = logging.getLogger(__name__)


def gen_time_str(t):
    t = int(t)
    minute = t // 60
    second = t % 60
    return '%2d:%02d' % (minute, second)


def output_value(epoch, mode, step, time, loss, info, end, config, lr="", otherinfo=""):
    try:
        delimiter = config.get("output", "delimiter")
    except Exception as e:
        delimiter = " "
    s = ""
    s = s + str(epoch) + " "
    while len(s) < 10:
        s += " "
    s = s + str(mode) + " "
    while len(s) < 18:
        s += " "
    s = s + str(step) + " "
    while len(s) < 30:
        s += " "
    s += str(time)
    while len(s) < 50:
        s += " "
    s += str(loss)
    while len(s) < 58:
        s += " "
    s += str(info)
    s = s.replace(" ", delimiter)
    s += "\t%s" % lr
    s += "\t%s" % otherinfo
    if not (end is None):
        print_rank(s, end=end)
    else:
        print_rank(s)


def valid(model, dataset, epoch, config, gpu_list, output_function, mode="valid"):
    model.eval()
    local_rank = bmt.rank() #config.getint('distributed', 'local_rank')
    acc_result = None
    total_loss = 0
    cnt = 0
    total_len = len(dataset)
    start_time = timer()
    output_info = ""

    output_time = config.getint("output", "output_time")
    step = -1

    if hasattr(dataset, "dataset") and isinstance(dataset.dataset, KaraPytorchDatasetBase): 
        dataset.dataset.set_epoch(0)

    for step, data in enumerate(dataset):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if len(gpu_list) > 0:
                    data[key] = Variable(data[key].cuda())
                else:
                    data[key] = Variable(data[key])
        results = model(data, config, gpu_list, acc_result, mode=mode)

        loss, acc_result = results["loss"], results["acc_result"]
        total_loss += bmt.sum_loss(loss).item()
        cnt += 1
        if step % output_time == 0:
            delta_t = timer() - start_time

            output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, None, config)
    if step == -1:
        logger.error("There is no data given to the model in this epoch, check your data.")
        raise NotImplementedError

    if acc_result is not None and config.getboolean("distributed", "use") and type(acc_result) != list:
        if "train" in acc_result:
            acc_result.pop("train")
        total_loss = bmt.sum_loss(torch.tensor(total_loss).cuda()).item()
        for key in acc_result:
            if type(acc_result[key]) == list:
                continue
            acc_result[key] = reduce(torch.tensor(acc_result[key]).cuda(), "sum").item()
        acc_result["train"] = False
    else:
        total_loss = bmt.sum_loss(torch.tensor(total_loss).cuda()).item()

    delta_t = timer() - start_time
    output_info = output_function(acc_result, config)
    output_value(epoch, mode, "%d/%d" % (step + 1, total_len), "%s/%s" % (
        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

    model.train()
