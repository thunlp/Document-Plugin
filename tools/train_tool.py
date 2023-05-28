
import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler as lrs
from timeit import default_timer as timer

from tools.eval_tool import valid, gen_time_str, output_value
from tools.init_tool import init_test_dataset, init_formatter
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast
from kara_storage.pytorch.base import KaraPytorchDatasetBase
from tools import output_log
import bmtrain as bmt
from bmtrain import print_rank
from bmtrain.store import _save_to_rank0
import math
logger = logging.getLogger(__name__)


def checkpoint(filename, model, optimizer, trained_epoch, config, global_step, lr_scheduler):
    torch.cuda.synchronize()
    save_params = _save_to_rank0(model)
    try:
        if bmt.rank() == 0:
            torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


def train(parameters, config, gpu_list, do_test=False, only_eval=False):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")
    output_grad = config.getboolean("output", "output_grad")

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        output_log(logger, "Output path exists, check whether need to change a name of model", logging.WARNING)
        # logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"] + 1
    optim_manager = bmt.optim.OptimManager(loss_scale=2**20)
    model, optimizer, dataset, lr_scheduler = parameters["model"], parameters["optimizer"], parameters["train_dataset"], parameters["lr_scheduler"]
    global_step, output_function = parameters["global_step"], parameters["output_function"]
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    if do_test:
        init_formatter(config, ["test"])
        test_dataset = init_test_dataset(config)
    grad_accumulate = config.getint("train", "grad_accumulate")
    output_grad_step = config.getint("output", "output_grad_step")

    max_grad_norm = config.getfloat('train', 'max_grad_norm')
    valid_mode = config.get('train', 'valid_mode')
    if valid_mode != 'step' and valid_mode != 'batch':
        raise ValueError('The value of valid_mode is invalid.')
    no_valid = config.getboolean("train", "no_valid")

    print_rank('valid_mode', valid_mode, "no_valid", no_valid)
    step_epoch = None
    if valid_mode == 'step':
        step_epoch = config.getint('train', 'step_epoch')
    save_step = config.getint("train", "save_step")

    print_rank('step_epoch', step_epoch)
    output_log(logger, "Start training", logging.INFO)

    print_rank("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")
    # print("begin iteration", bmt.rank())
    total_len = len(dataset)
    for epoch_num in range(trained_epoch, epoch):
        model.train()
        start_time = timer()
        current_epoch = epoch_num
        # print("begin iteration", bmt.rank())
        
        acc_result = None
        total_loss = 0

        output_info = ""
        step = -1
        if hasattr(dataset, "dataset") and isinstance(dataset.dataset, KaraPytorchDatasetBase): 
            dataset.dataset.set_epoch(epoch_num)

        if not only_eval:
            for step, data in enumerate(dataset):
                # print(bmt.rank(), "step")
                if epoch_num == 1 and step < parameters["skip-step"]:
                    print_rank("skip step %s" % step, end="\r")
                    continue
                for key in data.keys():
                    if isinstance(data[key], torch.Tensor):
                        if len(gpu_list) > 0:
                            data[key] = Variable(data[key].cuda())
                        else:
                            data[key] = Variable(data[key])
                # break
                results = model(data, config, gpu_list, acc_result, "train")

                loss, acc_result = results["loss"], results["acc_result"]
                total_loss += bmt.sum_loss(loss).item()

                loss = loss / grad_accumulate
                # loss = optimizer.loss_scale(loss)
                # loss.backward()
                optim_manager.backward(loss)

                if output_grad and step % output_grad_step == 0:
                    inpsector = config.get("train", "inspector_para")
                    bmt.print_rank(bmt.inspect.format_summary(bmt.inspect.inspect_model(model, inpsector)))
                    # bmt.print_rank(bmt.inspect.format_summary(bmt.inspect.inspect_model(model, "que_model.*")))
                
                grad_norm=None
                if (step + 1) % grad_accumulate == 0:
                    if max_grad_norm is not None and max_grad_norm > 0:
                        grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, max_grad_norm, norm_type=2)
                        grad_norm = grad_norm.item()
                    optim_manager.step()
                    optim_manager.zero_grad()

                if step % output_time == 0:
                    output_info = output_function(acc_result, config)

                    delta_t = timer() - start_time

                    output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                                "%.3lf" % (total_loss / (step + 1 - parameters["skip-step"])), output_info, None, config, lr_scheduler.current_lr, "grad_norm: %s" % grad_norm)

                if save_step > 0 and step > 0 and step % save_step == 0:
                    print_rank("=" * 10, "saving model", "=" * 10)
                    checkpoint(os.path.join(output_path, f"{current_epoch}-{step}.pkl"), model, optimizer, current_epoch, config, global_step, lr_scheduler)

                global_step += 1
                if (step + 1) % grad_accumulate == 0 and valid_mode == 'step' and int((step + 1) / grad_accumulate) % step_epoch == 0:
                    acc_result = None
                    checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config, global_step, lr_scheduler)
                    if not no_valid:
                        with torch.no_grad():
                            valid(model, parameters["valid_dataset"], current_epoch, config, gpu_list, output_function)
                            if do_test:
                                valid(model, test_dataset, current_epoch, config, gpu_list, output_function, mode="test")

            if step == -1:
                output_log(logger, "No data in this epoch", logging.ERROR)
                # logger.error("There is no data given to the model in this epoch, check your data.")
                raise NotImplementedError

            print_rank(valid_mode != "batch", no_valid)
            if (valid_mode != "batch") or no_valid:
                print_rank("skip validation")
                continue

            output_info = output_function(acc_result, config)
            delta_t = timer() - start_time
            output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                        "%.3lf" % (total_loss / (step + 1)), output_info, None, config)
            print_rank("==" * 10, "begin saving model and validation", "==" * 10)


            checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config, global_step, lr_scheduler)
        # if only_eval:
        #     ckp = "../checkpoint/ELI5PlugD/model-%s.pkl" % epoch_num
        #     print(bmt.load(model, ckp, strict=False))

        if current_epoch % test_time == 0:
            with torch.no_grad():
                valid(model, parameters["valid_dataset"], current_epoch, config, gpu_list, output_function)
                if do_test:
                    valid(model, test_dataset, current_epoch, config, gpu_list, output_function, mode="test")
