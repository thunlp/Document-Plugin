import argparse
import os
import torch
import logging

from tools.init_tool import init_all
from config_parser import create_config
from tools.train_tool import train
import bmtrain as bmt
from bmtrain import print_rank
import re

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def parse_hyper_para(setting, config):
    if setting is None:
        return None
    pat = re.compile("\+(.*?)=(.*?)=(.*?)\+")
    paras = pat.findall(setting)
    for para in paras:
        print_rank("add", para)
        config.set(para[0], para[1], para[2])

def print_config(config):
    for sec in config.sections():
        print_rank("[%s]" % sec)
        for op in config.options(sec):
            print_rank("%s: %s" % (op, config.get(sec, op)))
        print_rank("========")
    
from transformers import T5ForConditionalGeneration
from transformers import T5Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--only_eval', action="store_true")
    parser.add_argument('--do_test', help="do test while training or not", action="store_true")
    parser.add_argument('--hyper_para', "-hp", default=None)
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    parser.add_argument('--skip_step', type=int, default=0)
    
    args = parser.parse_args()

    configFilePath = args.config

    config = create_config(configFilePath)

    gpu_list = []

    use_gpu = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device_list = args.gpu.split(",")
    for a in range(0, len(device_list)):
        gpu_list.append(int(a))

    bmt.init_distributed(seed=1003)
    print_rank(args.hyper_para)
    parse_hyper_para(args.hyper_para, config)

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.error("CUDA is not available but specific gpu id")
        raise NotImplementedError

    import sys
    sys.setrecursionlimit(512 * 400 + 10)

    parameters = init_all(config, gpu_list, args.checkpoint, "train", skip_step=args.skip_step)
    parameters["skip-step"] = args.skip_step
    do_test = False
    if args.do_test:
        do_test = True
    
    print_config(config)

    train(parameters, config, gpu_list, do_test, args.only_eval)
