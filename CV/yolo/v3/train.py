import argparse
import os
from datetime import datetime
import tqdm
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import datasets

from utils import config


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path",        type=str,   default='configs/yolov3.yaml',  help="path to config file")
    parser.add_argument("--log-path",           type=str,   default="./logs",               help="path to log file")
    parser.add_argument("--resume",             type=str,   default=None,                   help="path to checkpoint file")
    parser.add_argument("--device",             type=str,   default="cuda",                 help="device to use", choices=["cuda", "cpu"])
    parser.add_argument("-j", "--num-workers",  type=int,   default=4,                      help="number of workers")
    parser.add_argument("--data-root",          type=str,   default="./data",               help="path to dataset")
    parser.add_argument("--image-size",         type=int,   default=416,                    help="size of each image dimension")
    parser.add_argument("--multiscale",         action="store_true",                        help="use multiscale")
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = config_parser()
    config_path     = args.config_path
    log_path        = args.log_path
    resume          = args.resume
    device          = args.device
    num_workers     = args.num_workers
    data_root       = args.data_root
    image_size      = args.image_size
    multiscale      = args.multiscale

    config.check_paths(config_path, log_path, resume, data_root)
    logger = SummaryWriter(os.path.join(log_path, datetime.now().strftime("%Y%m%d-%H%M%S")))
    model_config, data_config, train_config = config.get_config(config_path)