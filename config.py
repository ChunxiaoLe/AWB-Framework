# config.py
# 默认值（可被 train.py 覆盖）
import re

import numpy as np
import torch
import os
import argparse
import yaml
from datetime import datetime




# Input size
TRAIN_IMG_W, TRAIN_IMG_H = 128, 128
TEST_IMG_W, TEST_IMG_H = 128, 128

def get_device(DEVICE_TYPE) -> torch.device:
    if DEVICE_TYPE == "cpu":
        print("\n Running on device 'cpu' \n")
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", DEVICE_TYPE):
        if not torch.cuda.is_available():
            print("\n WARNING: running on cpu since device {} is not available \n".format(DEVICE_TYPE))
            return torch.device("cpu")

        print("\n Running on device '{}' \n".format(DEVICE_TYPE))
        return torch.device(DEVICE_TYPE)

    raise ValueError("ERROR: {} is not a valid device! Supported device are 'cpu' and 'cuda:n'".format(DEVICE_TYPE))





def parse_args():
    parser = argparse.ArgumentParser(description="Train illumination model")
    parser.add_argument( '--task',  default='train',  type=str,  choices=['train', 'test', 'demo'],
                         help='choose the task for running the model')

    "IE, IE-PS"
    parser.add_argument("--model_task", type=str, default="IE", help="path to config file")
    parser.add_argument("--config", type=str, default="configs.yaml", help="path to config file")
    parser.add_argument("--epochs", type=int, default=1500)

    "data realted"
    parser.add_argument("--train_bz", type=int, default=4)
    parser.add_argument("--test_bz", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=9)

    parser.add_argument("--type", type=str, default='original') # [original, re-parameterized]
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--init_para", type=bool, default=True)

    parser.add_argument("--device", type=str, default='cuda:1')

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_period', type=int, default=10)
    parser.add_argument('--lr_mult', type=int, default=2)
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--betas', type=float, default=(0.9, 0.999))
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min_delta', type=float, default=0.0001)

    "测试模型参数的路径，一般情况下是在./experiments"
    parser.add_argument("--test_model_para", type=str, default='/home/wsy/crosscamera/experiments/experiments/XXX')

    opt = parser.parse_args()

    opt = opt_format(opt)

    return opt


def load_yaml(path):
    with open(path, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    return model_config

def save_yaml(path, file_dict):
    with open(path, 'w') as f:
        f.write(yaml.dump(file_dict, allow_unicode=True))


def opt_format(opt):
    "将当前的工作目录的路径赋值给 `opt.root` 变量"
    opt.root = os.getcwd()
    # opt.config = r'{}/config/{}.yaml'.format(opt.root, opt.model_task)

    # 加载模型的预设参数并合并配置
    opt.config = load_yaml(opt.config)

    proper_time = str(datetime.now()).split('.')[0].replace(':', '-')

    # opt.config['exp_name'] = '{}_{}'.format(opt.task, opt.config['exp_name'])

    opt.experiments = r'{}/experiments/{}'.format(opt.root, '{} {} {}'.format(proper_time, opt.config['exp_name'], opt.task))
    if not os.path.exists(opt.experiments):
        os.mkdir(opt.experiments)

    config_path = r'{}/config.yaml'.format(opt.experiments)
    save_yaml(config_path, opt.config)

    if opt.task == 'demo' or (opt.task == 'test' and opt.config['test']['save'] != False):
        opt.save_image = True
        opt.save_image_dir = r'{}/{}'.format(opt.experiments, 'images')
        if not os.path.exists(opt.save_image_dir):
            os.mkdir(opt.save_image_dir)


    log_path = os.path.join(opt.experiments,opt.task)

    opt.log_path = r'{}/logger.log'.format(opt.experiments)


    opt.device = get_device(opt.device)

    if opt.task == 'train':
        opt.save_model = True
        opt.save_model_dir = r'{}/{}'.format(opt.experiments, opt.model_task, 'models')
        if not os.path.exists(opt.save_model_dir):
            os.mkdir(opt.save_model_dir)

    return opt



if __name__ == "__main__":
    opt, device = parse_args()
