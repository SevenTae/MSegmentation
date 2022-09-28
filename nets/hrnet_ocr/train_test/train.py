#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@project: 
@File    : train
@Author  : root
@create_time    : 2022/9/27 20:21
"""

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

# import _init_paths
import nets.hrnet_ocr.models as models

from nets.hrnet_ocr.config import config
from nets.hrnet_ocr.config import update_config


####我只想说  我还是看json看的习惯

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()  # 把yaml中对应的参数读进来
    update_config(config, args)  # 更新超参数

    return args


def main():
    args = parse_args()

    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)



    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)

    # config.MODEL.NAME  'seg_hrnet'
    # build model
    #
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)

    inpu = torch.randn(2, 3, 224, 224)
    out = model(inpu)
    print(out.shape)  # torch.Size([2, 19, 56, 56])   #19是类别  #注意hrnet保持原图的下采样4倍输出这里没有

    # ****其实源码的train是个完整的文件只不过这里给 简化我们只要model就行


if __name__ == '__main__':
    main()
