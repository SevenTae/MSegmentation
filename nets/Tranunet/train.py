#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@project: 
@File    : trainmodel
@Author  : root
@create_time    : 2022/9/25 14:01
"""
#
from nets.Tranunet.vit_seg_modeling import VisionTransformer as ViT_seg
from nets.Tranunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import  numpy as np

import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter






def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--train_batch-size', '-tb', dest='train_batch_size', metavar='TB', type=int, default=2, help='Train_Batch size')
    parser.add_argument('--val_batch-size', '-vb', dest='val_batch_size', metavar='VB', type=int, default=2, help='Val_Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default="", help='Load model from a .pth file')  # 有没有预训练。。
    parser.add_argument('--ignore-index', '-i', type=int, dest='ignore_index', default=255,
                        help='ignore index defult 100')  # 有没有预训练。。
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--resume', '-r', type=str, default=False, help='is use Resume')
    parser.add_argument('--classes', '-c', type=int, default=13, help='Number of classes')
    #transfomer专属参数
    parser.add_argument('--num_classes', type=int,
                        default=9, help='output channel of network')

    parser.add_argument('--img_size', type=int,
                        default=256, help='input patch size of network input')

    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')




    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))  # 16，16
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes)

    input= torch.randn(2,3,256,256)


    out= net(input)
    print(out.shape)
    # net.load_from(weights=np.load(config_vit.pretrained_path))
    # print("预训练加载成功aaa ")




