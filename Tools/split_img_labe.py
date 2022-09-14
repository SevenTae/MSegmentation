#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@project: 
@File    : split_img_labe
@Author  : root
@create_time    : 2022/8/30 18:25
划分pv数据集label和img的脚本  原来label和img是混在一起的
"""

import os
import  shutil
def split(source_dir,img_dir,label_dir):
    '''
    Args:
        source_dir: 源文件路径
        img_dir: 划分后img存放的文件夹
        label_dir: 划分后label存放的文件夹
    Returns:
    '''

    base_dir =os.path.abspath(source_dir)
    data =os.listdir(source_dir)
    label_str ="label"
    for i in data:
        if label_str in i:
            shutil.move(os.path.join(base_dir,i),os.path.join(label_dir,i))
        else:
            shutil.move(os.path.join(base_dir, i), os.path.join(img_dir, i))
pa ="D:\Yan\ReadCode\Segment\FarSeg-master\pv"
im = "D:\Yan\ReadCode\Segment\FarSeg-master\img"
lab = "D:\Yan\ReadCode\Segment\FarSeg-master\label"

split(pa,im,lab)