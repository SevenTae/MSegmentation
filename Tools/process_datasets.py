#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@project: 
@File    : process_datasets
@Author  : root
@create_time    : 2022/9/7 15:48
"""

'''1.处理pv数据集的标签不太对的情况，
2.将单通道灰度图彩色化
'''
import  os
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as  plt
from  tqdm import  tqdm
from PIL import Image
import argparse

def  process_pvlabel(label_path,save_path):
    data= os.listdir(label_path)
    imge_path =[]
    for i in data:

        img = Image.open(os.path.join(label_path,i))
        img_arr = np.array(img)
        img_arr[img_arr>0]=1
        new_img= Image.fromarray(img_arr)
        new_img.save(os.path.join(save_path,i))
    print("处理完成")








class gray8_to_color8:
    #这个的使用前提必须是已经形成了8位无误的灰度图
    def __init__(self,gray8_path,savepath):
        self.gray8_path =gray8_path
        self.savepath=savepath
        # 使用的时候注意修改这个
        self.Cls = namedtuple('cls', ['name', 'id', 'color'])
        self.Clss = [
            self.Cls('c1', 0, (0, 0, 0)),
            self.Cls('c2', 1, (255, 255, 255))
        ]


    def get_putpalette(self,Clss, color_other=[0, 0, 0]):
        '''
        灰度图转8bit彩色图
        :param Clss:颜色映射表
        :param color_other:其余颜色设置
        :return:
        '''
        putpalette = []
        for cls in Clss:
            putpalette += list(cls.color)
        putpalette += color_other * (255 - len(Clss))
        return putpalette


    def get_color8bit(self,):
        '''
        灰度图转8bit彩色图
        :param grays_path:  灰度图文件路径
        :param colors_path: 彩色图文件路径
        :return:
        '''
        if not os.path.exists( self.savepath):
            os.makedirs( self.savepath)
        file_names = os.listdir(self.gray8_path)
        bin_colormap = self.get_putpalette(  self.Clss)
        with tqdm(file_names) as pbar:
            for file_name in pbar:
                gray_path = os.path.join(self.gray8_path, file_name)
                color_path = os.path.join( self.savepath, file_name.replace('.bmp', '.bmp'))
                gt = Image.open(gray_path)
                gt = gt.convert("P")
                gt.putpalette(bin_colormap)
                gt.save(color_path)
                pbar.set_description('get color')
        print("转换完成")




if __name__ == '__main__':

    # ulablepath= r"D:\Yan\ReadCode\Segment\FarSeg-master\pv\label"
    # savepath =  r"D:\Yan\ReadCode\Segment\FarSeg-master\pv\img"
    # process_pvlabel(ulablepath,savepath)
    graypath=r"D:\Yan\ReadCode\Segment\FarSeg-master\pv\img"
    savepath=r"D:\Yan\ReadCode\Segment\FarSeg-master\pv\label"

    model =gray8_to_color8(graypath,savepath)
    model.get_color8bit()






