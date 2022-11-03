#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@project: 
@File    : 针对语义分割超大的图滑动窗口切分图片
@Author  : root
@create_time    : 2022/10/5 20:59
"""
'''
|------>x
|
|y
比如3164*2043
注意height是y， width是x
cv读进来是yx也就是 hw-->  (2043,3164)
所以数组是imge[y,x]
'''
import os
import cv2

# import numpy as np
'''滑动窗口'''


def sliding_window(image, stepSize, windowSize, height, width, count,save_dir,item_name,type=".tif"):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            if (y + windowSize[1]) <= height and (x + windowSize[0]) <= width:  # 没超出下边界，也超出下边界
                slide = image[y:y + windowSize[1], x:x + windowSize[0], :]
                # slide_shrink = cv2.resize(slide, (256, 256), interpolation=cv2.INTER_AREA)
                # slide_shrink_gray = cv2.cvtColor(slide_shrink,cv2.COLOR_BGR2GRAY)
                cv2.imwrite(save_dir+item_name+"_" + str(count) + type, slide)
                count = count + 1  # count持续加1

            #自己瞎写的比较麻烦哈

            if (x + windowSize[0]) > width and  (y + windowSize[1]) <= height: #1.超出右但是没超出下
                #直接靠着右边补
                slide = image[y:y + windowSize[1], width-windowSize[0]:width, :]
                cv2.imwrite(save_dir + item_name + "_" + str(count) + type, slide)
                count = count + 1  # count持续加1
            elif (x + windowSize[0]) <= width and  (y + windowSize[1]) > height: #2.超出下但是没超出右
              #直接靠着下边补
              slide = image[height-windowSize[0]:height, x:x + windowSize[0], :]
              cv2.imwrite(save_dir + item_name + "_" + str(count) + type, slide)
              count = count + 1  # count持续加1
            elif (x + windowSize[0]) > width and  (y + windowSize[1]) > height:  #3.都超出了
                #直接靠着右下角补
                slide = image[height-windowSize[0]:height, width-windowSize[1]:width, :]
                cv2.imwrite(save_dir + item_name + "_" + str(count) + type, slide)
                count = count + 1  # count持续加1



    return count


if __name__ == "__main__":
    stepSize = int(0.3 * 512)  # 步长就是0.5倍的滑窗大小
    windowSize = [512, 512]  # 滑窗大小
    #注意
    '''
    既然是用cv的那么滑动窗口必须是（height，width的形式）,,,其实一般现在基本上上正方形的了基本不用管也行
    '''
    path = r'D:\Yan\SomeTools\aboutDatasets\source'  # 文件路径
    save_path ="D:/Yan\SomeTools/aboutDatasets/save/"
    count = 0

    filelist = os.listdir(path)  # 列举图片名
    for item in filelist:
        total_num_file = len(filelist)  # 单个文件夹内图片的总数
        if item.endswith('.jpg') or item.endswith('.tif'):  # 查询文件后缀名
            item_path = os.path.join(path,item)
            item_name= item.split(".")[0]
            image = cv2.imread(item_path)  # #注意cv读进来是bgr的格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            print(height, width)

            totalcount = sliding_window(image, stepSize, windowSize, height, width, # count要返回，不然下一个图像滑窗会覆盖原来的图像
                                   count,save_dir=save_path,item_name=item_name)
            print(item_name,"共裁剪出{}张子图".format(totalcount))




