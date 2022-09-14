'''针对这个数据集的标签处理'''

from PIL import Image
import cv2 as cv
import numpy as np
import operator
import  matplotlib.pyplot as plt
import os
from tqdm import tqdm

source_label_path = r"D:\MSegmentation\data\VOCdevkit\berlin\labels"
source_label=os.listdir(source_label_path)
label_len =len(source_label)
#标记为建筑物、道路和背景的像素由 RGB 颜色 [255,0,0]、[0,0,255] 和 [255,255,255] 表示。
'''
3类：
0：背景[255,255,255]
1：建筑物  [255,0,0]
2：道路 [0,0,255]

'''
cato0= [255,255,255] #背景
cato1=[255,0,0] #建筑物
cato2=[0,0,255] #道路
dest = r"D:\MSegmentation\data\VOCdevkit\VOC2012\SegmentationClass"
for i in tqdm(source_label):
    label_name = i.split(".")[0]
    labelpath = os.path.join(source_label_path,i)
    labels = Image.open(labelpath)
    source=np.array(labels)
    reallabel = np.zeros((source.shape[0], source.shape[1]))
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            a = source[i, j, :].tolist()
            if operator.eq(a, cato0):
                reallabel[i, j] = 0
            elif operator.eq(a, cato1):
                reallabel[i, j] = 1
            elif operator.eq(a, cato2):
                reallabel[i, j] = 2
            else:
                print("你绝对有问题")
    lable = Image.fromarray(np.uint8(reallabel)).convert("L")  # 转换成单通道灰度图
    lable.save("D:/MSegmentation/data/VOCdevkit/VOC2012/SegmentationClass/{}.png".format(label_name))

print("完成")





