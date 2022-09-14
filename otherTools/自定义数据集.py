from PIL import Image
import cv2 as cv
import numpy as np
import operator
import  matplotlib.pyplot as plt
import os
import shutil



'''没用的。。。练手的东西'''
#标记为建筑物、道路和背景的像素由 RGB 颜色 [255,0,0]、[0,0,255] 和 [255,255,255] 表示。

#先把原来的数据集处理一下吧...

sour_path = r"D:\MSegmentation\data\VOCdevkit\berlin"
image_path = r"D:\MSegmentation\data\VOCdevkit\berlin\images"
label_path =r"D:\MSegmentation\data\VOCdevkit\berlin\labels"

files = os.listdir(sour_path)
for i in files:
    imagname = i.split(".")[0]
    ty = imagname.split("_")[1]
    if ty =="image":
        s =os.path.join(sour_path,i)
        d= os.path.join(image_path,i)
        shutil.move(s,d)
    elif ty =="labels":
        s = os.path.join(sour_path, i)
        d = os.path.join(label_path, i)
        shutil.move(s, d)
    else:
        print("你有病")

print("完成")




'''
img =Image.open(r"D:\迅雷下载\berlin\berlin\berlin1_labels.png")
# 
# 3类：
# 0：背景[255,255,255]
# 1：建筑物  [255,0,0]
# 2：道路 [0,0,255]
# 
# 
cato0= [255,255,255] #背景
cato1=[255,0,0] #建筑物
cato2=[0,0,255] #道路

source = np.array(img)
label=np.zeros((source.shape[0],source.shape[1]))
#把它转换为标签
#判断两个list是否想等
# l = source[0,0,:].tolist()
# c = [255,255,255]
# print(operator.eq(l,c))
for  i in range(source.shape[0]):
    for j in range(source.shape[1]):
        a = source[i,j,:].tolist()
        if operator.eq(a,cato0):
            label[i,j]=0
        elif operator.eq(a,cato1):
            label[i,j]=1
        elif operator.eq(a,cato2):
            label[i,j]=2
        else:
            print("你绝对有问题")

print("完成")
print(np.unique(label))
lable = Image.fromarray(np.uint8(label)).convert("L") #转换成单通道灰度图
lable.save("labelss/mylabels.png")

'''
