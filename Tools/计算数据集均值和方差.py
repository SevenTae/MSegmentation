from torchvision.datasets import ImageFolder
import torchvision
import torch
import numpy as np
'''数据集格式ImageFolder'''
def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    #首先先专门弄出一个train的数据集文件夹
    #这个玩意他要求数据格式是根目录：类别1文件夹 类别2文件夹
    train_dataset = ImageFolder(root=r'F:\openmmmlab\mmsegmentation\data\test', transform=torchvision.transforms.ToTensor())
    print(getStat(train_dataset))
    mean,std =getStat(train_dataset)
    #
    mean_255=[round(x*255,3) for x in mean]
    std_255=[round(x*255,3) for x in std]
    print(mean_255)
    print(std_255)


# torchvison.transforms.ToTensor()把一个取值为[0,255]的形状为（h,w ,c）的图片转换成一个[0,1]的（c,h,w）的tensor  貌似只有data的dtype为uint8时取值范围才会缩放到[0, 1.0]


# 注：这里涉及到两种算法
# 1. 计算每张图片的像素平均灰度值，再求和并除以总图片数
# 2. 计算总灰度值，将每张图像像素数求和，除以这个和
# 方法一更好，适用于图片大小不一样的情况



