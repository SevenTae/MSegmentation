from torchvision.datasets import ImageFolder
import torchvision
import torch
import  argparse
import numpy as np
from tqdm import tqdm
'''所有图片不管是全部的还是光训练集的必须再同一个文件夹里边'''
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
    for batch in tqdm(train_loader, total=len(train_data), desc='cal_md', unit='img',
                      leave=False):
        X,_=batch
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':

    #注意这里的数据集是没有归一化的 ，所以出来的均值和方差都是很大的那个数
    from dataloaders.datasets.cal_md import Cal_md
    dir_path =r"F:\一些数据集\aeroscapes\data\VOCdevkit\VOC2012\JPEGImages"

    dir_path =r"F:\一些数据集\Tree\VOCdevkit\VOC2012\JPEGImages"

    datasets=Cal_md(dir_path,)
    mean,std =getStat(datasets)
    mean_save3 = [round(x, 3) for x in mean]
    std_save3 = [round(x, 3) for x in std]
    print(mean_save3)
    print(std_save3)
    mean_255=[round(x*255,3) for x in mean]
    std_255=[round(x*255,3) for x in std]
    print(mean_255)
    print(std_255)


    with open("均值和方差.txt",mode='a',encoding='utf-8') as f:

        f.write("此数据集：\n")
        f.write("均值：{}\n".format(mean_255))
        f.write("方差：{}\n".format(std_255))
        f.write("归一化后的均值：{}\n".format(mean_save3))
        f.write("归一化后的方差：{}\n".format(std_save3))

    f.close()
    print("文件已经被保存")




# torchvison.transforms.ToTensor()把一个取值为[0,255]的形状为（h,w ,c）的图片转换成一个[0,1]的（c,h,w）的tensor  貌似只有data的dtype为uint8时取值范围才会缩放到[0, 1.0]


# 注：这里涉及到两种算法
# 1. 计算每张图片的像素平均灰度值，再求和并除以总图片数
# 2. 计算总灰度值，将每张图像像素数求和，除以这个和
# 方法一更好，适用于图片大小不一样的情况



