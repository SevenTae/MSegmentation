import os
from tqdm import tqdm
import numpy as np
from dataloaders.mypath import Path

'''
cr：http://t.csdn.cn/C9pWg
计算类别权重，前提是你的数据个是example的格式
'''
def calculate_weigths_labels( dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()

        y=y-1 #这个地方是有时候标签可能从1开始
        # y=y-1 #这个地方是有时候标签可能从1开始

        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    # classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    # np.save(classes_weights_path, ret)

    return ret

#EXample
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
# from dataloaders.datasets.Customer import CustomerSegmentation

from dataloaders.datasets .pascal_customer import Customer_VOCSegmentation
args = parser.parse_args()
# args.base_size = 512#这玩意干啥的
# args.crop_size = 512
args.resize=(256,256)

from dataloaders.datasets .pascal_customer2 import Customer_VOCSegmentation
args = parser.parse_args()
# args.base_size = 512#这玩意干啥的
# args.crop_size = 512
args.resize=(512,512)


customer_train = Customer_VOCSegmentation(args, split='train',isAug=False)
print(len(customer_train))
dataloader = DataLoader(customer_train, batch_size=1, shuffle=False)

num_classes = 6
re = calculate_weigths_labels(dataloader,num_classes)
print(re)

num_classes = 13
re = calculate_weigths_labels(dataloader,num_classes)
print(re)


#把这些数归一化到一个区间内比如0.5-2.5

#把不同的权重归一化到[a，b]的范围内（0.5,2.5）
#     将数据归一化到[a,b]区间范围的方法：
#
# （1）首先找到样本数据Y的最小值Min及最大值Max
# （2）计算系数为：k=（b-a)/(Max-Min)
# （3）得到归一化到[a,b]区间的数据：norY=a+k(Y-Min)
#


minn =re.min()
max = re.max()
a = 0.5
b = 2.5
k = (b-a)/(max-minn)
guiyi = a + k*(re-minn)
li = [round(i,2) for i in guiyi.tolist()]
print(li)

with open("权重.txt", mode='a', encoding='utf-8') as f:
    f.write("此数据集：\n")
    f.write("没归一化之前的权重：{}\n".format(re))
    f.write("归一化到0.5-2.5后的权重：{}\n".format(li))


f.close()
print("文件已经被保存")

