'''专门为计算均值和方差写的datsset处理
前提是图片再一个文件见里不能跟city一样分了好几个城市文件见
'''
import torch
import  torchvision
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from dataloaders.mypath import Path

'''voc数据集处理格式'''
class Cal_md(Dataset):
    """
    PascalVoc dataset
    """


    def __init__(self,absolute_path,isdvi=True):
        """
        :param base_dir: path to VOCdevkit dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()

        self.isdvi=isdvi#转换成tensor的时候是否将数据集先先进行/255的归一化
        self._image_dir = absolute_path
        self.imge_list = os.listdir(self._image_dir)
        self.images=[os.path.join(absolute_path,x) for x in self.imge_list]



    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        # return  _img,_target
        sam =self.ToTenso(sample)
        return sam['image'],sam["label"]




    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target =0
        return _img, _target

    def ToTenso(self,sample):


        img = sample['image']
        label = sample['label']

        if  self.isdvi:
            img = np.array(img).astype(np.float32).transpose((2, 0, 1))
            img/=255.0
        else:
            img = np.array(img).astype(np.float32).transpose((2, 0, 1))

        label = np.array(label).astype(np.float32)

        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()

        return {'image': img,
                'label': label}

    # def __str__(self):
    #     return 'VOC2012(split=' + str(self.split) + ')'



# if __name__ == '__main__':
#     datas = Cal_md(r"F:\一些数据集\aeroscapes\data\VOCdevkit\VOC2012\JPEGImages")
#     img,lab  =datas[0]
    # print(lab)
    # print(img.shape)
    # print("d")

