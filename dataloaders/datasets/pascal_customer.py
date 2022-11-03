from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from dataloaders.mypath2 import Path  #注意一下这个path
from torchvision import transforms
from dataloaders import custom_transforms as tr
'''voc数据集处理格式'''
#一定要设置随机种子要不然每次结果都不一样
import  numpy as np
import  random
import torch

seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子


'''pascal_customer2 是个老版本（以前的备份），一般用这个就行'''
class Customer_VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """


    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('Customer'),
                 split='train',
                 isAug =False
                 ):
        """
        :param base_dir: path to VOCdevkit dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.isAug=isAug
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        self.args = args
        self.crop_size = args.crop_size   #是单数比如256，
        self.flip_prob = args.flip_prob  #是0-1

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split


        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpeg") #注意这个地方有的图片可能是jpg有的可能是png自己看着改
                _cat = os.path.join(self._cat_dir, line + ".png")  #注意格式
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                if self.isAug: #训练的时候使用数据增强验证的时候不需要
                    return self.AugTrain(sample)
                else:
                   return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)
            elif split =='test':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(self.args.resize),  # 先缩放要不然原图太大了进不去
            # tr.Normalize(mean=(0.231, 0.217, 0.22), std=(0.104, 0.086, 0.085)),  # 针对whdld数据集的
            tr.Normalize_simple(),
            tr.ToTensor()])

        return composed_transforms(sample)
    #
    def AugTrain(self, sample):
        '''
        这个版本的AugTrain使用的 Zhenchao Jin版本对numpy进行操作
        :param sample:
        :return:

        随机裁剪，随机翻转，随机颜色增强，随机旋转这四个就够了以后别再浪费时间在这上边了
        '''
        composed_transforms = transforms.Compose([
            tr.Tonumpy(),
            tr.RandomCrop(crop_size=self.crop_size),
            tr.RandomFlip(),
            tr.PhotoMetricDistortion(),
            tr.RandomRotation(),

            tr.Resize2(self.args.resize,scale_range=None,keep_ratio=False),
            tr.Normalize_simple(),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.ResizeforValTest(self.args.resize),
            tr.Normalize_simple(),

            tr.ToTensor()])
        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

#
if __name__ == '__main__':
    a  =np.random.randn(3,3)
    # print(a)


    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.crop_size = 100
    args.flip_prob=0.5
    args.resize=(256,256)

    voc_train = Customer_VOCSegmentation(args, split='train',isAug=True)

    dataloader = DataLoader(voc_train, batch_size=1, shuffle=False, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()

            print("现在的尺寸：",img.shape)
            # gt=gt-1
            # print(gt)
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal_customer')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


