'''这种的数据集大概是和city的数据集差不多格式的，已经划分好训练验证集再分别的文件夹里了'''
'''且图片名字和标签名字一样（除了格式），否则就得自己改改'''
'''并且标签也是正常的'''

import os
import numpy as np
from PIL import Image
from torch.utils import data
from dataloaders.mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
#改成20类了 原来19个类+那个未知类当背景
class CustomerSegmentation(data.Dataset):
    NUM_CLASSES = 2

    def __init__(self, args, root=Path.db_root_dir('Customer'), split="train",isAugTrain =False):

        self.root = root
        self.split = split
        self.args = args
        self.isAugTrain =isAugTrain#是否使用数据增强
        self.files = {}


        self.images_base = os.path.join(self.root, 'images/', self.split) #
        self.annotations_base = os.path.join(self.root, 'annotations/', self.split) #

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')
        self.valid_classes = [0,1]#
        self.class_names = ['background', 'foreground']

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        #这个地方需要根据自己数据集的不同自己改
        img_path = self.files[self.split][index].rstrip() #获得图片的路径
        img_name = img_path.split(os.sep)[-1].split(".")[-2]  #使用os.sep的话，就不用考虑这个了，os.sep根据你所处的平台，自动采用相应的分隔符号
        lbl_path = os.path.join(self.annotations_base+'/',  #获得图片对应的标签图片的路径
                               img_name+'_2ndHO.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}
         #进行数据增强
        if self.split == 'train':
            if self.isAugTrain == True:
                return self.AugTrain(sample)
            else:
                return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir  #使用给定的后缀和 rootdir 执行递归 glob
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]
    #对训练集数据增强  这个数据增强有个问题 ，就是我的图片都会经过裁剪，最后网络输入的大小都是裁剪的大小。不是那种随机数据增强，
    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(self.args.resize),  #先缩放要不然原图太大了进不去
            tr.Normalize_simple(),
            tr.ToTensor()])

        return composed_transforms(sample)

    def AugTrain(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(self.args.resize),  # 先缩放要不然原图太大了进不去
            tr.RandomFixScaleCropMy(crop_size=self.args.crop_size,fill=255),
            tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255), #先不要了
            tr.RandomGaussianBlur(),
            tr.Normalize_simple(),
            tr.ToTensor()])

        return composed_transforms(sample)
    #
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(self.args.resize),
            tr.Normalize_simple(),
            tr.ToTensor()])
        return composed_transforms(sample)
    #测试集
    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(self.args.resize),
            tr.Normalize_simple(),
            tr.ToTensor()])

        return composed_transforms(sample)


'''
if __name__ == '__main__':
    from dataloaders.train_utils import decode_segmap
    from torch.train_utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512#这玩意干啥的
    args.crop_size = 512
    args.resize=(512,512)

    cityscapes_train = CityscapesSegmentation(args, split='train')
    print(len(cityscapes_train))
    dataloader = DataLoader(cityscapes_train, batch_size=1, shuffle=False, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            print("我看看你是什么尺寸",img.shape)
            gt = sample['label'].numpy()
            print(type(gt))
            print(np.unique(gt))
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='Customer')
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

'''
