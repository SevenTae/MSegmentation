import os
import numpy as np
from PIL import Image
from torch.utils import data
from dataloaders.mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
#改成20类了 原来19个类+那个未知类当背景
class CityscapesSegmentation(data.Dataset):
    NUM_CLASSES = 20

    def __init__(self, args, root=Path.db_root_dir('cityscapes'), split="train",isAugTrain=False):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}
        self.isAugTrain =isAugTrain#是否使用数据增强

        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        '''
        关于这个破数据集的标签的一些说明：
        那个labelTrainid确实是0-18 但是未知类是255 不好把它改成20个类
        关于那个labelids 里边村的是原数据集类别的id 原数据集给每个标签都编了个号比如roda是7 再这里边就是7
        '''
        self.void_classes = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1] ##这些都是数据集中那些用不着的id
        # self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33] #这是那真的19个类的id
        self.valid_classes = [0,7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]#我这次算上未知类当作背景算20个类
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        # self.ignore_index = 255
        self.set_ignore=0 #把所有用不着的也就是未知的设置成0
        self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))
        #{7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        # print(self.class_map)

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        # print(np.unique(_tmp))
        _tmp = self.encode_segmapmy(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}
         #进行数据增强
        if self.split == 'train':
            if self.isAugTrain==True:
               return self.AugTrain(sample)
            else:
                return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmapmy(self, mask):#我自己尝试给他编码
        # Put all void classes to zero
        for _voidc in self.void_classes: #把所有用着的设置为背景也就是0
            mask[mask == _voidc] = self.set_ignore
        for _validc in self.valid_classes:#用的找的给他对应的
            mask[mask == _validc] = self.class_map[_validc]
        # print(np.unique(mask))
        return mask

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
            tr.Resize(  self.args.resize),  #数据增强的时候先缩放
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), #这hcity特有的mean和std
            tr.ToTensor()])
        return composed_transforms(sample)

    def AugTrain(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(self.args.resize),  # 数据增强的时候先缩放

            tr.RandomFixScaleCropMy(crop_size=self.args.crop_size,fill=255),
            tr.RandomHorizontalFlip(),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255), #先不要了
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 这hcity特有的mean和std
            tr.ToTensor()])
        return composed_transforms(sample)

    #验证集和测试不要增强
    #验证集数据增强 验证集不用
    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(  self.args.resize),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)
    #
    def transform_ts(self, sample):
        composed_transforms = transforms.Compose([
            tr.Resize(  self.args.resize),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512#这玩意干啥的
    args.crop_size = 512
    args.resize=(2048,1024)

    cityscapes_train = CityscapesSegmentation(args, split='train',isAugTrain=True)
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
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
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


