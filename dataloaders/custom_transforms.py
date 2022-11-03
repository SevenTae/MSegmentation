

import numpy as np
import torchvision.transforms as transforms
#定义了一些数据增强的方法
from PIL import Image, ImageOps, ImageFilter,ImageEnhance

#一定要设置随机种子
import  random
import torch
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



################################################################
'''
一下这些除了归一化的数据增强都是在原图上进行操作的（没有转换成numpy）
'''
'''自定义一些数据增强的方法'''
class Normalize(object):  #归一化处理 根据数据集的均值和方差归一化
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}

class Tonumpy(object):
    def __init__(self,):
        pass


    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img)
        mask = np.array(mask)
        return {'image': img,
                'label': mask}

class Normalize_simple(object):  #简单归一化处理
    """Normalize a tensor image with/255
    Args:

    """
    def __init__(self,):
        self.num = 255.0

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0


        return {'image': img,
                'label': mask}


class Resize(object):
    '''太大了进不去'''
    def __init__(self, resizeshape=(1024,512)):
        self.resizeshape = resizeshape

    def __call__(self, sample):

        img = sample['image']
        mask = sample['label']
        if img.size ==  self.resizeshape:
            return {'image': img,
                    'label': mask}
        else:
            img =img.resize(self.resizeshape,Image.BILINEAR) #原图用双线性插值
            mask = mask.resize(self.resizeshape, Image.NEAREST)#标签图用最近邻，要不然就乱了

            return {'image': img,
                    'label': mask}

class ResizeforValTest(object):  #注意测试miu的时候gt是打死都不能动的只能缩放预测图  #允许你缩放原图，因为可能因为显存不够盛不下  ！！但是gt打死不能动
    '''太大了进不去'''
    def __init__(self, resizeshape=(1024,512)):
        self.resizeshape = resizeshape

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if img.size == self.resizeshape:
            return {'image': img,
                    'label': mask}
        else:
            img = img.resize(self.resizeshape, Image.BILINEAR)  # 原图用双线性插值
            # mask = mask.resize(self.resizeshape, Image.NEAREST)  # 标签图用最近邻，要不然就乱了

            return {'image': img,
                    'label': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):#这是真的随机翻转，有的翻转有的不反转 keyong
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

#随机色度增强
class Enhance_Color(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        enh_col = ImageEnhance.Color(img)
        color= np.random.uniform(0.4,2.6) #返回a,b之间的随机浮点数,控制图像的增强程度。变量factor为1将返回原始图像的拷贝；factor值越小，颜色越少（亮度，对比度等），更多的价值。对变量facotr没有限制。
        img_colored = enh_col.enhance(color)
        return {'image': img_colored,
                'label': mask}

#随机对比度增强
class Enhance_contrasted(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        enh_con = ImageEnhance.Color(img)
        contrast = np.random.uniform(0.6,1.6)
        img_contrasted = enh_con.enhance(contrast)
        return {'image': img_contrasted,
                'label': mask}

#随机锐度增强
class Enhance_sharped(object):
   def __call__(self, sample):
     img = sample['image']
     mask = sample['label']

     enh_sha = ImageEnhance.Sharpness(img)
     sharpness = np.random.uniform(0.4, 4)
     image_sharped = enh_sha.enhance(sharpness)
     return {'image': image_sharped,
            'label': mask}

class RandomRotate(object):  #随机 随机旋转率旋转 有的旋转有的不旋转，转转率也不一样
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            #这玩意旋转完了还好吗
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}

class RandomGaussianBlur(object): #随机高斯模糊，真的随机
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}

################################################################



################################################################
'''
下边这些的使用，前提你的img和label都转换成了numpy但是还没有进行归一化
'''
'''
Function:
    Define the transforms for data augmentations
Author:
    Zhenchao Jin
'''
import cv2
import torch
import numpy as np
import torch.nn.functional as F


'''Resize'''
class Resize2(object):
    #固定尺寸resize或者按照scale_range随机resize大小
    def __init__(self, output_size, scale_range=(0.5, 2.0), img_interpolation='bilinear', seg_interpolation='nearest', keep_ratio=True, min_size=None):
        # set attribute
        '''

        :param output_size: 输出的size
        :param scale_range: 缩放的范围
        :param img_interpolation:
        :param seg_interpolation:
        :param keep_ratio: 是否保持原比例缩放
        :param min_size:
        当scale_range=None,keep_ratio=False时，就直接粗暴的resize成想要的大小
        当scale_range=None,keep_ratio=True时不失真的缩放 比如原图(501, 750, 3) outsize=512--》 (342, 512, 3)
        当scale_range=[0.5,2],keep_ratio=True时按照这个随机比例不失真的缩放
        当scale_range=[0.5,2],keep_ratio=False时按照这个随机比例直接缩放
        '''
        self.output_size = output_size
        if isinstance(output_size, int): self.output_size = (output_size, output_size)
        self.scale_range = scale_range
        self.img_interpolation = img_interpolation
        self.seg_interpolation = seg_interpolation
        self.keep_ratio = keep_ratio
        self.min_size = min_size
        # interpolation to cv2 interpolation
        self.interpolation_dict = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
    '''call'''
    def __call__(self, sample):
        # parse
        image, segmentation = sample['image'].copy(), sample['label'].copy()
        if self.scale_range is not None:
            rand_scale = np.random.random_sample() * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0]
            output_size = int(self.output_size[0] * rand_scale), int(self.output_size[1] * rand_scale)
        else:
            output_size = self.output_size[0], self.output_size[1]
        # resize image and segmentation
        if self.keep_ratio:
            scale_factor = min(max(output_size) / max(image.shape[:2]), min(output_size) / min(image.shape[:2]))
            dsize = int(image.shape[1] * scale_factor + 0.5), int(image.shape[0] * scale_factor + 0.5)
            if self.min_size is not None and min(dsize) < self.min_size:
                scale_factor = self.min_size / min(image.shape[:2])
                dsize = int(image.shape[1] * scale_factor + 0.5), int(image.shape[0] * scale_factor + 0.5)
            image = cv2.resize(image, dsize=dsize, interpolation=self.interpolation_dict[self.img_interpolation])
            segmentation = cv2.resize(segmentation, dsize=dsize, interpolation=self.interpolation_dict[self.seg_interpolation])
        else:
            if image.shape[0] > image.shape[1]:
                dsize = min(output_size), max(output_size)
            else:
                dsize = max(output_size), min(output_size)
            image = cv2.resize(image, dsize=dsize, interpolation=self.interpolation_dict[self.img_interpolation])
            segmentation = cv2.resize(segmentation, dsize=dsize, interpolation=self.interpolation_dict[self.seg_interpolation])
        # update and return sample
        sample['image'], sample['label'] = image, segmentation
        return sample


'''RandomCrop'''
class RandomCrop(object):
    #原来是没有概率，对每张图都进行随机裁剪
    #现在我给他改成对每张图片以一定的概率进行随机裁剪,一般就默认0.5
    #prob 0就是不进行
    '''
    随机裁剪：具体代码就别读了，它大概的原理：
    比如你要输入网络的尺寸为512*512
    你的原图是（500，750）
    这个随机裁剪：比如随机裁剪的尺寸是crop_size=256*256
    那么将会从原图（500,750）随机裁剪出一块256*256的区域，然后再resize成网络输入的大小
    （注意这个class出来就是crop_size大小的区域，然后根据网络要求的输入在把这个crop的区域resize成512）

    这个代码是：在crop size的长和高都不超过范围的的情况下裁剪出来是crop size大小的
    比如原图（500，750） cropsize=256 那就会随机从原图中裁剪出256*256
    如果crop size超过原图了 比如原图500,750 你的crop size为800*800，那就直接返回原图（500，750）
    如果只有一个尺寸超了范围（比如crop size （700,700）），那么对应的那一边就会返回原图最小的那一边 （500,700)

    '''
    def __init__(self, crop_size, crop_prob = 0.5,ignore_index=255, one_category_max_ratio=0.75):
        self.crop_size = crop_size
        self.crop_prob = crop_prob
        if isinstance(crop_size, int): self.crop_size = (crop_size, crop_size)
        self.ignore_index = ignore_index
        self.one_category_max_ratio = one_category_max_ratio
    '''call'''
    def __call__(self, sample):
        # avoid the cropped image is filled by only one category
        if np.random.rand() > self.crop_prob: return sample
        for _ in range(10):
            # --parse
            image, segmentation = sample['image'].copy(), sample['label'].copy()
            h_ori, w_ori = image.shape[:2]
            h_out, w_out = min(self.crop_size[0], h_ori), min(self.crop_size[1], w_ori)
            # --random crop
            top, left = np.random.randint(0, h_ori - h_out + 1), np.random.randint(0, w_ori - w_out + 1)
            image = image[top: top + h_out, left: left + w_out]
            segmentation = segmentation[top: top + h_out, left: left + w_out]
            # --judge
            labels, counts = np.unique(segmentation, return_counts=True)
            counts = counts[labels != self.ignore_index]
            if len(counts) > 1 and np.max(counts) / np.sum(counts) < self.one_category_max_ratio: break
        # update and return sample
        if len(counts) == 0: return sample
        sample['image'], sample['label'] = image, segmentation
        return sample


'''RandomFlip'''
class RandomFlip(object):
    #以一定的额概率随机翻转
    def __init__(self, flip_prob=0.5, fix_ann_pairs=None):
        self.flip_prob = flip_prob
        self.fix_ann_pairs = fix_ann_pairs
    '''call'''
    def __call__(self, sample):
        if np.random.rand() > self.flip_prob: return sample
        image, segmentation = sample['image'].copy(), sample['label'].copy()
        image, segmentation = np.flip(image, axis=1), np.flip(segmentation, axis=1)
        if self.fix_ann_pairs:
            for (pair_a, pair_b) in self.fix_ann_pairs:
                pair_a_pos = np.where(segmentation == pair_a)
                pair_b_pos = np.where(segmentation == pair_b)
                segmentation[pair_a_pos[0], pair_a_pos[1]] = pair_b
                segmentation[pair_b_pos[0], pair_b_pos[1]] = pair_a
        sample['image'], sample['label'] = image, segmentation
        return sample


'''PhotoMetricDistortion'''
class PhotoMetricDistortion(object):
    '''一些颜色变换'''
    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
    '''call'''
    def __call__(self, sample):
        image = sample['image'].copy()
        image = self.brightness(image)
        mode = np.random.randint(2)
        if mode == 1: image = self.contrast(image)
        image = self.saturation(image)
        image = self.hue(image)
        if mode == 0: image = self.contrast(image)
        sample['image'] = image
        return sample
    '''brightness distortion（亮度）'''
    def brightness(self, image):
        if not np.random.randint(2): return image
        return self.convert(image, beta=np.random.uniform(-self.brightness_delta, self.brightness_delta))
    '''contrast distortion（对比度）'''
    def contrast(self, image):
        if not np.random.randint(2): return image
        return self.convert(image, alpha=np.random.uniform(self.contrast_lower, self.contrast_upper))
    '''rgb2hsv（RGB颜色空间转hsv颜色空间）'''
    def rgb2hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    '''hsv2rgb（hsv转rgb）'''
    def hsv2rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    '''saturation distortion（饱和度）'''
    def saturation(self, image):
        if not np.random.randint(2): return image
        image = self.rgb2hsv(image)
        image[..., 1] = self.convert(image[..., 1], alpha=np.random.uniform(self.saturation_lower, self.saturation_upper))
        image = self.hsv2rgb(image)
        return image
    '''hue distortion（色调）'''
    def hue(self, image):
        if not np.random.randint(2): return image
        image = self.rgb2hsv(image)
        image[..., 0] = (image[..., 0].astype(int) + np.random.randint(-self.hue_delta, self.hue_delta)) % 180
        image = self.hsv2rgb(image)
        return image
    '''multiple with alpha and add beat with clip'''
    def convert(self, image, alpha=1, beta=0):
        image = image.astype(np.float32) * alpha + beta
        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)


'''RandomRotation'''
class RandomRotation(object):
    '''随机旋转'''
    '''每一张图有rotation_prob的概率会进行angle_upper度的旋转'''
    def __init__(self, angle_upper=30, rotation_prob=0.5, img_fill_value=0.0, seg_fill_value=255, img_interpolation='bicubic', seg_interpolation='nearest'):
        # set attributes
        '''

        :param angle_upper: 旋转角度
        :param rotation_prob: 旋转概率
        :param img_fill_value: 原图旋转后用什么像素值填充
        :param seg_fill_value: 标签图旋转后用什么像素值填充
        :param img_interpolation:原图的插值方式
        :param seg_interpolation: 标签图的插值方式
        '''

        self.angle_upper = angle_upper
        self.rotation_prob = rotation_prob
        self.img_fill_value = img_fill_value
        self.seg_fill_value = seg_fill_value
        self.img_interpolation = img_interpolation
        self.seg_interpolation = seg_interpolation
        # interpolation to cv2 interpolation
        self.interpolation_dict = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
    '''call'''
    def __call__(self, sample):
        if np.random.rand() > self.rotation_prob: return sample
        image, segmentation = sample['image'].copy(), sample['label'].copy()
        h_ori, w_ori = image.shape[:2]
        rand_angle = np.random.randint(-self.angle_upper, self.angle_upper)
        matrix = cv2.getRotationMatrix2D(center=(w_ori / 2, h_ori / 2), angle=rand_angle, scale=1)
        image = cv2.warpAffine(image, matrix, (w_ori, h_ori), flags=self.interpolation_dict[self.img_interpolation], borderValue=self.img_fill_value)
        segmentation = cv2.warpAffine(segmentation, matrix, (w_ori, h_ori), flags=self.interpolation_dict[self.seg_interpolation], borderValue=self.seg_fill_value)
        sample['image'], sample['label'] = image, segmentation
        return sample


'''Padding'''
class Padding(object):
    def __init__(self, output_size, data_type='numpy', img_fill_value=0, seg_fill_value=255, output_size_auto_adaptive=True):
        self.output_size = output_size
        if isinstance(output_size, int): self.output_size = (output_size, output_size)
        assert data_type in ['numpy', 'tensor'], 'unsupport data type %s' % data_type
        self.data_type = data_type
        self.img_fill_value = img_fill_value
        self.seg_fill_value = seg_fill_value
        self.output_size_auto_adaptive = output_size_auto_adaptive
    '''call'''
    def __call__(self, sample):
        output_size = self.output_size[0], self.output_size[1]
        if self.output_size_auto_adaptive:
            if self.data_type == 'numpy':
                h_ori, w_ori = sample['image'].shape[:2]
            else:
                h_ori, w_ori = sample['image'].shape[1:]
            h_out, w_out = output_size
            if (h_ori > w_ori and h_out < w_out) or (h_ori < w_ori and h_out > w_out):
                output_size = (w_out, h_out)
        if self.data_type == 'numpy':
            image, segmentation, edge = sample['image'].copy(), sample['segmentation'].copy(), sample['edge'].copy()
            h_ori, w_ori = image.shape[:2]
            top = (output_size[0] - h_ori) // 2
            bottom = output_size[0] - h_ori - top
            left = (output_size[1] - w_ori) // 2
            right = output_size[1] - w_ori - left
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[self.img_fill_value, self.img_fill_value, self.img_fill_value])
            segmentation = cv2.copyMakeBorder(segmentation, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[self.seg_fill_value])
            edge = cv2.copyMakeBorder(edge, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[self.seg_fill_value])
            sample['image'], sample['segmentation'], sample['edge'] = image, segmentation, edge
        else:
            image, segmentation, edge = sample['image'], sample['segmentation'], sample['edge']
            h_ori, w_ori = image.shape[1:]
            top = (output_size[0] - h_ori) // 2
            bottom = output_size[0] - h_ori - top
            left = (output_size[1] - w_ori) // 2
            right = output_size[1] - w_ori - left
            image = F.pad(image, pad=(left, right, top, bottom), value=self.img_fill_value)
            segmentation = F.pad(segmentation, pad=(left, right, top, bottom), value=self.seg_fill_value)
            edge = F.pad(edge, pad=(left, right, top, bottom), value=self.seg_fill_value)
            sample['image'], sample['segmentation'], sample['edge'] = image, segmentation, edge
        return sample
################################################################

# '''ToTensor'''
# class ToTensor(object):
#     '''call'''
#     def __call__(self, sample):
#         for key in sample.keys():
#             if key == 'image':
#                 sample[key] = torch.from_numpy((sample[key].transpose((2, 0, 1))).astype(np.float32))
#             elif key in ['edge', 'groundtruth', 'segmentation']:
#                 sample[key] = torch.from_numpy(sample[key].astype(np.float32))
#         return sample
#
#
# '''Normalize'''
# class Normalize(object):
#     def __init__(self, mean, std, to_rgb=True):
#         self.mean = np.array(mean)
#         self.std = np.array(std)
#         self.to_rgb = to_rgb
#     '''call'''
#     def __call__(self, sample):
#         for key in sample.keys():
#             if key == 'image':
#                 image = sample[key].astype(np.float32)
#                 mean = np.float64(self.mean.reshape(1, -1))
#                 stdinv = 1 / np.float64(self.std.reshape(1, -1))
#                 if self.to_rgb: cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
#                 cv2.subtract(image, mean, image)
#                 cv2.multiply(image, stdinv, image)
#                 sample[key] = image
#         return sample












'''下边几个暂时别用了'''
class RandomCropResize(object):
    """ #裁剪出一块区域后再resize成原图大小
    Randomly crop and resize the given PIL image with a probability of 0.5
    """
    def __init__(self, crop_area):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between o and crop area
        '''
        self.cw = crop_area #裁剪区域大小 裁剪区域必须小于原图大小
        self.ch = crop_area

    def __call__(self,sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h-y1, x1:w-x1]
            label_crop = mask[y1:h-y1, x1:w-x1]

            img_crop = img_crop.resize( (w, h),interpolation=Image.BILINEAR)
            label_crop = label_crop.resize( (w,h), interpolation=Image.NEAREST)
            return {'image': img_crop,
                    'label': label_crop}
        else:
            return {'image': img,
                    'label': mask}
class RandomScaleCrop(object):
    '''先基于base size 将原图放缩到0.5-1.0/2.0倍的大小，再从这个放缩后的裁剪出crop size大小'''
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size #这个basesize是干啥的
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 1.0))#随机尺度！！！从base_size的0.5-1.0随机缩放
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size  #比如这里oh=503
            ow = int(1.0 * w * oh / h)  #就是长宽等比裁. 那么长就是503乘以原来长宽比也就是2  也就是1006
        img = img.resize((ow, oh), Image.BILINEAR)#原图先reisiz
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:  ##如果随机出来的这个尺寸小于cropsize的话就把它补成和cropsize一样的大小
            padh = self.crop_size - oh if oh < self.crop_size else 0  #如果图片的尺寸还小于crop的尺寸先计算要补的大小
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0) #得填充够crop的尺寸
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size  随机裁剪，这个随机是随机位置
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,  #反正最后出来的size是cropsize
                'label': mask}
class FixScaleCrop(object):#固定尺寸裁剪中心裁剪，从标标准准的的中心裁剪，裁剪大小是crop_size
    def __init__(self, crop_size):#他裁剪出来后貌似不进行resize或者填充了
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))  #坐上右下点的坐标，然后呢 裁剪完成之后不pading成统一的尺寸了吗
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}
class RandomFixScaleCropMy(object):  #随机中心裁剪裁剪出来的那一块再填充成原图大小 基于上边的那个给填充
    def __init__(self, crop_size,fill=0):
        self.crop_size = crop_size
        self.fill=fill
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        w, h = img.size
       #前提cropsize要小于resize的大小
        # center crop
        if random.random() < 0.5:
            x1 = int(round((w - self.crop_size) / 2.))
            y1 = int(round((h - self.crop_size) / 2.))
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))  # 坐上右下点的坐标，然后呢 裁剪完成之后不pading成统一的尺寸了吗
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            #裁剪玩成之后再补回原图（缩放后的原图）大小 #比如原图2048 1024 缩放后的原图是1024 512
            padw=w-self.crop_size-x1
            padh=h-self.crop_size-y1
            img = ImageOps.expand(img, border=(x1, y1, padw, padh), fill=0)  # 左，上，右，下
            mask = ImageOps.expand(mask, border=(x1, y1, padw, padh), fill=self.fill)

            return {'image': img,
                    'label': mask}
        else:

            return {'image': img,
                    'label': mask}
#缩放尺寸原图2048*1024太大了
class FixedResize(object):#固定尺寸
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,

                'label': mask}





