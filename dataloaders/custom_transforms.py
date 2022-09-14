
import torch
import random
import numpy as np
import torchvision.transforms as transforms
#定义了一些数据增强的方法
from PIL import Image, ImageOps, ImageFilter


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
        img =img.resize(self.resizeshape,Image.BILINEAR) #原图用双线性插值
        # mask = mask.resize(self.resizeshape, Image.NEAREST)#标签图用最近邻，要不然就乱了

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




#随机尺度裁剪  这个最后出来是cropsize 没有填充成原来的大小 貌似分割可以真的用多尺度训练，因为不影响
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





