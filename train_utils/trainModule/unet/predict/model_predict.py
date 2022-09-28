import colorsys
import copy
import time

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from utils.utils import cvtColor,preprocess_input

from nets.unet.unet_model import UNet




# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和num_classes都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的model_path和num_classes数的修改
# --------------------------------------------#
class PredictModel(object):
    _defaults = {
        # -------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        # -------------------------------------------------------------------#

        "model_path": r'F:\MSegmentation\customecompa\psp\checkpoints\best.pth',
        # --------------------------------#
        #   所需要区分的类的个数+1
        # --------------------------------#
        "num_classes": 6,
        # --------------------------------#
        #   所使用的的主干网络：vgg、resnet50
        # --------------------------------#

        # --------------------------------#
        #   输入图片的大小
        # --------------------------------#
        "input_shape": [256, 256],

        # --------------------------------#
        #   blend参数用于控制是否
        #   让识别结果和原图混合
        # --------------------------------#
        "blend": False,
        # --------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # --------------------------------#
        "cuda": True,
    }

    # ---------------------------------------------------#
    #   初始化UNET
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#

        # self.CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        #            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        #            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        #            'bicycle')

        self.CLASSES  = ('unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle')

        self.PALETTE = [[0,0,0],[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                   [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
                   [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                   [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                   [0, 80, 100], [0, 0, 230], [119, 11, 32]]


        # self.CLASSES  = ('unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
        #                     'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
        #                     'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        #                     'motorcycle', 'bicycle')

        # self.PALETTE = [[0,0,0],[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        #            [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        #            [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        #            [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        #            [0, 80, 100], [0, 0, 230], [119, 11, 32]]

        self.CLASSES  = ('building', 'road', 'pavement', 'vegetation', 'bare soil', 'water')
        self.PALETTE = [[255, 0, 0], [255, 255, 0], [192, 192, 0], [0, 255, 0],
                       [128, 128, 128], [0, 0, 255]]


        # ---------------------------------------------------#
        #   获得模型
        # ---------------------------------------------------#
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):

        self.net = UNet(n_channels=3,n_classes=self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(model_dict['net'])


        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize

        #   也可以直接resize进行识别

        #   也可以直接resize进行识别  这里是直接resize

        # ---------------------------------------------------------#
        image=image.resize((self.input_shape[0],self.input_shape[1]), Image.BICUBIC)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#

        #测试图片归一化，按照你你数据集的归一化方法归一化

        #测试图片归一化，如果有你自己数据集的均值和方差就可以使用他们

        # if mean_std:
        # else:
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data) #转成tensor
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]  # 输出 【1 nuclass 512 ，256 】---》torch.Size([21, 512, 512])
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()  #softmax
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)
            #到这里就剩下一个通道的里边是类别数0，1 2 3的一个二维数组了
            # ---------------------------------------------------#
            #   进行图片的resize成原大小
            # ---------------------------------------------------#

            pr = cv.resize(pr, (orininal_w, orininal_h), interpolation=cv.INTER_NEAREST) #使用最近邻
            #然后按照调色盘给他上色
            # ------------------------------------------------#
            #   创建一副新图，并根据每个像素点的种类赋予颜色
            # ------------------------------------------------#
            seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            for c in range(self.num_classes):
                seg_img[:, :, 0] += ((pr[:, :] == c) * (self.PALETTE[c][0])).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c) * (self.PALETTE[c][1])).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c) * (self.PALETTE[c][2])).astype('uint8')
            # ------------------------------------------------#
            #   将新图片转换成Image的形式
            # ------------------------------------------------#
            FinalImage = Image.fromarray(np.uint8(seg_img))
            # ------------------------------------------------#
            #   将新图片和原图片混合
            # ------------------------------------------------#
            if self.blend:
                FinalImage = Image.blend(old_img, FinalImage, 0.7)

        return FinalImage

    def get_miou_png(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #
        #   可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh =  image   = image.resize( (self.input_shape[1], self.input_shape[0]), Image.BICUBIC)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv.resize(pr, (orininal_w, orininal_h), interpolation=cv.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

    def get_FPS(self, image, test_interval):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image=image.resize((self.input_shape[0],self.input_shape[1]), Image.BICUBIC)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                print(self.cuda)
                images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)

        #上边这个算是对gpu进行了一个热身运动？

        #热身完了开始正式？
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                # ---------------------------------------------------#
                #   图片传入网络进行预测
                # ---------------------------------------------------#
                pr = self.net(images)[0]
                # ---------------------------------------------------#
                #   取出每一个像素点的种类
                # ---------------------------------------------------#
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                pr = pr.argmax(axis=-1)

                pr = cv.resize(pr, (orininal_w, orininal_h), interpolation=cv.INTER_NEAREST)  # 使用最近邻


        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time











    #-----------------------------------------------------------------------------------

    #
    # def get_FPS(self, image, test_interval):
    #     # ---------------------------------------------------------#
    #     #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #     #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #     # ---------------------------------------------------------#
    #     image = cvtColor(image)
    #     # ---------------------------------------------------------#
    #     #   给图像增加灰条，实现不失真的resize
    #     #   也可以直接resize进行识别
    #     # ---------------------------------------------------------#
    #     image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
    #     # ---------------------------------------------------------#
    #     #   添加上batch_size维度
    #     # ---------------------------------------------------------#
    #     image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    #
    #     with torch.no_grad():
    #         images = torch.from_numpy(image_data)
    #         if self.cuda:
    #             images = images.cuda()
    #
    #         # ---------------------------------------------------#
    #         #   图片传入网络进行预测
    #         # ---------------------------------------------------#

    #         pr = self.net(images)[0]

    #         pr = self.nets(images)[0]

    #         # ---------------------------------------------------#
    #         #   取出每一个像素点的种类
    #         # ---------------------------------------------------#
    #         pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
    #         # --------------------------------------#
    #         #   将灰条部分截取掉
    #         # --------------------------------------#
    #         pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
    #              int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
    #
    #     t1 = time.time()
    #     for _ in range(test_interval):
    #         with torch.no_grad():
    #             # ---------------------------------------------------#
    #             #   图片传入网络进行预测
    #             # ---------------------------------------------------#

    #             pr = self.net(images)[0]

    #             pr = self.nets(images)[0]

    #             # ---------------------------------------------------#
    #             #   取出每一个像素点的种类
    #             # ---------------------------------------------------#
    #             pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
    #             # --------------------------------------#
    #             #   将灰条部分截取掉
    #             # --------------------------------------#
    #             pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
    #                  int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
    #     t2 = time.time()
    #     tact_time = (t2 - t1) / test_interval
    #     return tact_time
    #
    # def get_miou_png(self, image):
    #     # ---------------------------------------------------------#
    #     #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #     #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #     # ---------------------------------------------------------#
    #     image = cvtColor(image)
    #     orininal_h = np.array(image).shape[0]
    #     orininal_w = np.array(image).shape[1]
    #     # ---------------------------------------------------------#
    #     #   给图像增加灰条，实现不失真的resize
    #     #   也可以直接resize进行识别
    #     # ---------------------------------------------------------#
    #     image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
    #     # ---------------------------------------------------------#
    #     #   添加上batch_size维度
    #     # ---------------------------------------------------------#
    #     image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    #
    #     with torch.no_grad():
    #         images = torch.from_numpy(image_data)
    #         if self.cuda:
    #             images = images.cuda()
    #
    #         # ---------------------------------------------------#
    #         #   图片传入网络进行预测
    #         # ---------------------------------------------------#

    #         pr = self.net(images)[0]

    #         pr = self.nets(images)[0]

    #         # ---------------------------------------------------#
    #         #   取出每一个像素点的种类
    #         # ---------------------------------------------------#
    #         pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
    #         # --------------------------------------#
    #         #   将灰条部分截取掉
    #         # --------------------------------------#
    #         pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
    #              int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
    #         # ---------------------------------------------------#
    #         #   进行图片的resize
    #         # ---------------------------------------------------#
    #         pr = cv.resize(pr, (orininal_w, orininal_h), interpolation=cv.INTER_LINEAR)
    #         # ---------------------------------------------------#
    #         #   取出每一个像素点的种类
    #         # ---------------------------------------------------#
    #         pr = pr.argmax(axis=-1)
    #
    #     image = Image.fromarray(np.uint8(pr))
    #     return image
