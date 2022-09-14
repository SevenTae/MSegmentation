##用来可视化一些黑乎乎的8位灰度图
import numpy as np
import torchvision.transforms as transforms
from collections import namedtuple  #namedtuple创建一个和tuple类似的对象，而且对象拥有可访问的属性
import matplotlib.pyplot as  plt
import  os
from tqdm import tqdm

from  PIL import Image
class gray8_to_color8:
    #八位灰度图转八位彩色图
    #这个的使用前提必须是已经形成了8位无误的灰度图
    def __init__(self,gray8_path,savepath):
        self.gray8_path =gray8_path
        self.savepath=savepath
        # 使用的时候注意修改这个
        self.Cls = namedtuple('cls', ['name', 'id', 'color'])
        self.Clss = [
            self.Cls('c1', 0, (0, 0, 0)),
            self.Cls('c2', 1, (255, 255, 255))
        ]


    def get_putpalette(self,Clss, color_other=[0, 0, 0]):
        '''
        灰度图转8bit彩色图
        :param Clss:颜色映射表
        :param color_other:其余颜色设置
        :return:
        '''
        putpalette = []
        for cls in Clss:
            putpalette += list(cls.color)
        putpalette += color_other * (255 - len(Clss))
        return putpalette


    def Convert_get_color8bit(self,):
        '''
        灰度图转8bit彩色图
        :param grays_path:  灰度图文件路径
        :param colors_path: 彩色图文件路径
        :return:
        '''
        if not os.path.exists( self.savepath):
            os.makedirs( self.savepath)
        file_names = os.listdir(self.gray8_path)
        bin_colormap = self.get_putpalette(  self.Clss)
        with tqdm(file_names) as pbar:
            for file_name in pbar:
                gray_path = os.path.join(self.gray8_path, file_name)
                color_path = os.path.join( self.savepath, file_name.replace('.bmp', '.bmp'))
                gt = Image.open(gray_path)
                gt = gt.convert("P")
                gt.putpalette(bin_colormap)
                gt.save(color_path)
                pbar.set_description('get color')
        print("转换完成")


if __name__ == '__main__':
    gray8_path=""
    savepath=""
    model = gray8_to_color8(gray8_path,savepath)
    model.Convert_get_color8bit()

