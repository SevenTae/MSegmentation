# -*- coding:utf-8 -*-

import os
import operator

class ImageRename():
    def __init__(self):
        self.path = r'D:\MSegmentation\data\VOCdevkit\VOC2012\SegmentationClass'  # 文件位置
        self.a = r"D:\MSegmentation\data\VOCdevkit\VOC2012\SegmentationClass"
    def rename(self):
        filelist = os.listdir(self.path)  # 源文件
        filelist.sort()  # 保证按照原文件夹图片的顺序进行重命名
        total_num = len(filelist)  # 原文件的大小
        print(filelist)
        print(len(filelist))
        #
        # b = os.listdir(self.path)  # 源文件
        # b.sort()  # 保证按照原文件夹图片的顺序进行重命名
        # total_num = len(b)  # 原文件的大小
        # print(b)
        # print(len(b))
        # print(operator.eq(filelist,b))


        # 文件的第一张图片命名
        i = 0

        for item in filelist:
            if item.endswith('.png'):  # 文件类型
                src = os.path.join(os.path.abspath(self.path),
                                   item)  # 源路径 #os.path.join路径拼接文件路径，可以传入多个参数。 os.path.abspath取指定文件或目录的绝对路径（完整路径）
                dst = os.path.join(os.path.abspath(self.path),  "Berlin_"+"00"+format(str(i)) + '.png')  # 指定的完整路径
                os.rename(src, dst)  # 重命名函数，
                print('converting %s to %s ...', src, dst)
                i = i + 1
        print('total %d to rename & converted %d pngs', total_num, i)
    

if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()
