import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
import  cv2 as cv

from Tools.gradcam.utils import GradCAM, show_cam_on_image, center_crop_img
from nets.unet.unet_model import UNet
#是你的网络训练好了以后使用这个进行可视化
def main():



    model = UNet(n_channels=3,n_classes=2)

    model_path = "./best.pth"
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['net'])
    print("权重加载")


    # 查看某一层的 比如resnet layer4(这其中就包括了各种cbr以及模块的堆叠)。
    # target_layers = [model.down4.maxpool_conv[1].double_conv]
    target_layers = [model.up3.conv.double_conv]




    data_transform = transforms.Compose([transforms.ToTensor(),
                                      ])
    # load image
    img_path = "./image/PV01_325500_1203825.bmp"
    image_name ="PV01_325500_1203825"  # 结果保存的名字 最好是与原图一样的前缀名字
    save_path ="./result/" #结果保存路径 注意末尾必须有/
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 256)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False) #把你的模型 taget_layer传进来 #这玩意当黑盒知道怎么用算了
    target_category = 1 #要可视化的类别
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :] #0就是取你第一张图 如果你有多个batch的话
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  #画图展示
                                      grayscale_cam,
                                      use_rgb=True)

    plt.imshow(visualization)
    plt.show()

    visualimage = Image.fromarray(visualization)
    visualimage.save( "{}/{}.png".format(save_path, image_name))






if __name__ == '__main__':
    main()
