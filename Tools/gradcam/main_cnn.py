import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from Tools.gradcam.utils import GradCAM, show_cam_on_image, center_crop_img
from nets3.SUnet4.su_model import SUNet,weights_init
#是你的网络训练好了以后使用这个进行可视化
def main():
    model = SUNet(n_channels=3,n_classes=6)

    model_path = r"F:\MSegmentation\customer\trainSunet4_1\checkpoints\best.pth"
    model_dict = torch.load(model_path)
    model.load_state_dict(model_dict['nets'])
    print("权重加载")

    target_layers = [model.backbon.stage4[3].branch2[7]] #这里用的是mobilenet最后一个卷积层




    data_transform = transforms.Compose([transforms.ToTensor(),
                                      ])
    # load image
    img_path = r"F:\MSegmentation\tools\gradcam\image\wh0017.jpg"
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
    target_category = 3
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :] #0就是取你第一张图 如果你有多个batch的话
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,  #画图展示
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
