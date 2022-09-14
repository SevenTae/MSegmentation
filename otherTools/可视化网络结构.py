import hiddenlayer as h
import torch
<<<<<<< HEAD
from nets.bisegnet.bisenetv1 import BiSeNetV1
from  nets.bisegnet.bisenetv2 import BiSeNetV2

# from nets.unet_improve1.unet_model import UNet
from torch import nn
import torch.nn.functional as F

from netCompaire.bisegnet import BiSeNetV2
# from nets.unet_improve1.unet_model import UNet


'''
一个例子
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.output = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = F.relu(self.block1(x) + identity)
        x = self.output(x)
        return x

if __name__ == '__main__':
    d = torch.rand(1, 3, 416, 416)
    m = model()
    o = m(d)
    vis_graph= h.build_graph(m, d)
    vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
    vis_graph.save("./images/demo1.png")   #
'''



net = BiSeNetV2(n_classes=20,aux_mode='eval')
net.eval()
inputs= torch.randn([1 ,3, 512, 512])
vis_graph = h.build_graph(net, inputs)   # 获取绘制图像的对象
vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
vis_graph.save("./images/demo1.png")   #
