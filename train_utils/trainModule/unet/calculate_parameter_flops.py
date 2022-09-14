'''
计算网络或者model的参数量和FLOPS（貌似是计算量）
'''
'''cr：https://blog.csdn.net/qq_35407318/article/details/109359006
      http://t.csdn.cn/prmSk

'''
import torch
import torch
from torchsummary import summary
import torch
from torch import  nn
from  nets.unet.unet_model import UNet,weights_init
import torch
from torchsummary import summary
from thop import profile

import  numpy as np
import  random
import torch
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子

'''计算网络的参数量和计算量'''

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model =UNet(n_channels=3,n_classes=2).to(device)
    input = torch.randn(1, 3, 512, 512).to(device)
    flops, params = profile(model, inputs=(input,))
    print("FLOPS:",flops)
    print("params:",params)

    print("Total FLOPS: %.2fG" % (flops/1e9))
    print("Total params: %.2fM" % (params/1e6))

    # filename = "shufffle1unet的测评指标"
    # with open(r"F:\MSegmentation\Test_log\{}.txt".format(filename), mode='a', encoding='utf-8') as f:
    #     f.write("模型的复杂度:\n")
    #     f.write("Total FLOPS: %.2fG\n" % (flops/1e9))
    #     f.write("Total FLOPS: %.2fM\n" % (params/1e6))
    # f.close()
    # print("结果已经保存")

    # torch.save(model.state_dict(),"test.pth")