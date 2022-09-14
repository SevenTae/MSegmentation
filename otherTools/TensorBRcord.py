from tensorboardX import SummaryWriter


from nets.unet.unet_model import UNet
from nets.bisegnet.bisenetv1 import BiSeNetV1

from netCompaire.bisegnet import BiSeNetV1
import torch
writer = SummaryWriter("./logs")
input_to_model=torch.randn([1,3,512,512])
# model = UNet(n_channels=3,n_classes=20)
model=BiSeNetV1(n_classes=20,aux_mode='eval')
model.eval()
writer.add_graph(model, input_to_model=input_to_model, verbose=False)
writer.close()

