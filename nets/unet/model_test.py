from unet import  unet_model
from unet import  unet_parts
import torch

input=torch.randn([1,3,512,512])
model = unet_model.UNet(n_channels=3,n_classes=19)
out = model(input)
print(out.shape)



# x5= torch.randn([1,1024,32])
# x4 = torch.randn([1,512,64,64])
#
# modelup1= unet_parts.Up()