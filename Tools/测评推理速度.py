import torch
import time
<<<<<<< HEAD
from torch import nn
import torchvision.models as models
from PIL import Image
from nets.CGNet.CGNet import Context_Guided_Network
import numpy as np
from train_utils.otherutils import cvtColor,preprocess_input
=======
from PIL import Image
from netCompaire.CGNet.CGNet import Context_Guided_Network
import numpy as np
from train_utils.otherutils import preprocess_input
####这个废了废了！！


>>>>>>> 8b6166e (大幅度更新)


'''使用真正的图片'''
img_true=True
if img_true:

    img_path =r"F:\MSegmentation\img\test_img.png"
    img= Image.open(img_path)
    input_shape=(1024,512)#w,h
    img=img.resize(input_shape)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(img, np.float32)), (2, 0, 1)), 0)
    inputd =  torch.from_numpy(image_data)
else: #否则使用假的 随机生成的
    inputd = torch.randn([1,3,512,512],dtype=torch.float32)
model = Context_Guided_Network(n_channels=3,classes=19)
cuda =True

'''废了废了 待完善'''


if cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)

input = inputd.to(device)
model = model.to(device)
#推理单个（未进行cuda预热）
# model.eval()
# time0=time.time()
# with torch.no_grad():
    # output = model(input)
# tim1=time.time()
# print("推理时间为:",tim1-time0)

#批量推理
ite=5000#推理多少次 ，理论上越多越好
model.eval()
tim3 = time.time()
with torch.no_grad():
    for i in range(ite):
        out=model(input)
tim4 = time.time()
Sing_Avg_time= (tim4-tim3)/ite
print("平均的时间:",Sing_Avg_time)
print("fps(@1batch):",1/Sing_Avg_time)

'''
cr:http://t.csdn.cn/0Pd3P
需要克服GPU异步执行和GPU预热两个问题，
<<<<<<< HEAD
下面例子使用 Efficient-net-b0，在进行任何时间测量之前，
=======
下面例子使用 Efficient-nets-b0，在进行任何时间测量之前，
>>>>>>> 8b6166e (大幅度更新)
我们通过网络运行一些虚拟示例来进行“GPU 预热”。
这将自动初始化 GPU 并防止它在我们测量时间时进入省电模式。
接下来，我们使用 tr.cuda.event 来测量 GPU 上的时间。
在这里使用 torch.cuda.synchronize() 至关重要。
这行代码执行主机和设备（即GPU和CPU）之间的同步，
因此只有在GPU上运行的进程完成后才会进行时间记录。这克服了不同步执行的问题。
'''

